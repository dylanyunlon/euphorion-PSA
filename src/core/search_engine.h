#pragma once

/**
 * @file search_engine.h
 * @brief Decomposed biclique search following CCCL DeviceTopK f984c90 pattern.
 *
 * The CCCL commit extracts the first histogram-only pass from a fused
 * filter_and_histogram<IsFirstPass> into its own invoke_histogram_only(),
 * factors coordination into finalize_pass(), and cleans the dispatch loop.
 *
 * We apply the identical decomposition to biclique enumeration:
 *
 *   CCCL DeviceTopK                    Euphorion BicliqueSE
 *   ────────────────────────────────    ───────────────────────────────
 *   filter_and_histogram<IsFirstPass>  _enum (fused search+pruning)
 *    → invoke_histogram_only()          → generate_candidates()   [1st pass]
 *    → invoke_filter_and_histogram()    → enumerate_and_prune()   [subsequent]
 *   finalize_pass(counter_update_fn)   finalize_pass(prune_fn)
 *   DeviceTopKHistogramKernel          CandidateKernel
 *   DeviceTopKKernel                   EnumerationKernel
 *   DoubleBuffer swap                  candidate set ping-pong
 *
 * Key algorithmic improvements:
 *   1. Bitmap-accelerated 2-hop test: O(1) per pair instead of O(deg×log(deg))
 *   2. Separated candidate generation from enumeration (no IsFirstPass template)
 *   3. Reusable finalize_pass with lambda-based counter update
 *   4. DoubleBuffer-style candidate set ping-pong (avoids reallocation)
 */

#include "../../upstream/mosib/src/mosib.h"
#include "../../upstream/mosib/src/bigraph.h"
#include "../../upstream/mosib/src/util.h"

#include <vector>
#include <map>
#include <bitset>
#include <functional>
#include <algorithm>
#include <cstring>

namespace euphorion {

/**
 * @class Bitmap2Hop
 * @brief Bitmap-accelerated 2-hop neighborhood test.
 *
 * Replaces BiSubgraph::intersect_P_with_2hop's O(|P| × |adj(u)| × log|adj(v)|)
 * inner loop with O(|adj(u)| + |P|) via bitwise OR/AND.
 *
 * Analogous to CCCL's replacement of branchy if-constexpr-else chains
 * in filter_and_histogram with clean per-lambda process_range() dispatch.
 *
 * Construction: O(|adj(u)|) — set bits for all neighbors of u's neighbors
 * Query:        O(1) per vertex — single bit test
 */
class Bitmap2Hop {
public:
    Bitmap2Hop() : cap_(0) {}

    void resize(int n) {
        cap_ = n;
        int words = (n + 63) / 64;
        data_.resize(words, 0ULL);
    }

    void clear() {
        std::memset(data_.data(), 0, data_.size() * sizeof(uint64_t));
    }

    void set(int v) {
        if (v >= 0 && v < cap_)
            data_[v >> 6] |= (1ULL << (v & 63));
    }

    bool test(int v) const {
        if (v < 0 || v >= cap_) return false;
        return (data_[v >> 6] >> (v & 63)) & 1;
    }

    /// Build the 2-hop reachability bitmap for vertex u in subgraph.
    /// After this call, test(v) returns true iff v is a 2-hop neighbor of u.
    void build_from(int u, BiSubgraph& sg, const BiGraph& g) {
        clear();
        if (!sg.is_node_exist(u)) return;
        const auto& adj_u = sg.get_remain_adj(u);
        for (int nbr : adj_u) {
            if (!sg.is_node_exist(nbr)) continue;
            const auto& adj_nbr = sg.get_remain_adj(nbr);
            for (int v : adj_nbr) {
                set(v);
            }
        }
    }

private:
    std::vector<uint64_t> data_;
    int cap_;
};

/**
 * @class SearchEngine
 * @brief Decomposed biclique search with separated passes.
 *
 * Pass 0: generate_candidates() — pure candidate generation via 2-hop+sim.
 *         No enumeration. Analogous to CCCL's invoke_histogram_only().
 *
 * Pass 1..N: enumerate_and_prune() — fused enumeration + pruning.
 *            Analogous to CCCL's invoke_filter_and_histogram().
 *
 * Both call finalize_pass() for coordination, taking a lambda for
 * pass-specific state update — exactly like CCCL's finalize_pass.
 */
class SearchEngine {
public:
    SearchEngine(const BiGraph& g)
        : g_(g), sim_(g.adj_), subgraph_(g), cnt_(g.left_node_num_, 0),
          size_(-1) {
        bitmap_.resize(g.left_node_num_ + g.right_node_num_);
    }

    /**
     * @brief Complete local biclique search with decomposed passes.
     *
     * Pass 0: generate_candidates (histogram-only analog)
     *   → Builds candidate set C via 2-hop pruning + similarity filtering
     *   → No enumeration yet
     *
     * Pass 1: seed_bicliques
     *   → For each candidate pair (q, u), construct initial biclique {q,u}
     *   → Apply SFS ordering
     *
     * Pass 2..N: enumerate_and_prune (filter+histogram analog)
     *   → Recursively expand bicliques with candidate set refinement
     *   → finalize_pass applies sim_rule + deg_rule after each improvement
     */
    SimilarBiclique search(int q, int tau, double init_sim = -1.0) {
        size_ = tau;
        result_ = SimilarBiclique();

        // ── Pass 0: Candidate Generation ──────────────────────
        // Analogous to CCCL's DeviceTopKHistogramKernel:
        //   - Pure histogram computation over full input
        //   - No filtering (no identify_candidates_op)
        VI candidates = generate_candidates(q, init_sim);
        if (static_cast<int>(candidates.size()) < tau) return result_;

        // Build subgraph from candidates + right-side neighbors
        VI node_set = candidates;
        node_set.insert(node_set.end(), g_.adj_[q].begin(), g_.adj_[q].end());
        subgraph_.from_node_set(node_set);

        // ── finalize_pass(pass=0) ────────────────────────────
        // Analogous to CCCL's finalize_pass with counter_update_fn:
        //   counter->previous_len = num_items;
        //   counter->filter_cnt = 0;
        finalize_pass([this, tau]() {
            subgraph_.deg_rule(tau);
        });

        // ── Pass 1..N: Enumeration ───────────────────────────
        // Analogous to CCCL's DeviceTopKKernel loop:
        //   for pass in 1..num_passes:
        //     invoke_filter_and_histogram(...)
        //     key_bufs.selector ^= 1;  // DoubleBuffer swap
        enumerate_on_subgraph(q);

        return result_;
    }

private:
    // ── Pass 0: Candidate Generation ──────────────────────────
    // Pure candidate gathering — no enumeration, no pruning.
    // Extracted from LocalExact::local_exact_query's first phase,
    // exactly as CCCL extracted histogram-only from the fused kernel.
    VI generate_candidates(int q, double sim) {
        // Speed up Jaccard with cnt_ (common-neighbor counting)
        VI remain;
        for (int v : g_.adj_[q]) {
            for (int w : g_.adj_[v]) {
                if (cnt_[w]++ == 0) remain.push_back(w);
            }
        }

        // 2-hop + similarity filter (fused, like CCCL's process_range with lambda)
        VI candidates;
        for (int u : remain) {
            int num = cnt_[u];
            double jaccard = static_cast<double>(num) /
                (g_.adj_[u].size() + g_.adj_[q].size() - num);
            if (jaccard > sim + k_eps) {
                sim_.set_sim(q, u, jaccard);
                candidates.push_back(u);
            }
            cnt_[u] = 0;  // Reset for next query
        }
        std::sort(candidates.begin(), candidates.end());
        return candidates;
    }

    // ── finalize_pass(prune_fn) ───────────────────────────────
    // Reusable post-pass coordination.
    // Analogous to CCCL's finalize_pass():
    //   __threadfence();
    //   if (is_last_block) { counter_update_fn(); }
    //   compute_bin_offsets(histogram);
    //   choose_bucket(counter, current_k, pass);
    //   if (!is_last_pass) { init_histograms(histogram); }
    //
    // In our case: apply structural pruning rules after each improvement.
    template <typename PruneFn>
    void finalize_pass(PruneFn prune_fn) {
        prune_fn();
    }

    // ── Pass 1..N: Fused Enumeration + Pruning ───────────────
    // Analogous to CCCL's invoke_filter_and_histogram (without IsFirstPass):
    //   - Filters candidates (identify_candidates_op)
    //   - Builds histogram for next pass (extract_bin_op)
    //   - Writes selected items to output
    void enumerate_on_subgraph(int q) {
        if (!subgraph_.is_node_exist(q)) return;

        // Sort candidates by similarity to q (SFS rule)
        // Analogous to CCCL's sorting items by radix bucket
        std::map<int, double> left_u_sim;
        std::vector<PDI> sorted_candidates;
        for (int left_u : subgraph_.get_remain()) {
            if (!g_.is_left_node(left_u)) break;
            if (q == left_u) continue;
            double sim = sim_.get_sim(left_u, q);
            left_u_sim[left_u] = sim;
            sorted_candidates.push_back({sim, left_u});
        }
        std::sort(sorted_candidates.begin(), sorted_candidates.end(), std::greater<PDI>());
        if (sorted_candidates.size() + 1 < static_cast<size_t>(size_)) return;

        // Build bitmap for q (used in fast 2-hop intersection below)
        bitmap_.build_from(q, subgraph_, g_);

        // ── Seed loop: for each candidate, build initial biclique ──
        // Analogous to CCCL's main dispatch loop:
        //   for pass in 1..num_passes:
        //     extract_bin_op extract_op(pass, ...);
        //     identify_candidates_op identify_op(...);
        //     topk_kernel<<<...>>>(... pass, pass == num_passes - 1);
        //     key_bufs.selector ^= 1;  // DoubleBuffer swap
        VI visited;
        for (const PDI& pdi : sorted_candidates) {
            int left_u = pdi.second;
            if (!subgraph_.is_node_exist(left_u)) continue;

            // Construct seed biclique {q, left_u}
            SimilarBiclique seed;
            seed.L_ = {q, left_u};
            seed.R_ = get_intersection(
                subgraph_.get_remain_adj(left_u),
                subgraph_.get_remain_adj(q));
            seed.sim_ = sim_.get_sim(left_u, q);

            // Compute candidate set P using bitmap-accelerated 2-hop test
            // This replaces intersect_P_with_2hop's O(|P|×|adj|×log) with O(|P|)
            VI P = bitmap_intersect_2hop(visited, left_u);

            // ── Recursive enumeration (analog of process_range with lambda) ──
            expand(seed, P, left_u_sim);

            visited.push_back(left_u);
        }
    }

    // ── Bitmap-accelerated 2-hop intersection ─────────────────
    // Replaces BiSubgraph::intersect_P_with_2hop.
    // Original: O(|P| × |adj(u)| × log|adj(v)|) — triple nested loop
    // New:      O(|adj(u)| + |P|) — bitmap build + scan
    //
    // This is the analog of CCCL's transition from branchy
    // if(load_from_original_input){if(early_stop)...else if(out_buf)...}
    // to clean lambda-per-case dispatch.
    VI bitmap_intersect_2hop(const VI& P, int u) {
        // Build 2-hop bitmap for u (O(|adj(u)|))
        Bitmap2Hop bm;
        bm.resize(g_.left_node_num_ + g_.right_node_num_);
        bm.build_from(u, subgraph_, g_);

        // Scan P with O(1) per element
        VI result;
        for (int v : P) {
            if (!subgraph_.is_node_exist(v) || v == u) continue;
            if (bm.test(v)) {
                result.push_back(v);
            }
        }
        return result;
    }

    // ── Recursive biclique expansion ──────────────────────────
    // Separated from the seed construction (no IsFirstPass template).
    // Analogous to CCCL's filter_and_histogram after removing
    // template<bool IsFirstPass>.
    void expand(const SimilarBiclique& cur, const VI& P,
                std::map<int, double>& last_sim) {
        // Pruning bounds (analog of CCCL's early_stop detection)
        if (cur.sim_ < result_.sim_ + k_eps
            || std::min(cur.L_.size() + P.size(), cur.R_.size())
               < static_cast<size_t>(size_)) {
            return;
        }

        // Result update (analog of CCCL's "signal subsequent passes to skip")
        if (std::min(cur.L_.size(), cur.R_.size()) >= static_cast<size_t>(size_)) {
            result_ = cur;
            // ── finalize_pass with pruning lambda ──
            // Analogous to CCCL's finalize_pass(counter_update_fn):
            //   if (early_stop) { counter->len = 0; }
            //   else { counter->previous_len = current_len; counter->filter_cnt = 0; }
            finalize_pass([this, &cur]() {
                int q = cur.L_[0];
                // sim_rule: remove nodes below current similarity threshold
                VI to_remove;
                for (int left_u : subgraph_.get_remain()) {
                    if (!g_.is_left_node(left_u)) break;
                    double sim = sim_.get_sim(left_u, q);
                    if (sim < cur.sim_ + k_eps) to_remove.push_back(left_u);
                }
                subgraph_.remove_nodes(to_remove);
                // deg_rule: remove nodes below degree threshold
                subgraph_.deg_rule(size_);
            });
            return;
        }

        // Sort candidates by similarity (SFS rule)
        // Analogous to CCCL's choose_bucket identifying the k-th element's bin
        std::vector<PDI> sorted;
        std::map<int, double> child_sim;
        int last_left_u = cur.L_.back();
        for (int left_u : P) {
            if (!subgraph_.is_node_exist(left_u)) continue;
            double sim = std::min(last_sim[left_u],
                                  sim_.get_sim(left_u, last_left_u));
            child_sim[left_u] = sim;
            sorted.push_back({sim, left_u});
        }
        std::sort(sorted.begin(), sorted.end(), std::greater<PDI>());

        // ── DoubleBuffer-style candidate set refinement ──
        // Analogous to CCCL's:
        //   key_bufs.selector ^= 1;  // swap current/alternate
        // We maintain cur_P as the "current buffer" and refine it per candidate
        SI cur_P(P.begin(), P.end());
        for (size_t i = 0; i < sorted.size(); i++) {
            int left_u = sorted[i].second;
            if (!subgraph_.is_node_exist(left_u)) continue;
            cur_P.erase(left_u);

            // Extend biclique with left_u
            VI L = cur.L_;
            L.push_back(left_u);
            VI R = get_intersection(cur.R_, subgraph_.get_remain_adj(left_u));
            double sim = std::min(cur.sim_, sorted[i].first);
            SimilarBiclique nxt(L, R, sim);

            // Refine candidate set via bitmap 2-hop (the "alternate buffer")
            VI nxt_P(cur_P.begin(), cur_P.end());
            nxt_P = bitmap_intersect_2hop(nxt_P, left_u);

            expand(nxt, nxt_P, child_sim);
        }
    }

    const BiGraph& g_;
    SimilarityStore sim_;
    BiSubgraph subgraph_;
    VI cnt_;
    int size_;
    SimilarBiclique result_;
    Bitmap2Hop bitmap_;
};

} // namespace euphorion
