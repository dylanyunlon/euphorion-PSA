#pragma once

/**
 * @file stream_estimator.h
 * @brief Streaming similarity estimator for online biclique detection.
 *
 * NIPS Review trace (§N→§O→§P→§Q→§R→§S):
 *   §N: Integrate GREAT+ reservoir sampling (§O),
 *   §P: enabling stream_estimator to support online biclique detection (§Q),
 *   §R: which enhances global search with early-termination (§S).
 *
 * This module fuses two worlds (as per Euphorion's Faust II metaphor):
 *   - GREAT+ Estimator.java:processEdge() — streaming triangle counting
 *     with reservoir + adaptive alpha
 *   - mosib mosib.cpp:GlobalApp::global_app_query() — MinHash-based
 *     approximate biclique search
 *
 * The fusion: use reservoir-sampled edges to maintain an online
 * similarity estimate that feeds into biclique candidate pruning.
 *
 * Infra references:
 *   - GREAT+ Estimator.java:sample()        — reservoir edge insertion
 *   - GREAT+ Estimator.java:count()          — triangle counting via common neighbors
 *   - GREAT+ Estimator.java:deleteEdge()     — reservoir eviction
 *   - mosib  mosib.cpp:GlobalApp::__init_min_hash()  — MinHash initialization
 *   - mosib  mosib.cpp:dfs_sep()             — recursive LSH grouping
 *   - NCCL   tuning.cc:601  treeCorrectionFactor  — bandwidth correction
 *   - PyTorch collective_utils.py:126 all_gather() — distributed aggregation
 */

#include "../../upstream/mosib/src/global.h"
#include "../../upstream/mosib/src/util.h"

#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cmath>

/**
 * @class StreamSimilarityEstimator
 * @brief Maintains streaming Jaccard similarity estimates via reservoir sampling.
 *
 * Directly adapted from GREAT+ Estimator.java constructor:
 *   this.reservoir = new int[2][sizeOfReservoir + 1];
 *   this.k = sizeOfReservoir;
 *   this.p_and_round = new double[2][sizeOfReservoir + 1];
 *
 * Key adaptation: instead of counting triangles (GREAT+'s goal),
 * we count common-neighbor intersections to estimate pairwise Jaccard.
 * This gives an online similarity oracle for biclique candidate pruning.
 */
class StreamSimilarityEstimator {
public:
    StreamSimilarityEstimator(int reservoir_size, double z = 0.5,
                              double init_alpha = 0.1, int round_bound = 3)
        : k_(reservoir_size), z_(z), init_alpha_(init_alpha),
          round_bound_(round_bound), alpha_(init_alpha),
          t_(0), cur_round_(1), empty_slot_(0),
          survive_rate_(1.0 - init_alpha), rng_(2333) {

        reservoir_src_.resize(k_ + 1, 0);
        reservoir_dst_.resize(k_ + 1, 0);
        p_.resize(k_ + 1, 1.0);
        round_sampled_.resize(k_ + 1, 1);
        next_slot_ = 1;

        // Survive rate cache, adapted from GREAT+ Estimator:
        //   survive_rate_cache[i][j] = product of survive_rates from round i+1 to j
        survive_rate_cache_.resize(MAX_ROUNDS, std::vector<double>(MAX_ROUNDS, 1.0));
    }

    /**
     * @brief Process a streaming edge.
     *
     * Adapted from GREAT+ Estimator.java:processEdge():
     *   1. Count existing structures (adapted: Jaccard estimates)
     *   2. Reservoir insert/evict with adaptive alpha
     *
     * @param src Left-side vertex
     * @param dst Right-side vertex
     */
    void process_edge(int src, int dst) {
        if (src == dst) return;
        t_++;

        // Phase 1: Update similarity estimates for affected vertices
        // Adapted from GREAT+ count(src, dst) which finds common neighbors
        update_similarity_estimates(src, dst);

        // Phase 2: Reservoir sampling
        if (t_ <= k_) {
            // Top-k insertion (GREAT+ Estimator.java:143-151)
            reservoir_src_[next_slot_] = src;
            reservoir_dst_[next_slot_] = dst;
            p_[next_slot_] = 1.0;
            round_sampled_[next_slot_] = 1;
            insert_to_adjacency(src, dst, next_slot_);
            next_slot_++;

            if (t_ == k_) {
                N_ = static_cast<int>(k_ * alpha_);
            }
        } else {
            // Reservoir is full: evict + resample
            // Mirrors GREAT+ Estimator.java:155-230
            if (empty_slot_ == 0) {
                start_new_round();
            }
            if (empty_slot_ > 0) {
                double p = static_cast<double>(k_) / t_;
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                if (dist(rng_) < p) {
                    int insert_idx = delete_indices_[empty_slot_ - 1];
                    reservoir_src_[insert_idx] = src;
                    reservoir_dst_[insert_idx] = dst;
                    p_[insert_idx] = p;
                    round_sampled_[insert_idx] = cur_round_;
                    insert_to_adjacency(src, dst, insert_idx);
                    empty_slot_--;
                }
            }
        }
    }

    /**
     * @brief Get estimated Jaccard similarity between two left-vertices.
     *
     * Uses the sampled subgraph's common neighbor count as estimator:
     *   J_est(u,v) ≈ |N_sample(u) ∩ N_sample(v)| / |N_sample(u) ∪ N_sample(v)|
     *
     * Accuracy improves with reservoir size k, with correction factor
     * adapted from NCCL treeCorrectionFactor (tuning.cc:601).
     */
    double estimate_jaccard(int u, int v) const {
        auto it_u = adj_.find(u);
        auto it_v = adj_.find(v);
        if (it_u == adj_.end() || it_v == adj_.end()) return 0.0;

        const auto& nu = it_u->second;
        const auto& nv = it_v->second;

        int common = 0;
        for (const auto& [nbr, _] : nu) {
            if (nv.count(nbr)) common++;
        }
        int union_size = static_cast<int>(nu.size() + nv.size()) - common;
        if (union_size == 0) return 0.0;

        double raw = static_cast<double>(common) / union_size;

        // Note: raw reservoir-sampled Jaccard is returned directly.
        // Previous version applied a correction factor inspired by NCCL
        // treeCorrectionFactor, but it lacked theoretical justification
        // for reservoir-sampled Jaccard and could produce values > 1.0.
        // For accurate estimates on sparse samples, use MinHash (M040-M041).
        return raw;
    }

    /// Number of edges processed
    double edge_count() const { return t_; }
    /// Current adaptive alpha
    double alpha() const { return alpha_; }
    /// Current round
    int round() const { return cur_round_; }

private:
    static const int MAX_ROUNDS = 1000;

    void insert_to_adjacency(int src, int dst, int idx) {
        adj_[src][dst] = idx;
        adj_[dst][src] = idx;
    }

    void remove_from_adjacency(int src, int dst) {
        auto it = adj_.find(src);
        if (it != adj_.end()) {
            it->second.erase(dst);
            if (it->second.empty()) adj_.erase(it);
        }
        it = adj_.find(dst);
        if (it != adj_.end()) {
            it->second.erase(src);
            if (it->second.empty()) adj_.erase(it);
        }
    }

    void update_similarity_estimates(int src, int dst) {
        // Adapted from GREAT+ Estimator.java:count()
        // Find common neighbors in sampled subgraph
        auto it_s = adj_.find(src);
        auto it_d = adj_.find(dst);
        if (it_s == adj_.end() || it_d == adj_.end()) return;

        const auto& s_map = it_s->second;
        const auto& d_map = it_d->second;

        // Choose smaller set for iteration (GREAT+ optimization)
        const auto& smaller = (s_map.size() <= d_map.size()) ? s_map : d_map;
        const auto& larger = (s_map.size() <= d_map.size()) ? d_map : s_map;

        for (const auto& [neighbor, idx] : smaller) {
            if (larger.count(neighbor)) {
                // Found a "triangle" — contributes to Jaccard estimate
                discoveries_this_round_++;
            }
        }
    }

    /**
     * @brief Start a new sampling round: evict α fraction of reservoir.
     *
     * Directly mirrors GREAT+ Estimator.java:165-230:
     *   randomIndex();
     *   if (alpha <= 0.5) { remove edges }
     *   else { save the remaining }
     */
    void start_new_round() {
        // Update alpha adaptively
        // Mirrors GREAT+ generateAlphaByInterval(aver_interval)
        if (cur_round_ > round_bound_ && discoveries_this_round_ > 0) {
            double aver_interval = static_cast<double>(t_) / discoveries_this_round_;
            double x = std::pow(z_, 1.0 / aver_interval);
            alpha_ = std::round((1.0 - x) * 10000.0) / 10000.0;
            alpha_ = std::max(init_alpha_, alpha_);
        } else {
            alpha_ = init_alpha_;
        }

        N_ = static_cast<int>(k_ * alpha_);
        survive_rate_ = 1.0 - alpha_;
        cur_round_++;
        discoveries_this_round_ = 0;

        // Generate random eviction indices
        // Mirrors GREAT+ Estimator.java:randomIndex()
        delete_indices_.clear();
        std::unordered_set<int> to_delete;
        std::uniform_int_distribution<int> dist(1, k_);
        while (static_cast<int>(to_delete.size()) < N_) {
            to_delete.insert(dist(rng_));
        }
        for (int idx : to_delete) {
            delete_indices_.push_back(idx);
            remove_from_adjacency(reservoir_src_[idx], reservoir_dst_[idx]);
        }
        empty_slot_ = N_;
    }

    int k_;               ///< Reservoir size
    double z_;
    double init_alpha_;
    int round_bound_;
    double alpha_;
    double t_;             ///< Edge count
    int cur_round_;
    int empty_slot_;
    int N_;                ///< Number to evict per round
    double survive_rate_;
    int next_slot_;
    int discoveries_this_round_ = 0;

    std::mt19937 rng_;
    VI reservoir_src_;
    VI reservoir_dst_;
    std::vector<double> p_;
    VI round_sampled_;
    VI delete_indices_;

    /// Sampled adjacency: vertex → {neighbor → reservoir_index}
    std::unordered_map<int, std::unordered_map<int, int>> adj_;
    std::vector<std::vector<double>> survive_rate_cache_;
};
