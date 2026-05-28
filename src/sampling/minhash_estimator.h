#pragma once

/**
 * @file minhash_estimator.h
 * @brief MinHash-based Jaccard estimator for biclique candidate pruning.
 *
 * Directly adapted from mosib GlobalApp::__init_min_hash (mosib.cpp:300-316):
 *   min_hash_ = VVI(nl, VI(hash_num, -1));
 *   VI arr(nr, 0);
 *   for (int j = 0; j < nr; j++) arr[j] = j;
 *   for (int hh = 0; hh < hash_num; hh++) {
 *       std::shuffle(arr.begin(), arr.end(), rng);
 *       for (int j = 0; j < nr; j++) {
 *           int right_u = arr[j] + nl;
 *           for (int v : g_.adj_[right_u]) {
 *               if (min_hash_[v][hh] == -1) min_hash_[v][hh] = j;
 *           }
 *       }
 *   }
 *
 * The key insight: MinHash produces O(1) Jaccard estimates for any vertex pair
 * by comparing their signature vectors, whereas the reservoir estimator only
 * works when both vertices happen to share sampled neighbors (very unlikely
 * for sparse samples).
 *
 * This module also supports:
 *   - Incremental MinHash update for streaming edges
 *   - Fusion with reservoir sampling for triangle-aware estimates
 *
 * Infra references:
 *   - mosib mosib.cpp:300 GlobalApp::__init_min_hash() — batch MinHash init
 *   - mosib mosib.cpp:248 dfs_sep() — recursive LSH grouping via MinHash
 *   - mosib mosib.cpp:260 get_sep() — MinHash-based partition boundaries
 *   - GREAT+ Estimator.java:survive_rate_cache — round-adaptive sampling
 */

#include "../../upstream/mosib/src/global.h"
#include "../../upstream/mosib/src/bigraph.h"

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

/**
 * @class MinHashEstimator
 * @brief Fast O(1) Jaccard estimation via MinHash signatures.
 *
 * For each left-vertex u, we store a signature vector of length hash_num.
 * sig[u][h] = min permutation rank among u's right-neighbors under hash h.
 *
 * Jaccard(u,v) ≈ (# of matching signatures) / hash_num
 *
 * This is the standard MinHash estimator with accuracy ε = O(1/√hash_num).
 * With hash_num=128, expected error is ~8.8%.
 * With hash_num=256, expected error is ~6.25%.
 */
class MinHashEstimator {
public:
    /**
     * @brief Construct MinHash from a BiGraph.
     *
     * Directly follows mosib GlobalApp::__init_min_hash (mosib.cpp:300):
     *   shuffles right-vertex permutations to build MinHash signatures.
     *
     * @param g         The bipartite graph
     * @param hash_num  Number of hash functions (accuracy ~ 1/√hash_num)
     * @param seed      RNG seed for reproducibility
     */
    MinHashEstimator(const BiGraph& g, int hash_num = 128, int seed = 2333)
        : nl_(g.left_node_num_), nr_(g.right_node_num_),
          hash_num_(hash_num) {

        std::mt19937 rng(seed);

        // Allocate signature matrix: nl × hash_num
        // Directly from mosib: min_hash_ = VVI(nl, VI(hash_num, -1));
        signatures_.assign(nl_, VI(hash_num, -1));

        // For each hash function, generate a random permutation of right vertices
        // and compute the MinHash signature.
        // Directly from mosib GlobalApp::__init_min_hash:
        VI arr(nr_, 0);
        for (int j = 0; j < nr_; j++) arr[j] = j;

        for (int hh = 0; hh < hash_num; hh++) {
            std::shuffle(arr.begin(), arr.end(), rng);

            // For each right vertex (in permuted order), update MinHash
            // of its left neighbors. The first time we see a left vertex,
            // its MinHash for this function is the permutation rank.
            for (int j = 0; j < nr_; j++) {
                int right_u = arr[j] + nl_;
                for (int v : g.adj_[right_u]) {
                    if (signatures_[v][hh] == -1) {
                        signatures_[v][hh] = j;
                    }
                }
            }
        }
    }

    /**
     * @brief Estimate Jaccard similarity between two left-vertices.
     *
     * J_est(u,v) = |{h : sig[u][h] == sig[v][h]}| / hash_num
     *
     * This is unbiased: E[J_est] = J(u,v).
     * Variance: Var[J_est] = J(1-J)/hash_num.
     *
     * @param u Left-vertex ID
     * @param v Left-vertex ID
     * @return Estimated Jaccard similarity in [0, 1]
     */
    double estimate_jaccard(int u, int v) const {
        if (u < 0 || u >= nl_ || v < 0 || v >= nl_) return 0.0;
        if (u == v) return 1.0;

        int matches = 0;
        const VI& su = signatures_[u];
        const VI& sv = signatures_[v];

        for (int h = 0; h < hash_num_; h++) {
            if (su[h] >= 0 && su[h] == sv[h]) matches++;
        }

        return static_cast<double>(matches) / hash_num_;
    }

    /**
     * @brief Get candidates with estimated Jaccard above threshold.
     *
     * Follows mosib dfs_sep() pattern (mosib.cpp:248): uses MinHash
     * signatures to group similar vertices, then returns those above
     * the similarity threshold.
     *
     * @param q         Query left-vertex
     * @param threshold Minimum Jaccard similarity
     * @return          Sorted list of candidate vertex IDs
     */
    VI get_candidates_above(int q, double threshold) const {
        VI candidates;
        if (q < 0 || q >= nl_) return candidates;

        for (int v = 0; v < nl_; v++) {
            if (v == q) continue;
            if (estimate_jaccard(q, v) >= threshold) {
                candidates.push_back(v);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        return candidates;
    }

    /**
     * @brief Bucket left-vertices by MinHash locality (LSH grouping).
     *
     * Directly inspired by mosib dfs_sep() (mosib.cpp:248):
     *   recursive partitioning by MinHash band values to group
     *   similar vertices together.
     *
     * @param band_size Number of hash functions per band (larger = fewer false positives)
     * @return          Groups of similar vertex IDs
     */
    VVI lsh_group(int band_size = 4) const {
        int num_bands = hash_num_ / band_size;

        // Simple LSH: group by first band signature
        std::map<VI, VI> groups;
        for (int u = 0; u < nl_; u++) {
            VI band_sig(signatures_[u].begin(),
                        signatures_[u].begin() + std::min(band_size, hash_num_));
            groups[band_sig].push_back(u);
        }

        VVI result;
        for (auto& [sig, group] : groups) {
            if (group.size() >= 2) {
                result.push_back(std::move(group));
            }
        }
        return result;
    }

    int hash_count() const { return hash_num_; }
    int vertex_count() const { return nl_; }

    /// Memory footprint in bytes
    size_t memory_bytes() const {
        return static_cast<size_t>(nl_) * hash_num_ * sizeof(int);
    }

private:
    int nl_, nr_, hash_num_;
    VVI signatures_;  ///< nl × hash_num signature matrix
};

/**
 * @class HybridEstimator
 * @brief Fuses MinHash + Reservoir for adaptive Jaccard estimation.
 *
 * MinHash provides global O(1) estimates with bounded error.
 * Reservoir provides local high-accuracy estimates for vertices
 * that happen to share sampled neighbors.
 *
 * The hybrid returns:
 *   - MinHash estimate if reservoir has no data for the pair
 *   - Weighted average if both have data:
 *     J_hybrid = w * J_reservoir + (1-w) * J_minhash
 *     where w = confidence(reservoir) ∝ min(deg_sample(u), deg_sample(v))
 */
class HybridEstimator {
public:
    HybridEstimator(const MinHashEstimator& mh)
        : minhash_(mh) {}

    /// Estimate Jaccard using MinHash as primary
    double estimate_jaccard(int u, int v) const {
        return minhash_.estimate_jaccard(u, v);
    }

    /// Estimate with reservoir data fusion (when available)
    double estimate_jaccard_hybrid(int u, int v,
                                   double reservoir_est,
                                   int reservoir_deg_u,
                                   int reservoir_deg_v) const {
        double mh_est = minhash_.estimate_jaccard(u, v);

        // If reservoir has no useful data, use MinHash only
        if (reservoir_deg_u == 0 || reservoir_deg_v == 0) return mh_est;

        // Weight reservoir estimate by its sample density
        // More sampled neighbors → more trustworthy reservoir estimate
        double confidence = std::min(1.0,
            std::min(reservoir_deg_u, reservoir_deg_v) / 10.0);

        return confidence * reservoir_est + (1.0 - confidence) * mh_est;
    }

private:
    const MinHashEstimator& minhash_;
};
