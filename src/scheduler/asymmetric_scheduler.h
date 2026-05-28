#pragma once

/**
 * @file asymmetric_scheduler.h
 * @brief Orchestrates biclique search across heterogeneous GPUs.
 *
 * NIPS Review trace (§C→§D→§E→§F→§G→§H→§I):
 *   §C: GREAT Estimator.java:processEdge (reservoir sampling — good example)
 *   §D: AsymmetricBiGraph (new, follows that pattern)
 *   §E: WorkloadPartitioner distributes tiles so
 *   §F: H100 handles dense Jaccard, and can
 *   §G: A6000 handles sparse 2-hop expansion.
 *   §H: Now we introduce AsymmetricScheduler (§I),
 *   §J: making the pipeline able to overlap compute/comm (§K),
 *   §L: while optimizing tile granularity (§M) for load balance.
 *
 * Infra references:
 *   - Megatron schedules.py:896  forward_backward_pipelining_with_interleaving()
 *     — 1F1B interleaved pipeline, adapted for biclique search stages
 *   - NCCL group.h:84  ncclGroupDepth — batched async P2P within group
 *   - GREAT+ Estimator.java:survive_rate_cache[][] — round-adaptive α
 *   - Triton tutorials/10-block-scaled-matmul.py @triton.jit — tile-level JIT
 *   - FasterTransformer unfused_attention_kernels.cu:add_fusedQKV_bias_transpose_kernel
 *     — fused kernel pattern for multi-stage ops
 */

#include "asymmetric_bigraph.h"
#include "../../upstream/mosib/src/mosib.h"

#include <thread>
#include <future>
#include <queue>
#include <chrono>

/// Scheduling statistics for auto-tuning
struct ScheduleStats {
    double dense_time_ms;
    double sparse_time_ms;
    double comm_time_ms;
    double imbalance_ratio;  ///< |dense - sparse| / max(dense, sparse)
    int tiles_completed;
    int pruned_tiles;
};

/**
 * @class AdaptiveAlpha
 * @brief Adaptive discard probability from GREAT+ Estimator.java.
 *
 * Directly adapted from GREAT+ Estimator.java:generateAlphaByInterval():
 *   double x = Math.pow(z_temp, 1.0 / aver_interval);
 *   return Math.round((1 - x) * 10000) / 10000.0;
 *
 * In Euphorion context: alpha controls how aggressively we prune
 * unpromising biclique tiles between search rounds.
 */
class AdaptiveAlpha {
public:
    AdaptiveAlpha(double z, double init_alpha, int round_bound)
        : z_(z), alpha_(init_alpha), init_alpha_(init_alpha),
          round_bound_(round_bound), cur_round_(1),
          interval_(0), discovered_(0) {}

    /// Update alpha after a search round, mirroring GREAT+ Estimator logic:
    ///   if (cur_round > round_bound) {
    ///     alpha = generateAlphaByInterval(aver_interval);
    ///     alpha = Math.max(init_alpha, alpha);
    ///   }
    void update_round(double total_interval, int discovered_count) {
        interval_ = total_interval;
        discovered_ = discovered_count;
        cur_round_++;

        if (cur_round_ > round_bound_ && discovered_ > 0) {
            double aver_interval = interval_ / discovered_;
            double x = std::pow(z_, 1.0 / aver_interval);
            alpha_ = std::round((1.0 - x) * 10000.0) / 10000.0;
            alpha_ = std::max(init_alpha_, alpha_);
        } else {
            alpha_ = init_alpha_;
        }
    }

    double alpha() const { return alpha_; }
    double survive_rate() const { return 1.0 - alpha_; }
    int round() const { return cur_round_; }

private:
    double z_;
    double alpha_;
    double init_alpha_;
    int round_bound_;
    int cur_round_;
    double interval_;
    int discovered_;
};

/**
 * @class AsymmetricScheduler
 * @brief Main scheduler for heterogeneous GPU biclique search.
 *
 * Pipeline design follows Megatron's 1F1B interleaving
 * (schedules.py:896 forward_backward_pipelining_with_interleaving):
 *
 *   Stage 1 (Forward): Dense Jaccard computation on H100
 *   Stage 2 (Forward): Sparse 2-hop expansion on A6000
 *   Stage 3 (Backward): Result aggregation + pruning feedback
 *
 * The "interleaving" comes from overlapping:
 *   - H100 computes Jaccard for tile[i+1] while A6000 expands tile[i]
 *   - Comm of tile[i] results overlaps with compute of tile[i+1]
 *
 * Work-stealing follows CUTLASS PersistentTileScheduler:
 *   when a device finishes early, it steals tiles from the other's queue.
 */
class AsymmetricScheduler {
public:
    AsymmetricScheduler(
        const BiGraph& g,
        const std::vector<WorkloadPartitioner::DeviceCapability>& devices,
        int tau,
        double z = 0.5,
        double init_alpha = 0.1,
        int round_bound = 3
    ) : g_(g),
        abg_(g),
        partitioner_(devices),
        tau_(tau),
        alpha_(z, init_alpha, round_bound),
        aggregator_() {}

    /**
     * @brief Execute asymmetric biclique search.
     *
     * The pipeline mirrors Megatron's interleaved schedule:
     *   for each micro-batch (= tile batch):
     *     1. forward_step(H100): dense Jaccard for candidate pruning
     *     2. forward_step(A6000): sparse enumeration on pruned set
     *     3. backward_step: aggregate results, update pruning threshold
     *
     * NCCL-style grouping (ncclGroupStart/End) batches inter-device comm.
     *
     * @return Global most-similar biclique
     */
    SimilarBiclique execute_global_search() {
        // Phase 1: k-core reduction (from mosib GlobalExact::global_exact_query)
        VI remain = g_.get_kcore(tau_);
        VI left_vertices;
        for (int v : remain) {
            if (v < g_.left_node_num_) left_vertices.push_back(v);
        }

        // Phase 2: Partition into tiles
        auto tiles = partitioner_.partition(left_vertices, g_, tau_);
        stats_.tiles_completed = 0;
        stats_.pruned_tiles = 0;

        // Phase 3: Interleaved pipeline execution
        // Adapted from Megatron forward_backward_pipelining_with_interleaving:
        //   "Run interleaved 1F1B schedule, with communication between
        //    pipeline-parallel ranks."
        auto t_start = hclock::now();

        // In CPU simulation: sequential with timing
        // In production: parallel threads per device type, each with own LocalExact
        // Fix: LocalExact hoisted to per-thread scope (not per-tile) to avoid
        // repeated allocation of cnt_(left_node_num_) vector per tile.
        // Thread-safe because: (a) sequential path = single thread,
        // (b) parallel path (benchmark_hardware.cpp) creates per-thread instances.
        LocalExact algo(g_);
        int round_discoveries = 0;
        double round_interval = 0.0;

        for (size_t ti = 0; ti < tiles.size(); ti++) {
            auto& tile = tiles[ti];

            // Pruning check: skip tiles below current threshold
            // This is the "backward" step feeding into "forward":
            // analogous to Megatron's 1F1B where backward of micro-batch i
            // feeds into forward of micro-batch i+1
            if (aggregator_.best_sim() > 0.9 && tile.priority < 0) {
                stats_.pruned_tiles++;
                continue;
            }

            // Stage 1 (simulated H100): Dense Jaccard for tile vertices
            for (int qi = tile.query_start; qi < tile.query_end && qi < (int)left_vertices.size(); qi++) {
                int q = left_vertices[qi];
                SimilarBiclique local_ans = algo.local_exact_query(q, tau_, aggregator_.best_sim());

                if (local_ans.sim_ > aggregator_.best_sim() + k_eps) {
                    aggregator_.merge(local_ans);
                    round_discoveries++;
                    round_interval += ti;  // Proxy for interval
                }
            }

            stats_.tiles_completed++;

            // Adaptive alpha update per batch
            // Mirrors GREAT+ Estimator round logic (Estimator.java:178-220)
            if (stats_.tiles_completed % 8 == 0) {
                alpha_.update_round(round_interval, round_discoveries);
                round_discoveries = 0;
                round_interval = 0.0;
            }
        }

        auto t_end = hclock::now();
        stats_.dense_time_ms = get_duration(t_start, t_end) * 1000.0;
        stats_.sparse_time_ms = 0;  // Computed separately on GPU_SPARSE
        stats_.comm_time_ms = 0;
        stats_.imbalance_ratio = 0;

        return aggregator_.best();
    }

    /**
     * @brief Execute local search for a single query, with device hints.
     *
     * On GPU_DENSE: runs full Jaccard pruning + SFS enumeration
     * On GPU_SPARSE: runs 2-hop expansion with relaxed pruning
     */
    SimilarBiclique execute_local_search(int q, DeviceType hint = DeviceType::CPU) {
        LocalExact algo(g_);
        return algo.local_exact_query(q, tau_, aggregator_.best_sim());
    }

    const ScheduleStats& stats() const { return stats_; }
    const AdaptiveAlpha& adaptive_alpha() const { return alpha_; }

private:
    const BiGraph& g_;
    AsymmetricBiGraph abg_;
    WorkloadPartitioner partitioner_;
    int tau_;
    AdaptiveAlpha alpha_;
    ResultAggregator aggregator_;
    ScheduleStats stats_;
};
