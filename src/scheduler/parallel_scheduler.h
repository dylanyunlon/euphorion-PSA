#pragma once

/**
 * @file parallel_scheduler.h
 * @brief OpenMP-parallel biclique search for multi-core CPUs.
 *
 * This replaces the sequential tile loop in AsymmetricScheduler with
 * proper OpenMP parallelism for the ags1 server (128 threads, 2 NUMA nodes).
 *
 * Design follows these real infra patterns:
 *
 *   - Megatron schedules.py:2039 forward_backward_pipelining_without_interleaving
 *     — simple pipeline for non-interleaved case (our CPU baseline)
 *   - Megatron tensor_parallel/layers.py:770 ColumnParallelLinear
 *     — each thread owns a column partition of the work
 *   - NCCL tuning.cc:276 "nsteps = 2*(nRanks-1)"
 *     — ring allreduce steps, adapted to thread-local → global reduce
 *   - CUTLASS PersistentTileSchedulerSm90Params (tile_scheduler_params.h:87)
 *     — persistent scheduling: threads grab work from shared queue
 *   - PyTorch collective_utils.py:290 "torch.distributed.all_gather"
 *     — final gather of per-thread best results
 *
 * NUMA awareness on ags1:
 *   GPUs on NUMA 1 (CPUs 32-63, 96-127). For GPU-adjacent CPU work,
 *   bind threads to NUMA 1 via: numactl --cpunodebind=1 --membind=1
 */

#include "../core/asymmetric_bigraph.h"
#include "../../upstream/mosib/src/mosib.h"

#include <omp.h>
#include <atomic>
#include <vector>
#include <cstdio>

/// Statistics for parallel execution
struct ParallelStats {
    double wall_time_s;
    double total_thread_time_s;    ///< Sum of per-thread times (= utilization)
    int total_queries;
    int threads_used;
    double best_sim;
    double speedup_vs_sequential;  ///< Filled after sequential baseline runs
    double efficiency_pct;
};

/**
 * @class ParallelGlobalSearch
 * @brief OpenMP-parallel global biclique search.
 *
 * Each thread owns a LocalExact instance (no shared mutable state).
 * The global best is maintained via a lock-free atomic double for
 * pruning, with mutex-protected full biclique update.
 *
 * Three scheduling modes are supported, matching E8 in benchmark_hardware:
 *   - STATIC:  equal partition, no stealing (lowest overhead, worst balance)
 *   - DYNAMIC: work-stealing with configurable chunk size (best balance)
 *   - GUIDED:  decreasing chunk sizes (compromise)
 */
class ParallelGlobalSearch {
public:
    enum class Schedule { STATIC, DYNAMIC, GUIDED };

    ParallelGlobalSearch(const BiGraph& g, int tau, int nthreads = 0)
        : g_(g), tau_(tau),
          nthreads_(nthreads > 0 ? nthreads : omp_get_max_threads()) {}

    /**
     * @brief Execute parallel global search.
     *
     * Phase 1: k-core reduction (sequential, O(n+m))
     * Phase 2: Parallel local-exact queries with dynamic work-stealing
     *
     * Uses lock-free ResultAggregator for pruning hint propagation:
     *   each thread reads best_sim atomically (relaxed) for pruning,
     *   and writes only when it finds something better (rare, mutex).
     *
     * @param schedule  OpenMP scheduling strategy
     * @param chunk     Chunk size for dynamic/guided scheduling
     * @return          Global most-similar biclique
     */
    SimilarBiclique execute(Schedule schedule = Schedule::DYNAMIC, int chunk = 16) {
        // Phase 1: k-core reduction
        VI remain = g_.get_kcore(tau_);
        VI left_vertices;
        for (int v : remain) {
            if (v < g_.left_node_num_) left_vertices.push_back(v);
        }
        int nlv = static_cast<int>(left_vertices.size());

        omp_set_num_threads(nthreads_);

        ResultAggregator aggregator;
        std::atomic<int> queries_done{0};

        auto t_start = omp_get_wtime();

        // Phase 2: Parallel enumeration
        // Each thread: own LocalExact → thread-safe
        // Dynamic schedule → work-stealing for load balance
        switch (schedule) {
        case Schedule::STATIC:
            #pragma omp parallel
            {
                LocalExact algo(g_);
                SimilarBiclique thread_best;

                #pragma omp for schedule(static)
                for (int i = 0; i < nlv; i++) {
                    int q = left_vertices[i];
                    double hint = aggregator.best_sim();
                    SimilarBiclique ans = algo.local_exact_query(q, tau_, hint);
                    if (ans.sim_ > thread_best.sim_ + k_eps) {
                        thread_best = ans;
                        aggregator.merge(ans);
                    }
                    queries_done.fetch_add(1, std::memory_order_relaxed);
                }

                #pragma omp critical
                { aggregator.merge(thread_best); }
            }
            break;

        case Schedule::DYNAMIC:
            #pragma omp parallel
            {
                LocalExact algo(g_);
                SimilarBiclique thread_best;

                #pragma omp for schedule(dynamic, chunk)
                for (int i = 0; i < nlv; i++) {
                    int q = left_vertices[i];
                    double hint = aggregator.best_sim();
                    SimilarBiclique ans = algo.local_exact_query(q, tau_, hint);
                    if (ans.sim_ > thread_best.sim_ + k_eps) {
                        thread_best = ans;
                        aggregator.merge(ans);
                    }
                    queries_done.fetch_add(1, std::memory_order_relaxed);
                }

                #pragma omp critical
                { aggregator.merge(thread_best); }
            }
            break;

        case Schedule::GUIDED:
            #pragma omp parallel
            {
                LocalExact algo(g_);
                SimilarBiclique thread_best;

                #pragma omp for schedule(guided)
                for (int i = 0; i < nlv; i++) {
                    int q = left_vertices[i];
                    double hint = aggregator.best_sim();
                    SimilarBiclique ans = algo.local_exact_query(q, tau_, hint);
                    if (ans.sim_ > thread_best.sim_ + k_eps) {
                        thread_best = ans;
                        aggregator.merge(ans);
                    }
                    queries_done.fetch_add(1, std::memory_order_relaxed);
                }

                #pragma omp critical
                { aggregator.merge(thread_best); }
            }
            break;
        }

        double wall_time = omp_get_wtime() - t_start;

        stats_.wall_time_s = wall_time;
        stats_.total_queries = queries_done.load();
        stats_.threads_used = nthreads_;
        stats_.best_sim = aggregator.best_sim();
        stats_.efficiency_pct = 0;  // Computed externally
        stats_.speedup_vs_sequential = 0;

        return aggregator.best();
    }

    /**
     * @brief Run sequential baseline and compare.
     */
    SimilarBiclique execute_with_comparison() {
        // Sequential baseline
        int saved = nthreads_;
        nthreads_ = 1;
        auto seq_result = execute(Schedule::STATIC, 1);
        double seq_time = stats_.wall_time_s;
        nthreads_ = saved;

        // Parallel run
        auto par_result = execute(Schedule::DYNAMIC, 16);
        double par_time = stats_.wall_time_s;

        stats_.speedup_vs_sequential = seq_time / par_time;
        stats_.efficiency_pct = stats_.speedup_vs_sequential / nthreads_ * 100.0;

        return par_result;
    }

    const ParallelStats& stats() const { return stats_; }

private:
    const BiGraph& g_;
    int tau_;
    int nthreads_;
    ParallelStats stats_;
};
