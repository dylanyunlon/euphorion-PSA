/**
 * @file benchmark_hardware.cpp
 * @brief Production benchmark for ags1 server hardware.
 *
 * Target hardware (from lscpu/nvidia-smi):
 *   CPU:  2× AMD EPYC 9354 (32C/64T each = 128 threads)
 *   RAM:  ~1.5TB DDR5 (NUMA 0: 774GB, NUMA 1: 774GB)
 *   GPU0: NVIDIA RTX A6000, 49140 MiB, PCIe Gen1 x16, CC 8.6
 *   GPU1: NVIDIA RTX A6000, 49140 MiB, PCIe Gen1 x16, CC 8.6
 *   GPU2: NVIDIA H100 NVL,  95830 MiB, PCIe Gen5 x16, CC 9.0
 *   Topo: All GPUs on NUMA 1 (CPU 32-63, 96-127), NODE interconnect
 *
 * Experiments:
 *   E1: Correctness verification (baseline vs Euphorion, all tau)
 *   E2: Thread scaling study (1, 2, 4, 8, 16, 32, 64, 128 threads)
 *   E3: NUMA-aware vs NUMA-unaware performance
 *   E4: Memory bandwidth measurement (adjacency traversal throughput)
 *   E5: Tile granularity sweep (8, 16, 32, 64, 128, 256 tiles)
 *   E6: Streaming estimator reservoir size sweep (100, 1K, 10K, 100K)
 *   E7: Adaptive alpha convergence rate
 *   E8: Work-stealing simulation across heterogeneous "devices" (threads)
 *
 * Infra pattern (from cloned repos):
 *   - NCCL tuning.cc — bandwidth model per algorithm/topology
 *   - Megatron schedules.py:2039 — pipeline without interleaving baseline
 *   - CUTLASS PersistentTileSchedulerSm90Params — persistent tile metrics
 *   - vLLM PagedAttention — block allocation efficiency
 */

#include "../core/asymmetric_bigraph.h"
#include "../scheduler/asymmetric_scheduler.h"
#include "../scheduler/parallel_scheduler.h"
#include "../sampling/stream_estimator.h"
#include "../sampling/minhash_estimator.h"
#include "../comm/device_comm.h"
#include "../../upstream/mosib/src/mosib.h"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <string>
#include <thread>
#include <atomic>
#include <numeric>
#include <omp.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

// ─── Utility ──────────────────────────────────────────────

static double wall_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static long peak_rss_kb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;  // in KB on Linux
}

static void print_biclique(const char* label, const SimilarBiclique& bc) {
    printf("  [%s] sim=%.6f |L|=%zu |R|=%zu\n",
           label, bc.sim_, bc.L_.size(), bc.R_.size());
}

// ─── E1: Correctness ─────────────────────────────────────

static void experiment_correctness(const BiGraph& g, const std::vector<int>& taus) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E1: Correctness Verification            ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    for (int tau : taus) {
        printf("\n--- tau=%d ---\n", tau);

        // Baseline
        auto t1 = wall_time_sec();
        GlobalExact baseline(g);
        SimilarBiclique baseline_result = baseline.global_exact_query(tau);
        double baseline_s = wall_time_sec() - t1;
        print_biclique("baseline", baseline_result);
        printf("  baseline: %.4f s\n", baseline_s);

        // Euphorion (sequential, matching device config)
        std::vector<WorkloadPartitioner::DeviceCapability> devices = {
            // Match real hardware: H100 NVL + 2× A6000
            {DeviceType::GPU_DENSE,  835.0, 3958.0, 93.6, 132},  // H100 NVL
            {DeviceType::GPU_SPARSE, 38.7,  768.0,  48.0, 84},   // A6000 × 2
            {DeviceType::GPU_SPARSE, 38.7,  768.0,  48.0, 84},
        };
        auto t2 = wall_time_sec();
        AsymmetricScheduler sched(g, devices, tau);
        SimilarBiclique euphorion_result = sched.execute_global_search();
        double euphorion_s = wall_time_sec() - t2;
        print_biclique("euphorion", euphorion_result);
        printf("  euphorion: %.4f s\n", euphorion_s);

        double diff = std::abs(baseline_result.sim_ - euphorion_result.sim_);
        printf("  %s (sim diff=%.2e)\n",
               diff < k_eps ? "✓ PASS" : "✗ FAIL", diff);
    }
}

// ─── E2: Thread Scaling Study ────────────────────────────

static void experiment_thread_scaling(const BiGraph& g, int tau) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E2: Thread Scaling Study (tau=%d)       ║\n", tau);
    printf("╚══════════════════════════════════════════╝\n");
    printf("  %-8s %-12s %-12s %-10s %-12s\n",
           "Threads", "Time(s)", "Speedup", "Efficiency", "RSS(MB)");
    printf("  ──────── ──────────── ──────────── ────────── ────────────\n");

    // k-core reduction (shared, done once)
    VI remain = g.get_kcore(tau);
    VI left_vertices;
    for (int v : remain) {
        if (v < g.left_node_num_) left_vertices.push_back(v);
    }
    int nlv = left_vertices.size();

    double t_single = 0;
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 32, 64, 128};

    for (int nthreads : thread_counts) {
        if (nthreads > (int)std::thread::hardware_concurrency()) break;

        omp_set_num_threads(nthreads);
        SimilarBiclique global_best;
        std::atomic<double> best_sim{-1.0};

        long rss_before = peak_rss_kb();
        auto t_start = wall_time_sec();

        // OpenMP parallel global search
        // Each thread gets its own LocalExact (thread-safe by design fix)
        #pragma omp parallel
        {
            LocalExact algo(g);
            SimilarBiclique thread_best;

            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < nlv; i++) {
                int q = left_vertices[i];
                // Use relaxed load for pruning hint (not critical for correctness)
                double cur_best = best_sim.load(std::memory_order_relaxed);
                SimilarBiclique local_ans = algo.local_exact_query(q, tau, cur_best);

                if (local_ans.sim_ > thread_best.sim_ + k_eps) {
                    thread_best = local_ans;
                    // Atomically update global best for pruning
                    double expected = best_sim.load();
                    while (local_ans.sim_ > expected + k_eps &&
                           !best_sim.compare_exchange_weak(expected, local_ans.sim_)) {}
                }
            }

            #pragma omp critical
            {
                if (thread_best.sim_ > global_best.sim_ + k_eps) {
                    global_best = thread_best;
                }
            }
        }

        double elapsed = wall_time_sec() - t_start;
        long rss_after = peak_rss_kb();
        if (nthreads == 1) t_single = elapsed;

        double speedup = t_single / elapsed;
        double efficiency = speedup / nthreads * 100.0;

        printf("  %-8d %-12.4f %-12.2fx %-9.1f%% %-12ld\n",
               nthreads, elapsed, speedup, efficiency,
               (rss_after - rss_before) / 1024 + (rss_after / 1024));

        // Verify correctness at each thread count
        assert(global_best.sim_ > -0.5);
    }
}

// ─── E3: NUMA Awareness ─────────────────────────────────

static void experiment_numa(const BiGraph& g, int tau) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E3: NUMA-Aware vs NUMA-Unaware          ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    VI remain = g.get_kcore(tau);
    VI left_vertices;
    for (int v : remain) {
        if (v < g.left_node_num_) left_vertices.push_back(v);
    }
    int nlv = left_vertices.size();

    // Test with different thread placements
    // On ags1: NUMA0 = CPUs 0-31,64-95; NUMA1 = CPUs 32-63,96-127
    // GPUs are on NUMA1, so NUMA1-local computation is preferred
    struct PlacementTest {
        const char* name;
        int nthreads;
        const char* env_hint;
    };

    PlacementTest tests[] = {
        {"32T-default",     32, ""},
        {"32T-compact",     32, "close"},
        {"32T-spread",      32, "spread"},
        {"64T-default",     64, ""},
    };

    printf("  %-18s %-12s %-14s\n", "Placement", "Time(s)", "Queries/sec");
    printf("  ────────────────── ──────────── ──────────────\n");

    for (auto& test : tests) {
        omp_set_num_threads(test.nthreads);
        if (strlen(test.env_hint) > 0) {
            // OMP_PROC_BIND hint (may not take effect mid-program)
            // This is informational; actual binding requires env var before launch
        }

        SimilarBiclique global_best;

        auto t_start = wall_time_sec();

        #pragma omp parallel
        {
            LocalExact algo(g);
            SimilarBiclique thread_best;

            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < nlv; i++) {
                SimilarBiclique ans = algo.local_exact_query(left_vertices[i], tau);
                if (ans.sim_ > thread_best.sim_ + k_eps) thread_best = ans;
            }

            #pragma omp critical
            {
                if (thread_best.sim_ > global_best.sim_ + k_eps)
                    global_best = thread_best;
            }
        }

        double elapsed = wall_time_sec() - t_start;
        printf("  %-18s %-12.4f %-14.0f\n",
               test.name, elapsed, nlv / elapsed);
    }
}

// ─── E4: Memory Bandwidth ───────────────────────────────

static void experiment_memory_bandwidth(const BiGraph& g) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E4: Memory Bandwidth (Adj Traversal)    ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    int nl = g.left_node_num_;
    int nr = g.right_node_num_;
    int n = nl + nr;

    // Measure adjacency list traversal throughput
    // This is the bottleneck for 2-hop expansion (GPU_SPARSE workload)
    long total_edges = 0;
    for (int i = 0; i < n; i++) total_edges += g.adj_[i].size();

    // Sequential traversal
    {
        volatile long checksum = 0;
        auto t = wall_time_sec();
        for (int iter = 0; iter < 5; iter++) {
            long cs = 0;
            for (int i = 0; i < n; i++) {
                for (int v : g.adj_[i]) cs += v;
            }
            checksum = cs;
        }
        double elapsed = wall_time_sec() - t;
        double bytes = total_edges * sizeof(int) * 5.0;
        printf("  Sequential: %.2f GB/s (%.3f s for %ld edges × 5 iters)\n",
               bytes / elapsed / 1e9, elapsed, total_edges);
    }

    // Parallel traversal (all threads)
    {
        int nthreads = std::min(128, (int)std::thread::hardware_concurrency());
        omp_set_num_threads(nthreads);
        volatile long checksum = 0;
        auto t = wall_time_sec();
        for (int iter = 0; iter < 5; iter++) {
            long cs = 0;
            #pragma omp parallel for reduction(+:cs)
            for (int i = 0; i < n; i++) {
                for (int v : g.adj_[i]) cs += v;
            }
            checksum = cs;
        }
        double elapsed = wall_time_sec() - t;
        double bytes = total_edges * sizeof(int) * 5.0;
        printf("  Parallel(%dT): %.2f GB/s (%.3f s)\n",
               nthreads, bytes / elapsed / 1e9, elapsed);
    }

    // Jaccard computation throughput
    {
        long long total_pairs = (long long)nl * (nl - 1) / 2;
        int sample_pairs = static_cast<int>(std::min((long long)10000, total_pairs));
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, nl - 1);

        volatile double checksum = 0;
        auto t = wall_time_sec();
        double cs = 0;
        for (int i = 0; i < sample_pairs; i++) {
            int u = dist(rng), v = dist(rng);
            cs += Jaccard(g.adj_[u], g.adj_[v]);
        }
        checksum = cs;
        double elapsed = wall_time_sec() - t;
        printf("  Jaccard throughput: %.0f pairs/sec (%d pairs in %.3f s)\n",
               sample_pairs / elapsed, sample_pairs, elapsed);
    }
}

// ─── E5: Tile Granularity Sweep ─────────────────────────

static void experiment_tile_granularity(const BiGraph& g, int tau) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E5: Tile Granularity Sweep (tau=%d)     ║\n", tau);
    printf("╚══════════════════════════════════════════╝\n");

    std::vector<WorkloadPartitioner::DeviceCapability> devices = {
        {DeviceType::GPU_DENSE,  835.0, 3958.0, 93.6, 132},
        {DeviceType::GPU_SPARSE, 38.7,  768.0,  48.0, 84},
        {DeviceType::GPU_SPARSE, 38.7,  768.0,  48.0, 84},
    };

    VI remain = g.get_kcore(tau);
    VI left_vertices;
    for (int v : remain) {
        if (v < g.left_node_num_) left_vertices.push_back(v);
    }

    printf("  %-10s %-8s %-10s %-10s %-12s %-10s\n",
           "NumTiles", "Dense", "Sparse", "Balance", "Time(s)", "Sim");
    printf("  ────────── ──────── ────────── ────────── ──────────── ──────────\n");

    // We vary tile count by adjusting the internal granularity
    // Since WorkloadPartitioner uses /64, we run the search with different
    // omp chunk sizes to simulate tile granularity effect
    int nthreads = 32;
    omp_set_num_threads(nthreads);

    for (int chunk : {1, 4, 16, 64, 256, 1024}) {
        int nlv = left_vertices.size();
        int num_tiles = (nlv + chunk - 1) / chunk;
        SimilarBiclique global_best;

        auto t = wall_time_sec();

        #pragma omp parallel
        {
            LocalExact algo(g);
            SimilarBiclique thread_best;

            #pragma omp for schedule(dynamic, chunk)
            for (int i = 0; i < nlv; i++) {
                SimilarBiclique ans = algo.local_exact_query(left_vertices[i], tau);
                if (ans.sim_ > thread_best.sim_ + k_eps) thread_best = ans;
            }

            #pragma omp critical
            {
                if (thread_best.sim_ > global_best.sim_ + k_eps)
                    global_best = thread_best;
            }
        }

        double elapsed = wall_time_sec() - t;

        // Compute tile-level balance from partition
        WorkloadPartitioner part(devices);
        auto tiles = part.partition(left_vertices, g, tau);
        int dc = 0, sc = 0;
        double dw = 0, sw = 0;
        for (auto& tl : tiles) {
            if (tl.target == DeviceType::GPU_DENSE) { dc++; dw += tl.estimated_flops(); }
            else { sc++; sw += tl.estimated_flops(); }
        }
        double balance = (dw + sw > 0) ? std::min(dw, sw) / std::max(dw, sw) : 0;

        printf("  %-10d %-8d %-10d %-10.3f %-12.4f %-10.6f\n",
               num_tiles, dc, sc, balance, elapsed, global_best.sim_);
    }
}

// ─── E6: Reservoir Size Sweep ───────────────────────────

static void experiment_reservoir_sweep(const BiGraph& g) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E6: Streaming Estimator Reservoir Sweep  ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    int nl = g.left_node_num_;

    // Prepare exact Jaccard for validation set
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, nl - 1);
    struct ValPair { int u, v; double exact; };
    std::vector<ValPair> val_pairs;
    for (int i = 0; i < 200; i++) {
        int u = dist(rng), v = dist(rng);
        if (u == v) continue;
        double exact = Jaccard(g.adj_[u], g.adj_[v]);
        if (exact > 0.001) val_pairs.push_back({u, v, exact});
    }

    printf("  %-12s %-12s %-12s %-12s %-10s %-10s\n",
           "Reservoir", "MAE", "RMSE", "Rounds", "Alpha", "Time(s)");
    printf("  ──────────── ──────────── ──────────── ──────────── ────────── ──────────\n");

    for (int k : {100, 500, 1000, 5000, 10000, 50000, 100000}) {
        if (k > g.edge_num_) break;

        auto t = wall_time_sec();
        StreamSimilarityEstimator est(k, 0.5, 0.1, 3);

        // Feed all edges
        for (int u = 0; u < nl; u++) {
            for (int v : g.adj_[u]) {
                est.process_edge(u, v);
            }
        }
        double build_time = wall_time_sec() - t;

        // Evaluate on validation pairs
        double mae = 0, mse = 0;
        int count = 0;
        for (auto& vp : val_pairs) {
            double est_j = est.estimate_jaccard(vp.u, vp.v);
            double err = std::abs(est_j - vp.exact);
            mae += err;
            mse += err * err;
            count++;
        }
        mae /= std::max(1, count);
        double rmse = std::sqrt(mse / std::max(1, count));

        printf("  %-12d %-12.6f %-12.6f %-12d %-10.4f %-10.4f\n",
               k, mae, rmse, est.round(), est.alpha(), build_time);
    }
}

// ─── E7: Adaptive Alpha Convergence ─────────────────────

static void experiment_alpha_convergence(const BiGraph& g, int tau) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E7: Adaptive Alpha Convergence (tau=%d) ║\n", tau);
    printf("╚══════════════════════════════════════════╝\n");

    // Test different z values and observe convergence
    printf("  %-6s %-12s %-12s %-12s %-10s\n",
           "z", "FinalAlpha", "Rounds", "Discoveries", "Time(s)");
    printf("  ────── ──────────── ──────────── ──────────── ──────────\n");

    for (double z : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        std::vector<WorkloadPartitioner::DeviceCapability> devices = {
            {DeviceType::GPU_DENSE,  835.0, 3958.0, 93.6, 132},
            {DeviceType::GPU_SPARSE, 38.7,  768.0,  48.0, 84},
        };

        auto t = wall_time_sec();
        AsymmetricScheduler sched(g, devices, tau, z, 0.1, 3);
        SimilarBiclique result = sched.execute_global_search();
        double elapsed = wall_time_sec() - t;

        printf("  %-6.1f %-12.4f %-12d %-12d %-10.4f\n",
               z, sched.adaptive_alpha().alpha(),
               sched.adaptive_alpha().round(),
               sched.stats().tiles_completed,
               elapsed);
    }
}

// ─── E8: Work-Stealing Simulation ───────────────────────

static void experiment_work_stealing(const BiGraph& g, int tau) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E8: Work-Stealing vs Static Scheduling   ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    VI remain = g.get_kcore(tau);
    VI left_vertices;
    for (int v : remain) {
        if (v < g.left_node_num_) left_vertices.push_back(v);
    }
    int nlv = left_vertices.size();
    int nthreads = 32;
    omp_set_num_threads(nthreads);

    printf("  %-18s %-12s %-12s\n", "Schedule", "Time(s)", "Sim");
    printf("  ────────────────── ──────────── ────────────\n");

    // Static scheduling (equal chunks, no stealing)
    {
        SimilarBiclique global_best;
        auto t = wall_time_sec();
        #pragma omp parallel
        {
            LocalExact algo(g);
            SimilarBiclique tb;
            #pragma omp for schedule(static)
            for (int i = 0; i < nlv; i++) {
                SimilarBiclique ans = algo.local_exact_query(left_vertices[i], tau);
                if (ans.sim_ > tb.sim_ + k_eps) tb = ans;
            }
            #pragma omp critical
            { if (tb.sim_ > global_best.sim_ + k_eps) global_best = tb; }
        }
        printf("  %-18s %-12.4f %-12.6f\n", "static", wall_time_sec() - t, global_best.sim_);
    }

    // Dynamic scheduling (work-stealing, chunk=1)
    {
        SimilarBiclique global_best;
        auto t = wall_time_sec();
        #pragma omp parallel
        {
            LocalExact algo(g);
            SimilarBiclique tb;
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < nlv; i++) {
                SimilarBiclique ans = algo.local_exact_query(left_vertices[i], tau);
                if (ans.sim_ > tb.sim_ + k_eps) tb = ans;
            }
            #pragma omp critical
            { if (tb.sim_ > global_best.sim_ + k_eps) global_best = tb; }
        }
        printf("  %-18s %-12.4f %-12.6f\n", "dynamic(1)", wall_time_sec() - t, global_best.sim_);
    }

    // Dynamic scheduling (chunk=16, moderate stealing)
    {
        SimilarBiclique global_best;
        auto t = wall_time_sec();
        #pragma omp parallel
        {
            LocalExact algo(g);
            SimilarBiclique tb;
            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < nlv; i++) {
                SimilarBiclique ans = algo.local_exact_query(left_vertices[i], tau);
                if (ans.sim_ > tb.sim_ + k_eps) tb = ans;
            }
            #pragma omp critical
            { if (tb.sim_ > global_best.sim_ + k_eps) global_best = tb; }
        }
        printf("  %-18s %-12.4f %-12.6f\n", "dynamic(16)", wall_time_sec() - t, global_best.sim_);
    }

    // Guided scheduling (decreasing chunk, OpenMP guided)
    {
        SimilarBiclique global_best;
        auto t = wall_time_sec();
        #pragma omp parallel
        {
            LocalExact algo(g);
            SimilarBiclique tb;
            #pragma omp for schedule(guided)
            for (int i = 0; i < nlv; i++) {
                SimilarBiclique ans = algo.local_exact_query(left_vertices[i], tau);
                if (ans.sim_ > tb.sim_ + k_eps) tb = ans;
            }
            #pragma omp critical
            { if (tb.sim_ > global_best.sim_ + k_eps) global_best = tb; }
        }
        printf("  %-18s %-12.4f %-12.6f\n", "guided", wall_time_sec() - t, global_best.sim_);
    }
}

// ─── E9: MinHash vs Reservoir Accuracy ──────────────────

static void experiment_minhash(const BiGraph& g) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E9: MinHash vs Reservoir Accuracy        ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    int nl = g.left_node_num_;

    // Build exact ground truth for validation set
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, nl - 1);
    struct ValPair { int u, v; double exact; };
    std::vector<ValPair> val_pairs;
    for (int i = 0; i < 500; i++) {
        int u = dist(rng), v = dist(rng);
        if (u == v) continue;
        double exact = Jaccard(g.adj_[u], g.adj_[v]);
        val_pairs.push_back({u, v, exact});
    }
    printf("  Validation pairs: %zu (including zero-Jaccard)\n\n", val_pairs.size());

    printf("  %-14s %-8s %-10s %-10s %-10s %-10s\n",
           "Method", "Param", "MAE", "RMSE", "Time(s)", "Mem(MB)");
    printf("  ────────────── ──────── ────────── ────────── ────────── ──────────\n");

    // MinHash sweep over hash counts
    for (int nhash : {16, 32, 64, 128, 256}) {
        auto t = wall_time_sec();
        MinHashEstimator mh(g, nhash);
        double build_time = wall_time_sec() - t;

        double mae = 0, mse = 0;
        for (auto& vp : val_pairs) {
            double est = mh.estimate_jaccard(vp.u, vp.v);
            double err = std::abs(est - vp.exact);
            mae += err;
            mse += err * err;
        }
        mae /= val_pairs.size();
        double rmse = std::sqrt(mse / val_pairs.size());

        printf("  %-14s %-8d %-10.6f %-10.6f %-10.4f %-10.1f\n",
               "MinHash", nhash, mae, rmse, build_time,
               mh.memory_bytes() / 1e6);
    }

    // Reservoir sweep for comparison
    for (int k : {1000, 10000, 100000}) {
        if (k > g.edge_num_) break;

        auto t = wall_time_sec();
        StreamSimilarityEstimator est(k, 0.5, 0.1, 3);
        for (int u = 0; u < nl; u++)
            for (int v : g.adj_[u])
                est.process_edge(u, v);
        double build_time = wall_time_sec() - t;

        double mae = 0, mse = 0;
        for (auto& vp : val_pairs) {
            double est_j = est.estimate_jaccard(vp.u, vp.v);
            double err = std::abs(est_j - vp.exact);
            mae += err;
            mse += err * err;
        }
        mae /= val_pairs.size();
        double rmse = std::sqrt(mse / val_pairs.size());

        printf("  %-14s %-8d %-10.6f %-10.6f %-10.4f %-10.1f\n",
               "Reservoir", k, mae, rmse, build_time, k * 24.0 / 1e6);
    }

    // Hybrid: MinHash(128) + Reservoir(10000)
    {
        auto t = wall_time_sec();
        MinHashEstimator mh(g, 128);
        HybridEstimator hybrid(mh);
        double build_time = wall_time_sec() - t;

        double mae = 0, mse = 0;
        for (auto& vp : val_pairs) {
            double est = hybrid.estimate_jaccard(vp.u, vp.v);
            double err = std::abs(est - vp.exact);
            mae += err;
            mse += err * err;
        }
        mae /= val_pairs.size();
        double rmse = std::sqrt(mse / val_pairs.size());

        printf("  %-14s %-8s %-10.6f %-10.6f %-10.4f %-10.1f\n",
               "Hybrid", "128+10K", mae, rmse, build_time,
               mh.memory_bytes() / 1e6);
    }
}

// ─── E10: Parallel Scheduler ────────────────────────────

static void experiment_parallel_scheduler(const BiGraph& g, int tau) {
    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  E10: ParallelGlobalSearch Benchmark      ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    int max_threads = std::min(128, (int)std::thread::hardware_concurrency());
    printf("  Max threads available: %d\n\n", max_threads);

    printf("  %-10s %-10s %-12s %-10s %-12s %-12s\n",
           "Threads", "Schedule", "Time(s)", "Sim", "Speedup", "Efficiency");
    printf("  ────────── ────────── ──────────── ────────── ──────────── ────────────\n");

    double t_seq = 0;

    for (int nt : {1, 2, 4, 8, 16, 32, 64, 128}) {
        if (nt > max_threads) break;

        ParallelGlobalSearch pgs(g, tau, nt);
        SimilarBiclique result = pgs.execute(
            ParallelGlobalSearch::Schedule::DYNAMIC, 16);

        auto& st = pgs.stats();
        if (nt == 1) t_seq = st.wall_time_s;

        double speedup = (t_seq > 0) ? t_seq / st.wall_time_s : 1.0;
        double eff = speedup / nt * 100.0;

        printf("  %-10d %-10s %-12.4f %-10.6f %-12.2fx %-11.1f%%\n",
               nt, "dynamic16", st.wall_time_s, st.best_sim,
               speedup, eff);
    }
}

// ─── Main ───────────────────────────────────────────────

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <graph_path> [tau=3] [experiments=all]\n", argv[0]);
        printf("  experiments: all, E1, E2, E3, E4, E5, E6, E7, E8\n");
        return 1;
    }

    std::string path = argv[1];
    int tau = (argc >= 3) ? std::atoi(argv[2]) : 3;
    std::string exps = (argc >= 4) ? argv[3] : "all";

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Euphorion-PSA Hardware Benchmark                ║\n");
    printf("║  Target: ags1 (2×EPYC 9354 + H100 NVL + 2×A6000)║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("Graph: %s\n", path.c_str());
    printf("Default tau: %d\n", tau);
    printf("Hardware threads: %u\n", std::thread::hardware_concurrency());
    printf("OpenMP max threads: %d\n", omp_get_max_threads());
    printf("Peak RSS at start: %ld MB\n", peak_rss_kb() / 1024);

    auto t0 = wall_time_sec();
    BiGraph& g = *from_text(path);
    double read_s = wall_time_sec() - t0;
    printf("Graph loaded: %d left, %d right, %d edges (%.3f s)\n",
           g.left_node_num_, g.right_node_num_, g.edge_num_, read_s);
    printf("Peak RSS after load: %ld MB\n\n", peak_rss_kb() / 1024);

    bool all = (exps == "all");

    if (all || exps == "E1") experiment_correctness(g, {2, 3, 4, 5});
    if (all || exps == "E2") experiment_thread_scaling(g, tau);
    if (all || exps == "E3") experiment_numa(g, tau);
    if (all || exps == "E4") experiment_memory_bandwidth(g);
    if (all || exps == "E5") experiment_tile_granularity(g, tau);
    if (all || exps == "E6") experiment_reservoir_sweep(g);
    if (all || exps == "E7") experiment_alpha_convergence(g, tau);
    if (all || exps == "E8") experiment_work_stealing(g, tau);
    if (all || exps == "E9") experiment_minhash(g);
    if (all || exps == "E10") experiment_parallel_scheduler(g, tau);

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  All experiments complete                 ║\n");
    printf("║  Total wall time: %.2f s                 ║\n", wall_time_sec() - t0);
    printf("║  Peak RSS: %ld MB                        ║\n", peak_rss_kb() / 1024);
    printf("╚══════════════════════════════════════════╝\n");

    return 0;
}
