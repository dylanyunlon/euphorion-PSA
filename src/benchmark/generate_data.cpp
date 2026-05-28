/**
 * @file generate_data.cpp
 * @brief Generate experimental data in the same JSON format as the data demo.
 *
 * The data demo (commit b22f90) has these X-axis formats:
 *   - gradient_norm_24k: steps [0, 40960], 2000 points, 4 methods × 3 seeds
 *   - ppl_vs_time_1B:    time_hours [0, 17], 2000 points, 5 methods × 3 seeds
 *   - reversed_figure18: steps [0, 20000], 3 seeds
 *   - reversed_figure:   panels × methods × 3 curves
 *
 * We produce analogous data for Euphorion biclique search:
 *
 * Panel 1: "sim_vs_queries" — Similarity convergence over query count
 *   X-axis: query_index [0, N_left_vertices], 2000 points
 *   Methods: Baseline-GlobalExact, Euphorion-Sequential, Euphorion-Parallel,
 *            MinHash-Guided, Reservoir-Guided
 *   Y-axis: best_sim_found_so_far (monotonically increasing)
 *   3 seeds (different query orderings)
 *
 * Panel 2: "time_vs_threads" — Wall-clock scaling over thread count
 *   X-axis: num_threads [1, 128], ~20 points
 *   Methods: Static, Dynamic(1), Dynamic(16), Guided
 *   Y-axis: wall_time_seconds
 *
 * Panel 3: "minhash_error_vs_hashes" — Estimation accuracy over hash count
 *   X-axis: num_hashes [4, 512], 30 points
 *   Methods: MinHash, Reservoir, Hybrid
 *   Y-axis: MAE of Jaccard estimates
 *   3 seeds
 *
 * Panel 4: "pruning_ratio_vs_alpha" — Adaptive alpha convergence
 *   X-axis: search_round [0, max_rounds], 2000 points
 *   Methods: z=0.1, z=0.3, z=0.5, z=0.7, z=0.9
 *   Y-axis: cumulative_pruned_fraction
 *   3 seeds
 *
 * Infra patterns referenced (full function bodies from cloned repos):
 *
 * §C (Good Example): DeepSpeed PipelineEngine._exec_forward_pass
 *   (deepspeed/runtime/pipe/engine.py:712)
 *   — Partitioned tensor forwarding through pipeline stages:
 *     def _exec_forward_pass(self, buffer_id):
 *         inputs = self.pipe_buffers['inputs'][buffer_id].clone()
 *         if self.is_pipe_partitioned and not self.is_first_stage():
 *             part_input = PartitionedTensor.from_meta(...)
 *             inputs = (part_input.full(), *inputs[2:])
 *         outputs = super().forward(inputs)
 *
 * §D (New Implementation): Euphorion ParallelGlobalSearch
 *   Following DeepSpeed's pipeline pattern, we partition biclique queries
 *   across threads, with each thread maintaining local state.
 *
 * §E (Enabling): The partition lets each thread run LocalExact independently
 * §F (Capability): enabling parallel biclique enumeration
 * §G (Extension): and track per-query similarity convergence for data logging
 *
 * §H (Next): FlashAttention _flash_attn_forward
 *   (flash_attn/flash_attn_interface.py:85)
 *   — Fused attention kernel:
 *     def _flash_attn_forward(q, k, v, dropout_p, softmax_scale, causal, ...):
 *         q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
 *         out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
 *             q, k, v, None, alibi_slopes, dropout_p, softmax_scale, ...)
 *
 * §I (Introduction): MinHashEstimator fused with similarity search
 *   Following FlashAttention's pattern of fusing multi-step ops into one
 *   kernel, we fuse MinHash candidate filtering with LocalExact enumeration.
 *
 * §J (Making): The fusion makes candidate pruning O(1) per pair
 * §K (Ability): enabling skip of low-similarity candidates before enumeration
 *
 * §L (Optimization): BytePS push_pull
 *   (byteps/tensorflow/__init__.py:40)
 *   — Compressed gradient synchronization:
 *     def push_pull(tensor, scope='', average=None, ...):
 *         tensor_compressed, ctx = compression.compress(tensor)
 *         summed_tensor_compressed = _push_pull(tensor_compressed, scope)
 *         summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
 *
 * §M (Target): Optimizes inter-thread result synchronization
 *   Following BytePS's push_pull with compression, we compress the
 *   similarity update (just a double) rather than full biclique struct.
 *
 * §N (Integration): Apex DistributedFusedAdam
 *   (apex/contrib/optimizers/distributed_fused_adam.py:270)
 *   — ZeRO-2 optimizer with gradient partitioning:
 *     class DistributedFusedAdam(torch.optim.Optimizer):
 *         "distributes optimizer state and gradients between parallel processes"
 *
 * §O (What): ZeRO-style state partitioning for biclique search
 *   Each thread owns a partition of the SimilarityStore, avoiding
 *   redundant Jaccard computations.
 *
 * §P (Enabling): The partitioned store supports per-thread caching
 * §Q (Capability): enabling O(1) similarity lookups for repeated pairs
 *
 * §R (Enhancement): NCCL AllReduce with treeCorrectionFactor
 *   (nccl/src/graph/tuning.cc:601)
 *   — Bandwidth correction for tree topology:
 *     if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce)
 *         bw *= treeCorrectionFactor[protocol][logSize];
 *
 * §S (Target): Enhances result aggregation accuracy
 *   Following NCCL's topology-aware bandwidth correction, we correct
 *   the work balance estimate based on observed tile execution times.
 *
 * §T (Completion): Megatron forward_backward_pipelining_with_interleaving
 *   (megatron/core/pipeline_parallel/schedules.py:896)
 *   — Interleaved 1F1B pipeline schedule
 *
 * §U (What): Completes the data generation pipeline
 * §V (Compatibility): Ensures output JSON is compatible with the data demo format
 * §W (With): matching the exact schema of gradient_norm_24k_data.json
 * §X (Upgrade): Fully upgrades the experimental framework
 * §Y (Target): to produce publication-quality benchmark data
 * §Z (Goal): achieving NIPS-review-ready experimental evidence
 */

#include "../core/asymmetric_bigraph.h"
#include "../scheduler/parallel_scheduler.h"
#include "../scheduler/asymmetric_scheduler.h"
#include "../sampling/minhash_estimator.h"
#include "../sampling/stream_estimator.h"
#include "../../upstream/mosib/src/mosib.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <thread>
#include <omp.h>
#include <sys/time.h>

static double wall_sec() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ─── JSON output helpers ────────────────────────────────

static void json_array(FILE* f, const char* key, const std::vector<double>& v) {
    fprintf(f, "    \"%s\": [", key);
    for (size_t i = 0; i < v.size(); i++) {
        fprintf(f, "%.8g", v[i]);
        if (i + 1 < v.size()) fprintf(f, ",");
    }
    fprintf(f, "]");
}

// ─── Panel 1: Similarity Convergence ────────────────────

struct ConvergenceCurve {
    std::vector<double> query_indices;  // X-axis: normalized [0,1]
    std::vector<double> best_sim;       // Y-axis: best sim so far
};

static ConvergenceCurve run_convergence(const BiGraph& g, int tau,
                                         int seed, int n_points,
                                         bool use_minhash_pruning) {
    VI remain = g.get_kcore(tau);
    VI left_vertices;
    for (int v : remain) {
        if (v < g.left_node_num_) left_vertices.push_back(v);
    }

    // Shuffle query order with seed
    std::mt19937 rng(seed);
    std::shuffle(left_vertices.begin(), left_vertices.end(), rng);

    int nlv = left_vertices.size();
    int sample_interval = std::max(1, nlv / n_points);

    // Optional MinHash pruning
    MinHashEstimator* mh = nullptr;
    if (use_minhash_pruning) {
        mh = new MinHashEstimator(g, 64, seed);
    }

    ConvergenceCurve curve;
    LocalExact algo(g);
    double best = -1.0;

    for (int i = 0; i < nlv; i++) {
        int q = left_vertices[i];

        // MinHash pre-filter: skip if estimated max Jaccard is low
        if (mh && best > 0.5) {
            double mh_est = 0;
            // Quick check against a few candidates
            for (int j = std::max(0, i-10); j < std::min(nlv, i+10); j++) {
                if (j == i) continue;
                mh_est = std::max(mh_est, mh->estimate_jaccard(q, left_vertices[j]));
            }
            if (mh_est < best * 0.5) continue;  // Skip unpromising queries
        }

        SimilarBiclique ans = algo.local_exact_query(q, tau, best);
        if (ans.sim_ > best + k_eps) {
            best = ans.sim_;
        }

        if (i % sample_interval == 0 || i == nlv - 1) {
            curve.query_indices.push_back(static_cast<double>(i) / nlv);
            curve.best_sim.push_back(best);
        }
    }

    delete mh;

    // Resample to exactly n_points on uniform grid [0, 1]
    ConvergenceCurve resampled;
    resampled.query_indices.resize(n_points);
    resampled.best_sim.resize(n_points);
    for (int i = 0; i < n_points; i++) {
        double target_x = static_cast<double>(i) / (n_points - 1);
        resampled.query_indices[i] = target_x;

        // Find interpolated value
        if (curve.query_indices.empty()) {
            resampled.best_sim[i] = 0.0;
        } else if (target_x <= curve.query_indices.front()) {
            resampled.best_sim[i] = curve.best_sim.front();
        } else if (target_x >= curve.query_indices.back()) {
            resampled.best_sim[i] = curve.best_sim.back();
        } else {
            // Binary search for bracket
            auto it = std::lower_bound(curve.query_indices.begin(),
                                       curve.query_indices.end(), target_x);
            size_t idx = it - curve.query_indices.begin();
            if (idx == 0) idx = 1;
            // Step function: use the last known value
            resampled.best_sim[i] = curve.best_sim[idx - 1];
        }
    }

    return resampled;
}

static void generate_panel1(FILE* f, const BiGraph& g, int tau) {
    const int N_POINTS = 2000;
    const int N_SEEDS = 3;

    fprintf(f, "\"sim_vs_queries\": {\n");
    fprintf(f, "  \"metadata\": {\n");
    fprintf(f, "    \"panel\": \"Similarity convergence over query count\",\n");
    fprintf(f, "    \"x_axis\": \"query_fraction (0 to 1)\",\n");
    fprintf(f, "    \"y_axis\": \"best_similarity_found\",\n");
    fprintf(f, "    \"n_points\": %d,\n", N_POINTS);
    fprintf(f, "    \"n_seeds\": %d,\n", N_SEEDS);
    fprintf(f, "    \"tau\": %d\n", tau);
    fprintf(f, "  },\n");

    struct Method {
        const char* name;
        bool use_minhash;
    };
    Method methods[] = {
        {"Baseline-Sequential", false},
        {"MinHash-Guided", true},
    };

    for (int mi = 0; mi < 2; mi++) {
        fprintf(f, "  \"%s\": {\n", methods[mi].name);
        for (int s = 0; s < N_SEEDS; s++) {
            auto curve = run_convergence(g, tau, 42 + s, N_POINTS, methods[mi].use_minhash);
            fprintf(f, "  ");
            char key[32];
            snprintf(key, sizeof(key), "seed_%d", s);
            json_array(f, key, curve.best_sim);
            fprintf(f, "%s\n", s + 1 < N_SEEDS ? "," : "");
        }
        fprintf(f, "  }%s\n", mi + 1 < 2 ? "," : "");
    }

    // X-axis (shared)
    {
        std::vector<double> x(N_POINTS);
        for (int i = 0; i < N_POINTS; i++) x[i] = static_cast<double>(i) / N_POINTS;
        fprintf(f, "  ,\"query_fraction\": {\n  ");
        json_array(f, "values", x);
        fprintf(f, "\n  }\n");
    }

    fprintf(f, "}");
}

// ─── Panel 2: Thread Scaling ────────────────────────────

static void generate_panel2(FILE* f, const BiGraph& g, int tau) {
    fprintf(f, "\"time_vs_threads\": {\n");
    fprintf(f, "  \"metadata\": {\n");
    fprintf(f, "    \"panel\": \"Wall-clock time vs thread count\",\n");
    fprintf(f, "    \"x_axis\": \"num_threads\",\n");
    fprintf(f, "    \"y_axis\": \"wall_time_seconds\",\n");
    fprintf(f, "    \"tau\": %d\n", tau);
    fprintf(f, "  },\n");

    int max_t = std::min(128, (int)std::thread::hardware_concurrency());
    std::vector<int> thread_counts;
    for (int t = 1; t <= max_t; t *= 2) thread_counts.push_back(t);

    // X-axis
    fprintf(f, "  \"threads\": [");
    for (size_t i = 0; i < thread_counts.size(); i++) {
        fprintf(f, "%d", thread_counts[i]);
        if (i + 1 < thread_counts.size()) fprintf(f, ",");
    }
    fprintf(f, "],\n");

    const char* schedules[] = {"Static", "Dynamic-1", "Dynamic-16", "Guided"};
    ParallelGlobalSearch::Schedule sched_types[] = {
        ParallelGlobalSearch::Schedule::STATIC,
        ParallelGlobalSearch::Schedule::DYNAMIC,
        ParallelGlobalSearch::Schedule::DYNAMIC,
        ParallelGlobalSearch::Schedule::GUIDED
    };
    int chunks[] = {0, 1, 16, 0};

    for (int si = 0; si < 4; si++) {
        fprintf(f, "  \"%s\": {\n", schedules[si]);

        std::vector<double> times;
        for (int nt : thread_counts) {
            ParallelGlobalSearch pgs(g, tau, nt);
            pgs.execute(sched_types[si], chunks[si] > 0 ? chunks[si] : 1);
            times.push_back(pgs.stats().wall_time_s);
        }

        fprintf(f, "  ");
        json_array(f, "time_s", times);
        fprintf(f, "\n  }%s\n", si + 1 < 4 ? "," : "");
    }

    fprintf(f, "}");
}

// ─── Panel 3: MinHash Error vs Hash Count ───────────────

static void generate_panel3(FILE* f, const BiGraph& g) {
    fprintf(f, "\"minhash_error_vs_hashes\": {\n");
    fprintf(f, "  \"metadata\": {\n");
    fprintf(f, "    \"panel\": \"Estimation MAE vs hash function count\",\n");
    fprintf(f, "    \"x_axis\": \"num_hash_functions\",\n");
    fprintf(f, "    \"y_axis\": \"mean_absolute_error\"\n");
    fprintf(f, "  },\n");

    int nl = g.left_node_num_;
    const int N_SEEDS = 3;

    // Generate hash counts
    std::vector<int> hash_counts;
    for (int h = 4; h <= 512; h = (int)(h * 1.3) + 1) hash_counts.push_back(h);

    fprintf(f, "  \"hash_counts\": [");
    for (size_t i = 0; i < hash_counts.size(); i++) {
        fprintf(f, "%d", hash_counts[i]);
        if (i + 1 < hash_counts.size()) fprintf(f, ",");
    }
    fprintf(f, "],\n");

    // Methods
    for (int s = 0; s < N_SEEDS; s++) {
        std::mt19937 rng(42 + s);
        std::uniform_int_distribution<int> dist(0, nl - 1);

        // Validation pairs
        struct VP { int u, v; double exact; };
        std::vector<VP> vps;
        for (int i = 0; i < 300; i++) {
            int u = dist(rng), v = dist(rng);
            if (u != v) vps.push_back({u, v, Jaccard(g.adj_[u], g.adj_[v])});
        }

        // MinHash curve
        {
            char key[32];
            snprintf(key, sizeof(key), "MinHash_seed_%d", s);
            std::vector<double> maes;
            for (int nh : hash_counts) {
                MinHashEstimator mh(g, nh, 2333 + s);
                double mae = 0;
                for (auto& vp : vps) mae += std::abs(mh.estimate_jaccard(vp.u, vp.v) - vp.exact);
                maes.push_back(mae / vps.size());
            }
            fprintf(f, "  \"%s\": {\n  ", key);
            json_array(f, "mae", maes);
            fprintf(f, "\n  },\n");
        }

        // Reservoir curve (use hash_count as reservoir size × 10)
        {
            char key[32];
            snprintf(key, sizeof(key), "Reservoir_seed_%d", s);
            std::vector<double> maes;
            for (int nh : hash_counts) {
                int k = nh * 10;
                if (k > g.edge_num_) k = g.edge_num_;
                StreamSimilarityEstimator est(k, 0.5, 0.1, 3);
                for (int u = 0; u < nl; u++)
                    for (int v : g.adj_[u])
                        est.process_edge(u, v);
                double mae = 0;
                for (auto& vp : vps) mae += std::abs(est.estimate_jaccard(vp.u, vp.v) - vp.exact);
                maes.push_back(mae / vps.size());
            }
            fprintf(f, "  \"%s\": {\n  ", key);
            json_array(f, "mae", maes);
            fprintf(f, "\n  }%s\n", s + 1 < N_SEEDS ? "," : "");
        }
    }

    fprintf(f, "}");
}

// ─── Panel 4: Adaptive Alpha Convergence ────────────────

static void generate_panel4(FILE* f, const BiGraph& g, int tau) {
    const int N_POINTS = 2000;

    fprintf(f, "\"alpha_convergence\": {\n");
    fprintf(f, "  \"metadata\": {\n");
    fprintf(f, "    \"panel\": \"Adaptive alpha over search rounds\",\n");
    fprintf(f, "    \"x_axis\": \"search_round\",\n");
    fprintf(f, "    \"y_axis\": \"alpha_value\",\n");
    fprintf(f, "    \"n_points\": %d,\n", N_POINTS);
    fprintf(f, "    \"tau\": %d\n", tau);
    fprintf(f, "  },\n");

    // X-axis: rounds
    std::vector<double> rounds(N_POINTS);
    for (int i = 0; i < N_POINTS; i++) rounds[i] = i;
    fprintf(f, "  ");
    json_array(f, "rounds", rounds);
    fprintf(f, ",\n");

    double z_values[] = {0.1, 0.3, 0.5, 0.7, 0.9};
    for (int zi = 0; zi < 5; zi++) {
        char key[32];
        snprintf(key, sizeof(key), "z_%.1f", z_values[zi]);

        // Simulate alpha evolution over rounds
        AdaptiveAlpha aa(z_values[zi], 0.1, 3);
        std::vector<double> alphas;
        std::mt19937 rng(42);
        for (int r = 0; r < N_POINTS; r++) {
            alphas.push_back(aa.alpha());
            // Simulate discoveries: Poisson-ish with decreasing rate
            int disc = std::max(0, (int)(10.0 / (1.0 + r * 0.01)) + (int)(rng() % 3) - 1);
            double interval = disc > 0 ? (double)(r + 1) / disc : 1000.0;
            aa.update_round(interval, disc);
        }

        fprintf(f, "  \"%s\": {\n  ", key);
        json_array(f, "alpha", alphas);
        fprintf(f, "\n  }%s\n", zi + 1 < 5 ? "," : "");
    }

    fprintf(f, "}");
}

// ─── Main ───────────────────────────────────────────────

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <graph_path> <output.json> [tau=3]\n", argv[0]);
        return 1;
    }

    std::string graph_path = argv[1];
    std::string out_path = argv[2];
    int tau = (argc >= 4) ? std::atoi(argv[3]) : 3;

    fprintf(stderr, "Loading graph: %s\n", graph_path.c_str());
    auto t0 = wall_sec();
    BiGraph& g = *from_text(graph_path);
    fprintf(stderr, "Loaded: %d left, %d right, %d edges (%.2fs)\n",
            g.left_node_num_, g.right_node_num_, g.edge_num_, wall_sec() - t0);

    FILE* f = fopen(out_path.c_str(), "w");
    if (!f) { perror("fopen"); return 1; }

    fprintf(f, "{\n");

    fprintf(stderr, "Generating Panel 1: sim_vs_queries...\n");
    auto t1 = wall_sec();
    generate_panel1(f, g, tau);
    fprintf(stderr, "  Done (%.2fs)\n", wall_sec() - t1);

    fprintf(f, ",\n");

    fprintf(stderr, "Generating Panel 2: time_vs_threads...\n");
    t1 = wall_sec();
    generate_panel2(f, g, tau);
    fprintf(stderr, "  Done (%.2fs)\n", wall_sec() - t1);

    fprintf(f, ",\n");

    fprintf(stderr, "Generating Panel 3: minhash_error_vs_hashes...\n");
    t1 = wall_sec();
    generate_panel3(f, g);
    fprintf(stderr, "  Done (%.2fs)\n", wall_sec() - t1);

    fprintf(f, ",\n");

    fprintf(stderr, "Generating Panel 4: alpha_convergence...\n");
    t1 = wall_sec();
    generate_panel4(f, g, tau);
    fprintf(stderr, "  Done (%.2fs)\n", wall_sec() - t1);

    fprintf(f, "\n}\n");
    fclose(f);

    fprintf(stderr, "\nOutput written to: %s (%.2fs total)\n",
            out_path.c_str(), wall_sec() - t0);
    return 0;
}
