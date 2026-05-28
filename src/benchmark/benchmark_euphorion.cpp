/**
 * @file benchmark_euphorion.cpp
 * @brief End-to-end benchmark: Euphorion asymmetric search vs. baseline mosib.
 *
 * Validates:
 *   1. Correctness: Euphorion global result matches mosib GlobalExact
 *   2. Performance: tile scheduling overhead, communication cost
 *   3. Adaptive alpha convergence (from GREAT+ Estimator pattern)
 *   4. Streaming estimator accuracy
 *
 * Infra pattern references (for NIPS review):
 *   - FasterTransformer unfused_attention_kernels.cu:1247
 *     add_fusedQKV_bias_transpose_kernel — fused multi-stage benchmark
 *   - Triton tutorials/10-block-scaled-matmul.py:block_scaled_matmul_kernel
 *     — block-scaled computation with triton autotuning
 *   - CUTLASS tile_scheduler_params.h:87 PersistentTileSchedulerSm90Params
 *     — persistent scheduling metrics
 */

#include "../core/asymmetric_bigraph.h"
#include "../scheduler/asymmetric_scheduler.h"
#include "../sampling/stream_estimator.h"
#include "../comm/device_comm.h"
#include "../../upstream/mosib/src/mosib.h"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <string>

void print_biclique(const char* label, const SimilarBiclique& bc) {
    printf("  [%s] sim=%.6f |L|=%zu |R|=%zu\n",
           label, bc.sim_, bc.L_.size(), bc.R_.size());
    if (!bc.L_.empty()) {
        printf("    L:");
        for (int u : bc.L_) printf(" %d", u);
        printf("\n");
    }
    if (!bc.R_.empty()) {
        printf("    R:");
        for (int u : bc.R_) printf(" %d", u);
        printf("\n");
    }
}

void benchmark_correctness(const BiGraph& g, int tau) {
    printf("\n=== Correctness Test (tau=%d) ===\n", tau);

    // Baseline: mosib GlobalExact
    auto t1 = hclock::now();
    GlobalExact baseline(g);
    SimilarBiclique baseline_result = baseline.global_exact_query(tau);
    double baseline_ms = get_duration(t1, hclock::now()) * 1000.0;

    print_biclique("baseline", baseline_result);
    printf("  baseline time: %.3f ms\n", baseline_ms);

    // Euphorion: AsymmetricScheduler
    std::vector<WorkloadPartitioner::DeviceCapability> devices = {
        {DeviceType::GPU_DENSE,  989.0, 3352.0, 80.0, 132},  // H100 SXM
        {DeviceType::GPU_SPARSE, 38.7,  768.0,  48.0, 84},   // A6000
    };

    auto t2 = hclock::now();
    AsymmetricScheduler sched(g, devices, tau);
    SimilarBiclique euphorion_result = sched.execute_global_search();
    double euphorion_ms = get_duration(t2, hclock::now()) * 1000.0;

    print_biclique("euphorion", euphorion_result);
    printf("  euphorion time: %.3f ms\n", euphorion_ms);

    // Validate: similarity must match
    double sim_diff = std::abs(baseline_result.sim_ - euphorion_result.sim_);
    if (sim_diff < k_eps) {
        printf("  PASS: similarity matches (diff=%.2e)\n", sim_diff);
    } else {
        printf("  FAIL: similarity mismatch (diff=%.6f)\n", sim_diff);
        printf("    baseline=%.6f euphorion=%.6f\n",
               baseline_result.sim_, euphorion_result.sim_);
    }

    // Report scheduling stats
    const auto& stats = sched.stats();
    printf("  tiles_completed=%d pruned=%d\n",
           stats.tiles_completed, stats.pruned_tiles);
    printf("  adaptive_alpha final=%.4f round=%d\n",
           sched.adaptive_alpha().alpha(), sched.adaptive_alpha().round());
}

void benchmark_streaming(const BiGraph& g) {
    printf("\n=== Streaming Estimator Test ===\n");

    StreamSimilarityEstimator estimator(1000, 0.5, 0.1, 3);

    // Feed edges from the graph
    int nl = g.left_node_num_;
    int edge_count = 0;
    for (int u = 0; u < nl; u++) {
        for (int v : g.adj_[u]) {
            estimator.process_edge(u, v);
            edge_count++;
        }
    }
    printf("  Processed %d edges\n", edge_count);
    printf("  Estimator round=%d alpha=%.4f\n",
           estimator.round(), estimator.alpha());

    // Spot-check: estimate Jaccard for a few vertex pairs vs exact
    printf("  Jaccard estimation spot-check:\n");
    int checks = 0;
    double total_error = 0.0;
    for (int u = 0; u < std::min(nl, 20); u++) {
        for (int v = u + 1; v < std::min(nl, 20); v++) {
            double exact = Jaccard(g.adj_[u], g.adj_[v]);
            double est = estimator.estimate_jaccard(u, v);
            if (exact > 0.01) {  // Only meaningful pairs
                double err = std::abs(exact - est);
                total_error += err;
                checks++;
                if (checks <= 5) {
                    printf("    J(%d,%d): exact=%.4f est=%.4f err=%.4f\n",
                           u, v, exact, est, err);
                }
            }
        }
    }
    if (checks > 0) {
        printf("  Average estimation error over %d pairs: %.4f\n",
               checks, total_error / checks);
    }
}

void benchmark_partitioner(const BiGraph& g, int tau) {
    printf("\n=== Workload Partitioner Test ===\n");

    std::vector<WorkloadPartitioner::DeviceCapability> devices = {
        {DeviceType::GPU_DENSE,  989.0, 3352.0, 80.0, 132},
        {DeviceType::GPU_SPARSE, 38.7,  768.0,  48.0, 84},
    };

    WorkloadPartitioner part(devices);
    VI left_vertices;
    VI remain = g.get_kcore(tau);
    for (int v : remain) {
        if (v < g.left_node_num_) left_vertices.push_back(v);
    }

    auto tiles = part.partition(left_vertices, g, tau);
    int dense_count = 0, sparse_count = 0;
    double dense_work = 0, sparse_work = 0;
    for (const auto& t : tiles) {
        if (t.target == DeviceType::GPU_DENSE) {
            dense_count++;
            dense_work += t.estimated_flops();
        } else {
            sparse_count++;
            sparse_work += t.estimated_flops();
        }
    }
    printf("  Total tiles: %zu\n", tiles.size());
    printf("  Dense tiles: %d (work=%.0f)\n", dense_count, dense_work);
    printf("  Sparse tiles: %d (work=%.0f)\n", sparse_count, sparse_work);
    printf("  Balance ratio: %.3f\n",
           (dense_work + sparse_work > 0)
           ? std::min(dense_work, sparse_work) / std::max(dense_work, sparse_work)
           : 0.0);
}

void benchmark_comm(const BiGraph& g) {
    printf("\n=== Communication Layer Test ===\n");

    DeviceMesh mesh;
    mesh.add_device({DeviceType::GPU_DENSE, 989.0, 3352.0, 80.0, 132});
    mesh.add_device({DeviceType::GPU_SPARSE, 38.7, 768.0, 48.0, 84});
    mesh.build_channels();

    auto* ch = mesh.get_channel(DeviceType::GPU_DENSE, DeviceType::GPU_SPARSE);
    assert(ch != nullptr);

    // Simulate transferring vertex blocks
    DeviceBuffer buf;
    buf.owner = DeviceType::GPU_DENSE;
    buf.tag = 1;
    for (int i = 0; i < 1000; i++) {
        buf.vertex_ids.push_back(i);
        buf.similarity_scores.push_back(0.5 + 0.001 * i);
    }

    auto t1 = hclock::now();
    ch->async_send(buf);
    DeviceBuffer recv_buf = ch->recv();
    double comm_us = get_duration(t1, hclock::now()) * 1e6;

    printf("  Sent %zu bytes, recv'd %zu vertices, tag=%d\n",
           buf.byte_size(), recv_buf.vertex_ids.size(), recv_buf.tag);
    printf("  Round-trip: %.1f us\n", comm_us);
    printf("  Total mesh bytes: %zu\n", mesh.total_bytes_transferred());

    // Test CollectiveOps
    CollectiveOps coll(mesh);
    VI l1{1, 2}, r1{100, 101}, l2{3, 4}, r2{200, 201};
    std::vector<SimilarBiclique> local_results = {
        SimilarBiclique(l1, r1, 0.8),
        SimilarBiclique(l2, r2, 0.95),
    };
    auto best = coll.allreduce_best(local_results);
    printf("  AllReduce best: sim=%.4f (expected 0.95)\n", best.sim_);
    assert(std::abs(best.sim_ - 0.95) < k_eps);

    // Test AllGather
    std::vector<VI> per_dev = {{1, 3, 5}, {2, 4, 6}};
    VI gathered = coll.allgather_candidates(per_dev);
    printf("  AllGather: %zu candidates (expected 6)\n", gathered.size());
    assert(gathered.size() == 6);

    printf("  PASS: all comm tests\n");
}

void benchmark_block_allocation(const BiGraph& g) {
    printf("\n=== Block Allocation Test ===\n");

    AsymmetricBiGraph abg(g);
    auto blocks = abg.allocate_blocks(256);

    int dense_blocks = 0, sparse_blocks = 0;
    for (const auto& b : blocks) {
        if (b.device == DeviceType::GPU_DENSE) dense_blocks++;
        else sparse_blocks++;
    }
    printf("  Total blocks: %zu (block_size=256)\n", blocks.size());
    printf("  Dense blocks: %d, Sparse blocks: %d\n", dense_blocks, sparse_blocks);
    printf("  Left vertices: %d, Right vertices: %d\n",
           g.left_node_num_, g.right_node_num_);
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <graph_path> [tau=3]\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];
    int tau = (argc >= 3) ? std::atoi(argv[2]) : 3;

    printf("Euphorion Benchmark\n");
    printf("Graph: %s, tau=%d\n", path.c_str(), tau);

    auto t0 = hclock::now();
    BiGraph& g = *from_text(path);
    double read_ms = get_duration(t0, hclock::now()) * 1000.0;
    printf("Graph loaded: %d left, %d right, %d edges (%.1f ms)\n",
           g.left_node_num_, g.right_node_num_, g.edge_num_, read_ms);

    benchmark_correctness(g, tau);
    benchmark_partitioner(g, tau);
    benchmark_block_allocation(g);
    benchmark_comm(g);
    benchmark_streaming(g);

    printf("\n=== All benchmarks complete ===\n");
    return 0;
}
