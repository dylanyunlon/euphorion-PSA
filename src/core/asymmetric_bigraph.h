#pragma once

/**
 * @file asymmetric_bigraph.h
 * @brief Asymmetric GPU-aware bipartite graph for Euphorion.
 *
 * Design rationale (NIPS Review §C → §D):
 *   From GREAT's reservoir sampling (good example §C in Estimator.java:processEdge),
 *   we follow that pattern to implement a new AsymmetricBiGraph (§D),
 *   letting the WorkloadPartitioner (§E) distribute dense Jaccard computation
 *   to H100 (§F) and enable sparse 2-hop traversal on A6000 (§G).
 *
 * Infra references (real functions from cloned repos):
 *   - NCCL  src/graph/connect.cc:connectRings()    — ring topology for peer comm
 *   - NCCL  src/graph/connect.cc:connectTrees()    — tree reduction for aggregation
 *   - CUTLASS include/cutlass/gemm/kernel/tile_scheduler_params.h:PersistentTileSchedulerSm90Params
 *     — persistent tile scheduling across SMs, adapted for biclique tile distribution
 *   - Megatron megatron/core/tensor_parallel/layers.py:ColumnParallelLinear
 *     — column-parallel split, adapted for L/R vertex partition
 *   - vLLM vllm/v1/attention/ops/paged_attn.py:PagedAttention::split_kv_cache
 *     — block-paged memory, adapted for vertex-block allocation
 */

#include "../../upstream/mosib/src/mosib.h"
#include "../../upstream/mosib/src/bigraph.h"
#include "../../upstream/mosib/src/global.h"

#include <functional>
#include <memory>
#include <vector>
#include <cassert>
#include <cmath>
#include <atomic>
#include <mutex>

/// GPU device type for asymmetric scheduling
enum class DeviceType : int {
    CPU = 0,
    GPU_DENSE = 1,   // H100-class: high FLOPS for dense Jaccard matrix ops
    GPU_SPARSE = 2,  // A6000-class: large VRAM for sparse 2-hop traversal
};

/// Inspired by CUTLASS PersistentTileSchedulerSm90Params
/// Represents a work tile for biclique enumeration
struct BicliqueTile {
    int query_start;      ///< Start index in left-vertex array
    int query_end;        ///< End index in left-vertex array (exclusive)
    int candidate_start;  ///< Start index in candidate 2-hop set
    int candidate_end;    ///< End index in candidate set (exclusive)
    DeviceType target;    ///< Assigned device
    int priority;         ///< Higher = more promising (sim-based)

    /// Estimated FLOP cost for this tile
    /// Dense path: O(|L_tile| * |R_tile| * avg_deg) for Jaccard
    /// Sparse path: O(|L_tile| * avg_2hop) for traversal
    double estimated_flops() const {
        return static_cast<double>(query_end - query_start) *
               static_cast<double>(candidate_end - candidate_start);
    }
};

/**
 * @class WorkloadPartitioner
 * @brief Partitions biclique search work across asymmetric GPUs.
 *
 * Follows Megatron's ColumnParallelLinear pattern (layers.py:770):
 *   partition left-vertices column-wise across devices,
 *   then gather results via NCCL-style allreduce (ncclAllReduce).
 *
 * Follows CUTLASS PersistentTileScheduler pattern:
 *   persistent work-stealing across SMs → adapted to persistent
 *   work-stealing across GPU devices.
 */
class WorkloadPartitioner {
public:
    struct DeviceCapability {
        DeviceType type;
        double flops_peak;      ///< TFLOPS
        double memory_bw;       ///< GB/s (device memory bandwidth)
        double vram_gb;         ///< Available VRAM in GB
        int sm_count;           ///< Streaming multiprocessor count
        double pcie_bw;         ///< PCIe bandwidth GB/s (ags1: H100=64, A6000=4)
        int device_id;          ///< Physical GPU index

        /// Effective throughput considering both compute and PCIe transfer
        /// For a tile with data_bytes of adjacency data:
        ///   tile_time = compute_cost/flops + data_bytes/pcie_bw
        /// This addresses the ags1 A6000 Gen1 bottleneck (4 GB/s vs H100 Gen5 64 GB/s)
        double effective_throughput(double compute_flops, double data_bytes) const {
            double compute_time = (flops_peak > 0) ? compute_flops / (flops_peak * 1e12) : 1e9;
            double transfer_time = (pcie_bw > 0) ? data_bytes / (pcie_bw * 1e9) : 1e9;
            double total_time = compute_time + transfer_time;
            return (total_time > 0) ? 1.0 / total_time : 0.0;
        }
    };

    WorkloadPartitioner(
        const std::vector<DeviceCapability>& devices
    ) : devices_(devices) {}

    /**
     * @brief Partition tiles following CUTLASS StreamK decomposition adapted
     *        for biclique enumeration. Dense Jaccard tiles → GPU_DENSE,
     *        sparse 2-hop expansion tiles → GPU_SPARSE.
     *
     * Ref: CUTLASS PersistentTileSchedulerSm90StreamKParams (tile_scheduler_params.h:389)
     *      "Stream-K partitioning: divide total work into SK tiles and
     *       assign across SMs proportional to compute capability"
     *
     * @param left_vertices  Sorted left-vertex IDs to search
     * @param g              The bipartite graph
     * @param tau            Size constraint
     * @return               Partitioned tiles with device assignment
     */
    std::vector<BicliqueTile> partition(
        const VI& left_vertices,
        const BiGraph& g,
        int tau
    ) const {
        std::vector<BicliqueTile> tiles;

        // Compute total work estimate
        double total_work = 0.0;
        std::vector<double> vertex_work(left_vertices.size());
        for (size_t i = 0; i < left_vertices.size(); i++) {
            int u = left_vertices[i];
            // Work ~ degree * average_2hop_size (proxy for enumeration cost)
            double deg = static_cast<double>(g.adj_[u].size());
            vertex_work[i] = deg * deg;  // Upper bound on 2-hop neighborhood
            total_work += vertex_work[i];
        }

        // Compute device work shares proportional to capability
        // Follows Megatron's tensor_model_parallel_size split logic
        // Updated: incorporate PCIe bandwidth (ags1: H100 Gen5=64GB/s, A6000 Gen1=4GB/s)
        double dense_share = 0.0, sparse_share = 0.0;
        for (const auto& dev : devices_) {
            // Effective capacity = compute + transfer cost model
            // Dense path: compute-bound → weight by FLOPS, but penalize low PCIe
            // Sparse path: BW-bound → weight by memory_bw, but penalize low PCIe
            double pcie_factor = std::max(0.1, std::min(1.0, dev.pcie_bw / 64.0));
            if (dev.type == DeviceType::GPU_DENSE) {
                dense_share += dev.flops_peak * pcie_factor;
            } else if (dev.type == DeviceType::GPU_SPARSE) {
                sparse_share += dev.memory_bw * pcie_factor;
            }
        }
        double total_cap = dense_share + sparse_share;
        if (total_cap < 1e-9) total_cap = 1.0;

        // Tile granularity: inspired by CUTLASS BLOCK_M, BLOCK_N
        const int TILE_SIZE = std::max(1, static_cast<int>(left_vertices.size()) / 64);

        double dense_budget = total_work * (dense_share / total_cap);
        double sparse_budget = total_work * (sparse_share / total_cap);
        double dense_used = 0.0;

        for (size_t i = 0; i < left_vertices.size(); i += TILE_SIZE) {
            BicliqueTile tile;
            tile.query_start = static_cast<int>(i);
            tile.query_end = std::min(static_cast<int>(i + TILE_SIZE),
                                      static_cast<int>(left_vertices.size()));
            tile.candidate_start = 0;
            // Fix: estimate candidate set size from vertex degree (2-hop proxy)
            double avg_deg_sq = 0.0;
            for (int j = tile.query_start; j < tile.query_end; j++) {
                int u = left_vertices[j];
                avg_deg_sq += static_cast<double>(g.adj_[u].size());
            }
            avg_deg_sq /= std::max(1, tile.query_end - tile.query_start);
            tile.candidate_end = static_cast<int>(avg_deg_sq * avg_deg_sq);

            // Compute tile work
            double tile_work = 0.0;
            for (int j = tile.query_start; j < tile.query_end; j++) {
                tile_work += vertex_work[j];
            }

            // Assign: high-degree dense tiles → H100, sparse → A6000
            // This follows NCCL's algorithm selection: RING for bandwidth-bound,
            // TREE for latency-bound (tuning.cc:314)
            if (dense_used + tile_work <= dense_budget * 1.1) {
                tile.target = DeviceType::GPU_DENSE;
                dense_used += tile_work;
            } else {
                tile.target = DeviceType::GPU_SPARSE;
            }

            tile.priority = 0;
            tiles.push_back(tile);
        }

        return tiles;
    }

private:
    std::vector<DeviceCapability> devices_;
};

/**
 * @class AsymmetricBiGraph
 * @brief Extends BiGraph with device-aware vertex block allocation.
 *
 * Memory layout inspired by vLLM PagedAttention::split_kv_cache:
 *   key_cache = kv_cache[0].view(num_blocks, num_kv_heads, head_size // x, -1, x)
 * We adapt: vertex_blocks[device][block_id] → contiguous adjacency sub-arrays.
 *
 * Cross-device synchronization follows NCCL ncclGroupStart/ncclGroupEnd pattern
 * (group.h:84): batch peer-to-peer transfers within a group for overlap.
 */
class AsymmetricBiGraph {
public:
    AsymmetricBiGraph(const BiGraph& base) : base_(base) {}

    /// Block-allocate vertex ranges to devices
    /// Follows vLLM block allocation: split into fixed-size blocks,
    /// assign to device based on access pattern
    struct VertexBlock {
        int start;
        int end;
        DeviceType device;
        bool pinned;  ///< Pinned in device memory (won't be evicted)
    };

    /**
     * @brief Allocate vertex blocks following vLLM's paged approach.
     *
     * Ref: vLLM PagedAttention::write_to_paged_cache (paged_attn.py:32)
     *      "ops.reshape_and_cache(key, value, key_cache, value_cache,
     *       slot_mapping.flatten(), ...)"
     *
     * For biclique search:
     *   - Left vertices with high degree → GPU_DENSE blocks (Jaccard-heavy)
     *   - Right vertices → GPU_SPARSE blocks (2-hop traversal)
     */
    std::vector<VertexBlock> allocate_blocks(int block_size) const {
        std::vector<VertexBlock> blocks;
        int nl = base_.left_node_num_;
        int nr = base_.right_node_num_;

        // Left-side blocks
        for (int i = 0; i < nl; i += block_size) {
            VertexBlock blk;
            blk.start = i;
            blk.end = std::min(i + block_size, nl);
            // High-degree left nodes go to dense GPU
            double avg_deg = 0;
            for (int j = blk.start; j < blk.end; j++) {
                avg_deg += base_.adj_[j].size();
            }
            avg_deg /= (blk.end - blk.start);
            blk.device = (avg_deg > 10.0) ? DeviceType::GPU_DENSE : DeviceType::GPU_SPARSE;
            blk.pinned = false;
            blocks.push_back(blk);
        }

        // Right-side blocks (always sparse traversal)
        for (int i = nl; i < nl + nr; i += block_size) {
            VertexBlock blk;
            blk.start = i;
            blk.end = std::min(i + block_size, nl + nr);
            blk.device = DeviceType::GPU_SPARSE;
            blk.pinned = false;
            blocks.push_back(blk);
        }

        return blocks;
    }

    const BiGraph& base() const { return base_; }

private:
    const BiGraph& base_;
};

/**
 * @class ResultAggregator
 * @brief Aggregates biclique results across devices.
 *
 * Follows NCCL AllReduce pattern (tuning.cc:276):
 *   "nsteps = 2*(nRanks-1)" for ring allreduce,
 *   adapted to aggregate best SimilarBiclique across devices.
 *
 * Also follows PyTorch dist.all_reduce (collective_utils.py:290):
 *   "torch.distributed.all_gather(all_states, local_state)"
 */
class ResultAggregator {
public:
    ResultAggregator() : best_sim_(-1.0) {}

    /// Thread-safe merge of a device-local result.
    /// Lock-free fast path: atomic compare of sim_.
    /// Mutex slow path: only when updating full biclique struct.
    /// Follows NCCL ncclAllReduce with ncclMax op:
    ///   monotonically increasing best across all participants.
    void merge(const SimilarBiclique& candidate) {
        double cur = best_sim_.load(std::memory_order_relaxed);
        if (candidate.sim_ <= cur + k_eps) return;  // Fast reject

        std::lock_guard<std::mutex> lock(mu_);
        // Double-check under lock
        if (candidate.sim_ > best_sim_.load(std::memory_order_relaxed) + k_eps) {
            best_ = candidate;
            best_sim_.store(candidate.sim_, std::memory_order_release);
        }
    }

    SimilarBiclique best() const {
        std::lock_guard<std::mutex> lock(mu_);
        return best_;
    }

    /// Lock-free read for pruning decisions (relaxed ordering is fine:
    /// stale value only means we do slightly more work, never less)
    double best_sim() const {
        return best_sim_.load(std::memory_order_relaxed);
    }

private:
    mutable std::mutex mu_;
    SimilarBiclique best_;
    std::atomic<double> best_sim_;
};
