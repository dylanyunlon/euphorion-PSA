#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Euphorion-PSA: Full Hardware Benchmark on ags1
# 
# Target: 2×EPYC 9354 + H100 NVL + 2×A6000
# NUMA1 (GPUs): CPUs 32-63, 96-127
# ═══════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
DATASET="$PROJECT_DIR/upstream/mosib/dataset/bi-github.txt"
LOGDIR="$PROJECT_DIR/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "═══ Euphorion-PSA Benchmark Suite ═══"
echo "Project: $PROJECT_DIR"
echo "Log dir: $LOGDIR"
echo ""

# ── Step 1: Build ──────────────────────────────────────
echo ">>> Building..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(nproc) 2>&1 | tail -5
echo ">>> Build complete"
echo ""

# ── Step 2: Hardware Info ──────────────────────────────
echo ">>> Collecting hardware info..."
{
    echo "=== Hardware Info ==="
    lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
    echo ""
    free -h
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,compute_cap --format=csv 2>/dev/null || echo "no nvidia-smi"
    echo ""
    nvidia-smi topo -m 2>/dev/null || echo "no topo"
    echo ""
    numactl --hardware 2>/dev/null || echo "no numactl"
} > "$LOGDIR/hardware_info.txt" 2>&1
echo ">>> Hardware info saved"
echo ""

# ── Step 3: Run Experiments ───────────────────────────
echo ">>> Running experiments..."

# E1: Correctness (moderate time)
echo "  E1: Correctness..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E1 \
    > "$LOGDIR/E1_correctness.txt" 2>&1
echo "  E1 done: $(tail -3 "$LOGDIR/E1_correctness.txt" | head -1)"

# E2: Thread scaling (can take 5-10 min)
echo "  E2: Thread scaling..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E2 \
    > "$LOGDIR/E2_thread_scaling.txt" 2>&1
echo "  E2 done"

# E2 with NUMA pinning: run on NUMA1 (where GPUs are)
echo "  E2+NUMA1: Thread scaling pinned to NUMA1..."
numactl --cpunodebind=1 --membind=1 \
    "$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E2 \
    > "$LOGDIR/E2_thread_scaling_numa1.txt" 2>&1 || echo "  (numactl failed, skipping)"
echo "  E2+NUMA1 done"

# E3: NUMA-aware
echo "  E3: NUMA awareness..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E3 \
    > "$LOGDIR/E3_numa.txt" 2>&1
echo "  E3 done"

# E4: Memory bandwidth
echo "  E4: Memory bandwidth..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E4 \
    > "$LOGDIR/E4_membw.txt" 2>&1
echo "  E4 done"

# E5: Tile granularity
echo "  E5: Tile granularity..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E5 \
    > "$LOGDIR/E5_tile_granularity.txt" 2>&1
echo "  E5 done"

# E6: Reservoir sweep
echo "  E6: Reservoir sweep..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E6 \
    > "$LOGDIR/E6_reservoir.txt" 2>&1
echo "  E6 done"

# E7: Alpha convergence
echo "  E7: Alpha convergence..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E7 \
    > "$LOGDIR/E7_alpha.txt" 2>&1
echo "  E7 done"

# E8: Work-stealing
echo "  E8: Work-stealing..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E8 \
    > "$LOGDIR/E8_work_stealing.txt" 2>&1
echo "  E8 done"

# E9: MinHash vs Reservoir accuracy
echo "  E9: MinHash vs Reservoir..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E9 \
    > "$LOGDIR/E9_minhash.txt" 2>&1
echo "  E9 done"

# E10: Parallel scheduler scaling
echo "  E10: Parallel scheduler..."
"$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E10 \
    > "$LOGDIR/E10_parallel_scheduler.txt" 2>&1
echo "  E10 done"

# E10 with NUMA1 pinning
echo "  E10+NUMA1: Parallel scheduler pinned to NUMA1..."
numactl --cpunodebind=1 --membind=1 \
    "$BUILD_DIR/benchmark_hardware" "$DATASET" 3 E10 \
    > "$LOGDIR/E10_parallel_scheduler_numa1.txt" 2>&1 || echo "  (numactl failed)"
echo "  E10+NUMA1 done"

# ── Step 4: Also run baseline for comparison ──────────
echo "  Baseline comparison..."
{
    echo "=== Baseline mosib GlobalExact ==="
    for tau in 2 3 4 5; do
        echo "--- tau=$tau ---"
        /usr/bin/time -v "$BUILD_DIR/global_exact" "$DATASET" "$tau" 2>&1
        echo ""
    done
} > "$LOGDIR/baseline_mosib.txt" 2>&1
echo "  Baseline done"

# ── Step 5: Aggregate Results ─────────────────────────
echo ""
echo "═══ Results Summary ═══"
echo "All logs in: $LOGDIR"
echo ""
echo "--- E1: Correctness ---"
grep -E "PASS|FAIL" "$LOGDIR/E1_correctness.txt" || true
echo ""
echo "--- E2: Thread Scaling ---"
grep -E "Threads|──|^  [0-9]" "$LOGDIR/E2_thread_scaling.txt" || true
echo ""
echo "--- E4: Memory Bandwidth ---"
grep -E "Sequential|Parallel|Jaccard" "$LOGDIR/E4_membw.txt" || true
echo ""
echo "--- E6: Reservoir Sweep ---"
grep -E "Reservoir|──|^  [0-9]" "$LOGDIR/E6_reservoir.txt" || true
echo ""
echo "--- E8: Work-Stealing ---"
grep -E "Schedule|──|static|dynamic|guided" "$LOGDIR/E8_work_stealing.txt" || true
echo ""
echo "═══ Complete ═══"
