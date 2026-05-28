/**
 * @file benchmark_engine.cpp
 * @brief Validates SearchEngine against baseline LocalExact + measures speedup.
 */

#include "../core/search_engine.h"
#include "../../upstream/mosib/src/mosib.h"

#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

using hclock = std::chrono::high_resolution_clock;

static double elapsed_ms(hclock::time_point s, hclock::time_point e) {
    return std::chrono::duration<double, std::milli>(e - s).count();
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <graph> [tau=3] [n_queries=100]\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];
    int tau = (argc >= 3) ? std::atoi(argv[2]) : 3;
    int n_queries = (argc >= 4) ? std::atoi(argv[3]) : 100;

    auto t0 = hclock::now();
    BiGraph& g = *from_text(path);
    printf("Graph: %d left, %d right, %d edges (%.1f ms)\n",
           g.left_node_num_, g.right_node_num_, g.edge_num_,
           elapsed_ms(t0, hclock::now()));

    // Prepare query set from k-core
    VI remain = g.get_kcore(tau);
    VI queries;
    for (int v : remain) {
        if (v < g.left_node_num_ && (int)queries.size() < n_queries)
            queries.push_back(v);
    }
    printf("Testing %zu queries with tau=%d\n\n", queries.size(), tau);

    // ── Baseline: original LocalExact ──────────────────────
    printf("%-20s %-12s %-10s\n", "Engine", "Time(ms)", "MaxSim");
    printf("──────────────────── ──────────── ──────────\n");

    double baseline_max_sim = -1.0;
    auto t1 = hclock::now();
    {
        LocalExact algo(g);
        for (int q : queries) {
            SimilarBiclique ans = algo.local_exact_query(q, tau, baseline_max_sim);
            if (ans.sim_ > baseline_max_sim + k_eps)
                baseline_max_sim = ans.sim_;
        }
    }
    double baseline_ms = elapsed_ms(t1, hclock::now());
    printf("%-20s %-12.2f %-10.6f\n", "LocalExact", baseline_ms, baseline_max_sim);

    // ── New: decomposed SearchEngine ──────────────────────
    double engine_max_sim = -1.0;
    auto t2 = hclock::now();
    {
        euphorion::SearchEngine engine(g);
        for (int q : queries) {
            SimilarBiclique ans = engine.search(q, tau, engine_max_sim);
            if (ans.sim_ > engine_max_sim + k_eps)
                engine_max_sim = ans.sim_;
        }
    }
    double engine_ms = elapsed_ms(t2, hclock::now());
    printf("%-20s %-12.2f %-10.6f\n", "SearchEngine", engine_ms, engine_max_sim);

    // ── Correctness check ─────────────────────────────────
    printf("\n");
    double diff = std::abs(baseline_max_sim - engine_max_sim);
    if (diff < k_eps) {
        printf("✓ PASS: results match (diff=%.2e)\n", diff);
    } else {
        printf("✗ FAIL: results differ (baseline=%.6f engine=%.6f diff=%.6f)\n",
               baseline_max_sim, engine_max_sim, diff);
    }

    double speedup = baseline_ms / engine_ms;
    printf("Speedup: %.2fx\n", speedup);

    // ── Per-query comparison ──────────────────────────────
    printf("\nPer-query spot check (first 10):\n");
    printf("%-8s %-12s %-12s %-8s\n", "Query", "Baseline", "Engine", "Match");
    printf("──────── ──────────── ──────────── ────────\n");

    LocalExact baseline_algo(g);
    euphorion::SearchEngine engine_check(g);
    int mismatches = 0;

    for (int i = 0; i < std::min((int)queries.size(), 10); i++) {
        int q = queries[i];
        SimilarBiclique b = baseline_algo.local_exact_query(q, tau);
        SimilarBiclique e = engine_check.search(q, tau);

        bool match = std::abs(b.sim_ - e.sim_) < k_eps;
        if (!match) mismatches++;
        printf("%-8d %-12.6f %-12.6f %-8s\n",
               q, b.sim_, e.sim_, match ? "✓" : "✗");
    }

    printf("\nMismatches: %d / %d\n", mismatches, std::min((int)queries.size(), 10));
    return mismatches > 0 ? 1 : 0;
}
