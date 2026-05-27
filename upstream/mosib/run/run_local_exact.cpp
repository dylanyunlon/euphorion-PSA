#include "mosib.h"

#include <string>
#include <iostream>

void print_similar_biclique(BiGraph& g, SimilarBiclique& ans)
{
	printf("\tsim: %.5f\n", ans.sim_);
	printf("\tL: ");
	for (int u : ans.L_) printf("%d ", u);
	puts("");
	printf("\tR: ");
	for (int u : ans.R_) printf("%d ", u);
	puts("");
	puts("");
}

int main(int argc, const char* argv[])
{
	if (argc >= 4) {
		std::string input_path = argv[1];
		int size_bound = std::atoi(argv[2]);

		auto s1 = hclock::now();
		BiGraph& g = *from_text(input_path);
		auto s2 = hclock::now();

		LocalExact algo(g);
		double max_time = 0.0;
		for (int i = 3; i < argc; i++) {
			auto start = hclock::now();
			int q = std::atoi(argv[i]);
			SimilarBiclique ans = algo.local_exact_query(q, size_bound);
			double time = get_duration(start, hclock::now());
			max_time = std::max(max_time, time);
			//printf("%d Time cost: %.6f (s)\n", q, time);
			//print_similar_biclique(g, ans);
		}
		auto s3 = hclock::now();

		double read_time = get_duration(s1, s2);
		double compute_time = get_duration(s2, s3);
		printf("Mosib local exact, dataset: %s\n", input_path.c_str());
		printf("read: %.6f (s), num_query: %d, total_query_time: %.6f (s), max_query_time: %.6f (s) \n\n"
			, read_time, argc - 3, compute_time, max_time);
	}
	return 0;
}