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
	if (argc >= 3) {
		std::string input_path = argv[1];

		auto s1 = hclock::now();
		BiGraph& g = *from_text(input_path);
		auto s2 = hclock::now();

		GlobalExact algo(g);
		auto s3 = hclock::now();

		double read_time = get_duration(s1, s2);
		printf("Mosib global exact, dataset: %s\n", input_path.c_str());
		printf("read: %.6f (s)\n\n", read_time);
		for (int i = 2; i < argc; i++) {
			auto start = hclock::now();
			int size = std::atoi(argv[i]);
			SimilarBiclique ans = algo.global_exact_query(size);
			double time = get_duration(start, hclock::now());
			printf("size bound=%d, compute: %.6f (s)\n", size, time);
			print_similar_biclique(g, ans);
		}
	}
	return 0;
}