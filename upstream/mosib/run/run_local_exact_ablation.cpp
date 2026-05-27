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
	if (argc >= 5) {
		std::string input_path = argv[1];
		int size_bound = std::atoi(argv[2]);
		std::string flag = argv[3];
		if (flag != "nodeg" && flag != "nohop" 
			&& flag != "nosim" && flag != "nosfs" && flag != "nosfs2") {
			printf("invalied parameter: %s\n", flag.c_str());
			return 0;
		}

		auto s1 = hclock::now();
		BiGraph& g = *from_text(input_path);
		auto s2 = hclock::now();
		double max_time = 0.0;

		LocalExact* algo;
		if (flag == "nodeg") {
			algo = new LocalExactNoDeg(g);
		}
		else if (flag == "nohop") {
			algo = new LocalExactNoHop(g);
		}
		else if (flag == "nosim") {
			algo = new LocalExactNoSim(g);
		}
		else if (flag == "nosfs") {
			algo = new LocalExactNoSFS(g);
		}
		else if (flag == "nosfs2") {
			algo = new LocalExactNoSFS2(g);
		}

		for (int i = 4; i < argc; i++) {
			auto start = hclock::now();
			int q = std::atoi(argv[i]);
			SimilarBiclique ans = algo->local_exact_query(q, size_bound);
			double time = get_duration(start, hclock::now());
			max_time = std::max(max_time, time);
		}
		auto s3 = hclock::now();

		double read_time = get_duration(s1, s2);
		double compute_time = get_duration(s2, s3);
		printf("Mosib local exact %s, dataset: %s\n", flag.c_str(), input_path.c_str());
		printf("read: %.6f (s), num_query: %d, total_query_time: %.6f (s), max_query_time: %.6f (s) \n\n"
			, read_time, argc - 4, compute_time, max_time);
	}
	return 0;
}