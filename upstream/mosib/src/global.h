#ifndef global_h
#define global_h

#include <vector>
#include <algorithm>
#include <string>
#include <random>
#include <chrono>
#include <set>


typedef std::vector<int> VI;
typedef std::vector<VI> VVI;
typedef std::pair<int, int> PII;
typedef std::pair<int, double> PID;
typedef std::pair<double, int> PDI;
typedef std::set<int> SI;

typedef std::chrono::high_resolution_clock hclock;

const double k_eps = 1e-6;

#endif /* global_h */