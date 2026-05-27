#include "global.h"
#include "util.h"


//Jaccard Similarity
double Jaccard(const VI &x, const VI &y)
{
	int ix = 0, iy = 0;
	int szx = x.size(), szy = y.size();
	int ans = 0;
	while (ix < szx && iy < szy) {
		if (x[ix] < y[iy]) {
			ix++;
		}
		else if (x[ix] == y[iy]) {
			ans++, ix++, iy++;
		}
		else {
			iy++;
		}
	}
	return double(ans) / (x.size() + y.size() - ans);
}

VI get_intersection(const VI& x, VI::const_iterator y_b, VI::const_iterator y_e)
{
	VI ans;
	std::set_intersection(x.begin(), x.end(),
		y_b, y_e,
		back_inserter(ans));
	return ans;
}

double get_duration(hclock::time_point s, hclock::time_point t)
{
	return std::chrono::duration_cast<std::chrono::duration<double>>(t - s).count();
}
