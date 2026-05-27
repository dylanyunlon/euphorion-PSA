#ifndef util_h
#define util_h

#include "global.h"


/// @brief the Jaccard similarity of ordered vectors
/// @param x an ordered vector of integers
/// @param y another ordered vector of integers
/// @return the Jaccard similarty
double Jaccard(const VI & x, const VI & y);

/// @brief the intersection of ordered containers
/// @param x the first ordered container
/// @param y the second ordered container 
/// @return the intersection
template <class _InIt1, class _InIt2>
VI get_intersection(const _InIt1& x, const _InIt2& y)
{
	VI ans;
	std::set_intersection(x.begin(), x.end(),
		y.begin(), y.end(),
		back_inserter(ans));
	return ans;
}

/// @brief the intersection of ordered lists.
/// The second one is a range of iterators.
VI get_intersection(const VI &x, VI::const_iterator y_b, VI::const_iterator y_e);

/// @brief get the time duration between s and t
/// @param s the begin time
/// @param t the end time
/// @return time duration
double get_duration(hclock::time_point s, hclock::time_point t);

#endif /* util_h */