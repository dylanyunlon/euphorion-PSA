#include "mosib.h"

#include <random>


double SimilarityStore::get_sim(int u, int v)
{
	if (u > v) std::swap(u, v);
	PII min_pair(u, v);
	if (sim_.find(min_pair) == sim_.end()) {
		sim_[min_pair] = Jaccard(adj_[u], adj_[v]);
	}
	return sim_[min_pair];
}

void SimilarityStore::set_sim(int u, int v, double sim)
{
	if (u > v) std::swap(u, v);
	sim_[{u, v}] = sim;
}

SimilarBiclique LocalExact::local_exact_query(int q, int size, double init_sim)
{
	// init local result and G_r with hop-based and similarity-based reduction rule
	this->size_ = size;
	this->result_ = SimilarBiclique();
	VI node_set = _get_2hop_and_sim(q, init_sim);
	if (node_set.size() < size) return this->result_;
	node_set.insert(node_set.end(), g_.adj_[q].begin(), g_.adj_[q].end());

	// apply degree-based reduction rule
	subgraph_.from_node_set(node_set);
	_deg_rule();

	// run similarity-first search on the subgraph
	_simiarity_first_search_on_subgraph(q);
	return this->result_;
}

SimilarBiclique LocalExact::local_exact_query_left(int q, int size, const VI& remain_left, double init_sim)
{
	// init local result and G_r with hop-based and similarity-based rule
	this->size_ = size;
	this->result_ = SimilarBiclique();

	VI node_set = _get_2hop_and_sim(q, remain_left, init_sim);
	if (node_set.size() < size) return this->result_;
	node_set.insert(node_set.end(), g_.adj_[q].begin(), g_.adj_[q].end());

	// apply degree-based rule
	subgraph_.from_node_set(node_set);
	_deg_rule();

	// run similarity-first search on the subgraph
	_simiarity_first_search_on_subgraph(q);
	return this->result_;
}

void LocalExact::_sim_rule(int q, double init_sim)
{
	if (init_sim <= 0) return;
	// remove the left-side nodes whose similarity to ${q} < ${init_sim}
	VI to_remove;
	for (int left_u : subgraph_.get_remain()) {
		if (!g_.is_left_node(left_u)) break;
		double sim = sim_.get_sim(left_u, q);
		if (sim < init_sim + k_eps) to_remove.push_back(left_u);
	}
	subgraph_.remove_nodes(to_remove);
}

void LocalExact::_deg_rule()
{
	// recursively remove nodes whose degree < ${size_}
	subgraph_.deg_rule(size_);
}

void LocalExact::_simiarity_first_search_on_subgraph(int q)
{
	if (!subgraph_.is_node_exist(q)) return;
	// sort nodes by decreasing order of sim(u,q)
	// The first similarity-first search rule, i.e., SFS
	std::map<int, double> left_u_sim;
	std::vector<PDI> to_sort_by_sim;
	for (int left_u : subgraph_.get_remain()) {
		if (!g_.is_left_node(left_u)) break;
		if (q == left_u) continue;

		double sim = sim_.get_sim(left_u, q);
		left_u_sim[left_u] = sim;
		to_sort_by_sim.push_back({ sim, left_u });
	}
	std::sort(to_sort_by_sim.begin(), to_sort_by_sim.end(), std::greater<PDI>());
	if (to_sort_by_sim.size() + 1 < size_) return;

	// attempt to move a node into the current biclique with the ordering
	VI visited;
	for (const PDI& pdi : to_sort_by_sim) {
		int left_u = pdi.second;
		if (!subgraph_.is_node_exist(left_u)) continue;

		// build biclique with left-side vertices q and left_u
		SimilarBiclique nxt;
		nxt.L_ = { q, left_u };
		nxt.R_ = get_intersection(subgraph_.get_remain_adj(left_u), subgraph_.get_remain_adj(q));
		nxt.sim_ = sim_.get_sim(left_u, q);

		// ${P} is the intersection of 2-hop neighbors of ${left_u} and visited vertices
		// The second similarity-first search rule, i.e., SFS2
		VI P = subgraph_.intersect_P_with_2hop(visited, left_u);
		_enum(nxt, P, left_u_sim);
		visited.push_back(left_u);
	}
}

VI LocalExact::_get_2hop_and_sim(int q, double sim)
{
	// speed up the jaccard computation with ${cnt_}
	// ${cnt_[w]} is the number of common neighbors of q and w
	VI remain;
	for (int v : g_.adj_[q]) {
		for (int w : g_.adj_[v]) {
			if (cnt_[w]++ == 0) remain.push_back(w);
		}
	}

	// ${remain} is the 2-hop neighbors (hop-based rule)
	// remove nodes with low similarity to q (similarity-based rule)
	VI ans;
	for (int u : remain) {
		int num = cnt_[u];
		double jaccard = double(num) / (g_.adj_[u].size() + g_.adj_[q].size() - num);
		if (jaccard > sim + k_eps) {
			sim_.set_sim(q, u, jaccard);
			ans.push_back(u);
		}
		cnt_[u] = 0;
	}
	std::sort(ans.begin(), ans.end());
	return ans;
}

VI LocalExact::_get_2hop_and_sim(int q, const VI& remain_left, double sim)
{
	// apply the similarity-based reduction rule within ${remain_left}
	VI ans;
	for (int left_u : remain_left) {
		double jaccard = sim_.get_sim(left_u, q);
		if (jaccard > sim + k_eps) {
			ans.push_back(left_u);
		}
	}
	return ans;
}

void LocalExact::_enum(const SimilarBiclique& cur, const VI& P, std::map<int, double>& last_sim)
{
	if (cur.sim_ < this->result_.sim_ + k_eps
		|| std::min(cur.L_.size() + P.size(), cur.R_.size()) < this->size_) {
		return;
	}
	if (std::min(cur.L_.size(), cur.R_.size()) >= this->size_) {
		// apply similarity-based and degree-based rule when ${cur}
		// has a higher similarity than ${this->result_}
		this->result_ = cur;
		int q = cur.L_[0];
		_sim_rule(q, cur.sim_);
		_deg_rule();
		return;
	}

	// order all candidate vertices, i.e., SFS rule
	std::vector<PDI> to_sort_by_sim;
	std::map<int, double> left_u_sim;
	int last_left_u = cur.L_.back();
	for (int left_u : P) {
		if (!subgraph_.is_node_exist(left_u)) continue;
		double sim = std::min(last_sim[left_u], this->sim_.get_sim(left_u, last_left_u));
		left_u_sim[left_u] = sim;
		to_sort_by_sim.push_back({ sim, left_u });
	}
	std::sort(to_sort_by_sim.begin(), to_sort_by_sim.end(), std::greater<PDI>());

	SI cur_P(P.begin(), P.end());
	for (int i = 0; i < to_sort_by_sim.size(); i++) {
		const PDI& pdi = to_sort_by_sim[i];
		int left_u = pdi.second;
		if (!subgraph_.is_node_exist(left_u)) continue;
		cur_P.erase(left_u);

		// attempt to add ${left_u} to the current biclique
		VI L = cur.L_;
		L.push_back(left_u);
		VI R = get_intersection(cur.R_, subgraph_.get_remain_adj(left_u));
		double sim = std::min(cur.sim_, pdi.first);
		SimilarBiclique nxt(L, R, sim);

		VI nxt_P(cur_P.begin(), cur_P.end());
		nxt_P = subgraph_.intersect_P_with_2hop(nxt_P, left_u);
		_enum(nxt, nxt_P, left_u_sim);
	}
}

SimilarBiclique GlobalExact::global_exact_query(int size)
{
	SimilarBiclique ans;
	LocalExact algo(g_);
	VI remain = g_.get_kcore(size);
	for (int left_u : remain) {
		if (left_u >= g_.left_node_num_) break;
		SimilarBiclique local_ans = algo.local_exact_query(left_u, size, ans.sim_);
		//if (local_ans.sim_ > ans.sim_ + k_eps
		//	|| (local_ans.sim_ > ans.sim_ - k_eps && local_ans.L_.size() > ans.L_.size())) {
		//	ans = local_ans;
		//}
		if (local_ans.sim_ > ans.sim_ + k_eps) {
			ans = local_ans;
		}
		//printf("finish %d, %.3f\n", left_u, ans.sim_);
		//if (ans.sim_ >= 1 - k_eps) break;
	}
	return ans;
}


const int max_cnt_down = 10;

void dfs_sep(int max_group, std::vector<PII>& to_sort, VVI& min_hash, VI& ans,
	int d, int l, int r)
{
	if (d == max_cnt_down || r - l <= max_group) {
		ans.push_back(l);
		return;
	}
	for (int i = l; i < r; i++) {
		int u = to_sort[i].second;
		to_sort[i].first = min_hash[u][d];
	}
	std::sort(to_sort.begin() + l, to_sort.begin() + r);

	int prev = l;
	for (int i = l + 1; i < r; i++) {
		if (to_sort[i].first != to_sort[i - 1].first) {
			dfs_sep(max_group, to_sort, min_hash, ans, d + 1, prev, i);
			prev = i;
		}
	}
	dfs_sep(max_group, to_sort, min_hash, ans, d + 1, prev, r);
}

VI get_sep(int max_group, std::vector<PII>& to_sort, VVI& min_hash)
{
	VI sep;
	dfs_sep(max_group, to_sort, min_hash, sep, 0, 0, to_sort.size());
	sep.push_back(to_sort.size());
	return sep;
}

SimilarBiclique GlobalApp::global_app_query(int size)
{
	std::mt19937 rng(2333);
	int max_group = size * 10;

	SimilarBiclique ans;
	LocalExact algo(g_);

	// divide the nodes in ${remain} into groups
	VI remain = g_.get_kcore(size);
	std::vector<PII> to_sort;
	for (int i = 0; i < remain.size(); i++) {
		if (remain[i] >= g_.left_node_num_) break;
		to_sort.push_back({0, remain[i] });
	}
	VI sep = get_sep(max_group, to_sort, min_hash_);

	// for each group, find the most similar biclique within the group
	for (int i = 0; i + 1 < sep.size(); i++) {
		int l = sep[i], r = sep[i + 1];
		if (r - l < size) continue;
		VI remain_left;
		for (int j = l; j < r; j++) {
			remain_left.push_back(to_sort[j].second);
		}
		std::sort(remain_left.begin(), remain_left.end());
		for (int left_u : remain_left) {
			SimilarBiclique local_ans = algo.local_exact_query_left(left_u, size, remain_left, ans.sim_);
			if (local_ans.sim_ > ans.sim_ + k_eps) {
				ans = local_ans;
			}
		}
		//printf("finish %d %d vertices, %.3f\n", i, r - l, ans.sim_);
	}
	return ans;
}

void GlobalApp::__init_min_hash(int hash_num)
{
	this->hash_num_ = hash_num;
	std::mt19937 rng(2333);
	int nl = g_.left_node_num_;
	int nr = g_.right_node_num_;

	// get the Min-Hash of ${hash_num} random hash functions
	min_hash_ = VVI(nl, VI(hash_num, -1));
	VI arr(nr, 0);
	for (int j = 0; j < nr; j++) arr[j] = j;
	for (int hh = 0; hh < hash_num; hh++) {
		std::shuffle(arr.begin(), arr.end(), rng);
		for (int j = 0; j < nr; j++) {
			int right_u = arr[j] + nl;
			for (int v : g_.adj_[right_u]) {
				if (min_hash_[v][hh] == -1) {
					min_hash_[v][hh] = j;
				}
			}
		}
	}
}

SimilarBiclique LocalExactNoHop::local_exact_query(int q, int size, double init_sim)
{
	// init local result and G_r
	this->size_ = size;
	this->result_ = SimilarBiclique();
	VI node_set = get_sim(q, init_sim);
	if (node_set.size() < size) return this->result_;
	node_set.insert(node_set.end(), g_.adj_[q].begin(), g_.adj_[q].end());

	subgraph_.from_node_set(node_set);
	_deg_rule();

	_simiarity_first_search_on_subgraph(q);
	return this->result_;
}

VI LocalExactNoHop::get_sim(int q, double sim)
{
	VI remain;
	for (int v : g_.adj_[q]) {
		for (int w : g_.adj_[v]) {
			if (cnt_[w]++ == 0) remain.push_back(w);
		}
	}

	VI ans;
	for (int u = 0; u < g_.left_node_num_; u++) {
		int num = cnt_[u];
		if (num) {
			double jaccard = Jaccard(g_.adj_[u], g_.adj_[q]);
			if (jaccard > sim + k_eps) {
				sim_.set_sim(q, u, jaccard);
				ans.push_back(u);
			}
		}
		cnt_[u] = 0;
	}
	return ans;
}

SimilarBiclique LocalExactNoSim::local_exact_query(int q, int size, double init_sim)
{
	// init local result and G_r
	this->size_ = size;
	this->result_ = SimilarBiclique();
	VI node_set = g_.get_12hop(q);

	subgraph_.from_node_set(node_set);
	_deg_rule();

	_simiarity_first_search_on_subgraph(q);
	return this->result_;
}

void LocalExactNoSFS2::_simiarity_first_search_on_subgraph(int q)
{
	if (!subgraph_.is_node_exist(q)) return;
	// sort nodes by decreasing order of sim(u,q)
	SimilarBiclique nxt;
	nxt.L_ = { q };
	auto se = subgraph_.get_remain_adj(q);
	nxt.R_ = VI(se.begin(), se.end());
	nxt.sim_ = 1;

	VI P;
	std::map<int, double> left_u_sim;
	for (int left_u : subgraph_.get_remain()) {
		if (!g_.is_left_node(left_u)) break;
		if (q == left_u) continue;

		double sim = sim_.get_sim(left_u, q);
		left_u_sim[left_u] = sim;
		P.push_back(left_u);
	}
	_enum(nxt, P, left_u_sim);
}

void LocalExactNoSFS::_enum(const SimilarBiclique& cur, const VI& P)
{
	if (cur.sim_ < this->result_.sim_ + k_eps
		|| std::min(cur.L_.size() + P.size(), cur.R_.size()) < this->size_) {
		return;
	}
	if (std::min(cur.L_.size(), cur.R_.size()) >= this->size_) {
		this->result_ = cur;
		int q = cur.L_[0];
		_sim_rule(q, cur.sim_);
		_deg_rule();
		return;
	}

	SI cur_P(P.begin(), P.end());
	for (int left_u : P) {
		if (!subgraph_.is_node_exist(left_u)) continue;
		cur_P.erase(left_u);

		VI L = cur.L_;
		L.push_back(left_u);
		VI R = get_intersection(cur.R_, subgraph_.get_remain_adj(left_u));
		double sim = cur.sim_;
		for (int uu : cur.L_) {
			sim = std::min(sim, this->sim_.get_sim(left_u, uu));
		}
		SimilarBiclique nxt(L, R, sim);

		VI nxt_P(cur_P.begin(), cur_P.end());
		nxt_P = subgraph_.intersect_P_with_2hop(nxt_P, left_u);
		_enum(nxt, nxt_P);
	}
}

void LocalExactNoSFS::_simiarity_first_search_on_subgraph(int q)
{
	if (!subgraph_.is_node_exist(q)) return;
	// sort nodes by decreasing order of sim(u,q)
	VI PP;
	for (int left_u : subgraph_.get_remain()) {
		if (!g_.is_left_node(left_u)) break;
		if (q == left_u) continue;
		PP.push_back(left_u);
	}
	if (PP.size() + 1 < size_) return;

	VI visited;
	for (int left_u : PP) {
		if (!subgraph_.is_node_exist(left_u)) continue;

		SimilarBiclique nxt;
		nxt.L_ = { q, left_u };
		nxt.R_ = get_intersection(subgraph_.get_remain_adj(left_u), subgraph_.get_remain_adj(q));
		nxt.sim_ = sim_.get_sim(left_u, q);

		VI P = subgraph_.intersect_P_with_2hop(visited, left_u);
		_enum(nxt, P);
		visited.push_back(left_u);
	}
}
