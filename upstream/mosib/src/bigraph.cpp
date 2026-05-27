#include "bigraph.h"


VI BiGraph::get_12hop(int u) const
{
	// insert into ${ans} 1-hop and 2-hop neighbors of u
	VI ans(adj_[u]);
	for (int nbr : adj_[u]) {
		const VI& vec = adj_[nbr];
		std::copy(vec.begin(), vec.end(), std::back_inserter(ans));
	}
	// make elements of ${ans} unique
	std::sort(ans.begin(), ans.end());
	auto unique_end_it = std::unique(ans.begin(), ans.end());
	ans.erase(unique_end_it, ans.end());
	return ans;
}

int BiGraph::is_left_node(int u) const
{
	return u < left_node_num_;
}

VI BiGraph::get_kcore(int k) const
{
	int nl = left_node_num_;
	int nr = right_node_num_;
	int n = nl + nr;

	// store into ${q} nodes with degree < k
	std::queue<int> q;
	VI deg(n, 0);
	for (int i = 0; i < n; i++) {
		deg[i] = adj_[i].size();
		if (deg[i] < k) {
			q.push(i);
		}
	}

	// recursively remove nodes with degree < k
	while (!q.empty()) {
		int u = q.front();
		q.pop();

		for (int nbr : adj_[u]) {
			auto& cur_adj = adj_[nbr];
			if (deg[nbr] == k) {
				q.push(nbr);
			}
			deg[nbr]--;
		}
	}

	// return k-core
	VI ans;
	for (int i = 0; i < n; i++) {
		if (deg[i] >= k) ans.push_back(i);
	}
	std::sort(ans.begin(), ans.end());
	return ans;
}

BiGraph* from_text(std::string input_path)
{
	std::vector<PII> edges;
	std::ifstream fin(input_path);
	if (!fin) {
		throw std::invalid_argument("no file: " + input_path);
		return nullptr;
	}
	int u, v;
	fin >> u >> u >> u;
	while (fin >> u >> v) {
		edges.push_back({ u, v });
	}
	return from_edges(edges);
}

BiGraph* from_edges(std::vector<PII> &edges)
{
	// build the number of left-side and right-side vertices
	int left_node_max = 0, right_node_max = 0;
	for (PII &e : edges) {
		left_node_max = std::max(left_node_max, e.first);
		right_node_max = std::max(right_node_max, e.second);
	}
	left_node_max++, right_node_max++;

	BiGraph *g_pt = new BiGraph;
	BiGraph &g = *g_pt;
	g.left_node_num_ = left_node_max;
	g.right_node_num_ = right_node_max - left_node_max;
	g.edge_num_ = edges.size();

	// build the adjacency lists of nodes l..r
	auto build_adj_from_sorted_edges = [](VVI& adj, std::vector<PII>& edges, int l, int r) {
		std::sort(edges.begin(), edges.end());
		int ie = 0;
		for (int i = l; i < r; i++) {
			int start = ie;
			while (ie < edges.size() && edges[ie].first == i) ie++;

			VI cur_adj(ie - start, 0);
			for (int j = 0; j < cur_adj.size(); j++) {
				cur_adj[j] = edges[start + j].second;
			}
			adj.push_back(cur_adj);
		}
	};

	// build the adjacency lists of left-side vertices
	build_adj_from_sorted_edges(g.adj_, edges, 0, left_node_max);
	// build the adjacency lists of right-side vertices
	for (PII &e : edges) {
		std::swap(e.first, e.second);
	}
	build_adj_from_sorted_edges(g.adj_, edges, left_node_max, right_node_max);
	return g_pt;
}

BiSubgraph::BiSubgraph(const BiGraph& g, const VI& node_set) : g_(g)
{
	from_node_set(node_set);
}

void BiSubgraph::from_node_set(const VI& node_set)
{
	// update the size of ${remain_adj_}
	if (node_set.empty()) return;
	clear();
	int max_id = node_set.back() + 1;
	if (max_id > remain_adj_.size()) remain_adj_.resize(max_id);

	// build the remaining adjacency lists of left-side vertices
	int right_pos = std::lower_bound(node_set.begin(), node_set.end(), g_.left_node_num_) - node_set.begin();
	for (int i = 0; i < right_pos; i++) {
		int u = node_set[i];
		VI remain_adj = get_intersection(g_.adj_[u], node_set.begin() + right_pos, node_set.end());
		remain_adj_[u] = SI(remain_adj.begin(), remain_adj.end());
	}
	// build the remaining adjacency lists of right-side vertices
	for (int i = right_pos; i < node_set.size(); i++) {
		int u = node_set[i];
		VI remain_adj = get_intersection(g_.adj_[u], node_set.begin(), node_set.begin() + right_pos);
		remain_adj_[u] = SI(remain_adj.begin(), remain_adj.end());
	}

	for (int u : node_set) {
		if (!remain_adj_[u].empty()) {
			remain_.insert(u);
		}
	}
}

void BiSubgraph::clear()
{
	for (int u : remain_) {
		remain_adj_[u].clear();
	}
	remain_.clear();
}

void BiSubgraph::deg_rule(int size_bound)
{
	// store into ${q} nodes with degree < ${size_bound}
	std::queue<int> q;
	for (int u : remain_) {
		int deg = remain_adj_[u].size();
		if (deg < size_bound) {
			q.push(u);
		}
	}

	// recursively remove nodes with degree < ${size_bound}
	while (!q.empty()) {
		int u = q.front();
		q.pop();

		for (int nbr : remain_adj_[u]) {
			auto& cur_adj = remain_adj_[nbr];
			if (cur_adj.size() == size_bound) {
				q.push(nbr);
			}
			cur_adj.erase(u);
		}
		remain_.erase(u);
		remain_adj_[u].clear();
	}
}

void BiSubgraph::remove_nodes(const VI& nodes)
{
	for (int u : nodes) {
		// remove u from any adjacency lists
		for (int nbr : remain_adj_[u]) {
			auto& cur_adj = remain_adj_[nbr];
			cur_adj.erase(u);
		}
		// remove u from ${remain_} and ${remain_adj_}
		remain_.erase(u);
		remain_adj_[u].clear();
	}
}

bool BiSubgraph::is_node_exist(int u)
{
	return remain_.find(u) != remain_.end();
}

const SI& BiSubgraph::get_remain_adj(int u)
{
	if (is_node_exist(u)) {
		return remain_adj_[u];
	}
	else {
		throw std::invalid_argument("remove non-exist node from subgraph");
	}
}

const SI& BiSubgraph::get_remain()
{
	return remain_;
}

VI BiSubgraph::intersect_P_with_2hop(const VI& P, int u)
{
	const auto& adj = remain_adj_[u];
	VI ans;
	for (int v : P) {
		if (!is_node_exist(v) || v == u) continue;
		// quickly check if v is 2-hop neighbor of u with ${remain_adj_}
		for (int nbr_u : adj) {
			if (remain_adj_[v].count(nbr_u)) {
				ans.push_back(v);
				break;
			}
		}
	}
	return ans;
}
