#pragma once

#include "global.h"
#include "util.h"

#include <map>
#include <set>
#include <fstream>
#include <queue>


/**
 * The input bipartite graph.
 * This graph will not be updated after being built.
 */
class BiGraph {
public:
	/// @return the 1-hop and 2-hop neighbors of u
	VI get_12hop(int u) const;

	/// @return if u is a left-side node
	int is_left_node(int u) const;

	/// @return get the k-core of the graph
	VI get_kcore(int k) const;

	/// the adjacency lists
	VVI adj_;

	/// the number of left-side and right-side nodes
	int left_node_num_;
	int right_node_num_;

	/// the number of edges
	int edge_num_;
};

/// @brief build the bipartite graph from file
/// @param input_path the path the input graph file
/// @return the bipartite graph built
BiGraph* from_text(std::string input_path);

/// @brief build the bipartite graph from a list of edges
/// @param edges a list of edges (integer pairs)
/// @return the bipartite graph built
BiGraph* from_edges(std::vector<PII> &edges);


/**
 * An induced subgraph of the input bipartite graph ${g_}.
 * It can support efficient node removals and neighbor set
 * retrieval on the subgraph.
 */
class BiSubgraph {
public:
	BiSubgraph(const BiGraph& g, const VI& node_set);
	BiSubgraph(const BiGraph& g) : g_(g) {}

	/// build subgraph induced by ${node_set}
	/// ${node_set} must be sorted
	void from_node_set(const VI& node_set);

	/// recursively remove all vertices whose
	/// degree < ${size_bound} from the subgraph 
	void deg_rule(int size_bound);
	
	/// remove a set of vertices ${nodes} from the subgraph
	void remove_nodes(const VI &nodes);

	/// @return if node u exists in the subgraph
	bool is_node_exist(int u);

	/// clear the subgraph
	void clear();
	
	/// @return the adjacency list of u in the subgraph
	const SI& get_remain_adj(int u);

	/// @return ${remain_}
	const SI& get_remain();

	/// @param P a set of vertices
	/// @param u a node
	/// @return the intersection of P and the 2-hop-neighbors of u
	VI intersect_P_with_2hop(const VI& P, int u);

private:
	/// the input graph
	const BiGraph& g_;
	/// the set of remaining vertices in the subgraph
	SI remain_;
	/// the adjacency list of a node in the subgraph
	std::vector<SI> remain_adj_;
};
