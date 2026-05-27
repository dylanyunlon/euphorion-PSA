#pragma once

#include "bigraph.h"

#include <map>

/**
 * This file contains the main algorithms, including the local exact
 * global exact, and global approximate algorithms.
 */


/**
 * The basic information of a similar biclique, such as the vertices
 * contained, and the similarity of the biclique.
 */
struct SimilarBiclique {
public:
	/// The vertices on the left side and right side
	VI L_;
	VI R_;
	/// The similarity of the biclique
	double sim_;

	SimilarBiclique(VI& L, VI &R, double sim) : L_(L), R_(R), sim_(sim) {}
	SimilarBiclique() : L_(VI()), R_(VI()), sim_(0) {}
};

/**
 * This class supports efficient retrieval of pair-wise Jaccard similarity
 * in a biparitite graph. It stores the similairity once computed.
 */
class SimilarityStore {
public:
	SimilarityStore(const VVI &adj) : adj_(adj) {}

	/// @brief compute the Jaccard similarity between u, v
	/// sim(u, v) is stored in ${sim_} once computed 
	/// @return the similarity between u, v
	double get_sim(int u, int v);
	void set_sim(int u, int v, double sim);

private:
	const VVI &adj_;
	std::map<PII, double> sim_;
};

/**
 * The exact algorithm for the local most similar biclique
 */
class LocalExact {
public:
	LocalExact(const BiGraph& g) : g_(g), sim_(g.adj_), size_(-1), subgraph_(g), cnt_(g.left_node_num_, 0) {}

	/// @brief compute the local most similar biclique exactly
	/// @param q the query node
	/// @param size the size constraint tau
	/// @param init_sim the starting similarity for pruning
	/// @return the local most similar biclique containing ${q}
	virtual SimilarBiclique local_exact_query(int q, int size, double init_sim = -1.0);

	/// @brief compute the local most similar biclique on a given set of left-side vertices ${remain_left}
	/// used in the global solution
	/// @param q the query node
	/// @param size the minimum size tau
	/// @param remain_left a set of left-side vertices
	/// @param init_sim the starting similarity for pruning
	/// @return the local most similar biclique within ${remain_left}
	SimilarBiclique local_exact_query_left(int q, int size, const VI& remain_left, double init_sim = -1.0);

protected:
	/// reduce ${subgraph_} with similarity-based reduction rule 
	virtual void _sim_rule(int q, double sim);
	/// reduce ${subgraph_} with degree-based reduction rule 
	virtual void _deg_rule();

	virtual void _enum(const SimilarBiclique &cur, const VI& P, std::map<int, double> &last_sim);
	virtual void _simiarity_first_search_on_subgraph(int q);

	/// @brief initialize the node set with hop-based and similarity-based pruning rule
	/// @param q the query node
	/// @param sim the initial similarity
	/// @return the node set after pruning
	VI _get_2hop_and_sim(int q, double sim);
	VI _get_2hop_and_sim(int q, const VI&remain_left, double sim);

	// the temporary result of the graph
	const BiGraph& g_;
	SimilarityStore sim_;
	VI cnt_;

	// the temporary result when computing each local query
	SimilarBiclique result_;
	BiSubgraph subgraph_;
	int size_;

};

/**
 * The exact algorithm for the global most similar biclique
 */
class GlobalExact {
public:
	GlobalExact(const BiGraph& g) : g_(g) {}

	/// @brief compute the global most similar biclique exactly
	/// @param size the size constraint tau
	/// @return the global most similar biclique
	SimilarBiclique global_exact_query(int size);

private:
	const BiGraph& g_;
};

/**
 * The approximate algorithm for the global most similar biclique
 */
class GlobalApp {
public:
	/// @brief initialize algorithm and Min-Hash
	/// @param g the input graph
	/// @param hash_num the number of hash functions
	GlobalApp(const BiGraph& g, int hash_num) : g_(g) { __init_min_hash(hash_num); }
	
	/// @brief compute the global most similar biclique approximately
	/// @param size the size constraint tau
	/// @return the resulting biclique
	SimilarBiclique global_app_query(int size);

private:
	/// @brief initialize Min-Hash
	/// @param hash_num the number of hash functions
	void __init_min_hash(int hash_num);

	/// the input graph
	const BiGraph& g_;
	/// the number of hash functions
	int hash_num_;
	/// the Min-Hash values of different functions
	VVI min_hash_;
};

/// <summary>
/// Local Exact Abalation Test (Experiment)
/// Test the performance without a rule
/// </summary>
class LocalExactNoDeg : public LocalExact {
public:
	LocalExactNoDeg(const BiGraph& g) : LocalExact(g) {}
protected:
	void _deg_rule() {
		return;
	}
};

class LocalExactNoHop : public LocalExact {
public:
	LocalExactNoHop(const BiGraph& g) : LocalExact(g) {}
	SimilarBiclique local_exact_query(int q, int size, double init_sim = -1.0);
	VI get_sim(int q, double sim);
};


class LocalExactNoSim : public LocalExact {
public:
	LocalExactNoSim(const BiGraph& g) : LocalExact(g) {}
	SimilarBiclique local_exact_query(int q, int size, double init_sim = -1.0);
protected:
	void _sim_rule(int q, double sim) {
		return;
	}
};

class LocalExactNoSFS : public LocalExact {
public:
	LocalExactNoSFS(const BiGraph& g) : LocalExact(g) {}
protected:
	void _enum(const SimilarBiclique& cur, const VI& P);
	void _simiarity_first_search_on_subgraph(int q);
};

class LocalExactNoSFS2 : public LocalExact {
public:
	LocalExactNoSFS2(const BiGraph& g) : LocalExact(g) {}
protected:
	void _simiarity_first_search_on_subgraph(int q);
};
