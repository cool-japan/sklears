//! Graph and network dataset generators
//!
//! This module provides generators for various types of network topologies and
//! graph structures commonly used in network analysis and graph machine learning.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, rng};
use sklears_core::error::{Result, SklearsError};

/// Generate a random graph using the Erdős–Rényi model
///
/// Creates a random graph where each possible edge is included with a given probability,
/// independently of other edges. This is one of the most fundamental random graph models.
///
/// # Parameters
/// - `n_nodes`: Number of nodes in the graph
/// - `edge_probability`: Probability of edge creation between any two nodes
/// - `directed`: Whether to generate a directed graph
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Adjacency matrix where 1.0 indicates an edge
pub fn make_erdos_renyi_graph(
    n_nodes: usize,
    edge_probability: f64,
    directed: bool,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_nodes == 0 {
        return Err(SklearsError::InvalidInput(
            "n_nodes must be positive".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&edge_probability) {
        return Err(SklearsError::InvalidInput(
            "edge_probability must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut adjacency = Array2::zeros((n_nodes, n_nodes));

    for i in 0..n_nodes {
        let start_j = if directed { 0 } else { i + 1 };
        for j in start_j..n_nodes {
            if i != j && rng.gen() < edge_probability {
                adjacency[[i, j]] = 1.0;
                if !directed {
                    adjacency[[j, i]] = 1.0;
                }
            }
        }
    }

    Ok(adjacency)
}

/// Generate a scale-free network using the Barabási–Albert model
///
/// Creates a scale-free network through preferential attachment, where new nodes
/// are more likely to connect to existing nodes with higher degrees. This produces
/// networks with heavy-tailed degree distributions.
///
/// # Parameters
/// - `n_nodes`: Total number of nodes in the final graph
/// - `m_edges`: Number of edges to attach from a new node to existing nodes
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Adjacency matrix of the generated scale-free network
pub fn make_barabasi_albert_graph(
    n_nodes: usize,
    m_edges: usize,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_nodes == 0 {
        return Err(SklearsError::InvalidInput(
            "n_nodes must be positive".to_string(),
        ));
    }

    if m_edges == 0 || m_edges >= n_nodes {
        return Err(SklearsError::InvalidInput(
            "m_edges must be positive and less than n_nodes".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut adjacency = Array2::zeros((n_nodes, n_nodes));
    let mut degrees = vec![0; n_nodes];

    // Start with a complete graph of m_edges + 1 nodes
    let initial_nodes = m_edges + 1;
    for i in 0..initial_nodes {
        for j in (i + 1)..initial_nodes {
            adjacency[[i, j]] = 1.0;
            adjacency[[j, i]] = 1.0;
            degrees[i] += 1;
            degrees[j] += 1;
        }
    }

    // Add remaining nodes with preferential attachment
    for new_node in initial_nodes..n_nodes {
        let total_degree: usize = degrees.iter().sum();
        let mut targets = Vec::new();

        // Select m_edges targets based on preferential attachment
        while targets.len() < m_edges {
            let mut cumulative_prob = 0.0;
            let rand_val = rng.gen();

            for (node, &degree) in degrees.iter().enumerate().take(new_node) {
                if targets.contains(&node) {
                    continue;
                }
                cumulative_prob += degree as f64 / total_degree as f64;
                if rand_val <= cumulative_prob {
                    targets.push(node);
                    break;
                }
            }
        }

        // Add edges to selected targets
        for &target in &targets {
            adjacency[[new_node, target]] = 1.0;
            adjacency[[target, new_node]] = 1.0;
            degrees[new_node] += 1;
            degrees[target] += 1;
        }
    }

    Ok(adjacency)
}

/// Generate a small-world network using the Watts-Strogatz model
///
/// Creates a small-world network by starting with a regular ring lattice
/// and randomly rewiring edges. This model exhibits both high clustering
/// and short path lengths characteristic of small-world networks.
///
/// # Parameters
/// - `n_nodes`: Number of nodes
/// - `k_neighbors`: Each node is initially connected to k nearest neighbors in ring topology
/// - `p_rewire`: Probability of rewiring each edge
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Adjacency matrix of the small-world network
pub fn make_watts_strogatz_graph(
    n_nodes: usize,
    k_neighbors: usize,
    p_rewire: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_nodes == 0 {
        return Err(SklearsError::InvalidInput(
            "n_nodes must be positive".to_string(),
        ));
    }

    if k_neighbors >= n_nodes || k_neighbors % 2 != 0 {
        return Err(SklearsError::InvalidInput(
            "k_neighbors must be even and less than n_nodes".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&p_rewire) {
        return Err(SklearsError::InvalidInput(
            "p_rewire must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut adjacency = Array2::zeros((n_nodes, n_nodes));

    // Start with a ring lattice
    for i in 0..n_nodes {
        for j in 1..=(k_neighbors / 2) {
            let neighbor = (i + j) % n_nodes;
            adjacency[[i, neighbor]] = 1.0;
            adjacency[[neighbor, i]] = 1.0;
        }
    }

    // Rewire edges with probability p_rewire
    for i in 0..n_nodes {
        for j in 1..=(k_neighbors / 2) {
            if rng.gen() < p_rewire {
                let old_neighbor = (i + j) % n_nodes;

                // Remove old edge
                adjacency[[i, old_neighbor]] = 0.0;
                adjacency[[old_neighbor, i]] = 0.0;

                // Add new random edge (avoiding self-loops and existing edges)
                loop {
                    let new_neighbor = rng.gen_range(0..n_nodes);
                    if new_neighbor != i && adjacency[[i, new_neighbor]] == 0.0 {
                        adjacency[[i, new_neighbor]] = 1.0;
                        adjacency[[new_neighbor, i]] = 1.0;
                        break;
                    }
                }
            }
        }
    }

    Ok(adjacency)
}

/// Generate a graph with community structure using the stochastic block model
///
/// Creates a graph where nodes are organized into communities, with different
/// probabilities for edges within and between communities. This is useful for
/// testing community detection algorithms.
///
/// # Parameters
/// - `community_sizes`: Vector of community sizes
/// - `p_within`: Probability of edges within communities
/// - `p_between`: Probability of edges between communities
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (adjacency matrix, community labels)
pub fn make_stochastic_block_graph(
    community_sizes: &[usize],
    p_within: f64,
    p_between: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if community_sizes.is_empty() {
        return Err(SklearsError::InvalidInput(
            "community_sizes cannot be empty".to_string(),
        ));
    }

    if community_sizes.iter().any(|&size| size == 0) {
        return Err(SklearsError::InvalidInput(
            "All community sizes must be positive".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&p_within) || !(0.0..=1.0).contains(&p_between) {
        return Err(SklearsError::InvalidInput(
            "Probabilities must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let n_nodes: usize = community_sizes.iter().sum();
    let mut adjacency = Array2::zeros((n_nodes, n_nodes));
    let mut labels = Array1::zeros(n_nodes);

    // Assign community labels
    let mut node_idx = 0;
    for (community_id, &size) in community_sizes.iter().enumerate() {
        for _ in 0..size {
            labels[node_idx] = community_id as i32;
            node_idx += 1;
        }
    }

    // Generate edges
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            let same_community = labels[i] == labels[j];
            let edge_prob = if same_community { p_within } else { p_between };

            if rng.gen() < edge_prob {
                adjacency[[i, j]] = 1.0;
                adjacency[[j, i]] = 1.0;
            }
        }
    }

    Ok((adjacency, labels))
}

/// Generate a random tree structure
///
/// Creates a random tree by connecting nodes in a way that ensures the result
/// is connected and acyclic. Uses a simple algorithm where each new node
/// connects to a randomly chosen existing node.
///
/// # Parameters
/// - `n_nodes`: Number of nodes in the tree
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Adjacency matrix of the random tree
pub fn make_random_tree(n_nodes: usize, random_state: Option<u64>) -> Result<Array2<f64>> {
    if n_nodes == 0 {
        return Err(SklearsError::InvalidInput(
            "n_nodes must be positive".to_string(),
        ));
    }

    if n_nodes == 1 {
        return Ok(Array2::zeros((1, 1)));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut adjacency = Array2::zeros((n_nodes, n_nodes));

    // Use a simple algorithm to generate a random spanning tree
    // Each new node connects to a randomly chosen existing node
    for i in 1..n_nodes {
        let parent = rng.gen_range(0..i);
        adjacency[[i, parent]] = 1.0;
        adjacency[[parent, i]] = 1.0;
    }

    Ok(adjacency)
}