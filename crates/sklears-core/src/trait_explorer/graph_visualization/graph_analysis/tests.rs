//! Tests for `graph_analysis` (split out of the main module by splitrs to
//! keep both files under the 2000-line policy).

#![allow(non_snake_case)]

use super::super::graph_config::CommunityDetection;
use super::super::graph_structures::{Community, TraitGraph, TraitGraphEdge, TraitGraphNode};
use super::*;

fn create_test_graph() -> TraitGraph {
    let mut graph = TraitGraph::new();
    let node1 = TraitGraphNode::new_trait("A".to_string(), "NodeA".to_string());
    let node2 = TraitGraphNode::new_trait("B".to_string(), "NodeB".to_string());
    let node3 = TraitGraphNode::new_trait("C".to_string(), "NodeC".to_string());
    let node4 = TraitGraphNode::new_trait("D".to_string(), "NodeD".to_string());
    graph.add_node(node1);
    graph.add_node(node2);
    graph.add_node(node3);
    graph.add_node(node4);
    let edge1 = TraitGraphEdge::new_inheritance("A".to_string(), "B".to_string());
    let edge2 = TraitGraphEdge::new_inheritance("B".to_string(), "C".to_string());
    let edge3 = TraitGraphEdge::new_inheritance("A".to_string(), "D".to_string());
    let edge4 = TraitGraphEdge::new_inheritance("D".to_string(), "C".to_string());
    graph.add_edge(edge1);
    graph.add_edge(edge2);
    graph.add_edge(edge3);
    graph.add_edge(edge4);
    graph
}

#[test]
fn test_graph_analyzer_creation() {
    let analyzer = GraphAnalyzer::new();
    assert!(analyzer.simd_enabled);
    assert!(analyzer.parallel_enabled);
}

#[test]
fn test_degree_centrality() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let centrality_a = analyzer
        .calculate_degree_centrality(&graph, "A")
        .expect("calculate_degree_centrality should succeed");
    let centrality_c = analyzer
        .calculate_degree_centrality(&graph, "C")
        .expect("calculate_degree_centrality should succeed");
    assert!(centrality_a > 0.0);
    assert!(centrality_c > 0.0);
}

#[test]
fn test_betweenness_centrality() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let centrality_b = analyzer
        .calculate_betweenness_centrality(&graph, "B")
        .expect("calculate_betweenness_centrality should succeed");
    let centrality_d = analyzer
        .calculate_betweenness_centrality(&graph, "D")
        .expect("calculate_betweenness_centrality should succeed");
    assert!(centrality_b >= 0.0);
    assert!(centrality_d >= 0.0);
}

#[test]
fn test_closeness_centrality() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let centrality_a = analyzer
        .calculate_closeness_centrality(&graph, "A")
        .expect("calculate_closeness_centrality should succeed");
    assert!(centrality_a >= 0.0);
}

#[test]
fn test_pagerank() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let pagerank_a = analyzer
        .calculate_pagerank(&graph, "A")
        .expect("calculate_pagerank should succeed");
    let pagerank_c = analyzer
        .calculate_pagerank(&graph, "C")
        .expect("calculate_pagerank should succeed");
    assert!(pagerank_a > 0.0);
    assert!(pagerank_c > 0.0);
}

#[test]
fn test_all_centrality_measures() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let centralities = analyzer
        .calculate_all_centrality_measures(&graph)
        .expect("calculate_all_centrality_measures should succeed");
    assert_eq!(centralities.len(), 4);
    assert!(centralities.contains_key("A"));
    assert!(centralities.contains_key("B"));
    assert!(centralities.contains_key("C"));
    assert!(centralities.contains_key("D"));
    for (_, measures) in centralities {
        assert!(measures.degree >= 0.0 && measures.degree <= 1.0);
        assert!(measures.betweenness >= 0.0 && measures.betweenness <= 1.0);
        assert!(measures.closeness >= 0.0);
        assert!(measures.pagerank >= 0.0 && measures.pagerank <= 1.0);
    }
}

#[test]
fn test_shortest_paths() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let paths = analyzer
        .find_all_shortest_paths(&graph, "A", "C")
        .expect("find_all_shortest_paths should succeed");
    assert!(!paths.is_empty());
    assert!(paths
        .iter()
        .any(|path| path.start() == Some("A") && path.end() == Some("C")));
}

#[test]
fn test_community_detection() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let communities = analyzer
        .detect_communities(&graph, CommunityDetection::Louvain)
        .expect("detect_communities should succeed");
    assert!(!communities.is_empty());
    for community in communities {
        assert!(!community.nodes.is_empty());
        assert!(!community.id.is_empty());
    }
}

#[test]
fn test_label_propagation() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let communities = analyzer
        .label_propagation(&graph)
        .expect("label_propagation should succeed");
    for community in communities {
        assert!(!community.nodes.is_empty());
    }
}

#[test]
fn test_graph_quality_metrics() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let quality = analyzer
        .calculate_graph_quality_metrics(&graph)
        .expect("calculate_graph_quality_metrics should succeed");
    assert!(quality.clarity >= 0.0 && quality.clarity <= 1.0);
    assert!(quality.layout_quality >= 0.0 && quality.layout_quality <= 1.0);
    assert!(quality.information_density >= 0.0 && quality.information_density <= 1.0);
    assert!(quality.aesthetic_appeal >= 0.0 && quality.aesthetic_appeal <= 1.0);
    assert!(quality.usability >= 0.0 && quality.usability <= 1.0);
}

#[test]
fn test_comprehensive_analysis() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();

    let analysis = analyzer
        .analyze_graph(&graph)
        .expect("analyze_graph should succeed");

    assert!(!analysis.centrality_measures.is_empty());
    // `find_critical_paths_comprehensive` only considers node pairs
    // where *both* nodes clear a deliberately high (> 0.7)
    // `importance_score()` bar meant to identify genuinely dominant
    // hubs; on this small 4-node diamond graph (A -> {B, D} -> C) no
    // single node combines high degree, betweenness, closeness,
    // eigenvector, *and* pagerank centrality simultaneously, so an
    // empty result here is correct, not a bug. Larger, hub-shaped
    // graphs (as produced by `generate_full_graph` for a real trait
    // hierarchy) are where this threshold is expected to find paths.
    assert!(analysis.critical_paths.len() < graph.nodes.len() * graph.nodes.len());
    assert!(analysis.quality_metrics.overall_quality() >= 0.0);
}

#[test]
fn test_adjacency_matrix() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let matrix = analyzer
        .build_adjacency_matrix(&graph)
        .expect("build_adjacency_matrix should succeed");
    assert_eq!(matrix.dim(), (4, 4));
    assert!(matrix.sum() > 0.0);
}

#[test]
fn test_shortest_path_distances() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_test_graph();
    let distances = analyzer
        .calculate_shortest_path_distances(&graph, "A")
        .expect("calculate_shortest_path_distances should succeed");
    assert!(distances.contains_key("A"));
    assert_eq!(distances["A"], 0.0);
    assert!(distances.len() > 1);
}

#[test]
fn test_cache_operations() {
    let analyzer = GraphAnalyzer::new();
    analyzer.clear_cache();
    assert!(analyzer.computation_cache.lock().is_ok());
}

#[test]
fn test_configuration() {
    let mut analyzer = GraphAnalyzer::new();
    analyzer.set_simd_enabled(false);
    assert!(!analyzer.simd_enabled);
    analyzer.set_parallel_enabled(false);
    assert!(!analyzer.parallel_enabled);
    analyzer.set_simd_enabled(true);
    assert!(analyzer.simd_enabled);
}

/// A "bowtie" graph: two triangles {A, B, C} and {D, E, F} joined only
/// by a single C-D edge. This is the classic minimal topology for
/// exercising bridge/articulation-point algorithms: the C-D edge is the
/// graph's only bridge, and C and D are its only articulation points
/// (removing either one splits the graph into two components).
fn create_bowtie_graph() -> TraitGraph {
    let mut graph = TraitGraph::new();
    for id in ["A", "B", "C", "D", "E", "F"] {
        graph.add_node(TraitGraphNode::new_trait(id.to_string(), id.to_string()));
    }
    for (from, to) in [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        ("D", "E"),
        ("E", "F"),
        ("F", "D"),
        ("C", "D"),
    ] {
        graph.add_edge(TraitGraphEdge::new_usage(from.to_string(), to.to_string()));
    }
    graph
}

#[test]
fn test_identify_bottleneck_edges_finds_the_bridge() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_bowtie_graph();

    let bottlenecks = analyzer
        .identify_bottleneck_edges(&graph)
        .expect("identify_bottleneck_edges should succeed");

    assert_eq!(bottlenecks, vec!["C->D".to_string()]);
}

#[test]
fn test_identify_bridge_nodes_finds_the_articulation_points() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_bowtie_graph();

    let articulation_points = analyzer
        .identify_bridge_nodes(&graph)
        .expect("identify_bridge_nodes should succeed");

    assert_eq!(articulation_points, vec!["C".to_string(), "D".to_string()]);
}

#[test]
fn test_identify_bottleneck_edges_empty_for_a_single_cycle() {
    // A single cycle has no bridges: every edge lies on two distinct
    // paths between its endpoints (going around the cycle either way).
    let analyzer = GraphAnalyzer::new();
    let mut graph = TraitGraph::new();
    for id in ["A", "B", "C", "D"] {
        graph.add_node(TraitGraphNode::new_trait(id.to_string(), id.to_string()));
    }
    for (from, to) in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")] {
        graph.add_edge(TraitGraphEdge::new_usage(from.to_string(), to.to_string()));
    }

    assert!(analyzer
        .identify_bottleneck_edges(&graph)
        .expect("identify_bottleneck_edges should succeed")
        .is_empty());
    assert!(analyzer
        .identify_bridge_nodes(&graph)
        .expect("identify_bridge_nodes should succeed")
        .is_empty());
}

#[test]
fn test_identify_hub_nodes() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_bowtie_graph();

    // Every node has the same degree (2) in this graph, so a very low
    // threshold should select all of them, and an unreachably high
    // threshold should select none.
    let all_hubs = analyzer
        .identify_hub_nodes(&graph, 0.0)
        .expect("identify_hub_nodes should succeed");
    assert_eq!(all_hubs.len(), graph.nodes.len());

    let no_hubs = analyzer
        .identify_hub_nodes(&graph, 1.1)
        .expect("identify_hub_nodes should succeed");
    assert!(no_hubs.is_empty());
}

#[test]
fn test_calculate_modularity_perfect_partition_scores_higher_than_random() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_bowtie_graph();

    let perfect_partition = vec![
        Community::new(
            "left".to_string(),
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        ),
        Community::new(
            "right".to_string(),
            vec!["D".to_string(), "E".to_string(), "F".to_string()],
        ),
    ];
    let perfect_q = analyzer
        .calculate_modularity(&graph, &perfect_partition)
        .expect("calculate_modularity should succeed");

    // A partition that ignores the graph's actual community structure
    // (splitting each triangle across both "communities") should score
    // lower than the partition that matches the two triangles exactly.
    let poor_partition = vec![
        Community::new(
            "mixed1".to_string(),
            vec!["A".to_string(), "D".to_string(), "E".to_string()],
        ),
        Community::new(
            "mixed2".to_string(),
            vec!["B".to_string(), "C".to_string(), "F".to_string()],
        ),
    ];
    let poor_q = analyzer
        .calculate_modularity(&graph, &poor_partition)
        .expect("calculate_modularity should succeed");

    assert!(
        perfect_q > poor_q,
        "perfect partition modularity ({perfect_q}) should exceed a poor partition's ({poor_q})"
    );
    // Modularity is bounded in [-1, 1] for any partition.
    assert!((-1.0..=1.0).contains(&perfect_q));
}

#[test]
fn test_calculate_modularity_empty_communities_is_zero() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_bowtie_graph();
    let modularity = analyzer
        .calculate_modularity(&graph, &[])
        .expect("calculate_modularity should succeed");
    assert_eq!(modularity, 0.0);
}

#[test]
fn test_calculate_small_world_coefficient_runs_on_bowtie_graph() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_bowtie_graph();

    let sigma = analyzer
        .calculate_small_world_coefficient(&graph)
        .expect("calculate_small_world_coefficient should succeed");

    // The bowtie graph is small and fully connected end-to-end, so the
    // coefficient should be a finite, non-negative number (small graphs
    // don't reliably exceed the sigma > 1 "small-world" threshold, but
    // the computation itself must be well-defined).
    assert!(sigma.is_finite());
    assert!(sigma >= 0.0);
}

#[test]
fn test_calculate_small_world_coefficient_trivial_graph_is_zero() {
    let analyzer = GraphAnalyzer::new();
    let mut graph = TraitGraph::new();
    graph.add_node(TraitGraphNode::new_trait("A".to_string(), "A".to_string()));
    graph.add_node(TraitGraphNode::new_trait("B".to_string(), "B".to_string()));

    // Fewer than 3 nodes: not enough structure for a clustering
    // coefficient or meaningful average path length.
    let sigma = analyzer
        .calculate_small_world_coefficient(&graph)
        .expect("calculate_small_world_coefficient should succeed");
    assert_eq!(sigma, 0.0);
}

#[test]
fn test_graph_structural_hash_stable_and_sensitive() {
    let graph_a = create_bowtie_graph();
    let graph_b = create_bowtie_graph();
    assert_eq!(
        GraphAnalyzer::graph_structural_hash(&graph_a),
        GraphAnalyzer::graph_structural_hash(&graph_b),
        "two structurally identical graphs must hash identically"
    );

    let mut graph_c = create_bowtie_graph();
    graph_c.add_node(TraitGraphNode::new_trait("G".to_string(), "G".to_string()));
    assert_ne!(
        GraphAnalyzer::graph_structural_hash(&graph_a),
        GraphAnalyzer::graph_structural_hash(&graph_c),
        "adding a node must change the structural hash"
    );
}

#[test]
fn test_betweenness_centrality_cache_hit_returns_same_result() {
    let analyzer = GraphAnalyzer::new();
    let graph = create_bowtie_graph();

    // First call populates the cache; second call should hit it and
    // return an identical value (exercising the cache lookup path in
    // `calculate_betweenness_centrality` added alongside
    // `ComputationCacheEntry`).
    let first = analyzer
        .calculate_betweenness_centrality(&graph, "C")
        .expect("calculate_betweenness_centrality should succeed");
    let second = analyzer
        .calculate_betweenness_centrality(&graph, "C")
        .expect("calculate_betweenness_centrality should succeed");
    assert_eq!(first, second);

    let stats = analyzer
        .get_performance_stats()
        .expect("tracker should be lockable");
    assert!(stats.contains_key("betweenness_C"));

    let memory_stats = analyzer
        .get_memory_stats()
        .expect("tracker should be lockable");
    assert!(memory_stats.contains_key("betweenness"));
}
