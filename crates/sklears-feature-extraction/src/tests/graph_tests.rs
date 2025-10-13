//! Graph analysis and manifold learning tests
//!
//! This module contains tests for graph feature extraction, network analysis,
//! and manifold learning algorithms including spectral features, centrality measures,
//! clustering coefficients, and dimensionality reduction techniques.

use crate::{graph, manifold};
use scirs2_core::ndarray::{array, Array2};

#[test]
fn test_graph_spectral_features() {
    // Create a simple graph: 0-1-2-3 (path graph)
    let edges = vec![(0, 1), (1, 2), (2, 3)];
    let n_nodes = 4;

    let extractor = graph::GraphSpectralFeatures::new()
        .n_eigenvalues(3)
        .include_eigenvector_centrality(true);

    let features = extractor.extract_features(n_nodes, &edges).unwrap();

    // Should have eigenvalues + eigenvector centrality values
    // n_eigenvalues + n_nodes = 3 + 4 = 7
    assert_eq!(features.len(), 7);

    // All features should be finite
    for &feat in features.iter() {
        assert!(feat.is_finite());
    }

    // Eigenvalues should be non-negative (for undirected graphs)
    for i in 0..3 {
        assert!(features[i] >= 0.0);
    }
}

#[test]
fn test_graph_centrality_features() {
    // Create a star graph: center node 0 connected to 1,2,3,4
    let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
    let n_nodes = 5;

    let extractor = graph::GraphCentralityFeatures::new()
        .include_degree_centrality(true)
        .include_betweenness_centrality(true)
        .include_closeness_centrality(true);

    let features = extractor
        .extract_features_from_edges(n_nodes, &edges)
        .unwrap();

    // Should have features for each node
    assert!(features.len() >= n_nodes);

    // All features should be finite and non-negative
    for &feat in features.iter() {
        assert!(feat.is_finite());
        assert!(feat >= 0.0);
    }
}

#[test]
fn test_graph_clustering_features() {
    // Create a triangle: 0-1-2-0
    let edges = vec![(0, 1), (1, 2), (2, 0)];
    let n_nodes = 3;

    let extractor = graph::GraphClusteringFeatures::new()
        .include_clustering_coefficient(true)
        .include_transitivity(true);

    let features = extractor.extract_features(n_nodes, &edges).unwrap();

    // Should have clustering coefficient for each node + global transitivity
    // n_nodes + 1 = 3 + 1 = 4
    assert_eq!(features.len(), 4);

    // All features should be finite and between 0 and 1
    for &feat in features.iter() {
        assert!(feat.is_finite());
        assert!(feat >= 0.0 && feat <= 1.0);
    }

    // In a triangle, all nodes should have clustering coefficient 1.0
    for i in 0..3 {
        assert!((features[i] - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_graph_motif_features() {
    // Create a small network with triangles and other motifs
    let edges = vec![(0, 1), (1, 2), (2, 0), (1, 3), (2, 3)];
    let n_nodes = 4;

    let extractor = graph::GraphMotifFeatures::new()
        .motif_size(3)
        .count_connected_motifs(true);

    let features = extractor.extract_features(n_nodes, &edges).unwrap();

    // Should count various 3-node motifs
    assert!(features.len() > 0);

    // All motif counts should be non-negative integers (but stored as f64)
    for &count in features.iter() {
        assert!(count.is_finite());
        assert!(count >= 0.0);
        assert!(count.fract() == 0.0); // Should be integer counts
    }
}

#[test]
fn test_graph_path_features() {
    // Create a path graph: 0-1-2-3-4
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
    let n_nodes = 5;

    let extractor = graph::GraphPathFeatures::new()
        .max_path_length(4)
        .include_shortest_paths(true)
        .include_path_distribution(true);

    let features = extractor.extract_features(n_nodes, &edges).unwrap();

    // Should include shortest path statistics and path length distribution
    assert!(features.len() > 0);

    // All features should be finite
    for &feat in features.iter() {
        assert!(feat.is_finite());
    }

    // In a path graph, diameter should be n_nodes - 1 = 4
    // (This would be in the shortest path statistics)
}

#[test]
fn test_graph_community_features() {
    // Create two connected components
    let edges = vec![(0, 1), (1, 2), (3, 4), (4, 5)]; // Two triangles
    let n_nodes = 6;

    let extractor = graph::GraphCommunityFeatures::new()
        .include_modularity(true)
        .include_community_sizes(true);

    let features = extractor.extract_features(n_nodes, &edges).unwrap();

    // Should include modularity score and community size statistics
    assert!(features.len() > 0);

    // All features should be finite
    for &feat in features.iter() {
        assert!(feat.is_finite());
    }
}

#[test]
fn test_graph_error_cases() {
    // Test with empty graph
    let empty_edges: Vec<(usize, usize)> = vec![];
    let extractor = graph::GraphSpectralFeatures::new();

    let result = extractor.extract_features(0, &empty_edges);
    // Should either succeed with empty features or fail gracefully
    match result {
        Ok(features) => assert!(features.is_empty()),
        Err(_) => (), // Error is acceptable for empty graph
    }

    // Test with invalid node indices
    let invalid_edges = vec![(0, 5)]; // Node 5 doesn't exist in 3-node graph
    let result = extractor.extract_features(3, &invalid_edges);
    assert!(result.is_err());
}

#[test]
fn test_manifold_isomap() {
    // Create data on a simple manifold (circle)
    let n_points = 50;
    let data: Vec<Vec<f64>> = (0..n_points)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
            vec![angle.cos(), angle.sin()]
        })
        .collect();

    let X = Array2::from_shape_vec((n_points, 2), data.into_iter().flatten().collect()).unwrap();

    let isomap = manifold::Isomap::new()
        .n_components(1) // Circle is 1D manifold
        .n_neighbors(5);

    let embedded = isomap.fit_transform(&X.view()).unwrap();

    assert_eq!(embedded.dim(), (n_points, 1));

    // All embedded values should be finite
    for &val in embedded.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_manifold_lle() {
    // Create data with local linear structure
    let X = array![
        [1.0, 1.0, 1.0],
        [1.1, 1.1, 1.0],
        [1.2, 1.2, 1.0],
        [2.0, 2.0, 2.0],
        [2.1, 2.1, 2.0],
        [2.2, 2.2, 2.0]
    ];

    let lle = manifold::LocallyLinearEmbedding::new()
        .n_components(2)
        .n_neighbors(3);

    let embedded = lle.fit_transform(&X.view()).unwrap();

    assert_eq!(embedded.dim(), (6, 2));

    // All embedded values should be finite
    for &val in embedded.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_manifold_tsne() {
    let X = array![
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [5.0, 6.0, 7.0],
        [5.1, 6.1, 7.1],
        [10.0, 11.0, 12.0],
        [10.1, 11.1, 12.1]
    ];

    let tsne = manifold::TSNE::new()
        .n_components(2)
        .perplexity(2.0)
        .max_iter(100); // Reduced for testing

    let embedded = tsne.fit_transform(&X.view()).unwrap();

    assert_eq!(embedded.dim(), (6, 2));

    // All embedded values should be finite
    for &val in embedded.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_manifold_umap() {
    let X = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 1.9],
        [5.0, 6.0],
        [5.1, 6.1],
        [4.9, 5.9],
        [10.0, 11.0],
        [10.1, 11.1],
        [9.9, 10.9]
    ];

    let umap = manifold::UMAP::new()
        .n_components(2)
        .n_neighbors(3)
        .min_dist(0.1)
        .n_epochs(50); // Reduced for testing

    let embedded = umap.fit_transform(&X.view()).unwrap();

    assert_eq!(embedded.dim(), (9, 2));

    // All embedded values should be finite
    for &val in embedded.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_manifold_mds() {
    let X = array![[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]];

    let mds = manifold::MDS::new().n_components(2).max_iter(100);

    let embedded = mds.fit_transform(&X.view()).unwrap();

    assert_eq!(embedded.dim(), (4, 2));

    // All embedded values should be finite
    for &val in embedded.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_manifold_spectral_embedding() {
    let X = array![
        [1.0, 0.0],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.0, 1.0],
        [0.9, 0.9],
        [1.0, 1.0]
    ];

    let spectral = manifold::SpectralEmbedding::new()
        .n_components(2)
        .gamma(1.0);

    let embedded = spectral.fit_transform(&X.view()).unwrap();

    assert_eq!(embedded.dim(), (6, 2));

    // All embedded values should be finite
    for &val in embedded.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_manifold_error_cases() {
    let X = array![[1.0, 2.0]]; // Only one sample

    // Most manifold methods should fail with too few samples
    let tsne = manifold::TSNE::new().n_components(2);
    let result = tsne.fit_transform(&X.view());
    assert!(result.is_err());

    let isomap = manifold::Isomap::new().n_neighbors(5);
    let result = isomap.fit_transform(&X.view());
    assert!(result.is_err()); // Can't have 5 neighbors with only 1 sample

    // Test with invalid parameters
    let X_normal = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    let tsne_invalid = manifold::TSNE::new().perplexity(10.0); // Too high for 4 samples
    let result = tsne_invalid.fit_transform(&X_normal.view());
    assert!(result.is_err());
}

#[test]
fn test_manifold_consistency() {
    let X = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [5.0, 6.0],
        [5.1, 6.1],
        [9.0, 10.0],
        [9.1, 10.1]
    ];

    // Test MDS consistency (deterministic)
    let mds = manifold::MDS::new().n_components(2).max_iter(50);

    let embedded1 = mds.fit_transform(&X.view()).unwrap();
    let embedded2 = mds.fit_transform(&X.view()).unwrap();

    assert_eq!(embedded1.dim(), embedded2.dim());

    // Results should be similar (allowing for some numerical differences)
    let mut differences = vec![];
    for (e1, e2) in embedded1.iter().zip(embedded2.iter()) {
        differences.push((e1 - e2).abs());
    }

    let max_diff = differences.iter().fold(0.0_f64, |a, &b| a.max(b));
    assert!(max_diff < 0.1); // Should be very similar for deterministic method
}

#[test]
fn test_graph_and_manifold_integration() {
    // Test workflow: Graph -> features -> manifold learning

    // Create a graph structure
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]; // Square with diagonal
    let n_nodes = 4;

    let graph_extractor = graph::GraphCentralityFeatures::new()
        .include_degree_centrality(true)
        .include_betweenness_centrality(true);

    let graph_features = graph_extractor
        .extract_features_from_edges(n_nodes, &edges)
        .unwrap();

    // Reshape features for manifold learning (each node as a sample)
    let feature_dim = graph_features.len() / n_nodes;
    let X = Array2::from_shape_vec((n_nodes, feature_dim), graph_features.to_vec()).unwrap();

    // Apply manifold learning
    let mds = manifold::MDS::new().n_components(2);
    let embedded = mds.fit_transform(&X.view()).unwrap();

    assert_eq!(embedded.dim(), (n_nodes, 2));

    for &val in embedded.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_large_graph_scalability() {
    // Test with a larger graph to ensure reasonable performance
    let n_nodes = 100;
    let mut edges = vec![];

    // Create a random sparse graph
    for i in 0..n_nodes {
        let next = (i + 1) % n_nodes;
        edges.push((i, next)); // Ring structure

        // Add some random connections
        if i % 10 == 0 && i + 10 < n_nodes {
            edges.push((i, i + 10));
        }
    }

    let extractor = graph::GraphSpectralFeatures::new()
        .n_eigenvalues(10)
        .include_eigenvector_centrality(false); // Skip expensive computation

    let start = std::time::Instant::now();
    let features = extractor.extract_features(n_nodes, &edges).unwrap();
    let duration = start.elapsed();

    // Should complete in reasonable time
    assert!(duration.as_secs() < 5);

    // Should produce meaningful features
    assert_eq!(features.len(), 10); // Just eigenvalues

    for &feat in features.iter() {
        assert!(feat.is_finite());
    }
}

#[test]
fn test_manifold_different_input_sizes() {
    // Test manifold methods with different input dimensionalities

    // 2D -> 1D
    let X_2d = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
    let mds_2d = manifold::MDS::new().n_components(1);
    let embedded_2d = mds_2d.fit_transform(&X_2d.view()).unwrap();
    assert_eq!(embedded_2d.dim(), (4, 1));

    // 3D -> 2D
    let X_3d = array![
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ];
    let mds_3d = manifold::MDS::new().n_components(2);
    let embedded_3d = mds_3d.fit_transform(&X_3d.view()).unwrap();
    assert_eq!(embedded_3d.dim(), (4, 2));

    // High-dimensional -> 2D
    let X_high = Array2::from_shape_fn((10, 20), |(i, j)| i as f64 + j as f64 * 0.1);
    let mds_high = manifold::MDS::new().n_components(2).max_iter(50);
    let embedded_high = mds_high.fit_transform(&X_high.view()).unwrap();
    assert_eq!(embedded_high.dim(), (10, 2));

    // All results should be finite
    for &val in embedded_2d
        .iter()
        .chain(embedded_3d.iter())
        .chain(embedded_high.iter())
    {
        assert!(val.is_finite());
    }
}
