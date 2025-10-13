//! Comprehensive Tests for Spatial Mixture Models
//!
//! This module contains all tests for the spatial mixture model components,
//! ensuring proper functionality and correctness of spatial algorithms.

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::super::{
        geographic_mixture::GeographicMixtureBuilder,
        markov_random_field::MarkovRandomFieldMixtureBuilder,
        spatial_constraints::SpatialConstraint,
        spatial_utils::{euclidean_distance, k_nearest_neighbors, pairwise_distances},
        spatially_constrained_gmm::SpatiallyConstrainedGMMBuilder,
    };
    use crate::common::CovarianceType;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Predict};

    #[test]
    fn test_spatially_constrained_gmm_creation() {
        let gmm = SpatiallyConstrainedGMMBuilder::new(2)
            .spatial_weight(0.2)
            .build();

        assert_eq!(gmm.get_config().n_components, 2);
        assert_eq!(gmm.get_config().spatial_weight, 0.2);
    }

    #[test]
    fn test_spatial_constraints() {
        let distance_constraint = SpatialConstraint::Distance { radius: 1.0 };
        let adjacency_constraint = SpatialConstraint::Adjacency;
        let grid_constraint = SpatialConstraint::Grid { rows: 10, cols: 10 };

        assert_ne!(distance_constraint, adjacency_constraint);
        assert_ne!(adjacency_constraint, grid_constraint);
    }

    #[test]
    fn test_markov_random_field_mixture_creation() {
        let mrf = MarkovRandomFieldMixtureBuilder::new(3)
            .interaction_strength(1.5)
            .neighborhood_size(6)
            .build();

        assert_eq!(mrf.n_components, 3);
        assert_eq!(mrf.interaction_strength, 1.5);
        assert_eq!(mrf.neighborhood_size, 6);
    }

    #[test]
    fn test_geographic_mixture_creation() {
        let landmarks = array![[0.0, 0.0], [1.0, 1.0]];

        let geo = GeographicMixtureBuilder::new(2)
            .use_elevation(true)
            .use_distance_features(true)
            .landmark_coordinates(landmarks)
            .build();

        assert_eq!(geo.n_components, 2);
        assert!(geo.use_elevation);
        assert!(geo.use_distance_features);
        assert!(geo.landmark_coordinates.is_some());
    }

    #[test]
    fn test_spatial_smoothness_computation() {
        let gmm = SpatiallyConstrainedGMMBuilder::new(2).build();
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let smoothness = gmm.compute_spatial_smoothness(&coords);
        assert!(smoothness.is_ok());

        let smoothness_matrix = smoothness.unwrap();
        assert_eq!(smoothness_matrix.shape(), &[4, 4]);
    }

    #[test]
    fn test_geographic_feature_extraction() {
        let landmarks = array![[0.5, 0.5]];
        let geo = GeographicMixtureBuilder::new(2)
            .use_elevation(true)
            .use_distance_features(true)
            .landmark_coordinates(landmarks)
            .build();

        let X = array![[0.0, 0.0, 10.0], [1.0, 1.0, 15.0], [0.5, 0.5, 12.0]];
        let features = geo.extract_geographic_features(&X);

        assert!(features.is_ok());
        let feature_matrix = features.unwrap();
        // Original 3 features + 1 elevation + 1 distance to landmark = 5 total
        assert_eq!(feature_matrix.ncols(), 5);
    }

    #[test]
    fn test_spatially_constrained_gmm_fitting() {
        let X = array![
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 1.1, 2.1],
            [0.0, 1.0, 0.9, 1.9],
            [1.0, 1.0, 1.2, 2.2],
            [5.0, 5.0, 5.1, 6.0],
            [6.0, 5.0, 5.2, 6.1],
            [5.0, 6.0, 4.9, 5.8],
            [6.0, 6.0, 5.3, 6.2]
        ];

        let coords = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0],
            [6.0, 6.0]
        ];

        let gmm = SpatiallyConstrainedGMMBuilder::new(2)
            .spatial_weight(0.3)
            .max_iter(50)
            .tolerance(1e-3)
            .build()
            .with_coordinates(coords);

        let result = gmm.fit(&X, &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_spatial_constraint_types() {
        let distance_constraint = SpatialConstraint::Distance { radius: 2.0 };
        let adjacency_constraint = SpatialConstraint::Adjacency;
        let grid_constraint = SpatialConstraint::Grid { rows: 5, cols: 5 };
        let custom_constraint = SpatialConstraint::Custom;

        // Test that different constraints are not equal
        assert_ne!(distance_constraint, adjacency_constraint);
        assert_ne!(adjacency_constraint, grid_constraint);
        assert_ne!(grid_constraint, custom_constraint);

        // Test distance constraint parameters
        if let SpatialConstraint::Distance { radius } = distance_constraint {
            assert_eq!(radius, 2.0);
        } else {
            panic!("Expected Distance constraint");
        }

        // Test grid constraint parameters
        if let SpatialConstraint::Grid { rows, cols } = grid_constraint {
            assert_eq!(rows, 5);
            assert_eq!(cols, 5);
        } else {
            panic!("Expected Grid constraint");
        }
    }

    #[test]
    fn test_pairwise_distances() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let distances = pairwise_distances(&coords);

        assert_eq!(distances.shape(), &[4, 4]);

        // Check diagonal is zero
        for i in 0..4 {
            assert_eq!(distances[[i, i]], 0.0);
        }

        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(distances[[i, j]], distances[[j, i]]);
            }
        }

        // Check specific distances
        assert!((distances[[0, 1]] - 1.0).abs() < 1e-10); // Distance from (0,0) to (1,0)
        assert!((distances[[0, 2]] - 1.0).abs() < 1e-10); // Distance from (0,0) to (0,1)
        assert!((distances[[0, 3]] - 2.0_f64.sqrt()).abs() < 1e-10); // Distance from (0,0) to (1,1)
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let coords = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0]
        ];

        let neighbors = k_nearest_neighbors(&coords, 2);

        assert_eq!(neighbors.len(), 6);

        // Each point should have 2 neighbors
        for neighbor_list in &neighbors {
            assert_eq!(neighbor_list.len(), 2);
        }

        // Check that point 0 has points 1 and 3 as nearest neighbors
        let point_0_neighbors = &neighbors[0];
        assert!(point_0_neighbors.contains(&1) || point_0_neighbors.contains(&3));
    }

    #[test]
    fn test_geographic_mixture_builder() {
        let landmarks = array![[10.0, 20.0], [30.0, 40.0]];

        let geo = GeographicMixtureBuilder::new(3)
            .use_elevation(true)
            .use_distance_features(true)
            .landmark_coordinates(landmarks.clone())
            .max_iter(200)
            .tolerance(1e-5)
            .random_state(42)
            .build();

        assert_eq!(geo.n_components, 3);
        assert!(geo.use_elevation);
        assert!(geo.use_distance_features);
        assert_eq!(geo.max_iter, 200);
        assert_eq!(geo.tol, 1e-5);
        assert_eq!(geo.random_state, Some(42));

        if let Some(coords) = geo.landmark_coordinates {
            assert_eq!(coords.shape(), &[2, 2]);
        } else {
            panic!("Expected landmark coordinates to be set");
        }
    }

    #[test]
    fn test_markov_random_field_mixture_builder() {
        let mrf = MarkovRandomFieldMixtureBuilder::new(4)
            .covariance_type(CovarianceType::Diagonal)
            .interaction_strength(2.5)
            .neighborhood_size(12)
            .max_iter(150)
            .tolerance(1e-6)
            .random_state(123)
            .build();

        assert_eq!(mrf.n_components, 4);
        assert_eq!(mrf.covariance_type, CovarianceType::Diagonal);
        assert_eq!(mrf.interaction_strength, 2.5);
        assert_eq!(mrf.neighborhood_size, 12);
        assert_eq!(mrf.max_iter, 150);
        assert_eq!(mrf.tol, 1e-6);
        assert_eq!(mrf.random_state, Some(123));
    }

    #[test]
    fn test_geographic_mixture_fitting() {
        let X = array![
            [0.0, 0.0, 100.0], // (lat, lon, elevation)
            [0.1, 0.1, 105.0],
            [0.05, 0.05, 102.0],
            [5.0, 5.0, 200.0],
            [5.1, 5.1, 205.0],
            [4.95, 4.95, 198.0]
        ];

        let landmarks = array![[2.5, 2.5]];

        let geo = GeographicMixtureBuilder::new(2)
            .use_elevation(true)
            .use_distance_features(true)
            .landmark_coordinates(landmarks)
            .max_iter(10) // Reduce iterations for test speed
            .build();

        let result = geo.fit(&X, &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_spatial_smoothness_matrix_properties() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];

        let gmm = SpatiallyConstrainedGMMBuilder::new(2)
            .spatial_constraint(SpatialConstraint::Distance { radius: 1.5 })
            .build();

        let smoothness = gmm.compute_spatial_smoothness(&coords).unwrap();

        // Check matrix is symmetric
        let (n, m) = smoothness.dim();
        assert_eq!(n, m);
        assert_eq!(n, 4);

        for i in 0..n {
            for j in 0..n {
                assert_eq!(smoothness[[i, j]], smoothness[[j, i]]);
            }
        }

        // Check diagonal is zero (no self-influence)
        for i in 0..n {
            assert_eq!(smoothness[[i, i]], 0.0);
        }

        // Check that nearby points have positive influence
        assert!(smoothness[[0, 1]] > 0.0); // Distance 1.0 < 1.5
        assert!(smoothness[[0, 2]] > 0.0); // Distance 1.0 < 1.5

        // Check that far points have no influence
        assert_eq!(smoothness[[0, 3]], 0.0); // Distance 2√2 ≈ 2.83 > 1.5
    }

    #[test]
    fn test_euclidean_distance_function() {
        let p1 = vec![0.0, 0.0];
        let p2 = vec![3.0, 4.0];
        let distance = euclidean_distance(&p1, &p2);

        assert!((distance - 5.0).abs() < 1e-10); // 3-4-5 triangle

        // Test with same point
        let distance_zero = euclidean_distance(&p1, &p1);
        assert_eq!(distance_zero, 0.0);

        // Test with negative coordinates
        let p3 = vec![-1.0, -1.0];
        let p4 = vec![2.0, 3.0];
        let distance_negative = euclidean_distance(&p3, &p4);
        assert!((distance_negative - 5.0).abs() < 1e-10); // Distance should be 5
    }

    #[test]
    fn test_spatial_constraint_adjacency() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];

        let gmm = SpatiallyConstrainedGMMBuilder::new(2)
            .spatial_constraint(SpatialConstraint::Adjacency)
            .build();

        let smoothness = gmm.compute_spatial_smoothness(&coords).unwrap();

        // Each point should be connected to its k=4 nearest neighbors
        // Since we have 5 points in a line, each should connect to all others
        for i in 0..5 {
            let mut connections = 0;
            for j in 0..5 {
                if i != j && smoothness[[i, j]] > 0.0 {
                    connections += 1;
                }
            }
            assert_eq!(connections, 4); // Should connect to k=4 nearest neighbors
        }
    }

    #[test]
    fn test_spatial_constraint_grid() {
        let gmm = SpatiallyConstrainedGMMBuilder::new(2)
            .spatial_constraint(SpatialConstraint::Grid { rows: 2, cols: 2 })
            .build();

        // Create 2x2 grid coordinates
        let coords = array![
            [0.0, 0.0],
            [0.0, 1.0], // Row 0
            [1.0, 0.0],
            [1.0, 1.0] // Row 1
        ];

        let smoothness = gmm.compute_spatial_smoothness(&coords).unwrap();

        // In a 2x2 grid:
        // Point 0 (0,0) connects to points 1 (0,1) and 2 (1,0)
        // Point 1 (0,1) connects to points 0 (0,0) and 3 (1,1)
        // Point 2 (1,0) connects to points 0 (0,0) and 3 (1,1)
        // Point 3 (1,1) connects to points 1 (0,1) and 2 (1,0)

        assert_eq!(smoothness[[0, 1]], 1.0); // Vertical neighbor
        assert_eq!(smoothness[[0, 2]], 1.0); // Horizontal neighbor
        assert_eq!(smoothness[[0, 3]], 0.0); // Diagonal - not connected in grid

        assert_eq!(smoothness[[1, 0]], 1.0); // Symmetry
        assert_eq!(smoothness[[1, 3]], 1.0); // Vertical neighbor
        assert_eq!(smoothness[[1, 2]], 0.0); // Diagonal - not connected
    }

    #[test]
    fn test_predict_basic_functionality() {
        let X_train = array![
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 1.0, 1.1, 2.1],
            [5.0, 5.0, 5.0, 6.0],
            [6.0, 6.0, 5.1, 6.1]
        ];

        let X_test = array![[0.5, 0.5, 1.05, 2.05], [5.5, 5.5, 5.05, 6.05]];

        let coords = array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]];

        let gmm = SpatiallyConstrainedGMMBuilder::new(2)
            .spatial_weight(0.1)
            .max_iter(10)
            .build()
            .with_coordinates(coords);

        let fitted = gmm.fit(&X_train, &()).unwrap();
        let predictions = fitted.predict(&X_test).unwrap();

        assert_eq!(predictions.len(), 2);
        // Predictions should be valid component indices
        for &pred in predictions.iter() {
            assert!(pred < 2);
        }
    }

    #[test]
    fn test_spatial_mixture_config_properties() {
        let distance_constraint = SpatialConstraint::Distance { radius: 2.0 };
        assert!(distance_constraint.requires_coordinates());
        assert!(!distance_constraint.requires_grid_dimensions());
        assert_eq!(distance_constraint.constraint_name(), "distance");

        let grid_constraint = SpatialConstraint::Grid { rows: 5, cols: 5 };
        assert!(!grid_constraint.requires_coordinates());
        assert!(grid_constraint.requires_grid_dimensions());
        assert_eq!(grid_constraint.constraint_name(), "grid");

        let adjacency_constraint = SpatialConstraint::Adjacency;
        assert!(!adjacency_constraint.requires_coordinates());
        assert!(!adjacency_constraint.requires_grid_dimensions());
        assert_eq!(adjacency_constraint.constraint_name(), "adjacency");
    }

    #[test]
    fn test_markov_random_field_fitting() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];

        let mrf = MarkovRandomFieldMixtureBuilder::new(2)
            .interaction_strength(0.5)
            .neighborhood_size(2)
            .max_iter(5) // Reduce for test speed
            .build();

        let result = mrf.fit(&X, &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_geographic_feature_config_validation() {
        use super::super::geographic_mixture::GeographicFeatureConfig;
        let config = GeographicFeatureConfig::default();

        assert!(config.normalize_features);
        assert_eq!(config.spatial_weight, 1.5);
        assert!(config.landmark_coordinates.is_none());
    }

    #[test]
    fn test_edge_cases_small_datasets() {
        // Test with minimal data
        let X = array![[0.0, 0.0], [1.0, 1.0]];
        let coords = array![[0.0, 0.0], [1.0, 1.0]];

        let gmm = SpatiallyConstrainedGMMBuilder::new(2)
            .max_iter(5)
            .build()
            .with_coordinates(coords);

        // Should work with n_samples == n_components
        let result = gmm.fit(&X, &());
        assert!(result.is_ok());

        // Test with n_samples < n_components (should fail)
        let X_small = array![[0.0, 0.0]];
        let coords_small = array![[0.0, 0.0]];

        let gmm_small = SpatiallyConstrainedGMMBuilder::new(2)
            .build()
            .with_coordinates(coords_small);

        let result_small = gmm_small.fit(&X_small, &());
        assert!(result_small.is_err());
    }
}
