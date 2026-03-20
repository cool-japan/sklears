//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

use super::types::{
    AdvancedManifoldLearning, DistanceMetric, EigenSolver, GeodesicMethod, ManifoldMethod,
    PathMethod,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    #[test]
    fn test_manifold_learning_creation() {
        let manifold = AdvancedManifoldLearning::new(2, 2);
        assert_eq!(manifold.intrinsic_dimension, 2);
        assert_eq!(manifold.embedding_dimension, 2);
    }
    #[test]
    fn test_distance_metrics() {
        let manifold =
            AdvancedManifoldLearning::new(2, 2).distance_metric(DistanceMetric::Euclidean);
        let x = array![1.0, 2.0];
        let y = array![4.0, 6.0];
        let distance = manifold
            .compute_distance(&x.view(), &y.view())
            .expect("computation should succeed");
        assert!((distance - 5.0).abs() < 1e-10);
    }
    #[test]
    fn test_nearest_neighbors_computation() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]];
        let manifold = AdvancedManifoldLearning::new(2, 2);
        let neighbors = manifold
            .compute_nearest_neighbors(&data.view(), 2)
            .expect("computation should succeed");
        assert_eq!(neighbors.len(), 5);
        for neighbor_list in &neighbors {
            assert_eq!(neighbor_list.len(), 2);
        }
    }
    #[test]
    fn test_tsne_manifold_learning() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];
        let manifold = AdvancedManifoldLearning::new(2, 2).method(ManifoldMethod::TSNE {
            perplexity: 2.0,
            early_exaggeration: 4.0,
            learning_rate: 100.0,
            n_iter: 10,
            min_grad_norm: 1e-6,
        });
        let result = manifold.fit_transform(data.view());
        assert!(result.is_ok());
        let manifold_result = result.expect("operation should succeed");
        assert_eq!(manifold_result.embedding.nrows(), 5);
        assert_eq!(manifold_result.embedding.ncols(), 2);
        assert!(manifold_result.reconstruction_error >= 0.0);
        assert!(manifold_result.neighborhood_preservation >= 0.0);
    }
    #[test]
    fn test_umap_manifold_learning() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let manifold = AdvancedManifoldLearning::new(2, 2).method(ManifoldMethod::UMAP {
            n_neighbors: 3,
            min_dist: 0.1,
            spread: 1.0,
            repulsion_strength: 1.0,
            n_epochs: 10,
        });
        let result = manifold.fit_transform(data.view());
        assert!(result.is_ok());
        let manifold_result = result.expect("operation should succeed");
        assert_eq!(manifold_result.embedding.nrows(), 5);
        assert_eq!(manifold_result.embedding.ncols(), 2);
    }
    #[test]
    fn test_laplacian_eigenmaps() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let manifold = AdvancedManifoldLearning::new(2, 2).n_neighbors(2).method(
            ManifoldMethod::LaplacianEigenmaps {
                sigma: 1.0,
                reg_parameter: 0.1,
                use_normalized_laplacian: true,
            },
        );
        let result = manifold.fit_transform(data.view());
        assert!(result.is_ok());
        let manifold_result = result.expect("operation should succeed");
        assert_eq!(manifold_result.embedding.nrows(), 4);
        assert_eq!(manifold_result.embedding.ncols(), 2);
    }
    #[test]
    fn test_isomap() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let manifold = AdvancedManifoldLearning::new(2, 2).method(ManifoldMethod::Isomap {
            n_neighbors: 2,
            geodesic_method: GeodesicMethod::Dijkstra,
            path_method: PathMethod::Shortest,
        });
        let result = manifold.fit_transform(data.view());
        assert!(result.is_ok());
        let manifold_result = result.expect("operation should succeed");
        assert_eq!(manifold_result.embedding.nrows(), 5);
        assert_eq!(manifold_result.embedding.ncols(), 2);
        assert!(manifold_result.stress.is_some());
    }
    #[test]
    fn test_locally_linear_embedding() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.5, 1.5], [2.5, 2.5]];
        let manifold =
            AdvancedManifoldLearning::new(2, 1).method(ManifoldMethod::LocallyLinearEmbedding {
                n_neighbors: 2,
                reg_parameter: 0.01,
                eigen_solver: EigenSolver::Standard,
            });
        let result = manifold.fit_transform(data.view());
        assert!(result.is_ok());
        let manifold_result = result.expect("operation should succeed");
        assert_eq!(manifold_result.embedding.nrows(), 5);
        assert_eq!(manifold_result.embedding.ncols(), 1);
    }
    #[test]
    fn test_diffusion_maps() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 1.1], [0.5, 0.5]];
        let manifold = AdvancedManifoldLearning::new(2, 2).method(ManifoldMethod::DiffusionMaps {
            n_neighbors: 3,
            alpha: 0.5,
            diffusion_time: 1,
            epsilon: 1.0,
        });
        let result = manifold.fit_transform(data.view());
        assert!(result.is_ok());
        let manifold_result = result.expect("operation should succeed");
        assert_eq!(manifold_result.embedding.nrows(), 5);
        assert_eq!(manifold_result.embedding.ncols(), 2);
    }
    #[test]
    fn test_manifold_properties_computation() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let embedding = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let manifold = AdvancedManifoldLearning::new(2, 2);
        let properties = manifold
            .compute_manifold_properties(&data.view(), &embedding)
            .expect("operation should succeed");
        assert!(properties.intrinsic_dimension > 0.0);
        assert_eq!(properties.curvature_estimates.len(), 4);
        assert_eq!(properties.density_estimates.len(), 4);
        assert_eq!(properties.tangent_spaces.dim(), (4, 2, 2));
    }
    #[test]
    fn test_quality_metrics() {
        let original_data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let embedding = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let manifold = AdvancedManifoldLearning::new(2, 2);
        let neighborhood_preservation = manifold
            .compute_neighborhood_preservation(&original_data.view(), &embedding)
            .expect("operation should succeed");
        assert!(neighborhood_preservation >= 0.0 && neighborhood_preservation <= 1.0);
        let distances = manifold
            .compute_pairwise_distances(&original_data.view())
            .expect("operation should succeed");
        let global_preservation = manifold
            .compute_global_preservation(&distances, &embedding)
            .expect("operation should succeed");
        assert!(global_preservation >= 0.0);
    }
    #[test]
    fn test_error_handling() {
        let manifold = AdvancedManifoldLearning::new(2, 2);
        let x = array![1.0, 2.0];
        let y = array![1.0];
        let result = manifold.compute_distance(&x.view(), &y.view());
        assert!(result.is_err());
        let empty_data = Array2::zeros((0, 2));
        let result = manifold.fit_transform(empty_data.view());
        assert!(result.is_err());
    }
}
