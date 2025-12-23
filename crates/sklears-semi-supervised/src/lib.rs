#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_doc_comments)]
#![allow(unused_parens)]
#![allow(unused_comparisons)]
//! Semi-supervised learning algorithms
//!
//! This module provides semi-supervised learning algorithms that can utilize
//! both labeled and unlabeled data for training.

// #![warn(missing_docs)]

mod active_learning;
mod adversarial_graph_learning;
mod approximate_graph_methods;
mod batch_active_learning;
mod bayesian_methods;
mod co_training;
mod composable_graph;
mod contrastive_learning;
mod convergence_tests;
mod cross_modal_contrastive;
mod deep_learning;
mod democratic_co_learning;
mod dynamic_graph_learning;
mod entropy_methods;
mod few_shot;
mod graph;
mod graph_learning;
mod harmonic_functions;
mod hierarchical_graph;
mod information_theory;
mod label_propagation;
mod label_spreading;
mod landmark_methods;
mod local_global_consistency;
mod manifold_regularization;
mod mixture_discriminant_analysis;
mod multi_armed_bandits;
mod multi_view_graph;
mod optimal_transport;
pub mod parallel_graph;
mod robust_graph_methods;
mod self_training;
mod self_training_classifier;
mod semi_supervised_gmm;
mod semi_supervised_naive_bayes;
pub mod simd_distances;
mod streaming_graph_learning;
mod tri_training;

pub use active_learning::*;
pub use adversarial_graph_learning::*;
pub use approximate_graph_methods::*;
pub use batch_active_learning::*;
pub use bayesian_methods::*;
pub use co_training::*;
pub use composable_graph::*;
pub use contrastive_learning::*;
pub use convergence_tests::*;
pub use cross_modal_contrastive::*;
pub use deep_learning::*;
pub use democratic_co_learning::*;
pub use dynamic_graph_learning::*;
pub use entropy_methods::*;
pub use few_shot::*;
pub use graph::*;
pub use graph_learning::*;
pub use harmonic_functions::*;
pub use hierarchical_graph::*;
pub use information_theory::*;
pub use label_propagation::*;
pub use label_spreading::*;
pub use landmark_methods::*;
pub use local_global_consistency::*;
pub use manifold_regularization::*;
pub use mixture_discriminant_analysis::*;
pub use multi_armed_bandits::*;
pub use multi_view_graph::*;
pub use optimal_transport::*;
pub use robust_graph_methods::*;
pub use self_training::*;
pub use self_training_classifier::*;
pub use semi_supervised_gmm::*;
pub use semi_supervised_naive_bayes::*;
pub use streaming_graph_learning::*;
pub use tri_training::*;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;
    use scirs2_core::ndarray_ext::Array2;
    use sklears_core::traits::{Fit, Predict, PredictProba};

    #[test]
    fn test_label_propagation() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let lp = LabelPropagation::new()
            .kernel("rbf".to_string())
            .gamma(20.0);
        let fitted = lp.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (4, 2));
    }

    #[test]
    fn test_label_spreading() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let ls = LabelSpreading::new()
            .kernel("rbf".to_string())
            .gamma(20.0)
            .alpha(0.2);
        let fitted = ls.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (4, 2));
    }

    #[test]
    fn test_self_training() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let stc = SelfTrainingClassifier::new().threshold(0.5).max_iter(5);
        let fitted = stc.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    // Note: Commented out test for private method
    // #[test]
    // fn test_affinity_matrix_rbf() {
    //     let lp = LabelPropagation::new().kernel("rbf".to_string()).gamma(1.0);
    //     let X = array![[1.0, 2.0], [3.0, 4.0]];
    //
    //     let W = lp.build_affinity_matrix(&X).unwrap();
    //     assert_eq!(W.dim(), (2, 2));
    //     assert_eq!(W[[0, 0]], 0.0); // Diagonal should be 0
    //     assert_eq!(W[[1, 1]], 0.0);
    //     assert!(W[[0, 1]] > 0.0); // Off-diagonal should be positive
    //     assert!(W[[1, 0]] > 0.0);
    // }

    // Note: Commented out test for private method
    // #[test]
    // fn test_affinity_matrix_knn() {
    //     let lp = LabelPropagation::new()
    //         .kernel("knn".to_string())
    //         .n_neighbors(1);
    //     let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    //
    //     let W = lp.build_affinity_matrix(&X).unwrap();
    //     assert_eq!(W.dim(), (3, 3));
    //
    //     // Check that each row has exactly n_neighbors non-zero entries
    //     for i in 0..3 {
    //         let non_zero_count = W.row(i).iter().filter(|&&x| x > 0.0).count();
    //         assert!(non_zero_count <= 2); // At most 2 (should be 1 for n_neighbors=1, but symmetric)
    //     }
    // }

    #[test]
    fn test_enhanced_self_training() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let est = EnhancedSelfTraining::new()
            .threshold(0.6)
            .confidence_method("entropy".to_string())
            .max_iter(5);
        let fitted = est.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_co_training() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let ct = CoTraining::new()
            .view1_features(vec![0, 1])
            .view2_features(vec![2, 3])
            .p(1)
            .n(1)
            .max_iter(5);
        let fitted = ct.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_tri_training() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let tt = TriTraining::new().max_iter(5).theta(0.2);
        let fitted = tt.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_knn_graph() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let W = knn_graph(&X, 1, "connectivity").unwrap();
        assert_eq!(W.dim(), (3, 3));

        // Check that each row has at most n_neighbors non-zero entries
        for i in 0..3 {
            let non_zero_count = W.row(i).iter().filter(|&&x| x > 0.0).count();
            assert!(non_zero_count <= 1);
        }
    }

    #[test]
    fn test_epsilon_graph() {
        let X = array![[1.0, 2.0], [1.1, 2.1], [5.0, 6.0]];

        let W = epsilon_graph(&X, 1.0, "connectivity").unwrap();
        assert_eq!(W.dim(), (3, 3));

        // Points 0 and 1 should be connected (distance < 1.0)
        assert!(W[[0, 1]] > 0.0 || W[[1, 0]] > 0.0);
    }

    #[test]
    fn test_graph_laplacian() {
        let W = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];

        let L = graph_laplacian(&W, false).unwrap();
        assert_eq!(L.dim(), (3, 3));

        // Check Laplacian properties
        assert_eq!(L[[0, 0]], 1.0); // degree of node 0
        assert_eq!(L[[1, 1]], 2.0); // degree of node 1
        assert_eq!(L[[0, 1]], -1.0); // -adjacency
    }

    #[test]
    fn test_democratic_co_learning() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let dcl = DemocraticCoLearning::new()
            .views(vec![vec![0, 1], vec![2, 3], vec![4, 5]])
            .k_add(1)
            .min_agreement(2)
            .max_iter(5);
        let fitted = dcl.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_harmonic_functions() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let hf = HarmonicFunctions::new()
            .kernel("rbf".to_string())
            .gamma(20.0)
            .max_iter(100);
        let fitted = hf.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (4, 2));
    }

    #[test]
    fn test_local_global_consistency() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let lgc = LocalGlobalConsistency::new()
            .kernel("rbf".to_string())
            .gamma(20.0)
            .alpha(0.99)
            .max_iter(100);
        let fitted = lgc.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (4, 2));
    }

    #[test]
    fn test_manifold_regularization() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let mr = ManifoldRegularization::new()
            .lambda_a(0.01)
            .lambda_i(0.1)
            .kernel("rbf".to_string())
            .gamma(1.0)
            .max_iter(100);
        let fitted = mr.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_semi_supervised_gmm() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let gmm = SemiSupervisedGMM::new()
            .n_components(2)
            .max_iter(50)
            .labeled_weight(10.0);
        let fitted = gmm.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (4, 2));
    }

    #[test]
    fn test_multi_view_co_training() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        ];
        let y = array![0, 1, -1, -1, -1, -1]; // -1 indicates unlabeled

        let mvct = MultiViewCoTraining::new()
            .views(vec![vec![0, 1], vec![2, 3], vec![4, 5]])
            .k_add(1)
            .confidence_threshold(0.5)
            .max_iter(5);
        let fitted = mvct.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 6);

        // Check that labeled samples maintain their labels
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_semi_supervised_naive_bayes() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let nb = SemiSupervisedNaiveBayes::new()
            .alpha(1.0)
            .max_iter(50)
            .class_weight(1.0);
        let fitted = nb.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let sum: f64 = probas.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_random_walk_laplacian() {
        let W = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];

        let L_rw = random_walk_laplacian(&W).unwrap();
        assert_eq!(L_rw.dim(), (3, 3));

        // Check that L_rw has 1s on the diagonal
        assert!((L_rw[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((L_rw[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((L_rw[[2, 2]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diffusion_matrix() {
        let W = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];

        let P = diffusion_matrix(&W, 2).unwrap();
        assert_eq!(P.dim(), (3, 3));

        // Check that probabilities are non-negative
        for i in 0..3 {
            for j in 0..3 {
                assert!(P[[i, j]] >= 0.0);
            }
        }
    }

    #[test]
    fn test_adaptive_knn_graph() {
        let X = array![[1.0, 2.0], [1.1, 2.1], [5.0, 6.0]];

        let W = adaptive_knn_graph(&X, "connectivity").unwrap();
        assert_eq!(W.dim(), (3, 3));

        // Check symmetry
        assert_eq!(W[[0, 1]], W[[1, 0]]);
        assert_eq!(W[[0, 2]], W[[2, 0]]);
        assert_eq!(W[[1, 2]], W[[2, 1]]);
    }

    #[test]
    fn test_sparsify_graph() {
        let W = array![
            [0.0, 0.8, 0.2, 0.1],
            [0.8, 0.0, 0.9, 0.3],
            [0.2, 0.9, 0.0, 0.7],
            [0.1, 0.3, 0.7, 0.0]
        ];

        let W_sparse = sparsify_graph(&W, 0.5).unwrap();
        assert_eq!(W_sparse.dim(), (4, 4));

        // Count non-zero edges in original and sparse graphs
        let original_edges = W.iter().filter(|&&x| x > 0.0).count();
        let sparse_edges = W_sparse.iter().filter(|&&x| x > 0.0).count();

        // Sparse graph should have fewer edges
        assert!(sparse_edges <= original_edges);
    }

    #[test]
    fn test_spectral_clustering() {
        let W = array![
            [0.0, 1.0, 0.1, 0.0, 0.0],
            [1.0, 0.0, 0.2, 0.0, 0.0],
            [0.1, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0]
        ];

        let labels = spectral_clustering(&W, 2, true, Some(42)).unwrap();
        assert_eq!(labels.len(), 5);

        // Check that labels are valid
        for &label in labels.iter() {
            assert!(label >= 0 && label < 2);
        }
    }

    #[test]
    fn test_spectral_embedding() {
        let W = array![[0.0, 1.0, 0.1], [1.0, 0.0, 0.2], [0.1, 0.2, 0.0]];

        let embedding = spectral_embedding(&W, 2, true).unwrap();
        assert_eq!(embedding.dim(), (3, 2));
    }

    // Robustness tests with label noise
    #[test]
    fn test_label_propagation_robustness() {
        use scirs2_core::random::Random;

        let mut rng = Random::seed(42);

        // Generate synthetic dataset - create it manually to avoid distribution conflicts
        let mut X = Array2::<f64>::zeros((50, 5));
        for i in 0..50 {
            for j in 0..5 {
                X[(i, j)] = rng.random_range(-1.0, 1.0);
            }
        }
        let y_true = array![
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1
        ];

        // Test without noise
        let lp = LabelPropagation::new()
            .kernel("rbf".to_string())
            .gamma(20.0);
        let fitted_clean = lp.fit(&X.view(), &y_true.view()).unwrap();
        let pred_clean = fitted_clean.predict(&X.view()).unwrap();

        // Test with label noise (flip some labels)
        let mut y_noisy = y_true.clone();
        y_noisy[0] = 1; // Flip first label
        y_noisy[2] = 1; // Flip third label

        let lp_noisy = LabelPropagation::new()
            .kernel("rbf".to_string())
            .gamma(20.0);
        let fitted_noisy = lp_noisy.fit(&X.view(), &y_noisy.view()).unwrap();
        let pred_noisy = fitted_noisy.predict(&X.view()).unwrap();

        // Calculate robustness (predictions should not change dramatically)
        let different = pred_clean
            .iter()
            .zip(pred_noisy.iter())
            .filter(|(a, b)| a != b)
            .count();
        let robustness = 1.0 - (different as f64 / pred_clean.len() as f64);

        assert!(
            robustness > 0.6,
            "Label propagation should be somewhat robust to label noise"
        );
    }

    #[test]
    fn test_self_training_robustness() {
        use scirs2_core::random::Random;

        let mut rng = Random::seed(42);

        // Generate synthetic dataset - create it manually to avoid distribution conflicts
        let mut X = Array2::<f64>::zeros((30, 4));
        for i in 0..30 {
            for j in 0..4 {
                X[(i, j)] = rng.random_range(-1.0, 1.0);
            }
        }
        let y_clean = array![
            0, 1, 0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1
        ];

        // Test without noise
        let stc_clean = SelfTrainingClassifier::new().threshold(0.8).max_iter(10);
        let fitted_clean = stc_clean.fit(&X.view(), &y_clean.view()).unwrap();
        let pred_clean = fitted_clean.predict(&X.view()).unwrap();

        // Test with noise
        let mut y_noisy = y_clean.clone();
        y_noisy[1] = 0; // Flip a label

        let stc_noisy = SelfTrainingClassifier::new().threshold(0.8).max_iter(10);
        let fitted_noisy = stc_noisy.fit(&X.view(), &y_noisy.view()).unwrap();
        let pred_noisy = fitted_noisy.predict(&X.view()).unwrap();

        // Check that algorithm still produces valid predictions
        assert!(
            pred_clean.iter().all(|&p| p >= 0 && p <= 1),
            "Clean predictions should be valid"
        );
        assert!(
            pred_noisy.iter().all(|&p| p >= 0 && p <= 1),
            "Noisy predictions should be valid"
        );
    }

    #[test]
    fn test_co_training_robustness() {
        use scirs2_core::random::Random;

        let mut rng = Random::seed(42);

        // Generate synthetic dataset - create it manually to avoid distribution conflicts
        let mut X = Array2::<f64>::zeros((20, 6));
        for i in 0..20 {
            for j in 0..6 {
                X[(i, j)] = rng.random_range(-1.0, 1.0);
            }
        }
        let y_clean =
            array![0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];

        // Test clean performance
        let ct_clean = CoTraining::new()
            .view1_features(vec![0, 1, 2])
            .view2_features(vec![3, 4, 5])
            .p(1)
            .n(1)
            .max_iter(5);
        let fitted_clean = ct_clean.fit(&X.view(), &y_clean.view()).unwrap();
        let pred_clean = fitted_clean.predict(&X.view()).unwrap();

        // Test with noise
        let mut y_noisy = y_clean.clone();
        y_noisy[0] = 1; // Flip a label

        let ct_noisy = CoTraining::new()
            .view1_features(vec![0, 1, 2])
            .view2_features(vec![3, 4, 5])
            .p(1)
            .n(1)
            .max_iter(5);
        let fitted_noisy = ct_noisy.fit(&X.view(), &y_noisy.view()).unwrap();
        let pred_noisy = fitted_noisy.predict(&X.view()).unwrap();

        // Check that predictions are valid
        assert!(
            pred_clean.iter().all(|&p| p >= 0 && p <= 1),
            "Clean predictions should be valid"
        );
        assert!(
            pred_noisy.iter().all(|&p| p >= 0 && p <= 1),
            "Noisy predictions should be valid"
        );
    }

    // Label efficiency tests
    #[test]
    fn test_label_efficiency_comparison() {
        use scirs2_core::random::Random;
        let mut rng = Random::seed(42);

        // Generate synthetic dataset - create it manually to avoid distribution conflicts
        let mut X = Array2::<f64>::zeros((40, 5));
        for i in 0..40 {
            for j in 0..5 {
                X[(i, j)] = rng.random_range(-1.0, 1.0);
            }
        }

        // Test with different labeling ratios
        let small_labeled = array![
            0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        ];

        let large_labeled = array![
            0, 1, 0, 1, 0, 1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        ];

        // Test label propagation with different label amounts
        let lp_small = LabelPropagation::new()
            .kernel("rbf".to_string())
            .gamma(20.0);
        let fitted_small = lp_small.fit(&X.view(), &small_labeled.view()).unwrap();
        let pred_small = fitted_small.predict(&X.view()).unwrap();

        let lp_large = LabelPropagation::new()
            .kernel("rbf".to_string())
            .gamma(20.0);
        let fitted_large = lp_large.fit(&X.view(), &large_labeled.view()).unwrap();
        let pred_large = fitted_large.predict(&X.view()).unwrap();

        // Both should produce valid predictions
        assert!(
            pred_small.iter().all(|&p| p >= 0 && p <= 1),
            "Small labeled predictions should be valid"
        );
        assert!(
            pred_large.iter().all(|&p| p >= 0 && p <= 1),
            "Large labeled predictions should be valid"
        );
    }

    // Convergence tests
    #[test]
    fn test_algorithm_convergence() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let y = array![0, 1, -1, -1, -1];

        // Test that harmonic functions converge
        let hf = HarmonicFunctions::new()
            .kernel("rbf".to_string())
            .gamma(20.0)
            .max_iter(100);
        let fitted = hf.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        // Check that predictions are stable and valid
        assert!(
            predictions.iter().all(|&p| p >= 0 && p <= 1),
            "Predictions should be stable and valid"
        );

        // Test that local-global consistency converges
        let lgc = LocalGlobalConsistency::new()
            .kernel("rbf".to_string())
            .gamma(20.0)
            .alpha(0.99)
            .max_iter(100);
        let fitted_lgc = lgc.fit(&X.view(), &y.view()).unwrap();
        let predictions_lgc = fitted_lgc.predict(&X.view()).unwrap();

        assert!(
            predictions_lgc.iter().all(|&p| p >= 0 && p <= 1),
            "LGC predictions should be stable and valid"
        );
    }
}
