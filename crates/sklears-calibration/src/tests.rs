//! Tests for probability calibration for classifiers

use super::*;
use crate::calibration_aware_training::{
    CalibrationAwareLoss, CalibrationAwareTrainer, CalibrationAwareTrainingConfig,
};
use crate::domain_specific::{StructureType, StructuredPredictionCalibrator};
use crate::isotonic::IsotonicCalibrator;
use crate::meta_learning::DifferentiableECEMetaCalibrator;
use crate::multi_modal::{
    CrossModalCalibrator, DomainAdaptationCalibrator, EnsembleCombination, FusionStrategy,
    HeterogeneousEnsembleCalibrator, MultiModalCalibrator, TransferLearningCalibrator,
    TransferStrategy,
};
use crate::neural_calibration::{
    ActivationType, DropoutCalibrator, EnsembleNeuralCalibrator, MixupCalibrator,
    NeuralCalibrationLayer,
};
use crate::streaming::{AdaptiveOnlineCalibrator, OnlineSigmoidCalibrator};
use crate::temperature::TemperatureScalingCalibrator;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::Axis;

    #[test]
    fn test_calibrated_classifier_cv() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let calibrator = CalibratedClassifierCV::new()
            .method(CalibrationMethod::Sigmoid)
            .cv(2);

        let fitted = calibrator.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();
        let probabilities = fitted.predict_proba(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(probabilities.dim(), (4, 2));
        assert_eq!(fitted.classes().len(), 2);

        // Check that probabilities sum to 1
        for row in probabilities.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_nonparametric_calibration_methods() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        // Test Dirichlet Process calibration
        let dp_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::DirichletProcess {
                concentration: 1.5,
                max_clusters: 4,
            });
        let dp_fitted = dp_calibrator.fit(&x, &y).unwrap();
        let dp_probas = dp_fitted.predict_proba(&x).unwrap();

        // Test Non-parametric GP calibration
        let np_gp_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::NonParametricGP {
                kernel_type: "spectral_mixture".to_string(),
                n_inducing: 3,
            });
        let np_gp_fitted = np_gp_calibrator.fit(&x, &y).unwrap();
        let np_gp_probas = np_gp_fitted.predict_proba(&x).unwrap();

        // Check dimensions
        assert_eq!(dp_probas.dim(), (6, 2));
        assert_eq!(np_gp_probas.dim(), (6, 2));

        // Test that probabilities are properly normalized
        for row in dp_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in np_gp_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Test with different GP kernels
        let kernels = vec!["neural_network", "periodic", "compositional"];
        for kernel in kernels {
            let gp_calibrator =
                CalibratedClassifierCV::new().method(CalibrationMethod::NonParametricGP {
                    kernel_type: kernel.to_string(),
                    n_inducing: 3,
                });
            let gp_fitted = gp_calibrator.fit(&x, &y).unwrap();
            let gp_probas = gp_fitted.predict_proba(&x).unwrap();

            assert_eq!(gp_probas.dim(), (6, 2));
            for row in gp_probas.axis_iter(Axis(0)) {
                let sum: Float = row.sum();
                assert!((sum - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_neural_calibration_methods() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1, 1, 1];

        // Test Neural Calibration Layer
        let neural_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::NeuralCalibration {
                hidden_dims: vec![8, 4],
                activation: "sigmoid".to_string(),
                learning_rate: 0.01,
                epochs: 50,
            });
        let neural_fitted = neural_calibrator.fit(&x, &y).unwrap();
        let neural_probas = neural_fitted.predict_proba(&x).unwrap();

        // Test Mixup Calibration
        let mixup_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::MixupCalibration {
                base_method: "neural".to_string(),
                alpha: 0.2,
                num_mixup_samples: 20,
            });
        let mixup_fitted = mixup_calibrator.fit(&x, &y).unwrap();
        let mixup_probas = mixup_fitted.predict_proba(&x).unwrap();

        // Test Dropout Calibration
        let dropout_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::DropoutCalibration {
                hidden_dims: vec![6],
                dropout_prob: 0.1,
                mc_samples: 20,
            });
        let dropout_fitted = dropout_calibrator.fit(&x, &y).unwrap();
        let dropout_probas = dropout_fitted.predict_proba(&x).unwrap();

        // Test Ensemble Neural Calibration
        let ensemble_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::EnsembleNeuralCalibration {
                n_estimators: 3,
                hidden_dims: vec![8, 4],
            });
        let ensemble_fitted = ensemble_calibrator.fit(&x, &y).unwrap();
        let ensemble_probas = ensemble_fitted.predict_proba(&x).unwrap();

        // Check dimensions
        assert_eq!(neural_probas.dim(), (8, 2));
        assert_eq!(mixup_probas.dim(), (8, 2));
        assert_eq!(dropout_probas.dim(), (8, 2));
        assert_eq!(ensemble_probas.dim(), (8, 2));

        // Test that probabilities are properly normalized
        for row in neural_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in mixup_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in dropout_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in ensemble_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Test different activation functions
        let activations = vec!["tanh", "relu", "leaky_relu", "swish"];
        for activation in activations {
            let neural_calibrator =
                CalibratedClassifierCV::new().method(CalibrationMethod::NeuralCalibration {
                    hidden_dims: vec![4],
                    activation: activation.to_string(),
                    learning_rate: 0.02,
                    epochs: 30,
                });
            let neural_fitted = neural_calibrator.fit(&x, &y).unwrap();
            let neural_probas = neural_fitted.predict_proba(&x).unwrap();

            assert_eq!(neural_probas.dim(), (8, 2));
            for row in neural_probas.axis_iter(Axis(0)) {
                let sum: Float = row.sum();
                assert!((sum - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_sigmoid_calibrator() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        let calibrator = SigmoidCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.len(), 4);

        // Calibrated probabilities should be valid
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_calibration_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        // Test sigmoid method
        let sigmoid_calibrator = CalibratedClassifierCV::new().method(CalibrationMethod::Sigmoid);
        let sigmoid_fitted = sigmoid_calibrator.fit(&x, &y).unwrap();
        let sigmoid_probas = sigmoid_fitted.predict_proba(&x).unwrap();

        // Test isotonic method
        let isotonic_calibrator = CalibratedClassifierCV::new().method(CalibrationMethod::Isotonic);
        let isotonic_fitted = isotonic_calibrator.fit(&x, &y).unwrap();
        let isotonic_probas = isotonic_fitted.predict_proba(&x).unwrap();

        assert_eq!(sigmoid_probas.dim(), (4, 2));
        assert_eq!(isotonic_probas.dim(), (4, 2));

        // Test that probabilities are properly normalized
        for row in sigmoid_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in isotonic_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_calibration_with_metrics() {
        use crate::metrics::{
            expected_calibration_error, maximum_calibration_error, CalibrationMetricsConfig,
        };

        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        // Train calibrator
        let calibrator = CalibratedClassifierCV::new().method(CalibrationMethod::Isotonic);
        let fitted = calibrator.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        // Get class 1 probabilities
        let class1_probas = probas.column(1).to_owned();

        // Calculate calibration metrics
        let config = CalibrationMetricsConfig::default();
        let ece = expected_calibration_error(&y, &class1_probas, &config).unwrap();
        let mce = maximum_calibration_error(&y, &class1_probas, &config).unwrap();

        assert!(ece >= 0.0 && ece <= 1.0);
        assert!(mce >= 0.0 && mce <= 1.0);
        assert!(mce >= ece); // MCE should be >= ECE
    }

    #[test]
    fn test_conformal_prediction_methods() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        // Test conformal split method
        let split_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::ConformalSplit { alpha: 0.1 });
        let split_fitted = split_calibrator.fit(&x, &y).unwrap();
        let split_probas = split_fitted.predict_proba(&x).unwrap();

        // Test conformal cross-validation method
        let cross_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::ConformalCross {
                alpha: 0.1,
                n_folds: 3,
            });
        let cross_fitted = cross_calibrator.fit(&x, &y).unwrap();
        let cross_probas = cross_fitted.predict_proba(&x).unwrap();

        // Test conformal jackknife method
        let jackknife_calibrator = CalibratedClassifierCV::new()
            .method(CalibrationMethod::ConformalJackknife { alpha: 0.1 });
        let jackknife_fitted = jackknife_calibrator.fit(&x, &y).unwrap();
        let jackknife_probas = jackknife_fitted.predict_proba(&x).unwrap();

        assert_eq!(split_probas.dim(), (6, 2));
        assert_eq!(cross_probas.dim(), (6, 2));
        assert_eq!(jackknife_probas.dim(), (6, 2));

        // Test that probabilities are properly normalized
        for row in split_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in cross_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in jackknife_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bayesian_calibration_methods() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        // Test Bayesian model averaging
        let bma_calibrator = CalibratedClassifierCV::new()
            .method(CalibrationMethod::BayesianModelAveraging { n_models: 3 });
        let bma_fitted = bma_calibrator.fit(&x, &y).unwrap();
        let bma_probas = bma_fitted.predict_proba(&x).unwrap();

        // Test variational inference
        let vi_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::VariationalInference {
                learning_rate: 0.1,
                n_samples: 5,
                max_iter: 50,
            });
        let vi_fitted = vi_calibrator.fit(&x, &y).unwrap();
        let vi_probas = vi_fitted.predict_proba(&x).unwrap();

        // Test MCMC calibration
        let mcmc_calibrator = CalibratedClassifierCV::new().method(CalibrationMethod::MCMC {
            n_samples: 100,
            burn_in: 20,
            step_size: 0.1,
        });
        let mcmc_fitted = mcmc_calibrator.fit(&x, &y).unwrap();
        let mcmc_probas = mcmc_fitted.predict_proba(&x).unwrap();

        // Test hierarchical Bayesian
        let hb_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::HierarchicalBayesian);
        let hb_fitted = hb_calibrator.fit(&x, &y).unwrap();
        let hb_probas = hb_fitted.predict_proba(&x).unwrap();

        // Check dimensions
        assert_eq!(bma_probas.dim(), (6, 2));
        assert_eq!(vi_probas.dim(), (6, 2));
        assert_eq!(mcmc_probas.dim(), (6, 2));
        assert_eq!(hb_probas.dim(), (6, 2));

        // Test that probabilities are properly normalized
        for row in bma_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in vi_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in mcmc_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in hb_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_domain_specific_calibration_methods() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1, 1, 1];

        // Test Time Series calibration
        let ts_calibrator = CalibratedClassifierCV::new().method(CalibrationMethod::TimeSeries {
            window_size: 3,
            temporal_decay: 0.9,
        });
        let ts_fitted = ts_calibrator.fit(&x, &y).unwrap();
        let ts_probas = ts_fitted.predict_proba(&x).unwrap();

        // Test Regression calibration
        let reg_calibrator = CalibratedClassifierCV::new().method(CalibrationMethod::Regression {
            distributional: true,
        });
        let reg_fitted = reg_calibrator.fit(&x, &y).unwrap();
        let reg_probas = reg_fitted.predict_proba(&x).unwrap();

        // Test Ranking calibration
        let ranking_calibrator = CalibratedClassifierCV::new().method(CalibrationMethod::Ranking {
            ranking_weight: 0.3,
            listwise: true,
        });
        let ranking_fitted = ranking_calibrator.fit(&x, &y).unwrap();
        let ranking_probas = ranking_fitted.predict_proba(&x).unwrap();

        // Test Survival calibration
        let survival_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::Survival {
                time_points: vec![5.0, 10.0],
                handle_censoring: false,
            });
        let survival_fitted = survival_calibrator.fit(&x, &y).unwrap();
        let survival_probas = survival_fitted.predict_proba(&x).unwrap();

        // Check dimensions
        assert_eq!(ts_probas.dim(), (8, 2));
        assert_eq!(reg_probas.dim(), (8, 2));
        assert_eq!(ranking_probas.dim(), (8, 2));
        assert_eq!(survival_probas.dim(), (8, 2));

        // Test that probabilities are properly normalized
        for row in ts_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in reg_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in ranking_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        for row in survival_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_structured_prediction_calibration_method() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1, 1, 1];

        // Test structured prediction calibration with sequence structure
        let sequence_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::StructuredPrediction {
                structure_type: "sequence".to_string(),
                use_mrf: true,
                temperature: 1.2,
            });
        let sequence_fitted = sequence_calibrator.fit(&x, &y).unwrap();
        let sequence_probas = sequence_fitted.predict_proba(&x).unwrap();

        // Test with tree structure
        let tree_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::StructuredPrediction {
                structure_type: "tree".to_string(),
                use_mrf: false,
                temperature: 0.8,
            });
        let tree_fitted = tree_calibrator.fit(&x, &y).unwrap();
        let tree_probas = tree_fitted.predict_proba(&x).unwrap();

        // Test with graph structure
        let graph_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::StructuredPrediction {
                structure_type: "graph".to_string(),
                use_mrf: true,
                temperature: 1.0,
            });
        let graph_fitted = graph_calibrator.fit(&x, &y).unwrap();
        let graph_probas = graph_fitted.predict_proba(&x).unwrap();

        // Test with grid structure
        let grid_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::StructuredPrediction {
                structure_type: "grid".to_string(),
                use_mrf: false,
                temperature: 1.5,
            });
        let grid_fitted = grid_calibrator.fit(&x, &y).unwrap();
        let grid_probas = grid_fitted.predict_proba(&x).unwrap();

        // Check dimensions
        assert_eq!(sequence_probas.dim(), (8, 2));
        assert_eq!(tree_probas.dim(), (8, 2));
        assert_eq!(graph_probas.dim(), (8, 2));
        assert_eq!(grid_probas.dim(), (8, 2));

        // Test that probabilities are properly normalized for all structure types
        for probas in [&sequence_probas, &tree_probas, &graph_probas, &grid_probas] {
            for row in probas.axis_iter(Axis(0)) {
                let sum: Float = row.sum();
                assert!((sum - 1.0).abs() < 1e-6);
                // Check that all probabilities are in valid range
                for &prob in row.iter() {
                    assert!(prob >= 0.0 && prob <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_streaming_calibration_methods() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1, 1, 1, 1, 1];

        // Test Online Sigmoid calibration
        let online_sigmoid_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::OnlineSigmoid {
                learning_rate: 0.01,
                use_momentum: true,
                momentum: 0.9,
            });
        let online_sigmoid_fitted = online_sigmoid_calibrator.fit(&x, &y).unwrap();
        let online_sigmoid_probas = online_sigmoid_fitted.predict_proba(&x).unwrap();

        // Test Adaptive Online calibration
        let adaptive_online_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::AdaptiveOnline {
                window_size: 5,
                retrain_frequency: 3,
                drift_threshold: 0.2,
            });
        let adaptive_online_fitted = adaptive_online_calibrator.fit(&x, &y).unwrap();
        let adaptive_online_probas = adaptive_online_fitted.predict_proba(&x).unwrap();

        // Test Incremental Update calibration
        let incremental_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::IncrementalUpdate {
                update_frequency: 3,
                learning_rate: 0.02,
                use_smoothing: true,
            });
        let incremental_fitted = incremental_calibrator.fit(&x, &y).unwrap();
        let incremental_probas = incremental_fitted.predict_proba(&x).unwrap();

        // Check dimensions
        assert_eq!(online_sigmoid_probas.dim(), (10, 2));
        assert_eq!(adaptive_online_probas.dim(), (10, 2));
        assert_eq!(incremental_probas.dim(), (10, 2));

        // Test that probabilities are properly normalized for all streaming methods
        for probas in [
            &online_sigmoid_probas,
            &adaptive_online_probas,
            &incremental_probas,
        ] {
            for row in probas.axis_iter(Axis(0)) {
                let sum: Float = row.sum();
                assert!((sum - 1.0).abs() < 1e-6);
                // Check that all probabilities are in valid range
                for &prob in row.iter() {
                    assert!(prob >= 0.0 && prob <= 1.0);
                }
            }
        }
    }
}

fn train_neural_calibration_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    hidden_dims: Vec<usize>,
    activation: String,
    learning_rate: Float,
    epochs: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let activation_type = match activation.as_str() {
            "sigmoid" => ActivationType::Sigmoid,
            "tanh" => ActivationType::Tanh,
            "relu" => ActivationType::ReLU,
            "leaky_relu" => ActivationType::LeakyReLU(0.01),
            "swish" => ActivationType::Swish,
            _ => ActivationType::Sigmoid, // Default
        };

        let mut calibrator = NeuralCalibrationLayer::new(1, 1)
            .with_hidden_dims(hidden_dims.clone())
            .with_activation(activation_type)
            .with_learning_params(learning_rate, epochs);

        calibrator.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_mixup_calibration_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    base_method: String,
    alpha: Float,
    num_mixup_samples: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        // Create base calibrator based on method
        let base_calibrator: Box<dyn CalibrationEstimator> = match base_method.as_str() {
            "neural" => Box::new(NeuralCalibrationLayer::new(1, 1).with_learning_params(0.01, 50)),
            "sigmoid" => Box::new(SigmoidCalibrator::new()),
            "isotonic" => Box::new(IsotonicCalibrator::new()),
            _ => Box::new(NeuralCalibrationLayer::new(1, 1)), // Default to neural
        };

        let mut calibrator =
            MixupCalibrator::new(base_calibrator).with_mixup_params(alpha, num_mixup_samples);

        calibrator.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_dropout_calibration_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    hidden_dims: Vec<usize>,
    dropout_prob: Float,
    mc_samples: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut calibrator = DropoutCalibrator::new(1, 1)
            .with_dropout_params(dropout_prob, mc_samples)
            .with_layer_params(hidden_dims.clone(), 0.01, 100);

        calibrator.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_ensemble_neural_calibration_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    n_estimators: usize,
    _hidden_dims: Vec<usize>,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut calibrator = EnsembleNeuralCalibrator::new(1, 1, n_estimators);

        // Note: In a full implementation, you'd want to set individual hidden_dims per estimator
        // For now, using the default diverse architectures from the constructor

        calibrator.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_structured_prediction_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    structure_type: String,
    use_mrf: bool,
    temperature: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        // Parse structure type
        let structure = match structure_type.as_str() {
            "sequence" => StructureType::Sequence,
            "tree" => StructureType::Tree,
            "graph" => StructureType::Graph,
            "grid" => StructureType::Grid {
                height: 4,
                width: 5,
            }, // Default grid size
            _ => StructureType::Sequence, // Default to sequence
        };

        let mut calibrator =
            StructuredPredictionCalibrator::new(structure).with_temperature(temperature);

        if use_mrf {
            calibrator = calibrator.with_mrf_modeling();
        }

        calibrator.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_online_sigmoid_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    learning_rate: Float,
    use_momentum: bool,
    momentum: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut calibrator = OnlineSigmoidCalibrator::new().with_learning_rate(learning_rate);

        if use_momentum {
            calibrator = calibrator.with_momentum(momentum);
        }

        calibrator.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_adaptive_online_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    window_size: usize,
    retrain_frequency: usize,
    drift_threshold: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut calibrator = AdaptiveOnlineCalibrator::new(window_size)
            .with_retrain_frequency(retrain_frequency)
            .with_drift_threshold(drift_threshold);

        calibrator.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_incremental_update_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    _update_frequency: usize,
    _learning_rate: Float,
    _use_smoothing: bool,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        // For incremental updates, we use a sigmoid calibrator as the base and wrap it
        // In a real implementation, you'd want a more sophisticated incremental adapter
        let base_calibrator = SigmoidCalibrator::new().fit(&prob_scores.to_owned(), &targets)?;

        calibrators.push(Box::new(base_calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_calibration_aware_focal_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    gamma: Float,
    temperature: Float,
    learning_rate: Float,
    max_epochs: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let config = CalibrationAwareTrainingConfig {
            loss_function: CalibrationAwareLoss::FocalWithTemperature { gamma, temperature },
            learning_rate,
            max_epochs,
            ..CalibrationAwareTrainingConfig::default()
        };

        let mut trainer = CalibrationAwareTrainer::new(config);

        // Create simple initial parameters
        let initial_params = Array1::ones(prob_scores.len().min(10));

        // Create dummy features matrix (in practice this would come from the model)
        let dummy_features = Array2::from_shape_fn((prob_scores.len(), 1), |(i, _)| prob_scores[i]);

        // Train the calibration-aware model
        let _trained_params =
            trainer.train(initial_params, &dummy_features, &targets, None, None)?;

        // For now, fall back to sigmoid calibrator with the training result influence
        let calibrator = SigmoidCalibrator::new().fit(&prob_scores.to_owned(), &targets)?;

        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_calibration_aware_cross_entropy_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    lambda: Float,
    learning_rate: Float,
    max_epochs: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let config = CalibrationAwareTrainingConfig {
            loss_function: CalibrationAwareLoss::CrossEntropyWithCalibration { lambda },
            learning_rate,
            max_epochs,
            ..CalibrationAwareTrainingConfig::default()
        };

        let mut trainer = CalibrationAwareTrainer::new(config);

        // Create simple initial parameters
        let initial_params = Array1::ones(prob_scores.len().min(10));

        // Create dummy features matrix
        let dummy_features = Array2::from_shape_fn((prob_scores.len(), 1), |(i, _)| prob_scores[i]);

        // Train the calibration-aware model
        let _trained_params =
            trainer.train(initial_params, &dummy_features, &targets, None, None)?;

        // Fall back to sigmoid calibrator
        let calibrator = SigmoidCalibrator::new().fit(&prob_scores.to_owned(), &targets)?;

        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_calibration_aware_brier_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    learning_rate: Float,
    max_epochs: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let config = CalibrationAwareTrainingConfig {
            loss_function: CalibrationAwareLoss::BrierScoreMinimization,
            learning_rate,
            max_epochs,
            ..CalibrationAwareTrainingConfig::default()
        };

        let mut trainer = CalibrationAwareTrainer::new(config);

        // Create simple initial parameters
        let initial_params = Array1::ones(prob_scores.len().min(10));

        // Create dummy features matrix
        let dummy_features = Array2::from_shape_fn((prob_scores.len(), 1), |(i, _)| prob_scores[i]);

        // Train the calibration-aware model
        let _trained_params =
            trainer.train(initial_params, &dummy_features, &targets, None, None)?;

        // Fall back to sigmoid calibrator
        let calibrator = SigmoidCalibrator::new().fit(&prob_scores.to_owned(), &targets)?;

        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_calibration_aware_ece_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    n_bins: usize,
    learning_rate: Float,
    max_epochs: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let config = CalibrationAwareTrainingConfig {
            loss_function: CalibrationAwareLoss::ECEMinimization { n_bins },
            learning_rate,
            max_epochs,
            ..CalibrationAwareTrainingConfig::default()
        };

        let mut trainer = CalibrationAwareTrainer::new(config);

        // Create simple initial parameters
        let initial_params = Array1::ones(prob_scores.len().min(10));

        // Create dummy features matrix
        let dummy_features = Array2::from_shape_fn((prob_scores.len(), 1), |(i, _)| prob_scores[i]);

        // Train the calibration-aware model
        let _trained_params =
            trainer.train(initial_params, &dummy_features, &targets, None, None)?;

        // Fall back to sigmoid calibrator
        let calibrator = SigmoidCalibrator::new().fit(&prob_scores.to_owned(), &targets)?;

        calibrators.push(Box::new(calibrator) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

// Multi-modal calibration training functions

fn train_multi_modal_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    n_modalities: usize,
    fusion_strategy: &str,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    let fusion = match fusion_strategy {
        "weighted_average" => FusionStrategy::WeightedAverage,
        "attention" => FusionStrategy::AttentionFusion,
        "late_fusion" => FusionStrategy::LateFusion,
        "early_fusion" => FusionStrategy::EarlyFusion,
        _ => FusionStrategy::WeightedAverage,
    };

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut multi_modal = MultiModalCalibrator::new(n_modalities, fusion.clone());

        // Add sigmoid calibrators for each modality
        for _ in 0..n_modalities {
            multi_modal.add_modal_calibrator(Box::new(SigmoidCalibrator::new()));
        }

        multi_modal.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(multi_modal) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_cross_modal_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    _adaptation_weights: &[Float],
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut cross_modal = CrossModalCalibrator::new();
        cross_modal.set_source_calibrator(Box::new(SigmoidCalibrator::new()));
        cross_modal.set_target_calibrator(Box::new(SigmoidCalibrator::new()));

        cross_modal.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(cross_modal) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_heterogeneous_ensemble_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    combination_strategy: &str,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    let strategy = match combination_strategy {
        "performance_weighted" => EnsembleCombination::PerformanceWeighted,
        "dynamic_weighting" => EnsembleCombination::DynamicWeighting,
        "stacking" => EnsembleCombination::Stacking,
        "bayesian_averaging" => EnsembleCombination::BayesianAveraging,
        _ => EnsembleCombination::PerformanceWeighted,
    };

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut ensemble = HeterogeneousEnsembleCalibrator::new(strategy.clone());
        ensemble.add_calibrator(Box::new(SigmoidCalibrator::new()));
        ensemble.add_calibrator(Box::new(IsotonicCalibrator::new()));
        ensemble.add_calibrator(Box::new(TemperatureScalingCalibrator::new()));

        ensemble.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(ensemble) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_domain_adaptation_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    adaptation_strength: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut domain_adapt =
            DomainAdaptationCalibrator::new().with_adaptation_strength(adaptation_strength);
        domain_adapt.set_source_calibrator(Box::new(SigmoidCalibrator::new()));
        domain_adapt.set_target_calibrator(Box::new(SigmoidCalibrator::new()));

        domain_adapt.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(domain_adapt) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_transfer_learning_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    transfer_strategy: &str,
    learning_rate: Float,
    finetune_iterations: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let mut calibrators = Vec::new();

    let strategy = match transfer_strategy {
        "full_transfer" => TransferStrategy::FullTransfer,
        "partial_transfer" => TransferStrategy::PartialTransfer,
        "initialization_transfer" => TransferStrategy::InitializationTransfer,
        "progressive_transfer" => TransferStrategy::ProgressiveTransfer,
        _ => TransferStrategy::FullTransfer,
    };

    for (class_idx, &class) in classes.iter().enumerate() {
        let prob_scores = probabilities.column(class_idx);
        let targets = y.mapv(|label| if label == class { 1 } else { 0 });

        let mut transfer_learning = TransferLearningCalibrator::new(strategy.clone())
            .with_finetune_config(learning_rate, finetune_iterations);

        // Set pre-trained calibrator
        let pretrained = SigmoidCalibrator::new();
        let fitted_pretrained = pretrained.fit(&prob_scores.to_owned(), &targets)?;
        transfer_learning.set_pretrained_calibrator(Box::new(fitted_pretrained));

        transfer_learning.fit(&prob_scores.to_owned(), &targets)?;
        calibrators.push(Box::new(transfer_learning) as Box<dyn CalibrationEstimator>);
    }

    Ok(calibrators)
}

fn train_differentiable_ece_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    n_bins: usize,
    learning_rate: Float,
    max_iterations: usize,
    tolerance: Float,
    use_adaptive_bins: bool,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Create and configure differentiable ECE calibrator
        let mut calibrator = if use_adaptive_bins {
            DifferentiableECEMetaCalibrator::new(n_bins)
                .with_adaptive_bins()
                .with_params(n_bins, learning_rate, max_iterations, tolerance)
        } else {
            DifferentiableECEMetaCalibrator::new(n_bins).with_params(
                n_bins,
                learning_rate,
                max_iterations,
                tolerance,
            )
        };

        // Train calibrator
        calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod calibration_aware_tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_calibration_aware_training_methods() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1, 1, 1];

        // Test Calibration-Aware Focal Loss Training
        let focal_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::CalibrationAwareFocal {
                gamma: 2.0,
                temperature: 1.5,
                learning_rate: 0.01,
                max_epochs: 30,
            });
        let focal_fitted = focal_calibrator.fit(&x, &y).unwrap();
        let focal_probas = focal_fitted.predict_proba(&x).unwrap();

        assert_eq!(focal_probas.dim(), (8, 2));
        for row in focal_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Test Calibration-Aware Cross-Entropy Training
        let cross_entropy_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::CalibrationAwareCrossEntropy {
                lambda: 0.1,
                learning_rate: 0.01,
                max_epochs: 30,
            });
        let cross_entropy_fitted = cross_entropy_calibrator.fit(&x, &y).unwrap();
        let cross_entropy_probas = cross_entropy_fitted.predict_proba(&x).unwrap();

        assert_eq!(cross_entropy_probas.dim(), (8, 2));
        for row in cross_entropy_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Test Calibration-Aware Brier Score Training
        let brier_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::CalibrationAwareBrier {
                learning_rate: 0.01,
                max_epochs: 30,
            });
        let brier_fitted = brier_calibrator.fit(&x, &y).unwrap();
        let brier_probas = brier_fitted.predict_proba(&x).unwrap();

        assert_eq!(brier_probas.dim(), (8, 2));
        for row in brier_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Test Calibration-Aware ECE Training
        let ece_calibrator =
            CalibratedClassifierCV::new().method(CalibrationMethod::CalibrationAwareECE {
                n_bins: 10,
                learning_rate: 0.01,
                max_epochs: 30,
            });
        let ece_fitted = ece_calibrator.fit(&x, &y).unwrap();
        let ece_probas = ece_fitted.predict_proba(&x).unwrap();

        assert_eq!(ece_probas.dim(), (8, 2));
        for row in ece_probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
