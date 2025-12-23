//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::{RandNormal, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
/// Trait extension for array operations
pub trait ArrayInverse<A> {
    fn inv(&self) -> Option<Array2<A>>;
}
impl ArrayInverse<Float> for Array2<Float> {
    fn inv(&self) -> Option<Array2<Float>> {
        if self.is_square() {
            let n = self.nrows();
            let mut inv = Array2::<Float>::eye(n);
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        inv[(i, j)] = 1.0 / self[(i, j)].max(1e-10);
                    } else {
                        inv[(i, j)] = 0.0;
                    }
                }
            }
            Some(inv)
        } else {
            None
        }
    }
}
/// Trait extension for array operations
pub trait ArrayDeterminant<A> {
    fn det(&self) -> Option<A>;
}
impl ArrayDeterminant<Float> for Array2<Float> {
    fn det(&self) -> Option<Float> {
        if self.is_square() {
            Some(self.diag().iter().product())
        } else {
            None
        }
    }
}
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_variational() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = BayesianMultiOutputConfig {
            inference_method: InferenceMethod::Variational,
            max_iter: 50,
            random_state: Some(42),
            ..Default::default()
        };
        let model = BayesianMultiOutputModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.shape(), &[3, 2]);
        assert!(trained.log_marginal_likelihood().is_finite());
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_mcmc() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = BayesianMultiOutputConfig {
            inference_method: InferenceMethod::MCMC,
            n_samples: 100,
            burn_in: 20,
            max_iter: 50,
            random_state: Some(42),
            ..Default::default()
        };
        let model = BayesianMultiOutputModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.shape(), &[3, 2]);
        assert!(trained.weight_posterior().samples.is_some());
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_em() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = BayesianMultiOutputConfig {
            inference_method: InferenceMethod::EM,
            max_iter: 100,
            random_state: Some(42),
            ..Default::default()
        };
        let model = BayesianMultiOutputModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.shape(), &[3, 2]);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_laplace() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = BayesianMultiOutputConfig {
            inference_method: InferenceMethod::Laplace,
            max_iter: 100,
            random_state: Some(42),
            ..Default::default()
        };
        let model = BayesianMultiOutputModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.shape(), &[3, 2]);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_exact() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = BayesianMultiOutputConfig {
            inference_method: InferenceMethod::Exact,
            random_state: Some(42),
            ..Default::default()
        };
        let model = BayesianMultiOutputModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.shape(), &[3, 2]);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_uncertainty() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = BayesianMultiOutputConfig {
            inference_method: InferenceMethod::Variational,
            max_iter: 50,
            random_state: Some(42),
            ..Default::default()
        };
        let model = BayesianMultiOutputModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let prediction_with_uncertainty = trained.predict_with_uncertainty(&X.view()).unwrap();
        assert_eq!(prediction_with_uncertainty.mean.shape(), &[3, 2]);
        assert_eq!(prediction_with_uncertainty.variance.shape(), &[3, 2]);
        assert_eq!(
            prediction_with_uncertainty.confidence_intervals.shape(),
            &[3, 2, 2]
        );
        assert_eq!(prediction_with_uncertainty.samples.shape(), &[3, 2, 100]);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_priors() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = BayesianMultiOutputConfig {
            weight_prior: PriorDistribution::Laplace(0.0, 0.5),
            bias_prior: PriorDistribution::Uniform(-1.0, 1.0),
            noise_prior: PriorDistribution::Gamma(2.0, 2.0),
            inference_method: InferenceMethod::Variational,
            max_iter: 50,
            random_state: Some(42),
            ..Default::default()
        };
        let model = BayesianMultiOutputModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.shape(), &[3, 2]);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_hierarchical() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = BayesianMultiOutputConfig {
            weight_prior: PriorDistribution::Hierarchical,
            inference_method: InferenceMethod::Variational,
            max_iter: 50,
            random_state: Some(42),
            ..Default::default()
        };
        let model = BayesianMultiOutputModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.shape(), &[3, 2]);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_invalid_input() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let model = BayesianMultiOutputModel::new();
        let result = model.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_bayesian_multi_output_model_prediction_shapes() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let model = BayesianMultiOutputModel::new();
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let X_test = array![[1.5, 2.5], [2.5, 3.5]];
        let predictions = trained.predict(&X_test.view()).unwrap();
        assert_eq!(predictions.shape(), &[2, 2]);
        let prediction_with_uncertainty = trained.predict_with_uncertainty(&X_test.view()).unwrap();
        assert_eq!(prediction_with_uncertainty.mean.shape(), &[2, 2]);
        assert_eq!(prediction_with_uncertainty.variance.shape(), &[2, 2]);
        assert_eq!(
            prediction_with_uncertainty.confidence_intervals.shape(),
            &[2, 2, 2]
        );
        assert_eq!(prediction_with_uncertainty.samples.shape(), &[2, 2, 100]);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_gaussian_process_multi_output_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let gp = GaussianProcessMultiOutput::new()
            .kernel(KernelFunction::RBF(1.0))
            .noise_level(0.1)
            .random_state(Some(42));
        let trained = gp.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (3, 2));
        let (X_train, y_train) = trained.training_data();
        assert_eq!(X_train.dim(), (3, 2));
        assert_eq!(y_train.dim(), (3, 2));
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_gaussian_process_multi_output_with_variance() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let gp = GaussianProcessMultiOutput::new()
            .kernel(KernelFunction::RBF(1.0))
            .noise_level(0.01);
        let trained = gp.fit(&X.view(), &y.view()).unwrap();
        let X_test = array![[1.5, 2.5], [2.5, 3.5]];
        let (mean, variance) = trained.predict_with_variance(&X_test.view()).unwrap();
        assert_eq!(mean.dim(), (2, 2));
        assert_eq!(variance.dim(), (2, 2));
        for i in 0..variance.nrows() {
            for j in 0..variance.ncols() {
                assert!(variance[[i, j]] >= 0.0);
            }
        }
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_gaussian_process_multi_output_kernel_functions() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 0.0], [0.0, 1.0]];
        let kernels = vec![
            KernelFunction::RBF(1.0),
            KernelFunction::Linear,
            KernelFunction::Polynomial(2, 1.0),
            KernelFunction::Matern(1.0, 1.5),
            KernelFunction::RationalQuadratic(1.0, 1.0),
        ];
        for kernel in kernels {
            let gp = GaussianProcessMultiOutput::new()
                .kernel(kernel.clone())
                .noise_level(0.1);
            let trained = gp.fit(&X.view(), &y.view()).unwrap();
            let predictions = trained.predict(&X.view()).unwrap();
            assert_eq!(predictions.dim(), (2, 2));
            assert_eq!(trained.kernel(), &kernel);
        }
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_gaussian_process_multi_output_normalization() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![[10.0, 5.0], [15.0, 8.0], [12.0, 6.0]];
        let gp = GaussianProcessMultiOutput::new()
            .kernel(KernelFunction::RBF(1.0))
            .normalize_y(true)
            .noise_level(0.1);
        let trained = gp.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (3, 2));
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                assert!(predictions[[i, j]] >= 0.0);
                assert!(predictions[[i, j]] <= 20.0);
            }
        }
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_gaussian_process_multi_output_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let gp = GaussianProcessMultiOutput::new();
        assert!(gp.fit(&X.view(), &y.view()).is_err());
        let X_train = array![[1.0, 2.0], [2.0, 3.0]];
        let y_train = array![[1.0, 0.0], [0.0, 1.0]];
        let X_test = array![[1.0]];
        let trained = gp.fit(&X_train.view(), &y_train.view()).unwrap();
        assert!(trained.predict(&X_test.view()).is_err());
    }
    #[test]
    fn test_gaussian_process_multi_output_configuration() {
        let gp = GaussianProcessMultiOutput::new()
            .kernel(KernelFunction::Matern(2.0, 2.5))
            .noise_level(0.05)
            .normalize_y(true)
            .random_state(Some(123));
        assert_eq!(gp.kernel, KernelFunction::Matern(2.0, 2.5));
        assert_eq!(gp.noise_level, 0.05);
        assert_eq!(gp.normalize_y, true);
        assert_eq!(gp.random_state, Some(123));
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_bayesian_model_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let config = EnsembleBayesianConfig {
            n_models: 3,
            strategy: EnsembleStrategy::BayesianAveraging,
            random_state: Some(42),
            base_config: BayesianMultiOutputConfig {
                max_iter: 20,
                inference_method: InferenceMethod::Exact,
                ..Default::default()
            },
            ..Default::default()
        };
        let model = EnsembleBayesianModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (4, 2));
        assert_eq!(trained.n_models(), 3);
        let weights_sum: Float = trained.model_weights().sum();
        assert_abs_diff_eq!(weights_sum, 1.0, epsilon = 1e-6);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_bayesian_strategies() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let strategies = vec![
            EnsembleStrategy::BayesianAveraging,
            EnsembleStrategy::EqualWeight,
            EnsembleStrategy::ProductOfExperts,
            EnsembleStrategy::CommitteeMachine,
            EnsembleStrategy::MixtureOfExperts,
        ];
        for strategy in strategies {
            let config = EnsembleBayesianConfig {
                n_models: 2,
                strategy: strategy.clone(),
                random_state: Some(42),
                base_config: BayesianMultiOutputConfig {
                    max_iter: 10,
                    inference_method: InferenceMethod::Exact,
                    ..Default::default()
                },
                ..Default::default()
            };
            let model = EnsembleBayesianModel::new().config(config);
            let trained = model.fit(&X.view(), &y.view()).unwrap();
            let predictions = trained.predict(&X.view()).unwrap();
            assert_eq!(predictions.dim(), (3, 2));
        }
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_bayesian_uncertainty() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = EnsembleBayesianConfig {
            n_models: 3,
            strategy: EnsembleStrategy::BayesianAveraging,
            random_state: Some(42),
            base_config: BayesianMultiOutputConfig {
                max_iter: 15,
                inference_method: InferenceMethod::Exact,
                ..Default::default()
            },
            ..Default::default()
        };
        let model = EnsembleBayesianModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let uncertainty = trained.predict_with_uncertainty(&X.view(), 0.95).unwrap();
        assert_eq!(uncertainty.mean.dim(), (3, 2));
        assert_eq!(uncertainty.variance.dim(), (3, 2));
        assert_eq!(uncertainty.confidence_intervals.dim(), (3, 2, 2));
        assert_eq!(uncertainty.samples.dim(), (100, 3, 2));
        for i in 0..3 {
            for j in 0..2 {
                let lower = uncertainty.confidence_intervals[[i, j, 0]];
                let upper = uncertainty.confidence_intervals[[i, j, 1]];
                assert!(lower <= uncertainty.mean[[i, j]]);
                assert!(upper >= uncertainty.mean[[i, j]]);
            }
        }
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_bayesian_builder_pattern() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0]];
        let model = EnsembleBayesianModel::new()
            .n_models(4)
            .strategy(EnsembleStrategy::CommitteeMachine)
            .random_state(123);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (2, 2));
        assert_eq!(trained.n_models(), 4);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_bayesian_equal_weight() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = EnsembleBayesianConfig {
            n_models: 5,
            strategy: EnsembleStrategy::EqualWeight,
            random_state: Some(42),
            base_config: BayesianMultiOutputConfig {
                max_iter: 10,
                inference_method: InferenceMethod::Exact,
                ..Default::default()
            },
            ..Default::default()
        };
        let model = EnsembleBayesianModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let expected_weight = 1.0 / 5.0;
        for &weight in trained.model_weights().iter() {
            assert_abs_diff_eq!(weight, expected_weight, epsilon = 1e-6);
        }
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_bayesian_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let model = EnsembleBayesianModel::new();
        assert!(model.fit(&X.view(), &y.view()).is_err());
        let X_train = array![[1.0, 2.0], [2.0, 3.0]];
        let y_train = array![[1.0, 2.0], [2.0, 3.0]];
        let X_test = array![[1.0]];
        let trained = EnsembleBayesianModel::new()
            .random_state(42)
            .fit(&X_train.view(), &y_train.view())
            .unwrap();
        assert!(trained.predict(&X_test.view()).is_err());
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_bayesian_committee_machine() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let config = EnsembleBayesianConfig {
            n_models: 5,
            strategy: EnsembleStrategy::CommitteeMachine,
            random_state: Some(42),
            base_config: BayesianMultiOutputConfig {
                max_iter: 15,
                inference_method: InferenceMethod::Exact,
                ..Default::default()
            },
            ..Default::default()
        };
        let model = EnsembleBayesianModel::new().config(config);
        let trained = model.fit(&X.view(), &y.view()).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (4, 2));
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                assert!(predictions[[i, j]].is_finite());
            }
        }
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_bayesian_reproducibility() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let config = EnsembleBayesianConfig {
            n_models: 3,
            strategy: EnsembleStrategy::BayesianAveraging,
            random_state: Some(42),
            base_config: BayesianMultiOutputConfig {
                max_iter: 10,
                inference_method: InferenceMethod::Exact,
                ..Default::default()
            },
            ..Default::default()
        };
        let model1 = EnsembleBayesianModel::new().config(config.clone());
        let trained1 = model1.fit(&X.view(), &y.view()).unwrap();
        let pred1 = trained1.predict(&X.view()).unwrap();
        let model2 = EnsembleBayesianModel::new().config(config);
        let trained2 = model2.fit(&X.view(), &y.view()).unwrap();
        let pred2 = trained2.predict(&X.view()).unwrap();
        for i in 0..pred1.nrows() {
            for j in 0..pred1.ncols() {
                assert_abs_diff_eq!(pred1[[i, j]], pred2[[i, j]], epsilon = 1e-6);
            }
        }
    }
}
