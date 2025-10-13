//! Comprehensive tests for sparse Gaussian Process implementation
//!
//! This module contains integration tests for all sparse GP components,
//! ensuring proper functionality and numerical correctness.

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_gp::kernels::RBFKernel;
    use scirs2_core::ndarray::array;
    use approx::assert_abs_diff_eq;
    use sklears_core::traits::{Fit, Predict};

    /// Test basic sparse GP functionality
    #[test]
    fn test_sparse_gp_basic() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        assert_eq!(pred.len(), x.nrows());
        assert!(pred.iter().all(|&val| val.is_finite()));
    }

    /// Test all sparse approximation methods
    #[test]
    fn test_all_approximation_methods() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let approximations = vec![
            SparseApproximation::SubsetOfRegressors,
            SparseApproximation::FullyIndependentConditional,
            SparseApproximation::PartiallyIndependentConditional { block_size: 2 },
            SparseApproximation::VariationalFreeEnergy {
                whitened: false,
                natural_gradients: false,
            },
        ];

        for approximation in approximations {
            let gp = SparseGaussianProcess::new(3, kernel.clone())
                .approximation(approximation)
                .noise_variance(0.1)
                .optimization_params(10, 1e-4); // Reduced iterations for testing

            let fitted = gp.fit(&x, &y).unwrap();
            let pred = fitted.predict(&x).unwrap();

            assert_eq!(pred.len(), x.nrows());
            assert!(pred.iter().all(|&val| val.is_finite()));
        }
    }

    /// Test all inducing point selection strategies
    #[test]
    fn test_all_inducing_strategies() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let x = array![
            [0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5],
            [2.0, 2.0], [2.5, 2.5], [3.0, 3.0], [3.5, 3.5]
        ];
        let y = array![0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0, 12.25];

        let strategies = vec![
            InducingPointStrategy::Random,
            InducingPointStrategy::KMeans,
            InducingPointStrategy::UniformGrid { grid_size: vec![2, 2] },
            InducingPointStrategy::GreedyVariance,
        ];

        for strategy in strategies {
            let num_inducing = match &strategy {
                InducingPointStrategy::UniformGrid { grid_size } => grid_size.iter().product(),
                _ => 3,
            };

            let gp = SparseGaussianProcess::new(num_inducing, kernel.clone())
                .inducing_strategy(strategy)
                .approximation(SparseApproximation::FullyIndependentConditional)
                .noise_variance(0.1);

            let fitted = gp.fit(&x, &y).unwrap();
            let pred = fitted.predict(&x).unwrap();

            assert_eq!(pred.len(), x.nrows());
            assert!(pred.iter().all(|&val| val.is_finite()));
        }
    }

    /// Test user-specified inducing points
    #[test]
    fn test_user_specified_inducing_points() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let custom_inducing = array![[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]];

        let gp = SparseGaussianProcess::new(3, kernel)
            .inducing_strategy(InducingPointStrategy::UserSpecified(custom_inducing.clone()))
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let fitted = gp.fit(&x, &y).unwrap();

        // Check that custom inducing points are used
        for (i, row) in custom_inducing.outer_iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert_abs_diff_eq!(fitted.inducing_points[(i, j)], val, epsilon = 1e-10);
            }
        }
    }

    /// Test prediction with variance
    #[test]
    fn test_prediction_with_variance() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let fitted = gp.fit(&x, &y).unwrap();
        let (mean, var) = fitted.predict_with_variance(&x).unwrap();

        assert_eq!(mean.len(), x.nrows());
        assert_eq!(var.len(), x.nrows());
        assert!(mean.iter().all(|&val| val.is_finite()));
        assert!(var.iter().all(|&val| val >= 0.0 && val.is_finite()));
    }

    /// Test scalable inference methods
    #[test]
    fn test_scalable_inference_methods() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let fitted = gp.fit(&x, &y).unwrap();

        let methods = vec![
            ScalableInferenceMethod::Direct,
            ScalableInferenceMethod::PreconditionedCG {
                max_iter: 50,
                tol: 1e-6,
                preconditioner: PreconditionerType::Diagonal,
            },
            ScalableInferenceMethod::Lanczos {
                num_vectors: 2,
                tol: 1e-6,
            },
        ];

        for method in methods {
            let pred = fitted.predict_scalable(&x, method).unwrap();
            assert_eq!(pred.len(), x.nrows());
            assert!(pred.iter().all(|&val| val.is_finite()));
        }
    }

    /// Test different preconditioner types
    #[test]
    fn test_preconditioner_types() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0.0, 1.0, 4.0];

        let fitted = gp.fit(&x, &y).unwrap();

        let preconditioners = vec![
            PreconditionerType::None,
            PreconditionerType::Diagonal,
            PreconditionerType::SSOR { omega: 1.5 },
        ];

        for precond in preconditioners {
            let pred = fitted
                .predict_scalable(
                    &x,
                    ScalableInferenceMethod::PreconditionedCG {
                        max_iter: 20,
                        tol: 1e-4,
                        preconditioner: precond,
                    },
                )
                .unwrap();

            assert_eq!(pred.len(), x.nrows());
            assert!(pred.iter().all(|&val| val.is_finite()));
        }
    }

    /// Test log marginal likelihood computation
    #[test]
    fn test_log_marginal_likelihood() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let fitted = gp.fit(&x, &y).unwrap();
        let log_ml = fitted.log_marginal_likelihood().unwrap();

        assert!(log_ml.is_finite());
    }

    /// Test variational free energy with different settings
    #[test]
    fn test_vfe_variants() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let vfe_variants = vec![
            (false, false), // Standard, standard gradients
            (true, false),  // Whitened, standard gradients
            (false, true),  // Standard, natural gradients
            (true, true),   // Whitened, natural gradients
        ];

        for (whitened, natural_gradients) in vfe_variants {
            let gp = SparseGaussianProcess::new(3, kernel.clone())
                .approximation(SparseApproximation::VariationalFreeEnergy {
                    whitened,
                    natural_gradients,
                })
                .noise_variance(0.1)
                .optimization_params(10, 1e-4); // Reduced iterations for testing

            let fitted = gp.fit(&x, &y).unwrap();
            let pred = fitted.predict(&x).unwrap();

            assert_eq!(pred.len(), x.nrows());
            assert!(pred.iter().all(|&val| val.is_finite()));

            // Check that variational parameters are stored
            assert!(fitted.variational_params.is_some());
            let vfe_params = fitted.variational_params.as_ref().unwrap();
            assert!(vfe_params.elbo.is_finite());
            assert!(vfe_params.kl_divergence >= 0.0);
        }
    }

    /// Test structured kernel interpolation
    #[test]
    fn test_structured_kernel_interpolation() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let ski = StructuredKernelInterpolation::new(vec![3, 3], kernel).noise_variance(0.1);

        let x = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]];
        let y = array![0.0, 0.25, 1.0, 2.25];

        let fitted = ski.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        assert_eq!(pred.len(), x.nrows());
        assert!(pred.iter().all(|&x| x.is_finite()));
    }

    /// Test SKI with cubic interpolation
    #[test]
    fn test_ski_cubic_interpolation() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let ski = StructuredKernelInterpolation::new(vec![4, 4], kernel)
            .noise_variance(0.1)
            .interpolation(InterpolationMethod::Cubic);

        let x = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]];
        let y = array![0.0, 0.25, 1.0];

        let fitted = ski.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        assert_eq!(pred.len(), x.nrows());
        assert!(pred.iter().all(|&x| x.is_finite()));
    }

    /// Test SIMD operations
    #[test]
    fn test_simd_operations() {
        let x1 = array![[0.0, 0.0], [1.0, 1.0]];
        let x2 = array![[0.0, 0.0], [2.0, 2.0]];

        // Test SIMD kernel matrix
        let kernel_matrix = simd_sparse_gp::simd_rbf_kernel_matrix(&x1, &x2, 1.0, 1.0);
        assert_eq!(kernel_matrix.shape(), &[2, 2]);
        assert_abs_diff_eq!(kernel_matrix[(0, 0)], 1.0, epsilon = 1e-10);

        // Test SIMD prediction
        let k_star_m = array![[0.8, 0.3], [0.5, 0.7]];
        let alpha = array![1.0, 2.0];
        let mean = simd_sparse_gp::simd_posterior_mean(&k_star_m, &alpha);
        assert_eq!(mean.len(), 2);
        assert!(mean.iter().all(|&x| x.is_finite()));

        // Test SIMD variance
        let k_star_star = array![1.0, 1.0];
        let v_matrix = array![[0.5, 0.3], [0.2, 0.4]];
        let variance = simd_sparse_gp::simd_posterior_variance(&k_star_star, &v_matrix);
        assert_eq!(variance.len(), 2);
        assert!(variance.iter().all(|&x| x > 0.0 && x.is_finite()));
    }

    /// Test posterior sampling
    #[test]
    fn test_posterior_sampling() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0.0, 1.0, 4.0];

        let fitted = gp.fit(&x, &y).unwrap();
        let samples = fitted.sample_posterior(&x, 5).unwrap();

        assert_eq!(samples.shape(), &[5, 3]);
        assert!(samples.iter().all(|&val| val.is_finite()));
    }

    /// Test acquisition functions for active learning
    #[test]
    fn test_acquisition_functions() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0.0, 1.0, 4.0];

        let fitted = gp.fit(&x, &y).unwrap();

        let acquisition_functions = vec![
            "variance",
            "entropy",
            "upper_confidence_bound",
        ];

        for acq_func in acquisition_functions {
            let acq_values = fitted.acquisition_function(&x, acq_func).unwrap();
            assert_eq!(acq_values.len(), x.nrows());
            assert!(acq_values.iter().all(|&val| val.is_finite()));
        }
    }

    /// Test effective degrees of freedom computation
    #[test]
    fn test_effective_degrees_of_freedom() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let fitted = gp.fit(&x, &y).unwrap();
        let eff_dof = utils::effective_degrees_of_freedom(&fitted, &x).unwrap();

        assert!(eff_dof > 0.0 && eff_dof <= x.nrows() as f64);
        assert!(eff_dof.is_finite());
    }

    /// Test model complexity penalty
    #[test]
    fn test_model_complexity_penalty() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0.0, 1.0, 4.0];

        let fitted = gp.fit(&x, &y).unwrap();
        let penalty = utils::model_complexity_penalty(&fitted);

        assert!(penalty > 0.0);
        assert!(penalty.is_finite());
    }

    /// Test hyperparameter optimization
    #[test]
    fn test_hyperparameter_optimization() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let mut gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0.0, 1.0, 4.0];

        let best_likelihood = utils::optimize_hyperparameters(
            &mut gp,
            &x,
            &y,
            5, // Reduced iterations for testing
            0.01,
        ).unwrap();

        assert!(best_likelihood.is_finite());
    }

    /// Test error handling and edge cases
    #[test]
    fn test_error_handling() {
        let kernel = RBFKernel::new(1.0, 1.0);

        // Test dimension mismatch
        let gp = SparseGaussianProcess::new(2, kernel.clone());
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![0.0]; // Wrong size

        let result = gp.fit(&x, &y);
        assert!(result.is_err());

        // Test too many inducing points
        let gp = SparseGaussianProcess::new(10, kernel.clone()); // More than data points
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![0.0, 1.0];

        let result = gp.fit(&x, &y);
        assert!(result.is_err());

        // Test invalid grid size for uniform grid strategy
        let gp = SparseGaussianProcess::new(6, kernel.clone())
            .inducing_strategy(InducingPointStrategy::UniformGrid {
                grid_size: vec![2, 3], // 2 × 3 = 6
            });
        let x = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]; // 3D data but 2D grid

        let result = gp.fit(&x, &y);
        assert!(result.is_err());
    }

    /// Test numerical stability with ill-conditioned data
    #[test]
    fn test_numerical_stability() {
        let kernel = RBFKernel::new(0.01, 1.0); // Very small length scale
        let gp = SparseGaussianProcess::new(3, kernel)
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(1e-8); // Very small noise

        // Create nearly identical points (ill-conditioned)
        let x = array![
            [0.0, 0.0],
            [1e-6, 1e-6],
            [2e-6, 2e-6],
            [1.0, 1.0],
            [2.0, 2.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 4.0];

        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        assert_eq!(pred.len(), x.nrows());
        assert!(pred.iter().all(|&val| val.is_finite()));
    }

    /// Test large scale performance (reduced size for testing)
    #[test]
    fn test_large_scale_performance() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(5, kernel) // Small for testing
            .approximation(SparseApproximation::FullyIndependentConditional)
            .noise_variance(0.1);

        // Create "large" dataset (actually small for testing)
        let n_samples = 50;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for i in 0..n_samples {
            let x_val = i as f64 / 10.0;
            x_data.push([x_val, x_val * 0.5]);
            y_data.push(x_val * x_val);
        }

        let x = Array2::from_shape_vec((n_samples, 2), x_data.into_iter().flatten().collect()).unwrap();
        let y = Array1::from_vec(y_data);

        let fitted = gp.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        assert_eq!(pred.len(), x.nrows());
        assert!(pred.iter().all(|&val| val.is_finite()));
    }

    /// Integration test combining multiple features
    #[test]
    fn test_integration_comprehensive() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let gp = SparseGaussianProcess::new(4, kernel)
            .approximation(SparseApproximation::VariationalFreeEnergy {
                whitened: true,
                natural_gradients: true,
            })
            .inducing_strategy(InducingPointStrategy::GreedyVariance)
            .noise_variance(0.05)
            .optimization_params(15, 1e-5);

        let x = array![
            [0.0, 0.0], [0.5, 0.2], [1.0, 0.8], [1.5, 1.2],
            [2.0, 1.8], [2.5, 2.2], [3.0, 2.8], [3.5, 3.2],
            [4.0, 3.8], [4.5, 4.2]
        ];
        let y = array![0.0, 0.3, 1.6, 3.9, 7.2, 11.5, 16.8, 23.1, 30.4, 38.7];

        // Fit the model
        let fitted = gp.fit(&x, &y).unwrap();

        // Test basic prediction
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), x.nrows());
        assert!(pred.iter().all(|&val| val.is_finite()));

        // Test prediction with variance
        let (mean, var) = fitted.predict_with_variance(&x).unwrap();
        assert_eq!(mean.len(), x.nrows());
        assert_eq!(var.len(), x.nrows());
        assert!(var.iter().all(|&val| val >= 0.0));

        // Test scalable inference
        let pred_scalable = fitted.predict_scalable(
            &x,
            ScalableInferenceMethod::PreconditionedCG {
                max_iter: 100,
                tol: 1e-8,
                preconditioner: PreconditionerType::Diagonal,
            },
        ).unwrap();
        assert_eq!(pred_scalable.len(), x.nrows());

        // Test log marginal likelihood
        let log_ml = fitted.log_marginal_likelihood().unwrap();
        assert!(log_ml.is_finite());

        // Test acquisition function
        let acq = fitted.acquisition_function(&x, "variance").unwrap();
        assert_eq!(acq.len(), x.nrows());
        assert!(acq.iter().all(|&val| val >= 0.0));

        // Check that VFE parameters are present and valid
        assert!(fitted.variational_params.is_some());
        let vfe_params = fitted.variational_params.as_ref().unwrap();
        assert!(vfe_params.elbo.is_finite());
        assert!(vfe_params.kl_divergence >= 0.0);
        assert!(vfe_params.log_likelihood.is_finite());
    }
}