//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::super::types::{ADBackend, ADVIGaussianMixture, ADVIOptimizer, Dual};
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, s};
    use sklears_core::traits::{Fit, Predict};
    #[test]
    fn test_advi_gaussian_mixture_creation() {
        let gmm = ADVIGaussianMixture::new()
            .n_components(3)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.001)
            .max_iter(500);
        assert_eq!(gmm.n_components, 3);
        assert_eq!(gmm.ad_backend, ADBackend::DualNumbers);
        assert_eq!(gmm.optimizer, ADVIOptimizer::Adam);
        assert_eq!(gmm.learning_rate, 0.001);
        assert_eq!(gmm.max_iter, 500);
    }
    #[test]
    fn test_dual_number_arithmetic() {
        let a = Dual::new(2.0, 1.0);
        let b = Dual::new(3.0, 0.0);
        let sum = a + b;
        assert_eq!(sum.value, 5.0);
        assert_eq!(sum.derivative, 1.0);
        let product = a * b;
        assert_eq!(product.value, 6.0);
        assert_eq!(product.derivative, 3.0);
        let exp_a = a.exp();
        assert_abs_diff_eq!(exp_a.value, 2.0_f64.exp(), epsilon = 1e-10);
        assert_abs_diff_eq!(exp_a.derivative, 2.0_f64.exp(), epsilon = 1e-10);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_fit_predict() {
        let X = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [10.0, 10.0],
            [10.5, 10.5],
            [11.0, 11.0]
        ];
        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(50);
        let fitted = gmm
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        let predictions = fitted
            .predict(&X.view())
            .expect("prediction should succeed");
        assert_eq!(predictions.len(), 6);
        assert!(predictions.iter().all(|&label| label < 2));
        let first_cluster = predictions[0];
        assert_eq!(predictions[1], first_cluster);
        assert_eq!(predictions[2], first_cluster);
        let second_cluster = predictions[3];
        assert_eq!(predictions[4], second_cluster);
        assert_eq!(predictions[5], second_cluster);
        assert_ne!(first_cluster, second_cluster);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_backends() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let backends = vec![
            ADBackend::FiniteDifferences,
            ADBackend::DualNumbers,
            ADBackend::ForwardMode,
        ];
        for backend in backends {
            let gmm = ADVIGaussianMixture::new()
                .n_components(2)
                .ad_backend(backend)
                .optimizer(ADVIOptimizer::SGD)
                .learning_rate(0.1)
                .random_state(42)
                .tol(1e-2)
                .max_iter(20);
            let fitted = gmm
                .fit(&X.view(), &())
                .expect("model fitting should succeed");
            let predictions = fitted
                .predict(&X.view())
                .expect("prediction should succeed");
            assert_eq!(predictions.len(), 4);
            assert!(predictions.iter().all(|&label| label < 2));
        }
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_optimizers() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let optimizers = vec![
            ADVIOptimizer::SGD,
            ADVIOptimizer::AdaGrad,
            ADVIOptimizer::RMSprop,
            ADVIOptimizer::Adam,
        ];
        for optimizer in optimizers {
            let gmm = ADVIGaussianMixture::new()
                .n_components(2)
                .ad_backend(ADBackend::DualNumbers)
                .optimizer(optimizer)
                .learning_rate(0.1)
                .random_state(42)
                .tol(1e-2)
                .max_iter(20);
            let fitted = gmm
                .fit(&X.view(), &())
                .expect("model fitting should succeed");
            let predictions = fitted
                .predict(&X.view())
                .expect("prediction should succeed");
            assert_eq!(predictions.len(), 4);
            assert!(predictions.iter().all(|&label| label < 2));
        }
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_probabilities() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);
        let fitted = gmm
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        let probabilities = fitted
            .predict_proba(&X.view())
            .expect("operation should succeed");
        assert_eq!(probabilities.dim(), (4, 2));
        for i in 0..4 {
            let sum: f64 = probabilities.slice(s![i, ..]).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }
        assert!(probabilities.iter().all(|&p| p >= 0.0));
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_score() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);
        let fitted = gmm
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        let score = fitted.score(&X.view()).expect("operation should succeed");
        assert!(score.is_finite());
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_model_selection() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);
        let fitted = gmm
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        let model_selection = fitted.model_selection();
        assert!(model_selection.aic.is_finite());
        assert!(model_selection.bic.is_finite());
        assert!(model_selection.log_likelihood.is_finite());
        assert!(model_selection.n_parameters > 0);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_parameter_access() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);
        let fitted = gmm
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        assert_eq!(fitted.mean_values().dim(), (2, 2));
        assert_eq!(fitted.precision_values().dim(), (2, 2, 2));
        assert_eq!(fitted.responsibilities().dim(), (4, 2));
        assert_eq!(fitted.ad_backend(), ADBackend::DualNumbers);
        assert_eq!(fitted.optimizer(), ADVIOptimizer::Adam);
        assert!(fitted.lower_bound().is_finite());
        assert!(!fitted.gradient_history().is_empty());
        assert!(!fitted.parameter_history().is_empty());
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_reproducibility() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let gmm1 = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);
        let gmm2 = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);
        let fitted1 = gmm1
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        let fitted2 = gmm2
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        let predictions1 = fitted1
            .predict(&X.view())
            .expect("prediction should succeed");
        let predictions2 = fitted2
            .predict(&X.view())
            .expect("prediction should succeed");
        assert_eq!(predictions1, predictions2);
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_error_handling() {
        let X = array![[0.0, 0.0], [1.0, 1.0]];
        let gmm = ADVIGaussianMixture::new()
            .n_components(5)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .random_state(42);
        let result = gmm.fit(&X.view(), &());
        assert!(result.is_err());
        let gmm2 = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .random_state(42)
            .tol(1e-2)
            .max_iter(20);
        let fitted = gmm2
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        let X_wrong = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let result = fitted.predict(&X_wrong.view());
        assert!(result.is_err());
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_gradient_clipping() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .grad_clip(0.5)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);
        let fitted = gmm
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        let predictions = fitted
            .predict(&X.view())
            .expect("prediction should succeed");
        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&label| label < 2));
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_advi_gaussian_mixture_natural_gradients() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let gmm = ADVIGaussianMixture::new()
            .n_components(2)
            .ad_backend(ADBackend::DualNumbers)
            .optimizer(ADVIOptimizer::Adam)
            .learning_rate(0.1)
            .use_natural_gradients(true)
            .random_state(42)
            .tol(1e-2)
            .max_iter(30);
        let fitted = gmm
            .fit(&X.view(), &())
            .expect("model fitting should succeed");
        let predictions = fitted
            .predict(&X.view())
            .expect("prediction should succeed");
        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&label| label < 2));
    }
}
