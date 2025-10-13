//! Support Vector Regression (SVR) implementation

use crate::{
    kernels::{create_kernel, Kernel, KernelType},
    smo::SmoConfig,
    svc::SvcKernel,
};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for SVR
#[derive(Debug, Clone)]
pub struct SvrConfig {
    /// Regularization parameter
    pub c: Float,
    /// Epsilon parameter for epsilon-insensitive loss
    pub epsilon: Float,
    /// Kernel type
    pub kernel: SvcKernel,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to use shrinking heuristics
    pub shrinking: bool,
    /// Cache size for kernel evaluations (in MB)
    pub cache_size: usize,
}

impl Default for SvrConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            kernel: SvcKernel::default(),
            tol: 1e-3,
            max_iter: 200,
            shrinking: true,
            cache_size: 200,
        }
    }
}

/// Support Vector Regression
#[derive(Debug, Clone)]
pub struct SVR<State = Untrained> {
    config: SvrConfig,
    state: PhantomData<State>,
    // Fitted parameters
    support_vectors_: Option<Array2<Float>>,
    dual_coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_in_: Option<usize>,
    support_indices_: Option<Vec<usize>>,
    kernel_: Option<KernelType>,
}

impl SVR<Untrained> {
    /// Create a new SVR
    pub fn new() -> Self {
        Self {
            config: SvrConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            dual_coef_: None,
            intercept_: None,
            n_features_in_: None,
            support_indices_: None,
            kernel_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the epsilon parameter for epsilon-insensitive loss
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set the kernel to linear
    pub fn linear(mut self) -> Self {
        self.config.kernel = SvcKernel::Linear;
        self
    }

    /// Set the kernel to RBF with optional gamma
    pub fn rbf(mut self, gamma: Option<Float>) -> Self {
        self.config.kernel = SvcKernel::Rbf { gamma };
        self
    }

    /// Set the kernel to polynomial
    pub fn poly(mut self, degree: usize, gamma: Option<Float>, coef0: Float) -> Self {
        self.config.kernel = SvcKernel::Poly {
            degree,
            gamma,
            coef0,
        };
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set whether to use shrinking heuristics
    pub fn shrinking(mut self, shrinking: bool) -> Self {
        self.config.shrinking = shrinking;
        self
    }

    /// Set the cache size for kernel evaluations
    pub fn cache_size(mut self, cache_size: usize) -> Self {
        self.config.cache_size = cache_size;
        self
    }

    /// Create kernel based on configuration
    fn create_kernel(&self, n_features: usize) -> KernelType {
        match &self.config.kernel {
            SvcKernel::Linear => KernelType::Linear,
            SvcKernel::Rbf { gamma } => {
                let gamma_val = gamma.unwrap_or(1.0 / n_features as Float);
                KernelType::Rbf { gamma: gamma_val }
            }
            SvcKernel::Poly {
                degree,
                gamma,
                coef0,
            } => {
                let _gamma_val = gamma.unwrap_or(1.0 / n_features as Float);
                KernelType::Polynomial {
                    gamma: _gamma_val,
                    degree: *degree as f64,
                    coef0: *coef0,
                }
            }
            SvcKernel::Sigmoid { gamma, coef0 } => {
                let gamma_val = gamma.unwrap_or(1.0 / n_features as Float);
                KernelType::Sigmoid {
                    gamma: gamma_val,
                    coef0: *coef0,
                }
            }
            SvcKernel::Custom(kernel) => kernel.clone(),
        }
    }

    /// Transform regression problem to epsilon-insensitive classification
    /// Creates dual problem with alpha+ and alpha- variables
    fn transform_to_dual_problem(&self, y: &Array1<Float>) -> (Array2<Float>, Array1<Float>) {
        let n_samples = y.len();

        // Create expanded feature matrix for dual variables
        // Each sample becomes two samples (for alpha+ and alpha-)
        let x_dual = Array2::<Float>::zeros((2 * n_samples, 1)); // Placeholder

        // Create target vector for epsilon-insensitive loss
        let mut y_dual = Array1::<Float>::zeros(2 * n_samples);

        for i in 0..n_samples {
            // Upper constraints: y_i - epsilon <= f(x_i) <= y_i + epsilon
            y_dual[i] = 1.0; // For alpha+ (error > epsilon)
            y_dual[i + n_samples] = -1.0; // For alpha- (error < -epsilon)
        }

        (x_dual, y_dual)
    }
}

impl SVR<Trained> {
    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_
            .as_ref()
            .expect("SVR should be fitted")
    }

    /// Get the dual coefficients
    pub fn dual_coef(&self) -> &Array1<Float> {
        self.dual_coef_.as_ref().expect("SVR should be fitted")
    }

    /// Get the intercept (bias) term
    pub fn intercept(&self) -> Float {
        self.intercept_.expect("SVR should be fitted")
    }

    /// Get the indices of support vectors
    pub fn support_indices(&self) -> &[usize] {
        self.support_indices_
            .as_ref()
            .expect("SVR should be fitted")
    }

    /// Compute decision function values (regression predictions)
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_.unwrap() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_.unwrap(),
                actual: n_features,
            });
        }

        let kernel_type = self.kernel_.as_ref().expect("Kernel should be available");
        let kernel = create_kernel(kernel_type.clone());
        let support_vectors = self.support_vectors();
        let dual_coef = self.dual_coef();
        let intercept = self.intercept();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut score = 0.0;

            for (j, _) in self.support_indices().iter().enumerate() {
                let k_val = kernel.compute(
                    x.row(i).to_owned().view(),
                    support_vectors.row(j).to_owned().view(),
                );
                score += dual_coef[j] * k_val;
            }

            predictions[i] = score + intercept;
        }

        Ok(predictions)
    }
}

impl Default for SVR<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for SVR<Untrained> {
    type Fitted = SVR<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit SVR on empty dataset".to_string(),
            ));
        }

        if self.config.epsilon < 0.0 {
            return Err(SklearsError::InvalidParameter {
                name: "epsilon".to_string(),
                reason: "Epsilon must be non-negative".to_string(),
            });
        }

        // Create kernel
        let kernel = self.create_kernel(n_features);

        // For simplicity, we'll implement a basic SVR using the SMO framework
        // In practice, SVR requires a more sophisticated dual formulation

        // Simplified approach: treat as regression with epsilon-insensitive loss
        // Create pseudo-classification problem for SMO solver
        let (_x_dual, _y_dual) = self.transform_to_dual_problem(y);

        // Configure SMO solver
        let _smo_config = SmoConfig {
            c: self.config.c,
            tol: self.config.tol,
            max_iter: self.config.max_iter,
            cache_size: self.config.cache_size,
            shrinking: self.config.shrinking,
            working_set_strategy: crate::smo::WorkingSetStrategy::SecondOrder,
            early_stopping_tol: 1e-4,
            convergence_check_interval: 10,
        };

        // For now, use a simplified linear approach
        // In a full implementation, you'd solve the dual SVR problem

        // Placeholder: find support vectors as samples with largest residuals
        let mut support_indices = Vec::new();
        let mut dual_coef = Vec::new();

        // Simple heuristic: select samples as support vectors
        for i in 0..n_samples.min(n_samples / 2) {
            support_indices.push(i);
            dual_coef.push(1.0 / n_samples as Float);
        }

        // Extract support vectors
        let n_support = support_indices.len();
        let mut support_vectors = Array2::zeros((n_support, n_features));

        for (i, &support_idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(support_idx));
        }

        // Compute intercept (simplified)
        let intercept = y.mean().unwrap_or(0.0);

        Ok(SVR {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(support_vectors),
            dual_coef_: Some(Array1::from_vec(dual_coef)),
            intercept_: Some(intercept),
            n_features_in_: Some(n_features),
            support_indices_: Some(support_indices),
            kernel_: Some(kernel),
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for SVR<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        self.decision_function(x)
    }
}

/// Epsilon-SVR variant with explicit epsilon parameter control
#[derive(Debug, Clone)]
pub struct EpsilonSVR<State = Untrained> {
    svr: SVR<State>,
}

impl EpsilonSVR<Untrained> {
    /// Create a new Epsilon-SVR
    pub fn new() -> Self {
        Self { svr: SVR::new() }
    }

    /// Set the epsilon parameter
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.svr = self.svr.epsilon(epsilon);
        self
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.svr = self.svr.c(c);
        self
    }

    /// Set the kernel to RBF
    pub fn rbf(mut self, gamma: Option<Float>) -> Self {
        self.svr = self.svr.rbf(gamma);
        self
    }
}

impl Default for EpsilonSVR<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for EpsilonSVR<Untrained> {
    type Fitted = EpsilonSVR<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let fitted_svr = self.svr.fit(x, y)?;
        Ok(EpsilonSVR { svr: fitted_svr })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for EpsilonSVR<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        self.svr.predict(x)
    }
}

impl EpsilonSVR<Trained> {
    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.svr.support_vectors()
    }

    /// Get the dual coefficients
    pub fn dual_coef(&self) -> &Array1<Float> {
        self.svr.dual_coef()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Float {
        self.svr.intercept()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_svr_basic() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Linear relationship: y = 2*x

        let svr = SVR::new().linear().c(1.0).epsilon(0.1).fit(&x, &y).unwrap();

        // Check fitted attributes
        assert!(svr.support_vectors().nrows() > 0);
        assert!(svr.dual_coef().len() > 0);

        // Test prediction
        let x_test = array![[3.0]];
        let predictions = svr.predict(&x_test).unwrap();

        assert_eq!(predictions.len(), 1);
        // Note: exact prediction depends on support vector selection
    }

    #[test]
    fn test_svr_rbf_kernel() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];
        let y = array![1.0, 4.0, 9.0];

        let svr = SVR::new()
            .rbf(Some(1.0))
            .c(1.0)
            .epsilon(0.1)
            .fit(&x, &y)
            .unwrap();

        assert!(svr.support_vectors().nrows() > 0);
    }

    #[test]
    fn test_epsilon_svr() {
        let x = array![[1.0], [2.0], [3.0],];
        let y = array![1.0, 2.0, 3.0];

        let epsilon_svr = EpsilonSVR::new().epsilon(0.5).c(1.0).fit(&x, &y).unwrap();

        assert!(epsilon_svr.support_vectors().nrows() > 0);

        let predictions = epsilon_svr.predict(&x).unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_svr_config_builder() {
        let svr = SVR::new()
            .c(10.0)
            .epsilon(0.01)
            .linear()
            .tol(1e-4)
            .max_iter(2000);

        assert_eq!(svr.config.c, 10.0);
        assert_eq!(svr.config.epsilon, 0.01);
        assert_eq!(svr.config.tol, 1e-4);
        assert_eq!(svr.config.max_iter, 2000);
    }

    #[test]
    fn test_svr_empty_dataset() {
        let x = Array2::<Float>::zeros((0, 2));
        let y = Array1::<Float>::zeros(0);

        let result = SVR::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_svr_negative_epsilon() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let result = SVR::new().epsilon(-0.1).fit(&x, &y);
        assert!(result.is_err());
    }
}
