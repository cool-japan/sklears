//! Support Vector Regression (SVR) implementation

use crate::{
    kernels::{create_kernel, Kernel, KernelType},
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

    /// Set the kernel directly from an [`SvcKernel`] specification.
    ///
    /// This preserves deferred parameters (such as a `None` gamma, which is
    /// resolved to `1 / n_features` at fit time) instead of forcing a concrete
    /// value up front.
    pub fn svc_kernel(mut self, kernel: SvcKernel) -> Self {
        self.config.kernel = kernel;
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

    /// Solve the epsilon-SVR dual problem using a Sequential Minimal
    /// Optimization (SMO) algorithm.
    ///
    /// The epsilon-SVR dual problem is:
    /// ```text
    /// minimize:  ½ (α - α*)ᵀ Q (α - α*) + ε Σ(α_i + α_i*) - Σ y_i (α_i - α_i*)
    /// subject to: Σ(α_i - α_i*) = 0
    ///             0 ≤ α_i, α_i* ≤ C
    /// ```
    /// where `Q_ij = K(x_i, x_j)`.
    ///
    /// We track the signed coefficient `w_i = α_i - α_i*` and the cached
    /// decision values `decision_i = Σ_j w_j K(i,j)`. A maximal-violating pair
    /// is selected from the KKT conditions, the 2-variable subproblem is solved
    /// analytically subject to the equality constraint `Σ w = 0` and the box
    /// constraints, and the decision cache is updated incrementally.
    ///
    /// Returns `(dual_coef, intercept)` where `dual_coef[i] = α_i - α_i*`.
    fn solve_dual_smo(
        &self,
        kernel: &KernelType,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Float)> {
        let n_samples = y.len();
        let c = self.config.c;
        let epsilon = self.config.epsilon;
        let tol = self.config.tol;

        // Precompute the kernel (Gram) matrix.
        let kernel_fn = create_kernel(kernel.clone())?;
        let mut q = Array2::<Float>::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i..n_samples {
                let val = kernel_fn.compute(x.row(i), x.row(j));
                q[[i, j]] = val;
                q[[j, i]] = val;
            }
        }

        // alpha_i and alpha_i* (both non-negative, bounded by C).
        let mut alpha = Array1::<Float>::zeros(n_samples);
        let mut alpha_star = Array1::<Float>::zeros(n_samples);
        // decision_i = Σ_j (alpha_j - alpha_j*) K(i,j).
        let mut decision = Array1::<Float>::zeros(n_samples);

        let max_iter = self.config.max_iter.max(1) * n_samples.max(1);

        for _iter in 0..max_iter {
            // Select the maximal-violating pair from the residual gradient.
            // r_i = decision_i - y_i. The epsilon-insensitive offsets give two
            // candidate gradients per sample, one for the "increase w" direction
            // and one for the "decrease w" direction.
            let mut i_up = None;
            let mut g_max = Float::NEG_INFINITY;
            let mut i_low = None;
            let mut g_min = Float::INFINITY;

            for i in 0..n_samples {
                let r_i = decision[i] - y[i];
                let grad_up = -r_i - epsilon; // gradient when increasing w_i
                let grad_down = -r_i + epsilon; // gradient when decreasing w_i

                let can_increase = alpha[i] < c - tol || alpha_star[i] > tol;
                let can_decrease = alpha_star[i] < c - tol || alpha[i] > tol;

                if can_increase && grad_up > g_max {
                    g_max = grad_up;
                    i_up = Some(i);
                }
                if can_decrease && grad_down < g_min {
                    g_min = grad_down;
                    i_low = Some(i);
                }
            }

            if g_max - g_min < tol {
                break;
            }

            let (i, j) = match (i_up, i_low) {
                (Some(i), Some(j)) if i != j => (i, j),
                _ => break,
            };

            // Curvature of the 2-variable subproblem.
            let eta = q[[i, i]] + q[[j, j]] - 2.0 * q[[i, j]];
            if eta <= 1e-12 {
                continue;
            }

            let w_i_old = alpha[i] - alpha_star[i];
            let w_j_old = alpha[j] - alpha_star[j];

            let r_i = decision[i] - y[i];
            let r_j = decision[j] - y[j];
            // Optimal step along the (i up, j down) direction for the
            // epsilon-insensitive quadratic.
            let delta = ((r_j - r_i) - 2.0 * epsilon) / eta;

            // Project signed coefficients back onto [-C, C].
            let w_i_new = (w_i_old + delta).clamp(-c, c);
            let w_j_new = (w_j_old - delta).clamp(-c, c);

            // Enforce Σ w = 0 by taking the smaller-magnitude feasible step.
            let actual_delta_i = w_i_new - w_i_old;
            let actual_delta_j = w_j_new - w_j_old;
            let step = if actual_delta_i.abs() <= actual_delta_j.abs() {
                actual_delta_i
            } else {
                -actual_delta_j
            };

            if step.abs() < 1e-12 {
                continue;
            }

            let w_i_final = w_i_old + step;
            let w_j_final = w_j_old - step;

            // Decompose signed coefficients into (alpha, alpha*).
            alpha[i] = w_i_final.max(0.0);
            alpha_star[i] = (-w_i_final).max(0.0);
            alpha[j] = w_j_final.max(0.0);
            alpha_star[j] = (-w_j_final).max(0.0);

            let dw_i = w_i_final - w_i_old;
            let dw_j = w_j_final - w_j_old;
            for k in 0..n_samples {
                decision[k] += dw_i * q[[k, i]] + dw_j * q[[k, j]];
            }
        }

        // Signed dual coefficients.
        let mut dual_coef = Array1::<Float>::zeros(n_samples);
        for i in 0..n_samples {
            dual_coef[i] = alpha[i] - alpha_star[i];
        }

        // Compute the bias b by averaging over free (unbounded) support vectors,
        // for which the KKT conditions fix the residual at +/- epsilon.
        let mut bias_sum = 0.0;
        let mut bias_count = 0usize;
        for i in 0..n_samples {
            let w_i = dual_coef[i];
            if w_i > tol && w_i < c - tol {
                bias_sum += y[i] - decision[i] - epsilon;
                bias_count += 1;
            } else if w_i < -tol && w_i > -(c - tol) {
                bias_sum += y[i] - decision[i] + epsilon;
                bias_count += 1;
            }
        }

        let intercept = if bias_count > 0 {
            bias_sum / bias_count as Float
        } else {
            let mut sum = 0.0;
            for i in 0..n_samples {
                sum += y[i] - decision[i];
            }
            sum / n_samples as Float
        };

        Ok((dual_coef, intercept))
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

        if n_features
            != self
                .n_features_in_
                .expect("n_features_in_ not available - model not fitted")
        {
            return Err(SklearsError::FeatureMismatch {
                expected: self
                    .n_features_in_
                    .expect("n_features_in_ not available - model not fitted"),
                actual: n_features,
            });
        }

        let kernel_type = self.kernel_.as_ref().expect("Kernel should be available");
        let kernel = create_kernel(kernel_type.clone())?;
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

        // Solve the epsilon-SVR dual problem via SMO to obtain the signed dual
        // coefficients w_i = (alpha_i - alpha_i*) and the bias.
        let (full_dual_coef, intercept) = self.solve_dual_smo(&kernel, x, y)?;

        // Support vectors are samples with non-negligible dual coefficient.
        let sv_threshold = 1e-8 * self.config.c.max(1.0);
        let mut support_indices = Vec::new();
        let mut dual_coef = Vec::new();
        for i in 0..n_samples {
            if full_dual_coef[i].abs() > sv_threshold {
                support_indices.push(i);
                dual_coef.push(full_dual_coef[i]);
            }
        }

        // Degenerate fallback: if the solver found no support vectors (e.g. a
        // perfectly fittable problem with large epsilon), keep at least the
        // sample with the largest coefficient so prediction stays well-defined.
        if support_indices.is_empty() {
            let best = (0..n_samples)
                .max_by(|&a, &b| {
                    full_dual_coef[a]
                        .abs()
                        .partial_cmp(&full_dual_coef[b].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            support_indices.push(best);
            dual_coef.push(full_dual_coef[best]);
        }

        // Extract support vectors.
        let n_support = support_indices.len();
        let mut support_vectors = Array2::zeros((n_support, n_features));
        for (i, &support_idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(support_idx));
        }

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

        let svr = SVR::new()
            .linear()
            .c(1.0)
            .epsilon(0.1)
            .fit(&x, &y)
            .expect("model fitting should succeed");

        // Check fitted attributes
        assert!(svr.support_vectors().nrows() > 0);
        assert!(!svr.dual_coef().is_empty());

        // Test prediction
        let x_test = array![[3.0]];
        let predictions = svr.predict(&x_test).expect("prediction should succeed");

        assert_eq!(predictions.len(), 1);
        // The SMO solver should recover the linear trend y = 2*x, so the
        // prediction at x = 3 should be close to 6 (within epsilon + slack).
        assert!(
            (predictions[0] - 6.0).abs() < 1.0,
            "SVR prediction {} too far from expected 6.0",
            predictions[0]
        );
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
            .expect("operation should succeed");

        assert!(svr.support_vectors().nrows() > 0);
    }

    #[test]
    fn test_epsilon_svr() {
        let x = array![[1.0], [2.0], [3.0],];
        let y = array![1.0, 2.0, 3.0];

        let epsilon_svr = EpsilonSVR::new()
            .epsilon(0.5)
            .c(1.0)
            .fit(&x, &y)
            .expect("model fitting should succeed");

        assert!(epsilon_svr.support_vectors().nrows() > 0);

        let predictions = epsilon_svr.predict(&x).expect("prediction should succeed");
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
