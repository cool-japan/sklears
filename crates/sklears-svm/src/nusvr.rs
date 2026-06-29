//! Nu Support Vector Regression
//!
//! This module implements Nu-SVR, an alternative formulation of SVM regression
//! that uses a parameter nu instead of C and epsilon for controlling the
//! regularization and error tolerance.

use crate::kernels::{Kernel, KernelType};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Nu Support Vector Regression Configuration
#[derive(Debug, Clone)]
pub struct NuSVRConfig {
    /// Nu parameter (0 < nu <= 1)
    pub nu: Float,
    /// Kernel function to use
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Random seed
    pub random_state: Option<u64>,
}

impl Default for NuSVRConfig {
    fn default() -> Self {
        Self {
            nu: 0.5,
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 200,
            random_state: None,
        }
    }
}

/// Nu Support Vector Regression
///
/// Nu-SVR is an alternative formulation of SVR that uses a parameter nu
/// instead of C and epsilon. The parameter nu controls the fraction of
/// support vectors and roughly corresponds to the fraction of training
/// points that lie outside the epsilon-tube.
#[derive(Debug)]
pub struct NuSVR<State = Untrained> {
    config: NuSVRConfig,
    state: PhantomData<State>,
    // Fitted attributes
    support_vectors_: Option<Array2<Float>>,
    support_: Option<Array1<usize>>,
    dual_coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_in_: Option<usize>,
    n_support_: Option<usize>,
    epsilon_: Option<Float>,
}

impl NuSVR<Untrained> {
    /// Create a new Nu-SVR regressor
    pub fn new() -> Self {
        Self {
            config: NuSVRConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            support_: None,
            dual_coef_: None,
            intercept_: None,
            n_features_in_: None,
            n_support_: None,
            epsilon_: None,
        }
    }

    /// Set the nu parameter (0 < nu <= 1)
    pub fn nu(mut self, nu: Float) -> Result<Self> {
        if nu <= 0.0 || nu > 1.0 {
            return Err(SklearsError::InvalidParameter {
                name: "nu".to_string(),
                reason: "must be in the range (0, 1]".to_string(),
            });
        }
        self.config.nu = nu;
        Ok(self)
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
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

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }
}

impl Default for NuSVR<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for NuSVR<Untrained> {
    type Fitted = NuSVR<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Shape mismatch: X has {} samples, y has {} samples",
                x.nrows(),
                y.len()
            )));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Box constraint C for the dual variables. In the nu-SVR formulation,
        // each dual variable is bounded by C/n so that the total mass relates
        // directly to the nu fraction (libsvm convention with C = 1).
        let c = 1.0 / n_samples as Float;

        // Solve the nu-SVR dual via SMO. This determines both the dual
        // coefficients and the effective epsilon-tube width.
        let (dual_coef_full, intercept, epsilon) = solve_nu_svr_dual(
            &self.config.kernel,
            x,
            y,
            self.config.nu,
            c,
            self.config.tol,
            self.config.max_iter,
        )?;

        // Support vectors are samples whose signed dual coefficient is non-zero.
        let sv_threshold = 1e-8 * c.max(1.0);
        let mut support_indices = Vec::new();
        let mut dual_coef_vec = Vec::new();
        for i in 0..n_samples {
            if dual_coef_full[i].abs() > sv_threshold {
                support_indices.push(i);
                dual_coef_vec.push(dual_coef_full[i]);
            }
        }

        if support_indices.is_empty() {
            // Degenerate fallback: retain the most influential sample so that
            // prediction remains well-defined rather than fabricating a mean.
            let best = (0..n_samples)
                .max_by(|&a, &b| {
                    dual_coef_full[a]
                        .abs()
                        .partial_cmp(&dual_coef_full[b].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            support_indices.push(best);
            dual_coef_vec.push(dual_coef_full[best]);
        }

        let n_support = support_indices.len();
        let mut support_vectors = Array2::zeros((n_support, n_features));
        for (i, &idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(idx));
        }
        let support = Array1::from_vec(support_indices);
        let dual_coef = Array1::from_vec(dual_coef_vec);

        Ok(NuSVR {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(support_vectors),
            support_: Some(support),
            dual_coef_: Some(dual_coef),
            intercept_: Some(intercept),
            n_features_in_: Some(n_features),
            n_support_: Some(n_support),
            epsilon_: Some(epsilon),
        })
    }
}

/// Solve the nu-SVR dual problem with Sequential Minimal Optimization.
///
/// The nu-SVR dual is:
/// ```text
/// minimize:  ½ (α - α*)ᵀ Q (α - α*) - Σ y_i (α_i - α_i*)
/// subject to: Σ(α_i - α_i*) = 0
///             Σ(α_i + α_i*) ≤ C · nu · n
///             0 ≤ α_i, α_i* ≤ C
/// ```
/// where `Q_ij = K(x_i, x_j)`. Unlike epsilon-SVR, the tube width epsilon is
/// not fixed; it emerges from the second (inequality) constraint, which we
/// enforce by bounding each coordinate step by the remaining nu-budget.
///
/// We track the signed coefficient `w_i = α_i - α_i*` and the total mass
/// `Σ(α_i + α_i*)`. The maximal-violating pair is selected from the gradient
/// `g_i = decision_i - y_i`. After convergence epsilon is recovered from the
/// free support vectors as the mean of `|decision_i - y_i|`.
#[allow(clippy::too_many_arguments)]
fn solve_nu_svr_dual(
    kernel: &KernelType,
    x: &Array2<Float>,
    y: &Array1<Float>,
    nu: Float,
    c: Float,
    tol: Float,
    max_iter: usize,
) -> Result<(Array1<Float>, Float, Float)> {
    let n_samples = y.len();

    let kernel_fn = crate::kernels::create_kernel(kernel.clone())?;
    let mut q = Array2::<Float>::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in i..n_samples {
            let val = kernel_fn.compute(x.row(i), x.row(j));
            q[[i, j]] = val;
            q[[j, i]] = val;
        }
    }

    let mut alpha = Array1::<Float>::zeros(n_samples);
    let mut alpha_star = Array1::<Float>::zeros(n_samples);
    let mut decision = Array1::<Float>::zeros(n_samples);

    // Total budget for Σ(α_i + α_i*).
    let budget = c * nu * n_samples as Float;

    let max_total_iter = max_iter.max(1) * n_samples.max(1);

    for _iter in 0..max_total_iter {
        let mass: Float = (0..n_samples).map(|k| alpha[k] + alpha_star[k]).sum();
        let mass_remaining = (budget - mass).max(0.0);

        // Select maximal-violating pair from the residual gradient.
        let mut i_up = None;
        let mut g_max = Float::NEG_INFINITY;
        let mut i_low = None;
        let mut g_min = Float::INFINITY;

        for i in 0..n_samples {
            let r_i = decision[i] - y[i];
            let can_increase = alpha[i] < c - tol || alpha_star[i] > tol;
            let can_decrease = alpha_star[i] < c - tol || alpha[i] > tol;
            if can_increase && -r_i > g_max {
                g_max = -r_i;
                i_up = Some(i);
            }
            if can_decrease && -r_i < g_min {
                g_min = -r_i;
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

        let eta = q[[i, i]] + q[[j, j]] - 2.0 * q[[i, j]];
        if eta <= 1e-12 {
            continue;
        }

        let w_i_old = alpha[i] - alpha_star[i];
        let w_j_old = alpha[j] - alpha_star[j];

        let r_i = decision[i] - y[i];
        let r_j = decision[j] - y[j];
        let unconstrained = (r_j - r_i) / eta;

        // Limit the step so the total mass does not exceed the nu-budget.
        let mass_step_cap = (mass_remaining / 2.0).max(0.0);
        let mut step = unconstrained.clamp(-mass_step_cap, mass_step_cap);

        // Box-project both coefficients into [-C, C].
        let w_i_new = (w_i_old + step).clamp(-c, c);
        let w_j_new = (w_j_old - step).clamp(-c, c);
        let actual_i = w_i_new - w_i_old;
        let actual_j = w_j_new - w_j_old;
        step = if actual_i.abs() <= actual_j.abs() {
            actual_i
        } else {
            -actual_j
        };

        if step.abs() < 1e-12 {
            if mass_remaining < tol {
                break;
            }
            continue;
        }

        let w_i_final = w_i_old + step;
        let w_j_final = w_j_old - step;

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

    let mut dual_coef = Array1::<Float>::zeros(n_samples);
    for i in 0..n_samples {
        dual_coef[i] = alpha[i] - alpha_star[i];
    }

    // Recover epsilon as the mean absolute residual over free support vectors,
    // and the bias as the mean residual over the same set.
    let mut eps_sum = 0.0;
    let mut eps_count = 0usize;
    let mut bias_sum = 0.0;
    let mut bias_count = 0usize;
    for i in 0..n_samples {
        let w_i = dual_coef[i];
        let is_free = (w_i > tol && w_i < c - tol) || (w_i < -tol && w_i > -(c - tol));
        if is_free {
            let residual = y[i] - decision[i];
            eps_sum += residual.abs();
            eps_count += 1;
            bias_sum += residual;
            bias_count += 1;
        }
    }

    let epsilon = if eps_count > 0 {
        eps_sum / eps_count as Float
    } else {
        0.0
    };

    let intercept = if bias_count > 0 {
        bias_sum / bias_count as Float
    } else {
        let mut sum = 0.0;
        for i in 0..n_samples {
            sum += y[i] - decision[i];
        }
        sum / n_samples as Float
    };

    Ok((dual_coef, intercept, epsilon))
}

impl Predict<Array2<Float>, Array1<Float>> for NuSVR<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols()
            != self
                .n_features_in_
                .expect("n_features_in_ not available - model not fitted")
        {
            return Err(SklearsError::InvalidInput(format!(
                "Feature mismatch: expected {} features, got {}",
                self.n_features_in_
                    .expect("n_features_in_ not available - model not fitted"),
                x.ncols()
            )));
        }

        let support_vectors = self
            .support_vectors_
            .as_ref()
            .expect("support_vectors_ not available - model not fitted");
        let dual_coef = self
            .dual_coef_
            .as_ref()
            .expect("dual_coef_ not available - model not fitted");
        let intercept = self
            .intercept_
            .expect("intercept_ not available - model not fitted");

        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>, // Default fallback
        };
        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut prediction = intercept;
            for (j, &coef) in dual_coef.iter().enumerate() {
                let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                prediction += coef * k_val;
            }
            predictions[i] = prediction;
        }

        Ok(predictions)
    }
}

impl NuSVR<Trained> {
    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_
            .as_ref()
            .expect("support_vectors_ not available - model not fitted")
    }

    /// Get the indices of support vectors
    pub fn support(&self) -> &Array1<usize> {
        self.support_
            .as_ref()
            .expect("support_ not available - model not fitted")
    }

    /// Get the dual coefficients
    pub fn dual_coef(&self) -> &Array1<Float> {
        self.dual_coef_
            .as_ref()
            .expect("dual_coef_ not available - model not fitted")
    }

    /// Get the intercept
    pub fn intercept(&self) -> Float {
        self.intercept_
            .expect("intercept_ not available - model not fitted")
    }

    /// Get the number of support vectors
    pub fn n_support(&self) -> usize {
        self.n_support_
            .expect("n_support_ not available - model not fitted")
    }

    /// Get the number of features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
            .expect("n_features_in_ not available - model not fitted")
    }

    /// Get the epsilon parameter
    pub fn epsilon(&self) -> Float {
        self.epsilon_
            .expect("epsilon_ not available - model not fitted")
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nusvr_creation() {
        let nusvr = NuSVR::new()
            .nu(0.3)
            .expect("valid parameter")
            .kernel(KernelType::Linear)
            .tol(1e-4)
            .max_iter(500)
            .random_state(42);

        assert_eq!(nusvr.config.nu, 0.3);
        assert_eq!(nusvr.config.tol, 1e-4);
        assert_eq!(nusvr.config.max_iter, 500);
        assert_eq!(nusvr.config.random_state, Some(42));
    }

    #[test]
    fn test_nusvr_invalid_nu() {
        let result = NuSVR::new().nu(1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_nusvr_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]; // y = 2*x

        let nusvr = NuSVR::new()
            .nu(0.5)
            .expect("valid parameter")
            .kernel(KernelType::Linear);
        let fitted_model = nusvr.fit(&x, &y).expect("model fitting should succeed");

        assert_eq!(fitted_model.n_features_in(), 1);
        assert!(fitted_model.epsilon() > 0.0);

        let predictions = fitted_model.predict(&x).expect("prediction should succeed");
        assert_eq!(predictions.len(), 6);

        // Check that predictions are finite
        for &pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_nusvr_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0]; // Wrong length

        let nusvr = NuSVR::new();
        let result = nusvr.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_nusvr_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![1.0, 2.0];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let nusvr = NuSVR::new();
        let fitted_model = nusvr
            .fit(&x_train, &y_train)
            .expect("model fitting should succeed");
        let result = fitted_model.predict(&x_test);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Feature"));
    }

    #[test]
    fn test_nusvr_empty_data() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);

        let nusvr = NuSVR::new();
        let result = nusvr.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Empty dataset"));
    }

    #[test]
    fn test_nusvr_different_kernels() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0]; // y = x^2

        let kernels = vec![
            KernelType::Linear,
            KernelType::Rbf { gamma: 0.1 },
            KernelType::Polynomial {
                gamma: 1.0,
                degree: 2.0,
                coef0: 0.0,
            },
        ];

        for kernel in kernels {
            let nusvr = NuSVR::new().kernel(kernel);
            let fitted_model = nusvr.fit(&x, &y).expect("model fitting should succeed");
            let predictions = fitted_model.predict(&x).expect("prediction should succeed");

            assert_eq!(predictions.len(), 4);
            for &pred in predictions.iter() {
                assert!(pred.is_finite());
            }
        }
    }
}
