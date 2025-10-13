//! Primal-Dual Methods for SVM Optimization
//!
//! This module implements primal-dual optimization algorithms for SVMs that
//! solve the primal and dual problems simultaneously, often leading to faster
//! convergence and better numerical stability.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};

/// Primal-Dual Support Vector Machine
///
/// This implementation uses primal-dual optimization methods that solve
/// both the primal and dual SVM formulations simultaneously. This approach
/// can offer better convergence properties and numerical stability compared
/// to traditional methods.
///
/// # Parameters
/// * `C` - Regularization parameter (default: 1.0)
/// * `loss` - Loss function type ('hinge' or 'squared_hinge', default: 'squared_hinge')
/// * `sigma` - Primal step size (default: 0.1)
/// * `tau` - Dual step size (default: 0.1)
/// * `theta` - Over-relaxation parameter (default: 1.0)
/// * `tol` - Tolerance for stopping criterion (default: 1e-4)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `adaptive_step_size` - Use adaptive step size adjustment (default: true)
/// * `verbose` - Enable verbose output (default: false)
///
/// # Example
/// ```rust
/// use sklears_svm::PrimalDualSVM;
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X_var = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
/// let y = array![0, 1, 1, 0];
///
/// let model = PrimalDualSVM::new()
///     .with_c(1.0)
///     .with_max_iter(1000);
///
/// let trained_model = model.fit(&X_var, &y).unwrap();
/// let predictions = trained_model.predict(&X_var).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PrimalDualSVM {
    /// Regularization parameter
    pub c: f64,
    /// Loss function ('hinge' or 'squared_hinge')
    pub loss: String,
    /// Primal step size
    pub sigma: f64,
    /// Dual step size
    pub tau: f64,
    /// Over-relaxation parameter
    pub theta: f64,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Use adaptive step size adjustment
    pub adaptive_step_size: bool,
    /// Verbose output
    pub verbose: bool,
    /// Random seed
    pub random_state: Option<u64>,
}

/// Trained Primal-Dual Support Vector Machine model
#[derive(Debug, Clone)]
pub struct TrainedPrimalDualSVM {
    /// Model weights (coefficients)
    pub coef_: Array2<f64>,
    /// Intercept terms
    pub intercept_: Array1<f64>,
    /// Unique class labels
    pub classes_: Array1<i32>,
    /// Number of features
    pub n_features_in_: usize,
    /// Final dual variables
    pub dual_coef_: Array1<f64>,
    /// Training parameters
    _params: PrimalDualSVM,
}

impl Default for PrimalDualSVM {
    fn default() -> Self {
        Self::new()
    }
}

impl PrimalDualSVM {
    /// Create a new PrimalDualSVM with default parameters
    pub fn new() -> Self {
        Self {
            c: 1.0,
            loss: "squared_hinge".to_string(),
            sigma: 0.1,
            tau: 0.1,
            theta: 1.0,
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            adaptive_step_size: true,
            verbose: false,
            random_state: None,
        }
    }

    /// Set the regularization parameter C
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the loss function
    pub fn with_loss(mut self, loss: &str) -> Self {
        self.loss = loss.to_string();
        self
    }

    /// Set the primal step size
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set the dual step size
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    /// Set the over-relaxation parameter
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set whether to fit an intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to use adaptive step size adjustment
    pub fn with_adaptive_step_size(mut self, adaptive: bool) -> Self {
        self.adaptive_step_size = adaptive;
        self
    }

    /// Set verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set random state for reproducible results
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Chambolle-Pock primal-dual algorithm for SVM
    fn chambolle_pock_algorithm(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        w: &mut Array1<f64>,
        dual_vars: &mut Array1<f64>,
        intercept: &mut f64,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        // Convert y to f64 with proper labels (-1, 1)
        let y_binary: Array1<f64> = y.map(|&label| if label == 1 { 1.0 } else { -1.0 });

        // Initialize variables
        let mut w_bar = w.clone(); // Over-relaxed primal variable
        let mut sigma = self.sigma;
        let mut tau = self.tau;

        // Precompute operator norm for step size bounds
        let mut operator_norm: f64 = 0.0;
        for i in 0..n_samples {
            let row_norm: f64 = x.row(i).iter().map(|&xi| xi * xi).sum();
            operator_norm = operator_norm.max(row_norm);
        }
        operator_norm = operator_norm.sqrt();

        // Ensure step size product is bounded for convergence
        if sigma * tau * operator_norm * operator_norm >= 1.0 {
            let scale = 0.99 / (operator_norm * operator_norm);
            sigma *= scale.sqrt();
            tau *= scale.sqrt();
        }

        for iteration in 0..self.max_iter {
            let old_w = w.clone();
            let old_dual = dual_vars.clone();

            // Update dual variables
            for i in 0..n_samples {
                let xi = x.row(i);
                let yi = y_binary[i];

                // Compute prediction with over-relaxed primal
                let mut prediction = if self.fit_intercept { *intercept } else { 0.0 };
                for j in 0..n_features {
                    prediction += w_bar[j] * xi[j];
                }

                // Dual update step
                let dual_update = dual_vars[i] + tau * yi * prediction;

                // Projection onto dual feasible set
                dual_vars[i] = match self.loss.as_str() {
                    "hinge" => {
                        // For hinge loss: 0 <= dual <= C
                        dual_update.max(0.0).min(self.c)
                    }
                    "squared_hinge" => {
                        // For squared hinge: dual can be negative
                        dual_update.max(-self.c).min(self.c)
                    }
                    _ => {
                        return Err(SklearsError::InvalidParameter {
                            name: "loss".to_string(),
                            reason: format!("Unknown loss: {}", self.loss),
                        });
                    }
                };
            }

            // Update primal variables
            let mut gradient: Array1<f64> = Array1::zeros(n_features);
            for i in 0..n_samples {
                let xi = x.row(i);
                let yi = y_binary[i];

                for j in 0..n_features {
                    gradient[j] += dual_vars[i] * yi * xi[j];
                }
            }

            // Primal update with L2 regularization
            for j in 0..n_features {
                w[j] = (w[j] - sigma * gradient[j]) / (1.0 + sigma / self.c);
            }

            // Update intercept if needed
            if self.fit_intercept {
                let mut intercept_gradient = 0.0;
                for i in 0..n_samples {
                    intercept_gradient += dual_vars[i] * y_binary[i];
                }
                *intercept -= sigma * intercept_gradient;
            }

            // Over-relaxation step
            for j in 0..n_features {
                w_bar[j] = w[j] + self.theta * (w[j] - old_w[j]);
            }

            // Adaptive step size adjustment
            if self.adaptive_step_size {
                let primal_change: f64 = w
                    .iter()
                    .zip(old_w.iter())
                    .map(|(new, old)| (new - old).powi(2))
                    .sum::<f64>()
                    .sqrt();

                let dual_change: f64 = dual_vars
                    .iter()
                    .zip(old_dual.iter())
                    .map(|(new, old)| (new - old).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if iteration > 10 {
                    if primal_change > 2.0 * dual_change {
                        tau *= 0.95;
                        sigma /= 0.95;
                    } else if dual_change > 2.0 * primal_change {
                        sigma *= 0.95;
                        tau /= 0.95;
                    }
                }
            }

            // Check convergence
            let primal_residual: f64 = w
                .iter()
                .zip(old_w.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum::<f64>()
                .sqrt();

            let dual_residual: f64 = dual_vars
                .iter()
                .zip(old_dual.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum::<f64>()
                .sqrt();

            if primal_residual < self.tol && dual_residual < self.tol {
                if self.verbose {
                    println!("Primal-Dual SVM converged at iteration {iteration}");
                }
                break;
            }

            if self.verbose && iteration % 100 == 0 {
                println!(
                    "PD iteration {iteration}, primal_res: {primal_residual:.6}, dual_res: {dual_residual:.6}"
                );
            }
        }

        Ok(())
    }

    /// ADMM (Alternating Direction Method of Multipliers) for SVM
    fn admm_algorithm(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        w: &mut Array1<f64>,
        dual_vars: &mut Array1<f64>,
        intercept: &mut f64,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        // Convert y to f64 with proper labels (-1, 1)
        let y_binary: Array1<f64> = y.map(|&label| if label == 1 { 1.0 } else { -1.0 });

        // ADMM auxiliary variables
        let mut z = Array1::zeros(n_samples); // Auxiliary variable for constraints
        let mut u = Array1::zeros(n_samples); // Dual variables (scaled Lagrange multipliers)

        let rho = 1.0; // Penalty parameter

        for iteration in 0..self.max_iter {
            let old_w = w.clone();
            let old_z = z.clone();

            // Update w (primal variable)
            // Solve: (X^T X + I/C + rho I) w = X^T (rho z - u)
            let mut xtx_plus_reg: Array2<f64> = Array2::zeros((n_features, n_features));
            let mut xty: Array1<f64> = Array1::zeros(n_features);

            // Compute X^T X + regularization
            for i in 0..n_features {
                for j in 0..n_features {
                    for k in 0..n_samples {
                        xtx_plus_reg[[i, j]] += x[[k, i]] * x[[k, j]];
                    }
                    if i == j {
                        xtx_plus_reg[[i, j]] += 1.0 / self.c + rho;
                    }
                }
            }

            // Compute X^T (rho z - u)
            for j in 0..n_features {
                for i in 0..n_samples {
                    xty[j] += x[[i, j]] * (rho * z[i] - u[i]);
                }
            }

            // Solve linear system (simplified - in practice would use Cholesky or QR)
            // For now, use gradient descent step
            let learning_rate = 0.01;
            for j in 0..n_features {
                let mut gradient = 0.0;
                for k in 0..n_features {
                    gradient += xtx_plus_reg[[j, k]] * w[k];
                }
                gradient -= xty[j];
                w[j] -= learning_rate * gradient;
            }

            // Update z (auxiliary variable with constraint)
            for i in 0..n_samples {
                let xi = x.row(i);
                let yi = y_binary[i];

                let mut prediction = if self.fit_intercept { *intercept } else { 0.0 };
                for j in 0..n_features {
                    prediction += w[j] * xi[j];
                }

                let z_update: f64 = prediction + u[i] / rho;

                // Soft thresholding for loss constraint
                match self.loss.as_str() {
                    "hinge" => {
                        // z >= 1 - yi * (Xw + b)
                        z[i] = z_update.max(1.0 - yi * prediction);
                    }
                    "squared_hinge" => {
                        // Quadratic constraint handling
                        let threshold = 1.0 / (rho * self.c);
                        z[i] = if z_update > threshold {
                            z_update - threshold
                        } else if z_update < -threshold {
                            z_update + threshold
                        } else {
                            0.0
                        };
                    }
                    _ => {
                        return Err(SklearsError::InvalidParameter {
                            name: "loss".to_string(),
                            reason: format!("Unknown loss: {}", self.loss),
                        })
                    }
                }
            }

            // Update dual variables u
            for i in 0..n_samples {
                let xi = x.row(i);
                let mut prediction = if self.fit_intercept { *intercept } else { 0.0 };
                for j in 0..n_features {
                    prediction += w[j] * xi[j];
                }
                u[i] += rho * (prediction - z[i]);
            }

            // Update intercept
            if self.fit_intercept {
                let mut intercept_update = 0.0;
                for i in 0..n_samples {
                    intercept_update += z[i] - u[i] / rho;
                }
                *intercept = intercept_update / n_samples as f64;
            }

            // Check convergence
            let primal_residual: f64 = w
                .iter()
                .zip(old_w.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum::<f64>()
                .sqrt();

            let dual_residual: f64 = z
                .iter()
                .zip(old_z.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum::<f64>()
                .sqrt();

            if primal_residual < self.tol && dual_residual < self.tol {
                if self.verbose {
                    println!("ADMM SVM converged at iteration {iteration}");
                }
                break;
            }

            if self.verbose && iteration % 100 == 0 {
                println!(
                    "ADMM iteration {iteration}, primal_res: {primal_residual:.6}, dual_res: {dual_residual:.6}"
                );
            }
        }

        // Store final dual variables for output
        for i in 0..n_samples {
            dual_vars[i] = u[i];
        }

        Ok(())
    }

    /// Solve binary classification problem using primal-dual methods
    fn solve_binary_problem(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        algorithm: &str,
    ) -> Result<(Array1<f64>, f64, Array1<f64>)> {
        let (n_samples, n_features) = x.dim();

        let mut w = Array1::zeros(n_features);
        let mut dual_vars = Array1::zeros(n_samples);
        let mut intercept = 0.0;

        match algorithm {
            "chambolle_pock" => {
                self.chambolle_pock_algorithm(x, y, &mut w, &mut dual_vars, &mut intercept)?;
            }
            "admm" => {
                self.admm_algorithm(x, y, &mut w, &mut dual_vars, &mut intercept)?;
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "algorithm".to_string(),
                    reason: format!("Unknown algorithm: {algorithm}"),
                })
            }
        }

        Ok((w, intercept, dual_vars))
    }
}

impl Estimator for PrimalDualSVM {
    type Config = Self;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<Array2<f64>, Array1<i32>> for PrimalDualSVM {
    type Fitted = TrainedPrimalDualSVM;

    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> Result<TrainedPrimalDualSVM> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        if x.len_of(Axis(0)) != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Get unique classes
        let mut classes: Vec<i32> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from(classes);

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        let n_classes = classes.len();

        // Use Chambolle-Pock by default (can be extended to allow selection)
        let algorithm = "chambolle_pock";

        // For binary classification, use single model
        if n_classes == 2 {
            // Convert labels to binary (0/1 -> -1/1)
            let y_binary = y.map(|&label| if label == classes[1] { 1 } else { -1 });

            let (w, intercept, dual_vars) =
                self.solve_binary_problem(x.view(), y_binary.view(), algorithm)?;

            let coef = w.insert_axis(Axis(0));
            let intercept_arr = Array1::from(vec![intercept]);

            Ok(TrainedPrimalDualSVM {
                coef_: coef,
                intercept_: intercept_arr,
                classes_: classes,
                n_features_in_: n_features,
                dual_coef_: dual_vars,
                _params: self,
            })
        } else {
            // Multi-class: One-vs-Rest approach
            let mut coef_matrix = Array2::zeros((n_classes, n_features));
            let mut intercept_vec = Array1::zeros(n_classes);
            let mut all_dual_vars = Array1::zeros(n_samples * n_classes);

            for (class_idx, &class_label) in classes.iter().enumerate() {
                // Create binary labels (current class vs rest)
                let y_binary = y.map(|&label| if label == class_label { 1 } else { -1 });

                let (w, intercept, dual_vars) =
                    self.solve_binary_problem(x.view(), y_binary.view(), algorithm)?;

                coef_matrix.row_mut(class_idx).assign(&w);
                intercept_vec[class_idx] = intercept;

                // Store dual variables for this class
                for (i, &dual_var) in dual_vars.iter().enumerate() {
                    all_dual_vars[class_idx * n_samples + i] = dual_var;
                }
            }

            Ok(TrainedPrimalDualSVM {
                coef_: coef_matrix,
                intercept_: intercept_vec,
                classes_: classes,
                n_features_in_: n_features,
                dual_coef_: all_dual_vars,
                _params: self,
            })
        }
    }
}

impl Predict<Array2<f64>, Array1<i32>> for TrainedPrimalDualSVM {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let decision_values = self.decision_function(x)?;

        if self.classes_.len() == 2 {
            // Binary classification
            let predictions = decision_values.map(|&score| {
                if score >= 0.0 {
                    self.classes_[1]
                } else {
                    self.classes_[0]
                }
            });
            Ok(predictions.remove_axis(Axis(1)))
        } else {
            // Multi-class: predict class with highest score
            let mut predictions = Array1::zeros(x.len_of(Axis(0)));
            for (i, row) in decision_values.axis_iter(Axis(0)).enumerate() {
                let best_class_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                predictions[i] = self.classes_[best_class_idx];
            }
            Ok(predictions)
        }
    }
}

impl TrainedPrimalDualSVM {
    /// Compute decision function values
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_ {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_,
                actual: n_features,
            });
        }

        if self.classes_.len() == 2 {
            // Binary classification: single decision function
            let mut scores = Array1::zeros(n_samples);
            let w = self.coef_.row(0);
            let intercept = self.intercept_[0];

            for (i, x_row) in x.axis_iter(Axis(0)).enumerate() {
                let mut score = intercept;
                for (j, &x_val) in x_row.iter().enumerate() {
                    score += w[j] * x_val;
                }
                scores[i] = score;
            }

            Ok(scores.insert_axis(Axis(1)))
        } else {
            // Multi-class: one score per class
            let mut scores = Array2::zeros((n_samples, self.classes_.len()));

            for (class_idx, coef_row) in self.coef_.axis_iter(Axis(0)).enumerate() {
                let intercept = self.intercept_[class_idx];

                for (i, x_row) in x.axis_iter(Axis(0)).enumerate() {
                    let mut score = intercept;
                    for (j, &x_val) in x_row.iter().enumerate() {
                        score += coef_row[j] * x_val;
                    }
                    scores[[i, class_idx]] = score;
                }
            }

            Ok(scores)
        }
    }

    /// Get the model coefficients
    pub fn coef(&self) -> &Array2<f64> {
        &self.coef_
    }

    /// Get the intercept terms
    pub fn intercept(&self) -> &Array1<f64> {
        &self.intercept_
    }

    /// Get the class labels
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes_
    }

    /// Get the dual coefficients from the optimization
    pub fn dual_coef(&self) -> &Array1<f64> {
        &self.dual_coef_
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_primal_dual_svm_binary_classification() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
        let y = array![0, 1, 1, 0];

        let model = PrimalDualSVM::new()
            .with_c(1.0)
            .with_max_iter(500)
            .with_verbose(true);
        let trained_model = model.fit(&X_var, &y).unwrap();

        let predictions = trained_model.predict(&X_var).unwrap();
        assert_eq!(predictions.len(), 4);

        // Test decision function
        let scores = trained_model.decision_function(&X_var).unwrap();
        assert_eq!(scores.dim(), (4, 1));

        // Test dual coefficients
        assert_eq!(trained_model.dual_coef().len(), 4);
    }

    #[test]
    fn test_primal_dual_svm_multiclass_classification() {
        let X_var = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let model = PrimalDualSVM::new().with_c(1.0).with_max_iter(500);
        let trained_model = model.fit(&X_var, &y).unwrap();

        let predictions = trained_model.predict(&X_var).unwrap();
        assert_eq!(predictions.len(), 6);

        // Test decision function for multiclass
        let scores = trained_model.decision_function(&X_var).unwrap();
        assert_eq!(scores.dim(), (6, 3)); // 6 samples, 3 classes
    }

    #[test]
    fn test_primal_dual_svm_parameters() {
        let model = PrimalDualSVM::new()
            .with_c(0.5)
            .with_loss("hinge")
            .with_sigma(0.05)
            .with_tau(0.05)
            .with_theta(0.8)
            .with_tol(1e-5)
            .with_max_iter(500)
            .with_fit_intercept(false)
            .with_adaptive_step_size(false);

        assert_eq!(model.c, 0.5);
        assert_eq!(model.loss, "hinge");
        assert_abs_diff_eq!(model.sigma, 0.05);
        assert_abs_diff_eq!(model.tau, 0.05);
        assert_abs_diff_eq!(model.theta, 0.8);
        assert_abs_diff_eq!(model.tol, 1e-5);
        assert_eq!(model.max_iter, 500);
        assert!(!model.fit_intercept);
        assert!(!model.adaptive_step_size);
    }

    #[test]
    fn test_primal_dual_convergence() {
        let X_var = array![[2.0, 3.0], [3.0, 3.0], [1.0, 1.0], [2.0, 1.0]];
        let y = array![1, 1, 0, 0];

        let model = PrimalDualSVM::new()
            .with_c(1.0)
            .with_max_iter(1000)
            .with_tol(1e-6)
            .with_adaptive_step_size(true);

        let trained = model.fit(&X_var, &y).unwrap();
        let predictions = trained.predict(&X_var).unwrap();

        // Check that we have reasonable accuracy
        let accuracy = predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &actual)| if pred == actual { 1.0 } else { 0.0 })
            .sum::<f64>()
            / predictions.len() as f64;

        assert!(
            accuracy >= 0.5,
            "Primal-Dual accuracy too low: {}",
            accuracy
        );
    }

    #[test]
    fn test_step_size_bounds() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 1];

        // Test with very large step sizes that should be automatically adjusted
        let model = PrimalDualSVM::new()
            .with_sigma(10.0)
            .with_tau(10.0)
            .with_max_iter(100);

        let result = model.fit(&X_var, &y);
        assert!(result.is_ok(), "Should handle large step sizes gracefully");
    }

    #[test]
    fn test_different_loss_functions() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
        let y = array![0, 1, 1, 0];

        // Test hinge loss
        let model_hinge = PrimalDualSVM::new()
            .with_loss("hinge")
            .with_c(1.0)
            .with_max_iter(500);

        let result_hinge = model_hinge.fit(&X_var, &y);
        assert!(result_hinge.is_ok());

        // Test squared hinge loss
        let model_squared = PrimalDualSVM::new()
            .with_loss("squared_hinge")
            .with_c(1.0)
            .with_max_iter(500);

        let result_squared = model_squared.fit(&X_var, &y);
        assert!(result_squared.is_ok());

        // Test invalid loss
        let model_invalid = PrimalDualSVM::new().with_loss("invalid_loss");
        let result_invalid = model_invalid.fit(&X_var, &y);
        assert!(result_invalid.is_err());
    }
}
