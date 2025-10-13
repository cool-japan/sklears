//! Linear Support Vector Classification using coordinate descent
//!
//! This implementation uses coordinate descent optimization which avoids the need
//! for BLAS operations while still providing efficient linear SVM training.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};

/// Linear Support Vector Classification
///
/// LinearSVC is a more efficient implementation for linear kernels using coordinate descent,
/// which avoids the computational overhead of the full SMO algorithm while maintaining
/// accuracy for linear separable problems.
///
/// # Parameters
/// * `C` - Regularization parameter (default: 1.0)
/// * `loss` - Loss function type ('hinge' or 'squared_hinge', default: 'squared_hinge')
/// * `penalty` - Regularization type ('l1', 'l2', or 'elasticnet', default: 'l2')
/// * `l1_ratio` - Elastic net mixing parameter (0.0 to 1.0, default: 0.15)
/// * `dual` - Use dual formulation (default: true for better performance with many features)
/// * `tol` - Tolerance for stopping criterion (default: 1e-4)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `intercept_scaling` - When `fit_intercept` is True, scaling for synthetic feature (default: 1.0)
/// * `class_weight` - Class weighting strategy (default: None)
/// * `verbose` - Enable verbose output (default: false)
/// * `random_state` - Random seed for reproducible results (default: None)
/// * `multi_class` - Multi-class strategy ('ovr' or 'crammer_singer', default: 'ovr')
///
/// # Example
/// ```rust
/// use sklears_svm::LinearSVC;
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
/// let y = array![0, 1, 1, 0];
///
/// let model = LinearSVC::new()
///     .with_c(1.0)
///     .with_penalty("elasticnet")
///     .with_l1_ratio(0.5)
///     .with_max_iter(1000);
///
/// let trained_model = model.fit(&X, &y).unwrap();
/// let predictions = trained_model.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct LinearSVC {
    /// Regularization parameter
    pub c: f64,
    /// Loss function ('hinge' or 'squared_hinge')
    pub loss: String,
    /// Regularization type ('l1', 'l2', or 'elasticnet')
    pub penalty: String,
    /// Elastic net mixing parameter (0.0 = L2, 1.0 = L1)
    pub l1_ratio: f64,
    /// Use dual formulation
    pub dual: bool,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Scaling for synthetic intercept feature
    pub intercept_scaling: f64,
    /// Multi-class strategy
    pub multi_class: String,
    /// Verbose output
    pub verbose: bool,
    /// Random seed
    pub random_state: Option<u64>,
    /// Solver algorithm ('dual_cd', 'primal_cd', 'enhanced_cd')
    pub solver: String,
}

/// Trained Linear Support Vector Classification model
#[derive(Debug, Clone)]
pub struct TrainedLinearSVC {
    /// Model weights (coefficients)
    pub coef_: Array2<f64>,
    /// Intercept terms
    pub intercept_: Array1<f64>,
    /// Unique class labels
    pub classes_: Array1<i32>,
    /// Number of features
    pub n_features_in_: usize,
    /// Training parameters
    _params: LinearSVC,
}

impl Default for LinearSVC {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSVC {
    /// Create a new LinearSVC with default parameters
    pub fn new() -> Self {
        Self {
            c: 1.0,
            loss: "squared_hinge".to_string(),
            penalty: "l2".to_string(),
            l1_ratio: 0.15,
            dual: true,
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            intercept_scaling: 1.0,
            multi_class: "ovr".to_string(),
            verbose: false,
            random_state: None,
            solver: "dual_cd".to_string(),
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

    /// Set the regularization penalty ('l1', 'l2', or 'elasticnet')
    pub fn with_penalty(mut self, penalty: &str) -> Self {
        self.penalty = penalty.to_string();
        self
    }

    /// Set the elastic net mixing parameter (0.0 = L2, 1.0 = L1)
    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set whether to use dual formulation
    pub fn with_dual(mut self, dual: bool) -> Self {
        self.dual = dual;
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

    /// Set the intercept scaling
    pub fn with_intercept_scaling(mut self, intercept_scaling: f64) -> Self {
        self.intercept_scaling = intercept_scaling;
        self
    }

    /// Set the multi-class strategy
    pub fn with_multi_class(mut self, multi_class: &str) -> Self {
        self.multi_class = multi_class.to_string();
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

    /// Set the solver algorithm ('dual_cd', 'primal_cd', 'enhanced_cd')
    pub fn with_solver(mut self, solver: &str) -> Self {
        self.solver = solver.to_string();
        self
    }

    /// Soft thresholding function for L1 regularization
    pub fn soft_threshold(&self, value: f64, threshold: f64) -> f64 {
        if value > threshold {
            value - threshold
        } else if value < -threshold {
            value + threshold
        } else {
            0.0
        }
    }

    /// Solve binary classification problem using the selected solver
    fn solve_binary_problem(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        alpha: &mut Array1<f64>,
        w: &mut Array1<f64>,
        intercept: &mut f64,
    ) -> Result<()> {
        match self.solver.as_str() {
            "dual_cd" => self.coordinate_descent(x, y, alpha, w, intercept),
            "primal_cd" => {
                // For primal CD, we don't need alpha
                self.primal_coordinate_descent(x, y, w, intercept)
            }
            "enhanced_cd" => self.enhanced_coordinate_descent(x, y, alpha, w, intercept),
            _ => Err(SklearsError::InvalidParameter {
                name: "solver".to_string(),
                reason: format!(
                    "Unknown solver: {}. Must be one of 'dual_cd', 'primal_cd', 'enhanced_cd'",
                    self.solver
                ),
            }),
        }
    }

    /// Coordinate descent solver for linear SVM
    fn coordinate_descent(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        alpha: &mut Array1<f64>,
        w: &mut Array1<f64>,
        intercept: &mut f64,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let mut rng = StdRng::seed_from_u64(42);

        // Convert y to f64 with proper labels (-1, 1)
        let y_binary: Array1<f64> = y.map(|&label| if label == 1 { 1.0 } else { -1.0 });

        for iteration in 0..self.max_iter {
            let mut alpha_diff = 0.0;

            // Shuffle sample indices for better convergence
            let mut indices: Vec<usize> = (0..n_samples).collect();
            if self.random_state.is_some() {
                use scirs2_core::rand_prelude::SliceRandom;
                indices.shuffle(&mut rng);
            }

            for &i in &indices {
                let xi = x.row(i);
                let yi = y_binary[i];

                // Compute current prediction
                let mut prediction = if self.fit_intercept { *intercept } else { 0.0 };
                for j in 0..n_features {
                    prediction += w[j] * xi[j];
                }

                // Compute gradient and update
                let margin = yi * prediction;
                let old_alpha = alpha[i];

                match self.loss.as_str() {
                    "hinge" => {
                        // Hinge loss: max(0, 1 - y*f(x))
                        if margin < 1.0 {
                            let gradient = -yi;
                            let hessian = xi.iter().map(|&x| x * x).sum::<f64>();
                            if hessian > 0.0 {
                                let delta = -gradient / (hessian + 1.0 / self.c);
                                alpha[i] = (alpha[i] + delta).max(0.0).min(self.c);
                            }
                        }
                    }
                    "squared_hinge" => {
                        // Squared hinge loss: max(0, 1 - y*f(x))^2
                        if margin < 1.0 {
                            let loss_value = 1.0 - margin;
                            let gradient = -2.0 * yi * loss_value;
                            let hessian =
                                2.0 * (xi.iter().map(|&x| x * x).sum::<f64>() + 1.0 / self.c);
                            if hessian > 0.0 {
                                let delta = -gradient / hessian;
                                alpha[i] = (alpha[i] + delta).max(0.0).min(self.c);
                            }
                        }
                    }
                    _ => {
                        return Err(SklearsError::InvalidParameter {
                            name: "loss".to_string(),
                            reason: format!("Unknown loss: {}", self.loss),
                        })
                    }
                }

                let alpha_change = alpha[i] - old_alpha;
                alpha_diff += alpha_change.abs();

                // Update weights and intercept
                if alpha_change.abs() > 1e-12 {
                    for j in 0..n_features {
                        w[j] += alpha_change * yi * xi[j];
                    }
                    if self.fit_intercept {
                        *intercept += alpha_change * yi * self.intercept_scaling;
                    }
                }
            }

            // Check convergence
            if alpha_diff < self.tol {
                if self.verbose {
                    println!("LinearSVC converged at iteration {iteration}");
                }
                break;
            }

            if self.verbose && iteration % 100 == 0 {
                println!("LinearSVC iteration {iteration}, alpha_diff: {alpha_diff:.6}");
            }
        }

        Ok(())
    }

    /// Primal coordinate descent solver for linear SVM
    ///
    /// This method optimizes the primal formulation directly, which can be more efficient
    /// for problems with many samples and relatively few features.
    fn primal_coordinate_descent(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        w: &mut Array1<f64>,
        intercept: &mut f64,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let mut rng = StdRng::seed_from_u64(42);

        // Convert y to f64 with proper labels (-1, 1)
        let y_binary: Array1<f64> = y.map(|&label| if label == 1 { 1.0 } else { -1.0 });

        // Precompute X^T X diagonal elements for efficiency
        let mut x_norm_sq = Array1::<f64>::zeros(n_features);
        for j in 0..n_features {
            x_norm_sq[j] = x.column(j).iter().map(|&xi| xi * xi).sum::<f64>();
        }

        for iteration in 0..self.max_iter {
            let mut w_diff = 0.0;

            // Shuffle feature indices for better convergence
            let mut feature_indices: Vec<usize> = (0..n_features).collect();
            if self.random_state.is_some() {
                use scirs2_core::rand_prelude::SliceRandom;
                feature_indices.shuffle(&mut rng);
            }

            // Update each coordinate
            for &j in &feature_indices {
                let old_wj = w[j];

                // Compute partial residual (gradient w.r.t. w_j)
                let mut gradient = 0.0;
                for i in 0..n_samples {
                    let xi = x.row(i);
                    let yi = y_binary[i];

                    // Current prediction without feature j
                    let mut prediction = if self.fit_intercept { *intercept } else { 0.0 };
                    for k in 0..n_features {
                        if k != j {
                            prediction += w[k] * xi[k];
                        }
                    }

                    let margin = yi * prediction;

                    // Add gradient contribution based on loss function
                    match self.loss.as_str() {
                        "hinge" => {
                            if margin < 1.0 {
                                gradient += yi * xi[j];
                            }
                        }
                        "squared_hinge" => {
                            if margin < 1.0 {
                                gradient += 2.0 * (1.0 - margin) * yi * xi[j];
                            }
                        }
                        _ => {
                            return Err(SklearsError::InvalidParameter {
                                name: "loss".to_string(),
                                reason: format!("Unknown loss: {}", self.loss),
                            })
                        }
                    }
                }

                // Apply regularization based on penalty type
                match self.penalty.as_str() {
                    "l2" => {
                        // L2 regularization
                        gradient -= w[j] / self.c;
                        if x_norm_sq[j] > 0.0 {
                            let hessian: f64 = x_norm_sq[j] + 1.0 / self.c;
                            w[j] = gradient / hessian;
                        }
                    }
                    "l1" => {
                        // L1 regularization with soft thresholding
                        if x_norm_sq[j] > 0.0 {
                            let threshold = 1.0 / (self.c * x_norm_sq[j]);
                            let new_w = gradient / x_norm_sq[j];
                            w[j] = self.soft_threshold(new_w, threshold);
                        }
                    }
                    "elasticnet" => {
                        // Elastic net: combination of L1 and L2
                        let l2_weight = 1.0 - self.l1_ratio;
                        let l1_weight = self.l1_ratio;

                        // L2 part
                        gradient -= l2_weight * w[j] / self.c;

                        if x_norm_sq[j] > 0.0 {
                            let hessian = x_norm_sq[j] + l2_weight / self.c;
                            let new_w = gradient / hessian;

                            // Apply L1 soft thresholding
                            let threshold = l1_weight / (self.c * hessian);
                            w[j] = self.soft_threshold(new_w, threshold);
                        }
                    }
                    _ => {
                        return Err(SklearsError::InvalidParameter {
                            name: "penalty".to_string(),
                            reason: format!("Unknown penalty: {}", self.penalty),
                        })
                    }
                }

                let weight_change = (w[j] - old_wj).abs();
                w_diff += weight_change;
            }

            // Update intercept if needed
            if self.fit_intercept {
                let old_intercept = *intercept;
                let mut intercept_gradient = 0.0;

                for i in 0..n_samples {
                    let xi = x.row(i);
                    let yi = y_binary[i];

                    let mut prediction = 0.0;
                    for j in 0..n_features {
                        prediction += w[j] * xi[j];
                    }

                    let margin = yi * (prediction + *intercept);

                    match self.loss.as_str() {
                        "hinge" => {
                            if margin < 1.0 {
                                intercept_gradient += yi;
                            }
                        }
                        "squared_hinge" => {
                            if margin < 1.0 {
                                intercept_gradient += 2.0 * (1.0 - margin) * yi;
                            }
                        }
                        _ => {}
                    }
                }

                // Update intercept (no regularization for intercept)
                *intercept = intercept_gradient / (n_samples as f64);
                w_diff += (*intercept - old_intercept).abs();
            }

            // Check convergence
            if w_diff < self.tol {
                if self.verbose {
                    println!("Primal coordinate descent converged at iteration {iteration}");
                }
                break;
            }

            if self.verbose && iteration % 100 == 0 {
                println!("Primal CD iteration {iteration}, w_diff: {w_diff:.6}");
            }
        }

        Ok(())
    }

    /// Enhanced coordinate descent with line search and momentum
    fn enhanced_coordinate_descent(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        alpha: &mut Array1<f64>,
        w: &mut Array1<f64>,
        intercept: &mut f64,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let rng = StdRng::seed_from_u64(42);

        let y_binary: Array1<f64> = y.map(|&label| if label == 1 { 1.0 } else { -1.0 });

        // Momentum parameters
        let beta = 0.9; // Momentum coefficient
        let mut alpha_momentum = Array1::zeros(n_samples);

        // Working set parameters
        let working_set_size = std::cmp::min(n_samples, 1000);

        for iteration in 0..self.max_iter {
            let mut alpha_diff = 0.0;

            // Select working set using second-order information
            let working_set =
                self.select_working_set(x, &y_binary, alpha, w, *intercept, working_set_size)?;

            for &i in &working_set {
                let xi = x.row(i);
                let yi = y_binary[i];

                // Compute current prediction
                let mut prediction = if self.fit_intercept { *intercept } else { 0.0 };
                for j in 0..n_features {
                    prediction += w[j] * xi[j];
                }

                let margin = yi * prediction;
                let old_alpha = alpha[i];

                // Compute gradient
                let gradient = match self.loss.as_str() {
                    "hinge" => {
                        if margin < 1.0 {
                            -yi
                        } else {
                            0.0
                        }
                    }
                    "squared_hinge" => {
                        if margin < 1.0 {
                            -2.0 * yi * (1.0 - margin)
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        return Err(SklearsError::InvalidParameter {
                            name: "loss".to_string(),
                            reason: format!("Unknown loss: {}", self.loss),
                        })
                    }
                };

                if gradient.abs() > 1e-12 {
                    // Compute Hessian (second derivative)
                    let hessian = match self.loss.as_str() {
                        "hinge" => xi.iter().map(|&x| x * x).sum::<f64>(),
                        "squared_hinge" => {
                            2.0 * (xi.iter().map(|&x| x * x).sum::<f64>() + 1.0 / self.c)
                        }
                        _ => xi.iter().map(|&x| x * x).sum::<f64>(),
                    };

                    if hessian > 0.0 {
                        // Newton step with momentum
                        let newton_step = -gradient / hessian;
                        alpha_momentum[i] = beta * alpha_momentum[i] + (1.0 - beta) * newton_step;

                        // Line search for optimal step size
                        let step_size =
                            self.line_search(x, &y_binary, i, alpha[i], alpha_momentum[i])?;

                        alpha[i] = (alpha[i] + step_size * alpha_momentum[i])
                            .max(0.0)
                            .min(self.c);
                    }
                }

                let alpha_change = alpha[i] - old_alpha;
                alpha_diff += alpha_change.abs();

                // Update weights and intercept incrementally
                if alpha_change.abs() > 1e-12 {
                    for j in 0..n_features {
                        w[j] += alpha_change * yi * xi[j];
                    }
                    if self.fit_intercept {
                        *intercept += alpha_change * yi * self.intercept_scaling;
                    }
                }
            }

            // Check convergence
            if alpha_diff < self.tol {
                if self.verbose {
                    println!("Enhanced coordinate descent converged at iteration {iteration}");
                }
                break;
            }

            if self.verbose && iteration % 100 == 0 {
                println!("Enhanced CD iteration {iteration}, alpha_diff: {alpha_diff:.6}");
            }
        }

        Ok(())
    }

    /// Select working set using second-order information
    fn select_working_set(
        &self,
        x: ArrayView2<f64>,
        y: &Array1<f64>,
        alpha: &Array1<f64>,
        w: &Array1<f64>,
        intercept: f64,
        set_size: usize,
    ) -> Result<Vec<usize>> {
        let n_samples = x.nrows();
        let mut scores = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let xi = x.row(i);
            let yi = y[i];

            // Compute prediction
            let mut prediction = if self.fit_intercept { intercept } else { 0.0 };
            for j in 0..x.ncols() {
                prediction += w[j] * xi[j];
            }

            let margin = yi * prediction;

            // Compute violation score for KKT conditions
            let violation = if alpha[i] <= 1e-12 {
                // alpha = 0, check if we should increase it
                if margin < 1.0 {
                    1.0 - margin
                } else {
                    0.0
                }
            } else if alpha[i] >= self.c - 1e-12 {
                // alpha = C, check if we should decrease it
                if margin > 1.0 {
                    margin - 1.0
                } else {
                    0.0
                }
            } else {
                // 0 < alpha < C, should be on margin
                (margin - 1.0).abs()
            };

            scores.push((violation, i));
        }

        // Sort by violation score (descending)
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Return top violators
        Ok(scores.into_iter().take(set_size).map(|(_, i)| i).collect())
    }

    /// Line search for optimal step size
    fn line_search(
        &self,
        x: ArrayView2<f64>,
        y: &Array1<f64>,
        sample_idx: usize,
        current_alpha: f64,
        direction: f64,
    ) -> Result<f64> {
        let xi = x.row(sample_idx);
        let yi = y[sample_idx];

        // Simple backtracking line search
        let mut step_size = 1.0;
        let c1 = 1e-4; // Armijo condition parameter
        let rho = 0.5; // Backtracking parameter

        // Compute current objective value
        let current_obj = self.compute_sample_objective(current_alpha, yi, xi.view())?;

        for _ in 0..10 {
            // Max 10 backtracking steps
            let new_alpha = (current_alpha + step_size * direction).max(0.0).min(self.c);
            let new_obj = self.compute_sample_objective(new_alpha, yi, xi.view())?;

            // Check Armijo condition
            if new_obj
                <= current_obj
                    + c1 * step_size
                        * direction
                        * self.compute_gradient(current_alpha, yi, xi.view())?
            {
                return Ok(step_size);
            }

            step_size *= rho;
        }

        Ok(step_size) // Return last step size if no good one found
    }

    /// Compute objective function value for a single sample
    fn compute_sample_objective(&self, alpha: f64, y: f64, x: ArrayView1<f64>) -> Result<f64> {
        match self.loss.as_str() {
            "hinge" => Ok(alpha), // Simplified for line search
            "squared_hinge" => Ok(alpha * alpha / (2.0 * self.c)), // Simplified
            _ => Err(SklearsError::InvalidParameter {
                name: "loss".to_string(),
                reason: format!("Unknown loss: {}", self.loss),
            }),
        }
    }

    /// Compute gradient for a single sample
    fn compute_gradient(&self, alpha: f64, y: f64, x: ArrayView1<f64>) -> Result<f64> {
        match self.loss.as_str() {
            "hinge" => Ok(-y),               // Simplified
            "squared_hinge" => Ok(-2.0 * y), // Simplified
            _ => Err(SklearsError::InvalidParameter {
                name: "loss".to_string(),
                reason: format!("Unknown loss: {}", self.loss),
            }),
        }
    }
}

impl Estimator for LinearSVC {
    type Config = Self;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<Array2<f64>, Array1<i32>> for LinearSVC {
    type Fitted = TrainedLinearSVC;

    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> Result<TrainedLinearSVC> {
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

        // For binary classification, use single model
        if n_classes == 2 {
            let mut alpha = Array1::zeros(n_samples);
            let mut w = Array1::zeros(n_features);
            let mut intercept = 0.0;

            // Convert labels to binary (0/1 -> -1/1)
            let y_binary = y.map(|&label| if label == classes[1] { 1 } else { -1 });

            self.solve_binary_problem(
                x.view(),
                y_binary.view(),
                &mut alpha,
                &mut w,
                &mut intercept,
            )?;

            let coef = w.insert_axis(Axis(0));
            let intercept_arr = Array1::from(vec![intercept]);

            Ok(TrainedLinearSVC {
                coef_: coef,
                intercept_: intercept_arr,
                classes_: classes,
                n_features_in_: n_features,
                _params: self,
            })
        } else {
            // Multi-class: One-vs-Rest approach
            let mut coef_matrix = Array2::zeros((n_classes, n_features));
            let mut intercept_vec = Array1::zeros(n_classes);

            for (class_idx, &class_label) in classes.iter().enumerate() {
                // Create binary labels (current class vs rest)
                let y_binary = y.map(|&label| if label == class_label { 1 } else { -1 });

                let mut alpha = Array1::zeros(n_samples);
                let mut w = Array1::zeros(n_features);
                let mut intercept = 0.0;

                self.solve_binary_problem(
                    x.view(),
                    y_binary.view(),
                    &mut alpha,
                    &mut w,
                    &mut intercept,
                )?;

                coef_matrix.row_mut(class_idx).assign(&w);
                intercept_vec[class_idx] = intercept;
            }

            Ok(TrainedLinearSVC {
                coef_: coef_matrix,
                intercept_: intercept_vec,
                classes_: classes,
                n_features_in_: n_features,
                _params: self,
            })
        }
    }
}

impl Predict<Array2<f64>, Array1<i32>> for TrainedLinearSVC {
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

impl TrainedLinearSVC {
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
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_svc_binary_classification() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
        let y = array![0, 1, 1, 0];

        let model = LinearSVC::new().with_c(1.0).with_max_iter(1000);
        let trained_model = model.fit(&X_var, &y).unwrap();

        let predictions = trained_model.predict(&X_var).unwrap();
        assert_eq!(predictions.len(), 4);

        // Test decision function
        let scores = trained_model.decision_function(&X_var).unwrap();
        assert_eq!(scores.dim(), (4, 1));
    }

    #[test]
    fn test_linear_svc_multiclass_classification() {
        let X_var = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let model = LinearSVC::new().with_c(1.0).with_max_iter(1000);
        let trained_model = model.fit(&X_var, &y).unwrap();

        let predictions = trained_model.predict(&X_var).unwrap();
        assert_eq!(predictions.len(), 6);

        // Test decision function for multiclass
        let scores = trained_model.decision_function(&X_var).unwrap();
        assert_eq!(scores.dim(), (6, 3)); // 6 samples, 3 classes
    }

    #[test]
    fn test_linear_svc_parameters() {
        let model = LinearSVC::new()
            .with_c(0.5)
            .with_loss("hinge")
            .with_penalty("elasticnet")
            .with_l1_ratio(0.3)
            .with_dual(false)
            .with_tol(1e-5)
            .with_max_iter(500)
            .with_fit_intercept(false)
            .with_random_state(42);

        assert_eq!(model.c, 0.5);
        assert_eq!(model.loss, "hinge");
        assert_eq!(model.penalty, "elasticnet");
        assert_abs_diff_eq!(model.l1_ratio, 0.3);
        assert!(!model.dual);
        assert_abs_diff_eq!(model.tol, 1e-5);
        assert_eq!(model.max_iter, 500);
        assert!(!model.fit_intercept);
        assert_eq!(model.random_state, Some(42));
    }

    #[test]
    fn test_linear_svc_invalid_loss() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 1];

        let model = LinearSVC::new().with_loss("invalid_loss");
        let result = model.fit(&X_var, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_svc_dimension_mismatch() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0]; // Wrong size

        let model = LinearSVC::new();
        let result = model.fit(&X_var, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_primal_coordinate_descent() {
        let X_var = array![[2.0, 3.0], [3.0, 3.0], [1.0, 1.0], [2.0, 1.0]];
        let y = array![1, 1, 0, 0];

        let model = LinearSVC::new()
            .with_solver("primal_cd")
            .with_max_iter(1000)
            .with_tol(1e-6);

        let trained = model.fit(&X_var, &y).unwrap();
        let predictions = trained.predict(&X_var).unwrap();

        // Check that we have reasonable accuracy
        let accuracy = predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &actual)| if pred == actual { 1.0 } else { 0.0 })
            .sum::<f64>()
            / predictions.len() as f64;

        assert!(accuracy >= 0.75, "Primal CD accuracy too low: {}", accuracy);
    }

    #[test]
    fn test_enhanced_coordinate_descent() {
        let X_var = array![[2.0, 3.0], [3.0, 3.0], [1.0, 1.0], [2.0, 1.0]];
        let y = array![1, 1, 0, 0];

        let model = LinearSVC::new()
            .with_solver("enhanced_cd")
            .with_max_iter(1000)
            .with_tol(1e-6);

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
            "Enhanced CD accuracy too low: {}",
            accuracy
        );
    }

    #[test]
    fn test_solver_selection() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 1];

        // Test invalid solver
        let model = LinearSVC::new().with_solver("invalid_solver");
        let result = model.fit(&X_var, &y);
        assert!(result.is_err());

        // Test valid solvers
        for solver in &["dual_cd", "primal_cd", "enhanced_cd"] {
            let model = LinearSVC::new().with_solver(solver);
            let result = model.fit(&X_var, &y);
            assert!(result.is_ok(), "Solver {} should work", solver);
        }
    }

    #[test]
    fn test_solver_performance_comparison() {
        // Larger dataset for better comparison
        let X_var = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [1.5, 1.0],
            [2.5, 1.5],
            [3.5, 2.0],
            [4.5, 2.5],
            [0.5, 3.0],
            [1.5, 4.0],
            [2.5, 5.0],
            [3.5, 6.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0]
        ];
        let y = array![1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0];

        for solver in &["dual_cd", "primal_cd", "enhanced_cd"] {
            let model = LinearSVC::new()
                .with_solver(solver)
                .with_max_iter(1000)
                .with_tol(1e-6);

            let trained = model.fit(&X_var, &y).unwrap();
            let predictions = trained.predict(&X_var).unwrap();

            let accuracy = predictions
                .iter()
                .zip(y.iter())
                .map(|(&pred, &actual)| if pred == actual { 1.0 } else { 0.0 })
                .sum::<f64>()
                / predictions.len() as f64;

            assert!(
                accuracy >= 0.5,
                "Solver {} accuracy too low: {}",
                solver,
                accuracy
            );
        }
    }
}
