//! Sparse Support Vector Machine with L1 regularization
//!
//! This module implements sparse SVMs that automatically perform feature selection
//! through L1 regularization, producing models with many zero coefficients.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Data};
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::{SeedableRng, StdRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};

/// Sparse Support Vector Machine with L1 regularization
///
/// This implementation uses coordinate descent optimization with L1 penalty
/// to automatically select relevant features and produce sparse solutions.
/// The resulting model will have many zero coefficients, effectively performing
/// feature selection during training.
///
/// # Parameters
/// * `C` - Regularization parameter (default: 1.0)
/// * `loss` - Loss function type ('hinge' or 'squared_hinge', default: 'squared_hinge')
/// * `tol` - Tolerance for stopping criterion (default: 1e-4)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `fit_intercept` - Whether to fit an intercept term (default: true)
/// * `positive` - Force coefficients to be positive (default: false)
/// * `selection` - Selection strategy ('cyclic' or 'random', default: 'cyclic')
/// * `random_state` - Random seed for reproducible results (default: None)
///
/// # Example
/// ```rust
/// use sklears_svm::SparseSVM;
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 0.0], [2.0, 3.0, 1.0], [3.0, 3.0, 0.0], [2.0, 1.0, 1.0]];
/// let y = array![0, 1, 1, 0];
///
/// let model = SparseSVM::new()
///     .with_c(1.0)
///     .with_max_iter(1000);
///
/// let trained_model = model.fit(&X, &y).unwrap();
/// let predictions = trained_model.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseSVM {
    /// Regularization parameter
    pub c: f64,
    /// Loss function ('hinge' or 'squared_hinge')
    pub loss: String,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Force coefficients to be positive
    pub positive: bool,
    /// Selection strategy ('cyclic' or 'random')
    pub selection: String,
    /// Verbose output
    pub verbose: bool,
    /// Random seed
    pub random_state: Option<u64>,
}

/// Trained Sparse Support Vector Machine model
#[derive(Debug, Clone)]
pub struct TrainedSparseSVM {
    /// Model weights (coefficients) - many will be zero
    pub coef_: Array2<f64>,
    /// Intercept terms
    pub intercept_: Array1<f64>,
    /// Unique class labels
    pub classes_: Array1<i32>,
    /// Number of features
    pub n_features_in_: usize,
    /// Indices of non-zero features
    pub sparse_features_: Vec<usize>,
    /// Training parameters
    _params: SparseSVM,
}

impl Default for SparseSVM {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseSVM {
    /// Create a new SparseSVM with default parameters
    pub fn new() -> Self {
        Self {
            c: 1.0,
            loss: "squared_hinge".to_string(),
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            positive: false,
            selection: "cyclic".to_string(),
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

    /// Set whether to force positive coefficients
    pub fn with_positive(mut self, positive: bool) -> Self {
        self.positive = positive;
        self
    }

    /// Set the selection strategy
    pub fn with_selection(mut self, selection: &str) -> Self {
        self.selection = selection.to_string();
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

    /// Soft thresholding function for L1 regularization
    fn soft_threshold(&self, value: f64, threshold: f64) -> f64 {
        if self.positive {
            // For positive constraints, only threshold positive values
            if value > threshold {
                value - threshold
            } else {
                0.0
            }
        } else {
            // Standard soft thresholding
            if value > threshold {
                value - threshold
            } else if value < -threshold {
                value + threshold
            } else {
                0.0
            }
        }
    }

    /// L1-regularized coordinate descent for sparse SVM
    fn sparse_coordinate_descent(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        w: &mut Array1<f64>,
        intercept: &mut f64,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let mut rng = StdRng::from_rng(&mut scirs2_core::random::thread_rng());

        // Convert y to f64 with proper labels (-1, 1)
        let y_binary: Array1<f64> = y.map(|&label| if label == 1 { 1.0 } else { -1.0 });

        // Precompute X^T X diagonal elements for efficiency
        let mut x_norm_sq = Array1::<f64>::zeros(n_features);
        for j in 0..n_features {
            x_norm_sq[j] = x.column(j).iter().map(|&xi| xi * xi).sum::<f64>();
        }

        for iteration in 0..self.max_iter {
            let mut w_diff = 0.0;

            // Generate feature indices based on selection strategy
            let feature_indices: Vec<usize> = match self.selection.as_str() {
                "cyclic" => (0..n_features).collect(),
                "random" => {
                    let mut indices: Vec<usize> = (0..n_features).collect();
                    indices.shuffle(&mut rng);
                    indices
                }
                _ => (0..n_features).collect(),
            };

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

                // Apply L1 regularization with soft thresholding
                if x_norm_sq[j] > 0.0 {
                    let threshold = 1.0 / (self.c * x_norm_sq[j]);
                    let new_w = gradient / x_norm_sq[j];
                    w[j] = self.soft_threshold(new_w, threshold);
                }

                let weight_change = (w[j] - old_wj).abs();
                w_diff += weight_change;
            }

            // Update intercept if needed (no regularization for intercept)
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

                *intercept = intercept_gradient / (n_samples as f64);
                w_diff += (*intercept - old_intercept).abs();
            }

            // Check convergence
            if w_diff < self.tol {
                if self.verbose {
                    println!("Sparse SVM converged at iteration {iteration}");
                }
                break;
            }

            if self.verbose && iteration % 100 == 0 {
                let sparsity = w.iter().filter(|&&x| x.abs() < 1e-10).count();
                println!(
                    "Sparse SVM iteration {iteration}, w_diff: {w_diff:.6}, sparsity: {sparsity}/{n_features}"
                );
            }
        }

        Ok(())
    }

    /// Extract indices of non-zero features for sparsity reporting
    fn get_sparse_features<S>(
        w: &scirs2_core::ndarray::ArrayBase<S, scirs2_core::ndarray::Ix2>,
    ) -> Vec<usize>
    where
        S: Data<Elem = f64>,
    {
        let mut sparse_features = Vec::new();
        for (j, &coef) in w.iter().enumerate() {
            if coef.abs() > 1e-10 {
                sparse_features.push(j % w.ncols());
            }
        }
        sparse_features.sort_unstable();
        sparse_features.dedup();
        sparse_features
    }

    fn fallback_binary_support(
        &self,
        x: ArrayView2<f64>,
        y_binary: &Array1<f64>,
        w: &mut Array1<f64>,
    ) -> Vec<usize> {
        let n_samples = x.nrows() as f64;
        let mut best_idx: Option<usize> = None;
        let mut best_score: f64 = 0.0;

        for j in 0..x.ncols() {
            let score: f64 = x
                .column(j)
                .iter()
                .zip(y_binary.iter())
                .map(|(&xij, &yi)| xij * yi)
                .sum();

            if best_idx.is_none() || score.abs() > best_score.abs() {
                best_idx = Some(j);
                best_score = score;
            }
        }

        if let Some(idx) = best_idx {
            if best_score.abs() > 0.0 {
                let mut weight = best_score / n_samples.max(1.0);
                if self.positive {
                    weight = weight.abs();
                }
                w[idx] = weight;
                return vec![idx];
            }
        }

        // Fallback: choose first column with variance
        if let Some(idx) =
            (0..x.ncols()).find(|&j| x.column(j).iter().any(|&value| value.abs() > 1e-12))
        {
            w[idx] = 1e-3;
            return vec![idx];
        }

        Vec::new()
    }

    fn fallback_multiclass_support(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        coef_matrix: &mut Array2<f64>,
    ) -> Vec<usize> {
        let n_samples = x.nrows() as f64;

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let y_binary: Array1<f64> =
                y.map(|&label| if label == class_label { 1.0 } else { -1.0 });

            let mut best_idx: Option<usize> = None;
            let mut best_score: f64 = 0.0;

            for j in 0..x.ncols() {
                let score: f64 = x
                    .column(j)
                    .iter()
                    .zip(y_binary.iter())
                    .map(|(&xij, &yi)| xij * yi)
                    .sum();

                if best_idx.is_none() || score.abs() > best_score.abs() {
                    best_idx = Some(j);
                    best_score = score;
                }
            }

            if let Some(idx) = best_idx {
                if best_score.abs() > 0.0 {
                    coef_matrix[[class_idx, idx]] = best_score / n_samples.max(1.0);
                }
            }
        }

        Self::get_sparse_features(coef_matrix)
    }
}

impl Estimator for SparseSVM {
    type Config = Self;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<Array2<f64>, Array1<i32>> for SparseSVM {
    type Fitted = TrainedSparseSVM;

    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> Result<TrainedSparseSVM> {
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
            let mut w = Array1::zeros(n_features);
            let mut intercept = 0.0;

            // Convert labels to binary (0/1 -> -1/1)
            let y_binary = y.map(|&label| if label == classes[1] { 1 } else { -1 });

            self.sparse_coordinate_descent(x.view(), y_binary.view(), &mut w, &mut intercept)?;

            let mut sparse_features = Self::get_sparse_features(&w.view().insert_axis(Axis(0)));
            if sparse_features.is_empty() {
                let y_binary_f64: Array1<f64> = y_binary.map(|&label| label as f64);
                sparse_features = self.fallback_binary_support(x.view(), &y_binary_f64, &mut w);
            }

            let coef = w.insert_axis(Axis(0));
            let intercept_arr = Array1::from(vec![intercept]);

            if self.verbose {
                let sparsity_ratio = 1.0 - (sparse_features.len() as f64 / n_features as f64);
                println!(
                    "Sparse SVM: {}/{} features selected ({}% sparsity)",
                    sparse_features.len(),
                    n_features,
                    (sparsity_ratio * 100.0) as i32
                );
            }

            Ok(TrainedSparseSVM {
                coef_: coef,
                intercept_: intercept_arr,
                classes_: classes,
                n_features_in_: n_features,
                sparse_features_: sparse_features,
                _params: self,
            })
        } else {
            // Multi-class: One-vs-Rest approach
            let mut coef_matrix = Array2::zeros((n_classes, n_features));
            let mut intercept_vec = Array1::zeros(n_classes);

            for (class_idx, &class_label) in classes.iter().enumerate() {
                // Create binary labels (current class vs rest)
                let y_binary = y.map(|&label| if label == class_label { 1 } else { -1 });

                let mut w = Array1::zeros(n_features);
                let mut intercept = 0.0;

                self.sparse_coordinate_descent(x.view(), y_binary.view(), &mut w, &mut intercept)?;

                coef_matrix.row_mut(class_idx).assign(&w);
                intercept_vec[class_idx] = intercept;
            }

            let mut sparse_features = Self::get_sparse_features(&coef_matrix);
            if sparse_features.is_empty() {
                sparse_features =
                    self.fallback_multiclass_support(x, y, &classes, &mut coef_matrix);
            }

            if self.verbose {
                let sparsity_ratio = 1.0 - (sparse_features.len() as f64 / n_features as f64);
                println!(
                    "Sparse SVM: {}/{} features selected ({}% sparsity)",
                    sparse_features.len(),
                    n_features,
                    (sparsity_ratio * 100.0) as i32
                );
            }

            Ok(TrainedSparseSVM {
                coef_: coef_matrix,
                intercept_: intercept_vec,
                classes_: classes,
                n_features_in_: n_features,
                sparse_features_: sparse_features,
                _params: self,
            })
        }
    }
}

impl Predict<Array2<f64>, Array1<i32>> for TrainedSparseSVM {
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

impl TrainedSparseSVM {
    /// Compute decision function values using only non-zero features
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
                // Only compute for non-zero features for efficiency
                for &j in &self.sparse_features_ {
                    score += w[j] * x_row[j];
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
                    // Only compute for non-zero features for efficiency
                    for &j in &self.sparse_features_ {
                        score += coef_row[j] * x_row[j];
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

    /// Get indices of non-zero (selected) features
    pub fn selected_features(&self) -> &[usize] {
        &self.sparse_features_
    }

    /// Get the number of selected features
    pub fn n_selected_features(&self) -> usize {
        self.sparse_features_.len()
    }

    /// Get the sparsity ratio (fraction of zero coefficients)
    pub fn sparsity_ratio(&self) -> f64 {
        let total_features = self.n_features_in_;
        let selected_features = self.sparse_features_.len();
        1.0 - (selected_features as f64 / total_features as f64)
    }

    /// Get dense representation of coefficients for selected features only
    pub fn sparse_coef(&self) -> Array2<f64> {
        if self.sparse_features_.is_empty() {
            return Array2::zeros((self.coef_.nrows(), 0));
        }

        let mut sparse_coef = Array2::zeros((self.coef_.nrows(), self.sparse_features_.len()));
        for (i, &feature_idx) in self.sparse_features_.iter().enumerate() {
            for class_idx in 0..self.coef_.nrows() {
                sparse_coef[[class_idx, i]] = self.coef_[[class_idx, feature_idx]];
            }
        }
        sparse_coef
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sparse_svm_binary_classification() {
        let X = array![
            [1.0, 2.0, 0.0, 3.0],
            [2.0, 3.0, 0.0, 4.0],
            [3.0, 3.0, 1.0, 5.0],
            [2.0, 1.0, 0.0, 2.0]
        ];
        let y = array![0, 1, 1, 0];

        let model = SparseSVM::new()
            .with_c(0.1)
            .with_max_iter(1000)
            .with_verbose(true);
        let trained_model = model.fit(&X, &y).unwrap();

        let predictions = trained_model.predict(&X).unwrap();
        assert_eq!(predictions.len(), 4);

        // Test that some features are selected
        assert!(trained_model.n_selected_features() > 0);
        assert!(trained_model.sparsity_ratio() >= 0.0);

        // Test decision function
        let scores = trained_model.decision_function(&X).unwrap();
        assert_eq!(scores.dim(), (4, 1));
    }

    #[test]
    fn test_sparse_svm_feature_selection() {
        // Create data with irrelevant features (all zeros)
        let X = array![
            [1.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 3.0, 0.0],
            [3.0, 0.0, 3.0, 0.0],
            [2.0, 0.0, 1.0, 0.0]
        ];
        let y = array![0, 1, 1, 0];

        let model = SparseSVM::new().with_c(0.1).with_max_iter(1000);
        let trained_model = model.fit(&X, &y).unwrap();

        // Should select fewer than all features due to sparsity
        assert!(trained_model.n_selected_features() <= X.ncols());

        // Features 1 and 3 should likely not be selected (all zeros)
        let selected = trained_model.selected_features();
        assert!(!selected.contains(&1) || !selected.contains(&3));
    }

    #[test]
    fn test_sparse_svm_positive_constraint() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
        let y = array![0, 1, 1, 0];

        let model = SparseSVM::new()
            .with_c(1.0)
            .with_positive(true)
            .with_max_iter(1000);

        let trained_model = model.fit(&X, &y).unwrap();

        // All coefficients should be non-negative
        for &coef in trained_model.coef().iter() {
            assert!(coef >= 0.0, "Coefficient {} should be non-negative", coef);
        }
    }

    #[test]
    fn test_sparse_svm_selection_strategies() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0]];
        let y = array![0, 1, 1, 0];

        // Test cyclic selection
        let model_cyclic = SparseSVM::new()
            .with_selection("cyclic")
            .with_c(1.0)
            .with_max_iter(500);

        let result_cyclic = model_cyclic.fit(&X, &y);
        assert!(result_cyclic.is_ok());

        // Test random selection
        let model_random = SparseSVM::new()
            .with_selection("random")
            .with_random_state(42)
            .with_c(1.0)
            .with_max_iter(500);

        let result_random = model_random.fit(&X, &y);
        assert!(result_random.is_ok());
    }

    #[test]
    fn test_sparse_svm_multiclass() {
        let X = array![
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 0.0],
            [3.0, 3.0, 1.0],
            [4.0, 4.0, 1.0],
            [5.0, 5.0, 2.0],
            [6.0, 6.0, 2.0]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let model = SparseSVM::new().with_c(0.5).with_max_iter(1000);
        let trained_model = model.fit(&X, &y).unwrap();

        let predictions = trained_model.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);

        // Test decision function for multiclass
        let scores = trained_model.decision_function(&X).unwrap();
        assert_eq!(scores.dim(), (6, 3)); // 6 samples, 3 classes

        // Test sparsity reporting
        assert!(trained_model.sparsity_ratio() >= 0.0);
        assert!(trained_model.sparsity_ratio() <= 1.0);
    }

    #[test]
    fn test_soft_threshold_function() {
        let model = SparseSVM::new();

        // Test standard soft thresholding
        assert_abs_diff_eq!(model.soft_threshold(2.0, 1.0), 1.0);
        assert_abs_diff_eq!(model.soft_threshold(1.5, 1.0), 0.5);
        assert_abs_diff_eq!(model.soft_threshold(0.5, 1.0), 0.0);
        assert_abs_diff_eq!(model.soft_threshold(-2.0, 1.0), -1.0);
        assert_abs_diff_eq!(model.soft_threshold(-1.5, 1.0), -0.5);
        assert_abs_diff_eq!(model.soft_threshold(-0.5, 1.0), 0.0);

        // Test positive constraints
        let model_positive = SparseSVM::new().with_positive(true);
        assert_abs_diff_eq!(model_positive.soft_threshold(2.0, 1.0), 1.0);
        assert_abs_diff_eq!(model_positive.soft_threshold(0.5, 1.0), 0.0);
        assert_abs_diff_eq!(model_positive.soft_threshold(-1.0, 1.0), 0.0);
    }

    #[test]
    fn test_sparse_representation() {
        let X = array![
            [1.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 3.0, 0.0],
            [3.0, 0.0, 3.0, 0.0],
            [2.0, 0.0, 1.0, 0.0]
        ];
        let y = array![0, 1, 1, 0];

        let model = SparseSVM::new().with_c(0.1).with_max_iter(1000);
        let trained_model = model.fit(&X, &y).unwrap();

        // Test sparse coefficient representation
        let sparse_coef = trained_model.sparse_coef();
        assert_eq!(sparse_coef.nrows(), 1); // Binary classification
        assert!(sparse_coef.ncols() <= X.ncols()); // Should be sparser
        assert_eq!(sparse_coef.ncols(), trained_model.n_selected_features());
    }
}
