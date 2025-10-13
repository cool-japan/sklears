//! Utility functions and shared types for multi-output learning
//!
//! This module contains common functionality used across multiple algorithms in the
//! multi-output learning suite, including mathematical operations, binary classifier
//! training, feature processing, and label reconstruction methods.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

/// Methods for label reconstruction in compressed sensing approaches
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconstructionMethod {
    /// Linear reconstruction using least squares
    Linear,
    /// Iterative soft thresholding
    IterativeThresholding,
    /// Orthogonal matching pursuit
    OrthogonalMatchingPursuit,
}

/// Strategies for pruning label combinations in label powerset methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningStrategy {
    /// Remove rare combinations and map to default (e.g., all zeros)
    Default,
    /// Map rare combinations to most similar frequent combination
    Similarity,
}

/// Classification criteria for decision trees
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClassificationCriterion {
    /// Gini impurity
    Gini,
    /// Information gain (entropy)
    Entropy,
}

/// Threshold strategies for binary relevance methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdStrategy {
    /// Fixed threshold for all labels
    Fixed,
    /// Per-label optimal thresholds
    PerLabel,
    /// Optimal thresholds based on validation data
    Optimal,
    /// F-score based thresholds
    FScore,
}

/// Calibration methods for probability calibration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationMethod {
    /// Sigmoid calibration (Platt scaling)
    Sigmoid,
    /// Isotonic regression calibration
    Isotonic,
}

/// Simple linear classifier for binary classification
#[derive(Debug, Clone)]
pub struct SimpleLinearClassifier {
    /// Weight vector
    pub weights: Array1<Float>,
    /// Bias term
    pub bias: Float,
}

/// Simple binary classification model
#[derive(Debug, Clone)]
pub struct SimpleBinaryModel {
    /// Feature weights
    pub weights: Array1<Float>,
    /// Bias term
    pub bias: Float,
    /// Training accuracy
    pub accuracy: Float,
}

/// Bayesian binary classification model
#[derive(Debug, Clone)]
pub struct BayesianBinaryModel {
    /// Posterior mean of weights
    pub weight_mean: Array1<Float>,
    /// Posterior covariance of weights
    pub weight_cov: Array2<Float>,
    /// Bias parameter
    pub bias_mean: Float,
    /// Bias variance
    pub bias_var: Float,
    /// Noise precision
    pub noise_precision: Float,
}

/// Cost matrix for cost-sensitive learning
#[derive(Debug, Clone)]
pub struct CostMatrix {
    /// False positive costs for each label
    pub fp_costs: Vec<Float>,
    /// False negative costs for each label
    pub fn_costs: Vec<Float>,
}

impl CostMatrix {
    /// Create cost matrix from false positive and false negative costs
    pub fn from_fp_fn_costs(fp_costs: Vec<Float>, fn_costs: Vec<Float>) -> SklResult<Self> {
        if fp_costs.len() != fn_costs.len() {
            return Err(SklearsError::InvalidInput(
                "False positive and false negative cost vectors must have the same length"
                    .to_string(),
            ));
        }

        if fp_costs.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cost vectors cannot be empty".to_string(),
            ));
        }

        // Check for non-negative costs
        for &cost in fp_costs.iter().chain(fn_costs.iter()) {
            if cost < 0.0 {
                return Err(SklearsError::InvalidInput(
                    "All costs must be non-negative".to_string(),
                ));
            }
        }

        Ok(Self { fp_costs, fn_costs })
    }

    /// Create balanced cost matrix (all costs = 1.0)
    pub fn balanced(n_labels: usize) -> Self {
        Self {
            fp_costs: vec![1.0; n_labels],
            fn_costs: vec![1.0; n_labels],
        }
    }

    /// Get optimal threshold for a given label based on cost ratio
    pub fn get_threshold(&self, label_idx: usize) -> Float {
        if label_idx >= self.fp_costs.len() {
            return 0.5; // Default threshold
        }

        let fp_cost = self.fp_costs[label_idx];
        let fn_cost = self.fn_costs[label_idx];

        // Optimal threshold = FP_cost / (FP_cost + FN_cost)
        if fp_cost + fn_cost > 0.0 {
            fp_cost / (fp_cost + fn_cost)
        } else {
            0.5
        }
    }
}

/// Calculate Euclidean distance between two points
pub fn euclidean_distance(x1: &ArrayView1<Float>, x2: &ArrayView1<Float>) -> Float {
    x1.iter()
        .zip(x2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<Float>()
        .sqrt()
}

/// Standardize features using provided means and standard deviations
pub fn standardize_features_simple(
    X: &ArrayView2<Float>,
    means: &Array1<Float>,
    stds: &Array1<Float>,
) -> Array2<Float> {
    let mut X_standardized = X.to_owned();

    for (mut col, (&mean, &std)) in X_standardized
        .axis_iter_mut(Axis(1))
        .zip(means.iter().zip(stds.iter()))
    {
        col.mapv_inplace(|x| (x - mean) / std);
    }

    X_standardized
}

/// Train a simple binary classifier using correlation-based approach
pub fn train_binary_classifier(
    X: &ArrayView2<Float>,
    y: &Array1<i32>,
) -> SklResult<SimpleBinaryModel> {
    let (n_samples, n_features) = X.dim();

    if n_samples != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same number of samples".to_string(),
        ));
    }

    // Convert labels to Float for computation
    let y_float: Array1<Float> = y.mapv(|x| x as Float);

    // Calculate means
    let x_means = X.mean_axis(Axis(0)).unwrap();
    let y_mean = y_float.mean().unwrap();

    // Calculate weights using correlation
    let mut weights = Array1::<Float>::zeros(n_features);

    for (i, weight) in weights.iter_mut().enumerate() {
        let x_col = X.column(i);
        let x_mean = x_means[i];

        let numerator: Float = x_col
            .iter()
            .zip(y_float.iter())
            .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
            .sum();

        let denominator: Float = x_col
            .iter()
            .map(|&x| (x - x_mean).powi(2))
            .sum::<Float>()
            .sqrt()
            * y_float
                .iter()
                .map(|&y| (y - y_mean).powi(2))
                .sum::<Float>()
                .sqrt();

        *weight = if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };
    }

    // Calculate bias
    let bias = y_mean - weights.dot(&x_means);

    // Calculate training accuracy
    let mut correct = 0;
    for i in 0..n_samples {
        let prediction = if weights.dot(&X.row(i)) + bias > 0.0 {
            1
        } else {
            0
        };
        if prediction == y[i] {
            correct += 1;
        }
    }
    let accuracy = correct as Float / n_samples as Float;

    Ok(SimpleBinaryModel {
        weights,
        bias,
        accuracy,
    })
}

/// Train a simple linear classifier using least squares
pub fn train_simple_linear_classifier(
    X: &ArrayView2<Float>,
    y: &Array1<Float>,
) -> SklResult<SimpleLinearClassifier> {
    let (n_samples, n_features) = X.dim();

    if n_samples != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same number of samples".to_string(),
        ));
    }

    // Add bias column to X
    let mut X_with_bias = Array2::ones((n_samples, n_features + 1));
    X_with_bias.slice_mut(s![.., ..n_features]).assign(X);

    // Solve normal equations: (X^T X) w = X^T y
    let xtx = X_with_bias.t().dot(&X_with_bias);
    let xty = X_with_bias.t().dot(y);

    let weights_with_bias = solve_linear_system(&xtx, &xty)?;

    let weights = weights_with_bias.slice(s![..n_features]).to_owned();
    let bias = weights_with_bias[n_features];

    Ok(SimpleLinearClassifier { weights, bias })
}

/// Predict using a simple linear classifier
pub fn predict_simple_linear(
    X: &ArrayView2<Float>,
    classifier: &SimpleLinearClassifier,
) -> Array1<Float> {
    X.dot(&classifier.weights) + classifier.bias
}

/// Solve linear system Ax = b using Gaussian elimination
pub fn solve_linear_system(A: &Array2<Float>, b: &Array1<Float>) -> SklResult<Array1<Float>> {
    let n = A.nrows();
    if A.ncols() != n || b.len() != n {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square and vector must match matrix size".to_string(),
        ));
    }

    let mut aug = Array2::<Float>::zeros((n, n + 1));
    aug.slice_mut(s![.., ..n]).assign(A);
    aug.slice_mut(s![.., n]).assign(b);

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singular matrix
        if aug[[i, i]].abs() < 1e-10 {
            // Add small regularization
            aug[[i, i]] += 1e-8;
        }

        // Eliminate
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..=n {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<Float>::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
    }

    Ok(x)
}

/// Generate random projection matrix for compressed sensing
pub fn generate_random_projection_matrix(
    n_compressed: usize,
    n_labels: usize,
    random_state: Option<u64>,
) -> Array2<Float> {
    // Use deterministic random generation based on seed
    let mut rng_state = random_state.unwrap_or(42);

    let mut matrix = Array2::<Float>::zeros((n_compressed, n_labels));

    for i in 0..n_compressed {
        for j in 0..n_labels {
            // Simple LCG for reproducible random numbers
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let random_val = (rng_state as Float) / (u64::MAX as Float);
            matrix[[i, j]] = (random_val - 0.5) * 2.0; // Range [-1, 1]
        }
    }

    // Normalize rows
    for mut row in matrix.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<Float>().sqrt();
        if norm > 1e-10 {
            row /= norm;
        }
    }

    matrix
}

/// Reconstruct labels using specified method
pub fn reconstruct_labels(
    compressed_labels: &Array1<Float>,
    projection_matrix: &Array2<Float>,
    method: ReconstructionMethod,
) -> SklResult<Array1<Float>> {
    match method {
        ReconstructionMethod::Linear => {
            // Pseudo-inverse reconstruction
            let pinv = compute_pseudoinverse(projection_matrix)?;
            Ok(pinv.dot(compressed_labels))
        }
        ReconstructionMethod::IterativeThresholding => {
            iterative_thresholding_reconstruction(compressed_labels, projection_matrix)
        }
        ReconstructionMethod::OrthogonalMatchingPursuit => {
            omp_reconstruction(compressed_labels, projection_matrix)
        }
    }
}

/// Compute pseudo-inverse of a matrix
fn compute_pseudoinverse(matrix: &Array2<Float>) -> SklResult<Array2<Float>> {
    let (m, n) = matrix.dim();

    if m >= n {
        // More rows than columns: (A^T A)^-1 A^T
        let ata = matrix.t().dot(matrix);
        let ata_inv = matrix_inverse(&ata)?;
        Ok(ata_inv.dot(&matrix.t()))
    } else {
        // More columns than rows: A^T (A A^T)^-1
        let aat = matrix.dot(&matrix.t());
        let aat_inv = matrix_inverse(&aat)?;
        Ok(matrix.t().dot(&aat_inv))
    }
}

/// Compute matrix inverse using Gaussian elimination
fn matrix_inverse(matrix: &Array2<Float>) -> SklResult<Array2<Float>> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    let mut aug = Array2::<Float>::zeros((n, 2 * n));
    aug.slice_mut(s![.., ..n]).assign(matrix);

    // Set up identity on the right side
    for i in 0..n {
        aug[[i, n + i]] = 1.0;
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singular matrix
        if aug[[i, i]].abs() < 1e-10 {
            return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
        }

        // Scale pivot row
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    Ok(aug.slice(s![.., n..]).to_owned())
}

/// Iterative thresholding reconstruction
pub fn iterative_thresholding_reconstruction(
    compressed_labels: &Array1<Float>,
    projection_matrix: &Array2<Float>,
) -> SklResult<Array1<Float>> {
    let n_labels = projection_matrix.ncols();
    let mut x = Array1::<Float>::zeros(n_labels);
    let step_size = 0.1;
    let threshold = 0.1;
    let max_iterations = 100;

    for _ in 0..max_iterations {
        // Gradient step
        let residual = projection_matrix.dot(&x) - compressed_labels;
        let gradient = projection_matrix.t().dot(&residual);
        x = &x - step_size * &gradient;

        // Soft thresholding
        x.mapv_inplace(|xi| {
            if xi > threshold {
                xi - threshold
            } else if xi < -threshold {
                xi + threshold
            } else {
                0.0
            }
        });
    }

    Ok(x)
}

/// Orthogonal Matching Pursuit reconstruction
pub fn omp_reconstruction(
    compressed_labels: &Array1<Float>,
    projection_matrix: &Array2<Float>,
) -> SklResult<Array1<Float>> {
    let n_labels = projection_matrix.ncols();
    let mut selected_indices = Vec::new();
    let mut residual = compressed_labels.clone();
    let max_iterations = std::cmp::min(10, n_labels); // Sparsity constraint

    for _ in 0..max_iterations {
        // Find column with maximum correlation to residual
        let mut max_corr = 0.0;
        let mut best_idx = 0;

        for j in 0..n_labels {
            if !selected_indices.contains(&j) {
                let column = projection_matrix.column(j);
                let corr = column.dot(&residual).abs();
                if corr > max_corr {
                    max_corr = corr;
                    best_idx = j;
                }
            }
        }

        if max_corr < 1e-6 {
            break;
        }

        selected_indices.push(best_idx);

        // Solve least squares on selected columns
        if let Ok(coeffs) =
            solve_least_squares_subset(compressed_labels, projection_matrix, &selected_indices)
        {
            // Update residual
            let mut reconstruction = Array1::<Float>::zeros(projection_matrix.nrows());
            for (i, &idx) in selected_indices.iter().enumerate() {
                let column = projection_matrix.column(idx);
                reconstruction = reconstruction + coeffs[i] * &column;
            }
            residual = compressed_labels - &reconstruction;
        }
    }

    // Construct final solution
    let mut x = Array1::<Float>::zeros(n_labels);
    if let Ok(coeffs) =
        solve_least_squares_subset(compressed_labels, projection_matrix, &selected_indices)
    {
        for (i, &idx) in selected_indices.iter().enumerate() {
            x[idx] = coeffs[i];
        }
    }

    Ok(x)
}

/// Solve least squares problem for a subset of columns
pub fn solve_least_squares_subset(
    y: &Array1<Float>,
    A: &Array2<Float>,
    indices: &[usize],
) -> SklResult<Array1<Float>> {
    if indices.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No indices provided".to_string(),
        ));
    }

    let n_rows = A.nrows();
    let n_selected = indices.len();
    let mut A_subset = Array2::<Float>::zeros((n_rows, n_selected));

    for (j, &idx) in indices.iter().enumerate() {
        A_subset.column_mut(j).assign(&A.column(idx));
    }

    // Solve normal equations
    let ata = A_subset.t().dot(&A_subset);
    let aty = A_subset.t().dot(y);

    solve_linear_system(&ata, &aty)
}

/// Train a weighted binary classifier
pub fn train_weighted_binary_classifier_simple(
    X: &ArrayView2<Float>,
    y: &Array1<i32>,
    sample_weights: &Array1<Float>,
) -> SklResult<SimpleBinaryModel> {
    let (n_samples, n_features) = X.dim();

    if n_samples != y.len() || n_samples != sample_weights.len() {
        return Err(SklearsError::InvalidInput(
            "X, y, and sample_weights must have the same number of samples".to_string(),
        ));
    }

    // Weighted means
    let total_weight = sample_weights.sum();
    let mut x_means = Array1::<Float>::zeros(n_features);
    let mut y_mean = 0.0;

    for i in 0..n_samples {
        let weight = sample_weights[i];
        y_mean += weight * y[i] as Float;
        for j in 0..n_features {
            x_means[j] += weight * X[[i, j]];
        }
    }

    x_means /= total_weight;
    y_mean /= total_weight;

    // Weighted covariance computation
    let mut weights = Array1::<Float>::zeros(n_features);

    for j in 0..n_features {
        let mut numerator = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for i in 0..n_samples {
            let weight = sample_weights[i];
            let x_diff = X[[i, j]] - x_means[j];
            let y_diff = y[i] as Float - y_mean;

            numerator += weight * x_diff * y_diff;
            x_var += weight * x_diff * x_diff;
            y_var += weight * y_diff * y_diff;
        }

        let denominator = (x_var * y_var).sqrt();
        weights[j] = if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };
    }

    let bias = y_mean - weights.dot(&x_means);

    // Calculate weighted accuracy
    let mut correct_weight = 0.0;
    for i in 0..n_samples {
        let prediction = if weights.dot(&X.row(i)) + bias > 0.0 {
            1
        } else {
            0
        };
        if prediction == y[i] {
            correct_weight += sample_weights[i];
        }
    }
    let accuracy = correct_weight / total_weight;

    Ok(SimpleBinaryModel {
        weights,
        bias,
        accuracy,
    })
}

/// Predict binary probabilities using sigmoid function
pub fn predict_binary_probabilities(
    X: &ArrayView2<Float>,
    model: &SimpleBinaryModel,
) -> Array1<Float> {
    let raw_scores = X.dot(&model.weights) + model.bias;
    raw_scores.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Compute cost-sensitive sample weights
pub fn compute_cost_sensitive_weights(y: &Array2<i32>, cost_matrix: &CostMatrix) -> Array1<Float> {
    let n_samples = y.nrows();
    let mut weights = Array1::ones(n_samples);

    for i in 0..n_samples {
        let mut sample_weight = 1.0;

        for (j, &label) in y.row(i).iter().enumerate() {
            if j < cost_matrix.fp_costs.len() {
                // Weight based on label imbalance and costs
                if label == 1 {
                    sample_weight *= cost_matrix.fn_costs[j];
                } else {
                    sample_weight *= cost_matrix.fp_costs[j];
                }
            }
        }

        weights[i] = sample_weight;
    }

    // Normalize weights to sum to n_samples
    let total_weight = weights.sum();
    if total_weight > 0.0 {
        weights *= n_samples as Float / total_weight;
    }

    weights
}

/// Generate random normal distribution using Box-Muller transform
pub fn random_normal() -> Float {
    use std::cell::RefCell;

    thread_local! {
        static SAVED: RefCell<Option<Float>> = const { RefCell::new(None) };
    }

    SAVED.with(|saved| {
        if let Some(value) = saved.borrow_mut().take() {
            return value;
        }

        // Generate two uniform random numbers
        let u1 = (rand_u32() as Float) / (u32::MAX as Float);
        let u2 = (rand_u32() as Float) / (u32::MAX as Float);

        // Box-Muller transform
        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2 as f64).cos() as Float;
        let z1 = mag * (2.0 * std::f64::consts::PI * u2 as f64).sin() as Float;

        *saved.borrow_mut() = Some(z1);
        z0
    })
}

/// Simple random number generator for reproducible results
fn rand_u32() -> u32 {
    use std::cell::RefCell;

    thread_local! {
        static STATE: RefCell<u32> = const { RefCell::new(42) };
    }

    STATE.with(|state| {
        let mut s = state.borrow_mut();
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *s
    })
}

/// Train a Bayesian binary classifier
pub fn train_bayesian_binary_classifier(
    X: &Array2<Float>,
    y: &Array1<i32>,
    alpha: Float,
) -> SklResult<BayesianBinaryModel> {
    let (n_samples, n_features) = X.dim();

    if n_samples != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same number of samples".to_string(),
        ));
    }

    // Convert labels to Float and to {-1, 1}
    let y_float: Array1<Float> = y.mapv(|x| if x == 1 { 1.0 } else { -1.0 });

    // Prior precision matrix (alpha * I)
    let prior_precision = Array2::<Float>::eye(n_features) * alpha;

    // Likelihood precision (simplified)
    let noise_precision = 1.0;

    // Posterior precision = prior_precision + X^T X * noise_precision
    let xtx = X.t().dot(X);
    let posterior_precision = &prior_precision + &xtx * noise_precision;

    // Posterior covariance = inv(posterior_precision)
    let weight_cov = matrix_inverse(&posterior_precision)?;

    // Posterior mean = posterior_cov * X^T * y * noise_precision
    let xty = X.t().dot(&y_float);
    let weight_mean = weight_cov.dot(&(&xty * noise_precision));

    // Bias parameters (simplified)
    let bias_mean = 0.0;
    let bias_var = 1.0 / alpha;

    Ok(BayesianBinaryModel {
        weight_mean,
        weight_cov,
        bias_mean,
        bias_var,
        noise_precision,
    })
}

/// Predict with Bayesian binary classifier (mean prediction)
pub fn predict_bayesian_binary(
    X: &ArrayView2<Float>,
    model: &BayesianBinaryModel,
) -> Array1<Float> {
    let raw_scores = X.dot(&model.weight_mean) + model.bias_mean;
    raw_scores.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Predict with uncertainty quantification
pub fn predict_bayesian_uncertainty(
    X: &ArrayView2<Float>,
    model: &BayesianBinaryModel,
) -> SklResult<(Array1<Float>, Array1<Float>)> {
    let n_samples = X.nrows();
    let mut means = Array1::<Float>::zeros(n_samples);
    let mut variances = Array1::<Float>::zeros(n_samples);

    for i in 0..n_samples {
        let x = X.row(i);

        // Mean prediction
        let mean_score = x.dot(&model.weight_mean) + model.bias_mean;
        means[i] = 1.0 / (1.0 + (-mean_score).exp());

        // Variance calculation
        let score_var =
            x.dot(&model.weight_cov.dot(&x)) + model.bias_var + 1.0 / model.noise_precision;
        variances[i] = score_var;
    }

    Ok((means, variances))
}

/// Predict mean of Bayesian model
pub fn predict_bayesian_mean(X: &ArrayView2<Float>, model: &BayesianBinaryModel) -> Array1<Float> {
    X.dot(&model.weight_mean) + model.bias_mean
}
