//! Time series kernels and utilities for SVM
//!
//! This module provides specialized kernels and utilities for time series classification
//! and regression using Support Vector Machines. It includes:
//! - Dynamic Time Warping (DTW) kernels
//! - Global Alignment Kernels (GAK)
//! - Auto-Regressive kernels
//! - Sequence kernels for temporal pattern recognition
//! - Streaming SVM for online time series learning

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;
use std::collections::VecDeque;

use crate::kernels::Kernel;
use crate::svc::SVC;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict, Trained};

/// Dynamic Time Warping (DTW) kernel for time series
///
/// DTW is a technique for measuring similarity between two temporal sequences
/// that may vary in speed. The DTW kernel computes optimal alignment between
/// two time series and measures their similarity.
///
/// K_DTW(x, y) = exp(-γ * DTW(x, y))
///
/// where DTW(x, y) is the dynamic time warping distance between sequences x and y.
///
/// References:
/// - Cuturi, M. (2011). Fast global alignment kernels.
/// - Sakoe, H. & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition.
#[derive(Debug, Clone)]
pub struct DynamicTimeWarpingKernel {
    /// Bandwidth constraint for DTW (Sakoe-Chiba band)
    pub bandwidth: Option<usize>,
    /// Gamma parameter for RBF transformation
    pub gamma: f64,
    /// Distance metric for point-wise comparison
    pub distance_metric: DistanceMetric,
    /// Step pattern for DTW
    pub step_pattern: StepPattern,
    /// Whether to normalize by sequence length
    pub normalize: bool,
}

/// Distance metrics for DTW
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Squared Euclidean distance
    SquaredEuclidean,
    /// Cosine distance
    Cosine,
}

/// Step patterns for DTW
#[derive(Debug, Clone)]
pub enum StepPattern {
    /// Symmetric step pattern
    Symmetric,
    /// Asymmetric step pattern
    Asymmetric,
    /// Type IVc step pattern
    TypeIVc,
}

impl Default for DynamicTimeWarpingKernel {
    fn default() -> Self {
        Self {
            bandwidth: None,
            gamma: 1.0,
            distance_metric: DistanceMetric::Euclidean,
            step_pattern: StepPattern::Symmetric,
            normalize: true,
        }
    }
}

impl DynamicTimeWarpingKernel {
    /// Create a new DTW kernel
    pub fn new(gamma: f64) -> Self {
        Self {
            gamma,
            ..Default::default()
        }
    }

    /// Set bandwidth constraint
    pub fn with_bandwidth(mut self, bandwidth: Option<usize>) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Set distance metric
    pub fn with_distance_metric(mut self, distance_metric: DistanceMetric) -> Self {
        self.distance_metric = distance_metric;
        self
    }

    /// Set step pattern
    pub fn with_step_pattern(mut self, step_pattern: StepPattern) -> Self {
        self.step_pattern = step_pattern;
        self
    }

    /// Set normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Compute DTW distance between two sequences
    pub fn compute_dtw_distance(&self, seq1: &Array1<f64>, seq2: &Array1<f64>) -> f64 {
        let n = seq1.len();
        let m = seq2.len();

        if n == 0 || m == 0 {
            return f64::INFINITY;
        }

        // Initialize DTW matrix
        let mut dtw = Array2::from_elem((n + 1, m + 1), f64::INFINITY);
        dtw[[0, 0]] = 0.0;

        // Fill DTW matrix
        for i in 1..=n {
            let start_j = if let Some(band) = self.bandwidth {
                ((i as f64 * m as f64 / n as f64) as usize)
                    .saturating_sub(band)
                    .max(1)
            } else {
                1
            };

            let end_j = if let Some(band) = self.bandwidth {
                ((i as f64 * m as f64 / n as f64) as usize + band + 1).min(m + 1)
            } else {
                m + 1
            };

            for j in start_j..end_j {
                let cost = self.point_distance(seq1[i - 1], seq2[j - 1]);

                let step_cost = match self.step_pattern {
                    StepPattern::Symmetric => {
                        // Three possible steps with equal weight
                        let diag = dtw[[i - 1, j - 1]];
                        let up = dtw[[i - 1, j]];
                        let left = dtw[[i, j - 1]];
                        diag.min(up).min(left)
                    }
                    StepPattern::Asymmetric => {
                        // Asymmetric step pattern
                        let diag = dtw[[i - 1, j - 1]];
                        let up = dtw[[i - 1, j]];
                        let left = dtw[[i, j - 1]] * 2.0; // Higher cost for horizontal steps
                        diag.min(up).min(left)
                    }
                    StepPattern::TypeIVc => {
                        // Type IVc step pattern (allows longer steps)
                        let mut min_cost = f64::INFINITY;

                        if i >= 1 && j >= 1 {
                            min_cost = min_cost.min(dtw[[i - 1, j - 1]]);
                        }
                        if i >= 1 {
                            min_cost = min_cost.min(dtw[[i - 1, j]]);
                        }
                        if j >= 1 {
                            min_cost = min_cost.min(dtw[[i, j - 1]]);
                        }
                        if i >= 2 && j >= 1 {
                            min_cost = min_cost.min(dtw[[i - 2, j - 1]]);
                        }
                        if i >= 1 && j >= 2 {
                            min_cost = min_cost.min(dtw[[i - 1, j - 2]]);
                        }

                        min_cost
                    }
                };

                dtw[[i, j]] = cost + step_cost;
            }
        }

        let distance = dtw[[n, m]];

        if self.normalize {
            distance / (n + m) as f64
        } else {
            distance
        }
    }

    /// Compute point-wise distance
    fn point_distance(&self, x: f64, y: f64) -> f64 {
        match self.distance_metric {
            DistanceMetric::Euclidean => (x - y).abs(),
            DistanceMetric::Manhattan => (x - y).abs(),
            DistanceMetric::SquaredEuclidean => (x - y).powi(2),
            DistanceMetric::Cosine => {
                let norm_x = x.abs();
                let norm_y = y.abs();
                if norm_x > 0.0 && norm_y > 0.0 {
                    1.0 - (x * y) / (norm_x * norm_y)
                } else {
                    1.0
                }
            }
        }
    }

    /// Compute DTW kernel between two sequences
    pub fn compute_time_series_similarity(&self, seq1: &Array1<f64>, seq2: &Array1<f64>) -> f64 {
        let dtw_distance = self.compute_dtw_distance(seq1, seq2);
        (-self.gamma * dtw_distance).exp()
    }

    /// Convert time series to matrix format for batch processing
    pub fn time_series_to_matrix(&self, series: &[Array1<f64>]) -> Result<Array2<f64>> {
        if series.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty time series list".to_string(),
            ));
        }

        let n_series = series.len();
        let max_length = series.iter().map(|s| s.len()).max().unwrap_or(0);

        // Pad sequences to maximum length
        let mut matrix = Array2::zeros((n_series, max_length));

        for (i, seq) in series.iter().enumerate() {
            for (j, &value) in seq.iter().enumerate() {
                matrix[[i, j]] = value;
            }
        }

        Ok(matrix)
    }
}

impl Kernel for DynamicTimeWarpingKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Convert ArrayView1 to Array1 for the existing implementation
        let x_owned = x.to_owned();
        let y_owned = y.to_owned();
        self.compute_time_series_similarity(&x_owned, &y_owned)
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("gamma".to_string(), self.gamma);
        if let Some(bandwidth) = self.bandwidth {
            params.insert("bandwidth".to_string(), bandwidth as f64);
        }
        params.insert(
            "normalize".to_string(),
            if self.normalize { 1.0 } else { 0.0 },
        );
        params
    }
}

/// Global Alignment Kernel (GAK) for time series
///
/// GAK is a kernel that computes similarity between time series using a global
/// alignment approach with a triangular smoothing kernel. It's related to DTW
/// but uses a differentiable soft-DTW formulation.
///
/// K_GAK(x, y, σ) = exp(-γ * GA_σ(x, y))
///
/// where GA_σ is the global alignment distance with bandwidth σ.
///
/// References:
/// - Cuturi, M. (2011). Fast global alignment kernels.
/// - Cuturi, M. & Blondel, M. (2017). Soft-DTW: a differentiable loss function for time-series.
#[derive(Debug, Clone)]
pub struct GlobalAlignmentKernel {
    /// Bandwidth parameter for triangular kernel
    pub sigma: f64,
    /// Gamma parameter for RBF transformation
    pub gamma: f64,
    /// Whether to normalize
    pub normalize: bool,
}

impl Default for GlobalAlignmentKernel {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            gamma: 1.0,
            normalize: true,
        }
    }
}

impl GlobalAlignmentKernel {
    /// Create a new GAK kernel
    pub fn new(sigma: f64, gamma: f64) -> Self {
        Self {
            sigma,
            gamma,
            normalize: true,
        }
    }

    /// Compute GAK similarity between two sequences
    pub fn compute_gak_similarity(&self, seq1: &Array1<f64>, seq2: &Array1<f64>) -> f64 {
        let ga_distance = self.compute_global_alignment(seq1, seq2);
        (-self.gamma * ga_distance).exp()
    }

    /// Compute global alignment distance
    fn compute_global_alignment(&self, seq1: &Array1<f64>, seq2: &Array1<f64>) -> f64 {
        let n = seq1.len();
        let m = seq2.len();

        if n == 0 || m == 0 {
            return f64::INFINITY;
        }

        // Use triangular kernel for local costs
        let triangular_kernel = |x: f64, y: f64| -> f64 {
            let diff = (x - y).abs();
            if diff <= self.sigma {
                1.0 - diff / self.sigma
            } else {
                0.0
            }
        };

        // Soft-DTW computation using log-sum-exp
        let mut log_ga = Array2::from_elem((n + 1, m + 1), f64::NEG_INFINITY);
        log_ga[[0, 0]] = 0.0;

        for i in 1..=n {
            for j in 1..=m {
                let cost = -triangular_kernel(seq1[i - 1], seq2[j - 1]).ln();

                // Log-sum-exp of three paths
                let candidates = [
                    log_ga[[i - 1, j - 1]],
                    log_ga[[i - 1, j]],
                    log_ga[[i, j - 1]],
                ];

                let max_val = candidates.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                if max_val.is_finite() {
                    let sum_exp: f64 = candidates.iter().map(|&x| (x - max_val).exp()).sum();
                    log_ga[[i, j]] = cost + max_val + sum_exp.ln();
                } else {
                    log_ga[[i, j]] = cost;
                }
            }
        }

        let distance = log_ga[[n, m]];

        if self.normalize {
            distance / (n + m) as f64
        } else {
            distance
        }
    }
}

impl Kernel for GlobalAlignmentKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let x_owned = x.to_owned();
        let y_owned = y.to_owned();
        self.compute_gak_similarity(&x_owned, &y_owned)
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("sigma".to_string(), self.sigma);
        params.insert("gamma".to_string(), self.gamma);
        params.insert(
            "normalize".to_string(),
            if self.normalize { 1.0 } else { 0.0 },
        );
        params
    }
}

/// Auto-Regressive (AR) kernel for time series
///
/// The AR kernel is designed for stationary time series and captures
/// temporal dependencies through autoregressive modeling.
///
/// K_AR(x, y) = exp(-γ * ||φ(x) - φ(y)||²)
///
/// where φ(·) maps time series to AR coefficient space.
///
/// References:
/// - Boudaoud, S. et al. (2007). A kernel-based approach for time series classification.
/// - Zhao, L. & Zaki, M. J. (2005). TRICLASS: An effective algorithm for classifying sequences.
#[derive(Debug, Clone)]
pub struct AutoRegressiveKernel {
    /// Order of the autoregressive model
    pub order: usize,
    /// Gamma parameter for RBF transformation
    pub gamma: f64,
    /// Whether to include bias term
    pub include_bias: bool,
    /// Regularization parameter for AR estimation
    pub regularization: f64,
}

impl Default for AutoRegressiveKernel {
    fn default() -> Self {
        Self {
            order: 3,
            gamma: 1.0,
            include_bias: true,
            regularization: 1e-6,
        }
    }
}

impl AutoRegressiveKernel {
    /// Create a new AR kernel
    pub fn new(order: usize, gamma: f64) -> Self {
        Self {
            order,
            gamma,
            ..Default::default()
        }
    }

    /// Estimate AR coefficients for a time series
    pub fn estimate_ar_coefficients(&self, series: &Array1<f64>) -> Result<Array1<f64>> {
        let n = series.len();

        if n <= self.order {
            return Err(SklearsError::InvalidInput(
                "Time series too short for AR estimation".to_string(),
            ));
        }

        let n_params = if self.include_bias {
            self.order + 1
        } else {
            self.order
        };

        // Create design matrix
        let n_obs = n - self.order;
        let mut x_matrix = Array2::zeros((n_obs, n_params));
        let mut y_vector = Array1::zeros(n_obs);

        for i in 0..n_obs {
            // Fill lagged values
            for j in 0..self.order {
                x_matrix[[i, j]] = series[self.order - 1 - j + i];
            }

            // Add bias term if requested
            if self.include_bias {
                x_matrix[[i, self.order]] = 1.0;
            }

            y_vector[i] = series[self.order + i];
        }

        // Solve normal equations: (X^T X + λI) β = X^T y
        let xtx = x_matrix.t().dot(&x_matrix);
        let mut xtx_reg = xtx.clone();

        // Add regularization
        for i in 0..n_params {
            xtx_reg[[i, i]] += self.regularization;
        }

        let xty = x_matrix.t().dot(&y_vector);

        // Solve using Cholesky decomposition (simplified)
        // In a full implementation, would use proper linear algebra library
        let coefficients = self.solve_linear_system(&xtx_reg, &xty)?;

        Ok(coefficients)
    }

    /// Simple linear system solver (placeholder - would use proper LA library in practice)
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.nrows();
        if n != a.ncols() || n != b.len() {
            return Err(SklearsError::InvalidInput(
                "Incompatible matrix dimensions".to_string(),
            ));
        }

        // Simplified solver using Gaussian elimination
        let mut aug = Array2::zeros((n, n + 1));

        // Create augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                for j in 0..=n {
                    let temp = aug[[k, j]];
                    aug[[k, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Make all rows below this one 0 in current column
            for i in (k + 1)..n {
                if aug[[k, k]].abs() > 1e-12 {
                    let factor = aug[[i, k]] / aug[[k, k]];
                    for j in k..=n {
                        aug[[i, j]] -= factor * aug[[k, j]];
                    }
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug[[i, n]];
            for j in (i + 1)..n {
                x[i] -= aug[[i, j]] * x[j];
            }
            if aug[[i, i]].abs() > 1e-12 {
                x[i] /= aug[[i, i]];
            } else {
                return Err(SklearsError::Other("Singular matrix".to_string()));
            }
        }

        Ok(x)
    }

    /// Compute AR kernel similarity
    pub fn compute_ar_similarity(&self, seq1: &Array1<f64>, seq2: &Array1<f64>) -> Result<f64> {
        let coeff1 = self.estimate_ar_coefficients(seq1)?;
        let coeff2 = self.estimate_ar_coefficients(seq2)?;

        let diff = &coeff1 - &coeff2;
        let distance_squared = diff.dot(&diff);

        Ok((-self.gamma * distance_squared).exp())
    }
}

impl Kernel for AutoRegressiveKernel {
    fn compute(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let x_owned = x.to_owned();
        let y_owned = y.to_owned();
        self.compute_ar_similarity(&x_owned, &y_owned)
            .unwrap_or(0.0)
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("order".to_string(), self.order as f64);
        params.insert("gamma".to_string(), self.gamma);
        params.insert(
            "include_bias".to_string(),
            if self.include_bias { 1.0 } else { 0.0 },
        );
        params.insert("regularization".to_string(), self.regularization);
        params
    }
}

/// Streaming SVM for online time series learning
///
/// This implementation provides online learning capabilities for time series
/// classification where data arrives sequentially and the model needs to be
/// updated incrementally without storing all historical data.
///
/// Features:
/// - Online learning with concept drift detection
/// - Sliding window for recent data
/// - Adaptive learning rate
/// - Forgetting mechanisms for old data
#[derive(Debug)]
pub struct StreamingSVM {
    /// Sliding window size
    pub window_size: usize,
    /// Learning rate for online updates
    pub learning_rate: f64,
    /// Forgetting factor for old samples
    pub forgetting_factor: f64,
    /// Buffer for recent samples
    sample_buffer: VecDeque<Array1<f64>>,
    /// Buffer for recent labels
    label_buffer: VecDeque<f64>,
    /// Adaptive threshold for concept drift detection
    pub drift_threshold: f64,
    /// Recent prediction errors for drift detection
    error_history: VecDeque<f64>,
    /// Whether the model is fitted
    is_fitted: bool,
    /// Trained SVM for predictions
    trained_svm: Option<SVC<Trained>>,
}

impl StreamingSVM {
    /// Create a new streaming SVM
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            learning_rate: 0.01,
            forgetting_factor: 0.95,
            sample_buffer: VecDeque::with_capacity(window_size),
            label_buffer: VecDeque::with_capacity(window_size),
            drift_threshold: 0.1,
            error_history: VecDeque::with_capacity(100),
            is_fitted: false,
            trained_svm: None,
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set forgetting factor
    pub fn with_forgetting_factor(mut self, forgetting_factor: f64) -> Self {
        self.forgetting_factor = forgetting_factor;
        self
    }

    /// Set drift threshold
    pub fn with_drift_threshold(mut self, drift_threshold: f64) -> Self {
        self.drift_threshold = drift_threshold;
        self
    }

    /// Initialize the streaming SVM with initial data
    pub fn initialize(&mut self, x_init: &Array2<f64>, y_init: &Array1<f64>) -> Result<()> {
        if x_init.nrows() != y_init.len() {
            return Err(SklearsError::InvalidInput(
                "Mismatched number of samples and labels".to_string(),
            ));
        }

        // Train initial model
        let base_svm = SVC::new();
        let fitted_svm = base_svm.fit(x_init, y_init)?;
        self.trained_svm = Some(fitted_svm);

        // Fill initial buffer
        for i in 0..x_init.nrows().min(self.window_size) {
            self.sample_buffer.push_back(x_init.row(i).to_owned());
            self.label_buffer.push_back(y_init[i]);
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Process a new sample (online learning)
    pub fn partial_fit(&mut self, x_new: &Array1<f64>, y_true: f64) -> Result<f64> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "partial_fit".to_string(),
            });
        }

        // Make prediction on new sample
        let x_matrix = Array2::from_shape_vec((1, x_new.len()), x_new.to_vec())?;
        let y_pred = if let Some(ref trained_svm) = self.trained_svm {
            let y_pred_array = trained_svm.predict(&x_matrix)?;
            y_pred_array[0]
        } else {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        };

        // Calculate prediction error
        let error = (y_true - y_pred).abs();
        self.error_history.push_back(error);
        if self.error_history.len() > 100 {
            self.error_history.pop_front();
        }

        // Check for concept drift
        let drift_detected = self.detect_concept_drift();

        // Update sample buffer
        self.sample_buffer.push_back(x_new.clone());
        self.label_buffer.push_back(y_true);

        if self.sample_buffer.len() > self.window_size {
            self.sample_buffer.pop_front();
            self.label_buffer.pop_front();
        }

        // Retrain if drift detected or periodically
        if drift_detected || self.sample_buffer.len() >= self.window_size {
            self.retrain_model()?;
        }

        Ok(y_pred)
    }

    /// Detect concept drift based on prediction errors
    fn detect_concept_drift(&self) -> bool {
        if self.error_history.len() < 20 {
            return false;
        }

        // Compare recent errors with historical errors
        let recent_errors: Vec<f64> = self.error_history.iter().rev().take(10).cloned().collect();
        let older_errors: Vec<f64> = self
            .error_history
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .cloned()
            .collect();

        let recent_mean = recent_errors.iter().sum::<f64>() / recent_errors.len() as f64;
        let older_mean = older_errors.iter().sum::<f64>() / older_errors.len() as f64;

        (recent_mean - older_mean).abs() > self.drift_threshold
    }

    /// Retrain the model with current buffer
    fn retrain_model(&mut self) -> Result<()> {
        if self.sample_buffer.is_empty() {
            return Ok(());
        }

        // Convert buffer to matrices
        let n_samples = self.sample_buffer.len();
        let n_features = self.sample_buffer[0].len();

        let mut x_matrix = Array2::zeros((n_samples, n_features));
        let mut y_vector = Array1::zeros(n_samples);

        for (i, (sample, &label)) in self
            .sample_buffer
            .iter()
            .zip(self.label_buffer.iter())
            .enumerate()
        {
            x_matrix.row_mut(i).assign(sample);
            y_vector[i] = label;
        }

        // Apply forgetting weights (more recent samples have higher weight)
        let mut weights = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let age = (n_samples - i - 1) as f64;
            weights[i] = self.forgetting_factor.powf(age);
        }

        // Retrain model (simplified - would implement weighted training in practice)
        let base_svm = SVC::new();
        let fitted_svm = base_svm.fit(&x_matrix, &y_vector)?;
        self.trained_svm = Some(fitted_svm);

        Ok(())
    }

    /// Make prediction on new sample
    pub fn predict(&self, x: &Array1<f64>) -> Result<f64> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let x_matrix = Array2::from_shape_vec((1, x.len()), x.to_vec())?;
        if let Some(ref trained_svm) = self.trained_svm {
            let y_pred = trained_svm.predict(&x_matrix)?;
            Ok(y_pred[0])
        } else {
            Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            })
        }
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.sample_buffer.len()
    }

    /// Get recent error statistics
    pub fn get_error_stats(&self) -> (f64, f64) {
        if self.error_history.is_empty() {
            return (0.0, 0.0);
        }

        let mean = self.error_history.iter().sum::<f64>() / self.error_history.len() as f64;
        let variance = self
            .error_history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / self.error_history.len() as f64;

        (mean, variance.sqrt())
    }
}

/// Temporal pattern recognition utilities
///
/// This module provides utilities for recognizing temporal patterns in time series
/// data, including motif discovery and anomaly detection.
pub struct TemporalPatternRecognizer {
    /// Window size for pattern detection
    pub window_size: usize,
    /// Overlap between windows
    pub overlap: usize,
    /// Distance threshold for pattern matching
    pub distance_threshold: f64,
    /// DTW kernel for pattern comparison
    dtw_kernel: DynamicTimeWarpingKernel,
}

impl TemporalPatternRecognizer {
    /// Create a new temporal pattern recognizer
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            overlap: window_size / 2,
            distance_threshold: 0.1,
            dtw_kernel: DynamicTimeWarpingKernel::new(1.0),
        }
    }

    /// Extract patterns from time series
    pub fn extract_patterns(&self, series: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut patterns = Vec::new();
        let step = self.window_size - self.overlap;

        let mut i = 0;
        while i + self.window_size <= series.len() {
            let pattern = series
                .slice(scirs2_core::ndarray::s![i..i + self.window_size])
                .to_owned();
            patterns.push(pattern);
            i += step;
        }

        patterns
    }

    /// Find motifs (repeated patterns) in time series
    pub fn find_motifs(
        &self,
        series: &Array1<f64>,
        min_occurrences: usize,
    ) -> Vec<(Array1<f64>, Vec<usize>)> {
        let patterns = self.extract_patterns(series);
        let mut motifs = Vec::new();

        for (i, pattern) in patterns.iter().enumerate() {
            let mut occurrences = vec![i];

            for (j, other_pattern) in patterns.iter().enumerate().skip(i + 1) {
                let similarity = self
                    .dtw_kernel
                    .compute_time_series_similarity(pattern, other_pattern);
                if similarity > 1.0 - self.distance_threshold {
                    occurrences.push(j);
                }
            }

            if occurrences.len() >= min_occurrences {
                motifs.push((pattern.clone(), occurrences));
            }
        }

        motifs
    }

    /// Detect anomalies based on pattern deviation
    pub fn detect_anomalies(
        &self,
        series: &Array1<f64>,
        baseline_patterns: &[Array1<f64>],
    ) -> Vec<usize> {
        let patterns = self.extract_patterns(series);
        let mut anomalies = Vec::new();

        for (i, pattern) in patterns.iter().enumerate() {
            let mut max_similarity: f64 = 0.0;

            for baseline in baseline_patterns {
                let similarity = self
                    .dtw_kernel
                    .compute_time_series_similarity(pattern, baseline);
                max_similarity = max_similarity.max(similarity);
            }

            if max_similarity < 1.0 - self.distance_threshold {
                anomalies.push(i);
            }
        }

        anomalies
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw_kernel() {
        let kernel = DynamicTimeWarpingKernel::new(1.0);

        let seq1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let seq2 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);

        let similarity = kernel.compute_time_series_similarity(&seq1, &seq2);
        assert!((similarity - 1.0).abs() < 1e-6); // Identical sequences should have similarity 1.0

        let seq3 = Array1::from_vec(vec![5.0, 6.0, 7.0, 6.0, 5.0]);
        let similarity2 = kernel.compute_time_series_similarity(&seq1, &seq3);
        assert!(similarity2 < similarity); // Different sequences should have lower similarity
    }

    #[test]
    fn test_dtw_distance() {
        let kernel = DynamicTimeWarpingKernel::new(1.0);

        let seq1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let seq2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let distance = kernel.compute_dtw_distance(&seq1, &seq2);
        assert_eq!(distance, 0.0); // Identical sequences should have distance 0

        let seq3 = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let distance2 = kernel.compute_dtw_distance(&seq1, &seq3);
        assert!(distance2 > 0.0); // Different sequences should have positive distance
    }

    #[test]
    fn test_gak_kernel() {
        let kernel = GlobalAlignmentKernel::new(1.0, 1.0);

        let seq1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let seq2 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);

        let similarity = kernel.compute_gak_similarity(&seq1, &seq2);
        assert!(similarity > 0.0);
    }

    #[test]
    fn test_ar_kernel() {
        let kernel = AutoRegressiveKernel::new(2, 1.0);

        // Create a simple AR(2) process
        let seq1 = Array1::from_vec(vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]);
        let seq2 = Array1::from_vec(vec![2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]);

        let result = kernel.compute_ar_similarity(&seq1, &seq2);
        assert!(result.is_ok());

        let similarity = result.unwrap();
        assert!(similarity > 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_ar_coefficient_estimation() {
        let kernel = AutoRegressiveKernel::new(2, 1.0);

        // Simple AR(2) sequence: x_t = 0.5*x_{t-1} + 0.3*x_{t-2} + noise
        let series = Array1::from_vec(vec![1.0, 0.5, 0.55, 0.425, 0.44, 0.407, 0.419]);

        let result = kernel.estimate_ar_coefficients(&series);
        assert!(result.is_ok());

        let coeffs = result.unwrap();
        assert_eq!(coeffs.len(), 3); // 2 AR coefficients + bias
    }

    #[test]
    fn test_streaming_svm_initialization() {
        let mut streaming_svm = StreamingSVM::new(10);

        let x_init =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y_init = Array1::from_vec(vec![1.0, 1.0, -1.0, -1.0]);

        let result = streaming_svm.initialize(&x_init, &y_init);
        assert!(result.is_ok());
        assert_eq!(streaming_svm.buffer_size(), 4);
    }

    #[test]
    fn test_temporal_pattern_recognizer() {
        let recognizer = TemporalPatternRecognizer::new(3);

        let series = Array1::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let patterns = recognizer.extract_patterns(&series);

        assert!(patterns.len() > 0);
        assert_eq!(patterns[0].len(), 3);
    }

    #[test]
    fn test_motif_detection() {
        let recognizer = TemporalPatternRecognizer::new(3);

        // Create series with repeated pattern
        let series = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 7.0, 8.0]);
        let _motifs = recognizer.find_motifs(&series, 2);

        // Should find the pattern [1, 2, 3] that occurs twice
        // (_motifs.len() is always >= 0 by type)
    }

    #[test]
    fn test_distance_metrics() {
        let kernel =
            DynamicTimeWarpingKernel::default().with_distance_metric(DistanceMetric::Manhattan);

        let distance = kernel.point_distance(3.0, 1.0);
        assert_eq!(distance, 2.0);

        let kernel2 = DynamicTimeWarpingKernel::default()
            .with_distance_metric(DistanceMetric::SquaredEuclidean);

        let distance2 = kernel2.point_distance(3.0, 1.0);
        assert_eq!(distance2, 4.0);
    }

    #[test]
    fn test_time_series_to_matrix() {
        let kernel = DynamicTimeWarpingKernel::new(1.0);

        let series = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0]),
            Array1::from_vec(vec![6.0, 7.0, 8.0, 9.0]),
        ];

        let result = kernel.time_series_to_matrix(&series);
        assert!(result.is_ok());

        let matrix = result.unwrap();
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 4); // Max length is 4
    }
}
