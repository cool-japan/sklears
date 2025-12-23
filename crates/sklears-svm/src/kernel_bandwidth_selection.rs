//! Kernel Bandwidth Selection Algorithms
//!
//! This module implements sophisticated algorithms for automatically selecting
//! optimal kernel bandwidth (gamma) parameters for RBF and related kernels.
//! Proper bandwidth selection is critical for SVM performance, and these
//! algorithms provide principled, data-driven approaches.
//!
//! Methods implemented:
//! - Median Heuristic (robust, widely used baseline)
//! - Scott's Rule (density estimation theory)
//! - Silverman's Rule (optimal for Gaussian data)
//! - Maximum Likelihood Cross-Validation (MLCV)
//! - Least Squares Cross-Validation (LSCV)
//! - Kernel Alignment Optimization
//! - Leave-One-Out Likelihood
//! - Grid Search with Cross-Validation

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{essentials::Uniform, seeded_rng, CoreRandom};
use std::collections::HashMap;
use thiserror::Error;

/// Errors for bandwidth selection
#[derive(Error, Debug)]
pub enum BandwidthError {
    #[error("Insufficient data: need at least {required} samples, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    #[error("Invalid bandwidth range: min={min}, max={max}")]
    InvalidRange { min: f64, max: f64 },
    #[error("Optimization failed to converge after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },
    #[error("Numerical instability: {message}")]
    NumericalInstability { message: String },
    #[error("Invalid kernel parameter: {message}")]
    InvalidParameter { message: String },
}

/// Result type for bandwidth selection
pub type BandwidthResult<T> = Result<T, BandwidthError>;

/// Bandwidth selection method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandwidthMethod {
    /// Median heuristic (robust, popular default)
    MedianHeuristic,
    /// Scott's rule from density estimation
    ScottRule,
    /// Silverman's rule (optimal for Gaussian)
    SilvermanRule,
    /// Maximum likelihood cross-validation
    MLCV { n_folds: usize },
    /// Least squares cross-validation
    LSCV { n_folds: usize },
    /// Kernel alignment maximization
    KernelAlignment,
    /// Leave-one-out likelihood
    LeaveOneOut,
    /// Grid search with cross-validation
    GridSearchCV { n_folds: usize, n_points: usize },
}

/// Configuration for bandwidth selection
#[derive(Debug, Clone)]
pub struct BandwidthConfig {
    pub method: BandwidthMethod,
    pub search_range: (f64, f64), // (min_gamma, max_gamma)
    pub max_iterations: usize,
    pub tolerance: f64,
    pub random_state: Option<u64>,
    pub verbose: bool,
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            method: BandwidthMethod::MedianHeuristic,
            search_range: (1e-6, 1e2),
            max_iterations: 100,
            tolerance: 1e-6,
            random_state: None,
            verbose: false,
        }
    }
}

/// Bandwidth selection result with diagnostics
#[derive(Debug, Clone)]
pub struct BandwidthSelection {
    pub gamma: f64,
    pub method: BandwidthMethod,
    pub score: Option<f64>,
    pub search_history: Vec<(f64, f64)>, // (gamma, score) pairs
    pub computation_time: std::time::Duration,
}

/// Kernel bandwidth selector
#[derive(Debug, Clone)]
pub struct KernelBandwidthSelector {
    config: BandwidthConfig,
}

impl KernelBandwidthSelector {
    /// Create a new bandwidth selector
    pub fn new(config: BandwidthConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(BandwidthConfig::default())
    }

    /// Create with specific method
    pub fn with_method(method: BandwidthMethod) -> Self {
        Self::new(BandwidthConfig {
            method,
            ..Default::default()
        })
    }

    /// Select optimal bandwidth for given data
    pub fn select(&self, X: &Array2<f64>) -> BandwidthResult<BandwidthSelection> {
        let start_time = std::time::Instant::now();

        let (gamma, score, history) = match self.config.method {
            BandwidthMethod::MedianHeuristic => {
                let gamma = self.median_heuristic(X)?;
                (gamma, None, vec![])
            }
            BandwidthMethod::ScottRule => {
                let gamma = self.scott_rule(X)?;
                (gamma, None, vec![])
            }
            BandwidthMethod::SilvermanRule => {
                let gamma = self.silverman_rule(X)?;
                (gamma, None, vec![])
            }
            BandwidthMethod::MLCV { n_folds } => {
                let (gamma, score, history) = self.mlcv(x, n_folds)?;
                (gamma, Some(score), history)
            }
            BandwidthMethod::LSCV { n_folds } => {
                let (gamma, score, history) = self.lscv(x, n_folds)?;
                (gamma, Some(score), history)
            }
            BandwidthMethod::KernelAlignment => {
                let (gamma, score, history) = self.kernel_alignment(X)?;
                (gamma, Some(score), history)
            }
            BandwidthMethod::LeaveOneOut => {
                let (gamma, score, history) = self.leave_one_out(X)?;
                (gamma, Some(score), history)
            }
            BandwidthMethod::GridSearchCV { n_folds, n_points } => {
                let (gamma, score, history) = self.grid_search_cv(x, n_folds, n_points)?;
                (gamma, Some(score), history)
            }
        };

        let computation_time = start_time.elapsed();

        Ok(BandwidthSelection {
            gamma,
            method: self.config.method,
            score,
            search_history: history,
            computation_time,
        })
    }

    /// Select bandwidth with labels (for supervised kernel alignment)
    pub fn select_supervised(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
    ) -> BandwidthResult<BandwidthSelection> {
        let start_time = std::time::Instant::now();

        let (gamma, score, history) = match self.config.method {
            BandwidthMethod::KernelAlignment => {
                let (gamma, score, history) = self.supervised_kernel_alignment(x, y)?;
                (gamma, Some(score), history)
            }
            _ => {
                // Fall back to unsupervised for other methods
                return self.select(X);
            }
        };

        let computation_time = start_time.elapsed();

        Ok(BandwidthSelection {
            gamma,
            method: self.config.method,
            score,
            search_history: history,
            computation_time,
        })
    }

    // Individual selection methods

    /// Median heuristic: gamma = 1 / (2 * median(pairwise_distances)²)
    fn median_heuristic(&self, X: &Array2<f64>) -> BandwidthResult<f64> {
        let (n_samples, _) = x.dim();

        if n_samples < 2 {
            return Err(BandwidthError::InsufficientData {
                required: 2,
                actual: n_samples,
            });
        }

        // Compute pairwise squared distances (sample subset for efficiency)
        let max_samples = 1000.min(n_samples);
        let mut distances = Vec::new();

        for i in 0..max_samples {
            for j in (i + 1)..max_samples {
                let dist_sq = self.squared_distance(&x.row(i), &x.row(j));
                distances.push(dist_sq);
            }
        }

        if distances.is_empty() {
            return Err(BandwidthError::InsufficientData {
                required: 2,
                actual: 0,
            });
        }

        // Compute median
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_dist_sq = distances[distances.len() / 2];

        if median_dist_sq <= 0.0 {
            return Err(BandwidthError::NumericalInstability {
                message: "Median distance is zero".to_string(),
            });
        }

        let gamma = 1.0 / (2.0 * median_dist_sq);

        Ok(gamma.clamp(self.config.search_range.0, self.config.search_range.1))
    }

    /// Scott's rule: h = n^(-1/(d+4)) * sigma
    fn scott_rule(&self, X: &Array2<f64>) -> BandwidthResult<f64> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(BandwidthError::InsufficientData {
                required: 2,
                actual: n_samples,
            });
        }

        // Compute standard deviation (average across features)
        let mut total_std = 0.0;

        for j in 0..n_features {
            let column = X.column(j);
            let mean = column.mean().unwrap_or(0.0);
            let variance: f64 =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
            total_std += variance.sqrt();
        }

        let avg_std = total_std / n_features as f64;

        // Scott's bandwidth
        let h = avg_std * (n_samples as f64).powf(-1.0 / (n_features as f64 + 4.0));

        // Convert to gamma
        let gamma = 1.0 / (2.0 * h * h);

        Ok(gamma.clamp(self.config.search_range.0, self.config.search_range.1))
    }

    /// Silverman's rule: h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * sigma
    fn silverman_rule(&self, X: &Array2<f64>) -> BandwidthResult<f64> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(BandwidthError::InsufficientData {
                required: 2,
                actual: n_samples,
            });
        }

        // Compute standard deviation
        let mut total_std = 0.0;

        for j in 0..n_features {
            let column = X.column(j);
            let mean = column.mean().unwrap_or(0.0);
            let variance: f64 =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
            total_std += variance.sqrt();
        }

        let avg_std = total_std / n_features as f64;

        // Silverman's bandwidth
        let d = n_features as f64;
        let n = n_samples as f64;
        let factor = (4.0 / (d + 2.0)).powf(1.0 / (d + 4.0));
        let h = factor * n.powf(-1.0 / (d + 4.0)) * avg_std;

        // Convert to gamma
        let gamma = 1.0 / (2.0 * h * h);

        Ok(gamma.clamp(self.config.search_range.0, self.config.search_range.1))
    }

    /// Maximum likelihood cross-validation
    fn mlcv(
        &self,
        X: &Array2<f64>,
        n_folds: usize,
    ) -> BandwidthResult<(f64, f64, Vec<(f64, f64)>)> {
        let n_samples = x.nrows();

        if n_samples < n_folds {
            return Err(BandwidthError::InsufficientData {
                required: n_folds,
                actual: n_samples,
            });
        }

        // Use golden section search
        self.golden_section_search(x, n_folds, |gamma, X_train, X_test| {
            self.compute_likelihood(X_train, X_test, gamma)
        })
    }

    /// Least squares cross-validation
    fn lscv(
        &self,
        X: &Array2<f64>,
        n_folds: usize,
    ) -> BandwidthResult<(f64, f64, Vec<(f64, f64)>)> {
        let n_samples = x.nrows();

        if n_samples < n_folds {
            return Err(BandwidthError::InsufficientData {
                required: n_folds,
                actual: n_samples,
            });
        }

        // Use golden section search with LSCV criterion
        self.golden_section_search(x, n_folds, |gamma, X_train, X_test| {
            self.compute_lscv_score(X_train, X_test, gamma)
        })
    }

    /// Kernel alignment maximization (unsupervised)
    fn kernel_alignment(&self, X: &Array2<f64>) -> BandwidthResult<(f64, f64, Vec<(f64, f64)>)> {
        // Compute ideal kernel (identity for unsupervised)
        let n = x.nrows();
        let ideal_kernel = Array2::eye(n);

        self.optimize_alignment(x, &ideal_kernel)
    }

    /// Supervised kernel alignment
    fn supervised_kernel_alignment(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
    ) -> BandwidthResult<(f64, f64, Vec<(f64, f64)>)> {
        let n = x.nrows();

        // Compute ideal kernel from labels: K_ideal[i,j] = y[i] * y[j]
        let mut ideal_kernel = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                ideal_kernel[[i, j]] = y[i] * y[j];
            }
        }

        // Normalize
        let norm: f64 = ideal_kernel.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            ideal_kernel /= norm;
        }

        self.optimize_alignment(x, &ideal_kernel)
    }

    /// Optimize kernel alignment
    fn optimize_alignment(
        &self,
        X: &Array2<f64>,
        ideal_kernel: &Array2<f64>,
    ) -> BandwidthResult<(f64, f64, Vec<(f64, f64)>)> {
        let mut best_gamma = self.config.search_range.0;
        let mut best_score = f64::NEG_INFINITY;
        let mut history = Vec::new();

        // Grid search for kernel alignment
        let n_points = 20;
        let log_min = self.config.search_range.0.ln();
        let log_max = self.config.search_range.1.ln();

        for i in 0..n_points {
            let log_gamma = log_min + (log_max - log_min) * i as f64 / (n_points - 1) as f64;
            let gamma = log_gamma.exp();

            // Compute kernel matrix
            let K = self.compute_rbf_kernel(x, gamma);

            // Compute alignment: <K, K_ideal> / (||K|| * ||K_ideal||)
            let alignment = self.kernel_alignment_score(&K, ideal_kernel);

            history.push((gamma, alignment));

            if alignment > best_score {
                best_score = alignment;
                best_gamma = gamma;
            }

            if self.config.verbose {
                eprintln!("Gamma: {:.6e}, Alignment: {:.6}", gamma, alignment);
            }
        }

        Ok((best_gamma, best_score, history))
    }

    /// Leave-one-out likelihood
    fn leave_one_out(&self, X: &Array2<f64>) -> BandwidthResult<(f64, f64, Vec<(f64, f64)>)> {
        let n_samples = x.nrows();

        if n_samples < 2 {
            return Err(BandwidthError::InsufficientData {
                required: 2,
                actual: n_samples,
            });
        }

        // Use golden section search
        self.golden_section_search(x, n_samples, |gamma, X_train, X_test| {
            // LOO: each test point is predicted from all others
            self.compute_likelihood(X_train, X_test, gamma)
        })
    }

    /// Grid search with cross-validation
    fn grid_search_cv(
        &self,
        X: &Array2<f64>,
        n_folds: usize,
        n_points: usize,
    ) -> BandwidthResult<(f64, f64, Vec<(f64, f64)>)> {
        let mut best_gamma = self.config.search_range.0;
        let mut best_score = f64::NEG_INFINITY;
        let mut history = Vec::new();

        // Logarithmic grid
        let log_min = self.config.search_range.0.ln();
        let log_max = self.config.search_range.1.ln();

        for i in 0..n_points {
            let log_gamma = log_min + (log_max - log_min) * i as f64 / (n_points - 1) as f64;
            let gamma = log_gamma.exp();

            // Cross-validation score
            let score = self.cross_validate(x, gamma, n_folds)?;

            history.push((gamma, score));

            if score > best_score {
                best_score = score;
                best_gamma = gamma;
            }

            if self.config.verbose {
                eprintln!("Gamma: {:.6e}, CV Score: {:.6}", gamma, score);
            }
        }

        Ok((best_gamma, best_score, history))
    }

    // Helper methods

    /// Golden section search for 1D optimization
    fn golden_section_search<F>(
        &self,
        X: &Array2<f64>,
        n_folds: usize,
        objective: F,
    ) -> BandwidthResult<(f64, f64, Vec<(f64, f64)>)>
    where
        F: Fn(f64, &Array2<f64>, &Array2<f64>) -> f64,
    {
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut a = self.config.search_range.0.ln();
        let mut b = self.config.search_range.1.ln();

        let mut history = Vec::new();

        for _ in 0..self.config.max_iterations {
            let c = b - (b - a) / golden_ratio;
            let d = a + (b - a) / golden_ratio;

            let gamma_c = c.exp();
            let gamma_d = d.exp();

            // Evaluate objective (simplified - using full data)
            let score_c = self.cross_validate_objective(x, gamma_c, n_folds, &objective)?;
            let score_d = self.cross_validate_objective(x, gamma_d, n_folds, &objective)?;

            history.push((gamma_c, score_c));
            history.push((gamma_d, score_d));

            if score_c > score_d {
                b = d;
            } else {
                a = c;
            }

            if (b - a).abs() < self.config.tolerance {
                let gamma = ((a + b) / 2.0).exp();
                let score = self.cross_validate_objective(x, gamma, n_folds, &objective)?;
                return Ok((gamma, score, history));
            }
        }

        let gamma = ((a + b) / 2.0).exp();
        let score = self.cross_validate_objective(x, gamma, n_folds, &objective)?;

        Ok((gamma, score, history))
    }

    /// Cross-validation objective evaluation
    fn cross_validate_objective<F>(
        &self,
        X: &Array2<f64>,
        gamma: f64,
        n_folds: usize,
        objective: &F,
    ) -> BandwidthResult<f64>
    where
        F: Fn(f64, &Array2<f64>, &Array2<f64>) -> f64,
    {
        let n_samples = x.nrows();
        let fold_size = n_samples / n_folds;
        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start = fold * fold_size;
            let end = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Split data
            let (X_train, X_test) = self.split_fold(x, start, end);

            let score = objective(gamma, &X_train, &X_test);
            scores.push(score);
        }

        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Split data into train/test for a fold
    fn split_fold(
        &self,
        X: &Array2<f64>,
        test_start: usize,
        test_end: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let train_size = n_samples - (test_end - test_start);
        let test_size = test_end - test_start;

        let mut X_train = Array2::zeros((train_size, n_features));
        let mut X_test = Array2::zeros((test_size, n_features));

        let mut train_idx = 0;
        let mut test_idx = 0;

        for i in 0..n_samples {
            if i >= test_start && i < test_end {
                X_test.row_mut(test_idx).assign(&x.row(i));
                test_idx += 1;
            } else {
                X_train.row_mut(train_idx).assign(&x.row(i));
                train_idx += 1;
            }
        }

        (X_train, X_test)
    }

    /// Compute likelihood score
    fn compute_likelihood(&self, X_train: &Array2<f64>, X_test: &Array2<f64>, gamma: f64) -> f64 {
        let n_test = X_test.nrows();
        let mut total_log_likelihood = 0.0;

        for i in 0..n_test {
            let x_test = X_test.row(i);
            let mut density = 0.0;

            for j in 0..X_train.nrows() {
                let x_train = X_train.row(j);
                let dist_sq = self.squared_distance(&x_test, &x_train);
                density += (-gamma * dist_sq).exp();
            }

            density /= X_train.nrows() as f64;

            if density > 1e-300 {
                total_log_likelihood += density.ln();
            } else {
                total_log_likelihood += -300.0; // Penalize very low density
            }
        }

        total_log_likelihood / n_test as f64
    }

    /// Compute LSCV score
    fn compute_lscv_score(&self, X_train: &Array2<f64>, X_test: &Array2<f64>, gamma: f64) -> f64 {
        // Simplified LSCV: negative of integrated squared error
        let K_train = self.compute_rbf_kernel(X_train, gamma);
        let K_test = self.compute_rbf_kernel(X_test, gamma);

        // Approximate integrated squared error
        let train_sum: f64 = K_train.iter().map(|&x| x * x).sum();
        let test_sum: f64 = K_test.iter().map(|&x| x * x).sum();

        -(train_sum + test_sum)
    }

    /// Cross-validation score
    fn cross_validate(&self, X: &Array2<f64>, gamma: f64, n_folds: usize) -> BandwidthResult<f64> {
        let n_samples = x.nrows();
        let fold_size = n_samples / n_folds;
        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start = fold * fold_size;
            let end = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let (X_train, X_test) = self.split_fold(x, start, end);
            let score = self.compute_likelihood(&X_train, &X_test, gamma);
            scores.push(score);
        }

        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Compute RBF kernel matrix
    fn compute_rbf_kernel(&self, X: &Array2<f64>, gamma: f64) -> Array2<f64> {
        let n = x.nrows();
        let mut K = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let dist_sq = self.squared_distance(&x.row(i), &x.row(j));
                K[[i, j]] = (-gamma * dist_sq).exp();
            }
        }

        K
    }

    /// Kernel alignment score
    fn kernel_alignment_score(&self, K: &Array2<f64>, K_ideal: &Array2<f64>) -> f64 {
        // Compute <K, K_ideal> / (||K|| * ||K_ideal||)
        let mut inner_product = 0.0;
        let mut norm_k = 0.0;
        let mut norm_ideal = 0.0;

        for i in 0..K.nrows() {
            for j in 0..K.ncols() {
                inner_product += K[[i, j]] * K_ideal[[i, j]];
                norm_k += K[[i, j]] * K[[i, j]];
                norm_ideal += K_ideal[[i, j]] * K_ideal[[i, j]];
            }
        }

        if norm_k > 0.0 && norm_ideal > 0.0 {
            inner_product / (norm_k.sqrt() * norm_ideal.sqrt())
        } else {
            0.0
        }
    }

    /// Squared Euclidean distance
    fn squared_distance(
        &self,
        x1: &scirs2_core::ndarray::ArrayView1<f64>,
        x2: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum()
    }
}

/// Convenience functions

/// Select bandwidth using median heuristic
pub fn median_heuristic(X: &Array2<f64>) -> BandwidthResult<f64> {
    KernelBandwidthSelector::with_method(BandwidthMethod::MedianHeuristic)
        .select(X)
        .map(|result| result.gamma)
}

/// Select bandwidth using Scott's rule
pub fn scott_rule(X: &Array2<f64>) -> BandwidthResult<f64> {
    KernelBandwidthSelector::with_method(BandwidthMethod::ScottRule)
        .select(X)
        .map(|result| result.gamma)
}

/// Select bandwidth using Silverman's rule
pub fn silverman_rule(X: &Array2<f64>) -> BandwidthResult<f64> {
    KernelBandwidthSelector::with_method(BandwidthMethod::SilvermanRule)
        .select(X)
        .map(|result| result.gamma)
}

/// Select bandwidth using cross-validation
pub fn cross_validated_bandwidth(X: &Array2<f64>, n_folds: usize) -> BandwidthResult<f64> {
    KernelBandwidthSelector::with_method(BandwidthMethod::GridSearchCV {
        n_folds,
        n_points: 20,
    })
    .select(X)
    .map(|result| result.gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_median_heuristic() {
        // Simple 2D dataset
        let X = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        )
        .unwrap();

        let result = median_heuristic(&x).unwrap();
        assert!(result > 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_scott_rule() {
        let X = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();

        let result = scott_rule(&x).unwrap();
        assert!(result > 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_silverman_rule() {
        let X = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();

        let result = silverman_rule(&x).unwrap();
        assert!(result > 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_bandwidth_selector() {
        let selector = KernelBandwidthSelector::with_method(BandwidthMethod::MedianHeuristic);

        let X = Array2::from_shape_vec((20, 3), (0..60).map(|x| x as f64).collect()).unwrap();

        let selection = selector.select(&x).unwrap();

        assert!(selection.gamma > 0.0);
        assert!(selection.gamma.is_finite());
        assert_eq!(selection.method, BandwidthMethod::MedianHeuristic);
    }

    #[test]
    fn test_insufficient_data() {
        let X = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();

        let result = median_heuristic(&x);
        assert!(result.is_err());
    }
}
