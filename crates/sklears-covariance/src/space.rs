//! SPACE (Sparse Partial Correlation Estimation)
//!
//! This module implements SPACE, a method for estimating sparse precision
//! matrices using adaptive thresholding of partial correlations with
//! iterative refinement and cross-validation for threshold selection.

use crate::empirical::EmpiricalCovariance;
use crate::utils::{matrix_determinant, matrix_inverse};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Configuration for SPACE estimator
#[derive(Debug, Clone)]
pub struct SPACEConfig {
    /// Initial threshold for partial correlations
    pub threshold: f64,
    /// Whether to store the precision matrix
    pub store_precision: bool,
    /// Whether to assume the data is centered
    pub assume_centered: bool,
    /// Maximum number of iterations for refinement
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to use adaptive thresholding
    pub adaptive: bool,
    /// Cross-validation folds for threshold selection
    pub cv_folds: usize,
    /// Range of thresholds to test (if adaptive)
    pub threshold_range: (f64, f64),
    /// Number of threshold candidates to test
    pub n_thresholds: usize,
}

/// SPACE (Sparse Partial Correlation Estimation) Estimator
///
/// Estimates sparse precision matrices using adaptive thresholding of
/// partial correlations. The method iteratively refines the estimate
/// and can use cross-validation for optimal threshold selection.
///
/// # Parameters
///
/// * `threshold` - Initial threshold for partial correlations (default: 0.1)
/// * `store_precision` - Whether to store the precision matrix (default: true)
/// * `assume_centered` - Whether to assume the data is centered (default: false)
/// * `max_iter` - Maximum number of iterations (default: 100)
/// * `tol` - Convergence tolerance (default: 1e-4)
/// * `adaptive` - Whether to use adaptive thresholding (default: true)
/// * `cv_folds` - Cross-validation folds (default: 5)
/// * `threshold_range` - Range of thresholds to test (default: (0.01, 0.5))
/// * `n_thresholds` - Number of threshold candidates (default: 20)
///
/// # Examples
///
/// ```
/// use sklears_covariance::SPACE;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 3.0]];
///
/// let space = SPACE::new().threshold(0.1);
/// let fitted = space.fit(&X.view(), &()).unwrap();
/// let precision = fitted.get_precision().unwrap();
/// let partial_corr = fitted.get_partial_correlations();
/// ```
#[derive(Debug, Clone)]
pub struct SPACE<S = Untrained> {
    state: S,
    config: SPACEConfig,
}

/// Trained state for SPACE
#[derive(Debug, Clone)]
pub struct SPACETrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The partial correlation matrix
    pub partial_correlations: Array2<f64>,
    /// The final threshold used
    pub final_threshold: f64,
    /// Number of iterations taken for convergence
    pub n_iter: usize,
    /// The sparsity level (fraction of zero elements)
    pub sparsity: f64,
}

impl SPACE<Untrained> {
    /// Create a new SPACE instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            config: SPACEConfig {
                threshold: 0.1,
                store_precision: true,
                assume_centered: false,
                max_iter: 100,
                tol: 1e-4,
                adaptive: true,
                cv_folds: 5,
                threshold_range: (0.01, 0.5),
                n_thresholds: 20,
            },
        }
    }

    /// Set the threshold for partial correlations
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Set whether to store the precision matrix
    pub fn store_precision(mut self, store_precision: bool) -> Self {
        self.config.store_precision = store_precision;
        self
    }

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.config.assume_centered = assume_centered;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set whether to use adaptive thresholding
    pub fn adaptive(mut self, adaptive: bool) -> Self {
        self.config.adaptive = adaptive;
        self
    }

    /// Set the number of cross-validation folds
    pub fn cv_folds(mut self, cv_folds: usize) -> Self {
        self.config.cv_folds = cv_folds;
        self
    }

    /// Set the range of thresholds to test
    pub fn threshold_range(mut self, range: (f64, f64)) -> Self {
        self.config.threshold_range = range;
        self
    }

    /// Set the number of threshold candidates
    pub fn n_thresholds(mut self, n_thresholds: usize) -> Self {
        self.config.n_thresholds = n_thresholds;
        self
    }
}

impl Default for SPACE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SPACE<Untrained> {
    type Config = SPACEConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for SPACE<Untrained> {
    type Fitted = SPACE<SPACETrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        if n_features < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 features".to_string(),
            ));
        }

        // Compute empirical covariance
        let emp_cov = EmpiricalCovariance::new()
            .assume_centered(self.config.assume_centered)
            .store_precision(false)
            .fit(&x, &())?;

        let covariance_emp = emp_cov.get_covariance().clone();
        let location = emp_cov.get_location().clone();

        // Select optimal threshold using cross-validation if adaptive
        let optimal_threshold = if self.config.adaptive {
            self.select_threshold_cv(&x, &covariance_emp)?
        } else {
            self.config.threshold
        };

        // Estimate sparse precision matrix using SPACE algorithm
        let (precision_matrix, partial_corr, n_iter) =
            self.space_algorithm(&covariance_emp, optimal_threshold)?;

        // Compute final covariance matrix
        let covariance = matrix_inverse(&precision_matrix)?;

        // Calculate sparsity
        let total_elements = (n_features * n_features) as f64;
        let zero_elements = precision_matrix
            .iter()
            .filter(|&&x| x.abs() < 1e-10)
            .count() as f64;
        let sparsity = zero_elements / total_elements;

        let precision = if self.config.store_precision {
            Some(precision_matrix)
        } else {
            None
        };

        Ok(SPACE {
            state: SPACETrained {
                covariance,
                precision,
                location,
                partial_correlations: partial_corr,
                final_threshold: optimal_threshold,
                n_iter,
                sparsity,
            },
            config: self.config,
        })
    }
}

impl SPACE<Untrained> {
    fn select_threshold_cv(
        &self,
        x: &ArrayView2<'_, Float>,
        _covariance: &Array2<f64>,
    ) -> SklResult<f64> {
        let (n_samples, _n_features) = x.dim();
        let fold_size = n_samples / self.config.cv_folds;

        // Generate threshold candidates
        let (min_thresh, max_thresh) = self.config.threshold_range;
        let thresholds: Vec<f64> = (0..self.config.n_thresholds)
            .map(|i| {
                let t = i as f64 / (self.config.n_thresholds - 1) as f64;
                min_thresh + t * (max_thresh - min_thresh)
            })
            .collect();

        let mut best_threshold = self.config.threshold;
        let mut best_score = f64::NEG_INFINITY;

        for &threshold in &thresholds {
            let mut cv_score = 0.0;

            // Cross-validation
            for fold in 0..self.config.cv_folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == self.config.cv_folds - 1 {
                    n_samples
                } else {
                    (fold + 1) * fold_size
                };

                // Create train/validation splits
                let mut train_indices = Vec::new();
                let mut val_indices = Vec::new();

                for i in 0..n_samples {
                    if i >= start_idx && i < end_idx {
                        val_indices.push(i);
                    } else {
                        train_indices.push(i);
                    }
                }

                if train_indices.len() < 2 || val_indices.is_empty() {
                    continue;
                }

                // Train on fold
                let score =
                    self.evaluate_threshold_on_fold(x, &train_indices, &val_indices, threshold)?;
                cv_score += score;
            }

            cv_score /= self.config.cv_folds as f64;

            if cv_score > best_score {
                best_score = cv_score;
                best_threshold = threshold;
            }
        }

        Ok(best_threshold)
    }

    fn evaluate_threshold_on_fold(
        &self,
        x: &ArrayView2<'_, Float>,
        train_indices: &[usize],
        val_indices: &[usize],
        threshold: f64,
    ) -> SklResult<f64> {
        let (_n_samples, n_features) = x.dim();

        // Create training data
        let mut x_train = Array2::zeros((train_indices.len(), n_features));
        for (i, &idx) in train_indices.iter().enumerate() {
            for j in 0..n_features {
                x_train[[i, j]] = x[[idx, j]];
            }
        }

        // Compute empirical covariance on training data
        let emp_cov = EmpiricalCovariance::new()
            .assume_centered(self.config.assume_centered)
            .store_precision(false)
            .fit(&x_train.view(), &())?;

        let train_cov = emp_cov.get_covariance();

        // Estimate precision matrix using SPACE
        let (precision_matrix, _, _) = self.space_algorithm(train_cov, threshold)?;

        // Evaluate on validation data
        let mut x_val = Array2::zeros((val_indices.len(), n_features));
        for (i, &idx) in val_indices.iter().enumerate() {
            for j in 0..n_features {
                x_val[[i, j]] = x[[idx, j]];
            }
        }

        // Compute log-likelihood on validation set
        let log_likelihood = self.compute_log_likelihood(&x_val, &precision_matrix)?;

        Ok(log_likelihood)
    }

    fn compute_log_likelihood(&self, x: &Array2<f64>, precision: &Array2<f64>) -> SklResult<f64> {
        let (n_samples, n_features) = x.dim();
        let location = if self.config.assume_centered {
            Array1::zeros(n_features)
        } else {
            x.mean_axis(Axis(0)).unwrap()
        };

        let mut log_likelihood = 0.0;
        let log_det = matrix_determinant(precision).ln();

        for i in 0..n_samples {
            let mut x_centered = Array1::zeros(n_features);
            for j in 0..n_features {
                x_centered[j] = x[[i, j]] - location[j];
            }

            // Compute quadratic form: x^T * Precision * x
            let mut quadratic_form = 0.0;
            for j in 0..n_features {
                for k in 0..n_features {
                    quadratic_form += x_centered[j] * precision[[j, k]] * x_centered[k];
                }
            }

            log_likelihood += 0.5 * log_det - 0.5 * quadratic_form;
        }

        log_likelihood -=
            0.5 * n_samples as f64 * n_features as f64 * (2.0 * std::f64::consts::PI).ln();

        Ok(log_likelihood / n_samples as f64)
    }

    fn space_algorithm(
        &self,
        covariance: &Array2<f64>,
        threshold: f64,
    ) -> SklResult<(Array2<f64>, Array2<f64>, usize)> {
        let n_features = covariance.nrows();

        // Initialize precision matrix as inverse of covariance
        let mut precision = matrix_inverse(covariance)?;

        let mut iter = 0;
        for _ in 0..self.config.max_iter {
            iter += 1;
            let old_precision = precision.clone();

            // Compute partial correlations
            let partial_corr = self.compute_partial_correlations(&precision)?;

            // Apply thresholding to partial correlations
            let mut thresholded_partial_corr = partial_corr.clone();
            for i in 0..n_features {
                for j in 0..n_features {
                    if i != j && thresholded_partial_corr[[i, j]].abs() < threshold {
                        thresholded_partial_corr[[i, j]] = 0.0;
                    }
                }
            }

            // Update precision matrix based on thresholded partial correlations
            precision = self.partial_correlations_to_precision(&thresholded_partial_corr)?;

            // Check convergence
            let mut max_change: f64 = 0.0;
            for i in 0..n_features {
                for j in 0..n_features {
                    max_change = max_change.max((precision[[i, j]] - old_precision[[i, j]]).abs());
                }
            }

            if max_change < self.config.tol {
                break;
            }
        }

        let final_partial_corr = self.compute_partial_correlations(&precision)?;
        Ok((precision, final_partial_corr, iter))
    }

    fn compute_partial_correlations(&self, precision: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_features = precision.nrows();
        let mut partial_corr = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    partial_corr[[i, j]] = 1.0;
                } else {
                    let denom: f64 = (precision[[i, i]] * precision[[j, j]]).sqrt();
                    if denom > 0.0 {
                        partial_corr[[i, j]] = -precision[[i, j]] / denom;
                    }
                }
            }
        }

        Ok(partial_corr)
    }

    fn partial_correlations_to_precision(
        &self,
        partial_corr: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_features = partial_corr.nrows();
        let mut precision = Array2::eye(n_features);

        // Start with identity and iteratively refine
        for _iter in 0..10 {
            // Fixed number of refinement iterations
            let mut new_precision = Array2::zeros((n_features, n_features));

            for i in 0..n_features {
                for j in 0..n_features {
                    if i == j {
                        new_precision[[i, j]] = 1.0; // Will be scaled later
                    } else if partial_corr[[i, j]] != 0.0 {
                        // Use partial correlation to estimate precision element
                        let scale: f64 = ((precision[[i, i]] * precision[[j, j]]) as f64).sqrt();
                        new_precision[[i, j]] = -partial_corr[[i, j]] * scale;
                    }
                }
            }

            // Adjust diagonal elements to maintain positive definiteness
            for i in 0..n_features {
                let mut row_sum = 0.0;
                for j in 0..n_features {
                    if i != j {
                        row_sum += new_precision[[i, j]].abs();
                    }
                }
                new_precision[[i, i]] = row_sum + 0.1; // Diagonal dominance
            }

            precision = new_precision;
        }

        Ok(precision)
    }
}

impl SPACE<SPACETrained> {
    /// Get the covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix (inverse covariance)
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the location (mean)
    pub fn get_location(&self) -> &Array1<f64> {
        &self.state.location
    }

    /// Get the partial correlations matrix
    pub fn get_partial_correlations(&self) -> &Array2<f64> {
        &self.state.partial_correlations
    }

    /// Get the final threshold used
    pub fn get_threshold(&self) -> f64 {
        self.state.final_threshold
    }

    /// Get the number of iterations taken
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the sparsity level
    pub fn get_sparsity(&self) -> f64 {
        self.state.sparsity
    }
}
