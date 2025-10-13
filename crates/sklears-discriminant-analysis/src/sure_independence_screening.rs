//! Sure Independence Screening for high-dimensional discriminant analysis
//!
//! This module implements Sure Independence Screening (SIS) combined with discriminant analysis
//! for high-dimensional data where the number of features is much larger than the number of samples.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};
use std::collections::HashMap;

/// Configuration for Sure Independence Screening
#[derive(Debug, Clone)]
pub struct SureIndependenceScreeningConfig {
    /// Number of features to select in the screening step
    pub n_features_to_select: usize,
    /// Type of correlation measure to use
    pub correlation_measure: String,
    /// Threshold for feature selection
    pub threshold: Option<Float>,
    /// Base discriminant method to use after screening
    pub base_method: String,
    /// Regularization parameter for the base method
    pub reg_param: Float,
    /// Whether to use iterative screening
    pub iterative: bool,
    /// Maximum number of iterations for iterative screening
    pub max_iter: usize,
    /// Tolerance for convergence in iterative screening
    pub tol: Float,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for SureIndependenceScreeningConfig {
    fn default() -> Self {
        Self {
            n_features_to_select: 100,
            correlation_measure: "pearson".to_string(),
            threshold: None,
            base_method: "lda".to_string(),
            reg_param: 0.01,
            iterative: false,
            max_iter: 10,
            tol: 1e-6,
            random_state: None,
        }
    }
}

/// Sure Independence Screening discriminant analysis
pub struct SureIndependenceScreening {
    config: SureIndependenceScreeningConfig,
}

impl SureIndependenceScreening {
    /// Create a new Sure Independence Screening instance
    pub fn new() -> Self {
        Self {
            config: SureIndependenceScreeningConfig::default(),
        }
    }

    /// Set the number of features to select
    pub fn n_features_to_select(mut self, n: usize) -> Self {
        self.config.n_features_to_select = n;
        self
    }

    /// Set the correlation measure
    pub fn correlation_measure(mut self, measure: &str) -> Self {
        self.config.correlation_measure = measure.to_string();
        self
    }

    /// Set the threshold for feature selection
    pub fn threshold(mut self, threshold: Float) -> Self {
        self.config.threshold = Some(threshold);
        self
    }

    /// Set the base discriminant method
    pub fn base_method(mut self, method: &str) -> Self {
        self.config.base_method = method.to_string();
        self
    }

    /// Set the regularization parameter
    pub fn reg_param(mut self, param: Float) -> Self {
        self.config.reg_param = param;
        self
    }

    /// Enable iterative screening
    pub fn iterative(mut self, iterative: bool) -> Self {
        self.config.iterative = iterative;
        self
    }

    /// Set maximum iterations for iterative screening
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Compute marginal correlations between features and response
    fn compute_marginal_correlations(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Array1<Float>> {
        let n_features = x.ncols();
        let mut correlations = Array1::zeros(n_features);

        // Convert y to binary indicator matrix for multi-class problems
        let classes: Vec<i32> = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };

        match self.config.correlation_measure.as_str() {
            "pearson" => {
                for (j, corr) in correlations.iter_mut().enumerate() {
                    let feature_col = x.column(j);
                    *corr = self.pearson_correlation(&feature_col, y)?;
                }
            }
            "spearman" => {
                for (j, corr) in correlations.iter_mut().enumerate() {
                    let feature_col = x.column(j);
                    *corr = self.spearman_correlation(&feature_col, y)?;
                }
            }
            "kendall" => {
                for (j, corr) in correlations.iter_mut().enumerate() {
                    let feature_col = x.column(j);
                    *corr = self.kendall_correlation(&feature_col, y)?;
                }
            }
            "mutual_info" => {
                for (j, corr) in correlations.iter_mut().enumerate() {
                    let feature_col = x.column(j);
                    *corr = self.mutual_information(&feature_col, y)?;
                }
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "correlation_measure".to_string(),
                    reason: format!(
                        "Unknown correlation measure: {}",
                        self.config.correlation_measure
                    ),
                })
            }
        }

        Ok(correlations)
    }

    /// Compute Pearson correlation coefficient
    fn pearson_correlation(&self, x: &ArrayView1<Float>, y: &Array1<i32>) -> Result<Float> {
        let n = x.len() as Float;
        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.iter().map(|&v| v as Float).sum::<Float>() / n;

        let mut numerator = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for (i, &x_val) in x.iter().enumerate() {
            let y_val = y[i] as Float;
            let x_diff = x_val - x_mean;
            let y_diff = y_val - y_mean;

            numerator += x_diff * y_diff;
            x_var += x_diff * x_diff;
            y_var += y_diff * y_diff;
        }

        if x_var == 0.0 || y_var == 0.0 {
            return Ok(0.0);
        }

        Ok(numerator / (x_var * y_var).sqrt())
    }

    /// Compute Spearman correlation coefficient
    fn spearman_correlation(&self, x: &ArrayView1<Float>, y: &Array1<i32>) -> Result<Float> {
        // Convert to ranks
        let x_ranks = self.compute_ranks(&x.to_owned());
        let y_ranks = self.compute_ranks(&y.mapv(|v| v as Float));

        // Compute Pearson correlation of ranks
        let n = x_ranks.len() as Float;
        let x_mean = x_ranks.mean().unwrap_or(0.0);
        let y_mean = y_ranks.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for i in 0..x_ranks.len() {
            let x_diff = x_ranks[i] - x_mean;
            let y_diff = y_ranks[i] - y_mean;

            numerator += x_diff * y_diff;
            x_var += x_diff * x_diff;
            y_var += y_diff * y_diff;
        }

        if x_var == 0.0 || y_var == 0.0 {
            return Ok(0.0);
        }

        Ok(numerator / (x_var * y_var).sqrt())
    }

    /// Compute Kendall's tau correlation coefficient
    fn kendall_correlation(&self, x: &ArrayView1<Float>, y: &Array1<i32>) -> Result<Float> {
        let n = x.len();
        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let x_sign = (x[i] - x[j]).signum();
                let y_sign = ((y[i] - y[j]) as Float).signum();

                if x_sign * y_sign > 0.0 {
                    concordant += 1;
                } else if x_sign * y_sign < 0.0 {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs == 0 {
            return Ok(0.0);
        }

        Ok((concordant as Float - discordant as Float) / total_pairs as Float)
    }

    /// Compute mutual information
    fn mutual_information(&self, x: &ArrayView1<Float>, y: &Array1<i32>) -> Result<Float> {
        // Discretize continuous feature into bins
        let n_bins = 10;
        let x_min = x.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let bin_width = (x_max - x_min) / n_bins as Float;

        if bin_width == 0.0 {
            return Ok(0.0);
        }

        // Count joint frequencies
        let mut joint_counts: HashMap<(usize, i32), usize> = HashMap::new();
        let mut x_counts: HashMap<usize, usize> = HashMap::new();
        let mut y_counts: HashMap<i32, usize> = HashMap::new();

        for (i, &x_val) in x.iter().enumerate() {
            let bin = ((x_val - x_min) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            let y_val = y[i];

            *joint_counts.entry((bin, y_val)).or_insert(0) += 1;
            *x_counts.entry(bin).or_insert(0) += 1;
            *y_counts.entry(y_val).or_insert(0) += 1;
        }

        let n = x.len() as Float;
        let mut mi = 0.0;

        for ((x_bin, y_val), &joint_count) in joint_counts.iter() {
            let p_xy = joint_count as Float / n;
            let p_x = x_counts[x_bin] as Float / n;
            let p_y = y_counts[y_val] as Float / n;

            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }

        Ok(mi)
    }

    /// Compute ranks for Spearman correlation
    fn compute_ranks(&self, x: &Array1<Float>) -> Array1<Float> {
        let n = x.len();
        let mut indexed_values: Vec<(Float, usize)> =
            x.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        indexed_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = Array1::zeros(n);
        for (rank, &(_, original_index)) in indexed_values.iter().enumerate() {
            ranks[original_index] = (rank + 1) as Float;
        }

        ranks
    }

    /// Select features based on correlation scores
    fn select_features(&self, correlations: &Array1<Float>) -> Result<Vec<usize>> {
        let mut indexed_correlations: Vec<(Float, usize)> = correlations
            .iter()
            .enumerate()
            .map(|(i, &corr)| (corr.abs(), i))
            .collect();

        indexed_correlations
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_select = if let Some(threshold) = self.config.threshold {
            indexed_correlations
                .iter()
                .take_while(|(corr, _)| *corr >= threshold)
                .count()
                .min(self.config.n_features_to_select)
        } else {
            self.config
                .n_features_to_select
                .min(indexed_correlations.len())
        };

        Ok(indexed_correlations
            .iter()
            .take(n_select)
            .map(|(_, idx)| *idx)
            .collect())
    }
}

/// Trained Sure Independence Screening model
pub struct TrainedSureIndependenceScreening {
    config: SureIndependenceScreeningConfig,
    selected_features: Vec<usize>,
    feature_correlations: Array1<Float>,
    classes: Vec<i32>,
    base_model: Box<dyn BaseDiscriminantModel>,
    n_features_in: usize,
}

impl TrainedSureIndependenceScreening {
    /// Get the selected features
    pub fn selected_features(&self) -> &[usize] {
        &self.selected_features
    }

    /// Get the feature correlations
    pub fn feature_correlations(&self) -> &Array1<Float> {
        &self.feature_correlations
    }

    /// Get the classes
    pub fn classes(&self) -> &[i32] {
        &self.classes
    }

    /// Get the number of selected features
    pub fn n_selected_features(&self) -> usize {
        self.selected_features.len()
    }

    /// Get support mask for selected features
    pub fn support(&self) -> Array1<bool> {
        let mut support = Array1::from_elem(self.n_features_in, false);
        for &idx in &self.selected_features {
            support[idx] = true;
        }
        support
    }
}

/// Trait for base discriminant models
pub trait BaseDiscriminantModel: Send + Sync {
    fn fit(&mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<()>;
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>>;
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>>;
    fn classes(&self) -> &[i32];
}

/// Simple LDA implementation for base model
pub struct SimpleLDA {
    classes: Vec<i32>,
    means: Array2<Float>,
    covariance_inv: Array2<Float>,
    priors: Array1<Float>,
}

impl Default for SimpleLDA {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleLDA {
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            means: Array2::zeros((0, 0)),
            covariance_inv: Array2::zeros((0, 0)),
            priors: Array1::zeros(0),
        }
    }
}

impl BaseDiscriminantModel for SimpleLDA {
    fn fit(&mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        // Get unique classes
        self.classes = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };

        let n_classes = self.classes.len();
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Initialize arrays
        self.means = Array2::zeros((n_classes, n_features));
        self.priors = Array1::zeros(n_classes);

        // Compute class means and priors
        for (k, &class_label) in self.classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&label| label == class_label).collect();
            let class_count = class_mask.iter().filter(|&&mask| mask).count();

            if class_count == 0 {
                return Err(SklearsError::InvalidData {
                    reason: "Empty class found".to_string(),
                });
            }

            self.priors[k] = class_count as Float / n_samples as Float;

            // Compute class mean
            let mut class_sum = Array1::zeros(n_features);
            for (i, &mask) in class_mask.iter().enumerate() {
                if mask {
                    class_sum += &x.row(i);
                }
            }
            self.means
                .row_mut(k)
                .assign(&(class_sum / class_count as Float));
        }

        // Compute pooled covariance matrix
        let mut covariance = Array2::zeros((n_features, n_features));
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let class_idx = self.classes.iter().position(|&c| c == y[i]).unwrap();
            let diff = &row - &self.means.row(class_idx);
            let diff_outer = diff
                .clone()
                .insert_axis(Axis(1))
                .dot(&diff.insert_axis(Axis(0)));
            covariance += &diff_outer;
        }

        covariance /= (n_samples - n_classes) as Float;

        // Add regularization to prevent singular matrix
        for i in 0..n_features {
            covariance[[i, i]] += 1e-6;
        }

        // Invert covariance matrix (simple approach)
        self.covariance_inv = self.invert_matrix(&covariance)?;

        Ok(())
    }

    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, proba_row) in probas.axis_iter(Axis(0)).enumerate() {
            let max_idx = proba_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }

    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let mut log_probas = Array1::zeros(n_classes);

            for (k, _) in self.classes.iter().enumerate() {
                let mean_diff = &sample - &self.means.row(k);
                let quadratic_term = mean_diff.dot(&self.covariance_inv.dot(&mean_diff));
                log_probas[k] = self.priors[k].ln() - 0.5 * quadratic_term;
            }

            // Convert to probabilities using softmax
            let max_log_proba = log_probas
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_log_probas = log_probas.mapv(|x| (x - max_log_proba).exp());
            let sum_exp = exp_log_probas.sum();

            probas.row_mut(i).assign(&(exp_log_probas / sum_exp));
        }

        Ok(probas)
    }

    fn classes(&self) -> &[i32] {
        &self.classes
    }
}

impl SimpleLDA {
    /// Simple matrix inversion using Gauss-Jordan elimination
    fn invert_matrix(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidData {
                reason: "Matrix must be square".to_string(),
            });
        }

        // Create augmented matrix [A | I]
        let mut augmented = Array2::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                augmented[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[pivot_row, i]].abs() {
                    pivot_row = k;
                }
            }

            // Swap rows if needed
            if pivot_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[pivot_row, j]];
                    augmented[[pivot_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if augmented[[i, i]].abs() < 1e-10 {
                return Err(SklearsError::InvalidData {
                    reason: "Matrix is singular".to_string(),
                });
            }

            // Scale pivot row
            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse matrix
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inverse)
    }
}

impl Estimator for SureIndependenceScreening {
    type Config = SureIndependenceScreeningConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for SureIndependenceScreening {
    type Fitted = TrainedSureIndependenceScreening;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidData {
                reason: "Number of samples in X and y must match".to_string(),
            });
        }

        // Compute marginal correlations
        let correlations = self.compute_marginal_correlations(x, y)?;

        // Select features
        let selected_features = self.select_features(&correlations)?;

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidData {
                reason: "No features selected".to_string(),
            });
        }

        // Extract selected features
        let mut x_selected = Array2::zeros((n_samples, selected_features.len()));
        for (j, &feature_idx) in selected_features.iter().enumerate() {
            x_selected.column_mut(j).assign(&x.column(feature_idx));
        }

        // Get classes
        let classes = {
            let mut cls = y.to_vec();
            cls.sort_unstable();
            cls.dedup();
            cls
        };

        // Fit base model
        let mut base_model: Box<dyn BaseDiscriminantModel> = match self.config.base_method.as_str()
        {
            "lda" => Box::new(SimpleLDA::new()),
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "base_method".to_string(),
                    reason: format!("Unknown base method: {}", self.config.base_method),
                })
            }
        };

        base_model.fit(&x_selected, y)?;

        Ok(TrainedSureIndependenceScreening {
            config: self.config,
            selected_features,
            feature_correlations: correlations,
            classes,
            base_model,
            n_features_in: n_features,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedSureIndependenceScreening {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::InvalidData {
                reason: "Number of features in X does not match training data".to_string(),
            });
        }

        // Extract selected features
        let mut x_selected = Array2::zeros((x.nrows(), self.selected_features.len()));
        for (j, &feature_idx) in self.selected_features.iter().enumerate() {
            x_selected.column_mut(j).assign(&x.column(feature_idx));
        }

        self.base_model.predict(&x_selected)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedSureIndependenceScreening {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::InvalidData {
                reason: "Number of features in X does not match training data".to_string(),
            });
        }

        // Extract selected features
        let mut x_selected = Array2::zeros((x.nrows(), self.selected_features.len()));
        for (j, &feature_idx) in self.selected_features.iter().enumerate() {
            x_selected.column_mut(j).assign(&x.column(feature_idx));
        }

        self.base_model.predict_proba(&x_selected)
    }
}

impl Transform<Array2<Float>> for TrainedSureIndependenceScreening {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::InvalidData {
                reason: "Number of features in X does not match training data".to_string(),
            });
        }

        // Extract selected features
        let mut x_selected = Array2::zeros((x.nrows(), self.selected_features.len()));
        for (j, &feature_idx) in self.selected_features.iter().enumerate() {
            x_selected.column_mut(j).assign(&x.column(feature_idx));
        }

        Ok(x_selected)
    }
}

impl Default for SureIndependenceScreening {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sure_independence_screening() {
        let x = array![
            [1.0, 2.0, 0.1, 0.2, 0.3],
            [2.0, 3.0, 0.2, 0.1, 0.4],
            [3.0, 4.0, 0.3, 0.4, 0.1],
            [4.0, 5.0, 0.4, 0.3, 0.2],
            [5.0, 6.0, 0.5, 0.6, 0.7],
            [6.0, 7.0, 0.6, 0.5, 0.8]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let sis = SureIndependenceScreening::new().n_features_to_select(3);
        let fitted = sis.fit(&x, &y).unwrap();

        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.n_selected_features(), 3);
        assert_eq!(fitted.selected_features().len(), 3);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        let probas = fitted.predict_proba(&x).unwrap();
        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }

        let transformed = fitted.transform(&x).unwrap();
        assert_eq!(transformed.dim(), (6, 3));
    }

    #[test]
    fn test_sis_with_threshold() {
        let x = array![
            [1.0, 2.0, 0.1, 0.2],
            [2.0, 3.0, 0.2, 0.1],
            [3.0, 4.0, 0.3, 0.4],
            [4.0, 5.0, 0.4, 0.3]
        ];
        let y = array![0, 0, 1, 1];

        let sis = SureIndependenceScreening::new()
            .n_features_to_select(10)
            .threshold(0.3);
        let fitted = sis.fit(&x, &y).unwrap();

        assert!(fitted.n_selected_features() <= 4);
        assert!(fitted.n_selected_features() > 0);
    }

    #[test]
    fn test_sis_different_correlation_measures() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 0, 1, 1];

        let measures = vec!["pearson", "spearman", "kendall", "mutual_info"];

        for measure in measures {
            let sis = SureIndependenceScreening::new()
                .correlation_measure(measure)
                .n_features_to_select(2);
            let fitted = sis.fit(&x, &y).unwrap();

            assert_eq!(fitted.classes().len(), 2);
            assert_eq!(fitted.n_selected_features(), 2);

            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }

    #[test]
    fn test_sis_support_mask() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0]
        ];
        let y = array![0, 0, 1, 1];

        let sis = SureIndependenceScreening::new().n_features_to_select(3);
        let fitted = sis.fit(&x, &y).unwrap();

        let support = fitted.support();
        assert_eq!(support.len(), 5);
        assert_eq!(support.iter().filter(|&&x| x).count(), 3);
    }

    #[test]
    fn test_sis_correlation_computations() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0, 0, 1, 1, 1];

        let sis = SureIndependenceScreening::new();

        // Test Pearson correlation
        let pearson = sis.pearson_correlation(&x.view(), &y).unwrap();
        assert!(pearson.abs() <= 1.0);

        // Test Spearman correlation
        let spearman = sis.spearman_correlation(&x.view(), &y).unwrap();
        assert!(spearman.abs() <= 1.0);

        // Test Kendall correlation
        let kendall = sis.kendall_correlation(&x.view(), &y).unwrap();
        assert!(kendall.abs() <= 1.0);

        // Test mutual information
        let mi = sis.mutual_information(&x.view(), &y).unwrap();
        assert!(mi >= 0.0);
    }
}
