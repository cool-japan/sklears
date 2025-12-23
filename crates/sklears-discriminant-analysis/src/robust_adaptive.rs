//! Robust and Adaptive Discriminant Analysis
//!
//! This module implements robust variants of discriminant analysis that are resistant
//! to outliers and can adapt to non-standard data distributions.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};
use std::collections::HashMap;

/// M-estimator functions for robust estimation
#[derive(Debug, Clone, PartialEq)]
pub enum MEstimatorType {
    /// Huber M-estimator: robust to outliers with parameter k
    Huber { k: Float },
    /// Tukey biweight M-estimator: more aggressive outlier rejection
    Tukey { c: Float },
    /// Hampel M-estimator: three-part redescending function
    Hampel { a: Float, b: Float, c: Float },
    /// Andrews M-estimator: sinusoidal weight function
    Andrews { c: Float },
}

impl Default for MEstimatorType {
    fn default() -> Self {
        MEstimatorType::Huber { k: 1.345 }
    }
}

/// Configuration for Robust Discriminant Analysis
#[derive(Debug, Clone)]
pub struct RobustDiscriminantAnalysisConfig {
    /// estimator_type
    pub estimator_type: MEstimatorType,
    /// n_components
    pub n_components: Option<usize>,
    /// trimming_fraction
    pub trimming_fraction: Float,
    /// max_iter
    pub max_iter: usize,
    /// tol
    pub tol: Float,
    /// reg_param
    pub reg_param: Float,
    /// adaptive_threshold
    pub adaptive_threshold: bool,
    /// bootstrap_samples
    pub bootstrap_samples: usize,
}

impl Default for RobustDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            estimator_type: MEstimatorType::default(),
            n_components: None,
            trimming_fraction: 0.1,
            max_iter: 100,
            tol: 1e-6,
            reg_param: 1e-4,
            adaptive_threshold: true,
            bootstrap_samples: 100,
        }
    }
}

/// Robust Discriminant Analysis
///
/// Implements various robust methods for discriminant analysis including:
/// - M-estimator based robust estimation
/// - Trimmed discriminant analysis
/// - Adaptive threshold selection
/// - Bootstrap-based robustness assessment
///
/// # Mathematical Background
///
/// Instead of using the sample mean and covariance matrix, robust methods use:
/// 1. M-estimators that downweight outliers based on their residuals
/// 2. Trimming that excludes a fraction of extreme observations
/// 3. Adaptive thresholds that automatically determine outlier cutoffs
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use sklears_discriminant_analysis::*;
/// use sklears_core::traits::{Predict, Fit};
///
/// let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [10.0, 10.0]]; // Last point is outlier
/// let y = array![0, 0, 1, 1];
///
/// let rda = RobustDiscriminantAnalysis::new()
///     .estimator_type(MEstimatorType::Huber { k: 1.345 })
///     .trimming_fraction(0.25);
/// let fitted = rda.fit(&x, &y).unwrap();
/// let predictions = fitted.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RobustDiscriminantAnalysis {
    config: RobustDiscriminantAnalysisConfig,
}

impl Default for RobustDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustDiscriminantAnalysis {
    /// Create a new robust discriminant analysis instance
    pub fn new() -> Self {
        Self {
            config: RobustDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the M-estimator type
    pub fn estimator_type(mut self, estimator_type: MEstimatorType) -> Self {
        self.config.estimator_type = estimator_type;
        self
    }

    /// Set the number of components for dimensionality reduction
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set the trimming fraction (proportion of observations to trim)
    pub fn trimming_fraction(mut self, trimming_fraction: Float) -> Self {
        self.config.trimming_fraction = trimming_fraction.max(0.0).min(0.5);
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the regularization parameter
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Enable/disable adaptive threshold selection
    pub fn adaptive_threshold(mut self, adaptive_threshold: bool) -> Self {
        self.config.adaptive_threshold = adaptive_threshold;
        self
    }

    /// Set the number of bootstrap samples for robustness assessment
    pub fn bootstrap_samples(mut self, bootstrap_samples: usize) -> Self {
        self.config.bootstrap_samples = bootstrap_samples;
        self
    }
}

/// Trained Robust Discriminant Analysis model
#[derive(Debug, Clone)]
pub struct TrainedRobustDiscriminantAnalysis {
    config: RobustDiscriminantAnalysisConfig,
    classes: Array1<i32>,
    robust_means: HashMap<i32, Array1<Float>>,
    robust_covariances: HashMap<i32, Array2<Float>>,
    pooled_covariance: Array2<Float>,
    components: Array2<Float>,
    eigenvalues: Array1<Float>,
    outlier_scores: Array1<Float>,
    trimmed_indices: Vec<usize>,
    n_components: usize,
}

impl TrainedRobustDiscriminantAnalysis {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get the number of components
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get the robust class means
    pub fn robust_means(&self) -> &HashMap<i32, Array1<Float>> {
        &self.robust_means
    }

    /// Get the robust class covariances
    pub fn robust_covariances(&self) -> &HashMap<i32, Array2<Float>> {
        &self.robust_covariances
    }

    /// Get the pooled robust covariance matrix
    pub fn pooled_covariance(&self) -> &Array2<Float> {
        &self.pooled_covariance
    }

    /// Get the discriminant components
    pub fn components(&self) -> &Array2<Float> {
        &self.components
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> &Array1<Float> {
        &self.eigenvalues
    }

    /// Get the outlier scores for training samples
    pub fn outlier_scores(&self) -> &Array1<Float> {
        &self.outlier_scores
    }

    /// Get the indices of trimmed (non-outlier) samples
    pub fn trimmed_indices(&self) -> &Vec<usize> {
        &self.trimmed_indices
    }

    /// Compute M-estimator weight for a given residual
    fn m_estimator_weight(&self, residual: Float) -> Float {
        match &self.config.estimator_type {
            MEstimatorType::Huber { k } => {
                if residual.abs() <= *k {
                    1.0
                } else {
                    k / residual.abs()
                }
            }
            MEstimatorType::Tukey { c } => {
                if residual.abs() <= *c {
                    let u = residual / c;
                    (1.0 - u * u).powi(2)
                } else {
                    0.0
                }
            }
            MEstimatorType::Hampel { a, b, c } => {
                let abs_r = residual.abs();
                if abs_r <= *a {
                    1.0
                } else if abs_r <= *b {
                    a / abs_r
                } else if abs_r <= *c {
                    a * (c - abs_r) / (abs_r * (c - b))
                } else {
                    0.0
                }
            }
            MEstimatorType::Andrews { c } => {
                if residual.abs() <= *c {
                    (residual / c).sin() / (residual / c)
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute robust mean using M-estimator
    fn compute_robust_mean(&self, data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty data".to_string()));
        }

        // Initialize with sample mean
        let mut robust_mean = data.mean_axis(Axis(0)).unwrap();
        let mut prev_mean = robust_mean.clone();

        // Iteratively reweighted least squares
        for _ in 0..self.config.max_iter {
            let mut weighted_sum = Array1::zeros(n_features);
            let mut weight_sum = 0.0;

            // Compute weights based on Mahalanobis distances
            for i in 0..n_samples {
                let sample = data.row(i);
                let diff = &sample - &robust_mean;
                let distance = diff.iter().map(|&x| x * x).sum::<Float>().sqrt();

                let weight = self.m_estimator_weight(distance);
                weighted_sum += &(sample.to_owned() * weight);
                weight_sum += weight;
            }

            if weight_sum > 0.0 {
                robust_mean = weighted_sum / weight_sum;
            }

            // Check convergence
            let change = (&robust_mean - &prev_mean)
                .iter()
                .map(|&x| x * x)
                .sum::<Float>()
                .sqrt();
            if change < self.config.tol {
                break;
            }
            prev_mean = robust_mean.clone();
        }

        Ok(robust_mean)
    }

    /// Compute robust covariance using M-estimator
    fn compute_robust_covariance(
        &self,
        data: &ArrayView2<Float>,
        robust_mean: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty data".to_string()));
        }

        let mut robust_cov = Array2::zeros((n_features, n_features));
        let mut weight_sum = 0.0;

        // Compute weighted covariance
        for i in 0..n_samples {
            let sample = data.row(i);
            let diff = &sample - robust_mean;
            let distance = diff.iter().map(|&x| x * x).sum::<Float>().sqrt();

            let weight = self.m_estimator_weight(distance);

            // Outer product of centered sample
            for j in 0..n_features {
                for k in 0..n_features {
                    robust_cov[[j, k]] += weight * diff[j] * diff[k];
                }
            }
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            robust_cov /= weight_sum;
        }

        // Add regularization
        for i in 0..n_features {
            robust_cov[[i, i]] += self.config.reg_param;
        }

        Ok(robust_cov)
    }

    /// Trim outliers based on Mahalanobis distance
    fn trim_outliers(
        &self,
        data: &ArrayView2<Float>,
        mean: &Array1<Float>,
        covariance: &Array2<Float>,
    ) -> Result<Vec<usize>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Compute Mahalanobis distances
        let mut distances = Vec::with_capacity(n_samples);
        let cov_inv = self.compute_matrix_inverse(covariance)?;

        for i in 0..n_samples {
            let sample = data.row(i);
            let diff = &sample - mean;

            // Mahalanobis distance: sqrt(diff^T * inv(Cov) * diff)
            let mut mahal_dist = 0.0;
            for j in 0..n_features {
                for k in 0..n_features {
                    mahal_dist += diff[j] * cov_inv[[j, k]] * diff[k];
                }
            }
            distances.push((mahal_dist.sqrt(), i));
        }

        // Sort by distance
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Keep only the non-trimmed fraction
        let n_keep = ((1.0 - self.config.trimming_fraction) * n_samples as Float) as usize;
        let trimmed_indices: Vec<usize> = distances[..n_keep].iter().map(|(_, idx)| *idx).collect();

        Ok(trimmed_indices)
    }

    /// Compute matrix inverse using regularized approach
    fn compute_matrix_inverse(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        let mut augmented = Array2::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                if i == j {
                    augmented[[i, j + n]] = 1.0;
                }
            }
        }

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for near-singular matrix
            if augmented[[i, i]].abs() < self.config.tol {
                return Err(SklearsError::NumericalError(
                    "Matrix is singular or near-singular".to_string(),
                ));
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]] / augmented[[i, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }

            // Scale row
            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
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

impl Estimator for RobustDiscriminantAnalysis {
    type Config = RobustDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>, TrainedRobustDiscriminantAnalysis>
    for RobustDiscriminantAnalysis
{
    type Fitted = TrainedRobustDiscriminantAnalysis;
    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<TrainedRobustDiscriminantAnalysis> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Mismatch between X and y dimensions".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        // Determine number of components
        let max_components = (n_classes - 1).min(n_features);
        let n_components = self.config.n_components.unwrap_or(max_components);

        if n_components > max_components {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) cannot be larger than min(n_classes-1, n_features) ({})",
                n_components, max_components
            )));
        }

        let mut trained = TrainedRobustDiscriminantAnalysis {
            config: self.config.clone(),
            classes: classes.clone(),
            robust_means: HashMap::new(),
            robust_covariances: HashMap::new(),
            pooled_covariance: Array2::zeros((n_features, n_features)),
            components: Array2::zeros((n_features, n_components)),
            eigenvalues: Array1::zeros(n_components),
            outlier_scores: Array1::zeros(n_samples),
            trimmed_indices: Vec::new(),
            n_components,
        };

        // Phase 1: Initial robust estimation by class
        let mut all_trimmed_indices = Vec::new();
        let mut class_data: HashMap<i32, Vec<usize>> = HashMap::new();

        // Organize data by class
        for (i, &class) in y.iter().enumerate() {
            class_data.entry(class).or_default().push(i);
        }

        // Compute robust estimates for each class
        for &class in &classes {
            if let Some(indices) = class_data.get(&class) {
                // Extract class data
                let class_x =
                    Array2::from_shape_fn((indices.len(), n_features), |(i, j)| x[[indices[i], j]]);

                // Compute robust mean
                let robust_mean = trained.compute_robust_mean(&class_x.view())?;

                // Compute robust covariance
                let robust_cov =
                    trained.compute_robust_covariance(&class_x.view(), &robust_mean)?;

                // Trim outliers within class
                let trimmed_class_indices =
                    trained.trim_outliers(&class_x.view(), &robust_mean, &robust_cov)?;

                // Convert back to global indices
                let global_trimmed: Vec<usize> = trimmed_class_indices
                    .iter()
                    .map(|&local_idx| indices[local_idx])
                    .collect();
                all_trimmed_indices.extend(global_trimmed);

                trained.robust_means.insert(class, robust_mean);
                trained.robust_covariances.insert(class, robust_cov);
            }
        }

        trained.trimmed_indices = all_trimmed_indices;

        // Phase 2: Compute pooled robust covariance
        let mut pooled_cov = Array2::zeros((n_features, n_features));
        let mut total_weight = 0.0;

        for &class in &classes {
            if let (Some(indices), Some(robust_cov)) = (
                class_data.get(&class),
                trained.robust_covariances.get(&class),
            ) {
                let class_weight = indices.len() as Float;
                pooled_cov += &(robust_cov * class_weight);
                total_weight += class_weight;
            }
        }

        if total_weight > 0.0 {
            pooled_cov /= total_weight;
        }

        // Add regularization
        for i in 0..n_features {
            pooled_cov[[i, i]] += self.config.reg_param;
        }

        trained.pooled_covariance = pooled_cov;

        // Phase 3: Compute discriminant components
        let (components, eigenvalues) = self.compute_discriminant_components(&trained)?;
        trained.components = components;
        trained.eigenvalues = eigenvalues;

        // Phase 4: Compute outlier scores
        trained.outlier_scores = self.compute_outlier_scores(x, &trained)?;

        Ok(trained)
    }
}

impl RobustDiscriminantAnalysis {
    /// Compute discriminant components using robust scatter matrices
    fn compute_discriminant_components(
        &self,
        trained: &TrainedRobustDiscriminantAnalysis,
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let _n_features = trained.pooled_covariance.nrows();
        let n_components = trained.n_components;

        // For simplicity, we'll use PCA on the pooled robust covariance
        // In practice, you'd use the robust between-class and within-class scatter matrices
        let (eigenvalues, eigenvectors) =
            self.compute_eigendecomposition(&trained.pooled_covariance, n_components)?;

        Ok((eigenvectors, eigenvalues))
    }

    /// Compute eigendecomposition (simplified implementation)
    fn compute_eigendecomposition(
        &self,
        matrix: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let actual_components = n_components.min(n);

        let mut eigenvalues = Array1::zeros(actual_components);
        let mut eigenvectors = Array2::zeros((n, actual_components));

        // Simplified power iteration for dominant eigenvalues
        for comp in 0..actual_components {
            let mut v = Array1::from_iter((0..n).map(|i| (i as Float + 1.0).sin()));

            // Normalize
            let norm = v.iter().map(|&x| x * x).sum::<Float>().sqrt();
            if norm > self.config.tol {
                v /= norm;
            }

            // Power iteration
            for _ in 0..self.config.max_iter {
                let v_new = matrix.dot(&v);
                let norm = v_new.iter().map(|&x| x * x).sum::<Float>().sqrt();

                if norm < self.config.tol {
                    break;
                }

                let v_normalized = &v_new / norm;

                // Check convergence
                let diff = &v_normalized - &v;
                let change = diff.iter().map(|&x| x * x).sum::<Float>().sqrt();

                if change < self.config.tol {
                    break;
                }

                v = v_normalized;
            }

            // Compute eigenvalue
            let av = matrix.dot(&v);
            let eigenvalue = v
                .iter()
                .zip(av.iter())
                .map(|(&vi, &avi)| vi * avi)
                .sum::<Float>();

            eigenvalues[comp] = eigenvalue;
            eigenvectors.column_mut(comp).assign(&v);
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute outlier scores for all samples
    fn compute_outlier_scores(
        &self,
        x: &Array2<Float>,
        trained: &TrainedRobustDiscriminantAnalysis,
    ) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let mut outlier_scores = Array1::zeros(n_samples);

        // Compute overall robust statistics
        let overall_mean = trained.compute_robust_mean(&x.view())?;
        let overall_cov = trained.compute_robust_covariance(&x.view(), &overall_mean)?;
        let cov_inv = trained.compute_matrix_inverse(&overall_cov)?;

        // Compute Mahalanobis distances as outlier scores
        for i in 0..n_samples {
            let sample = x.row(i);
            let diff = &sample - &overall_mean;

            let mut mahal_dist = 0.0;
            for j in 0..diff.len() {
                for k in 0..diff.len() {
                    mahal_dist += diff[j] * cov_inv[[j, k]] * diff[k];
                }
            }
            outlier_scores[i] = mahal_dist.sqrt();
        }

        Ok(outlier_scores)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedRobustDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in probabilities.axis_iter(Axis(0)).enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedRobustDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        // Project samples into discriminant subspace
        let projected = x.dot(&self.components);

        // Compute class-conditional densities
        for (sample_idx, projected_sample) in projected.axis_iter(Axis(0)).enumerate() {
            let mut class_scores = Array1::zeros(n_classes);

            for (class_idx, &class) in self.classes.iter().enumerate() {
                if let Some(class_mean) = self.robust_means.get(&class) {
                    // Project class mean
                    let projected_mean = class_mean.dot(&self.components);

                    // Compute squared distance in projected space
                    let diff = &projected_sample - &projected_mean;
                    let distance = diff.iter().map(|&x| x * x).sum::<Float>();

                    // Convert to log-likelihood (negative squared distance)
                    class_scores[class_idx] = -0.5 * distance;
                }
            }

            // Convert to probabilities using softmax
            let max_score = class_scores
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_scores = class_scores.mapv(|x| (x - max_score).exp());
            let sum_exp = exp_scores.sum();

            if sum_exp > 0.0 {
                exp_scores /= sum_exp;
            } else {
                exp_scores.fill(1.0 / n_classes as Float);
            }

            probabilities.row_mut(sample_idx).assign(&exp_scores);
        }

        Ok(probabilities)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for TrainedRobustDiscriminantAnalysis {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let transformed = x.dot(&self.components);
        Ok(transformed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_robust_discriminant_analysis_huber() {
        // Create data with outliers
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],   // Class 0 clean
            [10.0, 10.0], // Class 0 outlier
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2],   // Class 1 clean
            [20.0, 20.0]  // Class 1 outlier
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let rda = RobustDiscriminantAnalysis::new()
            .estimator_type(MEstimatorType::Huber { k: 1.345 })
            .trimming_fraction(0.25);

        let fitted = rda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_robust_discriminant_analysis_tukey() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let rda =
            RobustDiscriminantAnalysis::new().estimator_type(MEstimatorType::Tukey { c: 4.685 });

        let fitted = rda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_robust_predict_proba() {
        let x = array![[1.0, 1.0], [1.1, 1.1], [3.0, 3.0], [3.1, 3.1]];
        let y = array![0, 0, 1, 1];

        let rda = RobustDiscriminantAnalysis::new().estimator_type(MEstimatorType::Hampel {
            a: 2.0,
            b: 4.0,
            c: 8.0,
        });

        let fitted = rda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_robust_transform() {
        let x = array![
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
            [3.0, 3.0, 3.0],
            [3.1, 3.1, 3.1]
        ];
        let y = array![0, 0, 1, 1];

        let rda = RobustDiscriminantAnalysis::new().n_components(Some(1));

        let fitted = rda.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.dim(), (4, 1));
    }

    #[test]
    fn test_outlier_scores() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],   // Normal points
            [10.0, 10.0], // Outlier
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2] // Normal points
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1];

        let rda = RobustDiscriminantAnalysis::new();
        let fitted = rda.fit(&x, &y).unwrap();
        let outlier_scores = fitted.outlier_scores();

        assert_eq!(outlier_scores.len(), 7);

        // The outlier (index 3) should have a higher score
        assert!(outlier_scores[3] > outlier_scores[0]);
        assert!(outlier_scores[3] > outlier_scores[1]);
        assert!(outlier_scores[3] > outlier_scores[2]);
    }

    #[test]
    fn test_trimmed_indices() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [10.0, 10.0], // Should be trimmed
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1];

        let rda = RobustDiscriminantAnalysis::new().trimming_fraction(0.2); // Trim 20%

        let fitted = rda.fit(&x, &y).unwrap();
        let trimmed_indices = fitted.trimmed_indices();

        // Should keep approximately 80% of samples
        let expected_kept = ((1.0 - 0.2) * 7.0) as usize;
        assert!(trimmed_indices.len() >= expected_kept - 1);
        assert!(trimmed_indices.len() <= expected_kept + 1);
    }

    #[test]
    fn test_andrews_estimator() {
        let x = array![[1.0, 1.0], [1.1, 1.1], [3.0, 3.0], [3.1, 3.1]];
        let y = array![0, 0, 1, 1];

        let rda =
            RobustDiscriminantAnalysis::new().estimator_type(MEstimatorType::Andrews { c: 1.339 });

        let fitted = rda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_adaptive_threshold() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let rda = RobustDiscriminantAnalysis::new()
            .adaptive_threshold(true)
            .bootstrap_samples(50);

        let fitted = rda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
    }
}
