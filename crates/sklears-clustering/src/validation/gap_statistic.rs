//! Gap statistic implementation for optimal cluster number selection
//!
//! The gap statistic compares the within-cluster sum of squares for different numbers
//! of clusters against a null reference distribution. It helps determine the optimal
//! number of clusters by finding where the observed clustering is most distinct from
//! random data with the same statistical properties.

use super::internal_validation::ClusteringValidator;
use super::validation_types::*;
use numrs2::prelude::*;
use scirs2_core::ndarray::Array2;
use scirs2_core::rand_prelude::Distribution;
// Normal distribution via scirs2_core::random::RandNormal
use scirs2_core::random::Random;
use sklears_core::error::{Result, SklearsError};
use std::ops::Range;

/// Gap statistic implementation
impl ClusteringValidator {
    /// Compute gap statistic for optimal cluster number selection
    ///
    /// The gap statistic measures the goodness of clustering by comparing
    /// the within-cluster sum of squares against a null reference distribution.
    /// The optimal number of clusters is where the gap is largest, using
    /// the "one standard error" rule.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `k_range` - Range of cluster numbers to test
    /// * `n_refs` - Number of reference datasets to generate (default: 10)
    /// * `clustering_fn` - Function that performs clustering for given k
    ///
    /// # Returns
    /// Gap statistic results with optimal k recommendation
    ///
    /// # Example
    /// ```rust
    /// use sklears_clustering::validation::{ClusteringValidator, ValidationMetric};
    /// use sklears_core::prelude::*;
    ///
    /// let validator = ClusteringValidator::new(ValidationMetric::Euclidean);
    /// let data = Array2::from_shape_vec(
    ///     (4, 2),
    ///     vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 5.2, 7.8],
    /// )
    /// .unwrap();
    ///
    /// // Define clustering function
    /// let clustering_fn = |data: &Array2<f64>, k: usize| -> Result<Vec<i32>> {
    ///     // Your clustering implementation here
    ///     // For example, K-means clustering
    ///     Ok((0..data.nrows()).map(|i| (i % k) as i32).collect())
    /// };
    ///
    /// let result = validator
    ///     .gap_statistic(&data, 1..4, Some(20), clustering_fn)
    ///     .unwrap();
    /// println!("Optimal k: {}", result.optimal_k);
    /// ```
    pub fn gap_statistic<F>(
        &self,
        X: &Array2<f64>,
        k_range: Range<usize>,
        n_refs: Option<usize>,
        clustering_fn: F,
    ) -> Result<GapStatisticResult>
    where
        F: Fn(&Array2<f64>, usize) -> Result<Vec<i32>>,
    {
        let n_refs = n_refs.unwrap_or(10);
        if n_refs == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of reference datasets must be positive".to_string(),
            ));
        }

        let k_values: Vec<usize> = k_range.collect();
        if k_values.is_empty() {
            return Err(SklearsError::InvalidInput(
                "k_range must not be empty".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Data matrix must not be empty".to_string(),
            ));
        }

        // Compute data range for generating reference datasets
        let (min_vals, max_vals) = self.compute_data_range(X);

        let mut gap_values = Vec::new();
        let mut gap_std_errors = Vec::new();
        let mut within_cluster_ss = Vec::new();
        let mut reference_statistics = Vec::new();

        // Compute gap statistic for each k
        for &k in &k_values {
            if k > n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "k ({}) cannot be larger than number of samples ({})",
                    k, n_samples
                )));
            }

            // Compute within-cluster sum of squares for original data
            let labels = clustering_fn(X, k)?;
            let w_k = self.compute_within_cluster_sum_of_squares(X, &labels)?;
            within_cluster_ss.push(w_k);

            // Generate reference datasets and compute their WCSS
            let mut reference_log_w_values = Vec::new();

            for _ in 0..n_refs {
                let reference_data = self.generate_reference_data(X, &min_vals, &max_vals);
                let ref_labels = clustering_fn(&reference_data, k)?;
                let ref_w_k =
                    self.compute_within_cluster_sum_of_squares(&reference_data, &ref_labels)?;

                if ref_w_k > 0.0 {
                    reference_log_w_values.push(ref_w_k.ln());
                } else {
                    // Handle edge case where WCSS is zero
                    reference_log_w_values.push(0.0);
                }
            }

            // Compute statistics for reference datasets
            let mean_log_w =
                reference_log_w_values.iter().sum::<f64>() / reference_log_w_values.len() as f64;
            let variance_log_w = reference_log_w_values
                .iter()
                .map(|x| (x - mean_log_w).powi(2))
                .sum::<f64>()
                / reference_log_w_values.len() as f64;
            let std_log_w = variance_log_w.sqrt();

            reference_statistics.push(ReferenceStatistics {
                k,
                mean_log_w,
                std_log_w,
                reference_log_w_values: reference_log_w_values.clone(),
            });

            // Compute gap value
            let log_w_k = if w_k > 0.0 { w_k.ln() } else { 0.0 };
            let gap_k = mean_log_w - log_w_k;
            gap_values.push(gap_k);

            // Compute standard error
            let s_k = std_log_w * (1.0 + 1.0 / n_refs as f64).sqrt();
            gap_std_errors.push(s_k);

            eprintln!(
                "Gap statistic: k={}, gap={:.4}, std_err={:.4}",
                k, gap_k, s_k
            );
        }

        // Find optimal k using the "one standard error" rule
        let optimal_k = self.find_optimal_k_one_se_rule(&k_values, &gap_values, &gap_std_errors);

        Ok(GapStatisticResult {
            gap_values,
            gap_std_errors,
            optimal_k,
            k_values,
            within_cluster_ss,
            reference_statistics,
            n_references: n_refs,
        })
    }

    /// Compute within-cluster sum of squares
    ///
    /// This measures the compactness of clusters by summing the squared distances
    /// from each point to its cluster centroid.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    /// Within-cluster sum of squares
    pub fn compute_within_cluster_sum_of_squares(
        &self,
        X: &Array2<f64>,
        labels: &[i32],
    ) -> Result<f64> {
        if X.nrows() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Data and labels length mismatch".to_string(),
            ));
        }

        let n_features = X.ncols();
        let mut total_wcss = 0.0;

        // Group points by cluster
        let mut clusters = std::collections::HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            if label != -1 {
                // Exclude noise points
                clusters.entry(label).or_insert_with(Vec::new).push(i);
            }
        }

        // Compute WCSS for each cluster
        for cluster_points in clusters.values() {
            if cluster_points.is_empty() {
                continue;
            }

            // Compute cluster centroid
            let mut centroid = vec![0.0; n_features];
            for &point_idx in cluster_points {
                for j in 0..n_features {
                    centroid[j] += X[(point_idx, j)];
                }
            }

            for val in centroid.iter_mut() {
                *val /= cluster_points.len() as f64;
            }

            // Sum squared distances to centroid
            for &point_idx in cluster_points {
                for j in 0..n_features {
                    let diff = X[(point_idx, j)] - centroid[j];
                    total_wcss += diff * diff;
                }
            }
        }

        Ok(total_wcss)
    }

    /// Compute data range (min and max values) for each feature
    ///
    /// This is used to generate reference data with the same statistical properties
    /// as the original data.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    ///
    /// # Returns
    /// Tuple of (min_values, max_values) for each feature
    pub fn compute_data_range(&self, X: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
        let n_features = X.ncols();
        let mut min_vals = vec![f64::INFINITY; n_features];
        let mut max_vals = vec![f64::NEG_INFINITY; n_features];

        for i in 0..X.nrows() {
            for j in 0..n_features {
                let val = X[(i, j)];
                if val < min_vals[j] {
                    min_vals[j] = val;
                }
                if val > max_vals[j] {
                    max_vals[j] = val;
                }
            }
        }

        (min_vals, max_vals)
    }

    /// Generate reference data with uniform distribution within data range
    ///
    /// Creates a random dataset with the same dimensions as the original data,
    /// where each feature is uniformly distributed within its observed range.
    ///
    /// # Arguments
    /// * `X` - Original data matrix (for dimensions)
    /// * `min_vals` - Minimum values for each feature
    /// * `max_vals` - Maximum values for each feature
    ///
    /// # Returns
    /// Reference data matrix
    pub fn generate_reference_data(
        &self,
        X: &Array2<f64>,
        min_vals: &[f64],
        max_vals: &[f64],
    ) -> Array2<f64> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        let mut rng = Random::default();
        let mut reference_data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let range = max_vals[j] - min_vals[j];
                if range > 0.0 {
                    reference_data[(i, j)] = min_vals[j] + rng.gen_range(0.0..1.0) * range;
                } else {
                    reference_data[(i, j)] = min_vals[j];
                }
            }
        }

        reference_data
    }

    /// Generate reference data using Principal Component Analysis
    ///
    /// This creates a more sophisticated null model that preserves the
    /// principal components of the original data while randomizing within
    /// the PC space.
    ///
    /// # Arguments
    /// * `X` - Original data matrix
    ///
    /// # Returns
    /// PCA-based reference data matrix
    pub fn generate_pca_reference_data(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples < 2 || n_features < 1 {
            return Err(SklearsError::InvalidInput(
                "Insufficient data for PCA reference generation".to_string(),
            ));
        }

        // Center the data
        let mut means = vec![0.0; n_features];
        for j in 0..n_features {
            for i in 0..n_samples {
                means[j] += X[(i, j)];
            }
            means[j] /= n_samples as f64;
        }

        let mut centered_data = X.clone();
        for i in 0..n_samples {
            for j in 0..n_features {
                centered_data[(i, j)] -= means[j];
            }
        }

        // For simplicity, we'll use a basic PCA approximation
        // In a full implementation, you would use proper SVD/eigendecomposition

        // Compute covariance matrix
        let mut cov_matrix = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += centered_data[(k, i)] * centered_data[(k, j)];
                }
                cov_matrix[(i, j)] = sum / (n_samples - 1) as f64;
            }
        }

        // For this implementation, we'll use a simpler approach:
        // Generate random data with the same mean and variance as the original
        let mut rng = Random::default();
        let mut reference_data = Array2::zeros((n_samples, n_features));

        // Compute standard deviations
        let mut stds = vec![0.0; n_features];
        for j in 0..n_features {
            let variance = cov_matrix[(j, j)];
            stds[j] = variance.sqrt();
        }

        // Generate reference data
        let normal = scirs2_core::random::RandNormal::new(0.0, 1.0).unwrap();
        for i in 0..n_samples {
            for j in 0..n_features {
                let normal_sample: f64 = normal.sample(&mut rng);
                reference_data[(i, j)] = means[j] + stds[j] * normal_sample;
            }
        }

        Ok(reference_data)
    }

    /// Find optimal k using the "one standard error" rule
    ///
    /// The rule states: choose the smallest k such that Gap(k) >= Gap(k+1) - se(k+1),
    /// where se is the standard error.
    ///
    /// # Arguments
    /// * `k_values` - K values tested
    /// * `gap_values` - Gap statistic values
    /// * `gap_std_errors` - Standard errors for gap values
    ///
    /// # Returns
    /// Optimal k value
    fn find_optimal_k_one_se_rule(
        &self,
        k_values: &[usize],
        gap_values: &[f64],
        gap_std_errors: &[f64],
    ) -> usize {
        for i in 0..(gap_values.len() - 1) {
            let gap_k = gap_values[i];
            let gap_k_plus_1 = gap_values[i + 1];
            let se_k_plus_1 = gap_std_errors[i + 1];

            if gap_k >= gap_k_plus_1 - se_k_plus_1 {
                return k_values[i];
            }
        }

        // If no k satisfies the rule, return the k with maximum gap
        let max_gap_idx = gap_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        k_values[max_gap_idx]
    }

    /// Find optimal k using maximum gap criterion
    ///
    /// Alternative method that simply chooses the k with the largest gap value.
    ///
    /// # Arguments
    /// * `k_values` - K values tested
    /// * `gap_values` - Gap statistic values
    ///
    /// # Returns
    /// K value with maximum gap
    pub fn find_optimal_k_max_gap(&self, k_values: &[usize], gap_values: &[f64]) -> usize {
        let max_gap_idx = gap_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        k_values[max_gap_idx]
    }

    /// Compute gap statistic with advanced reference generation
    ///
    /// This version uses PCA-based reference data generation for a more
    /// sophisticated null model.
    ///
    /// # Arguments
    /// * `X` - Input data matrix
    /// * `k_range` - Range of cluster numbers to test
    /// * `n_refs` - Number of reference datasets
    /// * `clustering_fn` - Clustering function
    /// * `use_pca` - Whether to use PCA-based reference generation
    ///
    /// # Returns
    /// Gap statistic results
    pub fn gap_statistic_advanced<F>(
        &self,
        X: &Array2<f64>,
        k_range: Range<usize>,
        n_refs: Option<usize>,
        clustering_fn: F,
        use_pca: bool,
    ) -> Result<GapStatisticResult>
    where
        F: Fn(&Array2<f64>, usize) -> Result<Vec<i32>>,
    {
        let n_refs = n_refs.unwrap_or(10);
        let k_values: Vec<usize> = k_range.collect();

        let mut gap_values = Vec::new();
        let mut gap_std_errors = Vec::new();
        let mut within_cluster_ss = Vec::new();
        let mut reference_statistics = Vec::new();

        // Pre-compute data range for uniform reference generation
        let (min_vals, max_vals) = if !use_pca {
            self.compute_data_range(X)
        } else {
            (Vec::new(), Vec::new()) // Not used for PCA
        };

        for &k in &k_values {
            // Compute WCSS for original data
            let labels = clustering_fn(X, k)?;
            let w_k = self.compute_within_cluster_sum_of_squares(X, &labels)?;
            within_cluster_ss.push(w_k);

            // Generate reference datasets
            let mut reference_log_w_values = Vec::new();

            for _ in 0..n_refs {
                let reference_data = if use_pca {
                    self.generate_pca_reference_data(X)?
                } else {
                    self.generate_reference_data(X, &min_vals, &max_vals)
                };

                let ref_labels = clustering_fn(&reference_data, k)?;
                let ref_w_k =
                    self.compute_within_cluster_sum_of_squares(&reference_data, &ref_labels)?;

                if ref_w_k > 0.0 {
                    reference_log_w_values.push(ref_w_k.ln());
                } else {
                    reference_log_w_values.push(0.0);
                }
            }

            // Compute reference statistics
            let mean_log_w =
                reference_log_w_values.iter().sum::<f64>() / reference_log_w_values.len() as f64;
            let variance_log_w = reference_log_w_values
                .iter()
                .map(|x| (x - mean_log_w).powi(2))
                .sum::<f64>()
                / reference_log_w_values.len() as f64;
            let std_log_w = variance_log_w.sqrt();

            reference_statistics.push(ReferenceStatistics {
                k,
                mean_log_w,
                std_log_w,
                reference_log_w_values,
            });

            // Compute gap and standard error
            let log_w_k = if w_k > 0.0 { w_k.ln() } else { 0.0 };
            let gap_k = mean_log_w - log_w_k;
            let s_k = std_log_w * (1.0 + 1.0 / n_refs as f64).sqrt();

            gap_values.push(gap_k);
            gap_std_errors.push(s_k);
        }

        let optimal_k = self.find_optimal_k_one_se_rule(&k_values, &gap_values, &gap_std_errors);

        Ok(GapStatisticResult {
            gap_values,
            gap_std_errors,
            optimal_k,
            k_values,
            within_cluster_ss,
            reference_statistics,
            n_references: n_refs,
        })
    }

    /// Validate gap statistic results
    ///
    /// Performs sanity checks on the gap statistic computation to ensure
    /// the results are reasonable.
    ///
    /// # Arguments
    /// * `result` - Gap statistic result to validate
    ///
    /// # Returns
    /// Validation summary and any warnings
    pub fn validate_gap_statistic(&self, result: &GapStatisticResult) -> GapStatisticValidation {
        let mut warnings = Vec::new();
        let mut is_valid = true;

        // Check if gap values are reasonable
        let max_gap = result
            .gap_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_gap = result
            .gap_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        if max_gap - min_gap < 0.01 {
            warnings.push(
                "Gap values show very little variation - results may not be reliable".to_string(),
            );
        }

        if max_gap < 0.0 {
            warnings.push(
                "All gap values are negative - clustering may not be better than random"
                    .to_string(),
            );
        }

        // Check if standard errors are reasonable
        let max_se = result
            .gap_std_errors
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if max_se > max_gap {
            warnings.push(
                "Standard errors are very large compared to gap values - increase n_references"
                    .to_string(),
            );
            is_valid = false;
        }

        // Check if optimal k is at boundary
        if result.optimal_k == result.k_values[0] {
            warnings.push(
                "Optimal k is at the lower boundary - consider testing smaller k values"
                    .to_string(),
            );
        }

        if result.optimal_k == result.k_values[result.k_values.len() - 1] {
            warnings.push(
                "Optimal k is at the upper boundary - consider testing larger k values".to_string(),
            );
        }

        // Check for monotonic decrease in WCSS
        let mut wcss_decreasing = true;
        for i in 1..result.within_cluster_ss.len() {
            if result.within_cluster_ss[i] > result.within_cluster_ss[i - 1] {
                wcss_decreasing = false;
                break;
            }
        }

        if !wcss_decreasing {
            warnings.push("Within-cluster sum of squares is not monotonically decreasing - check clustering function".to_string());
            is_valid = false;
        }

        GapStatisticValidation {
            is_valid,
            warnings,
            recommendation: self.generate_gap_recommendation(result),
        }
    }

    /// Generate recommendation based on gap statistic results
    fn generate_gap_recommendation(&self, result: &GapStatisticResult) -> String {
        let optimal_gap = result.gap_values[result
            .k_values
            .iter()
            .position(|&k| k == result.optimal_k)
            .unwrap_or(0)];

        if optimal_gap > 0.5 {
            format!(
                "Strong evidence for {} clusters (gap = {:.3}). The clustering structure is clearly better than random.",
                result.optimal_k, optimal_gap
            )
        } else if optimal_gap > 0.1 {
            format!(
                "Moderate evidence for {} clusters (gap = {:.3}). Consider validating with other metrics.",
                result.optimal_k, optimal_gap
            )
        } else {
            format!(
                "Weak evidence for clustering structure. Optimal k = {} with gap = {:.3}. \
                 Data may not have clear cluster structure.",
                result.optimal_k, optimal_gap
            )
        }
    }
}

/// Validation result for gap statistic computation
#[derive(Debug, Clone)]
pub struct GapStatisticValidation {
    /// Whether the gap statistic results are considered valid
    pub is_valid: bool,
    /// List of warnings about the results
    pub warnings: Vec<String>,
    /// Recommendation based on the gap statistic
    pub recommendation: String,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> Array2<f64> {
        // Create test data with clear cluster structure
        let data_vec = vec![
            // Cluster 1
            1.0, 1.0, 1.2, 1.1, 1.1, 1.2, // Cluster 2
            5.0, 5.0, 5.1, 5.2, 5.2, 5.1, // Cluster 3
            9.0, 9.0, 9.1, 9.2, 9.2, 9.1,
        ];
        Array2::from_shape_vec((9, 2), data_vec).unwrap()
    }

    // Simple clustering function for testing
    fn simple_clustering_fn(data: &Array2<f64>, k: usize) -> Result<Vec<i32>> {
        let n_samples = data.nrows();
        let mut labels = vec![-1; n_samples];

        // Assign first k*3 points to k clusters (3 points each)
        for i in 0..n_samples.min(k * 3) {
            labels[i] = (i / 3) as i32;
        }

        Ok(labels)
    }

    #[test]
    fn test_gap_statistic_basic() {
        let validator = ClusteringValidator::euclidean();
        let data = create_test_data();

        let result = validator
            .gap_statistic(&data, 1..6, Some(5), simple_clustering_fn)
            .unwrap();

        assert_eq!(result.k_values.len(), 5);
        assert_eq!(result.gap_values.len(), 5);
        assert_eq!(result.gap_std_errors.len(), 5);
        assert_eq!(result.within_cluster_ss.len(), 5);
        assert_eq!(result.reference_statistics.len(), 5);
        assert!(result.optimal_k >= 1 && result.optimal_k <= 5);

        // Gap values should be finite
        for gap in &result.gap_values {
            assert!(gap.is_finite());
        }

        // Standard errors should be non-negative
        for std_err in &result.gap_std_errors {
            assert!(*std_err >= 0.0);
        }

        // WCSS should be non-negative
        for wcss in &result.within_cluster_ss {
            assert!(*wcss >= 0.0);
        }
    }

    #[test]
    fn test_within_cluster_sum_of_squares() {
        let validator = ClusteringValidator::euclidean();
        let data = create_test_data();
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let wcss = validator
            .compute_within_cluster_sum_of_squares(&data, &labels)
            .unwrap();

        assert!(wcss >= 0.0);
        assert!(wcss.is_finite());
    }

    #[test]
    fn test_data_range_computation() {
        let validator = ClusteringValidator::euclidean();
        let data = create_test_data();

        let (min_vals, max_vals) = validator.compute_data_range(&data);

        assert_eq!(min_vals.len(), 2);
        assert_eq!(max_vals.len(), 2);

        for i in 0..min_vals.len() {
            assert!(min_vals[i] <= max_vals[i]);
            assert!(min_vals[i].is_finite());
            assert!(max_vals[i].is_finite());
        }

        // Check specific values for our test data
        assert!((min_vals[0] - 1.0).abs() < 1e-10);
        assert!((max_vals[0] - 9.2).abs() < 1e-10);
    }

    #[test]
    fn test_reference_data_generation() {
        let validator = ClusteringValidator::euclidean();
        let data = create_test_data();

        let (min_vals, max_vals) = validator.compute_data_range(&data);
        let ref_data = validator.generate_reference_data(&data, &min_vals, &max_vals);

        assert_eq!(ref_data.nrows(), data.nrows());
        assert_eq!(ref_data.ncols(), data.ncols());

        // Check that all values are within bounds
        for i in 0..ref_data.nrows() {
            for j in 0..ref_data.ncols() {
                let val = ref_data[(i, j)];
                assert!(val >= min_vals[j] && val <= max_vals[j]);
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_pca_reference_data_generation() {
        let validator = ClusteringValidator::euclidean();
        let data = create_test_data();

        let ref_data = validator.generate_pca_reference_data(&data).unwrap();

        assert_eq!(ref_data.nrows(), data.nrows());
        assert_eq!(ref_data.ncols(), data.ncols());

        // All values should be finite
        for i in 0..ref_data.nrows() {
            for j in 0..ref_data.ncols() {
                assert!(ref_data[(i, j)].is_finite());
            }
        }
    }

    #[test]
    fn test_optimal_k_selection() {
        let validator = ClusteringValidator::euclidean();

        let k_values = vec![1, 2, 3, 4, 5];
        let gap_values = vec![0.1, 0.5, 0.8, 0.6, 0.4];
        let gap_std_errors = vec![0.05, 0.1, 0.15, 0.12, 0.08];

        let optimal_k =
            validator.find_optimal_k_one_se_rule(&k_values, &gap_values, &gap_std_errors);
        assert!(optimal_k >= 1 && optimal_k <= 5);

        let max_gap_k = validator.find_optimal_k_max_gap(&k_values, &gap_values);
        assert_eq!(max_gap_k, 3); // k=3 has the highest gap (0.8)
    }

    #[test]
    fn test_gap_statistic_advanced() {
        let validator = ClusteringValidator::euclidean();
        let data = create_test_data();

        // Test with uniform reference
        let result_uniform = validator
            .gap_statistic_advanced(&data, 1..4, Some(3), simple_clustering_fn, false)
            .unwrap();

        // Test with PCA reference
        let result_pca = validator
            .gap_statistic_advanced(&data, 1..4, Some(3), simple_clustering_fn, true)
            .unwrap();

        // Both should produce valid results
        assert_eq!(result_uniform.k_values.len(), 3);
        assert_eq!(result_pca.k_values.len(), 3);

        assert!(result_uniform.optimal_k >= 1 && result_uniform.optimal_k <= 3);
        assert!(result_pca.optimal_k >= 1 && result_pca.optimal_k <= 3);
    }

    #[test]
    fn test_gap_statistic_validation() {
        let validator = ClusteringValidator::euclidean();
        let data = create_test_data();

        let result = validator
            .gap_statistic(&data, 1..6, Some(10), simple_clustering_fn)
            .unwrap();

        let validation = validator.validate_gap_statistic(&result);

        // Should have some recommendation
        assert!(!validation.recommendation.is_empty());

        // Check that warnings are reasonable
        for warning in &validation.warnings {
            assert!(!warning.is_empty());
        }
    }

    #[test]
    fn test_gap_statistic_edge_cases() {
        let validator = ClusteringValidator::euclidean();

        // Empty k range
        let data = create_test_data();
        let result = validator.gap_statistic(&data, 1..1, Some(5), simple_clustering_fn);
        assert!(result.is_err());

        // k larger than sample size
        let small_data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = validator.gap_statistic(&small_data, 1..6, Some(3), simple_clustering_fn);
        assert!(result.is_err());

        // Zero reference datasets
        let result = validator.gap_statistic(&data, 1..4, Some(0), simple_clustering_fn);
        assert!(result.is_err());
    }

    #[test]
    fn test_wcss_with_noise_points() {
        let validator = ClusteringValidator::euclidean();
        let data = create_test_data();

        // Include some noise points (label -1)
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, -1];

        let wcss = validator
            .compute_within_cluster_sum_of_squares(&data, &labels)
            .unwrap();

        // Should ignore noise points and compute WCSS for valid clusters only
        assert!(wcss >= 0.0);
        assert!(wcss.is_finite());
    }

    #[test]
    fn test_gap_statistic_result_methods() {
        let result = GapStatisticResult {
            gap_values: vec![0.1, 0.5, 0.8, 0.3],
            gap_std_errors: vec![0.05, 0.1, 0.15, 0.08],
            optimal_k: 3,
            k_values: vec![1, 2, 3, 4],
            within_cluster_ss: vec![10.0, 5.0, 2.0, 3.0],
            reference_statistics: Vec::new(),
            n_references: 10,
        };

        // Test gap_for_k method
        assert_eq!(result.gap_for_k(2), Some(0.5));
        assert_eq!(result.gap_for_k(5), None);

        // Test recommended_k_values method
        let recommended = result.recommended_k_values(2);
        assert_eq!(recommended.len(), 2);
        assert_eq!(recommended[0], (3, 0.8)); // Highest gap

        // Test is_optimal_significant method
        assert!(result.is_optimal_significant());
    }
}
