//! Split implementations for decision trees
//!
//! This module contains various splitting strategies including hyperplane splits,
//! CHAID splits, and conditional inference splits with their statistical tests.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::thread_rng;
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

#[cfg(feature = "oblique")]
use scirs2_core::ndarray::s;

use crate::criteria::{ConditionalTestType, FeatureType};

/// Hyperplane split information for oblique trees
#[derive(Debug, Clone)]
pub struct HyperplaneSplit {
    /// Feature coefficients for the hyperplane (w^T x >= threshold)
    pub coefficients: Array1<f64>,
    /// Threshold for the hyperplane split
    pub threshold: f64,
    /// Bias term for the hyperplane
    pub bias: f64,
    /// Impurity decrease achieved by this split
    pub impurity_decrease: f64,
}

impl HyperplaneSplit {
    /// Evaluate the hyperplane split for a sample
    pub fn evaluate(&self, sample: &Array1<f64>) -> bool {
        let dot_product = self.coefficients.dot(sample) + self.bias;
        dot_product >= self.threshold
    }

    /// Create a random hyperplane with normalized coefficients
    pub fn random(n_features: usize, rng: &mut scirs2_core::CoreRandom) -> Self {
        let mut coefficients = Array1::zeros(n_features);
        for i in 0..n_features {
            coefficients[i] = rng.gen_range(-1.0..1.0);
        }

        // Normalize coefficients
        let dot_product: f64 = coefficients.dot(&coefficients);
        let norm = dot_product.sqrt();
        if norm > 1e-10_f64 {
            coefficients /= norm;
        }

        Self {
            coefficients,
            threshold: rng.gen_range(-1.0..1.0),
            bias: rng.gen_range(-0.1..0.1),
            impurity_decrease: 0.0,
        }
    }

    /// Find optimal hyperplane using ridge regression
    #[cfg(feature = "oblique")]
    pub fn from_ridge_regression(x: &Array2<f64>, y: &Array1<f64>, alpha: f64) -> Result<Self> {
        let n_features = x.ncols();
        if x.nrows() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for ridge regression".to_string(),
            ));
        }

        // Add bias column to X
        let mut x_bias = Array2::ones((x.nrows(), n_features + 1));
        x_bias.slice_mut(s![.., ..n_features]).assign(x);

        // Ridge regression: w = (X^T X + α I)^(-1) X^T y
        let xtx = x_bias.t().dot(&x_bias);
        let ridge_matrix = xtx + Array2::<f64>::eye(n_features + 1) * alpha;
        let xty = x_bias.t().dot(y);

        // Simple matrix inverse using Gauss-Jordan elimination
        match gauss_jordan_inverse(&ridge_matrix) {
            Ok(inv_matrix) => {
                let coefficients_full = inv_matrix.dot(&xty);

                let coefficients = coefficients_full.slice(s![..n_features]).to_owned();
                let bias = coefficients_full[n_features];

                Ok(Self {
                    coefficients,
                    threshold: 0.0, // Will be set during split evaluation
                    bias,
                    impurity_decrease: 0.0,
                })
            }
            Err(_) => {
                // Fallback to random hyperplane if matrix is singular
                let mut rng = thread_rng();
                Ok(Self::random(n_features, &mut rng))
            }
        }
    }
}

/// Simple Gauss-Jordan elimination for matrix inversion
#[cfg(feature = "oblique")]
fn gauss_jordan_inverse(matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    // Create augmented matrix [A | I]
    let mut augmented = Array2::zeros((n, 2 * n));

    // Copy matrix to left side
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = matrix[[i, j]];
        }
        // Identity matrix on right side
        augmented[[i, i + n]] = 1.0;
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Check for singular matrix
        if augmented[[max_row, i]].abs() < 1e-12 {
            return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = augmented[[i, j]];
                augmented[[i, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = temp;
            }
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

    // Extract inverse matrix from right side
    let mut inverse = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inverse[[i, j]] = augmented[[i, j + n]];
        }
    }

    Ok(inverse)
}

/// CHAID (Chi-squared Automatic Interaction Detection) split information
#[derive(Debug, Clone)]
pub struct ChaidSplit {
    /// Feature index
    pub feature_idx: usize,
    /// Category groups after merging based on chi-squared tests
    pub category_groups: Vec<Vec<String>>,
    /// Chi-squared statistic
    pub chi_squared: f64,
    /// P-value of the chi-squared test
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
    /// Significance level used
    pub significance_level: f64,
}

impl ChaidSplit {
    /// Perform CHAID splitting analysis for a categorical feature
    pub fn analyze_categorical_split(
        feature_values: &[String],
        target_values: &[i32],
        significance_level: f64,
    ) -> Result<Option<Self>> {
        if feature_values.len() != target_values.len() {
            return Err(SklearsError::InvalidInput(
                "Feature and target arrays must have the same length".to_string(),
            ));
        }

        if feature_values.is_empty() {
            return Ok(None);
        }

        // Build contingency table
        let contingency_table = build_contingency_table(feature_values, target_values)?;

        // Perform iterative category merging based on chi-squared tests
        let merged_categories = merge_categories_chaid(&contingency_table, significance_level)?;

        if merged_categories.len() <= 1 {
            return Ok(None); // No meaningful split possible
        }

        // Calculate final chi-squared statistic
        let (chi_squared, p_value, df) = calculate_chi_squared(&contingency_table)?;

        Ok(Some(ChaidSplit {
            feature_idx: 0, // Will be set by caller
            category_groups: merged_categories,
            chi_squared,
            p_value,
            degrees_of_freedom: df,
            significance_level,
        }))
    }

    /// Check if the split is statistically significant
    pub fn is_significant(&self) -> bool {
        self.p_value < self.significance_level
    }
}

/// Build contingency table for categorical feature vs target
fn build_contingency_table(
    feature_values: &[String],
    target_values: &[i32],
) -> Result<HashMap<String, HashMap<i32, usize>>> {
    let mut table: HashMap<String, HashMap<i32, usize>> = HashMap::new();

    for (feature_val, target_val) in feature_values.iter().zip(target_values.iter()) {
        let target_counts = table.entry(feature_val.clone()).or_default();
        *target_counts.entry(*target_val).or_insert(0) += 1;
    }

    Ok(table)
}

/// Merge categories based on chi-squared tests (CHAID algorithm)
fn merge_categories_chaid(
    contingency_table: &HashMap<String, HashMap<i32, usize>>,
    significance_level: f64,
) -> Result<Vec<Vec<String>>> {
    let categories: Vec<String> = contingency_table.keys().cloned().collect();
    let mut groups: Vec<Vec<String>> = categories.iter().map(|c| vec![c.clone()]).collect();

    if groups.len() <= 1 {
        return Ok(groups);
    }

    loop {
        let mut best_merge: Option<(usize, usize, f64)> = None;
        let mut min_chi_squared = f64::INFINITY;

        // Find the pair of adjacent categories with the smallest chi-squared statistic
        for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                // Create merged contingency table for these two groups
                let merged_table =
                    create_merged_contingency_table(contingency_table, &groups[i], &groups[j])?;

                if let Ok((chi_squared, p_value, _)) =
                    calculate_chi_squared_for_merged(&merged_table)
                {
                    // If not significant (p > significance_level), consider for merging
                    if p_value > significance_level && chi_squared < min_chi_squared {
                        min_chi_squared = chi_squared;
                        best_merge = Some((i, j, chi_squared));
                    }
                }
            }
        }

        // If no merge found, stop
        if let Some((i, j, _)) = best_merge {
            // Merge groups i and j
            let mut merged_group = groups[i].clone();
            merged_group.extend(groups[j].clone());

            // Remove the original groups and add the merged group
            if i < j {
                groups.remove(j);
                groups.remove(i);
            } else {
                groups.remove(i);
                groups.remove(j);
            }
            groups.push(merged_group);
        } else {
            break;
        }

        if groups.len() <= 1 {
            break;
        }
    }

    Ok(groups)
}

/// Create merged contingency table for two category groups
fn create_merged_contingency_table(
    original_table: &HashMap<String, HashMap<i32, usize>>,
    group1: &[String],
    group2: &[String],
) -> Result<HashMap<i32, usize>> {
    let mut merged_table = HashMap::new();

    // Add counts from group1
    for category in group1 {
        if let Some(target_counts) = original_table.get(category) {
            for (&target, &count) in target_counts {
                *merged_table.entry(target).or_insert(0) += count;
            }
        }
    }

    // Add counts from group2
    for category in group2 {
        if let Some(target_counts) = original_table.get(category) {
            for (&target, &count) in target_counts {
                *merged_table.entry(target).or_insert(0) += count;
            }
        }
    }

    Ok(merged_table)
}

/// Calculate chi-squared statistic for contingency table
fn calculate_chi_squared(
    contingency_table: &HashMap<String, HashMap<i32, usize>>,
) -> Result<(f64, f64, usize)> {
    use std::collections::HashSet;

    // Get all unique target values
    let mut all_targets: HashSet<i32> = HashSet::new();
    for target_counts in contingency_table.values() {
        all_targets.extend(target_counts.keys());
    }

    if all_targets.len() <= 1 {
        return Ok((0.0, 1.0, 0));
    }

    let categories: Vec<&String> = contingency_table.keys().collect();
    let targets: Vec<i32> = all_targets.into_iter().collect();

    if categories.len() <= 1 {
        return Ok((0.0, 1.0, 0));
    }

    // Calculate row and column totals
    let mut row_totals: HashMap<&String, usize> = HashMap::new();
    let mut col_totals: HashMap<i32, usize> = HashMap::new();
    let mut grand_total = 0;

    for category in &categories {
        let mut row_total = 0;
        if let Some(target_counts) = contingency_table.get(*category) {
            for (&target, &count) in target_counts {
                row_total += count;
                *col_totals.entry(target).or_insert(0) += count;
                grand_total += count;
            }
        }
        row_totals.insert(category, row_total);
    }

    if grand_total == 0 {
        return Ok((0.0, 1.0, 0));
    }

    // Calculate chi-squared statistic
    let mut chi_squared = 0.0;
    for category in &categories {
        for &target in &targets {
            let observed = contingency_table
                .get(*category)
                .and_then(|counts| counts.get(&target))
                .unwrap_or(&0);

            let expected = (*row_totals.get(category).unwrap_or(&0) as f64)
                * (*col_totals.get(&target).unwrap_or(&0) as f64)
                / (grand_total as f64);

            if expected > 0.0 {
                let diff = (*observed as f64) - expected;
                chi_squared += (diff * diff) / expected;
            }
        }
    }

    let degrees_of_freedom = (categories.len() - 1) * (targets.len() - 1);
    let p_value = chi_squared_p_value(chi_squared, degrees_of_freedom);

    Ok((chi_squared, p_value, degrees_of_freedom))
}

/// Calculate chi-squared statistic for merged contingency table
fn calculate_chi_squared_for_merged(
    merged_table: &HashMap<i32, usize>,
) -> Result<(f64, f64, usize)> {
    if merged_table.len() <= 1 {
        return Ok((0.0, 1.0, 0));
    }

    let total: usize = merged_table.values().sum();
    if total == 0 {
        return Ok((0.0, 1.0, 0));
    }

    // Simple chi-squared test for goodness of fit (equal expected frequencies)
    let expected = total as f64 / merged_table.len() as f64;
    let mut chi_squared = 0.0;

    for &observed in merged_table.values() {
        let diff = (observed as f64) - expected;
        chi_squared += (diff * diff) / expected;
    }

    let degrees_of_freedom = merged_table.len() - 1;
    let p_value = chi_squared_p_value(chi_squared, degrees_of_freedom);

    Ok((chi_squared, p_value, degrees_of_freedom))
}

/// Calculate approximate p-value for chi-squared statistic
fn chi_squared_p_value(chi_squared: f64, df: usize) -> f64 {
    if df == 0 || chi_squared <= 0.0 {
        return 1.0;
    }

    // Simple approximation using Wilson-Hilferty transformation
    // For more accuracy, consider using a proper statistical library
    let h = 2.0 / (9.0 * df as f64);
    let z = ((chi_squared / df as f64).powf(1.0 / 3.0) - 1.0 + h) / h.sqrt();

    // Approximate standard normal CDF
    if z > 0.0 {
        0.5 * (1.0 - (2.0 / std::f64::consts::PI).sqrt() * z * (-z * z / 2.0).exp())
    } else {
        0.5 * (1.0 + (2.0 / std::f64::consts::PI).sqrt() * (-z) * (-z * z / 2.0).exp())
    }
}

/// Conditional inference tree split information
#[derive(Debug, Clone)]
pub struct ConditionalInferenceSplit {
    /// Feature index that was selected for splitting
    pub feature_idx: usize,
    /// Split value for continuous features
    pub split_value: Option<f64>,
    /// Categories for the left branch (for categorical features)
    pub left_categories: Option<Vec<String>>,
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value of the statistical test
    pub p_value: f64,
    /// Type of test performed
    pub test_type: ConditionalTestType,
    /// Significance level used
    pub significance_level: f64,
}

impl ConditionalInferenceSplit {
    /// Perform conditional inference splitting analysis
    pub fn analyze_conditional_split(
        x: &Array2<f64>,
        y: &Array1<f64>,
        _feature_types: &[FeatureType],
        significance_level: f64,
        test_type: ConditionalTestType,
    ) -> Result<Option<Self>> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Feature and target arrays must have the same length".to_string(),
            ));
        }

        if x.nrows() < 4 {
            return Ok(None); // Need at least 4 samples for meaningful statistics
        }

        let n_features = x.ncols();
        let mut best_split: Option<ConditionalInferenceSplit> = None;
        let mut best_p_value = 1.0;

        // Test each feature for association with the target
        for feature_idx in 0..n_features {
            let feature_values = x.column(feature_idx);

            let (test_statistic, p_value) = match test_type {
                ConditionalTestType::QuadraticForm => {
                    compute_quadratic_form_test(&feature_values, y)?
                }
                ConditionalTestType::MaxType => compute_maxtype_test(&feature_values, y)?,
                ConditionalTestType::MonteCarlo { n_permutations } => {
                    compute_monte_carlo_test(&feature_values, y, n_permutations)?
                }
                ConditionalTestType::AsymptoticChiSquared => {
                    compute_asymptotic_chi_squared_test(&feature_values, y)?
                }
            };

            // Check if this is the most significant association
            if p_value < significance_level && p_value < best_p_value {
                // Find the best split point for this feature
                let split_value = find_best_split_point(&feature_values, y)?;

                best_split = Some(ConditionalInferenceSplit {
                    feature_idx,
                    split_value: Some(split_value),
                    left_categories: None,
                    test_statistic,
                    p_value,
                    test_type,
                    significance_level,
                });
                best_p_value = p_value;
            }
        }

        Ok(best_split)
    }

    /// Check if the split is statistically significant
    pub fn is_significant(&self) -> bool {
        self.p_value < self.significance_level
    }
}

/// Compute quadratic form test statistic for continuous features
fn compute_quadratic_form_test(
    feature_values: &ArrayView1<f64>,
    target_values: &Array1<f64>,
) -> Result<(f64, f64)> {
    let n = feature_values.len();
    if n < 4 {
        return Ok((0.0, 1.0));
    }

    // Compute correlation coefficient
    let feature_mean = feature_values.mean().unwrap_or(0.0);
    let target_mean = target_values.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut feature_var = 0.0;
    let mut target_var = 0.0;

    for i in 0..n {
        let feature_diff = feature_values[i] - feature_mean;
        let target_diff = target_values[i] - target_mean;

        numerator += feature_diff * target_diff;
        feature_var += feature_diff * feature_diff;
        target_var += target_diff * target_diff;
    }

    if feature_var == 0.0 || target_var == 0.0 {
        return Ok((0.0, 1.0));
    }

    let correlation = numerator / (feature_var * target_var).sqrt();

    // Transform to test statistic
    let test_statistic =
        correlation * correlation * (n - 2) as f64 / (1.0 - correlation * correlation);

    // Approximate p-value using t-distribution approximation
    let p_value = 2.0 * (1.0 - student_t_cdf(test_statistic.sqrt(), n - 2));

    Ok((test_statistic, p_value))
}

/// Compute maxtype test statistic for categorical features
fn compute_maxtype_test(
    feature_values: &ArrayView1<f64>,
    target_values: &Array1<f64>,
) -> Result<(f64, f64)> {
    // For simplicity, treat as continuous and use quadratic form
    // In practice, this would be more sophisticated for true categorical data
    compute_quadratic_form_test(feature_values, target_values)
}

/// Compute Monte Carlo permutation test
fn compute_monte_carlo_test(
    feature_values: &ArrayView1<f64>,
    target_values: &Array1<f64>,
    n_permutations: usize,
) -> Result<(f64, f64)> {
    // Compute original test statistic
    let (original_statistic, _) = compute_quadratic_form_test(feature_values, target_values)?;

    // Perform permutations
    let mut rng = thread_rng();
    let mut permuted_target = target_values.clone();
    let mut extreme_count = 0;

    for _ in 0..n_permutations {
        // Shuffle target values using Fisher-Yates algorithm
        let target_slice = permuted_target.as_slice_mut().unwrap();
        for i in (1..target_slice.len()).rev() {
            let j = rng.gen_range(0..=i);
            target_slice.swap(i, j);
        }

        // Compute test statistic for permuted data
        let (permuted_statistic, _) =
            compute_quadratic_form_test(feature_values, &permuted_target)?;

        if permuted_statistic >= original_statistic {
            extreme_count += 1;
        }
    }

    let p_value = (extreme_count + 1) as f64 / (n_permutations + 1) as f64;

    Ok((original_statistic, p_value))
}

/// Compute asymptotic chi-squared test
fn compute_asymptotic_chi_squared_test(
    feature_values: &ArrayView1<f64>,
    target_values: &Array1<f64>,
) -> Result<(f64, f64)> {
    // Use quadratic form test and chi-squared approximation
    let (test_statistic, _) = compute_quadratic_form_test(feature_values, target_values)?;

    // Degrees of freedom = 1 for single feature test
    let df = 1;
    let p_value = chi_squared_p_value(test_statistic, df);

    Ok((test_statistic, p_value))
}

/// Find the best split point for a feature using conditional inference
fn find_best_split_point(
    feature_values: &ArrayView1<f64>,
    target_values: &Array1<f64>,
) -> Result<f64> {
    if feature_values.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Empty feature values".to_string(),
        ));
    }

    // Find unique sorted values
    let mut values: Vec<f64> = feature_values.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values.dedup();

    if values.len() < 2 {
        return Ok(values[0]);
    }

    let mut best_split = values[0];
    let mut best_statistic = 0.0;

    // Try each potential split point
    for i in 0..(values.len() - 1) {
        let split_candidate = (values[i] + values[i + 1]) / 2.0;

        // Split data at this point
        let mut left_targets = Vec::new();
        let mut right_targets = Vec::new();

        for (j, &feature_val) in feature_values.iter().enumerate() {
            if feature_val <= split_candidate {
                left_targets.push(target_values[j]);
            } else {
                right_targets.push(target_values[j]);
            }
        }

        if left_targets.is_empty() || right_targets.is_empty() {
            continue;
        }

        // Compute separation statistic (simplified)
        let left_mean = left_targets.iter().sum::<f64>() / left_targets.len() as f64;
        let right_mean = right_targets.iter().sum::<f64>() / right_targets.len() as f64;
        let separation = (left_mean - right_mean).abs();

        if separation > best_statistic {
            best_statistic = separation;
            best_split = split_candidate;
        }
    }

    Ok(best_split)
}

/// Approximate Student's t-distribution CDF
fn student_t_cdf(t: f64, df: usize) -> f64 {
    if df == 0 {
        return 0.5;
    }

    // Simple approximation for t-distribution CDF
    // For production use, consider a proper statistical library
    let x = t / (df as f64).sqrt();
    0.5 * (1.0 + (2.0 / std::f64::consts::PI).sqrt() * x / (1.0 + x * x).sqrt())
}
