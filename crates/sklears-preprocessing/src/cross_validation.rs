//! Cross-Validation Utilities for Preprocessing
//!
//! Provides cross-validation support for preprocessing parameter tuning,
//! including grid search and random search for optimal preprocessing parameters.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::{seeded_rng, Distribution};
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// K-Fold cross-validation splitter
#[derive(Debug, Clone)]
pub struct KFold {
    /// Number of folds
    pub n_splits: usize,
    /// Whether to shuffle data before splitting
    pub shuffle: bool,
    /// Random seed for shuffling
    pub random_state: Option<u64>,
}

impl KFold {
    /// Create a new K-Fold splitter
    pub fn new(n_splits: usize, shuffle: bool, random_state: Option<u64>) -> Self {
        Self {
            n_splits,
            shuffle,
            random_state,
        }
    }

    /// Generate train/test splits
    pub fn split(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>, SklearsError> {
        if n_samples < self.n_splits {
            return Err(SklearsError::InvalidInput(format!(
                "Cannot split {} samples into {} folds",
                n_samples, self.n_splits
            )));
        }

        let mut indices: Vec<usize> = (0..n_samples).collect();

        if self.shuffle {
            use std::time::{SystemTime, UNIX_EPOCH};

            let seed = self.random_state.unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            });

            let mut rng = seeded_rng(seed);

            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                let uniform = Uniform::new(0, i + 1).unwrap();
                let j = uniform.sample(&mut rng);
                indices.swap(i, j);
            }
        }

        let fold_size = n_samples / self.n_splits;
        let mut splits = Vec::new();

        for fold_idx in 0..self.n_splits {
            let test_start = fold_idx * fold_size;
            let test_end = if fold_idx == self.n_splits - 1 {
                n_samples
            } else {
                (fold_idx + 1) * fold_size
            };

            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = indices[..test_start]
                .iter()
                .chain(&indices[test_end..])
                .copied()
                .collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }
}

/// Stratified K-Fold cross-validation splitter
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    /// Number of folds
    pub n_splits: usize,
    /// Whether to shuffle data before splitting
    pub shuffle: bool,
    /// Random seed for shuffling
    pub random_state: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new Stratified K-Fold splitter
    pub fn new(n_splits: usize, shuffle: bool, random_state: Option<u64>) -> Self {
        Self {
            n_splits,
            shuffle,
            random_state,
        }
    }

    /// Generate stratified train/test splits
    pub fn split(&self, y: &Array1<i32>) -> Result<Vec<(Vec<usize>, Vec<usize>)>, SklearsError> {
        let n_samples = y.len();

        if n_samples < self.n_splits {
            return Err(SklearsError::InvalidInput(format!(
                "Cannot split {} samples into {} folds",
                n_samples, self.n_splits
            )));
        }

        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }

        // Shuffle within each class
        if self.shuffle {
            use std::time::{SystemTime, UNIX_EPOCH};

            let seed = self.random_state.unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            });

            let mut rng = seeded_rng(seed);

            for indices in class_indices.values_mut() {
                for i in (1..indices.len()).rev() {
                    let uniform = Uniform::new(0, i + 1).unwrap();
                    let j = uniform.sample(&mut rng);
                    indices.swap(i, j);
                }
            }
        }

        // Create splits maintaining class distribution
        let mut splits: Vec<(Vec<usize>, Vec<usize>)> = vec![];

        for fold_idx in 0..self.n_splits {
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for indices in class_indices.values() {
                let fold_size = indices.len() / self.n_splits;
                let test_start = fold_idx * fold_size;
                let test_end = if fold_idx == self.n_splits - 1 {
                    indices.len()
                } else {
                    (fold_idx + 1) * fold_size
                };

                test_indices.extend(&indices[test_start..test_end]);
                train_indices.extend(&indices[..test_start]);
                train_indices.extend(&indices[test_end..]);
            }

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }
}

/// Cross-validation score result
#[derive(Debug, Clone)]
pub struct CVScore {
    /// Mean score across folds
    pub mean: f64,
    /// Standard deviation of scores
    pub std: f64,
    /// Individual fold scores
    pub scores: Vec<f64>,
}

/// Grid search parameter specification
#[derive(Debug, Clone)]
pub struct ParameterGrid {
    parameters: HashMap<String, Vec<f64>>,
}

impl ParameterGrid {
    /// Create a new parameter grid
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    /// Add a parameter with possible values
    pub fn add_parameter(mut self, name: String, values: Vec<f64>) -> Self {
        self.parameters.insert(name, values);
        self
    }

    /// Generate all parameter combinations
    pub fn combinations(&self) -> Vec<HashMap<String, f64>> {
        if self.parameters.is_empty() {
            return vec![HashMap::new()];
        }

        let mut result = vec![HashMap::new()];

        for (param_name, param_values) in &self.parameters {
            let mut new_result = Vec::new();

            for combination in &result {
                for &value in param_values {
                    let mut new_combination = combination.clone();
                    new_combination.insert(param_name.clone(), value);
                    new_result.push(new_combination);
                }
            }

            result = new_result;
        }

        result
    }

    /// Get total number of combinations
    pub fn n_combinations(&self) -> usize {
        if self.parameters.is_empty() {
            return 0;
        }

        self.parameters.values().map(|v| v.len()).product()
    }
}

impl Default for ParameterGrid {
    fn default() -> Self {
        Self::new()
    }
}

/// Random search parameter specification
#[derive(Debug, Clone)]
pub struct ParameterDistribution {
    parameters: HashMap<String, (f64, f64)>, // (min, max) for uniform distribution
}

impl ParameterDistribution {
    /// Create a new parameter distribution
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    /// Add a parameter with range
    pub fn add_parameter(mut self, name: String, min: f64, max: f64) -> Self {
        self.parameters.insert(name, (min, max));
        self
    }

    /// Sample random parameters
    pub fn sample(&self, n_iter: usize, random_state: Option<u64>) -> Vec<HashMap<String, f64>> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let seed = random_state.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

        let mut rng = seeded_rng(seed);

        (0..n_iter)
            .map(|_| {
                self.parameters
                    .iter()
                    .map(|(name, &(min, max))| {
                        let uniform = Uniform::new_inclusive(min, max).unwrap();
                        (name.clone(), uniform.sample(&mut rng))
                    })
                    .collect()
            })
            .collect()
    }
}

impl Default for ParameterDistribution {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluation metric for preprocessing quality
pub trait PreprocessingMetric {
    /// Evaluate preprocessing quality
    fn evaluate(&self, x_original: &Array2<f64>, x_transformed: &Array2<f64>) -> f64;
}

/// Variance preservation metric
pub struct VariancePreservationMetric;

impl PreprocessingMetric for VariancePreservationMetric {
    fn evaluate(&self, x_original: &Array2<f64>, x_transformed: &Array2<f64>) -> f64 {
        let mut total_variance_ratio = 0.0;

        for j in 0..x_original.ncols() {
            let original_col = x_original.column(j);
            let transformed_col = x_transformed.column(j);

            let original_var = Self::compute_variance(original_col);
            let transformed_var = Self::compute_variance(transformed_col);

            if original_var > 1e-10 {
                total_variance_ratio += transformed_var / original_var;
            }
        }

        total_variance_ratio / x_original.ncols() as f64
    }
}

impl VariancePreservationMetric {
    fn compute_variance<'a, I>(values: I) -> f64
    where
        I: IntoIterator<Item = &'a f64>,
    {
        let vals: Vec<f64> = values
            .into_iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect();

        if vals.is_empty() {
            return 0.0;
        }

        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64
    }
}

/// Information preservation metric (measures mutual information preservation)
pub struct InformationPreservationMetric;

impl PreprocessingMetric for InformationPreservationMetric {
    fn evaluate(&self, x_original: &Array2<f64>, x_transformed: &Array2<f64>) -> f64 {
        // Simplified: Use correlation as proxy for information preservation
        let mut total_correlation = 0.0;
        let mut count = 0;

        for j in 0..x_original.ncols().min(x_transformed.ncols()) {
            let corr = Self::compute_correlation(x_original, x_transformed, j);
            if !corr.is_nan() {
                total_correlation += corr.abs();
                count += 1;
            }
        }

        if count > 0 {
            total_correlation / count as f64
        } else {
            0.0
        }
    }
}

impl InformationPreservationMetric {
    fn compute_correlation(x1: &Array2<f64>, x2: &Array2<f64>, col_idx: usize) -> f64 {
        let col1 = x1.column(col_idx);
        let col2 = x2.column(col_idx);

        let pairs: Vec<(f64, f64)> = col1
            .iter()
            .zip(col2.iter())
            .filter(|(a, b)| !a.is_nan() && !b.is_nan())
            .map(|(&a, &b)| (a, b))
            .collect();

        if pairs.len() < 2 {
            return 0.0;
        }

        let mean1 = pairs.iter().map(|(a, _)| a).sum::<f64>() / pairs.len() as f64;
        let mean2 = pairs.iter().map(|(_, b)| b).sum::<f64>() / pairs.len() as f64;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (a, b) in &pairs {
            let d1 = a - mean1;
            let d2 = b - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        if var1 < 1e-10 || var2 < 1e-10 {
            return 0.0;
        }

        cov / (var1 * var2).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::essentials::Normal;
    use scirs2_core::random::{seeded_rng, Distribution};

    fn generate_test_data(nrows: usize, ncols: usize, seed: u64) -> Array2<f64> {
        let mut rng = seeded_rng(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let data: Vec<f64> = (0..nrows * ncols)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Array2::from_shape_vec((nrows, ncols), data).unwrap()
    }

    #[test]
    fn test_kfold_split() {
        let kfold = KFold::new(5, false, Some(42));
        let splits = kfold.split(100).unwrap();

        assert_eq!(splits.len(), 5);

        for (train, test) in &splits {
            assert!(train.len() > 0);
            assert!(test.len() > 0);
            assert_eq!(train.len() + test.len(), 100);
        }
    }

    #[test]
    fn test_kfold_shuffle() {
        let kfold1 = KFold::new(3, true, Some(42));
        let splits1 = kfold1.split(30).unwrap();

        let kfold2 = KFold::new(3, false, None);
        let splits2 = kfold2.split(30).unwrap();

        // Shuffled and non-shuffled should be different
        let different = splits1[0].0 != splits2[0].0;
        assert!(different);
    }

    #[test]
    fn test_stratified_kfold() {
        let y = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);

        let stratified = StratifiedKFold::new(3, false, Some(42));
        let splits = stratified.split(&y).unwrap();

        assert_eq!(splits.len(), 3);

        // Check that each split maintains class distribution
        for (train_indices, test_indices) in &splits {
            let _train_classes: Vec<i32> = train_indices.iter().map(|&i| y[i]).collect();
            let test_classes: Vec<i32> = test_indices.iter().map(|&i| y[i]).collect();

            // Count classes in test set
            let test_0 = test_classes.iter().filter(|&&c| c == 0).count();
            let test_1 = test_classes.iter().filter(|&&c| c == 1).count();
            let test_2 = test_classes.iter().filter(|&&c| c == 2).count();

            // Each class should appear roughly equally
            assert!(test_0 > 0);
            assert!(test_1 > 0);
            assert!(test_2 > 0);
        }
    }

    #[test]
    fn test_parameter_grid() {
        let grid = ParameterGrid::new()
            .add_parameter("alpha".to_string(), vec![0.1, 1.0, 10.0])
            .add_parameter("beta".to_string(), vec![0.5, 1.5]);

        let combinations = grid.combinations();

        assert_eq!(combinations.len(), 6); // 3 * 2 = 6
        assert_eq!(grid.n_combinations(), 6);

        // Check that all combinations are present
        let has_alpha_0_1 = combinations.iter().any(|c| c.get("alpha") == Some(&0.1));
        assert!(has_alpha_0_1);
    }

    #[test]
    fn test_parameter_distribution() {
        let dist = ParameterDistribution::new()
            .add_parameter("alpha".to_string(), 0.0, 1.0)
            .add_parameter("beta".to_string(), 0.0, 10.0);

        let samples = dist.sample(10, Some(42));

        assert_eq!(samples.len(), 10);

        for sample in &samples {
            let alpha = sample.get("alpha").unwrap();
            let beta = sample.get("beta").unwrap();

            assert!(*alpha >= 0.0 && *alpha <= 1.0);
            assert!(*beta >= 0.0 && *beta <= 10.0);
        }
    }

    #[test]
    fn test_variance_preservation_metric() {
        let x_original = generate_test_data(100, 5, 42);
        let x_transformed = x_original.clone();

        let metric = VariancePreservationMetric;
        let score = metric.evaluate(&x_original, &x_transformed);

        // Same data should have score close to 1.0
        assert!((score - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_information_preservation_metric() {
        let x_original = generate_test_data(100, 5, 123);
        let x_transformed = x_original.clone();

        let metric = InformationPreservationMetric;
        let score = metric.evaluate(&x_original, &x_transformed);

        // Same data should have high correlation
        assert!(score > 0.9);
    }

    #[test]
    fn test_kfold_edge_case_small_dataset() {
        let kfold = KFold::new(5, false, Some(42));
        let result = kfold.split(3);

        assert!(result.is_err());
    }

    #[test]
    fn test_empty_parameter_grid() {
        let grid = ParameterGrid::new();
        let combinations = grid.combinations();

        assert_eq!(combinations.len(), 1);
        assert!(combinations[0].is_empty());
        assert_eq!(grid.n_combinations(), 0);
    }
}
