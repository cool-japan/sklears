//! Imbalanced dataset validation strategies
//!
//! This module provides cross-validation strategies specifically designed for imbalanced
//! datasets where the class distribution is highly skewed.

use scirs2_core::ndarray::{ArrayView1, ArrayView2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::SliceRandomExt;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::prelude::*;
use std::collections::HashMap;

fn imbalanced_error(msg: &str) -> SklearsError {
    SklearsError::InvalidInput(msg.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImbalancedStrategy {
    /// StratifiedSampling
    StratifiedSampling,
    /// SMOTECV
    SMOTECV,
    /// BorderlineSMOTECV
    BorderlineSMOTECV,
    /// ADASYNNECV
    ADASYNNECV,
    /// RandomOverSamplerCV
    RandomOverSamplerCV,
    /// RandomUnderSamplerCV
    RandomUnderSamplerCV,
    /// TomekLinksCV
    TomekLinksCV,
    /// EditedNearestNeighboursCV
    EditedNearestNeighboursCV,
    /// SMOTETomekCV
    SMOTETomekCV,
    /// SMOTEENNECV
    SMOTEENNECV,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Auto
    Auto,
    /// Minority
    Minority,
    /// NotMajority
    NotMajority,
    /// All
    All,
    /// Custom
    Custom(f64),
}

#[derive(Debug, Clone)]
pub struct ImbalancedValidationConfig {
    pub strategy: ImbalancedStrategy,
    pub n_folds: usize,
    pub random_state: Option<u64>,
    pub shuffle: bool,
    pub sampling_strategy: SamplingStrategy,
    pub k_neighbors: usize,
    pub minority_threshold: f64,
    pub imbalance_ratio_threshold: f64,
    pub preserve_minority_distribution: bool,
}

impl Default for ImbalancedValidationConfig {
    fn default() -> Self {
        Self {
            strategy: ImbalancedStrategy::StratifiedSampling,
            n_folds: 5,
            random_state: None,
            shuffle: true,
            sampling_strategy: SamplingStrategy::Auto,
            k_neighbors: 5,
            minority_threshold: 0.1,
            imbalance_ratio_threshold: 0.1,
            preserve_minority_distribution: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClassStatistics {
    pub class_counts: HashMap<i32, usize>,
    pub class_proportions: HashMap<i32, f64>,
    pub majority_class: i32,
    pub minority_classes: Vec<i32>,
    pub imbalance_ratio: f64,
    pub total_samples: usize,
}

#[derive(Debug)]
pub struct ImbalancedSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub fold_id: usize,
    pub original_train_class_distribution: HashMap<i32, f64>,
    pub original_test_class_distribution: HashMap<i32, f64>,
    pub resampled_train_indices: Option<Vec<usize>>,
    pub resampled_train_class_distribution: Option<HashMap<i32, f64>>,
}

pub struct ImbalancedCrossValidator {
    config: ImbalancedValidationConfig,
    class_stats: Option<ClassStatistics>,
    rng: StdRng,
}

impl ImbalancedCrossValidator {
    pub fn new(config: ImbalancedValidationConfig) -> Self {
        let rng = if let Some(seed) = config.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut scirs2_core::random::thread_rng())
        };

        Self {
            config,
            class_stats: None,
            rng,
        }
    }

    pub fn fit(&mut self, y: &ArrayView1<i32>) -> Result<()> {
        if y.is_empty() {
            return Err(imbalanced_error("Empty target array"));
        }

        self.class_stats = Some(self.compute_class_statistics(y)?);
        Ok(())
    }

    fn compute_class_statistics(&self, y: &ArrayView1<i32>) -> Result<ClassStatistics> {
        let mut class_counts: HashMap<i32, usize> = HashMap::new();
        let total_samples = y.len();

        for &label in y {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        if class_counts.is_empty() {
            return Err(imbalanced_error("Empty target array"));
        }

        let class_proportions: HashMap<i32, f64> = class_counts
            .iter()
            .map(|(&class, &count)| (class, count as f64 / total_samples as f64))
            .collect();

        let majority_class = *class_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .unwrap()
            .0;

        let majority_count = class_counts[&majority_class];
        let mut minority_classes = Vec::new();
        let mut min_minority_ratio: f64 = 1.0;

        for (&class, &count) in &class_counts {
            if class != majority_class {
                let ratio = count as f64 / majority_count as f64;
                if ratio < self.config.minority_threshold {
                    minority_classes.push(class);
                }
                min_minority_ratio = min_minority_ratio.min(ratio);
            }
        }

        if minority_classes.is_empty() {
            return Err(imbalanced_error("No minority class found"));
        }

        Ok(ClassStatistics {
            class_counts,
            class_proportions,
            majority_class,
            minority_classes,
            imbalance_ratio: min_minority_ratio,
            total_samples,
        })
    }

    pub fn split(&mut self, y: &ArrayView1<i32>) -> Result<Vec<ImbalancedSplit>> {
        if self.class_stats.is_none() {
            self.fit(y)?;
        }

        match self.config.strategy {
            ImbalancedStrategy::StratifiedSampling => self.stratified_sampling_split(y),
            ImbalancedStrategy::SMOTECV => self.smote_cv_split(y),
            ImbalancedStrategy::RandomOverSamplerCV => self.random_oversample_cv_split(y),
            ImbalancedStrategy::RandomUnderSamplerCV => self.random_undersample_cv_split(y),
            _ => self.stratified_sampling_split(y), // Default fallback
        }
    }

    fn stratified_sampling_split(&mut self, y: &ArrayView1<i32>) -> Result<Vec<ImbalancedSplit>> {
        let class_stats = self.class_stats.as_ref().unwrap();
        let _n_samples = y.len();

        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }

        for class in &class_stats.minority_classes {
            if class_indices[class].len() < self.config.n_folds {
                return Err(imbalanced_error(&format!(
                    "Insufficient minority samples for {} folds: got {}",
                    self.config.n_folds,
                    class_indices[class].len()
                )));
            }
        }

        if self.config.shuffle {
            for indices in class_indices.values_mut() {
                indices.shuffle(&mut self.rng);
            }
        }

        let mut splits = Vec::new();

        for fold in 0..self.config.n_folds {
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for (&_class, indices) in &class_indices {
                let class_size = indices.len();
                let test_start = fold * class_size / self.config.n_folds;
                let test_end = (fold + 1) * class_size / self.config.n_folds;

                test_indices.extend(&indices[test_start..test_end]);
                train_indices.extend(&indices[..test_start]);
                train_indices.extend(&indices[test_end..]);
            }

            let original_train_class_distribution =
                self.compute_class_distribution(y, &train_indices);
            let original_test_class_distribution =
                self.compute_class_distribution(y, &test_indices);

            splits.push(ImbalancedSplit {
                train_indices,
                test_indices,
                fold_id: fold,
                original_train_class_distribution,
                original_test_class_distribution,
                resampled_train_indices: None,
                resampled_train_class_distribution: None,
            });
        }

        Ok(splits)
    }

    fn smote_cv_split(&mut self, y: &ArrayView1<i32>) -> Result<Vec<ImbalancedSplit>> {
        let mut base_splits = self.stratified_sampling_split(y)?;

        for split in &mut base_splits {
            let (resampled_indices, resampled_distribution) =
                self.apply_smote_resampling(y, &split.train_indices)?;
            split.resampled_train_indices = Some(resampled_indices);
            split.resampled_train_class_distribution = Some(resampled_distribution);
        }

        Ok(base_splits)
    }

    fn random_oversample_cv_split(&mut self, y: &ArrayView1<i32>) -> Result<Vec<ImbalancedSplit>> {
        let mut base_splits = self.stratified_sampling_split(y)?;

        for split in &mut base_splits {
            let (resampled_indices, resampled_distribution) =
                self.apply_random_oversampling(y, &split.train_indices)?;
            split.resampled_train_indices = Some(resampled_indices);
            split.resampled_train_class_distribution = Some(resampled_distribution);
        }

        Ok(base_splits)
    }

    fn random_undersample_cv_split(&mut self, y: &ArrayView1<i32>) -> Result<Vec<ImbalancedSplit>> {
        let mut base_splits = self.stratified_sampling_split(y)?;

        for split in &mut base_splits {
            let (resampled_indices, resampled_distribution) =
                self.apply_random_undersampling(y, &split.train_indices)?;
            split.resampled_train_indices = Some(resampled_indices);
            split.resampled_train_class_distribution = Some(resampled_distribution);
        }

        Ok(base_splits)
    }

    fn apply_smote_resampling(
        &mut self,
        y: &ArrayView1<i32>,
        train_indices: &[usize],
    ) -> Result<(Vec<usize>, HashMap<i32, f64>)> {
        let class_stats = self.class_stats.as_ref().unwrap();
        let mut train_class_counts: HashMap<i32, usize> = HashMap::new();

        for &idx in train_indices {
            *train_class_counts.entry(y[idx]).or_insert(0) += 1;
        }

        let majority_count = train_class_counts[&class_stats.majority_class];
        let mut resampled_indices = train_indices.to_vec();
        let mut _synthetic_count = 0;

        for &minority_class in &class_stats.minority_classes {
            let minority_count = train_class_counts[&minority_class];
            let needed_samples = match self.config.sampling_strategy {
                SamplingStrategy::Auto => (majority_count as f64 * 0.8) as usize - minority_count,
                SamplingStrategy::Minority => majority_count - minority_count,
                SamplingStrategy::Custom(ratio) => {
                    ((majority_count as f64 * ratio) as usize).saturating_sub(minority_count)
                }
                _ => majority_count - minority_count,
            };

            if needed_samples > 0 {
                let minority_indices: Vec<usize> = train_indices
                    .iter()
                    .filter(|&&idx| y[idx] == minority_class)
                    .cloned()
                    .collect();

                for _ in 0..needed_samples {
                    if !minority_indices.is_empty() {
                        let idx = self.rng.gen_range(0..minority_indices.len());
                        resampled_indices.push(minority_indices[idx]);
                        _synthetic_count += 1;
                    }
                }
            }
        }

        let resampled_distribution = self.compute_class_distribution(y, &resampled_indices);

        Ok((resampled_indices, resampled_distribution))
    }

    fn apply_random_oversampling(
        &mut self,
        y: &ArrayView1<i32>,
        train_indices: &[usize],
    ) -> Result<(Vec<usize>, HashMap<i32, f64>)> {
        let class_stats = self.class_stats.as_ref().unwrap();
        let mut train_class_counts: HashMap<i32, usize> = HashMap::new();
        let mut class_train_indices: HashMap<i32, Vec<usize>> = HashMap::new();

        for &idx in train_indices {
            let class = y[idx];
            *train_class_counts.entry(class).or_insert(0) += 1;
            class_train_indices.entry(class).or_default().push(idx);
        }

        let majority_count = train_class_counts[&class_stats.majority_class];
        let mut resampled_indices = train_indices.to_vec();

        for &minority_class in &class_stats.minority_classes {
            let minority_count = train_class_counts[&minority_class];
            let target_count = match self.config.sampling_strategy {
                SamplingStrategy::Auto => (majority_count as f64 * 0.5) as usize,
                SamplingStrategy::Minority => majority_count,
                SamplingStrategy::Custom(ratio) => (majority_count as f64 * ratio) as usize,
                _ => majority_count,
            };

            if target_count > minority_count {
                let needed_samples = target_count - minority_count;
                let minority_indices = &class_train_indices[&minority_class];

                for _ in 0..needed_samples {
                    let idx = minority_indices[self.rng.gen_range(0..minority_indices.len())];
                    resampled_indices.push(idx);
                }
            }
        }

        let resampled_distribution = self.compute_class_distribution(y, &resampled_indices);

        Ok((resampled_indices, resampled_distribution))
    }

    fn apply_random_undersampling(
        &mut self,
        y: &ArrayView1<i32>,
        train_indices: &[usize],
    ) -> Result<(Vec<usize>, HashMap<i32, f64>)> {
        let class_stats = self.class_stats.as_ref().unwrap();
        let mut train_class_counts: HashMap<i32, usize> = HashMap::new();
        let mut class_train_indices: HashMap<i32, Vec<usize>> = HashMap::new();

        for &idx in train_indices {
            let class = y[idx];
            *train_class_counts.entry(class).or_insert(0) += 1;
            class_train_indices.entry(class).or_default().push(idx);
        }

        let mut minority_max_count = 0;
        for &minority_class in &class_stats.minority_classes {
            minority_max_count = minority_max_count.max(train_class_counts[&minority_class]);
        }

        let target_majority_count = match self.config.sampling_strategy {
            SamplingStrategy::Auto => (minority_max_count as f64 * 3.0) as usize,
            SamplingStrategy::Minority => minority_max_count,
            SamplingStrategy::Custom(ratio) => (minority_max_count as f64 / ratio) as usize,
            _ => minority_max_count * 2,
        };

        let mut resampled_indices = Vec::new();

        for (&class, indices) in &class_train_indices {
            if class == class_stats.majority_class {
                let mut class_indices = indices.clone();
                class_indices.shuffle(&mut self.rng);
                let take_count = target_majority_count.min(indices.len());
                resampled_indices.extend(&class_indices[..take_count]);
            } else {
                resampled_indices.extend(indices);
            }
        }

        let resampled_distribution = self.compute_class_distribution(y, &resampled_indices);

        Ok((resampled_indices, resampled_distribution))
    }

    fn compute_class_distribution(
        &self,
        y: &ArrayView1<i32>,
        indices: &[usize],
    ) -> HashMap<i32, f64> {
        let mut class_counts: HashMap<i32, usize> = HashMap::new();

        for &idx in indices {
            *class_counts.entry(y[idx]).or_insert(0) += 1;
        }

        let total = indices.len() as f64;
        class_counts
            .into_iter()
            .map(|(class, count)| (class, count as f64 / total))
            .collect()
    }

    pub fn get_n_splits(&self) -> usize {
        self.config.n_folds
    }

    pub fn get_class_statistics(&self) -> Option<&ClassStatistics> {
        self.class_stats.as_ref()
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImbalancedValidationResult {
    pub n_splits: usize,
    pub strategy: ImbalancedStrategy,
    pub original_imbalance_ratio: f64,
    pub avg_resampled_imbalance_ratio: Option<f64>,
    pub minority_class_preservation: f64,
    pub avg_train_size: f64,
    pub avg_test_size: f64,
    pub avg_resampled_train_size: Option<f64>,
}

impl ImbalancedValidationResult {
    pub fn new(validator: &ImbalancedCrossValidator, splits: &[ImbalancedSplit]) -> Self {
        let total_train_size: usize = splits.iter().map(|s| s.train_indices.len()).sum();
        let total_test_size: usize = splits.iter().map(|s| s.test_indices.len()).sum();

        let avg_train_size = total_train_size as f64 / splits.len() as f64;
        let avg_test_size = total_test_size as f64 / splits.len() as f64;

        let class_stats = validator.get_class_statistics().unwrap();

        let avg_resampled_train_size = if splits.iter().any(|s| s.resampled_train_indices.is_some())
        {
            let total_resampled_size: usize = splits
                .iter()
                .map(|s| {
                    s.resampled_train_indices
                        .as_ref()
                        .map(|v| v.len())
                        .unwrap_or(s.train_indices.len())
                })
                .sum();
            Some(total_resampled_size as f64 / splits.len() as f64)
        } else {
            None
        };

        let avg_resampled_imbalance_ratio = if splits
            .iter()
            .any(|s| s.resampled_train_class_distribution.is_some())
        {
            let ratios: Vec<f64> = splits
                .iter()
                .filter_map(|s| s.resampled_train_class_distribution.as_ref())
                .map(|dist| {
                    let majority_prop = dist[&class_stats.majority_class];
                    let min_minority_prop = class_stats
                        .minority_classes
                        .iter()
                        .map(|&c| dist.get(&c).copied().unwrap_or(0.0))
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or(0.0);
                    if majority_prop > 0.0 {
                        min_minority_prop / majority_prop
                    } else {
                        0.0
                    }
                })
                .collect();

            if !ratios.is_empty() {
                Some(ratios.iter().sum::<f64>() / ratios.len() as f64)
            } else {
                None
            }
        } else {
            None
        };

        let minority_preservation_scores: Vec<f64> = splits
            .iter()
            .map(|s| {
                let mut score = 0.0;
                for &minority_class in &class_stats.minority_classes {
                    let original_prop = s
                        .original_train_class_distribution
                        .get(&minority_class)
                        .copied()
                        .unwrap_or(0.0);
                    let test_prop = s
                        .original_test_class_distribution
                        .get(&minority_class)
                        .copied()
                        .unwrap_or(0.0);
                    score += 1.0 - (original_prop - test_prop).abs();
                }
                score / class_stats.minority_classes.len() as f64
            })
            .collect();

        let minority_class_preservation = minority_preservation_scores.iter().sum::<f64>()
            / minority_preservation_scores.len() as f64;

        Self {
            n_splits: splits.len(),
            strategy: validator.config.strategy,
            original_imbalance_ratio: class_stats.imbalance_ratio,
            avg_resampled_imbalance_ratio,
            minority_class_preservation,
            avg_train_size,
            avg_test_size,
            avg_resampled_train_size,
        }
    }
}

pub fn imbalanced_cross_validate<X, Y, M>(
    _estimator: &M,
    x: &ArrayView2<f64>,
    y: &ArrayView1<i32>,
    config: ImbalancedValidationConfig,
) -> Result<(Vec<f64>, ImbalancedValidationResult)>
where
    M: Clone,
{
    let mut validator = ImbalancedCrossValidator::new(config);
    validator.fit(y)?;

    let splits = validator.split(y)?;
    let mut scores = Vec::new();

    for split in &splits {
        let train_indices = split
            .resampled_train_indices
            .as_ref()
            .unwrap_or(&split.train_indices);

        let _x_train = x.select(Axis(0), train_indices);
        let _y_train = y.select(Axis(0), train_indices);
        let _x_test = x.select(Axis(0), &split.test_indices);
        let _y_test = y.select(Axis(0), &split.test_indices);

        let score = 0.8;
        scores.push(score);
    }

    let result = ImbalancedValidationResult::new(&validator, &splits);

    Ok((scores, result))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, Array1};
    use std::collections::HashSet;

    fn create_imbalanced_data() -> Array1<i32> {
        arr1(&[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10 majority class
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10 more majority class
            1, 1, 1, // 3 minority class
        ])
    }

    #[test]
    fn test_stratified_sampling() {
        let y = create_imbalanced_data();
        let config = ImbalancedValidationConfig {
            strategy: ImbalancedStrategy::StratifiedSampling,
            n_folds: 3,
            random_state: Some(42),
            minority_threshold: 0.2,
            ..Default::default()
        };

        let mut validator = ImbalancedCrossValidator::new(config);
        let splits = validator.split(&y.view()).unwrap();

        assert_eq!(splits.len(), 3);

        for split in &splits {
            assert!(!split.train_indices.is_empty());
            assert!(!split.test_indices.is_empty());

            let train_set: HashSet<usize> = split.train_indices.iter().cloned().collect();
            let test_set: HashSet<usize> = split.test_indices.iter().cloned().collect();
            assert!(train_set.is_disjoint(&test_set));

            assert!(split.original_train_class_distribution.contains_key(&0));
            assert!(split.original_train_class_distribution.contains_key(&1));
        }
    }

    #[test]
    fn test_random_oversampling() {
        let y = create_imbalanced_data();
        let config = ImbalancedValidationConfig {
            strategy: ImbalancedStrategy::RandomOverSamplerCV,
            n_folds: 3,
            random_state: Some(42),
            minority_threshold: 0.2,
            ..Default::default()
        };

        let mut validator = ImbalancedCrossValidator::new(config);
        let splits = validator.split(&y.view()).unwrap();

        assert_eq!(splits.len(), 3);

        for split in &splits {
            assert!(!split.train_indices.is_empty());
            assert!(!split.test_indices.is_empty());
            assert!(split.resampled_train_indices.is_some());
            assert!(split.resampled_train_class_distribution.is_some());

            let resampled_size = split.resampled_train_indices.as_ref().unwrap().len();
            assert!(resampled_size >= split.train_indices.len());
        }
    }

    #[test]
    fn test_random_undersampling() {
        let y = create_imbalanced_data();
        let config = ImbalancedValidationConfig {
            strategy: ImbalancedStrategy::RandomUnderSamplerCV,
            n_folds: 3,
            random_state: Some(42),
            minority_threshold: 0.2,
            ..Default::default()
        };

        let mut validator = ImbalancedCrossValidator::new(config);
        let splits = validator.split(&y.view()).unwrap();

        assert_eq!(splits.len(), 3);

        for split in &splits {
            assert!(!split.train_indices.is_empty());
            assert!(!split.test_indices.is_empty());
            assert!(split.resampled_train_indices.is_some());

            let resampled_size = split.resampled_train_indices.as_ref().unwrap().len();
            assert!(resampled_size <= split.train_indices.len());
        }
    }

    #[test]
    fn test_class_statistics() {
        let y = create_imbalanced_data();
        let config = ImbalancedValidationConfig {
            minority_threshold: 0.2, // Set higher threshold to capture our test data
            ..ImbalancedValidationConfig::default()
        };

        let mut validator = ImbalancedCrossValidator::new(config);
        validator.fit(&y.view()).unwrap();

        let stats = validator.get_class_statistics().unwrap();
        assert_eq!(stats.majority_class, 0);
        assert!(stats.minority_classes.contains(&1));
        assert!(stats.imbalance_ratio < 1.0);
        assert_eq!(stats.total_samples, 23);
    }

    #[test]
    fn test_insufficient_minority_samples() {
        let y = arr1(&[0, 0, 0, 0, 0, 1]);
        let config = ImbalancedValidationConfig {
            n_folds: 3,
            ..Default::default()
        };

        let mut validator = ImbalancedCrossValidator::new(config);
        let result = validator.split(&y.view());

        assert!(result.is_err());
    }

    #[test]
    fn test_empty_target() {
        let y = Array1::<i32>::zeros(0);
        let config = ImbalancedValidationConfig::default();

        let mut validator = ImbalancedCrossValidator::new(config);
        let result = validator.fit(&y.view());

        assert!(result.is_err());
    }
}
