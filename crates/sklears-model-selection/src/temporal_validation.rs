//! Advanced time series validation with temporal dependencies
//!
//! This module provides sophisticated validation methods for time series data
//! that respect temporal dependencies and prevent data leakage.

use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

/// Configuration for temporal validation
#[derive(Debug, Clone)]
pub struct TemporalValidationConfig {
    /// Minimum gap between train and test sets (in time units)
    pub min_gap: usize,
    /// Maximum lookback window for features (prevents future data leakage)
    pub max_lookback: usize,
    /// Whether to use forward chaining validation
    pub forward_chaining: bool,
    /// Number of validation splits
    pub n_splits: usize,
    /// Percentage of data to use for testing in each split
    pub test_size: f64,
    /// Whether to allow overlapping training periods
    pub allow_overlap: bool,
    /// Seasonal period for seasonal adjustments
    pub seasonal_period: Option<usize>,
}

impl Default for TemporalValidationConfig {
    fn default() -> Self {
        Self {
            min_gap: 1,
            max_lookback: 10,
            forward_chaining: true,
            n_splits: 5,
            test_size: 0.2,
            allow_overlap: false,
            seasonal_period: None,
        }
    }
}

/// Time series cross-validator with temporal dependency awareness
#[derive(Debug, Clone)]
pub struct TemporalCrossValidator {
    config: TemporalValidationConfig,
}

impl TemporalCrossValidator {
    pub fn new(config: TemporalValidationConfig) -> Self {
        Self { config }
    }

    /// Generate temporal splits that respect time dependencies
    pub fn split(
        &self,
        n_samples: usize,
        time_index: &[usize],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if time_index.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Time index length must match number of samples".to_string(),
            ));
        }

        let sorted_indices = self.sort_by_time(time_index)?;

        let mut splits = if self.config.forward_chaining {
            self.forward_chaining_splits(&sorted_indices)?
        } else {
            self.sliding_window_splits(&sorted_indices)?
        };

        // Apply temporal constraints
        self.apply_temporal_constraints(&mut splits, time_index)?;

        Ok(splits)
    }

    /// Forward chaining: each training set includes all previous data
    fn forward_chaining_splits(
        &self,
        sorted_indices: &[usize],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();
        let n_samples = sorted_indices.len();
        let test_samples = (n_samples as f64 * self.config.test_size) as usize;

        for i in 0..self.config.n_splits {
            let test_start = n_samples - (self.config.n_splits - i) * test_samples;
            let test_end = test_start + test_samples;

            if test_start < self.config.max_lookback {
                continue;
            }

            let train_end = test_start.saturating_sub(self.config.min_gap);

            let train_indices = sorted_indices[0..train_end].to_vec();
            let test_indices = if test_end <= n_samples {
                sorted_indices[test_start..test_end].to_vec()
            } else {
                sorted_indices[test_start..].to_vec()
            };

            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }

        Ok(splits)
    }

    /// Sliding window: fixed-size training windows
    fn sliding_window_splits(
        &self,
        sorted_indices: &[usize],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();
        let n_samples = sorted_indices.len();
        let test_samples = (n_samples as f64 * self.config.test_size) as usize;
        let train_samples = n_samples - test_samples - self.config.min_gap;

        let step_size = if self.config.allow_overlap {
            test_samples / 2
        } else {
            test_samples
        };

        let mut start = self.config.max_lookback;

        while start + train_samples + self.config.min_gap + test_samples <= n_samples {
            let train_end = start + train_samples;
            let test_start = train_end + self.config.min_gap;
            let test_end = test_start + test_samples;

            let train_indices = sorted_indices[start..train_end].to_vec();
            let test_indices = sorted_indices[test_start..test_end].to_vec();

            splits.push((train_indices, test_indices));
            start += step_size;
        }

        Ok(splits)
    }

    /// Sort indices by time
    fn sort_by_time(&self, time_index: &[usize]) -> Result<Vec<usize>> {
        let mut indexed_times: Vec<(usize, usize)> = time_index
            .iter()
            .enumerate()
            .map(|(idx, &time)| (idx, time))
            .collect();

        indexed_times.sort_by_key(|&(_, time)| time);

        Ok(indexed_times.into_iter().map(|(idx, _)| idx).collect())
    }

    /// Apply temporal constraints to ensure no data leakage
    fn apply_temporal_constraints(
        &self,
        splits: &mut [(Vec<usize>, Vec<usize>)],
        time_index: &[usize],
    ) -> Result<()> {
        for (train_indices, test_indices) in splits.iter_mut() {
            // Ensure no future information in training set
            let max_test_time = test_indices
                .iter()
                .map(|&idx| time_index[idx])
                .min()
                .unwrap_or(0);

            train_indices.retain(|&idx| time_index[idx] + self.config.min_gap <= max_test_time);

            // Apply lookback window constraint
            if let Some(min_train_time) = train_indices.iter().map(|&idx| time_index[idx]).max() {
                let cutoff_time = min_train_time.saturating_sub(self.config.max_lookback);
                train_indices.retain(|&idx| time_index[idx] >= cutoff_time);
            }
        }

        Ok(())
    }
}

/// Seasonal cross-validator for time series with seasonal patterns
#[derive(Debug, Clone)]
pub struct SeasonalCrossValidator {
    config: TemporalValidationConfig,
    seasonal_period: usize,
}

impl SeasonalCrossValidator {
    pub fn new(config: TemporalValidationConfig, seasonal_period: usize) -> Self {
        Self {
            config,
            seasonal_period,
        }
    }

    /// Generate seasonal splits that maintain seasonal patterns
    pub fn split(
        &self,
        _n_samples: usize,
        time_index: &[usize],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();
        let sorted_indices = self.sort_by_time(time_index)?;

        // Group indices by seasonal period
        let seasonal_groups = self.group_by_season(&sorted_indices, time_index)?;

        // Create splits ensuring each split has representation from all seasons
        for split_idx in 0..self.config.n_splits {
            let (train_indices, test_indices) =
                self.create_seasonal_split(&seasonal_groups, split_idx, time_index)?;

            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }

        Ok(splits)
    }

    fn sort_by_time(&self, time_index: &[usize]) -> Result<Vec<usize>> {
        let mut indexed_times: Vec<(usize, usize)> = time_index
            .iter()
            .enumerate()
            .map(|(idx, &time)| (idx, time))
            .collect();

        indexed_times.sort_by_key(|&(_, time)| time);

        Ok(indexed_times.into_iter().map(|(idx, _)| idx).collect())
    }

    fn group_by_season(
        &self,
        sorted_indices: &[usize],
        time_index: &[usize],
    ) -> Result<HashMap<usize, Vec<usize>>> {
        let mut seasonal_groups: HashMap<usize, Vec<usize>> = HashMap::new();

        for &idx in sorted_indices {
            let season = time_index[idx] % self.seasonal_period;
            seasonal_groups.entry(season).or_default().push(idx);
        }

        Ok(seasonal_groups)
    }

    fn create_seasonal_split(
        &self,
        seasonal_groups: &HashMap<usize, Vec<usize>>,
        split_idx: usize,
        time_index: &[usize],
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        for indices in seasonal_groups.values() {
            let n_season_samples = indices.len();
            let test_size = (n_season_samples as f64 * self.config.test_size) as usize;
            let samples_per_split = test_size.max(1);

            let test_start = split_idx * samples_per_split;
            let test_end = ((split_idx + 1) * samples_per_split).min(n_season_samples);

            if test_start < n_season_samples {
                // Add test samples
                test_indices.extend_from_slice(&indices[test_start..test_end]);

                // Add training samples (excluding test and respecting temporal constraints)
                for (i, &idx) in indices.iter().enumerate() {
                    if i < test_start || i >= test_end {
                        // Check temporal constraints
                        let sample_time = time_index[idx];
                        let can_use_for_training = test_indices.iter().all(|&test_idx| {
                            sample_time + self.config.min_gap <= time_index[test_idx]
                        });

                        if can_use_for_training {
                            train_indices.push(idx);
                        }
                    }
                }
            }
        }

        Ok((train_indices, test_indices))
    }
}

/// Blocked temporal cross-validator for handling irregular time series
#[derive(Debug, Clone)]
pub struct BlockedTemporalCV {
    config: TemporalValidationConfig,
    block_size: usize,
}

impl BlockedTemporalCV {
    pub fn new(config: TemporalValidationConfig, block_size: usize) -> Self {
        Self { config, block_size }
    }

    /// Generate blocked temporal splits
    pub fn split(
        &self,
        _n_samples: usize,
        time_index: &[usize],
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let sorted_indices = self.sort_by_time(time_index)?;
        let blocks = self.create_blocks(&sorted_indices)?;

        let mut splits = Vec::new();

        for i in 0..self.config.n_splits {
            let (train_blocks, test_blocks) = self.select_blocks(&blocks, i)?;

            let train_indices: Vec<usize> = train_blocks.into_iter().flatten().collect();
            let test_indices: Vec<usize> = test_blocks.into_iter().flatten().collect();

            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }

        Ok(splits)
    }

    fn sort_by_time(&self, time_index: &[usize]) -> Result<Vec<usize>> {
        let mut indexed_times: Vec<(usize, usize)> = time_index
            .iter()
            .enumerate()
            .map(|(idx, &time)| (idx, time))
            .collect();

        indexed_times.sort_by_key(|&(_, time)| time);

        Ok(indexed_times.into_iter().map(|(idx, _)| idx).collect())
    }

    fn create_blocks(&self, sorted_indices: &[usize]) -> Result<Vec<Vec<usize>>> {
        let mut blocks = Vec::new();

        for chunk in sorted_indices.chunks(self.block_size) {
            blocks.push(chunk.to_vec());
        }

        Ok(blocks)
    }

    fn select_blocks(
        &self,
        blocks: &[Vec<usize>],
        split_idx: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        let n_blocks = blocks.len();
        let test_blocks_count = (n_blocks as f64 * self.config.test_size) as usize;
        let test_blocks_count = test_blocks_count.max(1);

        let test_start = split_idx * test_blocks_count;
        let test_end = ((split_idx + 1) * test_blocks_count).min(n_blocks);

        let mut train_blocks = Vec::new();
        let mut test_blocks = Vec::new();

        for (i, block) in blocks.iter().enumerate() {
            if i >= test_start && i < test_end {
                test_blocks.push(block.clone());
            } else if i < test_start.saturating_sub(self.config.min_gap) {
                train_blocks.push(block.clone());
            }
        }

        Ok((train_blocks, test_blocks))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_cross_validator() {
        let config = TemporalValidationConfig::default();
        let cv = TemporalCrossValidator::new(config);

        let time_index: Vec<usize> = (0..100).collect();
        let splits = cv.split(100, &time_index).unwrap();

        assert!(!splits.is_empty(), "Should generate at least one split");

        // Check temporal constraints
        for (train_indices, test_indices) in &splits {
            let max_train_time = train_indices
                .iter()
                .map(|&i| time_index[i])
                .max()
                .unwrap_or(0);
            let min_test_time = test_indices
                .iter()
                .map(|&i| time_index[i])
                .min()
                .unwrap_or(usize::MAX);

            assert!(
                max_train_time + 1 <= min_test_time,
                "Temporal constraint violated"
            );
        }
    }

    #[test]
    fn test_seasonal_cross_validator() {
        let config = TemporalValidationConfig::default();
        let cv = SeasonalCrossValidator::new(config, 12); // Monthly seasonality

        let time_index: Vec<usize> = (0..120).collect(); // 10 years of monthly data
        let splits = cv.split(120, &time_index).unwrap();

        assert!(!splits.is_empty(), "Should generate at least one split");
    }

    #[test]
    fn test_blocked_temporal_cv() {
        let config = TemporalValidationConfig::default();
        let cv = BlockedTemporalCV::new(config, 10);

        let time_index: Vec<usize> = (0..100).collect();
        let splits = cv.split(100, &time_index).unwrap();

        assert!(!splits.is_empty(), "Should generate at least one split");
    }
}
