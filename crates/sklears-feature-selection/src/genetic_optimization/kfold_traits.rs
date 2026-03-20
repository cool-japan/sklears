//! # KFold - Trait Implementations
//!
//! This module contains trait implementations for `KFold`.
//!
//! ## Implemented Traits
//!
//! - `CrossValidator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl CrossValidator for KFold {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();
        let fold_size = n_samples / self.n_splits;
        for i in 0..self.n_splits {
            let test_start = i * fold_size;
            let test_end = if i == self.n_splits - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            let train_indices: Vec<usize> = (0..test_start).chain(test_end..n_samples).collect();
            splits.push((train_indices, test_indices));
        }
        splits
    }
}
