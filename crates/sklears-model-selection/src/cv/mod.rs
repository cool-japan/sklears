//! Cross-validation iterators and utilities

pub mod basic_cv;
pub mod custom_cv;
pub mod group_cv;
pub mod regression_cv;
pub mod repeated_cv;
pub mod shuffle_cv;
pub mod time_series_cv;

use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;

/// Trait for cross-validation iterators
pub trait CrossValidator: Send + Sync {
    /// Returns the number of splits
    fn n_splits(&self) -> usize;

    /// Generate train/test indices for cross-validation
    ///
    /// For cross-validators that don't need y (like KFold), pass None.
    /// For stratified cross-validators, y should contain integer class labels.
    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)>;
}

/// Extended trait for regression cross-validation that works with continuous targets
pub trait RegressionCrossValidator: Send + Sync {
    /// Returns the number of splits
    fn n_splits(&self) -> usize;

    /// Generate train/test indices for cross-validation with continuous targets
    fn split_regression(
        &self,
        n_samples: usize,
        y: &Array1<Float>,
    ) -> Vec<(Vec<usize>, Vec<usize>)>;
}

// Re-export all cross-validators
pub use basic_cv::{KFold, LeaveOneOut, LeavePOut, StratifiedKFold};
pub use custom_cv::{BlockCrossValidator, CustomCrossValidator, PredefinedSplit};
pub use group_cv::{
    GroupKFold, GroupShuffleSplit, GroupStrategy, LeaveOneGroupOut, LeavePGroupsOut,
    StratifiedGroupKFold,
};
pub use regression_cv::StratifiedRegressionKFold;
pub use repeated_cv::{RepeatedKFold, RepeatedStratifiedKFold};
pub use shuffle_cv::{BootstrapCV, MonteCarloCV, ShuffleSplit, StratifiedShuffleSplit};
pub use time_series_cv::{BlockedTimeSeriesCV, PurgedGroupTimeSeriesSplit, TimeSeriesSplit};
