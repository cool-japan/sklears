//! Cross-validation iterators
//!
//! This module provides various cross-validation splitters organized into logical submodules:
//! - `cv::basic_cv` - Basic cross-validation (K-Fold, Stratified K-Fold, Leave-One-Out, Leave-P-Out)
//! - `cv::regression_cv` - Regression-specific cross-validation (Stratified Regression K-Fold)
//! - `cv::time_series_cv` - Time series cross-validation (Time Series Split, Blocked Time Series CV, Purged Group Time Series Split)
//! - `cv::group_cv` - Group-based cross-validation (Group K-Fold, Stratified Group K-Fold, etc.)
//! - `cv::shuffle_cv` - Shuffle-based cross-validation (Shuffle Split, Stratified Shuffle Split, Bootstrap CV, Monte Carlo CV)
//! - `cv::custom_cv` - Custom cross-validation (Custom Cross Validator, Block Cross Validator, Predefined Split)
//! - `cv::repeated_cv` - Repeated cross-validation (Repeated K-Fold, Repeated Stratified K-Fold)

use crate::cv;

// Re-export all cross-validation types for backward compatibility
pub use cv::{
    BlockCrossValidator,
    BlockedTimeSeriesCV,
    BootstrapCV,
    // Traits
    CrossValidator,
    // Custom CV
    CustomCrossValidator,
    // Group CV
    GroupKFold,
    GroupShuffleSplit,
    GroupStrategy,

    // Basic CV
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,

    MonteCarloCV,

    PredefinedSplit,

    PurgedGroupTimeSeriesSplit,

    RegressionCrossValidator,

    // Repeated CV
    RepeatedKFold,
    RepeatedStratifiedKFold,
    // Shuffle CV
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    // Regression CV
    StratifiedRegressionKFold,

    StratifiedShuffleSplit,
    // Time Series CV
    TimeSeriesSplit,
};
