//! View and feature group structures for Multi-View Discriminant Analysis

use super::types::{FeatureType, HeterogeneousDistance, PreprocessingMethod};
use sklears_core::types::Float;

/// Feature group specification for heterogeneous data
#[derive(Debug, Clone)]
pub struct FeatureGroup {
    /// Feature indices in this group
    pub feature_indices: Vec<usize>,
    /// Type of features in this group
    pub feature_type: FeatureType,
    /// Preprocessing method for this group
    pub preprocessing: PreprocessingMethod,
    /// Distance metric for this group
    pub distance_metric: HeterogeneousDistance,
    /// Weight of this group in the overall distance
    pub group_weight: Float,
    /// Group name or identifier
    pub name: String,
}

/// View information for multi-view data
#[derive(Debug, Clone)]
pub struct ViewInfo {
    /// Starting column index for this view
    pub start_col: usize,
    /// Ending column index for this view (exclusive)
    pub end_col: usize,
    /// View-specific weight
    pub weight: Float,
    /// View name or identifier
    pub name: String,
    /// Feature groups in this view (for heterogeneous features)
    pub feature_groups: Option<Vec<FeatureGroup>>,
    /// Whether this view contains heterogeneous features
    pub is_heterogeneous: bool,
}
