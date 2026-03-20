//! Base classes for feature selection

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result as SklResult, traits::Transform};

/// Base trait for feature selectors
pub trait SelectorMixin: Transform<Array2<f64>> {
    /// Get support mask
    fn get_support(&self) -> SklResult<Array1<bool>>;

    /// Transform by selecting features
    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>>;
}

/// Trait for getting selected feature indices from trained selectors
pub trait FeatureSelector {
    /// Get selected feature indices
    fn selected_features(&self) -> &Vec<usize>;
}
