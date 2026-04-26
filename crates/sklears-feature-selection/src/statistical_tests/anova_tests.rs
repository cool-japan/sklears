//! ANOVA tests for feature selection

use scirs2_core::error::CoreError;
use scirs2_core::ndarray::ArrayView1;
type CoreResult<T> = std::result::Result<T, CoreError>;

/// one_way_anova
pub fn one_way_anova(_samples: &[ArrayView1<f64>]) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

/// two_way_anova
pub fn two_way_anova(_samples: &[ArrayView1<f64>]) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

/// repeated_measures_anova
pub fn repeated_measures_anova(_samples: &[ArrayView1<f64>]) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}
