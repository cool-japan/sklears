//! # FeatureSelectionPipeline - Trait Implementations
//!
//! This module contains trait implementations for `FeatureSelectionPipeline`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Estimator`
//! - `Fit`
//! - `Transform`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use sklears_core::error::SklearsError;
use sklears_core::traits::{Estimator, Fit, Transform};

use super::functions::Result;
use super::types::{FeatureSelectionPipeline, Trained, Untrained};

impl Default for FeatureSelectionPipeline<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for FeatureSelectionPipeline<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, f64>> for FeatureSelectionPipeline<Untrained> {
    type Fitted = FeatureSelectionPipeline<Trained>;
    fn fit(self, X: &ArrayView2<'_, f64>, y: &ArrayView1<'_, f64>) -> Result<Self::Fitted> {
        self.fit(*X, *y)
    }
}

impl Transform<ArrayView2<'_, f64>, Array2<f64>> for FeatureSelectionPipeline<Trained> {
    fn transform(&self, X: &ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.transform(*X)
    }
}
