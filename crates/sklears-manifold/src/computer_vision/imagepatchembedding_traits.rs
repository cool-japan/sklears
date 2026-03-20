//! # ImagePatchEmbedding - Trait Implementations
//!
//! This module contains trait implementations for `ImagePatchEmbedding`.
//!
//! ## Implemented Traits
//!
//! - `Estimator`
//! - `Fit`
//! - `Transform`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

use super::types::{ImagePatchEmbedding, TrainedPatchEmbedding};

impl Estimator for ImagePatchEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for ImagePatchEmbedding<Untrained> {
    type Fitted = ImagePatchEmbedding<TrainedPatchEmbedding>;
    fn fit(self, image: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (height, width) = image.dim();
        if height == 0 || width == 0 {
            return Err(SklearsError::InvalidInput("Empty image".to_string()));
        }
        let patches = self.extract_patches(&image.view())?;
        let trained_state = self.compute_patch_embedding(&patches.view())?;
        Ok(ImagePatchEmbedding {
            patch_size: self.patch_size,
            stride: self.stride,
            n_components: self.n_components,
            embedding_method: self.embedding_method,
            state: trained_state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for ImagePatchEmbedding<TrainedPatchEmbedding> {
    fn transform(&self, image: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.transform_image(&image.view())
    }
}
