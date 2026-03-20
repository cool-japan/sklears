//! # ManifoldImageDenoising - Trait Implementations
//!
//! This module contains trait implementations for `ManifoldImageDenoising`.
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

use super::types::{ImagePatchEmbedding, ManifoldImageDenoising, TrainedImageDenoising};

impl Estimator for ManifoldImageDenoising<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for ManifoldImageDenoising<Untrained> {
    type Fitted = ManifoldImageDenoising<TrainedImageDenoising>;
    fn fit(self, clean_image: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let image_patch_embedding = ImagePatchEmbedding::<Untrained>::new(self.patch_size);
        let clean_patches = image_patch_embedding.extract_patches(&clean_image.view())?;
        let patch_embedding = Array2::eye(clean_patches.ncols());
        let trained_state = TrainedImageDenoising {
            patch_size: self.patch_size,
            n_components: self.n_components,
            overlap_threshold: self.overlap_threshold,
            patch_embedding,
            clean_patches,
        };
        Ok(ManifoldImageDenoising {
            patch_size: self.patch_size,
            n_components: self.n_components,
            overlap_threshold: self.overlap_threshold,
            state: trained_state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for ManifoldImageDenoising<TrainedImageDenoising> {
    fn transform(&self, noisy_image: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.denoise_image(&noisy_image.view())
    }
}
