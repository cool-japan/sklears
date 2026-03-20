//! # FaceManifoldLearning - Trait Implementations
//!
//! This module contains trait implementations for `FaceManifoldLearning`.
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

use super::types::{FaceManifoldLearning, TrainedFaceManifold};

impl Estimator for FaceManifoldLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array3<f64>, ()> for FaceManifoldLearning<Untrained> {
    type Fitted = FaceManifoldLearning<TrainedFaceManifold>;
    fn fit(self, faces: &Array3<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_faces, height, width) = faces.dim();
        if n_faces == 0 || height == 0 || width == 0 {
            return Err(SklearsError::InvalidInput("Empty face dataset".to_string()));
        }
        if (height, width) != self.image_size {
            return Err(SklearsError::InvalidInput(
                "Face image size doesn't match expected size".to_string(),
            ));
        }
        if n_faces < 2 {
            return Err(SklearsError::InvalidInput(
                "At least two faces are required to fit the manifold".to_string(),
            ));
        }
        let trained_state = self.compute_face_embedding(&faces.view())?;
        Ok(FaceManifoldLearning {
            image_size: self.image_size,
            n_components: trained_state.n_components,
            preprocessing: self.preprocessing,
            state: trained_state,
        })
    }
}

impl Transform<Array2<f64>, Array1<f64>> for FaceManifoldLearning<TrainedFaceManifold> {
    fn transform(&self, face: &Array2<f64>) -> SklResult<Array1<f64>> {
        self.encode_face(&face.view())
    }
}
