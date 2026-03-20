//! # PoseEstimationManifold - Trait Implementations
//!
//! This module contains trait implementations for `PoseEstimationManifold`.
//!
//! ## Implemented Traits
//!
//! - `Fit`
//! - `Transform`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

use super::types::{PoseEstimationManifold, TrainedPoseEstimation};

impl Fit<Array2<f64>, ()> for PoseEstimationManifold<Untrained> {
    type Fitted = PoseEstimationManifold<TrainedPoseEstimation>;
    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if x.ncols() != self.n_keypoints * 2 {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} columns ({}*2), got {}",
                self.n_keypoints * 2,
                self.n_keypoints,
                x.ncols()
            )));
        }
        let mean_pose = x.mean_axis(Axis(0)).expect("operation should succeed");
        let centered = x - &mean_pose.clone().insert_axis(Axis(0));
        let pose_manifold = self.compute_pose_embedding(x)?;
        let (_, _, vt) = centered
            .svd(false)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
        let v: Array2<f64> = vt.t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection_matrix = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();
        Ok(PoseEstimationManifold {
            n_keypoints: self.n_keypoints,
            n_components: self.n_components,
            embedding_method: self.embedding_method.clone(),
            bone_constraints: self.bone_constraints.clone(),
            state: TrainedPoseEstimation {
                n_keypoints: self.n_keypoints,
                n_components: self.n_components,
                embedding_method: self.embedding_method.clone(),
                bone_constraints: self.bone_constraints.clone(),
                pose_manifold,
                projection_matrix,
                mean_pose,
                reference_poses: x.to_owned(),
            },
        })
    }
}

impl Transform<Array1<f64>, Array1<f64>> for PoseEstimationManifold<TrainedPoseEstimation> {
    fn transform(&self, x: &Array1<f64>) -> SklResult<Array1<f64>> {
        self.estimate_pose(&x.view())
    }
}
