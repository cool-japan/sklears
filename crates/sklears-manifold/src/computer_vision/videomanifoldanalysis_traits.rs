//! # VideoManifoldAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `VideoManifoldAnalysis`.
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

use super::types::{TrainedVideoAnalysis, VideoManifoldAnalysis};

impl Fit<Array3<f64>, ()> for VideoManifoldAnalysis<Untrained> {
    type Fitted = VideoManifoldAnalysis<TrainedVideoAnalysis>;
    fn fit(self, x: &Array3<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_frames, height, width) = x.dim();
        if height != self.frame_size.0 || width != self.frame_size.1 {
            return Err(SklearsError::InvalidInput(format!(
                "Frame size mismatch: expected {:?}, got ({}, {})",
                self.frame_size, height, width
            )));
        }
        let features = self.extract_temporal_features(x)?;
        let mean_frame = features
            .mean_axis(Axis(0))
            .expect("operation should succeed");
        let centered = &features - &mean_frame.clone().insert_axis(Axis(0));
        let (_, _, vt) = centered
            .svd(false)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
        let v: Array2<f64> = vt.t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection_matrix = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();
        let temporal_embedding = centered.dot(&projection_matrix);
        Ok(VideoManifoldAnalysis {
            frame_size: self.frame_size,
            temporal_window: self.temporal_window,
            n_components: self.n_components,
            analysis_method: self.analysis_method.clone(),
            state: TrainedVideoAnalysis {
                frame_size: self.frame_size,
                temporal_window: self.temporal_window,
                n_components: self.n_components,
                analysis_method: self.analysis_method.clone(),
                temporal_embedding,
                projection_matrix,
                mean_frame,
            },
        })
    }
}

impl Transform<Array3<f64>, Array2<f64>> for VideoManifoldAnalysis<TrainedVideoAnalysis> {
    fn transform(&self, x: &Array3<f64>) -> SklResult<Array2<f64>> {
        self.analyze_video(x)
    }
}
