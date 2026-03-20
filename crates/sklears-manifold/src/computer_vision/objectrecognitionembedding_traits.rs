//! # ObjectRecognitionEmbedding - Trait Implementations
//!
//! This module contains trait implementations for `ObjectRecognitionEmbedding`.
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

use super::types::{ObjectEmbeddingMethod, ObjectRecognitionEmbedding, TrainedObjectRecognition};

impl Fit<Array2<f64>, Array1<usize>> for ObjectRecognitionEmbedding<Untrained> {
    type Fitted = ObjectRecognitionEmbedding<TrainedObjectRecognition>;
    fn fit(self, x: &Array2<f64>, y: &Array1<usize>) -> SklResult<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }
        let feature_mean = x.mean_axis(Axis(0)).expect("operation should succeed");
        let embedding_matrix = match &self.embedding_method {
            ObjectEmbeddingMethod::Contrastive { margin } => {
                self.learn_contrastive_embedding(x, y, *margin)?
            }
            ObjectEmbeddingMethod::Triplet { margin } => {
                self.learn_contrastive_embedding(x, y, *margin)?
            }
            ObjectEmbeddingMethod::Supervised => {
                let centered = x - &feature_mean.clone().insert_axis(Axis(0));
                let (_, _, vt) = centered
                    .svd(false)
                    .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
                let v: Array2<f64> = vt.t().to_owned();
                let n_comp = self.n_components.min(v.ncols());
                v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned()
            }
        };
        let mut class_prototypes = Array2::zeros((self.n_classes, self.n_components));
        let mut class_counts = vec![0; self.n_classes];
        for (sample_idx, &label) in y.iter().enumerate() {
            if label < self.n_classes {
                let features = x.row(sample_idx).to_owned() - &feature_mean;
                let embedded = embedding_matrix.t().dot(&features);
                for comp_idx in 0..self.n_components.min(embedded.len()) {
                    class_prototypes[[label, comp_idx]] += embedded[comp_idx];
                }
                class_counts[label] += 1;
            }
        }
        for class_idx in 0..self.n_classes {
            if class_counts[class_idx] > 0 {
                for comp_idx in 0..self.n_components {
                    class_prototypes[[class_idx, comp_idx]] /= class_counts[class_idx] as f64;
                }
            }
        }
        Ok(ObjectRecognitionEmbedding {
            n_components: self.n_components,
            embedding_method: self.embedding_method.clone(),
            n_classes: self.n_classes,
            state: TrainedObjectRecognition {
                n_components: self.n_components,
                embedding_method: self.embedding_method.clone(),
                n_classes: self.n_classes,
                embedding_matrix,
                class_prototypes,
                feature_mean,
            },
        })
    }
}

impl Transform<Array1<f64>, usize> for ObjectRecognitionEmbedding<TrainedObjectRecognition> {
    fn transform(&self, x: &Array1<f64>) -> SklResult<usize> {
        self.recognize(&x.view())
    }
}
