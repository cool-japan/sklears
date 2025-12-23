//! Trained Multi-View Discriminant Analysis implementation

use super::types::MultiViewDiscriminantAnalysisConfig;
use super::views::ViewInfo;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::Result,
    traits::{Estimator, Predict, PredictProba, Transform},
    types::Float,
};

/// Multi-View Discriminant Analysis (Trained state)
#[derive(Debug, Clone)]
pub struct MultiViewDiscriminantAnalysisTrained {
    pub(crate) config: MultiViewDiscriminantAnalysisConfig,
    // Trained state fields
    pub(crate) classes_: Array1<i32>,
    pub(crate) views_: Vec<ViewInfo>,
    pub(crate) view_means_: Vec<Array2<Float>>,
    pub(crate) view_components_: Vec<Array2<Float>>,
    pub(crate) fusion_weights_: Array1<Float>,
    pub(crate) shared_components_: Option<Array2<Float>>,
}

impl MultiViewDiscriminantAnalysisTrained {
    /// Get the classes found during training
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes_
    }

    /// Get the views information
    pub fn views(&self) -> &[ViewInfo] {
        &self.views_
    }

    /// Get the view-specific means
    pub fn view_means(&self) -> &[Array2<Float>] {
        &self.view_means_
    }

    /// Get the view-specific components
    pub fn view_components(&self) -> &[Array2<Float>] {
        &self.view_components_
    }

    /// Get the fusion weights
    pub fn fusion_weights(&self) -> &Array1<Float> {
        &self.fusion_weights_
    }

    /// Get the shared components (if available)
    pub fn shared_components(&self) -> Option<&Array2<Float>> {
        self.shared_components_.as_ref()
    }
}

impl Estimator for MultiViewDiscriminantAnalysisTrained {
    type Config = MultiViewDiscriminantAnalysisConfig;
    type Error = sklears_core::prelude::SklearsError;
    type Float = sklears_core::types::Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Predict<Vec<Array2<Float>>, Array1<i32>> for MultiViewDiscriminantAnalysisTrained {
    fn predict(&self, x: &Vec<Array2<Float>>) -> Result<Array1<i32>> {
        // Get probabilities and select most probable class
        let probabilities = self.predict_proba(x)?;
        let n_samples = probabilities.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let probs = probabilities.row(sample_idx);
            let mut max_prob = Float::NEG_INFINITY;
            let mut best_class_idx = 0;

            for (class_idx, &prob) in probs.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    best_class_idx = class_idx;
                }
            }

            predictions[sample_idx] = self.classes_[best_class_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Vec<Array2<Float>>, Array2<Float>> for MultiViewDiscriminantAnalysisTrained {
    fn predict_proba(&self, x: &Vec<Array2<Float>>) -> Result<Array2<Float>> {
        if x.len() != self.views_.len() {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(format!(
                "Expected {} views, got {}",
                self.views_.len(),
                x.len()
            )));
        }

        let n_samples = x[0].nrows();
        let n_classes = self.classes_.len();

        // Aggregate probabilities from all views
        let mut combined_scores: Array2<Float> = Array2::zeros((n_samples, n_classes));

        for view_data in x.iter() {
            // Get current view index by iterating again
            let view_idx = x.iter().position(|v| std::ptr::eq(v, view_data)).unwrap();
            let view_means = &self.view_means_[view_idx];
            let fusion_weight = self.views_[view_idx].weight;

            // Compute distances to class means for this view
            for sample_idx in 0..n_samples {
                let sample = view_data.row(sample_idx);

                for class_idx in 0..n_classes {
                    let mean = view_means.row(class_idx);
                    let mut distance_sq = 0.0;

                    for feat_idx in 0..view_data.ncols() {
                        let diff = sample[feat_idx] - mean[feat_idx];
                        distance_sq += diff * diff;
                    }

                    // Convert distance to similarity score (negative exp)
                    // Weighted by fusion weight for this view
                    combined_scores[[sample_idx, class_idx]] +=
                        fusion_weight * (-distance_sq).exp();
                }
            }
        }

        // Normalize to probabilities
        let mut probabilities: Array2<Float> = Array2::zeros((n_samples, n_classes));
        for sample_idx in 0..n_samples {
            let mut sum = 0.0;
            for class_idx in 0..n_classes {
                sum += combined_scores[[sample_idx, class_idx]];
            }

            if sum > 1e-10 {
                for class_idx in 0..n_classes {
                    probabilities[[sample_idx, class_idx]] =
                        combined_scores[[sample_idx, class_idx]] / sum;
                }
            } else {
                // Uniform distribution if all scores are zero
                for class_idx in 0..n_classes {
                    probabilities[[sample_idx, class_idx]] = 1.0 / n_classes as Float;
                }
            }
        }

        Ok(probabilities)
    }
}

impl Transform<Vec<Array2<Float>>, Array2<Float>> for MultiViewDiscriminantAnalysisTrained {
    fn transform(&self, x: &Vec<Array2<Float>>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(sklears_core::prelude::SklearsError::InvalidInput(
                "Empty views list".to_string(),
            ));
        }

        let n_samples = x[0].nrows();

        // If shared components are available, project onto them
        if let Some(ref shared_comps) = self.shared_components_ {
            let n_components = shared_comps.nrows();
            let mut transformed: Array2<Float> = Array2::zeros((n_samples, n_components));

            // For simplicity, concatenate all views and project
            // In full implementation, would use proper multi-view projection
            for sample_idx in 0..n_samples {
                for comp_idx in 0..n_components {
                    let mut projection = 0.0;
                    let mut feat_offset = 0;

                    for view_data in x.iter() {
                        let n_features = view_data.ncols();
                        for feat_idx in 0..n_features {
                            if feat_offset + feat_idx < shared_comps.ncols() {
                                projection += view_data[[sample_idx, feat_idx]]
                                    * shared_comps[[comp_idx, feat_offset + feat_idx]];
                            }
                        }
                        feat_offset += n_features;
                    }

                    transformed[[sample_idx, comp_idx]] = projection;
                }
            }

            Ok(transformed)
        } else {
            // No dimensionality reduction, concatenate views
            let total_features: usize = x.iter().map(|v| v.ncols()).sum();
            let mut concatenated: Array2<Float> = Array2::zeros((n_samples, total_features));

            let mut feat_offset = 0;
            for view_data in x.iter() {
                let n_features = view_data.ncols();
                for sample_idx in 0..n_samples {
                    for feat_idx in 0..n_features {
                        concatenated[[sample_idx, feat_offset + feat_idx]] =
                            view_data[[sample_idx, feat_idx]];
                    }
                }
                feat_offset += n_features;
            }

            Ok(concatenated)
        }
    }
}
