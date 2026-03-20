//! # MultiLayerStackingClassifier - Trait Implementations
//!
//! This module contains trait implementations for `MultiLayerStackingClassifier`.
//!
//! ## Implemented Traits
//!
//! - `Fit`
//! - `Predict`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::Predict,
    traits::{Fit, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use super::types::MultiLayerStackingClassifier;

impl Fit<Array2<Float>, Array1<i32>> for MultiLayerStackingClassifier<Untrained> {
    type Fitted = MultiLayerStackingClassifier<Trained>;
    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} samples", y.len()),
            });
        }
        let (n_samples, n_features) = x.dim();
        if n_samples < 20 {
            return Err(SklearsError::InvalidInput(
                "Multi-layer stacking requires at least 20 samples".to_string(),
            ));
        }
        let mut classes: Vec<i32> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes_array = Array1::from_vec(classes.clone());
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }
        let y_float: Array1<Float> = y.mapv(|v| v as Float);
        let mut layers = Vec::new();
        let mut current_features = x.clone();
        let mut layer_importances = Vec::new();
        for (layer_idx, layer_config) in self.config.layers.iter().enumerate() {
            let layer =
                self.train_stacking_layer(&current_features, &y_float, layer_config, layer_idx)?;
            let meta_features =
                self.generate_layer_meta_features(&current_features, &layer, layer_config)?;
            if layer_config.passthrough && layer_idx == 0 {
                current_features = Array2::zeros((n_samples, meta_features.ncols() + n_features));
                current_features.slice_mut(s![.., ..n_features]).assign(x);
                current_features
                    .slice_mut(s![.., n_features..])
                    .assign(&meta_features);
            } else {
                current_features = meta_features;
            }
            layer_importances.push(layer.feature_importances.clone());
            layers.push(layer);
        }
        let (final_meta_weights, final_meta_intercept) = self.train_final_meta_learner(
            &current_features,
            &y_float,
            &self.config.final_meta_strategy,
        )?;
        Ok(MultiLayerStackingClassifier {
            config: self.config,
            state: PhantomData,
            layers_: Some(layers),
            final_meta_weights_: Some(final_meta_weights),
            final_meta_intercept_: Some(final_meta_intercept),
            classes_: Some(classes_array),
            n_features_in_: Some(n_features),
            layer_feature_importances_: Some(layer_importances),
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for MultiLayerStackingClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().expect("operation should succeed");
        let predictions = probabilities
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("operation should succeed"))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect::<Vec<_>>();
        Ok(Array1::from_vec(predictions))
    }
}
