//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier as SmartCoreClassifier, RandomForestClassifierParameters,
};
use smartcore::tree::decision_tree_classifier::SplitCriterion as ClassifierCriterion;

use super::types::*;
use crate::{ndarray_to_dense_matrix, MaxFeatures, SplitCriterion};

impl Default for RandomForestClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RandomForestClassifier<Untrained> {
    type Config = RandomForestConfig;
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for RandomForestClassifier<Untrained> {
    type Fitted = RandomForestClassifier<Trained>;
    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, y.shape[0]={}", n_samples, y.len()),
            });
        }
        let x_matrix = ndarray_to_dense_matrix(x);
        let y_vec = y.to_vec();
        let _max_features = match self.config.max_features {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
            MaxFeatures::Log2 => (n_features as f64).log2().ceil() as usize,
            MaxFeatures::Number(n) => n.min(n_features),
            MaxFeatures::Fraction(f) => ((n_features as f64 * f).ceil() as usize).min(n_features),
        };
        let criterion = match self.config.criterion {
            SplitCriterion::Gini => ClassifierCriterion::Gini,
            SplitCriterion::Entropy => ClassifierCriterion::Entropy,
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "criterion".to_string(),
                    reason: "MSE and MAE are only valid for regression".to_string(),
                });
            }
        };
        let mut parameters = RandomForestClassifierParameters::default()
            .with_n_trees(self.config.n_estimators as u16)
            .with_min_samples_split(self.config.min_samples_split)
            .with_min_samples_leaf(self.config.min_samples_leaf)
            .with_criterion(criterion);
        if let Some(max_depth) = self.config.max_depth {
            parameters = parameters.with_max_depth(max_depth as u16);
        }
        let model = SmartCoreClassifier::fit(&x_matrix, &y_vec, parameters)
            .map_err(|e| SklearsError::FitError(format!("Random forest fit failed: {e:?}")))?;
        let mut classes: Vec<i32> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes_array = Array1::from_vec(classes.clone());
        let n_classes = classes.len();
        let (oob_score, oob_decision_function) = if self.config.oob_score && self.config.bootstrap {
            let (score, decisions) = Self::compute_oob_score(&model, x, y, &classes)?;
            (Some(score), Some(decisions))
        } else {
            (None, None)
        };
        Ok(RandomForestClassifier {
            config: self.config,
            state: PhantomData,
            model_: Some(model),
            classes_: Some(classes_array),
            n_classes_: Some(n_classes),
            n_features_: Some(n_features),
            n_outputs_: Some(1),
            oob_score_: oob_score,
            oob_decision_function_: oob_decision_function,
            proximity_matrix_: None,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for RandomForestClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let model = self.model_.as_ref().expect("Model should be fitted");
        if x.ncols() != self.n_features() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features(),
                actual: x.ncols(),
            });
        }
        let x_matrix = ndarray_to_dense_matrix(x);
        let predictions = model
            .predict(&x_matrix)
            .map_err(|e| SklearsError::PredictError(format!("Prediction failed: {e:?}")))?;
        Ok(Array1::from_vec(predictions))
    }
}
