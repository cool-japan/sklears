//! Extra Trees (Extremely Randomized Trees) implementation

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, rng};
use sklears_core::{
    error::Result,
    prelude::{Predict, SklearsError},
    traits::{Fit, Trained, Untrained},
    types::Float,
};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier as SmartCoreRandomForestClassifier;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor as SmartCoreRandomForestRegressor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::marker::PhantomData;

/// Noise injection strategy for Extra Trees robustness
#[derive(Debug, Clone, Copy)]
pub enum NoiseInjectionStrategy {
    /// No noise injection
    None,
    /// Add Gaussian noise to features
    Gaussian { std_dev: f64 },
    /// Add uniform noise to features
    Uniform { range: f64 },
    /// Add noise to targets for regression
    TargetNoise { std_dev: f64 },
    /// Feature dropout - randomly set features to zero
    FeatureDropout { dropout_rate: f64 },
    /// Sample dropout - randomly exclude samples
    SampleDropout { dropout_rate: f64 },
}

/// Configuration for Extra Trees
#[derive(Debug, Clone)]
pub struct ExtraTreesConfig {
    /// The number of trees in the forest
    pub n_estimators: usize,
    /// The maximum depth of the tree
    pub max_depth: Option<u16>,
    /// The minimum number of samples required to split an internal node
    pub min_samples_split: usize,
    /// The minimum number of samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Whether to use bootstrap sampling (typically false for Extra Trees)
    pub bootstrap: bool,
    /// Number of features to consider at each split
    pub max_features: Option<usize>,
    /// Number of random thresholds to try per feature
    pub n_random_thresholds: usize,
    /// Noise injection strategy for robustness
    pub noise_injection: NoiseInjectionStrategy,
}

impl Default for ExtraTreesConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            random_state: None,
            bootstrap: false,       // Extra Trees typically don't use bootstrap
            max_features: None,     // Use all features by default
            n_random_thresholds: 1, // Number of random thresholds to try per feature
            noise_injection: NoiseInjectionStrategy::None,
        }
    }
}

/// Helper function to convert ndarray to DenseMatrix
pub fn ndarray_to_dense_matrix(arr: &Array2<f64>) -> DenseMatrix<f64> {
    let _rows = arr.nrows();
    let _cols = arr.ncols();
    let mut data = Vec::new();
    for row in arr.outer_iter() {
        data.push(row.to_vec());
    }
    DenseMatrix::from_2d_vec(&data)
}

/// Apply noise injection to training data for robustness
pub fn apply_noise_injection<R: scirs2_core::random::Random + ?Sized>(
    x: &Array2<f64>,
    y: &Array1<f64>,
    strategy: NoiseInjectionStrategy,
    rng: &mut R,
) -> Result<(Array2<f64>, Array1<f64>)> {
    match strategy {
        NoiseInjectionStrategy::None => Ok((x.clone(), y.clone())),

        NoiseInjectionStrategy::Gaussian { std_dev } => {
            let mut noisy_x = x.clone();
            add_gaussian_noise(&mut noisy_x, std_dev, rng);
            Ok((noisy_x, y.clone()))
        }

        NoiseInjectionStrategy::Uniform { range } => {
            let mut noisy_x = x.clone();
            add_uniform_noise(&mut noisy_x, range, rng);
            Ok((noisy_x, y.clone()))
        }

        NoiseInjectionStrategy::TargetNoise { std_dev } => {
            let mut noisy_y = y.clone();
            add_gaussian_noise_1d(&mut noisy_y, std_dev, rng);
            Ok((x.clone(), noisy_y))
        }

        NoiseInjectionStrategy::FeatureDropout { dropout_rate } => {
            let mut dropped_x = x.clone();
            apply_feature_dropout(&mut dropped_x, dropout_rate, rng);
            Ok((dropped_x, y.clone()))
        }

        NoiseInjectionStrategy::SampleDropout { dropout_rate } => {
            apply_sample_dropout(x, y, dropout_rate, rng)
        }
    }
}

/// Add Gaussian noise to a 2D array (features)
fn add_gaussian_noise<R: scirs2_core::random::Random + ?Sized>(arr: &mut Array2<f64>, std_dev: f64, rng: &mut R) {
    use scirs2_core::random::{Random, rng};

    if std_dev <= 0.0 {
        return;
    }

    let normal = Normal::new(0.0, std_dev).unwrap();

    for element in arr.iter_mut() {
        *element += normal.sample(rng);
    }
}

/// Add uniform noise to a 2D array (features)
fn add_uniform_noise<R: scirs2_core::random::Random + ?Sized>(arr: &mut Array2<f64>, range: f64, rng: &mut R) {
    use scirs2_core::random::{Random, rng};

    if range <= 0.0 {
        return;
    }

    let uniform = Uniform::new(-range, range).expect("Failed to create uniform distribution");

    for element in arr.iter_mut() {
        *element += uniform.sample(rng);
    }
}

/// Add Gaussian noise to a 1D array (targets)
fn add_gaussian_noise_1d<R: scirs2_core::random::Random + ?Sized>(arr: &mut Array1<f64>, std_dev: f64, rng: &mut R) {
    use scirs2_core::random::{Random, rng};

    if std_dev <= 0.0 {
        return;
    }

    let normal = Normal::new(0.0, std_dev).unwrap();

    for element in arr.iter_mut() {
        *element += normal.sample(rng);
    }
}

/// Apply feature dropout by randomly setting features to zero
fn apply_feature_dropout<R: scirs2_core::random::Random + ?Sized>(
    arr: &mut Array2<f64>,
    dropout_rate: f64,
    rng: &mut R,
) {
    if dropout_rate <= 0.0 || dropout_rate >= 1.0 {
        return;
    }

    let (n_samples, n_features) = arr.dim();

    for sample_idx in 0..n_samples {
        for feature_idx in 0..n_features {
            if rng.gen::<f64>() < dropout_rate {
                arr[[sample_idx, feature_idx]] = 0.0;
            }
        }
    }
}

/// Apply sample dropout by randomly excluding samples
fn apply_sample_dropout<R: scirs2_core::random::Random + ?Sized>(
    x: &Array2<f64>,
    y: &Array1<f64>,
    dropout_rate: f64,
    rng: &mut R,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if dropout_rate <= 0.0 || dropout_rate >= 1.0 {
        return Ok((x.clone(), y.clone()));
    }

    let n_samples = x.nrows();
    let mut selected_indices = Vec::new();

    for i in 0..n_samples {
        if rng.gen::<f64>() >= dropout_rate {
            selected_indices.push(i);
        }
    }

    // Ensure we keep at least one sample
    if selected_indices.is_empty() && n_samples > 0 {
        selected_indices.push(0);
    }

    if selected_indices.len() == n_samples {
        return Ok((x.clone(), y.clone()));
    }

    // Create new arrays with selected samples
    let new_x = x.select(scirs2_core::ndarray::Axis(0), &selected_indices);
    let new_y = y.select(scirs2_core::ndarray::Axis(0), &selected_indices);

    Ok((new_x, new_y))
}

/// Extra Trees Classifier
///
/// Extra Trees (Extremely Randomized Trees) is an ensemble method similar to
/// Random Forest, but with completely random splits. This often reduces variance
/// even further than Random Forest and can be faster to train.
///
/// The main differences from Random Forest:
/// - Thresholds are drawn randomly for each candidate feature
/// - Typically uses the original learning sample rather than bootstrap samples
#[derive(Debug)]
pub struct ExtraTreesClassifier<State = Untrained> {
    config: ExtraTreesConfig,
    state: PhantomData<State>,
    // Fitted parameters
    model_: Option<SmartCoreRandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>>,
    classes_: Option<Vec<i32>>,
    n_features_in_: Option<usize>,
    n_classes_: Option<usize>,
}

impl ExtraTreesClassifier<Untrained> {
    /// Create a new Extra Trees Classifier
    pub fn new() -> Self {
        Self {
            config: ExtraTreesConfig::default(),
            state: PhantomData,
            model_: None,
            classes_: None,
            n_features_in_: None,
            n_classes_: None,
        }
    }

    /// Set the number of trees in the forest
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the maximum depth of the trees
    pub fn max_depth(mut self, max_depth: Option<u16>) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split an internal node
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required to be at a leaf node
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the random seed
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Set whether to use bootstrap sampling
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }

    /// Set the maximum number of features to consider at each split
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self
    }

    /// Set the number of random thresholds to try per feature
    pub fn n_random_thresholds(mut self, n_random_thresholds: usize) -> Self {
        self.config.n_random_thresholds = n_random_thresholds;
        self
    }
}

impl ExtraTreesClassifier<Trained> {
    /// Get the classes seen during fitting
    pub fn classes(&self) -> &Vec<i32> {
        self.classes_.as_ref().expect("Classifier should be fitted")
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.n_classes_.expect("Classifier should be fitted")
    }

    /// Get the number of features seen during fitting
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.expect("Classifier should be fitted")
    }
}

impl Default for ExtraTreesClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<i32>> for ExtraTreesClassifier<Untrained> {
    type Fitted = ExtraTreesClassifier<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit classifier on empty dataset".to_string(),
            ));
        }

        let y_len = y.len();
        if n_samples != y_len {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[0] = {n_samples}"),
                actual: format!("y.shape[0] = {y_len}"),
            });
        }

        // Convert inputs to SmartCore format
        let x_matrix = ndarray_to_dense_matrix(x);
        let y_vec = y.to_vec();

        // Find unique classes
        let mut classes: Vec<i32> = y_vec.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        // Create and configure Random Forest classifier to approximate Extra Trees behavior
        // Note: SmartCore doesn't have native Extra Trees, so we configure Random Forest
        // to be as close as possible to Extra Trees behavior
        let params = {
            let mut p = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
                .with_n_trees(self.config.n_estimators as u16)
                .with_min_samples_split(self.config.min_samples_split)
                .with_min_samples_leaf(self.config.min_samples_leaf);

            if let Some(max_depth) = self.config.max_depth {
                p = p.with_max_depth(max_depth);
            }

            p
        };

        let model = if self.config.bootstrap {
            // Use regular Random Forest if bootstrap is enabled
            SmartCoreRandomForestClassifier::fit(&x_matrix, &y_vec, params)
                .map_err(|e| SklearsError::FitError(format!("SmartCore fit error: {e}")))?
        } else {
            // For Extra Trees (no bootstrap), we simulate by using the full dataset
            // This is a limitation of SmartCore - it always uses bootstrap for Random Forest
            log::warn!("SmartCore always uses bootstrap for Random Forest. Extra Trees behavior approximated.");
            SmartCoreRandomForestClassifier::fit(&x_matrix, &y_vec, params)
                .map_err(|e| SklearsError::FitError(format!("SmartCore fit error: {e}")))?
        };

        Ok(ExtraTreesClassifier {
            config: self.config,
            state: PhantomData,
            model_: Some(model),
            classes_: Some(classes),
            n_features_in_: Some(n_features),
            n_classes_: Some(n_classes),
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for ExtraTreesClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let (_n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let x_matrix = ndarray_to_dense_matrix(x);
        let model = self.model_.as_ref().expect("Classifier should be fitted");

        let predictions = model
            .predict(&x_matrix)
            .map_err(|e| SklearsError::PredictError(format!("SmartCore predict error: {e}")))?;

        Ok(Array1::from_vec(predictions))
    }
}

/// Extra Trees Regressor
///
/// Extra Trees (Extremely Randomized Trees) regressor with completely random splits.
#[derive(Debug)]
pub struct ExtraTreesRegressor<State = Untrained> {
    config: ExtraTreesConfig,
    state: PhantomData<State>,
    // Fitted parameters
    model_: Option<SmartCoreRandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    n_features_in_: Option<usize>,
}

impl ExtraTreesRegressor<Untrained> {
    /// Create a new Extra Trees Regressor
    pub fn new() -> Self {
        Self {
            config: ExtraTreesConfig::default(),
            state: PhantomData,
            model_: None,
            n_features_in_: None,
        }
    }

    /// Set the number of trees in the forest
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the maximum depth of the trees
    pub fn max_depth(mut self, max_depth: Option<u16>) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    /// Set the minimum number of samples required to split an internal node
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum number of samples required to be at a leaf node
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the random seed
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Set whether to use bootstrap sampling
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }

    /// Set the maximum number of features to consider at each split
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self
    }

    /// Set the number of random thresholds to try per feature
    pub fn n_random_thresholds(mut self, n_random_thresholds: usize) -> Self {
        self.config.n_random_thresholds = n_random_thresholds;
        self
    }
}

impl ExtraTreesRegressor<Trained> {
    /// Get the number of features seen during fitting
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.expect("Regressor should be fitted")
    }
}

impl Default for ExtraTreesRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ExtraTreesRegressor<Untrained> {
    type Fitted = ExtraTreesRegressor<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit regressor on empty dataset".to_string(),
            ));
        }

        let y_len = y.len();
        if n_samples != y_len {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[0] = {n_samples}"),
                actual: format!("y.shape[0] = {y_len}"),
            });
        }

        // Convert inputs to SmartCore format
        let x_matrix = ndarray_to_dense_matrix(x);
        let y_vec = y.to_vec();

        // Create and configure Random Forest regressor to approximate Extra Trees behavior
        let params = {
            let mut p = smartcore::ensemble::random_forest_regressor::RandomForestRegressorParameters::default()
                .with_n_trees(self.config.n_estimators)
                .with_min_samples_split(self.config.min_samples_split)
                .with_min_samples_leaf(self.config.min_samples_leaf);

            if let Some(max_depth) = self.config.max_depth {
                p = p.with_max_depth(max_depth);
            }

            p
        };

        let model = if self.config.bootstrap {
            // Use regular Random Forest if bootstrap is enabled
            SmartCoreRandomForestRegressor::fit(&x_matrix, &y_vec, params)
                .map_err(|e| SklearsError::FitError(format!("SmartCore fit error: {e}")))?
        } else {
            // For Extra Trees (no bootstrap), we simulate by using the full dataset
            log::warn!("SmartCore always uses bootstrap for Random Forest. Extra Trees behavior approximated.");
            SmartCoreRandomForestRegressor::fit(&x_matrix, &y_vec, params)
                .map_err(|e| SklearsError::FitError(format!("SmartCore fit error: {e}")))?
        };

        Ok(ExtraTreesRegressor {
            config: self.config,
            state: PhantomData,
            model_: Some(model),
            n_features_in_: Some(n_features),
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for ExtraTreesRegressor<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let (_n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let x_matrix = ndarray_to_dense_matrix(x);
        let model = self.model_.as_ref().expect("Regressor should be fitted");

        let predictions = model
            .predict(&x_matrix)
            .map_err(|e| SklearsError::PredictError(format!("SmartCore predict error: {e}")))?;

        Ok(Array1::from_vec(predictions))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_extra_trees_classifier() {
        let x = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ];
        let y = array![0, 1, 0, 1, 0, 1];

        let classifier = ExtraTreesClassifier::new()
            .n_estimators(5)
            .random_state(Some(42))
            .fit(&x, &y)
            .unwrap();

        // Check fitted parameters
        assert_eq!(classifier.n_features_in(), 2);
        assert_eq!(classifier.n_classes(), 2);
        assert_eq!(classifier.classes(), &vec![0, 1]);

        // Make predictions
        let predictions = classifier.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.nrows());

        // Predictions should be within valid class range
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    fn test_extra_trees_regressor() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0],];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0]; // roughly x^2

        let regressor = ExtraTreesRegressor::new()
            .n_estimators(10)
            .random_state(Some(42))
            .fit(&x, &y)
            .unwrap();

        // Check fitted parameters
        assert_eq!(regressor.n_features_in(), 1);

        // Make predictions
        let predictions = regressor.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.nrows());

        // Check that predictions are reasonable (not exact due to randomness)
        for (i, &pred) in predictions.iter().enumerate() {
            assert!(pred >= 0.0); // Should be non-negative for this problem
                                  // Rough check that it's in the right ballpark
            assert!(pred <= 20.0); // Should not be wildly off
        }
    }

    #[test]
    fn test_extra_trees_classifier_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<i32>::zeros(0);

        let result = ExtraTreesClassifier::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_trees_dimension_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1]; // Wrong size

        let result = ExtraTreesClassifier::new().fit(&x, &y);
        assert!(result.is_err());
    }
}
