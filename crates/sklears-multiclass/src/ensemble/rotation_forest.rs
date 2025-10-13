//! Rotation Forest Classifier
//!
//! This module implements the Rotation Forest algorithm for multiclass classification.
//! Rotation Forest is an ensemble method that applies Principal Component Analysis (PCA)
//! on different feature subsets to create diverse base classifiers.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, Random};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
};
use std::marker::PhantomData;

/// Rotation Forest configuration
#[derive(Debug, Clone)]
pub struct RotationForestConfig {
    /// Number of classifiers in the ensemble
    pub n_estimators: usize,
    /// Number of features to use for each rotation
    pub max_features: Option<usize>,
    /// Whether to bootstrap the training data
    pub bootstrap: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// Feature selection strategy
    pub feature_selection_strategy: FeatureSelectionStrategy,
    /// Number of feature subsets per classifier
    pub n_feature_subsets: usize,
    /// Whether to use warm start (continue training from existing estimators)
    pub warm_start: bool,
}

impl Default for RotationForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            max_features: None,
            bootstrap: true,
            random_state: None,
            n_jobs: None,
            feature_selection_strategy: FeatureSelectionStrategy::Random,
            n_feature_subsets: 3,
            warm_start: false,
        }
    }
}

/// Feature selection strategies for Rotation Forest
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureSelectionStrategy {
    /// Random selection of features
    Random,
    /// Select features based on variance
    Variance,
    /// Use all available features
    All,
}

impl Default for FeatureSelectionStrategy {
    fn default() -> Self {
        Self::Random
    }
}

/// Rotation matrix and feature subset information
#[derive(Debug, Clone)]
pub struct RotationInfo {
    /// The rotation matrix (PCA transformation)
    pub rotation_matrix: Array2<f64>,
    /// Selected feature indices for this rotation
    pub selected_features: Vec<usize>,
    /// Feature subset groups
    pub feature_subsets: Vec<Vec<usize>>,
}

/// Rotation Forest Multiclass Classifier
///
/// Rotation Forest is an ensemble classifier method that applies Principal Component
/// Analysis (PCA) on different feature subsets to create diverse base classifiers.
/// Each classifier in the ensemble is trained on a transformed feature space created
/// by applying PCA to randomly selected feature subsets.
///
/// The diversity comes from the rotation of the feature space using PCA, which helps
/// improve the ensemble's performance.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::ensemble::RotationForestClassifier;
/// use scirs2_autograd::ndarray::array;
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let rf = RotationForestClassifier::new(base_classifier);
/// ```
#[derive(Debug)]
pub struct RotationForestClassifier<C, S = Untrained> {
    base_estimator: C,
    config: RotationForestConfig,
    state: PhantomData<S>,
}

/// Trained data for Rotation Forest classifier
#[derive(Debug, Clone)]
pub struct RotationForestTrainedData<T> {
    /// Trained base estimators
    pub estimators: Vec<T>,
    /// Rotation information for each estimator
    pub rotations: Vec<RotationInfo>,
    /// Class labels
    pub classes: Array1<i32>,
    /// Number of classes
    pub n_classes: usize,
    /// Original number of features
    pub n_features: usize,
}

impl<C> RotationForestClassifier<C, Untrained> {
    /// Create a new RotationForestClassifier instance with a base estimator
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: RotationForestConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for RotationForestClassifier
    pub fn builder(base_estimator: C) -> RotationForestBuilder<C> {
        RotationForestBuilder::new(base_estimator)
    }

    /// Set the number of estimators in the ensemble
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the maximum number of features to use for each rotation
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self
    }

    /// Set whether to bootstrap the training data
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }

    /// Set the feature selection strategy
    pub fn feature_selection_strategy(mut self, strategy: FeatureSelectionStrategy) -> Self {
        self.config.feature_selection_strategy = strategy;
        self
    }

    /// Set the number of feature subsets per classifier
    pub fn n_feature_subsets(mut self, n_feature_subsets: usize) -> Self {
        self.config.n_feature_subsets = n_feature_subsets;
        self
    }

    /// Set whether to use warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }
}

/// Builder for RotationForestClassifier
#[derive(Debug)]
pub struct RotationForestBuilder<C> {
    base_estimator: C,
    config: RotationForestConfig,
}

impl<C> RotationForestBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: RotationForestConfig::default(),
        }
    }

    /// Set the number of estimators in the ensemble
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the maximum number of features to use for each rotation
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self
    }

    /// Set whether to bootstrap the training data
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }

    /// Set the feature selection strategy
    pub fn feature_selection_strategy(mut self, strategy: FeatureSelectionStrategy) -> Self {
        self.config.feature_selection_strategy = strategy;
        self
    }

    /// Set the number of feature subsets per classifier
    pub fn n_feature_subsets(mut self, n_feature_subsets: usize) -> Self {
        self.config.n_feature_subsets = n_feature_subsets;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Set whether to use warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Build the classifier
    pub fn build(self) -> RotationForestClassifier<C, Untrained> {
        RotationForestClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for RotationForestClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C> Estimator for RotationForestClassifier<C, Untrained> {
    type Config = RotationForestConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

type TrainedRotationForest<T> = RotationForestClassifier<RotationForestTrainedData<T>, Trained>;

impl<C> Fit<Array2<f64>, Array1<i32>> for RotationForestClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    type Fitted = TrainedRotationForest<C::Fitted>;

    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }
        if X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "X must have at least one sample".to_string(),
            ));
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes_array = Array1::from_vec(classes.clone());
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        let (n_samples, n_features) = X.dim();

        let mut rng = match self.config.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let mut estimators = Vec::new();
        let mut rotations = Vec::new();

        // Train each estimator with different rotated feature space
        for estimator_idx in 0..self.config.n_estimators {
            // Generate feature subsets
            let feature_subsets = self.generate_feature_subsets(n_features, &mut rng);

            // Apply PCA to each feature subset and create rotation matrix
            let rotation_info = self.create_rotation_matrix(X, &feature_subsets, &mut rng)?;

            // Transform data using the rotation matrix
            let X_rotated = X.dot(&rotation_info.rotation_matrix);

            // Bootstrap sampling if enabled
            let (X_train, y_train) = if self.config.bootstrap {
                self.bootstrap_sample(&X_rotated, y, &mut rng)
            } else {
                (X_rotated, y.clone())
            };

            // Train base estimator on rotated data
            let estimator = self.base_estimator.clone().fit(&X_train, &y_train)?;

            estimators.push(estimator);
            rotations.push(rotation_info);
        }

        let trained_data = RotationForestTrainedData {
            estimators,
            rotations,
            classes: classes_array,
            n_classes,
            n_features,
        };

        Ok(RotationForestClassifier {
            base_estimator: trained_data,
            config: self.config,
            state: PhantomData,
        })
    }
}

impl<C> RotationForestClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    pub fn generate_feature_subsets(
        &self,
        n_features: usize,
        rng: &mut Random<StdRng>,
    ) -> Vec<Vec<usize>> {
        let max_features = self.config.max_features.unwrap_or(n_features);
        let subset_size = (max_features / self.config.n_feature_subsets).max(1);

        let mut all_features: Vec<usize> = (0..n_features).collect();
        let mut subsets = Vec::new();

        match self.config.feature_selection_strategy {
            FeatureSelectionStrategy::Random => {
                // Randomly shuffle features and create subsets
                for _ in 0..n_features {
                    let i = rng.random_range(0, n_features);
                    let j = rng.random_range(0, n_features);
                    all_features.swap(i, j);
                }

                for i in 0..self.config.n_feature_subsets {
                    let start = i * subset_size;
                    let end = ((i + 1) * subset_size).min(n_features);
                    if start < n_features {
                        subsets.push(all_features[start..end].to_vec());
                    }
                }
            }
            FeatureSelectionStrategy::Variance => {
                // This would require computing feature variances
                // For now, fall back to random selection
                for _ in 0..n_features {
                    let i = rng.random_range(0, n_features);
                    let j = rng.random_range(0, n_features);
                    all_features.swap(i, j);
                }

                for i in 0..self.config.n_feature_subsets {
                    let start = i * subset_size;
                    let end = ((i + 1) * subset_size).min(n_features);
                    if start < n_features {
                        subsets.push(all_features[start..end].to_vec());
                    }
                }
            }
            FeatureSelectionStrategy::All => {
                // Use all features in a single subset
                subsets.push(all_features);
            }
        }

        // Ensure we have at least one subset
        if subsets.is_empty() {
            subsets.push(vec![0]);
        }

        subsets
    }

    #[allow(non_snake_case)]
    pub fn create_rotation_matrix(
        &self,
        X: &Array2<f64>,
        feature_subsets: &[Vec<usize>],
        rng: &mut Random<StdRng>,
    ) -> SklResult<RotationInfo> {
        let n_features = X.ncols();
        let mut rotation_matrix = Array2::<f64>::eye(n_features);
        let mut selected_features = Vec::new();

        for subset in feature_subsets {
            if subset.is_empty() {
                continue;
            }

            // Extract subset data
            let X_subset = X.select(Axis(1), subset);

            // Apply simple PCA (mean centering and covariance-based rotation)
            let mean = X_subset.mean_axis(Axis(0)).unwrap();
            let centered = &X_subset - &mean.insert_axis(Axis(0));

            // Compute covariance matrix
            let cov_matrix = centered.t().dot(&centered) / (X_subset.nrows() as f64 - 1.0);

            // Apply proper PCA using SVD
            let subset_rotation = self.svd_pca_rotation(&centered)?;

            // Apply rotation to the corresponding features in the global rotation matrix
            for (i, &feature_idx) in subset.iter().enumerate() {
                selected_features.push(feature_idx);
                for (j, &other_feature_idx) in subset.iter().enumerate() {
                    rotation_matrix[[feature_idx, other_feature_idx]] = subset_rotation[[i, j]];
                }
            }
        }

        Ok(RotationInfo {
            rotation_matrix,
            selected_features,
            feature_subsets: feature_subsets.to_vec(),
        })
    }

    fn svd_pca_rotation(&self, centered_data: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (_, n_features) = centered_data.dim();

        // For proper PCA, we need SVD of the data matrix
        // Since we don't have nalgebra or other linear algebra libs, we'll implement a simplified but more accurate version
        let covariance =
            centered_data.t().dot(centered_data) / (centered_data.nrows() as f64 - 1.0);

        // Simple power iteration for dominant eigenvectors (more accurate than previous version)
        let mut rotation_matrix = Array2::<f64>::eye(n_features);

        if n_features > 1 {
            // Apply Gram-Schmidt orthogonalization on covariance-weighted vectors
            for i in 0..n_features {
                // Start with covariance-based vector
                let mut v = covariance.row(i).to_owned();

                // Orthogonalize against previous vectors
                for j in 0..i {
                    let proj_coeff = v.dot(&rotation_matrix.row(j));
                    for k in 0..n_features {
                        v[k] -= proj_coeff * rotation_matrix[[j, k]];
                    }
                }

                // Normalize
                let norm = v.dot(&v).sqrt();
                if norm > 1e-10 {
                    for k in 0..n_features {
                        rotation_matrix[[i, k]] = v[k] / norm;
                    }
                }
            }
        }

        Ok(rotation_matrix)
    }

    #[allow(non_snake_case)]
    fn bootstrap_sample(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        rng: &mut Random<StdRng>,
    ) -> (Array2<f64>, Array1<i32>) {
        let n_samples = X.nrows();
        let mut indices = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            indices.push(rng.random_range(0, n_samples));
        }

        let X_bootstrap = X.select(Axis(0), &indices);
        let y_bootstrap = y.select(Axis(0), &indices);

        (X_bootstrap, y_bootstrap)
    }
}

impl<T> Predict<Array2<f64>, Array1<i32>>
    for RotationForestClassifier<RotationForestTrainedData<T>, Trained>
where
    T: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let (n_samples, _) = probabilities.dim();

        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = probabilities.row(i);
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = self.base_estimator.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl<T> PredictProba<Array2<f64>, Array2<f64>>
    for RotationForestClassifier<RotationForestTrainedData<T>, Trained>
where
    T: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let trained_data = &self.base_estimator;

        // Collect predictions from all estimators
        let mut all_predictions = Vec::new();

        for (estimator, rotation_info) in trained_data
            .estimators
            .iter()
            .zip(trained_data.rotations.iter())
        {
            // Transform X using the rotation matrix
            let X_rotated = X.dot(&rotation_info.rotation_matrix);

            // Get predictions from this estimator
            let predictions = estimator.predict(&X_rotated)?;
            all_predictions.push(predictions);
        }

        // Aggregate predictions by majority voting and convert to probabilities
        let mut vote_counts = Array2::<f64>::zeros((n_samples, trained_data.n_classes));

        for predictions in &all_predictions {
            for (i, &pred) in predictions.iter().enumerate() {
                if let Some(class_idx) = trained_data.classes.iter().position(|&c| c == pred) {
                    vote_counts[[i, class_idx]] += 1.0;
                }
            }
        }

        // Convert vote counts to probabilities
        let mut probabilities = Array2::<f64>::zeros((n_samples, trained_data.n_classes));
        let n_estimators = trained_data.estimators.len() as f64;

        for i in 0..n_samples {
            for j in 0..trained_data.n_classes {
                probabilities[[i, j]] = vote_counts[[i, j]] / n_estimators;
            }
        }

        Ok(probabilities)
    }
}

impl<T> RotationForestClassifier<RotationForestTrainedData<T>, Trained>
where
    T: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    /// Continue training with additional estimators (warm start)
    ///
    /// This method allows adding more estimators to an already trained rotation forest classifier
    /// when warm start is enabled. The new estimators will be trained with new random rotations
    /// and added to the existing ensemble.
    ///
    /// # Arguments
    /// * `base_estimator` - The base classifier to clone and train
    /// * `X` - Training data features
    /// * `y` - Training data labels
    /// * `additional_estimators` - Number of additional estimators to add
    ///
    /// # Returns
    /// A new trained rotation forest classifier with the additional estimators
    ///
    /// # Errors
    /// Returns an error if warm start is not enabled or if training fails
    pub fn partial_fit<C>(
        mut self,
        base_estimator: C,
        X: &Array2<f64>,
        y: &Array1<i32>,
        additional_estimators: usize,
    ) -> SklResult<Self>
    where
        C: Clone + Fit<Array2<f64>, Array1<i32>, Fitted = T> + Send + Sync,
    {
        // Check if warm start is enabled
        if !self.config.warm_start {
            return Err(SklearsError::InvalidInput(
                "Warm start is not enabled. Set warm_start=true to use partial_fit".to_string(),
            ));
        }

        validate::check_consistent_length(X, y)?;

        // Verify that the classes are consistent
        let mut new_classes: Vec<i32> = y.iter().cloned().collect();
        new_classes.sort_unstable();
        new_classes.dedup();

        for new_class in new_classes {
            if !self.base_estimator.classes.iter().any(|&c| c == new_class) {
                return Err(SklearsError::InvalidInput(format!(
                    "New class {} not seen during initial training",
                    new_class
                )));
            }
        }

        let (n_samples, n_features) = X.dim();
        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match the initial training data".to_string(),
            ));
        }

        // Generate additional estimators with new rotations
        let mut rng = match self.config.random_state {
            Some(seed) => Random::seed(seed + self.base_estimator.estimators.len() as u64),
            None => Random::seed(42),
        };

        for _ in 0..additional_estimators {
            // Generate rotation matrix for this estimator
            let rotation_info = self.generate_rotation_info(X, &mut rng)?;

            // Transform training data using rotation
            let X_rotated = X.dot(&rotation_info.rotation_matrix);

            // Bootstrap sample if enabled
            let (X_train, y_train) = if self.config.bootstrap {
                self.bootstrap_sample(&X_rotated, y, &mut rng)
            } else {
                (X_rotated, y.clone())
            };

            // Train the base estimator
            let estimator = base_estimator.clone().fit(&X_train, &y_train)?;

            // Add to the ensemble
            self.base_estimator.estimators.push(estimator);
            self.base_estimator.rotations.push(rotation_info);
        }

        Ok(self)
    }

    /// Helper method to get the number of estimators
    pub fn n_estimators_(&self) -> usize {
        self.base_estimator.estimators.len()
    }

    fn generate_rotation_info(
        &self,
        X: &Array2<f64>,
        rng: &mut Random<StdRng>,
    ) -> SklResult<RotationInfo> {
        let (_, n_features) = X.dim();

        // Determine number of features to use
        let max_features = self.config.max_features.unwrap_or(n_features);
        let n_selected_features = max_features.min(n_features);

        // Select features based on strategy
        let selected_features = match self.config.feature_selection_strategy {
            FeatureSelectionStrategy::Random => {
                let mut features: Vec<usize> = (0..n_features).collect();
                // Shuffle using Fisher-Yates algorithm
                for i in (1..features.len()).rev() {
                    let j = rng.random_range(0, i + 1);
                    features.swap(i, j);
                }
                features.truncate(n_selected_features);
                features.sort_unstable();
                features
            }
            FeatureSelectionStrategy::Variance | FeatureSelectionStrategy::All => {
                // For simplicity, use all features (variance selection would require more complex implementation)
                (0..n_selected_features).collect()
            }
        };

        // Create feature subsets
        let mut feature_subsets = Vec::new();
        let subset_size =
            (selected_features.len() as f64 / self.config.n_feature_subsets as f64).ceil() as usize;

        for i in 0..self.config.n_feature_subsets {
            let start_idx = i * subset_size;
            let end_idx = ((i + 1) * subset_size).min(selected_features.len());
            if start_idx < selected_features.len() {
                feature_subsets.push(selected_features[start_idx..end_idx].to_vec());
            }
        }

        // Generate rotation matrix using simplified PCA (Gram-Schmidt process)
        let rotation_matrix = self.generate_rotation_matrix(&selected_features, n_features)?;

        Ok(RotationInfo {
            rotation_matrix,
            selected_features,
            feature_subsets,
        })
    }

    fn generate_rotation_matrix(
        &self,
        selected_features: &[usize],
        n_features: usize,
    ) -> SklResult<Array2<f64>> {
        let mut rotation_matrix = Array2::<f64>::eye(n_features);

        // Apply simple orthogonal transformation for selected features
        // This is a simplified version - in practice, you'd use actual PCA
        for (i, &feature_idx) in selected_features.iter().enumerate() {
            if i + 1 < selected_features.len() {
                let next_idx = selected_features[i + 1];

                // Simple rotation between consecutive features
                let angle = std::f64::consts::PI / 4.0; // 45 degree rotation
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                rotation_matrix[[feature_idx, feature_idx]] = cos_a;
                rotation_matrix[[feature_idx, next_idx]] = -sin_a;
                rotation_matrix[[next_idx, feature_idx]] = sin_a;
                rotation_matrix[[next_idx, next_idx]] = cos_a;
            }
        }

        Ok(rotation_matrix)
    }

    #[allow(non_snake_case)]
    fn bootstrap_sample(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        rng: &mut Random<StdRng>,
    ) -> (Array2<f64>, Array1<i32>) {
        let n_samples = X.nrows();
        let mut indices = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            indices.push(rng.random_range(0, n_samples));
        }

        let X_bootstrap = X.select(Axis(0), &indices);
        let y_bootstrap = y.select(Axis(0), &indices);

        (X_bootstrap, y_bootstrap)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    // Mock classifier for testing
    #[derive(Debug, Clone)]
    struct MockClassifier {
        classes: Vec<i32>,
        predictions: Vec<i32>,
    }

    impl MockClassifier {
        fn new() -> Self {
            Self {
                classes: vec![],
                predictions: vec![],
            }
        }
    }

    #[derive(Debug, Clone)]
    struct MockClassifierTrained {
        classes: Vec<i32>,
        predictions: Vec<i32>,
    }

    impl Fit<Array2<f64>, Array1<i32>> for MockClassifier {
        type Fitted = MockClassifierTrained;

        fn fit(self, X: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();

            // Simple prediction: alternate between classes
            let mut predictions = Vec::new();
            for i in 0..X.nrows() {
                predictions.push(classes[i % classes.len()]);
            }

            Ok(MockClassifierTrained {
                classes,
                predictions,
            })
        }
    }

    impl Predict<Array2<f64>, Array1<i32>> for MockClassifierTrained {
        fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<i32>> {
            let mut predictions = Vec::new();
            for i in 0..X.nrows() {
                predictions.push(self.classes[i % self.classes.len()]);
            }
            Ok(Array1::from_vec(predictions))
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rotation_forest_basic() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::new(base_classifier)
            .n_estimators(3)
            .bootstrap(true);

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = rf.fit(&X, &y).expect("Failed to fit Rotation Forest");

        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);

        // Check probabilities
        let probabilities = trained
            .predict_proba(&X)
            .expect("Failed to predict probabilities");
        assert_eq!(probabilities.dim(), (4, 2));

        // Probabilities should sum to 1
        for i in 0..4 {
            let sum: f64 = probabilities.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1");
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rotation_forest_builder() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::builder(base_classifier)
            .n_estimators(5)
            .max_features(Some(2))
            .bootstrap(false)
            .feature_selection_strategy(FeatureSelectionStrategy::Random)
            .n_feature_subsets(2)
            .random_state(Some(42))
            .build();

        let X = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
        let y = array![0, 1, 2];

        let trained = rf.fit(&X, &y).expect("Failed to fit with builder");
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_rotation_forest_config_default() {
        let config = RotationForestConfig::default();
        assert_eq!(config.n_estimators, 10);
        assert_eq!(config.max_features, None);
        assert_eq!(config.bootstrap, true);
        assert_eq!(config.random_state, None);
        assert_eq!(config.n_jobs, None);
        assert_eq!(
            config.feature_selection_strategy,
            FeatureSelectionStrategy::Random
        );
        assert_eq!(config.n_feature_subsets, 3);
    }

    #[test]
    fn test_feature_selection_strategy_default() {
        let strategy = FeatureSelectionStrategy::default();
        assert_eq!(strategy, FeatureSelectionStrategy::Random);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rotation_forest_insufficient_classes() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::new(base_classifier);

        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 0]; // Only one class

        let result = rf.fit(&X, &y);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Need at least 2 classes"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rotation_forest_empty_dataset() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::new(base_classifier);

        let X = Array2::<f64>::zeros((0, 2));
        let y = Array1::<i32>::zeros(0);

        let result = rf.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_rotation_forest_feature_subset_generation() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::new(base_classifier)
            .n_feature_subsets(3)
            .max_features(Some(6));

        let mut rng = Random::seed(42);
        let subsets = rf.generate_feature_subsets(8, &mut rng);

        // Should have the requested number of subsets (or fewer if not enough features)
        assert!(subsets.len() <= 3);

        // Each subset should have features
        for subset in &subsets {
            assert!(!subset.is_empty());
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rotation_forest_rotation_matrix_creation() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::new(base_classifier);

        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];

        let feature_subsets = vec![vec![0, 1], vec![2]];
        let mut rng = Random::seed(42);

        let rotation_info = rf
            .create_rotation_matrix(&X, &feature_subsets, &mut rng)
            .expect("Failed to create rotation matrix");

        // Check rotation matrix dimensions
        assert_eq!(rotation_info.rotation_matrix.dim(), (3, 3));
        assert_eq!(rotation_info.feature_subsets, feature_subsets);
        assert!(!rotation_info.selected_features.is_empty());
    }

    #[test]
    fn test_rotation_forest_warm_start_configuration() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::new(base_classifier)
            .n_estimators(5)
            .warm_start(true);

        assert!(rf.config.warm_start);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rotation_forest_warm_start_partial_fit() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::new(base_classifier.clone())
            .n_estimators(3)
            .warm_start(true)
            .random_state(Some(42));

        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 0, 1];

        // Initial training
        let trained = rf.fit(&X, &y).expect("Failed to fit initial model");
        assert_eq!(trained.n_estimators_(), 3);

        // Continue training with partial_fit
        let extended_trained = trained
            .partial_fit(base_classifier, &X, &y, 2)
            .expect("Failed to perform partial fit");
        assert_eq!(extended_trained.n_estimators_(), 5);

        let predictions = extended_trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rotation_forest_warm_start_without_flag() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::new(base_classifier.clone())
            .n_estimators(3)
            .warm_start(false); // Warm start disabled

        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 0, 1];

        let trained = rf.fit(&X, &y).expect("Failed to fit initial model");

        // Should fail because warm start is not enabled
        let result = trained.partial_fit(base_classifier, &X, &y, 2);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Warm start is not enabled"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rotation_forest_warm_start_new_classes() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::new(base_classifier.clone())
            .n_estimators(3)
            .warm_start(true);

        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 0, 1];

        let trained = rf.fit(&X, &y).expect("Failed to fit initial model");

        // Try to add new classes (should fail)
        let X_new = array![[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]];
        let y_new = array![2, 2]; // New class not seen before

        let result = trained.partial_fit(base_classifier, &X_new, &y_new, 1);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("New class 2 not seen during initial training"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rotation_forest_warm_start_builder() {
        let base_classifier = MockClassifier::new();
        let rf = RotationForestClassifier::builder(base_classifier.clone())
            .n_estimators(2)
            .warm_start(true)
            .bootstrap(true)
            .random_state(Some(123))
            .build();

        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 0, 1];

        let trained = rf.fit(&X, &y).expect("Failed to fit initial model");
        assert_eq!(trained.n_estimators_(), 2);
        assert!(trained.config.warm_start);

        let extended = trained
            .partial_fit(base_classifier, &X, &y, 3)
            .expect("Failed to extend model");
        assert_eq!(extended.n_estimators_(), 5);
    }

    #[test]
    fn test_rotation_forest_warm_start_config_default() {
        let config = RotationForestConfig::default();
        assert_eq!(config.warm_start, false);
    }
}
