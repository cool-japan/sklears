//! Rotation Forest ensemble classifier
//!
//! This module implements the Rotation Forest algorithm for multiclass classification.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
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
    /// StdRng state for reproducibility
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// Feature selection strategy
    pub feature_selection_strategy: FeatureSelectionStrategy,
    /// Number of feature subsets per classifier
    pub n_feature_subsets: usize,
}

impl Default for RotationForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            max_features: None,
            bootstrap: true,
            random_state: None,
            n_jobs: None,
            feature_selection_strategy: FeatureSelectionStrategy::StdRng,
            n_feature_subsets: 3,
        }
    }
}

/// Feature selection strategies for Rotation Forest
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureSelectionStrategy {
    /// StdRng selection of features
    StdRng,
    /// Select features based on variance
    Variance,
    /// Use all available features
    All,
}

impl Default for FeatureSelectionStrategy {
    fn default() -> Self {
        Self::StdRng
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
/// use sklears_multiclass::RotationForestClassifier;
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

pub type TrainedRotationForest<T> = RotationForestClassifier<RotationForestTrainedData<T>, Trained>;

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

        let (_n_samples, n_features) = X.dim();

        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        let mut estimators = Vec::new();
        let mut rotations = Vec::new();

        // Train each estimator with different rotated feature space
        for _estimator_idx in 0..self.config.n_estimators {
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
    fn generate_feature_subsets(
        &self,
        n_features: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> Vec<Vec<usize>> {
        let max_features = self.config.max_features.unwrap_or(n_features);
        let subset_size = (max_features / self.config.n_feature_subsets).max(1);

        let mut all_features: Vec<usize> = (0..n_features).collect();
        let mut subsets = Vec::new();

        match self.config.feature_selection_strategy {
            FeatureSelectionStrategy::StdRng => {
                // StdRngly shuffle features and create subsets
                for _ in 0..n_features {
                    let i = rng.gen_range(0..n_features);
                    let j = rng.gen_range(0..n_features);
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
                    let i = rng.gen_range(0..n_features);
                    let j = rng.gen_range(0..n_features);
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
    fn create_rotation_matrix(
        &self,
        X: &Array2<f64>,
        feature_subsets: &[Vec<usize>],
        _rng: &mut CoreRandom<StdRng>,
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

            // For simplicity, we'll use a simplified rotation based on the covariance matrix
            // In a full implementation, this would use proper SVD/eigendecomposition
            let subset_rotation = self.simple_pca_rotation(&cov_matrix, subset.len());

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

    fn simple_pca_rotation(&self, cov_matrix: &Array2<f64>, size: usize) -> Array2<f64> {
        // Simplified PCA rotation - in practice, this would use proper eigendecomposition
        // For now, we'll create a rotation based on the diagonal elements
        let mut rotation = Array2::<f64>::eye(size);

        // Add some rotation based on covariance matrix structure
        for i in 0..size {
            for j in 0..size {
                if i != j && cov_matrix.dim().0 > i && cov_matrix.dim().1 > j {
                    let cov_val = cov_matrix[[i, j]];
                    rotation[[i, j]] = cov_val * 0.1; // Small rotation factor
                }
            }
        }

        // Normalize to maintain orthogonality (simplified)
        let norm = rotation
            .mapv(|x| x * x)
            .sum_axis(Axis(1))
            .mapv(|x| x.sqrt());
        for i in 0..size {
            if norm[i] > 1e-10 {
                rotation.row_mut(i).map_inplace(|x| *x /= norm[i]);
            }
        }

        rotation
    }

    #[allow(non_snake_case)]
    fn bootstrap_sample(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        rng: &mut CoreRandom<StdRng>,
    ) -> (Array2<f64>, Array1<i32>) {
        let n_samples = X.nrows();
        let mut indices = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            indices.push(rng.gen_range(0..n_samples));
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

impl<T> TrainedRotationForest<T> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.base_estimator.n_classes
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.base_estimator.n_features
    }

    /// Get the fitted estimators
    pub fn estimators(&self) -> &[T] {
        &self.base_estimator.estimators
    }

    /// Get the rotation information
    pub fn rotations(&self) -> &[RotationInfo] {
        &self.base_estimator.rotations
    }
}
