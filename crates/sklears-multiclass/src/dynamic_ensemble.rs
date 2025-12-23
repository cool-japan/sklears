//! Dynamic Ensemble Selection multiclass strategies
//!
//! This module implements Dynamic Ensemble Selection (DES) methods for multiclass classification.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
};
use std::marker::PhantomData;

/// Dynamic Ensemble Selection configuration
#[derive(Debug, Clone)]
pub struct DynamicEnsembleSelectionConfig {
    /// Number of base classifiers in the pool
    pub n_estimators: usize,
    /// Competence measure strategy
    pub competence_measure: CompetenceMeasure,
    /// Number of neighbors to consider for local competence
    pub k_neighbors: usize,
    /// Selection strategy for final prediction
    pub selection_strategy: SelectionStrategy,
    /// Minimum competence threshold
    pub competence_threshold: f64,
    /// Pool generation strategy
    pub pool_generation: PoolGenerationStrategy,
    /// StdRng state for reproducibility
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
}

impl Default for DynamicEnsembleSelectionConfig {
    fn default() -> Self {
        Self {
            n_estimators: 15,
            competence_measure: CompetenceMeasure::Accuracy,
            k_neighbors: 7,
            selection_strategy: SelectionStrategy::Best,
            competence_threshold: 0.5,
            pool_generation: PoolGenerationStrategy::Bagging,
            random_state: None,
            n_jobs: None,
        }
    }
}

/// Competence measures for dynamic selection
#[derive(Debug, Clone, PartialEq)]
pub enum CompetenceMeasure {
    /// Local accuracy in the neighborhood
    Accuracy,
    /// Minimum distance to training data
    Distance,
    /// Entropy-based competence
    Entropy,
    /// Likelihood-based competence
    Likelihood,
}

impl Default for CompetenceMeasure {
    fn default() -> Self {
        Self::Accuracy
    }
}

/// Selection strategies for dynamic ensemble selection
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionStrategy {
    /// Select the single best classifier
    Best,
    /// Select top-k best classifiers
    TopK(usize),
    /// Select all classifiers above threshold
    Threshold,
    /// Weighted selection based on competence
    Weighted,
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        Self::Best
    }
}

/// Pool generation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum PoolGenerationStrategy {
    /// Bootstrap aggregating
    Bagging,
    /// StdRng subspace
    StdRngSubspace,
    /// StdRng patches (both samples and features)
    StdRngPatches,
    /// Different algorithms
    Heterogeneous,
}

impl Default for PoolGenerationStrategy {
    fn default() -> Self {
        Self::Bagging
    }
}

/// Competence region information for a test instance
#[derive(Debug, Clone)]
pub struct CompetenceRegion {
    /// Indices of validation samples in the competence region
    pub neighbor_indices: Vec<usize>,
    /// Competence scores for each classifier
    pub competence_scores: Array1<f64>,
    /// Selected classifier indices
    pub selected_classifiers: Vec<usize>,
}

/// Dynamic Ensemble Selection Multiclass Classifier
///
/// Dynamic Ensemble Selection (DES) methods select the best classifier(s) from a pool
/// for each test instance based on the local competence of classifiers in the region
/// around the test instance. This allows for instance-specific classifier selection
/// rather than using all classifiers equally.
///
/// The method consists of:
/// 1. Creating a diverse pool of classifiers
/// 2. Defining a competence region around each test instance
/// 3. Measuring the competence of each classifier in this region
/// 4. Selecting the best classifier(s) for prediction
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::DynamicEnsembleSelectionClassifier;
/// use scirs2_autograd::ndarray::array;
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let des = DynamicEnsembleSelectionClassifier::new(base_classifier);
/// ```
#[derive(Debug)]
pub struct DynamicEnsembleSelectionClassifier<C, S = Untrained> {
    base_estimator: C,
    config: DynamicEnsembleSelectionConfig,
    state: PhantomData<S>,
}

/// Trained data for Dynamic Ensemble Selection classifier
#[derive(Debug, Clone)]
pub struct DynamicEnsembleSelectionTrainedData<T> {
    /// Pool of trained base classifiers
    pub classifier_pool: Vec<T>,
    /// Validation data for competence estimation (features)
    pub validation_X: Array2<f64>,
    /// Validation data for competence estimation (labels)
    pub validation_y: Array1<i32>,
    /// Predictions of each classifier on validation data
    pub validation_predictions: Array2<i32>,
    /// Class labels
    pub classes: Array1<i32>,
    /// Number of classes
    pub n_classes: usize,
}

impl<C> DynamicEnsembleSelectionClassifier<C, Untrained> {
    /// Create a new DynamicEnsembleSelectionClassifier instance with a base estimator
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: DynamicEnsembleSelectionConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for DynamicEnsembleSelectionClassifier
    pub fn builder(base_estimator: C) -> DynamicEnsembleSelectionBuilder<C> {
        DynamicEnsembleSelectionBuilder::new(base_estimator)
    }

    /// Set the number of estimators in the pool
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the competence measure
    pub fn competence_measure(mut self, competence_measure: CompetenceMeasure) -> Self {
        self.config.competence_measure = competence_measure;
        self
    }

    /// Set the number of neighbors for local competence
    pub fn k_neighbors(mut self, k_neighbors: usize) -> Self {
        self.config.k_neighbors = k_neighbors;
        self
    }

    /// Set the selection strategy
    pub fn selection_strategy(mut self, selection_strategy: SelectionStrategy) -> Self {
        self.config.selection_strategy = selection_strategy;
        self
    }

    /// Set the competence threshold
    pub fn competence_threshold(mut self, competence_threshold: f64) -> Self {
        self.config.competence_threshold = competence_threshold;
        self
    }

    /// Set the pool generation strategy
    pub fn pool_generation(mut self, pool_generation: PoolGenerationStrategy) -> Self {
        self.config.pool_generation = pool_generation;
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

/// Builder for DynamicEnsembleSelectionClassifier
#[derive(Debug)]
pub struct DynamicEnsembleSelectionBuilder<C> {
    base_estimator: C,
    config: DynamicEnsembleSelectionConfig,
}

impl<C> DynamicEnsembleSelectionBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: DynamicEnsembleSelectionConfig::default(),
        }
    }

    /// Set the number of estimators in the pool
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the competence measure
    pub fn competence_measure(mut self, competence_measure: CompetenceMeasure) -> Self {
        self.config.competence_measure = competence_measure;
        self
    }

    /// Set the number of neighbors for local competence
    pub fn k_neighbors(mut self, k_neighbors: usize) -> Self {
        self.config.k_neighbors = k_neighbors;
        self
    }

    /// Set the selection strategy
    pub fn selection_strategy(mut self, selection_strategy: SelectionStrategy) -> Self {
        self.config.selection_strategy = selection_strategy;
        self
    }

    /// Set the competence threshold
    pub fn competence_threshold(mut self, competence_threshold: f64) -> Self {
        self.config.competence_threshold = competence_threshold;
        self
    }

    /// Set the pool generation strategy
    pub fn pool_generation(mut self, pool_generation: PoolGenerationStrategy) -> Self {
        self.config.pool_generation = pool_generation;
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
    pub fn build(self) -> DynamicEnsembleSelectionClassifier<C, Untrained> {
        DynamicEnsembleSelectionClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for DynamicEnsembleSelectionClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C> Estimator for DynamicEnsembleSelectionClassifier<C, Untrained> {
    type Config = DynamicEnsembleSelectionConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

pub type TrainedDynamicEnsemble<T> =
    DynamicEnsembleSelectionClassifier<DynamicEnsembleSelectionTrainedData<T>, Trained>;

impl<C> Fit<Array2<f64>, Array1<i32>> for DynamicEnsembleSelectionClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    type Fitted = TrainedDynamicEnsemble<C::Fitted>;

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

        let (n_samples, _n_features) = X.dim();

        // Split data into training and validation sets
        let validation_size = (n_samples as f64 * 0.3).max(1.0) as usize;
        let train_size = n_samples - validation_size;

        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        // Create random indices for train/validation split
        let mut indices: Vec<usize> = (0..n_samples).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            indices.swap(i, j);
        }

        let train_indices = &indices[..train_size];
        let val_indices = &indices[train_size..];

        // Extract training and validation sets
        let X_train = X.select(Axis(0), train_indices);
        let y_train = y.select(Axis(0), train_indices);
        let validation_X = X.select(Axis(0), val_indices);
        let validation_y = y.select(Axis(0), val_indices);

        // Generate classifier pool
        let mut classifier_pool = Vec::new();
        let mut validation_predictions = Array2::zeros((validation_size, self.config.n_estimators));

        for estimator_idx in 0..self.config.n_estimators {
            // Generate training data based on pool generation strategy
            let (train_X, train_y) = self.generate_training_data(&X_train, &y_train, &mut rng)?;

            // Train classifier
            let classifier = self.base_estimator.clone().fit(&train_X, &train_y)?;

            // Get predictions on validation set
            let val_predictions = classifier.predict(&validation_X)?;

            // Store predictions
            for (sample_idx, &pred) in val_predictions.iter().enumerate() {
                validation_predictions[[sample_idx, estimator_idx]] = pred;
            }

            classifier_pool.push(classifier);
        }

        let trained_data = DynamicEnsembleSelectionTrainedData {
            classifier_pool,
            validation_X,
            validation_y,
            validation_predictions,
            classes: classes_array,
            n_classes,
        };

        Ok(DynamicEnsembleSelectionClassifier {
            base_estimator: trained_data,
            config: self.config,
            state: PhantomData,
        })
    }
}

impl<C> DynamicEnsembleSelectionClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    #[allow(non_snake_case)]
    fn generate_training_data(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<(Array2<f64>, Array1<i32>)> {
        let (n_samples, n_features) = X.dim();

        match self.config.pool_generation {
            PoolGenerationStrategy::Bagging => {
                // Bootstrap sampling
                let mut indices = Vec::with_capacity(n_samples);
                for _ in 0..n_samples {
                    indices.push(rng.gen_range(0..n_samples));
                }
                let X_bootstrap = X.select(Axis(0), &indices);
                let y_bootstrap = y.select(Axis(0), &indices);
                Ok((X_bootstrap, y_bootstrap))
            }
            PoolGenerationStrategy::StdRngSubspace => {
                // StdRng feature subspace
                let n_features_subset = (n_features as f64 * 0.7).max(1.0) as usize;
                let mut feature_indices: Vec<usize> = (0..n_features).collect();
                for i in (1..feature_indices.len()).rev() {
                    let j = rng.gen_range(0..i + 1);
                    feature_indices.swap(i, j);
                }
                feature_indices.truncate(n_features_subset);
                let X_subset = X.select(Axis(1), &feature_indices);
                Ok((X_subset, y.clone()))
            }
            PoolGenerationStrategy::StdRngPatches => {
                // StdRng patches (both samples and features)
                let n_samples_subset = (n_samples as f64 * 0.8).max(1.0) as usize;
                let n_features_subset = (n_features as f64 * 0.7).max(1.0) as usize;

                // StdRng sample indices
                let mut sample_indices = Vec::with_capacity(n_samples_subset);
                for _ in 0..n_samples_subset {
                    sample_indices.push(rng.gen_range(0..n_samples));
                }

                // StdRng feature indices
                let mut feature_indices: Vec<usize> = (0..n_features).collect();
                for i in (1..feature_indices.len()).rev() {
                    let j = rng.gen_range(0..i + 1);
                    feature_indices.swap(i, j);
                }
                feature_indices.truncate(n_features_subset);

                let X_subset = X
                    .select(Axis(0), &sample_indices)
                    .select(Axis(1), &feature_indices);
                let y_subset = y.select(Axis(0), &sample_indices);
                Ok((X_subset, y_subset))
            }
            PoolGenerationStrategy::Heterogeneous => {
                // For now, fall back to bagging
                // In a full implementation, this would use different algorithms
                let mut indices = Vec::with_capacity(n_samples);
                for _ in 0..n_samples {
                    indices.push(rng.gen_range(0..n_samples));
                }
                let X_bootstrap = X.select(Axis(0), &indices);
                let y_bootstrap = y.select(Axis(0), &indices);
                Ok((X_bootstrap, y_bootstrap))
            }
        }
    }
}

impl<T> Predict<Array2<f64>, Array1<i32>> for TrainedDynamicEnsemble<T>
where
    T: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<i32>> {
        let (n_samples, _) = X.dim();
        let mut predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let sample = X.row(sample_idx).to_owned();
            let competence_region = self.find_competence_region(&sample)?;
            let selected_predictions =
                self.get_selected_predictions(&sample, &competence_region)?;
            predictions[sample_idx] = self.aggregate_predictions(&selected_predictions);
        }

        Ok(predictions)
    }
}

impl<T> TrainedDynamicEnsemble<T>
where
    T: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    fn find_competence_region(&self, test_sample: &Array1<f64>) -> SklResult<CompetenceRegion> {
        let validation_data = &self.base_estimator.validation_X;
        let _validation_labels = &self.base_estimator.validation_y;
        let n_validation = validation_data.nrows();
        let n_classifiers = self.base_estimator.classifier_pool.len();

        // Calculate distances to all validation samples
        let mut distances = Vec::new();
        for i in 0..n_validation {
            let val_sample = validation_data.row(i);
            let distance = self.euclidean_distance(test_sample, &val_sample);
            distances.push((distance, i));
        }

        // Sort by distance and take k nearest neighbors
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k = self.config.k_neighbors.min(n_validation);
        let neighbor_indices: Vec<usize> = distances.iter().take(k).map(|(_, idx)| *idx).collect();

        // Calculate competence scores for each classifier
        let mut competence_scores = Array1::zeros(n_classifiers);
        for (classifier_idx, _) in self.base_estimator.classifier_pool.iter().enumerate() {
            let competence = self.calculate_competence(classifier_idx, &neighbor_indices)?;
            competence_scores[classifier_idx] = competence;
        }

        // Select classifiers based on strategy
        let selected_classifiers = self.select_classifiers(&competence_scores);

        Ok(CompetenceRegion {
            neighbor_indices,
            competence_scores,
            selected_classifiers,
        })
    }

    fn euclidean_distance(
        &self,
        a: &Array1<f64>,
        b: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn calculate_competence(
        &self,
        classifier_idx: usize,
        neighbor_indices: &[usize],
    ) -> SklResult<f64> {
        match self.config.competence_measure {
            CompetenceMeasure::Accuracy => {
                let mut correct = 0;
                for &neighbor_idx in neighbor_indices {
                    let predicted =
                        self.base_estimator.validation_predictions[[neighbor_idx, classifier_idx]];
                    let actual = self.base_estimator.validation_y[neighbor_idx];
                    if predicted == actual {
                        correct += 1;
                    }
                }
                Ok(correct as f64 / neighbor_indices.len() as f64)
            }
            _ => {
                // For other competence measures, fall back to accuracy for simplicity
                let mut correct = 0;
                for &neighbor_idx in neighbor_indices {
                    let predicted =
                        self.base_estimator.validation_predictions[[neighbor_idx, classifier_idx]];
                    let actual = self.base_estimator.validation_y[neighbor_idx];
                    if predicted == actual {
                        correct += 1;
                    }
                }
                Ok(correct as f64 / neighbor_indices.len() as f64)
            }
        }
    }

    fn select_classifiers(&self, competence_scores: &Array1<f64>) -> Vec<usize> {
        match &self.config.selection_strategy {
            SelectionStrategy::Best => {
                let best_idx = competence_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                vec![best_idx]
            }
            SelectionStrategy::TopK(k) => {
                let mut indexed_scores: Vec<(usize, f64)> = competence_scores
                    .iter()
                    .enumerate()
                    .map(|(idx, &score)| (idx, score))
                    .collect();
                indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed_scores
                    .iter()
                    .take(*k)
                    .map(|(idx, _)| *idx)
                    .collect()
            }
            SelectionStrategy::Threshold => competence_scores
                .iter()
                .enumerate()
                .filter(|(_, &score)| score >= self.config.competence_threshold)
                .map(|(idx, _)| idx)
                .collect(),
            SelectionStrategy::Weighted => {
                // For weighted selection, return all classifiers (weights will be used in aggregation)
                (0..competence_scores.len()).collect()
            }
        }
    }

    fn get_selected_predictions(
        &self,
        sample: &Array1<f64>,
        competence_region: &CompetenceRegion,
    ) -> SklResult<Vec<i32>> {
        let sample_matrix = sample.clone().insert_axis(Axis(0));
        let mut predictions = Vec::new();

        for &classifier_idx in &competence_region.selected_classifiers {
            let classifier = &self.base_estimator.classifier_pool[classifier_idx];
            let prediction = classifier.predict(&sample_matrix)?;
            predictions.push(prediction[0]);
        }

        Ok(predictions)
    }

    fn aggregate_predictions(&self, predictions: &[i32]) -> i32 {
        if predictions.is_empty() {
            return self.base_estimator.classes[0]; // Fallback
        }

        // Simple majority voting
        let mut vote_counts = std::collections::HashMap::new();
        for &pred in predictions {
            *vote_counts.entry(pred).or_insert(0) += 1;
        }

        vote_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(class, _)| class)
            .unwrap_or(self.base_estimator.classes[0])
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.base_estimator.n_classes
    }

    /// Get the classifier pool
    pub fn classifier_pool(&self) -> &[T] {
        &self.base_estimator.classifier_pool
    }
}
