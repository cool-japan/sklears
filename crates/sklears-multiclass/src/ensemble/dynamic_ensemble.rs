//! Dynamic Ensemble Selection for multiclass classification
//!
//! This module provides Dynamic Ensemble Selection (DES) methods that select the best
//! classifier(s) from a pool for each test instance based on the local competence of
//! classifiers in the region around the test instance.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
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

type TrainedDynamicEnsembleSelection<T> =
    DynamicEnsembleSelectionClassifier<DynamicEnsembleSelectionTrainedData<T>, Trained>;

impl<C> Fit<Array2<f64>, Array1<i32>> for DynamicEnsembleSelectionClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    type Fitted = TrainedDynamicEnsembleSelection<C::Fitted>;

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

        // Split data into training and validation for competence estimation
        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        // Use 70% for training, 30% for validation
        let validation_size = (n_samples as f64 * 0.3) as usize;
        let mut indices: Vec<usize> = (0..n_samples).collect();
        for i in 0..n_samples {
            let j = rng.gen_range(i..n_samples);
            indices.swap(i, j);
        }

        let validation_indices = &indices[0..validation_size];
        let training_indices = &indices[validation_size..];

        let X_train = X.select(Axis(0), training_indices);
        let y_train = y.select(Axis(0), training_indices);
        let X_val = X.select(Axis(0), validation_indices);
        let y_val = y.select(Axis(0), validation_indices);

        // Generate classifier pool based on strategy
        let classifier_pool = self.generate_classifier_pool(&X_train, &y_train, &mut rng)?;

        // Get predictions on validation set for competence estimation
        let mut validation_predictions =
            Array2::<i32>::zeros((validation_size, self.config.n_estimators));
        for (i, classifier) in classifier_pool.iter().enumerate() {
            let predictions = classifier.predict(&X_val)?;
            for (j, &pred) in predictions.iter().enumerate() {
                validation_predictions[[j, i]] = pred;
            }
        }

        let trained_data = DynamicEnsembleSelectionTrainedData {
            classifier_pool,
            validation_X: X_val,
            validation_y: y_val,
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
    fn generate_classifier_pool(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Vec<C::Fitted>> {
        let mut pool = Vec::new();

        for _ in 0..self.config.n_estimators {
            let (X_subset, y_subset) = match self.config.pool_generation {
                PoolGenerationStrategy::Bagging => {
                    // Bootstrap sampling
                    self.bootstrap_sample(X, y, rng)
                }
                PoolGenerationStrategy::StdRngSubspace => {
                    // StdRng feature subset
                    self.random_subspace_sample(X, y, rng)
                }
                PoolGenerationStrategy::StdRngPatches => {
                    // Both random samples and features
                    let (X_boot, y_boot) = self.bootstrap_sample(X, y, rng);
                    self.random_subspace_sample(&X_boot, &y_boot, rng)
                }
                PoolGenerationStrategy::Heterogeneous => {
                    // For now, use bagging (in practice, this would use different algorithms)
                    self.bootstrap_sample(X, y, rng)
                }
            };

            let classifier = self.base_estimator.clone().fit(&X_subset, &y_subset)?;
            pool.push(classifier);
        }

        Ok(pool)
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

    #[allow(non_snake_case)]
    fn random_subspace_sample(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        rng: &mut CoreRandom<StdRng>,
    ) -> (Array2<f64>, Array1<i32>) {
        let n_features = X.ncols();
        let n_subset_features = (n_features as f64 * 0.7) as usize; // Use 70% of features

        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        for i in 0..n_subset_features {
            let j = rng.gen_range(i..n_features);
            feature_indices.swap(i, j);
        }
        feature_indices.truncate(n_subset_features);

        let X_subset = X.select(Axis(1), &feature_indices);
        (X_subset, y.clone())
    }
}

impl<T> Predict<Array2<f64>, Array1<i32>>
    for DynamicEnsembleSelectionClassifier<DynamicEnsembleSelectionTrainedData<T>, Trained>
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
    for DynamicEnsembleSelectionClassifier<DynamicEnsembleSelectionTrainedData<T>, Trained>
where
    T: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    fn predict_proba(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let trained_data = &self.base_estimator;

        let mut probabilities = Array2::<f64>::zeros((n_samples, trained_data.n_classes));

        // For each test instance, select classifiers and make prediction
        for i in 0..n_samples {
            let test_instance = X.row(i).to_owned();
            let test_instance_2d = test_instance.clone().insert_axis(Axis(0));

            // Find competence region for this test instance
            let competence_region = self.find_competence_region(&test_instance, trained_data);

            // Get predictions from selected classifiers
            let mut class_votes = Array1::<f64>::zeros(trained_data.n_classes);

            if competence_region.selected_classifiers.is_empty() {
                // Fallback: use all classifiers with equal weight
                for classifier in &trained_data.classifier_pool {
                    let pred = classifier.predict(&test_instance_2d)?;
                    if let Some(class_idx) = trained_data.classes.iter().position(|&c| c == pred[0])
                    {
                        class_votes[class_idx] += 1.0;
                    }
                }
            } else {
                // Use selected classifiers with competence-based weighting
                for &classifier_idx in competence_region.selected_classifiers.iter() {
                    let classifier = &trained_data.classifier_pool[classifier_idx];
                    let pred = classifier.predict(&test_instance_2d)?;
                    let weight = competence_region.competence_scores[classifier_idx];

                    if let Some(class_idx) = trained_data.classes.iter().position(|&c| c == pred[0])
                    {
                        class_votes[class_idx] += weight;
                    }
                }
            }

            // Normalize to probabilities
            let total_votes: f64 = class_votes.sum();
            if total_votes > 0.0 {
                for j in 0..trained_data.n_classes {
                    probabilities[[i, j]] = class_votes[j] / total_votes;
                }
            } else {
                // Uniform distribution fallback
                for j in 0..trained_data.n_classes {
                    probabilities[[i, j]] = 1.0 / trained_data.n_classes as f64;
                }
            }
        }

        Ok(probabilities)
    }
}

impl<T> DynamicEnsembleSelectionClassifier<DynamicEnsembleSelectionTrainedData<T>, Trained>
where
    T: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    fn find_competence_region(
        &self,
        test_instance: &Array1<f64>,
        trained_data: &DynamicEnsembleSelectionTrainedData<T>,
    ) -> CompetenceRegion {
        // Find k-nearest neighbors in validation set
        let mut distances = Vec::new();

        for (i, val_instance) in trained_data.validation_X.axis_iter(Axis(0)).enumerate() {
            let distance = self.euclidean_distance(test_instance, &val_instance.to_owned());
            distances.push((distance, i));
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbor_indices: Vec<usize> = distances
            .iter()
            .take(self.config.k_neighbors.min(distances.len()))
            .map(|(_, idx)| *idx)
            .collect();

        // Compute competence scores for each classifier
        let mut competence_scores = Array1::<f64>::zeros(trained_data.classifier_pool.len());

        for (classifier_idx, _) in trained_data.classifier_pool.iter().enumerate() {
            let competence =
                self.compute_competence(classifier_idx, &neighbor_indices, trained_data);
            competence_scores[classifier_idx] = competence;
        }

        // Select classifiers based on strategy
        let selected_classifiers = self.select_classifiers(&competence_scores);

        CompetenceRegion {
            neighbor_indices,
            competence_scores,
            selected_classifiers,
        }
    }

    fn euclidean_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn compute_competence(
        &self,
        classifier_idx: usize,
        neighbor_indices: &[usize],
        trained_data: &DynamicEnsembleSelectionTrainedData<T>,
    ) -> f64 {
        if neighbor_indices.is_empty() {
            return 0.0;
        }

        match self.config.competence_measure {
            CompetenceMeasure::Accuracy => {
                let mut correct = 0;
                for &idx in neighbor_indices {
                    let predicted = trained_data.validation_predictions[[idx, classifier_idx]];
                    let actual = trained_data.validation_y[idx];
                    if predicted == actual {
                        correct += 1;
                    }
                }
                correct as f64 / neighbor_indices.len() as f64
            }
            CompetenceMeasure::Distance => {
                // For simplicity, use inverse of average distance as competence
                // In practice, this would be more sophisticated
                let mut total_distance = 0.0;
                for &_idx in neighbor_indices {
                    // Simplified distance-based competence
                    total_distance += 1.0; // Placeholder
                }
                1.0 / (1.0 + total_distance / neighbor_indices.len() as f64)
            }
            CompetenceMeasure::Entropy | CompetenceMeasure::Likelihood => {
                // Fallback to accuracy for these measures
                let mut correct = 0;
                for &idx in neighbor_indices {
                    let predicted = trained_data.validation_predictions[[idx, classifier_idx]];
                    let actual = trained_data.validation_y[idx];
                    if predicted == actual {
                        correct += 1;
                    }
                }
                correct as f64 / neighbor_indices.len() as f64
            }
        }
    }

    fn select_classifiers(&self, competence_scores: &Array1<f64>) -> Vec<usize> {
        match &self.config.selection_strategy {
            SelectionStrategy::Best => {
                // Select the single best classifier
                if let Some((best_idx, _)) = competence_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                {
                    vec![best_idx]
                } else {
                    vec![]
                }
            }
            SelectionStrategy::TopK(k) => {
                // Select top-k classifiers
                let mut indexed_scores: Vec<(usize, f64)> = competence_scores
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| (i, score))
                    .collect();

                indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed_scores
                    .iter()
                    .take(*k)
                    .map(|(idx, _)| *idx)
                    .collect()
            }
            SelectionStrategy::Threshold => {
                // Select all classifiers above threshold
                competence_scores
                    .iter()
                    .enumerate()
                    .filter(|(_, &score)| score >= self.config.competence_threshold)
                    .map(|(idx, _)| idx)
                    .collect()
            }
            SelectionStrategy::Weighted => {
                // Select all classifiers (weighting handled in prediction)
                (0..competence_scores.len()).collect()
            }
        }
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
    }

    impl MockClassifier {
        fn new() -> Self {
            Self { classes: vec![] }
        }
    }

    #[derive(Debug, Clone)]
    struct TrainedMockClassifier {
        classes: Array1<i32>,
    }

    impl Fit<Array2<f64>, Array1<i32>> for MockClassifier {
        type Fitted = TrainedMockClassifier;

        fn fit(self, _X: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            Ok(TrainedMockClassifier {
                classes: Array1::from_vec(classes),
            })
        }
    }

    impl Predict<Array2<f64>, Array1<i32>> for TrainedMockClassifier {
        fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<i32>> {
            let n_samples = X.nrows();
            let mut predictions = Array1::zeros(n_samples);

            // Simple prediction: alternate between classes based on first feature
            for i in 0..n_samples {
                let class_idx = if X[[i, 0]] > 2.0 { 1 } else { 0 };
                predictions[i] = self.classes[class_idx.min(self.classes.len() - 1)];
            }

            Ok(predictions)
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dynamic_ensemble_selection_basic() {
        let base_classifier = MockClassifier::new();
        let des = DynamicEnsembleSelectionClassifier::new(base_classifier)
            .n_estimators(5)
            .k_neighbors(3);

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0]
        ];
        let y = array![0, 1, 0, 1, 0, 1, 0, 1];

        let trained = des
            .fit(&X, &y)
            .expect("Failed to fit Dynamic Ensemble Selection");

        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 8);

        // Check probabilities
        let probabilities = trained
            .predict_proba(&X)
            .expect("Failed to predict probabilities");
        assert_eq!(probabilities.dim(), (8, 2));

        // Probabilities should sum to 1
        for i in 0..8 {
            let sum: f64 = probabilities.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1");
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dynamic_ensemble_selection_builder() {
        let base_classifier = MockClassifier::new();
        let des = DynamicEnsembleSelectionClassifier::builder(base_classifier)
            .n_estimators(8)
            .competence_measure(CompetenceMeasure::Accuracy)
            .k_neighbors(5)
            .selection_strategy(SelectionStrategy::TopK(3))
            .competence_threshold(0.6)
            .pool_generation(PoolGenerationStrategy::StdRngSubspace)
            .random_state(Some(42))
            .build();

        let X = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 1, 2, 0];

        let trained = des.fit(&X, &y).expect("Failed to fit with builder");
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_dynamic_ensemble_selection_config_default() {
        let config = DynamicEnsembleSelectionConfig::default();
        assert_eq!(config.n_estimators, 15);
        assert_eq!(config.competence_measure, CompetenceMeasure::Accuracy);
        assert_eq!(config.k_neighbors, 7);
        assert_eq!(config.selection_strategy, SelectionStrategy::Best);
        assert_eq!(config.competence_threshold, 0.5);
        assert_eq!(config.pool_generation, PoolGenerationStrategy::Bagging);
        assert_eq!(config.random_state, None);
        assert_eq!(config.n_jobs, None);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dynamic_ensemble_selection_insufficient_classes() {
        let base_classifier = MockClassifier::new();
        let des = DynamicEnsembleSelectionClassifier::new(base_classifier);

        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 0]; // Only one class

        let result = des.fit(&X, &y);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Need at least 2 classes"));
    }

    #[test]
    fn test_dynamic_ensemble_selection_clone() {
        let base_classifier = MockClassifier::new();
        let des = DynamicEnsembleSelectionClassifier::new(base_classifier);

        let _cloned = des.clone();
        // If clone works, this test passes
    }

    #[test]
    fn test_dynamic_ensemble_selection_estimator_trait() {
        let classifier = MockClassifier::new();
        let des = DynamicEnsembleSelectionClassifier::new(classifier);

        let config = des.config();
        assert_eq!(config.n_estimators, 15);
        assert_eq!(config.k_neighbors, 7);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dynamic_ensemble_selection_reproducibility() {
        let base_classifier = MockClassifier::new();

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0]
        ];
        let y = array![0, 1, 0, 1, 0, 1, 0, 1];

        // Train two models with same random state
        let des1 = DynamicEnsembleSelectionClassifier::new(base_classifier.clone())
            .n_estimators(4)
            .random_state(Some(42));
        let trained1 = des1.fit(&X, &y).expect("Failed to fit first model");

        let des2 = DynamicEnsembleSelectionClassifier::new(base_classifier)
            .n_estimators(4)
            .random_state(Some(42));
        let trained2 = des2.fit(&X, &y).expect("Failed to fit second model");

        let pred1 = trained1.predict(&X).expect("Failed to predict first");
        let pred2 = trained2.predict(&X).expect("Failed to predict second");

        // Results should be identical with same random state
        assert_eq!(pred1.to_vec(), pred2.to_vec());
    }
}
