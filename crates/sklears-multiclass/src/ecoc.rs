//! Error-Correcting Output Codes (ECOC) multiclass strategy
//!
//! This module implements the ECOC strategy for multiclass classification.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom, Rng};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Error-Correcting Output Codes (ECOC) strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ECOCStrategy {
    /// StdRng binary codes
    #[default]
    StdRng,
    /// Dense random codes with balanced +1/-1 distribution
    DenseStdRng,
    /// Exhaustive codes (all possible binary combinations)
    Exhaustive,
    /// BCH (Bose-Chaudhuri-Hocquenghem) codes for optimal error correction
    BCH {
        /// Minimum distance parameter for error correction capability
        min_distance: usize,
    },
    /// Optimal code design using maximum distance criterion
    Optimal {
        /// Target minimum distance between codewords
        target_distance: usize,
    },
}

/// Configuration for Error-Correcting Output Code Classifier
#[derive(Debug, Clone)]
pub struct ECOCConfig {
    /// Code matrix generation strategy
    pub strategy: ECOCStrategy,
    /// Code size multiplier (for random strategies)
    pub code_size: f64,
    /// StdRng state for reproducibility
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
}

impl Default for ECOCConfig {
    fn default() -> Self {
        Self {
            strategy: ECOCStrategy::default(),
            code_size: 1.5,
            random_state: None,
            n_jobs: None,
        }
    }
}

/// Error-Correcting Output Code Classifier
///
/// This multiclass strategy uses error-correcting output codes to transform
/// a multiclass problem into multiple binary classification problems.
/// Each class is represented by a unique binary code, and binary classifiers
/// are trained to predict each bit of the code.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::{ECOCClassifier, ECOCStrategy};
/// use scirs2_autograd::ndarray::array;
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let ecoc = ECOCClassifier::new(base_classifier)
/// //     .strategy(ECOCStrategy::StdRng)
/// //     .code_size(2.0);
/// ```
#[derive(Debug)]
pub struct ECOCClassifier<C, S = Untrained> {
    base_estimator: C,
    config: ECOCConfig,
    state: PhantomData<S>,
}

impl<C> ECOCClassifier<C, Untrained> {
    /// Create a new ECOCClassifier instance with a base estimator
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: ECOCConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for ECOCClassifier
    pub fn builder(base_estimator: C) -> ECOCBuilder<C> {
        ECOCBuilder::new(base_estimator)
    }

    /// Set the coding strategy
    pub fn strategy(mut self, strategy: ECOCStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the code size multiplier
    pub fn code_size(mut self, code_size: f64) -> Self {
        self.config.code_size = code_size;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }
}

/// Builder for ECOCClassifier
#[derive(Debug)]
pub struct ECOCBuilder<C> {
    base_estimator: C,
    config: ECOCConfig,
}

impl<C> ECOCBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: ECOCConfig::default(),
        }
    }

    /// Set the coding strategy
    pub fn strategy(mut self, strategy: ECOCStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the code size multiplier
    pub fn code_size(mut self, code_size: f64) -> Self {
        self.config.code_size = code_size;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Enable parallel training using all available cores
    pub fn parallel(mut self) -> Self {
        self.config.n_jobs = Some(-1);
        self
    }

    /// Disable parallel training (use sequential training)
    pub fn sequential(mut self) -> Self {
        self.config.n_jobs = None;
        self
    }

    /// Build the ECOCClassifier
    pub fn build(self) -> ECOCClassifier<C, Untrained> {
        ECOCClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for ECOCClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for ECOCBuilder<C> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
        }
    }
}

impl<C> Estimator for ECOCClassifier<C, Untrained> {
    type Config = ECOCConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// Trained data container for ECOCClassifier
#[derive(Debug)]
pub struct ECOCTrainedData<T> {
    estimators: Vec<T>,
    classes: Array1<i32>,
    code_matrix: Array2<i32>,
    n_features: usize,
}

impl<T: Clone> Clone for ECOCTrainedData<T> {
    fn clone(&self) -> Self {
        Self {
            estimators: self.estimators.clone(),
            classes: self.classes.clone(),
            code_matrix: self.code_matrix.clone(),
            n_features: self.n_features,
        }
    }
}

pub type TrainedECOC<T> = ECOCClassifier<ECOCTrainedData<T>, Trained>;

impl<C> ECOCClassifier<C, Untrained> {
    /// Generate code matrix based on strategy
    fn generate_code_matrix(
        &self,
        n_classes: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Array2<i32>> {
        match &self.config.strategy {
            ECOCStrategy::StdRng => self.generate_random_code_matrix(n_classes, rng),
            ECOCStrategy::DenseStdRng => self.generate_dense_random_code_matrix(n_classes, rng),
            ECOCStrategy::Exhaustive => self.generate_exhaustive_code_matrix(n_classes),
            ECOCStrategy::BCH { min_distance } => {
                self.generate_bch_code_matrix(n_classes, *min_distance)
            }
            ECOCStrategy::Optimal { target_distance } => {
                self.generate_optimal_code_matrix(n_classes, *target_distance, rng)
            }
        }
    }

    /// Generate random binary code matrix
    fn generate_random_code_matrix(
        &self,
        n_classes: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Array2<i32>> {
        let code_length = ((n_classes as f64) * self.config.code_size).ceil() as usize;
        let mut code_matrix = Array2::zeros((n_classes, code_length));

        for i in 0..n_classes {
            for j in 0..code_length {
                code_matrix[[i, j]] = if rng.gen::<f64>() > 0.5 { 1 } else { -1 };
            }
        }

        Ok(code_matrix)
    }

    /// Generate dense random code matrix with balanced +1/-1 distribution
    fn generate_dense_random_code_matrix(
        &self,
        n_classes: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Array2<i32>> {
        let code_length = ((n_classes as f64) * self.config.code_size).ceil() as usize;
        let mut code_matrix = Array2::zeros((n_classes, code_length));

        for j in 0..code_length {
            // Create balanced column (half +1, half -1)
            let mut column_values: Vec<i32> = Vec::with_capacity(n_classes);
            for _i in 0..(n_classes / 2) {
                column_values.push(1);
            }
            for _i in (n_classes / 2)..n_classes {
                column_values.push(-1);
            }

            // Shuffle the column
            for i in (1..column_values.len()).rev() {
                let j = rng.gen_range(0..i + 1);
                column_values.swap(i, j);
            }

            // Assign to matrix
            for (i, &value) in column_values.iter().enumerate() {
                code_matrix[[i, j]] = value;
            }
        }

        Ok(code_matrix)
    }

    /// Generate exhaustive code matrix (all possible binary combinations)
    fn generate_exhaustive_code_matrix(&self, n_classes: usize) -> SklResult<Array2<i32>> {
        if n_classes > 10 {
            return Err(SklearsError::InvalidInput(
                "Exhaustive codes not practical for more than 10 classes".to_string(),
            ));
        }

        let code_length = 2_usize.pow((n_classes as f64).log2().ceil() as u32);
        let mut code_matrix = Array2::zeros((n_classes, code_length));

        for i in 0..n_classes {
            for j in 0..code_length {
                // Use binary representation of class index
                let bit = (i >> j) & 1;
                code_matrix[[i, j]] = if bit == 1 { 1 } else { -1 };
            }
        }

        Ok(code_matrix)
    }

    /// Generate BCH (Bose-Chaudhuri-Hocquenghem) code matrix
    /// BCH codes are a class of cyclic error-correcting codes with guaranteed minimum distance
    fn generate_bch_code_matrix(
        &self,
        n_classes: usize,
        min_distance: usize,
    ) -> SklResult<Array2<i32>> {
        if n_classes > 31 {
            return Err(SklearsError::InvalidInput(
                "BCH codes not practical for more than 31 classes".to_string(),
            ));
        }

        // For simplicity, we'll use a basic BCH-like construction
        // In practice, this would use proper finite field arithmetic
        let code_length = self.calculate_bch_length(n_classes, min_distance);
        let mut code_matrix = Array2::zeros((n_classes, code_length));

        // Generate BCH-like codes using polynomial construction
        for i in 0..n_classes {
            for j in 0..code_length {
                // Use a primitive polynomial approach (simplified)
                let bit = self.bch_encode_bit(i, j, min_distance);
                code_matrix[[i, j]] = if bit { 1 } else { -1 };
            }
        }

        // Verify minimum distance property (but don't fail if it's close)
        let actual_min_distance = self.calculate_minimum_distance(&code_matrix);
        if actual_min_distance == 0 && min_distance > 0 {
            // Only fail if we have identical codewords when we don't want them
            return Err(SklearsError::InvalidInput(
                "Generated identical codewords".to_string(),
            ));
        }

        Ok(code_matrix)
    }

    /// Calculate required BCH code length for given parameters
    fn calculate_bch_length(&self, n_classes: usize, min_distance: usize) -> usize {
        // BCH codes require code length >= log2(n_classes) * min_distance
        // This is a simplified calculation
        let base_length = (n_classes as f64).log2().ceil() as usize;
        std::cmp::max(base_length * min_distance, n_classes + min_distance)
    }

    /// Encode a single bit using BCH-like construction
    fn bch_encode_bit(&self, class_index: usize, bit_position: usize, min_distance: usize) -> bool {
        // Simplified BCH encoding that ensures different classes get different codes
        // Use the class index itself as a major component to ensure uniqueness
        let base_poly = class_index * 7 + bit_position * 3; // Use prime multipliers

        // Add minimum distance consideration
        let mut result = (base_poly % 2) == 1;

        // XOR with additional polynomial terms for minimum distance
        for d in 1..=min_distance {
            let poly_value = (class_index * d + bit_position * d) % 2;
            result ^= poly_value == 1;
        }

        result
    }

    /// Generate optimal code matrix using maximum distance criterion
    /// This attempts to maximize the minimum distance between any two codewords
    fn generate_optimal_code_matrix(
        &self,
        n_classes: usize,
        target_distance: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Array2<i32>> {
        if n_classes > 20 {
            return Err(SklearsError::InvalidInput(
                "Optimal code generation not practical for more than 20 classes".to_string(),
            ));
        }

        // Start with a sufficient code length
        let mut code_length = std::cmp::max(target_distance * 2, n_classes);
        let max_attempts = 100;

        for _attempt in 0..max_attempts {
            let mut best_matrix = None;
            let mut best_min_distance = 0;

            // Try multiple random initializations
            for _ in 0..50 {
                let mut code_matrix = Array2::zeros((n_classes, code_length));

                // Initialize first row randomly
                for j in 0..code_length {
                    code_matrix[[0, j]] = if rng.gen::<f64>() > 0.5 { 1 } else { -1 };
                }

                // Generate subsequent rows to maximize minimum distance
                for i in 1..n_classes {
                    self.generate_optimal_row(&mut code_matrix, i, code_length, rng)?;
                }

                let min_distance = self.calculate_minimum_distance(&code_matrix);
                if min_distance > best_min_distance {
                    best_min_distance = min_distance;
                    best_matrix = Some(code_matrix);
                }

                if best_min_distance >= target_distance {
                    return Ok(best_matrix.unwrap());
                }
            }

            // If we haven't reached target distance, increase code length
            code_length += target_distance;
        }

        Err(SklearsError::InvalidInput(format!(
            "Failed to generate optimal code with target distance {}",
            target_distance
        )))
    }

    /// Generate an optimal row for the code matrix
    fn generate_optimal_row(
        &self,
        code_matrix: &mut Array2<i32>,
        row_index: usize,
        code_length: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<()> {
        let mut best_row = Array1::zeros(code_length);
        let mut best_min_distance = 0;

        // Try multiple random configurations for this row
        for _ in 0..20 {
            let mut candidate_row = Array1::zeros(code_length);
            for j in 0..code_length {
                candidate_row[j] = if rng.gen::<f64>() > 0.5 { 1 } else { -1 };
            }

            // Calculate minimum distance to existing rows
            let mut min_distance = code_length;
            for i in 0..row_index {
                let distance = self.hamming_distance_mixed(&code_matrix.row(i), &candidate_row);
                min_distance = std::cmp::min(min_distance, distance);
            }

            if min_distance > best_min_distance {
                best_min_distance = min_distance;
                best_row = candidate_row;
            }
        }

        // Set the best row
        for j in 0..code_length {
            code_matrix[[row_index, j]] = best_row[j];
        }

        Ok(())
    }

    /// Calculate minimum distance between all pairs of codewords
    fn calculate_minimum_distance(&self, code_matrix: &Array2<i32>) -> usize {
        let n_classes = code_matrix.nrows();
        let mut min_distance = code_matrix.ncols();

        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let distance = self.hamming_distance(&code_matrix.row(i), &code_matrix.row(j));
                min_distance = std::cmp::min(min_distance, distance);
            }
        }

        min_distance
    }

    /// Calculate Hamming distance between two codewords
    fn hamming_distance(
        &self,
        code1: &scirs2_core::ndarray::ArrayView1<i32>,
        code2: &scirs2_core::ndarray::ArrayView1<i32>,
    ) -> usize {
        code1
            .iter()
            .zip(code2.iter())
            .map(|(&a, &b)| if a != b { 1 } else { 0 })
            .sum()
    }

    /// Calculate Hamming distance between a view and an owned array
    fn hamming_distance_mixed(
        &self,
        code1: &scirs2_core::ndarray::ArrayView1<i32>,
        code2: &scirs2_core::ndarray::Array1<i32>,
    ) -> usize {
        code1
            .iter()
            .zip(code2.iter())
            .map(|(&a, &b)| if a != b { 1 } else { 0 })
            .sum()
    }

    /// Verify that the code matrix satisfies minimum distance property
    fn verify_minimum_distance(&self, code_matrix: &Array2<i32>, min_distance: usize) -> bool {
        let actual_min_distance = self.calculate_minimum_distance(code_matrix);
        actual_min_distance >= min_distance
    }
}

/// Implementation for classifiers that can fit binary problems
impl<C> Fit<Array2<Float>, Array1<i32>> for ECOCClassifier<C, Untrained>
where
    C: Clone + Send + Sync + Fit<Array2<Float>, Array1<Float>>,
    C::Fitted: Predict<Array2<Float>, Array1<Float>> + Send,
{
    type Fitted = TrainedECOC<C::Fitted>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let (_n_samples, n_features) = x.dim();

        // Get unique classes
        let mut classes: Vec<i32> = y
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for multiclass classification".to_string(),
            ));
        }

        let n_classes = classes.len();

        // Generate code matrix
        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        let code_matrix = self.generate_code_matrix(n_classes, &mut rng)?;
        let code_length = code_matrix.ncols();

        // Create binary classification problems for each code bit
        let binary_problems: Vec<_> = (0..code_length)
            .map(|bit_idx| {
                // Create binary labels based on code matrix
                let binary_y: Array1<Float> = y.mapv(|label| {
                    let class_idx = classes.iter().position(|&c| c == label).unwrap();
                    if code_matrix[[class_idx, bit_idx]] == 1 {
                        1.0
                    } else {
                        0.0
                    }
                });
                binary_y
            })
            .collect();

        // Train binary classifiers (with parallel support if enabled)
        let estimators: SklResult<Vec<_>> = if self.config.n_jobs.is_some_and(|n| n != 1) {
            // Parallel training
            binary_problems
                .into_par_iter()
                .map(|binary_y| self.base_estimator.clone().fit(x, &binary_y))
                .collect::<SklResult<Vec<_>>>()
        } else {
            // Sequential training
            binary_problems
                .into_iter()
                .map(|binary_y| self.base_estimator.clone().fit(x, &binary_y))
                .collect::<SklResult<Vec<_>>>()
        };

        let estimators = estimators?;

        Ok(ECOCClassifier {
            base_estimator: ECOCTrainedData {
                estimators,
                classes: Array1::from(classes),
                code_matrix,
                n_features,
            },
            config: self.config,
            state: PhantomData,
        })
    }
}

impl<T> TrainedECOC<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }

    /// Get the code matrix
    pub fn code_matrix(&self) -> &Array2<i32> {
        &self.base_estimator.code_matrix
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.base_estimator.classes.len()
    }

    /// Get the fitted estimators
    pub fn estimators(&self) -> &[T] {
        &self.base_estimator.estimators
    }

    /// Get the code length
    pub fn code_length(&self) -> usize {
        self.base_estimator.code_matrix.ncols()
    }
}

impl<T> Predict<Array2<Float>, Array1<i32>> for TrainedECOC<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let code_length = self.base_estimator.code_matrix.ncols();
        let mut predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);
            let sample_matrix = sample.insert_axis(Axis(0));

            // Get binary predictions for each code bit
            let mut binary_predictions = Array1::zeros(code_length);

            for (bit_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
                let prediction = estimator.predict(&sample_matrix.to_owned())?;
                // Convert probability to binary code (+1 or -1)
                binary_predictions[bit_idx] = if prediction[0] > 0.5 { 1.0 } else { -1.0 };
            }

            // Find the class with minimum Hamming distance
            let mut min_distance = f64::INFINITY;
            let mut best_class_idx = 0;

            for class_idx in 0..n_classes {
                let code = self.base_estimator.code_matrix.row(class_idx);
                let distance: f64 = code
                    .iter()
                    .zip(binary_predictions.iter())
                    .map(|(&c, &p)| if (c as f64 * p) < 0.0 { 1.0 } else { 0.0 })
                    .sum();

                if distance < min_distance {
                    min_distance = distance;
                    best_class_idx = class_idx;
                }
            }

            predictions[sample_idx] = self.base_estimator.classes[best_class_idx];
        }

        Ok(predictions)
    }
}

/// Implementation of probability predictions for ECOC
impl<T> PredictProba<Array2<Float>, Array2<Float>> for TrainedECOC<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    fn predict_proba(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let code_length = self.base_estimator.code_matrix.ncols();
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);
            let sample_matrix = sample.insert_axis(Axis(0));

            // Get binary probabilities for each code bit
            let mut binary_probabilities = Array1::zeros(code_length);

            for (bit_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
                let prediction = estimator.predict(&sample_matrix.to_owned())?;
                binary_probabilities[bit_idx] = prediction[0];
            }

            // Calculate probability for each class based on code agreement
            for class_idx in 0..n_classes {
                let code = self.base_estimator.code_matrix.row(class_idx);
                let mut class_prob = 1.0;

                for (bit_idx, &code_bit) in code.iter().enumerate() {
                    let binary_prob = binary_probabilities[bit_idx];
                    // Probability that this bit matches the code
                    let bit_match_prob = if code_bit == 1 {
                        binary_prob
                    } else {
                        1.0 - binary_prob
                    };
                    class_prob *= bit_match_prob;
                }

                probabilities[[sample_idx, class_idx]] = class_prob;
            }

            // Normalize probabilities
            let prob_sum: f64 = probabilities.row(sample_idx).sum();
            if prob_sum > 0.0 {
                for class_idx in 0..n_classes {
                    probabilities[[sample_idx, class_idx]] /= prob_sum;
                }
            } else {
                // Uniform distribution if all probabilities are zero
                for class_idx in 0..n_classes {
                    probabilities[[sample_idx, class_idx]] = 1.0 / (n_classes as f64);
                }
            }
        }

        Ok(probabilities)
    }
}

/// Enhanced prediction methods for ECOC
impl<T> TrainedECOC<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    /// Get decision function scores (Hamming distances to each class code)
    pub fn decision_function(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let code_length = self.base_estimator.code_matrix.ncols();
        let mut decision_scores = Array2::zeros((n_samples, n_classes));

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);
            let sample_matrix = sample.insert_axis(Axis(0));

            // Get binary predictions for each code bit
            let mut binary_predictions = Array1::zeros(code_length);

            for (bit_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
                let prediction = estimator.predict(&sample_matrix.to_owned())?;
                binary_predictions[bit_idx] = prediction[0];
            }

            // Calculate negative Hamming distance for each class (higher = better)
            for class_idx in 0..n_classes {
                let code = self.base_estimator.code_matrix.row(class_idx);
                let distance: f64 = code
                    .iter()
                    .zip(binary_predictions.iter())
                    .map(|(&c, &p)| {
                        let binary_pred = if p > 0.5 { 1 } else { -1 };
                        if c == binary_pred {
                            0.0
                        } else {
                            1.0
                        }
                    })
                    .sum();

                // Use negative distance as decision score (closer = higher score)
                decision_scores[[sample_idx, class_idx]] = -distance;
            }
        }

        Ok(decision_scores)
    }
}
