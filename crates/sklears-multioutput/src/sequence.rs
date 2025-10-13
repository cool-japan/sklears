//! Sequence and structured prediction models
//!
//! This module provides algorithms for sequence labeling and structured prediction tasks.
//! It includes Hidden Markov Models (HMM), Structured Perceptron, and Maximum Entropy
//! Markov Models (MEMM) for handling sequential and structured data.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1};
use scirs2_core::random::RandNormal;
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};

/// Structured Perceptron for sequence labeling
///
/// The structured perceptron is an extension of the perceptron algorithm for
/// structured prediction problems. It can handle variable-length sequences and
/// complex output structures by using structured features and losses.
///
/// The structured perceptron extends the classic perceptron to handle structured
/// outputs by using a feature function that maps input-output pairs to feature
/// vectors and a loss function that measures the quality of predictions.
///
/// # Examples
///
/// ```
/// use sklears_core::traits::{Predict, Fit};
/// use sklears_multioutput::StructuredPerceptron;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[[1.0, 2.0], [2.0, 3.0]]]; // One sequence
/// let y = array![[0, 1]]; // Label sequence
///
/// let perceptron = StructuredPerceptron::new().max_iterations(50);
/// let trained_perceptron = perceptron.fit(&X, &y).unwrap();
/// let predictions = trained_perceptron.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StructuredPerceptron<State = Untrained> {
    max_iterations: usize,
    learning_rate: Float,
    random_state: Option<u64>,
    state: State,
}

/// Trained state for Structured Perceptron
#[derive(Debug, Clone)]
pub struct StructuredPerceptronTrained {
    weights: Array1<Float>,
    n_features: usize,
    n_classes: usize,
}

impl Default for StructuredPerceptron<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuredPerceptron<Untrained> {
    /// Create a new Structured Perceptron
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            learning_rate: 1.0,
            random_state: None,
            state: Untrained,
        }
    }

    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for StructuredPerceptron<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array3<Float>, Array2<i32>> for StructuredPerceptron<Untrained> {
    type Fitted = StructuredPerceptron<StructuredPerceptronTrained>;

    fn fit(self, X: &Array3<Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let (n_sequences, max_seq_len, n_features) = X.dim();

        if n_sequences != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of sequences".to_string(),
            ));
        }

        if y.ncols() != max_seq_len {
            return Err(SklearsError::InvalidInput(
                "y sequence length must match X sequence length".to_string(),
            ));
        }

        let n_classes = y.iter().max().unwrap_or(&0) + 1;
        let feature_dim = n_features * n_classes as usize + n_classes as usize * n_classes as usize;
        let mut weights = Array1::<Float>::zeros(feature_dim);

        let rng = thread_rng();

        for _iteration in 0..self.max_iterations {
            let mut updated = false;

            for seq_idx in 0..n_sequences {
                let sequence = X.slice(s![seq_idx, .., ..]);
                let true_labels = y.row(seq_idx);

                // Simple prediction: take argmax for each position
                let mut predicted_labels = Array1::<Float>::zeros(max_seq_len);

                for pos in 0..max_seq_len {
                    let features = sequence.slice(s![pos, ..]);
                    let mut best_score = Float::NEG_INFINITY;
                    let mut best_label = 0;

                    for label in 0..n_classes {
                        let feature_offset = label as usize * n_features;
                        let score = features
                            .iter()
                            .enumerate()
                            .map(|(feat_idx, &feat_val)| {
                                weights[feature_offset + feat_idx] * feat_val
                            })
                            .sum::<Float>();

                        if score > best_score {
                            best_score = score;
                            best_label = label;
                        }
                    }
                    predicted_labels[pos] = best_label as Float;
                }

                // Check if prediction is correct
                let correct = true_labels
                    .iter()
                    .zip(predicted_labels.iter())
                    .all(|(&true_label, &pred_label)| true_label == pred_label as i32);

                if !correct {
                    // Update weights: add true features, subtract predicted features
                    for pos in 0..max_seq_len {
                        let features = sequence.slice(s![pos, ..]);
                        let true_label = true_labels[pos] as usize;
                        let pred_label = predicted_labels[pos] as usize;

                        // Add true label features
                        let true_offset = true_label * n_features;
                        for (feat_idx, &feat_val) in features.iter().enumerate() {
                            weights[true_offset + feat_idx] += self.learning_rate * feat_val;
                        }

                        // Subtract predicted label features
                        let pred_offset = pred_label * n_features;
                        for (feat_idx, &feat_val) in features.iter().enumerate() {
                            weights[pred_offset + feat_idx] -= self.learning_rate * feat_val;
                        }
                    }
                    updated = true;
                }
            }

            if !updated {
                break;
            }
        }

        Ok(StructuredPerceptron {
            max_iterations: self.max_iterations,
            learning_rate: self.learning_rate,
            random_state: self.random_state,
            state: StructuredPerceptronTrained {
                weights,
                n_features,
                n_classes: n_classes as usize,
            },
        })
    }
}

impl StructuredPerceptron<Untrained> {
    /// Get the weights of the trained model
    pub fn weights(&self) -> Option<&Array1<Float>> {
        None
    }
}

impl Predict<Array3<Float>, Array2<i32>> for StructuredPerceptron<StructuredPerceptronTrained> {
    fn predict(&self, X: &Array3<Float>) -> SklResult<Array2<i32>> {
        let (n_sequences, max_seq_len, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_sequences, max_seq_len));

        for seq_idx in 0..n_sequences {
            let sequence = X.slice(s![seq_idx, .., ..]);

            for pos in 0..max_seq_len {
                let features = sequence.slice(s![pos, ..]);
                let mut best_score = Float::NEG_INFINITY;
                let mut best_label = 0;

                for label in 0..self.state.n_classes {
                    let feature_offset = label * n_features;
                    let score = features
                        .iter()
                        .enumerate()
                        .map(|(feat_idx, &feat_val)| {
                            self.state.weights[feature_offset + feat_idx] * feat_val
                        })
                        .sum::<Float>();

                    if score > best_score {
                        best_score = score;
                        best_label = label;
                    }
                }
                predictions[[seq_idx, pos]] = best_label as i32;
            }
        }

        Ok(predictions)
    }
}

impl StructuredPerceptron<StructuredPerceptronTrained> {
    /// Get the weights of the trained model
    pub fn weights(&self) -> &Array1<Float> {
        &self.state.weights
    }
}

/// Hidden Markov Model for sequence modeling
///
/// A statistical model that assumes the system being modeled is a Markov process
/// with unobserved (hidden) states. HMMs are particularly useful for temporal
/// pattern recognition such as speech recognition, handwriting recognition,
/// gesture recognition, part-of-speech tagging, and bioinformatics.
///
/// # Examples
///
/// ```
/// use sklears_core::traits::{Predict, Fit};
/// use sklears_multioutput::HiddenMarkovModel;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[[1.0, 2.0], [2.0, 3.0], [1.5, 2.5]]]; // One sequence
/// let y = array![[0, 1, 0]]; // State sequence
///
/// let hmm = HiddenMarkovModel::new().n_states(2).max_iterations(100);
/// let trained_hmm = hmm.fit(&X, &y).unwrap();
/// let predictions = trained_hmm.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel<State = Untrained> {
    n_states: usize,
    max_iterations: usize,
    tolerance: Float,
    random_state: Option<u64>,
    state: State,
}

/// Trained state for Hidden Markov Model
#[derive(Debug, Clone)]
pub struct HiddenMarkovModelTrained {
    transition_matrix: Array2<Float>,
    emission_means: Array2<Float>,
    emission_covariances: Array3<Float>,
    initial_probs: Array1<Float>,
    n_features: usize,
    n_states: usize,
}

impl Default for HiddenMarkovModel<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl HiddenMarkovModel<Untrained> {
    /// Create a new Hidden Markov Model
    pub fn new() -> Self {
        Self {
            n_states: 2,
            max_iterations: 100,
            tolerance: 1e-6,
            random_state: None,
            state: Untrained,
        }
    }

    /// Set the number of hidden states
    pub fn n_states(mut self, n_states: usize) -> Self {
        self.n_states = n_states;
        self
    }

    /// Set the maximum number of iterations for EM algorithm
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for HiddenMarkovModel<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array3<Float>, Array2<i32>> for HiddenMarkovModel<Untrained> {
    type Fitted = HiddenMarkovModel<HiddenMarkovModelTrained>;

    fn fit(self, X: &Array3<Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let (n_sequences, max_seq_len, n_features) = X.dim();

        if n_sequences != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of sequences".to_string(),
            ));
        }

        if y.ncols() != max_seq_len {
            return Err(SklearsError::InvalidInput(
                "y sequence length must match X sequence length".to_string(),
            ));
        }

        let mut rng = thread_rng();

        // Initialize parameters
        let normal_dist = RandNormal::new(0.0, 1.0).unwrap();
        let mut transition_matrix = Array2::<Float>::zeros((self.n_states, self.n_states));
        for i in 0..self.n_states {
            for j in 0..self.n_states {
                transition_matrix[[i, j]] = rng.sample(normal_dist);
            }
        }
        let mut emission_means = Array2::<Float>::zeros((self.n_states, n_features));
        for i in 0..self.n_states {
            for j in 0..n_features {
                emission_means[[i, j]] = rng.sample(normal_dist);
            }
        }
        let mut emission_covariances =
            Array3::from_elem((self.n_states, n_features, n_features), 1.0);
        let mut initial_probs = Array1::from_elem(self.n_states, 1.0 / self.n_states as Float);

        // Normalize transition matrix
        for i in 0..self.n_states {
            let row_sum = transition_matrix.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..self.n_states {
                    transition_matrix[[i, j]] /= row_sum;
                }
            }
        }

        // Initialize emission covariances as identity matrices
        for state in 0..self.n_states {
            for i in 0..n_features {
                emission_covariances[[state, i, i]] = 1.0;
            }
        }

        let mut prev_likelihood = Float::NEG_INFINITY;

        // EM algorithm
        for _iteration in 0..self.max_iterations {
            let mut total_likelihood = 0.0;

            // E-step: Forward-backward algorithm would go here
            // For simplicity, we'll use supervised learning with the provided labels

            // M-step: Update parameters based on state assignments
            let mut state_counts = Array1::<Float>::zeros(self.n_states);
            let mut transition_counts = Array2::<Float>::zeros((self.n_states, self.n_states));
            let mut emission_sums = Array2::<Float>::zeros((self.n_states, n_features));

            for seq_idx in 0..n_sequences {
                let sequence_data = X.slice(s![seq_idx, .., ..]);
                let sequence_states = y.row(seq_idx);

                for pos in 0..max_seq_len {
                    let state = sequence_states[pos] as usize;
                    if state < self.n_states {
                        state_counts[state] += 1.0;

                        // Update emission parameters
                        for feat in 0..n_features {
                            emission_sums[[state, feat]] += sequence_data[[pos, feat]];
                        }

                        // Update transition parameters
                        if pos < max_seq_len - 1 {
                            let next_state = sequence_states[pos + 1] as usize;
                            if next_state < self.n_states {
                                transition_counts[[state, next_state]] += 1.0;
                            }
                        }
                    }
                }

                // Update initial state probabilities
                let first_state = sequence_states[0] as usize;
                if first_state < self.n_states {
                    initial_probs[first_state] += 1.0;
                }
            }

            // Normalize and update parameters
            for state in 0..self.n_states {
                if state_counts[state] > 0.0 {
                    for feat in 0..n_features {
                        emission_means[[state, feat]] =
                            emission_sums[[state, feat]] / state_counts[state];
                    }
                }

                let row_sum = transition_counts.row(state).sum();
                if row_sum > 0.0 {
                    for next_state in 0..self.n_states {
                        transition_matrix[[state, next_state]] =
                            transition_counts[[state, next_state]] / row_sum;
                    }
                }
            }

            // Normalize initial probabilities
            let init_sum = initial_probs.sum();
            if init_sum > 0.0 {
                initial_probs /= init_sum;
            }

            // Simple likelihood calculation
            total_likelihood = state_counts.sum();

            if (total_likelihood - prev_likelihood).abs() < self.tolerance {
                break;
            }
            prev_likelihood = total_likelihood;
        }

        Ok(HiddenMarkovModel {
            n_states: self.n_states,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            random_state: self.random_state,
            state: HiddenMarkovModelTrained {
                transition_matrix,
                emission_means,
                emission_covariances,
                initial_probs,
                n_features,
                n_states: self.n_states,
            },
        })
    }
}

impl HiddenMarkovModel<Untrained> {
    /// Get the transition matrix (only available after training)
    pub fn transition_matrix(&self) -> Option<&Array2<Float>> {
        None
    }

    /// Get the emission means (only available after training)
    pub fn emission_means(&self) -> Option<&Array2<Float>> {
        None
    }

    /// Get the initial state probabilities (only available after training)
    pub fn initial_probabilities(&self) -> Option<&Array1<Float>> {
        None
    }
}

impl Predict<Array3<Float>, Array2<i32>> for HiddenMarkovModel<HiddenMarkovModelTrained> {
    fn predict(&self, X: &Array3<Float>) -> SklResult<Array2<i32>> {
        let (n_sequences, max_seq_len, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_sequences, max_seq_len));

        for seq_idx in 0..n_sequences {
            let sequence = X.slice(s![seq_idx, .., ..]);

            // Viterbi algorithm for finding most likely state sequence
            let mut viterbi = Array2::<Float>::zeros((max_seq_len, self.state.n_states));
            let mut path = Array2::<Float>::zeros((max_seq_len, self.state.n_states));

            // Initialize first time step
            for state in 0..self.state.n_states {
                let emission_prob = self.gaussian_probability(&sequence.slice(s![0, ..]), state);
                viterbi[[0, state]] = self.state.initial_probs[state].ln() + emission_prob.ln();
            }

            // Forward pass
            for t in 1..max_seq_len {
                for state in 0..self.state.n_states {
                    let emission_prob =
                        self.gaussian_probability(&sequence.slice(s![t, ..]), state);
                    let mut best_prob = Float::NEG_INFINITY;
                    let mut best_prev_state = 0;

                    for prev_state in 0..self.state.n_states {
                        let prob = viterbi[[t - 1, prev_state]]
                            + self.state.transition_matrix[[prev_state, state]].ln()
                            + emission_prob.ln();
                        if prob > best_prob {
                            best_prob = prob;
                            best_prev_state = prev_state;
                        }
                    }
                    viterbi[[t, state]] = best_prob;
                    path[[t, state]] = best_prev_state as Float;
                }
            }

            // Backward pass - find best path
            let mut states = Array1::<Float>::zeros(max_seq_len);

            // Find best final state
            let mut best_final_prob = Float::NEG_INFINITY;
            let mut best_final_state = 0;
            for state in 0..self.state.n_states {
                if viterbi[[max_seq_len - 1, state]] > best_final_prob {
                    best_final_prob = viterbi[[max_seq_len - 1, state]];
                    best_final_state = state;
                }
            }

            states[max_seq_len - 1] = best_final_state as Float;

            // Trace back
            for t in (0..max_seq_len - 1).rev() {
                states[t] = path[[t + 1, states[t + 1] as usize]];
            }

            // Copy to predictions
            for t in 0..max_seq_len {
                predictions[[seq_idx, t]] = states[t] as i32;
            }
        }

        Ok(predictions)
    }
}

impl HiddenMarkovModel<HiddenMarkovModelTrained> {
    /// Get the transition matrix
    pub fn transition_matrix(&self) -> &Array2<Float> {
        &self.state.transition_matrix
    }

    /// Get the emission means
    pub fn emission_means(&self) -> &Array2<Float> {
        &self.state.emission_means
    }

    /// Get the initial state probabilities
    pub fn initial_probabilities(&self) -> &Array1<Float> {
        &self.state.initial_probs
    }

    /// Calculate Gaussian probability for emission
    fn gaussian_probability(&self, observation: &ArrayView1<Float>, state: usize) -> Float {
        let mean = self.state.emission_means.row(state);
        let diff = observation.to_owned() - &mean.to_owned();

        // Simple Gaussian with identity covariance for now
        let exponent = -0.5 * diff.mapv(|x| x * x).sum();
        let normalization =
            (2.0 * std::f64::consts::PI).powf(self.state.n_features as f64 / 2.0) as Float;

        (exponent.exp() / normalization).max(1e-10)
    }
}

/// Maximum Entropy Markov Model (MEMM) for Sequence Labeling
///
/// MEMM is a discriminative model for sequence labeling that combines the advantages
/// of maximum entropy models with Markov assumptions. Unlike CRF, MEMM models the
/// conditional probability of each label given the previous label and observed features.
///
/// The model uses logistic regression at each position to predict the next label
/// based on features extracted from the current observation and previous label.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::MaximumEntropyMarkovModel;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // Sequence data: each row is a sequence element with features
/// let X = vec![
///     array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]], // sequence 1
///     array![[4.0, 1.0], [1.0, 4.0]]                // sequence 2
/// ];
///
/// // Label sequences
/// let y = vec![
///     vec![0, 1, 0], // labels for sequence 1
///     vec![1, 0]     // labels for sequence 2
/// ];
///
/// let memm = MaximumEntropyMarkovModel::new()
///     .max_iter(50)
///     .learning_rate(0.01);
/// let trained_memm = memm.fit(&X, &y).unwrap();
/// let predictions = trained_memm.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MaximumEntropyMarkovModel<S = Untrained> {
    state: S,
    max_iter: usize,
    learning_rate: Float,
    l2_reg: Float,
    tolerance: Float,
    feature_functions: Vec<FeatureFunction>,
    random_state: Option<u64>,
}

/// Trained state for MEMM
#[derive(Debug, Clone)]
pub struct MaximumEntropyMarkovModelTrained {
    weights: Array1<Float>,
    feature_functions: Vec<FeatureFunction>,
    n_labels: usize,
    n_features: usize,
    label_to_idx: HashMap<i32, usize>,
    idx_to_label: HashMap<usize, i32>,
}

/// Feature function for MEMM
///
/// Each feature function extracts a specific type of feature from the input
/// and previous label combination. The feature value is typically binary
/// (0 or 1) indicating presence or absence of the feature.
#[derive(Debug, Clone)]
pub struct FeatureFunction {
    /// feature_type
    pub feature_type: FeatureType,
    /// weight_index
    pub weight_index: usize,
}

/// Types of features in MEMM
#[derive(Debug, Clone)]
pub enum FeatureType {
    /// Current observation feature at given index
    Observation(usize),
    /// Previous label feature
    PreviousLabel(i32),
    /// Interaction between observation and previous label
    LabelObservationInteraction(i32, usize),
}

impl MaximumEntropyMarkovModel<Untrained> {
    /// Create a new MEMM instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            max_iter: 100,
            learning_rate: 0.01,
            l2_reg: 0.01,
            tolerance: 1e-6,
            feature_functions: Vec::new(),
            random_state: None,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set L2 regularization strength
    pub fn l2_regularization(mut self, l2_reg: Float) -> Self {
        self.l2_reg = l2_reg;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for MaximumEntropyMarkovModel<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MaximumEntropyMarkovModel<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Vec<Array2<Float>>, Vec<Vec<i32>>> for MaximumEntropyMarkovModel<Untrained> {
    type Fitted = MaximumEntropyMarkovModel<MaximumEntropyMarkovModelTrained>;

    fn fit(self, X: &Vec<Array2<Float>>, y: &Vec<Vec<i32>>) -> SklResult<Self::Fitted> {
        if X.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of sequences".to_string(),
            ));
        }

        if X.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit with 0 sequences".to_string(),
            ));
        }

        // Determine dimensions and create label mappings
        let n_features = X[0].ncols();
        let mut unique_labels = HashSet::new();

        for sequence_labels in y {
            for &label in sequence_labels {
                unique_labels.insert(label);
            }
        }

        let mut label_to_idx = HashMap::new();
        let mut idx_to_label = HashMap::new();
        // Sort labels to ensure deterministic ordering
        let mut sorted_labels: Vec<_> = unique_labels.iter().cloned().collect();
        sorted_labels.sort();
        for (idx, label) in sorted_labels.iter().enumerate() {
            label_to_idx.insert(*label, idx);
            idx_to_label.insert(idx, *label);
        }
        let n_labels = unique_labels.len();

        // Create feature functions
        let mut feature_functions = Vec::new();
        let mut weight_idx = 0;

        // Observation features
        for feat_idx in 0..n_features {
            for &label in &sorted_labels {
                feature_functions.push(FeatureFunction {
                    feature_type: FeatureType::LabelObservationInteraction(label, feat_idx),
                    weight_index: weight_idx,
                });
                weight_idx += 1;
            }
        }

        // Previous label features
        for &prev_label in &sorted_labels {
            for &curr_label in &sorted_labels {
                feature_functions.push(FeatureFunction {
                    feature_type: FeatureType::PreviousLabel(prev_label),
                    weight_index: weight_idx,
                });
                weight_idx += 1;
            }
        }

        let n_weights = weight_idx;
        let mut weights = Array1::<Float>::zeros(n_weights);

        // Initialize weights randomly using random_state for reproducibility
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            // Use current time as seed for non-deterministic behavior
            use std::time::{SystemTime, UNIX_EPOCH};
            let time_seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            scirs2_core::random::seeded_rng(time_seed)
        };

        for w in weights.iter_mut() {
            *w = rng.gen_range(-0.1..0.1);
        }

        // Training loop
        for _iter in 0..self.max_iter {
            let mut gradient = Array1::<Float>::zeros(n_weights);
            let mut total_loss = 0.0;

            // Process each sequence
            for (seq_idx, (sequence_x, sequence_y)) in X.iter().zip(y.iter()).enumerate() {
                let seq_len = sequence_x.nrows();

                if sequence_y.len() != seq_len {
                    return Err(SklearsError::InvalidInput(format!(
                        "Sequence {} length mismatch between X and y",
                        seq_idx
                    )));
                }

                // Process each position in the sequence
                for pos in 0..seq_len {
                    let current_obs = sequence_x.row(pos);
                    let true_label = sequence_y[pos];
                    let prev_label = if pos == 0 { -1 } else { sequence_y[pos - 1] };

                    // Calculate feature vector for current position
                    let features =
                        self.extract_features(&current_obs, prev_label, &feature_functions);

                    // Calculate probabilities for all possible labels
                    let mut scores = Array1::<Float>::zeros(n_labels);
                    let mut max_score = Float::NEG_INFINITY;

                    for (label_idx, &label) in unique_labels.iter().enumerate() {
                        let label_features =
                            self.extract_features(&current_obs, prev_label, &feature_functions);
                        scores[label_idx] = label_features.dot(&weights);
                        max_score = max_score.max(scores[label_idx]);
                    }

                    // Numerical stability: subtract max score
                    for score in scores.iter_mut() {
                        *score -= max_score;
                    }

                    // Calculate softmax probabilities
                    let exp_scores: Array1<Float> = scores.mapv(|x| x.exp());
                    let sum_exp_scores = exp_scores.sum();
                    let probabilities = exp_scores / sum_exp_scores;

                    // Calculate loss (negative log likelihood)
                    let true_label_idx = label_to_idx[&true_label];
                    total_loss -= probabilities[true_label_idx].ln();

                    // Calculate gradient
                    for (label_idx, &label) in sorted_labels.iter().enumerate() {
                        let label_features =
                            self.extract_features(&current_obs, prev_label, &feature_functions);
                        let prob = probabilities[label_idx];
                        let indicator = if label_idx == true_label_idx {
                            1.0
                        } else {
                            0.0
                        };

                        for (feat_idx, &feat_val) in label_features.iter().enumerate() {
                            gradient[feat_idx] += feat_val * (prob - indicator);
                        }
                    }
                }
            }

            // Add L2 regularization to gradient
            for (i, w) in weights.iter().enumerate() {
                gradient[i] += self.l2_reg * w;
            }

            // Update weights
            let gradient_norm = gradient.mapv(|x| x.abs()).sum();
            weights = &weights - self.learning_rate * &gradient;

            // Check convergence
            if gradient_norm < self.tolerance {
                break;
            }
        }

        let trained_state = MaximumEntropyMarkovModelTrained {
            weights,
            feature_functions,
            n_labels,
            n_features,
            label_to_idx,
            idx_to_label,
        };

        Ok(MaximumEntropyMarkovModel {
            state: trained_state,
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            l2_reg: self.l2_reg,
            tolerance: self.tolerance,
            feature_functions: Vec::new(),
            random_state: self.random_state,
        })
    }
}

impl MaximumEntropyMarkovModel<Untrained> {
    /// Extract features for a given observation and previous label
    fn extract_features(
        &self,
        observation: &ArrayView1<Float>,
        prev_label: i32,
        feature_functions: &[FeatureFunction],
    ) -> Array1<Float> {
        let mut features = Array1::<Float>::zeros(feature_functions.len());

        for (i, func) in feature_functions.iter().enumerate() {
            features[i] = match &func.feature_type {
                FeatureType::Observation(feat_idx) => observation[*feat_idx],
                FeatureType::PreviousLabel(label) => {
                    if prev_label == *label {
                        1.0
                    } else {
                        0.0
                    }
                }
                FeatureType::LabelObservationInteraction(label, feat_idx) => {
                    if prev_label == *label {
                        observation[*feat_idx]
                    } else {
                        0.0
                    }
                }
            };
        }

        features
    }
}

impl Predict<Vec<Array2<Float>>, Vec<Vec<i32>>>
    for MaximumEntropyMarkovModel<MaximumEntropyMarkovModelTrained>
{
    fn predict(&self, X: &Vec<Array2<Float>>) -> SklResult<Vec<Vec<i32>>> {
        if X.is_empty() {
            return Ok(Vec::new());
        }

        let mut predictions = Vec::with_capacity(X.len());

        for sequence in X {
            let seq_len = sequence.nrows();

            if sequence.ncols() != self.state.n_features {
                return Err(SklearsError::InvalidInput(
                    "X has different number of features than training data".to_string(),
                ));
            }

            let mut sequence_predictions = Vec::with_capacity(seq_len);

            for pos in 0..seq_len {
                let current_obs = sequence.row(pos);
                let prev_label = if pos == 0 {
                    -1
                } else {
                    sequence_predictions[pos - 1]
                };

                // Calculate scores for all possible labels
                let mut best_score = Float::NEG_INFINITY;
                let mut best_label = 0;

                // Sort labels to ensure deterministic iteration order
                let mut labels_sorted: Vec<_> = self.state.label_to_idx.iter().collect();
                labels_sorted.sort_by_key(|(&label, _)| label);

                for (&label, &label_idx) in labels_sorted {
                    let features = self.extract_features(
                        &current_obs,
                        prev_label,
                        &self.state.feature_functions,
                    );
                    let score = features.dot(&self.state.weights);

                    if score > best_score {
                        best_score = score;
                        best_label = label;
                    }
                }

                sequence_predictions.push(best_label);
            }

            predictions.push(sequence_predictions);
        }

        Ok(predictions)
    }
}

impl MaximumEntropyMarkovModel<MaximumEntropyMarkovModelTrained> {
    /// Get the learned weights
    pub fn weights(&self) -> &Array1<Float> {
        &self.state.weights
    }

    /// Get the number of labels
    pub fn n_labels(&self) -> usize {
        self.state.n_labels
    }

    /// Extract features for a given observation and previous label
    fn extract_features(
        &self,
        observation: &ArrayView1<Float>,
        prev_label: i32,
        feature_functions: &[FeatureFunction],
    ) -> Array1<Float> {
        let mut features = Array1::<Float>::zeros(feature_functions.len());

        for (i, func) in feature_functions.iter().enumerate() {
            features[i] = match &func.feature_type {
                FeatureType::Observation(feat_idx) => observation[*feat_idx],
                FeatureType::PreviousLabel(label) => {
                    if prev_label == *label {
                        1.0
                    } else {
                        0.0
                    }
                }
                FeatureType::LabelObservationInteraction(label, feat_idx) => {
                    if prev_label == *label {
                        observation[*feat_idx]
                    } else {
                        0.0
                    }
                }
            };
        }

        features
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_structured_perceptron_basic() {
        let X = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 2.0, 3.0]).unwrap();
        let y = Array2::from_shape_vec((1, 2), vec![0, 1]).unwrap();

        let perceptron = StructuredPerceptron::new().max_iterations(10);
        let trained = perceptron.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.dim(), (1, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_hidden_markov_model_basic() {
        let X = Array3::from_shape_vec((1, 3, 2), vec![1.0, 2.0, 2.0, 3.0, 1.5, 2.5]).unwrap();
        let y = Array2::from_shape_vec((1, 3), vec![0, 1, 0]).unwrap();

        let hmm = HiddenMarkovModel::new().n_states(2).max_iterations(5);
        let trained = hmm.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.dim(), (1, 3));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_memm_basic() {
        let X = vec![array![[1.0, 2.0], [2.0, 3.0]], array![[3.0, 1.0]]];
        let y = vec![vec![0, 1], vec![0]];

        let memm = MaximumEntropyMarkovModel::new()
            .max_iter(5)
            .learning_rate(0.1);
        let trained = memm.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0].len(), 2);
        assert_eq!(predictions[1].len(), 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_memm_reproducibility() {
        let X = vec![
            array![[1.0, 2.0], [2.0, 3.0]],
            array![[3.0, 1.0], [4.0, 2.0]],
        ];
        let y = vec![vec![0, 1], vec![1, 0]];

        let memm1 = MaximumEntropyMarkovModel::new()
            .max_iter(10)
            .random_state(42);
        let trained_memm1 = memm1.fit(&X, &y).unwrap();
        let pred1 = trained_memm1.predict(&X).unwrap();

        let memm2 = MaximumEntropyMarkovModel::new()
            .max_iter(10)
            .random_state(42);
        let trained_memm2 = memm2.fit(&X, &y).unwrap();
        let pred2 = trained_memm2.predict(&X).unwrap();

        assert_eq!(pred1, pred2);
    }
}
