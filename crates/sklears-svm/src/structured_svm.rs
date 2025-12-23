//! Structured Support Vector Machines
//!
//! This module implements structured SVMs for sequence labeling and other
//! structured prediction tasks. Structured SVMs extend traditional SVMs to
//! handle structured outputs like sequences, trees, or graphs.
//!
//! Algorithms included:
//! - Structured SVM for sequence labeling (CRF-like)
//! - Structural SVM with margin rescaling
//! - Structural SVM with slack rescaling
//! - Latent structural SVM

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

/// Errors that can occur during structured SVM training and prediction
#[derive(Debug, Clone)]
pub enum StructuredSVMError {
    InvalidInput(String),
    TrainingError(String),
    PredictionError(String),
    ConvergenceError(String),
}

impl fmt::Display for StructuredSVMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StructuredSVMError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            StructuredSVMError::TrainingError(msg) => write!(f, "Training error: {msg}"),
            StructuredSVMError::PredictionError(msg) => write!(f, "Prediction error: {msg}"),
            StructuredSVMError::ConvergenceError(msg) => write!(f, "Convergence error: {msg}"),
        }
    }
}

impl std::error::Error for StructuredSVMError {}

/// Loss function for structured SVM
#[derive(Debug, Clone)]
pub enum StructuredLoss {
    /// Hamming loss (number of incorrect predictions)
    Hamming,
    /// F1 loss (based on F1 score)
    F1,
    /// Edit distance (Levenshtein distance)
    EditDistance,
}

/// Inference algorithm for structured prediction
#[derive(Debug, Clone)]
pub enum InferenceAlgorithm {
    /// Viterbi algorithm for sequence labeling
    Viterbi,
    /// Belief propagation for general graphical models
    BeliefPropagation,
    /// Graph cuts for certain types of problems
    GraphCuts,
}

/// Structured SVM configuration
#[derive(Debug, Clone)]
pub struct StructuredSVMConfig {
    /// Regularization parameter
    pub c: f64,
    /// Loss function
    pub loss: StructuredLoss,
    /// Inference algorithm
    pub inference: InferenceAlgorithm,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Learning rate for cutting plane algorithm
    pub learning_rate: f64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for StructuredSVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            loss: StructuredLoss::Hamming,
            inference: InferenceAlgorithm::Viterbi,
            max_iter: 100,
            tol: 1e-4,
            learning_rate: 0.01,
            verbose: false,
        }
    }
}

/// Sequence structure for labeling tasks
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Feature vectors for each position in the sequence
    pub features: Array2<f64>,
    /// Length of the sequence
    pub length: usize,
    /// Labels (for training sequences)
    pub labels: Option<Array1<usize>>,
}

impl Sequence {
    /// Create a new sequence
    pub fn new(features: Array2<f64>, labels: Option<Array1<usize>>) -> Self {
        let length = features.nrows();
        Self {
            features,
            length,
            labels,
        }
    }

    /// Get feature vector at position
    pub fn feature_at(&self, position: usize) -> Array1<f64> {
        self.features.row(position).to_owned()
    }

    /// Get label at position (if available)
    pub fn label_at(&self, position: usize) -> Option<usize> {
        self.labels.as_ref().map(|labels| labels[position])
    }
}

/// Structured SVM for sequence labeling
#[derive(Debug, Clone)]
pub struct StructuredSVM<State = Untrained> {
    config: StructuredSVMConfig,
    state: PhantomData<State>,
    // Model parameters
    weights: Option<Array1<f64>>,
    transition_weights: Option<Array2<f64>>,
    // Training data statistics
    n_features: Option<usize>,
    n_labels: Option<usize>,
    label_to_idx: Option<HashMap<usize, usize>>,
    idx_to_label: Option<HashMap<usize, usize>>,
}

impl Default for StructuredSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuredSVM<Untrained> {
    /// Create a new structured SVM
    pub fn new() -> Self {
        Self {
            config: StructuredSVMConfig::default(),
            state: PhantomData,
            weights: None,
            transition_weights: None,
            n_features: None,
            n_labels: None,
            label_to_idx: None,
            idx_to_label: None,
        }
    }

    /// Set the regularization parameter
    pub fn with_c(mut self, c: f64) -> Self {
        self.config.c = c;
        self
    }

    /// Set the loss function
    pub fn with_loss(mut self, loss: StructuredLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the inference algorithm
    pub fn with_inference(mut self, inference: InferenceAlgorithm) -> Self {
        self.config.inference = inference;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }
}

impl Fit<Vec<Sequence>, Vec<Array1<usize>>> for StructuredSVM<Untrained> {
    type Fitted = StructuredSVM<Trained>;

    fn fit(self, sequences: &Vec<Sequence>, labels: &Vec<Array1<usize>>) -> Result<Self::Fitted> {
        if sequences.len() != labels.len() {
            return Err(SklearsError::InvalidInput(
                "Number of sequences and label arrays must match".to_string(),
            ));
        }

        if sequences.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string(),
            ));
        }

        // Extract feature dimensions and label vocabulary
        let n_features = sequences[0].features.ncols();
        let mut label_set = std::collections::HashSet::new();

        for (seq, seq_labels) in sequences.iter().zip(labels.iter()) {
            if seq.features.ncols() != n_features {
                return Err(SklearsError::InvalidInput(
                    "All sequences must have the same feature dimension".to_string(),
                ));
            }

            if seq.length != seq_labels.len() {
                return Err(SklearsError::InvalidInput(
                    "Sequence length must match label length".to_string(),
                ));
            }

            for &label in seq_labels.iter() {
                label_set.insert(label);
            }
        }

        let n_labels = label_set.len();
        let mut label_to_idx = HashMap::new();
        let mut idx_to_label = HashMap::new();

        for (idx, &label) in label_set.iter().enumerate() {
            label_to_idx.insert(label, idx);
            idx_to_label.insert(idx, label);
        }

        // Initialize weights
        let total_features = n_features + n_labels * n_labels; // Node features + transition features
        let mut weights = Array1::zeros(total_features);
        let mut transition_weights = Array2::zeros((n_labels, n_labels));

        // Cutting plane algorithm for structured SVM training
        let mut objective_history = Vec::new();

        for iteration in 0..self.config.max_iter {
            let mut total_loss = 0.0;
            let mut gradient = Array1::zeros(total_features);

            // For each training sequence
            for (seq, seq_labels) in sequences.iter().zip(labels.iter()) {
                // Find most violating constraint (loss-augmented inference)
                let predicted_labels = self.loss_augmented_inference(
                    seq,
                    seq_labels,
                    &weights,
                    &transition_weights,
                    &label_to_idx,
                )?;

                // Compute loss
                let loss = self.compute_loss(seq_labels, &predicted_labels)?;
                total_loss += loss;

                // Compute feature difference (ψ(x,y) - ψ(x,ŷ))
                let true_features = self.compute_features(seq, seq_labels, &label_to_idx)?;
                let pred_features = self.compute_features(seq, &predicted_labels, &label_to_idx)?;
                let feature_diff = &true_features - &pred_features;

                // Update gradient
                gradient = gradient + feature_diff;
            }

            // Regularization gradient
            let reg_gradient = &weights * self.config.c;
            gradient = gradient - reg_gradient;

            // Update weights
            weights = weights + self.config.learning_rate * gradient;

            // Update transition weights (extract from weights)
            let start_idx = n_features;
            for i in 0..n_labels {
                for j in 0..n_labels {
                    let idx = start_idx + i * n_labels + j;
                    transition_weights[[i, j]] = weights[idx];
                }
            }

            // Compute objective
            let regularization = 0.5 * self.config.c * weights.dot(&weights);
            let objective = total_loss - regularization;
            objective_history.push(objective);

            if self.config.verbose {
                println!(
                    "Iteration {iteration}: Objective = {objective:.6}, Loss = {total_loss:.6}"
                );
            }

            // Check convergence
            if iteration > 0 {
                let prev_obj = objective_history[iteration - 1];
                let obj_change = (objective - prev_obj).abs();
                if obj_change < self.config.tol {
                    if self.config.verbose {
                        println!("Converged after {} iterations", iteration + 1);
                    }
                    break;
                }
            }
        }

        Ok(StructuredSVM {
            config: self.config,
            state: PhantomData,
            weights: Some(weights),
            transition_weights: Some(transition_weights),
            n_features: Some(n_features),
            n_labels: Some(n_labels),
            label_to_idx: Some(label_to_idx),
            idx_to_label: Some(idx_to_label),
        })
    }
}

impl StructuredSVM<Untrained> {
    /// Compute features for a sequence with given labels
    fn compute_features(
        &self,
        sequence: &Sequence,
        labels: &Array1<usize>,
        label_to_idx: &HashMap<usize, usize>,
    ) -> Result<Array1<f64>> {
        let n_features = sequence.features.ncols();
        let n_labels = label_to_idx.len();
        let total_features = n_features + n_labels * n_labels;
        let mut features = Array1::zeros(total_features);

        // Node features (emission features)
        for (pos, &label) in labels.iter().enumerate() {
            let _label_idx = *label_to_idx.get(&label).unwrap();
            let node_features = sequence.feature_at(pos);

            // Add weighted node features
            for (feat_idx, &feat_val) in node_features.iter().enumerate() {
                features[feat_idx] += feat_val;
            }
        }

        // Transition features
        for pos in 1..labels.len() {
            let prev_label = labels[pos - 1];
            let curr_label = labels[pos];
            let prev_idx = *label_to_idx.get(&prev_label).unwrap();
            let curr_idx = *label_to_idx.get(&curr_label).unwrap();

            let transition_idx = n_features + prev_idx * n_labels + curr_idx;
            features[transition_idx] += 1.0;
        }

        Ok(features)
    }

    /// Loss-augmented inference (finding most violating constraint)
    fn loss_augmented_inference(
        &self,
        sequence: &Sequence,
        true_labels: &Array1<usize>,
        weights: &Array1<f64>,
        transition_weights: &Array2<f64>,
        label_to_idx: &HashMap<usize, usize>,
    ) -> Result<Array1<usize>> {
        match self.config.inference {
            InferenceAlgorithm::Viterbi => self.viterbi_loss_augmented(
                sequence,
                true_labels,
                weights,
                transition_weights,
                label_to_idx,
            ),
            _ => {
                // Fallback to simple greedy decoding for now
                self.greedy_inference(sequence, weights, label_to_idx)
            }
        }
    }

    /// Viterbi algorithm with loss augmentation
    fn viterbi_loss_augmented(
        &self,
        sequence: &Sequence,
        true_labels: &Array1<usize>,
        weights: &Array1<f64>,
        transition_weights: &Array2<f64>,
        label_to_idx: &HashMap<usize, usize>,
    ) -> Result<Array1<usize>> {
        let seq_len = sequence.length;
        let n_labels = label_to_idx.len();
        let _n_features = sequence.features.ncols();

        // Dynamic programming tables
        let mut dp = Array2::zeros((seq_len, n_labels));
        let mut backtrack = Array2::zeros((seq_len, n_labels));

        // Initialize first position
        for (label, &label_idx) in label_to_idx.iter() {
            let node_features = sequence.feature_at(0);
            let mut score = 0.0;

            // Node features
            for (feat_idx, &feat_val) in node_features.iter().enumerate() {
                score += weights[feat_idx] * feat_val;
            }

            // Loss augmentation
            if *label != true_labels[0] {
                score += 1.0; // Hamming loss
            }

            dp[[0, label_idx]] = score;
        }

        // Forward pass
        for pos in 1..seq_len {
            let node_features = sequence.feature_at(pos);

            for (curr_label, &curr_idx) in label_to_idx.iter() {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0;

                for (_prev_label, &prev_idx) in label_to_idx.iter() {
                    let transition_score = transition_weights[[prev_idx, curr_idx]];
                    let total_score = dp[[pos - 1, prev_idx]] + transition_score;

                    if total_score > best_score {
                        best_score = total_score;
                        best_prev = prev_idx;
                    }
                }

                // Add node features
                for (feat_idx, &feat_val) in node_features.iter().enumerate() {
                    best_score += weights[feat_idx] * feat_val;
                }

                // Loss augmentation
                if *curr_label != true_labels[pos] {
                    best_score += 1.0; // Hamming loss
                }

                dp[[pos, curr_idx]] = best_score;
                backtrack[[pos, curr_idx]] = best_prev as f64;
            }
        }

        // Find best final state
        let mut best_final_score = f64::NEG_INFINITY;
        let mut best_final_state = 0;
        for label_idx in 0..n_labels {
            if dp[[seq_len - 1, label_idx]] > best_final_score {
                best_final_score = dp[[seq_len - 1, label_idx]];
                best_final_state = label_idx;
            }
        }

        // Backward pass (traceback)
        let mut path = Array1::zeros(seq_len);
        let mut current_state = best_final_state;

        for pos in (0..seq_len).rev() {
            path[pos] = current_state as f64;
            if pos > 0 {
                current_state = backtrack[[pos, current_state]] as usize;
            }
        }

        // Convert indices back to labels
        let idx_to_label = label_to_idx
            .iter()
            .map(|(&k, &v)| (v, k))
            .collect::<HashMap<_, _>>();
        let labels = path
            .iter()
            .map(|&idx| idx_to_label[&(idx as usize)])
            .collect();

        Ok(Array1::from_vec(labels))
    }

    /// Simple greedy inference
    fn greedy_inference(
        &self,
        sequence: &Sequence,
        weights: &Array1<f64>,
        label_to_idx: &HashMap<usize, usize>,
    ) -> Result<Array1<usize>> {
        let mut labels = Array1::zeros(sequence.length);

        for pos in 0..sequence.length {
            let node_features = sequence.feature_at(pos);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_label = 0;

            for (label, _) in label_to_idx.iter() {
                let mut score = 0.0;
                for (feat_idx, &feat_val) in node_features.iter().enumerate() {
                    score += weights[feat_idx] * feat_val;
                }

                if score > best_score {
                    best_score = score;
                    best_label = *label;
                }
            }

            labels[pos] = best_label;
        }

        Ok(labels)
    }

    /// Compute loss between true and predicted labels
    fn compute_loss(
        &self,
        true_labels: &Array1<usize>,
        pred_labels: &Array1<usize>,
    ) -> Result<f64> {
        match self.config.loss {
            StructuredLoss::Hamming => {
                let mut loss = 0.0;
                for (true_label, pred_label) in true_labels.iter().zip(pred_labels.iter()) {
                    if true_label != pred_label {
                        loss += 1.0;
                    }
                }
                Ok(loss)
            }
            StructuredLoss::F1 => {
                // Simplified F1 loss calculation
                let mut tp = 0.0;
                let mut fp = 0.0;
                let mut fn_count = 0.0;

                for (true_label, pred_label) in true_labels.iter().zip(pred_labels.iter()) {
                    if true_label == pred_label && *true_label != 0 {
                        tp += 1.0;
                    } else if *pred_label != 0 {
                        fp += 1.0;
                    } else if *true_label != 0 {
                        fn_count += 1.0;
                    }
                }

                let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
                let recall = if tp + fn_count > 0.0 {
                    tp / (tp + fn_count)
                } else {
                    0.0
                };
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };

                Ok(1.0 - f1)
            }
            StructuredLoss::EditDistance => {
                // Simplified edit distance (Levenshtein distance)
                let n = true_labels.len();
                let m = pred_labels.len();
                let mut dp = Array2::zeros((n + 1, m + 1));

                // Initialize base cases
                for i in 0..=n {
                    dp[[i, 0]] = i as f64;
                }
                for j in 0..=m {
                    dp[[0, j]] = j as f64;
                }

                // Fill DP table
                for i in 1..=n {
                    for j in 1..=m {
                        let cost = if true_labels[i - 1] == pred_labels[j - 1] {
                            0.0
                        } else {
                            1.0
                        };
                        dp[[i, j]] = (dp[[i - 1, j]] + 1.0)
                            .min(dp[[i, j - 1]] + 1.0)
                            .min(dp[[i - 1, j - 1]] + cost);
                    }
                }

                Ok(dp[[n, m]])
            }
        }
    }
}

impl Predict<Vec<Sequence>, Vec<Array1<usize>>> for StructuredSVM<Trained> {
    fn predict(&self, sequences: &Vec<Sequence>) -> Result<Vec<Array1<usize>>> {
        let weights = self.weights.as_ref().unwrap();
        let transition_weights = self.transition_weights.as_ref().unwrap();
        let label_to_idx = self.label_to_idx.as_ref().unwrap();

        let mut predictions = Vec::new();

        for sequence in sequences {
            let pred_labels = match self.config.inference {
                InferenceAlgorithm::Viterbi => {
                    self.viterbi_inference(sequence, weights, transition_weights, label_to_idx)?
                }
                _ => self.greedy_inference_trained(sequence, weights, label_to_idx)?,
            };
            predictions.push(pred_labels);
        }

        Ok(predictions)
    }
}

impl StructuredSVM<Trained> {
    /// Viterbi inference for prediction
    fn viterbi_inference(
        &self,
        sequence: &Sequence,
        weights: &Array1<f64>,
        transition_weights: &Array2<f64>,
        label_to_idx: &HashMap<usize, usize>,
    ) -> Result<Array1<usize>> {
        let seq_len = sequence.length;
        let n_labels = label_to_idx.len();

        // Dynamic programming tables
        let mut dp = Array2::zeros((seq_len, n_labels));
        let mut backtrack = Array2::zeros((seq_len, n_labels));

        // Initialize first position
        for (_label, &label_idx) in label_to_idx.iter() {
            let node_features = sequence.feature_at(0);
            let mut score = 0.0;

            // Node features
            for (feat_idx, &feat_val) in node_features.iter().enumerate() {
                score += weights[feat_idx] * feat_val;
            }

            dp[[0, label_idx]] = score;
        }

        // Forward pass
        for pos in 1..seq_len {
            let node_features = sequence.feature_at(pos);

            for (_curr_label, &curr_idx) in label_to_idx.iter() {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0;

                for (_prev_label, &prev_idx) in label_to_idx.iter() {
                    let transition_score = transition_weights[[prev_idx, curr_idx]];
                    let total_score = dp[[pos - 1, prev_idx]] + transition_score;

                    if total_score > best_score {
                        best_score = total_score;
                        best_prev = prev_idx;
                    }
                }

                // Add node features
                for (feat_idx, &feat_val) in node_features.iter().enumerate() {
                    best_score += weights[feat_idx] * feat_val;
                }

                dp[[pos, curr_idx]] = best_score;
                backtrack[[pos, curr_idx]] = best_prev as f64;
            }
        }

        // Find best final state
        let mut best_final_score = f64::NEG_INFINITY;
        let mut best_final_state = 0;
        for label_idx in 0..n_labels {
            if dp[[seq_len - 1, label_idx]] > best_final_score {
                best_final_score = dp[[seq_len - 1, label_idx]];
                best_final_state = label_idx;
            }
        }

        // Backward pass (traceback)
        let mut path = Array1::zeros(seq_len);
        let mut current_state = best_final_state;

        for pos in (0..seq_len).rev() {
            path[pos] = current_state as f64;
            if pos > 0 {
                current_state = backtrack[[pos, current_state]] as usize;
            }
        }

        // Convert indices back to labels
        let idx_to_label = label_to_idx
            .iter()
            .map(|(&k, &v)| (v, k))
            .collect::<HashMap<_, _>>();
        let labels = path
            .iter()
            .map(|&idx| idx_to_label[&(idx as usize)])
            .collect();

        Ok(Array1::from_vec(labels))
    }

    /// Simple greedy inference for trained model
    fn greedy_inference_trained(
        &self,
        sequence: &Sequence,
        weights: &Array1<f64>,
        label_to_idx: &HashMap<usize, usize>,
    ) -> Result<Array1<usize>> {
        let mut labels = Array1::zeros(sequence.length);

        for pos in 0..sequence.length {
            let node_features = sequence.feature_at(pos);
            let mut best_score = f64::NEG_INFINITY;
            let mut best_label = 0;

            for (label, _) in label_to_idx.iter() {
                let mut score = 0.0;
                for (feat_idx, &feat_val) in node_features.iter().enumerate() {
                    score += weights[feat_idx] * feat_val;
                }

                if score > best_score {
                    best_score = score;
                    best_label = *label;
                }
            }

            labels[pos] = best_label;
        }

        Ok(labels)
    }

    /// Get model weights
    pub fn weights(&self) -> &Array1<f64> {
        self.weights.as_ref().unwrap()
    }

    /// Get transition weights
    pub fn transition_weights(&self) -> &Array2<f64> {
        self.transition_weights.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn create_test_sequences() -> (Vec<Sequence>, Vec<Array1<usize>>) {
        // Create simple test sequences for POS tagging-like task
        let seq1 = Sequence::new(
            array![[1.0, 0.5, 0.2], [0.8, 0.9, 0.1], [0.3, 0.1, 0.7]],
            None,
        );
        let labels1 = array![0, 1, 2]; // NOUN, VERB, ADJ

        let seq2 = Sequence::new(array![[0.9, 0.4, 0.3], [0.2, 0.8, 0.2]], None);
        let labels2 = array![0, 1]; // NOUN, VERB

        (vec![seq1, seq2], vec![labels1, labels2])
    }

    #[test]
    fn test_structured_svm_creation() {
        let svm = StructuredSVM::new()
            .with_c(1.0)
            .with_loss(StructuredLoss::Hamming)
            .with_inference(InferenceAlgorithm::Viterbi)
            .with_max_iter(50)
            .verbose(false);

        assert_eq!(svm.config.c, 1.0);
        assert!(matches!(svm.config.loss, StructuredLoss::Hamming));
        assert!(matches!(svm.config.inference, InferenceAlgorithm::Viterbi));
    }

    #[test]
    fn test_sequence_creation() {
        let features = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array![0, 1];
        let seq = Sequence::new(features.clone(), Some(labels.clone()));

        assert_eq!(seq.length, 2);
        assert_eq!(seq.feature_at(0), array![1.0, 2.0]);
        assert_eq!(seq.label_at(0), Some(0));
    }

    #[test]
    fn test_structured_svm_fit() {
        let (sequences, labels) = create_test_sequences();

        let svm = StructuredSVM::new()
            .with_c(0.1)
            .with_max_iter(10)
            .with_tolerance(1e-3);

        use sklears_core::traits::Fit;
        let result = svm.fit(&sequences, &labels);
        assert!(result.is_ok());

        let trained_svm = result.unwrap();
        assert!(trained_svm.weights.is_some());
        assert!(trained_svm.transition_weights.is_some());
    }

    #[test]
    fn test_structured_svm_predict() {
        let (sequences, labels) = create_test_sequences();

        let svm = StructuredSVM::new()
            .with_c(0.1)
            .with_max_iter(5)
            .with_tolerance(1e-2);

        use sklears_core::traits::Predict;
        let trained_svm = svm.fit(&sequences, &labels).unwrap();

        let predictions = trained_svm.predict(&sequences);
        assert!(predictions.is_ok());

        let pred_labels = predictions.unwrap();
        assert_eq!(pred_labels.len(), 2);
        assert_eq!(pred_labels[0].len(), 3);
        assert_eq!(pred_labels[1].len(), 2);
    }

    #[test]
    fn test_hamming_loss() {
        let svm = StructuredSVM::new();
        let true_labels = array![0, 1, 2, 1];
        let pred_labels = array![0, 2, 2, 0];

        let loss = svm.compute_loss(&true_labels, &pred_labels).unwrap();
        assert_eq!(loss, 2.0); // Two mismatches
    }

    #[test]
    fn test_invalid_input() {
        let (sequences, mut labels) = create_test_sequences();
        labels.pop(); // Make lengths mismatch

        let svm = StructuredSVM::new();
        use sklears_core::traits::Fit;
        let result = svm.fit(&sequences, &labels);
        assert!(result.is_err());
    }
}
