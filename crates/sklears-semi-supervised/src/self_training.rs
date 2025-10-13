//! Enhanced Self-Training implementation with multiple confidence measures

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};

/// Enhanced Self-Training Classifier with multiple confidence measures
#[derive(Debug, Clone)]
pub struct EnhancedSelfTraining<S = Untrained> {
    state: S,
    threshold: f64,
    criterion: String,
    k_best: usize,
    max_iter: usize,
    verbose: bool,
    confidence_method: String,
    min_samples_per_class: usize,
}

impl EnhancedSelfTraining<Untrained> {
    /// Create a new EnhancedSelfTraining instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            threshold: 0.75,
            criterion: "threshold".to_string(),
            k_best: 10,
            max_iter: 10,
            verbose: false,
            confidence_method: "max_proba".to_string(),
            min_samples_per_class: 1,
        }
    }

    /// Set the confidence threshold
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the selection criterion
    pub fn criterion(mut self, criterion: String) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set the number of best samples to select
    pub fn k_best(mut self, k_best: usize) -> Self {
        self.k_best = k_best;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set confidence computation method
    pub fn confidence_method(mut self, method: String) -> Self {
        self.confidence_method = method;
        self
    }

    /// Set minimum samples per class threshold
    pub fn min_samples_per_class(mut self, min_samples: usize) -> Self {
        self.min_samples_per_class = min_samples;
        self
    }

    fn compute_confidence(&self, probabilities: &Array1<f64>) -> f64 {
        match self.confidence_method.as_str() {
            "max_proba" => probabilities.iter().fold(0.0, |a, &b| a.max(b)),
            "entropy" => {
                let entropy = -probabilities
                    .iter()
                    .filter(|&&p| p > 0.0)
                    .map(|&p| p * p.ln())
                    .sum::<f64>();
                let max_entropy = (probabilities.len() as f64).ln();
                if max_entropy > 0.0 {
                    1.0 - (entropy / max_entropy)
                } else {
                    1.0
                }
            }
            "margin" => {
                let mut sorted_probs: Vec<f64> = probabilities.to_vec();
                sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
                if sorted_probs.len() >= 2 {
                    sorted_probs[0] - sorted_probs[1]
                } else {
                    sorted_probs[0]
                }
            }
            _ => probabilities.iter().fold(0.0, |a, &b| a.max(b)),
        }
    }
}

impl Default for EnhancedSelfTraining<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for EnhancedSelfTraining<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for EnhancedSelfTraining<Untrained> {
    type Fitted = EnhancedSelfTraining<EnhancedSelfTrainingTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let mut y = y.to_owned();

        // Identify labeled and unlabeled samples
        let mut labeled_mask = Array1::from_elem(y.len(), false);
        let mut classes = HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label != -1 {
                labeled_mask[i] = true;
                classes.insert(label);
            }
        }

        if labeled_mask.iter().all(|&x| !x) {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Enhanced self-training with better confidence measures
        for iter in 0..self.max_iter {
            let labeled_indices: Vec<usize> = labeled_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_labeled)| is_labeled)
                .map(|(i, _)| i)
                .collect();

            if labeled_indices.len() == y.len() {
                break;
            }

            // Check class balance
            let mut class_counts = HashMap::new();
            for &idx in &labeled_indices {
                *class_counts.entry(y[idx]).or_insert(0) += 1;
            }

            // Skip if any class has too few samples
            if class_counts
                .values()
                .any(|&count| count < self.min_samples_per_class)
            {
                if iter == 0 {
                    return Err(SklearsError::InvalidInput(
                        "Insufficient labeled samples per class".to_string(),
                    ));
                }
                continue;
            }

            // Compute pseudo-labels with enhanced confidence
            let mut candidates = Vec::new();

            for (i, &is_labeled) in labeled_mask.iter().enumerate() {
                if !is_labeled {
                    // Find k nearest labeled neighbors
                    let mut distances: Vec<(usize, f64, i32)> = Vec::new();
                    for &labeled_idx in &labeled_indices {
                        let diff = &X.row(i) - &X.row(labeled_idx);
                        let dist = diff.mapv(|x| x * x).sum().sqrt();
                        distances.push((labeled_idx, dist, y[labeled_idx]));
                    }

                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    // Compute class probabilities based on k-NN
                    let k = (labeled_indices.len().min(7)).max(1);
                    let mut class_probs = vec![0.0; n_classes];
                    let mut total_weight = 0.0;

                    for &(_, dist, label) in distances.iter().take(k) {
                        let weight = if dist > 0.0 { 1.0 / (1.0 + dist) } else { 1.0 };
                        if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                            class_probs[class_idx] += weight;
                            total_weight += weight;
                        }
                    }

                    if total_weight > 0.0 {
                        for prob in &mut class_probs {
                            *prob /= total_weight;
                        }
                    }

                    let proba_array = Array1::from(class_probs);
                    let confidence = self.compute_confidence(&proba_array);
                    let predicted_class_idx = proba_array
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0;

                    candidates.push((i, classes[predicted_class_idx], confidence));
                }
            }

            // Select samples to pseudo-label
            let mut selected_indices = Vec::new();

            match self.criterion.as_str() {
                "threshold" => {
                    for (i, label, confidence) in candidates {
                        if confidence >= self.threshold {
                            selected_indices.push((i, label));
                        }
                    }
                }
                "k_best" => {
                    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                    for (i, label, _) in candidates.into_iter().take(self.k_best) {
                        selected_indices.push((i, label));
                    }
                }
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown criterion: {}",
                        self.criterion
                    )));
                }
            }

            if selected_indices.is_empty() {
                if self.verbose {
                    println!("Iteration {}: No confident predictions, stopping", iter + 1);
                }
                break;
            }

            // Add pseudo-labels
            for (i, label) in selected_indices {
                y[i] = label;
                labeled_mask[i] = true;
            }

            if self.verbose {
                let n_labeled = labeled_mask.iter().filter(|&&x| x).count();
                println!("Iteration {}: {} labeled samples", iter + 1, n_labeled);
            }
        }

        Ok(EnhancedSelfTraining {
            state: EnhancedSelfTrainingTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes),
                labeled_mask,
            },
            threshold: self.threshold,
            criterion: self.criterion,
            k_best: self.k_best,
            max_iter: self.max_iter,
            verbose: self.verbose,
            confidence_method: self.confidence_method,
            min_samples_per_class: self.min_samples_per_class,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for EnhancedSelfTraining<EnhancedSelfTrainingTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        let labeled_indices: Vec<usize> = self
            .state
            .labeled_mask
            .iter()
            .enumerate()
            .filter(|(_, &is_labeled)| is_labeled)
            .map(|(i, _)| i)
            .collect();

        for i in 0..n_test {
            let mut min_dist = f64::INFINITY;
            let mut best_label = 0;

            for &labeled_idx in &labeled_indices {
                let diff = &X.row(i) - &self.state.X_train.row(labeled_idx);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    best_label = self.state.y_train[labeled_idx];
                }
            }

            predictions[i] = best_label;
        }

        Ok(predictions)
    }
}

/// Trained state for EnhancedSelfTraining
#[derive(Debug, Clone)]
pub struct EnhancedSelfTrainingTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// labeled_mask
    pub labeled_mask: Array1<bool>,
}
