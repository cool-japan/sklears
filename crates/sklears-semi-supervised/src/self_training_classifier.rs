//! Self-Training Classifier implementation

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashSet;

/// Self-Training Classifier
///
/// Self-training is a wrapper method for semi-supervised learning where a
/// supervised classifier is trained on labeled data, then used to classify
/// unlabeled data. The most confident predictions are added to the training set.
///
/// # Parameters
///
/// * `base_classifier` - The base classifier to use
/// * `threshold` - Confidence threshold for pseudo-labeling
/// * `criterion` - Criterion for selecting samples ('threshold' or 'k_best')
/// * `k_best` - Number of best samples to select per iteration
/// * `max_iter` - Maximum number of iterations
/// * `verbose` - Whether to print progress information
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::SelfTrainingClassifier;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let stc = SelfTrainingClassifier::new()
///     .threshold(0.75)
///     .max_iter(10);
/// let fitted = stc.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SelfTrainingClassifier<S = Untrained> {
    state: S,
    threshold: f64,
    criterion: String,
    k_best: usize,
    max_iter: usize,
    verbose: bool,
}

impl SelfTrainingClassifier<Untrained> {
    /// Create a new SelfTrainingClassifier instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            threshold: 0.75,
            criterion: "threshold".to_string(),
            k_best: 10,
            max_iter: 10,
            verbose: false,
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
}

impl Default for SelfTrainingClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SelfTrainingClassifier<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for SelfTrainingClassifier<Untrained> {
    type Fitted = SelfTrainingClassifier<SelfTrainingTrained>;

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

        // Simple self-training iteration
        for _iter in 0..self.max_iter {
            // Find labeled samples
            let labeled_indices: Vec<usize> = labeled_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_labeled)| is_labeled)
                .map(|(i, _)| i)
                .collect();

            if labeled_indices.len() == y.len() {
                break; // All samples are labeled
            }

            // Extract labeled data
            let _X_labeled: Vec<Vec<f64>> =
                labeled_indices.iter().map(|&i| X.row(i).to_vec()).collect();
            let y_labeled: Vec<i32> = labeled_indices.iter().map(|&i| y[i]).collect();

            // Simple nearest neighbor classifier for pseudo-labeling
            let mut new_labels = Vec::new();
            let mut confidences = Vec::new();

            for (i, &is_labeled) in labeled_mask.iter().enumerate() {
                if !is_labeled {
                    // Find nearest labeled neighbor
                    let mut min_dist = f64::INFINITY;
                    let mut best_label = 0;

                    for (j, &labeled_idx) in labeled_indices.iter().enumerate() {
                        let diff = &X.row(i) - &X.row(labeled_idx);
                        let dist = diff.mapv(|x: f64| x * x).sum().sqrt();
                        if dist < min_dist {
                            min_dist = dist;
                            best_label = y_labeled[j];
                        }
                    }

                    // Simple confidence based on distance
                    let confidence = 1.0 / (1.0 + min_dist);
                    new_labels.push((i, best_label, confidence));
                    confidences.push(confidence);
                }
            }

            // Select samples to pseudo-label
            let mut selected_indices = Vec::new();

            match self.criterion.as_str() {
                "threshold" => {
                    for (i, label, confidence) in new_labels {
                        if confidence >= self.threshold {
                            selected_indices.push((i, label));
                        }
                    }
                }
                "k_best" => {
                    new_labels.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                    for (i, label, _) in new_labels.into_iter().take(self.k_best) {
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
                break; // No confident predictions
            }

            // Add pseudo-labels
            for (i, label) in selected_indices {
                y[i] = label;
                labeled_mask[i] = true;
            }

            if self.verbose {
                let n_labeled = labeled_mask.iter().filter(|&&x| x).count();
                println!("Iteration {}: {} labeled samples", _iter + 1, n_labeled);
            }
        }

        Ok(SelfTrainingClassifier {
            state: SelfTrainingTrained {
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
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for SelfTrainingClassifier<SelfTrainingTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        // Get labeled training samples
        let labeled_indices: Vec<usize> = self
            .state
            .labeled_mask
            .iter()
            .enumerate()
            .filter(|(_, &is_labeled)| is_labeled)
            .map(|(i, _)| i)
            .collect();

        for i in 0..n_test {
            // Find nearest labeled neighbor
            let mut min_dist = f64::INFINITY;
            let mut best_label = 0;

            for &labeled_idx in &labeled_indices {
                let diff = &X.row(i) - &self.state.X_train.row(labeled_idx);
                let dist = diff.mapv(|x: f64| x * x).sum().sqrt();
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

/// Trained state for SelfTrainingClassifier
#[derive(Debug, Clone)]
pub struct SelfTrainingTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// labeled_mask
    pub labeled_mask: Array1<bool>,
}
