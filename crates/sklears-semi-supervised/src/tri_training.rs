//! Tri-Training implementation for semi-supervised learning

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Random, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};

/// Tri-Training classifier for semi-supervised learning
///
/// Tri-training uses three classifiers trained on different bootstrap samples
/// of the labeled data. Two classifiers vote to label examples for the third.
///
/// # Parameters
///
/// * `max_iter` - Maximum number of iterations
/// * `verbose` - Whether to print progress information
/// * `theta` - Threshold for noise rate estimation
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::TriTraining;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let tt = TriTraining::new()
///     .max_iter(10)
///     .verbose(true);
/// let fitted = tt.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TriTraining<S = Untrained> {
    state: S,
    max_iter: usize,
    verbose: bool,
    theta: f64,
}

impl TriTraining<Untrained> {
    /// Create a new TriTraining instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            max_iter: 30,
            verbose: false,
            theta: 0.1,
        }
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

    /// Set theta parameter for noise rate estimation
    pub fn theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    fn bootstrap_sample(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        labeled_indices: &[usize],
    ) -> (Array2<f64>, Array1<i32>) {
        let n_labeled = labeled_indices.len();
        let mut bootstrap_X = Array2::zeros((n_labeled, X.ncols()));
        let mut bootstrap_y = Array1::zeros(n_labeled);

        // Bootstrap sampling with replacement
        let mut rng = Random::seed(42);
        for i in 0..n_labeled {
            let random_idx = rng.gen_range(0..n_labeled);
            let idx = labeled_indices[random_idx];
            bootstrap_X.row_mut(i).assign(&X.row(idx));
            bootstrap_y[i] = y[idx];
        }

        (bootstrap_X, bootstrap_y)
    }

    fn simple_classifier_fit_predict(
        &self,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
    ) -> Array1<i32> {
        let n_test = X_test.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            // Simple k-NN classifier
            let mut distances: Vec<(f64, i32)> = Vec::new();
            for j in 0..X_train.nrows() {
                let diff = &X_test.row(i) - &X_train.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((dist, y_train[j]));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Use k=5 nearest neighbors with majority vote
            let k = distances.len().min(5).max(1);
            let mut class_votes: HashMap<i32, usize> = HashMap::new();

            for &(_, label) in distances.iter().take(k) {
                *class_votes.entry(label).or_insert(0) += 1;
            }

            let best_class = class_votes
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&class, _)| class)
                .unwrap_or(y_train[0]);

            predictions[i] = best_class;
        }

        predictions
    }

    fn estimate_error_rate(
        &self,
        classifier_i: &Array2<f64>,
        y_i: &Array1<i32>,
        classifier_j: &Array2<f64>,
        y_j: &Array1<i32>,
        X_labeled: &Array2<f64>,
        y_labeled: &Array1<i32>,
    ) -> f64 {
        let n_labeled = X_labeled.nrows();
        let mut errors = 0;
        let mut total = 0;

        for k in 0..n_labeled {
            let test_sample = X_labeled
                .row(k)
                .to_owned()
                .insert_axis(scirs2_core::ndarray::Axis(0));
            let pred_i = self.simple_classifier_fit_predict(classifier_i, y_i, &test_sample);
            let pred_j = self.simple_classifier_fit_predict(classifier_j, y_j, &test_sample);

            if pred_i[0] == pred_j[0] {
                total += 1;
                if pred_i[0] != y_labeled[k] {
                    errors += 1;
                }
            }
        }

        if total > 0 {
            errors as f64 / total as f64
        } else {
            1.0 // Conservative estimate if no agreement
        }
    }
}

impl Default for TriTraining<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TriTraining<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for TriTraining<Untrained> {
    type Fitted = TriTraining<TriTrainingTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let mut y = y.to_owned();

        // Identify labeled and unlabeled samples
        let mut labeled_indices: Vec<usize> = y
            .iter()
            .enumerate()
            .filter(|(_, &label)| label != -1)
            .map(|(i, _)| i)
            .collect();

        let mut unlabeled_indices: Vec<usize> = y
            .iter()
            .enumerate()
            .filter(|(_, &label)| label == -1)
            .map(|(i, _)| i)
            .collect();

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let mut classes = HashSet::new();
        for &idx in &labeled_indices {
            classes.insert(y[idx]);
        }
        let classes: Vec<i32> = classes.into_iter().collect();

        // Initialize three classifiers with bootstrap samples
        let mut classifiers: Vec<(Array2<f64>, Array1<i32>)> = Vec::new();
        for _ in 0..3 {
            let (bootstrap_X, bootstrap_y) = self.bootstrap_sample(&X, &y, &labeled_indices);
            classifiers.push((bootstrap_X, bootstrap_y));
        }

        let mut e_prime = [0.5; 3]; // Previous error rates
        let mut l_prime = [0; 3]; // Previous unlabeled set sizes

        // Tri-training iterations
        for iter in 0..self.max_iter {
            let mut any_changes = false;

            for i in 0..3 {
                let j = (i + 1) % 3;
                let k = (i + 2) % 3;

                // Extract labeled data for current training set
                let X_labeled_i: Vec<Vec<f64>> = labeled_indices
                    .iter()
                    .map(|&idx| X.row(idx).to_vec())
                    .collect();
                let y_labeled_i: Vec<i32> = labeled_indices.iter().map(|&idx| y[idx]).collect();

                let X_labeled_array = Array2::from_shape_vec(
                    (X_labeled_i.len(), X.ncols()),
                    X_labeled_i.into_iter().flatten().collect(),
                )
                .map_err(|_| {
                    SklearsError::InvalidInput("Failed to create labeled training data".to_string())
                })?;

                let y_labeled_array = Array1::from(y_labeled_i);

                // Estimate error rate between classifiers j and k
                let e_jk = self.estimate_error_rate(
                    &classifiers[j].0,
                    &classifiers[j].1,
                    &classifiers[k].0,
                    &classifiers[k].1,
                    &X_labeled_array,
                    &y_labeled_array,
                );

                if e_jk < e_prime[i] && e_jk < self.theta {
                    // Get predictions from classifiers j and k on unlabeled data
                    if !unlabeled_indices.is_empty() {
                        let X_unlabeled: Vec<Vec<f64>> = unlabeled_indices
                            .iter()
                            .map(|&idx| X.row(idx).to_vec())
                            .collect();

                        let X_unlabeled_array = Array2::from_shape_vec(
                            (X_unlabeled.len(), X.ncols()),
                            X_unlabeled.into_iter().flatten().collect(),
                        )
                        .map_err(|_| {
                            SklearsError::InvalidInput(
                                "Failed to create unlabeled data".to_string(),
                            )
                        })?;

                        let pred_j = self.simple_classifier_fit_predict(
                            &classifiers[j].0,
                            &classifiers[j].1,
                            &X_unlabeled_array,
                        );
                        let pred_k = self.simple_classifier_fit_predict(
                            &classifiers[k].0,
                            &classifiers[k].1,
                            &X_unlabeled_array,
                        );

                        // Find samples where j and k agree
                        let mut new_labeled_for_i = Vec::new();
                        for (idx, (&p_j, &p_k)) in pred_j.iter().zip(pred_k.iter()).enumerate() {
                            if p_j == p_k {
                                let original_idx = unlabeled_indices[idx];
                                new_labeled_for_i.push((original_idx, p_j));
                            }
                        }

                        if !new_labeled_for_i.is_empty() {
                            // Add agreed-upon labels
                            for (idx, label) in new_labeled_for_i {
                                y[idx] = label;
                                labeled_indices.push(idx);
                                any_changes = true;
                            }

                            // Update unlabeled indices
                            unlabeled_indices.retain(|&idx| y[idx] == -1);

                            // Retrain classifier i with new data
                            let (new_bootstrap_X, new_bootstrap_y) =
                                self.bootstrap_sample(&X, &y, &labeled_indices);
                            classifiers[i] = (new_bootstrap_X, new_bootstrap_y);

                            e_prime[i] = e_jk;
                            l_prime[i] = unlabeled_indices.len();
                        }
                    }
                }
            }

            if !any_changes {
                if self.verbose {
                    println!("Iteration {}: No changes, stopping", iter + 1);
                }
                break;
            }

            if self.verbose {
                let n_labeled = labeled_indices.len();
                let n_unlabeled = unlabeled_indices.len();
                println!(
                    "Iteration {}: {} labeled, {} unlabeled",
                    iter + 1,
                    n_labeled,
                    n_unlabeled
                );
            }

            if unlabeled_indices.is_empty() {
                if self.verbose {
                    println!("All samples labeled, stopping");
                }
                break;
            }
        }

        Ok(TriTraining {
            state: TriTrainingTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes),
                classifiers,
            },
            max_iter: self.max_iter,
            verbose: self.verbose,
            theta: self.theta,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for TriTraining<TriTrainingTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        // Use ensemble prediction from all three classifiers
        for i in 0..n_test {
            let test_sample = X
                .row(i)
                .to_owned()
                .insert_axis(scirs2_core::ndarray::Axis(0));
            let mut votes: HashMap<i32, usize> = HashMap::new();

            // Get prediction from each classifier
            for (classifier_X, classifier_y) in &self.state.classifiers {
                let pred = TriTraining::<TriTrainingTrained>::simple_classifier_fit_predict_static(
                    classifier_X,
                    classifier_y,
                    &test_sample,
                );
                *votes.entry(pred[0]).or_insert(0) += 1;
            }

            // Majority vote
            let best_class = votes
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&class, _)| class)
                .unwrap_or(self.state.classes[0]);

            predictions[i] = best_class;
        }

        Ok(predictions)
    }
}

impl TriTraining<TriTrainingTrained> {
    /// Static version of the classifier for predictions
    fn simple_classifier_fit_predict_static(
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
    ) -> Array1<i32> {
        let n_test = X_test.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            // Simple k-NN classifier
            let mut distances: Vec<(f64, i32)> = Vec::new();
            for j in 0..X_train.nrows() {
                let diff = &X_test.row(i) - &X_train.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((dist, y_train[j]));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Use k=5 nearest neighbors with majority vote
            let k = distances.len().min(5).max(1);
            let mut class_votes: HashMap<i32, usize> = HashMap::new();

            for &(_, label) in distances.iter().take(k) {
                *class_votes.entry(label).or_insert(0) += 1;
            }

            let best_class = class_votes
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&class, _)| class)
                .unwrap_or(y_train[0]);

            predictions[i] = best_class;
        }

        predictions
    }
}

/// Trained state for TriTraining
#[derive(Debug, Clone)]
pub struct TriTrainingTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// classifiers
    pub classifiers: Vec<(Array2<f64>, Array1<i32>)>,
}
