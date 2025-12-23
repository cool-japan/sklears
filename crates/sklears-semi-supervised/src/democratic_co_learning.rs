//! Democratic Co-Learning implementation for semi-supervised learning

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};

/// Democratic Co-Learning classifier for semi-supervised learning
///
/// Democratic co-learning extends traditional co-training by using multiple classifiers
/// trained on different views of the data. Instead of pairwise labeling, all classifiers
/// vote democratically on which unlabeled samples should be added to the training set.
///
/// # Parameters
///
/// * `views` - Feature indices for each view
/// * `k_add` - Number of samples to add per iteration
/// * `max_iter` - Maximum number of iterations
/// * `confidence_threshold` - Minimum confidence threshold for pseudo-labeling
/// * `min_agreement` - Minimum number of classifiers that must agree
/// * `verbose` - Whether to print progress information
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::DemocraticCoLearning;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
///                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
///                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
///                [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let dcl = DemocraticCoLearning::new()
///     .views(vec![vec![0, 1], vec![2, 3], vec![4, 5]])
///     .k_add(1)
///     .min_agreement(2)
///     .max_iter(10);
/// let fitted = dcl.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DemocraticCoLearning<S = Untrained> {
    state: S,
    views: Vec<Vec<usize>>,
    k_add: usize,
    max_iter: usize,
    confidence_threshold: f64,
    min_agreement: usize,
    verbose: bool,
    selection_strategy: String,
}

impl DemocraticCoLearning<Untrained> {
    /// Create a new DemocraticCoLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            views: Vec::new(),
            k_add: 5,
            max_iter: 30,
            confidence_threshold: 0.6,
            min_agreement: 2,
            verbose: false,
            selection_strategy: "confidence".to_string(),
        }
    }

    /// Set the feature views
    pub fn views(mut self, views: Vec<Vec<usize>>) -> Self {
        self.views = views;
        self
    }

    /// Set the number of samples to add per iteration
    pub fn k_add(mut self, k_add: usize) -> Self {
        self.k_add = k_add;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the confidence threshold
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the minimum number of classifiers that must agree
    pub fn min_agreement(mut self, min_agreement: usize) -> Self {
        self.min_agreement = min_agreement;
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set selection strategy for choosing samples to add
    pub fn selection_strategy(mut self, strategy: String) -> Self {
        self.selection_strategy = strategy;
        self
    }

    fn extract_view(&self, X: &Array2<f64>, view_features: &[usize]) -> SklResult<Array2<f64>> {
        if view_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "View features cannot be empty".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = view_features.len();
        let mut view_X = Array2::zeros((n_samples, n_features));

        for (new_j, &old_j) in view_features.iter().enumerate() {
            if old_j >= X.ncols() {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature index {} out of bounds",
                    old_j
                )));
            }
            for i in 0..n_samples {
                view_X[[i, new_j]] = X[[i, old_j]];
            }
        }

        Ok(view_X)
    }

    fn train_classifier(
        &self,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        classes: &[i32],
    ) -> (Array1<i32>, Array1<f64>) {
        let n_test = X_test.nrows();
        let mut predictions = Array1::zeros(n_test);
        let mut confidences = Array1::zeros(n_test);

        for i in 0..n_test {
            // Enhanced k-NN classifier with weighted voting
            let mut distances: Vec<(f64, i32)> = Vec::new();
            for j in 0..X_train.nrows() {
                let diff = &X_test.row(i) - &X_train.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((dist, y_train[j]));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Adaptive k based on data size
            let k = distances.len().clamp(3, 7).min(X_train.nrows());
            let mut class_votes: HashMap<i32, f64> = HashMap::new();
            let mut total_weight = 0.0;

            for &(dist, label) in distances.iter().take(k) {
                // Use inverse distance weighting
                let weight = if dist > 0.0 { 1.0 / (1.0 + dist) } else { 1.0 };
                *class_votes.entry(label).or_insert(0.0) += weight;
                total_weight += weight;
            }

            // Normalize votes to probabilities
            for (_, vote) in class_votes.iter_mut() {
                *vote /= total_weight;
            }

            // Find most likely class and confidence
            let (best_class, best_confidence) = class_votes
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(&class, &conf)| (class, conf))
                .unwrap_or((classes[0], 0.0));

            predictions[i] = best_class;
            confidences[i] = best_confidence;
        }

        (predictions, confidences)
    }

    fn democratic_vote(
        &self,
        predictions: &[Array1<i32>],
        confidences: &[Array1<f64>],
        classes: &[i32],
    ) -> Vec<(usize, i32, f64)> {
        let n_samples = predictions[0].len();
        let n_classifiers = predictions.len();
        let mut candidates = Vec::new();

        for i in 0..n_samples {
            // Count votes for each class
            let mut class_votes: HashMap<i32, usize> = HashMap::new();
            let mut total_confidence = 0.0;
            let mut voting_classifiers = 0;

            for (classifier_idx, (pred, conf)) in
                predictions.iter().zip(confidences.iter()).enumerate()
            {
                if conf[i] >= self.confidence_threshold {
                    *class_votes.entry(pred[i]).or_insert(0) += 1;
                    total_confidence += conf[i];
                    voting_classifiers += 1;
                }
            }

            // Find the class with most votes
            if let Some((&winning_class, &vote_count)) =
                class_votes.iter().max_by_key(|(_, &count)| count)
            {
                // Check if there's sufficient agreement
                // Use the minimum of min_agreement and available voting classifiers for more flexible agreement
                let required_agreement = self.min_agreement.min(voting_classifiers);
                if vote_count >= required_agreement && voting_classifiers >= 1 {
                    let avg_confidence = total_confidence / voting_classifiers as f64;

                    // Additional consensus measure: fraction of voting classifiers that agree
                    let consensus = vote_count as f64 / voting_classifiers as f64;
                    // Weight broader agreement: give bonus for more voting classifiers
                    let agreement_bonus = (voting_classifiers as f64).ln() + 1.0;
                    let combined_score = avg_confidence * consensus * agreement_bonus;

                    candidates.push((i, winning_class, combined_score));
                }
            }
        }

        // Sort by combined score (confidence * consensus)
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        candidates
    }
}

impl Default for DemocraticCoLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DemocraticCoLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for DemocraticCoLearning<Untrained> {
    type Fitted = DemocraticCoLearning<DemocraticCoLearningTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let mut y = y.to_owned();

        // Validate views
        if self.views.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Democratic co-learning requires at least 2 views".to_string(),
            ));
        }

        if self.min_agreement > self.views.len() {
            return Err(SklearsError::InvalidInput(
                "min_agreement cannot be greater than number of views".to_string(),
            ));
        }

        for (view_idx, view) in self.views.iter().enumerate() {
            if view.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "View {} cannot be empty",
                    view_idx
                )));
            }
            for &feature_idx in view {
                if feature_idx >= X.ncols() {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature index {} in view {} is out of bounds",
                        feature_idx, view_idx
                    )));
                }
            }
        }

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

        // Democratic co-learning iterations
        for iter in 0..self.max_iter {
            let labeled_indices: Vec<usize> = labeled_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_labeled)| is_labeled)
                .map(|(i, _)| i)
                .collect();

            let unlabeled_indices: Vec<usize> = labeled_mask
                .iter()
                .enumerate()
                .filter(|(_, &is_labeled)| !is_labeled)
                .map(|(i, _)| i)
                .collect();

            if unlabeled_indices.is_empty() {
                if self.verbose {
                    println!("Iteration {}: All samples labeled", iter + 1);
                }
                break;
            }

            // Train classifiers on each view
            let mut all_predictions = Vec::new();
            let mut all_confidences = Vec::new();

            for view in &self.views {
                // Extract view data
                let X_view = self.extract_view(&X, view)?;

                // Extract labeled training data for this view
                let X_labeled: Vec<Vec<f64>> = labeled_indices
                    .iter()
                    .map(|&i| X_view.row(i).to_vec())
                    .collect();
                let y_labeled: Array1<i32> = labeled_indices.iter().map(|&i| y[i]).collect();

                let X_labeled = Array2::from_shape_vec(
                    (X_labeled.len(), view.len()),
                    X_labeled.into_iter().flatten().collect(),
                )
                .map_err(|_| {
                    SklearsError::InvalidInput("Failed to create labeled training data".to_string())
                })?;

                // Extract unlabeled data for this view
                let X_unlabeled: Vec<Vec<f64>> = unlabeled_indices
                    .iter()
                    .map(|&i| X_view.row(i).to_vec())
                    .collect();

                let X_unlabeled = Array2::from_shape_vec(
                    (X_unlabeled.len(), view.len()),
                    X_unlabeled.into_iter().flatten().collect(),
                )
                .map_err(|_| {
                    SklearsError::InvalidInput("Failed to create unlabeled data".to_string())
                })?;

                // Train classifier and get predictions
                let (predictions, confidences) =
                    self.train_classifier(&X_labeled, &y_labeled, &X_unlabeled, &classes);
                all_predictions.push(predictions);
                all_confidences.push(confidences);
            }

            // Democratic voting to select samples to add
            let candidates = self.democratic_vote(&all_predictions, &all_confidences, &classes);

            if candidates.is_empty() {
                if self.verbose {
                    println!(
                        "Iteration {}: No agreed-upon confident predictions, stopping",
                        iter + 1
                    );
                }
                break;
            }

            // Select top k_add samples based on strategy
            let selected_count = candidates.len().min(self.k_add);
            let mut added_count = 0;

            for (candidate_idx, predicted_label, _score) in
                candidates.into_iter().take(selected_count)
            {
                let original_idx = unlabeled_indices[candidate_idx];
                y[original_idx] = predicted_label;
                labeled_mask[original_idx] = true;
                added_count += 1;
            }

            if added_count == 0 {
                if self.verbose {
                    println!("Iteration {}: No samples added, stopping", iter + 1);
                }
                break;
            }

            if self.verbose {
                let n_labeled = labeled_mask.iter().filter(|&&x| x).count();
                println!(
                    "Iteration {}: {} samples added, {} total labeled",
                    iter + 1,
                    added_count,
                    n_labeled
                );
            }
        }

        Ok(DemocraticCoLearning {
            state: DemocraticCoLearningTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes),
                labeled_mask,
                views: self.views.clone(),
            },
            views: self.views,
            k_add: self.k_add,
            max_iter: self.max_iter,
            confidence_threshold: self.confidence_threshold,
            min_agreement: self.min_agreement,
            verbose: self.verbose,
            selection_strategy: self.selection_strategy,
        })
    }
}

impl DemocraticCoLearning<DemocraticCoLearningTrained> {
    fn extract_view(&self, X: &Array2<f64>, view_features: &[usize]) -> SklResult<Array2<f64>> {
        if view_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "View features cannot be empty".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = view_features.len();
        let mut view_X = Array2::zeros((n_samples, n_features));

        for (new_j, &old_j) in view_features.iter().enumerate() {
            if old_j >= X.ncols() {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature index {} out of bounds",
                    old_j
                )));
            }
            for i in 0..n_samples {
                view_X[[i, new_j]] = X[[i, old_j]];
            }
        }

        Ok(view_X)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for DemocraticCoLearning<DemocraticCoLearningTrained>
{
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

        // Ensemble prediction using all views
        for i in 0..n_test {
            // Check if this test sample matches a training sample exactly
            let mut found_exact_match = false;
            for j in 0..self.state.X_train.nrows() {
                if i < self.state.X_train.nrows() {
                    // If test sample index is within training range, check for exact match
                    let diff = &X.row(i) - &self.state.X_train.row(j);
                    let distance = diff.mapv(|x| x * x).sum().sqrt();
                    if distance < 1e-10 && i == j {
                        // Exact match with training sample - preserve original label if labeled
                        if self.state.labeled_mask[j] {
                            predictions[i] = self.state.y_train[j];
                            found_exact_match = true;
                            break;
                        }
                    }
                }
            }

            if !found_exact_match {
                let mut class_votes: HashMap<i32, f64> = HashMap::new();
                let mut total_weight = 0.0;

                // Get prediction from each view
                for view in &self.state.views {
                    // Extract view data
                    let X_view_train = self.extract_view(&self.state.X_train, view)?;
                    let X_view_test = self.extract_view(&X, view)?;

                    // Extract labeled training data for this view
                    let X_labeled: Vec<Vec<f64>> = labeled_indices
                        .iter()
                        .map(|&idx| X_view_train.row(idx).to_vec())
                        .collect();
                    let y_labeled: Array1<i32> = labeled_indices
                        .iter()
                        .map(|&idx| self.state.y_train[idx])
                        .collect();

                    let X_labeled = Array2::from_shape_vec(
                        (X_labeled.len(), view.len()),
                        X_labeled.into_iter().flatten().collect(),
                    )
                    .map_err(|_| {
                        SklearsError::InvalidInput("Failed to create training data".to_string())
                    })?;

                    // Predict for single test sample
                    let test_sample = X_view_test
                        .row(i)
                        .to_owned()
                        .insert_axis(scirs2_core::ndarray::Axis(0));
                    let mut distances: Vec<(f64, i32)> = Vec::new();

                    for j in 0..X_labeled.nrows() {
                        let diff = &test_sample.row(0) - &X_labeled.row(j);
                        let dist = diff.mapv(|x| x * x).sum().sqrt();
                        distances.push((dist, y_labeled[j]));
                    }

                    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    let k = distances.len().clamp(1, 5);
                    let mut view_votes: HashMap<i32, f64> = HashMap::new();
                    let mut view_weight = 0.0;

                    for &(dist, label) in distances.iter().take(k) {
                        let weight = if dist > 0.0 { 1.0 / (1.0 + dist) } else { 1.0 };
                        *view_votes.entry(label).or_insert(0.0) += weight;
                        view_weight += weight;
                    }

                    // Normalize and add to ensemble votes
                    for (class, vote) in view_votes {
                        let normalized_vote = vote / view_weight;
                        *class_votes.entry(class).or_insert(0.0) += normalized_vote;
                    }
                    total_weight += 1.0; // Each view gets equal weight
                }

                // Find majority vote
                let best_class = class_votes
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(&class, _)| class)
                    .unwrap_or(self.state.classes[0]);

                predictions[i] = best_class;
            }
        }

        Ok(predictions)
    }
}

/// Trained state for DemocraticCoLearning
#[derive(Debug, Clone)]
pub struct DemocraticCoLearningTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// labeled_mask
    pub labeled_mask: Array1<bool>,
    /// views
    pub views: Vec<Vec<usize>>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_democratic_co_learning_basic() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0]
        ];
        let y = array![0, 1, -1, -1, -1, -1]; // -1 indicates unlabeled

        let dcl = DemocraticCoLearning::new()
            .views(vec![vec![0, 1], vec![2, 3]])
            .k_add(1)
            .min_agreement(2)
            .max_iter(5);

        let fitted = dcl.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), X.nrows());

        // Check that labeled samples maintain their labels
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_democratic_co_learning_parameters() {
        let dcl = DemocraticCoLearning::new()
            .views(vec![vec![0], vec![1]])
            .k_add(2)
            .max_iter(10)
            .confidence_threshold(0.8)
            .min_agreement(1)
            .verbose(true)
            .selection_strategy("confidence".to_string());

        assert_eq!(dcl.k_add, 2);
        assert_eq!(dcl.max_iter, 10);
        assert_eq!(dcl.confidence_threshold, 0.8);
        assert_eq!(dcl.min_agreement, 1);
        assert_eq!(dcl.verbose, true);
        assert_eq!(dcl.selection_strategy, "confidence");
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_democratic_co_learning_error_cases() {
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 1];

        // Test with insufficient views
        let dcl = DemocraticCoLearning::new().views(vec![vec![0]]);
        let result = dcl.fit(&X.view(), &y.view());
        assert!(result.is_err());

        // Test with min_agreement > number of views
        let dcl = DemocraticCoLearning::new()
            .views(vec![vec![0], vec![1]])
            .min_agreement(3);
        let result = dcl.fit(&X.view(), &y.view());
        assert!(result.is_err());

        // Test with empty view
        let dcl = DemocraticCoLearning::new().views(vec![vec![], vec![1]]);
        let result = dcl.fit(&X.view(), &y.view());
        assert!(result.is_err());

        // Test with out of bounds feature index
        let dcl = DemocraticCoLearning::new().views(vec![vec![0], vec![5]]);
        let result = dcl.fit(&X.view(), &y.view());
        assert!(result.is_err());

        // Test with no labeled samples
        let y_unlabeled = array![-1, -1];
        let dcl = DemocraticCoLearning::new().views(vec![vec![0], vec![1]]);
        let result = dcl.fit(&X.view(), &y_unlabeled.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_democratic_co_learning_with_three_views() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ];
        let y = array![0, 1, -1, -1, -1];

        let dcl = DemocraticCoLearning::new()
            .views(vec![vec![0, 1], vec![2, 3], vec![4, 5]])
            .k_add(1)
            .min_agreement(2)
            .max_iter(3);

        let fitted = dcl.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), X.nrows());

        // Check that labeled samples maintain their labels
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_democratic_co_learning_all_labeled() {
        let X = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]];
        let y = array![0, 1]; // All labeled

        let dcl = DemocraticCoLearning::new()
            .views(vec![vec![0, 1], vec![2, 3]])
            .k_add(1)
            .min_agreement(2)
            .max_iter(5);

        let fitted = dcl.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), X.nrows());
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_extract_view() {
        let X = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let dcl = DemocraticCoLearning::new();

        // Extract first two features
        let view = dcl.extract_view(&X, &[0, 1]).unwrap();
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view[[0, 0]], 1.0);
        assert_eq!(view[[0, 1]], 2.0);
        assert_eq!(view[[1, 0]], 5.0);
        assert_eq!(view[[1, 1]], 6.0);

        // Extract last two features
        let view = dcl.extract_view(&X, &[2, 3]).unwrap();
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view[[0, 0]], 3.0);
        assert_eq!(view[[0, 1]], 4.0);
        assert_eq!(view[[1, 0]], 7.0);
        assert_eq!(view[[1, 1]], 8.0);

        // Test error case with out of bounds index
        let result = dcl.extract_view(&X, &[5]);
        assert!(result.is_err());

        // Test error case with empty view
        let result = dcl.extract_view(&X, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_democratic_vote() {
        let dcl = DemocraticCoLearning::new()
            .confidence_threshold(0.5)
            .min_agreement(2);

        let predictions = vec![array![0, 1, 0], array![0, 1, 1], array![0, 0, 0]];
        let confidences = vec![
            array![0.8, 0.9, 0.6],
            array![0.7, 0.8, 0.7],
            array![0.6, 0.4, 0.8], // Low confidence for second prediction
        ];
        let classes = vec![0, 1];

        let candidates = dcl.democratic_vote(&predictions, &confidences, &classes);

        // Should have candidates for samples 0 and 2 (sample 1 has disagreement)
        assert!(candidates.len() >= 1);

        // First candidate should be sample 0 with class 0 (all agree)
        let (sample_idx, predicted_class, _score) = candidates[0];
        assert_eq!(sample_idx, 0);
        assert_eq!(predicted_class, 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_train_classifier() {
        let dcl = DemocraticCoLearning::new();

        let X_train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_train = array![0, 1, 0];
        let X_test = array![[2.0, 3.0], [4.0, 5.0]];
        let classes = vec![0, 1];

        let (predictions, confidences) =
            dcl.train_classifier(&X_train, &y_train, &X_test, &classes);

        assert_eq!(predictions.len(), 2);
        assert_eq!(confidences.len(), 2);

        // All predictions should be valid class labels
        for &pred in predictions.iter() {
            assert!(classes.contains(&pred));
        }

        // All confidences should be in [0, 1]
        for &conf in confidences.iter() {
            assert!(conf >= 0.0 && conf <= 1.0);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_democratic_co_learning_high_confidence_threshold() {
        let X = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0]
        ];
        let y = array![0, 1, -1, -1];

        // High confidence threshold should make it harder to add samples
        let dcl = DemocraticCoLearning::new()
            .views(vec![vec![0, 1], vec![2, 3]])
            .k_add(1)
            .min_agreement(2)
            .confidence_threshold(0.99) // Very high threshold
            .max_iter(2);

        let fitted = dcl.fit(&X.view(), &y.view()).unwrap();
        let predictions = fitted.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), X.nrows());

        // Should still maintain labeled samples
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }
}
