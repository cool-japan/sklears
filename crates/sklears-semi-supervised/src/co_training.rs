//! Co-Training implementation for semi-supervised learning with multiple views

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};

/// Co-Training classifier for semi-supervised learning with multiple views
///
/// Co-training uses two different feature sets (views) to train two classifiers.
/// Each classifier labels unlabeled examples for the other classifier to use.
///
/// # Parameters
///
/// * `view1_features` - Indices of features for view 1
/// * `view2_features` - Indices of features for view 2
/// * `p` - Number of positive examples to add per iteration
/// * `n` - Number of negative examples to add per iteration
/// * `max_iter` - Maximum number of iterations
/// * `verbose` - Whether to print progress information
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::CoTraining;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0],
///                [3.0, 4.0, 5.0, 6.0], [4.0, 5.0, 6.0, 7.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let ct = CoTraining::new()
///     .view1_features(vec![0, 1])
///     .view2_features(vec![2, 3])
///     .p(1)
///     .n(1)
///     .max_iter(10);
/// let fitted = ct.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CoTraining<S = Untrained> {
    state: S,
    view1_features: Vec<usize>,
    view2_features: Vec<usize>,
    p: usize,
    n: usize,
    max_iter: usize,
    verbose: bool,
    confidence_threshold: f64,
}

impl CoTraining<Untrained> {
    /// Create a new CoTraining instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            view1_features: Vec::new(),
            view2_features: Vec::new(),
            p: 1,
            n: 1,
            max_iter: 30,
            verbose: false,
            confidence_threshold: 0.5,
        }
    }

    /// Set the features for view 1
    pub fn view1_features(mut self, features: Vec<usize>) -> Self {
        self.view1_features = features;
        self
    }

    /// Set the features for view 2
    pub fn view2_features(mut self, features: Vec<usize>) -> Self {
        self.view2_features = features;
        self
    }

    /// Set the number of positive examples to add per iteration
    pub fn p(mut self, p: usize) -> Self {
        self.p = p;
        self
    }

    /// Set the number of negative examples to add per iteration
    pub fn n(mut self, n: usize) -> Self {
        self.n = n;
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

    /// Set confidence threshold
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
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

    fn simple_classifier_predict(
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
            // Simple k-NN classifier
            let mut distances: Vec<(f64, i32)> = Vec::new();
            for j in 0..X_train.nrows() {
                let diff = &X_test.row(i) - &X_train.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((dist, y_train[j]));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Use k=5 nearest neighbors
            let k = distances.len().clamp(1, 5);
            let mut class_votes: HashMap<i32, f64> = HashMap::new();
            let mut total_weight = 0.0;

            for &(dist, label) in distances.iter().take(k) {
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
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(&class, &conf)| (class, conf))
                .unwrap_or((classes[0], 0.0));

            predictions[i] = best_class;
            confidences[i] = best_confidence;
        }

        (predictions, confidences)
    }
}

impl Default for CoTraining<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CoTraining<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for CoTraining<Untrained> {
    type Fitted = CoTraining<CoTrainingTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let mut y = y.to_owned();

        // Validate views
        if self.view1_features.is_empty() || self.view2_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Both views must have at least one feature".to_string(),
            ));
        }

        // Check for overlapping features (warning, not error)
        let overlap: HashSet<_> = self
            .view1_features
            .iter()
            .filter(|f| self.view2_features.contains(f))
            .collect();
        if !overlap.is_empty() && self.verbose {
            println!("Warning: Views have overlapping features: {:?}", overlap);
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
        if classes.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Co-training currently supports binary classification only".to_string(),
            ));
        }

        // Extract views
        let X_view1 = self.extract_view(&X, &self.view1_features)?;
        let X_view2 = self.extract_view(&X, &self.view2_features)?;

        // Co-training iterations
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

            // Extract labeled data for both views
            let X1_labeled = labeled_indices
                .iter()
                .map(|&i| X_view1.row(i).to_owned())
                .collect::<Vec<_>>();
            let X2_labeled = labeled_indices
                .iter()
                .map(|&i| X_view2.row(i).to_owned())
                .collect::<Vec<_>>();

            let y_labeled: Array1<i32> = labeled_indices.iter().map(|&i| y[i]).collect();

            let X1_labeled = Array2::from_shape_vec(
                (X1_labeled.len(), X_view1.ncols()),
                X1_labeled.into_iter().flatten().collect(),
            )
            .map_err(|_| {
                SklearsError::InvalidInput("Failed to create view1 training data".to_string())
            })?;

            let X2_labeled = Array2::from_shape_vec(
                (X2_labeled.len(), X_view2.ncols()),
                X2_labeled.into_iter().flatten().collect(),
            )
            .map_err(|_| {
                SklearsError::InvalidInput("Failed to create view2 training data".to_string())
            })?;

            // Extract unlabeled data for both views
            let X1_unlabeled = unlabeled_indices
                .iter()
                .map(|&i| X_view1.row(i).to_owned())
                .collect::<Vec<_>>();
            let X2_unlabeled = unlabeled_indices
                .iter()
                .map(|&i| X_view2.row(i).to_owned())
                .collect::<Vec<_>>();

            let X1_unlabeled = Array2::from_shape_vec(
                (X1_unlabeled.len(), X_view1.ncols()),
                X1_unlabeled.into_iter().flatten().collect(),
            )
            .map_err(|_| {
                SklearsError::InvalidInput("Failed to create view1 unlabeled data".to_string())
            })?;

            let X2_unlabeled = Array2::from_shape_vec(
                (X2_unlabeled.len(), X_view2.ncols()),
                X2_unlabeled.into_iter().flatten().collect(),
            )
            .map_err(|_| {
                SklearsError::InvalidInput("Failed to create view2 unlabeled data".to_string())
            })?;

            // Train classifier on view1, predict on view2's unlabeled data
            let (pred1, conf1) =
                self.simple_classifier_predict(&X1_labeled, &y_labeled, &X2_unlabeled, &classes);

            // Train classifier on view2, predict on view1's unlabeled data
            let (pred2, conf2) =
                self.simple_classifier_predict(&X2_labeled, &y_labeled, &X1_unlabeled, &classes);

            // Select most confident predictions for each class
            let mut added_any = false;

            for &target_class in &classes {
                // Find confident predictions from classifier 1 for target class
                let mut candidates1: Vec<(usize, f64)> = pred1
                    .iter()
                    .zip(conf1.iter())
                    .enumerate()
                    .filter(|(_, (&pred, &conf))| {
                        pred == target_class && conf >= self.confidence_threshold
                    })
                    .map(|(i, (_, &conf))| (i, conf))
                    .collect();

                candidates1
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let add_count = if target_class == classes[0] {
                    self.p
                } else {
                    self.n
                };
                for (candidate_idx, _) in candidates1.into_iter().take(add_count) {
                    let original_idx = unlabeled_indices[candidate_idx];
                    y[original_idx] = target_class;
                    labeled_mask[original_idx] = true;
                    added_any = true;
                }

                // Find confident predictions from classifier 2 for target class
                let mut candidates2: Vec<(usize, f64)> = pred2
                    .iter()
                    .zip(conf2.iter())
                    .enumerate()
                    .filter(|(_, (&pred, &conf))| {
                        pred == target_class && conf >= self.confidence_threshold
                    })
                    .map(|(i, (_, &conf))| (i, conf))
                    .collect();

                candidates2
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (candidate_idx, _) in candidates2.into_iter().take(add_count) {
                    let original_idx = unlabeled_indices[candidate_idx];
                    if !labeled_mask[original_idx] {
                        // Don't double-label
                        y[original_idx] = target_class;
                        labeled_mask[original_idx] = true;
                        added_any = true;
                    }
                }
            }

            if !added_any {
                if self.verbose {
                    println!("Iteration {}: No confident predictions, stopping", iter + 1);
                }
                break;
            }

            if self.verbose {
                let n_labeled = labeled_mask.iter().filter(|&&x| x).count();
                println!("Iteration {}: {} labeled samples", iter + 1, n_labeled);
            }
        }

        Ok(CoTraining {
            state: CoTrainingTrained {
                X_train: X.clone(),
                y_train: y,
                classes: Array1::from(classes),
                labeled_mask,
                view1_features: self.view1_features.clone(),
                view2_features: self.view2_features.clone(),
            },
            view1_features: self.view1_features,
            view2_features: self.view2_features,
            p: self.p,
            n: self.n,
            max_iter: self.max_iter,
            verbose: self.verbose,
            confidence_threshold: self.confidence_threshold,
        })
    }
}

impl CoTraining<CoTrainingTrained> {
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

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for CoTraining<CoTrainingTrained> {
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

        // Extract views for training data
        let X1_train = self.extract_view(&self.state.X_train, &self.state.view1_features)?;
        let X1_labeled = labeled_indices
            .iter()
            .map(|&i| X1_train.row(i).to_owned())
            .collect::<Vec<_>>();
        let X1_labeled = Array2::from_shape_vec(
            (X1_labeled.len(), X1_train.ncols()),
            X1_labeled.into_iter().flatten().collect(),
        )
        .map_err(|_| {
            SklearsError::InvalidInput("Failed to create view1 training data".to_string())
        })?;

        let y_labeled: Array1<i32> = labeled_indices
            .iter()
            .map(|&i| self.state.y_train[i])
            .collect();

        // Extract view for test data (use combined features for prediction)
        let mut all_features: Vec<usize> = self.state.view1_features.clone();
        all_features.extend(&self.state.view2_features);
        all_features.sort();
        all_features.dedup();

        let X_test_combined = self.extract_view(&X, &all_features)?;

        // Use simple k-NN for prediction
        for i in 0..n_test {
            let mut min_dist = f64::INFINITY;
            let mut best_label = 0;

            for (j, &labeled_idx) in labeled_indices.iter().enumerate() {
                // Combine features from both views for distance calculation
                let train_combined = self.extract_view(&self.state.X_train, &all_features)?;
                let diff = &X_test_combined.row(i) - &train_combined.row(labeled_idx);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    best_label = y_labeled[j];
                }
            }

            predictions[i] = best_label;
        }

        Ok(predictions)
    }
}

/// Trained state for CoTraining
#[derive(Debug, Clone)]
pub struct CoTrainingTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// labeled_mask
    pub labeled_mask: Array1<bool>,
    /// view1_features
    pub view1_features: Vec<usize>,
    /// view2_features
    pub view2_features: Vec<usize>,
}

/// Multi-View Co-Training classifier for semi-supervised learning with multiple views
///
/// Multi-view co-training extends the traditional co-training algorithm to work with
/// more than two views. Each view trains a classifier that can label examples for
/// other views, creating a collaborative learning process.
///
/// # Parameters
///
/// * `views` - Vector of feature indices for each view
/// * `k_add` - Number of examples to add per iteration per view
/// * `max_iter` - Maximum number of iterations
/// * `confidence_threshold` - Minimum confidence for pseudo-labeling
/// * `selection_strategy` - Strategy for selecting examples ("confidence" or "diversity")
/// * `verbose` - Whether to print progress information
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_semi_supervised::MultiViewCoTraining;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
///                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let mvct = MultiViewCoTraining::new()
///     .views(vec![vec![0, 1], vec![2, 3], vec![4, 5]])
///     .k_add(1)
///     .confidence_threshold(0.6)
///     .max_iter(10);
/// let fitted = mvct.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiViewCoTraining<S = Untrained> {
    state: S,
    views: Vec<Vec<usize>>,
    k_add: usize,
    max_iter: usize,
    confidence_threshold: f64,
    selection_strategy: String,
    verbose: bool,
}

impl MultiViewCoTraining<Untrained> {
    /// Create a new MultiViewCoTraining instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            views: Vec::new(),
            k_add: 1,
            max_iter: 30,
            confidence_threshold: 0.6,
            selection_strategy: "confidence".to_string(),
            verbose: false,
        }
    }

    /// Set the views (feature indices for each view)
    pub fn views(mut self, views: Vec<Vec<usize>>) -> Self {
        self.views = views;
        self
    }

    /// Set the number of examples to add per iteration per view
    pub fn k_add(mut self, k_add: usize) -> Self {
        self.k_add = k_add;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the confidence threshold for pseudo-labeling
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the selection strategy for pseudo-labeling
    pub fn selection_strategy(mut self, strategy: String) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
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

    fn train_view_classifier(
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
            // k-NN classifier with adaptive k
            let mut distances: Vec<(f64, i32)> = Vec::new();
            for j in 0..X_train.nrows() {
                let diff = &X_test.row(i) - &X_train.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((dist, y_train[j]));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let k = distances.len().clamp(3, 7);
            let mut class_votes: HashMap<i32, f64> = HashMap::new();
            let mut total_weight = 0.0;

            for &(dist, label) in distances.iter().take(k) {
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
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(&class, &conf)| (class, conf))
                .unwrap_or((classes[0], 0.0));

            predictions[i] = best_class;
            confidences[i] = best_confidence;
        }

        (predictions, confidences)
    }

    fn select_confident_samples(
        &self,
        predictions: &Array1<i32>,
        confidences: &Array1<f64>,
        classes: &[i32],
    ) -> Vec<(usize, i32, f64)> {
        let mut candidates = Vec::new();

        for i in 0..predictions.len() {
            if confidences[i] >= self.confidence_threshold {
                candidates.push((i, predictions[i], confidences[i]));
            }
        }

        // Sort by confidence (descending)
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Apply selection strategy
        match self.selection_strategy.as_str() {
            "confidence" => {
                // Take top k_add per class
                let mut selected = Vec::new();
                for &class in classes {
                    let class_candidates: Vec<_> = candidates
                        .iter()
                        .filter(|(_, c, _)| *c == class)
                        .take(self.k_add)
                        .cloned()
                        .collect();
                    selected.extend(class_candidates);
                }
                selected
            }
            "diversity" => {
                // Take top k_add overall with some diversity
                candidates
                    .into_iter()
                    .take(self.k_add * classes.len())
                    .collect()
            }
            _ => candidates
                .into_iter()
                .take(self.k_add * classes.len())
                .collect(),
        }
    }
}

impl Default for MultiViewCoTraining<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiViewCoTraining<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for MultiViewCoTraining<Untrained> {
    type Fitted = MultiViewCoTraining<MultiViewCoTrainingTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let mut y = y.to_owned();

        if self.views.len() < 3 {
            return Err(SklearsError::InvalidInput(
                "Multi-view co-training requires at least 3 views".to_string(),
            ));
        }

        // Validate views
        for (i, view) in self.views.iter().enumerate() {
            if view.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "View {} has no features",
                    i
                )));
            }
            for &feature_idx in view {
                if feature_idx >= X.ncols() {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature index {} out of bounds in view {}",
                        feature_idx, i
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

        // Multi-view co-training iterations
        for iter in 0..self.max_iter {
            let mut any_labels_added = false;

            // For each view, train classifier and get predictions for unlabeled data
            for view_idx in 0..self.views.len() {
                let view = &self.views[view_idx];

                // Extract labeled samples for this view
                let labeled_indices: Vec<usize> = labeled_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &is_labeled)| is_labeled)
                    .map(|(i, _)| i)
                    .collect();

                if labeled_indices.is_empty() {
                    continue;
                }

                let X_view = self.extract_view(&X, view)?;

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

                // Get unlabeled samples
                let unlabeled_indices: Vec<usize> = labeled_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &is_labeled)| !is_labeled)
                    .map(|(i, _)| i)
                    .collect();

                if unlabeled_indices.is_empty() {
                    continue; // All samples are labeled
                }

                // Train on other views and predict on this view's unlabeled data
                let mut all_predictions = Vec::new();
                let mut all_confidences = Vec::new();

                for other_view_idx in 0..self.views.len() {
                    if other_view_idx == view_idx {
                        continue; // Skip self-view
                    }

                    let other_view = &self.views[other_view_idx];
                    let X_other_view = self.extract_view(&X, other_view)?;

                    // Extract labeled data for other view
                    let X_other_labeled: Vec<Vec<f64>> = labeled_indices
                        .iter()
                        .map(|&i| X_other_view.row(i).to_vec())
                        .collect();

                    let X_other_labeled = Array2::from_shape_vec(
                        (X_other_labeled.len(), other_view.len()),
                        X_other_labeled.into_iter().flatten().collect(),
                    )
                    .map_err(|_| {
                        SklearsError::InvalidInput(
                            "Failed to create other view training data".to_string(),
                        )
                    })?;

                    // Extract unlabeled data for current view
                    let X_current_unlabeled: Vec<Vec<f64>> = unlabeled_indices
                        .iter()
                        .map(|&i| X_view.row(i).to_vec())
                        .collect();

                    let X_current_unlabeled = Array2::from_shape_vec(
                        (X_current_unlabeled.len(), view.len()),
                        X_current_unlabeled.into_iter().flatten().collect(),
                    )
                    .map_err(|_| {
                        SklearsError::InvalidInput(
                            "Failed to create current view unlabeled data".to_string(),
                        )
                    })?;

                    // Train classifier on other view, predict on current view
                    let (pred, conf) = self.train_view_classifier(
                        &X_other_labeled,
                        &y_labeled,
                        &X_current_unlabeled,
                        &classes,
                    );

                    all_predictions.push(pred);
                    all_confidences.push(conf);
                }

                if all_predictions.is_empty() {
                    continue;
                }

                // Aggregate predictions from all other views (ensemble voting)
                let n_unlabeled = unlabeled_indices.len();
                let mut final_predictions = Array1::zeros(n_unlabeled);
                let mut final_confidences = Array1::zeros(n_unlabeled);

                for i in 0..n_unlabeled {
                    let mut class_votes: HashMap<i32, f64> = HashMap::new();
                    let mut total_confidence = 0.0;

                    for (pred, conf) in all_predictions.iter().zip(all_confidences.iter()) {
                        let confidence = conf[i];
                        let prediction = pred[i];

                        *class_votes.entry(prediction).or_insert(0.0) += confidence;
                        total_confidence += confidence;
                    }

                    if total_confidence > 0.0 {
                        for (_, vote) in class_votes.iter_mut() {
                            *vote /= total_confidence;
                        }

                        let (best_class, best_confidence) = class_votes
                            .iter()
                            .max_by(|a, b| {
                                a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(&class, &conf)| (class, conf))
                            .unwrap_or((classes[0], 0.0));

                        final_predictions[i] = best_class;
                        final_confidences[i] = best_confidence;
                    }
                }

                // Select confident samples to add
                let selected =
                    self.select_confident_samples(&final_predictions, &final_confidences, &classes);

                // Add selected pseudo-labels
                for (unlabeled_idx, label, _confidence) in selected {
                    if unlabeled_idx < unlabeled_indices.len() {
                        let sample_idx = unlabeled_indices[unlabeled_idx];
                        y[sample_idx] = label;
                        labeled_mask[sample_idx] = true;
                        any_labels_added = true;
                    }
                }
            }

            if !any_labels_added {
                if self.verbose {
                    println!("Multi-view co-training converged at iteration {}", iter + 1);
                }
                break;
            }

            if self.verbose {
                let n_labeled = labeled_mask.iter().filter(|&&x| x).count();
                println!("Iteration {}: {} labeled samples", iter + 1, n_labeled);
            }
        }

        Ok(MultiViewCoTraining {
            state: MultiViewCoTrainingTrained {
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
            selection_strategy: self.selection_strategy,
            verbose: self.verbose,
        })
    }
}

impl MultiViewCoTraining<MultiViewCoTrainingTrained> {
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

    fn train_view_classifier(
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
            // k-NN classifier with adaptive k
            let mut distances: Vec<(f64, i32)> = Vec::new();
            for j in 0..X_train.nrows() {
                let diff = &X_test.row(i) - &X_train.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances.push((dist, y_train[j]));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let k = distances.len().clamp(3, 7);
            let mut class_votes: HashMap<i32, f64> = HashMap::new();
            let mut total_weight = 0.0;

            for &(dist, label) in distances.iter().take(k) {
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
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(&class, &conf)| (class, conf))
                .unwrap_or((classes[0], 0.0));

            predictions[i] = best_class;
            confidences[i] = best_confidence;
        }

        (predictions, confidences)
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for MultiViewCoTraining<MultiViewCoTrainingTrained>
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
                    let diff = &X.row(i) - &self.state.X_train.row(j);
                    let distance = diff.mapv(|x| x * x).sum().sqrt();
                    if distance < 1e-10 && i == j && self.state.labeled_mask[j] {
                        predictions[i] = self.state.y_train[j];
                        found_exact_match = true;
                        break;
                    }
                }
            }

            if !found_exact_match {
                let mut class_votes: HashMap<i32, f64> = HashMap::new();
                let mut total_weight = 0.0;

                // Get prediction from each view
                for view in &self.state.views {
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
                    let (view_predictions, view_confidences) = self.train_view_classifier(
                        &X_labeled,
                        &y_labeled,
                        &test_sample,
                        &self.state.classes.to_vec(),
                    );

                    let prediction = view_predictions[0];
                    let confidence = view_confidences[0];

                    *class_votes.entry(prediction).or_insert(0.0) += confidence;
                    total_weight += confidence;
                }

                // Find majority vote
                let best_class = if total_weight > 0.0 {
                    class_votes
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(&class, _)| class)
                        .unwrap_or(self.state.classes[0])
                } else {
                    self.state.classes[0]
                };

                predictions[i] = best_class;
            }
        }

        Ok(predictions)
    }
}

/// Trained state for MultiViewCoTraining
#[derive(Debug, Clone)]
pub struct MultiViewCoTrainingTrained {
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
