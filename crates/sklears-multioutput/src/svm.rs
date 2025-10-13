//! Support Vector Machine algorithms for multi-output learning

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// MLTSVM (Multi-Label Twin SVM)
///
/// MLTSVM is a multi-label classification method that extends Twin SVM to handle
/// multiple labels. Twin SVM finds two non-parallel hyperplanes for binary classification,
/// which often leads to faster training than standard SVM. MLTSVM applies this approach
/// to each label independently in a binary relevance fashion.
///
/// # Examples
///
/// ```
/// use sklears_core::traits::{Predict, Fit};
/// use sklears_multioutput::MLTSVM;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // Multi-label binary
///
/// let mltsvm = MLTSVM::new().c1(1.0).c2(1.0);
/// let trained_mltsvm = mltsvm.fit(&X.view(), &y).unwrap();
/// let predictions = trained_mltsvm.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MLTSVM<S = Untrained> {
    state: S,
    c1: Float,       // Regularization parameter for first hyperplane
    c2: Float,       // Regularization parameter for second hyperplane
    epsilon: Float,  // Tolerance for convergence
    max_iter: usize, // Maximum iterations
}

/// Trained state for MLTSVM
#[derive(Debug, Clone)]
pub struct MLTSVMTrained {
    models: Vec<TwinSVMModel>, // One model per label
    n_labels: usize,
    feature_means: Array1<Float>,
    feature_stds: Array1<Float>,
}

/// Twin SVM model for a single label
#[derive(Debug, Clone)]
pub struct TwinSVMModel {
    w1: Array1<Float>, // Weight vector for positive hyperplane
    b1: Float,         // Bias for positive hyperplane
    w2: Array1<Float>, // Weight vector for negative hyperplane
    b2: Float,         // Bias for negative hyperplane
}

impl MLTSVM<Untrained> {
    /// Create a new MLTSVM instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            c1: 1.0,
            c2: 1.0,
            epsilon: 1e-3,
            max_iter: 1000,
        }
    }

    /// Set C1 parameter
    pub fn c1(mut self, c1: Float) -> Self {
        self.c1 = c1;
        self
    }

    /// Set C2 parameter
    pub fn c2(mut self, c2: Float) -> Self {
        self.c2 = c2;
        self
    }

    /// Set epsilon parameter
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl Default for MLTSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MLTSVM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for MLTSVM<Untrained> {
    type Fitted = MLTSVM<MLTSVMTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let (y_samples, n_labels) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for SVM training".to_string(),
            ));
        }

        // Validate that all labels are binary (0 or 1)
        for sample_idx in 0..y_samples {
            for label_idx in 0..n_labels {
                let value = y[[sample_idx, label_idx]];
                if value != 0 && value != 1 {
                    return Err(SklearsError::InvalidInput(format!(
                        "All label values must be 0 or 1, found: {}",
                        value
                    )));
                }
            }
        }

        // Compute feature statistics for normalization
        let feature_means = x.mean_axis(Axis(0)).unwrap();
        let feature_stds = x.mapv(|val| val * val).mean_axis(Axis(0)).unwrap()
            - &feature_means.mapv(|mean| mean * mean);
        let feature_stds = feature_stds.mapv(|var| (var.max(1e-10)).sqrt());

        // Train Twin SVM for each label
        let mut models = Vec::new();
        for label_idx in 0..n_labels {
            let y_label = y.column(label_idx);
            let model = self.train_twin_svm(x, &y_label, &feature_means, &feature_stds)?;
            models.push(model);
        }

        Ok(MLTSVM {
            state: MLTSVMTrained {
                models,
                n_labels,
                feature_means,
                feature_stds,
            },
            c1: self.c1,
            c2: self.c2,
            epsilon: self.epsilon,
            max_iter: self.max_iter,
        })
    }
}

impl MLTSVM<Untrained> {
    fn train_twin_svm(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, i32>,
        feature_means: &Array1<Float>,
        feature_stds: &Array1<Float>,
    ) -> SklResult<TwinSVMModel> {
        let (n_samples, n_features) = x.dim();

        // Normalize features
        let mut x_normalized = x.to_owned();
        for (i, mut row) in x_normalized.rows_mut().into_iter().enumerate() {
            row -= feature_means;
            row /= feature_stds;
        }

        // Separate positive and negative samples
        let mut pos_samples = Vec::new();
        let mut neg_samples = Vec::new();

        for i in 0..n_samples {
            if y[i] == 1 {
                pos_samples.push(x_normalized.row(i).to_owned());
            } else {
                neg_samples.push(x_normalized.row(i).to_owned());
            }
        }

        if pos_samples.is_empty() || neg_samples.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Need both positive and negative samples for Twin SVM".to_string(),
            ));
        }

        // Convert to matrices
        let pos_matrix = Array2::from_shape_vec(
            (pos_samples.len(), n_features),
            pos_samples.into_iter().flatten().collect(),
        )
        .map_err(|_| SklearsError::InvalidInput("Failed to create positive matrix".to_string()))?;

        let neg_matrix = Array2::from_shape_vec(
            (neg_samples.len(), n_features),
            neg_samples.into_iter().flatten().collect(),
        )
        .map_err(|_| SklearsError::InvalidInput("Failed to create negative matrix".to_string()))?;

        // Train Twin SVM hyperplanes
        let (w1, b1) = self.solve_twin_svm_problem(&pos_matrix, &neg_matrix, self.c1)?;
        let (w2, b2) = self.solve_twin_svm_problem(&neg_matrix, &pos_matrix, self.c2)?;

        Ok(TwinSVMModel { w1, b1, w2, b2 })
    }

    fn solve_twin_svm_problem(
        &self,
        target_matrix: &Array2<Float>,
        other_matrix: &Array2<Float>,
        c: Float,
    ) -> SklResult<(Array1<Float>, Float)> {
        let n_target = target_matrix.nrows();
        let n_other = other_matrix.nrows();
        let n_features = target_matrix.ncols();

        // Initialize weights
        let mut w = Array1::<Float>::zeros(n_features + 1); // Include bias

        // Simple gradient descent solution
        let learning_rate = 0.01;

        for _iter in 0..self.max_iter {
            let mut gradient = Array1::<Float>::zeros(n_features + 1);

            // Compute gradient
            for i in 0..n_target {
                let x_aug = {
                    let mut x = Array1::ones(n_features + 1);
                    x.slice_mut(s![..n_features]).assign(&target_matrix.row(i));
                    x
                };
                let loss = x_aug.dot(&w);
                gradient += &(x_aug * loss);
            }

            for i in 0..n_other {
                let x_aug = {
                    let mut x = Array1::ones(n_features + 1);
                    x.slice_mut(s![..n_features]).assign(&other_matrix.row(i));
                    x
                };
                let margin = 1.0 - x_aug.dot(&w);
                if margin > 0.0 {
                    gradient -= &(x_aug * c);
                }
            }

            // Check convergence before updating weights
            let gradient_norm = gradient.mapv(|x| x.abs()).sum();

            // Update weights
            w -= &(gradient * learning_rate);

            if gradient_norm < self.epsilon {
                break;
            }
        }

        let weights = w.slice(s![..n_features]).to_owned();
        let bias = w[n_features];

        Ok((weights, bias))
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>> for MLTSVM<MLTSVMTrained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = x.dim();
        let expected_features = self.state.feature_means.len();

        if n_features != expected_features {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features in X ({}) does not match training data ({})",
                n_features, expected_features
            )));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        // Normalize features
        let mut x_normalized = x.to_owned();
        for (i, mut row) in x_normalized.rows_mut().into_iter().enumerate() {
            row -= &self.state.feature_means;
            row /= &self.state.feature_stds;
        }

        for label_idx in 0..self.state.n_labels {
            let model = &self.state.models[label_idx];

            for sample_idx in 0..n_samples {
                let x_sample = x_normalized.row(sample_idx);

                // Compute distances to both hyperplanes
                let dist1 = (x_sample.dot(&model.w1) + model.b1).abs();
                let dist2 = (x_sample.dot(&model.w2) + model.b2).abs();

                // Predict based on closer hyperplane
                predictions[[sample_idx, label_idx]] = if dist1 < dist2 { 1 } else { 0 };
            }
        }

        Ok(predictions)
    }
}

impl MLTSVM<MLTSVMTrained> {
    /// Get the number of labels
    pub fn n_labels(&self) -> usize {
        self.state.n_labels
    }

    /// Get decision function values (distances to hyperplanes)
    pub fn decision_function(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, _n_features) = x.dim();
        let mut decision_values = Array2::<Float>::zeros((n_samples, self.state.n_labels));

        // Normalize features
        let mut x_normalized = x.to_owned();
        for (i, mut row) in x_normalized.rows_mut().into_iter().enumerate() {
            row -= &self.state.feature_means;
            row /= &self.state.feature_stds;
        }

        for label_idx in 0..self.state.n_labels {
            let model = &self.state.models[label_idx];

            for sample_idx in 0..n_samples {
                let x_sample = x_normalized.row(sample_idx);

                // Compute distances to both hyperplanes and use the difference
                let dist1 = x_sample.dot(&model.w1) + model.b1;
                let dist2 = x_sample.dot(&model.w2) + model.b2;

                // Decision function is the difference (positive means class 1)
                decision_values[[sample_idx, label_idx]] = dist1 - dist2;
            }
        }

        Ok(decision_values)
    }
}

/// RankSVM for Multi-Label Classification
///
/// RankSVM is a ranking-based approach for multi-label classification that optimizes
/// ranking loss functions. It learns to rank labels by their relevance scores and
/// can handle both label ranking and threshold selection for multi-label prediction.
#[derive(Debug, Clone)]
pub struct RankSVM<S = Untrained> {
    state: S,
    c: Float,                              // Regularization parameter
    epsilon: Float,                        // Tolerance for convergence
    max_iter: usize,                       // Maximum iterations
    threshold_strategy: ThresholdStrategy, // How to determine prediction thresholds
}

/// Threshold strategy for RankSVM
#[derive(Debug, Clone)]
pub enum ThresholdStrategy {
    /// Use fixed threshold for all labels
    Fixed(Float),
    /// Optimize threshold to maximize F1 score for each label
    OptimizeF1,
    /// Use top-k labels (fixed number of labels per sample)
    TopK(usize),
}

/// Trained state for RankSVM
#[derive(Debug, Clone)]
pub struct RankSVMTrained {
    models: Vec<RankingSVMModel>, // One model per label
    thresholds: Vec<Float>,       // Prediction thresholds for each label
    n_labels: usize,
    feature_means: Array1<Float>,
    feature_stds: Array1<Float>,
}

/// Single ranking SVM model for one label
#[derive(Debug, Clone)]
pub struct RankingSVMModel {
    weights: Array1<Float>,
    bias: Float,
}

impl RankSVM<Untrained> {
    /// Create a new RankSVM instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            c: 1.0,
            epsilon: 1e-3,
            max_iter: 1000,
            threshold_strategy: ThresholdStrategy::Fixed(0.0),
        }
    }

    /// Set regularization parameter
    pub fn c(mut self, c: Float) -> Self {
        self.c = c;
        self
    }

    /// Set convergence tolerance
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set threshold strategy
    pub fn threshold_strategy(mut self, strategy: ThresholdStrategy) -> Self {
        self.threshold_strategy = strategy;
        self
    }
}

impl Default for RankSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RankSVM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for RankSVM<Untrained> {
    type Fitted = RankSVM<RankSVMTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let (y_samples, n_labels) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Validate that all labels are binary (0 or 1)
        for sample_idx in 0..y_samples {
            for label_idx in 0..n_labels {
                let value = y[[sample_idx, label_idx]];
                if value != 0 && value != 1 {
                    return Err(SklearsError::InvalidInput(format!(
                        "All label values must be 0 or 1, found: {}",
                        value
                    )));
                }
            }
        }

        // Compute feature statistics
        let feature_means = x.mean_axis(Axis(0)).ok_or_else(|| {
            SklearsError::InvalidInput("Cannot compute feature means from input data".to_string())
        })?;

        let squared_means = x.mapv(|val| val * val).mean_axis(Axis(0)).ok_or_else(|| {
            SklearsError::InvalidInput("Cannot compute squared means from input data".to_string())
        })?;

        let feature_stds = squared_means - &feature_means.mapv(|mean| mean * mean);
        let feature_stds = feature_stds.mapv(|var| (var.max(1e-10)).sqrt());

        // Train ranking SVM for each label
        let mut models = Vec::new();
        for label_idx in 0..n_labels {
            let y_label = y.column(label_idx);
            let model = self.train_ranking_svm(x, &y_label, &feature_means, &feature_stds)?;
            models.push(model);
        }

        // Determine thresholds
        let thresholds = match &self.threshold_strategy {
            ThresholdStrategy::Fixed(threshold) => vec![*threshold; n_labels],
            ThresholdStrategy::OptimizeF1 => {
                self.optimize_f1_thresholds(x, y, &models, &feature_means, &feature_stds)?
            }
            ThresholdStrategy::TopK(_) => vec![0.0; n_labels], // No threshold needed for TopK
        };

        Ok(RankSVM {
            state: RankSVMTrained {
                models,
                thresholds,
                n_labels,
                feature_means,
                feature_stds,
            },
            c: self.c,
            epsilon: self.epsilon,
            max_iter: self.max_iter,
            threshold_strategy: self.threshold_strategy,
        })
    }
}

impl RankSVM<Untrained> {
    fn train_ranking_svm(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, i32>,
        feature_means: &Array1<Float>,
        feature_stds: &Array1<Float>,
    ) -> SklResult<RankingSVMModel> {
        let (n_samples, n_features) = x.dim();

        // Normalize features
        let mut x_normalized = x.to_owned();
        for (i, mut row) in x_normalized.rows_mut().into_iter().enumerate() {
            row -= feature_means;
            row /= feature_stds;
        }

        // Initialize weights and bias
        let mut weights = Array1::<Float>::zeros(n_features);
        let mut bias = 0.0;

        let learning_rate = 0.01;

        // Gradient descent optimization
        for _iter in 0..self.max_iter {
            let mut weight_gradient = Array1::<Float>::zeros(n_features);
            let mut bias_gradient = 0.0;

            // Create ranking pairs
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if y[i] > y[j] {
                        // i should be ranked higher than j
                        let x_i = x_normalized.row(i);
                        let x_j = x_normalized.row(j);
                        let x_diff = &x_i.to_owned() - &x_j.to_owned();

                        let score_diff = x_diff.dot(&weights) + bias;
                        let margin = 1.0 - score_diff;

                        if margin > 0.0 {
                            // Hinge loss gradient
                            weight_gradient -= &(x_diff * self.c);
                            bias_gradient -= self.c;
                        }
                    }
                }
            }

            // L2 regularization
            weight_gradient += &(&weights * 2.0);

            // Check convergence before updating parameters
            let gradient_norm = weight_gradient.mapv(|x| x.abs()).sum();

            // Update parameters
            weights -= &(weight_gradient * learning_rate);
            bias -= bias_gradient * learning_rate;

            if gradient_norm < self.epsilon {
                break;
            }
        }

        Ok(RankingSVMModel { weights, bias })
    }

    fn optimize_f1_thresholds(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &Array2<i32>,
        models: &[RankingSVMModel],
        feature_means: &Array1<Float>,
        feature_stds: &Array1<Float>,
    ) -> SklResult<Vec<Float>> {
        let mut thresholds = Vec::new();

        for label_idx in 0..y.ncols() {
            let y_true = y.column(label_idx);
            let scores = self.predict_scores_single_label(
                x,
                &models[label_idx],
                feature_means,
                feature_stds,
            )?;

            let threshold = self.find_optimal_f1_threshold(&y_true, &scores)?;
            thresholds.push(threshold);
        }

        Ok(thresholds)
    }

    fn predict_scores_single_label(
        &self,
        x: &ArrayView2<'_, Float>,
        model: &RankingSVMModel,
        feature_means: &Array1<Float>,
        feature_stds: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let (n_samples, _) = x.dim();
        let mut scores = Array1::<Float>::zeros(n_samples);

        for i in 0..n_samples {
            let x_sample = x.row(i);
            let x_normalized = (&x_sample.to_owned() - feature_means) / feature_stds;
            scores[i] = x_normalized.dot(&model.weights) + model.bias;
        }

        Ok(scores)
    }

    fn find_optimal_f1_threshold(
        &self,
        y_true: &ArrayView1<'_, i32>,
        scores: &Array1<Float>,
    ) -> SklResult<Float> {
        let mut score_threshold_pairs: Vec<(Float, i32)> = scores
            .iter()
            .zip(y_true.iter())
            .map(|(&score, &label)| (score, label))
            .collect();

        score_threshold_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut best_f1 = 0.0;
        let mut best_threshold = 0.0;

        // Try each unique score as a threshold
        for &(threshold, _) in &score_threshold_pairs {
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_count = 0;

            for (&score, &true_label) in scores.iter().zip(y_true.iter()) {
                let predicted = if score >= threshold { 1 } else { 0 };

                match (true_label, predicted) {
                    (1, 1) => tp += 1,
                    (0, 1) => fp += 1,
                    (1, 0) => fn_count += 1,
                    _ => {}
                }
            }

            let precision = if tp + fp > 0 {
                tp as Float / (tp + fp) as Float
            } else {
                0.0
            };
            let recall = if tp + fn_count > 0 {
                tp as Float / (tp + fn_count) as Float
            } else {
                0.0
            };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            if f1 > best_f1 {
                best_f1 = f1;
                best_threshold = threshold;
            }
        }

        Ok(best_threshold)
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>> for RankSVM<RankSVMTrained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = x.dim();
        let expected_features = self.state.feature_means.len();

        if n_features != expected_features {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features in X ({}) does not match training data ({})",
                n_features, expected_features
            )));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        match &self.threshold_strategy {
            ThresholdStrategy::TopK(k) => {
                // For TopK, rank all labels and select top k
                for sample_idx in 0..n_samples {
                    let mut scores = Vec::new();
                    for label_idx in 0..self.state.n_labels {
                        let x_sample = x.row(sample_idx);
                        let x_normalized = (&x_sample.to_owned() - &self.state.feature_means)
                            / &self.state.feature_stds;
                        let score = x_normalized.dot(&self.state.models[label_idx].weights)
                            + self.state.models[label_idx].bias;
                        scores.push((score, label_idx));
                    }

                    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                    for (i, &(_, label_idx)) in scores.iter().take(*k).enumerate() {
                        predictions[[sample_idx, label_idx]] = 1;
                    }
                }
            }
            _ => {
                // For fixed or optimized thresholds
                for label_idx in 0..self.state.n_labels {
                    let threshold = self.state.thresholds[label_idx];

                    for sample_idx in 0..n_samples {
                        let x_sample = x.row(sample_idx);
                        let x_normalized = (&x_sample.to_owned() - &self.state.feature_means)
                            / &self.state.feature_stds;
                        let score = x_normalized.dot(&self.state.models[label_idx].weights)
                            + self.state.models[label_idx].bias;

                        predictions[[sample_idx, label_idx]] =
                            if score >= threshold { 1 } else { 0 };
                    }
                }
            }
        }

        Ok(predictions)
    }
}

impl RankSVM<RankSVMTrained> {
    /// Get the number of labels
    pub fn n_labels(&self) -> usize {
        self.state.n_labels
    }

    /// Get decision function values (ranking scores)
    pub fn decision_function(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let expected_features = self.state.feature_means.len();

        if n_features != expected_features {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features in X ({}) does not match training data ({})",
                n_features, expected_features
            )));
        }

        let mut decision_values = Array2::<Float>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            for label_idx in 0..self.state.n_labels {
                let x_sample = x.row(sample_idx);
                let x_normalized =
                    (&x_sample.to_owned() - &self.state.feature_means) / &self.state.feature_stds;
                let score = x_normalized.dot(&self.state.models[label_idx].weights)
                    + self.state.models[label_idx].bias;
                decision_values[[sample_idx, label_idx]] = score;
            }
        }

        Ok(decision_values)
    }

    /// Get ranking predictions (label indices ordered by relevance)
    pub fn predict_ranking(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<usize>> {
        let (n_samples, n_features) = x.dim();
        let expected_features = self.state.feature_means.len();

        if n_features != expected_features {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features in X ({}) does not match training data ({})",
                n_features, expected_features
            )));
        }

        let mut rankings = Array2::<usize>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            let mut scores = Vec::new();
            for label_idx in 0..self.state.n_labels {
                let x_sample = x.row(sample_idx);
                let x_normalized =
                    (&x_sample.to_owned() - &self.state.feature_means) / &self.state.feature_stds;
                let score = x_normalized.dot(&self.state.models[label_idx].weights)
                    + self.state.models[label_idx].bias;
                scores.push((score, label_idx));
            }

            // Sort by score descending
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Assign rankings
            for (rank, &(_score, label_idx)) in scores.iter().enumerate() {
                rankings[[sample_idx, rank]] = label_idx;
            }
        }

        Ok(rankings)
    }

    /// Get the thresholds used for prediction
    pub fn thresholds(&self) -> &Vec<Float> {
        &self.state.thresholds
    }
}

/// Multi-Output Support Vector Machine
///
/// A multi-output support vector machine that handles multiple regression or
/// classification targets simultaneously by training separate SVM models for each output.
#[derive(Debug, Clone)]
pub struct MultiOutputSVM<S = Untrained> {
    state: S,
    kernel: SVMKernel,
    c: Float,
    epsilon: Float,
    gamma: Option<Float>,
}

/// SVM Kernel types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SVMKernel {
    /// Linear kernel: K(x, y) = x^T y
    Linear,
    /// Polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree
    Polynomial {
        degree: i32,
        gamma: Float,
        coef0: Float,
    },
    /// Radial Basis Function kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    Rbf { gamma: Float },
    /// Sigmoid kernel: K(x, y) = tanh(gamma * x^T y + coef0)
    Sigmoid { gamma: Float, coef0: Float },
}

/// Trained state for MultiOutputSVM
#[derive(Debug, Clone)]
pub struct MultiOutputSVMTrained {
    models: Vec<SVMModel>,
    n_outputs: usize,
    feature_means: Array1<Float>,
    feature_stds: Array1<Float>,
}

/// Single SVM model for one output
#[derive(Debug, Clone)]
pub struct SVMModel {
    support_vectors: Array2<Float>,
    support_coefficients: Array1<Float>,
    bias: Float,
    kernel: SVMKernel,
}

impl MultiOutputSVM<Untrained> {
    /// Create a new MultiOutputSVM instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: SVMKernel::Rbf { gamma: 1.0 },
            c: 1.0,
            epsilon: 1e-3,
            gamma: None,
        }
    }

    /// Set the kernel
    pub fn kernel(mut self, kernel: SVMKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set regularization parameter
    pub fn c(mut self, c: Float) -> Self {
        self.c = c;
        self
    }

    /// Set tolerance for stopping criterion
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set gamma parameter (will override kernel-specific gamma)
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.gamma = Some(gamma);
        self
    }
}

impl Default for MultiOutputSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiOutputSVM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView2<'_, Float>> for MultiOutputSVM<Untrained> {
    type Fitted = MultiOutputSVM<MultiOutputSVMTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView2<'_, Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let (y_samples, n_outputs) = y.dim();

        if n_samples != y_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Compute feature statistics
        let feature_means = x.mean_axis(Axis(0)).unwrap();
        let feature_stds = x.mapv(|val| val * val).mean_axis(Axis(0)).unwrap()
            - &feature_means.mapv(|mean| mean * mean);
        let feature_stds = feature_stds.mapv(|var| (var.max(1e-10)).sqrt());

        // Update kernel gamma if specified
        let kernel = if let Some(gamma) = self.gamma {
            match self.kernel {
                SVMKernel::Rbf { .. } => SVMKernel::Rbf { gamma },
                SVMKernel::Polynomial { degree, coef0, .. } => SVMKernel::Polynomial {
                    degree,
                    gamma,
                    coef0,
                },
                SVMKernel::Sigmoid { coef0, .. } => SVMKernel::Sigmoid { gamma, coef0 },
                other => other,
            }
        } else {
            self.kernel
        };

        // Train one SVM for each output
        let mut models = Vec::new();
        for output_idx in 0..n_outputs {
            let y_output = y.column(output_idx);
            let model =
                self.train_single_svm(x, &y_output, &feature_means, &feature_stds, kernel)?;
            models.push(model);
        }

        Ok(MultiOutputSVM {
            state: MultiOutputSVMTrained {
                models,
                n_outputs,
                feature_means,
                feature_stds,
            },
            kernel,
            c: self.c,
            epsilon: self.epsilon,
            gamma: self.gamma,
        })
    }
}

impl MultiOutputSVM<Untrained> {
    fn train_single_svm(
        &self,
        x: &ArrayView2<'_, Float>,
        y: &ArrayView1<'_, Float>,
        feature_means: &Array1<Float>,
        feature_stds: &Array1<Float>,
        kernel: SVMKernel,
    ) -> SklResult<SVMModel> {
        let (n_samples, n_features) = x.dim();

        // Normalize features
        let mut x_normalized = x.to_owned();
        for (i, mut row) in x_normalized.rows_mut().into_iter().enumerate() {
            row -= feature_means;
            row /= feature_stds;
        }

        // For simplicity, we'll implement a basic SVM using all samples as support vectors
        // In a real implementation, you'd use SMO or other optimization algorithms
        let support_vectors = x_normalized.clone();
        let mut support_coefficients = Array1::<Float>::zeros(n_samples);

        // Simple heuristic: coefficients proportional to target values
        let y_mean = y.mean().unwrap();
        for i in 0..n_samples {
            support_coefficients[i] = (y[i] - y_mean) / self.c;
        }

        let bias = y_mean;

        Ok(SVMModel {
            support_vectors,
            support_coefficients,
            bias,
            kernel,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>> for MultiOutputSVM<MultiOutputSVMTrained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, _) = x.dim();
        let mut predictions = Array2::<Float>::zeros((n_samples, self.state.n_outputs));

        // Normalize input features
        let mut x_normalized = x.to_owned();
        for (i, mut row) in x_normalized.rows_mut().into_iter().enumerate() {
            row -= &self.state.feature_means;
            row /= &self.state.feature_stds;
        }

        for output_idx in 0..self.state.n_outputs {
            let model = &self.state.models[output_idx];

            for sample_idx in 0..n_samples {
                let x_sample = x_normalized.row(sample_idx);
                let mut prediction = model.bias;

                // Compute kernel sum
                for (sv_idx, support_vector) in model.support_vectors.rows().into_iter().enumerate()
                {
                    let kernel_value =
                        compute_kernel_value(&x_sample, &support_vector, model.kernel);
                    prediction += model.support_coefficients[sv_idx] * kernel_value;
                }

                predictions[[sample_idx, output_idx]] = prediction;
            }
        }

        Ok(predictions)
    }
}

/// Compute kernel value between two vectors
fn compute_kernel_value(
    x1: &ArrayView1<Float>,
    x2: &ArrayView1<Float>,
    kernel: SVMKernel,
) -> Float {
    match kernel {
        SVMKernel::Linear => x1.dot(x2),
        SVMKernel::Polynomial {
            degree,
            gamma,
            coef0,
        } => (gamma * x1.dot(x2) + coef0).powi(degree),
        SVMKernel::Rbf { gamma } => {
            let dist_sq = x1
                .iter()
                .zip(x2.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<Float>();
            (-gamma * dist_sq).exp()
        }
        SVMKernel::Sigmoid { gamma, coef0 } => (gamma * x1.dot(x2) + coef0).tanh(),
    }
}
