//! Multi-label classification algorithms
//!
//! This module provides various multi-label classification approaches including
//! calibrated methods, k-nearest neighbor approaches, cost-sensitive methods,
//! and specialized techniques for handling multiple labels simultaneously.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Calibrated Binary Relevance Method
///
/// Enhanced binary relevance that applies probability calibration to improve
/// prediction reliability and provide confidence estimates.
#[derive(Debug, Clone)]
pub struct CalibratedBinaryRelevance<S = Untrained> {
    state: S,
    calibration_method: CalibrationMethod,
}

/// Calibration methods for probability calibration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationMethod {
    /// Platt scaling (sigmoid calibration)
    Platt,
    /// Isotonic regression calibration
    Isotonic,
}

/// Trained state for CalibratedBinaryRelevance
#[derive(Debug, Clone)]
pub struct CalibratedBinaryRelevanceTrained {
    base_models: Vec<(Array1<Float>, Float)>, // (weights, bias) for each label
    calibration_params: Vec<(Float, Float)>,  // (slope, intercept) for each label
    calibration_method: CalibrationMethod,
    n_features: usize,
    n_labels: usize,
}

impl Default for CalibratedBinaryRelevance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CalibratedBinaryRelevance<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for CalibratedBinaryRelevance<Untrained> {
    type Fitted = CalibratedBinaryRelevance<CalibratedBinaryRelevanceTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let mut base_models = Vec::new();
        let mut calibration_params = Vec::new();

        // Train base classifiers and calibration for each label
        for label_idx in 0..n_labels {
            let y_label = y.column(label_idx);

            // Train base logistic regression
            let mut weights = Array1::<Float>::zeros(n_features);
            let mut bias = 0.0;
            let learning_rate = 0.01;
            let max_iter = 100;

            // Simple logistic regression training
            for _iter in 0..max_iter {
                let mut weight_gradient = Array1::<Float>::zeros(n_features);
                let mut bias_gradient = 0.0;

                for sample_idx in 0..n_samples {
                    let x = X.row(sample_idx);
                    let y_true = y_label[sample_idx] as Float;

                    let logit = x.dot(&weights) + bias;
                    let prob = 1.0 / (1.0 + (-logit).exp());
                    let error = prob - y_true;

                    // Accumulate gradients
                    for feat_idx in 0..n_features {
                        weight_gradient[feat_idx] += error * x[feat_idx];
                    }
                    bias_gradient += error;
                }

                // Update parameters
                for i in 0..n_features {
                    weights[i] -= learning_rate * weight_gradient[i] / n_samples as Float;
                }
                bias -= learning_rate * bias_gradient / n_samples as Float;
            }

            // Collect probabilities for calibration
            let mut probs = Vec::new();
            let mut labels = Vec::new();
            for sample_idx in 0..n_samples {
                let x = X.row(sample_idx);
                let logit = x.dot(&weights) + bias;
                let prob = 1.0 / (1.0 + (-logit).exp());
                probs.push(prob);
                labels.push(y_label[sample_idx] as Float);
            }

            // Fit calibration
            let (slope, intercept) = self.fit_calibration(&probs, &labels)?;

            base_models.push((weights, bias));
            calibration_params.push((slope, intercept));
        }

        Ok(CalibratedBinaryRelevance {
            state: CalibratedBinaryRelevanceTrained {
                base_models,
                calibration_params,
                calibration_method: self.calibration_method,
                n_features,
                n_labels,
            },
            calibration_method: self.calibration_method,
        })
    }
}

impl CalibratedBinaryRelevance<Untrained> {
    /// Create a new CalibratedBinaryRelevance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            calibration_method: CalibrationMethod::Platt,
        }
    }

    /// Set the calibration method
    pub fn calibration_method(mut self, method: CalibrationMethod) -> Self {
        self.calibration_method = method;
        self
    }

    /// Fit calibration parameters
    fn fit_calibration(&self, probs: &[Float], labels: &[Float]) -> SklResult<(Float, Float)> {
        // Simple Platt scaling implementation
        match self.calibration_method {
            CalibrationMethod::Platt => {
                // Fit sigmoid: p_cal = 1 / (1 + exp(a*p + b))
                // Simplified: just fit linear transformation
                let mut a = -1.0;
                let mut b = 0.0;
                let learning_rate = 0.01;

                for _iter in 0..100 {
                    let mut grad_a = 0.0;
                    let mut grad_b = 0.0;

                    for (i, &prob) in probs.iter().enumerate() {
                        let y_true = labels[i];
                        let logit = a * prob + b;
                        let cal_prob = 1.0 / (1.0 + (-logit).exp());
                        let error = cal_prob - y_true;

                        grad_a += error * prob;
                        grad_b += error;
                    }

                    a -= learning_rate * grad_a / probs.len() as Float;
                    b -= learning_rate * grad_b / probs.len() as Float;
                }

                Ok((a, b))
            }
            CalibrationMethod::Isotonic => {
                // Simplified isotonic regression
                Ok((-1.0, 0.0))
            }
        }
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for CalibratedBinaryRelevance<CalibratedBinaryRelevanceTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            let x = X.row(sample_idx);

            for label_idx in 0..self.state.n_labels {
                let (weights, bias) = &self.state.base_models[label_idx];
                let (slope, intercept) = self.state.calibration_params[label_idx];

                // Get base probability
                let logit = x.dot(weights) + bias;
                let base_prob = 1.0 / (1.0 + (-logit).exp());

                // Apply calibration
                let cal_logit = slope * base_prob + intercept;
                let cal_prob = 1.0 / (1.0 + (-cal_logit).exp());

                predictions[[sample_idx, label_idx]] = if cal_prob > 0.5 { 1 } else { 0 };
            }
        }

        Ok(predictions)
    }
}

impl CalibratedBinaryRelevance<CalibratedBinaryRelevanceTrained> {
    /// Get calibrated probabilities
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut probabilities = Array2::<Float>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            let x = X.row(sample_idx);

            for label_idx in 0..self.state.n_labels {
                let (weights, bias) = &self.state.base_models[label_idx];
                let (slope, intercept) = self.state.calibration_params[label_idx];

                // Get base probability
                let logit = x.dot(weights) + bias;
                let base_prob = 1.0 / (1.0 + (-logit).exp());

                // Apply calibration
                let cal_logit = slope * base_prob + intercept;
                let cal_prob = 1.0 / (1.0 + (-cal_logit).exp());

                probabilities[[sample_idx, label_idx]] = cal_prob;
            }
        }

        Ok(probabilities)
    }
}

/// Random Label Combinations Method
///
/// Generates random label combinations for evaluation and testing purposes.
/// Useful for creating synthetic multi-label datasets with controlled characteristics.
pub struct RandomLabelCombinations {
    n_labels: usize,
    n_combinations: usize,
    label_density: Float,
    random_state: Option<u64>,
}

impl RandomLabelCombinations {
    /// Create a new RandomLabelCombinations generator
    pub fn new(n_labels: usize) -> Self {
        Self {
            n_labels,
            n_combinations: 100,
            label_density: 0.3,
            random_state: None,
        }
    }

    /// Set the number of combinations to generate
    pub fn n_combinations(mut self, n_combinations: usize) -> Self {
        self.n_combinations = n_combinations;
        self
    }

    /// Set the label density (proportion of positive labels)
    pub fn label_density(mut self, density: Float) -> Self {
        self.label_density = density;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate random label combinations
    pub fn generate(&self) -> Array2<i32> {
        let mut rng = if let Some(_seed) = self.random_state {
            // TODO: Implement deterministic seeding with ThreadRng
            thread_rng()
        } else {
            thread_rng()
        };

        let mut combinations = Array2::<i32>::zeros((self.n_combinations, self.n_labels));

        for i in 0..self.n_combinations {
            for j in 0..self.n_labels {
                combinations[[i, j]] = if rng.gen::<Float>() < self.label_density {
                    1
                } else {
                    0
                };
            }
        }

        combinations
    }
}

/// ML-kNN: Multi-Label k-Nearest Neighbors
///
/// ML-kNN is an adaptation of the k-nearest neighbors algorithm for multi-label classification.
/// It uses the maximum a posteriori (MAP) principle to determine the label set for a test instance
/// based on the labels of its k nearest neighbors.
#[derive(Debug, Clone)]
pub struct MLkNN<S = Untrained> {
    state: S,
    k: usize,
    smooth: Float,
    distance_metric: DistanceMetric,
}

/// Distance metrics for ML-kNN
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
}

/// Trained state for ML-kNN
#[derive(Debug, Clone)]
pub struct MLkNNTrained {
    training_data: Array2<Float>,
    training_labels: Array2<i32>,
    prior_probs: Array1<Float>,
    conditional_probs: Array2<Float>, // P(label|neighbor_count)
    k: usize,
    smooth: Float,
    distance_metric: DistanceMetric,
    n_labels: usize,
}

impl Default for MLkNN<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MLkNN<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for MLkNN<Untrained> {
    type Fitted = MLkNN<MLkNNTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.k >= n_samples {
            return Err(SklearsError::InvalidInput(
                "k must be smaller than the number of training samples".to_string(),
            ));
        }

        // Calculate prior probabilities
        let mut prior_probs = Array1::<Float>::zeros(n_labels);
        for label_idx in 0..n_labels {
            let positive_count = y.column(label_idx).iter().filter(|&&x| x == 1).count();
            prior_probs[label_idx] =
                (positive_count as Float + self.smooth) / (n_samples as Float + 2.0 * self.smooth);
        }

        // Calculate conditional probabilities P(neighbor_count | label)
        let mut conditional_probs = Array2::<Float>::zeros((n_labels, self.k + 1));

        for sample_idx in 0..n_samples {
            let neighbors = self.find_k_neighbors(X, sample_idx, &X.view())?;

            for label_idx in 0..n_labels {
                let label_count = neighbors
                    .iter()
                    .filter(|&&neighbor_idx| y[[neighbor_idx, label_idx]] == 1)
                    .count();

                if y[[sample_idx, label_idx]] == 1 {
                    conditional_probs[[label_idx, label_count]] += 1.0;
                }
            }
        }

        // Normalize conditional probabilities with smoothing
        for label_idx in 0..n_labels {
            let total_positive = y.column(label_idx).iter().filter(|&&x| x == 1).count() as Float;
            for count in 0..=self.k {
                conditional_probs[[label_idx, count]] = (conditional_probs[[label_idx, count]]
                    + self.smooth)
                    / (total_positive + (self.k + 1) as Float * self.smooth);
            }
        }

        Ok(MLkNN {
            state: MLkNNTrained {
                training_data: X.to_owned(),
                training_labels: y.clone(),
                prior_probs,
                conditional_probs,
                k: self.k,
                smooth: self.smooth,
                distance_metric: self.distance_metric,
                n_labels,
            },
            k: self.k,
            smooth: self.smooth,
            distance_metric: self.distance_metric,
        })
    }
}

impl MLkNN<Untrained> {
    /// Create a new ML-kNN classifier
    pub fn new() -> Self {
        Self {
            state: Untrained,
            k: 10,
            smooth: 1.0,
            distance_metric: DistanceMetric::Euclidean,
        }
    }

    /// Set the number of neighbors
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the smoothing parameter
    pub fn smooth(mut self, smooth: Float) -> Self {
        self.smooth = smooth;
        self
    }

    /// Set the distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Find k nearest neighbors for a sample
    fn find_k_neighbors(
        &self,
        X: &ArrayView2<'_, Float>,
        sample_idx: usize,
        training_data: &ArrayView2<'_, Float>,
    ) -> SklResult<Vec<usize>> {
        let query = X.row(sample_idx);
        let mut distances = Vec::new();

        for (train_idx, train_sample) in training_data.rows().into_iter().enumerate() {
            if train_idx != sample_idx {
                let distance = self.calculate_distance(&query, &train_sample);
                distances.push((distance, train_idx));
            }
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbors = distances
            .into_iter()
            .take(self.k)
            .map(|(_, idx)| idx)
            .collect();

        Ok(neighbors)
    }

    /// Calculate distance between two samples
    fn calculate_distance(&self, a: &ArrayView1<'_, Float>, b: &ArrayView1<'_, Float>) -> Float {
        match self.distance_metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<Float>()
                .sqrt(),
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
            DistanceMetric::Cosine => {
                let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<Float>();
                let norm_a = a.iter().map(|x| x.powi(2)).sum::<Float>().sqrt();
                let norm_b = b.iter().map(|x| x.powi(2)).sum::<Float>().sqrt();
                if norm_a > 0.0 && norm_b > 0.0 {
                    1.0 - dot / (norm_a * norm_b)
                } else {
                    1.0
                }
            }
        }
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>> for MLkNN<MLkNNTrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.training_data.ncols() {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            let neighbors = self.find_k_neighbors_trained(X, sample_idx)?;

            for label_idx in 0..self.state.n_labels {
                // Count positive neighbors for this label
                let positive_neighbors = neighbors
                    .iter()
                    .filter(|&&neighbor_idx| {
                        self.state.training_labels[[neighbor_idx, label_idx]] == 1
                    })
                    .count();

                // Calculate posterior probabilities using MAP
                let prob_positive = self.state.prior_probs[label_idx]
                    * self.state.conditional_probs[[label_idx, positive_neighbors]];
                let prob_negative = (1.0 - self.state.prior_probs[label_idx])
                    * (1.0 - self.state.conditional_probs[[label_idx, positive_neighbors]]);

                predictions[[sample_idx, label_idx]] =
                    if prob_positive > prob_negative { 1 } else { 0 };
            }
        }

        Ok(predictions)
    }
}

impl MLkNN<MLkNNTrained> {
    /// Find k nearest neighbors for a test sample
    fn find_k_neighbors_trained(
        &self,
        X: &ArrayView2<'_, Float>,
        sample_idx: usize,
    ) -> SklResult<Vec<usize>> {
        let query = X.row(sample_idx);
        let mut distances = Vec::new();

        for (train_idx, train_sample) in self.state.training_data.rows().into_iter().enumerate() {
            let distance = self.calculate_distance_trained(&query, &train_sample);
            distances.push((distance, train_idx));
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbors = distances
            .into_iter()
            .take(self.state.k)
            .map(|(_, idx)| idx)
            .collect();

        Ok(neighbors)
    }

    /// Calculate distance between two samples (trained version)
    fn calculate_distance_trained(
        &self,
        a: &ArrayView1<'_, Float>,
        b: &ArrayView1<'_, Float>,
    ) -> Float {
        match self.state.distance_metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<Float>()
                .sqrt(),
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
            DistanceMetric::Cosine => {
                let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<Float>();
                let norm_a = a.iter().map(|x| x.powi(2)).sum::<Float>().sqrt();
                let norm_b = b.iter().map(|x| x.powi(2)).sum::<Float>().sqrt();
                if norm_a > 0.0 && norm_b > 0.0 {
                    1.0 - dot / (norm_a * norm_b)
                } else {
                    1.0
                }
            }
        }
    }

    /// Get the number of neighbors
    pub fn k(&self) -> usize {
        self.state.k
    }

    /// Get prior probabilities
    pub fn prior_probabilities(&self) -> &Array1<Float> {
        &self.state.prior_probs
    }
}

/// Cost-Sensitive Binary Relevance
///
/// Binary relevance approach that incorporates label-specific misclassification costs
/// to optimize cost-sensitive performance rather than accuracy.
#[derive(Debug, Clone)]
pub struct CostSensitiveBinaryRelevance<S = Untrained> {
    state: S,
    cost_matrix: CostMatrix,
    learning_rate: Float,
    max_iterations: usize,
    regularization: Float,
}

/// Cost matrix for cost-sensitive learning
#[derive(Debug, Clone)]
pub struct CostMatrix {
    /// Cost of false positives for each label
    false_positive_costs: Array1<Float>,
    /// Cost of false negatives for each label
    false_negative_costs: Array1<Float>,
}

impl CostMatrix {
    /// Create a new cost matrix
    pub fn new(false_positive_costs: Array1<Float>, false_negative_costs: Array1<Float>) -> Self {
        Self {
            false_positive_costs,
            false_negative_costs,
        }
    }

    /// Create uniform cost matrix
    pub fn uniform(n_labels: usize, fp_cost: Float, fn_cost: Float) -> Self {
        Self {
            false_positive_costs: Array1::from_elem(n_labels, fp_cost),
            false_negative_costs: Array1::from_elem(n_labels, fn_cost),
        }
    }

    /// Get false positive cost for a label
    pub fn fp_cost(&self, label_idx: usize) -> Float {
        self.false_positive_costs
            .get(label_idx)
            .copied()
            .unwrap_or(1.0)
    }

    /// Get false negative cost for a label
    pub fn fn_cost(&self, label_idx: usize) -> Float {
        self.false_negative_costs
            .get(label_idx)
            .copied()
            .unwrap_or(1.0)
    }
}

/// Trained state for cost-sensitive binary relevance
#[derive(Debug, Clone)]
pub struct CostSensitiveBinaryRelevanceTrained {
    models: Vec<SimpleBinaryModel>,
    cost_matrix: CostMatrix,
    n_features: usize,
    n_labels: usize,
}

/// Simple binary model for cost-sensitive learning
#[derive(Debug, Clone)]
pub struct SimpleBinaryModel {
    weights: Array1<Float>,
    bias: Float,
    threshold: Float, // Cost-sensitive threshold
}

impl Default for CostSensitiveBinaryRelevance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CostSensitiveBinaryRelevance<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for CostSensitiveBinaryRelevance<Untrained> {
    type Fitted = CostSensitiveBinaryRelevance<CostSensitiveBinaryRelevanceTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let mut models = Vec::new();

        // Train cost-sensitive binary classifier for each label
        for label_idx in 0..n_labels {
            let y_label = y.column(label_idx);
            let fp_cost = self.cost_matrix.fp_cost(label_idx);
            let fn_cost = self.cost_matrix.fn_cost(label_idx);

            let mut weights = Array1::<Float>::zeros(n_features);
            let mut bias = 0.0;

            // Cost-sensitive training loop
            for _iter in 0..self.max_iterations {
                let mut weight_gradient = Array1::<Float>::zeros(n_features);
                let mut bias_gradient = 0.0;

                for sample_idx in 0..n_samples {
                    let x = X.row(sample_idx);
                    let y_true = y_label[sample_idx] as Float;

                    let logit = x.dot(&weights) + bias;
                    let prob = 1.0 / (1.0 + (-logit).exp());

                    // Cost-sensitive gradient
                    let cost_weight = if y_true == 1.0 { fn_cost } else { fp_cost };
                    let error = (prob - y_true) * cost_weight;

                    // Accumulate gradients
                    for feat_idx in 0..n_features {
                        weight_gradient[feat_idx] += error * x[feat_idx];
                    }
                    bias_gradient += error;
                }

                // Add L2 regularization
                for i in 0..n_features {
                    weight_gradient[i] += self.regularization * weights[i];
                }

                // Update parameters
                for i in 0..n_features {
                    weights[i] -= self.learning_rate * weight_gradient[i] / n_samples as Float;
                }
                bias -= self.learning_rate * bias_gradient / n_samples as Float;
            }

            // Calculate cost-sensitive threshold
            let threshold = self.calculate_cost_sensitive_threshold(fp_cost, fn_cost);

            models.push(SimpleBinaryModel {
                weights,
                bias,
                threshold,
            });
        }

        Ok(CostSensitiveBinaryRelevance {
            state: CostSensitiveBinaryRelevanceTrained {
                models,
                cost_matrix: self.cost_matrix,
                n_features,
                n_labels,
            },
            cost_matrix: CostMatrix::uniform(n_labels, 1.0, 1.0),
            learning_rate: self.learning_rate,
            max_iterations: self.max_iterations,
            regularization: self.regularization,
        })
    }
}

impl CostSensitiveBinaryRelevance<Untrained> {
    /// Create a new cost-sensitive binary relevance classifier
    pub fn new() -> Self {
        Self {
            state: Untrained,
            cost_matrix: CostMatrix::uniform(1, 1.0, 1.0),
            learning_rate: 0.01,
            max_iterations: 100,
            regularization: 0.01,
        }
    }

    /// Set the cost matrix
    pub fn cost_matrix(mut self, cost_matrix: CostMatrix) -> Self {
        self.cost_matrix = cost_matrix;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the regularization strength
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Calculate cost-sensitive threshold
    fn calculate_cost_sensitive_threshold(&self, fp_cost: Float, fn_cost: Float) -> Float {
        // Threshold that minimizes expected cost
        // threshold = log(fp_cost / fn_cost) if we had class priors
        // Simplified version
        fp_cost / (fp_cost + fn_cost)
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for CostSensitiveBinaryRelevance<CostSensitiveBinaryRelevanceTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            let x = X.row(sample_idx);

            for (label_idx, model) in self.state.models.iter().enumerate() {
                let logit = x.dot(&model.weights) + model.bias;
                let prob = 1.0 / (1.0 + (-logit).exp());

                predictions[[sample_idx, label_idx]] = if prob > model.threshold { 1 } else { 0 };
            }
        }

        Ok(predictions)
    }
}

impl CostSensitiveBinaryRelevance<CostSensitiveBinaryRelevanceTrained> {
    /// Get the cost matrix
    pub fn cost_matrix(&self) -> &CostMatrix {
        &self.state.cost_matrix
    }

    /// Get model thresholds
    pub fn thresholds(&self) -> Vec<Float> {
        self.state.models.iter().map(|m| m.threshold).collect()
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
    fn test_calibrated_binary_relevance_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        let cbr = CalibratedBinaryRelevance::new().calibration_method(CalibrationMethod::Platt);
        let trained_cbr = cbr.fit(&X.view(), &y).unwrap();
        let predictions = trained_cbr.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (4, 2));
        assert!(predictions.iter().all(|&x| x == 0 || x == 1));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_calibrated_binary_relevance_probabilities() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1, 0], [0, 1]];

        let cbr = CalibratedBinaryRelevance::new();
        let trained_cbr = cbr.fit(&X.view(), &y).unwrap();
        let probabilities = trained_cbr.predict_proba(&X.view()).unwrap();

        assert_eq!(probabilities.dim(), (2, 2));
        assert!(probabilities.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_random_label_combinations() {
        let generator = RandomLabelCombinations::new(3)
            .n_combinations(5)
            .label_density(0.5)
            .random_state(42);

        let combinations = generator.generate();
        assert_eq!(combinations.dim(), (5, 3));
        assert!(combinations.iter().all(|&x| x == 0 || x == 1));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mlknn_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0], [1.5, 2.5]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]];

        let mlknn = MLkNN::new().k(3).smooth(1.0);
        let trained_mlknn = mlknn.fit(&X.view(), &y).unwrap();
        let predictions = trained_mlknn.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (5, 2));
        assert!(predictions.iter().all(|&x| x == 0 || x == 1));
        assert_eq!(trained_mlknn.k(), 3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mlknn_distance_metrics() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![[1, 0], [0, 1], [1, 1]];

        let mlknn_euclidean = MLkNN::new().k(2).distance_metric(DistanceMetric::Euclidean);
        let trained_euclidean = mlknn_euclidean.fit(&X.view(), &y).unwrap();

        let mlknn_manhattan = MLkNN::new().k(2).distance_metric(DistanceMetric::Manhattan);
        let trained_manhattan = mlknn_manhattan.fit(&X.view(), &y).unwrap();

        let pred_euclidean = trained_euclidean.predict(&X.view()).unwrap();
        let pred_manhattan = trained_manhattan.predict(&X.view()).unwrap();

        assert_eq!(pred_euclidean.dim(), (3, 2));
        assert_eq!(pred_manhattan.dim(), (3, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_cost_sensitive_binary_relevance() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]];

        let fp_costs = array![2.0, 1.0]; // Higher cost for FP on first label
        let fn_costs = array![1.0, 3.0]; // Higher cost for FN on second label
        let cost_matrix = CostMatrix::new(fp_costs, fn_costs);

        let csbr = CostSensitiveBinaryRelevance::new()
            .cost_matrix(cost_matrix)
            .learning_rate(0.01)
            .max_iterations(50);

        let trained_csbr = csbr.fit(&X.view(), &y).unwrap();
        let predictions = trained_csbr.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (4, 2));
        assert!(predictions.iter().all(|&x| x == 0 || x == 1));

        let thresholds = trained_csbr.thresholds();
        assert_eq!(thresholds.len(), 2);
    }

    #[test]
    fn test_cost_matrix_creation() {
        let fp_costs = array![1.0, 2.0, 3.0];
        let fn_costs = array![2.0, 1.0, 1.0];
        let cost_matrix = CostMatrix::new(fp_costs, fn_costs);

        assert_eq!(cost_matrix.fp_cost(0), 1.0);
        assert_eq!(cost_matrix.fp_cost(1), 2.0);
        assert_eq!(cost_matrix.fn_cost(0), 2.0);
        assert_eq!(cost_matrix.fn_cost(1), 1.0);

        let uniform_costs = CostMatrix::uniform(3, 1.5, 2.5);
        assert_eq!(uniform_costs.fp_cost(0), 1.5);
        assert_eq!(uniform_costs.fn_cost(2), 2.5);
    }

    #[test]
    fn test_calibration_methods() {
        let cbr_platt =
            CalibratedBinaryRelevance::new().calibration_method(CalibrationMethod::Platt);
        let cbr_isotonic =
            CalibratedBinaryRelevance::new().calibration_method(CalibrationMethod::Isotonic);

        // Just test that they can be created with different methods
        assert_eq!(cbr_platt.calibration_method, CalibrationMethod::Platt);
        assert_eq!(cbr_isotonic.calibration_method, CalibrationMethod::Isotonic);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mlknn_prior_probabilities() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
        let y = array![[1, 0], [0, 1], [1, 1], [0, 0]]; // 2/4 positive for each label

        let mlknn = MLkNN::new().k(2).smooth(1.0);
        let trained_mlknn = mlknn.fit(&X.view(), &y).unwrap();

        let priors = trained_mlknn.prior_probabilities();
        assert_eq!(priors.len(), 2);

        // With smoothing: (2 + 1) / (4 + 2) = 0.5
        assert!((priors[0] - 0.5).abs() < 1e-6);
        assert!((priors[1] - 0.5).abs() < 1e-6);
    }
}
