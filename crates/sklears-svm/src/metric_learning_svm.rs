//! Metric Learning Support Vector Machines
//!
//! This module implements metric learning SVMs that learn optimal distance metrics
//! for improved classification performance. These methods learn to transform the
//! feature space to make classification easier.
//!
//! Algorithms included:
//! - Large Margin Nearest Neighbor (LMNN)
//! - Information Theoretic Metric Learning (ITML)
//! - Neighborhood Component Analysis (NCA)
//! - Metric Learning for SVM (ML-SVM)

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Transform, Untrained},
};
use std::fmt;
use std::marker::PhantomData;

/// Errors that can occur during metric learning SVM training and prediction
#[derive(Debug, Clone)]
pub enum MetricLearningError {
    InvalidInput(String),
    TrainingError(String),
    PredictionError(String),
    ConvergenceError(String),
}

impl fmt::Display for MetricLearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricLearningError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            MetricLearningError::TrainingError(msg) => write!(f, "Training error: {msg}"),
            MetricLearningError::PredictionError(msg) => write!(f, "Prediction error: {msg}"),
            MetricLearningError::ConvergenceError(msg) => write!(f, "Convergence error: {msg}"),
        }
    }
}

impl std::error::Error for MetricLearningError {}

/// Metric learning algorithm type
#[derive(Debug, Clone)]
pub enum MetricLearningAlgorithm {
    /// Large Margin Nearest Neighbor
    LMNN {
        k: usize,            // Number of target neighbors
        regularization: f64, // Regularization parameter
    },
    /// Information Theoretic Metric Learning
    ITML {
        gamma: f64,             // Regularization parameter
        max_constraints: usize, // Maximum number of constraints
    },
    /// Neighborhood Component Analysis
    NCA {
        regularization: f64, // L2 regularization
    },
    /// Metric Learning for SVM
    MLSVM {
        c: f64,          // SVM regularization parameter
        metric_reg: f64, // Metric regularization parameter
    },
}

/// Configuration for metric learning SVM
#[derive(Debug, Clone)]
pub struct MetricLearningSVMConfig {
    /// Metric learning algorithm
    pub algorithm: MetricLearningAlgorithm,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for MetricLearningSVMConfig {
    fn default() -> Self {
        Self {
            algorithm: MetricLearningAlgorithm::LMNN {
                k: 3,
                regularization: 0.1,
            },
            max_iter: 1000,
            tol: 1e-6,
            learning_rate: 0.01,
            verbose: false,
        }
    }
}

/// Metric Learning SVM
#[derive(Debug, Clone)]
pub struct MetricLearningSVM<State = Untrained> {
    config: MetricLearningSVMConfig,
    state: PhantomData<State>,
    // Learned metric (transformation matrix)
    metric_matrix: Option<Array2<f64>>,
    // SVM classifier weights
    svm_weights: Option<Array1<f64>>,
    svm_intercept: Option<f64>,
    // Training data statistics
    n_features: Option<usize>,
    classes: Option<Array1<f64>>,
    // Preprocessing
    mean: Option<Array1<f64>>,
    std: Option<Array1<f64>>,
}

impl Default for MetricLearningSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricLearningSVM<Untrained> {
    /// Create a new metric learning SVM
    pub fn new() -> Self {
        Self {
            config: MetricLearningSVMConfig::default(),
            state: PhantomData,
            metric_matrix: None,
            svm_weights: None,
            svm_intercept: None,
            n_features: None,
            classes: None,
            mean: None,
            std: None,
        }
    }

    /// Set the metric learning algorithm
    pub fn with_algorithm(mut self, algorithm: MetricLearningAlgorithm) -> Self {
        self.config.algorithm = algorithm;
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

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>> for MetricLearningSVM<Untrained> {
    type Fitted = MetricLearningSVM<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string(),
            ));
        }

        // Find unique classes
        let classes = Self::find_classes(y);
        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Standardize data
        let (x_std, mean, std) = Self::standardize_data(x);

        // Learn metric transformation
        let metric_matrix = self.learn_metric(&x_std, y, &classes)?;

        // Transform data using learned metric
        let x_transformed = self.transform_data(&x_std, &metric_matrix);

        // Train SVM on transformed data
        let (svm_weights, svm_intercept) = self.train_svm(&x_transformed, y, &classes)?;

        Ok(MetricLearningSVM {
            config: self.config,
            state: PhantomData,
            metric_matrix: Some(metric_matrix),
            svm_weights: Some(svm_weights),
            svm_intercept: Some(svm_intercept),
            n_features: Some(n_features),
            classes: Some(classes),
            mean: Some(mean),
            std: Some(std),
        })
    }
}

impl MetricLearningSVM<Untrained> {
    /// Find unique classes in labels
    fn find_classes(y: &Array1<f64>) -> Array1<f64> {
        let mut classes: Vec<f64> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        Array1::from_vec(classes)
    }

    /// Standardize data (zero mean, unit variance)
    fn standardize_data(x: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let (n_samples, n_features) = x.dim();
        let mut mean = Array1::zeros(n_features);
        let mut std = Array1::zeros(n_features);

        // Compute mean
        for j in 0..n_features {
            mean[j] = x.column(j).mean().unwrap();
        }

        // Compute standard deviation
        for j in 0..n_features {
            let variance = x
                .column(j)
                .iter()
                .map(|&val| (val - mean[j]).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            std[j] = variance.sqrt().max(1e-8); // Avoid division by zero
        }

        // Standardize
        let mut x_std = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                x_std[[i, j]] = (x[[i, j]] - mean[j]) / std[j];
            }
        }

        (x_std, mean, std)
    }

    /// Learn metric transformation matrix
    fn learn_metric(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        match &self.config.algorithm {
            MetricLearningAlgorithm::LMNN { k, regularization } => {
                self.learn_lmnn_metric(x, y, *k, *regularization)
            }
            MetricLearningAlgorithm::ITML {
                gamma,
                max_constraints,
            } => self.learn_itml_metric(x, y, *gamma, *max_constraints),
            MetricLearningAlgorithm::NCA { regularization } => {
                self.learn_nca_metric(x, y, *regularization)
            }
            MetricLearningAlgorithm::MLSVM { c, metric_reg } => {
                self.learn_mlsvm_metric(x, y, *c, *metric_reg)
            }
        }
    }

    /// Learn Large Margin Nearest Neighbor (LMNN) metric
    fn learn_lmnn_metric(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        k: usize,
        regularization: f64,
    ) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut metric = Array2::eye(n_features); // Initialize as identity

        // Find k nearest neighbors for each sample
        let target_neighbors = self.find_target_neighbors(x, y, k)?;

        for iteration in 0..self.config.max_iter {
            let mut gradient = Array2::zeros((n_features, n_features));

            // Pull force: encourage target neighbors to be close
            for (i, neighbors) in target_neighbors.iter().enumerate() {
                for &j in neighbors {
                    let diff = &x.row(i) - &x.row(j);
                    let outer_product = self.outer_product(&diff, &diff);
                    gradient = gradient + outer_product.clone();
                }
            }

            // Push force: encourage impostors to be far
            for i in 0..n_samples {
                for &j in &target_neighbors[i] {
                    let diff_ij = &x.row(i) - &x.row(j);
                    let dist_ij = self.mahalanobis_distance(&diff_ij, &metric);

                    // Find impostors (samples with different labels that are too close)
                    for l in 0..n_samples {
                        if y[l] != y[i] {
                            let diff_il = &x.row(i) - &x.row(l);
                            let dist_il = self.mahalanobis_distance(&diff_il, &metric);

                            let margin = 1.0; // Margin parameter
                            if dist_il - dist_ij < margin {
                                // This is an impostor
                                let diff_outer = self.outer_product(&diff_il, &diff_il);
                                let target_outer = self.outer_product(&diff_ij, &diff_ij);
                                gradient = gradient - diff_outer + target_outer;
                            }
                        }
                    }
                }
            }

            // Add regularization
            gradient = gradient + regularization * &metric;

            // Update metric matrix
            metric = &metric - self.config.learning_rate * &gradient;

            // Ensure positive semi-definite (project onto PSD cone)
            metric = self.project_psd(&metric);

            if self.config.verbose && iteration % 100 == 0 {
                println!("LMNN Iteration {iteration}: Metric updated");
            }

            // Check convergence (simplified)
            if gradient.iter().map(|&x| x.abs()).sum::<f64>() < self.config.tol {
                if self.config.verbose {
                    println!("LMNN converged after {} iterations", iteration + 1);
                }
                break;
            }
        }

        Ok(metric)
    }

    /// Learn Information Theoretic Metric Learning (ITML) metric
    fn learn_itml_metric(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        gamma: f64,
        max_constraints: usize,
    ) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut metric = Array2::eye(n_features); // Initialize as identity

        // Generate constraints
        let constraints = self.generate_constraints(x, y, max_constraints)?;

        for iteration in 0..self.config.max_iter {
            let mut violated_constraints = 0;

            for constraint in &constraints {
                let (i, j, should_be_close, target_distance) = *constraint;
                let diff = &x.row(i) - &x.row(j);
                let current_distance = self.mahalanobis_distance(&diff, &metric);

                let violation = if should_be_close {
                    (current_distance - target_distance).max(0.0)
                } else {
                    (target_distance - current_distance).max(0.0)
                };

                if violation > self.config.tol {
                    violated_constraints += 1;

                    // Update metric using Bregman projection
                    let update_rate = violation / (gamma + violation);
                    let outer_product = self.outer_product(&diff, &diff);

                    if should_be_close {
                        // Move closer
                        metric = &metric - update_rate * &outer_product;
                    } else {
                        // Move farther
                        metric = &metric + update_rate * &outer_product;
                    }

                    // Ensure positive semi-definite
                    metric = self.project_psd(&metric);
                }
            }

            if self.config.verbose && iteration % 100 == 0 {
                println!(
                    "ITML Iteration {}: {} violated constraints",
                    iteration, violated_constraints
                );
            }

            if violated_constraints == 0 {
                if self.config.verbose {
                    println!("ITML converged after {} iterations", iteration + 1);
                }
                break;
            }
        }

        Ok(metric)
    }

    /// Learn Neighborhood Component Analysis (NCA) metric
    fn learn_nca_metric(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        regularization: f64,
    ) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut transformation = Array2::eye(n_features);

        for iteration in 0..self.config.max_iter {
            let mut gradient = Array2::zeros((n_features, n_features));
            let mut objective = 0.0;

            for i in 0..n_samples {
                let mut numerator_sum = 0.0;
                let mut denominator_sum = 0.0;
                let mut grad_i: Array2<f64> = Array2::zeros((n_features, n_features));

                // Compute softmax probabilities
                for j in 0..n_samples {
                    if i != j {
                        let diff = &x.row(i) - &x.row(j);
                        let transformed_diff = transformation.dot(&diff);
                        let distance_sq = transformed_diff.iter().map(|&x| x * x).sum::<f64>();
                        let exp_neg_dist = (-distance_sq).exp();

                        denominator_sum += exp_neg_dist;

                        if y[i] == y[j] {
                            numerator_sum += exp_neg_dist;
                        }
                    }
                }

                // Compute probability and update objective
                let prob = if denominator_sum > 0.0 {
                    numerator_sum / denominator_sum
                } else {
                    0.0
                };
                objective += prob;

                // Compute gradient contribution
                for j in 0..n_samples {
                    if i != j {
                        let diff = &x.row(i) - &x.row(j);
                        let transformed_diff = transformation.dot(&diff);
                        let distance_sq = transformed_diff.iter().map(|&x| x * x).sum::<f64>();
                        let exp_neg_dist = (-distance_sq).exp();

                        let p_ij = if denominator_sum > 0.0 {
                            exp_neg_dist / denominator_sum
                        } else {
                            0.0
                        };

                        let weight = if y[i] == y[j] {
                            p_ij * (1.0 - prob)
                        } else {
                            -p_ij * prob
                        };

                        let outer_product = self.outer_product(&diff, &transformed_diff);
                        grad_i = grad_i + weight * outer_product;
                    }
                }

                gradient = gradient + grad_i;
            }

            // Add regularization
            gradient = gradient - regularization * &transformation;

            // Update transformation matrix
            transformation = &transformation + self.config.learning_rate * &gradient;

            if self.config.verbose && iteration % 100 == 0 {
                println!("NCA Iteration {}: Objective = {:.6}", iteration, objective);
            }

            // Check convergence
            if gradient.iter().map(|&x| x.abs()).sum::<f64>() < self.config.tol {
                if self.config.verbose {
                    println!("NCA converged after {} iterations", iteration + 1);
                }
                break;
            }
        }

        // Convert transformation to metric (M = A^T A)
        let metric = transformation.t().dot(&transformation);
        Ok(metric)
    }

    /// Learn joint metric and SVM parameters
    fn learn_mlsvm_metric(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        c: f64,
        metric_reg: f64,
    ) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut metric = Array2::eye(n_features);
        let mut svm_weights = Array1::zeros(n_features);
        let mut svm_bias = 0.0;

        // Convert labels to binary {-1, +1} for now (simplified)
        let binary_y: Array1<f64> = y
            .iter()
            .map(|&label| if label > 0.5 { 1.0 } else { -1.0 })
            .collect();

        for iteration in 0..self.config.max_iter {
            // Update metric given current SVM
            let mut metric_gradient = Array2::zeros((n_features, n_features));

            for i in 0..n_samples {
                let xi = x.row(i);
                let xi_vec = xi.to_owned();
                let transformed_xi = metric.dot(&xi_vec);
                let score = svm_weights.dot(&transformed_xi) + svm_bias;
                let margin = binary_y[i] * score;

                if margin < 1.0 {
                    // This is a support vector
                    let outer_product = self.outer_product(&xi_vec, &svm_weights);
                    metric_gradient = metric_gradient + binary_y[i] * c * outer_product;
                }
            }

            // Add metric regularization
            metric_gradient = metric_gradient - metric_reg * &metric;

            // Update metric
            metric = &metric + self.config.learning_rate * &metric_gradient;
            metric = self.project_psd(&metric);

            // Update SVM given current metric
            let mut svm_gradient = Array1::zeros(n_features);
            let mut bias_gradient = 0.0;

            for i in 0..n_samples {
                let xi = x.row(i);
                let xi_vec = xi.to_owned();
                let transformed_xi = metric.dot(&xi_vec);
                let score = svm_weights.dot(&transformed_xi) + svm_bias;
                let margin = binary_y[i] * score;

                if margin < 1.0 {
                    svm_gradient = svm_gradient + binary_y[i] * c * transformed_xi.clone();
                    bias_gradient += binary_y[i] * c;
                }
            }

            // Add SVM regularization
            svm_gradient -= &svm_weights;

            // Update SVM parameters
            svm_weights = &svm_weights + self.config.learning_rate * &svm_gradient;
            svm_bias += self.config.learning_rate * bias_gradient;

            if self.config.verbose && iteration % 100 == 0 {
                println!("ML-SVM Iteration {iteration}: Metric and SVM updated");
            }

            // Check convergence (simplified)
            let total_gradient_norm = metric_gradient.iter().map(|&x| x.abs()).sum::<f64>()
                + svm_gradient.iter().map(|&x| x.abs()).sum::<f64>();

            if total_gradient_norm < self.config.tol {
                if self.config.verbose {
                    println!("ML-SVM converged after {} iterations", iteration + 1);
                }
                break;
            }
        }

        Ok(metric)
    }

    /// Helper functions
    fn find_target_neighbors(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        k: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let n_samples = x.nrows();
        let mut target_neighbors = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut same_class_neighbors = Vec::new();
            let mut distances = Vec::new();

            // Find all samples with the same class
            for j in 0..n_samples {
                if i != j && y[i] == y[j] {
                    let diff = &x.row(i) - &x.row(j);
                    let distance = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
                    distances.push((j, distance));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            same_class_neighbors.extend(
                distances
                    .iter()
                    .take(k.min(distances.len()))
                    .map(|(idx, _)| *idx),
            );

            target_neighbors.push(same_class_neighbors);
        }

        Ok(target_neighbors)
    }

    fn generate_constraints(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        max_constraints: usize,
    ) -> Result<Vec<(usize, usize, bool, f64)>> {
        let n_samples = x.nrows();
        let mut constraints = Vec::new();

        // Generate must-link and cannot-link constraints
        let mut count = 0;
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                if count >= max_constraints {
                    break;
                }

                let diff = &x.row(i) - &x.row(j);
                let distance = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();

                if y[i] == y[j] {
                    // Must-link constraint
                    constraints.push((i, j, true, distance * 0.8)); // Target closer distance
                } else {
                    // Cannot-link constraint
                    constraints.push((i, j, false, distance * 1.2)); // Target farther distance
                }

                count += 1;
            }
            if count >= max_constraints {
                break;
            }
        }

        Ok(constraints)
    }

    fn mahalanobis_distance(&self, diff: &Array1<f64>, metric: &Array2<f64>) -> f64 {
        let transformed = metric.dot(diff);
        diff.dot(&transformed)
    }

    fn outer_product(&self, a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((a.len(), b.len()));
        for i in 0..a.len() {
            for j in 0..b.len() {
                result[[i, j]] = a[i] * b[j];
            }
        }
        result
    }

    fn project_psd(&self, matrix: &Array2<f64>) -> Array2<f64> {
        // Simplified PSD projection: just make diagonal positive
        let mut result = matrix.clone();
        for i in 0..matrix.nrows() {
            if result[[i, i]] <= 0.0 {
                result[[i, i]] = 1e-6;
            }
        }
        result
    }

    /// Transform data using learned metric
    fn transform_data(&self, x: &Array2<f64>, metric: &Array2<f64>) -> Array2<f64> {
        // For simplicity, we'll use the metric as a linear transformation
        // In practice, you might want to use Cholesky decomposition
        x.dot(metric)
    }

    /// Train SVM on transformed data
    fn train_svm(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64)> {
        let (n_samples, n_features) = x.dim();

        // Simplified SVM training using SGD
        let mut weights = Array1::zeros(n_features);
        let mut bias = 0.0;
        let c = 1.0; // Regularization parameter

        // Convert to binary classification for simplicity
        let binary_y: Array1<f64> = y
            .iter()
            .map(|&label| if label == classes[0] { -1.0 } else { 1.0 })
            .collect();

        for iteration in 0..1000 {
            for i in 0..n_samples {
                let xi = x.row(i);
                let score = weights.dot(&xi) + bias;
                let margin = binary_y[i] * score;

                if margin < 1.0 {
                    // Update weights and bias
                    let learning_rate = 0.01 / (1.0 + iteration as f64 * 0.01);
                    weights =
                        &weights * (1.0 - learning_rate / c) + learning_rate * binary_y[i] * &xi;
                    bias += learning_rate * binary_y[i];
                } else {
                    // Only regularization update
                    let learning_rate = 0.01 / (1.0 + iteration as f64 * 0.01);
                    weights = &weights * (1.0 - learning_rate / c);
                }
            }
        }

        Ok((weights, bias))
    }
}

impl Predict<Array2<f64>, Array1<f64>> for MetricLearningSVM<Trained> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let (n_samples, n_features) = x.dim();

        if n_features != *self.n_features.as_ref().unwrap() {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        // Standardize input data
        let x_std = self.standardize_input(x);

        // Transform using learned metric
        let metric = self.metric_matrix.as_ref().unwrap();
        let x_transformed = self.transform_data(&x_std, metric);

        // Predict using SVM
        let weights = self.svm_weights.as_ref().unwrap();
        let bias = *self.svm_intercept.as_ref().unwrap();
        let classes = self.classes.as_ref().unwrap();

        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let score = weights.dot(&x_transformed.row(i)) + bias;
            predictions[i] = if score >= 0.0 { classes[1] } else { classes[0] };
        }

        Ok(predictions)
    }
}

impl Transform<Array2<f64>, Array2<f64>> for MetricLearningSVM<Trained> {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();

        if n_features != *self.n_features.as_ref().unwrap() {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        // Standardize input data
        let x_std = self.standardize_input(x);

        // Transform using learned metric
        let metric = self.metric_matrix.as_ref().unwrap();
        let x_transformed = self.transform_data(&x_std, metric);

        Ok(x_transformed)
    }
}

impl MetricLearningSVM<Trained> {
    /// Standardize input data using training statistics
    fn standardize_input(&self, x: &Array2<f64>) -> Array2<f64> {
        let (n_samples, n_features) = x.dim();
        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        let mut x_std = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                x_std[[i, j]] = (x[[i, j]] - mean[j]) / std[j];
            }
        }
        x_std
    }

    /// Transform data using learned metric
    fn transform_data(&self, x: &Array2<f64>, metric: &Array2<f64>) -> Array2<f64> {
        x.dot(metric)
    }

    /// Get the learned metric matrix
    pub fn metric_matrix(&self) -> &Array2<f64> {
        self.metric_matrix.as_ref().unwrap()
    }

    /// Get SVM weights
    pub fn svm_weights(&self) -> &Array1<f64> {
        self.svm_weights.as_ref().unwrap()
    }

    /// Get SVM intercept
    pub fn svm_intercept(&self) -> f64 {
        *self.svm_intercept.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn create_test_data() -> (Array2<f64>, Array1<f64>) {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 5.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_metric_learning_svm_creation() {
        let ml_svm = MetricLearningSVM::new()
            .with_algorithm(MetricLearningAlgorithm::LMNN {
                k: 2,
                regularization: 0.1,
            })
            .with_max_iter(100)
            .verbose(false);

        assert!(matches!(
            ml_svm.config.algorithm,
            MetricLearningAlgorithm::LMNN { .. }
        ));
        assert_eq!(ml_svm.config.max_iter, 100);
    }

    #[test]
    fn test_metric_learning_svm_fit() {
        let (x, y) = create_test_data();

        let ml_svm = MetricLearningSVM::new()
            .with_algorithm(MetricLearningAlgorithm::LMNN {
                k: 1,
                regularization: 0.1,
            })
            .with_max_iter(10)
            .with_tolerance(1e-2);

        use sklears_core::traits::Fit;
        let result = ml_svm.fit(&x, &y);
        assert!(result.is_ok());

        let trained_model = result.unwrap();
        assert!(trained_model.metric_matrix.is_some());
        assert!(trained_model.svm_weights.is_some());
    }

    #[test]
    fn test_metric_learning_svm_predict() {
        let (x, y) = create_test_data();

        let ml_svm = MetricLearningSVM::new()
            .with_algorithm(MetricLearningAlgorithm::NCA {
                regularization: 0.1,
            })
            .with_max_iter(5)
            .with_tolerance(1e-1);

        use sklears_core::traits::Predict;
        let trained_model = ml_svm.fit(&x, &y).unwrap();

        let predictions = trained_model.predict(&x);
        assert!(predictions.is_ok());

        let pred_labels = predictions.unwrap();
        assert_eq!(pred_labels.len(), x.nrows());
    }

    #[test]
    fn test_metric_learning_transform() {
        let (x, y) = create_test_data();

        let ml_svm = MetricLearningSVM::new()
            .with_algorithm(MetricLearningAlgorithm::LMNN {
                k: 1,
                regularization: 0.1,
            })
            .with_max_iter(5);

        use sklears_core::traits::Transform;
        let trained_model = ml_svm.fit(&x, &y).unwrap();

        let transformed_x = trained_model.transform(&x);
        assert!(transformed_x.is_ok());

        let x_new = transformed_x.unwrap();
        assert_eq!(x_new.dim(), x.dim());
    }

    #[test]
    fn test_invalid_input() {
        let x = Array2::zeros((5, 2));
        let y = Array1::zeros(6); // Wrong number of samples

        let ml_svm = MetricLearningSVM::new();
        use sklears_core::traits::Fit;
        let result = ml_svm.fit(&x, &y);
        assert!(result.is_err());
    }
}
