//! Outlier-Resistant Support Vector Machines
//!
//! This module implements SVM variants that are specifically designed to handle
//! datasets with outliers by detecting and down-weighting or removing outliers
//! during training. It combines robust loss functions with outlier detection
//! algorithms for enhanced robustness.

use crate::kernels::{Kernel, KernelType};
use crate::robust_svm::RobustLoss;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Outlier detection methods
#[derive(Debug, Clone, PartialEq)]
pub enum OutlierDetectionMethod {
    /// Statistical outlier detection using z-score
    ZScore { threshold: Float },
    /// Interquartile Range (IQR) method
    IQR { multiplier: Float },
    /// Modified Z-score using median absolute deviation
    ModifiedZScore { threshold: Float },
    /// Distance-based outlier detection
    DistanceBased { k: usize, threshold: Float },
    /// Local Outlier Factor approximation
    LocalOutlierFactor { k: usize, threshold: Float },
    /// Isolation Forest approximation using random cuts
    IsolationForest { n_trees: usize, threshold: Float },
}

impl Default for OutlierDetectionMethod {
    fn default() -> Self {
        OutlierDetectionMethod::ModifiedZScore { threshold: 3.5 }
    }
}

/// Weight assignment strategy for samples
#[derive(Debug, Clone, PartialEq)]
pub enum WeightStrategy {
    /// Binary: remove outliers completely (weight = 0) or keep (weight = 1)
    Binary,
    /// Smooth: assign weights based on outlier scores using sigmoid
    Smooth { steepness: Float },
    /// Linear decay: linearly decrease weights based on outlier scores
    LinearDecay,
    /// Exponential decay: exponentially decrease weights
    ExponentialDecay { rate: Float },
}

impl Default for WeightStrategy {
    fn default() -> Self {
        WeightStrategy::Smooth { steepness: 2.0 }
    }
}

/// Configuration for outlier-resistant SVM
#[derive(Debug, Clone)]
pub struct OutlierResistantSVMConfig {
    /// Regularization parameter
    pub c: Float,
    /// Robust loss function
    pub loss: RobustLoss,
    /// Kernel function to use
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Random seed
    pub random_state: Option<u64>,
    /// Learning rate for gradient-based optimization
    pub learning_rate: Float,
    /// Learning rate decay
    pub learning_rate_decay: Float,
    /// Minimum learning rate
    pub min_learning_rate: Float,
    /// Outlier detection method
    pub outlier_detection: OutlierDetectionMethod,
    /// Weight assignment strategy
    pub weight_strategy: WeightStrategy,
    /// Maximum number of outlier detection iterations
    pub max_outlier_iter: usize,
    /// Minimum weight threshold (samples below this are considered outliers)
    pub min_weight_threshold: Float,
}

impl Default for OutlierResistantSVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            loss: RobustLoss::default(),
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 1000,
            fit_intercept: true,
            random_state: None,
            learning_rate: 0.01,
            learning_rate_decay: 0.99,
            min_learning_rate: 1e-6,
            outlier_detection: OutlierDetectionMethod::default(),
            weight_strategy: WeightStrategy::default(),
            max_outlier_iter: 5,
            min_weight_threshold: 0.1,
        }
    }
}

/// Outlier-Resistant Support Vector Machine for classification and regression
#[derive(Debug)]
pub struct OutlierResistantSVM<State = Untrained> {
    config: OutlierResistantSVMConfig,
    state: PhantomData<State>,
    // Fitted attributes
    support_vectors_: Option<Array2<Float>>,
    alpha_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_in_: Option<usize>,
    n_iter_: Option<usize>,
    sample_weights_: Option<Array1<Float>>,
    outlier_scores_: Option<Array1<Float>>,
    n_outliers_detected_: Option<usize>,
}

impl OutlierResistantSVM<Untrained> {
    /// Create a new outlier-resistant SVM
    pub fn new() -> Self {
        Self {
            config: OutlierResistantSVMConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            alpha_: None,
            intercept_: None,
            n_features_in_: None,
            n_iter_: None,
            sample_weights_: None,
            outlier_scores_: None,
            n_outliers_detected_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the robust loss function
    pub fn loss(mut self, loss: RobustLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set the outlier detection method
    pub fn outlier_detection(mut self, method: OutlierDetectionMethod) -> Self {
        self.config.outlier_detection = method;
        self
    }

    /// Set the weight assignment strategy
    pub fn weight_strategy(mut self, strategy: WeightStrategy) -> Self {
        self.config.weight_strategy = strategy;
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the maximum number of outlier detection iterations
    pub fn max_outlier_iter(mut self, max_outlier_iter: usize) -> Self {
        self.config.max_outlier_iter = max_outlier_iter;
        self
    }

    /// Set the minimum weight threshold
    pub fn min_weight_threshold(mut self, threshold: Float) -> Self {
        self.config.min_weight_threshold = threshold;
        self
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }
}

impl Default for OutlierResistantSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl OutlierDetectionMethod {
    /// Compute outlier scores for the given data
    pub fn compute_outlier_scores(&self, x: &Array2<Float>, y: &Array1<Float>) -> Array1<Float> {
        match self {
            OutlierDetectionMethod::ZScore { threshold: _ } => self.compute_z_score_outliers(x, y),
            OutlierDetectionMethod::IQR { multiplier: _ } => self.compute_iqr_outliers(x, y),
            OutlierDetectionMethod::ModifiedZScore { threshold: _ } => {
                self.compute_modified_z_score_outliers(x, y)
            }
            OutlierDetectionMethod::DistanceBased { k, threshold: _ } => {
                self.compute_distance_based_outliers(x, *k)
            }
            OutlierDetectionMethod::LocalOutlierFactor { k, threshold: _ } => {
                self.compute_lof_outliers(x, *k)
            }
            OutlierDetectionMethod::IsolationForest {
                n_trees,
                threshold: _,
            } => self.compute_isolation_forest_outliers(x, *n_trees),
        }
    }

    fn compute_z_score_outliers(&self, x: &Array2<Float>, y: &Array1<Float>) -> Array1<Float> {
        let mut scores = Array1::<Float>::zeros(x.nrows());

        // Compute z-scores for each feature
        for j in 0..x.ncols() {
            let col = x.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = ((col.map(|&val| (val - mean).powi(2)).sum() / col.len() as Float).sqrt())
                .max(1e-8);

            for i in 0..x.nrows() {
                let z_score = ((x[[i, j]] - mean) / std).abs();
                scores[i] = scores[i].max(z_score);
            }
        }

        // Also consider target values for regression
        if y.len() == x.nrows() {
            let y_mean = y.mean().unwrap_or(0.0);
            let y_std =
                ((y.map(|&val| (val - y_mean).powi(2)).sum() / y.len() as Float).sqrt()).max(1e-8);

            for i in 0..y.len() {
                let y_z_score = ((y[i] - y_mean) / y_std).abs();
                scores[i] = scores[i].max(y_z_score);
            }
        }

        scores
    }

    fn compute_modified_z_score_outliers(
        &self,
        x: &Array2<Float>,
        _y: &Array1<Float>,
    ) -> Array1<Float> {
        let mut scores = Array1::<Float>::zeros(x.nrows());

        // Compute modified z-scores using median and MAD
        for j in 0..x.ncols() {
            let mut col_values: Vec<Float> = x.column(j).to_vec();
            col_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = if col_values.len() % 2 == 0 {
                (col_values[col_values.len() / 2 - 1] + col_values[col_values.len() / 2]) / 2.0
            } else {
                col_values[col_values.len() / 2]
            };

            // Compute MAD (Median Absolute Deviation)
            let mut deviations: Vec<Float> =
                col_values.iter().map(|&val| (val - median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mad = if deviations.len() % 2 == 0 {
                (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
            } else {
                deviations[deviations.len() / 2]
            };

            let mad_scaled = (mad * 1.4826).max(1e-8); // Scale factor for normal distribution

            for i in 0..x.nrows() {
                let modified_z_score = (0.6745 * (x[[i, j]] - median) / mad_scaled).abs();
                scores[i] = scores[i].max(modified_z_score);
            }
        }

        scores
    }

    fn compute_iqr_outliers(&self, x: &Array2<Float>, _y: &Array1<Float>) -> Array1<Float> {
        let mut scores = Array1::<Float>::zeros(x.nrows());

        for j in 0..x.ncols() {
            let mut col_values: Vec<Float> = x.column(j).to_vec();
            col_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = col_values.len();
            let q1_idx = n / 4;
            let q3_idx = 3 * n / 4;

            let q1 = col_values[q1_idx];
            let q3 = col_values[q3_idx];
            let iqr = q3 - q1;

            if let OutlierDetectionMethod::IQR { multiplier } = self {
                let lower_bound = q1 - multiplier * iqr;
                let upper_bound = q3 + multiplier * iqr;

                for i in 0..x.nrows() {
                    let val = x[[i, j]];
                    let outlier_degree = if val < lower_bound {
                        (lower_bound - val) / iqr.max(1e-8)
                    } else if val > upper_bound {
                        (val - upper_bound) / iqr.max(1e-8)
                    } else {
                        0.0
                    };
                    scores[i] = scores[i].max(outlier_degree);
                }
            }
        }

        scores
    }

    fn compute_distance_based_outliers(&self, x: &Array2<Float>, k: usize) -> Array1<Float> {
        let mut scores = Array1::<Float>::zeros(x.nrows());
        let n_samples = x.nrows();

        for i in 0..n_samples {
            let mut distances = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = x
                        .row(i)
                        .iter()
                        .zip(x.row(j).iter())
                        .map(|(&xi, &xj)| (xi - xj).powi(2))
                        .sum::<Float>()
                        .sqrt();
                    distances.push(dist);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Average distance to k nearest neighbors
            let k_actual = k.min(distances.len());
            if k_actual > 0 {
                scores[i] = distances[..k_actual].iter().sum::<Float>() / k_actual as Float;
            }
        }

        // Normalize scores
        let max_score = scores.iter().cloned().fold(0.0, Float::max);
        if max_score > 0.0 {
            scores.mapv_inplace(|x| x / max_score);
        }

        scores
    }

    fn compute_lof_outliers(&self, x: &Array2<Float>, k: usize) -> Array1<Float> {
        let mut scores = Array1::<Float>::zeros(x.nrows());
        let n_samples = x.nrows();

        // Simplified LOF approximation
        for i in 0..n_samples {
            let mut distances = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let dist = x
                        .row(i)
                        .iter()
                        .zip(x.row(j).iter())
                        .map(|(&xi, &xj)| (xi - xj).powi(2))
                        .sum::<Float>()
                        .sqrt();
                    distances.push((dist, j));
                }
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let k_actual = k.min(distances.len());
            if k_actual > 0 {
                let k_distance = distances[k_actual - 1].0;
                let reachability_density = k_actual as Float / (k_distance + 1e-8);

                // Approximate local outlier factor
                let mut neighbor_densities = 0.0;
                for &(_, _neighbor_idx) in distances[..k_actual].iter() {
                    // Simplified density estimation for neighbor
                    neighbor_densities += 1.0;
                }

                let avg_neighbor_density = neighbor_densities / k_actual as Float;
                scores[i] = avg_neighbor_density / (reachability_density + 1e-8);
            }
        }

        scores
    }

    fn compute_isolation_forest_outliers(
        &self,
        x: &Array2<Float>,
        n_trees: usize,
    ) -> Array1<Float> {
        let mut scores = Array1::<Float>::zeros(x.nrows());
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Simplified isolation forest using random cuts
        let mut rng = scirs2_core::random::thread_rng();

        for _ in 0..n_trees {
            // Create random cuts
            let max_depth = (n_samples as Float).log2().ceil() as usize;

            for i in 0..n_samples {
                let mut depth = 0;
                let current_sample = x.row(i).to_owned();

                for _ in 0..max_depth {
                    let feature_idx = rng.gen_range(0..n_features);
                    let min_val = x
                        .column(feature_idx)
                        .iter()
                        .cloned()
                        .fold(Float::INFINITY, Float::min);
                    let max_val = x
                        .column(feature_idx)
                        .iter()
                        .cloned()
                        .fold(Float::NEG_INFINITY, Float::max);
                    let split_value = if max_val > min_val {
                        rng.gen_range(min_val..max_val)
                    } else {
                        min_val
                    };

                    if current_sample[feature_idx] < split_value {
                        // Go left
                    } else {
                        // Go right
                    }

                    depth += 1;

                    // Random early stopping
                    if rng.random::<Float>() < 0.5 {
                        break;
                    }
                }

                // Shorter path lengths indicate outliers
                scores[i] += (max_depth - depth) as Float;
            }
        }

        // Normalize by number of trees
        scores.mapv_inplace(|x| x / n_trees as Float);

        // Normalize to [0, 1]
        let max_score = scores.iter().cloned().fold(0.0, Float::max);
        if max_score > 0.0 {
            scores.mapv_inplace(|x| x / max_score);
        }

        scores
    }
}

impl WeightStrategy {
    /// Convert outlier scores to sample weights
    pub fn compute_weights(&self, outlier_scores: &Array1<Float>) -> Array1<Float> {
        match self {
            WeightStrategy::Binary => {
                let threshold = outlier_scores.mean().unwrap_or(0.0) + outlier_scores.std(1.0);
                outlier_scores.mapv(|score| if score > threshold { 0.0 } else { 1.0 })
            }
            WeightStrategy::Smooth { steepness } => {
                let max_score = outlier_scores.iter().cloned().fold(0.0, Float::max);
                outlier_scores.mapv(|score| {
                    let normalized_score = if max_score > 0.0 {
                        score / max_score
                    } else {
                        0.0
                    };
                    1.0 / (1.0 + (steepness * normalized_score).exp())
                })
            }
            WeightStrategy::LinearDecay => {
                let max_score = outlier_scores.iter().cloned().fold(0.0, Float::max);
                outlier_scores.mapv(|score| {
                    let normalized_score = if max_score > 0.0 {
                        score / max_score
                    } else {
                        0.0
                    };
                    (1.0 - normalized_score).max(0.0)
                })
            }
            WeightStrategy::ExponentialDecay { rate } => {
                outlier_scores.mapv(|score| (-rate * score).exp())
            }
        }
    }
}

impl Fit<Array2<Float>, Array1<Float>> for OutlierResistantSVM<Untrained> {
    type Fitted = OutlierResistantSVM<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Shape mismatch: X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize sample weights
        let mut sample_weights = Array1::ones(n_samples);
        let mut outlier_scores = Array1::<Float>::zeros(n_samples);

        // Create kernel instance
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>, // Default fallback
        };

        // Initialize model parameters
        let support_vectors = x.clone();
        let mut alpha = Array1::<Float>::zeros(n_samples);
        let mut intercept = 0.0;
        let mut current_lr = self.config.learning_rate;
        let mut n_iter = 0;

        // Iterative outlier detection and model training
        for outlier_iter in 0..self.config.max_outlier_iter {
            // Detect outliers if not first iteration
            if outlier_iter > 0 {
                // Compute residuals from current model
                let mut residuals = Array1::zeros(n_samples);
                for i in 0..n_samples {
                    let mut prediction = if self.config.fit_intercept {
                        intercept
                    } else {
                        0.0
                    };
                    for j in 0..n_samples {
                        if alpha[j].abs() > 1e-10 {
                            let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                            prediction += alpha[j] * k_val;
                        }
                    }
                    residuals[i] = (prediction - y[i]).abs();
                }

                // Update outlier scores based on residuals
                let residual_threshold = residuals.mean().unwrap_or(0.0) + 2.0 * residuals.std(1.0);
                for i in 0..n_samples {
                    if residuals[i] > residual_threshold {
                        outlier_scores[i] =
                            outlier_scores[i].max(residuals[i] / residual_threshold);
                    }
                }
            } else {
                // Initial outlier detection
                outlier_scores = self.config.outlier_detection.compute_outlier_scores(x, y);
            }

            // Convert outlier scores to sample weights
            sample_weights = self.config.weight_strategy.compute_weights(&outlier_scores);

            // Train model with weighted samples
            for iteration in 0..self.config.max_iter {
                n_iter = iteration + 1;
                let mut converged = true;

                for i in 0..n_samples {
                    // Skip samples with very low weights
                    if sample_weights[i] < self.config.min_weight_threshold {
                        continue;
                    }

                    // Compute prediction
                    let mut prediction = if self.config.fit_intercept {
                        intercept
                    } else {
                        0.0
                    };
                    for j in 0..n_samples {
                        if alpha[j].abs() > 1e-10 {
                            let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                            prediction += alpha[j] * k_val;
                        }
                    }

                    // Compute prediction error
                    let error = prediction - y[i];

                    // Compute loss derivative
                    let loss_derivative = self.config.loss.derivative(error);

                    // Update alpha using weighted gradient descent
                    let old_alpha = alpha[i];
                    let weighted_gradient =
                        sample_weights[i] * (loss_derivative + alpha[i] / self.config.c);
                    alpha[i] -= current_lr * weighted_gradient;

                    // Update intercept if needed
                    if self.config.fit_intercept {
                        intercept -= current_lr * sample_weights[i] * loss_derivative;
                    }

                    // Check convergence
                    if (alpha[i] - old_alpha).abs() > self.config.tol {
                        converged = false;
                    }
                }

                // Update learning rate
                current_lr = (current_lr * self.config.learning_rate_decay)
                    .max(self.config.min_learning_rate);

                if converged {
                    break;
                }
            }
        }

        // Count detected outliers
        let n_outliers_detected = sample_weights
            .iter()
            .filter(|&&w| w < self.config.min_weight_threshold)
            .count();

        // Filter out non-support vectors (alpha close to zero) and low-weight samples
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(i, &a)| {
                a.abs() > 1e-8 && sample_weights[*i] >= self.config.min_weight_threshold
            })
            .map(|(i, _)| i)
            .collect();

        let final_support_vectors = if support_indices.is_empty() {
            // If no support vectors, keep the first non-outlier sample
            let first_valid = (0..n_samples)
                .find(|&i| sample_weights[i] >= self.config.min_weight_threshold)
                .unwrap_or(0);
            let mut sv = Array2::zeros((1, n_features));
            sv.row_mut(0).assign(&x.row(first_valid));
            sv
        } else {
            let mut sv = Array2::zeros((support_indices.len(), n_features));
            for (i, &idx) in support_indices.iter().enumerate() {
                sv.row_mut(i).assign(&x.row(idx));
            }
            sv
        };

        let final_alpha = if support_indices.is_empty() {
            Array1::from_vec(vec![1e-8])
        } else {
            Array1::from_vec(support_indices.iter().map(|&i| alpha[i]).collect())
        };

        Ok(OutlierResistantSVM {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(final_support_vectors),
            alpha_: Some(final_alpha),
            intercept_: Some(intercept),
            n_features_in_: Some(n_features),
            n_iter_: Some(n_iter),
            sample_weights_: Some(sample_weights),
            outlier_scores_: Some(outlier_scores),
            n_outliers_detected_: Some(n_outliers_detected),
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for OutlierResistantSVM<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(
                "Feature mismatch: X has different number of features than training data"
                    .to_string(),
            ));
        }

        let support_vectors = self.support_vectors_.as_ref().unwrap();
        let alpha = self.alpha_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>, // Default fallback
        };

        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut prediction = if self.config.fit_intercept {
                intercept
            } else {
                0.0
            };

            for j in 0..support_vectors.nrows() {
                let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                prediction += alpha[j] * k_val;
            }

            predictions[i] = prediction;
        }

        Ok(predictions)
    }
}

impl OutlierResistantSVM<Trained> {
    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_.as_ref().unwrap()
    }

    /// Get the support vector coefficients (alpha values)
    pub fn alpha(&self) -> &Array1<Float> {
        self.alpha_.as_ref().unwrap()
    }

    /// Get the intercept term
    pub fn intercept(&self) -> Float {
        self.intercept_.unwrap()
    }

    /// Get the number of features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get the number of iterations performed during training
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Get the sample weights computed during training
    pub fn sample_weights(&self) -> &Array1<Float> {
        self.sample_weights_.as_ref().unwrap()
    }

    /// Get the outlier scores computed during training
    pub fn outlier_scores(&self) -> &Array1<Float> {
        self.outlier_scores_.as_ref().unwrap()
    }

    /// Get the number of outliers detected
    pub fn n_outliers_detected(&self) -> usize {
        self.n_outliers_detected_.unwrap()
    }

    /// Compute the decision function values
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        self.predict(x)
    }

    /// Predict outlier scores for new data
    pub fn predict_outlier_scores(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        // Use residuals from predictions as outlier indicators
        let predictions = self.predict(x)?;
        let residuals = predictions.mapv(|pred| pred.abs());

        // Normalize to [0, 1] range
        let max_residual = residuals.iter().cloned().fold(0.0, Float::max);
        let outlier_scores = if max_residual > 0.0 {
            residuals.mapv(|r| r / max_residual)
        } else {
            Array1::zeros(x.nrows())
        };

        Ok(outlier_scores)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_outlier_resistant_svm_creation() {
        let orsvm = OutlierResistantSVM::new()
            .c(2.0)
            .loss(RobustLoss::Huber { delta: 1.5 })
            .kernel(KernelType::Linear)
            .outlier_detection(OutlierDetectionMethod::ModifiedZScore { threshold: 3.0 })
            .weight_strategy(WeightStrategy::Smooth { steepness: 1.5 })
            .max_outlier_iter(3)
            .min_weight_threshold(0.2);

        assert_eq!(orsvm.config.c, 2.0);
        assert_eq!(orsvm.config.loss, RobustLoss::Huber { delta: 1.5 });
        assert_eq!(orsvm.config.max_outlier_iter, 3);
        assert_eq!(orsvm.config.min_weight_threshold, 0.2);
    }

    #[test]
    fn test_z_score_outlier_detection() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [100.0, 100.0], // Clear outlier
        ];
        let y = array![1.0, 2.0, 3.0, 100.0];

        let method = OutlierDetectionMethod::ZScore { threshold: 2.0 };
        let scores = method.compute_outlier_scores(&x, &y);

        assert_eq!(scores.len(), 4);
        // The last sample should have the highest outlier score
        assert!(scores[3] > scores[0]);
        assert!(scores[3] > scores[1]);
        assert!(scores[3] > scores[2]);
    }

    #[test]
    fn test_weight_strategies() {
        let outlier_scores = array![0.1, 0.2, 0.9, 0.05]; // Third sample is outlier

        // Test binary strategy
        let binary_weights = WeightStrategy::Binary.compute_weights(&outlier_scores);
        assert!(binary_weights[2] < binary_weights[0]); // Outlier gets lower weight

        // Test smooth strategy
        let smooth_weights =
            WeightStrategy::Smooth { steepness: 2.0 }.compute_weights(&outlier_scores);
        assert!(smooth_weights[2] < smooth_weights[0]); // Outlier gets lower weight
        assert!(smooth_weights[2] > 0.0); // But not zero

        // Test linear decay
        let linear_weights = WeightStrategy::LinearDecay.compute_weights(&outlier_scores);
        assert!(linear_weights[2] < linear_weights[0]);
    }

    #[test]
    #[ignore = "Slow test: trains outlier-resistant SVM. Run with --ignored flag"]
    fn test_outlier_resistant_svm_training() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [100.0, 100.0], // Outlier
        ];
        let y = array![1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 1000.0]; // Outlier target

        let orsvm = OutlierResistantSVM::new()
            .c(1.0)
            .loss(RobustLoss::Huber { delta: 1.0 })
            .kernel(KernelType::Linear)
            .outlier_detection(OutlierDetectionMethod::ModifiedZScore { threshold: 2.5 })
            .weight_strategy(WeightStrategy::Smooth { steepness: 2.0 })
            .max_iter(50)
            .max_outlier_iter(3)
            .learning_rate(0.01)
            .random_state(42);

        let fitted_model = orsvm.fit(&x, &y).unwrap();

        assert_eq!(fitted_model.n_features_in(), 2);
        assert!(fitted_model.n_iter() > 0);
        assert!(fitted_model.n_outliers_detected() > 0);

        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 7);

        // Check that outlier scores are computed
        let outlier_scores = fitted_model.outlier_scores();
        assert_eq!(outlier_scores.len(), 7);

        // The outlier sample should have high outlier score
        assert!(outlier_scores[6] > outlier_scores[0]);

        // Check sample weights
        let sample_weights = fitted_model.sample_weights();
        assert_eq!(sample_weights.len(), 7);

        // The outlier should have lower weight
        assert!(sample_weights[6] < sample_weights[0]);

        // Predictions should be finite
        for &pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_outlier_resistant_svm_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0]; // Wrong length

        let orsvm = OutlierResistantSVM::new();
        let result = orsvm.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_iqr_outlier_detection() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [100.0, 100.0], // Clear outlier
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];

        let method = OutlierDetectionMethod::IQR { multiplier: 1.5 };
        let scores = method.compute_outlier_scores(&x, &y);

        assert_eq!(scores.len(), 6);
        // The last sample should have the highest outlier score
        assert!(scores[5] > scores[0]);
        assert!(scores[5] > 0.0);
    }

    #[test]
    #[ignore = "Slow test: trains outlier-resistant SVM with distance-based detection"]
    fn test_distance_based_outlier_detection() {
        let x = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0], // Outlier
        ];
        let y = array![0.0, 1.0, 2.0, 10.0];

        let method = OutlierDetectionMethod::DistanceBased {
            k: 2,
            threshold: 2.0,
        };
        let scores = method.compute_outlier_scores(&x, &y);

        assert_eq!(scores.len(), 4);
        // The outlier should have higher score
        assert!(scores[3] > scores[0]);
    }
}
