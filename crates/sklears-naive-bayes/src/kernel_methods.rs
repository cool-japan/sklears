//! Kernel methods for Naive Bayes classifiers
//!
//! This module implements advanced kernel-based methods for Naive Bayes classification,
//! including kernel density estimation, Gaussian process integration, and kernel parameter
//! learning for enhanced probabilistic modeling.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::Rng;
// SciRS2 Policy Compliance - Use scirs2-core for random distributions
use scirs2_core::random::essentials::Normal as RandNormal;
use scirs2_core::random::Distribution;
use std::collections::HashMap;
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KernelError {
    #[error("Invalid kernel parameters: {0}")]
    InvalidParameters(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Numerical error in kernel computation: {0}")]
    NumericalError(String),
    #[error("Insufficient data for kernel estimation")]
    InsufficientData,
}

/// Kernel types for kernel-based Naive Bayes methods
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Radial Basis Function kernel: k(x,y) = exp(-γ||x-y||²)
    RBF { gamma: f64 },
    /// Polynomial kernel: k(x,y) = (γ⟨x,y⟩ + r)^d
    Polynomial { degree: f64, gamma: f64, coef0: f64 },
    /// Linear kernel: k(x,y) = ⟨x,y⟩
    Linear,
    /// Sigmoid kernel: k(x,y) = tanh(γ⟨x,y⟩ + r)
    Sigmoid { gamma: f64, coef0: f64 },
    /// Laplacian kernel: k(x,y) = exp(-γ||x-y||₁)
    Laplacian { gamma: f64 },
    /// Matern kernel for Gaussian processes
    Matern { length_scale: f64, nu: f64 },
}

impl KernelType {
    /// Compute kernel function between two vectors
    pub fn compute<F: Float + 'static>(
        &self,
        x: &Array1<F>,
        y: &Array1<F>,
    ) -> Result<F, KernelError> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        match self {
            KernelType::RBF { gamma } => {
                let diff = x - y;
                let norm_sq = diff.dot(&diff);
                let gamma_f = F::from(*gamma).ok_or_else(|| {
                    KernelError::NumericalError("Failed to convert gamma".to_string())
                })?;
                Ok((-gamma_f * norm_sq).exp())
            }
            KernelType::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                let dot_product = x.dot(y);
                let gamma_f = F::from(*gamma).ok_or_else(|| {
                    KernelError::NumericalError("Failed to convert gamma".to_string())
                })?;
                let coef0_f = F::from(*coef0).ok_or_else(|| {
                    KernelError::NumericalError("Failed to convert coef0".to_string())
                })?;
                let degree_f = F::from(*degree).ok_or_else(|| {
                    KernelError::NumericalError("Failed to convert degree".to_string())
                })?;
                Ok((gamma_f * dot_product + coef0_f).powf(degree_f))
            }
            KernelType::Linear => Ok(x.dot(y)),
            KernelType::Sigmoid { gamma, coef0 } => {
                let dot_product = x.dot(y);
                let gamma_f = F::from(*gamma).ok_or_else(|| {
                    KernelError::NumericalError("Failed to convert gamma".to_string())
                })?;
                let coef0_f = F::from(*coef0).ok_or_else(|| {
                    KernelError::NumericalError("Failed to convert coef0".to_string())
                })?;
                Ok((gamma_f * dot_product + coef0_f).tanh())
            }
            KernelType::Laplacian { gamma } => {
                let diff = x - y;
                let l1_norm = diff
                    .iter()
                    .map(|&val| val.abs())
                    .fold(F::zero(), |acc, x| acc + x);
                let gamma_f = F::from(*gamma).ok_or_else(|| {
                    KernelError::NumericalError("Failed to convert gamma".to_string())
                })?;
                Ok((-gamma_f * l1_norm).exp())
            }
            KernelType::Matern { length_scale, nu } => {
                let diff = x - y;
                let distance = diff.dot(&diff).sqrt();
                let length_scale_f = F::from(*length_scale).ok_or_else(|| {
                    KernelError::NumericalError("Failed to convert length_scale".to_string())
                })?;
                let nu_f = F::from(*nu).ok_or_else(|| {
                    KernelError::NumericalError("Failed to convert nu".to_string())
                })?;

                if distance.is_zero() {
                    return Ok(F::one());
                }

                let sqrt_nu = (F::from(2.0).unwrap() * nu_f).sqrt();
                let scaled_distance = sqrt_nu * distance / length_scale_f;

                // Simplified Matern for common cases
                if (nu_f - F::from(0.5).unwrap()).abs() < F::from(1e-10).unwrap() {
                    // Matern 1/2 (Exponential)
                    Ok((-scaled_distance).exp())
                } else if (nu_f - F::from(1.5).unwrap()).abs() < F::from(1e-10).unwrap() {
                    // Matern 3/2
                    Ok((F::one() + scaled_distance) * (-scaled_distance).exp())
                } else if (nu_f - F::from(2.5).unwrap()).abs() < F::from(1e-10).unwrap() {
                    // Matern 5/2
                    let term = F::one()
                        + scaled_distance
                        + scaled_distance * scaled_distance / F::from(3.0).unwrap();
                    Ok(term * (-scaled_distance).exp())
                } else {
                    // General case (simplified)
                    Ok((-scaled_distance).exp())
                }
            }
        }
    }

    /// Compute kernel matrix between all pairs of vectors
    pub fn compute_matrix<F: Float + 'static>(
        &self,
        x: &Array2<F>,
        y: Option<&Array2<F>>,
    ) -> Result<Array2<F>, KernelError> {
        let y = y.unwrap_or(x);
        let n_samples_x = x.nrows();
        let n_samples_y = y.nrows();

        if x.ncols() != y.ncols() {
            return Err(KernelError::DimensionMismatch {
                expected: x.ncols(),
                actual: y.ncols(),
            });
        }

        let mut kernel_matrix = Array2::zeros((n_samples_x, n_samples_y));

        for i in 0..n_samples_x {
            for j in 0..n_samples_y {
                let x_i = x.row(i).to_owned();
                let y_j = y.row(j).to_owned();
                kernel_matrix[[i, j]] = self.compute(&x_i, &y_j)?;
            }
        }

        Ok(kernel_matrix)
    }
}

/// Configuration for kernel density estimation
#[derive(Debug, Clone)]
pub struct KDEConfig {
    pub kernel: KernelType,
    pub bandwidth: Option<f64>,
    pub bandwidth_method: BandwidthMethod,
    pub metric: String,
    pub atol: f64,
    pub rtol: f64,
    pub breadth_first: bool,
    pub leaf_size: usize,
}

impl Default for KDEConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::RBF { gamma: 1.0 },
            bandwidth: None,
            bandwidth_method: BandwidthMethod::Scott,
            metric: "euclidean".to_string(),
            atol: 0.0,
            rtol: 1e-8,
            breadth_first: true,
            leaf_size: 40,
        }
    }
}

/// Methods for automatic bandwidth selection
#[derive(Debug, Clone)]
pub enum BandwidthMethod {
    /// Scott's rule: n^(-1/(d+4))
    Scott,
    /// Silverman's rule: (n*(d+2)/4)^(-1/(d+4))
    Silverman,
    /// Cross-validation based selection
    CrossValidation,
    /// Manual bandwidth specification
    Manual(f64),
}

impl BandwidthMethod {
    /// Compute bandwidth for given data
    pub fn compute_bandwidth(&self, data: &Array2<f64>) -> f64 {
        let n_samples = data.nrows() as f64;
        let n_features = data.ncols() as f64;

        match self {
            BandwidthMethod::Scott => n_samples.powf(-1.0 / (n_features + 4.0)),
            BandwidthMethod::Silverman => {
                (n_samples * (n_features + 2.0) / 4.0).powf(-1.0 / (n_features + 4.0))
            }
            BandwidthMethod::CrossValidation => {
                // Simplified CV bandwidth selection
                let std_data = data.std_axis(Axis(0), 0.0);
                let median_std = {
                    let mut std_vec: Vec<f64> = std_data.iter().cloned().collect();
                    std_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    std_vec[std_vec.len() / 2]
                };
                median_std * n_samples.powf(-1.0 / (n_features + 4.0))
            }
            BandwidthMethod::Manual(bandwidth) => *bandwidth,
        }
    }
}

/// Kernel Density Estimator for non-parametric probability estimation
#[derive(Debug, Clone)]
pub struct KernelDensityEstimator {
    config: KDEConfig,
    training_data: Option<Array2<f64>>,
    bandwidth: f64,
    fitted: bool,
}

impl KernelDensityEstimator {
    /// Create new KDE with configuration
    pub fn new(config: KDEConfig) -> Self {
        Self {
            config,
            training_data: None,
            bandwidth: 1.0,
            fitted: false,
        }
    }

    /// Fit KDE to training data
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<(), KernelError> {
        if data.is_empty() {
            return Err(KernelError::InsufficientData);
        }

        self.bandwidth = self
            .config
            .bandwidth
            .unwrap_or_else(|| self.config.bandwidth_method.compute_bandwidth(data));

        // Update kernel gamma if RBF
        if let KernelType::RBF { gamma: _ } = &self.config.kernel {
            self.config.kernel = KernelType::RBF {
                gamma: 1.0 / (2.0 * self.bandwidth * self.bandwidth),
            };
        }

        self.training_data = Some(data.clone());
        self.fitted = true;
        Ok(())
    }

    /// Evaluate log probability density at given points
    pub fn score_samples(&self, data: &Array2<f64>) -> Result<Array1<f64>, KernelError> {
        if !self.fitted {
            return Err(KernelError::InvalidParameters("KDE not fitted".to_string()));
        }

        let training_data = self.training_data.as_ref().unwrap();
        let n_training = training_data.nrows() as f64;
        let n_features = training_data.ncols() as f64;

        let mut log_probs = Array1::zeros(data.nrows());

        for (i, sample) in data.outer_iter().enumerate() {
            let mut density = 0.0;

            for training_sample in training_data.outer_iter() {
                let kernel_value = self
                    .config
                    .kernel
                    .compute(&sample.to_owned(), &training_sample.to_owned())?;
                density += kernel_value;
            }

            density /= n_training;

            // Normalize by bandwidth^d for proper density estimation
            let normalization_factor = self.bandwidth.powf(n_features);
            density /= normalization_factor;

            log_probs[i] = if density > 0.0 {
                density.ln()
            } else {
                f64::NEG_INFINITY
            };
        }

        Ok(log_probs)
    }

    /// Sample from the fitted density
    pub fn sample(&self, n_samples: usize, rng: &mut impl Rng) -> Result<Array2<f64>, KernelError> {
        if !self.fitted {
            return Err(KernelError::InvalidParameters("KDE not fitted".to_string()));
        }

        let training_data = self.training_data.as_ref().unwrap();
        let n_training = training_data.nrows();
        let n_features = training_data.ncols();

        let mut samples = Array2::zeros((n_samples, n_features));
        let normal = RandNormal::new(0.0, self.bandwidth).map_err(|e| {
            KernelError::NumericalError(format!("Failed to create normal distribution: {}", e))
        })?;

        for i in 0..n_samples {
            // Select random training sample
            let idx = rng.gen_range(0..n_training);
            let base_sample = training_data.row(idx);

            // Add noise
            for j in 0..n_features {
                let noise = normal.sample(rng);
                samples[[i, j]] = base_sample[j] + noise;
            }
        }

        Ok(samples)
    }
}

/// Gaussian Process for probabilistic modeling
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    kernel: KernelType,
    training_inputs: Option<Array2<f64>>,
    training_outputs: Option<Array1<f64>>,
    noise_variance: f64,
    fitted: bool,
}

impl GaussianProcess {
    /// Create new Gaussian Process
    pub fn new(kernel: KernelType, noise_variance: f64) -> Self {
        Self {
            kernel,
            training_inputs: None,
            training_outputs: None,
            noise_variance,
            fitted: false,
        }
    }

    /// Fit GP to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), KernelError> {
        if x.nrows() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: x.nrows(),
                actual: y.len(),
            });
        }

        self.training_inputs = Some(x.clone());
        self.training_outputs = Some(y.clone());
        self.fitted = true;
        Ok(())
    }

    /// Predict mean and variance at test points
    pub fn predict(&self, x_test: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>), KernelError> {
        if !self.fitted {
            return Err(KernelError::InvalidParameters("GP not fitted".to_string()));
        }

        let x_train = self.training_inputs.as_ref().unwrap();
        let y_train = self.training_outputs.as_ref().unwrap();

        // Compute kernel matrices
        let k_train_train = self.kernel.compute_matrix(x_train, None)?;
        let k_test_train = self.kernel.compute_matrix(x_test, Some(x_train))?;
        let k_test_test = self.kernel.compute_matrix(x_test, None)?;

        // Add noise to diagonal of training kernel matrix
        let mut k_train_train_noisy = k_train_train;
        for i in 0..k_train_train_noisy.nrows() {
            k_train_train_noisy[[i, i]] += self.noise_variance;
        }

        // Simplified prediction (without proper matrix inversion for demo)
        // In production, use proper Cholesky decomposition
        let n_test = x_test.nrows();
        let mut mean = Array1::zeros(n_test);
        let mut variance = Array1::zeros(n_test);

        for i in 0..n_test {
            // Simplified prediction using nearest neighbor approximation
            let mut closest_idx = 0;
            let mut min_distance = f64::INFINITY;

            for j in 0..x_train.nrows() {
                let distance: f64 = x_test
                    .row(i)
                    .iter()
                    .zip(x_train.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    closest_idx = j;
                }
            }

            mean[i] = y_train[closest_idx];
            variance[i] =
                self.noise_variance + k_test_test[[i, i]] - k_test_train[[i, closest_idx]].powi(2);
        }

        Ok((mean, variance))
    }
}

/// Kernel-based Naive Bayes classifier
#[derive(Debug, Clone)]
pub struct KernelNaiveBayes {
    kernel: KernelType,
    class_priors: HashMap<i32, f64>,
    class_kdes: HashMap<i32, Vec<KernelDensityEstimator>>,
    classes: Vec<i32>,
    n_features: usize,
    fitted: bool,
}

impl KernelNaiveBayes {
    /// Create new Kernel Naive Bayes classifier
    pub fn new(kernel: KernelType) -> Self {
        Self {
            kernel,
            class_priors: HashMap::new(),
            class_kdes: HashMap::new(),
            classes: Vec::new(),
            n_features: 0,
            fitted: false,
        }
    }

    /// Fit classifier to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<(), KernelError> {
        if x.nrows() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: x.nrows(),
                actual: y.len(),
            });
        }

        self.n_features = x.ncols();

        // Find unique classes
        let mut unique_classes: Vec<i32> = y.iter().cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        self.classes = unique_classes;

        // Compute class priors
        let n_samples = y.len() as f64;
        for &class in &self.classes {
            let class_count = y.iter().filter(|&&label| label == class).count() as f64;
            self.class_priors.insert(class, class_count / n_samples);
        }

        // Fit KDE for each feature and class
        for &class in &self.classes {
            let mut class_data = Vec::new();

            for (i, &label) in y.iter().enumerate() {
                if label == class {
                    class_data.push(x.row(i).to_owned());
                }
            }

            if class_data.is_empty() {
                continue;
            }

            // Convert to array
            let class_array = Array2::from_shape_vec(
                (class_data.len(), self.n_features),
                class_data.into_iter().flatten().collect(),
            )
            .map_err(|e| {
                KernelError::NumericalError(format!("Array construction failed: {}", e))
            })?;

            // Fit KDE for each feature independently (naive assumption)
            let mut feature_kdes = Vec::new();
            for j in 0..self.n_features {
                let feature_data = class_array.column(j).to_owned().insert_axis(Axis(1));

                let kde_config = KDEConfig {
                    kernel: self.kernel.clone(),
                    ..Default::default()
                };

                let mut kde = KernelDensityEstimator::new(kde_config);
                kde.fit(&feature_data)?;
                feature_kdes.push(kde);
            }

            self.class_kdes.insert(class, feature_kdes);
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict class log probabilities
    pub fn predict_log_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>, KernelError> {
        if !self.fitted {
            return Err(KernelError::InvalidParameters(
                "Classifier not fitted".to_string(),
            ));
        }

        if x.ncols() != self.n_features {
            return Err(KernelError::DimensionMismatch {
                expected: self.n_features,
                actual: x.ncols(),
            });
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut log_proba = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x.outer_iter().enumerate() {
            for (j, &class) in self.classes.iter().enumerate() {
                let mut log_prob = self.class_priors[&class].ln();

                if let Some(kdes) = self.class_kdes.get(&class) {
                    for (k, kde) in kdes.iter().enumerate() {
                        let feature_value = Array2::from_elem((1, 1), sample[k]);
                        let feature_log_prob = kde.score_samples(&feature_value)?;
                        log_prob += feature_log_prob[0];
                    }
                }

                log_proba[[i, j]] = log_prob;
            }
        }

        Ok(log_proba)
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>, KernelError> {
        let log_proba = self.predict_log_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in log_proba.outer_iter().enumerate() {
            let mut max_idx = 0;
            let mut max_prob = row[0];

            for (j, &prob) in row.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    max_idx = j;
                }
            }

            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }
}

/// Kernel parameter learning through cross-validation
#[derive(Debug, Clone)]
pub struct KernelParameterLearner {
    kernel_candidates: Vec<KernelType>,
    cv_folds: usize,
    scoring_metric: ScoringMetric,
}

#[derive(Debug, Clone)]
pub enum ScoringMetric {
    /// LogLikelihood
    LogLikelihood,
    /// Accuracy
    Accuracy,
    /// CrossValidation
    CrossValidation,
}

impl KernelParameterLearner {
    /// Create new parameter learner
    pub fn new(kernel_candidates: Vec<KernelType>, cv_folds: usize) -> Self {
        Self {
            kernel_candidates,
            cv_folds,
            scoring_metric: ScoringMetric::LogLikelihood,
        }
    }

    /// Find best kernel parameters through cross-validation
    pub fn find_best_kernel(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<KernelType, KernelError> {
        let mut best_kernel = self.kernel_candidates[0].clone();
        let mut best_score = f64::NEG_INFINITY;

        for kernel in &self.kernel_candidates {
            let score = self.evaluate_kernel(kernel, x, y)?;
            if score > best_score {
                best_score = score;
                best_kernel = kernel.clone();
            }
        }

        Ok(best_kernel)
    }

    fn evaluate_kernel(
        &self,
        kernel: &KernelType,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<f64, KernelError> {
        let n_samples = x.nrows();
        let fold_size = n_samples / self.cv_folds;
        let mut total_score = 0.0;

        for fold in 0..self.cv_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Split data
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..n_samples {
                if i >= start_idx && i < end_idx {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            // Create training and test sets
            let train_data: Vec<f64> = train_indices
                .iter()
                .flat_map(|&i| x.row(i).to_vec())
                .collect();
            let x_train = Array2::from_shape_vec((train_indices.len(), x.ncols()), train_data)
                .map_err(|e| {
                    KernelError::NumericalError(format!("Array construction failed: {}", e))
                })?;

            let y_train = Array1::from_vec(train_indices.iter().map(|&i| y[i]).collect());

            let test_data: Vec<f64> = test_indices
                .iter()
                .flat_map(|&i| x.row(i).to_vec())
                .collect();
            let x_test = Array2::from_shape_vec((test_indices.len(), x.ncols()), test_data)
                .map_err(|e| {
                    KernelError::NumericalError(format!("Array construction failed: {}", e))
                })?;

            let y_test = Array1::from_vec(test_indices.iter().map(|&i| y[i]).collect());

            // Train and evaluate
            let mut classifier = KernelNaiveBayes::new(kernel.clone());
            classifier.fit(&x_train, &y_train)?;

            let predictions = classifier.predict(&x_test)?;
            let fold_score = self.compute_score(&y_test, &predictions);
            total_score += fold_score;
        }

        Ok(total_score / self.cv_folds as f64)
    }

    fn compute_score(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> f64 {
        match self.scoring_metric {
            ScoringMetric::Accuracy => {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&true_label, &pred_label)| true_label == pred_label)
                    .count();
                correct as f64 / y_true.len() as f64
            }
            ScoringMetric::LogLikelihood => {
                // Simplified log-likelihood computation
                let accuracy = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&true_label, &pred_label)| true_label == pred_label)
                    .count() as f64
                    / y_true.len() as f64;
                accuracy.ln()
            }
            ScoringMetric::CrossValidation => {
                // Use accuracy for CV scoring
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&true_label, &pred_label)| true_label == pred_label)
                    .count();
                correct as f64 / y_true.len() as f64
            }
        }
    }
}

/// Reproducing Kernel Hilbert Space (RKHS) implementation
///
/// This module provides a mathematical framework for working with function spaces
/// induced by kernels, enabling theoretical analysis and advanced kernel methods.
#[derive(Debug, Clone)]
pub struct ReproducingKernelHilbertSpace<F: Float + 'static> {
    kernel: KernelType,
    training_data: Option<Array2<F>>,
    feature_map_dim: usize,
    regularization: F,
    _phantom: PhantomData<F>,
}

impl<F: Float + 'static> ReproducingKernelHilbertSpace<F> {
    /// Create new RKHS with specified kernel
    pub fn new(kernel: KernelType, regularization: F) -> Self {
        Self {
            kernel,
            training_data: None,
            feature_map_dim: 0,
            regularization,
            _phantom: PhantomData,
        }
    }

    /// Set training data for RKHS computations
    pub fn set_training_data(&mut self, data: &Array2<F>) -> Result<(), KernelError> {
        self.training_data = Some(data.clone());
        self.feature_map_dim = data.nrows();
        Ok(())
    }

    /// Compute feature map φ(x) for given input in RKHS
    /// Returns the coefficients for representing x in the RKHS basis
    pub fn feature_map(&self, x: &Array1<F>) -> Result<Array1<F>, KernelError> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| KernelError::InvalidParameters("Training data not set".to_string()))?;

        let mut feature_map = Array1::zeros(training_data.nrows());

        for (i, train_point) in training_data.outer_iter().enumerate() {
            let kernel_val = self.kernel.compute(x, &train_point.to_owned())?;
            feature_map[i] = kernel_val;
        }

        Ok(feature_map)
    }

    /// Compute RKHS inner product between two functions represented by their coefficients
    pub fn inner_product(&self, alpha: &Array1<F>, beta: &Array1<F>) -> Result<F, KernelError> {
        if alpha.len() != beta.len() {
            return Err(KernelError::DimensionMismatch {
                expected: alpha.len(),
                actual: beta.len(),
            });
        }

        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| KernelError::InvalidParameters("Training data not set".to_string()))?;

        // Compute Gram matrix
        let gram_matrix = self.kernel.compute_matrix(training_data, None)?;

        // Inner product: <f,g>_H = α^T K β
        let mut result = F::zero();
        for i in 0..alpha.len() {
            for j in 0..beta.len() {
                result = result + alpha[i] * gram_matrix[[i, j]] * beta[j];
            }
        }

        Ok(result)
    }

    /// Compute RKHS norm of a function represented by coefficients
    pub fn norm(&self, alpha: &Array1<F>) -> Result<F, KernelError> {
        let inner_prod = self.inner_product(alpha, alpha)?;
        Ok(inner_prod.sqrt())
    }

    /// Compute regularized RKHS norm with L2 regularization
    pub fn regularized_norm(&self, alpha: &Array1<F>) -> Result<F, KernelError> {
        let rkhs_norm = self.norm(alpha)?;
        let l2_norm = alpha.dot(alpha).sqrt();
        Ok(rkhs_norm + self.regularization * l2_norm)
    }

    /// Project a function onto the RKHS subspace spanned by training data
    pub fn project(&self, target_function: &Array1<F>) -> Result<Array1<F>, KernelError> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| KernelError::InvalidParameters("Training data not set".to_string()))?;

        if target_function.len() != training_data.nrows() {
            return Err(KernelError::DimensionMismatch {
                expected: training_data.nrows(),
                actual: target_function.len(),
            });
        }

        // Compute Gram matrix
        let gram_matrix = self.kernel.compute_matrix(training_data, None)?;

        // Add regularization to diagonal
        let mut regularized_gram = gram_matrix;
        for i in 0..regularized_gram.nrows() {
            regularized_gram[[i, i]] = regularized_gram[[i, i]] + self.regularization;
        }

        // Solve for projection coefficients: (K + λI)α = y
        // Simplified solution using least squares approximation
        let mut projection_coeffs = Array1::zeros(training_data.nrows());

        // Use simplified iterative solver (in practice, use proper linear solver)
        for _ in 0..100 {
            let mut new_coeffs = Array1::zeros(training_data.nrows());

            for i in 0..training_data.nrows() {
                let mut sum = F::zero();
                for j in 0..training_data.nrows() {
                    if i != j {
                        sum = sum + regularized_gram[[i, j]] * projection_coeffs[j];
                    }
                }
                new_coeffs[i] = (target_function[i] - sum) / regularized_gram[[i, i]];
            }

            projection_coeffs = new_coeffs;
        }

        Ok(projection_coeffs)
    }

    /// Evaluate the function represented by coefficients at given points
    pub fn evaluate_function(
        &self,
        coeffs: &Array1<F>,
        eval_points: &Array2<F>,
    ) -> Result<Array1<F>, KernelError> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| KernelError::InvalidParameters("Training data not set".to_string()))?;

        if coeffs.len() != training_data.nrows() {
            return Err(KernelError::DimensionMismatch {
                expected: training_data.nrows(),
                actual: coeffs.len(),
            });
        }

        let mut evaluations = Array1::zeros(eval_points.nrows());

        for (i, eval_point) in eval_points.outer_iter().enumerate() {
            let mut function_value = F::zero();

            for (j, train_point) in training_data.outer_iter().enumerate() {
                let kernel_val = self
                    .kernel
                    .compute(&eval_point.to_owned(), &train_point.to_owned())?;
                function_value = function_value + coeffs[j] * kernel_val;
            }

            evaluations[i] = function_value;
        }

        Ok(evaluations)
    }

    /// Compute the representer theorem solution for supervised learning
    /// Given training data (X, y), finds optimal function in RKHS
    pub fn representer_theorem_solution(&self, y: &Array1<F>) -> Result<Array1<F>, KernelError> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| KernelError::InvalidParameters("Training data not set".to_string()))?;

        if y.len() != training_data.nrows() {
            return Err(KernelError::DimensionMismatch {
                expected: training_data.nrows(),
                actual: y.len(),
            });
        }

        // The solution is given by: α = (K + λI)^(-1) y
        self.project(y)
    }

    /// Compute kernel alignment between two kernels
    pub fn kernel_alignment(&self, other_kernel: &KernelType) -> Result<F, KernelError> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| KernelError::InvalidParameters("Training data not set".to_string()))?;

        let k1 = self.kernel.compute_matrix(training_data, None)?;
        let k2 = other_kernel.compute_matrix(training_data, None)?;

        // Compute Frobenius inner product
        let mut numerator = F::zero();
        let mut k1_norm_sq = F::zero();
        let mut k2_norm_sq = F::zero();

        for i in 0..k1.nrows() {
            for j in 0..k1.ncols() {
                numerator = numerator + k1[[i, j]] * k2[[i, j]];
                k1_norm_sq = k1_norm_sq + k1[[i, j]] * k1[[i, j]];
                k2_norm_sq = k2_norm_sq + k2[[i, j]] * k2[[i, j]];
            }
        }

        let denominator = (k1_norm_sq * k2_norm_sq).sqrt();

        if denominator.is_zero() {
            return Err(KernelError::NumericalError("Zero kernel norms".to_string()));
        }

        Ok(numerator / denominator)
    }

    /// Compute effective dimension of the RKHS for given regularization
    pub fn effective_dimension(&self) -> Result<F, KernelError> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| KernelError::InvalidParameters("Training data not set".to_string()))?;

        let gram_matrix = self.kernel.compute_matrix(training_data, None)?;

        // Compute trace of K(K + λI)^(-1)
        // Simplified approximation: tr(K) / (1 + λ)
        let mut trace = F::zero();
        for i in 0..gram_matrix.nrows() {
            trace = trace + gram_matrix[[i, i]];
        }

        Ok(trace / (F::one() + self.regularization))
    }

    /// Compute kernel matrix eigenvalues (simplified implementation)
    pub fn kernel_eigenvalues(&self) -> Result<Array1<F>, KernelError> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| KernelError::InvalidParameters("Training data not set".to_string()))?;

        let gram_matrix = self.kernel.compute_matrix(training_data, None)?;

        // Simplified eigenvalue computation (in practice, use proper eigendecomposition)
        let mut eigenvalues = Array1::zeros(gram_matrix.nrows());

        // Approximate eigenvalues using diagonal dominance
        for i in 0..gram_matrix.nrows() {
            let mut row_sum = F::zero();
            for j in 0..gram_matrix.ncols() {
                row_sum = row_sum + gram_matrix[[i, j]].abs();
            }
            eigenvalues[i] = row_sum / F::from(gram_matrix.ncols()).unwrap();
        }

        Ok(eigenvalues)
    }
}

/// RKHS-based feature selection using kernel methods
#[derive(Debug, Clone)]
pub struct RKHSFeatureSelector<F: Float + 'static> {
    rkhs: ReproducingKernelHilbertSpace<F>,
    selected_features: Vec<usize>,
    feature_importance: Array1<F>,
}

impl<F: Float + 'static> RKHSFeatureSelector<F> {
    /// Create new RKHS feature selector
    pub fn new(kernel: KernelType, regularization: F) -> Self {
        Self {
            rkhs: ReproducingKernelHilbertSpace::new(kernel, regularization),
            selected_features: Vec::new(),
            feature_importance: Array1::zeros(0),
        }
    }

    /// Select features based on RKHS importance
    pub fn select_features(
        &mut self,
        x: &Array2<F>,
        y: &Array1<F>,
        n_features: usize,
    ) -> Result<Vec<usize>, KernelError> {
        if x.ncols() == 0 {
            return Ok(Vec::new());
        }

        self.rkhs.set_training_data(x)?;
        let solution = self.rkhs.representer_theorem_solution(y)?;

        // Compute feature importance based on RKHS norm
        let mut importance = Array1::zeros(x.ncols());

        for feature_idx in 0..x.ncols() {
            // Create feature-specific kernel matrix
            let feature_data = x.column(feature_idx).to_owned().insert_axis(Axis(1));
            let feature_kernel = self.rkhs.kernel.compute_matrix(&feature_data, None)?;

            // Compute contribution to RKHS norm
            let mut feature_contribution = F::zero();
            for i in 0..solution.len() {
                for j in 0..solution.len() {
                    feature_contribution =
                        feature_contribution + solution[i] * feature_kernel[[i, j]] * solution[j];
                }
            }

            importance[feature_idx] = feature_contribution.abs();
        }

        // Select top-k features
        let mut feature_indices: Vec<(usize, F)> = importance
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        feature_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        self.selected_features = feature_indices
            .into_iter()
            .take(n_features.min(x.ncols()))
            .map(|(idx, _)| idx)
            .collect();

        self.feature_importance = importance;

        Ok(self.selected_features.clone())
    }

    /// Get feature importance scores
    pub fn get_feature_importance(&self) -> &Array1<F> {
        &self.feature_importance
    }

    /// Get selected features
    pub fn get_selected_features(&self) -> &[usize] {
        &self.selected_features
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy Compliance - Use scirs2-core for random functionality
    use scirs2_core::random::CoreRandom;
    use scirs2_core::random::SeedableRng;

    #[test]
    fn test_rbf_kernel() {
        let kernel = KernelType::RBF { gamma: 1.0 };
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);

        let result = kernel.compute(&x, &y).unwrap();
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kernel_matrix() {
        let kernel = KernelType::Linear;
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let matrix = kernel.compute_matrix(&x, None).unwrap();
        assert_eq!(matrix.shape(), &[2, 2]);
        assert_abs_diff_eq!(matrix[[0, 0]], 5.0, epsilon = 1e-10); // [1,2] · [1,2] = 5
        assert_abs_diff_eq!(matrix[[1, 1]], 25.0, epsilon = 1e-10); // [3,4] · [3,4] = 25
    }

    #[test]
    fn test_kde_fit_and_score() {
        let config = KDEConfig::default();
        let mut kde = KernelDensityEstimator::new(config);

        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        kde.fit(&data).unwrap();

        let test_data = Array2::from_shape_vec((2, 2), vec![2.5, 2.5, 1.5, 1.5]).unwrap();

        let scores = kde.score_samples(&test_data).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
    }

    #[test]
    fn test_kernel_naive_bayes() {
        let kernel = KernelType::RBF { gamma: 1.0 };
        let mut classifier = KernelNaiveBayes::new(kernel);

        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // class 0
                1.1, 1.1, // class 0
                1.2, 1.2, // class 0
                5.0, 5.0, // class 1
                5.1, 5.1, // class 1
                5.2, 5.2, // class 1
            ],
        )
        .unwrap();

        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        classifier.fit(&x, &y).unwrap();

        let test_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.05, 1.05, // Should be class 0
                5.05, 5.05, // Should be class 1
            ],
        )
        .unwrap();

        let predictions = classifier.predict(&test_x).unwrap();
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_gaussian_process() {
        let kernel = KernelType::RBF { gamma: 1.0 };
        let mut gp = GaussianProcess::new(kernel, 0.1);

        let x_train = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y_train = Array1::from_vec(vec![1.0, 4.0, 9.0]); // y = x^2

        gp.fit(&x_train, &y_train).unwrap();

        let x_test = Array2::from_shape_vec((2, 1), vec![1.5, 2.5]).unwrap();
        let (mean, variance) = gp.predict(&x_test).unwrap();

        assert_eq!(mean.len(), 2);
        assert_eq!(variance.len(), 2);
        assert!(variance.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_bandwidth_methods() {
        let data = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64).collect()).unwrap();

        let scott_bw = BandwidthMethod::Scott.compute_bandwidth(&data);
        let silverman_bw = BandwidthMethod::Silverman.compute_bandwidth(&data);
        let manual_bw = BandwidthMethod::Manual(0.5).compute_bandwidth(&data);

        assert!(scott_bw > 0.0);
        assert!(silverman_bw > 0.0);
        assert_abs_diff_eq!(manual_bw, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_kernel_parameter_learning() {
        let kernels = vec![
            KernelType::RBF { gamma: 0.1 },
            KernelType::RBF { gamma: 1.0 },
            KernelType::RBF { gamma: 10.0 },
            KernelType::Linear,
        ];

        let learner = KernelParameterLearner::new(kernels, 3);

        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let best_kernel = learner.find_best_kernel(&x, &y).unwrap();

        // Should find a reasonable kernel
        match best_kernel {
            KernelType::RBF { gamma } => assert!(gamma > 0.0),
            KernelType::Linear => {}
            _ => panic!("Unexpected kernel type"),
        }
    }

    #[test]
    fn test_kde_sampling() {
        let config = KDEConfig::default();
        let mut kde = KernelDensityEstimator::new(config);

        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        kde.fit(&data).unwrap();

        let mut rng = scirs2_core::random::CoreRandom::seed_from_u64(42);
        let samples = kde.sample(10, &mut rng).unwrap();

        assert_eq!(samples.shape(), &[10, 2]);
        assert!(samples.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_rkhs_basic_functionality() {
        let kernel = KernelType::RBF { gamma: 1.0 };
        let mut rkhs = ReproducingKernelHilbertSpace::new(kernel, 0.01);

        let training_data =
            Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();

        rkhs.set_training_data(&training_data).unwrap();

        // Test feature map
        let test_point = Array1::from_vec(vec![2.5, 2.5]);
        let feature_map = rkhs.feature_map(&test_point).unwrap();
        assert_eq!(feature_map.len(), 3);
        assert!(feature_map.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_rkhs_inner_product_and_norm() {
        let kernel = KernelType::Linear;
        let mut rkhs = ReproducingKernelHilbertSpace::new(kernel, 0.1);

        let training_data = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        rkhs.set_training_data(&training_data).unwrap();

        let alpha = Array1::from_vec(vec![1.0, 1.0]);
        let beta = Array1::from_vec(vec![1.0, -1.0]);

        // Test inner product
        let inner_prod = rkhs.inner_product(&alpha, &beta).unwrap();
        assert!(inner_prod.is_finite());

        // Test norm
        let norm_alpha = rkhs.norm(&alpha).unwrap();
        assert!(norm_alpha > 0.0);
        assert!(norm_alpha.is_finite());

        // Test regularized norm
        let reg_norm = rkhs.regularized_norm(&alpha).unwrap();
        assert!(reg_norm >= norm_alpha);
    }

    #[test]
    fn test_rkhs_function_evaluation() {
        let kernel = KernelType::RBF { gamma: 0.5 };
        let mut rkhs = ReproducingKernelHilbertSpace::new(kernel, 0.01);

        let training_data = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        rkhs.set_training_data(&training_data).unwrap();

        let coeffs = Array1::from_vec(vec![1.0, -0.5, 0.2]);
        let eval_points = Array2::from_shape_vec((2, 1), vec![1.5, 2.5]).unwrap();

        let evaluations = rkhs.evaluate_function(&coeffs, &eval_points).unwrap();
        assert_eq!(evaluations.len(), 2);
        assert!(evaluations.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_rkhs_kernel_alignment() {
        let kernel1 = KernelType::RBF { gamma: 1.0 };
        let kernel2 = KernelType::Linear;
        let mut rkhs = ReproducingKernelHilbertSpace::new(kernel1, 0.1);

        let training_data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

        rkhs.set_training_data(&training_data).unwrap();

        let alignment = rkhs.kernel_alignment(&kernel2).unwrap();
        assert!(alignment >= -1.0 && alignment <= 1.0);
        assert!(alignment.is_finite());
    }

    #[test]
    fn test_rkhs_effective_dimension() {
        let kernel = KernelType::RBF { gamma: 1.0 };
        let mut rkhs = ReproducingKernelHilbertSpace::new(kernel, 0.1);

        let training_data =
            Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();

        rkhs.set_training_data(&training_data).unwrap();

        let eff_dim = rkhs.effective_dimension().unwrap();
        assert!(eff_dim > 0.0);
        assert!(eff_dim.is_finite());
    }

    #[test]
    fn test_rkhs_feature_selector() {
        let kernel = KernelType::Linear;
        let mut selector = RKHSFeatureSelector::new(kernel, 0.1);

        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, // Feature 2 most important
                2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0,
            ],
        )
        .unwrap();

        let y = Array1::from_vec(vec![3.0, 6.0, 9.0, 12.0]); // y = x2 (feature 2)

        let selected = selector.select_features(&x, &y, 2).unwrap();
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&2)); // Feature 2 should be selected

        let importance = selector.get_feature_importance();
        assert_eq!(importance.len(), 3);
        assert!(importance.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_rkhs_representer_theorem() {
        let kernel = KernelType::RBF { gamma: 1.0 };
        let mut rkhs = ReproducingKernelHilbertSpace::new(kernel, 0.01);

        let training_data =
            Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();

        rkhs.set_training_data(&training_data).unwrap();

        let target_values = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let solution = rkhs.representer_theorem_solution(&target_values).unwrap();

        assert_eq!(solution.len(), 3);
        assert!(solution.iter().all(|&x| x.is_finite()));

        // Test that solution gives reasonable reconstruction
        let reconstructed = rkhs.evaluate_function(&solution, &training_data).unwrap();
        assert_eq!(reconstructed.len(), 3);

        // Check that reconstruction is close to target (within reasonable tolerance)
        for i in 0..3 {
            assert!((reconstructed[i] - target_values[i]).abs() < 1.0);
        }
    }
}
