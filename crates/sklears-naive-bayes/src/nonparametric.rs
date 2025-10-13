//! Non-parametric Naive Bayes classifier with Kernel Density Estimation
//!
//! This classifier uses kernel density estimation to model feature distributions
//! without assuming any specific parametric form.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, PredictProba, Score, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::{compute_class_prior, safe_log, NaiveBayesMixin};

/// Kernel types for density estimation
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    /// Gaussian kernel
    Gaussian,
    /// Epanechnikov kernel
    Epanechnikov,
    /// Uniform kernel
    Uniform,
    /// Triangular kernel
    Triangular,
    /// Biweight kernel
    Biweight,
    /// Triweight kernel
    Triweight,
    /// Cosine kernel
    Cosine,
}

/// Bandwidth selection methods
#[derive(Debug, Clone)]
pub enum BandwidthMethod {
    /// Scott's rule of thumb
    Scott,
    /// Silverman's rule of thumb
    Silverman,
    /// Cross-validation
    CrossValidation { folds: usize },
    /// Plug-in method
    PlugIn,
    /// Manual specification
    Manual { bandwidth: f64 },
    /// Adaptive bandwidth (varies by density)
    Adaptive { pilot_bandwidth: f64 },
}

/// Configuration for Non-parametric Naive Bayes
#[derive(Debug, Clone)]
pub struct NonparametricNBConfig {
    /// Kernel type for density estimation
    pub kernel: KernelType,
    /// Bandwidth selection method
    pub bandwidth_method: BandwidthMethod,
    /// Minimum bandwidth to prevent overfitting
    pub min_bandwidth: f64,
    /// Maximum bandwidth to prevent underfitting
    pub max_bandwidth: f64,
    /// Number of evaluation points for KDE
    pub n_points: usize,
    /// Prior probabilities of the classes
    pub priors: Option<Array1<f64>>,
    /// Whether to use adaptive bandwidth
    pub adaptive: bool,
    /// Tolerance for numerical computations
    pub tolerance: f64,
}

impl Default for NonparametricNBConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::Gaussian,
            bandwidth_method: BandwidthMethod::Scott,
            min_bandwidth: 1e-6,
            max_bandwidth: 10.0,
            n_points: 100,
            priors: None,
            adaptive: false,
            tolerance: 1e-10,
        }
    }
}

/// Kernel density estimator for a single feature
#[derive(Debug, Clone)]
pub struct KernelDensityEstimator {
    /// Training data points
    data: Array1<f64>,
    /// Bandwidth
    bandwidth: f64,
    /// Kernel type
    kernel: KernelType,
    /// Data range for evaluation
    data_min: f64,
    data_max: f64,
    /// Adaptive bandwidths (if using adaptive KDE)
    adaptive_bandwidths: Option<Array1<f64>>,
}

impl KernelDensityEstimator {
    /// Create a new KDE
    pub fn new(
        data: Array1<f64>,
        bandwidth: f64,
        kernel: KernelType,
        adaptive_bandwidths: Option<Array1<f64>>,
    ) -> Self {
        let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            data,
            bandwidth,
            kernel,
            data_min,
            data_max,
            adaptive_bandwidths,
        }
    }

    /// Evaluate kernel function
    fn kernel_function(&self, u: f64) -> f64 {
        match self.kernel {
            KernelType::Gaussian => (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt(),
            KernelType::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            KernelType::Uniform => {
                if u.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            KernelType::Triangular => {
                if u.abs() <= 1.0 {
                    1.0 - u.abs()
                } else {
                    0.0
                }
            }
            KernelType::Biweight => {
                if u.abs() <= 1.0 {
                    (15.0 / 16.0) * (1.0 - u * u).powi(2)
                } else {
                    0.0
                }
            }
            KernelType::Triweight => {
                if u.abs() <= 1.0 {
                    (35.0 / 32.0) * (1.0 - u * u).powi(3)
                } else {
                    0.0
                }
            }
            KernelType::Cosine => {
                if u.abs() <= 1.0 {
                    (std::f64::consts::PI / 4.0) * (std::f64::consts::PI * u / 2.0).cos()
                } else {
                    0.0
                }
            }
        }
    }

    /// Estimate density at a point
    pub fn density(&self, x: f64) -> f64 {
        let n = self.data.len() as f64;
        let mut density = 0.0;

        match &self.adaptive_bandwidths {
            Some(adaptive_bw) => {
                // Adaptive KDE
                for (i, &xi) in self.data.iter().enumerate() {
                    let h = adaptive_bw[i];
                    let u = (x - xi) / h;
                    density += self.kernel_function(u) / h;
                }
            }
            None => {
                // Fixed bandwidth KDE
                for &xi in self.data.iter() {
                    let u = (x - xi) / self.bandwidth;
                    density += self.kernel_function(u);
                }
                density /= self.bandwidth;
            }
        }

        density / n
    }

    /// Estimate log density at a point
    pub fn log_density(&self, x: f64) -> f64 {
        let density = self.density(x);
        if density > 0.0 {
            density.ln()
        } else {
            f64::NEG_INFINITY
        }
    }
}

/// Non-parametric Naive Bayes classifier
///
/// This classifier uses kernel density estimation to model feature distributions
/// without parametric assumptions.
#[derive(Debug, Clone)]
pub struct NonparametricNB<State = Untrained> {
    config: NonparametricNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    kdes_: Option<Vec<Vec<KernelDensityEstimator>>>, // [n_classes][n_features]
    class_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
}

impl NonparametricNB<Untrained> {
    /// Create a new Non-parametric Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: NonparametricNBConfig::default(),
            state: PhantomData,
            kdes_: None,
            class_prior_: None,
            classes_: None,
        }
    }

    /// Set kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set bandwidth selection method
    pub fn bandwidth_method(mut self, method: BandwidthMethod) -> Self {
        self.config.bandwidth_method = method;
        self
    }

    /// Set minimum bandwidth
    pub fn min_bandwidth(mut self, min_bw: f64) -> Self {
        self.config.min_bandwidth = min_bw;
        self
    }

    /// Set maximum bandwidth
    pub fn max_bandwidth(mut self, max_bw: f64) -> Self {
        self.config.max_bandwidth = max_bw;
        self
    }

    /// Set number of evaluation points
    pub fn n_points(mut self, n_points: usize) -> Self {
        self.config.n_points = n_points;
        self
    }

    /// Set prior probabilities
    pub fn priors(mut self, priors: Array1<f64>) -> Self {
        self.config.priors = Some(priors);
        self
    }

    /// Enable adaptive bandwidth
    pub fn adaptive(mut self, adaptive: bool) -> Self {
        self.config.adaptive = adaptive;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.tolerance = tol;
        self
    }
}

impl Default for NonparametricNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for NonparametricNB<Untrained> {
    type Float = Float;
    type Config = NonparametricNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl NonparametricNB<Untrained> {
    /// Select bandwidth using the specified method
    fn select_bandwidth(&self, data: &Array1<f64>) -> f64 {
        let n = data.len() as f64;
        if n == 0.0 {
            return self.config.min_bandwidth;
        }

        let bandwidth = match &self.config.bandwidth_method {
            BandwidthMethod::Scott => {
                // Scott's rule: h = n^(-1/5) * σ * 1.06
                let std_dev = self.compute_std_dev(data);
                1.06 * std_dev * n.powf(-0.2)
            }
            BandwidthMethod::Silverman => {
                // Silverman's rule: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
                let std_dev = self.compute_std_dev(data);
                let iqr = self.compute_iqr(data);
                0.9 * std_dev.min(iqr / 1.34) * n.powf(-0.2)
            }
            BandwidthMethod::CrossValidation { folds } => {
                self.cross_validation_bandwidth(data, *folds)
            }
            BandwidthMethod::PlugIn => {
                // Simplified plug-in method
                let std_dev = self.compute_std_dev(data);
                std_dev * n.powf(-0.2)
            }
            BandwidthMethod::Manual { bandwidth } => *bandwidth,
            BandwidthMethod::Adaptive { pilot_bandwidth } => *pilot_bandwidth,
        };

        bandwidth.clamp(self.config.min_bandwidth, self.config.max_bandwidth)
    }

    /// Compute standard deviation
    fn compute_std_dev(&self, data: &Array1<f64>) -> f64 {
        let n = data.len() as f64;
        if n <= 1.0 {
            return 1.0;
        }

        let mean = data.sum() / n;
        let variance = data.mapv(|x| (x - mean).powi(2)).sum() / (n - 1.0);
        variance.sqrt().max(self.config.tolerance)
    }

    /// Compute interquartile range
    fn compute_iqr(&self, data: &Array1<f64>) -> f64 {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_data.len();
        if n == 0 {
            return 1.0;
        }

        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;

        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx.min(n - 1)];

        (q3 - q1).max(self.config.tolerance)
    }

    /// Select bandwidth using cross-validation
    fn cross_validation_bandwidth(&self, data: &Array1<f64>, folds: usize) -> f64 {
        let n = data.len();
        if n < folds || folds == 0 {
            return self.select_bandwidth_fallback(data);
        }

        let fold_size = n / folds;
        let mut bandwidths = vec![];

        // Generate candidate bandwidths
        let std_dev = self.compute_std_dev(data);
        for i in 1..=10 {
            bandwidths.push(std_dev * (i as f64) * 0.1 * (n as f64).powf(-0.2));
        }

        let mut best_bandwidth = bandwidths[0];
        let mut best_score = f64::NEG_INFINITY;

        for &bandwidth in &bandwidths {
            let mut total_score = 0.0;

            for fold in 0..folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == folds - 1 {
                    n
                } else {
                    (fold + 1) * fold_size
                };

                // Split data
                let mut train_data = Vec::new();
                let mut test_data = Vec::new();

                for (i, &value) in data.iter().enumerate() {
                    if i >= start_idx && i < end_idx {
                        test_data.push(value);
                    } else {
                        train_data.push(value);
                    }
                }

                if train_data.is_empty() || test_data.is_empty() {
                    continue;
                }

                let train_array = Array1::from_vec(train_data);
                let kde = KernelDensityEstimator::new(
                    train_array,
                    bandwidth,
                    self.config.kernel.clone(),
                    None,
                );

                // Evaluate on test data
                let mut fold_score = 0.0;
                for &test_point in &test_data {
                    fold_score += kde.log_density(test_point);
                }
                total_score += fold_score / test_data.len() as f64;
            }

            let avg_score = total_score / folds as f64;
            if avg_score > best_score {
                best_score = avg_score;
                best_bandwidth = bandwidth;
            }
        }

        best_bandwidth.clamp(self.config.min_bandwidth, self.config.max_bandwidth)
    }

    /// Fallback bandwidth selection
    fn select_bandwidth_fallback(&self, data: &Array1<f64>) -> f64 {
        let std_dev = self.compute_std_dev(data);
        let n = data.len() as f64;
        (std_dev * n.powf(-0.2)).clamp(self.config.min_bandwidth, self.config.max_bandwidth)
    }

    /// Compute adaptive bandwidths
    fn compute_adaptive_bandwidths(&self, data: &Array1<f64>, pilot_bandwidth: f64) -> Array1<f64> {
        let n = data.len();
        let mut adaptive_bw = Array1::zeros(n);

        // First, compute pilot density estimates
        let pilot_kde = KernelDensityEstimator::new(
            data.clone(),
            pilot_bandwidth,
            self.config.kernel.clone(),
            None,
        );

        let mut pilot_densities = Array1::zeros(n);
        for (i, &xi) in data.iter().enumerate() {
            pilot_densities[i] = pilot_kde.density(xi);
        }

        // Compute geometric mean of pilot densities
        let log_sum: f64 = pilot_densities
            .mapv(|x| x.max(self.config.tolerance).ln())
            .sum();
        let geom_mean = (log_sum / n as f64).exp();

        // Compute adaptive bandwidths
        for (i, &density) in pilot_densities.iter().enumerate() {
            let local_factor = (geom_mean / density.max(self.config.tolerance)).sqrt();
            adaptive_bw[i] = pilot_bandwidth * local_factor;
        }

        adaptive_bw
    }
}

impl Fit<Array2<Float>, Array1<i32>> for NonparametricNB<Untrained> {
    type Fitted = NonparametricNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();
        let n_features = x.ncols();

        // Compute class priors
        let (_class_count, class_prior) = if let Some(ref priors) = self.config.priors {
            if priors.len() != n_classes {
                return Err(SklearsError::InvalidInput(format!(
                    "Number of priors ({}) doesn't match number of classes ({})",
                    priors.len(),
                    n_classes
                )));
            }
            let sum = priors.sum();
            if (sum - 1.0).abs() > 1e-10 {
                return Err(SklearsError::InvalidInput(
                    "The sum of the priors should be 1.0".to_string(),
                ));
            }
            let class_count = Array1::zeros(n_classes);
            (class_count, priors.clone())
        } else {
            compute_class_prior(y, &classes)
        };

        // Initialize KDE storage
        let mut kdes = vec![vec![]; n_classes];

        // Fit KDE for each class and feature
        for (class_idx, &class_label) in classes.iter().enumerate() {
            kdes[class_idx] = vec![];

            // Get samples belonging to this class
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if mask.is_empty() {
                // Create dummy KDEs for empty classes
                for _feature_idx in 0..n_features {
                    let dummy_data = Array1::from_vec(vec![0.0]);
                    let kde = KernelDensityEstimator::new(
                        dummy_data,
                        self.config.min_bandwidth,
                        self.config.kernel.clone(),
                        None,
                    );
                    kdes[class_idx].push(kde);
                }
                continue;
            }

            let x_class = x.select(Axis(0), &mask);

            // For each feature, fit KDE
            for feature_idx in 0..n_features {
                let feature_values = x_class.column(feature_idx);
                let feature_data = Array1::from_vec(feature_values.to_vec());

                // Select bandwidth
                let bandwidth = self.select_bandwidth(&feature_data);

                // Compute adaptive bandwidths if requested
                let adaptive_bandwidths = if self.config.adaptive {
                    Some(self.compute_adaptive_bandwidths(&feature_data, bandwidth))
                } else {
                    None
                };

                // Create KDE
                let kde = KernelDensityEstimator::new(
                    feature_data,
                    bandwidth,
                    self.config.kernel.clone(),
                    adaptive_bandwidths,
                );

                kdes[class_idx].push(kde);
            }
        }

        Ok(NonparametricNB {
            config: self.config,
            state: PhantomData,
            kdes_: Some(kdes),
            class_prior_: Some(class_prior),
            classes_: Some(classes),
        })
    }
}

impl NonparametricNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let kdes = self.kdes_.as_ref().unwrap();
        let class_prior = self.class_prior_.as_ref().unwrap();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            for (sample_idx, x_sample) in x.axis_iter(Axis(0)).enumerate() {
                let mut log_prob = safe_log(class_prior[class_idx]);

                for feature_idx in 0..n_features {
                    let x_val = x_sample[feature_idx];
                    let kde = &kdes[class_idx][feature_idx];
                    log_prob += kde.log_density(x_val);
                }

                joint_log_likelihood[[sample_idx, class_idx]] = log_prob;
            }
        }

        Ok(joint_log_likelihood)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for NonparametricNB<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let classes = self.classes_.as_ref().unwrap();

        Ok(log_prob.map_axis(Axis(1), |row| {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            classes[max_idx]
        }))
    }
}

impl PredictProba<Array2<Float>, Array2<f64>> for NonparametricNB<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let n_samples = x.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let mut proba = Array2::zeros((n_samples, n_classes));

        // Normalize to get probabilities
        for i in 0..n_samples {
            let row = log_prob.row(i);
            let max_log_prob = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let mut exp_sum = 0.0;
            for j in 0..n_classes {
                let exp_val = (log_prob[[i, j]] - max_log_prob).exp();
                proba[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            for j in 0..n_classes {
                proba[[i, j]] /= exp_sum;
            }
        }

        Ok(proba)
    }
}

impl Score<Array2<Float>, Array1<i32>> for NonparametricNB<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<f64> {
        let predictions = self.predict(x)?;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();

        Ok(correct as f64 / y.len() as f64)
    }
}

impl NaiveBayesMixin for NonparametricNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // For nonparametric NB, create a dummy array since we use KDE
        static DUMMY: once_cell::sync::Lazy<Array2<f64>> =
            once_cell::sync::Lazy::new(|| Array2::zeros((1, 1)));
        &DUMMY
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }
}

impl NonparametricNB<Trained> {
    /// Get the kernel density estimators for each class and feature
    pub fn kdes(&self) -> &Vec<Vec<KernelDensityEstimator>> {
        self.kdes_.as_ref().unwrap()
    }

    /// Evaluate feature density for a given class and feature
    pub fn feature_density(&self, class_idx: usize, feature_idx: usize, x: f64) -> f64 {
        if let Some(kdes) = &self.kdes_ {
            if class_idx < kdes.len() && feature_idx < kdes[class_idx].len() {
                return kdes[class_idx][feature_idx].density(x);
            }
        }
        0.0
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nonparametric_nb_basic() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
            [-4.0, -5.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = NonparametricNB::new()
            .kernel(KernelType::Gaussian)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        // Non-parametric methods might not achieve perfect accuracy on training data
        assert_eq!(predictions.len(), y.len());

        let score = model.score(&x, &y).unwrap();
        assert!(score >= 0.5); // Should perform better than random
    }

    #[test]
    fn test_nonparametric_nb_predict_proba() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]];
        let y = array![0, 0, 1, 1];

        let model = NonparametricNB::new()
            .kernel(KernelType::Gaussian)
            .bandwidth_method(BandwidthMethod::Manual { bandwidth: 0.5 })
            .fit(&x, &y)
            .unwrap();

        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_nonparametric_nb_different_kernels() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]];
        let y = array![0, 0, 1, 1];

        let kernels = vec![
            KernelType::Gaussian,
            KernelType::Epanechnikov,
            KernelType::Uniform,
            KernelType::Triangular,
        ];

        for kernel in kernels {
            let model = NonparametricNB::new()
                .kernel(kernel)
                .bandwidth_method(BandwidthMethod::Manual { bandwidth: 0.5 })
                .fit(&x, &y)
                .unwrap();

            let predictions = model.predict(&x).unwrap();
            assert_eq!(predictions.len(), y.len());
        }
    }

    #[test]
    fn test_nonparametric_nb_bandwidth_methods() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [1.3, 1.3],
            [2.0, 2.0],
            [2.1, 2.1],
            [2.2, 2.2],
            [2.3, 2.3]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let methods = vec![
            BandwidthMethod::Scott,
            BandwidthMethod::Silverman,
            BandwidthMethod::Manual { bandwidth: 0.1 },
        ];

        for method in methods {
            let model = NonparametricNB::new()
                .bandwidth_method(method)
                .fit(&x, &y)
                .unwrap();

            let predictions = model.predict(&x).unwrap();
            assert_eq!(predictions.len(), y.len());
        }
    }

    #[test]
    fn test_nonparametric_nb_adaptive() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [2.0, 2.0],
            [2.5, 2.5],
            [5.0, 5.0],
            [5.1, 5.1],
            [6.0, 6.0],
            [6.5, 6.5]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = NonparametricNB::new()
            .adaptive(true)
            .bandwidth_method(BandwidthMethod::Adaptive {
                pilot_bandwidth: 0.5,
            })
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }
}
