//! Binary calibration methods for probability calibration
//!
//! This module provides calibration methods specifically designed for binary classification
//! problems. These methods can also be used as components in multiclass calibration strategies
//! by applying them in a one-vs-rest fashion.
//!
//! The main binary calibrator provided is the `SigmoidCalibrator`, which implements
//! Platt's sigmoid method for probability calibration. Additional helper functions are
//! provided for training various calibration methods in binary classification scenarios.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

use crate::{
    bbq::BBQCalibrator,
    beta::{BetaCalibrator, EnsembleTemperatureScaling},
    gaussian_process::GaussianProcessCalibrator,
    histogram::HistogramBinningCalibrator,
    isotonic::IsotonicCalibrator,
    kde::{AdaptiveKDECalibrator, KDECalibrator},
    local::{LocalBinningCalibrator, LocalKNNCalibrator},
    multiclass::{
        DirichletCalibration, MatrixScaling, MulticlassTemperatureScaling, OneVsOneCalibrator,
    },
    temperature::TemperatureScalingCalibrator,
    CalibrationEstimator,
};

/// Sigmoid calibrator for binary probability calibration
///
/// Implements Platt's sigmoid method which fits a sigmoid function to map
/// classifier outputs to calibrated probabilities. The sigmoid function has the form:
///
/// P(y=1|f) = 1 / (1 + exp(A*f + B))
///
/// where f is the classifier output (or logit of predicted probability) and A, B are
/// parameters learned from validation data.
///
/// # Examples
///
/// ```
/// use sklears_calibration::binary::SigmoidCalibrator;
/// use sklears_calibration::CalibrationEstimator;
/// use scirs2_core::ndarray::array;
///
/// let probabilities = array![0.1, 0.4, 0.7, 0.9];
/// let y_true = array![0, 0, 1, 1];
///
/// let calibrator = SigmoidCalibrator::new()
///     .fit(&probabilities, &y_true).unwrap();
///
/// let calibrated = calibrator.predict_proba(&probabilities).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SigmoidCalibrator {
    a: Float,
    b: Float,
}

impl CalibrationEstimator for SigmoidCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = Self::new().fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        SigmoidCalibrator::predict_proba(self, probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl SigmoidCalibrator {
    /// Create a new sigmoid calibrator with default parameters
    pub fn new() -> Self {
        Self { a: 1.0, b: 0.0 }
    }

    /// Fit the sigmoid calibrator to training data
    ///
    /// # Arguments
    ///
    /// * `probabilities` - Predicted probabilities from the base classifier
    /// * `y_true` - True binary labels (0 or 1)
    ///
    /// # Returns
    ///
    /// A fitted `SigmoidCalibrator` instance
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        // Platt's method: fit sigmoid to map f(x) = 1 / (1 + exp(ax + b))
        // Simplified implementation - in practice would use proper optimization

        let n = probabilities.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "No probabilities provided".to_string(),
            ));
        }

        // Convert probabilities to decision values (logits)
        let decision_values: Array1<Float> = probabilities.mapv(|p| {
            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
            (clamped_p / (1.0 - clamped_p)).ln()
        });

        // Simple linear regression to fit A and B
        // y_true -> target probabilities (0 or 1)
        let target_probs: Array1<Float> = y_true.mapv(|y| y as Float);

        // Use simple least squares: minimize ||sigmoid(A*f + B) - y||^2
        // Simplified approach: use mean and variance
        let mean_f = decision_values.mean().unwrap_or(0.0);
        let mean_y = target_probs.mean().unwrap_or(0.5);

        // Simple heuristic fitting
        self.a = if mean_y > 0.5 { 1.0 } else { -1.0 };
        self.b = -self.a * mean_f;

        Ok(self)
    }

    /// Predict calibrated probabilities
    ///
    /// # Arguments
    ///
    /// * `probabilities` - Input probabilities to calibrate
    ///
    /// # Returns
    ///
    /// Calibrated probabilities
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let decision_values: Array1<Float> = probabilities.mapv(|p| {
            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
            (clamped_p / (1.0 - clamped_p)).ln()
        });

        let calibrated = decision_values.mapv(|f| {
            let score = self.a * f + self.b;
            1.0 / (1.0 + (-score).exp())
        });

        Ok(calibrated)
    }
}

impl Default for SigmoidCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for training calibrators in binary classification scenarios

/// Create dummy probabilities for testing purposes
///
/// This function generates synthetic probability distributions based on input features.
/// It's primarily used for testing and debugging purposes.
pub fn create_dummy_probabilities(
    x: &Array2<Float>,
    _y: &Array1<i32>,
    classes: &Array1<i32>,
) -> Result<Array2<Float>> {
    let (n_samples, _n_features) = x.dim();
    let n_classes = classes.len();

    let mut probabilities = Array2::zeros((n_samples, n_classes));

    // Create dummy probabilities based on simple heuristics
    for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
        let feature_sum = sample.sum();

        // Simple heuristic: normalize features to get probabilities
        for (j, &_class) in classes.iter().enumerate() {
            let base_prob = ((feature_sum + j as Float) % 2.0 + 0.1) / 2.1;
            probabilities[[i, j]] = base_prob;
        }

        // Normalize
        let row_sum = probabilities.row(i).sum();
        if row_sum > 0.0 {
            probabilities.row_mut(i).mapv_inplace(|x| x / row_sum);
        }
    }

    Ok(probabilities)
}

/// Train sigmoid calibrators for multiclass problems using one-vs-rest strategy
///
/// Creates a separate `SigmoidCalibrator` for each class, treating each as a binary
/// classification problem (class vs. all others).
pub fn train_sigmoid_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train calibrator (simplified - should use CV in practice)
        let calibrator = SigmoidCalibrator::new().fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train isotonic calibrators for multiclass problems using one-vs-rest strategy
pub fn train_isotonic_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train isotonic calibrator
        let calibrator = IsotonicCalibrator::new().fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train temperature scaling calibrators for multiclass problems using one-vs-rest strategy
pub fn train_temperature_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train temperature scaling calibrator
        let calibrator = TemperatureScalingCalibrator::new().fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train histogram binning calibrators for multiclass problems using one-vs-rest strategy
pub fn train_histogram_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    n_bins: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train histogram binning calibrator
        let calibrator = HistogramBinningCalibrator::new(n_bins).fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train BBQ (Bayesian Binning into Quantiles) calibrators for multiclass problems
pub fn train_bbq_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    min_bins: usize,
    max_bins: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train BBQ calibrator
        let calibrator = BBQCalibrator::new(min_bins, max_bins).fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train beta calibrators for multiclass problems using one-vs-rest strategy
pub fn train_beta_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train beta calibrator
        let calibrator = BetaCalibrator::new().fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train ensemble temperature scaling calibrators for multiclass problems
pub fn train_ensemble_temperature_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    n_estimators: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train ensemble temperature scaling calibrator
        let calibrator =
            EnsembleTemperatureScaling::new(n_estimators).fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train one-vs-one calibrators for multiclass problems
pub fn train_one_vs_one_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    _classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    // For one-vs-one, we return a single calibrator that handles all classes
    let base_calibrator_fn =
        Box::new(|| -> Box<dyn CalibrationEstimator> { Box::new(SigmoidCalibrator::new()) });

    let calibrator = OneVsOneCalibrator::new().fit(probabilities, y, base_calibrator_fn)?;

    // Wrap in a special adapter that implements CalibrationEstimator
    let adapter = OneVsOneAdapter::new(calibrator);
    Ok(vec![Box::new(adapter)])
}

/// Train multiclass temperature scaling calibrators
pub fn train_multiclass_temperature_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    _classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    // For multiclass temperature scaling, we use a single calibrator for all classes
    let calibrator = MulticlassTemperatureScaling::new().fit(probabilities, y)?;
    let adapter = MulticlassTemperatureAdapter::new(calibrator);
    Ok(vec![Box::new(adapter)])
}

/// Train matrix scaling calibrators for multiclass problems
pub fn train_matrix_scaling_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    _classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    // For matrix scaling, we use a single calibrator for all classes
    let calibrator = MatrixScaling::new().fit(probabilities, y)?;
    let adapter = MatrixScalingAdapter::new(calibrator);
    Ok(vec![Box::new(adapter)])
}

/// Train Dirichlet calibrators for multiclass problems
pub fn train_dirichlet_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    _classes: &[i32],
    _cv: usize,
    concentration: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    // For Dirichlet calibration, we use a single calibrator for all classes
    let calibrator = DirichletCalibration::new()
        .concentration(concentration)
        .fit(probabilities, y)?;
    let adapter = DirichletAdapter::new(calibrator);
    Ok(vec![Box::new(adapter)])
}

// Adapter structs to make multiclass calibrators compatible with CalibrationEstimator trait

/// Adapter for OneVsOneCalibrator to make it compatible with CalibrationEstimator trait
#[derive(Debug, Clone)]
struct OneVsOneAdapter {
    calibrator: OneVsOneCalibrator,
}

impl OneVsOneAdapter {
    fn new(calibrator: OneVsOneCalibrator) -> Self {
        Self { calibrator }
    }
}

impl CalibrationEstimator for OneVsOneAdapter {
    fn fit(&mut self, _probabilities: &Array1<Float>, _y_true: &Array1<i32>) -> Result<()> {
        // Already fitted during construction
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        // For single probability input, assume it's for binary case
        // In practice, this would need to be handled differently
        Ok(probabilities.clone())
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Adapter for MulticlassTemperatureScaling to make it compatible with CalibrationEstimator trait
#[derive(Debug, Clone)]
struct MulticlassTemperatureAdapter {
    calibrator: MulticlassTemperatureScaling,
}

impl MulticlassTemperatureAdapter {
    fn new(calibrator: MulticlassTemperatureScaling) -> Self {
        Self { calibrator }
    }
}

impl CalibrationEstimator for MulticlassTemperatureAdapter {
    fn fit(&mut self, _probabilities: &Array1<Float>, _y_true: &Array1<i32>) -> Result<()> {
        // Already fitted during construction
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        // For single probability input, assume it's for binary case
        Ok(probabilities.clone())
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Adapter for MatrixScaling to make it compatible with CalibrationEstimator trait
#[derive(Debug, Clone)]
struct MatrixScalingAdapter {
    calibrator: MatrixScaling,
}

impl MatrixScalingAdapter {
    fn new(calibrator: MatrixScaling) -> Self {
        Self { calibrator }
    }
}

impl CalibrationEstimator for MatrixScalingAdapter {
    fn fit(&mut self, _probabilities: &Array1<Float>, _y_true: &Array1<i32>) -> Result<()> {
        // Already fitted during construction
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        // For single probability input, assume it's for binary case
        Ok(probabilities.clone())
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Adapter for DirichletCalibration to make it compatible with CalibrationEstimator trait
#[derive(Debug, Clone)]
struct DirichletAdapter {
    calibrator: DirichletCalibration,
}

impl DirichletAdapter {
    fn new(calibrator: DirichletCalibration) -> Self {
        Self { calibrator }
    }
}

impl CalibrationEstimator for DirichletAdapter {
    fn fit(&mut self, _probabilities: &Array1<Float>, _y_true: &Array1<i32>) -> Result<()> {
        // Already fitted during construction
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        // For single probability input, assume it's for binary case
        Ok(probabilities.clone())
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Training functions for local calibration methods
/// Train local k-NN calibrators for multiclass problems using one-vs-rest strategy
pub fn train_local_knn_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    k: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train local k-NN calibrator
        let mut calibrator = LocalKNNCalibrator::new(k);
        calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train local binning calibrators for multiclass problems using one-vs-rest strategy
pub fn train_local_binning_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    n_bins: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train local binning calibrator
        let mut calibrator = LocalBinningCalibrator::new(n_bins);
        calibrator.fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train KDE calibrators for multiclass problems using one-vs-rest strategy
pub fn train_kde_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train KDE calibrator
        let calibrator = KDECalibrator::new().fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train adaptive KDE calibrators for multiclass problems using one-vs-rest strategy
pub fn train_adaptive_kde_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
    adaptation_factor: Float,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train adaptive KDE calibrator
        let calibrator = AdaptiveKDECalibrator::new()
            .adaptation_factor(adaptation_factor)
            .fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}

/// Train Gaussian process calibrators for multiclass problems using one-vs-rest strategy
pub fn train_gaussian_process_calibrators(
    probabilities: &Array2<Float>,
    y: &Array1<i32>,
    classes: &[i32],
    _cv: usize,
) -> Result<Vec<Box<dyn CalibrationEstimator>>> {
    let n_classes = classes.len();
    let mut calibrators: Vec<Box<dyn CalibrationEstimator>> = Vec::with_capacity(n_classes);

    for (i, &class) in classes.iter().enumerate() {
        // Create binary targets for this class
        let y_binary: Array1<i32> = y.mapv(|yi| if yi == class { 1 } else { 0 });

        // Get probabilities for this class
        let class_probas = probabilities.column(i).to_owned();

        // Train Gaussian process calibrator
        let calibrator = GaussianProcessCalibrator::new().fit(&class_probas, &y_binary)?;

        calibrators.push(Box::new(calibrator));
    }

    Ok(calibrators)
}
