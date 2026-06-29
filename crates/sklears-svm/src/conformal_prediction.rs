//! Conformal Prediction for Support Vector Machines
//!
//! This module implements conformal prediction methods for SVMs, providing
//! statistically valid prediction regions with guaranteed coverage probability.
//! It includes:
//! - Inductive Conformal Prediction (ICP)
//! - Transductive Conformal Prediction (TCP)
//! - Cross-Conformal Prediction
//! - Mondrian Conformal Prediction (class-conditional)
//! - Normalized and unnormalized nonconformity measures
//! - Prediction intervals for regression
//! - Prediction sets for classification
//! - Efficiency measures and calibration diagnostics

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::{essentials::Uniform, seeded_rng};
use std::collections::HashMap;
use thiserror::Error;

use crate::svc::{SvcKernel, SVC};
use crate::svr::SVR;
use sklears_core::traits::{Fit, Predict, Trained};

/// Errors for conformal prediction
#[derive(Error, Debug)]
pub enum ConformalError {
    #[error("Invalid significance level: must be in (0, 1)")]
    InvalidSignificanceLevel,
    #[error("Invalid calibration set size: must be positive")]
    InvalidCalibrationSize,
    #[error("Model not trained")]
    NotTrained,
    #[error("Dimension mismatch: {message}")]
    DimensionMismatch { message: String },
    #[error("Invalid nonconformity measure")]
    InvalidNonconformityMeasure,
    #[error("Insufficient calibration data")]
    InsufficientCalibrationData,
    #[error("Empty prediction set")]
    EmptyPredictionSet,
}

/// Result type for conformal prediction
pub type ConformalResult<T> = Result<T, ConformalError>;

/// Nonconformity measure for classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClassificationNonconformity {
    /// Margin-based: distance to decision boundary
    Margin,
    /// Inverse probability: 1 - P(y|x)
    InverseProbability,
    /// Distance to k-nearest neighbors of same class
    Knn { k: usize },
    /// Normalized margin by local density
    NormalizedMargin,
}

/// Nonconformity measure for regression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegressionNonconformity {
    /// Absolute residual |y - y_hat|
    AbsoluteResidual,
    /// Normalized residual by local density
    NormalizedResidual,
    /// Squared residual (y - y_hat)^2
    SquaredResidual,
    /// Quantile-normalized residual
    QuantileNormalized,
}

/// Conformal prediction method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConformalMethod {
    /// Inductive Conformal Prediction (split calibration)
    Inductive,
    /// Transductive Conformal Prediction (leave-one-out)
    Transductive,
    /// Cross-Conformal Prediction (k-fold)
    CrossConformal { n_folds: usize },
    /// Mondrian Conformal Prediction (class-conditional)
    Mondrian,
}

/// Conformal predictor configuration
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    pub method: ConformalMethod,
    pub significance_level: f64,   // alpha (typically 0.05 or 0.1)
    pub calibration_fraction: f64, // Fraction of data for calibration in ICP
    pub random_state: Option<u64>,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            method: ConformalMethod::Inductive,
            significance_level: 0.1, // 90% confidence
            calibration_fraction: 0.3,
            random_state: None,
        }
    }
}

/// Conformal classifier using SVM
#[derive(Debug, Clone)]
pub struct ConformalClassifier {
    config: ConformalConfig,
    nonconformity: ClassificationNonconformity,

    // Base SVM hyper-parameters used to (re)build untrained models.
    // The fitted `SVC<Trained>` is a distinct type from `SVC<Untrained>`,
    // so we keep the configuration here and store the fitted estimator
    // separately in `model`.
    kernel: SvcKernel,
    regularization: f64,

    // Fitted base SVM model (None until `fit` succeeds).
    model: Option<SVC<Trained>>,

    // Calibration data
    calibration_scores: Vec<f64>,
    calibration_labels: Vec<i32>,

    // For Mondrian CP: class-specific calibration scores
    class_calibration_scores: HashMap<i32, Vec<f64>>,

    // Classes
    classes: Vec<i32>,

    is_trained: bool,
}

impl ConformalClassifier {
    /// Create a new conformal classifier
    pub fn new(
        config: ConformalConfig,
        nonconformity: ClassificationNonconformity,
        kernel: SvcKernel,
        regularization: f64,
    ) -> Self {
        Self {
            config,
            nonconformity,
            kernel,
            regularization,
            model: None,
            calibration_scores: Vec::new(),
            calibration_labels: Vec::new(),
            class_calibration_scores: HashMap::new(),
            classes: Vec::new(),
            is_trained: false,
        }
    }

    /// Create a conformal classifier with default configuration and an RBF kernel.
    pub fn with_default(regularization: f64) -> Self {
        Self::new(
            ConformalConfig::default(),
            ClassificationNonconformity::Margin,
            SvcKernel::default(),
            regularization,
        )
    }

    /// Create conformal classifier with RBF kernel
    pub fn with_rbf(regularization: f64, gamma: Option<f64>) -> Self {
        Self::new(
            ConformalConfig::default(),
            ClassificationNonconformity::Margin,
            SvcKernel::Rbf { gamma },
            regularization,
        )
    }

    /// Create conformal classifier with linear kernel
    pub fn with_linear(regularization: f64) -> Self {
        Self::new(
            ConformalConfig::default(),
            ClassificationNonconformity::Margin,
            SvcKernel::Linear,
            regularization,
        )
    }

    /// Whether the predictor has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.is_trained
    }

    /// Build a fresh untrained base SVM from the stored hyper-parameters.
    fn build_estimator(&self) -> SVC {
        SVC::new()
            .c(self.regularization)
            .svc_kernel(self.kernel.clone())
    }

    /// Fit the conformal predictor
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> ConformalResult<()> {
        let n_samples = x.nrows();

        if y.len() != n_samples {
            return Err(ConformalError::DimensionMismatch {
                message: format!("X has {} samples but y has {}", n_samples, y.len()),
            });
        }

        if self.config.significance_level <= 0.0 || self.config.significance_level >= 1.0 {
            return Err(ConformalError::InvalidSignificanceLevel);
        }

        // Get unique classes
        self.classes = Self::get_unique_classes(y);

        match self.config.method {
            ConformalMethod::Inductive => self.fit_inductive(x, y)?,
            ConformalMethod::Transductive => self.fit_transductive(x, y)?,
            ConformalMethod::CrossConformal { n_folds } => {
                self.fit_cross_conformal(x, y, n_folds)?
            }
            ConformalMethod::Mondrian => self.fit_mondrian(x, y)?,
        }

        self.is_trained = true;
        Ok(())
    }

    /// Train the base SVM and return the fitted estimator.
    fn train_estimator(&self, x: &Array2<f64>, y: &Array1<i32>) -> ConformalResult<SVC<Trained>> {
        let y_float = Self::labels_to_float(y);
        self.build_estimator()
            .fit(x, &y_float)
            .map_err(|_| ConformalError::NotTrained)
    }

    /// Inductive conformal prediction (split into training and calibration)
    fn fit_inductive(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> ConformalResult<()> {
        let n_samples = x.nrows();
        let n_calibration = (n_samples as f64 * self.config.calibration_fraction) as usize;

        if n_calibration < 2 {
            return Err(ConformalError::InsufficientCalibrationData);
        }

        let n_training = n_samples - n_calibration;

        // Split data
        let indices = self.shuffle_indices(n_samples);
        let train_indices = &indices[..n_training];
        let calib_indices = &indices[n_training..];

        // Extract training and calibration sets
        let (x_train, y_train) = Self::extract_subset(x, y, train_indices);
        let (x_calib, y_calib) = Self::extract_subset(x, y, calib_indices);

        // Train model on training set and store the fitted estimator so that
        // calibration scoring and prediction can use it. The fitted type is
        // `SVC<Trained>`, which is why `model` holds that concrete type.
        let trained_model = self.train_estimator(&x_train, &y_train)?;
        self.model = Some(trained_model);

        // Compute nonconformity scores on the held-out calibration set using
        // the now-trained model.
        self.calibration_scores.clear();
        self.calibration_labels = y_calib.to_vec();

        for i in 0..x_calib.nrows() {
            let row = x_calib.row(i);
            let y_true = y_calib[i];
            let score = self.compute_nonconformity_score(&row, y_true)?;
            self.calibration_scores.push(score);
        }

        Ok(())
    }

    /// Transductive conformal prediction (leave-one-out)
    fn fit_transductive(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> ConformalResult<()> {
        // For transductive, we compute the nonconformity score for each training
        // point by temporarily excluding it and retraining the model.
        self.calibration_scores.clear();
        self.calibration_labels = y.to_vec();

        for i in 0..x.nrows() {
            // Create dataset excluding i-th point
            let (x_loo, y_loo) = Self::leave_one_out(x, y, i);

            // Train a temporary model on the leave-one-out set.
            // This is expensive - O(n) training operations.
            let temp_model = self.train_estimator(&x_loo, &y_loo)?;

            // Compute the nonconformity score for the left-out point.
            let row = x.row(i);
            let y_true = y[i];

            let prediction = temp_model
                .decision_function(&row.insert_axis(Axis(0)).to_owned())
                .map_err(|_| ConformalError::NotTrained)?;

            let score = if y_true == self.positive_label() {
                -prediction[0] // Margin from decision boundary
            } else {
                prediction[0]
            };

            self.calibration_scores.push(score);
        }

        // Finally train on the full dataset for predictions.
        self.model = Some(self.train_estimator(x, y)?);

        Ok(())
    }

    /// Cross-conformal prediction (k-fold)
    fn fit_cross_conformal(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        n_folds: usize,
    ) -> ConformalResult<()> {
        if n_folds < 2 {
            return Err(ConformalError::InvalidCalibrationSize);
        }

        let n_samples = x.nrows();
        self.calibration_scores.clear();
        self.calibration_labels = y.to_vec();

        // Create folds
        let indices = self.shuffle_indices(n_samples);
        let fold_size = n_samples / n_folds;

        for fold in 0..n_folds {
            let start = fold * fold_size;
            let end = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let test_indices = &indices[start..end];
            let train_indices: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < start || *i >= end)
                .map(|(_, &idx)| idx)
                .collect();

            // Train on the rest of the data, score the held-out fold.
            let (x_train, y_train) = Self::extract_subset(x, y, &train_indices);
            let (x_test, y_test) = Self::extract_subset(x, y, test_indices);

            let fold_model = self.train_estimator(&x_train, &y_train)?;

            for i in 0..x_test.nrows() {
                let row = x_test.row(i);
                let y_true = y_test[i];

                let prediction = fold_model
                    .decision_function(&row.insert_axis(Axis(0)).to_owned())
                    .map_err(|_| ConformalError::NotTrained)?;

                let score = if y_true == self.positive_label() {
                    -prediction[0]
                } else {
                    prediction[0]
                };

                self.calibration_scores.push(score);
            }
        }

        // Train final model on all data.
        self.model = Some(self.train_estimator(x, y)?);

        Ok(())
    }

    /// Mondrian conformal prediction (class-conditional)
    fn fit_mondrian(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> ConformalResult<()> {
        // Similar to inductive, but maintain separate calibration scores per class.
        let n_samples = x.nrows();
        let n_calibration = (n_samples as f64 * self.config.calibration_fraction) as usize;

        if n_calibration < 2 {
            return Err(ConformalError::InsufficientCalibrationData);
        }

        let n_training = n_samples - n_calibration;

        // Split data
        let indices = self.shuffle_indices(n_samples);
        let train_indices = &indices[..n_training];
        let calib_indices = &indices[n_training..];

        let (x_train, y_train) = Self::extract_subset(x, y, train_indices);
        let (x_calib, y_calib) = Self::extract_subset(x, y, calib_indices);

        // Train model and store it for subsequent scoring/prediction.
        self.model = Some(self.train_estimator(&x_train, &y_train)?);

        // Compute class-specific nonconformity scores.
        self.class_calibration_scores.clear();
        for &class in &self.classes {
            self.class_calibration_scores.insert(class, Vec::new());
        }

        for i in 0..x_calib.nrows() {
            let row = x_calib.row(i);
            let y_true = y_calib[i];
            let score = self.compute_nonconformity_score(&row, y_true)?;

            self.class_calibration_scores
                .entry(y_true)
                .or_default()
                .push(score);
        }

        Ok(())
    }

    /// Predict with conformal prediction (returns set of plausible labels)
    pub fn predict(&self, x: &Array2<f64>) -> ConformalResult<Vec<Vec<i32>>> {
        if !self.is_trained {
            return Err(ConformalError::NotTrained);
        }

        let mut prediction_sets = Vec::new();

        for i in 0..x.nrows() {
            let row = x.row(i);
            let pred_set = self.predict_set(&row)?;
            prediction_sets.push(pred_set);
        }

        Ok(prediction_sets)
    }

    /// Predict set for a single point
    fn predict_set(&self, x: &ArrayView1<f64>) -> ConformalResult<Vec<i32>> {
        let mut prediction_set = Vec::new();

        match self.config.method {
            ConformalMethod::Mondrian => {
                // Class-specific p-values
                for &class in &self.classes {
                    let p_value = self.compute_p_value_mondrian(x, class)?;
                    if p_value > self.config.significance_level {
                        prediction_set.push(class);
                    }
                }
            }
            _ => {
                // Standard conformal prediction
                for &class in &self.classes {
                    let p_value = self.compute_p_value(x, class)?;
                    if p_value > self.config.significance_level {
                        prediction_set.push(class);
                    }
                }
            }
        }

        if prediction_set.is_empty() {
            // Should rarely happen; if it does, return the most likely class.
            let model = self.model.as_ref().ok_or(ConformalError::NotTrained)?;
            let pred = model
                .predict(&x.insert_axis(Axis(0)).to_owned())
                .map_err(|_| ConformalError::EmptyPredictionSet)?;
            prediction_set.push(Self::float_to_label(pred[0]));
        }

        Ok(prediction_set)
    }

    /// Compute p-value for a class label
    fn compute_p_value(&self, x: &ArrayView1<f64>, y_candidate: i32) -> ConformalResult<f64> {
        let score = self.compute_nonconformity_score(x, y_candidate)?;

        // Count how many calibration scores are >= this score
        let n_greater_equal = self
            .calibration_scores
            .iter()
            .filter(|&&s| s >= score)
            .count();

        let n_calibration = self.calibration_scores.len();
        if n_calibration == 0 {
            return Err(ConformalError::InsufficientCalibrationData);
        }
        let p_value = (n_greater_equal + 1) as f64 / (n_calibration + 1) as f64;

        Ok(p_value)
    }

    /// Compute p-value for Mondrian CP (class-conditional)
    fn compute_p_value_mondrian(
        &self,
        x: &ArrayView1<f64>,
        y_candidate: i32,
    ) -> ConformalResult<f64> {
        let score = self.compute_nonconformity_score(x, y_candidate)?;

        let class_scores = self
            .class_calibration_scores
            .get(&y_candidate)
            .ok_or(ConformalError::InvalidNonconformityMeasure)?;

        let n_greater_equal = class_scores.iter().filter(|&&s| s >= score).count();

        let n_calibration = class_scores.len();
        if n_calibration == 0 {
            return Ok(0.0); // Conservative: exclude this class
        }

        let p_value = (n_greater_equal + 1) as f64 / (n_calibration + 1) as f64;

        Ok(p_value)
    }

    /// Compute nonconformity score for a point with given label
    fn compute_nonconformity_score(&self, x: &ArrayView1<f64>, y: i32) -> ConformalResult<f64> {
        let model = self.model.as_ref().ok_or(ConformalError::NotTrained)?;
        let x_single = x.insert_axis(Axis(0)).to_owned();

        let decision = model
            .decision_function(&x_single)
            .map_err(|_| ConformalError::NotTrained)?;
        let is_positive = y == self.positive_label();

        match self.nonconformity {
            ClassificationNonconformity::Margin => {
                // Nonconformity is negative margin (small for the correct class).
                let score = if is_positive {
                    -decision[0]
                } else {
                    decision[0]
                };
                Ok(score)
            }
            ClassificationNonconformity::InverseProbability => {
                // Use the sigmoid of the decision value as a probability proxy.
                let prob = 1.0 / (1.0 + (-decision[0]).exp());
                let score = if is_positive { 1.0 - prob } else { prob };
                Ok(score)
            }
            ClassificationNonconformity::NormalizedMargin => {
                let margin = if is_positive {
                    -decision[0]
                } else {
                    decision[0]
                };
                // Normalize by an approximate local density (|decision| proxy).
                let density = decision[0].abs().max(0.1);
                Ok(margin / density)
            }
            ClassificationNonconformity::Knn { .. } => {
                // k-NN nonconformity is not available without stored training
                // neighbours; fall back to the margin measure rather than
                // fabricating a value.
                let score = if is_positive {
                    -decision[0]
                } else {
                    decision[0]
                };
                Ok(score)
            }
        }
    }

    /// Compute efficiency (average prediction set size)
    pub fn efficiency(&self, x: &Array2<f64>) -> ConformalResult<f64> {
        let prediction_sets = self.predict(x)?;
        if prediction_sets.is_empty() {
            return Err(ConformalError::EmptyPredictionSet);
        }
        let avg_size = prediction_sets
            .iter()
            .map(|set| set.len() as f64)
            .sum::<f64>()
            / prediction_sets.len() as f64;
        Ok(avg_size)
    }

    /// Compute empirical coverage on test set
    pub fn empirical_coverage(&self, x: &Array2<f64>, y: &Array1<i32>) -> ConformalResult<f64> {
        let prediction_sets = self.predict(x)?;

        let mut n_covered = 0;
        for (i, pred_set) in prediction_sets.iter().enumerate() {
            if pred_set.contains(&y[i]) {
                n_covered += 1;
            }
        }

        Ok(n_covered as f64 / x.nrows() as f64)
    }

    // Helper methods

    /// The positive class label ({+1} side of the binary decision function).
    ///
    /// The base `SVC` orders classes ascending and treats the larger label as
    /// the positive class. We mirror that convention here.
    fn positive_label(&self) -> i32 {
        self.classes.iter().copied().max().unwrap_or(1)
    }

    fn labels_to_float(y: &Array1<i32>) -> Array1<f64> {
        y.mapv(|v| v as f64)
    }

    fn float_to_label(value: f64) -> i32 {
        value.round() as i32
    }

    fn get_unique_classes(y: &Array1<i32>) -> Vec<i32> {
        let mut classes: Vec<i32> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        classes
    }

    fn shuffle_indices(&self, n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        // Fisher-Yates shuffle
        if let Ok(uniform) = Uniform::new(0.0, 1.0) {
            for i in (1..n).rev() {
                let j = (rng.sample(uniform) * (i + 1) as f64) as usize;
                indices.swap(i, j.min(i));
            }
        }

        indices
    }

    fn extract_subset(
        x: &Array2<f64>,
        y: &Array1<i32>,
        indices: &[usize],
    ) -> (Array2<f64>, Array1<i32>) {
        let n_samples = indices.len();
        let n_features = x.ncols();

        let mut x_subset = Array2::zeros((n_samples, n_features));
        let mut y_subset = Array1::zeros(n_samples);

        for (i, &idx) in indices.iter().enumerate() {
            x_subset.row_mut(i).assign(&x.row(idx));
            y_subset[i] = y[idx];
        }

        (x_subset, y_subset)
    }

    fn leave_one_out(
        x: &Array2<f64>,
        y: &Array1<i32>,
        exclude_idx: usize,
    ) -> (Array2<f64>, Array1<i32>) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut x_loo = Array2::zeros((n_samples - 1, n_features));
        let mut y_loo = Array1::zeros(n_samples - 1);

        let mut j = 0;
        for i in 0..n_samples {
            if i != exclude_idx {
                x_loo.row_mut(j).assign(&x.row(i));
                y_loo[j] = y[i];
                j += 1;
            }
        }

        (x_loo, y_loo)
    }
}

/// Conformal regressor using SVR
#[derive(Debug, Clone)]
pub struct ConformalRegressor {
    config: ConformalConfig,
    nonconformity: RegressionNonconformity,

    // Base SVR hyper-parameters used to (re)build untrained models.
    kernel: SvcKernel,
    regularization: f64,
    epsilon: f64,

    // Fitted base SVR model (None until `fit` succeeds).
    model: Option<SVR<Trained>>,

    // Calibration data
    calibration_residuals: Vec<f64>,

    is_trained: bool,
}

impl ConformalRegressor {
    /// Create a new conformal regressor
    pub fn new(
        config: ConformalConfig,
        nonconformity: RegressionNonconformity,
        kernel: SvcKernel,
        regularization: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            config,
            nonconformity,
            kernel,
            regularization,
            epsilon,
            model: None,
            calibration_residuals: Vec::new(),
            is_trained: false,
        }
    }

    /// Create conformal regressor with RBF kernel
    pub fn with_rbf(regularization: f64, epsilon: f64, gamma: Option<f64>) -> Self {
        Self::new(
            ConformalConfig::default(),
            RegressionNonconformity::AbsoluteResidual,
            SvcKernel::Rbf { gamma },
            regularization,
            epsilon,
        )
    }

    /// Create conformal regressor with linear kernel
    pub fn with_linear(regularization: f64, epsilon: f64) -> Self {
        Self::new(
            ConformalConfig::default(),
            RegressionNonconformity::AbsoluteResidual,
            SvcKernel::Linear,
            regularization,
            epsilon,
        )
    }

    /// Whether the predictor has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.is_trained
    }

    /// Build a fresh untrained base SVR from the stored hyper-parameters.
    fn build_estimator(&self) -> SVR {
        SVR::new()
            .c(self.regularization)
            .epsilon(self.epsilon)
            .svc_kernel(self.kernel.clone())
    }

    /// Fit the conformal regressor
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> ConformalResult<()> {
        let n_samples = x.nrows();

        if y.len() != n_samples {
            return Err(ConformalError::DimensionMismatch {
                message: format!("X has {} samples but y has {}", n_samples, y.len()),
            });
        }

        if self.config.significance_level <= 0.0 || self.config.significance_level >= 1.0 {
            return Err(ConformalError::InvalidSignificanceLevel);
        }

        let n_calibration = (n_samples as f64 * self.config.calibration_fraction) as usize;

        if n_calibration < 2 {
            return Err(ConformalError::InsufficientCalibrationData);
        }

        let n_training = n_samples - n_calibration;

        // Split data
        let indices = self.shuffle_indices(n_samples);
        let train_indices = &indices[..n_training];
        let calib_indices = &indices[n_training..];

        let (x_train, y_train) = Self::extract_subset(x, y, train_indices);
        let (x_calib, y_calib) = Self::extract_subset(x, y, calib_indices);

        // Train model and store the fitted estimator.
        let trained = self
            .build_estimator()
            .fit(&x_train, &y_train)
            .map_err(|_| ConformalError::NotTrained)?;
        self.model = Some(trained);

        // Compute nonconformity scores on the calibration set.
        self.calibration_residuals.clear();

        let predictions = self
            .model
            .as_ref()
            .ok_or(ConformalError::NotTrained)?
            .predict(&x_calib)
            .map_err(|_| ConformalError::NotTrained)?;

        for i in 0..x_calib.nrows() {
            let residual = match self.nonconformity {
                RegressionNonconformity::AbsoluteResidual => (y_calib[i] - predictions[i]).abs(),
                RegressionNonconformity::SquaredResidual => (y_calib[i] - predictions[i]).powi(2),
                RegressionNonconformity::NormalizedResidual => {
                    // Normalize by an approximate local scale.
                    let abs_residual = (y_calib[i] - predictions[i]).abs();
                    let local_scale = 1.0 + predictions[i].abs() * 0.1;
                    abs_residual / local_scale
                }
                RegressionNonconformity::QuantileNormalized => (y_calib[i] - predictions[i]).abs(),
            };

            self.calibration_residuals.push(residual);
        }

        // Sort residuals for quantile computation.
        self.calibration_residuals
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        self.is_trained = true;
        Ok(())
    }

    /// Predict with prediction intervals
    pub fn predict_interval(
        &self,
        x: &Array2<f64>,
    ) -> ConformalResult<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        if !self.is_trained {
            return Err(ConformalError::NotTrained);
        }

        if self.calibration_residuals.is_empty() {
            return Err(ConformalError::InsufficientCalibrationData);
        }

        let predictions = self
            .model
            .as_ref()
            .ok_or(ConformalError::NotTrained)?
            .predict(x)
            .map_err(|_| ConformalError::NotTrained)?;

        // Compute the quantile for the prediction interval.
        let quantile_level = 1.0 - self.config.significance_level;
        let quantile_idx =
            ((self.calibration_residuals.len() as f64 + 1.0) * quantile_level).ceil() as usize;
        let quantile_idx = quantile_idx
            .saturating_sub(1)
            .min(self.calibration_residuals.len() - 1);

        let interval_width = self.calibration_residuals[quantile_idx];

        let lower = &predictions - interval_width;
        let upper = &predictions + interval_width;

        Ok((predictions, lower, upper))
    }

    /// Compute empirical coverage on test set
    pub fn empirical_coverage(&self, x: &Array2<f64>, y: &Array1<f64>) -> ConformalResult<f64> {
        let (_predictions, lower, upper) = self.predict_interval(x)?;

        let mut n_covered = 0;
        for i in 0..x.nrows() {
            if y[i] >= lower[i] && y[i] <= upper[i] {
                n_covered += 1;
            }
        }

        Ok(n_covered as f64 / x.nrows() as f64)
    }

    /// Compute average interval width (efficiency)
    pub fn average_interval_width(&self, x: &Array2<f64>) -> ConformalResult<f64> {
        let (_predictions, lower, upper) = self.predict_interval(x)?;

        let avg_width = (&upper - &lower).mean().unwrap_or(0.0);
        Ok(avg_width)
    }

    // Helper methods

    fn shuffle_indices(&self, n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        if let Ok(uniform) = Uniform::new(0.0, 1.0) {
            for i in (1..n).rev() {
                let j = (rng.sample(uniform) * (i + 1) as f64) as usize;
                indices.swap(i, j.min(i));
            }
        }

        indices
    }

    fn extract_subset(
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
    ) -> (Array2<f64>, Array1<f64>) {
        let n_samples = indices.len();
        let n_features = x.ncols();

        let mut x_subset = Array2::zeros((n_samples, n_features));
        let mut y_subset = Array1::zeros(n_samples);

        for (i, &idx) in indices.iter().enumerate() {
            x_subset.row_mut(i).assign(&x.row(idx));
            y_subset[i] = y[idx];
        }

        (x_subset, y_subset)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_conformal_classifier_creation() {
        let classifier = ConformalClassifier::with_linear(1.0);
        assert!(!classifier.is_fitted());
        assert_eq!(classifier.config.significance_level, 0.1);
    }

    #[test]
    fn test_conformal_regressor_creation() {
        let regressor = ConformalRegressor::new(
            ConformalConfig::default(),
            RegressionNonconformity::AbsoluteResidual,
            SvcKernel::Rbf { gamma: Some(0.1) },
            1.0,
            0.1,
        );
        assert!(!regressor.is_fitted());
    }

    #[test]
    fn test_unfitted_predict_is_err() {
        // An unfitted predictor must return an honest error, not panic.
        let classifier = ConformalClassifier::with_linear(1.0);
        let x =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("array shape mismatch");
        let result = classifier.predict(&x);
        assert!(matches!(result, Err(ConformalError::NotTrained)));

        let regressor = ConformalRegressor::with_linear(1.0, 0.1);
        let result = regressor.predict_interval(&x);
        assert!(matches!(result, Err(ConformalError::NotTrained)));
    }

    #[test]
    fn test_simple_classification() {
        let mut classifier = ConformalClassifier::with_linear(1.0);

        // Simple linearly separable dataset
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                -2.0, -2.0, -2.0, 2.0, -1.0, -1.0, -1.0, 1.0, -0.5, -0.5, 2.0, 2.0, 2.0, -2.0, 1.0,
                1.0, 1.0, -1.0, 0.5, 0.5,
            ],
        )
        .expect("operation should succeed");

        let y = Array1::from_vec(vec![-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]);

        // Fit
        classifier
            .fit(&x, &y)
            .expect("model fitting should succeed");
        assert!(classifier.is_fitted());

        // The trained model state must be populated after fitting.
        assert!(classifier.model.is_some());
        assert!(!classifier.calibration_scores.is_empty());

        // Predict
        let pred_sets = classifier.predict(&x).expect("prediction should succeed");
        assert_eq!(pred_sets.len(), 10);

        // Every prediction set must be non-empty.
        for pred_set in &pred_sets {
            assert!(!pred_set.is_empty());
        }
    }

    #[test]
    fn test_conformal_regression() {
        let mut regressor = ConformalRegressor::with_linear(1.0, 0.1);

        // Simple linear relationship: y = 2x + 1
        let x = Array2::from_shape_vec((20, 1), (0..20).map(|i| i as f64).collect())
            .expect("array shape mismatch");

        let y = Array1::from_vec((0..20).map(|i| 2.0 * i as f64 + 1.0).collect());

        // Fit
        regressor.fit(&x, &y).expect("model fitting should succeed");
        assert!(regressor.is_fitted());
        assert!(regressor.model.is_some());

        // Predict with intervals
        let x_test = Array2::from_shape_vec((5, 1), vec![0.0, 5.0, 10.0, 15.0, 20.0])
            .expect("array shape mismatch");
        let (_predictions, lower, upper) = regressor
            .predict_interval(&x_test)
            .expect("operation should succeed");

        // Check that intervals are valid (lower <= upper).
        for i in 0..5 {
            assert!(lower[i] <= upper[i]);
        }
    }

    #[test]
    fn test_cross_conformal() {
        let config = ConformalConfig {
            method: ConformalMethod::CrossConformal { n_folds: 3 },
            ..ConformalConfig::default()
        };
        let mut classifier = ConformalClassifier::new(
            config,
            ClassificationNonconformity::Margin,
            SvcKernel::Linear,
            1.0,
        );

        let x = Array::from_shape_fn((12, 2), |(i, j)| {
            if i < 6 {
                -1.0 + (i as f64) * 0.1 + (j as f64) * 0.01
            } else {
                1.0 + (i as f64) * 0.1 + (j as f64) * 0.01
            }
        });
        let y = Array1::from_vec(vec![-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1]);

        classifier
            .fit(&x, &y)
            .expect("cross-conformal fit should succeed");
        assert!(classifier.is_fitted());
        assert!(classifier.model.is_some());
        assert!(!classifier.calibration_scores.is_empty());
    }
}
