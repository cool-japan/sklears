//! Probability Calibration for Multiclass Classification
//!
//! This module provides various probability calibration methods for multiclass classifiers.
//! Calibration is crucial for obtaining well-calibrated probability estimates that
//! reflect the true likelihood of class membership.
//!
//! ## Implemented Methods
//!
//! - **Platt Scaling**: Sigmoid calibration using logistic regression
//! - **Isotonic Regression**: Non-parametric calibration preserving monotonicity
//! - **Temperature Scaling**: Simple scaling method using temperature parameter
//! - **Dirichlet Calibration**: Multinomial calibration for multiclass problems
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sklears_multiclass::calibration::{CalibratedClassifier, CalibrationMethod};
//!
//! // Create a calibrated classifier
//! let calibrated = CalibratedClassifier::new(base_classifier)
//!     .method(CalibrationMethod::PlattScaling)
//!     .cv_folds(5)
//!     .build();
//! ```

pub mod dirichlet_calibration;
pub mod isotonic_regression;
pub mod platt_scaling;
pub mod temperature_scaling;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
};
use std::marker::PhantomData;

pub use dirichlet_calibration::DirichletCalibration;
pub use isotonic_regression::IsotonicRegression;
pub use platt_scaling::PlattScaling;
pub use temperature_scaling::TemperatureScaling;

/// Calibration method configuration
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationMethod {
    /// Platt scaling using sigmoid calibration
    PlattScaling,
    /// Isotonic regression calibration
    IsotonicRegression,
    /// Temperature scaling calibration
    TemperatureScaling { initial_temperature: f64 },
    /// Dirichlet calibration for multiclass
    DirichletCalibration { l2_reg: f64 },
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        Self::PlattScaling
    }
}

/// Configuration for calibrated classifier
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Calibration method to use
    pub method: CalibrationMethod,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Ensemble calibration across CV folds
    pub ensemble: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            method: CalibrationMethod::default(),
            cv_folds: 5,
            random_state: None,
            ensemble: false,
        }
    }
}

/// Calibrated classifier wrapper
///
/// This classifier wraps any base classifier and calibrates its probability outputs
/// using one of several calibration methods. The calibration is performed using
/// cross-validation to avoid overfitting.
///
/// # Type Parameters
///
/// * `C` - Base classifier type implementing Fit and PredictProba
/// * `S` - State type (Untrained or Trained)
#[derive(Debug)]
pub struct CalibratedClassifier<C, S = Untrained> {
    base_classifier: C,
    config: CalibrationConfig,
    state: PhantomData<S>,
}

/// Trained calibrated classifier data
#[derive(Debug, Clone)]
pub struct CalibratedTrainedData<T> {
    /// Base classifiers from cross-validation
    pub base_classifiers: Vec<T>,
    /// Calibrators for each fold and class
    pub calibrators: Vec<Vec<Box<dyn CalibrationFunction>>>,
    /// Class labels
    pub classes: Array1<i32>,
    /// Number of classes
    pub n_classes: usize,
}

/// Trait for calibration functions
pub trait CalibrationFunction: std::fmt::Debug + Send + Sync {
    /// Calibrate probabilities
    fn calibrate(&self, uncalibrated_probs: &Array1<f64>) -> SklResult<Array1<f64>>;

    /// Clone the calibration function
    fn clone_box(&self) -> Box<dyn CalibrationFunction>;
}

impl Clone for Box<dyn CalibrationFunction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl<C> CalibratedClassifier<C, Untrained> {
    /// Create a new calibrated classifier
    pub fn new(base_classifier: C) -> Self {
        Self {
            base_classifier,
            config: CalibrationConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for the calibrated classifier
    pub fn builder(base_classifier: C) -> CalibratedClassifierBuilder<C> {
        CalibratedClassifierBuilder::new(base_classifier)
    }

    /// Set the calibration method
    pub fn method(mut self, method: CalibrationMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set the number of CV folds
    pub fn cv_folds(mut self, cv_folds: usize) -> Self {
        self.config.cv_folds = cv_folds;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Enable ensemble calibration
    pub fn ensemble(mut self, ensemble: bool) -> Self {
        self.config.ensemble = ensemble;
        self
    }
}

/// Builder for calibrated classifier
#[derive(Debug)]
pub struct CalibratedClassifierBuilder<C> {
    base_classifier: C,
    config: CalibrationConfig,
}

impl<C> CalibratedClassifierBuilder<C> {
    /// Create a new builder
    pub fn new(base_classifier: C) -> Self {
        Self {
            base_classifier,
            config: CalibrationConfig::default(),
        }
    }

    /// Set the calibration method
    pub fn method(mut self, method: CalibrationMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set the number of CV folds
    pub fn cv_folds(mut self, cv_folds: usize) -> Self {
        self.config.cv_folds = cv_folds;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Enable ensemble calibration
    pub fn ensemble(mut self, ensemble: bool) -> Self {
        self.config.ensemble = ensemble;
        self
    }

    /// Build the calibrated classifier
    pub fn build(self) -> CalibratedClassifier<C, Untrained> {
        CalibratedClassifier {
            base_classifier: self.base_classifier,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for CalibratedClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_classifier: self.base_classifier.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C> Estimator for CalibratedClassifier<C, Untrained> {
    type Config = CalibrationConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Helper function to create stratified splits for cross-validation
pub fn stratified_kfold_split(
    y: &Array1<i32>,
    n_splits: usize,
    random_state: Option<u64>,
) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
    use scirs2_core::random::Random;

    let n_samples = y.len();
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "Cannot create splits for empty dataset".to_string(),
        ));
    }

    // Validate minimum requirements
    if n_samples < 2 {
        return Err(SklearsError::InvalidInput(
            "Need at least 2 samples for cross-validation".to_string(),
        ));
    }

    // Adjust n_splits if necessary
    let effective_n_splits = std::cmp::min(n_splits, n_samples);

    if effective_n_splits < 2 {
        return Err(SklearsError::InvalidInput(
            "Number of splits must be at least 2".to_string(),
        ));
    }

    // Get unique classes and their indices
    let mut class_indices: std::collections::HashMap<i32, Vec<usize>> =
        std::collections::HashMap::new();

    for (idx, &class) in y.iter().enumerate() {
        class_indices.entry(class).or_default().push(idx);
    }

    // Shuffle indices for each class
    let mut rng = match random_state {
        Some(seed) => Random::seed(seed),
        None => Random::seed(42),
    };

    for indices in class_indices.values_mut() {
        // Shuffle indices using Fisher-Yates algorithm
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0, i + 1);
            indices.swap(i, j);
        }
    }

    // Create stratified splits
    let mut splits = Vec::new();

    for fold in 0..effective_n_splits {
        let mut test_indices = Vec::new();
        let mut train_indices = Vec::new();

        for indices in class_indices.values() {
            if indices.is_empty() {
                continue;
            }

            // Ensure each fold gets at least one sample from each class if possible
            if indices.len() >= effective_n_splits {
                let fold_size = indices.len() / effective_n_splits;
                let remainder = indices.len() % effective_n_splits;

                let start = fold * fold_size + std::cmp::min(fold, remainder);
                let end = start + fold_size + if fold < remainder { 1 } else { 0 };

                for (i, &idx) in indices.iter().enumerate() {
                    if i >= start && i < end {
                        test_indices.push(idx);
                    } else {
                        train_indices.push(idx);
                    }
                }
            } else {
                // For classes with fewer samples than folds, distribute round-robin
                for (i, &idx) in indices.iter().enumerate() {
                    if i % effective_n_splits == fold {
                        test_indices.push(idx);
                    } else {
                        train_indices.push(idx);
                    }
                }
            }
        }

        // Only add splits that have both training and test data
        if !train_indices.is_empty() && !test_indices.is_empty() {
            splits.push((train_indices, test_indices));
        }
    }

    // If we didn't get enough splits, adjust the approach
    if splits.len() < effective_n_splits && n_samples >= effective_n_splits {
        splits.clear();

        // Simple round-robin assignment to ensure we get the requested number of splits
        let mut all_indices: Vec<usize> = (0..n_samples).collect();
        // Shuffle indices using Fisher-Yates algorithm
        for i in (1..all_indices.len()).rev() {
            let j = rng.random_range(0, i + 1);
            all_indices.swap(i, j);
        }

        let base_size = n_samples / effective_n_splits;
        let remainder = n_samples % effective_n_splits;

        let mut start = 0;
        for fold in 0..effective_n_splits {
            let fold_size = base_size + if fold < remainder { 1 } else { 0 };
            let end = start + fold_size;

            if end <= n_samples && fold_size > 0 {
                let test_indices = all_indices[start..end].to_vec();
                let mut train_indices = Vec::new();
                train_indices.extend_from_slice(&all_indices[0..start]);
                train_indices.extend_from_slice(&all_indices[end..]);

                if !train_indices.is_empty() && !test_indices.is_empty() {
                    splits.push((train_indices, test_indices));
                }
            }
            start = end;
        }
    }

    // If still no valid splits, create at least one simple split
    if splits.is_empty() && n_samples > 1 {
        let test_size = std::cmp::max(1, n_samples / 3);
        let mut all_indices: Vec<usize> = (0..n_samples).collect();
        // Shuffle indices using Fisher-Yates algorithm
        for i in (1..all_indices.len()).rev() {
            let j = rng.random_range(0, i + 1);
            all_indices.swap(i, j);
        }

        let test_indices = all_indices.split_off(all_indices.len() - test_size);
        splits.push((all_indices, test_indices));
    }

    Ok(splits)
}

type TrainedCalibratedClassifier<T> = CalibratedClassifier<CalibratedTrainedData<T>, Trained>;

impl<C> Fit<Array2<f64>, Array1<i32>> for CalibratedClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: PredictProba<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    type Fitted = TrainedCalibratedClassifier<C::Fitted>;

    #[allow(non_snake_case)]
    fn fit(
        self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> SklResult<TrainedCalibratedClassifier<C::Fitted>> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }
        if X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "X must have at least one sample".to_string(),
            ));
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes_array = Array1::from_vec(classes.clone());
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        // Get stratified CV splits
        let splits = stratified_kfold_split(y, self.config.cv_folds, self.config.random_state)?;

        let mut base_classifiers = Vec::new();
        let mut calibrators = Vec::new();

        for (train_indices, val_indices) in splits {
            if train_indices.is_empty() || val_indices.is_empty() {
                continue;
            }

            // Create training data
            let X_train = X.select(scirs2_core::ndarray::Axis(0), &train_indices);
            let y_train = Array1::from_vec(train_indices.iter().map(|&i| y[i]).collect());

            // Create validation data
            let X_val = X.select(scirs2_core::ndarray::Axis(0), &val_indices);
            let y_val = Array1::from_vec(val_indices.iter().map(|&i| y[i]).collect());

            // Fit base classifier
            let fitted_classifier = self.base_classifier.clone().fit(&X_train, &y_train)?;

            // Get uncalibrated probabilities on validation set
            let uncalibrated_probs = fitted_classifier.predict_proba(&X_val)?;

            // Fit calibrators for each class
            let mut fold_calibrators = Vec::new();

            match &self.config.method {
                CalibrationMethod::PlattScaling => {
                    for class_idx in 0..n_classes {
                        let target_class = classes[class_idx];

                        // Create binary targets
                        let binary_targets = Array1::from_vec(
                            y_val
                                .iter()
                                .map(|&label| if label == target_class { 1.0 } else { 0.0 })
                                .collect(),
                        );

                        // Extract class probabilities - handle case where base classifier
                        // doesn't return probabilities for all original classes
                        let class_probs = if class_idx < uncalibrated_probs.ncols() {
                            uncalibrated_probs.column(class_idx).to_owned()
                        } else {
                            // If this class wasn't present in the training fold,
                            // create uniform low probabilities
                            Array1::from_elem(uncalibrated_probs.nrows(), 1e-6)
                        };

                        // Convert to decision values
                        let mut decision_values = Array1::zeros(class_probs.len());
                        for (i, &prob) in class_probs.iter().enumerate() {
                            let p_clipped = prob.max(1e-15).min(1.0 - 1e-15);
                            decision_values[i] = (p_clipped / (1.0 - p_clipped)).ln();
                        }

                        // Fit Platt scaler
                        let mut platt_scaler = PlattScaling::new();
                        platt_scaler.fit(
                            &decision_values,
                            &binary_targets.mapv(|x| x as i32),
                            100,
                            1e-6,
                        )?;

                        fold_calibrators
                            .push(Box::new(platt_scaler) as Box<dyn CalibrationFunction>);
                    }
                }
                CalibrationMethod::IsotonicRegression => {
                    for class_idx in 0..n_classes {
                        let target_class = classes[class_idx];

                        // Create binary targets
                        let binary_targets = Array1::from_vec(
                            y_val
                                .iter()
                                .map(|&label| if label == target_class { 1.0 } else { 0.0 })
                                .collect(),
                        );

                        // Extract class probabilities - handle case where base classifier
                        // doesn't return probabilities for all original classes
                        let class_probs = if class_idx < uncalibrated_probs.ncols() {
                            uncalibrated_probs.column(class_idx).to_owned()
                        } else {
                            // If this class wasn't present in the training fold,
                            // create uniform low probabilities
                            Array1::from_elem(uncalibrated_probs.nrows(), 1e-6)
                        };

                        // Fit isotonic regressor
                        let mut isotonic_regressor = IsotonicRegression::new(true);
                        isotonic_regressor.fit(&class_probs, &binary_targets, None)?;

                        fold_calibrators
                            .push(Box::new(isotonic_regressor) as Box<dyn CalibrationFunction>);
                    }
                }
                CalibrationMethod::TemperatureScaling {
                    initial_temperature,
                } => {
                    // Temperature scaling uses a single calibrator for all classes
                    let mut temp_scaler = temperature_scaling::MulticlassTemperatureScaling::new(
                        *initial_temperature,
                    );
                    temp_scaler.fit_probabilities(&uncalibrated_probs, &y_val, 100, 1e-6, 0.01)?;

                    // We store it for each class for consistency, but it's the same calibrator
                    let scaler_func = temp_scaler.scaler.clone();
                    for _class_idx in 0..n_classes {
                        fold_calibrators
                            .push(Box::new(scaler_func.clone()) as Box<dyn CalibrationFunction>);
                    }
                }
                CalibrationMethod::DirichletCalibration { l2_reg } => {
                    // Dirichlet calibration uses a single calibrator for all classes
                    let mut dirichlet_calibrator = DirichletCalibration::new(*l2_reg);
                    dirichlet_calibrator.fit(&uncalibrated_probs, &y_val, 100, 1e-6, 0.01)?;

                    // We store it for each class for consistency, but it's the same calibrator
                    for _class_idx in 0..n_classes {
                        fold_calibrators
                            .push(Box::new(dirichlet_calibrator.clone())
                                as Box<dyn CalibrationFunction>);
                    }
                }
            }

            base_classifiers.push(fitted_classifier);
            calibrators.push(fold_calibrators);
        }

        let trained_data = CalibratedTrainedData {
            base_classifiers,
            calibrators,
            classes: classes_array,
            n_classes,
        };

        Ok(CalibratedClassifier {
            base_classifier: trained_data,
            config: self.config,
            state: PhantomData,
        })
    }
}

impl<T> Predict<Array2<f64>, Array1<i32>>
    for CalibratedClassifier<CalibratedTrainedData<T>, Trained>
where
    T: PredictProba<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let (n_samples, _) = probabilities.dim();

        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = probabilities.row(i);
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = self.base_classifier.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl<T> PredictProba<Array2<f64>, Array2<f64>>
    for CalibratedClassifier<CalibratedTrainedData<T>, Trained>
where
    T: PredictProba<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    fn predict_proba(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let trained_data = &self.base_classifier;
        let n_folds = trained_data.base_classifiers.len();

        if n_folds == 0 {
            return Err(SklearsError::InvalidInput(
                "No trained classifiers available".to_string(),
            ));
        }

        // Average predictions across all folds
        let mut accumulated_probs = Array2::zeros((n_samples, trained_data.n_classes));
        let mut fold_count = 0;

        for (base_classifier, fold_calibrators) in trained_data
            .base_classifiers
            .iter()
            .zip(trained_data.calibrators.iter())
        {
            // Get uncalibrated probabilities from base classifier
            let uncalibrated_probs = base_classifier.predict_proba(X)?;

            // Apply calibration
            let calibrated_probs = match &self.config.method {
                CalibrationMethod::TemperatureScaling { .. }
                | CalibrationMethod::DirichletCalibration { .. } => {
                    // For methods that calibrate all classes together
                    if let Some(calibrator) = fold_calibrators.first() {
                        // For temperature scaling and Dirichlet, we stored the same calibrator for all classes
                        // We need to handle this differently - let's apply the multiclass calibration directly
                        match &self.config.method {
                            CalibrationMethod::TemperatureScaling { .. } => {
                                // Apply temperature scaling to all classes at once
                                if let Some(temp_scaler) = fold_calibrators.first() {
                                    // For now, apply per-class calibration as a fallback
                                    let mut result =
                                        Array2::zeros((n_samples, trained_data.n_classes));

                                    // Handle case where uncalibrated_probs has fewer columns than expected
                                    let available_classes = uncalibrated_probs.ncols();

                                    for class_idx in 0..trained_data.n_classes {
                                        if class_idx < available_classes {
                                            let class_probs =
                                                uncalibrated_probs.column(class_idx).to_owned();
                                            let calibrated_class =
                                                temp_scaler.calibrate(&class_probs)?;
                                            for i in 0..n_samples {
                                                result[[i, class_idx]] = calibrated_class[i];
                                            }
                                        } else {
                                            // Use uniform probability for missing classes
                                            let uniform_prob = 1.0 / trained_data.n_classes as f64;
                                            for i in 0..n_samples {
                                                result[[i, class_idx]] = uniform_prob;
                                            }
                                        }
                                    }
                                    result
                                } else {
                                    // Handle case where uncalibrated_probs has fewer columns than expected
                                    if uncalibrated_probs.ncols() < trained_data.n_classes {
                                        let mut result =
                                            Array2::zeros((n_samples, trained_data.n_classes));
                                        let available_classes = uncalibrated_probs.ncols();

                                        // Copy available probabilities
                                        for class_idx in 0..available_classes {
                                            for i in 0..n_samples {
                                                result[[i, class_idx]] =
                                                    uncalibrated_probs[[i, class_idx]];
                                            }
                                        }

                                        // Fill missing classes with uniform probability
                                        let uniform_prob = 1.0 / trained_data.n_classes as f64;
                                        for class_idx in available_classes..trained_data.n_classes {
                                            for i in 0..n_samples {
                                                result[[i, class_idx]] = uniform_prob;
                                            }
                                        }
                                        result
                                    } else {
                                        uncalibrated_probs
                                    }
                                }
                            }
                            CalibrationMethod::DirichletCalibration { .. } => {
                                // Apply Dirichlet calibration to all classes at once
                                if let Some(dirichlet_cal) = fold_calibrators.first() {
                                    // For now, apply per-class calibration as a fallback
                                    let mut result =
                                        Array2::zeros((n_samples, trained_data.n_classes));

                                    // Handle case where uncalibrated_probs has fewer columns than expected
                                    let available_classes = uncalibrated_probs.ncols();

                                    for class_idx in 0..trained_data.n_classes {
                                        if class_idx < available_classes {
                                            let class_probs =
                                                uncalibrated_probs.column(class_idx).to_owned();
                                            let calibrated_class =
                                                dirichlet_cal.calibrate(&class_probs)?;
                                            for i in 0..n_samples {
                                                result[[i, class_idx]] = calibrated_class[i];
                                            }
                                        } else {
                                            // Use uniform probability for missing classes
                                            let uniform_prob = 1.0 / trained_data.n_classes as f64;
                                            for i in 0..n_samples {
                                                result[[i, class_idx]] = uniform_prob;
                                            }
                                        }
                                    }
                                    result
                                } else {
                                    // Handle case where uncalibrated_probs has fewer columns than expected
                                    if uncalibrated_probs.ncols() < trained_data.n_classes {
                                        let mut result =
                                            Array2::zeros((n_samples, trained_data.n_classes));
                                        let available_classes = uncalibrated_probs.ncols();

                                        // Copy available probabilities
                                        for class_idx in 0..available_classes {
                                            for i in 0..n_samples {
                                                result[[i, class_idx]] =
                                                    uncalibrated_probs[[i, class_idx]];
                                            }
                                        }

                                        // Fill missing classes with uniform probability
                                        let uniform_prob = 1.0 / trained_data.n_classes as f64;
                                        for class_idx in available_classes..trained_data.n_classes {
                                            for i in 0..n_samples {
                                                result[[i, class_idx]] = uniform_prob;
                                            }
                                        }
                                        result
                                    } else {
                                        uncalibrated_probs
                                    }
                                }
                            }
                            _ => uncalibrated_probs,
                        }
                    } else {
                        uncalibrated_probs
                    }
                }
                _ => {
                    // For per-class calibration methods (Platt, Isotonic)
                    let mut calibrated = Array2::zeros((n_samples, trained_data.n_classes));

                    for class_idx in 0..trained_data.n_classes {
                        if let Some(calibrator) = fold_calibrators.get(class_idx) {
                            // Handle case where base classifier doesn't return probabilities for all classes
                            let class_probs = if class_idx < uncalibrated_probs.ncols() {
                                uncalibrated_probs.column(class_idx).to_owned()
                            } else {
                                // If this class wasn't present in the prediction,
                                // create uniform low probabilities
                                Array1::from_elem(uncalibrated_probs.nrows(), 1e-6)
                            };
                            let calibrated_class = calibrator.calibrate(&class_probs)?;

                            for i in 0..n_samples {
                                calibrated[[i, class_idx]] = calibrated_class[i];
                            }
                        } else {
                            // Fallback to uncalibrated probabilities
                            for i in 0..n_samples {
                                calibrated[[i, class_idx]] =
                                    if class_idx < uncalibrated_probs.ncols() {
                                        uncalibrated_probs[[i, class_idx]]
                                    } else {
                                        1e-6 // Low probability for missing classes
                                    };
                            }
                        }
                    }

                    calibrated
                }
            };

            // Normalize probabilities to sum to 1
            let mut normalized_probs = Array2::zeros((n_samples, trained_data.n_classes));
            for i in 0..n_samples {
                let row_sum: f64 = calibrated_probs.row(i).sum();
                if row_sum > 0.0 {
                    for j in 0..trained_data.n_classes {
                        normalized_probs[[i, j]] = calibrated_probs[[i, j]] / row_sum;
                    }
                } else {
                    // Uniform distribution fallback
                    for j in 0..trained_data.n_classes {
                        normalized_probs[[i, j]] = 1.0 / trained_data.n_classes as f64;
                    }
                }
            }

            // Accumulate probabilities
            for i in 0..n_samples {
                for j in 0..trained_data.n_classes {
                    accumulated_probs[[i, j]] += normalized_probs[[i, j]];
                }
            }
            fold_count += 1;
        }

        // Average across folds
        if fold_count > 0 {
            for i in 0..n_samples {
                for j in 0..trained_data.n_classes {
                    accumulated_probs[[i, j]] /= fold_count as f64;
                }
            }
        }

        Ok(accumulated_probs)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_calibration_method_default() {
        let method = CalibrationMethod::default();
        assert_eq!(method, CalibrationMethod::PlattScaling);
    }

    #[test]
    fn test_calibration_config_default() {
        let config = CalibrationConfig::default();
        assert_eq!(config.method, CalibrationMethod::PlattScaling);
        assert_eq!(config.cv_folds, 5);
        assert_eq!(config.random_state, None);
        assert!(!config.ensemble);
    }

    #[test]
    fn test_stratified_kfold_split() {
        let y = array![0, 0, 1, 1, 2, 2];
        let splits = stratified_kfold_split(&y, 3, Some(42)).unwrap();

        assert_eq!(splits.len(), 3);

        // Each split should have roughly equal class distribution
        for (train_indices, test_indices) in splits {
            assert!(!train_indices.is_empty());
            assert!(!test_indices.is_empty());
            assert_eq!(train_indices.len() + test_indices.len(), 6);
        }
    }

    #[test]
    fn test_stratified_kfold_invalid_splits() {
        let y = array![0, 1];
        let result = stratified_kfold_split(&y, 5, Some(42));
        // Should now succeed with adjusted number of splits
        assert!(result.is_ok());
        let splits = result.unwrap();
        // Should have at most 2 splits (same as number of samples)
        assert!(splits.len() <= 2);

        // Test with truly invalid case (fewer than 2 samples)
        let y_invalid = array![0];
        let result_invalid = stratified_kfold_split(&y_invalid, 5, Some(42));
        assert!(result_invalid.is_err());
    }

    #[test]
    fn test_calibrated_classifier_builder() {
        // Mock classifier for testing
        #[derive(Debug, Clone)]
        struct MockClassifier;

        let classifier = CalibratedClassifier::builder(MockClassifier)
            .method(CalibrationMethod::TemperatureScaling {
                initial_temperature: 1.5,
            })
            .cv_folds(3)
            .random_state(Some(42))
            .ensemble(true)
            .build();

        let config = classifier.config();
        assert_eq!(config.cv_folds, 3);
        assert_eq!(config.random_state, Some(42));
        assert!(config.ensemble);

        if let CalibrationMethod::TemperatureScaling {
            initial_temperature,
        } = &config.method
        {
            assert_eq!(*initial_temperature, 1.5);
        } else {
            panic!("Expected TemperatureScaling method");
        }
    }

    #[test]
    fn test_calibrated_classifier_comprehensive() {
        use sklears_core::traits::{Fit, PredictProba};

        // Mock classifier that implements required traits
        #[derive(Debug, Clone)]
        struct MockClassifier;

        #[derive(Debug, Clone)]
        struct MockClassifierTrained {
            classes: Array1<i32>,
        }

        impl Fit<Array2<f64>, Array1<i32>> for MockClassifier {
            type Fitted = MockClassifierTrained;

            fn fit(self, _X: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
                let mut classes = y.to_vec();
                classes.sort_unstable();
                classes.dedup();
                Ok(MockClassifierTrained {
                    classes: Array1::from_vec(classes),
                })
            }
        }

        impl PredictProba<Array2<f64>, Array2<f64>> for MockClassifierTrained {
            fn predict_proba(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
                let (n_samples, _) = X.dim();
                let n_classes = self.classes.len();

                // Return somewhat realistic probabilities based on features
                let mut probs = Array2::zeros((n_samples, n_classes));
                for i in 0..n_samples {
                    let feature_sum: f64 = X.row(i).sum();
                    for j in 0..n_classes {
                        probs[[i, j]] = (feature_sum + j as f64 + 1.0)
                            / ((n_classes as f64) * feature_sum + n_classes as f64);
                    }
                    // Normalize
                    let row_sum: f64 = probs.row(i).sum();
                    if row_sum > 0.0 {
                        for j in 0..n_classes {
                            probs[[i, j]] /= row_sum;
                        }
                    }
                }
                Ok(probs)
            }
        }

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, 0, 1, 2, 2];

        // Test with Platt scaling
        let platt_calibrator = CalibratedClassifier::new(MockClassifier)
            .method(CalibrationMethod::PlattScaling)
            .cv_folds(3);

        let trained_platt = platt_calibrator.fit(&X, &y).unwrap();
        let predictions = trained_platt.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);

        let probabilities = trained_platt.predict_proba(&X).unwrap();
        assert_eq!(probabilities.dim(), (6, 3));

        // Check probability constraints
        for i in 0..6 {
            let row_sum: f64 = probabilities.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }

        // Test with isotonic regression
        let isotonic_calibrator = CalibratedClassifier::new(MockClassifier)
            .method(CalibrationMethod::IsotonicRegression)
            .cv_folds(3);

        let trained_isotonic = isotonic_calibrator.fit(&X, &y).unwrap();
        let iso_predictions = trained_isotonic.predict(&X).unwrap();
        assert_eq!(iso_predictions.len(), 6);

        // Test with temperature scaling
        let temp_calibrator = CalibratedClassifier::new(MockClassifier)
            .method(CalibrationMethod::TemperatureScaling {
                initial_temperature: 1.0,
            })
            .cv_folds(3);

        let trained_temp = temp_calibrator.fit(&X, &y).unwrap();
        let temp_predictions = trained_temp.predict(&X).unwrap();
        assert_eq!(temp_predictions.len(), 6);

        // Test with Dirichlet calibration
        let dirichlet_calibrator = CalibratedClassifier::new(MockClassifier)
            .method(CalibrationMethod::DirichletCalibration { l2_reg: 0.01 })
            .cv_folds(3);

        let trained_dirichlet = dirichlet_calibrator.fit(&X, &y).unwrap();
        let dirichlet_predictions = trained_dirichlet.predict(&X).unwrap();
        assert_eq!(dirichlet_predictions.len(), 6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_calibration_edge_cases() {
        use sklears_core::traits::{Fit, PredictProba};

        #[derive(Debug, Clone)]
        struct MockClassifier;

        #[derive(Debug, Clone)]
        struct MockClassifierTrained;

        impl Fit<Array2<f64>, Array1<i32>> for MockClassifier {
            type Fitted = MockClassifierTrained;

            fn fit(self, _X: &Array2<f64>, _y: &Array1<i32>) -> SklResult<Self::Fitted> {
                Ok(MockClassifierTrained)
            }
        }

        impl PredictProba<Array2<f64>, Array2<f64>> for MockClassifierTrained {
            fn predict_proba(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
                let (n_samples, _) = X.dim();
                // Return uniform probabilities
                let probs = Array2::from_elem((n_samples, 2), 0.5);
                Ok(probs)
            }
        }

        // Test with insufficient data
        let X_small = array![[1.0], [2.0]];
        let y_small = array![0, 1];

        let calibrator = CalibratedClassifier::new(MockClassifier).cv_folds(3); // More folds than samples

        let result = calibrator.fit(&X_small, &y_small);
        // Should still work, just with fewer actual folds
        assert!(result.is_ok());

        // Test with single class (should fail)
        let y_single = array![0, 0];
        let calibrator_single = CalibratedClassifier::new(MockClassifier);
        let result_single = calibrator_single.fit(&X_small, &y_single);
        assert!(result_single.is_err());

        // Test with empty data (should fail)
        let X_empty = Array2::<f64>::zeros((0, 2));
        let y_empty = Array1::<i32>::zeros(0);
        let calibrator_empty = CalibratedClassifier::new(MockClassifier);
        let result_empty = calibrator_empty.fit(&X_empty, &y_empty);
        assert!(result_empty.is_err());
    }

    #[test]
    fn test_all_calibration_methods() {
        // Test that all calibration methods can be created and configured
        let methods = vec![
            CalibrationMethod::PlattScaling,
            CalibrationMethod::IsotonicRegression,
            CalibrationMethod::TemperatureScaling {
                initial_temperature: 2.0,
            },
            CalibrationMethod::DirichletCalibration { l2_reg: 0.1 },
        ];

        for method in methods {
            #[derive(Debug, Clone)]
            struct DummyClassifier;

            let calibrator = CalibratedClassifier::builder(DummyClassifier)
                .method(method)
                .cv_folds(5)
                .random_state(Some(42))
                .build();

            // Just verify the configuration is stored correctly
            assert_eq!(calibrator.config().cv_folds, 5);
            assert_eq!(calibrator.config().random_state, Some(42));
        }
    }
}
