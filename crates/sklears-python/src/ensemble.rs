//! Python bindings for ensemble methods
//!
//! This module provides PyO3-based Python bindings for sklears ensemble algorithms.
//! It includes implementations for Gradient Boosting, AdaBoost, Voting, and Stacking classifiers.

use crate::utils::{numpy_to_ndarray1, numpy_to_ndarray2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict, Trained, Untrained};
use sklears_ensemble::gradient_boosting::{
    TrainedGradientBoostingClassifier, TrainedGradientBoostingRegressor,
};
use sklears_ensemble::{
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, GradientBoostingConfig,
    GradientBoostingRegressor, LossFunction, VotingClassifier, VotingClassifierConfig,
    VotingStrategy,
};

/// Python wrapper for GradientBoostingClassifier
#[pyclass(name = "GradientBoostingClassifier")]
pub struct PyGradientBoostingClassifier {
    inner: Option<GradientBoostingClassifier>,
    trained: Option<TrainedGradientBoostingClassifier>,
}

#[pymethods]
impl PyGradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        loss="squared_loss",
        random_state=None,
        validation_fraction=0.1
    ))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        subsample: f64,
        loss: &str,
        random_state: Option<u64>,
        validation_fraction: f64,
    ) -> PyResult<Self> {
        let loss_function = match loss {
            "squared_loss" => LossFunction::SquaredLoss,
            "absolute_loss" => LossFunction::AbsoluteLoss,
            "huber" => LossFunction::HuberLoss,
            "quantile" => LossFunction::QuantileLoss,
            "logistic" => LossFunction::LogisticLoss,
            "deviance" => LossFunction::DevianceLoss,
            "exponential" => LossFunction::ExponentialLoss,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown loss function: {}",
                    loss
                )))
            }
        };

        let config = GradientBoostingConfig {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            subsample,
            loss_function,
            random_state,
            validation_fraction,
            ..Default::default()
        };

        Ok(Self {
            inner: Some(GradientBoostingClassifier::new(config)),
            trained: None,
        })
    }

    /// Fit the gradient boosting classifier
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        let predictions: Array1<f64> =
            Predict::<Array2<f64>, Array1<f64>>::predict(trained_model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        Ok(predictions.into_pyarray(py).unbind())
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        let importances = trained_model.feature_importances_gain();
        Ok(importances.clone().into_pyarray(py).unbind())
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "GradientBoostingClassifier(fitted=True)".to_string()
        } else {
            "GradientBoostingClassifier(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for GradientBoostingRegressor
#[pyclass(name = "GradientBoostingRegressor")]
pub struct PyGradientBoostingRegressor {
    inner: Option<GradientBoostingRegressor>,
    trained: Option<TrainedGradientBoostingRegressor>,
}

#[pymethods]
impl PyGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        loss="squared_loss",
        random_state=None,
        validation_fraction=0.1
    ))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        subsample: f64,
        loss: &str,
        random_state: Option<u64>,
        validation_fraction: f64,
    ) -> PyResult<Self> {
        let loss_function = match loss {
            "squared_loss" => LossFunction::SquaredLoss,
            "absolute_loss" => LossFunction::AbsoluteLoss,
            "huber" => LossFunction::HuberLoss,
            "quantile" => LossFunction::QuantileLoss,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown loss function for regression: {}",
                    loss
                )))
            }
        };

        let config = GradientBoostingConfig {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            subsample,
            loss_function,
            random_state,
            validation_fraction,
            ..Default::default()
        };

        Ok(Self {
            inner: Some(GradientBoostingRegressor::new(config)),
            trained: None,
        })
    }

    /// Fit the gradient boosting regressor
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        let predictions: Array1<f64> =
            Predict::<Array2<f64>, Array1<f64>>::predict(trained_model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        Ok(predictions.into_pyarray(py).unbind())
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "GradientBoostingRegressor(fitted=True)".to_string()
        } else {
            "GradientBoostingRegressor(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for AdaBoost Classifier
#[pyclass(name = "AdaBoostClassifier")]
pub struct PyAdaBoostClassifier {
    inner: Option<AdaBoostClassifier<Untrained>>,
    trained: Option<AdaBoostClassifier<Trained>>,
}

#[pymethods]
impl PyAdaBoostClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=50, learning_rate=1.0, random_state=None))]
    fn new(n_estimators: usize, learning_rate: f64, random_state: Option<u64>) -> PyResult<Self> {
        let mut model = AdaBoostClassifier::new()
            .n_estimators(n_estimators)
            .learning_rate(learning_rate);

        if let Some(seed) = random_state {
            model = model.random_state(seed);
        }

        Ok(Self {
            inner: Some(model),
            trained: None,
        })
    }

    /// Fit the AdaBoost classifier
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        let predictions: Array1<f64> =
            Predict::<Array2<f64>, Array1<f64>>::predict(trained_model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        Ok(predictions.into_pyarray(py).unbind())
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "AdaBoostClassifier(fitted=True)".to_string()
        } else {
            "AdaBoostClassifier(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Voting Classifier
#[pyclass(name = "VotingClassifier")]
pub struct PyVotingClassifier {
    inner: Option<VotingClassifier<Untrained>>,
    trained: Option<VotingClassifier<Trained>>,
}

#[pymethods]
impl PyVotingClassifier {
    #[new]
    #[pyo3(signature = (_estimators, voting="hard", weights=None))]
    fn new(
        _estimators: &Bound<'_, PyList>,
        voting: &str,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let voting_strategy = match voting {
            "hard" => VotingStrategy::Hard,
            "soft" => VotingStrategy::Soft,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown voting strategy: {}",
                    voting
                )))
            }
        };

        let config = VotingClassifierConfig {
            voting: voting_strategy,
            weights,
            ..Default::default()
        };

        Ok(Self {
            inner: Some(VotingClassifier::new(config)),
            trained: None,
        })
    }

    /// Fit the voting classifier
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        let predictions: Array1<f64> =
            Predict::<Array2<f64>, Array1<f64>>::predict(trained_model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        Ok(predictions.into_pyarray(py).unbind())
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "VotingClassifier(fitted=True)".to_string()
        } else {
            "VotingClassifier(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Bagging Classifier
#[pyclass(name = "BaggingClassifier")]
pub struct PyBaggingClassifier {
    inner: Option<BaggingClassifier<Untrained>>,
    trained: Option<BaggingClassifier<Trained>>,
}

#[pymethods]
impl PyBaggingClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=None,
        max_features=None,
        bootstrap=true,
        bootstrap_features=false,
        random_state=None
    ))]
    fn new(
        n_estimators: usize,
        max_samples: Option<usize>,
        max_features: Option<usize>,
        bootstrap: bool,
        bootstrap_features: bool,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let mut model = BaggingClassifier::new()
            .n_estimators(n_estimators)
            .bootstrap(bootstrap)
            .bootstrap_features(bootstrap_features);

        if let Some(samples) = max_samples {
            model = model.max_samples(Some(samples));
        }

        if let Some(features) = max_features {
            model = model.max_features(Some(features));
        }

        if let Some(seed) = random_state {
            model = model.random_state(seed);
        }

        Ok(Self {
            inner: Some(model),
            trained: None,
        })
    }

    /// Fit the bagging classifier
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        // Convert y to integer array for BaggingClassifier (Fit<Array2<Float>, Array1<Int>>)
        let y_int: Vec<i32> = y_array.iter().map(|&val| val as i32).collect();
        let y_int_array = Array1::from_vec(y_int);

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_int_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        let predictions: Array1<i32> =
            Predict::<Array2<f64>, Array1<i32>>::predict(trained_model, &x_array)
                .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        // Convert i32 predictions to f64
        let predictions_f64: Vec<f64> = predictions.iter().map(|&v| v as f64).collect();
        Ok(PyArray1::from_vec(py, predictions_f64).unbind())
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "BaggingClassifier(fitted=True)".to_string()
        } else {
            "BaggingClassifier(fitted=False)".to_string()
        }
    }
}
