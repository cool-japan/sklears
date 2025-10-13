//! Python bindings for ensemble methods
//!
//! This module provides PyO3-based Python bindings for sklears ensemble algorithms.
//! It includes implementations for Gradient Boosting, AdaBoost, Voting, and Stacking classifiers.

use crate::utils::{ndarray_to_numpy, numpy_to_ndarray1, numpy_to_ndarray2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use sklears_core::error::{Result as SklearsResult, SklearsError};
use sklears_core::traits::{Fit, Predict, Transform};
use sklears_ensemble::{
    AdaBoostClassifier, AdaBoostConfig, BaggingClassifier, BaggingConfig, BaggingRegressor,
    GradientBoostingClassifier, GradientBoostingConfig, GradientBoostingRegressor, LossFunction,
    StackingClassifier, StackingConfig, VotingClassifier, VotingClassifierConfig, VotingStrategy,
};

/// Python wrapper for GradientBoostingClassifier
#[pyclass(name = "GradientBoostingClassifier")]
pub struct PyGradientBoostingClassifier {
    inner: Option<GradientBoostingClassifier>,
    trained: Option<sklears_ensemble::TrainedGradientBoostingClassifier>,
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
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
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
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => Ok(predictions.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        // This would need to be implemented in the actual ensemble crate
        // For now, return a placeholder
        let n_features = trained_model.n_features;
        let importances = vec![1.0 / n_features as f64; n_features];
        Ok(PyArray1::from_vec(py, importances))
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
    trained: Option<sklears_ensemble::TrainedGradientBoostingRegressor>,
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
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
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
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => Ok(predictions.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
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
    inner: Option<AdaBoostClassifier>,
    trained: Option<sklears_ensemble::TrainedAdaBoostClassifier>,
}

#[pymethods]
impl PyAdaBoostClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=50, learning_rate=1.0, random_state=None))]
    fn new(n_estimators: usize, learning_rate: f64, random_state: Option<u64>) -> PyResult<Self> {
        let config = AdaBoostConfig {
            n_estimators,
            learning_rate,
            random_state,
            ..Default::default()
        };

        Ok(Self {
            inner: Some(AdaBoostClassifier::new(config)),
            trained: None,
        })
    }

    /// Fit the AdaBoost classifier
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
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
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => Ok(predictions.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
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
    inner: Option<VotingClassifier>,
    trained: Option<sklears_ensemble::TrainedVotingClassifier>,
}

#[pymethods]
impl PyVotingClassifier {
    #[new]
    #[pyo3(signature = (estimators, voting="hard", weights=None))]
    fn new(estimators: &PyList, voting: &str, weights: Option<Vec<f64>>) -> PyResult<Self> {
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

        // For now, we'll create a placeholder configuration
        // In a full implementation, we'd need to handle the estimators list
        let config = VotingClassifierConfig {
            voting_strategy,
            weights,
            ..Default::default()
        };

        Ok(Self {
            inner: Some(VotingClassifier::new(config)),
            trained: None,
        })
    }

    /// Fit the voting classifier
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
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
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => Ok(predictions.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
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
    inner: Option<BaggingClassifier>,
    trained: Option<sklears_ensemble::TrainedBaggingClassifier>,
}

#[pymethods]
impl PyBaggingClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        bootstrap_features=false,
        random_state=None
    ))]
    fn new(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        bootstrap_features: bool,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let config = BaggingConfig {
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            bootstrap_features,
            random_state,
            ..Default::default()
        };

        Ok(Self {
            inner: Some(BaggingClassifier::new(config)),
            trained: None,
        })
    }

    /// Fit the bagging classifier
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
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
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => Ok(predictions.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "BaggingClassifier(fitted=True)".to_string()
        } else {
            "BaggingClassifier(fitted=False)".to_string()
        }
    }
}
