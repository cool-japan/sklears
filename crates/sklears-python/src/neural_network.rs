//! Python bindings for neural network models
//!
//! This module provides PyO3-based Python bindings for sklears neural network algorithms,
//! including Multi-Layer Perceptron (MLP) classifiers and regressors.

use crate::utils::{numpy_to_ndarray1, numpy_to_ndarray2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use sklears_core::traits::{Fit, Predict};
use sklears_neural::solvers::LearningRateSchedule;
use sklears_neural::{Activation, MLPClassifier, MLPRegressor, Solver};

/// Python wrapper for MLP Classifier
#[pyclass(name = "MLPClassifier")]
pub struct PyMLPClassifier {
    inner: Option<MLPClassifier<sklears_core::traits::Untrained>>,
    trained: Option<MLPClassifier<sklears_neural::TrainedMLPClassifier>>,
}

#[pymethods]
impl PyMLPClassifier {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        hidden_layer_sizes=None,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=None,
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=true,
        random_state=None,
        tol=1e-4,
        verbose=false,
        warm_start=false,
        momentum=0.9,
        nesterovs_momentum=true,
        early_stopping=false,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000
    ))]
    fn new(
        hidden_layer_sizes: Option<Vec<usize>>,
        activation: &str,
        solver: &str,
        alpha: f64,
        batch_size: Option<usize>,
        learning_rate: &str,
        learning_rate_init: f64,
        power_t: f64,
        max_iter: usize,
        shuffle: bool,
        random_state: Option<u64>,
        tol: f64,
        verbose: bool,
        warm_start: bool,
        momentum: f64,
        nesterovs_momentum: bool,
        early_stopping: bool,
        validation_fraction: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        n_iter_no_change: usize,
        max_fun: usize,
    ) -> PyResult<Self> {
        let activation = match activation {
            "identity" => Activation::Identity,
            "logistic" => Activation::Logistic,
            "tanh" => Activation::Tanh,
            "relu" => Activation::Relu,
            "elu" => Activation::Elu,
            "swish" => Activation::Swish,
            "gelu" => Activation::Gelu,
            "mish" => Activation::Mish,
            "leaky_relu" => Activation::LeakyRelu,
            "prelu" => Activation::PRelu,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown activation: {}",
                    activation
                )))
            }
        };

        let solver = match solver {
            "lbfgs" => Solver::Lbfgs,
            "sgd" => Solver::Sgd,
            "adam" => Solver::Adam,
            _ => return Err(PyValueError::new_err(format!("Unknown solver: {}", solver))),
        };

        let learning_rate_schedule = match learning_rate {
            "constant" => LearningRateSchedule::Constant,
            "invscaling" => LearningRateSchedule::InvScaling,
            "adaptive" => LearningRateSchedule::Adaptive,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown learning rate schedule: {}",
                    learning_rate
                )))
            }
        };

        let hidden_sizes = hidden_layer_sizes.unwrap_or_else(|| vec![100]);

        let mut mlp = MLPClassifier::new();
        mlp.hidden_layer_sizes = hidden_sizes;
        mlp.activation = activation;
        mlp.solver = solver;
        mlp.alpha = alpha;
        mlp.batch_size = batch_size;
        mlp.learning_rate = learning_rate_schedule;
        mlp.learning_rate_init = learning_rate_init;
        mlp.power_t = power_t;
        mlp.max_iter = max_iter;
        mlp.shuffle = shuffle;
        mlp.random_state = random_state;
        mlp.tol = tol;
        mlp.verbose = verbose;
        mlp.warm_start = warm_start;
        mlp.momentum = momentum;
        mlp.nesterovs_momentum = nesterovs_momentum;
        mlp.early_stopping = early_stopping;
        mlp.validation_fraction = validation_fraction;
        mlp.beta_1 = beta_1;
        mlp.beta_2 = beta_2;
        mlp.epsilon = epsilon;
        mlp.n_iter_no_change = n_iter_no_change;
        mlp.max_fun = max_fun;

        Ok(Self {
            inner: Some(mlp),
            trained: None,
        })
    }

    /// Fit the MLP classifier
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        // Convert y to integer vector for classification
        let y_int: Vec<usize> = y_array.iter().map(|&val| val as usize).collect();

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_int) {
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

        match trained_model.predict(&x_array) {
            Ok(predictions) => {
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
                Ok(PyArray1::from_vec(py, predictions_f64).unbind())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Predict class probabilities
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict_proba(&x_array) {
            Ok(probabilities) => Ok(probabilities.into_pyarray(py).unbind()),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Probability prediction failed: {}",
                e
            ))),
        }
    }

    /// Get the loss after training
    fn loss_(&self) -> PyResult<f64> {
        let trained_model = self
            .trained
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model must be fitted before accessing loss"))?;

        Ok(trained_model.loss())
    }

    /// Get number of iterations
    fn n_iter_(&self) -> PyResult<usize> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing n_iter")
        })?;

        Ok(trained_model.n_iter())
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "MLPClassifier(fitted=True)".to_string()
        } else {
            "MLPClassifier(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for MLP Regressor
#[pyclass(name = "MLPRegressor")]
pub struct PyMLPRegressor {
    inner: Option<MLPRegressor<sklears_core::traits::Untrained>>,
    trained: Option<MLPRegressor<sklears_neural::TrainedMLPRegressor>>,
}

#[pymethods]
impl PyMLPRegressor {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        hidden_layer_sizes=None,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=None,
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=true,
        random_state=None,
        tol=1e-4,
        verbose=false,
        warm_start=false,
        momentum=0.9,
        nesterovs_momentum=true,
        early_stopping=false,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000
    ))]
    fn new(
        hidden_layer_sizes: Option<Vec<usize>>,
        activation: &str,
        solver: &str,
        alpha: f64,
        batch_size: Option<usize>,
        learning_rate: &str,
        learning_rate_init: f64,
        power_t: f64,
        max_iter: usize,
        shuffle: bool,
        random_state: Option<u64>,
        tol: f64,
        verbose: bool,
        warm_start: bool,
        momentum: f64,
        nesterovs_momentum: bool,
        early_stopping: bool,
        validation_fraction: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        n_iter_no_change: usize,
        max_fun: usize,
    ) -> PyResult<Self> {
        let activation = match activation {
            "identity" => Activation::Identity,
            "logistic" => Activation::Logistic,
            "tanh" => Activation::Tanh,
            "relu" => Activation::Relu,
            "elu" => Activation::Elu,
            "swish" => Activation::Swish,
            "gelu" => Activation::Gelu,
            "mish" => Activation::Mish,
            "leaky_relu" => Activation::LeakyRelu,
            "prelu" => Activation::PRelu,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown activation: {}",
                    activation
                )))
            }
        };

        let solver = match solver {
            "lbfgs" => Solver::Lbfgs,
            "sgd" => Solver::Sgd,
            "adam" => Solver::Adam,
            _ => return Err(PyValueError::new_err(format!("Unknown solver: {}", solver))),
        };

        let learning_rate_schedule = match learning_rate {
            "constant" => LearningRateSchedule::Constant,
            "invscaling" => LearningRateSchedule::InvScaling,
            "adaptive" => LearningRateSchedule::Adaptive,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown learning rate schedule: {}",
                    learning_rate
                )))
            }
        };

        let hidden_sizes = hidden_layer_sizes.unwrap_or_else(|| vec![100]);

        let mut mlp = MLPRegressor::new();
        mlp.hidden_layer_sizes = hidden_sizes;
        mlp.activation = activation;
        mlp.solver = solver;
        mlp.alpha = alpha;
        mlp.batch_size = batch_size;
        mlp.learning_rate = learning_rate_schedule;
        mlp.learning_rate_init = learning_rate_init;
        mlp.power_t = power_t;
        mlp.max_iter = max_iter;
        mlp.shuffle = shuffle;
        mlp.random_state = random_state;
        mlp.tol = tol;
        mlp.verbose = verbose;
        mlp.warm_start = warm_start;
        mlp.momentum = momentum;
        mlp.nesterovs_momentum = nesterovs_momentum;
        mlp.early_stopping = early_stopping;
        mlp.validation_fraction = validation_fraction;
        mlp.beta_1 = beta_1;
        mlp.beta_2 = beta_2;
        mlp.epsilon = epsilon;
        mlp.n_iter_no_change = n_iter_no_change;
        mlp.max_fun = max_fun;

        Ok(Self {
            inner: Some(mlp),
            trained: None,
        })
    }

    /// Fit the MLP regressor
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array_1d = numpy_to_ndarray1(y)?;

        // Convert y from 1D to 2D array (n_samples, 1)
        let y_array = y_array_1d.insert_axis(scirs2_core::ndarray::Axis(1));

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

        match trained_model.predict(&x_array) {
            Ok(predictions_2d) => {
                // Convert from Array2 (n_samples, 1) to Array1 (n_samples,)
                let predictions_1d = predictions_2d
                    .index_axis(scirs2_core::ndarray::Axis(1), 0)
                    .to_owned();
                Ok(predictions_1d.into_pyarray(py).unbind())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Get the loss after training
    fn loss_(&self) -> PyResult<f64> {
        let trained_model = self
            .trained
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model must be fitted before accessing loss"))?;

        Ok(trained_model.loss())
    }

    /// Get number of iterations
    fn n_iter_(&self) -> PyResult<usize> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing n_iter")
        })?;

        Ok(trained_model.n_iter())
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "MLPRegressor(fitted=True)".to_string()
        } else {
            "MLPRegressor(fitted=False)".to_string()
        }
    }
}
