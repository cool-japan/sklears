//! Python bindings for Naive Bayes classifiers
//!
//! This module provides PyO3-based Python bindings for sklears Naive Bayes algorithms,
//! including Gaussian, Multinomial, Bernoulli, Complement, and Categorical Naive Bayes.

use crate::utils::{numpy_to_ndarray1, numpy_to_ndarray2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use sklears_core::error::{Result as SklearsResult, SklearsError};
use sklears_core::traits::{Fit, Predict, PredictProba};
use sklears_naive_bayes::{BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB};

/// Python wrapper for Gaussian Naive Bayes
#[pyclass(name = "GaussianNB")]
pub struct PyGaussianNB {
    inner: Option<GaussianNB>,
    trained: Option<sklears_naive_bayes::TrainedGaussianNB>,
}

#[pymethods]
impl PyGaussianNB {
    #[new]
    #[pyo3(signature = (priors=None, var_smoothing=1e-9))]
    fn new(priors: Option<Vec<f64>>, var_smoothing: f64) -> PyResult<Self> {
        let mut nb = GaussianNB::new().var_smoothing(var_smoothing);

        if let Some(prior_values) = priors {
            let priors_array = ndarray::Array1::from_vec(prior_values);
            nb = nb.priors(priors_array);
        }

        Ok(Self {
            inner: Some(nb),
            trained: None,
        })
    }

    /// Fit the Gaussian Naive Bayes classifier
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        // Convert y to integer vector for classification
        let y_int: Vec<i32> = y_array.iter().map(|&val| val as i32).collect();
        let y_int_array = ndarray::Array1::from_vec(y_int);

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
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => {
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
                Ok(PyArray1::from_vec(py, predictions_f64))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Predict class probabilities
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &PyArray2<f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict_proba(&x_array) {
            Ok(probabilities) => Ok(probabilities.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Probability prediction failed: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "GaussianNB(fitted=True)".to_string()
        } else {
            "GaussianNB(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Multinomial Naive Bayes
#[pyclass(name = "MultinomialNB")]
pub struct PyMultinomialNB {
    inner: Option<MultinomialNB>,
    trained: Option<sklears_naive_bayes::TrainedMultinomialNB>,
}

#[pymethods]
impl PyMultinomialNB {
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_prior=true, class_prior=None))]
    fn new(alpha: f64, fit_prior: bool, class_prior: Option<Vec<f64>>) -> PyResult<Self> {
        let mut nb = MultinomialNB::new().alpha(alpha).fit_prior(fit_prior);

        if let Some(prior_values) = class_prior {
            let priors_array = ndarray::Array1::from_vec(prior_values);
            nb = nb.class_prior(priors_array);
        }

        Ok(Self {
            inner: Some(nb),
            trained: None,
        })
    }

    /// Fit the Multinomial Naive Bayes classifier
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        // Convert y to integer vector for classification
        let y_int: Vec<i32> = y_array.iter().map(|&val| val as i32).collect();
        let y_int_array = ndarray::Array1::from_vec(y_int);

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
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => {
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
                Ok(PyArray1::from_vec(py, predictions_f64))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Predict class probabilities
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &PyArray2<f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict_proba(&x_array) {
            Ok(probabilities) => Ok(probabilities.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Probability prediction failed: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "MultinomialNB(fitted=True)".to_string()
        } else {
            "MultinomialNB(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Bernoulli Naive Bayes
#[pyclass(name = "BernoulliNB")]
pub struct PyBernoulliNB {
    inner: Option<BernoulliNB>,
    trained: Option<sklears_naive_bayes::TrainedBernoulliNB>,
}

#[pymethods]
impl PyBernoulliNB {
    #[new]
    #[pyo3(signature = (alpha=1.0, binarize=0.0, fit_prior=true, class_prior=None))]
    fn new(
        alpha: f64,
        binarize: f64,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mut nb = BernoulliNB::new()
            .alpha(alpha)
            .binarize(binarize)
            .fit_prior(fit_prior);

        if let Some(prior_values) = class_prior {
            let priors_array = ndarray::Array1::from_vec(prior_values);
            nb = nb.class_prior(priors_array);
        }

        Ok(Self {
            inner: Some(nb),
            trained: None,
        })
    }

    /// Fit the Bernoulli Naive Bayes classifier
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        // Convert y to integer vector for classification
        let y_int: Vec<i32> = y_array.iter().map(|&val| val as i32).collect();
        let y_int_array = ndarray::Array1::from_vec(y_int);

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
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => {
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
                Ok(PyArray1::from_vec(py, predictions_f64))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Predict class probabilities
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &PyArray2<f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict_proba(&x_array) {
            Ok(probabilities) => Ok(probabilities.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Probability prediction failed: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "BernoulliNB(fitted=True)".to_string()
        } else {
            "BernoulliNB(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Complement Naive Bayes
#[pyclass(name = "ComplementNB")]
pub struct PyComplementNB {
    inner: Option<ComplementNB>,
    trained: Option<sklears_naive_bayes::TrainedComplementNB>,
}

#[pymethods]
impl PyComplementNB {
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_prior=true, class_prior=None, norm=false))]
    fn new(
        alpha: f64,
        fit_prior: bool,
        class_prior: Option<Vec<f64>>,
        norm: bool,
    ) -> PyResult<Self> {
        let mut nb = ComplementNB::new()
            .alpha(alpha)
            .fit_prior(fit_prior)
            .norm(norm);

        if let Some(prior_values) = class_prior {
            let priors_array = ndarray::Array1::from_vec(prior_values);
            nb = nb.class_prior(priors_array);
        }

        Ok(Self {
            inner: Some(nb),
            trained: None,
        })
    }

    /// Fit the Complement Naive Bayes classifier
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        // Convert y to integer vector for classification
        let y_int: Vec<i32> = y_array.iter().map(|&val| val as i32).collect();
        let y_int_array = ndarray::Array1::from_vec(y_int);

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
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => {
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
                Ok(PyArray1::from_vec(py, predictions_f64))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "ComplementNB(fitted=True)".to_string()
        } else {
            "ComplementNB(fitted=False)".to_string()
        }
    }
}
