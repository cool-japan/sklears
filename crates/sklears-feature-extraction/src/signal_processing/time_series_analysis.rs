//! Time series analysis and feature extraction
//!
//! This module provides comprehensive time series analysis tools including:
//! - Autoregressive (AR) model fitting and coefficient extraction
//! - Cross-correlation analysis with templates and autocorrelation
//! - Statistical analysis of temporal patterns
//! - Advanced time domain feature extraction methods

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};

/// Autoregressive (AR) features extractor
///
/// Fits an autoregressive model to the signal and uses the AR coefficients as features.
#[derive(Debug, Clone)]
pub struct AutoregressiveExtractor {
    order: usize,
    method: String, // "yule_walker", "burg", "least_squares"
    normalize: bool,
}

impl AutoregressiveExtractor {
    /// Create a new AR extractor
    pub fn new() -> Self {
        Self {
            order: 10,
            method: "yule_walker".to_string(),
            normalize: true,
        }
    }

    /// Set the AR model order
    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Set the estimation method
    pub fn method(mut self, method: String) -> Self {
        self.method = method;
        self
    }

    /// Set whether to normalize coefficients
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Extract AR features from signal
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        if signal.len() <= self.order {
            return Err(SklearsError::InvalidInput(
                "Signal too short for AR order".to_string(),
            ));
        }

        let ar_coeffs = match self.method.as_str() {
            "yule_walker" => self.yule_walker(signal)?,
            "least_squares" => self.least_squares(signal)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown AR method: {}",
                    self.method
                )))
            }
        };

        let mut result = Array1::from_vec(ar_coeffs);

        if self.normalize {
            let norm = result.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                result.mapv_inplace(|x| x / norm);
            }
        }

        Ok(result)
    }

    /// Yule-Walker estimation
    fn yule_walker(&self, signal: &ArrayView1<f64>) -> SklResult<Vec<f64>> {
        // Compute autocorrelation
        let autocorr = self.autocorrelation(signal, self.order + 1);

        // Set up Yule-Walker equations: R * a = -r
        let mut R = Array2::zeros((self.order, self.order));
        let mut r = Array1::zeros(self.order);

        for i in 0..self.order {
            r[i] = autocorr[i + 1];
            for j in 0..self.order {
                R[[i, j]] = autocorr[(i as i32 - j as i32).unsigned_abs() as usize];
            }
        }

        // Solve the system using simple Gaussian elimination
        let coeffs = self.solve_linear_system(&R, &r)?;

        Ok(coeffs.to_vec())
    }

    /// Least squares estimation
    fn least_squares(&self, signal: &ArrayView1<f64>) -> SklResult<Vec<f64>> {
        let n = signal.len();
        let p = self.order;

        if n <= p {
            return Err(SklearsError::InvalidInput(
                "Signal too short for least squares AR".to_string(),
            ));
        }

        // Build design matrix X and target vector y
        let mut X = Array2::zeros((n - p, p));
        let mut y = Array1::zeros(n - p);

        for i in 0..(n - p) {
            y[i] = signal[i + p];
            for j in 0..p {
                X[[i, j]] = signal[i + p - 1 - j];
            }
        }

        // Solve X^T X a = X^T y
        let XtX = X.t().dot(&X);
        let Xty = X.t().dot(&y);

        let coeffs = self.solve_linear_system(&XtX, &Xty)?;

        // Negate coefficients (AR convention)
        Ok(coeffs.iter().map(|&x| -x).collect())
    }

    /// Compute autocorrelation
    fn autocorrelation(&self, signal: &ArrayView1<f64>, max_lag: usize) -> Vec<f64> {
        let n = signal.len();
        let mut autocorr = vec![0.0; max_lag + 1];

        for lag in 0..=max_lag {
            let mut sum = 0.0;
            let count = n - lag;

            for i in 0..count {
                sum += signal[i] * signal[i + lag];
            }

            autocorr[lag] = sum / count as f64;
        }

        autocorr
    }

    /// Simple linear system solver using Gaussian elimination
    fn solve_linear_system(&self, A: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
        let n = A.nrows();
        if A.ncols() != n || b.len() != n {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions mismatch".to_string(),
            ));
        }

        let mut aug = Array2::zeros((n, n + 1));

        // Create augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = A[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug[[i, i]].abs() < 1e-10 {
                return Err(SklearsError::InvalidInput("Singular matrix".to_string()));
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = aug[[k, i]] / aug[[i, i]];
                for j in i..=n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug[[i, n]];
            for j in (i + 1)..n {
                x[i] -= aug[[i, j]] * x[j];
            }
            x[i] /= aug[[i, i]];
        }

        Ok(x)
    }
}

impl Default for AutoregressiveExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-correlation features extractor
///
/// Computes cross-correlation features between signals or between a signal and templates.
#[derive(Debug, Clone)]
pub struct CrossCorrelationExtractor {
    max_lag: usize,
    normalize: bool,
    mode: String, // "full", "valid", "same"
    templates: Option<Vec<Array1<f64>>>,
}

impl CrossCorrelationExtractor {
    /// Create a new cross-correlation extractor
    pub fn new() -> Self {
        Self {
            max_lag: 50,
            normalize: true,
            mode: "valid".to_string(),
            templates: None,
        }
    }

    /// Set the maximum lag for correlation
    pub fn max_lag(mut self, max_lag: usize) -> Self {
        self.max_lag = max_lag;
        self
    }

    /// Set whether to normalize correlations
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the correlation mode
    pub fn mode(mut self, mode: String) -> Self {
        self.mode = mode;
        self
    }

    /// Set template signals for correlation
    pub fn templates(mut self, templates: Vec<Array1<f64>>) -> Self {
        self.templates = Some(templates);
        self
    }

    /// Extract cross-correlation features
    pub fn extract_features(&self, signal: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        if signal.is_empty() {
            return Err(SklearsError::InvalidInput("Empty signal".to_string()));
        }

        let mut features = Vec::new();

        if let Some(ref templates) = self.templates {
            // Cross-correlation with templates
            for template in templates {
                let corr = self.cross_correlate(signal, &template.view())?;

                // Extract features from correlation
                let max_corr = corr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let max_idx = corr
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                features.push(max_corr);
                features.push(max_idx as f64);

                // Additional statistics
                let mean_corr = corr.mean().unwrap_or(0.0);
                let std_corr = corr.std(0.0);
                features.push(mean_corr);
                features.push(std_corr);
            }
        } else {
            // Auto-correlation features
            let autocorr = self.autocorrelation(signal);

            // Extract features from autocorrelation
            features.extend_from_slice(&autocorr.to_vec());
        }

        Ok(Array1::from_vec(features))
    }

    /// Compute cross-correlation between two signals
    fn cross_correlate(
        &self,
        signal1: &ArrayView1<f64>,
        signal2: &ArrayView1<f64>,
    ) -> SklResult<Array1<f64>> {
        let n1 = signal1.len();
        let n2 = signal2.len();

        match self.mode.as_str() {
            "full" => {
                let n_out = n1 + n2 - 1;
                let mut result = Array1::zeros(n_out);

                for i in 0..n_out {
                    let mut sum = 0.0;
                    let start1 = i.saturating_sub(n2 - 1);
                    let end1 = (i + 1).min(n1);

                    for j in start1..end1 {
                        let k = i - j;
                        if k < n2 {
                            sum += signal1[j] * signal2[n2 - 1 - k];
                        }
                    }

                    result[i] = sum;
                }

                if self.normalize {
                    let norm1 = signal1.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let norm2 = signal2.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm1 > 0.0 && norm2 > 0.0 {
                        result.mapv_inplace(|x| x / (norm1 * norm2));
                    }
                }

                Ok(result)
            }
            "valid" => {
                if n2 > n1 {
                    return Err(SklearsError::InvalidInput(
                        "Template longer than signal in valid mode".to_string(),
                    ));
                }

                let n_out = n1 - n2 + 1;
                let mut result = Array1::zeros(n_out);

                for i in 0..n_out {
                    let mut sum = 0.0;
                    for j in 0..n2 {
                        sum += signal1[i + j] * signal2[j];
                    }
                    result[i] = sum;
                }

                if self.normalize {
                    let norm1 = signal1.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let norm2 = signal2.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm1 > 0.0 && norm2 > 0.0 {
                        result.mapv_inplace(|x| x / (norm1 * norm2));
                    }
                }

                Ok(result)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown correlation mode: {}",
                self.mode
            ))),
        }
    }

    /// Compute autocorrelation
    fn autocorrelation(&self, signal: &ArrayView1<f64>) -> Array1<f64> {
        let n = signal.len();
        let max_lag = self.max_lag.min(n - 1);
        let mut autocorr = Array1::zeros(max_lag + 1);

        for lag in 0..=max_lag {
            let mut sum = 0.0;
            let count = n - lag;

            for i in 0..count {
                sum += signal[i] * signal[i + lag];
            }

            autocorr[lag] = sum / count as f64;
        }

        if self.normalize && autocorr[0] > 0.0 {
            let norm_factor = autocorr[0];
            autocorr.mapv_inplace(|x| x / norm_factor);
        }

        autocorr
    }
}

impl Default for CrossCorrelationExtractor {
    fn default() -> Self {
        Self::new()
    }
}
