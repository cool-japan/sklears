//! Power transformations to make data more Gaussian-like
//!
//! This module provides Box-Cox and Yeo-Johnson power transformations that can help
//! make data more normally distributed, improving the performance of many ML algorithms.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for PowerTransformer
#[derive(Debug, Clone)]
pub struct PowerTransformerConfig {
    /// Power transformation method
    pub method: PowerMethod,
    /// Whether to apply zero-mean, unit-variance normalization
    pub standardize: bool,
    /// Small constant to add for numerical stability
    pub eps: Float,
}

impl Default for PowerTransformerConfig {
    fn default() -> Self {
        Self {
            method: PowerMethod::YeoJohnson,
            standardize: true,
            eps: 1e-6,
        }
    }
}

/// Power transformation methods
#[derive(Debug, Clone, Copy)]
pub enum PowerMethod {
    /// Box-Cox transformation (requires positive data)
    BoxCox,
    /// Yeo-Johnson transformation (works with any data)
    YeoJohnson,
}

/// PowerTransformer applies power transformations to make data more Gaussian
///
/// This transformer applies power transformations to each feature to make
/// the data more Gaussian-like, which can improve the performance of many
/// machine learning algorithms.
#[derive(Debug, Clone)]
pub struct PowerTransformer<State = Untrained> {
    config: PowerTransformerConfig,
    state: PhantomData<State>,
    // Fitted parameters
    n_features_in_: Option<usize>,
    lambdas_: Option<Array1<Float>>, // Optimal lambda parameters for each feature
    means_: Option<Array1<Float>>,   // Means for standardization
    stds_: Option<Array1<Float>>,    // Standard deviations for standardization
}

impl PowerTransformer<Untrained> {
    /// Create a new PowerTransformer
    pub fn new() -> Self {
        Self {
            config: PowerTransformerConfig::default(),
            state: PhantomData,
            n_features_in_: None,
            lambdas_: None,
            means_: None,
            stds_: None,
        }
    }

    /// Create a new PowerTransformer with custom configuration
    pub fn with_config(config: PowerTransformerConfig) -> Self {
        Self {
            config,
            state: PhantomData,
            n_features_in_: None,
            lambdas_: None,
            means_: None,
            stds_: None,
        }
    }

    /// Set the transformation method
    pub fn method(mut self, method: PowerMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set whether to standardize after transformation
    pub fn standardize(mut self, standardize: bool) -> Self {
        self.config.standardize = standardize;
        self
    }

    /// Set the epsilon for numerical stability
    pub fn eps(mut self, eps: Float) -> Self {
        self.config.eps = eps;
        self
    }

    /// Find optimal lambda using maximum likelihood estimation (simplified)
    fn find_optimal_lambda(&self, x: &Array1<Float>) -> Float {
        let mut best_lambda = 0.0;
        let mut best_log_likelihood = Float::NEG_INFINITY;

        // Search over a range of lambda values
        let lambda_range = Array1::linspace(-2.0, 2.0, 41);

        for &lambda in lambda_range.iter() {
            let log_likelihood = self.compute_log_likelihood(x, lambda);
            if log_likelihood > best_log_likelihood {
                best_log_likelihood = log_likelihood;
                best_lambda = lambda;
            }
        }

        best_lambda
    }

    /// Compute log-likelihood for a given lambda
    fn compute_log_likelihood(&self, x: &Array1<Float>, lambda: Float) -> Float {
        let transformed = self.apply_power_transform(x, lambda);

        // Simple log-likelihood based on normality assumption
        let mean = transformed.mean().unwrap_or(0.0);
        let variance = transformed.var(0.0);

        if variance <= 0.0 {
            return Float::NEG_INFINITY;
        }

        let n = x.len() as Float;
        let log_likelihood = -0.5 * n * (2.0 * std::f64::consts::PI * variance).ln()
            - 0.5
                * transformed
                    .iter()
                    .map(|&val| (val - mean).powi(2))
                    .sum::<Float>()
                / variance;

        // Add Jacobian term
        let jacobian = match self.config.method {
            PowerMethod::BoxCox => {
                (lambda - 1.0)
                    * x.iter()
                        .map(|&val| val.max(self.config.eps).ln())
                        .sum::<Float>()
            }
            PowerMethod::YeoJohnson => x
                .iter()
                .map(|&val| {
                    if val >= 0.0 {
                        (lambda - 1.0) * (val + 1.0).ln()
                    } else {
                        (1.0 - lambda) * (-val + 1.0).ln()
                    }
                })
                .sum::<Float>(),
        };

        log_likelihood + jacobian
    }

    /// Apply power transformation with given lambda
    fn apply_power_transform(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        match self.config.method {
            PowerMethod::BoxCox => self.box_cox_transform(x, lambda),
            PowerMethod::YeoJohnson => self.yeo_johnson_transform(x, lambda),
        }
    }

    /// Box-Cox transformation
    fn box_cox_transform(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        x.mapv(|val| {
            let val_pos = val.max(self.config.eps);
            if lambda.abs() < 1e-8 {
                val_pos.ln()
            } else {
                (val_pos.powf(lambda) - 1.0) / lambda
            }
        })
    }

    /// Yeo-Johnson transformation
    fn yeo_johnson_transform(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        x.mapv(|val| {
            if val >= 0.0 {
                if lambda.abs() < 1e-8 {
                    (val + 1.0).ln()
                } else {
                    ((val + 1.0).powf(lambda) - 1.0) / lambda
                }
            } else if (lambda - 2.0).abs() < 1e-8 {
                -(-val + 1.0).ln()
            } else {
                -((-val + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
            }
        })
    }
}

impl PowerTransformer<Trained> {
    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
            .expect("PowerTransformer should be fitted")
    }

    /// Get the optimal lambda parameters
    pub fn lambdas(&self) -> &Array1<Float> {
        self.lambdas_
            .as_ref()
            .expect("PowerTransformer should be fitted")
    }

    /// Get the means used for standardization
    pub fn means(&self) -> Option<&Array1<Float>> {
        self.means_.as_ref()
    }

    /// Get the standard deviations used for standardization
    pub fn stds(&self) -> Option<&Array1<Float>> {
        self.stds_.as_ref()
    }

    /// Apply Box-Cox transformation
    fn box_cox_transform_fitted(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        // FIXME: SIMD implementation disabled for compilation
        // Use CPU fallback for now
        self.box_cox_transform_cpu(x, lambda)
    }

    /// Apply Yeo-Johnson transformation
    fn yeo_johnson_transform_fitted(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        // FIXME: SIMD implementation disabled for compilation
        // Use CPU fallback for now
        self.yeo_johnson_transform_cpu(x, lambda)
    }

    /// CPU implementation of Box-Cox transformation
    fn box_cox_transform_cpu(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        x.mapv(|val| {
            let val_pos = val.max(self.config.eps);
            if lambda.abs() < 1e-8 {
                val_pos.ln()
            } else {
                (val_pos.powf(lambda) - 1.0) / lambda
            }
        })
    }

    /// CPU implementation of Yeo-Johnson transformation
    fn yeo_johnson_transform_cpu(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        x.mapv(|val| {
            if val >= 0.0 {
                if lambda.abs() < 1e-8 {
                    (val + 1.0).ln()
                } else {
                    ((val + 1.0).powf(lambda) - 1.0) / lambda
                }
            } else if (lambda - 2.0).abs() < 1e-8 {
                -(-val + 1.0).ln()
            } else {
                -((-val + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
            }
        })
    }

    /// Inverse Box-Cox transformation
    pub fn inverse_box_cox(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        x.mapv(|val| {
            if lambda.abs() < 1e-8 {
                val.exp()
            } else {
                (lambda * val + 1.0).powf(1.0 / lambda).max(self.config.eps)
            }
        })
    }

    /// Inverse Yeo-Johnson transformation
    pub fn inverse_yeo_johnson(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        x.mapv(|val| {
            if val >= 0.0 {
                if lambda.abs() < 1e-8 {
                    val.exp() - 1.0
                } else {
                    (lambda * val + 1.0).powf(1.0 / lambda) - 1.0
                }
            } else if (lambda - 2.0).abs() < 1e-8 {
                -(-val).exp() + 1.0
            } else {
                -((2.0 - lambda) * (-val) + 1.0).powf(1.0 / (2.0 - lambda)) + 1.0
            }
        })
    }

    /// Apply inverse transformation to the entire dataset
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let lambdas = self.lambdas();
        let mut result = x.clone();

        // Apply inverse standardization if it was applied
        if self.config.standardize {
            if let (Some(means), Some(stds)) = (&self.means_, &self.stds_) {
                for j in 0..n_features {
                    let mut column = result.column_mut(j);
                    column.mapv_inplace(|val| val * stds[j] + means[j]);
                }
            }
        }

        // Apply inverse power transformation to each feature
        for j in 0..n_features {
            let feature_column = result.column(j).to_owned();
            let inverse_transformed_column = match self.config.method {
                PowerMethod::BoxCox => self.inverse_box_cox(&feature_column, lambdas[j]),
                PowerMethod::YeoJohnson => self.inverse_yeo_johnson(&feature_column, lambdas[j]),
            };
            result.column_mut(j).assign(&inverse_transformed_column);
        }

        Ok(result)
    }
}

impl Default for PowerTransformer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, ()> for PowerTransformer<Untrained> {
    type Fitted = PowerTransformer<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit PowerTransformer on empty dataset".to_string(),
            ));
        }

        // Check for positive values if using Box-Cox
        if matches!(self.config.method, PowerMethod::BoxCox) {
            for &val in x.iter() {
                if val <= 0.0 {
                    return Err(SklearsError::InvalidInput(
                        "Box-Cox transformation requires strictly positive data".to_string(),
                    ));
                }
            }
        }

        // Find optimal lambda for each feature
        let mut lambdas = Array1::<Float>::zeros(n_features);
        for j in 0..n_features {
            let feature_column = x.column(j).to_owned();
            lambdas[j] = self.find_optimal_lambda(&feature_column);
        }

        // Compute means and stds for standardization if requested
        let (means, stds) = if self.config.standardize {
            // Transform data first, then compute statistics
            let mut transformed_data = Array2::<Float>::zeros(x.dim());
            for j in 0..n_features {
                let feature_column = x.column(j).to_owned();
                let transformed_column = self.apply_power_transform(&feature_column, lambdas[j]);
                transformed_data.column_mut(j).assign(&transformed_column);
            }

            let means = transformed_data.mean_axis(Axis(0)).unwrap();
            let stds = transformed_data.std_axis(Axis(0), 0.0);
            (Some(means), Some(stds))
        } else {
            (None, None)
        };

        Ok(PowerTransformer {
            config: self.config,
            state: PhantomData,
            n_features_in_: Some(n_features),
            lambdas_: Some(lambdas),
            means_: means,
            stds_: stds,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PowerTransformer<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let lambdas = self.lambdas();
        let mut result = Array2::<Float>::zeros(x.dim());

        // Apply power transformation to each feature
        for j in 0..n_features {
            let feature_column = x.column(j).to_owned();
            let transformed_column = match self.config.method {
                PowerMethod::BoxCox => self.box_cox_transform_fitted(&feature_column, lambdas[j]),
                PowerMethod::YeoJohnson => {
                    self.yeo_johnson_transform_fitted(&feature_column, lambdas[j])
                }
            };
            result.column_mut(j).assign(&transformed_column);
        }

        // Apply standardization if requested
        if self.config.standardize {
            if let (Some(means), Some(stds)) = (&self.means_, &self.stds_) {
                for j in 0..n_features {
                    let mut column = result.column_mut(j);
                    let mean = means[j];
                    let std = stds[j];
                    if std > 0.0 {
                        column.mapv_inplace(|val| (val - mean) / std);
                    }
                }
            }
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_power_transformer_yeo_johnson() -> Result<()> {
        let x = array![[-1.0], [0.0], [1.0], [2.0]];
        let transformer = PowerTransformer::new()
            .method(PowerMethod::YeoJohnson)
            .standardize(false);

        let fitted = transformer.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        // Should transform without errors
        assert_eq!(transformed.nrows(), 4);
        assert_eq!(transformed.ncols(), 1);

        Ok(())
    }

    #[test]
    fn test_power_transformer_box_cox() -> Result<()> {
        let x = array![[0.1], [1.0], [2.0], [5.0]]; // Positive values only
        let transformer = PowerTransformer::new()
            .method(PowerMethod::BoxCox)
            .standardize(false);

        let fitted = transformer.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        // Should transform without errors
        assert_eq!(transformed.nrows(), 4);
        assert_eq!(transformed.ncols(), 1);

        Ok(())
    }

    #[test]
    fn test_box_cox_negative_data_error() {
        let x = array![[-1.0], [1.0], [2.0]]; // Contains negative value
        let transformer = PowerTransformer::new().method(PowerMethod::BoxCox);

        let result = transformer.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_power_transformer_with_standardization() -> Result<()> {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let transformer = PowerTransformer::new()
            .method(PowerMethod::YeoJohnson)
            .standardize(true);

        let fitted = transformer.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        // Check that standardization was applied (mean ≈ 0, std ≈ 1)
        let mean = transformed.mean_axis(Axis(0)).unwrap();
        let std = transformed.std_axis(Axis(0), 0.0);

        assert_abs_diff_eq!(mean[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(std[0], 1.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_inverse_transform() -> Result<()> {
        let x = array![[0.5], [1.0], [1.5], [2.0]];
        let transformer = PowerTransformer::new()
            .method(PowerMethod::YeoJohnson)
            .standardize(false);

        let fitted = transformer.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;
        let inverse_transformed = fitted.inverse_transform(&transformed)?;

        // Should recover original data (approximately)
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_abs_diff_eq!(x[[i, j]], inverse_transformed[[i, j]], epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_optimal_lambda_finding() -> Result<()> {
        let x = array![[1.0], [2.0], [4.0], [8.0]]; // Exponential-like data
        let transformer = PowerTransformer::new().method(PowerMethod::BoxCox);

        let fitted = transformer.fit(&x, &())?;
        let lambdas = fitted.lambdas();

        // Lambda should be found (exact value depends on optimization)
        assert!(lambdas.len() == 1);
        assert!(lambdas[0].is_finite());

        Ok(())
    }

    #[test]
    fn test_multiple_features() -> Result<()> {
        let x = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]];
        let transformer = PowerTransformer::new().method(PowerMethod::YeoJohnson);

        let fitted = transformer.fit(&x, &())?;
        let transformed = fitted.transform(&x)?;

        assert_eq!(transformed.nrows(), 4);
        assert_eq!(transformed.ncols(), 2);
        assert_eq!(fitted.lambdas().len(), 2);

        Ok(())
    }
}
