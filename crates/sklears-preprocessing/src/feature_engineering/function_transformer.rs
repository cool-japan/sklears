//! Function transformer for applying arbitrary functions to data

use scirs2_core::ndarray::{s, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for FunctionTransformer
#[derive(Clone)]
pub struct FunctionTransformerConfig<F, G>
where
    F: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
    G: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
{
    /// The function to apply forward transformation
    pub func: F,
    /// The function to apply inverse transformation (optional)
    pub inverse_func: Option<G>,
    /// Whether to check that func(x) and inverse_func(func(x)) are equal
    pub check_inverse: bool,
    /// Whether to validate input
    pub validate: bool,
}

/// FunctionTransformer applies arbitrary functions to transform data
///
/// This transformer is useful for stateless transformations such as taking
/// the log, square root, or any other function. Unlike most other transformers,
/// FunctionTransformer does not require fitting.
pub struct FunctionTransformer<F, G, State = Untrained>
where
    F: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
    G: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
{
    config: FunctionTransformerConfig<F, G>,
    state: PhantomData<State>,
    // Fitted parameters
    n_features_in_: Option<usize>,
}

impl<F, G> FunctionTransformer<F, G, Untrained>
where
    F: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
    G: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
{
    /// Create a new FunctionTransformer with the given function
    pub fn new(func: F) -> Self {
        Self {
            config: FunctionTransformerConfig {
                func,
                inverse_func: None,
                check_inverse: false,
                validate: true,
            },
            state: PhantomData,
            n_features_in_: None,
        }
    }

    /// Set the inverse function
    pub fn inverse_func(mut self, inverse_func: G) -> Self {
        self.config.inverse_func = Some(inverse_func);
        self
    }

    /// Set whether to check inverse
    pub fn check_inverse(mut self, check_inverse: bool) -> Self {
        self.config.check_inverse = check_inverse;
        self
    }

    /// Set whether to validate input
    pub fn validate(mut self, validate: bool) -> Self {
        self.config.validate = validate;
        self
    }
}

impl<F, G> FunctionTransformer<F, G, Trained>
where
    F: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
    G: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
{
    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
            .expect("FunctionTransformer should be fitted")
    }

    /// Apply the inverse transformation
    pub fn inverse_transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        match &self.config.inverse_func {
            Some(inverse_func) => inverse_func(x),
            None => Err(SklearsError::InvalidInput(
                "No inverse function provided".to_string(),
            )),
        }
    }
}

impl<F, G> Fit<Array2<Float>, ()> for FunctionTransformer<F, G, Untrained>
where
    F: Fn(&Array2<Float>) -> Result<Array2<Float>> + Clone + Send + Sync,
    G: Fn(&Array2<Float>) -> Result<Array2<Float>> + Clone + Send + Sync,
{
    type Fitted = FunctionTransformer<F, G, Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if self.config.validate && n_samples == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Cannot fit FunctionTransformer on empty dataset".to_string(),
            });
        }

        // Check that forward and inverse functions work correctly if requested
        if self.config.check_inverse {
            if let Some(ref inverse_func) = self.config.inverse_func {
                // Try transforming and then inverse transforming a small sample
                let sample_size = n_samples.min(10);
                let x_sample = x.slice(s![..sample_size, ..]).to_owned();

                let x_transformed = (self.config.func)(&x_sample)?;
                let x_restored = inverse_func(&x_transformed)?;

                // Check dimensions match
                if x_sample.dim() != x_restored.dim() {
                    return Err(SklearsError::InvalidParameter {
                        name: "func_inverse".to_string(),
                        reason: "func and inverse_func do not produce consistent dimensions"
                            .to_string(),
                    });
                }

                // Check values are approximately equal
                let max_diff = (x_sample - x_restored)
                    .mapv(Float::abs)
                    .fold(0.0_f64, |a, &b| a.max(b));
                if max_diff > 1e-6 {
                    return Err(SklearsError::InvalidParameter {
                        name: "func_inverse".to_string(),
                        reason: format!(
                            "func and inverse_func are not inverses. Max difference: {max_diff}"
                        ),
                    });
                }
            }
        }

        Ok(FunctionTransformer {
            config: self.config,
            state: PhantomData,
            n_features_in_: Some(n_features),
        })
    }
}

impl<F, G> Transform<Array2<Float>, Array2<Float>> for FunctionTransformer<F, G, Trained>
where
    F: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
    G: Fn(&Array2<Float>) -> Result<Array2<Float>> + Send + Sync,
{
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_, n_features) = x.dim();

        if self.config.validate && n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        (self.config.func)(x)
    }
}

/// Common transformation functions that can be used with FunctionTransformer
pub mod transforms {
    use super::*;

    /// Natural logarithm transformation
    pub fn log(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| val.ln()))
    }

    /// Exponential transformation
    pub fn exp(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| val.exp()))
    }

    /// Square root transformation
    pub fn sqrt(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| val.sqrt()))
    }

    /// Square transformation
    pub fn square(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| val.powi(2)))
    }

    /// Reciprocal transformation
    pub fn reciprocal(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| 1.0 / val))
    }

    /// Log1p transformation (log(1 + x))
    pub fn log1p(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| (1.0 + val).ln()))
    }

    /// Expm1 transformation (exp(x) - 1)
    pub fn expm1(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| val.exp() - 1.0))
    }

    /// Logit transformation
    pub fn logit(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| {
            if val <= 0.0 || val >= 1.0 {
                Float::NAN
            } else {
                (val / (1.0 - val)).ln()
            }
        }))
    }

    /// Sigmoid transformation (inverse of logit)
    pub fn sigmoid(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| 1.0 / (1.0 + (-val).exp())))
    }

    /// Absolute value transformation
    pub fn abs(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| val.abs()))
    }

    /// Sign transformation
    pub fn sign(x: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(x.mapv(|val| {
            if val > 0.0 {
                1.0
            } else if val < 0.0 {
                -1.0
            } else {
                0.0
            }
        }))
    }

    /// Clip values to a range
    pub fn clip(min: Float, max: Float) -> impl Fn(&Array2<Float>) -> Result<Array2<Float>> {
        move |x: &Array2<Float>| Ok(x.mapv(|val| val.clamp(min, max)))
    }

    /// Add a constant
    pub fn add_constant(constant: Float) -> impl Fn(&Array2<Float>) -> Result<Array2<Float>> {
        move |x: &Array2<Float>| Ok(x.mapv(|val| val + constant))
    }

    /// Multiply by a constant
    pub fn multiply_constant(constant: Float) -> impl Fn(&Array2<Float>) -> Result<Array2<Float>> {
        move |x: &Array2<Float>| Ok(x.mapv(|val| val * constant))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_function_transformer_log() {
        let transformer: FunctionTransformer<_, _> =
            FunctionTransformer::new(transforms::log).inverse_func(transforms::exp);

        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let fitted = transformer.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Check that log transformation was applied
        let expected = array![[1.0_f64.ln(), 2.0_f64.ln()], [3.0_f64.ln(), 4.0_f64.ln()]];

        for (actual, expected) in transformed.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_function_transformer_square() {
        let transformer: FunctionTransformer<_, fn(&Array2<Float>) -> Result<Array2<Float>>> =
            FunctionTransformer::new(transforms::square);

        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let fitted = transformer.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Check that square transformation was applied
        let expected = array![[1.0, 4.0], [9.0, 16.0]];

        for (actual, expected) in transformed.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_function_transformer_inverse() {
        let transformer: FunctionTransformer<_, _> = FunctionTransformer::new(transforms::log)
            .inverse_func(transforms::exp)
            .check_inverse(true);

        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let fitted = transformer.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();
        let restored = fitted.inverse_transform(&transformed).unwrap();

        // Check that inverse transformation restores original
        for (original, restored) in x.iter().zip(restored.iter()) {
            assert!((original - restored).abs() < 1e-6);
        }
    }

    #[test]
    fn test_custom_function() {
        let custom_fn =
            |x: &Array2<Float>| -> Result<Array2<Float>> { Ok(x.mapv(|val| val * 2.0 + 1.0)) };

        let transformer: FunctionTransformer<_, fn(&Array2<Float>) -> Result<Array2<Float>>> =
            FunctionTransformer::new(custom_fn);

        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let fitted = transformer.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Check that custom transformation was applied: 2x + 1
        let expected = array![[3.0, 5.0], [7.0, 9.0]];

        for (actual, expected) in transformed.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}
