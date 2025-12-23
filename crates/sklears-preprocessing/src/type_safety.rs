//! Advanced type safety for preprocessing transformers
//!
//! This module provides compile-time guarantees for transformation states and pipeline composition
//! using Rust's advanced type system features including:
//! - Phantom types for tracking fitted/unfitted states
//! - Const generics for compile-time dimension checking
//! - Type-level programming for pipeline validation
//! - Zero-cost abstractions for transformation composition

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::prelude::*;
use std::marker::PhantomData;

// ================================================================================================
// State Markers
// ================================================================================================

/// Marker trait for transformation states
pub trait TransformState: sealed::Sealed {}

/// Unfitted state - transformer has not been fitted to data
#[derive(Debug, Clone, Copy)]
pub struct Unfitted;

/// Fitted state - transformer has been fitted to data and can transform
#[derive(Debug, Clone, Copy)]
pub struct Fitted;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Unfitted {}
    impl Sealed for super::Fitted {}

    pub trait DimensionSealed {}
    impl DimensionSealed for super::Dynamic {}
    impl<const N: usize> DimensionSealed for super::Known<N> {}
}

impl TransformState for Unfitted {}
impl TransformState for Fitted {}

// ================================================================================================
// Dimension Markers
// ================================================================================================

/// Marker for unknown dimensions (determined at runtime)
pub struct Dynamic;

/// Marker for known dimensions (determined at compile time)
pub struct Known<const N: usize>;

/// Trait for dimension types
pub trait Dimension: sealed::DimensionSealed {
    /// Get the dimension value if known at compile time
    fn value() -> Option<usize>;
}

impl Dimension for Dynamic {
    fn value() -> Option<usize> {
        None
    }
}

impl<const N: usize> Dimension for Known<N> {
    fn value() -> Option<usize> {
        Some(N)
    }
}

// ================================================================================================
// Type-Safe Transformer
// ================================================================================================

/// Type-safe transformer with compile-time state and dimension tracking
///
/// # Type Parameters
/// * `S` - State marker (Unfitted or Fitted)
/// * `InDim` - Input dimension marker (Dynamic or Known<N>)
/// * `OutDim` - Output dimension marker (Dynamic or Known<N>)
#[derive(Debug, Clone)]
pub struct TypeSafeTransformer<S: TransformState, InDim: Dimension, OutDim: Dimension> {
    /// Configuration
    config: TypeSafeConfig,
    /// Input dimension (runtime value)
    input_dim: Option<usize>,
    /// Output dimension (runtime value)
    output_dim: Option<usize>,
    /// Fitted parameters (only available in Fitted state)
    parameters: Option<TransformParameters>,
    /// State marker (zero-sized, compile-time only)
    _state: PhantomData<S>,
    /// Input dimension marker (zero-sized, compile-time only)
    _in_dim: PhantomData<InDim>,
    /// Output dimension marker (zero-sized, compile-time only)
    _out_dim: PhantomData<OutDim>,
}

/// Configuration for type-safe transformer
#[derive(Debug, Clone)]
pub struct TypeSafeConfig {
    /// Whether to validate dimensions at runtime
    pub validate_dimensions: bool,
    /// Whether to normalize outputs
    pub normalize: bool,
}

impl Default for TypeSafeConfig {
    fn default() -> Self {
        Self {
            validate_dimensions: true,
            normalize: false,
        }
    }
}

/// Fitted parameters for the transformer
#[derive(Debug, Clone)]
struct TransformParameters {
    /// Mean for normalization
    mean: Array1<f64>,
    /// Standard deviation for normalization
    std: Array1<f64>,
}

// ================================================================================================
// Implementation for Unfitted State
// ================================================================================================

impl<InDim: Dimension, OutDim: Dimension> TypeSafeTransformer<Unfitted, InDim, OutDim> {
    /// Create a new unfitted transformer with dynamic dimensions
    pub fn new(config: TypeSafeConfig) -> TypeSafeTransformer<Unfitted, Dynamic, Dynamic> {
        TypeSafeTransformer {
            config,
            input_dim: None,
            output_dim: None,
            parameters: None,
            _state: PhantomData,
            _in_dim: PhantomData,
            _out_dim: PhantomData,
        }
    }

    /// Create a new unfitted transformer with known input dimension
    pub fn with_input_dim<const N: usize>(
        config: TypeSafeConfig,
    ) -> TypeSafeTransformer<Unfitted, Known<N>, Dynamic> {
        TypeSafeTransformer {
            config,
            input_dim: Some(N),
            output_dim: None,
            parameters: None,
            _state: PhantomData,
            _in_dim: PhantomData,
            _out_dim: PhantomData,
        }
    }

    /// Create a new unfitted transformer with known input and output dimensions
    pub fn with_dimensions<const IN: usize, const OUT: usize>(
        config: TypeSafeConfig,
    ) -> TypeSafeTransformer<Unfitted, Known<IN>, Known<OUT>> {
        TypeSafeTransformer {
            config,
            input_dim: Some(IN),
            output_dim: Some(OUT),
            parameters: None,
            _state: PhantomData,
            _in_dim: PhantomData,
            _out_dim: PhantomData,
        }
    }
}

// Fit for dynamic dimensions
impl TypeSafeTransformer<Unfitted, Dynamic, Dynamic> {
    /// Fit the transformer to data
    pub fn fit(self, X: &Array2<f64>) -> Result<TypeSafeTransformer<Fitted, Dynamic, Dynamic>> {
        let input_dim = X.ncols();
        let output_dim = X.ncols(); // Identity transform for this example

        let parameters = if self.config.normalize {
            let mean = X
                .mean_axis(scirs2_core::ndarray::Axis(0))
                .ok_or_else(|| SklearsError::InvalidInput("Failed to compute mean".to_string()))?;
            let std = X.std_axis(scirs2_core::ndarray::Axis(0), 0.0);
            Some(TransformParameters { mean, std })
        } else {
            None
        };

        Ok(TypeSafeTransformer {
            config: self.config,
            input_dim: Some(input_dim),
            output_dim: Some(output_dim),
            parameters,
            _state: PhantomData,
            _in_dim: PhantomData,
            _out_dim: PhantomData,
        })
    }
}

// Fit for known input dimension
impl<const N: usize> TypeSafeTransformer<Unfitted, Known<N>, Dynamic> {
    /// Fit the transformer to data with compile-time input dimension check
    pub fn fit(self, X: &Array2<f64>) -> Result<TypeSafeTransformer<Fitted, Known<N>, Dynamic>> {
        if X.ncols() != N {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} input features, got {}",
                N,
                X.ncols()
            )));
        }

        let output_dim = X.ncols();

        let parameters = if self.config.normalize {
            let mean = X
                .mean_axis(scirs2_core::ndarray::Axis(0))
                .ok_or_else(|| SklearsError::InvalidInput("Failed to compute mean".to_string()))?;
            let std = X.std_axis(scirs2_core::ndarray::Axis(0), 0.0);
            Some(TransformParameters { mean, std })
        } else {
            None
        };

        Ok(TypeSafeTransformer {
            config: self.config,
            input_dim: Some(N),
            output_dim: Some(output_dim),
            parameters,
            _state: PhantomData,
            _in_dim: PhantomData,
            _out_dim: PhantomData,
        })
    }
}

// Fit for known input and output dimensions
impl<const IN: usize, const OUT: usize> TypeSafeTransformer<Unfitted, Known<IN>, Known<OUT>> {
    /// Fit the transformer to data with compile-time dimension checks
    pub fn fit(
        self,
        X: &Array2<f64>,
    ) -> Result<TypeSafeTransformer<Fitted, Known<IN>, Known<OUT>>> {
        if X.ncols() != IN {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} input features, got {}",
                IN,
                X.ncols()
            )));
        }

        let parameters = if self.config.normalize {
            let mean = X
                .mean_axis(scirs2_core::ndarray::Axis(0))
                .ok_or_else(|| SklearsError::InvalidInput("Failed to compute mean".to_string()))?;
            let std = X.std_axis(scirs2_core::ndarray::Axis(0), 0.0);
            Some(TransformParameters { mean, std })
        } else {
            None
        };

        Ok(TypeSafeTransformer {
            config: self.config,
            input_dim: Some(IN),
            output_dim: Some(OUT),
            parameters,
            _state: PhantomData,
            _in_dim: PhantomData,
            _out_dim: PhantomData,
        })
    }
}

// ================================================================================================
// Implementation for Fitted State
// ================================================================================================

// Transform for dynamic dimensions
impl TypeSafeTransformer<Fitted, Dynamic, Dynamic> {
    /// Transform data using the fitted transformer
    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if let Some(input_dim) = self.input_dim {
            if X.ncols() != input_dim {
                return Err(SklearsError::InvalidInput(format!(
                    "Expected {} input features, got {}",
                    input_dim,
                    X.ncols()
                )));
            }
        }

        let mut result = X.clone();

        if let Some(ref params) = self.parameters {
            for i in 0..result.nrows() {
                for j in 0..result.ncols() {
                    result[[i, j]] = (result[[i, j]] - params.mean[j]) / params.std[j].max(1e-10);
                }
            }
        }

        Ok(result)
    }
}

// Transform for known input dimension
impl<const N: usize> TypeSafeTransformer<Fitted, Known<N>, Dynamic> {
    /// Transform data with compile-time input dimension check
    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != N {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} input features, got {}",
                N,
                X.ncols()
            )));
        }

        let mut result = X.clone();

        if let Some(ref params) = self.parameters {
            for i in 0..result.nrows() {
                for j in 0..result.ncols() {
                    result[[i, j]] = (result[[i, j]] - params.mean[j]) / params.std[j].max(1e-10);
                }
            }
        }

        Ok(result)
    }
}

// Transform for known input and output dimensions
impl<const IN: usize, const OUT: usize> TypeSafeTransformer<Fitted, Known<IN>, Known<OUT>> {
    /// Transform data with compile-time dimension checks
    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        if X.ncols() != IN {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} input features, got {}",
                IN,
                X.ncols()
            )));
        }

        let mut result = X.clone();

        if let Some(ref params) = self.parameters {
            for i in 0..result.nrows() {
                for j in 0..result.ncols() {
                    result[[i, j]] = (result[[i, j]] - params.mean[j]) / params.std[j].max(1e-10);
                }
            }
        }

        // Compile-time check: output must have OUT columns
        if result.ncols() != OUT {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} output features, got {}",
                OUT,
                result.ncols()
            )));
        }

        Ok(result)
    }
}

// ================================================================================================
// Type-Safe Pipeline
// ================================================================================================

/// Type-safe pipeline that chains transformers with compile-time validation
pub struct TypeSafePipeline<S1, S2, D1, D2, D3>
where
    S1: TransformState,
    S2: TransformState,
    D1: Dimension,
    D2: Dimension,
    D3: Dimension,
{
    /// First transformer
    first: TypeSafeTransformer<S1, D1, D2>,
    /// Second transformer
    second: TypeSafeTransformer<S2, D2, D3>,
}

/// Type-safe pipeline in unfitted state
impl<D1: Dimension, D2: Dimension, D3: Dimension> TypeSafePipeline<Unfitted, Unfitted, D1, D2, D3> {
    /// Create a new pipeline by chaining two unfitted transformers
    pub fn new(
        first: TypeSafeTransformer<Unfitted, D1, D2>,
        second: TypeSafeTransformer<Unfitted, D2, D3>,
    ) -> Self {
        Self { first, second }
    }
}

/// Fit pipeline with dynamic dimensions
impl TypeSafePipeline<Unfitted, Unfitted, Dynamic, Dynamic, Dynamic> {
    /// Fit the entire pipeline to data
    pub fn fit(
        self,
        X: &Array2<f64>,
    ) -> Result<TypeSafePipeline<Fitted, Fitted, Dynamic, Dynamic, Dynamic>> {
        let first_fitted = self.first.fit(X)?;
        let X_transformed = first_fitted.transform(X)?;
        let second_fitted = self.second.fit(&X_transformed)?;

        Ok(TypeSafePipeline {
            first: first_fitted,
            second: second_fitted,
        })
    }
}

/// Fit pipeline with known dimensions
impl<const D1: usize, const D2: usize, const D3: usize>
    TypeSafePipeline<Unfitted, Unfitted, Known<D1>, Known<D2>, Known<D3>>
{
    /// Fit the entire pipeline to data with compile-time dimension validation
    pub fn fit(
        self,
        X: &Array2<f64>,
    ) -> Result<TypeSafePipeline<Fitted, Fitted, Known<D1>, Known<D2>, Known<D3>>> {
        let first_fitted = self.first.fit(X)?;
        let X_transformed = first_fitted.transform(X)?;
        let second_fitted = self.second.fit(&X_transformed)?;

        Ok(TypeSafePipeline {
            first: first_fitted,
            second: second_fitted,
        })
    }
}

/// Transform for fitted pipeline with dynamic dimensions
impl TypeSafePipeline<Fitted, Fitted, Dynamic, Dynamic, Dynamic> {
    /// Transform data through the entire pipeline
    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let X_intermediate = self.first.transform(X)?;
        self.second.transform(&X_intermediate)
    }
}

/// Transform for fitted pipeline with known dimensions
impl<const D1: usize, const D2: usize, const D3: usize>
    TypeSafePipeline<Fitted, Fitted, Known<D1>, Known<D2>, Known<D3>>
{
    /// Transform data through the entire pipeline with compile-time dimension validation
    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let X_intermediate = self.first.transform(X)?;
        self.second.transform(&X_intermediate)
    }
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_dynamic_dimensions() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let config = TypeSafeConfig::default();
        let transformer: TypeSafeTransformer<Unfitted, Dynamic, Dynamic> =
            TypeSafeTransformer::<Unfitted, Dynamic, Dynamic>::new(config);
        let fitted = transformer.fit(&X).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_known_input_dimension() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let config = TypeSafeConfig::default();
        let transformer: TypeSafeTransformer<Unfitted, Known<2>, Dynamic> =
            TypeSafeTransformer::<Unfitted, Known<2>, Dynamic>::with_input_dim(config);
        let fitted = transformer.fit(&X).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_known_input_dimension_mismatch() {
        let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let config = TypeSafeConfig::default();
        let transformer: TypeSafeTransformer<Unfitted, Known<2>, Dynamic> =
            TypeSafeTransformer::<Unfitted, Known<2>, Dynamic>::with_input_dim(config);

        // This should fail because X has 3 columns, but we expect 2
        assert!(transformer.fit(&X).is_err());
    }

    #[test]
    fn test_known_dimensions() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let config = TypeSafeConfig::default();
        let transformer: TypeSafeTransformer<Unfitted, Known<2>, Known<2>> =
            TypeSafeTransformer::<Unfitted, Known<2>, Known<2>>::with_dimensions(config);
        let fitted = transformer.fit(&X).unwrap();
        let result = fitted.transform(&X).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_normalization() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let config = TypeSafeConfig {
            validate_dimensions: true,
            normalize: true,
        };
        let transformer: TypeSafeTransformer<Unfitted, Dynamic, Dynamic> =
            TypeSafeTransformer::<Unfitted, Dynamic, Dynamic>::new(config);
        let fitted = transformer.fit(&X).unwrap();
        let result = fitted.transform(&X).unwrap();

        // Verify normalization: mean should be approximately 0
        let mean = result.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        for &val in mean.iter() {
            assert!((val.abs()) < 1e-10);
        }
    }

    #[test]
    fn test_pipeline_dynamic() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let config1 = TypeSafeConfig {
            validate_dimensions: true,
            normalize: true,
        };
        let config2 = TypeSafeConfig::default();

        let transformer1: TypeSafeTransformer<Unfitted, Dynamic, Dynamic> =
            TypeSafeTransformer::<Unfitted, Dynamic, Dynamic>::new(config1);
        let transformer2: TypeSafeTransformer<Unfitted, Dynamic, Dynamic> =
            TypeSafeTransformer::<Unfitted, Dynamic, Dynamic>::new(config2);

        let pipeline = TypeSafePipeline::new(transformer1, transformer2);
        let fitted_pipeline = pipeline.fit(&X).unwrap();
        let result = fitted_pipeline.transform(&X).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_pipeline_known_dimensions() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let config1 = TypeSafeConfig::default();
        let config2 = TypeSafeConfig::default();

        let transformer1: TypeSafeTransformer<Unfitted, Known<2>, Known<2>> =
            TypeSafeTransformer::<Unfitted, Known<2>, Known<2>>::with_dimensions(config1);
        let transformer2: TypeSafeTransformer<Unfitted, Known<2>, Known<2>> =
            TypeSafeTransformer::<Unfitted, Known<2>, Known<2>>::with_dimensions(config2);

        let pipeline = TypeSafePipeline::new(transformer1, transformer2);
        let fitted_pipeline = pipeline.fit(&X).unwrap();
        let result = fitted_pipeline.transform(&X).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_state_transitions() {
        let X = array![[1.0, 2.0], [3.0, 4.0]];

        // Start in Unfitted state
        let unfitted: TypeSafeTransformer<Unfitted, Dynamic, Dynamic> =
            TypeSafeTransformer::<Unfitted, Dynamic, Dynamic>::new(TypeSafeConfig::default());

        // Fit transitions to Fitted state
        let fitted = unfitted.fit(&X).unwrap();

        // Can transform in Fitted state
        let _result = fitted.transform(&X).unwrap();

        // Cannot call fit() on fitted transformer (compile error if uncommented)
        // let _refitted = fitted.fit(&X); // This would not compile
    }

    #[test]
    fn test_dimension_markers() {
        assert_eq!(Dynamic::value(), None);
        assert_eq!(Known::<5>::value(), Some(5));
        assert_eq!(Known::<10>::value(), Some(10));
    }
}
