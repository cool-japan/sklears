//! Trait-based SIMD framework for modular and composable operations
//!
//! This module provides a comprehensive trait system for SIMD operations,
//! enabling modular design, runtime dispatch, and easy extensibility.

#[cfg(not(feature = "no-std"))]
use std::any;
#[cfg(not(feature = "no-std"))]
use std::boxed::Box;
#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;
#[cfg(not(feature = "no-std"))]
use std::fmt::Debug;
#[cfg(not(feature = "no-std"))]
use std::string::{String, ToString};
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

#[cfg(feature = "no-std")]
use alloc::boxed::Box;
#[cfg(feature = "no-std")]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "no-std")]
use alloc::string::{String, ToString};
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(feature = "no-std")]
use alloc::{format, vec};
#[cfg(feature = "no-std")]
use core::any;
#[cfg(feature = "no-std")]
use core::fmt::Debug;

/// Core trait for all SIMD operations
pub trait SimdOperation<T> {
    /// The output type of the operation
    type Output;

    /// The error type for operation failures
    type Error;

    /// Execute the SIMD operation
    fn execute(&self, input: &[T]) -> Result<Self::Output, Self::Error>;

    /// Get the optimal SIMD width for this operation on the current platform
    fn optimal_width(&self) -> usize;

    /// Check if the operation can be performed with SIMD on the current platform
    fn is_supported(&self) -> bool;

    /// Get a human-readable name for this operation
    fn name(&self) -> &'static str;
}

/// Trait for vectorized arithmetic operations
pub trait VectorArithmetic<T> {
    /// Add two vectors element-wise
    fn add(&self, a: &[T], b: &[T]) -> Result<Vec<T>, SimdError>;

    /// Subtract two vectors element-wise
    fn sub(&self, a: &[T], b: &[T]) -> Result<Vec<T>, SimdError>;

    /// Multiply two vectors element-wise
    fn mul(&self, a: &[T], b: &[T]) -> Result<Vec<T>, SimdError>;

    /// Divide two vectors element-wise
    fn div(&self, a: &[T], b: &[T]) -> Result<Vec<T>, SimdError>;

    /// Compute fused multiply-add: a * b + c
    fn fma(&self, a: &[T], b: &[T], c: &[T]) -> Result<Vec<T>, SimdError>;

    /// Scale a vector by a scalar
    fn scale(&self, vector: &[T], scalar: T) -> Result<Vec<T>, SimdError>;
}

/// Trait for vector reduction operations
pub trait VectorReduction<T> {
    /// Sum all elements in the vector
    fn sum(&self, vector: &[T]) -> Result<T, SimdError>;

    /// Find the minimum element
    fn min(&self, vector: &[T]) -> Result<T, SimdError>;

    /// Find the maximum element
    fn max(&self, vector: &[T]) -> Result<T, SimdError>;

    /// Compute the dot product of two vectors
    fn dot_product(&self, a: &[T], b: &[T]) -> Result<T, SimdError>;

    /// Compute the L2 norm of a vector
    fn norm(&self, vector: &[T]) -> Result<T, SimdError>;

    /// Compute the mean of all elements
    fn mean(&self, vector: &[T]) -> Result<T, SimdError>;
}

/// Trait for distance computations
pub trait DistanceMetric<T> {
    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, a: &[T], b: &[T]) -> Result<T, SimdError>;

    /// Compute Manhattan (L1) distance
    fn manhattan_distance(&self, a: &[T], b: &[T]) -> Result<T, SimdError>;

    /// Compute Cosine distance
    fn cosine_distance(&self, a: &[T], b: &[T]) -> Result<T, SimdError>;

    /// Compute squared Euclidean distance (avoiding square root)
    fn squared_euclidean_distance(&self, a: &[T], b: &[T]) -> Result<T, SimdError>;
}

/// Trait for activation functions used in neural networks
pub trait ActivationFunction<T: Copy> {
    /// Apply the activation function
    fn apply(&self, input: &[T]) -> Result<Vec<T>, SimdError>;

    /// Apply the derivative of the activation function
    fn derivative(&self, input: &[T]) -> Result<Vec<T>, SimdError>;

    /// Get the name of the activation function
    fn name(&self) -> &'static str;

    /// Check if this activation function supports in-place operations
    fn supports_inplace(&self) -> bool;

    /// Apply the activation function in-place (if supported)
    fn apply_inplace(&self, input: &mut [T]) -> Result<(), SimdError> {
        if !self.supports_inplace() {
            return Err(SimdError::UnsupportedOperation(
                "In-place operation not supported".to_string(),
            ));
        }
        let result = self.apply(input)?;
        input.copy_from_slice(&result);
        Ok(())
    }
}

/// Trait for kernel functions used in SVM and other algorithms
pub trait KernelFunction<T> {
    /// Compute the kernel function between two vectors
    fn compute(&self, a: &[T], b: &[T]) -> Result<T, SimdError>;

    /// Compute kernel matrix for a set of vectors
    fn kernel_matrix(&self, vectors: &[&[T]]) -> Result<Vec<Vec<T>>, SimdError>;

    /// Get the name of the kernel function
    fn name(&self) -> &'static str;

    /// Check if kernel supports hyperparameters
    fn has_parameters(&self) -> bool;
}

/// Trait for matrix operations
pub trait MatrixOperations<T> {
    /// Matrix-vector multiplication
    fn matrix_vector_multiply(&self, matrix: &[Vec<T>], vector: &[T]) -> Result<Vec<T>, SimdError>;

    /// Matrix-matrix multiplication
    fn matrix_multiply(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Result<Vec<Vec<T>>, SimdError>;

    /// Matrix transpose
    fn transpose(&self, matrix: &[Vec<T>]) -> Result<Vec<Vec<T>>, SimdError>;

    /// Element-wise matrix operations
    fn elementwise_add(&self, a: &[Vec<T>], b: &[Vec<T>]) -> Result<Vec<Vec<T>>, SimdError>;
}

/// Trait for clustering operations
pub trait ClusteringOperations<T> {
    /// Compute distances from points to centroids
    fn point_to_centroid_distances(
        &self,
        points: &[&[T]],
        centroids: &[&[T]],
    ) -> Result<Vec<Vec<T>>, SimdError>;

    /// Update centroids based on point assignments
    fn update_centroids(
        &self,
        points: &[&[T]],
        assignments: &[usize],
        k: usize,
    ) -> Result<Vec<Vec<T>>, SimdError>;

    /// Compute within-cluster sum of squares
    fn wcss(
        &self,
        points: &[&[T]],
        centroids: &[&[T]],
        assignments: &[usize],
    ) -> Result<T, SimdError>;
}

/// Common error types for SIMD operations
#[derive(Debug, Clone)]
pub enum SimdError {
    /// Input vectors have mismatched dimensions
    DimensionMismatch { expected: usize, actual: usize },

    /// Input data is empty
    EmptyInput,

    /// SIMD operation is not supported on this platform
    UnsupportedPlatform,

    /// Operation is not implemented for this type
    UnsupportedOperation(String),

    /// Numerical error (overflow, underflow, NaN)
    NumericalError(String),

    /// Invalid parameter value
    InvalidParameter { name: String, value: String },

    /// Memory allocation error
    AllocationError,

    /// External library integration error
    ExternalLibraryError(String),

    /// Invalid input data
    InvalidInput(String),

    /// Invalid argument provided
    InvalidArgument(String),

    /// Feature not implemented
    NotImplemented(String),

    /// Other generic errors
    Other(String),
}

impl core::fmt::Display for SimdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SimdError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            SimdError::EmptyInput => write!(f, "Input data is empty"),
            SimdError::UnsupportedPlatform => {
                write!(f, "SIMD operation not supported on this platform")
            }
            SimdError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            SimdError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            SimdError::InvalidParameter { name, value } => {
                write!(f, "Invalid parameter {}: {}", name, value)
            }
            SimdError::AllocationError => write!(f, "Memory allocation failed"),
            SimdError::ExternalLibraryError(msg) => write!(f, "External library error: {}", msg),
            SimdError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            SimdError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            SimdError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            SimdError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

#[cfg(not(feature = "no-std"))]
impl std::error::Error for SimdError {}

#[cfg(feature = "no-std")]
impl core::error::Error for SimdError {}

/// Dispatcher trait for runtime SIMD implementation selection
pub trait SimdDispatcher<T> {
    /// The operation type this dispatcher handles
    type Operation;

    /// Select the best implementation for the current platform
    fn select_implementation(
        &self,
    ) -> Box<dyn SimdOperation<T, Output = Self::Operation, Error = SimdError>>;

    /// Get all available implementations
    fn available_implementations(&self) -> Vec<&'static str>;

    /// Force a specific implementation (for testing/benchmarking)
    fn force_implementation(
        &self,
        name: &str,
    ) -> Option<Box<dyn SimdOperation<T, Output = Self::Operation, Error = SimdError>>>;
}

/// Configuration trait for SIMD operations
pub trait SimdConfig {
    /// Set the preferred SIMD width
    fn set_simd_width(&mut self, width: usize);

    /// Get the current SIMD width
    fn simd_width(&self) -> usize;

    /// Enable/disable automatic fallback to scalar
    fn set_scalar_fallback(&mut self, enabled: bool);

    /// Check if scalar fallback is enabled
    fn scalar_fallback_enabled(&self) -> bool;

    /// Set numerical precision requirements
    fn set_precision_tolerance(&mut self, tolerance: f64);

    /// Get current precision tolerance
    fn precision_tolerance(&self) -> f64;
}

/// Default configuration for SIMD operations
#[derive(Debug, Clone)]
pub struct DefaultSimdConfig {
    pub simd_width: usize,
    pub scalar_fallback: bool,
    pub precision_tolerance: f64,
}

impl Default for DefaultSimdConfig {
    fn default() -> Self {
        Self {
            simd_width: crate::SIMD_CAPS.best_f32_width(),
            scalar_fallback: true,
            precision_tolerance: 1e-6,
        }
    }
}

impl SimdConfig for DefaultSimdConfig {
    fn set_simd_width(&mut self, width: usize) {
        self.simd_width = width;
    }

    fn simd_width(&self) -> usize {
        self.simd_width
    }

    fn set_scalar_fallback(&mut self, enabled: bool) {
        self.scalar_fallback = enabled;
    }

    fn scalar_fallback_enabled(&self) -> bool {
        self.scalar_fallback
    }

    fn set_precision_tolerance(&mut self, tolerance: f64) {
        self.precision_tolerance = tolerance;
    }

    fn precision_tolerance(&self) -> f64 {
        self.precision_tolerance
    }
}

/// Trait for composable SIMD operations
pub trait ComposableOperation<T>: SimdOperation<T> {
    /// Compose this operation with another operation
    fn compose<Other>(self, other: Other) -> ComposedOperation<Self, Other>
    where
        Self: Sized,
        Other: SimdOperation<T>;

    /// Apply a transformation to the output
    fn map<F, U>(self, f: F) -> MappedOperation<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Output) -> U;
}

/// A composed operation that applies two operations in sequence
pub struct ComposedOperation<First, Second> {
    first: First,
    second: Second,
}

impl<First, Second> ComposedOperation<First, Second> {
    pub fn new(first: First, second: Second) -> Self {
        Self { first, second }
    }
}

/// An operation with a mapped output transformation
pub struct MappedOperation<Op, F> {
    operation: Op,
    mapper: F,
}

impl<Op, F> MappedOperation<Op, F> {
    pub fn new(operation: Op, mapper: F) -> Self {
        Self { operation, mapper }
    }
}

/// Trait for operations that can be parallelized
pub trait ParallelSimdOperation<T>: SimdOperation<T> {
    /// Execute the operation in parallel across multiple chunks
    fn execute_parallel(&self, input: &[T], chunk_size: usize)
        -> Result<Self::Output, Self::Error>;

    /// Get the optimal chunk size for parallel execution
    fn optimal_chunk_size(&self, input_size: usize) -> usize;

    /// Check if parallel execution is beneficial for the given input size
    fn should_parallelize(&self, input_size: usize) -> bool;
}

/// Registry for SIMD operation implementations
pub struct SimdRegistry {
    #[cfg(not(feature = "no-std"))]
    operations: HashMap<String, Box<dyn any::Any + Send + Sync>>,
    #[cfg(feature = "no-std")]
    operations: HashMap<String, Box<dyn any::Any + Send + Sync>>,
}

impl Default for SimdRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
        }
    }

    /// Register a new operation implementation
    pub fn register<T: 'static + Send + Sync>(&mut self, name: String, operation: T) {
        self.operations.insert(name, Box::new(operation));
    }

    /// Get a registered operation
    pub fn get<T: 'static>(&self, name: &str) -> Option<&T> {
        self.operations
            .get(name)
            .and_then(|op| op.downcast_ref::<T>())
    }

    /// List all registered operations
    pub fn list_operations(&self) -> Vec<&String> {
        self.operations.keys().collect()
    }
}

/// Macro for implementing basic SIMD operation traits
#[macro_export]
macro_rules! impl_simd_operation {
    ($type:ty, $output:ty, $name:literal) => {
        impl SimdOperation<f32> for $type {
            type Output = $output;
            type Error = SimdError;

            fn execute(&self, input: &[f32]) -> Result<Self::Output, Self::Error> {
                if input.is_empty() {
                    return Err(SimdError::EmptyInput);
                }
                self.compute(input)
            }

            fn optimal_width(&self) -> usize {
                $crate::SIMD_CAPS.best_f32_width()
            }

            fn is_supported(&self) -> bool {
                self.optimal_width() > 1
            }

            fn name(&self) -> &'static str {
                $name
            }
        }
    };
}

/// Utility functions for common trait implementations
pub mod utils {
    use super::*;

    /// Validate that two slices have the same length
    pub fn validate_same_length<T>(a: &[T], b: &[T]) -> Result<(), SimdError> {
        if a.len() != b.len() {
            Err(SimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            })
        } else {
            Ok(())
        }
    }

    /// Validate that a slice is not empty
    pub fn validate_not_empty<T>(slice: &[T]) -> Result<(), SimdError> {
        if slice.is_empty() {
            Err(SimdError::EmptyInput)
        } else {
            Ok(())
        }
    }

    /// Check if all values are finite (no NaN or infinity)
    pub fn validate_finite(slice: &[f32]) -> Result<(), SimdError> {
        for &value in slice {
            if !value.is_finite() {
                return Err(SimdError::NumericalError(format!(
                    "Non-finite value encountered: {}",
                    value
                )));
            }
        }
        Ok(())
    }

    /// Create a chunked iterator for parallel processing
    pub fn create_chunks<T>(slice: &[T], chunk_size: usize) -> impl Iterator<Item = &[T]> {
        slice.chunks(chunk_size)
    }

    /// Compute optimal chunk size based on input size and hardware
    pub fn optimal_chunk_size(input_size: usize, simd_width: usize) -> usize {
        let base_chunk = simd_width * 64; // Process 64 SIMD vectors per chunk
        let max_chunk = input_size / 4; // Use at most 4 chunks

        if max_chunk < base_chunk {
            max_chunk.max(simd_width)
        } else {
            base_chunk
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementation for testing
    struct MockVectorAdd;

    impl MockVectorAdd {
        fn compute(&self, input: &[f32]) -> Result<Vec<f32>, SimdError> {
            Ok(input.iter().map(|&x| x + 1.0).collect())
        }
    }

    impl_simd_operation!(MockVectorAdd, Vec<f32>, "mock_vector_add");

    #[test]
    fn test_simd_operation_trait() {
        let op = MockVectorAdd;
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = op.execute(&input).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);

        assert_eq!(op.name(), "mock_vector_add");
        assert!(op.optimal_width() >= 1);
    }

    #[test]
    fn test_simd_error_display() {
        let error = SimdError::DimensionMismatch {
            expected: 4,
            actual: 3,
        };
        assert!(error.to_string().contains("Dimension mismatch"));

        let error = SimdError::EmptyInput;
        assert!(error.to_string().contains("empty"));
    }

    #[test]
    fn test_default_simd_config() {
        let mut config = DefaultSimdConfig::default();

        assert!(config.simd_width() >= 1);
        assert!(config.scalar_fallback_enabled());
        assert_eq!(config.precision_tolerance(), 1e-6);

        config.set_simd_width(8);
        assert_eq!(config.simd_width(), 8);

        config.set_scalar_fallback(false);
        assert!(!config.scalar_fallback_enabled());

        config.set_precision_tolerance(1e-8);
        assert_eq!(config.precision_tolerance(), 1e-8);
    }

    #[test]
    fn test_simd_registry() {
        let mut registry = SimdRegistry::new();

        registry.register("test_op".to_string(), MockVectorAdd);

        let operations = registry.list_operations();
        assert_eq!(operations.len(), 1);
        assert_eq!(operations[0], "test_op");

        let op = registry.get::<MockVectorAdd>("test_op");
        assert!(op.is_some());

        let nonexistent = registry.get::<MockVectorAdd>("nonexistent");
        assert!(nonexistent.is_none());
    }

    #[test]
    fn test_validation_utils() {
        use utils::*;

        // Test same length validation
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = vec![7.0, 8.0];

        assert!(validate_same_length(&a, &b).is_ok());
        assert!(validate_same_length(&a, &c).is_err());

        // Test empty validation
        assert!(validate_not_empty(&a).is_ok());
        assert!(validate_not_empty(&Vec::<f32>::new()).is_err());

        // Test finite validation
        let finite = vec![1.0, 2.0, 3.0];
        let infinite = vec![1.0, f32::INFINITY, 3.0];
        let nan = vec![1.0, f32::NAN, 3.0];

        assert!(validate_finite(&finite).is_ok());
        assert!(validate_finite(&infinite).is_err());
        assert!(validate_finite(&nan).is_err());
    }

    #[test]
    fn test_chunk_utilities() {
        use utils::*;

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let chunks: Vec<&[i32]> = create_chunks(&data, 3).collect();

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], &[1, 2, 3]);
        assert_eq!(chunks[1], &[4, 5, 6]);
        assert_eq!(chunks[2], &[7, 8, 9]);
        assert_eq!(chunks[3], &[10]);

        let chunk_size = optimal_chunk_size(1000, 8);
        assert!(chunk_size >= 8);
        assert!(chunk_size <= 1000);
    }
}
