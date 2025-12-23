//! Type-safe distance metrics using phantom types
//!
//! This module provides compile-time guarantees about distance metric properties
//! through the use of phantom types and zero-cost abstractions.

use crate::distance::Distance;
use scirs2_core::ndarray::ArrayView1;
use sklears_core::types::Float;
use std::marker::PhantomData;

/// Marker trait for metric distance functions
///
/// A metric distance satisfies:
/// - Non-negativity: d(x, y) >= 0
/// - Identity: d(x, y) = 0 iff x = y
/// - Symmetry: d(x, y) = d(y, x)
/// - Triangle inequality: d(x, z) <= d(x, y) + d(y, z)
pub trait MetricDistance {}

/// Marker trait for non-metric distance functions
///
/// These distances don't satisfy all metric properties
/// (e.g., KL divergence is not symmetric)
pub trait NonMetricDistance {}

/// Marker trait for normalized distances (range [0, 1])
pub trait NormalizedDistance {}

/// Zero-sized type for Euclidean distance
pub struct EuclideanMetric;
impl MetricDistance for EuclideanMetric {}

/// Zero-sized type for Manhattan distance
pub struct ManhattanMetric;
impl MetricDistance for ManhattanMetric {}

/// Zero-sized type for Minkowski distance
pub struct MinkowskiMetric<const P: usize>;
impl<const P: usize> MetricDistance for MinkowskiMetric<P> {}

/// Zero-sized type for Cosine distance
pub struct CosineMetric;
impl NormalizedDistance for CosineMetric {}
// Note: Cosine similarity is not a proper metric (violates triangle inequality)

/// Zero-sized type for Chebyshev distance
pub struct ChebyshevMetric;
impl MetricDistance for ChebyshevMetric {}

/// Type-safe distance computer with compile-time guarantees
///
/// This struct uses phantom types to provide compile-time guarantees
/// about the properties of the distance metric being used.
///
/// # Example
/// ```ignore
/// use sklears_neighbors::type_safe_distance::{TypeSafeDistance, EuclideanMetric};
///
/// let distance_fn = TypeSafeDistance::<EuclideanMetric>::new();
/// // Compiler knows this is a metric distance
/// ```
pub struct TypeSafeDistance<M> {
    _marker: PhantomData<M>,
}

impl<M> TypeSafeDistance<M> {
    /// Create a new type-safe distance computer
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<M> Default for TypeSafeDistance<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M> Clone for TypeSafeDistance<M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M> Copy for TypeSafeDistance<M> {}

/// Trait for computing distances with specific metrics
pub trait ComputeDistance<M> {
    /// Compute distance between two points
    fn compute(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float;

    /// Convert to the underlying Distance enum
    fn to_distance(&self) -> Distance;
}

impl ComputeDistance<EuclideanMetric> for TypeSafeDistance<EuclideanMetric> {
    fn compute(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        Distance::Euclidean.calculate(x, y)
    }

    fn to_distance(&self) -> Distance {
        Distance::Euclidean
    }
}

impl ComputeDistance<ManhattanMetric> for TypeSafeDistance<ManhattanMetric> {
    fn compute(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        Distance::Manhattan.calculate(x, y)
    }

    fn to_distance(&self) -> Distance {
        Distance::Manhattan
    }
}

impl<const P: usize> ComputeDistance<MinkowskiMetric<P>> for TypeSafeDistance<MinkowskiMetric<P>> {
    fn compute(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        Distance::Minkowski(P as Float).calculate(x, y)
    }

    fn to_distance(&self) -> Distance {
        Distance::Minkowski(P as Float)
    }
}

impl ComputeDistance<CosineMetric> for TypeSafeDistance<CosineMetric> {
    fn compute(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        Distance::Cosine.calculate(x, y)
    }

    fn to_distance(&self) -> Distance {
        Distance::Cosine
    }
}

impl ComputeDistance<ChebyshevMetric> for TypeSafeDistance<ChebyshevMetric> {
    fn compute(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
        Distance::Chebyshev.calculate(x, y)
    }

    fn to_distance(&self) -> Distance {
        Distance::Chebyshev
    }
}

/// Type-safe KNN configuration that ensures metric properties at compile time
///
/// This allows writing generic code that only works with proper metrics:
///
/// # Example
/// ```ignore
/// fn only_metrics<M: MetricDistance>(config: &TypeSafeKnnConfig<M>) {
///     // This function can only be called with proper metric distances
/// }
/// ```
pub struct TypeSafeKnnConfig<M> {
    pub k: usize,
    pub distance: TypeSafeDistance<M>,
}

impl<M> TypeSafeKnnConfig<M> {
    /// Create a new type-safe KNN configuration
    pub const fn new(k: usize) -> Self {
        Self {
            k,
            distance: TypeSafeDistance::new(),
        }
    }
}

impl<M: MetricDistance> TypeSafeKnnConfig<M> {
    /// This method is only available for proper metric distances
    ///
    /// It provides additional guarantees that can be relied upon
    /// when the distance is known to be a metric.
    pub fn with_metric_guarantees(&self) -> &Self {
        // The type system guarantees M implements MetricDistance
        self
    }
}

impl<M: NormalizedDistance> TypeSafeKnnConfig<M> {
    /// This method is only available for normalized distances
    ///
    /// It can assume distance values are in [0, 1] range.
    pub fn with_normalized_guarantees(&self) -> &Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_euclidean_type_safety() {
        let dist = TypeSafeDistance::<EuclideanMetric>::new();
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];

        let d = dist.compute(&x.view(), &y.view());
        assert!(d > 0.0);

        // Can convert to Distance enum
        assert!(matches!(dist.to_distance(), Distance::Euclidean));
    }

    #[test]
    fn test_manhattan_type_safety() {
        let dist = TypeSafeDistance::<ManhattanMetric>::new();
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];

        let d = dist.compute(&x.view(), &y.view());
        assert!(d > 0.0);

        assert!(matches!(dist.to_distance(), Distance::Manhattan));
    }

    #[test]
    fn test_minkowski_const_generic() {
        // Minkowski with p=3 as compile-time constant
        let dist = TypeSafeDistance::<MinkowskiMetric<3>>::new();
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];

        let d = dist.compute(&x.view(), &y.view());
        assert!(d > 0.0);
    }

    #[test]
    fn test_knn_config_with_metrics() {
        let config = TypeSafeKnnConfig::<EuclideanMetric>::new(5);
        assert_eq!(config.k, 5);

        // This compiles because EuclideanMetric implements MetricDistance
        let _with_guarantees = config.with_metric_guarantees();
    }

    #[test]
    fn test_zero_cost_abstraction() {
        // Verify that TypeSafeDistance has zero size
        assert_eq!(std::mem::size_of::<TypeSafeDistance<EuclideanMetric>>(), 0);
        assert_eq!(std::mem::size_of::<TypeSafeDistance<ManhattanMetric>>(), 0);

        // Verify TypeSafeKnnConfig only stores k (plus potential padding)
        assert_eq!(
            std::mem::size_of::<TypeSafeKnnConfig<EuclideanMetric>>(),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn test_metric_distance_trait() {
        // Demonstrate that we can write generic functions over MetricDistance
        fn use_metric<M: MetricDistance>(_config: TypeSafeKnnConfig<M>) -> bool {
            // This function can only be called with proper metrics
            true
        }

        let euclidean_config = TypeSafeKnnConfig::<EuclideanMetric>::new(5);
        assert!(use_metric(euclidean_config));

        let manhattan_config = TypeSafeKnnConfig::<ManhattanMetric>::new(10);
        assert!(use_metric(manhattan_config));
    }

    #[test]
    fn test_cosine_normalized() {
        let config = TypeSafeKnnConfig::<CosineMetric>::new(5);

        // This compiles because CosineMetric implements NormalizedDistance
        let _normalized = config.with_normalized_guarantees();
    }
}
