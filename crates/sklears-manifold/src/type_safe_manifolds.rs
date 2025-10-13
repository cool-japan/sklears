//! Type-safe manifold abstractions with phantom types
//!
//! This module provides compile-time type safety for manifold learning algorithms
//! using phantom types to encode manifold structure, dimensionality, and properties
//! at the type level. This prevents many runtime errors and improves API safety.

use scirs2_core::ndarray::{Array2, ArrayView2};
/// Phantom type markers for manifold structure types
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::marker::PhantomData;
pub mod structure {
    /// Marker for Euclidean manifolds (flat geometry)
    pub struct Euclidean;

    /// Marker for Riemannian manifolds (curved geometry)
    pub struct Riemannian;

    /// Marker for topological manifolds (general topology)
    pub struct Topological;

    /// Marker for discrete manifolds (graph-like structures)
    pub struct Discrete;

    /// Marker for unknown/generic manifold structure
    pub struct Unknown;
}

/// Phantom type markers for manifold properties
pub mod properties {
    /// Marker for manifolds with known curvature
    pub struct HasCurvature;

    /// Marker for manifolds without curvature (flat)
    pub struct NoCurvature;

    /// Marker for orientable manifolds
    pub struct Orientable;

    /// Marker for non-orientable manifolds
    pub struct NonOrientable;

    /// Marker for compact manifolds
    pub struct Compact;

    /// Marker for non-compact manifolds
    pub struct NonCompact;

    /// Marker for connected manifolds
    pub struct Connected;

    /// Marker for disconnected manifolds
    pub struct Disconnected;
}

/// Phantom type markers for dimensionality
pub mod dimension {
    use std::marker::PhantomData;

    /// Compile-time dimension representation
    pub struct Dim<const N: usize>(PhantomData<[(); N]>);

    /// Dynamic dimension (unknown at compile time)
    pub struct Dynamic;

    /// Type alias for common dimensions
    pub type Dim1 = Dim<1>;
    pub type Dim2 = Dim<2>;
    pub type Dim3 = Dim<3>;
    pub type Dim4 = Dim<4>;
    pub type DimN<const N: usize> = Dim<N>;
}

/// Type-safe manifold wrapper that encodes structure and properties
///
/// This struct uses phantom types to encode manifold characteristics at compile time,
/// providing type safety and preventing invalid operations.
///
/// # Type Parameters
///
/// * `S` - Structure type (Euclidean, Riemannian, etc.)
/// * `P` - Properties type (HasCurvature, Orientable, etc.)
/// * `D` - Dimension type (Dim<N> or Dynamic)
///
/// # Examples
///
/// ```
/// use sklears_manifold::type_safe_manifolds::*;
/// use scirs2_core::ndarray::{array, ArrayView2};
///
/// // Create a 2D Euclidean manifold
/// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let manifold = TypeSafeManifold::<structure::Euclidean, properties::NoCurvature, dimension::Dim2>::new(data);
/// ```
#[derive(Debug, Clone)]
pub struct TypeSafeManifold<S, P, D> {
    data: Array2<Float>,
    _structure: PhantomData<S>,
    _properties: PhantomData<P>,
    _dimension: PhantomData<D>,
}

impl<S, P, D> TypeSafeManifold<S, P, D> {
    /// Create a new type-safe manifold
    ///
    /// # Arguments
    ///
    /// * `data` - The manifold data matrix
    ///
    /// # Returns
    ///
    /// A type-safe manifold wrapper
    pub fn new(data: Array2<Float>) -> Self {
        Self {
            data,
            _structure: PhantomData,
            _properties: PhantomData,
            _dimension: PhantomData,
        }
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &Array2<Float> {
        &self.data
    }

    /// Get a view of the underlying data
    pub fn view(&self) -> ArrayView2<Float> {
        self.data.view()
    }

    /// Get the number of points on the manifold
    pub fn n_points(&self) -> usize {
        self.data.nrows()
    }

    /// Get the ambient dimension of the manifold
    pub fn ambient_dim(&self) -> usize {
        self.data.ncols()
    }
}

/// Implementation for manifolds with compile-time known dimensions
impl<S, P, const N: usize> TypeSafeManifold<S, P, dimension::Dim<N>> {
    /// Get the compile-time known dimension
    pub const fn intrinsic_dim() -> usize {
        N
    }

    /// Validate that the data has the correct dimension
    pub fn validate_dimension(&self) -> SklResult<()> {
        if self.ambient_dim() < N {
            return Err(SklearsError::InvalidParameter {
                name: "ambient_dimension".to_string(),
                reason: format!(
                    "Ambient dimension {} is less than intrinsic dimension {}",
                    self.ambient_dim(),
                    N
                ),
            });
        }
        Ok(())
    }
}

/// Implementation for dynamic dimension manifolds
impl<S, P> TypeSafeManifold<S, P, dimension::Dynamic> {
    /// Set the intrinsic dimension dynamically
    pub fn with_intrinsic_dim(self, intrinsic_dim: usize) -> SklResult<Self> {
        if intrinsic_dim > self.ambient_dim() {
            return Err(SklearsError::InvalidParameter {
                name: "intrinsic_dimension".to_string(),
                reason: format!(
                    "Intrinsic dimension {} exceeds ambient dimension {}",
                    intrinsic_dim,
                    self.ambient_dim()
                ),
            });
        }
        Ok(self)
    }
}

/// Euclidean manifold specific operations
impl<P, D> TypeSafeManifold<structure::Euclidean, P, D> {
    /// Compute Euclidean distances between points
    pub fn euclidean_distances(&self) -> Array2<Float> {
        let n = self.n_points();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let dist = self.euclidean_distance_pair(i, j);
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance_pair(&self, i: usize, j: usize) -> Float {
        let row_i = self.data.row(i);
        let row_j = self.data.row(j);
        row_i
            .iter()
            .zip(row_j.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<Float>()
            .sqrt()
    }

    /// Project onto a linear subspace (valid for Euclidean manifolds)
    pub fn linear_projection(&self, target_dim: usize) -> SklResult<Array2<Float>> {
        if target_dim > self.ambient_dim() {
            return Err(SklearsError::InvalidParameter {
                name: "target_dimension".to_string(),
                reason: format!(
                    "Target dimension {} exceeds ambient dimension {}",
                    target_dim,
                    self.ambient_dim()
                ),
            });
        }

        // Simple projection by taking the first target_dim columns
        Ok(self
            .data
            .slice(scirs2_core::ndarray::s![.., ..target_dim])
            .to_owned())
    }
}

/// Riemannian manifold specific operations
impl<P, D> TypeSafeManifold<structure::Riemannian, P, D> {
    /// Estimate geodesic distances (placeholder implementation)
    pub fn geodesic_distances(&self) -> SklResult<Array2<Float>> {
        // This would implement geodesic distance computation
        // For now, we'll use Euclidean as an approximation
        Ok(self.euclidean_approximation())
    }

    /// Compute metric tensor at a point (placeholder)
    pub fn metric_tensor(&self, point_idx: usize) -> SklResult<Array2<Float>> {
        if point_idx >= self.n_points() {
            return Err(SklearsError::InvalidParameter {
                name: "point_index".to_string(),
                reason: format!("Point index {} out of bounds", point_idx),
            });
        }

        // Placeholder: return identity matrix as metric tensor
        let dim = self.ambient_dim();
        Ok(Array2::eye(dim))
    }

    /// Euclidean approximation for Riemannian distances
    fn euclidean_approximation(&self) -> Array2<Float> {
        let n = self.n_points();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let row_i = self.data.row(i);
                let row_j = self.data.row(j);
                let dist = row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }
}

/// Discrete manifold operations (for graph-like structures)
impl<P, D> TypeSafeManifold<structure::Discrete, P, D> {
    /// Build adjacency matrix based on k-nearest neighbors
    pub fn knn_adjacency(&self, k: usize) -> SklResult<Array2<Float>> {
        if k >= self.n_points() {
            return Err(SklearsError::InvalidParameter {
                name: "k".to_string(),
                reason: format!(
                    "k={} must be less than number of points {}",
                    k,
                    self.n_points()
                ),
            });
        }

        let n = self.n_points();
        let mut adjacency = Array2::zeros((n, n));

        // Compute distances and find k-nearest neighbors
        for i in 0..n {
            let mut distances: Vec<(Float, usize)> = Vec::new();

            for j in 0..n {
                if i != j {
                    let row_i = self.data.row(i);
                    let row_j = self.data.row(j);
                    let dist = row_i
                        .iter()
                        .zip(row_j.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<Float>()
                        .sqrt();
                    distances.push((dist, j));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (dist, neighbor) in distances.iter().take(k) {
                adjacency[[i, *neighbor]] = 1.0;
                adjacency[[*neighbor, i]] = 1.0; // Make symmetric
            }
        }

        Ok(adjacency)
    }

    /// Compute shortest path distances using Floyd-Warshall
    pub fn shortest_path_distances(&self, adjacency: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n = self.n_points();
        if adjacency.shape() != [n, n] {
            return Err(SklearsError::InvalidParameter {
                name: "adjacency_shape".to_string(),
                reason: format!(
                    "Adjacency matrix shape {:?} doesn't match data shape [{}x{}]",
                    adjacency.shape(),
                    n,
                    n
                ),
            });
        }

        let mut distances = adjacency.clone();

        // Initialize with large values for non-connected pairs
        for i in 0..n {
            for j in 0..n {
                if i != j && distances[[i, j]] == 0.0 {
                    distances[[i, j]] = Float::INFINITY;
                }
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let through_k = distances[[i, k]] + distances[[k, j]];
                    if through_k < distances[[i, j]] {
                        distances[[i, j]] = through_k;
                    }
                }
            }
        }

        Ok(distances)
    }
}

/// Type conversion utilities
impl<S, P, D> TypeSafeManifold<S, P, D> {
    /// Convert to a different structure type
    pub fn cast_structure<NewS>(self) -> TypeSafeManifold<NewS, P, D> {
        TypeSafeManifold {
            data: self.data,
            _structure: PhantomData,
            _properties: PhantomData,
            _dimension: PhantomData,
        }
    }

    /// Convert to a different property type
    pub fn cast_properties<NewP>(self) -> TypeSafeManifold<S, NewP, D> {
        TypeSafeManifold {
            data: self.data,
            _structure: PhantomData,
            _properties: PhantomData,
            _dimension: PhantomData,
        }
    }

    /// Convert to dynamic dimension
    pub fn to_dynamic_dim(self) -> TypeSafeManifold<S, P, dimension::Dynamic> {
        TypeSafeManifold {
            data: self.data,
            _structure: PhantomData,
            _properties: PhantomData,
            _dimension: PhantomData,
        }
    }
}

/// Manifold builder for constructing type-safe manifolds
#[derive(Debug)]
pub struct ManifoldBuilder<S, P, D> {
    _structure: PhantomData<S>,
    _properties: PhantomData<P>,
    _dimension: PhantomData<D>,
}

impl Default for ManifoldBuilder<structure::Unknown, properties::NoCurvature, dimension::Dynamic> {
    fn default() -> Self {
        Self::new()
    }
}

impl ManifoldBuilder<structure::Unknown, properties::NoCurvature, dimension::Dynamic> {
    /// Create a new manifold builder
    pub fn new() -> Self {
        Self {
            _structure: PhantomData,
            _properties: PhantomData,
            _dimension: PhantomData,
        }
    }
}

impl<S, P, D> ManifoldBuilder<S, P, D> {
    /// Set the structure type
    pub fn structure<NewS>(self) -> ManifoldBuilder<NewS, P, D> {
        ManifoldBuilder {
            _structure: PhantomData,
            _properties: PhantomData,
            _dimension: PhantomData,
        }
    }

    /// Set the properties type
    pub fn properties<NewP>(self) -> ManifoldBuilder<S, NewP, D> {
        ManifoldBuilder {
            _structure: PhantomData,
            _properties: PhantomData,
            _dimension: PhantomData,
        }
    }

    /// Set the dimension type
    pub fn dimension<NewD>(self) -> ManifoldBuilder<S, P, NewD> {
        ManifoldBuilder {
            _structure: PhantomData,
            _properties: PhantomData,
            _dimension: PhantomData,
        }
    }

    /// Build the manifold with the given data
    pub fn build(self, data: Array2<Float>) -> TypeSafeManifold<S, P, D> {
        TypeSafeManifold::new(data)
    }
}

/// Trait for validating manifold properties at compile time
pub trait ManifoldValidator<S, P, D> {
    /// Validate manifold structure and properties
    fn validate(&self) -> SklResult<()>;
}

/// Default validation for dynamic dimension manifolds
impl<S, P> ManifoldValidator<S, P, dimension::Dynamic>
    for TypeSafeManifold<S, P, dimension::Dynamic>
{
    fn validate(&self) -> SklResult<()> {
        if self.n_points() == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_points".to_string(),
                reason: "Manifold must have at least one point".to_string(),
            });
        }

        if self.ambient_dim() == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "ambient_dimension".to_string(),
                reason: "Manifold must have positive ambient dimension".to_string(),
            });
        }

        Ok(())
    }
}

/// Specialized validation for fixed-dimension manifolds
impl<S, P, const N: usize> ManifoldValidator<S, P, dimension::Dim<N>>
    for TypeSafeManifold<S, P, dimension::Dim<N>>
{
    fn validate(&self) -> SklResult<()> {
        // Basic validation
        if self.n_points() == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_points".to_string(),
                reason: "Manifold must have at least one point".to_string(),
            });
        }

        if self.ambient_dim() == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "ambient_dimension".to_string(),
                reason: "Manifold must have positive ambient dimension".to_string(),
            });
        }

        // Additional dimension validation
        self.validate_dimension()?;

        Ok(())
    }
}

/// Type alias helpers for common manifold types
pub type EuclideanManifold2D =
    TypeSafeManifold<structure::Euclidean, properties::NoCurvature, dimension::Dim2>;
pub type EuclideanManifold3D =
    TypeSafeManifold<structure::Euclidean, properties::NoCurvature, dimension::Dim3>;
pub type RiemannianManifold<const N: usize> =
    TypeSafeManifold<structure::Riemannian, properties::HasCurvature, dimension::Dim<N>>;
pub type DiscreteManifold =
    TypeSafeManifold<structure::Discrete, properties::Disconnected, dimension::Dynamic>;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, ArrayView2};

    #[test]
    fn test_euclidean_manifold_creation() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let manifold = EuclideanManifold2D::new(data);

        assert_eq!(manifold.n_points(), 3);
        assert_eq!(manifold.ambient_dim(), 2);
        assert_eq!(EuclideanManifold2D::intrinsic_dim(), 2);
    }

    #[test]
    fn test_euclidean_distances() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let manifold = EuclideanManifold2D::new(data);

        let distances = manifold.euclidean_distances();

        assert_abs_diff_eq!(distances[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[[0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[[1, 2]], 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_manifold_builder() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let manifold = ManifoldBuilder::new()
            .structure::<structure::Euclidean>()
            .properties::<properties::NoCurvature>()
            .dimension::<dimension::Dim3>()
            .build(data);

        assert_eq!(manifold.n_points(), 2);
        assert_eq!(manifold.ambient_dim(), 3);
    }

    #[test]
    fn test_dimension_validation() {
        let data = array![[1.0, 2.0], [3.0, 4.0]]; // 2D data
        let manifold = TypeSafeManifold::<
            structure::Euclidean,
            properties::NoCurvature,
            dimension::Dim3,
        >::new(data);

        // Should fail because data is 2D but we claim it's 3D
        assert!(manifold.validate_dimension().is_err());
    }

    #[test]
    fn test_discrete_manifold_adjacency() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let manifold = DiscreteManifold::new(data);

        let adjacency = manifold.knn_adjacency(2).unwrap();

        // Each point should be connected to its 2 nearest neighbors
        // For 4 points in a line: [0,0], [1,0], [2,0], [3,0] with k=2
        // Point 0 → neighbors 1,2
        // Point 1 → neighbors 0,2
        // Point 2 → neighbors 1,3
        // Point 3 → neighbors 2,1
        // This creates 10 total connections (5 unique edges × 2 for symmetry)
        assert_eq!(adjacency.sum(), 10.0);
    }

    #[test]
    fn test_type_conversion() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let euclidean_manifold = TypeSafeManifold::<
            structure::Euclidean,
            properties::NoCurvature,
            dimension::Dim2,
        >::new(data);

        // Convert to Riemannian manifold
        let riemannian_manifold = euclidean_manifold.cast_structure::<structure::Riemannian>();

        assert_eq!(riemannian_manifold.n_points(), 2);
        assert_eq!(riemannian_manifold.ambient_dim(), 2);
    }

    #[test]
    fn test_validation() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let manifold = EuclideanManifold2D::new(data);

        assert!(manifold.validate().is_ok());
    }

    #[test]
    fn test_empty_manifold_validation() {
        let data = Array2::zeros((0, 2));
        let manifold = EuclideanManifold2D::new(data);

        assert!(manifold.validate().is_err());
    }
}
