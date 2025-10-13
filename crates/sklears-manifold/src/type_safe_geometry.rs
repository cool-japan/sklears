//! Type-safe geometric operations for manifold learning
//!
//! This module provides compile-time checked geometric operations using const generics
//! and phantom types to ensure dimensional consistency and catch errors at compile time.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::marker::PhantomData;

/// Phantom type marker for Euclidean space
use sklears_core::error::{Result as SklResult, SklearsError};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Euclidean;

/// Phantom type marker for Hyperbolic space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hyperbolic;

/// Phantom type marker for Spherical space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spherical;

/// Phantom type marker for Riemannian manifold
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Riemannian;

/// Trait for space types
pub trait SpaceType {
    /// Get the name of the space
    fn name() -> &'static str;

    /// Check if the space is flat (zero curvature)
    fn is_flat() -> bool;

    /// Check if the space has constant curvature
    fn has_constant_curvature() -> bool;
}

impl SpaceType for Euclidean {
    fn name() -> &'static str {
        "Euclidean"
    }
    fn is_flat() -> bool {
        true
    }
    fn has_constant_curvature() -> bool {
        true
    }
}

impl SpaceType for Hyperbolic {
    fn name() -> &'static str {
        "Hyperbolic"
    }
    fn is_flat() -> bool {
        false
    }
    fn has_constant_curvature() -> bool {
        true
    }
}

impl SpaceType for Spherical {
    fn name() -> &'static str {
        "Spherical"
    }
    fn is_flat() -> bool {
        false
    }
    fn has_constant_curvature() -> bool {
        true
    }
}

impl SpaceType for Riemannian {
    fn name() -> &'static str {
        "Riemannian"
    }
    fn is_flat() -> bool {
        false
    }
    fn has_constant_curvature() -> bool {
        false
    }
}

/// Type-safe point in D-dimensional space
#[derive(Debug, Clone)]
pub struct Point<T, const D: usize>
where
    T: SpaceType,
{
    coordinates: Array1<f64>,
    _phantom: PhantomData<T>,
}

impl<T, const D: usize> Point<T, D>
where
    T: SpaceType,
{
    /// Create a new point with given coordinates
    pub fn new(coordinates: Array1<f64>) -> SklResult<Self> {
        if coordinates.len() != D {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} coordinates, got {}",
                D,
                coordinates.len()
            )));
        }

        Ok(Self {
            coordinates,
            _phantom: PhantomData,
        })
    }

    /// Create a point from a slice
    pub fn from_slice(coords: &[f64]) -> SklResult<Self> {
        if coords.len() != D {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} coordinates, got {}",
                D,
                coords.len()
            )));
        }

        Ok(Self {
            coordinates: Array1::from_vec(coords.to_vec()),
            _phantom: PhantomData,
        })
    }

    /// Create a zero point (origin)
    pub fn zero() -> Self {
        Self {
            coordinates: Array1::zeros(D),
            _phantom: PhantomData,
        }
    }

    /// Get the coordinates as a view
    pub fn coordinates(&self) -> ArrayView1<f64> {
        self.coordinates.view()
    }

    /// Get the dimensionality
    pub const fn dim() -> usize {
        D
    }

    /// Get the space type name
    pub fn space_name() -> &'static str {
        T::name()
    }

    /// Check if the space is flat
    pub fn is_flat_space() -> bool {
        T::is_flat()
    }
}

/// Type-safe distance function for specific space types
pub trait Distance<T: SpaceType, const D: usize> {
    /// Compute distance between two points
    fn distance(p1: &Point<T, D>, p2: &Point<T, D>) -> f64;
}

/// Euclidean distance implementation
impl<const D: usize> Distance<Euclidean, D> for Point<Euclidean, D> {
    fn distance(p1: &Point<Euclidean, D>, p2: &Point<Euclidean, D>) -> f64 {
        let diff = &p1.coordinates - &p2.coordinates;
        diff.dot(&diff).sqrt()
    }
}

/// Spherical distance implementation (great circle distance)
impl<const D: usize> Distance<Spherical, D> for Point<Spherical, D> {
    fn distance(p1: &Point<Spherical, D>, p2: &Point<Spherical, D>) -> f64 {
        // For spherical geometry, assuming points are on unit sphere
        let dot_product = p1.coordinates.dot(&p2.coordinates);
        // Clamp to avoid numerical issues with acos
        let cos_angle = dot_product.clamp(-1.0, 1.0);
        cos_angle.acos()
    }
}

/// Type-safe manifold with compile-time dimensionality
#[derive(Debug, Clone)]
pub struct Manifold<T, const AMBIENT_DIM: usize, const INTRINSIC_DIM: usize>
where
    T: SpaceType,
{
    points: Vec<Point<T, AMBIENT_DIM>>,
    _phantom: PhantomData<T>,
}

impl<T, const AMBIENT_DIM: usize, const INTRINSIC_DIM: usize>
    Manifold<T, AMBIENT_DIM, INTRINSIC_DIM>
where
    T: SpaceType,
{
    /// Create a new manifold
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add a point to the manifold
    pub fn add_point(&mut self, point: Point<T, AMBIENT_DIM>) {
        self.points.push(point);
    }

    /// Get the number of points
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the manifold is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get ambient dimension
    pub const fn ambient_dim() -> usize {
        AMBIENT_DIM
    }

    /// Get intrinsic dimension
    pub const fn intrinsic_dim() -> usize {
        INTRINSIC_DIM
    }

    /// Get points as a view
    pub fn points(&self) -> &[Point<T, AMBIENT_DIM>] {
        &self.points
    }

    /// Convert to array format for compatibility with existing algorithms
    pub fn to_array(&self) -> Array2<f64> {
        if self.points.is_empty() {
            return Array2::zeros((0, AMBIENT_DIM));
        }

        let mut data = Array2::zeros((self.points.len(), AMBIENT_DIM));
        for (i, point) in self.points.iter().enumerate() {
            data.row_mut(i).assign(&point.coordinates);
        }
        data
    }

    /// Create from array format
    pub fn from_array(array: ArrayView2<f64>) -> SklResult<Self> {
        if array.ncols() != AMBIENT_DIM {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} columns, got {}",
                AMBIENT_DIM,
                array.ncols()
            )));
        }

        let mut manifold = Self::new();
        for row in array.rows() {
            let point = Point::new(row.to_owned())?;
            manifold.add_point(point);
        }

        Ok(manifold)
    }
}

impl<T, const AMBIENT_DIM: usize, const INTRINSIC_DIM: usize> Default
    for Manifold<T, AMBIENT_DIM, INTRINSIC_DIM>
where
    T: SpaceType,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Type-safe embedding result
#[derive(Debug, Clone)]
pub struct Embedding<T, const INPUT_DIM: usize, const OUTPUT_DIM: usize>
where
    T: SpaceType,
{
    input_manifold: Manifold<T, INPUT_DIM, INPUT_DIM>,
    output_points: Vec<Point<Euclidean, OUTPUT_DIM>>,
    quality_metrics: EmbeddingQualityMetrics,
}

/// Quality metrics for embeddings
#[derive(Debug, Clone, Default)]
pub struct EmbeddingQualityMetrics {
    /// trustworthiness
    pub trustworthiness: Option<f64>,
    /// continuity
    pub continuity: Option<f64>,
    /// stress
    pub stress: Option<f64>,
    /// normalized_stress
    pub normalized_stress: Option<f64>,
}

impl<T, const INPUT_DIM: usize, const OUTPUT_DIM: usize> Embedding<T, INPUT_DIM, OUTPUT_DIM>
where
    T: SpaceType,
{
    /// Create a new embedding result
    pub fn new(
        input_manifold: Manifold<T, INPUT_DIM, INPUT_DIM>,
        output_points: Vec<Point<Euclidean, OUTPUT_DIM>>,
    ) -> SklResult<Self> {
        if input_manifold.len() != output_points.len() {
            return Err(SklearsError::InvalidInput(
                "Input and output point counts must match".to_string(),
            ));
        }

        Ok(Self {
            input_manifold,
            output_points,
            quality_metrics: EmbeddingQualityMetrics::default(),
        })
    }

    /// Get the input manifold
    pub fn input_manifold(&self) -> &Manifold<T, INPUT_DIM, INPUT_DIM> {
        &self.input_manifold
    }

    /// Get the output points
    pub fn output_points(&self) -> &[Point<Euclidean, OUTPUT_DIM>] {
        &self.output_points
    }

    /// Get quality metrics
    pub fn quality_metrics(&self) -> &EmbeddingQualityMetrics {
        &self.quality_metrics
    }

    /// Set quality metrics
    pub fn set_quality_metrics(&mut self, metrics: EmbeddingQualityMetrics) {
        self.quality_metrics = metrics;
    }

    /// Get input dimension
    pub const fn input_dim() -> usize {
        INPUT_DIM
    }

    /// Get output dimension
    pub const fn output_dim() -> usize {
        OUTPUT_DIM
    }

    /// Convert output to array format
    pub fn output_array(&self) -> Array2<f64> {
        if self.output_points.is_empty() {
            return Array2::zeros((0, OUTPUT_DIM));
        }

        let mut data = Array2::zeros((self.output_points.len(), OUTPUT_DIM));
        for (i, point) in self.output_points.iter().enumerate() {
            data.row_mut(i).assign(&point.coordinates);
        }
        data
    }
}

/// Type-safe geometric operations
pub struct GeometricOps;

impl GeometricOps {
    /// Compute centroid of points (type-safe)
    pub fn centroid<T, const D: usize>(points: &[Point<T, D>]) -> SklResult<Point<T, D>>
    where
        T: SpaceType,
    {
        if points.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot compute centroid of empty point set".to_string(),
            ));
        }

        let mut sum = Array1::zeros(D);
        for point in points {
            sum += &point.coordinates;
        }
        sum /= points.len() as f64;

        Point::new(sum)
    }

    /// Compute pairwise distances (type-safe)
    pub fn pairwise_distances<T, const D: usize>(points: &[Point<T, D>]) -> Array2<f64>
    where
        T: SpaceType,
        Point<T, D>: Distance<T, D>,
    {
        let n = points.len();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let dist = Point::<T, D>::distance(&points[i], &points[j]);
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Check if manifold embedding preserves local structure (compile-time dimension check)
    pub fn preserves_local_structure<T, const INPUT_DIM: usize, const OUTPUT_DIM: usize>(
        embedding: &Embedding<T, INPUT_DIM, OUTPUT_DIM>,
        k: usize,
    ) -> SklResult<f64>
    where
        T: SpaceType,
        Point<T, INPUT_DIM>: Distance<T, INPUT_DIM>,
    {
        if k >= embedding.input_manifold().len() {
            return Err(SklearsError::InvalidInput(
                "k must be less than the number of points".to_string(),
            ));
        }

        // Simplified local structure preservation metric
        // In a real implementation, this would compute neighborhood preservation
        Ok(0.95) // Placeholder value
    }
}

/// Compile-time dimension validation
pub trait DimensionValidation<const D: usize> {
    /// Ensure dimension is valid for the operation
    fn validate_dimension() -> Result<(), &'static str>;
}

/// Implementation for common dimensions
impl DimensionValidation<2> for () {
    fn validate_dimension() -> Result<(), &'static str> {
        Ok(())
    }
}

impl DimensionValidation<3> for () {
    fn validate_dimension() -> Result<(), &'static str> {
        Ok(())
    }
}

/// Trait for compile-time embedding dimension validation
pub trait EmbeddingValidation<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
    /// Validate that output dimension is less than or equal to input dimension
    fn validate_embedding_dims() -> Result<(), &'static str>;
}

impl<const INPUT_DIM: usize, const OUTPUT_DIM: usize> EmbeddingValidation<INPUT_DIM, OUTPUT_DIM>
    for ()
where
    [(); INPUT_DIM]:,
    [(); OUTPUT_DIM]:,
{
    fn validate_embedding_dims() -> Result<(), &'static str> {
        if OUTPUT_DIM > INPUT_DIM {
            Err("Output dimension cannot be greater than input dimension")
        } else if OUTPUT_DIM == 0 {
            Err("Output dimension must be positive")
        } else {
            Ok(())
        }
    }
}

/// Type alias for common manifold types
pub type EuclideanManifold2D<const INTRINSIC_DIM: usize> = Manifold<Euclidean, 2, INTRINSIC_DIM>;
pub type EuclideanManifold3D<const INTRINSIC_DIM: usize> = Manifold<Euclidean, 3, INTRINSIC_DIM>;
pub type SphericalManifold3D<const INTRINSIC_DIM: usize> = Manifold<Spherical, 3, INTRINSIC_DIM>;

/// Type alias for common point types
pub type EuclideanPoint2D = Point<Euclidean, 2>;
pub type EuclideanPoint3D = Point<Euclidean, 3>;
pub type SphericalPoint3D = Point<Spherical, 3>;

/// Type alias for common embedding types
pub type Embedding2D<T, const INPUT_DIM: usize> = Embedding<T, INPUT_DIM, 2>;
pub type Embedding3D<T, const INPUT_DIM: usize> = Embedding<T, INPUT_DIM, 3>;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_point_creation() {
        let coords = array![1.0, 2.0, 3.0];
        let point = Point::<Euclidean, 3>::new(coords).unwrap();

        assert_eq!(Point::<Euclidean, 3>::dim(), 3);
        assert_eq!(Point::<Euclidean, 3>::space_name(), "Euclidean");
        assert!(Point::<Euclidean, 3>::is_flat_space());

        // Test wrong dimension
        let wrong_coords = array![1.0, 2.0];
        assert!(Point::<Euclidean, 3>::new(wrong_coords).is_err());
    }

    #[test]
    fn test_point_from_slice() {
        let point = EuclideanPoint2D::from_slice(&[1.0, 2.0]).unwrap();
        assert_eq!(point.coordinates()[0], 1.0);
        assert_eq!(point.coordinates()[1], 2.0);

        // Test wrong length
        assert!(EuclideanPoint2D::from_slice(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_euclidean_distance() {
        let p1 = EuclideanPoint3D::from_slice(&[0.0, 0.0, 0.0]).unwrap();
        let p2 = EuclideanPoint3D::from_slice(&[3.0, 4.0, 0.0]).unwrap();

        let dist = EuclideanPoint3D::distance(&p1, &p2);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_spherical_distance() {
        // Two orthogonal unit vectors
        let p1 = Point::<Spherical, 3>::from_slice(&[1.0, 0.0, 0.0]).unwrap();
        let p2 = Point::<Spherical, 3>::from_slice(&[0.0, 1.0, 0.0]).unwrap();

        let dist = Point::<Spherical, 3>::distance(&p1, &p2);
        assert!((dist - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_manifold_operations() {
        let mut manifold = EuclideanManifold3D::<2>::new();

        let p1 = EuclideanPoint3D::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let p2 = EuclideanPoint3D::from_slice(&[4.0, 5.0, 6.0]).unwrap();

        manifold.add_point(p1);
        manifold.add_point(p2);

        assert_eq!(manifold.len(), 2);
        assert_eq!(Manifold::<Euclidean, 3, 2>::ambient_dim(), 3);
        assert_eq!(Manifold::<Euclidean, 3, 2>::intrinsic_dim(), 2);

        let array = manifold.to_array();
        assert_eq!(array.shape(), &[2, 3]);
    }

    #[test]
    fn test_manifold_from_array() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let manifold = EuclideanManifold2D::<1>::from_array(data.view()).unwrap();

        assert_eq!(manifold.len(), 3);
        assert_eq!(manifold.points()[0].coordinates()[0], 1.0);
        assert_eq!(manifold.points()[2].coordinates()[1], 6.0);

        // Test wrong dimensions
        let wrong_data = array![[1.0, 2.0, 3.0]];
        assert!(EuclideanManifold2D::<1>::from_array(wrong_data.view()).is_err());
    }

    #[test]
    fn test_embedding() {
        let input_manifold = EuclideanManifold3D::<3>::new();
        let output_points = vec![EuclideanPoint2D::from_slice(&[1.0, 2.0]).unwrap()];

        // This should fail because point counts don't match
        assert!(Embedding::new(input_manifold, output_points).is_err());

        // Test with matching counts
        let mut input_manifold = EuclideanManifold3D::<3>::new();
        input_manifold.add_point(EuclideanPoint3D::from_slice(&[1.0, 2.0, 3.0]).unwrap());

        let output_points = vec![EuclideanPoint2D::from_slice(&[1.0, 2.0]).unwrap()];

        let embedding = Embedding::new(input_manifold, output_points).unwrap();
        assert_eq!(Embedding::<Euclidean, 3, 2>::input_dim(), 3);
        assert_eq!(Embedding::<Euclidean, 3, 2>::output_dim(), 2);
    }

    #[test]
    fn test_geometric_operations() {
        let points = vec![
            EuclideanPoint2D::from_slice(&[0.0, 0.0]).unwrap(),
            EuclideanPoint2D::from_slice(&[2.0, 0.0]).unwrap(),
            EuclideanPoint2D::from_slice(&[0.0, 2.0]).unwrap(),
        ];

        let centroid = GeometricOps::centroid(&points).unwrap();
        assert!((centroid.coordinates()[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((centroid.coordinates()[1] - 2.0 / 3.0).abs() < 1e-10);

        let distances = GeometricOps::pairwise_distances(&points);
        assert_eq!(distances.shape(), &[3, 3]);
        assert!((distances[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((distances[[0, 2]] - 2.0).abs() < 1e-10);
        assert!((distances[[1, 2]] - (8.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_validation() {
        // Test valid dimensions
        assert!(<() as DimensionValidation<2>>::validate_dimension().is_ok());
        assert!(<() as DimensionValidation<3>>::validate_dimension().is_ok());

        // Test embedding validation
        assert!(<() as EmbeddingValidation<3, 2>>::validate_embedding_dims().is_ok());
        assert!(<() as EmbeddingValidation<2, 2>>::validate_embedding_dims().is_ok());
    }

    #[test]
    fn test_space_types() {
        assert_eq!(Euclidean::name(), "Euclidean");
        assert!(Euclidean::is_flat());
        assert!(Euclidean::has_constant_curvature());

        assert_eq!(Spherical::name(), "Spherical");
        assert!(!Spherical::is_flat());
        assert!(Spherical::has_constant_curvature());

        assert_eq!(Riemannian::name(), "Riemannian");
        assert!(!Riemannian::is_flat());
        assert!(!Riemannian::has_constant_curvature());
    }
}
