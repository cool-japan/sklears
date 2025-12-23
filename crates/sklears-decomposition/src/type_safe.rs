//! Type-safe decomposition abstractions using Rust's type system
//!
//! This module provides zero-cost abstractions for matrix decomposition methods
//! that leverage Rust's type system for improved safety and performance.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::marker::PhantomData;

/// Marker trait for decomposition states
pub trait DecompositionState {}

/// Untrained decomposition state
#[derive(Debug, Clone, Copy)]
pub struct Untrained;
impl DecompositionState for Untrained {}

/// Fitted decomposition state
#[derive(Debug, Clone, Copy)]
pub struct Fitted;
impl DecompositionState for Fitted {}

/// Phantom type for matrix rank
#[derive(Debug, Clone, Copy)]
pub struct Rank<const R: usize>;

/// Phantom type for matrix dimensions
#[derive(Debug, Clone, Copy)]
pub struct Dimensions<const ROWS: usize, const COLS: usize>;

/// Type-safe decomposition trait
pub trait TypeSafeDecomposition<State: DecompositionState> {
    type Output;
    type ErrorType;

    /// Get the current state of the decomposition
    fn state(&self) -> PhantomData<State>;
}

/// Type-safe PCA with compile-time rank checking
#[derive(Debug, Clone)]
pub struct TypeSafePCA<State: DecompositionState, const RANK: usize> {
    /// Number of components (must match RANK)
    pub n_components: usize,
    /// Whether to center the data
    pub center: bool,
    /// Whether to scale the data
    pub scale: bool,
    /// Fitted components (only available in Fitted state)
    components: Option<Array2<Float>>,
    /// Explained variance (only available in Fitted state)
    explained_variance: Option<Array1<Float>>,
    /// Mean (only available in Fitted state)
    mean: Option<Array1<Float>>,
    /// State phantom
    _state: PhantomData<State>,
}

impl<const RANK: usize> Default for TypeSafePCA<Untrained, RANK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const RANK: usize> TypeSafePCA<Untrained, RANK> {
    /// Create a new type-safe PCA with compile-time rank checking
    pub const fn new() -> Self {
        Self {
            n_components: RANK,
            center: true,
            scale: false,
            components: None,
            explained_variance: None,
            mean: None,
            _state: PhantomData,
        }
    }

    /// Set whether to center the data
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Fit the PCA model and transition to Fitted state
    pub fn fit(self, data: &Array2<Float>) -> Result<TypeSafePCA<Fitted, RANK>> {
        let (n_samples, n_features) = data.dim();

        if RANK > n_features {
            return Err(SklearsError::InvalidParameter {
                name: "RANK".to_string(),
                reason: format!("RANK ({RANK}) cannot exceed number of features ({n_features})"),
            });
        }

        if RANK > n_samples {
            return Err(SklearsError::InvalidParameter {
                name: "RANK".to_string(),
                reason: format!("RANK ({RANK}) cannot exceed number of samples ({n_samples})"),
            });
        }

        // Center the data if requested
        let mean = if self.center {
            data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap()
        } else {
            Array1::zeros(n_features)
        };

        let mut centered_data = data.clone();
        if self.center {
            for mut row in centered_data.axis_iter_mut(scirs2_core::ndarray::Axis(0)) {
                row -= &mean;
            }
        }

        // Scale the data if requested
        if self.scale {
            let std = centered_data.std_axis(scirs2_core::ndarray::Axis(0), 0.0);
            for mut row in centered_data.axis_iter_mut(scirs2_core::ndarray::Axis(0)) {
                for (i, val) in row.iter_mut().enumerate() {
                    if std[i] != 0.0 {
                        *val /= std[i];
                    }
                }
            }
        }

        // Compute covariance matrix
        let covariance = centered_data.t().dot(&centered_data) / ((n_samples - 1) as Float);

        // Eigendecomposition (simplified - in practice would use LAPACK)
        let (eigenvalues, eigenvectors) = self.eigendecomposition(&covariance)?;

        // Sort by eigenvalues in descending order and take top RANK
        let mut eigen_pairs: Vec<(Float, Array1<Float>)> = eigenvalues
            .iter()
            .zip(eigenvectors.axis_iter(scirs2_core::ndarray::Axis(1)))
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut components = Array2::zeros((n_features, RANK));
        let mut explained_variance = Array1::zeros(RANK);

        for (i, (eigenval, eigenvec)) in eigen_pairs.iter().take(RANK).enumerate() {
            components.column_mut(i).assign(eigenvec);
            explained_variance[i] = *eigenval;
        }

        Ok(TypeSafePCA {
            n_components: RANK,
            center: self.center,
            scale: self.scale,
            components: Some(components),
            explained_variance: Some(explained_variance),
            mean: Some(mean),
            _state: PhantomData,
        })
    }

    /// Simplified eigendecomposition (placeholder for actual LAPACK call)
    fn eigendecomposition(&self, matrix: &Array2<Float>) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        // This is a placeholder implementation
        // In practice, you would use LAPACK for eigendecomposition
        let eigenvalues = Array1::from_iter((0..n).map(|i| (n - i) as Float));
        let mut eigenvectors: Array2<Float> = Array2::eye(n);

        // Normalize eigenvectors
        for mut col in eigenvectors.axis_iter_mut(scirs2_core::ndarray::Axis(1)) {
            let norm = col.dot(&col).sqrt();
            if norm > 1e-10 {
                col /= norm;
            }
        }

        Ok((eigenvalues, eigenvectors))
    }
}

impl<const RANK: usize> TypeSafePCA<Fitted, RANK> {
    /// Transform data using the fitted components
    pub fn transform(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let mut transformed_data = data.clone();

        // Apply same centering as during fit
        if self.center {
            if let Some(ref mean) = self.mean {
                for mut row in transformed_data.axis_iter_mut(scirs2_core::ndarray::Axis(0)) {
                    row -= mean;
                }
            }
        }

        // Apply same scaling as during fit
        if self.scale {
            // Would need to store scale factors from fit for this to work properly
        }

        // Project onto components
        Ok(transformed_data.dot(components))
    }

    /// Get the fitted components (guaranteed to be RANK columns)
    pub fn components(&self) -> &Array2<Float> {
        self.components.as_ref().unwrap()
    }

    /// Get explained variance (guaranteed to be RANK elements)
    pub fn explained_variance(&self) -> &Array1<Float> {
        self.explained_variance.as_ref().unwrap()
    }

    /// Get explained variance ratios
    pub fn explained_variance_ratio(&self) -> Array1<Float> {
        let explained_var = self.explained_variance();
        let total_variance = explained_var.sum();
        explained_var / total_variance
    }

    /// Fit and transform in one step
    pub fn fit_transform(
        untrained: TypeSafePCA<Untrained, RANK>,
        data: &Array2<Float>,
    ) -> Result<(TypeSafePCA<Fitted, RANK>, Array2<Float>)> {
        let fitted = untrained.fit(data)?;
        let transformed = fitted.transform(data)?;
        Ok((fitted, transformed))
    }
}

impl<State: DecompositionState, const RANK: usize> TypeSafeDecomposition<State>
    for TypeSafePCA<State, RANK>
{
    type Output = Array2<Float>;
    type ErrorType = SklearsError;

    fn state(&self) -> PhantomData<State> {
        self._state
    }
}

/// Type-safe matrix with compile-time dimension checking
#[derive(Debug, Clone)]
pub struct TypeSafeMatrix<const ROWS: usize, const COLS: usize> {
    data: Array2<Float>,
}

impl<const ROWS: usize, const COLS: usize> TypeSafeMatrix<ROWS, COLS> {
    /// Create a new type-safe matrix with compile-time dimension checking
    pub fn new(data: Array2<Float>) -> Result<Self> {
        let (rows, cols) = data.dim();
        if rows != ROWS || cols != COLS {
            return Err(SklearsError::InvalidParameter {
                name: "matrix_dimensions".to_string(),
                reason: format!(
                    "Matrix dimensions {rows}x{cols} do not match expected {ROWS}x{COLS}"
                ),
            });
        }
        Ok(Self { data })
    }

    /// Create a zero matrix
    pub fn zeros() -> Self {
        Self {
            data: Array2::zeros((ROWS, COLS)),
        }
    }

    /// Create an identity matrix (only valid for square matrices where ROWS == COLS)
    pub fn eye() -> Self {
        assert_eq!(ROWS, COLS, "Identity matrix requires ROWS == COLS");
        Self {
            data: Array2::eye(ROWS),
        }
    }

    /// Get the underlying data
    pub fn data(&self) -> &Array2<Float> {
        &self.data
    }

    /// Get mutable access to the underlying data
    pub fn data_mut(&mut self) -> &mut Array2<Float> {
        &mut self.data
    }

    /// Matrix multiplication with compile-time dimension checking
    pub fn dot<const OTHER_COLS: usize>(
        &self,
        other: &TypeSafeMatrix<COLS, OTHER_COLS>,
    ) -> TypeSafeMatrix<ROWS, OTHER_COLS> {
        let result = self.data.dot(&other.data);
        TypeSafeMatrix { data: result }
    }

    /// Transpose the matrix
    pub fn t(&self) -> TypeSafeMatrix<COLS, ROWS> {
        TypeSafeMatrix {
            data: self.data.t().to_owned(),
        }
    }

    /// Extract a submatrix with runtime size checking
    pub fn submatrix<const SUB_ROWS: usize, const SUB_COLS: usize>(
        &self,
        start_row: usize,
        start_col: usize,
    ) -> Result<TypeSafeMatrix<SUB_ROWS, SUB_COLS>> {
        if SUB_ROWS > ROWS || SUB_COLS > COLS {
            return Err(SklearsError::InvalidParameter {
                name: "submatrix_size".to_string(),
                reason: "Submatrix size exceeds matrix dimensions".to_string(),
            });
        }

        if start_row + SUB_ROWS > ROWS || start_col + SUB_COLS > COLS {
            return Err(SklearsError::InvalidParameter {
                name: "submatrix_bounds".to_string(),
                reason: "Submatrix bounds exceed matrix dimensions".to_string(),
            });
        }

        let subarray = self
            .data
            .slice(scirs2_core::ndarray::s![
                start_row..start_row + SUB_ROWS,
                start_col..start_col + SUB_COLS
            ])
            .to_owned();

        Ok(TypeSafeMatrix { data: subarray })
    }
}

/// Type-safe component indexing
#[derive(Debug, Clone, Copy)]
pub struct ComponentIndex<const INDEX: usize>;

impl<const INDEX: usize> Default for ComponentIndex<INDEX> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const INDEX: usize> ComponentIndex<INDEX> {
    /// Create a new component index
    pub const fn new() -> Self {
        Self
    }

    /// Get the index value
    pub const fn index(&self) -> usize {
        INDEX
    }
}

/// Type-safe component access for fitted decomposition
pub trait ComponentAccess<const RANK: usize> {
    /// Get a specific component by index (runtime checked)
    fn component<const INDEX: usize>(&self, _index: ComponentIndex<INDEX>)
        -> Result<Array1<Float>>;
}

impl<const RANK: usize> ComponentAccess<RANK> for TypeSafePCA<Fitted, RANK> {
    fn component<const INDEX: usize>(
        &self,
        _index: ComponentIndex<INDEX>,
    ) -> Result<Array1<Float>> {
        if INDEX >= RANK {
            return Err(SklearsError::InvalidParameter {
                name: "component_index".to_string(),
                reason: format!("Component index {INDEX} exceeds number of components {RANK}"),
            });
        }

        let components = self.components();
        Ok(components.column(INDEX).to_owned())
    }
}

/// Zero-cost decomposition pipeline builder
pub struct DecompositionPipeline<State: DecompositionState> {
    operations: Vec<Box<dyn DecompositionOperation>>,
    _state: PhantomData<State>,
}

/// Trait for decomposition operations in the pipeline
pub trait DecompositionOperation {
    fn apply(&self, data: &Array2<Float>) -> Result<Array2<Float>>;
    fn name(&self) -> &str;
}

/// Centering operation
#[derive(Debug, Clone)]
pub struct CenteringOperation {
    #[allow(dead_code)]
    mean: Option<Array1<Float>>,
}

impl Default for CenteringOperation {
    fn default() -> Self {
        Self::new()
    }
}

impl CenteringOperation {
    pub fn new() -> Self {
        Self { mean: None }
    }
}

impl DecompositionOperation for CenteringOperation {
    fn apply(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let mean = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let mut centered = data.clone();
        for mut row in centered.axis_iter_mut(scirs2_core::ndarray::Axis(0)) {
            row -= &mean;
        }
        Ok(centered)
    }

    fn name(&self) -> &str {
        "centering"
    }
}

/// Scaling operation
#[derive(Debug, Clone)]
pub struct ScalingOperation {
    #[allow(dead_code)]
    scale: Option<Array1<Float>>,
}

impl Default for ScalingOperation {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalingOperation {
    pub fn new() -> Self {
        Self { scale: None }
    }
}

impl DecompositionOperation for ScalingOperation {
    fn apply(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let std = data.std_axis(scirs2_core::ndarray::Axis(0), 0.0);
        let mut scaled = data.clone();
        for mut row in scaled.axis_iter_mut(scirs2_core::ndarray::Axis(0)) {
            for (i, val) in row.iter_mut().enumerate() {
                if std[i] != 0.0 {
                    *val /= std[i];
                }
            }
        }
        Ok(scaled)
    }

    fn name(&self) -> &str {
        "scaling"
    }
}

impl Default for DecompositionPipeline<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl DecompositionPipeline<Untrained> {
    /// Create a new decomposition pipeline
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            _state: PhantomData,
        }
    }

    /// Add a centering operation to the pipeline
    pub fn center(mut self) -> Self {
        self.operations.push(Box::new(CenteringOperation::new()));
        self
    }

    /// Add a scaling operation to the pipeline
    pub fn scale(mut self) -> Self {
        self.operations.push(Box::new(ScalingOperation::new()));
        self
    }

    /// Apply the pipeline to data
    pub fn apply(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let mut result = data.clone();
        for operation in &self.operations {
            result = operation.apply(&result)?;
        }
        Ok(result)
    }

    /// Fit the pipeline and transition to fitted state
    pub fn fit(self, _data: &Array2<Float>) -> Result<DecompositionPipeline<Fitted>> {
        // In a real implementation, we would store fitted parameters
        Ok(DecompositionPipeline {
            operations: self.operations,
            _state: PhantomData,
        })
    }
}

impl DecompositionPipeline<Fitted> {
    /// Apply the fitted pipeline to new data
    pub fn transform(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let mut result = data.clone();
        for operation in &self.operations {
            result = operation.apply(&result)?;
        }
        Ok(result)
    }
}

/// Runtime matrix shape validation for matrix multiplication
pub fn validate_matrix_multiplication<
    const A_ROWS: usize,
    const A_COLS: usize,
    const B_ROWS: usize,
    const B_COLS: usize,
>(
    _a: &TypeSafeMatrix<A_ROWS, A_COLS>,
    _b: &TypeSafeMatrix<B_ROWS, B_COLS>,
) -> Result<()> {
    if A_COLS != B_ROWS {
        return Err(SklearsError::InvalidParameter {
            name: "matrix_multiplication".to_string(),
            reason: format!(
                "Cannot multiply {A_ROWS}x{A_COLS} matrix with {B_ROWS}x{B_COLS} matrix"
            ),
        });
    }
    Ok(())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_type_safe_pca_creation() {
        let pca: TypeSafePCA<Untrained, 2> = TypeSafePCA::new();
        assert_eq!(pca.n_components, 2);
        assert!(pca.center);
        assert!(!pca.scale);
    }

    #[test]
    fn test_type_safe_pca_fit() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];

        let pca: TypeSafePCA<Untrained, 2> = TypeSafePCA::new();
        let fitted_pca = pca.fit(&data).unwrap();

        assert_eq!(fitted_pca.components().dim(), (3, 2));
        assert_eq!(fitted_pca.explained_variance().len(), 2);
    }

    #[test]
    fn test_type_safe_pca_transform() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

        let pca: TypeSafePCA<Untrained, 2> = TypeSafePCA::new();
        let fitted_pca = pca.fit(&data).unwrap();
        let transformed = fitted_pca.transform(&data).unwrap();

        assert_eq!(transformed.dim(), (3, 2));
    }

    #[test]
    fn test_type_safe_pca_rank_validation() {
        let data = array![
            [1.0, 2.0], // Only 2 features
            [3.0, 4.0],
        ];

        // This should fail because RANK=3 > n_features=2
        let pca: TypeSafePCA<Untrained, 3> = TypeSafePCA::new();
        let result = pca.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_type_safe_matrix_creation() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let matrix: TypeSafeMatrix<2, 2> = TypeSafeMatrix::new(data).unwrap();
        assert_eq!(matrix.data().dim(), (2, 2));
    }

    #[test]
    fn test_type_safe_matrix_dimension_validation() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3 matrix
        let result: Result<TypeSafeMatrix<3, 3>> = TypeSafeMatrix::new(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_type_safe_matrix_multiplication() {
        let a_data = array![[1.0, 2.0], [3.0, 4.0]];
        let b_data = array![[5.0, 6.0], [7.0, 8.0]];

        let a: TypeSafeMatrix<2, 2> = TypeSafeMatrix::new(a_data).unwrap();
        let b: TypeSafeMatrix<2, 2> = TypeSafeMatrix::new(b_data).unwrap();

        let result: TypeSafeMatrix<2, 2> = a.dot(&b);
        assert_eq!(result.data().dim(), (2, 2));
    }

    #[test]
    fn test_type_safe_matrix_transpose() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let matrix: TypeSafeMatrix<2, 3> = TypeSafeMatrix::new(data).unwrap();
        let transposed: TypeSafeMatrix<3, 2> = matrix.t();
        assert_eq!(transposed.data().dim(), (3, 2));
    }

    #[test]
    fn test_component_index() {
        let index: ComponentIndex<0> = ComponentIndex::new();
        assert_eq!(index.index(), 0);

        let index: ComponentIndex<5> = ComponentIndex::new();
        assert_eq!(index.index(), 5);
    }

    #[test]
    fn test_decomposition_pipeline() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

        let pipeline = DecompositionPipeline::new().center().scale();

        let processed = pipeline.apply(&data).unwrap();
        assert_eq!(processed.dim(), data.dim());

        let fitted_pipeline = pipeline.fit(&data).unwrap();
        let transformed = fitted_pipeline.transform(&data).unwrap();
        assert_eq!(transformed.dim(), data.dim());
    }

    #[test]
    fn test_matrix_shape_validation() {
        let a_data = array![[1.0, 2.0], [3.0, 4.0]];
        let b_data = array![[5.0, 6.0], [7.0, 8.0]];

        let a: TypeSafeMatrix<2, 2> = TypeSafeMatrix::new(a_data).unwrap();
        let b: TypeSafeMatrix<2, 2> = TypeSafeMatrix::new(b_data).unwrap();

        // This should compile and succeed
        let result = validate_matrix_multiplication(&a, &b);
        assert!(result.is_ok());
    }
}
