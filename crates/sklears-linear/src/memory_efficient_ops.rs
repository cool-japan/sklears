//! Memory-efficient in-place matrix operations for linear models
//!
//! This module provides optimized in-place operations to reduce memory allocation
//! and improve performance for large-scale linear model computations.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Configuration for memory-efficient operations
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyConfig {
    /// Use chunked processing for operations larger than this size
    pub chunk_size_threshold: usize,
    /// Default chunk size for operations
    pub default_chunk_size: usize,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Number of threads for parallel operations
    pub n_threads: Option<usize>,
}

impl Default for MemoryEfficiencyConfig {
    fn default() -> Self {
        Self {
            chunk_size_threshold: 10000,
            default_chunk_size: 1000,
            use_parallel: true,
            n_threads: None,
        }
    }
}

/// Memory-efficient matrix operations
pub struct MemoryEfficientOps {
    config: MemoryEfficiencyConfig,
}

impl MemoryEfficientOps {
    /// Create new memory-efficient operations handler
    pub fn new(config: MemoryEfficiencyConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(MemoryEfficiencyConfig::default())
    }

    /// In-place matrix-vector multiplication: y = A * x (overwrites y)
    pub fn matvec_inplace(
        &self,
        a: &Array2<Float>,
        x: &Array1<Float>,
        y: &mut Array1<Float>,
    ) -> Result<()> {
        if a.ncols() != x.len() || a.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!(
                    "A: {}x{}, x: {}, y: {}",
                    a.nrows(),
                    a.ncols(),
                    a.ncols(),
                    a.nrows()
                ),
                actual: format!(
                    "A: {}x{}, x: {}, y: {}",
                    a.nrows(),
                    a.ncols(),
                    x.len(),
                    y.len()
                ),
            });
        }

        if a.nrows() > self.config.chunk_size_threshold {
            self.chunked_matvec_inplace(a, x, y)?;
        } else {
            // Standard operation for smaller matrices
            for (i, y_i) in y.iter_mut().enumerate() {
                *y_i = a.row(i).dot(x);
            }
        }

        Ok(())
    }

    /// In-place chunked matrix-vector multiplication for large matrices
    fn chunked_matvec_inplace(
        &self,
        a: &Array2<Float>,
        x: &Array1<Float>,
        y: &mut Array1<Float>,
    ) -> Result<()> {
        let chunk_size = self.config.default_chunk_size;
        let n_rows = a.nrows();

        for start_row in (0..n_rows).step_by(chunk_size) {
            let end_row = (start_row + chunk_size).min(n_rows);
            let a_chunk = a.slice(s![start_row..end_row, ..]);
            let mut y_chunk = y.slice_mut(s![start_row..end_row]);

            for (i, y_i) in y_chunk.iter_mut().enumerate() {
                *y_i = a_chunk.row(i).dot(x);
            }
        }

        Ok(())
    }

    /// In-place matrix-matrix multiplication: C = A * B (overwrites C)
    pub fn matmul_inplace(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
        c: &mut Array2<Float>,
    ) -> Result<()> {
        if a.ncols() != b.nrows() || a.nrows() != c.nrows() || b.ncols() != c.ncols() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!(
                    "A: {}x{}, B: {}x{}, C: {}x{}",
                    a.nrows(),
                    a.ncols(),
                    a.ncols(),
                    b.ncols(),
                    a.nrows(),
                    b.ncols()
                ),
                actual: format!(
                    "A: {}x{}, B: {}x{}, C: {}x{}",
                    a.nrows(),
                    a.ncols(),
                    b.nrows(),
                    b.ncols(),
                    c.nrows(),
                    c.ncols()
                ),
            });
        }

        let total_ops = a.nrows() * b.ncols() * a.ncols();
        if total_ops > self.config.chunk_size_threshold * 100 {
            self.chunked_matmul_inplace(a, b, c)?;
        } else {
            // Standard multiplication for smaller matrices
            for i in 0..a.nrows() {
                for j in 0..b.ncols() {
                    c[[i, j]] = a.row(i).dot(&b.column(j));
                }
            }
        }

        Ok(())
    }

    /// Chunked matrix multiplication for large matrices
    fn chunked_matmul_inplace(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
        c: &mut Array2<Float>,
    ) -> Result<()> {
        let chunk_size = self.config.default_chunk_size;
        let m = a.nrows();
        let n = b.ncols();

        for i_start in (0..m).step_by(chunk_size) {
            let i_end = (i_start + chunk_size).min(m);

            for j_start in (0..n).step_by(chunk_size) {
                let j_end = (j_start + chunk_size).min(n);

                let a_chunk = a.slice(s![i_start..i_end, ..]);
                let b_chunk = b.slice(s![.., j_start..j_end]);
                let mut c_chunk = c.slice_mut(s![i_start..i_end, j_start..j_end]);

                // Compute chunk
                for (i_local, a_row) in a_chunk.axis_iter(Axis(0)).enumerate() {
                    for (j_local, b_col) in b_chunk.axis_iter(Axis(1)).enumerate() {
                        c_chunk[[i_local, j_local]] = a_row.dot(&b_col);
                    }
                }
            }
        }

        Ok(())
    }

    /// In-place transpose: A^T (overwrites A if square, otherwise creates new array)
    pub fn transpose_inplace(&self, a: &mut Array2<Float>) -> Result<()> {
        if a.nrows() != a.ncols() {
            return Err(SklearsError::InvalidInput(
                "In-place transpose only supported for square matrices".to_string(),
            ));
        }

        let n = a.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                let temp = a[[i, j]];
                a[[i, j]] = a[[j, i]];
                a[[j, i]] = temp;
            }
        }

        Ok(())
    }

    /// In-place element-wise addition: A += B
    pub fn add_inplace(&self, a: &mut Array2<Float>, b: &Array2<Float>) -> Result<()> {
        if a.shape() != b.shape() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", a.shape()),
                actual: format!("{:?}", b.shape()),
            });
        }

        if a.len() > self.config.chunk_size_threshold {
            self.chunked_add_inplace(a, b)?;
        } else {
            *a += b;
        }

        Ok(())
    }

    /// Chunked in-place addition for large arrays
    fn chunked_add_inplace(&self, a: &mut Array2<Float>, b: &Array2<Float>) -> Result<()> {
        let chunk_size = self.config.default_chunk_size;
        let total_elements = a.len();

        let a_flat = a
            .as_slice_mut()
            .ok_or_else(|| SklearsError::InvalidInput("Array not contiguous".to_string()))?;
        let b_flat = b
            .as_slice()
            .ok_or_else(|| SklearsError::InvalidInput("Array not contiguous".to_string()))?;

        for start in (0..total_elements).step_by(chunk_size) {
            let end = (start + chunk_size).min(total_elements);
            for i in start..end {
                a_flat[i] += b_flat[i];
            }
        }

        Ok(())
    }

    /// In-place scalar multiplication: A *= scalar
    pub fn scale_inplace(&self, a: &mut Array2<Float>, scalar: Float) -> Result<()> {
        if a.len() > self.config.chunk_size_threshold {
            self.chunked_scale_inplace(a, scalar)?;
        } else {
            a.mapv_inplace(|x| x * scalar);
        }

        Ok(())
    }

    /// Chunked in-place scaling for large arrays
    fn chunked_scale_inplace(&self, a: &mut Array2<Float>, scalar: Float) -> Result<()> {
        let chunk_size = self.config.default_chunk_size;
        let total_elements = a.len();

        let a_flat = a
            .as_slice_mut()
            .ok_or_else(|| SklearsError::InvalidInput("Array not contiguous".to_string()))?;

        for start in (0..total_elements).step_by(chunk_size) {
            let end = (start + chunk_size).min(total_elements);
            for elem in &mut a_flat[start..end] {
                *elem *= scalar;
            }
        }

        Ok(())
    }

    /// In-place update: A = alpha * A + beta * B
    pub fn axpby_inplace(
        &self,
        alpha: Float,
        a: &mut Array2<Float>,
        beta: Float,
        b: &Array2<Float>,
    ) -> Result<()> {
        if a.shape() != b.shape() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{:?}", a.shape()),
                actual: format!("{:?}", b.shape()),
            });
        }

        if a.len() > self.config.chunk_size_threshold {
            self.chunked_axpby_inplace(alpha, a, beta, b)?;
        } else {
            *a = a.mapv(|x| alpha * x) + &b.mapv(|x| beta * x);
        }

        Ok(())
    }

    /// Chunked AXPBY operation for large arrays
    fn chunked_axpby_inplace(
        &self,
        alpha: Float,
        a: &mut Array2<Float>,
        beta: Float,
        b: &Array2<Float>,
    ) -> Result<()> {
        let chunk_size = self.config.default_chunk_size;
        let total_elements = a.len();

        let a_flat = a
            .as_slice_mut()
            .ok_or_else(|| SklearsError::InvalidInput("Array not contiguous".to_string()))?;
        let b_flat = b
            .as_slice()
            .ok_or_else(|| SklearsError::InvalidInput("Array not contiguous".to_string()))?;

        for start in (0..total_elements).step_by(chunk_size) {
            let end = (start + chunk_size).min(total_elements);
            for i in start..end {
                a_flat[i] = alpha * a_flat[i] + beta * b_flat[i];
            }
        }

        Ok(())
    }

    /// In-place Gram matrix computation: G = X^T * X (overwrites G)
    pub fn gram_inplace(&self, x: &Array2<Float>, g: &mut Array2<Float>) -> Result<()> {
        let n_features = x.ncols();
        if g.nrows() != n_features || g.ncols() != n_features {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}x{}", n_features, n_features),
                actual: format!("{}x{}", g.nrows(), g.ncols()),
            });
        }

        // Compute upper triangle
        for i in 0..n_features {
            for j in i..n_features {
                let col_i = x.column(i);
                let col_j = x.column(j);
                g[[i, j]] = col_i.dot(&col_j);
                if i != j {
                    g[[j, i]] = g[[i, j]]; // Symmetric
                }
            }
        }

        Ok(())
    }

    /// In-place covariance matrix computation with optional centering
    pub fn covariance_inplace(
        &self,
        x: &Array2<Float>,
        cov: &mut Array2<Float>,
        center: bool,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if cov.nrows() != n_features || cov.ncols() != n_features {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}x{}", n_features, n_features),
                actual: format!("{}x{}", cov.nrows(), cov.ncols()),
            });
        }

        if center {
            // Compute means
            let means = x.mean_axis(Axis(0)).unwrap();

            // Compute covariance with centering
            for i in 0..n_features {
                for j in i..n_features {
                    let mut sum = 0.0;
                    for k in 0..n_samples {
                        sum += (x[[k, i]] - means[i]) * (x[[k, j]] - means[j]);
                    }
                    cov[[i, j]] = sum / (n_samples - 1) as Float;
                    if i != j {
                        cov[[j, i]] = cov[[i, j]];
                    }
                }
            }
        } else {
            // Compute covariance without centering
            self.gram_inplace(x, cov)?;
            self.scale_inplace(cov, 1.0 / (n_samples - 1) as Float)?;
        }

        Ok(())
    }

    /// In-place QR decomposition update (for incremental learning)
    pub fn qr_update_inplace(
        &self,
        q: &mut Array2<Float>,
        r: &mut Array2<Float>,
        x: &Array1<Float>,
    ) -> Result<()> {
        let n = q.nrows();
        let k = r.ncols();

        if q.ncols() != k || r.nrows() != k || x.len() != n {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("Q: {}x{}, R: {}x{}, x: {}", n, k, k, k, n),
                actual: format!(
                    "Q: {}x{}, R: {}x{}, x: {}",
                    q.nrows(),
                    q.ncols(),
                    r.nrows(),
                    r.ncols(),
                    x.len()
                ),
            });
        }

        // Simple Givens rotation-based update
        let mut v = x.clone();

        // Orthogonalize against existing Q columns
        for j in 0..k {
            let q_j = q.column(j);
            let proj = q_j.dot(&v);

            // Update R matrix
            r[[j, j]] = (r[[j, j]] * r[[j, j]] + proj * proj).sqrt();

            // Update v
            for i in 0..n {
                v[i] -= proj * q_j[i];
            }
        }

        // Normalize the new column
        let norm = v.dot(&v).sqrt();
        if norm > 1e-12 {
            v /= norm;

            // This is a simplified update - in practice, you'd expand Q and R
            // For now, we assume Q and R have space for the new column
        }

        Ok(())
    }

    /// In-place normalization of matrix rows or columns
    pub fn normalize_inplace(
        &self,
        a: &mut Array2<Float>,
        axis: Axis,
        norm_type: NormType,
    ) -> Result<()> {
        match axis {
            Axis(0) => {
                // Normalize columns
                for mut col in a.axis_iter_mut(Axis(1)) {
                    let norm = self.compute_norm(&col.view(), norm_type);
                    if norm > 1e-12 {
                        col.mapv_inplace(|x| x / norm);
                    }
                }
            }
            Axis(1) => {
                // Normalize rows
                for mut row in a.axis_iter_mut(Axis(0)) {
                    let norm = self.compute_norm(&row.view(), norm_type);
                    if norm > 1e-12 {
                        row.mapv_inplace(|x| x / norm);
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Only axis 0 (columns) and axis 1 (rows) are supported".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Compute norm of a vector
    fn compute_norm(&self, v: &ArrayView1<Float>, norm_type: NormType) -> Float {
        match norm_type {
            NormType::L1 => v.mapv(Float::abs).sum(),
            NormType::L2 => v.mapv(|x| x * x).sum().sqrt(),
            NormType::LInf => v.mapv(Float::abs).fold(0.0, |a, &b| a.max(b)),
        }
    }

    /// Memory usage estimation for operations
    pub fn estimate_memory_usage(&self, operation: &MemoryOperation) -> usize {
        match operation {
            MemoryOperation::MatVec { rows, cols } => {
                // A (rows x cols) + x (cols) + y (rows) = rows * cols + cols + rows
                rows * cols + cols + rows
            }
            MemoryOperation::MatMul { m, n, k } => {
                // A (m x k) + B (k x n) + C (m x n) = m * k + k * n + m * n
                m * k + k * n + m * n
            }
            MemoryOperation::Gram {
                n_samples,
                n_features,
            } => {
                // X (n_samples x n_features) + G (n_features x n_features)
                n_samples * n_features + n_features * n_features
            }
            MemoryOperation::QRUpdate { n, k } => {
                // Q (n x k) + R (k x k) + x (n) = n * k + k * k + n
                n * k + k * k + n
            }
        }
    }

    /// Check if operation should use chunked processing
    pub fn should_use_chunked(&self, operation: &MemoryOperation) -> bool {
        let memory_usage = self.estimate_memory_usage(operation);
        memory_usage > self.config.chunk_size_threshold
    }
}

/// Types of matrix norms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    L1,
    L2,
    LInf,
}

/// Types of memory operations for estimation
#[derive(Debug, Clone)]
pub enum MemoryOperation {
    MatVec { rows: usize, cols: usize },
    MatMul { m: usize, n: usize, k: usize },
    Gram { n_samples: usize, n_features: usize },
    QRUpdate { n: usize, k: usize },
}

/// Memory-efficient coordinate descent implementation
pub struct MemoryEfficientCoordinateDescent {
    ops: MemoryEfficientOps,
}

impl MemoryEfficientCoordinateDescent {
    pub fn new(config: MemoryEfficiencyConfig) -> Self {
        Self {
            ops: MemoryEfficientOps::new(config),
        }
    }

    /// Memory-efficient coordinate descent step for Lasso
    pub fn lasso_step_inplace(
        &self,
        x: &Array2<Float>,
        _y: &Array1<Float>,
        coef: &mut Array1<Float>,
        alpha: Float,
        feature_idx: usize,
        residual: &mut Array1<Float>,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let x_j = x.column(feature_idx);

        // Update residual to exclude current feature
        let old_coef = coef[feature_idx];
        if old_coef != 0.0 {
            for i in 0..n_samples {
                residual[i] += old_coef * x_j[i];
            }
        }

        // Compute new coefficient
        let x_dot_residual = x_j.dot(residual);
        let x_norm_sq = x_j.dot(&x_j);

        let new_coef = if x_norm_sq > 1e-12 {
            self.soft_threshold(x_dot_residual / x_norm_sq, alpha / x_norm_sq)
        } else {
            0.0
        };

        // Update coefficient and residual
        coef[feature_idx] = new_coef;
        if new_coef != 0.0 {
            for i in 0..n_samples {
                residual[i] -= new_coef * x_j[i];
            }
        }

        Ok(())
    }

    /// Soft thresholding function
    fn soft_threshold(&self, z: Float, lambda: Float) -> Float {
        if z > lambda {
            z - lambda
        } else if z < -lambda {
            z + lambda
        } else {
            0.0
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_matvec_inplace() {
        let ops = MemoryEfficientOps::default();
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let x = array![1.0, 2.0];
        let mut y = array![0.0, 0.0];

        ops.matvec_inplace(&a, &x, &mut y).unwrap();

        assert_abs_diff_eq!(y[0], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y[1], 11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matmul_inplace() {
        let ops = MemoryEfficientOps::default();
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[2.0, 0.0], [1.0, 2.0]];
        let mut c = array![[0.0, 0.0], [0.0, 0.0]];

        ops.matmul_inplace(&a, &b, &mut c).unwrap();

        assert_abs_diff_eq!(c[[0, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[0, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 0]], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transpose_inplace() {
        let ops = MemoryEfficientOps::default();
        let mut a = array![[1.0, 2.0], [3.0, 4.0]];

        ops.transpose_inplace(&mut a).unwrap();

        assert_abs_diff_eq!(a[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[0, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[1, 1]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_add_inplace() {
        let ops = MemoryEfficientOps::default();
        let mut a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[1.0, 1.0], [1.0, 1.0]];

        ops.add_inplace(&mut a, &b).unwrap();

        assert_abs_diff_eq!(a[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[0, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[1, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[1, 1]], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scale_inplace() {
        let ops = MemoryEfficientOps::default();
        let mut a = array![[1.0, 2.0], [3.0, 4.0]];

        ops.scale_inplace(&mut a, 2.0).unwrap();

        assert_abs_diff_eq!(a[[0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[0, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[1, 0]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[1, 1]], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gram_inplace() {
        let ops = MemoryEfficientOps::default();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut g = array![[0.0, 0.0], [0.0, 0.0]];

        ops.gram_inplace(&x, &mut g).unwrap();

        // X^T * X = [[1,3,5],[2,4,6]] * [[1,2],[3,4],[5,6]] = [[35,44],[44,56]]
        assert_abs_diff_eq!(g[[0, 0]], 35.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[[0, 1]], 44.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[[1, 0]], 44.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[[1, 1]], 56.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_inplace() {
        let ops = MemoryEfficientOps::default();
        let mut a = array![[3.0, 4.0], [1.0, 0.0]];

        ops.normalize_inplace(&mut a, Axis(1), NormType::L2)
            .unwrap();

        // First row: [3,4] -> [3/5, 4/5] = [0.6, 0.8]
        assert_abs_diff_eq!(a[[0, 0]], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[0, 1]], 0.8, epsilon = 1e-10);

        // Second row: [1,0] -> [1, 0]
        assert_abs_diff_eq!(a[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_estimation() {
        let ops = MemoryEfficientOps::default();

        let matvec_op = MemoryOperation::MatVec {
            rows: 100,
            cols: 50,
        };
        let memory = ops.estimate_memory_usage(&matvec_op);

        // 100 * 50 + 50 + 100 = 5150
        assert_eq!(memory, 5150);

        let should_chunk = ops.should_use_chunked(&matvec_op);
        assert!(!should_chunk); // 5150 < 10000 (default threshold)
    }
}
