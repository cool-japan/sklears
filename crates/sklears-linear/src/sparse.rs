//! Sparse matrix support for linear models
//!
//! This module provides efficient sparse matrix operations for large-scale
//! linear models using the sprs crate. It includes:
//! - Sparse matrix traits and wrappers
//! - Conversion utilities between dense and sparse formats
//! - Sparse implementations of key algorithms (coordinate descent, etc.)
//! - Memory-efficient operations for high-dimensional sparse data

use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "sparse")]
use scirs2_core::numeric::{One, SparseElement, Zero};
#[cfg(feature = "sparse")]
use scirs2_sparse::CsrMatrix;
use sklears_core::{
    error::{Result, SklearsError},
    types::FloatBounds,
};

/// Either type for sparse matrix operations
#[derive(Debug, Clone)]
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

/// Configuration for sparse matrix operations
#[derive(Debug, Clone)]
pub struct SparseConfig {
    pub sparsity_threshold: f64,
    pub min_sparsity_ratio: f64,
    pub max_dense_memory_ratio: f64,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 1e-8,
            min_sparsity_ratio: 0.1,
            max_dense_memory_ratio: 0.5,
        }
    }
}

/// Analysis of matrix sparsity patterns
#[derive(Debug, Clone)]
pub struct SparsityAnalysis {
    /// Fraction of non-zero elements (0.0 = fully sparse, 1.0 = fully dense)
    pub sparsity_ratio: f64,
    /// Estimated memory savings ratio when using sparse over dense storage
    pub memory_savings: f64,
    /// Recommended storage format based on the analysis
    pub recommended_format: String,
    /// Total number of elements in the matrix
    pub total_elements: usize,
    /// Number of non-zero elements (above threshold)
    pub nonzero_count: usize,
    /// Number of rows in the analysed matrix
    pub n_rows: usize,
    /// Number of columns in the analysed matrix
    pub n_cols: usize,
}

impl SparsityAnalysis {
    /// Estimated ratio of memory saved by using sparse format over dense format.
    ///
    /// Returns a value in [0.0, 1.0]: 0.0 means no savings, 1.0 means all savings.
    /// Formula: `1.0 - (nnz / total_elements)` (rough estimate ignoring index overhead).
    pub fn memory_savings_ratio(&self, _nrows: usize, _ncols: usize) -> f64 {
        self.memory_savings
    }
}

/// Sparse matrix operations trait
#[cfg(feature = "sparse")]
pub trait SparseMatrix<T: FloatBounds> {
    /// Number of rows
    fn nrows(&self) -> usize;

    /// Number of columns
    fn ncols(&self) -> usize;

    /// Number of non-zero elements
    fn nnz(&self) -> usize;

    /// Sparsity ratio (nnz / (nrows * ncols))
    fn sparsity(&self) -> f64 {
        let total_elements = self.nrows() as f64 * self.ncols() as f64;
        if total_elements > 0.0 {
            self.nnz() as f64 / total_elements
        } else {
            0.0
        }
    }

    /// Matrix-vector multiplication: self * x
    fn matvec(&self, x: &Array1<T>) -> Result<Array1<T>>;

    /// Transposed matrix-vector multiplication: self^T * x
    fn transp_matvec(&self, x: &Array1<T>) -> Result<Array1<T>>;

    /// Get a specific row as a sparse vector
    fn row(&self, i: usize) -> Result<CsrMatrix<T>>;

    /// Get a specific column as a sparse vector
    fn col(&self, j: usize) -> Result<CsrMatrix<T>>;

    /// Convert to dense matrix (use with caution for large matrices)
    fn to_dense(&self) -> Result<Array2<T>>;
}

/// Wrapper for CSR sparse matrices
#[cfg(feature = "sparse")]
#[derive(Clone)]
pub struct SparseMatrixCSR<T: FloatBounds> {
    inner: CsrMatrix<T>,
}

#[cfg(feature = "sparse")]
impl<T: FloatBounds + SparseElement> SparseMatrixCSR<T> {
    /// Create a new sparse matrix from CSR format
    pub fn new(inner: CsrMatrix<T>) -> Self {
        Self { inner }
    }

    /// Create from triplet format (row indices, col indices, values)
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(usize, usize, T)],
    ) -> Result<Self> {
        let csmat = CsrMatrix::try_from_triplets(nrows, ncols, triplets)
            .map_err(|e| SklearsError::Other(format!("Failed to create sparse matrix: {:?}", e)))?;
        Ok(Self::new(csmat))
    }

    /// Get inner CSR matrix
    pub fn inner(&self) -> &CsrMatrix<T> {
        &self.inner
    }

    /// Create from dense matrix with sparsity threshold
    pub fn from_dense(dense: &Array2<T>, threshold: T) -> Self {
        let (nrows, ncols) = dense.dim();
        let mut triplets = Vec::new();

        // Iterate through dense matrix and collect non-zero elements
        for i in 0..nrows {
            for j in 0..ncols {
                let val = dense[[i, j]];
                // Check if value is above threshold (non-zero)
                if val.abs() > threshold {
                    triplets.push((i, j, val));
                }
            }
        }

        // Use try_from_triplets to construct CSR matrix
        let csmat = CsrMatrix::try_from_triplets(nrows, ncols, &triplets).unwrap_or_else(|_| {
            // Create empty matrix as fallback
            CsrMatrix::try_from_triplets(nrows, ncols, &[]).expect("operation should succeed")
        });

        Self::new(csmat)
    }
}

#[cfg(feature = "sparse")]
impl<T: FloatBounds + SparseElement> SparseMatrix<T> for SparseMatrixCSR<T> {
    fn nrows(&self) -> usize {
        self.inner.rows()
    }

    fn ncols(&self) -> usize {
        self.inner.cols()
    }

    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    fn matvec(&self, x: &Array1<T>) -> Result<Array1<T>> {
        if x.len() != self.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Vector length {} does not match matrix columns {}",
                x.len(),
                self.ncols()
            )));
        }

        let mut result = Array1::zeros(self.nrows());

        // Standard CSR matrix-vector multiplication: y = A * x
        for row_idx in 0..self.nrows() {
            let row_start = self.inner.indptr[row_idx];
            let row_end = self.inner.indptr[row_idx + 1];

            let mut sum = T::default();
            for j in row_start..row_end {
                let col_idx = self.inner.indices[j];
                let val = self.inner.data[j];
                sum += val * x[col_idx];
            }
            result[row_idx] = sum;
        }

        Ok(result)
    }

    fn transp_matvec(&self, x: &Array1<T>) -> Result<Array1<T>> {
        if x.len() != self.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Vector length {} does not match matrix rows {}",
                x.len(),
                self.nrows()
            )));
        }

        let mut result = Array1::zeros(self.ncols());

        // Transposed CSR matrix-vector multiplication: y = A^T * x
        // For each row in A, scatter x[row] * A[row,:] into result
        for row_idx in 0..self.nrows() {
            let row_start = self.inner.indptr[row_idx];
            let row_end = self.inner.indptr[row_idx + 1];

            let x_val = x[row_idx];
            for j in row_start..row_end {
                let col_idx = self.inner.indices[j];
                let val = self.inner.data[j];
                result[col_idx] += val * x_val;
            }
        }

        Ok(result)
    }

    fn row(&self, i: usize) -> Result<CsrMatrix<T>> {
        if i >= self.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Row index {} out of bounds for matrix with {} rows",
                i,
                self.nrows()
            )));
        }

        // For now, return a simple error - implementing row extraction requires more complex logic
        Err(SklearsError::NotImplemented(
            "Row extraction not yet implemented".to_string(),
        ))
    }

    fn col(&self, j: usize) -> Result<CsrMatrix<T>> {
        if j >= self.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Column index {} out of bounds for matrix with {} columns",
                j,
                self.ncols()
            )));
        }

        // For now, return a simple error - implementing column extraction requires more complex logic
        Err(SklearsError::NotImplemented(
            "Column extraction not yet implemented".to_string(),
        ))
    }

    fn to_dense(&self) -> Result<Array2<T>> {
        let mut dense = Array2::zeros((self.nrows(), self.ncols()));

        // Convert CSR sparse matrix to dense format
        for row_idx in 0..self.nrows() {
            let row_start = self.inner.indptr[row_idx];
            let row_end = self.inner.indptr[row_idx + 1];

            for j in row_start..row_end {
                let col_idx = self.inner.indices[j];
                let val = self.inner.data[j];
                dense[[row_idx, col_idx]] = val;
            }
        }

        Ok(dense)
    }
}

/// Coordinate descent solver for sparse matrices
#[cfg(feature = "sparse")]
pub struct SparseCoordinateDescentSolver<T> {
    pub alpha: T,
    pub l1_ratio: T,
    pub max_iter: usize,
    pub tol: T,
    pub cyclic: bool,
    pub sparse_config: SparseConfig,
}

#[cfg(feature = "sparse")]
impl<T: FloatBounds + SparseElement> SparseCoordinateDescentSolver<T> {
    pub fn new(alpha: T, l1_ratio: T, max_iter: usize, tol: T) -> Self {
        Self {
            alpha,
            l1_ratio,
            max_iter,
            tol,
            cyclic: true,
            sparse_config: SparseConfig::default(),
        }
    }

    /// Scalar soft-thresholding operator: sign(z) * max(|z| - threshold, 0)
    fn soft_threshold(z: T, threshold: T) -> T {
        let abs_z = z.abs();
        if abs_z <= threshold {
            <T as Zero>::zero()
        } else if z > <T as Zero>::zero() {
            abs_z - threshold
        } else {
            -(abs_z - threshold)
        }
    }

    /// Build a column-oriented index from CSR for efficient column access during
    /// coordinate descent.  Returns `(col_indptr, col_row_indices, col_values)`.
    ///
    /// After this call, column `j` has row indices `col_row_indices[col_indptr[j]..col_indptr[j+1]]`
    /// and values `col_values[col_indptr[j]..col_indptr[j+1]]`.
    fn build_column_index(x: &SparseMatrixCSR<T>) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        let ncols = x.ncols();
        let nnz = x.inner().nnz();

        // Count nnz per column
        let mut col_counts = vec![0usize; ncols];
        for &c in &x.inner().indices {
            col_counts[c] += 1;
        }

        // Build exclusive prefix-sum for column start pointers
        let mut col_indptr = vec![0usize; ncols + 1];
        for j in 0..ncols {
            col_indptr[j + 1] = col_indptr[j] + col_counts[j];
        }

        // Scatter values into column-oriented storage
        let mut col_row_indices = vec![0usize; nnz];
        let mut col_values = vec![<T as Zero>::zero(); nnz];
        let mut fill = vec![0usize; ncols]; // per-column fill cursor

        for row in 0..x.nrows() {
            let row_start = x.inner().indptr[row];
            let row_end = x.inner().indptr[row + 1];
            for k in row_start..row_end {
                let col = x.inner().indices[k];
                let val = x.inner().data[k];
                let dest = col_indptr[col] + fill[col];
                col_row_indices[dest] = row;
                col_values[dest] = val;
                fill[col] += 1;
            }
        }

        (col_indptr, col_row_indices, col_values)
    }

    /// Compute column norms squared: `||X[:,j]||^2` for each column j.
    fn column_norms_sq(col_indptr: &[usize], col_values: &[T], ncols: usize) -> Vec<T> {
        let mut norms_sq = vec![<T as Zero>::zero(); ncols];
        for j in 0..ncols {
            norms_sq[j] = col_values[col_indptr[j]..col_indptr[j + 1]]
                .iter()
                .fold(<T as Zero>::zero(), |acc, &v| acc + v * v);
        }
        norms_sq
    }

    /// Compute `X[:,j]^T * r` — the inner product of column j with the residual vector.
    fn col_dot_residual(
        j: usize,
        residual: &Array1<T>,
        col_indptr: &[usize],
        col_row_indices: &[usize],
        col_values: &[T],
    ) -> T {
        let mut dot = <T as Zero>::zero();
        for k in col_indptr[j]..col_indptr[j + 1] {
            let row = col_row_indices[k];
            dot += col_values[k] * residual[row];
        }
        dot
    }

    /// Update the residual in place when coordinate j changes from `beta_old` to `beta_new`:
    /// `r -= X[:,j] * (beta_new - beta_old)`
    fn update_residual(
        j: usize,
        delta: T,
        residual: &mut Array1<T>,
        col_indptr: &[usize],
        col_row_indices: &[usize],
        col_values: &[T],
    ) {
        for k in col_indptr[j]..col_indptr[j + 1] {
            let row = col_row_indices[k];
            residual[row] -= col_values[k] * delta;
        }
    }

    /// Sparse LASSO via cyclic coordinate descent.
    ///
    /// Minimises `(1/2n)||y - Xβ||^2 + α||β||_1` using coordinate descent.
    ///
    /// For each coordinate `j` the optimal update is the soft-threshold:
    /// ```text
    /// β_j ← S(X[:,j]ᵀr / ||X[:,j]||², αn / ||X[:,j]||²)
    /// ```
    /// where `r = y - Xβ + X[:,j]β_j` is the partial residual and `n` is the
    /// number of samples.
    pub fn fit_lasso(&self, x: &SparseMatrixCSR<T>, y: &Array1<T>) -> Result<Array1<T>> {
        let n = x.nrows();
        let p = x.ncols();

        if n == 0 || p == 0 {
            return Err(SklearsError::InvalidInput(
                "fit_lasso: matrix must have at least one row and one column".to_string(),
            ));
        }
        if y.len() != n {
            return Err(SklearsError::InvalidInput(format!(
                "fit_lasso: y length {} does not match matrix rows {}",
                y.len(),
                n
            )));
        }

        let n_t = T::from(n)
            .ok_or_else(|| SklearsError::InvalidInput("Cannot convert n to T".to_string()))?;

        // Build column-oriented view (O(nnz))
        let (col_indptr, col_row_indices, col_values) = Self::build_column_index(x);
        let norms_sq = Self::column_norms_sq(&col_indptr, &col_values, p);

        let mut beta = Array1::zeros(p);
        let mut residual = y.to_owned(); // r = y - X*beta; starts at y since beta=0

        for _iter in 0..self.max_iter {
            let mut max_change = <T as Zero>::zero();

            for j in 0..p {
                let norm_sq_j = norms_sq[j];
                if norm_sq_j <= <T as Zero>::zero() {
                    // Zero column — skip
                    continue;
                }

                // Partial residual: r_partial = r + X[:,j]*beta[j]
                // Instead of computing a new vector, we add the contribution back to residual
                // temporarily through the dot product.
                // rho = X[:,j]^T * (r + X[:,j]*beta[j]) = X[:,j]^T*r + norm_sq_j*beta[j]
                let xtj_r = Self::col_dot_residual(
                    j,
                    &residual,
                    &col_indptr,
                    &col_row_indices,
                    &col_values,
                );
                let rho = xtj_r + norm_sq_j * beta[j];

                // Threshold parameter: alpha * n / ||X[:,j]||^2
                let threshold = self.alpha * n_t / norm_sq_j;

                // Soft-threshold update
                let beta_new = Self::soft_threshold(rho / norm_sq_j, threshold);
                let delta = beta_new - beta[j];

                if delta.abs() > <T as Zero>::zero() {
                    Self::update_residual(
                        j,
                        delta,
                        &mut residual,
                        &col_indptr,
                        &col_row_indices,
                        &col_values,
                    );
                    let abs_delta = delta.abs();
                    if abs_delta > max_change {
                        max_change = abs_delta;
                    }
                    beta[j] = beta_new;
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        Ok(beta)
    }

    /// Sparse Ridge regression via Conjugate Gradient (CG).
    ///
    /// Solves the normal equations `(XᵀX + λI)β = Xᵀy` without materialising
    /// `XᵀX`, using only `SparseMatrixCSR::matvec` and `transp_matvec`.
    ///
    /// The operator application `A·p = Xᵀ(X·p) + λ·p` is O(nnz) per CG step.
    pub fn fit_ridge(&self, x: &SparseMatrixCSR<T>, y: &Array1<T>) -> Result<Array1<T>> {
        let n = x.nrows();
        let p = x.ncols();

        if n == 0 || p == 0 {
            return Err(SklearsError::InvalidInput(
                "fit_ridge: matrix must have at least one row and one column".to_string(),
            ));
        }
        if y.len() != n {
            return Err(SklearsError::InvalidInput(format!(
                "fit_ridge: y length {} does not match matrix rows {}",
                y.len(),
                n
            )));
        }

        let lambda = self.alpha;

        // RHS: b = Xᵀ y
        let b = x.transp_matvec(y)?;

        // Closure: apply A = XᵀX + λI to a vector v
        let apply_a = |v: &Array1<T>| -> Result<Array1<T>> {
            let xv = x.matvec(v)?;
            let xt_xv = x.transp_matvec(&xv)?;
            let result = xt_xv + v.mapv(|vi| vi * lambda);
            Ok(result)
        };

        // CG initialisation
        let mut beta = Array1::zeros(p);
        let mut r = b.clone(); // r_0 = b - A*beta_0 = b  (beta_0 = 0)
        let mut p_vec = r.clone();

        let r_dot = |a: &Array1<T>| a.iter().fold(<T as Zero>::zero(), |acc, &x| acc + x * x);
        let mut rs_old = r_dot(&r);

        // Short-circuit if RHS is already zero
        if rs_old <= <T as Zero>::zero() {
            return Ok(beta);
        }

        for _iter in 0..self.max_iter {
            let ap = apply_a(&p_vec)?;
            let p_ap = p_vec
                .iter()
                .zip(ap.iter())
                .fold(<T as Zero>::zero(), |acc, (&pi, &api)| acc + pi * api);

            if p_ap <= <T as Zero>::zero() {
                // Curvature loss — system is numerically singular or ill-conditioned
                break;
            }

            let alpha_step = rs_old / p_ap;
            beta = beta + p_vec.mapv(|pj| pj * alpha_step);
            r = r - ap.mapv(|aj| aj * alpha_step);

            let rs_new = r_dot(&r);

            if rs_new < self.tol * self.tol {
                break;
            }

            let beta_cg = rs_new / rs_old;
            p_vec = r.clone() + p_vec.mapv(|pj| pj * beta_cg);
            rs_old = rs_new;
        }

        Ok(beta)
    }

    /// Sparse Elastic Net via cyclic coordinate descent.
    ///
    /// Minimises `(1/2n)||y - Xβ||^2 + α·l1·||β||_1 + (α/2)·(1-l1)·||β||^2`
    ///
    /// Coordinate-wise closed form:
    /// ```text
    /// β_j ← S(rho, α·l1·n / ||X[:,j]||²) / (1 + α·(1-l1)·n / ||X[:,j]||²)
    /// ```
    pub fn fit_elastic_net(&self, x: &SparseMatrixCSR<T>, y: &Array1<T>) -> Result<Array1<T>> {
        let n = x.nrows();
        let p = x.ncols();

        if n == 0 || p == 0 {
            return Err(SklearsError::InvalidInput(
                "fit_elastic_net: matrix must have at least one row and one column".to_string(),
            ));
        }
        if y.len() != n {
            return Err(SklearsError::InvalidInput(format!(
                "fit_elastic_net: y length {} does not match matrix rows {}",
                y.len(),
                n
            )));
        }

        let n_t = T::from(n)
            .ok_or_else(|| SklearsError::InvalidInput("Cannot convert n to T".to_string()))?;

        let (col_indptr, col_row_indices, col_values) = Self::build_column_index(x);
        let norms_sq = Self::column_norms_sq(&col_indptr, &col_values, p);

        let mut beta = Array1::zeros(p);
        let mut residual = y.to_owned();

        for _iter in 0..self.max_iter {
            let mut max_change = <T as Zero>::zero();

            for j in 0..p {
                let norm_sq_j = norms_sq[j];
                if norm_sq_j <= <T as Zero>::zero() {
                    continue;
                }

                let xtj_r = Self::col_dot_residual(
                    j,
                    &residual,
                    &col_indptr,
                    &col_row_indices,
                    &col_values,
                );
                let rho = xtj_r + norm_sq_j * beta[j];

                // L1 threshold: α * l1_ratio * n / ||X[:,j]||^2
                let l1_thresh = self.alpha * self.l1_ratio * n_t / norm_sq_j;
                // L2 shrinkage denominator factor: 1 + α * (1 - l1_ratio) * n / ||X[:,j]||^2
                let l2_factor = <T as One>::one()
                    + self.alpha * (<T as One>::one() - self.l1_ratio) * n_t / norm_sq_j;

                let beta_new = Self::soft_threshold(rho / norm_sq_j, l1_thresh) / l2_factor;
                let delta = beta_new - beta[j];

                if delta.abs() > <T as Zero>::zero() {
                    Self::update_residual(
                        j,
                        delta,
                        &mut residual,
                        &col_indptr,
                        &col_row_indices,
                        &col_values,
                    );
                    let abs_delta = delta.abs();
                    if abs_delta > max_change {
                        max_change = abs_delta;
                    }
                    beta[j] = beta_new;
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        Ok(beta)
    }

    /// Compute feature column sums `Σ_i X[i,j]` for each column `j`.
    /// Accumulates over non-zero entries only — O(nnz), never materialises X.
    fn column_sums(x: &SparseMatrixCSR<T>, n_t: T) -> Vec<T> {
        let p = x.ncols();
        let n = x.nrows();
        let mut sums = vec![<T as Zero>::zero(); p];
        for row in 0..n {
            let row_start = x.inner().indptr[row];
            let row_end = x.inner().indptr[row + 1];
            for k in row_start..row_end {
                let col = x.inner().indices[k];
                sums[col] += x.inner().data[k];
            }
        }
        // Divide by n to get column means (reuse buffer)
        for s in &mut sums {
            *s /= n_t;
        }
        sums
    }

    /// Sparse LASSO with optional intercept via **virtual centering**.
    ///
    /// When `fit_intercept = true`, the coordinate-descent loop operates on
    /// the *virtually centred* design `X̃[:,j] = X[:,j] - x̄_j·1_n` without
    /// ever materialising the dense centred matrix.  This preserves sparsity.
    ///
    /// Key invariants maintained inside the CD loop:
    ///
    /// - `residual = y_centred - X·β`  (standard CSR sparse update)
    /// - `sum_r = Σ_i residual[i]` (scalar running sum; starts at 0 and
    ///   stays 0 because y_centred is zero-mean and X has zero virtual
    ///   contribution to the mean — maintained as rolling scalar)
    ///
    /// Effective per-column quantities:
    /// ```text
    /// norm_eff_j = ||X[:,j]||² - n·x̄_j²            (centred norm squared)
    /// rho_j      = X[:,j]ᵀ·r - x̄_j·sum(r) + norm_eff_j·β_j
    /// ```
    /// Soft-threshold is applied to `rho_j / norm_eff_j` with threshold
    /// `α·n / norm_eff_j`.  Intercept recovery: `b₀ = ȳ - x̄ᵀβ`.
    pub fn solve_sparse_lasso(
        &self,
        x: &SparseMatrixCSR<T>,
        y: &Array1<T>,
        alpha: T,
        fit_intercept: bool,
    ) -> Result<(Array1<T>, T)> {
        let n = x.nrows();
        let p = x.ncols();

        if n == 0 || p == 0 {
            return Err(SklearsError::InvalidInput(
                "solve_sparse_lasso: empty matrix".to_string(),
            ));
        }
        if y.len() != n {
            return Err(SklearsError::InvalidInput(format!(
                "solve_sparse_lasso: y length {} does not match matrix rows {}",
                y.len(),
                n
            )));
        }

        let n_t = T::from(n)
            .ok_or_else(|| SklearsError::InvalidInput("Cannot convert n to T".to_string()))?;

        // --- centering setup (O(n + nnz)) --------------------------------
        let y_mean = if fit_intercept {
            y.iter().fold(<T as Zero>::zero(), |acc, &v| acc + v) / n_t
        } else {
            <T as Zero>::zero()
        };
        let x_means = if fit_intercept {
            Self::column_sums(x, n_t)
        } else {
            vec![<T as Zero>::zero(); p]
        };
        // y_centred = y - y_mean  (O(n) dense; y_centred has sum = 0)
        let y_centred = y.mapv(|v| v - y_mean);

        // --- column index & norms (O(nnz)) --------------------------------
        let (col_indptr, col_row_indices, col_values) = Self::build_column_index(x);
        let norms_sq = Self::column_norms_sq(&col_indptr, &col_values, p);

        // Effective centred norms: ||X̃[:,j]||² = ||X[:,j]||² - n·x̄_j²
        let eff_norms_sq: Vec<T> = norms_sq
            .iter()
            .zip(x_means.iter())
            .map(|(&ns, &xm)| ns - n_t * xm * xm)
            .collect();

        // --- CD main loop -------------------------------------------------
        let mut beta = vec![<T as Zero>::zero(); p];
        // residual = y_centred - X·β  (starts as y_centred since β = 0)
        let mut residual = y_centred.to_owned();
        // sum_r = Σ residual[i]; for zero-mean y_centred and zero-mean X this
        // drifts only due to floating point; we recompute it each sweep for
        // accuracy rather than tracking it incrementally.
        let mut sum_r: T = residual.iter().fold(<T as Zero>::zero(), |a, &v| a + v);

        for _iter in 0..self.max_iter {
            let mut max_change = <T as Zero>::zero();

            for j in 0..p {
                let eff_ns = eff_norms_sq[j];
                if eff_ns <= <T as Zero>::zero() {
                    continue;
                }

                // rho_j = X[:,j]ᵀ·r - x̄_j·Σr + eff_ns·β_j
                let xtj_r = Self::col_dot_residual(
                    j,
                    &residual,
                    &col_indptr,
                    &col_row_indices,
                    &col_values,
                );
                let rho = xtj_r - x_means[j] * sum_r + eff_ns * beta[j];

                let threshold = alpha * n_t / eff_ns;
                let beta_new = Self::soft_threshold(rho / eff_ns, threshold);
                let delta = beta_new - beta[j];

                if delta.abs() > <T as Zero>::zero() {
                    // Update sparse residual: r -= X[:,j] * delta
                    Self::update_residual(
                        j,
                        delta,
                        &mut residual,
                        &col_indptr,
                        &col_row_indices,
                        &col_values,
                    );
                    // Update running Σr: subtract delta * col_j_sum where
                    // col_j_sum = n_t * x_means[j]  (= Σ_i X[i,j])
                    sum_r -= delta * (n_t * x_means[j]);
                    beta[j] = beta_new;

                    let abs_delta = delta.abs();
                    if abs_delta > max_change {
                        max_change = abs_delta;
                    }
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        let beta_arr = Array1::from(beta);

        // Intercept: b₀ = ȳ - x̄ᵀβ
        let intercept = if fit_intercept {
            let xbar_dot_beta = x_means
                .iter()
                .zip(beta_arr.iter())
                .fold(<T as Zero>::zero(), |acc, (&xm, &bj)| acc + xm * bj);
            y_mean - xbar_dot_beta
        } else {
            <T as Zero>::zero()
        };

        Ok((beta_arr, intercept))
    }

    /// Sparse Elastic Net with optional intercept via **virtual centering**.
    ///
    /// Minimises `(1/2n)||y - Xβ||^2 + α·ρ·||β||_1 + (α/2)·(1-ρ)·||β||^2`
    /// using the virtually centred design `X̃[:,j] = X[:,j] - x̄_j·1_n` so
    /// that sparsity of the stored `X` is never broken.
    ///
    /// See `solve_sparse_lasso` for invariant details.
    pub fn solve_sparse_elastic_net(
        &self,
        x: &SparseMatrixCSR<T>,
        y: &Array1<T>,
        alpha: T,
        l1_ratio: T,
        fit_intercept: bool,
    ) -> Result<(Array1<T>, T)> {
        let n = x.nrows();
        let p = x.ncols();

        if n == 0 || p == 0 {
            return Err(SklearsError::InvalidInput(
                "solve_sparse_elastic_net: empty matrix".to_string(),
            ));
        }
        if y.len() != n {
            return Err(SklearsError::InvalidInput(format!(
                "solve_sparse_elastic_net: y length {} does not match matrix rows {}",
                y.len(),
                n
            )));
        }

        let n_t = T::from(n)
            .ok_or_else(|| SklearsError::InvalidInput("Cannot convert n to T".to_string()))?;

        let y_mean = if fit_intercept {
            y.iter().fold(<T as Zero>::zero(), |acc, &v| acc + v) / n_t
        } else {
            <T as Zero>::zero()
        };
        let x_means = if fit_intercept {
            Self::column_sums(x, n_t)
        } else {
            vec![<T as Zero>::zero(); p]
        };
        let y_centred = y.mapv(|v| v - y_mean);

        let (col_indptr, col_row_indices, col_values) = Self::build_column_index(x);
        let norms_sq = Self::column_norms_sq(&col_indptr, &col_values, p);
        let eff_norms_sq: Vec<T> = norms_sq
            .iter()
            .zip(x_means.iter())
            .map(|(&ns, &xm)| ns - n_t * xm * xm)
            .collect();

        let mut beta = vec![<T as Zero>::zero(); p];
        let mut residual = y_centred.to_owned();
        let mut sum_r: T = residual.iter().fold(<T as Zero>::zero(), |a, &v| a + v);

        for _iter in 0..self.max_iter {
            let mut max_change = <T as Zero>::zero();

            for j in 0..p {
                let eff_ns = eff_norms_sq[j];
                if eff_ns <= <T as Zero>::zero() {
                    continue;
                }

                let xtj_r = Self::col_dot_residual(
                    j,
                    &residual,
                    &col_indptr,
                    &col_row_indices,
                    &col_values,
                );
                let rho = xtj_r - x_means[j] * sum_r + eff_ns * beta[j];

                // L1 threshold and L2 denominator use eff_ns
                let l1_thresh = alpha * l1_ratio * n_t / eff_ns;
                let l2_factor =
                    <T as One>::one() + alpha * (<T as One>::one() - l1_ratio) * n_t / eff_ns;
                let beta_new = Self::soft_threshold(rho / eff_ns, l1_thresh) / l2_factor;
                let delta = beta_new - beta[j];

                if delta.abs() > <T as Zero>::zero() {
                    Self::update_residual(
                        j,
                        delta,
                        &mut residual,
                        &col_indptr,
                        &col_row_indices,
                        &col_values,
                    );
                    sum_r -= delta * (n_t * x_means[j]);
                    beta[j] = beta_new;

                    let abs_delta = delta.abs();
                    if abs_delta > max_change {
                        max_change = abs_delta;
                    }
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        let beta_arr = Array1::from(beta);

        let intercept = if fit_intercept {
            let xbar_dot_beta = x_means
                .iter()
                .zip(beta_arr.iter())
                .fold(<T as Zero>::zero(), |acc, (&xm, &bj)| acc + xm * bj);
            y_mean - xbar_dot_beta
        } else {
            <T as Zero>::zero()
        };

        Ok((beta_arr, intercept))
    }
}

/// Convenience functions for sparse matrix operations
#[cfg(feature = "sparse")]
pub mod utils {
    use super::*;

    /// Check if a dense matrix should be converted to sparse based on sparsity ratio
    pub fn should_use_sparse<T: FloatBounds>(dense: &Array2<T>, config: &SparseConfig) -> bool {
        let total_elements = dense.len() as f64;
        let non_zero = dense
            .iter()
            .filter(|&&x| {
                x.abs() > T::from(config.sparsity_threshold).expect("operation should succeed")
            })
            .count() as f64;
        let sparsity = non_zero / total_elements;
        sparsity < config.min_sparsity_ratio
    }

    /// Convert dense to sparse if beneficial
    pub fn auto_sparse<T: FloatBounds + SparseElement>(
        dense: &Array2<T>,
        threshold: T,
    ) -> Result<SparseMatrixCSR<T>> {
        let config = SparseConfig::default();
        if should_use_sparse(dense, &config) {
            Ok(SparseMatrixCSR::from_dense(dense, threshold))
        } else {
            Err(SklearsError::InvalidInput(
                "Matrix is not sparse enough to benefit from sparse format".to_string(),
            ))
        }
    }

    /// Analyse the sparsity pattern of a dense matrix.
    ///
    /// # Arguments
    /// * `dense` - The dense matrix to analyse.
    /// * `threshold` - Values with absolute value ≤ `threshold` are considered zero.
    ///
    /// # Returns
    /// A [`SparsityAnalysis`] describing the matrix's sparsity characteristics
    /// and a recommended storage format.
    pub fn analyze_sparsity(dense: &Array2<f64>, threshold: f64) -> super::SparsityAnalysis {
        let n_rows = dense.nrows();
        let n_cols = dense.ncols();
        let total_elements = n_rows * n_cols;

        if total_elements == 0 {
            return super::SparsityAnalysis {
                sparsity_ratio: 0.0,
                memory_savings: 0.0,
                recommended_format: "dense".to_string(),
                total_elements: 0,
                nonzero_count: 0,
                n_rows,
                n_cols,
            };
        }

        let nonzero_count = dense.iter().filter(|&&v| v.abs() > threshold).count();
        let sparsity_ratio = nonzero_count as f64 / total_elements as f64;

        // Rough memory savings estimate:
        // Dense storage: total_elements * 8 bytes
        // CSR storage: nonzero_count * (8 + 4 + 4) bytes + (n_rows+1)*4 bytes (indptr)
        // savings = 1 - (csr_bytes / dense_bytes)
        let dense_bytes = total_elements as f64 * 8.0;
        let csr_bytes =
            nonzero_count as f64 * 16.0 + (n_rows as f64 + 1.0) * 4.0 + nonzero_count as f64 * 4.0;
        let memory_savings = if dense_bytes > 0.0 {
            (1.0 - csr_bytes / dense_bytes).max(0.0)
        } else {
            0.0
        };

        let recommended_format = if sparsity_ratio < 0.1 {
            "CSR (sparse)".to_string()
        } else if sparsity_ratio < 0.3 {
            "CSR or dense (borderline)".to_string()
        } else {
            "dense".to_string()
        };

        super::SparsityAnalysis {
            sparsity_ratio,
            memory_savings,
            recommended_format,
            total_elements,
            nonzero_count,
            n_rows,
            n_cols,
        }
    }
}

#[cfg(all(test, feature = "sparse"))]
mod tests {
    use super::*;

    /// Build a small sparse matrix from a dense 2-D array.
    fn make_sparse(rows: &[[f64; 3]]) -> SparseMatrixCSR<f64> {
        let nrows = rows.len();
        let ncols = rows[0].len();
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v.abs() > 1e-12 {
                    triplets.push((i, j, v));
                }
            }
        }
        SparseMatrixCSR::from_triplets(nrows, ncols, &triplets).expect("build_sparse failed")
    }

    /// -----------------------------------------------------------------------
    /// Test 1: sparse LASSO coordinate descent on a simple 5×3 problem.
    ///
    /// True beta = [2.0, 0.0, -1.5].  The non-sparse column (col 1) is
    /// zeroed in y so LASSO should shrink its coefficient to zero.
    #[test]
    fn test_sparse_lasso_fit() {
        // X: 5 rows, 3 columns — column 1 is entirely zero (sparse)
        let x_data: [[f64; 3]; 5] = [
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [3.0, 0.0, 1.0],
        ];
        let x = make_sparse(&x_data);

        // y = X * [2, 0, -1.5] + tiny noise
        let true_beta = [2.0_f64, 0.0, -1.5];
        let y_vals: Vec<f64> = x_data
            .iter()
            .map(|row| {
                row.iter()
                    .zip(true_beta.iter())
                    .map(|(xi, bi)| xi * bi)
                    .sum()
            })
            .collect();
        let y = Array1::from(y_vals);

        let solver = SparseCoordinateDescentSolver::new(
            0.01_f64, // alpha (LASSO)
            1.0_f64,  // l1_ratio
            1000_usize, 1e-8_f64,
        );

        let beta = solver.fit_lasso(&x, &y).expect("fit_lasso failed");

        assert_eq!(beta.len(), 3);
        // With small alpha, recovered betas should be close to true values
        assert!((beta[0] - 2.0).abs() < 0.1, "beta[0]={}", beta[0]);
        // beta[1] should stay near 0 (no signal)
        assert!(beta[1].abs() < 1e-6, "beta[1]={}", beta[1]);
        assert!((beta[2] - (-1.5)).abs() < 0.1, "beta[2]={}", beta[2]);
    }

    /// -----------------------------------------------------------------------
    /// Test 2: sparse Ridge regression via CG on the same 5×3 problem.
    #[test]
    fn test_sparse_ridge_fit() {
        let x_data: [[f64; 3]; 5] = [
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [3.0, 0.0, 1.0],
        ];
        let x = make_sparse(&x_data);

        let true_beta = [2.0_f64, 0.0, -1.5];
        let y_vals: Vec<f64> = x_data
            .iter()
            .map(|row| {
                row.iter()
                    .zip(true_beta.iter())
                    .map(|(xi, bi)| xi * bi)
                    .sum()
            })
            .collect();
        let y = Array1::from(y_vals);

        let solver = SparseCoordinateDescentSolver::new(
            0.001_f64, // very small ridge lambda
            0.0_f64,   // l1_ratio (unused for ridge)
            2000_usize, 1e-10_f64,
        );

        let beta = solver.fit_ridge(&x, &y).expect("fit_ridge failed");

        assert_eq!(beta.len(), 3);
        assert!((beta[0] - 2.0).abs() < 0.2, "beta[0]={}", beta[0]);
        assert!((beta[2] - (-1.5)).abs() < 0.2, "beta[2]={}", beta[2]);
    }

    /// -----------------------------------------------------------------------
    /// Test 3: sparse Elastic Net on the 5×3 problem with l1_ratio=0.5.
    #[test]
    fn test_sparse_elastic_net_fit() {
        let x_data: [[f64; 3]; 5] = [
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [3.0, 0.0, 1.0],
        ];
        let x = make_sparse(&x_data);

        let true_beta = [2.0_f64, 0.0, -1.5];
        let y_vals: Vec<f64> = x_data
            .iter()
            .map(|row| {
                row.iter()
                    .zip(true_beta.iter())
                    .map(|(xi, bi)| xi * bi)
                    .sum()
            })
            .collect();
        let y = Array1::from(y_vals);

        let solver = SparseCoordinateDescentSolver::new(
            0.01_f64, // alpha
            0.5_f64,  // l1_ratio = 0.5 (balanced EN)
            1000_usize, 1e-8_f64,
        );

        let beta = solver
            .fit_elastic_net(&x, &y)
            .expect("fit_elastic_net failed");

        assert_eq!(beta.len(), 3);
        assert!((beta[0] - 2.0).abs() < 0.15, "beta[0]={}", beta[0]);
        assert!(beta[1].abs() < 1e-6, "beta[1]={}", beta[1]);
        assert!((beta[2] - (-1.5)).abs() < 0.15, "beta[2]={}", beta[2]);
    }

    /// -----------------------------------------------------------------------
    /// Test 4: solve_sparse_lasso without intercept (fit_intercept=false).
    #[test]
    fn test_solve_sparse_lasso_no_intercept() {
        let x_data: [[f64; 3]; 5] = [
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [3.0, 0.0, 1.0],
        ];
        let x = make_sparse(&x_data);

        let true_beta = [2.0_f64, 0.0, -1.5];
        let y_vals: Vec<f64> = x_data
            .iter()
            .map(|row| {
                row.iter()
                    .zip(true_beta.iter())
                    .map(|(xi, bi)| xi * bi)
                    .sum()
            })
            .collect();
        let y = Array1::from(y_vals);

        let solver = SparseCoordinateDescentSolver::new(0.01_f64, 1.0_f64, 1000, 1e-8_f64);
        let (beta, intercept) = solver
            .solve_sparse_lasso(&x, &y, 0.01_f64, false)
            .expect("solve_sparse_lasso failed");

        assert_eq!(beta.len(), 3);
        assert!(intercept.abs() < 1e-12, "intercept should be 0");
        assert!((beta[0] - 2.0).abs() < 0.1, "beta[0]={}", beta[0]);
        assert!((beta[2] - (-1.5)).abs() < 0.1, "beta[2]={}", beta[2]);
    }

    /// -----------------------------------------------------------------------
    /// Test 5: solve_sparse_elastic_net with intercept (fit_intercept=true).
    ///
    /// Uses the original non-zero-mean design to verify that virtual centering
    /// correctly recovers the true intercept `b₀ = ȳ - x̄ᵀβ` even when X
    /// has non-zero column means.
    #[test]
    fn test_solve_sparse_elastic_net_with_intercept() {
        let x_data: [[f64; 3]; 5] = [
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [3.0, 0.0, 1.0],
        ];
        // col0 mean = 7/5 = 1.4, col2 mean = 5/5 = 1.0
        let x = make_sparse(&x_data);

        let true_beta = [2.0_f64, 0.0, -1.5];
        let true_intercept = 5.0_f64;
        let y_vals: Vec<f64> = x_data
            .iter()
            .map(|row| {
                true_intercept
                    + row
                        .iter()
                        .zip(true_beta.iter())
                        .map(|(xi, bi)| xi * bi)
                        .sum::<f64>()
            })
            .collect();
        let y = Array1::from(y_vals);

        // Use very small alpha so regularisation bias on betas is tiny.
        // Virtual centering ensures the intercept is recovered correctly.
        let solver = SparseCoordinateDescentSolver::new(1e-5_f64, 0.5_f64, 2000, 1e-10_f64);
        let (beta, intercept) = solver
            .solve_sparse_elastic_net(&x, &y, 1e-5_f64, 0.5_f64, true)
            .expect("solve_sparse_elastic_net failed");

        assert_eq!(beta.len(), 3);
        // With virtual centering, intercept ≈ true_intercept.
        assert!(
            (intercept - true_intercept).abs() < 0.1,
            "intercept={} expected ~{}",
            intercept,
            true_intercept
        );
        // Betas close to true values with small regularisation.
        assert!((beta[0] - 2.0).abs() < 0.2, "beta[0]={}", beta[0]);
    }
}
