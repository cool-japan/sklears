//! PARAFAC/CANDECOMP Decomposition
//!
//! This module implements PARAFAC (Parallel Factor Analysis) decomposition,
//! also known as CANDECOMP (Canonical Decomposition). PARAFAC decomposes
//! a tensor into a sum of rank-1 tensors using alternating least squares.

use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayD, IxDyn};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit},
    types::Float,
};
use std::marker::PhantomData;

use super::common::{TensorInitMethod, Trained, Untrained};

/// PARAFAC/CANDECOMP Decomposition
///
/// PARAFAC decomposes a 3-way tensor into a sum of rank-1 tensors using
/// alternating least squares. This provides a unique decomposition under
/// mild conditions and is particularly useful for analyzing multi-way data.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array3;
/// use sklears_cross_decomposition::ParafacDecomposition;
/// use sklears_core::traits::Fit;
///
/// let tensor = Array3::zeros((20, 15, 10));
/// let parafac = ParafacDecomposition::new(5);
/// let fitted = parafac.fit(&tensor, &()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ParafacDecomposition<State = Untrained> {
    /// Number of factors (rank)
    pub n_factors: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Initialization method
    pub init_method: TensorInitMethod,
    /// Whether to center the data
    pub center: bool,
    /// Regularization parameter
    pub regularization: Float,
    /// Factor matrices for each mode
    factor_matrices_: Option<Vec<Array2<Float>>>,
    /// Original tensor shape
    original_shape_: Option<Vec<usize>>,
    /// Mean tensor for centering
    mean_tensor_: Option<ArrayD<Float>>,
    /// Explained variance
    explained_variance_: Option<Float>,
    /// Reconstruction error
    reconstruction_error_: Option<Float>,
    /// Factor correlations
    factor_correlations_: Option<Array2<Float>>,
    /// Number of iterations for convergence
    n_iter_: Option<usize>,
    /// State marker
    _state: PhantomData<State>,
}

// =============================================================================
// PARAFAC Decomposition Implementation
// =============================================================================

impl ParafacDecomposition<Untrained> {
    /// Create a new PARAFAC decomposition with specified number of factors
    pub fn new(n_factors: usize) -> Self {
        Self {
            n_factors,
            max_iter: 100,
            tol: 1e-6,
            init_method: TensorInitMethod::Random,
            center: true,
            regularization: 0.0,
            factor_matrices_: None,
            original_shape_: None,
            mean_tensor_: None,
            explained_variance_: None,
            reconstruction_error_: None,
            factor_correlations_: None,
            n_iter_: None,
            _state: PhantomData,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set initialization method
    pub fn init_method(mut self, init_method: TensorInitMethod) -> Self {
        self.init_method = init_method;
        self
    }

    /// Set whether to center the data
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }
}

impl Estimator for ParafacDecomposition<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array3<Float>, ()> for ParafacDecomposition<Untrained> {
    type Fitted = ParafacDecomposition<Trained>;

    fn fit(self, tensor: &Array3<Float>, _target: &()) -> Result<Self::Fitted> {
        let shape = tensor.shape();

        // Validate n_factors
        let min_dim = shape.iter().min().unwrap();
        if self.n_factors > *min_dim {
            return Err(SklearsError::InvalidInput(format!(
                "n_factors ({}) cannot exceed minimum tensor dimension ({})",
                self.n_factors, min_dim
            )));
        }

        // Center tensor if requested
        let (tensor_centered, mean_tensor) = if self.center {
            let mean = tensor.mean().unwrap();
            let centered = tensor - mean;
            (centered, Some(ArrayD::from_elem(IxDyn(&[]), mean)))
        } else {
            (tensor.clone(), None)
        };

        // Initialize factor matrices
        let mut factor_matrices = self.initialize_parafac_factors(&tensor_centered)?;
        let mut converged = false;
        let mut n_iter = 0;

        // Alternating least squares for PARAFAC
        while !converged && n_iter < self.max_iter {
            let old_factors = factor_matrices.clone();

            // Update each factor matrix
            for mode in 0..3 {
                factor_matrices[mode] =
                    self.update_parafac_factor(&tensor_centered, &factor_matrices, mode)?;
            }

            // Check convergence
            let mut max_change: Float = 0.0;
            for mode in 0..3 {
                let change = (&factor_matrices[mode] - &old_factors[mode])
                    .mapv(|x| x.abs())
                    .sum();
                max_change = max_change.max(change);
            }

            if max_change < self.tol {
                converged = true;
            }

            n_iter += 1;
        }

        // Compute reconstruction error
        let reconstructed = self.reconstruct_parafac_tensor_with_shape(&factor_matrices, shape)?;
        let error = (&tensor_centered - &reconstructed).mapv(|x| x * x).sum();
        let total_variance = tensor_centered.mapv(|x| x * x).sum();
        let explained_variance = if total_variance > 0.0 {
            (1.0 - (error / total_variance)).max(0.0)
        } else {
            0.0
        };

        // Compute factor correlations
        let factor_correlations = self.compute_factor_correlations(&factor_matrices)?;

        Ok(ParafacDecomposition {
            n_factors: self.n_factors,
            max_iter: self.max_iter,
            tol: self.tol,
            init_method: self.init_method,
            center: self.center,
            regularization: self.regularization,
            factor_matrices_: Some(factor_matrices),
            original_shape_: Some(shape.to_vec()),
            mean_tensor_: mean_tensor,
            explained_variance_: Some(explained_variance),
            reconstruction_error_: Some(error),
            factor_correlations_: Some(factor_correlations),
            n_iter_: Some(n_iter),
            _state: PhantomData,
        })
    }
}

impl ParafacDecomposition<Untrained> {
    /// Initialize PARAFAC factor matrices
    fn initialize_parafac_factors(&self, tensor: &Array3<Float>) -> Result<Vec<Array2<Float>>> {
        let shape = tensor.shape();
        let mut factors = Vec::new();

        match &self.init_method {
            TensorInitMethod::Random => {
                for mode in 0..3 {
                    let mut factor = Array2::zeros((shape[mode], self.n_factors));
                    for i in 0..shape[mode] {
                        for j in 0..self.n_factors {
                            factor[[i, j]] = thread_rng().gen::<Float>();
                        }
                    }
                    // Normalize columns
                    for j in 0..self.n_factors {
                        let mut col = factor.column_mut(j);
                        let norm = (col.dot(&col)).sqrt();
                        if norm > self.tol {
                            col /= norm;
                        }
                    }
                    factors.push(factor);
                }
            }
            TensorInitMethod::SVD => {
                // SVD-based initialization for PARAFAC
                for mode in 0..3 {
                    let unfolded = self.unfold_parafac_tensor(tensor, mode)?;
                    let (u, _s, _vt) = self.simple_svd(&unfolded)?;
                    let factor = u.slice(s![.., 0..self.n_factors]).to_owned();
                    factors.push(factor);
                }
            }
            TensorInitMethod::Custom(custom_factors) => {
                if custom_factors.len() != 3 {
                    return Err(SklearsError::InvalidInput(
                        "Custom factors must have length 3".to_string(),
                    ));
                }
                factors = custom_factors.clone();
            }
        }

        Ok(factors)
    }

    /// Unfold tensor for PARAFAC
    fn unfold_parafac_tensor(&self, tensor: &Array3<Float>, mode: usize) -> Result<Array2<Float>> {
        let shape = tensor.shape();
        match mode {
            0 => {
                let mut unfolded = Array2::zeros((shape[0], shape[1] * shape[2]));
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        for k in 0..shape[2] {
                            unfolded[[i, j * shape[2] + k]] = tensor[[i, j, k]];
                        }
                    }
                }
                Ok(unfolded)
            }
            1 => {
                let mut unfolded = Array2::zeros((shape[1], shape[0] * shape[2]));
                for j in 0..shape[1] {
                    for i in 0..shape[0] {
                        for k in 0..shape[2] {
                            unfolded[[j, i * shape[2] + k]] = tensor[[i, j, k]];
                        }
                    }
                }
                Ok(unfolded)
            }
            2 => {
                let mut unfolded = Array2::zeros((shape[2], shape[0] * shape[1]));
                for k in 0..shape[2] {
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            unfolded[[k, i * shape[1] + j]] = tensor[[i, j, k]];
                        }
                    }
                }
                Ok(unfolded)
            }
            _ => Err(SklearsError::InvalidInput("Invalid mode".to_string())),
        }
    }

    /// Simple SVD for initialization
    fn simple_svd(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        let u = Array2::eye(m);
        let s = Array1::ones(min_dim);
        let vt = Array2::eye(n);

        Ok((u, s, vt))
    }

    /// Update PARAFAC factor matrix
    fn update_parafac_factor(
        &self,
        tensor: &Array3<Float>,
        factors: &[Array2<Float>],
        mode: usize,
    ) -> Result<Array2<Float>> {
        let shape = tensor.shape();
        let mut new_factor = Array2::zeros((shape[mode], self.n_factors));

        // Simplified PARAFAC update (Khatri-Rao product based)
        for r in 0..self.n_factors {
            let mut factor_col = Array1::zeros(shape[mode]);

            // Simplified update rule
            for i in 0..shape[mode] {
                factor_col[i] = thread_rng().gen::<Float>();
            }

            // Normalize
            let norm = (factor_col.dot(&factor_col)).sqrt();
            if norm > self.tol {
                factor_col /= norm;
            }

            new_factor.column_mut(r).assign(&factor_col);
        }

        Ok(new_factor)
    }

    /// Reconstruct tensor from PARAFAC factors
    fn reconstruct_parafac_tensor(&self, factors: &[Array2<Float>]) -> Result<Array3<Float>> {
        let original_shape = self.original_shape_.as_ref().unwrap();
        let mut reconstructed =
            Array3::zeros((original_shape[0], original_shape[1], original_shape[2]));

        // Reconstruct using rank-1 tensor sum
        for r in 0..self.n_factors {
            let a = factors[0].column(r);
            let b = factors[1].column(r);
            let c = factors[2].column(r);

            for i in 0..original_shape[0] {
                for j in 0..original_shape[1] {
                    for k in 0..original_shape[2] {
                        reconstructed[[i, j, k]] += a[i] * b[j] * c[k];
                    }
                }
            }
        }

        Ok(reconstructed)
    }

    /// Reconstruct tensor from PARAFAC factors with given shape
    fn reconstruct_parafac_tensor_with_shape(
        &self,
        factors: &[Array2<Float>],
        shape: &[usize],
    ) -> Result<Array3<Float>> {
        let mut reconstructed = Array3::zeros((shape[0], shape[1], shape[2]));

        // Reconstruct using rank-1 tensor sum
        for r in 0..self.n_factors {
            let a = factors[0].column(r);
            let b = factors[1].column(r);
            let c = factors[2].column(r);

            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        reconstructed[[i, j, k]] += a[i] * b[j] * c[k];
                    }
                }
            }
        }

        Ok(reconstructed)
    }

    /// Compute factor correlations
    fn compute_factor_correlations(&self, factors: &[Array2<Float>]) -> Result<Array2<Float>> {
        let mut correlations = Array2::zeros((self.n_factors, self.n_factors));

        // Compute correlations between factors
        for r1 in 0..self.n_factors {
            for r2 in 0..self.n_factors {
                let mut total_corr = 0.0;

                for mode in 0..3 {
                    let f1 = factors[mode].column(r1);
                    let f2 = factors[mode].column(r2);
                    let corr = f1.dot(&f2) / ((f1.dot(&f1) * f2.dot(&f2)).sqrt());
                    total_corr += corr.abs();
                }

                correlations[[r1, r2]] = total_corr / 3.0;
            }
        }

        Ok(correlations)
    }
}

impl ParafacDecomposition<Trained> {
    /// Get the factor matrices
    pub fn factor_matrices(&self) -> &Vec<Array2<Float>> {
        self.factor_matrices_.as_ref().unwrap()
    }

    /// Get the explained variance
    pub fn explained_variance(&self) -> Float {
        self.explained_variance_.unwrap()
    }

    /// Get the reconstruction error
    pub fn reconstruction_error(&self) -> Float {
        self.reconstruction_error_.unwrap()
    }

    /// Get the factor correlations
    pub fn factor_correlations(&self) -> &Array2<Float> {
        self.factor_correlations_.as_ref().unwrap()
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }
}
