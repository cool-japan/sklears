//! Sparse Tensor Decomposition implementation

use super::common::{Trained, Untrained};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::marker::PhantomData;

/// Sparse Tensor Decomposition
///
/// Decomposes a sparse tensor using CP decomposition with sparsity constraints.
/// Handles tensors with many zero entries efficiently and can enforce sparsity
/// in the factor matrices through L1 regularization.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array3;
/// use sklears_cross_decomposition::SparseTensorDecomposition;
/// use sklears_core::traits::Fit;
///
/// let tensor = Array3::zeros((20, 15, 10));
/// let sparse_decomp = SparseTensorDecomposition::new(5).sparsity_penalty(0.1);
/// let fitted = sparse_decomp.fit(&tensor, &()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseTensorDecomposition<State = Untrained> {
    /// Number of factors
    pub n_factors: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// L1 sparsity penalty
    pub sparsity_penalty: Float,
    /// L2 regularization
    pub regularization: Float,
    /// Sparsity threshold (values below this are set to zero)
    pub sparsity_threshold: Float,
    /// Factor matrices
    factor_matrices_: Option<Vec<Array2<Float>>>,
    /// Original tensor shape
    original_shape_: Option<Vec<usize>>,
    /// Sparsity levels achieved
    sparsity_levels_: Option<Array1<Float>>,
    /// Reconstruction error
    reconstruction_error_: Option<Float>,
    /// Number of iterations
    n_iter_: Option<usize>,
    /// State marker
    _state: PhantomData<State>,
}

impl SparseTensorDecomposition<Untrained> {
    /// Create a new sparse tensor decomposition
    pub fn new(n_factors: usize) -> Self {
        Self {
            n_factors,
            max_iter: 100,
            tol: 1e-6,
            sparsity_penalty: 0.01,
            regularization: 0.001,
            sparsity_threshold: 1e-8,
            factor_matrices_: None,
            original_shape_: None,
            sparsity_levels_: None,
            reconstruction_error_: None,
            n_iter_: None,
            _state: PhantomData,
        }
    }

    /// Set sparsity penalty (L1 regularization)
    pub fn sparsity_penalty(mut self, penalty: Float) -> Self {
        self.sparsity_penalty = penalty;
        self
    }

    /// Set L2 regularization
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set sparsity threshold
    pub fn sparsity_threshold(mut self, threshold: Float) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }
}

impl Estimator for SparseTensorDecomposition<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array3<Float>, ()> for SparseTensorDecomposition<Untrained> {
    type Fitted = SparseTensorDecomposition<Trained>;

    fn fit(self, tensor: &Array3<Float>, _target: &()) -> Result<Self::Fitted> {
        let shape = tensor.shape();

        // Initialize factor matrices with small random values
        let mut factor_matrices = Vec::new();
        for mode in 0..3 {
            let mut factor = Array2::zeros((shape[mode], self.n_factors));
            for i in 0..shape[mode] {
                for j in 0..self.n_factors {
                    factor[[i, j]] = thread_rng().gen::<Float>() * 0.01;
                }
            }
            factor_matrices.push(factor);
        }

        let mut converged = false;
        let mut n_iter = 0;
        let mut prev_error = Float::INFINITY;

        // Sparse alternating least squares
        while !converged && n_iter < self.max_iter {
            let old_factors = factor_matrices.clone();

            // Update each factor matrix with sparsity constraints
            for mode in 0..3 {
                factor_matrices[mode] =
                    self.update_sparse_factor(tensor, &factor_matrices, mode)?;

                // Apply soft thresholding for sparsity
                self.apply_soft_thresholding(&mut factor_matrices[mode]);
            }

            // Compute reconstruction error
            let reconstructed = self.reconstruct_sparse_tensor(&factor_matrices, shape)?;
            let error = (tensor - &reconstructed).mapv(|x| x * x).sum().sqrt();

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                converged = true;
            }

            // Also check factor convergence
            let mut max_factor_change: Float = 0.0;
            for mode in 0..3 {
                let change = (&factor_matrices[mode] - &old_factors[mode])
                    .mapv(|x| x.abs())
                    .sum();
                max_factor_change = max_factor_change.max(change);
            }

            if max_factor_change < self.tol {
                converged = true;
            }

            prev_error = error;
            n_iter += 1;
        }

        // Compute sparsity levels
        let mut sparsity_levels = Array1::zeros(3);
        for mode in 0..3 {
            let total_elements = factor_matrices[mode].len();
            let sparse_elements = factor_matrices[mode]
                .iter()
                .filter(|&&x| x.abs() < self.sparsity_threshold)
                .count();
            sparsity_levels[mode] = sparse_elements as Float / total_elements as Float;
        }

        Ok(SparseTensorDecomposition {
            n_factors: self.n_factors,
            max_iter: self.max_iter,
            tol: self.tol,
            sparsity_penalty: self.sparsity_penalty,
            regularization: self.regularization,
            sparsity_threshold: self.sparsity_threshold,
            factor_matrices_: Some(factor_matrices),
            original_shape_: Some(shape.to_vec()),
            sparsity_levels_: Some(sparsity_levels),
            reconstruction_error_: Some(prev_error),
            n_iter_: Some(n_iter),
            _state: PhantomData,
        })
    }
}

impl SparseTensorDecomposition<Untrained> {
    /// Update factor matrix with sparsity constraints
    fn update_sparse_factor(
        &self,
        tensor: &Array3<Float>,
        factors: &[Array2<Float>],
        mode: usize,
    ) -> Result<Array2<Float>> {
        let shape = tensor.shape();
        let mut new_factor = Array2::zeros((shape[mode], self.n_factors));

        // Simplified sparse update using coordinate descent
        for r in 0..self.n_factors {
            let mut factor_col = Array1::zeros(shape[mode]);

            match mode {
                0 => {
                    for i in 0..shape[0] {
                        let mut numerator = 0.0;
                        let mut denominator = 0.0;

                        for j in 0..shape[1] {
                            for k in 0..shape[2] {
                                let coeff = factors[1][[j, r]] * factors[2][[k, r]];
                                numerator += tensor[[i, j, k]] * coeff;
                                denominator += coeff * coeff;
                            }
                        }

                        if denominator > self.tol {
                            factor_col[i] = numerator / (denominator + self.regularization);
                        }
                    }
                }
                1 => {
                    for j in 0..shape[1] {
                        let mut numerator = 0.0;
                        let mut denominator = 0.0;

                        for i in 0..shape[0] {
                            for k in 0..shape[2] {
                                let coeff = factors[0][[i, r]] * factors[2][[k, r]];
                                numerator += tensor[[i, j, k]] * coeff;
                                denominator += coeff * coeff;
                            }
                        }

                        if denominator > self.tol {
                            factor_col[j] = numerator / (denominator + self.regularization);
                        }
                    }
                }
                2 => {
                    for k in 0..shape[2] {
                        let mut numerator = 0.0;
                        let mut denominator = 0.0;

                        for i in 0..shape[0] {
                            for j in 0..shape[1] {
                                let coeff = factors[0][[i, r]] * factors[1][[j, r]];
                                numerator += tensor[[i, j, k]] * coeff;
                                denominator += coeff * coeff;
                            }
                        }

                        if denominator > self.tol {
                            factor_col[k] = numerator / (denominator + self.regularization);
                        }
                    }
                }
                _ => return Err(SklearsError::InvalidInput("Invalid mode".to_string())),
            }

            new_factor.column_mut(r).assign(&factor_col);
        }

        Ok(new_factor)
    }

    /// Apply soft thresholding for L1 sparsity
    fn apply_soft_thresholding(&self, factor: &mut Array2<Float>) {
        let threshold = self.sparsity_penalty;
        factor.mapv_inplace(|x| {
            if x > threshold {
                x - threshold
            } else if x < -threshold {
                x + threshold
            } else {
                0.0
            }
        });
    }

    /// Reconstruct tensor from sparse factors
    fn reconstruct_sparse_tensor(
        &self,
        factors: &[Array2<Float>],
        shape: &[usize],
    ) -> Result<Array3<Float>> {
        let mut reconstructed = Array3::zeros((shape[0], shape[1], shape[2]));

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
}

impl Transform<Array3<Float>, Array3<Float>> for SparseTensorDecomposition<Trained> {
    /// Reconstruct tensor using sparse factors
    fn transform(&self, tensor: &Array3<Float>) -> Result<Array3<Float>> {
        let factors = self.factor_matrices_.as_ref().unwrap();
        let shape = tensor.shape();
        self.reconstruct_sparse_tensor(factors, shape)
    }
}

impl SparseTensorDecomposition<Trained> {
    /// Get the factor matrices
    pub fn factor_matrices(&self) -> &Vec<Array2<Float>> {
        self.factor_matrices_.as_ref().unwrap()
    }

    /// Get the sparsity levels for each mode
    pub fn sparsity_levels(&self) -> &Array1<Float> {
        self.sparsity_levels_.as_ref().unwrap()
    }

    /// Get the reconstruction error
    pub fn reconstruction_error(&self) -> Float {
        self.reconstruction_error_.unwrap()
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Helper method for sparse reconstruction
    fn reconstruct_sparse_tensor(
        &self,
        factors: &[Array2<Float>],
        shape: &[usize],
    ) -> Result<Array3<Float>> {
        let mut reconstructed = Array3::zeros((shape[0], shape[1], shape[2]));

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
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use sklears_core::traits::Fit;

    #[test]
    fn test_sparse_tensor_decomposition_basic() {
        let tensor = Array3::from_shape_fn((5, 4, 3), |(i, j, k)| {
            if (i + j + k) % 3 == 0 {
                (i + j + k) as Float
            } else {
                0.0
            }
        });

        let sparse_decomp = SparseTensorDecomposition::new(2)
            .sparsity_penalty(0.1)
            .max_iter(50);
        let fitted = sparse_decomp.fit(&tensor, &()).unwrap();

        assert_eq!(fitted.factor_matrices().len(), 3);
        assert_eq!(fitted.factor_matrices()[0].shape(), &[5, 2]);
        assert_eq!(fitted.factor_matrices()[1].shape(), &[4, 2]);
        assert_eq!(fitted.factor_matrices()[2].shape(), &[3, 2]);
        assert!(fitted.n_iter() > 0);
        assert!(fitted.reconstruction_error() >= 0.0);

        // Check sparsity levels
        let sparsity = fitted.sparsity_levels();
        assert_eq!(sparsity.len(), 3);
        for &level in sparsity.iter() {
            assert!(level >= 0.0 && level <= 1.0);
        }
    }

    #[test]
    fn test_sparse_tensor_decomposition_sparsity() {
        let tensor = Array3::from_shape_fn(
            (4, 4, 4),
            |(i, j, k)| {
                if i == j && j == k {
                    1.0
                } else {
                    0.0
                }
            },
        );

        let sparse_decomp = SparseTensorDecomposition::new(1)
            .sparsity_penalty(0.05)
            .regularization(0.01)
            .sparsity_threshold(1e-6);
        let fitted = sparse_decomp.fit(&tensor, &()).unwrap();

        // Should achieve some level of sparsity
        let sparsity = fitted.sparsity_levels();
        let avg_sparsity = sparsity.mean().unwrap();
        assert!(
            avg_sparsity > 0.0,
            "Expected some sparsity but got {}",
            avg_sparsity
        );
    }

    #[test]
    fn test_sparse_tensor_decomposition_transform() {
        let tensor = Array3::from_shape_fn((4, 3, 2), |(i, j, k)| (i + j + k) as Float * 0.1);

        let sparse_decomp = SparseTensorDecomposition::new(2);
        let fitted = sparse_decomp.fit(&tensor, &()).unwrap();

        let reconstructed = fitted.transform(&tensor).unwrap();
        assert_eq!(reconstructed.shape(), tensor.shape());
    }

    #[test]
    fn test_sparse_tensor_configuration() {
        let tensor = Array3::ones((3, 3, 3));

        let sparse_decomp = SparseTensorDecomposition::new(1)
            .sparsity_penalty(0.2)
            .regularization(0.05)
            .sparsity_threshold(1e-5)
            .max_iter(20)
            .tol(1e-4);

        let fitted = sparse_decomp.fit(&tensor, &()).unwrap();
        assert!(fitted.n_iter() <= 20);
    }
}
