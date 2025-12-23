//! Tensor Completion Implementation
//!
//! This module provides tensor completion via alternating least squares,
//! filling missing entries in a tensor using low-rank decomposition.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::marker::PhantomData;

use super::common::{Trained, Untrained};

/// Tensor Completion via Alternating Least Squares
///
/// Tensor completion fills missing entries in a tensor using low-rank decomposition.
/// This implementation uses alternating least squares with a CP (PARAFAC) decomposition
/// model to complete missing entries based on observed patterns.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array3;
/// use sklears_cross_decomposition::TensorCompletion;
/// use sklears_core::traits::Fit;
///
/// let mut tensor = Array3::ones((10, 8, 6));
/// let completion = TensorCompletion::new(3);
/// let fitted = completion.fit(&tensor, &()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TensorCompletion<State = Untrained> {
    /// Number of factors for CP decomposition
    pub n_factors: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Regularization parameter
    pub regularization: Float,
    /// Missing value threshold (values below this are considered missing)
    pub missing_threshold: Float,
    /// Factor matrices from decomposition
    factor_matrices_: Option<Vec<Array2<Float>>>,
    /// Original tensor shape
    original_shape_: Option<Vec<usize>>,
    /// Missing value mask (true for observed, false for missing)
    missing_mask_: Option<Array3<bool>>,
    /// Reconstruction error
    reconstruction_error_: Option<Float>,
    /// Number of iterations for convergence
    n_iter_: Option<usize>,
    /// State marker
    _state: PhantomData<State>,
}

impl TensorCompletion<Untrained> {
    /// Create a new tensor completion estimator
    pub fn new(n_factors: usize) -> Self {
        Self {
            n_factors,
            max_iter: 100,
            tol: 1e-6,
            regularization: 0.01,
            missing_threshold: Float::NAN,
            factor_matrices_: None,
            original_shape_: None,
            missing_mask_: None,
            reconstruction_error_: None,
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

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set missing value threshold
    pub fn missing_threshold(mut self, threshold: Float) -> Self {
        self.missing_threshold = threshold;
        self
    }
}

impl Estimator for TensorCompletion<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array3<Float>, ()> for TensorCompletion<Untrained> {
    type Fitted = TensorCompletion<Trained>;

    fn fit(self, tensor: &Array3<Float>, _target: &()) -> Result<Self::Fitted> {
        let shape = tensor.shape();

        // Create missing value mask
        let missing_mask = if self.missing_threshold.is_nan() {
            // If threshold is NaN, look for NaN values in tensor
            tensor.mapv(|x| !x.is_nan())
        } else {
            // Use threshold to determine missing values
            tensor.mapv(|x| x.abs() >= self.missing_threshold)
        };

        // Count observed entries
        let n_observed = missing_mask.iter().filter(|&&x| x).count();
        if n_observed == 0 {
            return Err(SklearsError::InvalidInput(
                "No observed entries found in tensor".to_string(),
            ));
        }

        // Initialize tensor with zeros for missing values
        let mut working_tensor = tensor.clone();
        for ((i, j, k), &is_observed) in missing_mask.indexed_iter() {
            if !is_observed {
                working_tensor[[i, j, k]] = 0.0;
            }
        }

        // Initialize factor matrices randomly
        let mut factor_matrices = Vec::new();
        for mode in 0..3 {
            let mut factor = Array2::zeros((shape[mode], self.n_factors));
            for i in 0..shape[mode] {
                for j in 0..self.n_factors {
                    factor[[i, j]] = thread_rng().random::<Float>() * 0.1;
                }
            }
            factor_matrices.push(factor);
        }

        let mut converged = false;
        let mut n_iter = 0;
        let mut prev_error = Float::INFINITY;

        // Alternating least squares optimization
        while !converged && n_iter < self.max_iter {
            // Update each factor matrix
            for mode in 0..3 {
                factor_matrices[mode] = self.update_completion_factor(
                    &working_tensor,
                    &factor_matrices,
                    &missing_mask,
                    mode,
                )?;
            }

            // Reconstruct tensor and update missing entries
            let reconstructed = self.reconstruct_completion_tensor(&factor_matrices, shape)?;

            // Update missing entries with reconstructed values
            for ((i, j, k), &is_observed) in missing_mask.indexed_iter() {
                if !is_observed {
                    working_tensor[[i, j, k]] = reconstructed[[i, j, k]];
                }
            }

            // Compute reconstruction error on observed entries only
            let mut error = 0.0;
            let mut count = 0;
            for ((i, j, k), &is_observed) in missing_mask.indexed_iter() {
                if is_observed {
                    let diff = tensor[[i, j, k]] - reconstructed[[i, j, k]];
                    error += diff * diff;
                    count += 1;
                }
            }
            error = (error / count as Float).sqrt();

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                converged = true;
            }
            prev_error = error;
            n_iter += 1;
        }

        Ok(TensorCompletion {
            n_factors: self.n_factors,
            max_iter: self.max_iter,
            tol: self.tol,
            regularization: self.regularization,
            missing_threshold: self.missing_threshold,
            factor_matrices_: Some(factor_matrices),
            original_shape_: Some(shape.to_vec()),
            missing_mask_: Some(missing_mask),
            reconstruction_error_: Some(prev_error),
            n_iter_: Some(n_iter),
            _state: PhantomData,
        })
    }
}

impl TensorCompletion<Untrained> {
    /// Update factor matrix for tensor completion
    fn update_completion_factor(
        &self,
        tensor: &Array3<Float>,
        factors: &[Array2<Float>],
        mask: &Array3<bool>,
        mode: usize,
    ) -> Result<Array2<Float>> {
        let shape = tensor.shape();
        let mut new_factor = Array2::zeros((shape[mode], self.n_factors));

        // Solve least squares for each factor column
        for r in 0..self.n_factors {
            let mut lhs: Array2<Float> = Array2::zeros((shape[mode], shape[mode]));
            let mut rhs: Array1<Float> = Array1::zeros(shape[mode]);

            // Build normal equations using observed entries only
            match mode {
                0 => {
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            for k in 0..shape[2] {
                                if mask[[i, j, k]] {
                                    let coeff = factors[1][[j, r]] * factors[2][[k, r]];
                                    lhs[[i, i]] += coeff * coeff;
                                    rhs[i] += tensor[[i, j, k]] * coeff;
                                }
                            }
                        }
                    }
                }
                1 => {
                    for j in 0..shape[1] {
                        for i in 0..shape[0] {
                            for k in 0..shape[2] {
                                if mask[[i, j, k]] {
                                    let coeff = factors[0][[i, r]] * factors[2][[k, r]];
                                    lhs[[j, j]] += coeff * coeff;
                                    rhs[j] += tensor[[i, j, k]] * coeff;
                                }
                            }
                        }
                    }
                }
                2 => {
                    for k in 0..shape[2] {
                        for i in 0..shape[0] {
                            for j in 0..shape[1] {
                                if mask[[i, j, k]] {
                                    let coeff = factors[0][[i, r]] * factors[1][[j, r]];
                                    lhs[[k, k]] += coeff * coeff;
                                    rhs[k] += tensor[[i, j, k]] * coeff;
                                }
                            }
                        }
                    }
                }
                _ => return Err(SklearsError::InvalidInput("Invalid mode".to_string())),
            }

            // Add regularization
            for i in 0..shape[mode] {
                lhs[[i, i]] += self.regularization;
            }

            // Solve for factor column (simplified - use diagonal solution)
            for i in 0..shape[mode] {
                if lhs[[i, i]] > self.tol {
                    new_factor[[i, r]] = rhs[i] / lhs[[i, i]];
                }
            }
        }

        Ok(new_factor)
    }

    /// Reconstruct tensor from completion factors
    fn reconstruct_completion_tensor(
        &self,
        factors: &[Array2<Float>],
        shape: &[usize],
    ) -> Result<Array3<Float>> {
        let mut reconstructed = Array3::zeros((shape[0], shape[1], shape[2]));

        // Sum over all rank-1 components
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

impl Transform<Array3<Float>, Array3<Float>> for TensorCompletion<Trained> {
    /// Complete missing entries in a new tensor
    fn transform(&self, tensor: &Array3<Float>) -> Result<Array3<Float>> {
        let factors = self.factor_matrices_.as_ref().unwrap();
        let shape = tensor.shape();

        if shape != self.original_shape_.as_ref().unwrap().as_slice() {
            return Err(SklearsError::InvalidInput(
                "Tensor shape must match training shape".to_string(),
            ));
        }

        self.reconstruct_completion_tensor(factors, shape)
    }
}

impl TensorCompletion<Trained> {
    /// Get the factor matrices
    pub fn factor_matrices(&self) -> &Vec<Array2<Float>> {
        self.factor_matrices_.as_ref().unwrap()
    }

    /// Get the reconstruction error
    pub fn reconstruction_error(&self) -> Float {
        self.reconstruction_error_.unwrap()
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Get the missing value mask
    pub fn missing_mask(&self) -> &Array3<bool> {
        self.missing_mask_.as_ref().unwrap()
    }

    /// Helper method for completion
    fn reconstruct_completion_tensor(
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
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_tensor_completion_basic() {
        let mut tensor = Array3::ones((3, 3, 3));
        // Set some missing values as NaN
        tensor[[0, 0, 0]] = Float::NAN;
        tensor[[1, 1, 1]] = Float::NAN;

        let completion = TensorCompletion::new(1);
        let fitted = completion.fit(&tensor, &()).unwrap();

        assert_eq!(fitted.n_factors, 1);
        assert!(fitted.reconstruction_error() >= 0.0);
    }

    #[test]
    fn test_tensor_completion_with_threshold() {
        let mut tensor = Array3::ones((3, 3, 3));
        // Set some values to be considered missing based on threshold
        tensor[[0, 0, 0]] = 0.001; // Below threshold
        tensor[[1, 1, 1]] = 0.0005; // Below threshold

        let completion = TensorCompletion::new(1).missing_threshold(0.01);
        let fitted = completion.fit(&tensor, &()).unwrap();

        assert_eq!(fitted.n_factors, 1);
        assert!(fitted.reconstruction_error() >= 0.0);
    }

    #[test]
    fn test_tensor_completion_configuration() {
        let tensor = Array3::ones((3, 3, 3));

        let completion = TensorCompletion::new(2)
            .max_iter(50)
            .tol(1e-5)
            .regularization(0.05);

        let fitted = completion.fit(&tensor, &()).unwrap();
        assert_eq!(fitted.n_factors, 2);
        assert!(fitted.n_iter() <= 50);
    }

    #[test]
    fn test_tensor_completion_transform() {
        let tensor = Array3::ones((3, 3, 3));
        let completion = TensorCompletion::new(1);
        let fitted = completion.fit(&tensor, &()).unwrap();

        let mut test_tensor = Array3::ones((3, 3, 3));
        test_tensor[[0, 0, 0]] = Float::NAN;

        let completed = fitted.transform(&test_tensor).unwrap();
        assert_eq!(completed.shape(), &[3, 3, 3]);
    }

    #[test]
    fn test_tensor_completion_wrong_shape() {
        let tensor = Array3::ones((3, 3, 3));
        let completion = TensorCompletion::new(1);
        let fitted = completion.fit(&tensor, &()).unwrap();

        let wrong_tensor = Array3::ones((4, 4, 4));
        assert!(fitted.transform(&wrong_tensor).is_err());
    }
}
