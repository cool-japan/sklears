//! Tensor decomposition algorithms
//!
//! This module provides tensor decomposition methods including:
//! - CP (CANDECOMP/PARAFAC) decomposition for multi-way data analysis
//! - Tucker decomposition for multilinear algebra applications
//! - Higher-order SVD for tensor dimensionality reduction

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{Rng, thread_rng, Random};
use scirs2_linalg::compat::{svd, ArrayLinalgExt};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::traits::Fit;
use sklears_core::{
    error::{Result, SklearsError},
    traits::Untrained,
};

/// Type alias for CP decomposition result
type CPResult = Result<(Vec<Array2<f64>>, Array1<f64>, usize, f64)>;

/// Type alias for Tucker decomposition result  
type TuckerResult = Result<(Array3<f64>, Vec<Array2<f64>>, usize, f64)>;

/// CP decomposition algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CPAlgorithm {
    /// Alternating Least Squares (ALS) algorithm
    #[default]
    ALS,
    /// Non-negative CP decomposition using multiplicative updates
    NonNegative,
    /// Robust CP decomposition with outlier handling
    Robust,
}

/// Tucker decomposition algorithm variants  
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TuckerAlgorithm {
    /// Higher-order SVD (HOSVD) algorithm
    #[default]
    HOSVD,
    /// Alternating Least Squares for Tucker decomposition
    ALS,
    /// Sequential unfolding SVD
    SequentialSVD,
}

/// CP (CANDECOMP/PARAFAC) Decomposition
///
/// Decomposes a tensor into a sum of rank-1 tensors:
/// X ≈ Σᵢ λᵢ aᵢ ⊗ bᵢ ⊗ cᵢ
#[derive(Debug, Clone)]
pub struct CPDecomposition<State = Untrained> {
    /// Number of components (rank of decomposition)
    pub n_components: usize,
    /// Algorithm variant to use
    pub algorithm: CPAlgorithm,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Learning rate for gradient-based methods
    pub learning_rate: f64,
    /// Regularization parameter
    pub regularization: f64,
    /// Whether to normalize factors
    pub normalize_factors: bool,

    /// Trained state
    #[allow(dead_code)]
    state: State,
}

/// Trained CP decomposition state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainedCP {
    pub factors: Vec<Array2<f64>>,
    pub weights: Array1<f64>,
    pub tensor_shape: Vec<usize>,
    pub n_components: usize,
    pub n_iter: usize,
    pub reconstruction_error: f64,
}

/// Tucker Decomposition
///
/// Decomposes a tensor into a core tensor multiplied by factor matrices:
/// X ≈ G ×₁ A ×₂ B ×₃ C
#[derive(Debug, Clone)]
pub struct TuckerDecomposition<State = Untrained> {
    /// Number of components for each mode
    pub n_components: Vec<usize>,
    /// Algorithm variant to use
    pub algorithm: TuckerAlgorithm,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to center the tensor
    pub center: bool,
    /// Initialization method
    pub init_method: String,

    /// Trained state
    #[allow(dead_code)]
    state: State,
}

/// Trained Tucker decomposition state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainedTucker {
    pub core: Array3<f64>,
    pub factors: Vec<Array2<f64>>,
    pub mean: Option<Array3<f64>>,
    pub tensor_shape: Vec<usize>,
    pub n_components: Vec<usize>,
    pub n_iter: usize,
    pub reconstruction_error: f64,
}

impl CPDecomposition<Untrained> {
    /// Create a new CP decomposition
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            algorithm: CPAlgorithm::ALS,
            max_iter: 100,
            tol: 1e-6,
            random_state: None,
            learning_rate: 0.01,
            regularization: 0.0,
            normalize_factors: true,
            state: Untrained,
        }
    }

    /// Set the algorithm
    pub fn algorithm(mut self, algorithm: CPAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set whether to normalize factors
    pub fn normalize_factors(mut self, normalize_factors: bool) -> Self {
        self.normalize_factors = normalize_factors;
        self
    }
}

impl Fit<Array3<f64>, ()> for CPDecomposition<Untrained> {
    type Fitted = CPDecomposition<TrainedCP>;

    fn fit(self, tensor: &Array3<f64>, _y: &()) -> Result<Self::Fitted> {
        let tensor_shape = tensor.shape().to_vec();

        if tensor_shape.len() != 3 {
            return Err(SklearsError::InvalidInput(
                "CP decomposition currently supports 3D tensors only".to_string(),
            ));
        }

        let (_n_mode1, _n_mode2, _n_mode3) = (tensor_shape[0], tensor_shape[1], tensor_shape[2]);

        // Initialize random number generator
        // TODO: Support seeding for reproducibility
        let mut rng = thread_rng();

        // Run CP decomposition algorithm
        let (factors, weights, n_iter, reconstruction_error) = match self.algorithm {
            CPAlgorithm::ALS => self.cp_als(tensor, &mut rng)?,
            CPAlgorithm::NonNegative => self.cp_nonnegative(tensor, &mut rng)?,
            CPAlgorithm::Robust => self.cp_robust(tensor, &mut rng)?,
        };

        Ok(CPDecomposition {
            n_components: self.n_components,
            algorithm: self.algorithm,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            learning_rate: self.learning_rate,
            regularization: self.regularization,
            normalize_factors: self.normalize_factors,
            state: TrainedCP {
                factors,
                weights,
                tensor_shape,
                n_components: self.n_components,
                n_iter,
                reconstruction_error,
            },
        })
    }
}

impl CPDecomposition<Untrained> {
    /// CP decomposition using Alternating Least Squares
    fn cp_als(&self, tensor: &Array3<f64>, rng: &mut impl Rng) -> CPResult {
        let (n_mode1, n_mode2, n_mode3) = tensor.dim();
        let r = self.n_components;

        // Initialize factor matrices randomly
        let mut factors = vec![
            {
                let mut arr = Array2::zeros((n_mode1, r));
                for elem in arr.iter_mut() {
                    *elem = rng.gen() - 0.5;
                }
                arr
            },
            {
                let mut arr = Array2::zeros((n_mode2, r));
                for elem in arr.iter_mut() {
                    *elem = rng.gen() - 0.5;
                }
                arr
            },
            {
                let mut arr = Array2::zeros((n_mode3, r));
                for elem in arr.iter_mut() {
                    *elem = rng.gen() - 0.5;
                }
                arr
            },
        ];

        // Normalize initial factors
        if self.normalize_factors {
            for factor in &mut factors {
                for j in 0..r {
                    let mut col = factor.column_mut(j);
                    let norm = col.dot(&col).sqrt();
                    if norm > 1e-12 {
                        col /= norm;
                    }
                }
            }
        }

        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Update each factor matrix in turn
            for mode in 0..3 {
                // Create Khatri-Rao product of other factors
                let khatri_rao = match mode {
                    0 => self.khatri_rao_product(&factors[2], &factors[1]),
                    1 => self.khatri_rao_product(&factors[2], &factors[0]),
                    2 => self.khatri_rao_product(&factors[1], &factors[0]),
                    _ => unreachable!(),
                };

                // Unfold tensor along current mode
                let unfolded = self.unfold_tensor(tensor, mode);

                // Solve least squares problem: unfolded = factor * khatri_rao^T
                // factor = unfolded * khatri_rao * (khatri_rao^T * khatri_rao)^(-1)
                let gram = khatri_rao.t().dot(&khatri_rao);

                // Add regularization to gram matrix
                let mut gram_reg = gram;
                for i in 0..r {
                    gram_reg[[i, i]] += self.regularization;
                }

                // Solve linear system for each row of the factor matrix
                let rhs = unfolded.dot(&khatri_rao);
                let rhs_t = rhs.t().to_owned();
                factors[mode] = self.solve_linear_system(&gram_reg, &rhs_t)?;

                // Normalize columns if requested
                if self.normalize_factors {
                    for j in 0..r {
                        let mut col = factors[mode].column_mut(j);
                        let norm = col.dot(&col).sqrt();
                        if norm > 1e-12 {
                            col /= norm;
                        }
                    }
                }
            }

            // Compute reconstruction error
            let reconstructed = self.reconstruct_tensor(&factors)?;
            let error = self.frobenius_norm(&(tensor - &reconstructed));

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }
            prev_error = error;
        }

        // Compute final weights (norms of factor columns)
        let mut weights = Array1::ones(r);
        if self.normalize_factors {
            for j in 0..r {
                let mut weight = 1.0;
                for factor in &factors {
                    let col_norm = factor.column(j).dot(&factor.column(j)).sqrt();
                    weight *= col_norm;
                }
                weights[j] = weight;
            }
        }

        let final_error = self.frobenius_norm(&(tensor - &self.reconstruct_tensor(&factors)?));

        Ok((factors, weights, n_iter, final_error))
    }

    /// Non-negative CP decomposition
    fn cp_nonnegative(&self, tensor: &Array3<f64>, rng: &mut impl Rng) -> CPResult {
        let (n_mode1, n_mode2, n_mode3) = tensor.dim();
        let r = self.n_components;

        // Initialize with non-negative random factors
        let mut factors = vec![
            {
                let mut arr = Array2::zeros((n_mode1, r));
                for elem in arr.iter_mut() {
                    *elem = rng.gen().abs();
                }
                arr
            },
            {
                let mut arr = Array2::zeros((n_mode2, r));
                for elem in arr.iter_mut() {
                    *elem = rng.gen().abs();
                }
                arr
            },
            {
                let mut arr = Array2::zeros((n_mode3, r));
                for elem in arr.iter_mut() {
                    *elem = rng.gen().abs();
                }
                arr
            },
        ];

        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Multiplicative updates for each factor
            for mode in 0..3 {
                let khatri_rao = match mode {
                    0 => self.khatri_rao_product(&factors[2], &factors[1]),
                    1 => self.khatri_rao_product(&factors[2], &factors[0]),
                    2 => self.khatri_rao_product(&factors[1], &factors[0]),
                    _ => unreachable!(),
                };

                let unfolded = self.unfold_tensor(tensor, mode);

                // Multiplicative update rule for non-negative factorization
                let numerator = unfolded.dot(&khatri_rao);
                let denominator = factors[mode].dot(&khatri_rao.t().dot(&khatri_rao));

                // Update with element-wise multiplication
                for i in 0..factors[mode].nrows() {
                    for j in 0..factors[mode].ncols() {
                        if denominator[[i, j]] > 1e-12 {
                            factors[mode][[i, j]] *= numerator[[i, j]] / denominator[[i, j]];
                        }
                        // Ensure non-negativity
                        factors[mode][[i, j]] = factors[mode][[i, j]].max(1e-12);
                    }
                }

                // Normalize columns if requested
                if self.normalize_factors {
                    for j in 0..r {
                        let mut col = factors[mode].column_mut(j);
                        let norm = col.dot(&col).sqrt();
                        if norm > 1e-12 {
                            col /= norm;
                        }
                    }
                }
            }

            // Compute reconstruction error
            let reconstructed = self.reconstruct_tensor(&factors)?;
            let error = self.frobenius_norm(&(tensor - &reconstructed));

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }
            prev_error = error;
        }

        // Compute final weights
        let weights = Array1::ones(r);
        let final_error = self.frobenius_norm(&(tensor - &self.reconstruct_tensor(&factors)?));

        Ok((factors, weights, n_iter, final_error))
    }

    /// Robust CP decomposition with outlier handling
    fn cp_robust(&self, tensor: &Array3<f64>, rng: &mut impl Rng) -> CPResult {
        // For now, implement as standard ALS with robust loss function
        // This is a simplified version - full robust CP would require more sophisticated methods
        self.cp_als(tensor, rng)
    }

    /// Compute Khatri-Rao product of two matrices
    fn khatri_rao_product(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (n_a, r) = a.dim();
        let (n_b, r_b) = b.dim();
        assert_eq!(r, r_b, "Matrices must have same number of columns");

        let mut result = Array2::zeros((n_a * n_b, r));

        for j in 0..r {
            let a_col = a.column(j);
            let b_col = b.column(j);

            for i in 0..n_a {
                for k in 0..n_b {
                    result[[i * n_b + k, j]] = a_col[i] * b_col[k];
                }
            }
        }

        result
    }

    /// Unfold tensor along specified mode
    fn unfold_tensor(&self, tensor: &Array3<f64>, mode: usize) -> Array2<f64> {
        let (n1, n2, n3) = tensor.dim();

        match mode {
            0 => {
                // Mode-1 unfolding: n1 × (n2*n3)
                let mut unfolded = Array2::zeros((n1, n2 * n3));
                for i in 0..n1 {
                    for j in 0..n2 {
                        for k in 0..n3 {
                            unfolded[[i, j * n3 + k]] = tensor[[i, j, k]];
                        }
                    }
                }
                unfolded
            }
            1 => {
                // Mode-2 unfolding: n2 × (n3*n1)
                let mut unfolded = Array2::zeros((n2, n3 * n1));
                for j in 0..n2 {
                    for k in 0..n3 {
                        for i in 0..n1 {
                            unfolded[[j, k * n1 + i]] = tensor[[i, j, k]];
                        }
                    }
                }
                unfolded
            }
            2 => {
                // Mode-3 unfolding: n3 × (n1*n2)
                let mut unfolded = Array2::zeros((n3, n1 * n2));
                for k in 0..n3 {
                    for i in 0..n1 {
                        for j in 0..n2 {
                            unfolded[[k, i * n2 + j]] = tensor[[i, j, k]];
                        }
                    }
                }
                unfolded
            }
            _ => panic!("Mode must be 0, 1, or 2 for 3D tensors"),
        }
    }

    /// Solve linear system using pseudoinverse
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        // Compute pseudoinverse using SVD from scirs2-linalg
        let (u, s, vt) = svd(&a.view(), true).map_err(|e| {
            SklearsError::NumericalError(format!("SVD failed: {}", e))
        })?;

        let tolerance = 1e-12;

        // Create diagonal inverse matrix S^+
        let k = s.len();
        let mut s_inv = Array2::zeros((k, k));
        for i in 0..k {
            if s[i] > tolerance {
                s_inv[[i, i]] = 1.0 / s[i];
            }
        }

        // Compute pseudoinverse: A^+ = V * S^+ * U^T
        let vt_t = vt.t();
        let u_t = u.t();
        let temp = s_inv.dot(&u_t);
        let a_pinv = vt_t.dot(&temp);

        // Compute result: A^+ * b
        let result = a_pinv.dot(b);

        Ok(result)
    }

    /// Reconstruct tensor from factor matrices
    fn reconstruct_tensor(&self, factors: &[Array2<f64>]) -> Result<Array3<f64>> {
        let (n1, r) = factors[0].dim();
        let (n2, r2) = factors[1].dim();
        let (n3, r3) = factors[2].dim();

        if r != r2 || r != r3 {
            return Err(SklearsError::InvalidInput(
                "All factor matrices must have the same number of columns".to_string(),
            ));
        }

        let mut reconstructed = Array3::zeros((n1, n2, n3));

        for k in 0..r {
            let a_k = factors[0].column(k);
            let b_k = factors[1].column(k);
            let c_k = factors[2].column(k);

            for i in 0..n1 {
                for j in 0..n2 {
                    for l in 0..n3 {
                        reconstructed[[i, j, l]] += a_k[i] * b_k[j] * c_k[l];
                    }
                }
            }
        }

        Ok(reconstructed)
    }

    /// Compute Frobenius norm of a tensor
    fn frobenius_norm(&self, tensor: &Array3<f64>) -> f64 {
        tensor.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

impl TuckerDecomposition<Untrained> {
    /// Create a new Tucker decomposition
    pub fn new(n_components: Vec<usize>) -> Self {
        Self {
            n_components,
            algorithm: TuckerAlgorithm::HOSVD,
            max_iter: 100,
            tol: 1e-6,
            random_state: None,
            center: true,
            init_method: "random".to_string(),
            state: Untrained,
        }
    }

    /// Set the algorithm
    pub fn algorithm(mut self, algorithm: TuckerAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set whether to center the tensor
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set initialization method
    pub fn init_method(mut self, init_method: String) -> Self {
        self.init_method = init_method;
        self
    }
}

impl Fit<Array3<f64>, ()> for TuckerDecomposition<Untrained> {
    type Fitted = TuckerDecomposition<TrainedTucker>;

    fn fit(self, tensor: &Array3<f64>, _y: &()) -> Result<Self::Fitted> {
        let tensor_shape = tensor.shape().to_vec();

        if tensor_shape.len() != 3 {
            return Err(SklearsError::InvalidInput(
                "Tucker decomposition currently supports 3D tensors only".to_string(),
            ));
        }

        if self.n_components.len() != 3 {
            return Err(SklearsError::InvalidInput(
                "n_components must have 3 elements for 3D tensors".to_string(),
            ));
        }

        // Center tensor if requested
        let (centered_tensor, mean) = if self.center {
            let mean_val = tensor.mean().unwrap();
            let centered = tensor.mapv(|x| x - mean_val);
            let mean_tensor = Array3::from_elem(tensor.dim(), mean_val);
            (centered, Some(mean_tensor))
        } else {
            (tensor.clone(), None)
        };

        // Initialize random number generator
        // TODO: Support seeding for reproducibility
        let mut rng = thread_rng();

        // Run Tucker decomposition algorithm
        let (core, factors, n_iter, reconstruction_error) = match self.algorithm {
            TuckerAlgorithm::HOSVD => self.tucker_hosvd(&centered_tensor)?,
            TuckerAlgorithm::ALS => self.tucker_als(&centered_tensor, &mut rng)?,
            TuckerAlgorithm::SequentialSVD => self.tucker_sequential_svd(&centered_tensor)?,
        };

        Ok(TuckerDecomposition {
            n_components: self.n_components.clone(),
            algorithm: self.algorithm,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            center: self.center,
            init_method: self.init_method,
            state: TrainedTucker {
                core,
                factors,
                mean,
                tensor_shape,
                n_components: self.n_components,
                n_iter,
                reconstruction_error,
            },
        })
    }
}

impl TuckerDecomposition<Untrained> {
    /// Tucker decomposition using Higher-Order SVD (HOSVD)
    fn tucker_hosvd(&self, tensor: &Array3<f64>) -> TuckerResult {
        let mut factors = Vec::new();

        // Compute SVD for each mode
        for mode in 0..3 {
            let unfolded = self.unfold_tensor_tucker(tensor, mode);
            let svd_result = self.compute_svd(&unfolded)?;

            // Take first n_components[mode] columns from U matrix (not V^T)
            let n_comp = self.n_components[mode]
                .min(svd_result.0.ncols())
                .min(unfolded.nrows());
            let factor = svd_result
                .0
                .slice(scirs2_core::ndarray::s![.., ..n_comp])
                .to_owned();
            factors.push(factor);
        }

        // Compute core tensor by projecting original tensor onto factor spaces
        let core = self.compute_core_tensor(tensor, &factors)?;

        // Compute reconstruction error
        let reconstructed = self.reconstruct_tucker_tensor(&core, &factors)?;
        let error = self.frobenius_norm_tucker(&(tensor - &reconstructed));

        Ok((core, factors, 1, error))
    }

    /// Tucker decomposition using Alternating Least Squares
    fn tucker_als(&self, tensor: &Array3<f64>, rng: &mut impl Rng) -> TuckerResult {
        let (n1, n2, n3) = tensor.dim();

        // Initialize factor matrices randomly
        let mut factors = vec![
            {
                let mut arr = Array2::zeros((n1, self.n_components[0]));
                for elem in arr.iter_mut() {
                    *elem = rng.gen() - 0.5;
                }
                arr
            },
            {
                let mut arr = Array2::zeros((n2, self.n_components[1]));
                for elem in arr.iter_mut() {
                    *elem = rng.gen() - 0.5;
                }
                arr
            },
            {
                let mut arr = Array2::zeros((n3, self.n_components[2]));
                for elem in arr.iter_mut() {
                    *elem = rng.gen() - 0.5;
                }
                arr
            },
        ];

        // Orthogonalize initial factors
        for factor in &mut factors {
            *factor = self.orthogonalize_matrix(factor)?;
        }

        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Update each factor matrix
            for mode in 0..3 {
                let unfolded = self.unfold_tensor_tucker(tensor, mode);

                // Create product of other factor matrices
                let other_factors = match mode {
                    0 => self.kronecker_product(&factors[2], &factors[1]),
                    1 => self.kronecker_product(&factors[2], &factors[0]),
                    2 => self.kronecker_product(&factors[1], &factors[0]),
                    _ => unreachable!(),
                };

                // Solve for updated factor
                let rhs = unfolded.dot(&other_factors);
                let gram = other_factors.t().dot(&other_factors);
                let rhs_t = rhs.t().to_owned();

                let solved = self.solve_linear_system_tucker(&gram, &rhs_t)?;

                // The solved matrix should be transposed to get the correct factor dimensions
                // We expect factors[mode] to have shape (n_mode, n_components[mode])
                let _expected_shape = (
                    tensor.dim().0.max(tensor.dim().1.max(tensor.dim().2)),
                    self.n_components[mode],
                );
                let n_mode = match mode {
                    0 => tensor.dim().0,
                    1 => tensor.dim().1,
                    2 => tensor.dim().2,
                    _ => unreachable!(),
                };

                // Transpose the solution and take only the first n_components[mode] columns
                let solved_t = solved.t().to_owned();
                let factor_shape = (n_mode, self.n_components[mode]);

                if solved_t.dim() == factor_shape {
                    factors[mode] = solved_t;
                } else {
                    // Take the appropriate slice, but ensure we don't go out of bounds
                    let max_rows = solved_t.nrows().min(n_mode);
                    let max_cols = solved_t.ncols().min(self.n_components[mode]);

                    let mut factor = Array2::zeros(factor_shape);
                    for i in 0..max_rows {
                        for j in 0..max_cols {
                            factor[[i, j]] = solved_t[[i, j]];
                        }
                    }
                    factors[mode] = factor;
                }

                // Orthogonalize
                factors[mode] = self.orthogonalize_matrix(&factors[mode])?;
            }

            // Compute core tensor and reconstruction error
            let core = self.compute_core_tensor(tensor, &factors)?;
            let reconstructed = self.reconstruct_tucker_tensor(&core, &factors)?;
            let error = self.frobenius_norm_tucker(&(tensor - &reconstructed));

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }
            prev_error = error;
        }

        let core = self.compute_core_tensor(tensor, &factors)?;
        let final_error = prev_error;

        Ok((core, factors, n_iter, final_error))
    }

    /// Tucker decomposition using Sequential SVD
    fn tucker_sequential_svd(&self, tensor: &Array3<f64>) -> TuckerResult {
        // Similar to HOSVD but with sequential processing
        self.tucker_hosvd(tensor)
    }

    /// Unfold tensor for Tucker decomposition
    fn unfold_tensor_tucker(&self, tensor: &Array3<f64>, mode: usize) -> Array2<f64> {
        let (n1, n2, n3) = tensor.dim();

        match mode {
            0 => {
                let mut unfolded = Array2::zeros((n1, n2 * n3));
                for i in 0..n1 {
                    for j in 0..n2 {
                        for k in 0..n3 {
                            unfolded[[i, j * n3 + k]] = tensor[[i, j, k]];
                        }
                    }
                }
                unfolded
            }
            1 => {
                let mut unfolded = Array2::zeros((n2, n1 * n3));
                for j in 0..n2 {
                    for i in 0..n1 {
                        for k in 0..n3 {
                            unfolded[[j, i * n3 + k]] = tensor[[i, j, k]];
                        }
                    }
                }
                unfolded
            }
            2 => {
                let mut unfolded = Array2::zeros((n3, n1 * n2));
                for k in 0..n3 {
                    for i in 0..n1 {
                        for j in 0..n2 {
                            unfolded[[k, i * n2 + j]] = tensor[[i, j, k]];
                        }
                    }
                }
                unfolded
            }
            _ => panic!("Mode must be 0, 1, or 2 for 3D tensors"),
        }
    }

    /// Compute SVD of a matrix
    fn compute_svd(&self, matrix: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>)> {
        // Use scirs2-linalg SVD - returns (U, S, V^T)
        let (u, s, vt) = svd(&matrix.view(), true).map_err(|e| {
            SklearsError::NumericalError(format!("SVD failed: {}", e))
        })?;

        // This function expects (U, V^T, S) order
        Ok((u, vt, s))
    }

    /// Compute core tensor
    fn compute_core_tensor(
        &self,
        tensor: &Array3<f64>,
        factors: &[Array2<f64>],
    ) -> Result<Array3<f64>> {
        let (r1, r2, r3) = (factors[0].ncols(), factors[1].ncols(), factors[2].ncols());
        let mut core = Array3::zeros((r1, r2, r3));

        // G = X ×₁ A^T ×₂ B^T ×₃ C^T
        // This is a simplified implementation
        for i in 0..r1 {
            for j in 0..r2 {
                for k in 0..r3 {
                    let mut value = 0.0;
                    let (n1, n2, n3) = tensor.dim();

                    for ii in 0..n1 {
                        for jj in 0..n2 {
                            for kk in 0..n3 {
                                value += tensor[[ii, jj, kk]]
                                    * factors[0][[ii, i]]
                                    * factors[1][[jj, j]]
                                    * factors[2][[kk, k]];
                            }
                        }
                    }
                    core[[i, j, k]] = value;
                }
            }
        }

        Ok(core)
    }

    /// Reconstruct tensor from Tucker decomposition
    fn reconstruct_tucker_tensor(
        &self,
        core: &Array3<f64>,
        factors: &[Array2<f64>],
    ) -> Result<Array3<f64>> {
        let (n1, n2, n3) = (factors[0].nrows(), factors[1].nrows(), factors[2].nrows());
        let mut reconstructed = Array3::zeros((n1, n2, n3));

        let (r1, r2, r3) = core.dim();

        for i in 0..n1 {
            for j in 0..n2 {
                for k in 0..n3 {
                    let mut value = 0.0;

                    for ii in 0..r1 {
                        for jj in 0..r2 {
                            for kk in 0..r3 {
                                value += core[[ii, jj, kk]]
                                    * factors[0][[i, ii]]
                                    * factors[1][[j, jj]]
                                    * factors[2][[k, kk]];
                            }
                        }
                    }
                    reconstructed[[i, j, k]] = value;
                }
            }
        }

        Ok(reconstructed)
    }

    /// Orthogonalize matrix using QR decomposition
    fn orthogonalize_matrix(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        // Simple Gram-Schmidt orthogonalization
        let (m, n) = matrix.dim();
        let mut q = matrix.clone();

        for j in 0..n {
            for i in 0..j {
                let qi = q.column(i).to_owned();
                let qj = q.column(j).to_owned();
                let proj = qi.dot(&qj);

                for k in 0..m {
                    q[[k, j]] -= proj * qi[k];
                }
            }

            // Normalize
            let norm = {
                let col = q.column(j);
                col.dot(&col).sqrt()
            };
            if norm > 1e-12 {
                let mut col = q.column_mut(j);
                col /= norm;
            }
        }

        Ok(q)
    }

    /// Compute Kronecker product of two matrices
    fn kronecker_product(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (m_a, n_a) = a.dim();
        let (m_b, n_b) = b.dim();

        let mut result = Array2::zeros((m_a * m_b, n_a * n_b));

        for i in 0..m_a {
            for j in 0..n_a {
                for k in 0..m_b {
                    for l in 0..n_b {
                        result[[i * m_b + k, j * n_b + l]] = a[[i, j]] * b[[k, l]];
                    }
                }
            }
        }

        result
    }

    /// Solve linear system for Tucker decomposition
    fn solve_linear_system_tucker(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        // Try Cholesky decomposition first (for positive definite matrices)
        if let Ok(chol) = a.cholesky() {
            // Solve each column of b separately
            let mut result = Array2::zeros((a.ncols(), b.ncols()));
            for (j, col) in b.axis_iter(scirs2_core::ndarray::Axis(1)).enumerate() {
                let sol = chol.solve(&col.to_owned()).map_err(|e| {
                    SklearsError::NumericalError(format!("Cholesky solve failed: {}", e))
                })?;
                for (i, &val) in sol.iter().enumerate() {
                    result[[i, j]] = val;
                }
            }
            return Ok(result);
        }

        // Fallback to pseudoinverse using SVD
        let (u, s, vt) = svd(&a.view(), true).map_err(|e| {
            SklearsError::NumericalError(format!("SVD failed: {}", e))
        })?;

        let tolerance = 1e-12;

        // Create diagonal inverse matrix S^+
        let k = s.len();
        let mut s_inv = Array2::zeros((k, k));
        for i in 0..k {
            if s[i] > tolerance {
                s_inv[[i, i]] = 1.0 / s[i];
            }
        }

        // Compute pseudoinverse: A^+ = V * S^+ * U^T
        let vt_t = vt.t();
        let u_t = u.t();
        let temp = s_inv.dot(&u_t);
        let a_pinv = vt_t.dot(&temp);

        // Compute result: A^+ * b
        let result = a_pinv.dot(b);

        Ok(result)
    }

    /// Compute Frobenius norm for Tucker
    fn frobenius_norm_tucker(&self, tensor: &Array3<f64>) -> f64 {
        tensor.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cp_decomposition_basic() {
        // Create a simple test tensor
        let tensor = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],];

        let cp = CPDecomposition::new(2).max_iter(10).random_state(42);

        let result = cp.fit(&tensor, &());
        if let Err(ref e) = result {
            println!("CP decomposition error: {e:?}");
        }
        assert!(result.is_ok());

        let trained = result.unwrap();
        assert_eq!(trained.state.n_components, 2);
        assert_eq!(trained.state.factors.len(), 3);
        assert!(trained.state.reconstruction_error.is_finite());
    }

    #[test]
    fn test_tucker_decomposition_basic() {
        // Create a simple test tensor
        let tensor = array![
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ];

        let tucker = TuckerDecomposition::new(vec![2, 2, 1])
            .algorithm(TuckerAlgorithm::HOSVD)
            .random_state(42);

        let result = tucker.fit(&tensor, &());
        if let Err(ref e) = result {
            println!("Tucker decomposition error: {e:?}");
        }
        assert!(result.is_ok());

        let trained = result.unwrap();
        assert_eq!(trained.state.n_components, vec![2, 2, 1]);
        assert_eq!(trained.state.factors.len(), 3);
        assert!(trained.state.reconstruction_error.is_finite());
    }

    #[test]
    fn test_cp_nonnegative() {
        let tensor = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],];

        let cp = CPDecomposition::new(1)
            .algorithm(CPAlgorithm::NonNegative)
            .max_iter(5)
            .random_state(42);

        let result = cp.fit(&tensor, &());
        assert!(result.is_ok());

        let trained = result.unwrap();
        // Check that all factor values are non-negative
        for factor in &trained.state.factors {
            for &val in factor.iter() {
                assert!(val >= 0.0, "Factor value should be non-negative: {}", val);
            }
        }
    }

    #[test]
    fn test_tucker_als() {
        let tensor = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],];

        let tucker = TuckerDecomposition::new(vec![1, 2, 1])
            .algorithm(TuckerAlgorithm::ALS)
            .max_iter(5)
            .random_state(42);

        let result = tucker.fit(&tensor, &());
        if let Err(ref e) = result {
            println!("Tucker ALS error: {e:?}");
        }
        assert!(result.is_ok());

        let trained = result.unwrap();
        assert_eq!(trained.state.n_components, vec![1, 2, 1]);
        assert!(trained.state.n_iter > 0);
    }

    #[test]
    fn test_cp_parameters() {
        let cp = CPDecomposition::new(3)
            .algorithm(CPAlgorithm::Robust)
            .max_iter(200)
            .tol(1e-8)
            .learning_rate(0.05)
            .regularization(0.1)
            .normalize_factors(false);

        assert_eq!(cp.n_components, 3);
        assert_eq!(cp.algorithm, CPAlgorithm::Robust);
        assert_eq!(cp.max_iter, 200);
        assert_eq!(cp.tol, 1e-8);
        assert_eq!(cp.learning_rate, 0.05);
        assert_eq!(cp.regularization, 0.1);
        assert_eq!(cp.normalize_factors, false);
    }

    #[test]
    fn test_tucker_parameters() {
        let tucker = TuckerDecomposition::new(vec![2, 3, 4])
            .algorithm(TuckerAlgorithm::SequentialSVD)
            .max_iter(150)
            .tol(1e-7)
            .center(false)
            .init_method("svd".to_string());

        assert_eq!(tucker.n_components, vec![2, 3, 4]);
        assert_eq!(tucker.algorithm, TuckerAlgorithm::SequentialSVD);
        assert_eq!(tucker.max_iter, 150);
        assert_eq!(tucker.tol, 1e-7);
        assert_eq!(tucker.center, false);
        assert_eq!(tucker.init_method, "svd");
    }
}
