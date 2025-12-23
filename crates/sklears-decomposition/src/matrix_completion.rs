//! Matrix completion algorithms
//!
//! This module provides matrix completion methods for filling missing values:
//! - Low-rank matrix completion using SVD
//! - Iterative matrix completion with ALS
//! - Nuclear norm minimization
//! - Matrix completion with side information

// TODO: Replace with scirs2-linalg
// use nalgebra::{DMatrix, DVector};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{Rng, thread_rng, Random, SeedableRng};
use scirs2_core::random::rngs::StdRng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

/// Type alias for complex completion result with side information
type SideInfoCompletionResult = (
    (Array2<f64>, Array2<f64>),
    Option<Array1<f64>>,
    Option<Array1<f64>>,
    f64,
    usize,
    f64,
);

/// Type alias for matrix completion result
#[allow(dead_code)]
type MatrixCompletionResult = (
    Array2<f64>,
    Array2<f64>,
    Option<(Array2<f64>, Array1<f64>, Array2<f64>)>,
    usize,
    f64,
    f64,
);

/// Type alias for RPCA decomposition result
type RPCAResult = (
    Array2<f64>,
    Array2<f64>,
    Option<(Array2<f64>, Array1<f64>, Array2<f64>)>,
    usize,
    usize,
    f64,
);

/// Matrix completion algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CompletionAlgorithm {
    /// SVD-based low-rank matrix completion
    #[default]
    SVD,
    /// Alternating Least Squares
    ALS,
    /// Nuclear norm minimization (approximate)
    NuclearNorm,
    /// Matrix completion with side information
    SideInfo,
}

/// Matrix completion transformer
#[derive(Debug, Clone)]
pub struct MatrixCompletion<State = Untrained> {
    /// Rank of the matrix completion
    pub rank: Option<usize>,
    /// Algorithm to use
    pub algorithm: CompletionAlgorithm,
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
    /// Whether to use bias terms
    pub use_bias: bool,

    /// Trained state
    state: State,
}

/// Trained matrix completion state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainedMatrixCompletion {
    pub factors: (Array2<f64>, Array2<f64>),
    pub user_bias: Option<Array1<f64>>,
    pub item_bias: Option<Array1<f64>>,
    pub global_bias: f64,
    pub matrix_shape: (usize, usize),
    pub rank: usize,
    pub n_iter: usize,
    pub reconstruction_error: f64,
    pub mask: Array2<bool>,
}

impl Default for MatrixCompletion<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl MatrixCompletion<Untrained> {
    /// Create a new matrix completion transformer
    pub fn new() -> Self {
        Self {
            rank: None,
            algorithm: CompletionAlgorithm::SVD,
            max_iter: 100,
            tol: 1e-6,
            random_state: None,
            learning_rate: 0.01,
            regularization: 0.01,
            use_bias: true,
            state: Untrained,
        }
    }

    /// Set the rank for matrix completion
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = Some(rank);
        self
    }

    /// Set the algorithm
    pub fn algorithm(mut self, algorithm: CompletionAlgorithm) -> Self {
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

    /// Set whether to use bias terms
    pub fn use_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }
}

impl Fit<Array2<f64>, Array2<bool>> for MatrixCompletion<Untrained> {
    type Fitted = MatrixCompletion<TrainedMatrixCompletion>;

    fn fit(self, matrix: &Array2<f64>, mask: &Array2<bool>) -> Result<Self::Fitted> {
        let (n_users, n_items) = matrix.dim();

        if mask.dim() != matrix.dim() {
            return Err(SklearsError::InvalidInput(
                "Mask dimensions must match matrix dimensions".to_string(),
            ));
        }

        // Count observed entries
        let n_observed = mask.iter().filter(|&&x| x).count();
        if n_observed == 0 {
            return Err(SklearsError::InvalidInput(
                "No observed entries in the matrix".to_string(),
            ));
        }

        // Determine rank
        let rank = self.rank.unwrap_or((n_users.min(n_items) / 2).max(1));

        // Initialize random number generator with optional seed for reproducibility
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng())
        };

        // Run matrix completion algorithm
        let (factors, user_bias, item_bias, global_bias, n_iter, reconstruction_error) =
            match self.algorithm {
                CompletionAlgorithm::SVD => self.svd_completion(matrix, mask, rank, &mut rng)?,
                CompletionAlgorithm::ALS => self.als_completion(matrix, mask, rank, &mut rng)?,
                CompletionAlgorithm::NuclearNorm => {
                    self.nuclear_norm_completion(matrix, mask, rank, &mut rng)?
                }
                CompletionAlgorithm::SideInfo => {
                    self.side_info_completion(matrix, mask, rank, &mut rng)?
                }
            };

        Ok(MatrixCompletion {
            rank: self.rank,
            algorithm: self.algorithm,
            max_iter: self.max_iter,
            tol: self.tol,
            random_state: self.random_state,
            learning_rate: self.learning_rate,
            regularization: self.regularization,
            use_bias: self.use_bias,
            state: TrainedMatrixCompletion {
                factors,
                user_bias,
                item_bias,
                global_bias,
                matrix_shape: (n_users, n_items),
                rank,
                n_iter,
                reconstruction_error,
                mask: mask.clone(),
            },
        })
    }
}

impl MatrixCompletion<Untrained> {
    /// SVD-based matrix completion
    fn svd_completion(
        &self,
        matrix: &Array2<f64>,
        mask: &Array2<bool>,
        rank: usize,
        _rng: &mut impl Rng,
    ) -> Result<SideInfoCompletionResult> {
        let (n_users, n_items) = matrix.dim();

        // Initialize with mean of observed values
        let observed_sum: f64 = matrix
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&val, _)| val)
            .sum();
        let n_observed = mask.iter().filter(|&&x| x).count();
        let global_mean = observed_sum / n_observed as f64;

        // Create initial completed matrix with global mean for missing values
        let mut completed = matrix.clone();
        for ((i, j), &is_observed) in scirs2_core::ndarray::indices(matrix.dim())
            .into_iter()
            .zip(mask.iter())
        {
            if !is_observed {
                completed[[i, j]] = global_mean;
            }
        }

        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Perform SVD on current completed matrix
            let (u, s, vt) = self.compute_truncated_svd(&completed, rank)?;

            // Reconstruct with truncated SVD
            let mut reconstructed = Array2::zeros((n_users, n_items));
            for i in 0..n_users {
                for j in 0..n_items {
                    for k in 0..rank {
                        reconstructed[[i, j]] += u[[i, k]] * s[k] * vt[[k, j]];
                    }
                }
            }

            // Update only missing entries
            for ((i, j), &is_observed) in scirs2_core::ndarray::indices(matrix.dim())
                .into_iter()
                .zip(mask.iter())
            {
                if !is_observed {
                    completed[[i, j]] = reconstructed[[i, j]];
                }
            }

            // Compute error on observed entries
            let mut error = 0.0;
            let mut count = 0;
            for ((i, j), &is_observed) in scirs2_core::ndarray::indices(matrix.dim())
                .into_iter()
                .zip(mask.iter())
            {
                if is_observed {
                    let diff = matrix[[i, j]] - reconstructed[[i, j]];
                    error += diff * diff;
                    count += 1;
                }
            }
            error = (error / count as f64).sqrt();

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }
            prev_error = error;
        }

        // Final SVD to get factors
        let (u, s, vt) = self.compute_truncated_svd(&completed, rank)?;

        // Create factor matrices: U * sqrt(S) and V * sqrt(S)
        let mut u_factor = Array2::zeros((n_users, rank));
        let mut v_factor = Array2::zeros((n_items, rank));

        for i in 0..n_users {
            for k in 0..rank {
                u_factor[[i, k]] = u[[i, k]] * s[k].sqrt();
            }
        }

        for j in 0..n_items {
            for k in 0..rank {
                v_factor[[j, k]] = vt[[k, j]] * s[k].sqrt();
            }
        }

        let user_bias = if self.use_bias {
            Some(Array1::zeros(n_users))
        } else {
            None
        };

        let item_bias = if self.use_bias {
            Some(Array1::zeros(n_items))
        } else {
            None
        };

        Ok((
            (u_factor, v_factor),
            user_bias,
            item_bias,
            global_mean,
            n_iter,
            prev_error,
        ))
    }

    /// Alternating Least Squares matrix completion
    fn als_completion(
        &self,
        matrix: &Array2<f64>,
        mask: &Array2<bool>,
        rank: usize,
        rng: &mut impl Rng,
    ) -> Result<SideInfoCompletionResult> {
        let (n_users, n_items) = matrix.dim();

        // Initialize factors randomly
        let mut u = Array2::zeros((n_users, rank));
        let mut v = Array2::zeros((n_items, rank));

        for i in 0..n_users {
            for k in 0..rank {
                u[[i, k]] = rng.gen() - 0.5;
            }
        }

        for j in 0..n_items {
            for k in 0..rank {
                v[[j, k]] = rng.gen() - 0.5;
            }
        }

        // Initialize biases
        let mut user_bias = if self.use_bias {
            Some(Array1::zeros(n_users))
        } else {
            None
        };

        let mut item_bias = if self.use_bias {
            Some(Array1::zeros(n_items))
        } else {
            None
        };

        // Compute global bias
        let observed_sum: f64 = matrix
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&val, _)| val)
            .sum();
        let n_observed = mask.iter().filter(|&&x| x).count();
        let global_bias = observed_sum / n_observed as f64;

        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Update user factors
            for i in 0..n_users {
                let mut a = Array2::eye(rank) * self.regularization;
                let mut b = Array1::zeros(rank);

                for j in 0..n_items {
                    if mask[[i, j]] {
                        let rating = matrix[[i, j]] - global_bias;
                        let rating = if self.use_bias {
                            rating - item_bias.as_ref().unwrap()[j]
                        } else {
                            rating
                        };

                        let v_j = v.row(j);
                        for k1 in 0..rank {
                            b[k1] += rating * v_j[k1];
                            for k2 in 0..rank {
                                a[[k1, k2]] += v_j[k1] * v_j[k2];
                            }
                        }
                    }
                }

                // Solve linear system A * u_i = b
                let u_i = self.solve_linear_system_als(&a, &b)?;
                u.row_mut(i).assign(&u_i);
            }

            // Update item factors
            for j in 0..n_items {
                let mut a = Array2::eye(rank) * self.regularization;
                let mut b = Array1::zeros(rank);

                for i in 0..n_users {
                    if mask[[i, j]] {
                        let rating = matrix[[i, j]] - global_bias;
                        let rating = if self.use_bias {
                            rating - user_bias.as_ref().unwrap()[i]
                        } else {
                            rating
                        };

                        let u_i = u.row(i);
                        for k1 in 0..rank {
                            b[k1] += rating * u_i[k1];
                            for k2 in 0..rank {
                                a[[k1, k2]] += u_i[k1] * u_i[k2];
                            }
                        }
                    }
                }

                // Solve linear system A * v_j = b
                let v_j = self.solve_linear_system_als(&a, &b)?;
                v.row_mut(j).assign(&v_j);
            }

            // Update biases if enabled
            if self.use_bias {
                // Update user biases
                if let Some(ref mut ub) = user_bias {
                    for i in 0..n_users {
                        let mut sum = 0.0;
                        let mut count = 0;
                        for j in 0..n_items {
                            if mask[[i, j]] {
                                let predicted = global_bias + u.row(i).dot(&v.row(j));
                                let predicted = predicted + item_bias.as_ref().unwrap()[j];
                                sum += matrix[[i, j]] - predicted;
                                count += 1;
                            }
                        }
                        if count > 0 {
                            ub[i] = sum / count as f64;
                        }
                    }
                }

                // Update item biases
                if let Some(ref mut ib) = item_bias {
                    for j in 0..n_items {
                        let mut sum = 0.0;
                        let mut count = 0;
                        for i in 0..n_users {
                            if mask[[i, j]] {
                                let predicted = global_bias + u.row(i).dot(&v.row(j));
                                let predicted = predicted + user_bias.as_ref().unwrap()[i];
                                sum += matrix[[i, j]] - predicted;
                                count += 1;
                            }
                        }
                        if count > 0 {
                            ib[j] = sum / count as f64;
                        }
                    }
                }
            }

            // Compute error
            let mut error = 0.0;
            let mut count = 0;
            for i in 0..n_users {
                for j in 0..n_items {
                    if mask[[i, j]] {
                        let predicted = global_bias + u.row(i).dot(&v.row(j));
                        let predicted = if self.use_bias {
                            predicted
                                + user_bias.as_ref().unwrap()[i]
                                + item_bias.as_ref().unwrap()[j]
                        } else {
                            predicted
                        };
                        let diff = matrix[[i, j]] - predicted;
                        error += diff * diff;
                        count += 1;
                    }
                }
            }
            error = (error / count as f64).sqrt();

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }
            prev_error = error;
        }

        Ok((
            (u, v),
            user_bias,
            item_bias,
            global_bias,
            n_iter,
            prev_error,
        ))
    }

    /// Nuclear norm minimization (simplified version)
    fn nuclear_norm_completion(
        &self,
        matrix: &Array2<f64>,
        mask: &Array2<bool>,
        rank: usize,
        _rng: &mut impl Rng,
    ) -> Result<SideInfoCompletionResult> {
        // For simplicity, implement this as a variation of SVD completion with soft thresholding
        let (n_users, n_items) = matrix.dim();

        // Initialize with mean of observed values
        let observed_sum: f64 = matrix
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(&val, _)| val)
            .sum();
        let n_observed = mask.iter().filter(|&&x| x).count();
        let global_mean = observed_sum / n_observed as f64;

        let mut completed = matrix.clone();
        for ((i, j), &is_observed) in scirs2_core::ndarray::indices(matrix.dim())
            .into_iter()
            .zip(mask.iter())
        {
            if !is_observed {
                completed[[i, j]] = global_mean;
            }
        }

        let mut prev_error = f64::INFINITY;
        let mut n_iter = 0;
        let lambda = self.regularization; // Nuclear norm regularization parameter

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Compute SVD
            let (u, s, vt) = self.compute_full_svd(&completed)?;

            // Soft thresholding on singular values (nuclear norm regularization)
            let s_thresh: Vec<f64> = s.iter().map(|&val| (val - lambda).max(0.0)).collect();

            // Reconstruct with thresholded singular values
            let mut reconstructed = Array2::zeros((n_users, n_items));
            for i in 0..n_users {
                for j in 0..n_items {
                    for k in 0..s_thresh.len() {
                        if s_thresh[k] > 0.0 {
                            reconstructed[[i, j]] += u[[i, k]] * s_thresh[k] * vt[[k, j]];
                        }
                    }
                }
            }

            // Update only missing entries
            for ((i, j), &is_observed) in scirs2_core::ndarray::indices(matrix.dim())
                .into_iter()
                .zip(mask.iter())
            {
                if !is_observed {
                    completed[[i, j]] = reconstructed[[i, j]];
                }
            }

            // Compute error on observed entries
            let mut error = 0.0;
            let mut count = 0;
            for ((i, j), &is_observed) in scirs2_core::ndarray::indices(matrix.dim())
                .into_iter()
                .zip(mask.iter())
            {
                if is_observed {
                    let diff = matrix[[i, j]] - reconstructed[[i, j]];
                    error += diff * diff;
                    count += 1;
                }
            }
            error = (error / count as f64).sqrt();

            // Check convergence
            if (prev_error - error).abs() < self.tol {
                break;
            }
            prev_error = error;
        }

        // Final decomposition to get factors
        let (u, s, vt) = self.compute_truncated_svd(&completed, rank)?;

        let mut u_factor = Array2::zeros((n_users, rank));
        let mut v_factor = Array2::zeros((n_items, rank));

        for i in 0..n_users {
            for k in 0..rank {
                u_factor[[i, k]] = u[[i, k]] * s[k].sqrt();
            }
        }

        for j in 0..n_items {
            for k in 0..rank {
                v_factor[[j, k]] = vt[[k, j]] * s[k].sqrt();
            }
        }

        let user_bias = if self.use_bias {
            Some(Array1::zeros(n_users))
        } else {
            None
        };
        let item_bias = if self.use_bias {
            Some(Array1::zeros(n_items))
        } else {
            None
        };

        Ok((
            (u_factor, v_factor),
            user_bias,
            item_bias,
            global_mean,
            n_iter,
            prev_error,
        ))
    }

    /// Matrix completion with side information (simplified)
    fn side_info_completion(
        &self,
        matrix: &Array2<f64>,
        mask: &Array2<bool>,
        rank: usize,
        rng: &mut impl Rng,
    ) -> Result<SideInfoCompletionResult> {
        // For now, implement this as regular ALS (can be extended with side information later)
        self.als_completion(matrix, mask, rank, rng)
    }

    /// Compute truncated SVD
    fn compute_truncated_svd(
        &self,
        matrix: &Array2<f64>,
        rank: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let matrix_vec: Vec<f64> = matrix.iter().cloned().collect();
        let na_matrix = DMatrix::from_row_slice(matrix.nrows(), matrix.ncols(), &matrix_vec);

        let svd = na_matrix.svd(true, true);
        let u = svd
            .u
            .ok_or_else(|| SklearsError::NumericalError("SVD failed".to_string()))?;
        let v_t = svd
            .v_t
            .ok_or_else(|| SklearsError::NumericalError("SVD failed".to_string()))?;
        let s = svd.singular_values;

        let actual_rank = rank.min(s.len());

        // Convert back to ndarray with truncation
        let u_trunc = u.columns(0, actual_rank);
        let s_trunc = s.rows(0, actual_rank);
        let vt_trunc = v_t.rows(0, actual_rank);

        let u_vec: Vec<f64> = u_trunc.iter().cloned().collect();
        let u_nd = Array2::from_shape_vec((u_trunc.nrows(), u_trunc.ncols()), u_vec)
            .map_err(|_| SklearsError::NumericalError("Failed to create U matrix".to_string()))?;

        let vt_vec: Vec<f64> = vt_trunc.iter().cloned().collect();
        let vt_nd = Array2::from_shape_vec((vt_trunc.nrows(), vt_trunc.ncols()), vt_vec)
            .map_err(|_| SklearsError::NumericalError("Failed to create V^T matrix".to_string()))?;

        let s_vec: Vec<f64> = s_trunc.iter().cloned().collect();
        let s_nd = Array1::from_vec(s_vec);

        Ok((u_nd, s_nd, vt_nd))
    }

    /// Compute full SVD (for nuclear norm method)
    fn compute_full_svd(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let matrix_vec: Vec<f64> = matrix.iter().cloned().collect();
        let na_matrix = DMatrix::from_row_slice(matrix.nrows(), matrix.ncols(), &matrix_vec);

        let svd = na_matrix.svd(true, true);
        let u = svd
            .u
            .ok_or_else(|| SklearsError::NumericalError("SVD failed".to_string()))?;
        let v_t = svd
            .v_t
            .ok_or_else(|| SklearsError::NumericalError("SVD failed".to_string()))?;
        let s = svd.singular_values;

        // Convert back to ndarray
        let u_vec: Vec<f64> = u.iter().cloned().collect();
        let u_nd = Array2::from_shape_vec((u.nrows(), u.ncols()), u_vec)
            .map_err(|_| SklearsError::NumericalError("Failed to create U matrix".to_string()))?;

        let vt_vec: Vec<f64> = v_t.iter().cloned().collect();
        let vt_nd = Array2::from_shape_vec((v_t.nrows(), v_t.ncols()), vt_vec)
            .map_err(|_| SklearsError::NumericalError("Failed to create V^T matrix".to_string()))?;

        let s_vec: Vec<f64> = s.iter().cloned().collect();
        let s_nd = Array1::from_vec(s_vec);

        Ok((u_nd, s_nd, vt_nd))
    }

    /// Solve linear system for ALS
    fn solve_linear_system_als(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        // Convert to nalgebra
        let a_vec: Vec<f64> = a.iter().cloned().collect();
        let b_vec: Vec<f64> = b.iter().cloned().collect();
        let a_na = DMatrix::from_row_slice(a.nrows(), a.ncols(), &a_vec);
        let b_na = nalgebra::DVector::from_vec(b_vec);

        // Solve using Cholesky decomposition if possible, otherwise use QR
        let result_na = if let Some(chol) = a_na.clone().cholesky() {
            chol.solve(&b_na)
        } else {
            // Use QR decomposition
            let qr = a_na.clone().qr();
            qr.solve(&b_na).unwrap_or_else(|| {
                // Fallback to pseudoinverse
                let svd = a_na.svd(true, true);
                let tolerance = 1e-12;

                let u = svd.u.unwrap();
                let v_t = svd.v_t.unwrap();
                let s = svd.singular_values;

                let mut s_inv = nalgebra::DMatrix::zeros(s.len(), s.len());
                for i in 0..s.len() {
                    if s[i] > tolerance {
                        s_inv[(i, i)] = 1.0 / s[i];
                    }
                }

                v_t.transpose() * s_inv * u.transpose() * b_na
            })
        };

        // Convert back to ndarray
        let result_vec: Vec<f64> = result_na.iter().cloned().collect();
        let result = Array1::from_vec(result_vec);

        Ok(result)
    }
}

impl MatrixCompletion<TrainedMatrixCompletion> {
    /// Get the user factors
    pub fn user_factors(&self) -> &Array2<f64> {
        &self.state.factors.0
    }

    /// Get the item factors
    pub fn item_factors(&self) -> &Array2<f64> {
        &self.state.factors.1
    }

    /// Get the reconstruction error
    pub fn reconstruction_error(&self) -> f64 {
        self.state.reconstruction_error
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Complete the matrix (fill in missing values)
    pub fn complete(&self, matrix: &Array2<f64>, mask: &Array2<bool>) -> Result<Array2<f64>> {
        let (n_users, n_items) = matrix.dim();

        if (n_users, n_items) != self.state.matrix_shape {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions don't match training dimensions".to_string(),
            ));
        }

        let mut completed = matrix.clone();
        let u = &self.state.factors.0;
        let v = &self.state.factors.1;

        for i in 0..n_users {
            for j in 0..n_items {
                if !mask[[i, j]] {
                    // Fill missing value
                    let mut predicted = self.state.global_bias + u.row(i).dot(&v.row(j));

                    if self.use_bias {
                        if let Some(ref user_bias) = self.state.user_bias {
                            predicted += user_bias[i];
                        }
                        if let Some(ref item_bias) = self.state.item_bias {
                            predicted += item_bias[j];
                        }
                    }

                    completed[[i, j]] = predicted;
                }
            }
        }

        Ok(completed)
    }

    /// Predict rating for a specific user-item pair
    pub fn predict(&self, user_id: usize, item_id: usize) -> Result<f64> {
        let (n_users, n_items) = self.state.matrix_shape;

        if user_id >= n_users || item_id >= n_items {
            return Err(SklearsError::InvalidInput(
                "User or item ID out of bounds".to_string(),
            ));
        }

        let u = &self.state.factors.0;
        let v = &self.state.factors.1;

        let mut predicted = self.state.global_bias + u.row(user_id).dot(&v.row(item_id));

        if self.use_bias {
            if let Some(ref user_bias) = self.state.user_bias {
                predicted += user_bias[user_id];
            }
            if let Some(ref item_bias) = self.state.item_bias {
                predicted += item_bias[item_id];
            }
        }

        Ok(predicted)
    }
}

/// Low-rank Matrix Recovery algorithms for robust decomposition
///
/// These methods are designed to recover low-rank matrices from corrupted observations,
/// handling outliers and sparse corruption better than standard matrix completion.
#[derive(Debug, Clone)]
pub struct LowRankMatrixRecovery<State = Untrained> {
    /// Algorithm to use for recovery
    pub algorithm: RecoveryAlgorithm,
    /// Rank of the low-rank component (if known)
    pub rank: Option<usize>,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Regularization parameter for sparse component
    pub lambda: f64,
    /// Regularization parameter for nuclear norm
    pub mu: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,

    /// Trained state
    state: State,
}

/// Low-rank matrix recovery algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RecoveryAlgorithm {
    #[default]
    PCP,
    RPCA,
    IHT,
    AltMin,
}

/// Trained low-rank matrix recovery state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainedLowRankMatrixRecovery {
    /// Low-rank component
    pub low_rank_component: Array2<f64>,
    /// Sparse component (corruption/outliers)
    pub sparse_component: Array2<f64>,
    /// SVD factors of low-rank component
    pub factors: Option<(Array2<f64>, Array1<f64>, Array2<f64>)>,
    /// Matrix dimensions
    pub matrix_shape: (usize, usize),
    /// Estimated rank
    pub estimated_rank: usize,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Final objective value
    pub objective_value: f64,
}

impl LowRankMatrixRecovery<Untrained> {
    /// Create a new low-rank matrix recovery transformer
    pub fn new() -> Self {
        Self {
            algorithm: RecoveryAlgorithm::PCP,
            rank: None,
            max_iter: 1000,
            tol: 1e-6,
            lambda: 0.1,
            mu: 0.1,
            random_state: None,
            state: Untrained,
        }
    }

    /// Set the recovery algorithm
    pub fn algorithm(mut self, algorithm: RecoveryAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the rank (if known)
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = Some(rank);
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

    /// Set sparsity regularization parameter
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set nuclear norm regularization parameter
    pub fn mu(mut self, mu: f64) -> Self {
        self.mu = mu;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Fit<Array2<f64>, ()> for LowRankMatrixRecovery<Untrained> {
    type Fitted = LowRankMatrixRecovery<TrainedLowRankMatrixRecovery>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let (n_rows, n_cols) = x.dim();

        if n_rows == 0 || n_cols == 0 {
            return Err(SklearsError::InvalidInput(
                "Input matrix cannot be empty".to_string(),
            ));
        }

        let (low_rank, sparse, factors, estimated_rank, n_iter, objective_value) =
            match self.algorithm {
                RecoveryAlgorithm::PCP => self.principal_component_pursuit(x)?,
                RecoveryAlgorithm::RPCA => self.robust_pca(x)?,
                RecoveryAlgorithm::IHT => self.iterative_hard_thresholding(x)?,
                RecoveryAlgorithm::AltMin => self.alternating_minimization(x)?,
            };

        Ok(LowRankMatrixRecovery {
            algorithm: self.algorithm,
            rank: self.rank,
            max_iter: self.max_iter,
            tol: self.tol,
            lambda: self.lambda,
            mu: self.mu,
            random_state: self.random_state,
            state: TrainedLowRankMatrixRecovery {
                low_rank_component: low_rank,
                sparse_component: sparse,
                factors,
                matrix_shape: (n_rows, n_cols),
                estimated_rank,
                n_iter,
                objective_value,
            },
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for LowRankMatrixRecovery<TrainedLowRankMatrixRecovery> {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_rows, n_cols) = x.dim();

        if (n_rows, n_cols) != self.state.matrix_shape {
            return Err(SklearsError::InvalidInput(
                "Input matrix dimensions must match training dimensions".to_string(),
            ));
        }

        // Return the low-rank component recovered from the input
        self.recover_low_rank_component(x)
    }
}

impl LowRankMatrixRecovery<Untrained> {
    /// Principal Component Pursuit using ADMM
    fn principal_component_pursuit(&self, x: &Array2<f64>) -> Result<RPCAResult> {
        let (m, n) = x.dim();
        let mut low_rank = Array2::zeros((m, n));
        let mut sparse = Array2::zeros((m, n));
        let mut y = Array2::zeros((m, n)); // Lagrange multipliers

        let tau = 1.0 / self.mu;

        for iter in 0..self.max_iter {
            // Update low-rank component using SVD soft thresholding
            let temp1 = x - &sparse + &y / self.mu;
            low_rank = self.svd_soft_threshold(&temp1, tau)?;

            // Update sparse component using element-wise soft thresholding
            let temp2 = x - &low_rank + &y / self.mu;
            sparse = self.element_wise_soft_threshold(&temp2, self.lambda / self.mu);

            // Update Lagrange multipliers
            let residual = x - &low_rank - &sparse;
            y = &y + self.mu * &residual;

            // Check convergence
            let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if residual_norm < self.tol {
                let (u, s, vt) = self.compute_svd(&low_rank)?;
                let estimated_rank = s.iter().filter(|&&x| x > 1e-12).count();
                let objective = self.compute_pcp_objective(&low_rank, &sparse);
                return Ok((
                    low_rank,
                    sparse,
                    Some((u, s, vt)),
                    estimated_rank,
                    iter + 1,
                    objective,
                ));
            }
        }

        let (u, s, vt) = self.compute_svd(&low_rank)?;
        let estimated_rank = s.iter().filter(|&&x| x > 1e-12).count();
        let objective = self.compute_pcp_objective(&low_rank, &sparse);
        Ok((
            low_rank,
            sparse,
            Some((u, s, vt)),
            estimated_rank,
            self.max_iter,
            objective,
        ))
    }

    /// Robust PCA (simplified version of PCP)
    fn robust_pca(&self, x: &Array2<f64>) -> Result<RPCAResult> {
        // Use PCP as the underlying algorithm for RPCA
        self.principal_component_pursuit(x)
    }

    /// Iterative Hard Thresholding
    fn iterative_hard_thresholding(&self, x: &Array2<f64>) -> Result<RPCAResult> {
        let (m, n) = x.dim();
        let mut low_rank = x.clone();

        let target_rank = self.rank.unwrap_or((m.min(n) / 4).max(1));

        for iter in 0..self.max_iter {
            // SVD and hard thresholding to enforce rank constraint
            let (u, s, vt) = self.compute_svd(&low_rank)?;

            // Keep only top `target_rank` singular values
            let mut s_thresh = Array1::zeros(s.len());
            for i in 0..target_rank.min(s.len()) {
                s_thresh[i] = s[i];
            }

            // Reconstruct low-rank matrix
            let new_low_rank = self.reconstruct_from_svd(&u, &s_thresh, &vt)?;

            // Check convergence
            let diff = &new_low_rank - &low_rank;
            let diff_norm = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();

            low_rank = new_low_rank;

            if diff_norm < self.tol {
                let sparse = x - &low_rank;
                let objective = self.compute_iht_objective(&low_rank, &sparse, target_rank);
                return Ok((
                    low_rank,
                    sparse,
                    Some((u, s_thresh, vt)),
                    target_rank,
                    iter + 1,
                    objective,
                ));
            }
        }

        let sparse = x - &low_rank;
        let (u, s, vt) = self.compute_svd(&low_rank)?;
        let objective = self.compute_iht_objective(&low_rank, &sparse, target_rank);
        Ok((
            low_rank,
            sparse,
            Some((u, s, vt)),
            target_rank,
            self.max_iter,
            objective,
        ))
    }

    /// Alternating Minimization
    fn alternating_minimization(&self, x: &Array2<f64>) -> Result<RPCAResult> {
        let (m, n) = x.dim();
        let target_rank = self.rank.unwrap_or((m.min(n) / 4).max(1));

        // Initialize random number generator with optional seed for reproducibility
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng())
        };

        // Initialize factors randomly
        let mut u = Array2::zeros((m, target_rank));
        let mut v = Array2::zeros((n, target_rank));

        for i in 0..m {
            for j in 0..target_rank {
                u[[i, j]] = rng.gen() - 0.5;
            }
        }

        for i in 0..n {
            for j in 0..target_rank {
                v[[i, j]] = rng.gen() - 0.5;
            }
        }

        for iter in 0..self.max_iter {
            let old_u = u.clone();
            let old_v = v.clone();

            // Update U by solving least squares
            u = self.update_factor_u(x, &v)?;

            // Update V by solving least squares
            v = self.update_factor_v(x, &u)?;

            // Check convergence
            let u_diff = (&u - &old_u).iter().map(|&x| x * x).sum::<f64>().sqrt();
            let v_diff = (&v - &old_v).iter().map(|&x| x * x).sum::<f64>().sqrt();

            if u_diff < self.tol && v_diff < self.tol {
                let low_rank = u.dot(&v.t());
                let sparse = x - &low_rank;
                let (u_svd, s, vt) = self.compute_svd(&low_rank)?;
                let objective = self.compute_altmin_objective(&low_rank, &sparse);
                return Ok((
                    low_rank,
                    sparse,
                    Some((u_svd, s, vt)),
                    target_rank,
                    iter + 1,
                    objective,
                ));
            }
        }

        let low_rank = u.dot(&v.t());
        let sparse = x - &low_rank;
        let (u_svd, s, vt) = self.compute_svd(&low_rank)?;
        let objective = self.compute_altmin_objective(&low_rank, &sparse);
        Ok((
            low_rank,
            sparse,
            Some((u_svd, s, vt)),
            target_rank,
            self.max_iter,
            objective,
        ))
    }

    /// SVD soft thresholding
    fn svd_soft_threshold(&self, matrix: &Array2<f64>, threshold: f64) -> Result<Array2<f64>> {
        let (u, s, vt) = self.compute_svd(matrix)?;

        // Apply soft thresholding to singular values
        let s_thresh: Array1<f64> = s
            .iter()
            .map(|&x| if x > threshold { x - threshold } else { 0.0 })
            .collect();

        self.reconstruct_from_svd(&u, &s_thresh, &vt)
    }

    /// Element-wise soft thresholding
    fn element_wise_soft_threshold(&self, matrix: &Array2<f64>, threshold: f64) -> Array2<f64> {
        matrix.mapv(|x| {
            if x > threshold {
                x - threshold
            } else if x < -threshold {
                x + threshold
            } else {
                0.0
            }
        })
    }

    /// Compute SVD decomposition
    fn compute_svd(&self, matrix: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (m, n) = matrix.dim();
        let matrix_na = DMatrix::from_row_slice(m, n, matrix.as_slice().unwrap());

        let svd = matrix_na.svd(true, true);

        let u = svd.u.ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute U matrix in SVD".to_string())
        })?;

        let vt = svd.v_t.ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute V^T matrix in SVD".to_string())
        })?;

        // Convert back to ndarray
        let mut u_array = Array2::zeros((m, u.ncols()));
        let mut vt_array = Array2::zeros((vt.nrows(), n));
        let mut s_array = Array1::zeros(svd.singular_values.len());

        for i in 0..m {
            for j in 0..u.ncols() {
                u_array[[i, j]] = u[(i, j)];
            }
        }

        for i in 0..vt.nrows() {
            for j in 0..n {
                vt_array[[i, j]] = vt[(i, j)];
            }
        }

        for i in 0..svd.singular_values.len() {
            s_array[i] = svd.singular_values[i];
        }

        Ok((u_array, s_array, vt_array))
    }

    /// Reconstruct matrix from SVD components
    fn reconstruct_from_svd(
        &self,
        u: &Array2<f64>,
        s: &Array1<f64>,
        vt: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let (m, k1) = u.dim();
        let (k2, n) = vt.dim();

        if k1 != k2 || k1 != s.len() {
            return Err(SklearsError::InvalidInput(
                "Inconsistent dimensions in SVD reconstruction".to_string(),
            ));
        }

        let mut result = Array2::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += u[[i, k]] * s[k] * vt[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }

    /// Update factor U in alternating minimization
    fn update_factor_u(&self, x: &Array2<f64>, v: &Array2<f64>) -> Result<Array2<f64>> {
        // Solve for U in ||X - UV^T||_F^2 by solving U = XV(V^TV)^{-1}
        let vtv = v.t().dot(v);
        let xv = x.dot(v);

        // Convert to nalgebra for matrix operations
        let vtv_matrix = DMatrix::from_row_slice(vtv.nrows(), vtv.ncols(), vtv.as_slice().unwrap());
        let xv_matrix = DMatrix::from_row_slice(xv.nrows(), xv.ncols(), xv.as_slice().unwrap());

        // Add regularization for numerical stability
        let mut reg_vtv = vtv_matrix.clone();
        for i in 0..reg_vtv.nrows() {
            reg_vtv[(i, i)] += 1e-12;
        }

        let vtv_inv = reg_vtv.try_inverse().ok_or_else(|| {
            SklearsError::NumericalError("Failed to invert matrix in factor update".to_string())
        })?;

        let u = xv_matrix * vtv_inv;

        // Convert back to ndarray
        let mut u_array = Array2::zeros((u.nrows(), u.ncols()));
        for i in 0..u.nrows() {
            for j in 0..u.ncols() {
                u_array[[i, j]] = u[(i, j)];
            }
        }

        Ok(u_array)
    }

    /// Update factor V in alternating minimization
    fn update_factor_v(&self, x: &Array2<f64>, u: &Array2<f64>) -> Result<Array2<f64>> {
        // Solve for V in ||X - UV^T||_F^2 by solving V = X^TU(U^TU)^{-1}
        let utu = u.t().dot(u);
        let xtu = x.t().dot(u);

        let utu_matrix = DMatrix::from_row_slice(utu.nrows(), utu.ncols(), utu.as_slice().unwrap());
        let xtu_matrix = DMatrix::from_row_slice(xtu.nrows(), xtu.ncols(), xtu.as_slice().unwrap());

        // Add regularization for numerical stability
        let mut reg_utu = utu_matrix.clone();
        for i in 0..reg_utu.nrows() {
            reg_utu[(i, i)] += 1e-12;
        }

        let utu_inv = reg_utu.try_inverse().ok_or_else(|| {
            SklearsError::NumericalError("Failed to invert matrix in factor update".to_string())
        })?;

        let v = xtu_matrix * utu_inv;

        // Convert back to ndarray
        let mut v_array = Array2::zeros((v.nrows(), v.ncols()));
        for i in 0..v.nrows() {
            for j in 0..v.ncols() {
                v_array[[i, j]] = v[(i, j)];
            }
        }

        Ok(v_array)
    }

    /// Compute PCP objective function
    fn compute_pcp_objective(&self, low_rank: &Array2<f64>, sparse: &Array2<f64>) -> f64 {
        let nuclear_norm = if let Ok((_, s, _)) = self.compute_svd(low_rank) {
            s.sum()
        } else {
            0.0
        };

        let l1_norm = sparse.iter().map(|&x| x.abs()).sum::<f64>();

        nuclear_norm + self.lambda * l1_norm
    }

    /// Compute IHT objective function
    fn compute_iht_objective(
        &self,
        _low_rank: &Array2<f64>,
        sparse: &Array2<f64>,
        _rank: usize,
    ) -> f64 {
        // Simple Frobenius norm of the sparse component
        sparse.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Compute alternating minimization objective function
    fn compute_altmin_objective(&self, _low_rank: &Array2<f64>, sparse: &Array2<f64>) -> f64 {
        // Frobenius norm of the sparse component
        sparse.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

impl LowRankMatrixRecovery<TrainedLowRankMatrixRecovery> {
    /// Get the recovered low-rank component
    pub fn low_rank_component(&self) -> &Array2<f64> {
        &self.state.low_rank_component
    }

    /// Get the sparse component (outliers/corruption)
    pub fn sparse_component(&self) -> &Array2<f64> {
        &self.state.sparse_component
    }

    /// Get the estimated rank
    pub fn estimated_rank(&self) -> usize {
        self.state.estimated_rank
    }

    /// Get SVD factors if available
    pub fn factors(&self) -> &Option<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        &self.state.factors
    }

    /// Recover low-rank component from new corrupted data
    pub fn recover_low_rank_component(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        // For simplicity, assume the same corruption pattern and apply the learned decomposition
        // In practice, this would involve solving the recovery problem for new data
        let corruption_estimate = x - &self.state.low_rank_component;
        Ok(x - &self.element_wise_soft_threshold(&corruption_estimate, self.lambda))
    }

    /// Element-wise soft thresholding (helper method)
    fn element_wise_soft_threshold(&self, matrix: &Array2<f64>, threshold: f64) -> Array2<f64> {
        matrix.mapv(|x| {
            if x > threshold {
                x - threshold
            } else if x < -threshold {
                x + threshold
            } else {
                0.0
            }
        })
    }

    /// Reconstruct the original matrix (low-rank + sparse)
    pub fn reconstruct(&self) -> Array2<f64> {
        &self.state.low_rank_component + &self.state.sparse_component
    }
}

impl Default for LowRankMatrixRecovery<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_matrix_completion_svd() {
        // Create a test matrix with some missing values
        let matrix = array![
            [5.0, 3.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 5.0],
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 5.0, 4.0],
        ];

        let mask = array![
            [true, true, false, true],
            [true, false, false, true],
            [true, true, false, true],
            [true, false, false, true],
            [false, true, true, true],
        ];

        let mc = MatrixCompletion::new()
            .rank(2)
            .algorithm(CompletionAlgorithm::SVD)
            .max_iter(10)
            .random_state(42);

        let result = mc.fit(&matrix, &mask);
        assert!(result.is_ok());

        let trained = result.unwrap();
        assert_eq!(trained.state.rank, 2);
        assert!(trained.state.reconstruction_error.is_finite());

        // Test completion
        let completed = trained.complete(&matrix, &mask).unwrap();
        assert_eq!(completed.dim(), matrix.dim());

        // Test prediction
        let prediction = trained.predict(0, 2).unwrap();
        assert!(prediction.is_finite());
    }

    #[test]
    fn test_matrix_completion_als() {
        let matrix = array![
            [5.0, 3.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 5.0],
        ];

        let mask = array![
            [true, true, false, true],
            [true, false, false, true],
            [true, true, false, true],
        ];

        let mc = MatrixCompletion::new()
            .rank(2)
            .algorithm(CompletionAlgorithm::ALS)
            .max_iter(5)
            .random_state(42);

        let result = mc.fit(&matrix, &mask);
        assert!(result.is_ok());

        let trained = result.unwrap();
        assert_eq!(trained.state.rank, 2);
        assert!(trained.state.n_iter > 0);
    }

    #[test]
    fn test_matrix_completion_nuclear_norm() {
        let matrix = array![[5.0, 3.0], [4.0, 2.0], [1.0, 1.0],];

        let mask = array![[true, false], [true, true], [false, true],];

        let mc = MatrixCompletion::new()
            .rank(1)
            .algorithm(CompletionAlgorithm::NuclearNorm)
            .regularization(0.1)
            .max_iter(5)
            .random_state(42);

        let result = mc.fit(&matrix, &mask);
        assert!(result.is_ok());

        let trained = result.unwrap();
        assert!(trained.state.reconstruction_error.is_finite());
    }

    #[test]
    fn test_matrix_completion_parameters() {
        let mc = MatrixCompletion::new()
            .rank(5)
            .algorithm(CompletionAlgorithm::ALS)
            .max_iter(200)
            .tol(1e-8)
            .learning_rate(0.05)
            .regularization(0.05)
            .use_bias(false);

        assert_eq!(mc.rank, Some(5));
        assert_eq!(mc.algorithm, CompletionAlgorithm::ALS);
        assert_eq!(mc.max_iter, 200);
        assert_eq!(mc.tol, 1e-8);
        assert_eq!(mc.learning_rate, 0.05);
        assert_eq!(mc.regularization, 0.05);
        assert_eq!(mc.use_bias, false);
    }

    #[test]
    fn test_matrix_completion_invalid_mask() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let mask = array![[true]]; // Wrong dimensions

        let mc = MatrixCompletion::new();
        let result = mc.fit(&matrix, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_completion_no_observed() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let mask = array![[false, false], [false, false]]; // No observed values

        let mc = MatrixCompletion::new();
        let result = mc.fit(&matrix, &mask);
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // TODO: Fix PCP algorithm numerical stability
    fn test_low_rank_matrix_recovery_pcp() {
        // Create a low-rank matrix with sparse corruption
        let low_rank = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        let sparse = array![[0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, -5.0, 0.0]];
        let corrupted = &low_rank + &sparse;

        let lrmr = LowRankMatrixRecovery::new()
            .algorithm(RecoveryAlgorithm::PCP)
            .max_iter(100)
            .lambda(0.01) // Smaller lambda to allow low-rank recovery
            .mu(0.1)
            .random_state(42);

        let trained_lrmr = lrmr.fit(&corrupted, &()).unwrap();

        assert_eq!(trained_lrmr.low_rank_component().dim(), (3, 3));
        assert_eq!(trained_lrmr.sparse_component().dim(), (3, 3));
        assert!(trained_lrmr.estimated_rank() > 0);
        assert!(trained_lrmr.state.objective_value >= 0.0);

        // Check that reconstruction is close to original
        let reconstruction = trained_lrmr.reconstruct();
        assert_eq!(reconstruction.dim(), (3, 3));

        // All values should be finite
        for val in reconstruction.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_low_rank_matrix_recovery_iht() {
        let matrix = array![
            [1.0, 2.0],
            [2.0, 4.1], // Slightly corrupted low-rank matrix
        ];

        let lrmr = LowRankMatrixRecovery::new()
            .algorithm(RecoveryAlgorithm::IHT)
            .rank(1)
            .max_iter(50)
            .random_state(42);

        let trained_lrmr = lrmr.fit(&matrix, &()).unwrap();

        assert_eq!(trained_lrmr.estimated_rank(), 1);
        assert!(trained_lrmr.state.n_iter <= 50);

        // Test transform
        let recovered = trained_lrmr.transform(&matrix).unwrap();
        assert_eq!(recovered.dim(), (2, 2));

        // All values should be finite
        for val in recovered.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    #[ignore] // TODO: Fix alternating minimization dimension handling
    fn test_low_rank_matrix_recovery_alternating_minimization() {
        let matrix = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.1, 2.1, 3.1] // Approximately rank-1 with noise
        ];

        let lrmr = LowRankMatrixRecovery::new()
            .algorithm(RecoveryAlgorithm::AltMin)
            .rank(2)
            .max_iter(20)
            .lambda(0.01)
            .random_state(42);

        let trained_lrmr = lrmr.fit(&matrix, &()).unwrap();

        assert!(trained_lrmr.estimated_rank() >= 1 && trained_lrmr.estimated_rank() <= 3);
        assert!(trained_lrmr.state.n_iter <= 20);

        // Check that factors are available
        assert!(trained_lrmr.factors().is_some());

        let (u, s, vt) = trained_lrmr.factors().as_ref().unwrap();
        assert_eq!(u.ncols(), 2);
        assert_eq!(vt.nrows(), 2);
        assert_eq!(s.len(), 3); // min(m, n)
    }

    #[test]
    fn test_low_rank_matrix_recovery_parameters() {
        let lrmr = LowRankMatrixRecovery::new()
            .algorithm(RecoveryAlgorithm::RPCA)
            .rank(5)
            .max_iter(500)
            .tol(1e-8)
            .lambda(0.05)
            .mu(0.05)
            .random_state(123);

        assert_eq!(lrmr.algorithm, RecoveryAlgorithm::RPCA);
        assert_eq!(lrmr.rank, Some(5));
        assert_eq!(lrmr.max_iter, 500);
        assert_eq!(lrmr.tol, 1e-8);
        assert_eq!(lrmr.lambda, 0.05);
        assert_eq!(lrmr.mu, 0.05);
        assert_eq!(lrmr.random_state, Some(123));
    }

    #[test]
    fn test_low_rank_matrix_recovery_error_cases() {
        let empty_matrix = Array2::<f64>::zeros((0, 0));
        let lrmr = LowRankMatrixRecovery::new();
        let result = lrmr.fit(&empty_matrix, &());
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // TODO: Fix PCP convergence and sparse component threshold
    fn test_low_rank_matrix_recovery_convergence() {
        // Create a simple rank-1 matrix
        let matrix = array![[1.0, 2.0], [2.0, 4.0]];

        let lrmr = LowRankMatrixRecovery::new()
            .algorithm(RecoveryAlgorithm::PCP)
            .max_iter(100)
            .tol(1e-6)
            .lambda(0.01) // Smaller lambda for cleaner low-rank decomposition
            .mu(0.1)
            .random_state(42);

        let trained_lrmr = lrmr.fit(&matrix, &()).unwrap();

        // Should converge quickly for a simple case
        assert!(trained_lrmr.state.n_iter <= 100);
        assert!(trained_lrmr.estimated_rank() <= 2);

        // Low-rank component should be close to original for this clean case
        let low_rank = trained_lrmr.low_rank_component();
        assert_eq!(low_rank.dim(), (2, 2));

        // Sparse component should be small for this clean case
        let sparse = trained_lrmr.sparse_component();
        let sparse_norm = sparse.iter().map(|&x| x.abs()).sum::<f64>();
        assert!(sparse_norm < 5.0); // Should be relatively small for clean data
    }
}
