//! Orthogonal Matching Pursuit (OMP) algorithms for sparse coding

use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{error::Result, types::Float};

/// Configuration for OMP algorithm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OMPConfig {
    /// Maximum number of non-zero coefficients
    pub n_nonzero_coefs: Option<usize>,
    /// Tolerance for residual norm
    pub tol: Option<Float>,
}

impl Default for OMPConfig {
    fn default() -> Self {
        Self {
            n_nonzero_coefs: None,
            tol: Some(1e-4),
        }
    }
}

/// OMP algorithm result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OMPResult {
    /// Sparse coefficients
    pub coefficients: Array1<Float>,
    /// Residual norm
    pub residual_norm: Float,
    /// Number of iterations
    pub n_iter: usize,
}

/// Orthogonal Matching Pursuit encoder
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OMPEncoder {
    config: OMPConfig,
}

impl OMPEncoder {
    pub fn new(config: OMPConfig) -> Self {
        Self { config }
    }

    /// Perform OMP sparse coding
    ///
    /// Orthogonal Matching Pursuit is a greedy algorithm for sparse approximation.
    /// It iteratively selects the dictionary atom most correlated with the residual,
    /// then updates coefficients via least squares projection.
    ///
    /// # Arguments
    /// * `dictionary` - Dictionary matrix where each column is an atom (n_features × n_atoms)
    /// * `signal` - Signal to encode (n_features)
    ///
    /// # Returns
    /// Sparse representation with coefficients, residual norm, and iteration count
    pub fn encode(&self, dictionary: &Array2<Float>, signal: &Array1<Float>) -> Result<OMPResult> {
        use sklears_core::error::SklearsError;

        let (n_features, n_atoms) = dictionary.dim();

        // Validate dimensions
        if signal.len() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Signal length {} doesn't match dictionary features {}",
                signal.len(),
                n_features
            )));
        }

        // Normalize dictionary atoms (columns) for correlation computation
        let mut dict_normalized = dictionary.clone();
        for j in 0..n_atoms {
            let col = dictionary.column(j);
            let norm = col.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                for i in 0..n_features {
                    dict_normalized[[i, j]] /= norm;
                }
            }
        }

        let mut coefficients = Array1::zeros(n_atoms);
        let mut residual = signal.clone();
        let mut selected_atoms: Vec<usize> = Vec::new();
        let mut n_iter = 0;

        // Determine maximum iterations
        let max_iter = if let Some(k) = self.config.n_nonzero_coefs {
            k.min(n_atoms)
        } else {
            n_atoms
        };

        let tol = self.config.tol.unwrap_or(1e-4);

        // OMP iterations
        for iteration in 0..max_iter {
            // Compute residual norm
            let residual_norm = residual.mapv(|x| x * x).sum().sqrt();

            // Check stopping criterion (residual tolerance)
            if residual_norm < tol {
                n_iter = iteration;
                break;
            }

            // Find atom with maximum absolute correlation with residual
            let mut max_corr = 0.0;
            let mut best_atom = 0;

            for j in 0..n_atoms {
                // Skip already selected atoms
                if selected_atoms.contains(&j) {
                    continue;
                }

                // Compute correlation: <dict[:,j], residual>
                let mut corr = 0.0;
                for i in 0..n_features {
                    corr += dict_normalized[[i, j]] * residual[i];
                }

                let abs_corr = corr.abs();
                if abs_corr > max_corr {
                    max_corr = abs_corr;
                    best_atom = j;
                }
            }

            // Add best atom to selected set
            selected_atoms.push(best_atom);

            // Solve least squares on selected atoms:
            // coeffs = (D_sel^T D_sel)^-1 D_sel^T signal
            // Using normal equations for small problems
            let k = selected_atoms.len();

            // Build D_sel: n_features × k matrix
            let mut d_sel = Array2::zeros((n_features, k));
            for (col_idx, &atom_idx) in selected_atoms.iter().enumerate() {
                for row_idx in 0..n_features {
                    d_sel[[row_idx, col_idx]] = dictionary[[row_idx, atom_idx]];
                }
            }

            // Compute Gram matrix: G = D_sel^T D_sel (k × k)
            let mut gram = Array2::zeros((k, k));
            for i in 0..k {
                for j in 0..k {
                    let mut sum = 0.0;
                    for row in 0..n_features {
                        sum += d_sel[[row, i]] * d_sel[[row, j]];
                    }
                    gram[[i, j]] = sum;
                }
            }

            // Compute rhs: b = D_sel^T signal (k × 1)
            let mut rhs = Array1::zeros(k);
            for i in 0..k {
                let mut sum = 0.0;
                for row in 0..n_features {
                    sum += d_sel[[row, i]] * signal[row];
                }
                rhs[i] = sum;
            }

            // Solve G @ x = rhs using Cholesky decomposition
            // For small k, use simple Gaussian elimination
            let selected_coeffs = solve_linear_system(&gram, &rhs)?;

            // Update coefficients
            coefficients.fill(0.0);
            for (i, &atom_idx) in selected_atoms.iter().enumerate() {
                coefficients[atom_idx] = selected_coeffs[i];
            }

            // Update residual: residual = signal - D_sel @ selected_coeffs
            residual = signal.clone();
            for (i, &atom_idx) in selected_atoms.iter().enumerate() {
                for row in 0..n_features {
                    residual[row] -= dictionary[[row, atom_idx]] * selected_coeffs[i];
                }
            }

            n_iter = iteration + 1;

            // Check if we've selected enough atoms
            if let Some(k_max) = self.config.n_nonzero_coefs {
                if selected_atoms.len() >= k_max {
                    break;
                }
            }
        }

        let final_residual_norm = residual.mapv(|x| x * x).sum().sqrt();

        Ok(OMPResult {
            coefficients,
            residual_norm: final_residual_norm,
            n_iter,
        })
    }
}

/// Solve linear system Ax = b using Gaussian elimination with partial pivoting
fn solve_linear_system(a: &Array2<Float>, b: &Array1<Float>) -> Result<Array1<Float>> {
    use sklears_core::error::SklearsError;

    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square and match RHS dimension".to_string(),
        ));
    }

    // Create augmented matrix [A|b]
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Check for singularity
        if max_val < 1e-10 {
            return Err(SklearsError::NumericalError(
                "Singular matrix in OMP least squares".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != col {
            for j in 0..=n {
                let temp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate column
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / aug[[col, col]];
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_omp_simple_sparse_signal() {
        // Create a simple 2-atom dictionary and sparse signal
        // Dictionary: [[1, 0], [0, 1]] (identity)
        // Signal: [3, 4]
        // Expected: coefficients = [3, 4]
        let dictionary = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let signal = Array1::from_vec(vec![3.0, 4.0]);

        let config = OMPConfig {
            n_nonzero_coefs: Some(2),
            tol: Some(1e-6),
        };

        let encoder = OMPEncoder::new(config);
        let result = encoder.encode(&dictionary, &signal).unwrap();

        // Should find both atoms
        assert!((result.coefficients[0] - 3.0).abs() < 1e-6);
        assert!((result.coefficients[1] - 4.0).abs() < 1e-6);
        assert!(result.residual_norm < 1e-6);
    }

    #[test]
    fn test_omp_sparse_representation() {
        // Create overcomplete dictionary (more atoms than features)
        // Dictionary: 3 features, 5 atoms
        let dictionary = Array2::from_shape_vec(
            (3, 5),
            vec![
                // Atom 0   Atom 1   Atom 2   Atom 3   Atom 4
                1.0, 0.5, 0.2, 0.8, 0.3, // Feature 0
                0.0, 1.0, 0.3, 0.1, 0.7, // Feature 1
                0.0, 0.0, 1.0, 0.2, 0.4, // Feature 2
            ],
        )
        .unwrap();

        // Signal is a linear combination of atoms 0 and 2: signal = 2*atom0 + 3*atom2
        let signal = Array1::from_vec(vec![
            2.0 * 1.0 + 3.0 * 0.2, // 2.6
            2.0 * 0.0 + 3.0 * 0.3, // 0.9
            2.0 * 0.0 + 3.0 * 1.0, // 3.0
        ]);

        let config = OMPConfig {
            n_nonzero_coefs: Some(2),
            tol: Some(1e-4),
        };

        let encoder = OMPEncoder::new(config);
        let result = encoder.encode(&dictionary, &signal).unwrap();

        // Should identify atoms 0 and 2
        assert!(result.coefficients[0].abs() > 1.0); // Atom 0 selected
        assert!(result.coefficients[2].abs() > 1.0); // Atom 2 selected
        assert!(result.residual_norm < 0.1); // Low residual
    }

    #[test]
    fn test_omp_tolerance_stopping() {
        // Test that OMP stops when residual falls below tolerance
        let dictionary =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                .unwrap();
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let config = OMPConfig {
            n_nonzero_coefs: None, // No limit on atoms
            tol: Some(1e-6),       // Very tight tolerance
        };

        let encoder = OMPEncoder::new(config);
        let result = encoder.encode(&dictionary, &signal).unwrap();

        // Should stop when residual is small
        assert!(result.residual_norm < 1e-5);
        assert!(result.n_iter <= 3); // Should select all needed atoms
    }

    #[test]
    fn test_omp_max_nonzero_coefs() {
        // Test that OMP respects maximum non-zero coefficients
        let dictionary = Array2::from_shape_vec(
            (3, 5),
            vec![
                1.0, 0.5, 0.2, 0.8, 0.3, 0.0, 1.0, 0.3, 0.1, 0.7, 0.0, 0.0, 1.0, 0.2, 0.4,
            ],
        )
        .unwrap();
        let signal = Array1::from_vec(vec![2.5, 1.5, 2.0]);

        let config = OMPConfig {
            n_nonzero_coefs: Some(2), // Limit to 2 atoms
            tol: None,
        };

        let encoder = OMPEncoder::new(config);
        let result = encoder.encode(&dictionary, &signal).unwrap();

        // Count non-zero coefficients
        let nnz = result
            .coefficients
            .iter()
            .filter(|&&x| x.abs() > 1e-10)
            .count();
        assert_eq!(nnz, 2);
    }

    #[test]
    fn test_omp_exact_representation() {
        // Test that OMP can exactly represent a signal in the span of dictionary
        let dictionary = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        // Signal = 2*atom0 + 3*atom1
        let signal = Array1::from_vec(vec![2.0, 3.0, 0.0, 0.0]);

        let config = OMPConfig {
            n_nonzero_coefs: Some(3),
            tol: Some(1e-6),
        };

        let encoder = OMPEncoder::new(config);
        let result = encoder.encode(&dictionary, &signal).unwrap();

        // Reconstruct signal
        let mut reconstructed = Array1::zeros(4);
        for (atom_idx, &coef) in result.coefficients.iter().enumerate() {
            for row in 0..4 {
                reconstructed[row] += coef * dictionary[[row, atom_idx]];
            }
        }

        // Check reconstruction error
        let recon_error: Float = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b): (&Float, &Float)| (a - b).powi(2))
            .sum::<Float>()
            .sqrt();
        assert!(recon_error < 1e-3);
        assert!(result.residual_norm < 1e-3);
    }

    #[test]
    fn test_omp_dimension_validation() {
        // Test error handling for mismatched dimensions
        let dictionary =
            Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let signal = Array1::from_vec(vec![1.0, 2.0]); // Wrong size (should be 3)

        let config = OMPConfig::default();
        let encoder = OMPEncoder::new(config);
        let result = encoder.encode(&dictionary, &signal);

        assert!(result.is_err());
    }

    #[test]
    fn test_solve_linear_system() {
        // Test the linear system solver with a simple 2x2 system
        // [2 1] [x1]   [5]
        // [1 3] [x2] = [6]
        // Solution: x1 = 9/5, x2 = 7/5
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 6.0]);

        let x = solve_linear_system(&a, &b).unwrap();

        assert!((x[0] - 9.0 / 5.0).abs() < 1e-6);
        assert!((x[1] - 7.0 / 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_solve_linear_system_identity() {
        // Test with identity matrix
        let a = Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let x = solve_linear_system(&a, &b).unwrap();

        assert!((x[0] - 4.0).abs() < 1e-10);
        assert!((x[1] - 5.0).abs() < 1e-10);
        assert!((x[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_with_pivoting() {
        // Test system that requires pivoting
        // [0 1] [x1]   [2]
        // [1 1] [x2] = [3]
        // Solution: x1 = 1, x2 = 2
        let a = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 1.0]).unwrap();
        let b = Array1::from_vec(vec![2.0, 3.0]);

        let x = solve_linear_system(&a, &b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 2.0).abs() < 1e-6);
    }
}
