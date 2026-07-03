//! Graphical Lasso Estimator

use crate::empirical::EmpiricalCovariance;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_linalg::compat::ArrayLinalgExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Graphical Lasso Estimator
///
/// Sparse inverse covariance estimation with L1-penalized maximum likelihood.
/// The Graphical Lasso estimates a sparse inverse covariance matrix (precision matrix)
/// by solving an L1-penalized maximum likelihood problem.
///
/// # Parameters
///
/// * `alpha` - Regularization parameter for L1 penalty
/// * `mode` - Algorithm mode ('cd' for coordinate descent, 'lars' for LARS)
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum number of iterations
/// * `assume_centered` - Whether to assume the data is centered
///
/// # Examples
///
/// ```
/// use sklears_covariance::GraphicalLasso;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let x = array![
///     [1.0, 0.1, 0.0], [2.0, 1.9, 0.1], [3.0, 2.8, 0.2],
///     [4.0, 4.1, 0.3], [5.0, 4.9, 0.4]
/// ];
///
/// let estimator = GraphicalLasso::new().alpha(0.1);
/// match estimator.fit(&x.view(), &()) {
///     Ok(fitted) => {
///         let precision = fitted.get_precision();
///         assert_eq!(precision.dim(), (3, 3));
///     }
///     Err(_) => {
///         // May fail due to numerical sensitivity
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GraphicalLasso<S = Untrained> {
    state: S,
    alpha: f64,
    mode: String,
    tol: f64,
    max_iter: usize,
    assume_centered: bool,
}

/// Trained state for GraphicalLasso
#[derive(Debug, Clone)]
pub struct GraphicalLassoTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The sparse precision matrix
    pub precision: Array2<f64>,
    /// The location (mean) vector
    pub location: Array1<f64>,
    /// The regularization parameter used
    pub alpha: f64,
    /// Number of iterations performed
    pub n_iter: usize,
}

impl GraphicalLasso<Untrained> {
    /// Create a new GraphicalLasso instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            alpha: 0.01,
            mode: "cd".to_string(),
            tol: 1e-4,
            max_iter: 100,
            assume_centered: false,
        }
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.max(0.0);
        self
    }

    /// Set the algorithm mode
    pub fn mode(mut self, mode: String) -> Self {
        self.mode = mode;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.assume_centered = assume_centered;
        self
    }
}

impl Default for GraphicalLasso<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GraphicalLasso<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for GraphicalLasso<Untrained> {
    type Fitted = GraphicalLasso<GraphicalLassoTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, _n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Compute empirical covariance
        let emp_cov = EmpiricalCovariance::new()
            .assume_centered(self.assume_centered)
            .fit(&x, &())?;
        let empirical_covariance = emp_cov.get_covariance().clone();

        // Solve the graphical lasso problem for the sparse precision matrix.
        let (precision, n_iter) = self.solve_graphical_lasso(&empirical_covariance)?;

        // The reported covariance is the one *implied* by the fitted,
        // regularized precision matrix (covariance = precision^-1), not the
        // raw empirical covariance. This keeps any code that scores the
        // model via `get_covariance()` -- including this crate's own
        // `ScoringMetric`s -- sensitive to `alpha`.
        let covariance = invert_with_ridge(&precision)?;

        Ok(GraphicalLasso {
            state: GraphicalLassoTrained {
                covariance,
                precision,
                location: emp_cov.get_location().clone(),
                alpha: self.alpha,
                n_iter,
            },
            alpha: self.alpha,
            mode: self.mode,
            tol: self.tol,
            max_iter: self.max_iter,
            assume_centered: self.assume_centered,
        })
    }
}

impl GraphicalLasso<Untrained> {
    /// Solve the L1-penalized (graphical lasso) precision matrix estimation
    /// problem, returning `(precision, n_iter)`.
    ///
    /// This is the block coordinate descent algorithm of Friedman, Hastie &
    /// Tibshirani (2008), "Sparse inverse covariance estimation with the
    /// graphical lasso": maintain a working covariance estimate `w`
    /// (initialized to the empirical covariance `emp_cov`), and cycle
    /// through each variable `j`, solving a Lasso regression of `j` against
    /// every other variable -- using the *current* working-covariance
    /// submatrix as the Gram matrix -- to refine `w`'s off-diagonal row/
    /// column `j`. The diagonal of `w` is left fixed at `emp_cov`'s diagonal
    /// throughout: that is the exact stationarity condition of the
    /// (off-diagonal-only penalized) graphical lasso objective
    /// `max_theta log det(theta) - tr(S theta) - alpha * sum_{i != j} |theta_ij|`,
    /// so unlike the off-diagonal, the diagonal needs no iterative solve.
    ///
    /// The fitted precision matrix is then the exact matrix inverse of the
    /// converged working covariance (`theta = w^-1`), which is the defining
    /// fixed-point relationship of the algorithm and reproduces the
    /// sparsity pattern induced by the per-row lasso solves above, to
    /// numerical precision.
    ///
    /// Convergence is checked every sweep via the largest absolute change
    /// in `w` (independent of `max_iter`'s parity), so the solver can also
    /// exit early once converged.
    fn solve_graphical_lasso(&self, emp_cov: &Array2<f64>) -> SklResult<(Array2<f64>, usize)> {
        let n = emp_cov.nrows();

        if n == 0 {
            return Ok((Array2::zeros((0, 0)), 0));
        }

        if n == 1 {
            // No off-diagonal terms exist to penalize; the precision is just
            // the reciprocal variance.
            let variance = emp_cov[[0, 0]].max(1e-12);
            let mut precision = Array2::zeros((1, 1));
            precision[[0, 0]] = 1.0 / variance;
            return Ok((precision, 0));
        }

        // `alpha <= 0` is the unregularized limit, where graphical lasso
        // reduces to the ordinary maximum-likelihood precision matrix:
        // skip the iterative solve and invert the empirical covariance
        // directly.
        if self.alpha <= 0.0 {
            let precision = invert_with_ridge(emp_cov)?;
            return Ok((precision, 0));
        }

        let mut w = emp_cov.clone();
        for i in 0..n {
            if w[[i, i]] <= 0.0 {
                w[[i, i]] = 1e-12;
            }
        }

        // Warm-started lasso coefficients for each row, refined sweep over
        // sweep so consecutive sweeps converge quickly.
        let mut beta_rows: Vec<Array1<f64>> = vec![Array1::zeros(n - 1); n];

        let mut n_iter = 0usize;
        for iter in 0..self.max_iter {
            let w_prev = w.clone();

            for j in 0..n {
                let (w_rr, s_rj, idx_map) = extract_row_block(&w, emp_cov, j);
                solve_lasso_cd(&w_rr, &s_rj, &mut beta_rows[j], self.alpha, self.tol);

                let w_rj_new = w_rr.dot(&beta_rows[j]);
                for (local_k, &global_k) in idx_map.iter().enumerate() {
                    w[[global_k, j]] = w_rj_new[local_k];
                    w[[j, global_k]] = w_rj_new[local_k];
                }
            }

            n_iter = iter + 1;

            let max_change = w
                .iter()
                .zip(w_prev.iter())
                .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));

            if max_change < self.tol {
                break;
            }
        }

        let precision = invert_with_ridge(&w)?;
        Ok((precision, n_iter))
    }
}

/// Maximum number of inner coordinate-descent sweeps used to solve each
/// row's per-variable Lasso regression subproblem.
const LASSO_INNER_MAX_ITER: usize = 100;

/// For row/column `j`, extract the `(n-1) x (n-1)` submatrix of the working
/// covariance `w` spanning every other row/column, and the `(n-1)`-vector of
/// empirical covariances between variable `j` and each other variable. Also
/// returns the mapping from the compacted local indices used in those two
/// objects back to global variable indices.
fn extract_row_block(
    w: &Array2<f64>,
    emp_cov: &Array2<f64>,
    j: usize,
) -> (Array2<f64>, Array1<f64>, Vec<usize>) {
    let n = w.nrows();
    let idx_map: Vec<usize> = (0..n).filter(|&k| k != j).collect();
    let m = idx_map.len();

    let mut w_rr = Array2::zeros((m, m));
    let mut s_rj = Array1::zeros(m);
    for (local_row, &global_row) in idx_map.iter().enumerate() {
        s_rj[local_row] = emp_cov[[global_row, j]];
        for (local_col, &global_col) in idx_map.iter().enumerate() {
            w_rr[[local_row, local_col]] = w[[global_row, global_col]];
        }
    }

    (w_rr, s_rj, idx_map)
}

/// Solve `argmin_beta 0.5 * beta' gram beta - target' beta + lambda * ||beta||_1`
/// by cyclic coordinate descent, warm-started from (and written back into)
/// `beta`. This is the per-row Lasso regression subproblem at the heart of
/// the graphical lasso block coordinate descent algorithm: `gram` is the
/// current working-covariance submatrix for the other variables and
/// `target` is the empirical covariance between the target variable and
/// each of them.
fn solve_lasso_cd(
    gram: &Array2<f64>,
    target: &Array1<f64>,
    beta: &mut Array1<f64>,
    lambda: f64,
    tol: f64,
) {
    let p = target.len();
    if p == 0 {
        return;
    }

    for _ in 0..LASSO_INNER_MAX_ITER {
        let mut max_change = 0.0_f64;

        for k in 0..p {
            let gram_kk = gram[[k, k]];
            if gram_kk <= 0.0 {
                // A degenerate (zero-variance) coordinate: leave it at zero
                // rather than dividing by zero.
                max_change = max_change.max(beta[k].abs());
                beta[k] = 0.0;
                continue;
            }

            let mut dot = 0.0;
            for l in 0..p {
                if l != k {
                    dot += gram[[k, l]] * beta[l];
                }
            }
            let partial_residual = target[k] - dot;
            let new_val = soft_threshold(partial_residual, lambda) / gram_kk;

            max_change = max_change.max((new_val - beta[k]).abs());
            beta[k] = new_val;
        }

        if max_change < tol {
            break;
        }
    }
}

/// Invert `matrix`, adding a small (geometrically escalating) ridge to the
/// diagonal only if the direct inversion fails because the matrix is
/// (near-)singular.
fn invert_with_ridge(matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = matrix.nrows();
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    if let Ok(inv) = matrix.inv() {
        return Ok(inv);
    }

    let mut ridge = 1e-10;
    for _ in 0..12 {
        let mut regularized = matrix.clone();
        for i in 0..n {
            regularized[[i, i]] += ridge;
        }
        if let Ok(inv) = regularized.inv() {
            return Ok(inv);
        }
        ridge *= 10.0;
    }

    Err(SklearsError::NumericalError(
        "GraphicalLasso: failed to invert matrix even after ridge regularization".to_string(),
    ))
}

/// Element-wise soft-thresholding operator: `sign(x) * max(|x| - threshold, 0)`.
fn soft_threshold(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

impl GraphicalLasso<GraphicalLassoTrained> {
    /// Reconstruct a fitted estimator from previously computed parameters.
    ///
    /// Used by the serialization layer to rebuild a fitted model from its
    /// stored state. The hyperparameters (`mode`, `tol`, `max_iter`,
    /// `assume_centered`) are restored from the supplied values; `assume_centered`
    /// is inferred from a zero location.
    #[allow(clippy::too_many_arguments)]
    pub fn from_fitted(
        covariance: Array2<f64>,
        precision: Array2<f64>,
        location: Array1<f64>,
        alpha: f64,
        n_iter: usize,
        mode: String,
        tol: f64,
        max_iter: usize,
    ) -> Self {
        let assume_centered = location.iter().all(|&value| value == 0.0);
        Self {
            state: GraphicalLassoTrained {
                covariance,
                precision,
                location,
                alpha,
                n_iter,
            },
            alpha,
            mode,
            tol,
            max_iter,
            assume_centered,
        }
    }

    /// Get the covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix (sparse inverse covariance)
    pub fn get_precision(&self) -> &Array2<f64> {
        &self.state.precision
    }

    /// Get the location (mean)
    pub fn get_location(&self) -> &Array1<f64> {
        &self.state.location
    }

    /// Get the regularization parameter used
    pub fn get_alpha(&self) -> f64 {
        self.state.alpha
    }

    /// Get the number of iterations performed
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::essentials::Normal;
    use scirs2_core::random::{Distribution, SeedableRng, StdRng};

    /// Synthetic data with a known block-sparse dependency structure:
    /// features `0..block_size` all load on one shared latent factor (a
    /// real, recoverable partial-correlation edge between every pair inside
    /// the block), and the remaining features are independent noise (no
    /// edges, either within the block-vs-noise boundary or among the noise
    /// features themselves).
    fn generate_block_sparse_data(
        n_samples: usize,
        n_features: usize,
        block_size: usize,
        seed: u64,
    ) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).expect("standard normal parameters are always valid");

        let mut data = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            let factor = normal.sample(&mut rng);
            for j in 0..n_features {
                let noise = normal.sample(&mut rng);
                data[[i, j]] = if j < block_size {
                    0.9 * factor + 0.3 * noise
                } else {
                    noise
                };
            }
        }
        data
    }

    fn off_diagonal_mass(matrix: &Array2<f64>) -> f64 {
        let n = matrix.nrows();
        (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j)
            .map(|(i, j)| matrix[[i, j]].abs())
            .sum()
    }

    /// Regression test for the coordinate-descent parity bug: an even
    /// `max_iter` (including the documented default, 100) must no longer
    /// silently degenerate the fitted precision matrix toward the identity
    /// matrix, and fits with consecutive even/odd `max_iter` values must
    /// agree once both have converged.
    #[test]
    fn test_parity_bug_fixed_even_max_iter_matches_odd() {
        let data = generate_block_sparse_data(200, 4, 2, 123);

        let fit_even = GraphicalLasso::new()
            .alpha(0.1)
            .fit(&data.view(), &())
            .expect("fit with the default (even) max_iter should succeed");
        assert_eq!(
            fit_even.max_iter, 100,
            "this test assumes the documented default max_iter"
        );

        let fit_odd = GraphicalLasso::new()
            .alpha(0.1)
            .max_iter(101)
            .fit(&data.view(), &())
            .expect("fit with an odd max_iter should succeed");

        let precision_even = fit_even.get_precision();
        let precision_odd = fit_odd.get_precision();

        // The fitted precision must reflect the real dependency structure,
        // not collapse to a trivial identity-like matrix (the pre-fix
        // behavior for every even `max_iter`).
        let identity_distance: f64 = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .map(|(i, j)| {
                let target = if i == j { 1.0 } else { 0.0 };
                (precision_even[[i, j]] - target).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            identity_distance > 0.1,
            "precision matrix from an even max_iter should not collapse to the identity \
             (distance from identity = {identity_distance})"
        );
        assert!(
            precision_even[[0, 1]].abs() > 0.05,
            "the genuine edge between the two correlated features should survive fitting, \
             got precision[0,1] = {}",
            precision_even[[0, 1]]
        );

        // Parity independence: an even and an odd max_iter must agree once
        // both have converged, rather than oscillating between two
        // different answers.
        for i in 0..4 {
            for j in 0..4 {
                let diff = (precision_even[[i, j]] - precision_odd[[i, j]]).abs();
                assert!(
                    diff < 1e-3,
                    "precision[{i},{j}] differs between even/odd max_iter: \
                     even={}, odd={}, diff={diff}",
                    precision_even[[i, j]],
                    precision_odd[[i, j]]
                );
            }
        }
    }

    /// Required regression test: `get_covariance()` must be consistent with
    /// `get_precision()` (`covariance ~= precision^-1`), not the alpha-
    /// invariant raw empirical covariance.
    #[test]
    fn test_covariance_consistent_with_precision_inverse() {
        let data = generate_block_sparse_data(150, 5, 3, 77);

        let fitted = GraphicalLasso::new()
            .alpha(0.2)
            .fit(&data.view(), &())
            .expect("fit should succeed");

        let covariance = fitted.get_covariance();
        let precision = fitted.get_precision();
        let n = covariance.nrows();

        let product = covariance.dot(precision);
        for i in 0..n {
            for j in 0..n {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[[i, j]] - target).abs() < 1e-6,
                    "covariance @ precision should be ~identity at [{i},{j}], got {}",
                    product[[i, j]]
                );
            }
        }
    }

    /// `get_covariance()` must respond to `alpha`: a heavily regularized fit
    /// should shrink toward a diagonal covariance, and must differ
    /// noticeably from the raw empirical covariance (the pre-fix bug had
    /// `get_covariance()` always equal to the latter, regardless of alpha).
    #[test]
    fn test_get_covariance_responds_to_alpha() {
        let data = generate_block_sparse_data(150, 5, 3, 77);

        let heavily_regularized = GraphicalLasso::new()
            .alpha(0.5)
            .fit(&data.view(), &())
            .expect("fit should succeed");
        let lightly_regularized = GraphicalLasso::new()
            .alpha(1e-4)
            .fit(&data.view(), &())
            .expect("fit should succeed");

        let heavy_mass = off_diagonal_mass(heavily_regularized.get_covariance());
        let light_mass = off_diagonal_mass(lightly_regularized.get_covariance());
        assert!(
            heavy_mass < light_mass,
            "get_covariance() should shrink toward a diagonal matrix as alpha grows, i.e. \
             respond to alpha rather than returning a fixed covariance: \
             heavy_mass={heavy_mass}, light_mass={light_mass}"
        );

        let empirical = EmpiricalCovariance::new()
            .fit(&data.view(), &())
            .expect("empirical covariance should fit");
        let n = heavily_regularized.get_covariance().nrows();
        let mut max_abs_diff = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let diff = (heavily_regularized.get_covariance()[[i, j]]
                    - empirical.get_covariance()[[i, j]])
                    .abs();
                max_abs_diff = max_abs_diff.max(diff);
            }
        }
        assert!(
            max_abs_diff > 1e-3,
            "a heavily regularized GraphicalLasso's covariance should differ noticeably \
             from the raw empirical covariance, max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn test_precision_matrix_is_symmetric() {
        let data = generate_block_sparse_data(120, 6, 3, 9);
        let fitted = GraphicalLasso::new()
            .alpha(0.15)
            .fit(&data.view(), &())
            .expect("fit should succeed");

        let precision = fitted.get_precision();
        let n = precision.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (precision[[i, j]] - precision[[j, i]]).abs() < 1e-8,
                    "precision matrix should be symmetric at [{i},{j}]: {} vs {}",
                    precision[[i, j]],
                    precision[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_single_feature_trivial_case() {
        let data = array![[1.0], [2.0], [3.0], [1.5], [2.5], [4.0]];
        let fitted = GraphicalLasso::new()
            .alpha(0.1)
            .fit(&data.view(), &())
            .expect("single-feature fit should succeed");

        let precision = fitted.get_precision();
        let covariance = fitted.get_covariance();
        assert_eq!(precision.dim(), (1, 1));
        assert_eq!(covariance.dim(), (1, 1));
        assert!(precision[[0, 0]] > 0.0);
        assert!((covariance[[0, 0]] * precision[[0, 0]] - 1.0).abs() < 1e-6);
    }

    /// `alpha <= 0` should recover the ordinary (unregularized) maximum
    /// likelihood precision matrix, i.e. the direct inverse of the
    /// empirical covariance -- exercising the early-return path added for
    /// that limit.
    #[test]
    fn test_unregularized_alpha_matches_empirical_inverse() {
        let data = generate_block_sparse_data(100, 4, 2, 55);

        let fitted = GraphicalLasso::new()
            .alpha(0.0)
            .fit(&data.view(), &())
            .expect("unregularized fit should succeed");

        let empirical = EmpiricalCovariance::new()
            .fit(&data.view(), &())
            .expect("empirical covariance should fit");
        let expected_precision = invert_with_ridge(empirical.get_covariance())
            .expect("empirical covariance should be invertible");

        let precision = fitted.get_precision();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (precision[[i, j]] - expected_precision[[i, j]]).abs() < 1e-6,
                    "unregularized GraphicalLasso should match the raw MLE precision at \
                     [{i},{j}]: got {}, expected {}",
                    precision[[i, j]],
                    expected_precision[[i, j]]
                );
            }
        }
    }
}
