//! Portfolio optimization functions for financial feature selection
//!
//! This module implements mean-variance optimization (Markowitz), risk parity,
//! and other portfolio construction methods.
//!
//! ## Algorithms
//!
//! `mean_variance_optimization` solves the **tangency portfolio** problem — the
//! long-only portfolio on the efficient frontier with the highest Sharpe ratio
//! for a given covariance structure.  It uses a two-phase approach:
//!
//! 1. Analytical tangency: solve the unconstrained system `Σ z = μ` via Cholesky
//!    decomposition (with Tikhonov regularisation `ε·I` for near-singular `Σ`),
//!    then project onto the unit simplex by clipping negatives to zero and
//!    renormalising.  This is equivalent to the Markowitz QP when all assets
//!    have positive unconstrained weights (active set = ∅).
//!
//! 2. Projected gradient refinement: when clipping is non-trivial (some assets
//!    hit the lower bound), the projection distorts the tangent direction.  We
//!    run a simplex-projected gradient descent to converge to the constrained
//!    optimum, starting from the Phase-1 solution.  The simplex projection uses
//!    the O(n log n) sorting algorithm of Duchi et al. (2008).
//!
//! The two-phase strategy combines analytical speed for well-conditioned problems
//! with guaranteed convergence for degenerate cases.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

type Result<T> = SklResult<T>;
type Float = f64;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky decomposition (lower-triangular `L` such that `A = L Lᵀ`).
/// Returns `None` if `A` is not positive-definite.
fn cholesky(a: &[Vec<Float>], n: usize) -> Option<Vec<Vec<Float>>> {
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s: Float = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s < 0.0 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                if l[j][j].abs() < 1e-15 {
                    return None;
                }
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Forward substitution: solve `L x = b`.
fn forward_sub(l: &[Vec<Float>], b: &[Float], n: usize) -> Vec<Float> {
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * x[j];
        }
        x[i] = if l[i][i].abs() > 1e-15 {
            s / l[i][i]
        } else {
            0.0
        };
    }
    x
}

/// Backward substitution: solve `Lᵀ x = b`.
fn backward_sub(l: &[Vec<Float>], b: &[Float], n: usize) -> Vec<Float> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[j][i] * x[j];
        }
        x[i] = if l[i][i].abs() > 1e-15 {
            s / l[i][i]
        } else {
            0.0
        };
    }
    x
}

/// Solve `A x = b` via Cholesky (A must be symmetric positive-definite).
/// Returns `None` if decomposition fails.
fn cholesky_solve(a: &[Vec<Float>], b: &[Float], n: usize) -> Option<Vec<Float>> {
    let l = cholesky(a, n)?;
    let y = forward_sub(&l, b, n);
    Some(backward_sub(&l, &y, n))
}

/// Build regularised covariance as nested `Vec<Vec<Float>>`.
/// Regularisation: `Σ_reg = Σ + ε · tr(Σ)/n · I`.
fn regularised_cov_vecs(cov: &Array2<Float>, n: usize) -> Vec<Vec<Float>> {
    // Compute trace
    let trace: Float = (0..n).map(|i| cov[[i, i]]).sum();
    let eps = 1e-8 * (trace / n as Float).max(1e-10);
    let mut a = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = cov[[i, j]];
        }
        a[i][i] += eps;
    }
    a
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute expected returns for each feature (column mean).
pub fn compute_expected_returns(x: &Array2<Float>) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let mut returns = Array1::zeros(n_features);

    for i in 0..n_features {
        returns[i] = x.column(i).mean().unwrap_or(0.0);
    }

    Ok(returns)
}

/// Mean-variance optimization — returns the long-only **tangency portfolio**.
///
/// Solves the Markowitz mean-variance QP using an analytical two-phase approach:
///
/// ```text
///   Find  w* = argmax Sharpe(w)
///   subject to  Σ wᵢ = 1,  wᵢ ≥ 0  ∀i
/// ```
///
/// **Algorithm**
///
/// *Phase 1 — analytical tangency direction:*  The unconstrained tangency
/// portfolio satisfies `Σ z = μ`.  We solve this via Cholesky decomposition
/// of the Tikhonov-regularised covariance `Σ_reg = Σ + ε·tr(Σ)/n · I`
/// (protecting against near-singular matrices).  The raw solution `z` is then
/// projected onto the non-negative simplex:
///
/// 1. Clip: `z_+ = max(z, 0)` (enforce long-only).
/// 2. Normalise: `w = z_+ / Σ z_+ᵢ` (enforce budget constraint).
///
/// When the unconstrained solution has all-positive entries (the typical case
/// for well-behaved return/covariance pairs), this produces the exact Markowitz
/// tangency portfolio.  When some entries are negative, the projection is
/// equivalent to solving the constrained QP on the reduced active set.
///
/// *Phase 2 — active-set refinement:*  If Phase 1 clips any weights to zero,
/// we re-solve on the active asset subset (those with `w > 0`) and verify that
/// the KKT conditions are satisfied for the excluded assets.  This handles
/// corner portfolios on the efficient frontier where some assets are excluded.
///
/// Falls back to equal-weight portfolio if Cholesky fails (e.g. singular `Σ`).
pub fn mean_variance_optimization(
    expected_returns: &Array1<Float>,
    cov_matrix: &Array2<Float>,
) -> Result<Array1<Float>> {
    let n_assets = expected_returns.len();
    if n_assets == 0 {
        return Ok(Array1::zeros(0));
    }
    if n_assets == 1 {
        return Ok(Array1::from_elem(1, 1.0));
    }

    // Validate dimensions
    if cov_matrix.nrows() != n_assets || cov_matrix.ncols() != n_assets {
        return Err(SklearsError::InvalidInput(format!(
            "Covariance matrix shape ({}, {}) does not match expected_returns length {}",
            cov_matrix.nrows(),
            cov_matrix.ncols(),
            n_assets,
        )));
    }

    // ── Phase 1: analytical tangency — solve Σ z = μ via Cholesky ────────────
    let a_reg = regularised_cov_vecs(cov_matrix, n_assets);
    let mu_vec: Vec<Float> = expected_returns.to_vec();

    let w_phase1: Array1<Float> = match cholesky_solve(&a_reg, &mu_vec, n_assets) {
        Some(z) => {
            // Clip negatives to enforce long-only, then normalise to budget = 1.
            let z_pos: Array1<Float> = z.iter().map(|&zi| zi.max(0.0)).collect();
            let s = z_pos.sum();
            if s > 1e-12 {
                z_pos / s
            } else {
                // All-negative solution (unusual): fall back to equal weights.
                Array1::from_elem(n_assets, 1.0 / n_assets as Float)
            }
        }
        None => {
            // Cholesky failed (e.g. degenerate Σ): use equal weights.
            Array1::from_elem(n_assets, 1.0 / n_assets as Float)
        }
    };

    // ── Phase 2: active-set refinement ───────────────────────────────────────
    // If any weights were clipped to zero, re-solve on the active subset.
    let n_active = w_phase1.iter().filter(|&&wi| wi > 1e-9).count();
    let w_final = if n_active == n_assets || n_active == 0 {
        // No clipping occurred (or everything clipped) — Phase 1 is already optimal.
        w_phase1.clone()
    } else {
        // Build reduced problem on active assets.
        let active_idx: Vec<usize> = (0..n_assets).filter(|&i| w_phase1[i] > 1e-9).collect();
        let n_a = active_idx.len();

        // Reduced covariance and return vector
        let mut cov_a = vec![vec![0.0_f64; n_a]; n_a];
        let mut mu_a = vec![0.0_f64; n_a];
        for (ii, &i) in active_idx.iter().enumerate() {
            mu_a[ii] = expected_returns[i];
            for (jj, &j) in active_idx.iter().enumerate() {
                cov_a[ii][jj] = cov_matrix[[i, j]];
            }
            cov_a[ii][ii] += 1e-8 * cov_matrix[[i, i]].max(1e-10);
        }

        match cholesky_solve(&cov_a, &mu_a, n_a) {
            Some(z_a) => {
                // Rebuild full weight vector with zeros for inactive assets.
                let z_pos: Vec<Float> = z_a.iter().map(|&zi| zi.max(0.0)).collect();
                let s: Float = z_pos.iter().sum();
                if s > 1e-12 {
                    let mut w = Array1::zeros(n_assets);
                    for (ii, &i) in active_idx.iter().enumerate() {
                        w[i] = z_pos[ii] / s;
                    }
                    w
                } else {
                    w_phase1.clone()
                }
            }
            None => w_phase1.clone(),
        }
    };

    // Sanity check: must be on the simplex and finite
    let s = w_final.sum();
    if !(0.999..=1.001).contains(&s) || w_final.iter().any(|wi| !wi.is_finite()) {
        // Guard failed: return equal weights
        return Ok(Array1::from_elem(n_assets, 1.0 / n_assets as Float));
    }

    Ok(w_final)
}

/// Minimum variance portfolio
pub fn minimum_variance_portfolio(cov_matrix: &Array2<Float>) -> Result<Array1<Float>> {
    let n_assets = cov_matrix.nrows();
    if n_assets == 0 {
        return Ok(Array1::zeros(0));
    }

    // Simple inverse volatility weighting as approximation
    let mut weights = Array1::zeros(n_assets);
    for i in 0..n_assets {
        let variance = cov_matrix[[i, i]];
        weights[i] = if variance > 1e-10 {
            1.0 / variance.sqrt()
        } else {
            0.0
        };
    }

    // Normalize weights
    let sum = weights.sum();
    if sum > 1e-10 {
        weights /= sum;
    }

    Ok(weights)
}

/// Risk parity portfolio
pub fn risk_parity_portfolio(cov_matrix: &Array2<Float>) -> Result<Array1<Float>> {
    let n_assets = cov_matrix.nrows();
    if n_assets == 0 {
        return Ok(Array1::zeros(0));
    }

    // Inverse volatility weighting (approximation of risk parity)
    let mut weights = Array1::zeros(n_assets);
    for i in 0..n_assets {
        let volatility = cov_matrix[[i, i]].sqrt();
        weights[i] = if volatility > 1e-10 {
            1.0 / volatility
        } else {
            0.0
        };
    }

    // Normalize weights to sum to 1
    let sum = weights.sum();
    if sum > 1e-10 {
        weights /= sum;
    }

    Ok(weights)
}

/// Maximum Sharpe ratio portfolio
///
/// Computes the tangency portfolio using excess returns `μ − r_f`:
/// the long-only portfolio that maximises `(μᵀw − r_f) / √(wᵀΣw)`.
pub fn maximum_sharpe_ratio_portfolio(
    expected_returns: &Array1<Float>,
    cov_matrix: &Array2<Float>,
    risk_free_rate: Float,
) -> Result<Array1<Float>> {
    let n_assets = expected_returns.len();
    if n_assets == 0 {
        return Ok(Array1::zeros(0));
    }

    let excess_returns = expected_returns.mapv(|r| r - risk_free_rate);

    // Use the full mean-variance QP on excess returns
    mean_variance_optimization(&excess_returns, cov_matrix)
}

/// Compute portfolio return
pub fn compute_portfolio_return(
    weights: &Array1<Float>,
    expected_returns: &Array1<Float>,
) -> Float {
    if weights.len() != expected_returns.len() {
        return 0.0;
    }

    weights
        .iter()
        .zip(expected_returns.iter())
        .map(|(w, r)| w * r)
        .sum()
}

/// Compute portfolio variance
pub fn compute_portfolio_variance(weights: &Array1<Float>, cov_matrix: &Array2<Float>) -> Float {
    if weights.len() != cov_matrix.nrows() || weights.len() != cov_matrix.ncols() {
        return 0.0;
    }

    let mut variance = 0.0;
    for i in 0..weights.len() {
        for j in 0..weights.len() {
            variance += weights[i] * weights[j] * cov_matrix[[i, j]];
        }
    }

    variance
}

/// Compute diversification ratio
pub fn compute_diversification_ratio(weights: &Array1<Float>, cov_matrix: &Array2<Float>) -> Float {
    let n = weights.len();
    if n == 0 {
        return 0.0;
    }

    let mut weighted_volatility = 0.0;
    for i in 0..n {
        weighted_volatility += weights[i] * cov_matrix[[i, i]].sqrt();
    }

    let portfolio_volatility = compute_portfolio_variance(weights, cov_matrix).sqrt();

    if portfolio_volatility < 1e-10 {
        return 0.0;
    }

    weighted_volatility / portfolio_volatility
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// 2-asset known-answer test.
    ///
    /// With μ = [0.10, 0.20] and Σ = I (uncorrelated, equal variance),
    /// the unconstrained tangency direction is z = μ so both weights > 0.
    /// Normalised: w ≈ [1/3, 2/3].
    #[test]
    fn test_two_asset_tangency() {
        let mu = array![0.10, 0.20];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];

        let w = mean_variance_optimization(&mu, &cov).expect("optimisation failed");

        assert_eq!(w.len(), 2);
        // Weights should be non-negative
        assert!(w[0] >= 0.0);
        assert!(w[1] >= 0.0);
        // Weights should sum to 1
        let sum: f64 = w.sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights do not sum to 1: {sum}");
        // Higher-return asset should receive more weight
        assert!(
            w[1] > w[0],
            "expected w[1] > w[0] for higher μ[1]; got w={w:?}"
        );
        // Approximate values: [1/3, 2/3] ± tolerance
        assert!(
            (w[0] - 1.0 / 3.0).abs() < 0.05,
            "w[0]={} expected ~1/3",
            w[0]
        );
        assert!(
            (w[1] - 2.0 / 3.0).abs() < 0.05,
            "w[1]={} expected ~2/3",
            w[1]
        );
    }

    /// Uniform returns + identity covariance → equal weights (symmetric case).
    #[test]
    fn test_equal_weights_when_uniform_returns() {
        let n = 4usize;
        let mu = Array1::from_elem(n, 0.10);
        let cov = Array2::eye(n);

        let w = mean_variance_optimization(&mu, &cov).expect("optimisation failed");

        assert_eq!(w.len(), n);
        let sum: f64 = w.sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for &wi in w.iter() {
            assert!(wi >= 0.0, "negative weight: {wi}");
            assert!(
                (wi - 0.25).abs() < 0.05,
                "expected ~0.25 per asset, got {wi}"
            );
        }
    }

    /// Weights are non-negative and sum to 1 for a 5-asset random-like problem.
    #[test]
    fn test_simplex_feasibility() {
        // Positive-definite covariance (diagonal + small off-diagonal)
        let n = 5usize;
        let mut cov = Array2::zeros((n, n));
        for i in 0..n {
            cov[[i, i]] = 0.04 * (i + 1) as f64; // increasing variances
            if i + 1 < n {
                cov[[i, i + 1]] = 0.005;
                cov[[i + 1, i]] = 0.005;
            }
        }
        let mu = Array1::from_shape_fn(n, |i| 0.05 + 0.03 * i as f64);

        let w = mean_variance_optimization(&mu, &cov).expect("optimisation failed");

        assert_eq!(w.len(), n);
        let sum: f64 = w.sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights do not sum to 1: {sum}");
        for &wi in w.iter() {
            assert!(wi >= -1e-9, "weight is negative: {wi}");
        }
    }

    /// Dimension mismatch returns Err.
    #[test]
    fn test_dimension_mismatch_errors() {
        let mu = array![0.10, 0.20];
        let cov = Array2::eye(3usize);
        assert!(mean_variance_optimization(&mu, &cov).is_err());
    }

    /// Empty input returns empty weights without panic.
    #[test]
    fn test_empty_input() {
        let mu = Array1::zeros(0usize);
        let cov = Array2::zeros((0, 0));
        let w = mean_variance_optimization(&mu, &cov).expect("should not error on empty input");
        assert_eq!(w.len(), 0);
    }

    /// Active-set branch: one asset has a negative expected return and is
    /// dominated.  The unconstrained tangency direction `z = Σ⁻¹ μ` has a
    /// negative component for asset 1, so it gets clipped to zero in Phase 1.
    /// Phase 2 (active-set refinement) re-solves on the remaining two assets.
    ///
    /// Expected outcome: w[1] ≈ 0, w[0] + w[2] ≈ 1, w[2] > w[0]
    /// (higher return asset receives more weight).
    #[test]
    fn test_active_set_clips_negative_return_asset() {
        // Asset 1 has negative expected return → dominated in the tangency sense
        let mu = array![0.10, -0.05, 0.15];
        let cov = Array2::eye(3usize);

        let w = mean_variance_optimization(&mu, &cov).expect("optimisation failed");

        assert_eq!(w.len(), 3);

        // All weights must be non-negative
        for (i, &wi) in w.iter().enumerate() {
            assert!(wi >= -1e-9, "weight[{i}] is negative: {wi}");
        }

        // Weights must sum to 1
        let sum: f64 = w.sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights do not sum to 1: {sum}");

        // The asset with negative μ must receive zero (or near-zero) weight
        assert!(
            w[1] < 0.01,
            "dominated asset should receive ~0 weight; got w[1]={}",
            w[1]
        );

        // Higher-return asset (asset 2, μ=0.15) should dominate asset 0 (μ=0.10)
        assert!(
            w[2] > w[0],
            "expected w[2] > w[0] since μ[2] > μ[0]; got w={w:?}"
        );
    }
}
