//! Advanced statistical distributions and methods

use scirs2_core::linalg::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{RngExt, StandardNormal};
use sklears_core::error::{Result, SklearsError};

/// Draws one row `mean + L @ z` where `z` is a fresh standard-normal vector,
/// given a precomputed Cholesky lower-triangular factor `l` of the covariance.
fn sample_mvn_row(rng: &mut StdRng, mean: &Array1<f64>, l: &Array2<f64>) -> Array1<f64> {
    let n = mean.len();
    let mut z = Array1::zeros(n);
    for k in 0..n {
        z[k] = rng.sample::<f64, _>(StandardNormal);
    }
    let mut x = Array1::zeros(n);
    for r in 0..n {
        let mut acc = mean[r];
        for c in 0..=r {
            acc += l[[r, c]] * z[c];
        }
        x[r] = acc;
    }
    x
}

/// Generates samples from a multivariate normal distribution `N(mean, cov)`
/// via Cholesky factorization of the covariance matrix.
pub fn make_multivariate_normal(
    n_samples: usize,
    mean: &Array1<f64>,
    cov: &Array2<f64>,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be greater than zero".to_string(),
        ));
    }
    if mean.is_empty() {
        return Err(SklearsError::InvalidInput(
            "mean must have at least one element".to_string(),
        ));
    }
    if cov.nrows() != cov.ncols() {
        return Err(SklearsError::InvalidInput(format!(
            "covariance matrix must be square, found {}x{}",
            cov.nrows(),
            cov.ncols()
        )));
    }
    if cov.nrows() != mean.len() {
        return Err(SklearsError::InvalidInput(format!(
            "covariance matrix dimensions ({0}x{0}) must match the length of mean ({1})",
            cov.nrows(),
            mean.len()
        )));
    }

    let chol = cholesky_ndarray(cov).map_err(|e| {
        SklearsError::InvalidInput(format!(
            "covariance matrix must be positive definite for Cholesky sampling: {e}"
        ))
    })?;

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let n = mean.len();
    let mut x = Array2::zeros((n_samples, n));
    for i in 0..n_samples {
        let row = sample_mvn_row(&mut rng, mean, &chol.l);
        for j in 0..n {
            x[[i, j]] = row[j];
        }
    }
    Ok(x)
}

/// Generates samples from a Gaussian mixture model with the given per-component
/// weights, means, and covariances. Returns the sampled features together with
/// the integer component label each sample was drawn from.
pub fn make_gaussian_mixture(
    n_samples: usize,
    weights: &[f64],
    means: &[Array1<f64>],
    covs: &[Array2<f64>],
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    let k = weights.len();
    if k == 0 {
        return Err(SklearsError::InvalidInput(
            "weights must contain at least one component".to_string(),
        ));
    }
    if means.len() != k {
        return Err(SklearsError::InvalidInput(format!(
            "means must contain exactly {k} components to match weights, found {}",
            means.len()
        )));
    }
    if covs.len() != k {
        return Err(SklearsError::InvalidInput(format!(
            "covs must contain exactly {k} components to match weights, found {}",
            covs.len()
        )));
    }
    if weights.iter().any(|&w| w < 0.0) {
        return Err(SklearsError::InvalidInput(
            "all component weights must be non-negative".to_string(),
        ));
    }
    let weight_sum: f64 = weights.iter().sum();
    if (weight_sum - 1.0).abs() > 1e-6 {
        return Err(SklearsError::InvalidInput(format!(
            "component weights must sum to 1.0, found {weight_sum}"
        )));
    }
    let n_features = means[0].len();
    if n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "component means must have at least one feature".to_string(),
        ));
    }
    for (idx, component_mean) in means.iter().enumerate() {
        if component_mean.len() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "component {idx} mean has {} features, expected {n_features}",
                component_mean.len()
            )));
        }
    }
    for (idx, component_cov) in covs.iter().enumerate() {
        if component_cov.nrows() != n_features || component_cov.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "component {idx} covariance must be {n_features}x{n_features}, found {}x{}",
                component_cov.nrows(),
                component_cov.ncols()
            )));
        }
    }
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be greater than zero".to_string(),
        ));
    }

    // Precompute a Cholesky factor per component once (O(k * n^3)) rather than
    // re-factorizing on every sample draw (which would be O(n_samples * n^3)).
    let cholesky_factors: Vec<Array2<f64>> = covs
        .iter()
        .map(|c| {
            cholesky_ndarray(c).map(|res| res.l).map_err(|e| {
                SklearsError::InvalidInput(format!(
                    "component covariance must be positive definite: {e}"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut cumulative = Vec::with_capacity(weights.len());
    let mut running = 0.0;
    for &w in weights {
        running += w;
        cumulative.push(running);
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut x = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let u: f64 = rng.random::<f64>();
        let comp = cumulative
            .iter()
            .position(|&c| u <= c)
            .unwrap_or(cumulative.len() - 1);
        labels[i] = comp as i32;
        let row = sample_mvn_row(&mut rng, &means[comp], &cholesky_factors[comp]);
        for j in 0..n_features {
            x[[i, j]] = row[j];
        }
    }
    Ok((x, labels))
}

/// Generates `n_features` equicorrelated Gaussian features (unit variance,
/// pairwise Pearson correlation equal to `correlation`) for `n_samples` rows.
pub fn make_correlated_features(
    n_samples: usize,
    n_features: usize,
    correlation: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be greater than zero".to_string(),
        ));
    }
    if n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_features must be greater than zero".to_string(),
        ));
    }
    if n_features > 1 {
        let lower_bound = -1.0 / (n_features as f64 - 1.0);
        if !(lower_bound < correlation && correlation < 1.0) {
            return Err(SklearsError::InvalidInput(format!(
                "correlation must lie strictly between {lower_bound} and 1.0 for n_features = \
                 {n_features} to yield a positive-definite equicorrelated covariance, found \
                 {correlation}"
            )));
        }
    }

    let mean = Array1::zeros(n_features);
    let mut cov = Array2::<f64>::eye(n_features);
    for i in 0..n_features {
        for j in 0..n_features {
            if i != j {
                cov[[i, j]] = correlation;
            }
        }
    }

    make_multivariate_normal(n_samples, &mean, &cov, random_state)
}

/// Generates a `n_samples x n_features` low-rank matrix `X = U * diag(s) * V^T`
/// where `U`, `V` have random orthonormal columns (obtained via QR of random
/// Gaussian matrices) and the singular-value profile `s` blends a fast
/// Gaussian-decay "low rank" part with a slow exponential-decay "tail" part,
/// mirroring scikit-learn's `make_low_rank_matrix`.
pub fn make_low_rank_matrix(
    n_samples: usize,
    n_features: usize,
    effective_rank: usize,
    tail_strength: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be greater than zero".to_string(),
        ));
    }
    if n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_features must be greater than zero".to_string(),
        ));
    }
    if effective_rank == 0 {
        return Err(SklearsError::InvalidInput(
            "effective_rank must be greater than zero".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&tail_strength) {
        return Err(SklearsError::InvalidInput(format!(
            "tail_strength must lie in [0.0, 1.0], found {tail_strength}"
        )));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };
    let n = n_samples.min(n_features);

    let mut g_u = Array2::zeros((n_samples, n));
    for i in 0..n_samples {
        for j in 0..n {
            g_u[[i, j]] = rng.sample::<f64, _>(StandardNormal);
        }
    }
    let mut g_v = Array2::zeros((n_features, n));
    for i in 0..n_features {
        for j in 0..n {
            g_v[[i, j]] = rng.sample::<f64, _>(StandardNormal);
        }
    }

    let q_u = qr_ndarray(&g_u).expect("QR decomposition succeeds for a non-empty matrix");
    let q_v = qr_ndarray(&g_v).expect("QR decomposition succeeds for a non-empty matrix");
    let u = q_u.q.slice(s![.., 0..n]).to_owned();
    let v = q_v.q.slice(s![.., 0..n]).to_owned();

    let mut s_diag = Array2::zeros((n, n));
    for k in 0..n {
        let idx = k as f64;
        let low_rank = (1.0 - tail_strength) * (-(idx / effective_rank as f64).powi(2)).exp();
        let tail = tail_strength * (-0.1 * idx / effective_rank as f64).exp();
        s_diag[[k, k]] = low_rank + tail;
    }

    Ok(u.dot(&s_diag).dot(&v.t()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_multivariate_normal_shape_and_empirical_mean() {
        let mean = scirs2_core::ndarray::array![5.0, -2.0];
        let cov = scirs2_core::ndarray::array![[2.0, 0.5], [0.5, 1.0]];
        let x = make_multivariate_normal(5000, &mean, &cov, Some(42))
            .expect("operation should succeed");
        assert_eq!(x.shape(), &[5000, 2]);

        let mean0 = x.column(0).mean().expect("operation should succeed");
        let mean1 = x.column(1).mean().expect("operation should succeed");
        assert!((mean0 - 5.0).abs() < 0.2, "mean0 = {mean0}");
        assert!((mean1 - (-2.0)).abs() < 0.2, "mean1 = {mean1}");
    }

    #[test]
    fn test_make_multivariate_normal_seed_determinism() {
        let mean = scirs2_core::ndarray::array![5.0, -2.0];
        let cov = scirs2_core::ndarray::array![[2.0, 0.5], [0.5, 1.0]];
        let x1 =
            make_multivariate_normal(200, &mean, &cov, Some(7)).expect("operation should succeed");
        let x2 =
            make_multivariate_normal(200, &mean, &cov, Some(7)).expect("operation should succeed");
        assert_eq!(x1, x2);
    }

    #[test]
    fn test_make_multivariate_normal_rejects_non_positive_definite_covariance() {
        let mean = scirs2_core::ndarray::array![0.0, 0.0];
        let cov = scirs2_core::ndarray::array![[1.0, 2.0], [2.0, 1.0]];
        let result = make_multivariate_normal(10, &mean, &cov, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_gaussian_mixture_shape_labels_and_component_means() {
        let weights = [0.3, 0.7];
        let means = [
            scirs2_core::ndarray::array![0.0, 0.0],
            scirs2_core::ndarray::array![10.0, 10.0],
        ];
        let covs = [Array2::<f64>::eye(2), Array2::<f64>::eye(2)];
        let n_samples = 3000;
        let (x, labels) = make_gaussian_mixture(n_samples, &weights, &means, &covs, Some(11))
            .expect("operation should succeed");
        assert_eq!(x.shape(), &[n_samples, 2]);
        assert_eq!(labels.len(), n_samples);
        assert!(labels.iter().all(|&l| l == 0 || l == 1));

        let n_label1 = labels.iter().filter(|&&l| l == 1).count();
        let frac1 = n_label1 as f64 / n_samples as f64;
        assert!((frac1 - 0.7).abs() < 0.05, "frac1 = {frac1}");

        let mut sum0 = [0.0f64, 0.0];
        let mut count0 = 0usize;
        let mut sum1 = [0.0f64, 0.0];
        let mut count1 = 0usize;
        for i in 0..n_samples {
            if labels[i] == 0 {
                sum0[0] += x[[i, 0]];
                sum0[1] += x[[i, 1]];
                count0 += 1;
            } else {
                sum1[0] += x[[i, 0]];
                sum1[1] += x[[i, 1]];
                count1 += 1;
            }
        }
        let mean0 = [sum0[0] / count0 as f64, sum0[1] / count0 as f64];
        let mean1 = [sum1[0] / count1 as f64, sum1[1] / count1 as f64];
        assert!(
            mean0[0].abs() < 1.0 && mean0[1].abs() < 1.0,
            "mean0 = {mean0:?}"
        );
        assert!(
            (mean1[0] - 10.0).abs() < 1.0 && (mean1[1] - 10.0).abs() < 1.0,
            "mean1 = {mean1:?}"
        );
    }

    #[test]
    fn test_make_gaussian_mixture_rejects_invalid_weights() {
        let weights = [0.8, 0.7];
        let means = [
            scirs2_core::ndarray::array![0.0, 0.0],
            scirs2_core::ndarray::array![10.0, 10.0],
        ];
        let covs = [Array2::<f64>::eye(2), Array2::<f64>::eye(2)];
        let result = make_gaussian_mixture(100, &weights, &means, &covs, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_correlated_features_empirical_pearson_correlation() {
        let n_samples = 4000;
        let x =
            make_correlated_features(n_samples, 3, 0.6, Some(3)).expect("operation should succeed");
        assert_eq!(x.shape(), &[n_samples, 3]);

        let col0 = x.column(0);
        let col1 = x.column(1);
        let m0 = col0.mean().expect("operation should succeed");
        let m1 = col1.mean().expect("operation should succeed");

        let mut cov01 = 0.0;
        let mut var0 = 0.0;
        let mut var1 = 0.0;
        for i in 0..n_samples {
            let d0 = col0[i] - m0;
            let d1 = col1[i] - m1;
            cov01 += d0 * d1;
            var0 += d0 * d0;
            var1 += d1 * d1;
        }
        let corr = cov01 / (var0.sqrt() * var1.sqrt());
        assert!((corr - 0.6).abs() < 0.15, "corr = {corr}");
    }

    #[test]
    fn test_make_correlated_features_rejects_out_of_bounds_correlation() {
        let result = make_correlated_features(100, 3, -0.9, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_correlated_features_rejects_zero_samples() {
        let result = make_correlated_features(0, 3, 0.5, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_low_rank_matrix_shape_and_singular_value_decay() {
        let x = make_low_rank_matrix(50, 30, 5, 0.3, Some(5)).expect("operation should succeed");
        assert_eq!(x.shape(), &[50, 30]);

        let svd = svd_ndarray(&x).expect("operation should succeed");
        let top = svd.s[0];
        let bottom = svd.s[svd.s.len() - 1];
        assert!(top > 5.0 * bottom, "top = {top}, bottom = {bottom}");
    }

    #[test]
    fn test_make_low_rank_matrix_rejects_zero_effective_rank() {
        let result = make_low_rank_matrix(20, 10, 0, 0.5, Some(1));
        assert!(result.is_err());
    }
}
