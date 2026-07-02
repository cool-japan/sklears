//! Adversarial examples and robust dataset generators

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Normal, RngExt};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashSet;

/// Standard deviation of the small Gaussian jitter added on top of a deterministic
/// covariate shift in [`make_covariate_shift`]. This keeps repeated calls with
/// different seeds from being bit-identical to a pure deterministic translation.
const COVARIATE_SHIFT_JITTER_STD: f64 = 0.01;

/// Generate a label-noise-corrupted copy of `y` by randomly flipping labels.
///
/// Each label is, independently and with probability `flip_rate`, replaced by a
/// uniformly random *different* class (the original class is never chosen as its
/// own replacement). Labels are otherwise left unchanged.
///
/// # Errors
/// Returns [`SklearsError::InvalidInput`] if `flip_rate` is not in `[0, 1]` or if
/// `n_classes < 2` (at least two classes are required so a different class always
/// exists to flip to).
pub fn make_label_noise(
    y: &Array1<i32>,
    flip_rate: f64,
    n_classes: usize,
    random_state: Option<u64>,
) -> Result<Array1<i32>> {
    if !(0.0..=1.0).contains(&flip_rate) {
        return Err(SklearsError::InvalidInput(
            "flip_rate must be in [0, 1]".to_string(),
        ));
    }

    if n_classes < 2 {
        return Err(SklearsError::InvalidInput(
            "n_classes must be at least 2".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut out = y.clone();
    for label in out.iter_mut() {
        if rng.random::<f64>() < flip_rate {
            // Defensive: treat the input label modulo n_classes so out-of-range
            // labels never cause an out-of-bounds class index.
            let original = (*label).rem_euclid(n_classes as i32) as usize;
            let offset = rng.random_range(1..n_classes); // in [1, n_classes)
            *label = ((original + offset) % n_classes) as i32; // guaranteed != original
        }
    }

    Ok(out)
}

/// Contaminate a dataset with outliers drawn from a scaled Gaussian centered on the
/// per-feature mean/std of the original (uncontaminated) data.
///
/// A random subset of rows (size `round(n_samples * contamination_frac)`) is
/// replaced with points sampled as `mean[j] + outlier_scale * std[j] * N(0, 1)` per
/// feature `j`, where `mean`/`std` are computed from the original `x`. The returned
/// boolean mask marks which rows were replaced.
///
/// # Errors
/// Returns [`SklearsError::InvalidInput`] if `contamination_frac` is not in
/// `[0, 1]`, if `outlier_scale <= 0`, or if `x` has no rows.
pub fn make_outlier_contamination(
    x: &Array2<f64>,
    contamination_frac: f64,
    outlier_scale: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<bool>)> {
    if !(0.0..=1.0).contains(&contamination_frac) {
        return Err(SklearsError::InvalidInput(
            "contamination_frac must be in [0, 1]".to_string(),
        ));
    }

    if outlier_scale <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "outlier_scale must be positive".to_string(),
        ));
    }

    if x.nrows() == 0 {
        return Err(SklearsError::InvalidInput(
            "x must have at least one row".to_string(),
        ));
    }

    let (n_samples, n_features) = (x.nrows(), x.ncols());
    let mut means = vec![0.0_f64; n_features];
    let mut stds = vec![0.0_f64; n_features];
    for j in 0..n_features {
        let col = x.column(j);
        let m = col.mean().expect("column has at least one element");
        let v = col.var(0.0);
        means[j] = m;
        stds[j] = v.sqrt().max(1e-12); // guard against a zero-variance (constant) column
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let n_outliers = (n_samples as f64 * contamination_frac) as usize;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    let outlier_indices: HashSet<usize> = indices[..n_outliers].iter().copied().collect();

    let mut out = x.clone();
    let mut is_outlier = Array1::from_elem(n_samples, false);
    let normal_unit = Normal::new(0.0, 1.0).expect("operation should succeed");
    for &row in &outlier_indices {
        is_outlier[row] = true;
        for j in 0..n_features {
            let z: f64 = rng.sample(normal_unit);
            out[[row, j]] = means[j] + outlier_scale * stds[j] * z;
        }
    }

    Ok((out, is_outlier))
}

/// Apply a covariate shift to `x_train` by adding a fixed per-feature `shift`
/// vector to every row, plus a small fixed Gaussian jitter (std
/// [`COVARIATE_SHIFT_JITTER_STD`]) so repeated calls with different seeds are not
/// bit-identical to a purely deterministic shift.
///
/// # Errors
/// Returns [`SklearsError::InvalidInput`] if `shift.len() != x_train.ncols()`.
pub fn make_covariate_shift(
    x_train: &Array2<f64>,
    shift: &Array1<f64>,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if shift.len() != x_train.ncols() {
        return Err(SklearsError::InvalidInput(
            "shift length must match the number of columns in x_train".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = Normal::new(0.0, COVARIATE_SHIFT_JITTER_STD).expect("operation should succeed");
    let (n_samples, n_features) = (x_train.nrows(), x_train.ncols());
    let mut out = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            out[[i, j]] = x_train[[i, j]] + shift[j] + rng.sample::<f64, _>(normal);
        }
    }

    Ok(out)
}

/// Apply a classic Fast Gradient Sign Method (FGSM) style perturbation:
/// `x + epsilon * sign(gradient_sign)`. This is fully deterministic and takes no
/// `random_state`.
///
/// # Errors
/// Returns [`SklearsError::InvalidInput`] if `x` and `gradient_sign` have
/// different shapes, or if `epsilon < 0`.
pub fn make_fgsm_style_perturbation(
    x: &Array2<f64>,
    gradient_sign: &Array2<f64>,
    epsilon: f64,
) -> Result<Array2<f64>> {
    if x.shape() != gradient_sign.shape() {
        return Err(SklearsError::InvalidInput(
            "x and gradient_sign must have the same shape".to_string(),
        ));
    }

    if epsilon < 0.0 {
        return Err(SklearsError::InvalidInput(
            "epsilon must be non-negative".to_string(),
        ));
    }

    let mut out = x.clone();
    for (o, &g) in out.iter_mut().zip(gradient_sign.iter()) {
        let sign = if g > 0.0 {
            1.0
        } else if g < 0.0 {
            -1.0
        } else {
            0.0
        };
        *o += epsilon * sign;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::StandardNormal;

    fn iid_standard_normal(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.sample::<f64, _>(StandardNormal);
            }
        }
        data
    }

    // ---- make_label_noise ----

    #[test]
    fn test_make_label_noise_zero_flip_rate_is_identity() {
        let y = Array1::from_elem(500, 2);
        let out = make_label_noise(&y, 0.0, 5, Some(0)).expect("operation should succeed");
        assert_eq!(out, y);
    }

    #[test]
    fn test_make_label_noise_full_flip_rate_always_changes() {
        let y = Array1::from_elem(2000, 0);
        let out = make_label_noise(&y, 1.0, 5, Some(1)).expect("operation should succeed");
        assert!(out.iter().zip(y.iter()).all(|(a, b)| a != b));
    }

    #[test]
    fn test_make_label_noise_half_flip_rate_fraction_in_range() {
        let y = Array1::from_elem(4000, 0);
        let out = make_label_noise(&y, 0.5, 4, Some(2)).expect("operation should succeed");
        let changed = out.iter().zip(y.iter()).filter(|(a, b)| a != b).count();
        let frac = changed as f64 / y.len() as f64;
        assert!(
            (0.35..=0.65).contains(&frac),
            "fraction changed {frac} outside expected range"
        );
    }

    #[test]
    fn test_make_label_noise_invalid_flip_rate() {
        let y = Array1::from_elem(10, 0);
        assert!(make_label_noise(&y, 1.5, 5, Some(0)).is_err());
    }

    #[test]
    fn test_make_label_noise_invalid_n_classes() {
        let y = Array1::from_elem(10, 0);
        assert!(make_label_noise(&y, 0.5, 1, Some(0)).is_err());
    }

    #[test]
    fn test_make_label_noise_deterministic_with_seed() {
        let y = Array1::from_elem(1000, 1);
        let out1 = make_label_noise(&y, 0.5, 5, Some(42)).expect("operation should succeed");
        let out2 = make_label_noise(&y, 0.5, 5, Some(42)).expect("operation should succeed");
        assert_eq!(out1, out2);
    }

    // ---- make_outlier_contamination ----

    #[test]
    fn test_make_outlier_contamination_shape_and_count() {
        let x = iid_standard_normal(1000, 4, 1);
        let (out, is_outlier) =
            make_outlier_contamination(&x, 0.1, 15.0, Some(7)).expect("operation should succeed");

        assert_eq!(out.shape(), x.shape());
        assert_eq!(is_outlier.len(), 1000);
        assert_eq!(is_outlier.iter().filter(|&&b| b).count(), 100);
    }

    #[test]
    fn test_make_outlier_contamination_outliers_are_far_out() {
        let x = iid_standard_normal(1000, 4, 1);
        let col0 = x.column(0);
        let mean0 = col0.mean().expect("column has at least one element");
        let std0 = col0.var(0.0).sqrt();

        let (out, is_outlier) =
            make_outlier_contamination(&x, 0.1, 15.0, Some(7)).expect("operation should succeed");

        for row in 0..out.nrows() {
            if is_outlier[row] {
                assert!(
                    (out[[row, 0]] - mean0).abs() > 3.0 * std0,
                    "contaminated row {row} is not far enough from the mean"
                );
            }
        }
    }

    #[test]
    fn test_make_outlier_contamination_invalid_input() {
        let x = iid_standard_normal(10, 2, 1);
        assert!(make_outlier_contamination(&x, 1.5, 15.0, Some(0)).is_err());
        assert!(make_outlier_contamination(&x, 0.1, 0.0, Some(0)).is_err());
        assert!(make_outlier_contamination(&x, 0.1, -1.0, Some(0)).is_err());

        let empty = Array2::<f64>::zeros((0, 2));
        assert!(make_outlier_contamination(&empty, 0.1, 15.0, Some(0)).is_err());
    }

    #[test]
    fn test_make_outlier_contamination_deterministic_with_seed() {
        let x = iid_standard_normal(200, 3, 5);
        let (out1, mask1) =
            make_outlier_contamination(&x, 0.2, 10.0, Some(99)).expect("operation should succeed");
        let (out2, mask2) =
            make_outlier_contamination(&x, 0.2, 10.0, Some(99)).expect("operation should succeed");
        assert_eq!(out1, out2);
        assert_eq!(mask1, mask2);
    }

    // ---- make_covariate_shift ----

    #[test]
    fn test_make_covariate_shift_shape_and_mean() {
        let x_train = Array2::<f64>::zeros((100, 2));
        let shift = scirs2_core::ndarray::array![5.0, -3.0];

        let out =
            make_covariate_shift(&x_train, &shift, Some(3)).expect("operation should succeed");
        assert_eq!(out.shape(), x_train.shape());

        for j in 0..x_train.ncols() {
            let input_mean = x_train
                .column(j)
                .mean()
                .expect("column has at least one element");
            let output_mean = out
                .column(j)
                .mean()
                .expect("column has at least one element");
            let expected = input_mean + shift[j];
            assert!(
                (output_mean - expected).abs() < 0.05,
                "column {j} mean {output_mean} not within 0.05 of expected {expected}"
            );
        }
    }

    #[test]
    fn test_make_covariate_shift_invalid_input() {
        let x_train = Array2::<f64>::zeros((10, 3));
        let shift = scirs2_core::ndarray::array![1.0, 2.0];
        assert!(make_covariate_shift(&x_train, &shift, Some(0)).is_err());
    }

    #[test]
    fn test_make_covariate_shift_deterministic_with_seed() {
        let x_train = iid_standard_normal(50, 2, 9);
        let shift = scirs2_core::ndarray::array![1.0, -1.0];
        let out1 =
            make_covariate_shift(&x_train, &shift, Some(11)).expect("operation should succeed");
        let out2 =
            make_covariate_shift(&x_train, &shift, Some(11)).expect("operation should succeed");
        assert_eq!(out1, out2);
    }

    // ---- make_fgsm_style_perturbation ----

    #[test]
    fn test_make_fgsm_style_perturbation_known_signs() {
        let x = scirs2_core::ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let gradient_sign =
            scirs2_core::ndarray::array![[1.0, -1.0, 0.0], [-5.0, 0.5, 0.0], [0.0, -2.0, 3.0]];
        let epsilon = 0.1;

        let out = make_fgsm_style_perturbation(&x, &gradient_sign, epsilon)
            .expect("operation should succeed");

        for i in 0..3 {
            for j in 0..3 {
                let g = gradient_sign[[i, j]];
                let expected = if g > 0.0 {
                    x[[i, j]] + epsilon
                } else if g < 0.0 {
                    x[[i, j]] - epsilon
                } else {
                    x[[i, j]]
                };
                assert!(
                    (out[[i, j]] - expected).abs() < 1e-12,
                    "mismatch at ({i}, {j}): got {}, expected {expected}",
                    out[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_make_fgsm_style_perturbation_invalid_shape() {
        let x = Array2::<f64>::zeros((3, 3));
        let gradient_sign = Array2::<f64>::zeros((2, 3));
        assert!(make_fgsm_style_perturbation(&x, &gradient_sign, 0.1).is_err());
    }

    #[test]
    fn test_make_fgsm_style_perturbation_invalid_epsilon() {
        let x = Array2::<f64>::zeros((3, 3));
        let gradient_sign = Array2::<f64>::zeros((3, 3));
        assert!(make_fgsm_style_perturbation(&x, &gradient_sign, -0.1).is_err());
    }
}
