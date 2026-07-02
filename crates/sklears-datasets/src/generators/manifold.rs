//! Manifold learning and geometric pattern generators
//!
//! Every generator in this module returns `(coordinates, intrinsic_position)`, where
//! `coordinates` is an `(n_samples, 3)` embedding of the samples in 3-D space and
//! `intrinsic_position` is the manifold's intrinsic 1-D parameter for each sample --
//! useful as a regression target or coloring value for manifold-learning algorithms
//! such as Isomap or LLE.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{RngExt, StandardNormal};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Generate a Swiss roll dataset.
///
/// A 2-D sheet is rolled up into 3-D space following the classic scikit-learn
/// `make_swiss_roll` construction: for each sample a roll parameter `t` is drawn and
/// used to place the point on the spiral `(t * cos(t), y, t * sin(t))`, with `y` drawn
/// independently and uniformly. The intrinsic coordinate returned for each sample is
/// `t`, measured before any noise is added, since that is the manifold's true unrolled
/// position.
pub fn make_swiss_roll(
    n_samples: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(SklearsError::InvalidInput(
            "noise must be non-negative".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut coordinates = Array2::zeros((n_samples, 3));
    let mut intrinsic = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let t = 1.5 * PI * (1.0 + 2.0 * rng.random::<f64>());

        coordinates[[i, 0]] = t * t.cos();
        coordinates[[i, 1]] = 21.0 * rng.random::<f64>();
        coordinates[[i, 2]] = t * t.sin();

        if noise > 0.0 {
            coordinates[[i, 0]] += noise * rng.sample::<f64, _>(StandardNormal);
            coordinates[[i, 1]] += noise * rng.sample::<f64, _>(StandardNormal);
            coordinates[[i, 2]] += noise * rng.sample::<f64, _>(StandardNormal);
        }

        intrinsic[i] = t;
    }

    Ok((coordinates, intrinsic))
}

/// Generate an S-curve dataset.
///
/// Points lie on a 2-D sheet folded into an "S" shape embedded in 3-D space, mirroring
/// scikit-learn's `make_s_curve`. The intrinsic coordinate `t` parameterizes position
/// along the S and is measured before any noise is added.
pub fn make_s_curve(
    n_samples: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(SklearsError::InvalidInput(
            "noise must be non-negative".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut coordinates = Array2::zeros((n_samples, 3));
    let mut intrinsic = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let t = 3.0 * PI * (rng.random::<f64>() - 0.5);

        coordinates[[i, 0]] = t.sin();
        coordinates[[i, 1]] = 2.0 * rng.random::<f64>();
        coordinates[[i, 2]] = t.signum() * (t.cos() - 1.0);

        if noise > 0.0 {
            coordinates[[i, 0]] += noise * rng.sample::<f64, _>(StandardNormal);
            coordinates[[i, 1]] += noise * rng.sample::<f64, _>(StandardNormal);
            coordinates[[i, 2]] += noise * rng.sample::<f64, _>(StandardNormal);
        }

        intrinsic[i] = t;
    }

    Ok((coordinates, intrinsic))
}

/// Generate a severed-sphere dataset.
///
/// This is a fixed-`n_samples` adaptation of scikit-learn's manifold-learning example
/// gallery "severed sphere": points are sampled on the unit sphere with both a polar cap
/// (colatitude restricted away from the poles) and a longitudinal wedge (azimuth
/// restricted away from a full turn) removed, so the resulting surface can genuinely be
/// severed and isometrically unrolled/flattened by manifold-learning algorithms. Rather
/// than scikit-learn's original generate-then-reject approach -- which shrinks the
/// requested sample count whenever a candidate point falls inside the removed cap or
/// wedge -- the azimuth and colatitude are sampled directly from their restricted valid
/// ranges here, so exactly `n_samples` rows are always returned. The intrinsic
/// coordinate returned for each sample is the colatitude `t` (the angle from the north
/// pole).
pub fn make_severed_sphere(
    n_samples: usize,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut coordinates = Array2::zeros((n_samples, 3));
    let mut intrinsic = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let p = rng.random_range(0.0..(2.0 * PI - 0.55));
        let t = rng.random_range((PI / 8.0)..(PI - PI / 8.0));

        coordinates[[i, 0]] = t.sin() * p.cos();
        coordinates[[i, 1]] = t.sin() * p.sin();
        coordinates[[i, 2]] = t.cos();

        intrinsic[i] = t;
    }

    Ok((coordinates, intrinsic))
}

/// Generate points sampled along a helix.
///
/// The helix has unit radius in the xy-plane and completes `n_turns` full rotations
/// while its height rises linearly with the intrinsic parameter `t`, which is the
/// sample's position along the helix in `[0, 1)`.
pub fn make_helix(
    n_samples: usize,
    n_turns: f64,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }
    if n_turns <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "n_turns must be positive".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(SklearsError::InvalidInput(
            "noise must be non-negative".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut coordinates = Array2::zeros((n_samples, 3));
    let mut intrinsic = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let t = rng.random::<f64>();
        let angle = 2.0 * PI * n_turns * t;

        coordinates[[i, 0]] = angle.cos();
        coordinates[[i, 1]] = angle.sin();
        coordinates[[i, 2]] = t;

        if noise > 0.0 {
            coordinates[[i, 0]] += noise * rng.sample::<f64, _>(StandardNormal);
            coordinates[[i, 1]] += noise * rng.sample::<f64, _>(StandardNormal);
            coordinates[[i, 2]] += noise * rng.sample::<f64, _>(StandardNormal);
        }

        intrinsic[i] = t;
    }

    Ok((coordinates, intrinsic))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_swiss_roll_shape() {
        let (coords, target) =
            make_swiss_roll(150, 0.0, Some(42)).expect("operation should succeed");
        assert_eq!(coords.shape(), &[150, 3]);
        assert_eq!(target.len(), 150);
    }

    #[test]
    fn test_make_swiss_roll_deterministic_with_seed() {
        let (coords1, target1) =
            make_swiss_roll(80, 0.05, Some(7)).expect("operation should succeed");
        let (coords2, target2) =
            make_swiss_roll(80, 0.05, Some(7)).expect("operation should succeed");
        assert_eq!(coords1, coords2);
        assert_eq!(target1, target2);
    }

    #[test]
    fn test_make_swiss_roll_invalid_inputs() {
        assert!(make_swiss_roll(0, 0.0, Some(42)).is_err());
        assert!(make_swiss_roll(10, -1.0, Some(42)).is_err());
    }

    #[test]
    fn test_make_swiss_roll_geometry() {
        let (coords, target) =
            make_swiss_roll(200, 0.0, Some(3)).expect("operation should succeed");
        for i in 0..coords.nrows() {
            let xi = coords[[i, 0]];
            let zi = coords[[i, 2]];
            let ti = target[i];
            assert!((xi * xi + zi * zi - ti * ti).abs() < 1e-6);
        }
    }

    #[test]
    fn test_make_s_curve_shape() {
        let (coords, target) = make_s_curve(150, 0.0, Some(42)).expect("operation should succeed");
        assert_eq!(coords.shape(), &[150, 3]);
        assert_eq!(target.len(), 150);
    }

    #[test]
    fn test_make_s_curve_deterministic_with_seed() {
        let (coords1, target1) = make_s_curve(80, 0.05, Some(7)).expect("operation should succeed");
        let (coords2, target2) = make_s_curve(80, 0.05, Some(7)).expect("operation should succeed");
        assert_eq!(coords1, coords2);
        assert_eq!(target1, target2);
    }

    #[test]
    fn test_make_s_curve_invalid_inputs() {
        assert!(make_s_curve(0, 0.0, Some(42)).is_err());
        assert!(make_s_curve(10, -1.0, Some(42)).is_err());
    }

    #[test]
    fn test_make_s_curve_geometry() {
        let (coords, target) = make_s_curve(200, 0.0, Some(3)).expect("operation should succeed");
        for i in 0..coords.nrows() {
            let xi = coords[[i, 0]];
            assert!(xi.abs() <= 1.0 + 1e-9);
        }

        let baseline = target[0];
        let has_variation = target.iter().any(|&value| (value - baseline).abs() > 1e-9);
        assert!(has_variation);
    }

    #[test]
    fn test_make_severed_sphere_shape() {
        let (coords, target) =
            make_severed_sphere(150, Some(42)).expect("operation should succeed");
        assert_eq!(coords.shape(), &[150, 3]);
        assert_eq!(target.len(), 150);
    }

    #[test]
    fn test_make_severed_sphere_deterministic_with_seed() {
        let (coords1, target1) =
            make_severed_sphere(80, Some(7)).expect("operation should succeed");
        let (coords2, target2) =
            make_severed_sphere(80, Some(7)).expect("operation should succeed");
        assert_eq!(coords1, coords2);
        assert_eq!(target1, target2);
    }

    #[test]
    fn test_make_severed_sphere_invalid_inputs() {
        assert!(make_severed_sphere(0, Some(42)).is_err());
    }

    #[test]
    fn test_make_severed_sphere_geometry() {
        let (coords, target) = make_severed_sphere(200, Some(3)).expect("operation should succeed");
        for i in 0..coords.nrows() {
            let xi = coords[[i, 0]];
            let yi = coords[[i, 1]];
            let zi = coords[[i, 2]];
            let norm = (xi * xi + yi * yi + zi * zi).sqrt();
            assert!((norm - 1.0).abs() < 1e-9);

            let ti = target[i];
            assert!(ti > PI / 8.0);
            assert!(ti < PI - PI / 8.0);
        }
    }

    #[test]
    fn test_make_helix_shape() {
        let (coords, target) =
            make_helix(150, 3.0, 0.0, Some(42)).expect("operation should succeed");
        assert_eq!(coords.shape(), &[150, 3]);
        assert_eq!(target.len(), 150);
    }

    #[test]
    fn test_make_helix_deterministic_with_seed() {
        let (coords1, target1) =
            make_helix(80, 3.0, 0.05, Some(7)).expect("operation should succeed");
        let (coords2, target2) =
            make_helix(80, 3.0, 0.05, Some(7)).expect("operation should succeed");
        assert_eq!(coords1, coords2);
        assert_eq!(target1, target2);
    }

    #[test]
    fn test_make_helix_invalid_inputs() {
        assert!(make_helix(0, 3.0, 0.0, Some(42)).is_err());
        assert!(make_helix(10, 0.0, 0.0, Some(42)).is_err());
        assert!(make_helix(10, 3.0, -1.0, Some(42)).is_err());
    }

    #[test]
    fn test_make_helix_geometry() {
        let (coords, target) =
            make_helix(200, 4.0, 0.0, Some(3)).expect("operation should succeed");
        for i in 0..coords.nrows() {
            let xi = coords[[i, 0]];
            let yi = coords[[i, 1]];
            assert!((xi * xi + yi * yi - 1.0).abs() < 1e-6);

            let ti = target[i];
            assert!((0.0..1.0).contains(&ti));
        }
    }
}
