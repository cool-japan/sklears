//! Basic synthetic data generators
//!
//! This module contains fundamental dataset generation functions including
//! blobs, classification, regression, circles, moons, and other basic patterns.

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Normal, StandardNormal};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 || n_features == 0 || centers == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, n_features, and centers must be positive".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Generate random centers
    let mut center_points = Array2::zeros((centers, n_features));
    for i in 0..centers {
        for j in 0..n_features {
            center_points[[i, j]] = rng.gen_range(-10.0..10.0);
        }
    }

    // Assign samples to centers
    let samples_per_center = n_samples / centers;
    let extra_samples = n_samples % centers;

    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    let mut sample_idx = 0;

    for center_idx in 0..centers {
        let n_samples_for_center = if center_idx < extra_samples {
            samples_per_center + 1
        } else {
            samples_per_center
        };

        let normal = Normal::new(0.0, cluster_std).unwrap();

        for _ in 0..n_samples_for_center {
            y[sample_idx] = center_idx as i32;

            for feature_idx in 0..n_features {
                let center_value = center_points[[center_idx, feature_idx]];
                let noise: f64 = rng.sample(normal);
                x[[sample_idx, feature_idx]] = center_value + noise;
            }

            sample_idx += 1;
        }
    }

    Ok((x, y))
}

pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    n_redundant: usize,
    n_classes: usize,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 || n_features == 0 || n_classes < 2 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_features must be positive, n_classes must be >= 2".to_string(),
        ));
    }

    if n_informative + n_redundant > n_features {
        return Err(SklearsError::InvalidInput(
            "n_informative + n_redundant cannot exceed n_features".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let normal = StandardNormal;

    // Generate informative features
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    // Assign classes randomly
    for i in 0..n_samples {
        y[i] = rng.gen_range(0..n_classes) as i32;
    }

    // Generate informative features based on class
    for i in 0..n_samples {
        let class = y[i] as usize;
        for j in 0..n_informative {
            let class_offset = (class as f64 - (n_classes as f64 - 1.0) / 2.0) * 2.0;
            x[[i, j]] = rng.sample::<f64, _>(normal) + class_offset;
        }
    }

    // Generate redundant features as linear combinations of informative features
    for j in n_informative..(n_informative + n_redundant) {
        let informative_idx = rng.gen_range(0..n_informative);
        let weight = rng.gen_range(-1.0..1.0);
        for i in 0..n_samples {
            x[[i, j]] = x[[i, informative_idx]] * weight + rng.sample::<f64, _>(normal) * 0.1;
        }
    }

    // Fill remaining features with random noise
    for j in (n_informative + n_redundant)..n_features {
        for i in 0..n_samples {
            x[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    Ok((x, y))
}

pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_features must be positive".to_string(),
        ));
    }

    if n_informative > n_features {
        return Err(SklearsError::InvalidInput(
            "n_informative cannot exceed n_features".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let normal = StandardNormal;

    // Generate feature matrix
    let mut x = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    // Generate true coefficients for informative features
    let mut coef = Array1::zeros(n_features);
    for i in 0..n_informative {
        coef[i] = rng.gen_range(-1.0..1.0) * 100.0;
    }

    // Compute target values
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let mut target = 0.0;
        for j in 0..n_informative {
            target += x[[i, j]] * coef[j];
        }

        // Add noise
        if noise > 0.0 {
            let noise_dist = Normal::new(0.0, noise).unwrap();
            target += rng.sample(noise_dist);
        }

        y[i] = target;
    }

    Ok((x, y))
}

pub fn make_circles(
    n_samples: usize,
    noise: Option<f64>,
    factor: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    if factor <= 0.0 || factor >= 1.0 {
        return Err(SklearsError::InvalidInput(
            "factor must be between 0 and 1".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let n_samples_out = n_samples / 2;
    let n_samples_in = n_samples - n_samples_out;

    let mut x = Array2::zeros((n_samples, 2));
    let mut y = Array1::zeros(n_samples);

    // Generate outer circle
    for i in 0..n_samples_out {
        let angle = rng.gen::<f64>() * 2.0 * PI;
        x[[i, 0]] = angle.cos();
        x[[i, 1]] = angle.sin();
        y[i] = 0;
    }

    // Generate inner circle
    for i in 0..n_samples_in {
        let angle = rng.gen::<f64>() * 2.0 * PI;
        x[[n_samples_out + i, 0]] = factor * angle.cos();
        x[[n_samples_out + i, 1]] = factor * angle.sin();
        y[n_samples_out + i] = 1;
    }

    // Add noise if specified
    if let Some(noise_level) = noise {
        if noise_level > 0.0 {
            let noise_dist = Normal::new(0.0, noise_level).unwrap();
            for i in 0..n_samples {
                x[[i, 0]] += rng.sample(noise_dist);
                x[[i, 1]] += rng.sample(noise_dist);
            }
        }
    }

    Ok((x, y))
}

pub fn make_moons(
    n_samples: usize,
    noise: Option<f64>,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let n_samples_out = n_samples / 2;
    let n_samples_in = n_samples - n_samples_out;

    let mut x = Array2::zeros((n_samples, 2));
    let mut y = Array1::zeros(n_samples);

    // Generate outer moon
    for i in 0..n_samples_out {
        let t = rng.gen::<f64>() * PI;
        x[[i, 0]] = t.cos();
        x[[i, 1]] = t.sin();
        y[i] = 0;
    }

    // Generate inner moon
    for i in 0..n_samples_in {
        let t = rng.gen::<f64>() * PI;
        x[[n_samples_out + i, 0]] = 1.0 - t.cos();
        x[[n_samples_out + i, 1]] = 1.0 - t.sin() - 0.5;
        y[n_samples_out + i] = 1;
    }

    // Add noise if specified
    if let Some(noise_level) = noise {
        if noise_level > 0.0 {
            let noise_dist = Normal::new(0.0, noise_level).unwrap();
            for i in 0..n_samples {
                x[[i, 0]] += rng.sample(noise_dist);
                x[[i, 1]] += rng.sample(noise_dist);
            }
        }
    }

    Ok((x, y))
}

pub fn make_gaussian_quantiles(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 || n_features == 0 || n_classes < 2 {
        return Err(SklearsError::InvalidInput(
            "n_samples, n_features must be positive, n_classes must be >= 2".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let normal = StandardNormal;

    // Generate multivariate normal data
    let mut x = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    // Calculate norm for each sample
    let mut norms = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let norm = x.slice(s![i, ..]).mapv(|x| x * x).sum().sqrt();
        norms[i] = norm;
    }

    // Sort by norm to create quantiles
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.sort_by(|&a, &b| norms[a].partial_cmp(&norms[b]).unwrap());

    // Assign classes based on quantiles
    let mut y = Array1::zeros(n_samples);
    let samples_per_class = n_samples / n_classes;

    for (class_idx, chunk) in indices.chunks(samples_per_class).enumerate() {
        let class = (class_idx.min(n_classes - 1)) as i32;
        for &sample_idx in chunk {
            y[sample_idx] = class;
        }
    }

    Ok((x, y))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_blobs() {
        let (x, y) = make_blobs(100, 2, 3, 1.0, Some(42)).unwrap();
        assert_eq!(x.shape(), &[100, 2]);
        assert_eq!(y.len(), 100);

        // Check that we have the right number of classes
        let mut classes = y.iter().cloned().collect::<Vec<_>>();
        classes.sort();
        classes.dedup();
        assert_eq!(classes.len(), 3);
    }

    #[test]
    fn test_make_classification() {
        let (x, y) = make_classification(100, 20, 10, 5, 3, Some(42)).unwrap();
        assert_eq!(x.shape(), &[100, 20]);
        assert_eq!(y.len(), 100);

        // Check that we have the right number of classes
        let mut classes = y.iter().cloned().collect::<Vec<_>>();
        classes.sort();
        classes.dedup();
        assert!(classes.len() <= 3);
    }

    #[test]
    fn test_make_regression() {
        let (x, y) = make_regression(50, 10, 5, 0.1, Some(42)).unwrap();
        assert_eq!(x.shape(), &[50, 10]);
        assert_eq!(y.len(), 50);

        // Check that target values have some variation
        let mean = y.mean().unwrap();
        let variance = y.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        assert!(variance > 0.0);
    }

    #[test]
    fn test_make_circles() {
        let (x, y) = make_circles(100, Some(0.1), 0.4, Some(42)).unwrap();
        assert_eq!(x.shape(), &[100, 2]);
        assert_eq!(y.len(), 100);

        // Check that we have two classes
        let mut classes = y.iter().cloned().collect::<Vec<_>>();
        classes.sort();
        classes.dedup();
        assert_eq!(classes.len(), 2);
    }

    #[test]
    fn test_make_moons() {
        let (x, y) = make_moons(80, Some(0.15), Some(42)).unwrap();
        assert_eq!(x.shape(), &[80, 2]);
        assert_eq!(y.len(), 80);

        // Check that we have two classes
        let mut classes = y.iter().cloned().collect::<Vec<_>>();
        classes.sort();
        classes.dedup();
        assert_eq!(classes.len(), 2);
    }

    #[test]
    fn test_make_gaussian_quantiles() {
        let (x, y) = make_gaussian_quantiles(120, 5, 3, Some(42)).unwrap();
        assert_eq!(x.shape(), &[120, 5]);
        assert_eq!(y.len(), 120);

        // Check that we have the right number of classes
        let mut classes = y.iter().cloned().collect::<Vec<_>>();
        classes.sort();
        classes.dedup();
        assert!(classes.len() <= 3);
    }

    #[test]
    fn test_invalid_inputs() {
        // Test invalid n_samples
        assert!(make_blobs(0, 2, 3, 1.0, Some(42)).is_err());

        // Test invalid factor for circles
        assert!(make_circles(100, Some(0.1), 1.5, Some(42)).is_err());

        // Test invalid n_informative for classification
        assert!(make_classification(100, 5, 10, 0, 3, Some(42)).is_err());
    }
}
