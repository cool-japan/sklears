//! Synthetic data generators

use scirs2_core::ndarray::{s, Array1, Array2, Array3};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Gamma, Normal, StandardNormal};
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
    factor: Option<f64>,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let noise = noise.unwrap_or(0.0);
    let factor = factor.unwrap_or(0.8);

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
    let _n_samples_in = n_samples - n_samples_out;

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
    for i in n_samples_out..n_samples {
        let angle = rng.gen::<f64>() * 2.0 * PI;
        x[[i, 0]] = angle.cos() * factor;
        x[[i, 1]] = angle.sin() * factor;
        y[i] = 1;
    }

    // Add noise if specified
    if noise > 0.0 {
        let noise_dist = Normal::new(0.0, noise).unwrap();
        for i in 0..n_samples {
            for j in 0..2 {
                x[[i, j]] += rng.sample(noise_dist);
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

    let noise = noise.unwrap_or(0.0);

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let n_samples_out = n_samples / 2;
    let _n_samples_in = n_samples - n_samples_out;

    let mut x = Array2::zeros((n_samples, 2));
    let mut y = Array1::zeros(n_samples);

    // Generate outer moon
    for i in 0..n_samples_out {
        let angle = rng.gen::<f64>() * PI;
        x[[i, 0]] = angle.cos();
        x[[i, 1]] = angle.sin();
        y[i] = 0;
    }

    // Generate inner moon (rotated and shifted)
    for i in n_samples_out..n_samples {
        let angle = rng.gen::<f64>() * PI;
        x[[i, 0]] = 1.0 - angle.cos();
        x[[i, 1]] = 1.0 - angle.sin() - 0.5;
        y[i] = 1;
    }

    // Add noise if specified
    if noise > 0.0 {
        let noise_dist = Normal::new(0.0, noise).unwrap();
        for i in 0..n_samples {
            for j in 0..2 {
                x[[i, j]] += rng.sample(noise_dist);
            }
        }
    }

    Ok((x, y))
}

pub fn make_gaussian_quantiles(
    mean: Option<Array1<f64>>,
    cov: Option<f64>,
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 || n_features == 0 || n_classes < 2 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_features must be positive, n_classes must be >= 2".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let normal = StandardNormal;
    let cov_value = cov.unwrap_or(1.0);

    // Set mean to zero vector if not provided
    let mean_vec = mean.unwrap_or_else(|| Array1::zeros(n_features));

    if mean_vec.len() != n_features {
        return Err(SklearsError::InvalidInput(
            "Mean vector length must match n_features".to_string(),
        ));
    }

    // Generate multivariate normal samples
    let mut x = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample::<f64, _>(normal) * cov_value.sqrt() + mean_vec[j];
        }
    }

    // Calculate distances from the mean for each sample
    let mut distances: Vec<(f64, usize)> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut distance_squared = 0.0;
        for j in 0..n_features {
            let diff = x[[i, j]] - mean_vec[j];
            distance_squared += diff * diff;
        }
        distances.push((distance_squared.sqrt() as f64, i));
    }

    // Sort by distance
    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign classes based on quantiles
    let mut y = Array1::zeros(n_samples);
    let samples_per_class = n_samples / n_classes;
    let extra_samples = n_samples % n_classes;

    let mut current_idx = 0;
    for class in 0..n_classes {
        let n_samples_for_class = if class < extra_samples {
            samples_per_class + 1
        } else {
            samples_per_class
        };

        for _ in 0..n_samples_for_class {
            if current_idx < distances.len() {
                let sample_idx = distances[current_idx].1;
                y[sample_idx] = class as i32;
                current_idx += 1;
            }
        }
    }

    Ok((x, y))
}

pub fn make_hastie_10_2(
    n_samples: usize,
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

    let normal = StandardNormal;
    let n_features = 10;

    // Generate feature matrix with standard normal distribution
    let mut x = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    // Compute target using Hastie et al. formula:
    // y = 1 if sum(X_i^2) > 9.34, else 0
    // where 9.34 is chosen so that P(y=1) ≈ 0.5 for 10-dimensional standard normal
    let mut y = Array1::zeros(n_samples);
    let threshold = 9.34;

    for i in 0..n_samples {
        let mut sum_squares = 0.0;
        for j in 0..n_features {
            sum_squares += x[[i, j]] * x[[i, j]];
        }
        y[i] = if sum_squares > threshold { 1 } else { 0 };
    }

    Ok((x, y))
}

pub fn make_friedman1(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 || n_features < 5 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive and n_features must be >= 5".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Generate features uniformly between 0 and 1
    let mut x = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.gen::<f64>();
        }
    }

    // Compute target using Friedman's formula:
    // y = 10 * sin(π * x1 * x2) + 20 * (x3 - 0.5)² + 10 * x4 + 5 * x5 + noise
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let x1 = x[[i, 0]];
        let x2 = x[[i, 1]];
        let x3 = x[[i, 2]];
        let x4 = x[[i, 3]];
        let x5 = x[[i, 4]];

        y[i] = 10.0 * (PI * x1 * x2).sin() + 20.0 * (x3 - 0.5).powi(2) + 10.0 * x4 + 5.0 * x5;

        // Add noise
        if noise > 0.0 {
            let noise_dist = Normal::new(0.0, noise).unwrap();
            y[i] += rng.sample(noise_dist);
        }
    }

    Ok((x, y))
}

pub fn make_friedman2(
    n_samples: usize,
    noise: f64,
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
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Generate features - Friedman2 uses 4 features
    let n_features = 4;
    let mut x = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        x[[i, 0]] = rng.gen_range(0.0..100.0); // x1: uniform [0, 100]
        x[[i, 1]] = rng.gen_range(40.0 * PI..560.0 * PI); // x2: uniform [40π, 560π]
        x[[i, 2]] = rng.gen::<f64>(); // x3: uniform [0, 1]
        x[[i, 3]] = rng.gen_range(1.0..11.0); // x4: uniform [1, 11]
    }

    // Compute target using Friedman's formula:
    // y = (x1² + (x2 * x3 - 1/(x2 * x4))²)^0.5 + noise
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let x1 = x[[i, 0]];
        let x2 = x[[i, 1]];
        let x3 = x[[i, 2]];
        let x4 = x[[i, 3]];

        let term1 = x1 * x1;
        let term2 = x2 * x3 - 1.0 / (x2 * x4);
        let term2_squared = term2 * term2;

        y[i] = (term1 + term2_squared).sqrt();

        // Add noise
        if noise > 0.0 {
            let noise_dist = Normal::new(0.0, noise).unwrap();
            y[i] += rng.sample(noise_dist);
        }
    }

    Ok((x, y))
}

pub fn make_friedman3(
    n_samples: usize,
    noise: f64,
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
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Generate features - Friedman3 uses 4 features
    let n_features = 4;
    let mut x = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        x[[i, 0]] = rng.gen_range(0.0..100.0); // x1: uniform [0, 100]
        x[[i, 1]] = rng.gen_range(40.0 * PI..560.0 * PI); // x2: uniform [40π, 560π]
        x[[i, 2]] = rng.gen::<f64>(); // x3: uniform [0, 1]
        x[[i, 3]] = rng.gen_range(1.0..11.0); // x4: uniform [1, 11]
    }

    // Compute target using Friedman's formula:
    // y = atan((x2 * x3 - 1/(x2 * x4)) / x1) + noise
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let x1 = x[[i, 0]];
        let x2 = x[[i, 1]];
        let x3 = x[[i, 2]];
        let x4 = x[[i, 3]];

        let numerator = x2 * x3 - 1.0 / (x2 * x4);
        let denominator = x1;

        y[i] = (numerator / denominator).atan();

        // Add noise
        if noise > 0.0 {
            let noise_dist = Normal::new(0.0, noise).unwrap();
            y[i] += rng.sample(noise_dist);
        }
    }

    Ok((x, y))
}
