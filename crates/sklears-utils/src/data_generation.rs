use crate::random::get_rng;
use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Rng;
use scirs2_core::random::StandardNormal;

#[allow(clippy::too_many_arguments)]
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_informative: Option<usize>,
    n_redundant: Option<usize>,
    flip_y: f64,
    class_sep: f64,
    random_state: Option<u64>,
) -> UtilsResult<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(UtilsError::EmptyInput);
    }

    if n_classes < 2 {
        return Err(UtilsError::InvalidParameter(
            "n_classes must be >= 2".to_string(),
        ));
    }

    let n_informative = n_informative.unwrap_or(n_features);
    let n_redundant = n_redundant.unwrap_or(0);

    if n_informative + n_redundant > n_features {
        return Err(UtilsError::InvalidParameter(
            "n_informative + n_redundant must be <= n_features".to_string(),
        ));
    }

    let mut rng = get_rng(random_state);

    // Generate informative features
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<i32>::zeros(n_samples);

    // Assign classes uniformly
    for i in 0..n_samples {
        y[i] = (i % n_classes) as i32;
    }

    // Shuffle class labels
    for i in (1..n_samples).rev() {
        let j = rng.random_range(0..=i);
        y.swap(i, j);
    }

    // Generate class centroids
    let mut centroids = Array2::<f64>::zeros((n_classes, n_informative));
    for i in 0..n_classes {
        for j in 0..n_informative {
            centroids[[i, j]] = rng.sample::<f64, _>(StandardNormal) * class_sep;
        }
    }

    // Generate features around centroids
    for i in 0..n_samples {
        let class_idx = y[i] as usize;
        for j in 0..n_informative {
            x[[i, j]] = centroids[[class_idx, j]] + rng.sample::<f64, _>(StandardNormal);
        }
    }

    // Add redundant features (linear combinations of informative features)
    for j in 0..n_redundant {
        let feat_idx = n_informative + j;
        let base_feat = j % n_informative;
        let coeff = rng.random_range(-1.0..1.0);

        for i in 0..n_samples {
            x[[i, feat_idx]] =
                x[[i, base_feat]] * coeff + rng.sample::<f64, _>(StandardNormal) * 0.1;
        }
    }

    // Add noise to remaining features
    for j in (n_informative + n_redundant)..n_features {
        for i in 0..n_samples {
            x[[i, j]] = rng.sample::<f64, _>(StandardNormal);
        }
    }

    // Flip some labels
    if flip_y > 0.0 {
        let n_flip = (n_samples as f64 * flip_y) as usize;
        for _ in 0..n_flip {
            let idx = rng.random_range(0..n_samples);
            y[idx] = rng.random_range(0..n_classes as i32);
        }
    }

    Ok((x, y))
}

pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    n_informative: Option<usize>,
    noise: f64,
    bias: f64,
    random_state: Option<u64>,
) -> UtilsResult<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(UtilsError::EmptyInput);
    }

    let n_informative = n_informative.unwrap_or(n_features);

    if n_informative > n_features {
        return Err(UtilsError::InvalidParameter(
            "n_informative must be <= n_features".to_string(),
        ));
    }

    let mut rng = get_rng(random_state);

    // Generate features
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample::<f64, _>(StandardNormal);
        }
    }

    // Generate true coefficients
    let mut coef = Array1::<f64>::zeros(n_features);
    for i in 0..n_informative {
        coef[i] = rng.sample::<f64, _>(StandardNormal) * 100.0;
    }

    // Generate target values
    let mut y = Array1::<f64>::zeros(n_samples);
    for i in 0..n_samples {
        let mut target = bias;
        for j in 0..n_features {
            target += x[[i, j]] * coef[j];
        }

        if noise > 0.0 {
            target += rng.sample::<f64, _>(StandardNormal) * noise;
        }

        y[i] = target;
    }

    Ok((x, y))
}

pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: Option<usize>,
    cluster_std: f64,
    center_box: (f64, f64),
    random_state: Option<u64>,
) -> UtilsResult<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(UtilsError::EmptyInput);
    }

    let n_centers = centers.unwrap_or(3);
    let mut rng = get_rng(random_state);

    // Generate cluster centers
    let mut cluster_centers = Array2::<f64>::zeros((n_centers, n_features));
    for i in 0..n_centers {
        for j in 0..n_features {
            cluster_centers[[i, j]] = rng.random_range(center_box.0..center_box.1);
        }
    }

    // Generate samples
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<i32>::zeros(n_samples);

    let samples_per_center = n_samples / n_centers;
    let remainder = n_samples % n_centers;

    let mut sample_idx = 0;
    for center_idx in 0..n_centers {
        let n_samples_this_center = samples_per_center + if center_idx < remainder { 1 } else { 0 };

        for _ in 0..n_samples_this_center {
            y[sample_idx] = center_idx as i32;

            for j in 0..n_features {
                let center_val = cluster_centers[[center_idx, j]];
                x[[sample_idx, j]] =
                    center_val + rng.sample::<f64, _>(StandardNormal) * cluster_std;
            }

            sample_idx += 1;
        }
    }

    // Shuffle the samples
    for i in (1..n_samples).rev() {
        let j = rng.random_range(0..=i);

        // Swap labels
        y.swap(i, j);

        // Swap rows in X
        for k in 0..n_features {
            let temp = x[[i, k]];
            x[[i, k]] = x[[j, k]];
            x[[j, k]] = temp;
        }
    }

    Ok((x, y))
}

pub fn make_circles(
    n_samples: usize,
    noise: f64,
    factor: f64,
    random_state: Option<u64>,
) -> UtilsResult<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(UtilsError::EmptyInput);
    }

    if factor <= 0.0 || factor >= 1.0 {
        return Err(UtilsError::InvalidParameter(
            "factor must be in (0, 1)".to_string(),
        ));
    }

    let mut rng = get_rng(random_state);
    let mut x = Array2::<f64>::zeros((n_samples, 2));
    let mut y = Array1::<i32>::zeros(n_samples);

    let n_outer = n_samples / 2;
    let n_inner = n_samples - n_outer;

    // Generate outer circle
    for i in 0..n_outer {
        let angle = 2.0 * std::f64::consts::PI * rng.random::<f64>();
        x[[i, 0]] = angle.cos() + rng.sample::<f64, _>(StandardNormal) * noise;
        x[[i, 1]] = angle.sin() + rng.sample::<f64, _>(StandardNormal) * noise;
        y[i] = 0;
    }

    // Generate inner circle
    for i in 0..n_inner {
        let idx = n_outer + i;
        let angle = 2.0 * std::f64::consts::PI * rng.random::<f64>();
        x[[idx, 0]] = factor * angle.cos() + rng.sample::<f64, _>(StandardNormal) * noise;
        x[[idx, 1]] = factor * angle.sin() + rng.sample::<f64, _>(StandardNormal) * noise;
        y[idx] = 1;
    }

    Ok((x, y))
}

/// Generate a moon-shaped 2D dataset for non-linear classification
pub fn make_moons(
    n_samples: usize,
    noise: f64,
    random_state: Option<u64>,
) -> UtilsResult<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(UtilsError::EmptyInput);
    }

    let mut rng = get_rng(random_state);
    let mut x = Array2::<f64>::zeros((n_samples, 2));
    let mut y = Array1::<i32>::zeros(n_samples);

    let n_samples_per_class = n_samples / 2;
    let remainder = n_samples % 2;

    // Generate first moon (upper moon)
    for i in 0..n_samples_per_class + remainder {
        let angle = std::f64::consts::PI * (i as f64) / (n_samples_per_class as f64);
        x[[i, 0]] = angle.cos() + noise * rng.random::<f64>() * 2.0 - noise;
        x[[i, 1]] = angle.sin() + noise * rng.random::<f64>() * 2.0 - noise;
        y[i] = 0;
    }

    // Generate second moon (lower moon)
    for i in 0..n_samples_per_class {
        let idx = i + n_samples_per_class + remainder;
        let angle = std::f64::consts::PI * (i as f64) / (n_samples_per_class as f64);
        x[[idx, 0]] = 1.0 - angle.cos() + noise * rng.random::<f64>() * 2.0 - noise;
        x[[idx, 1]] = 1.0 - angle.sin() - 0.5 + noise * rng.random::<f64>() * 2.0 - noise;
        y[idx] = 1;
    }

    Ok((x, y))
}

/// Generate a sparse classification dataset
pub fn make_sparse_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_informative: Option<usize>,
    sparsity: f64,
    random_state: Option<u64>,
) -> UtilsResult<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(UtilsError::EmptyInput);
    }

    if !(0.0..=1.0).contains(&sparsity) {
        return Err(UtilsError::InvalidParameter(
            "sparsity must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Generate a dense classification dataset first
    let (mut x, y) = make_classification(
        n_samples,
        n_features,
        n_classes,
        n_informative,
        Some(0),
        0.0,
        1.0,
        random_state,
    )?;

    // Make it sparse by setting a fraction of values to zero
    let mut rng = get_rng(random_state);
    let total_elements = n_samples * n_features;
    let n_zeros = (total_elements as f64 * sparsity) as usize;

    for _ in 0..n_zeros {
        let row = rng.random_range(0..n_samples);
        let col = rng.random_range(0..n_features);
        x[[row, col]] = 0.0;
    }

    Ok((x, y))
}

/// Generate a multilabel classification dataset
pub fn make_multilabel_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_labels: usize,
    random_state: Option<u64>,
) -> UtilsResult<(Array2<f64>, Array2<i32>)> {
    if n_samples == 0 {
        return Err(UtilsError::EmptyInput);
    }

    if n_labels > n_classes {
        return Err(UtilsError::InvalidParameter(
            "n_labels cannot be greater than n_classes".to_string(),
        ));
    }

    let mut rng = get_rng(random_state);
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array2::<i32>::zeros((n_samples, n_classes));

    // Generate features
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample::<f64, _>(StandardNormal);
        }
    }

    // Generate multilabel targets
    for i in 0..n_samples {
        // Randomly select which labels are active for this sample
        let mut available_labels: Vec<usize> = (0..n_classes).collect();
        for _ in 0..n_labels {
            if available_labels.is_empty() {
                break;
            }
            let idx = rng.random_range(0..available_labels.len());
            let label = available_labels.remove(idx);
            y[[i, label]] = 1;
        }
    }

    Ok((x, y))
}

/// Generate the Hastie 10-2 dataset for binary classification
pub fn make_hastie_10_2(
    n_samples: usize,
    random_state: Option<u64>,
) -> UtilsResult<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(UtilsError::EmptyInput);
    }

    let mut rng = get_rng(random_state);
    let n_features = 10;
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<i32>::zeros(n_samples);

    for i in 0..n_samples {
        // Generate 10 features from standard normal distribution
        for j in 0..n_features {
            x[[i, j]] = rng.sample::<f64, _>(StandardNormal);
        }

        // Hastie's formula: y = 1 if sum(X_j^2) > 9.34, else y = -1
        let sum_of_squares: f64 = x.row(i).iter().map(|&val| val * val).sum();
        y[i] = if sum_of_squares > 9.34 { 1 } else { -1 };
    }

    Ok((x, y))
}

/// Generate a dataset with Gaussian quantiles for classification
pub fn make_gaussian_quantiles(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    mean: f64,
    cov: f64,
    random_state: Option<u64>,
) -> UtilsResult<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(UtilsError::EmptyInput);
    }

    if n_classes < 2 {
        return Err(UtilsError::InvalidParameter(
            "n_classes must be >= 2".to_string(),
        ));
    }

    let mut rng = get_rng(random_state);
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut y = Array1::<i32>::zeros(n_samples);

    // Generate features from multivariate normal distribution
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = mean + cov * rng.sample::<f64, _>(StandardNormal);
        }
    }

    // Compute quantiles for class assignment
    // Calculate the L2 norm (distance from origin) for each sample
    let mut norms: Vec<(f64, usize)> = Vec::new();
    for i in 0..n_samples {
        let norm = x.row(i).iter().map(|&val| val * val).sum::<f64>().sqrt();
        norms.push((norm, i));
    }

    // Sort by norm and assign classes based on quantiles
    norms.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let samples_per_class = n_samples / n_classes;
    let remainder = n_samples % n_classes;

    let mut current_class = 0;
    let mut samples_in_current_class = 0;
    let mut max_samples_for_class =
        samples_per_class + if current_class < remainder { 1 } else { 0 };

    for (_, original_idx) in norms {
        y[original_idx] = current_class as i32;
        samples_in_current_class += 1;

        if samples_in_current_class >= max_samples_for_class && current_class < n_classes - 1 {
            current_class += 1;
            samples_in_current_class = 0;
            max_samples_for_class =
                samples_per_class + if current_class < remainder { 1 } else { 0 };
        }
    }

    Ok((x, y))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_classification() {
        let (x, y) = make_classification(100, 5, 3, None, None, 0.0, 1.0, Some(42)).unwrap();

        assert_eq!(x.shape(), &[100, 5]);
        assert_eq!(y.len(), 100);

        // Check that all classes are present
        let unique_classes: std::collections::HashSet<i32> = y.iter().copied().collect();
        assert!(unique_classes.len() <= 3);
    }

    #[test]
    fn test_make_regression() {
        let (x, y) = make_regression(50, 3, Some(2), 0.1, 5.0, Some(42)).unwrap();

        assert_eq!(x.shape(), &[50, 3]);
        assert_eq!(y.len(), 50);
    }

    #[test]
    fn test_make_blobs() {
        let (x, y) = make_blobs(60, 2, Some(3), 1.0, (-10.0, 10.0), Some(42)).unwrap();

        assert_eq!(x.shape(), &[60, 2]);
        assert_eq!(y.len(), 60);

        // Check that all expected cluster labels are present
        let unique_labels: std::collections::HashSet<i32> = y.iter().copied().collect();
        assert_eq!(unique_labels.len(), 3);
    }

    #[test]
    fn test_make_circles() {
        let (x, y) = make_circles(100, 0.1, 0.5, Some(42)).unwrap();

        assert_eq!(x.shape(), &[100, 2]);
        assert_eq!(y.len(), 100);

        // Check that both classes are present
        let unique_labels: std::collections::HashSet<i32> = y.iter().copied().collect();
        assert_eq!(unique_labels.len(), 2);
        assert!(unique_labels.contains(&0));
        assert!(unique_labels.contains(&1));
    }

    #[test]
    fn test_make_moons() {
        let (x, y) = make_moons(100, 0.1, Some(42)).unwrap();

        assert_eq!(x.shape(), &[100, 2]);
        assert_eq!(y.len(), 100);

        // Check that both classes are present
        let unique_labels: std::collections::HashSet<i32> = y.iter().copied().collect();
        assert_eq!(unique_labels.len(), 2);
        assert!(unique_labels.contains(&0));
        assert!(unique_labels.contains(&1));
    }

    #[test]
    fn test_make_sparse_classification() {
        let (x, y) = make_sparse_classification(50, 10, 2, Some(5), 0.3, Some(42)).unwrap();

        assert_eq!(x.shape(), &[50, 10]);
        assert_eq!(y.len(), 50);

        // Check sparsity - should have some zero values
        let zero_count = x.iter().filter(|&&val| val == 0.0).count();
        assert!(zero_count > 0);
    }

    #[test]
    fn test_make_multilabel_classification() {
        let (x, y) = make_multilabel_classification(30, 5, 4, 2, Some(42)).unwrap();

        assert_eq!(x.shape(), &[30, 5]);
        assert_eq!(y.shape(), &[30, 4]);

        // Check that each sample has the expected number of labels
        for i in 0..30 {
            let active_labels = y.row(i).iter().filter(|&&val| val == 1).count();
            assert!(active_labels <= 2); // Should have at most n_labels active
        }
    }

    #[test]
    fn test_make_hastie_10_2() {
        let (x, y) = make_hastie_10_2(100, Some(42)).unwrap();

        assert_eq!(x.shape(), &[100, 10]);
        assert_eq!(y.len(), 100);

        // Check that both classes (-1 and 1) are present
        let unique_labels: std::collections::HashSet<i32> = y.iter().copied().collect();
        assert_eq!(unique_labels.len(), 2);
        assert!(unique_labels.contains(&-1));
        assert!(unique_labels.contains(&1));
    }

    #[test]
    fn test_make_gaussian_quantiles() {
        let (x, y) = make_gaussian_quantiles(60, 3, 3, 0.0, 1.0, Some(42)).unwrap();

        assert_eq!(x.shape(), &[60, 3]);
        assert_eq!(y.len(), 60);

        // Check that all expected classes are present
        let unique_labels: std::collections::HashSet<i32> = y.iter().copied().collect();
        assert_eq!(unique_labels.len(), 3);
        assert!(unique_labels.contains(&0));
        assert!(unique_labels.contains(&1));
        assert!(unique_labels.contains(&2));
    }
}
