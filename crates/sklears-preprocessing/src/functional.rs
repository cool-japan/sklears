//! Functional APIs for preprocessing
//!
//! This module provides functional APIs that directly transform data without
//! needing to create and fit transformer objects.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform},
    types::Float,
};

use crate::{Binarizer, LabelBinarizer, NormType, Normalizer};

/// Standardize a dataset along any axis
///
/// Center to the mean and component wise scale to unit variance.
///
/// # Arguments
/// * `x` - The data to scale
/// * `axis` - Axis along which to compute mean and std (0 for features, 1 for samples)
/// * `with_mean` - If true, center the data before scaling
/// * `with_std` - If true, scale the data to unit variance
///
/// # Returns
/// Scaled data
pub fn scale(
    x: &Array2<Float>,
    axis: usize,
    with_mean: bool,
    with_std: bool,
) -> Result<Array2<Float>> {
    // Manual implementation since StandardScaler is a placeholder
    let mut result = x.clone();

    if axis == 0 {
        // Scale along features
        for j in 0..x.ncols() {
            let column = x.column(j);
            let mean = if with_mean {
                column.mean().unwrap_or(0.0)
            } else {
                0.0
            };
            let std = if with_std {
                column.std(0.0).max(1e-8)
            } else {
                1.0
            };

            for i in 0..x.nrows() {
                result[[i, j]] = (x[[i, j]] - mean) / std;
            }
        }
    } else if axis == 1 {
        // Scale along samples
        for i in 0..x.nrows() {
            let row = x.row(i);
            let mean = if with_mean {
                row.mean().unwrap_or(0.0)
            } else {
                0.0
            };
            let std = if with_std {
                row.std(0.0).max(1e-8)
            } else {
                1.0
            };

            for j in 0..x.ncols() {
                result[[i, j]] = (x[[i, j]] - mean) / std;
            }
        }
    } else {
        return Err(SklearsError::InvalidInput(format!(
            "axis must be 0 or 1, got {axis}"
        )));
    }

    Ok(result)
}

/// Scale samples individually to unit norm
///
/// Each sample (i.e. each row of the data matrix) with at least one
/// non-zero component is rescaled independently of other samples so
/// that its norm equals one.
///
/// # Arguments
/// * `x` - The data to normalize
/// * `norm` - The norm to use ('l1', 'l2', or 'max')
/// * `axis` - Axis along which to normalize (1 for samples, 0 for features)
///
/// # Returns
/// Normalized data
pub fn normalize(x: &Array2<Float>, norm: NormType, axis: usize) -> Result<Array2<Float>> {
    if axis == 1 {
        // Normalize along samples (standard behavior)
        let normalizer = Normalizer::new().norm(norm);
        normalizer.transform(x)
    } else if axis == 0 {
        // Normalize along features (transpose, normalize, transpose back)
        let x_t = x.t().to_owned();
        let normalizer = Normalizer::new().norm(norm);
        let normalized = normalizer.transform(&x_t)?;
        Ok(normalized.t().to_owned())
    } else {
        Err(SklearsError::InvalidInput(format!(
            "axis must be 0 or 1, got {axis}"
        )))
    }
}

/// Boolean thresholding of array-like or scipy.sparse matrix
///
/// # Arguments
/// * `x` - The data to binarize
/// * `threshold` - Feature values below or equal to this are replaced by 0, above it by 1
///
/// # Returns
/// Binarized data
pub fn binarize(x: &Array2<Float>, threshold: Float) -> Result<Array2<Float>> {
    let binarizer = Binarizer::new().threshold(threshold);
    let fitted = binarizer.fit(x, &())?;
    fitted.transform(x)
}

/// Scale each feature by its maximum absolute value
///
/// # Arguments
/// * `x` - The data to scale
/// * `axis` - Axis along which to scale (0 for features)
///
/// # Returns
/// Scaled data
pub fn maxabs_scale(x: &Array2<Float>, axis: usize) -> Result<Array2<Float>> {
    if axis != 0 {
        return Err(SklearsError::InvalidInput(
            "maxabs_scale only supports axis=0".to_string(),
        ));
    }

    let mut result = x.clone();

    for j in 0..x.ncols() {
        let column = x.column(j);
        let max_abs = column.iter().map(|&v| v.abs()).fold(0.0, Float::max);

        if max_abs > 1e-8 {
            for i in 0..x.nrows() {
                result[[i, j]] = x[[i, j]] / max_abs;
            }
        }
    }

    Ok(result)
}

/// Transform features to range [0, 1]
///
/// # Arguments
/// * `x` - The data to scale
/// * `feature_range` - Desired range of transformed data
/// * `axis` - Axis along which to scale (0 for features)
///
/// # Returns
/// Scaled data
pub fn minmax_scale(
    x: &Array2<Float>,
    feature_range: (Float, Float),
    axis: usize,
) -> Result<Array2<Float>> {
    if axis != 0 {
        return Err(SklearsError::InvalidInput(
            "minmax_scale only supports axis=0".to_string(),
        ));
    }

    let mut result = x.clone();
    let (min_range, max_range) = feature_range;

    for j in 0..x.ncols() {
        let column = x.column(j);
        let min_val = column.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_val = column.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range > 1e-8 {
            for i in 0..x.nrows() {
                let normalized = (x[[i, j]] - min_val) / range;
                result[[i, j]] = normalized * (max_range - min_range) + min_range;
            }
        } else {
            // If range is zero, set all values to midpoint of desired range
            let midpoint = (min_range + max_range) / 2.0;
            for i in 0..x.nrows() {
                result[[i, j]] = midpoint;
            }
        }
    }

    Ok(result)
}

/// Scale features using statistics that are robust to outliers
///
/// # Arguments
/// * `x` - The data to scale
/// * `axis` - Axis along which to scale (0 for features)
/// * `with_centering` - If true, center the data before scaling
/// * `with_scaling` - If true, scale the data to interquartile range
/// * `quantile_range` - Quantile range used to calculate scale
///
/// # Returns
/// Scaled data
pub fn robust_scale(
    x: &Array2<Float>,
    axis: usize,
    with_centering: bool,
    with_scaling: bool,
    quantile_range: (Float, Float),
) -> Result<Array2<Float>> {
    if axis != 0 {
        return Err(SklearsError::InvalidInput(
            "robust_scale only supports axis=0".to_string(),
        ));
    }

    let mut result = x.clone();

    for j in 0..x.ncols() {
        let mut column: Vec<Float> = x.column(j).to_vec();
        column.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = column.len();
        let q1_idx = ((n as Float) * quantile_range.0) as usize;
        let q3_idx = ((n as Float) * quantile_range.1) as usize;

        let q1 = column[q1_idx.min(n - 1)];
        let q3 = column[q3_idx.min(n - 1)];
        let median = column[n / 2];

        let center = if with_centering { median } else { 0.0 };
        let scale = if with_scaling && (q3 - q1) > 1e-8 {
            q3 - q1
        } else {
            1.0
        };

        for i in 0..x.nrows() {
            result[[i, j]] = (x[[i, j]] - center) / scale;
        }
    }

    Ok(result)
}

// FIXME: Commenting out complex transformations until proper implementations are available

// /// Transform features to uniform or normal distribution
// ///
// /// # Arguments
// /// * `x` - The data to transform
// /// * `n_quantiles` - Number of quantiles to estimate
// /// * `output_distribution` - Marginal distribution for transformed data
// /// * `subsample` - Maximum number of samples to use for quantile estimation
// ///
// /// # Returns
// /// Transformed data
// pub fn quantile_transform(
//     x: &Array2<Float>,
//     n_quantiles: usize,
//     output_distribution: QuantileOutput,
//     subsample: Option<usize>,
// ) -> Result<Array2<Float>> {
//     let transformer = QuantileTransformer::new()
//         .n_quantiles(n_quantiles)
//         .output_distribution(output_distribution)
//         .subsample(subsample);
//     let fitted = transformer.fit(x, &())?;
//     fitted.transform(x)
// }

// /// Apply a power transform to make data more Gaussian-like
// ///
// /// # Arguments
// /// * `x` - The data to transform
// /// * `method` - The power transform method ('yeo-johnson' or 'box-cox')
// /// * `standardize` - Apply zero-mean, unit-variance normalization
// ///
// /// # Returns
// /// Transformed data
// pub fn power_transform(
//     x: &Array2<Float>,
//     method: PowerMethod,
//     standardize: bool,
// ) -> Result<Array2<Float>> {
//     let transformer = PowerTransformer::new()
//         .method(method)
//         .standardize(standardize);
//     let fitted = transformer.fit(x, &())?;
//     fitted.transform(x)
// }

/// Add a dummy feature to the data
///
/// This is useful for fitting an intercept term with implementations which
/// cannot otherwise fit it directly.
///
/// # Arguments
/// * `x` - The data to add dummy feature to
/// * `value` - Value of the dummy feature
///
/// # Returns
/// Data with dummy feature added as first column
pub fn add_dummy_feature(x: &Array2<Float>, value: Float) -> Result<Array2<Float>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Create new array with extra column
    let mut x_with_dummy = Array2::zeros((n_samples, n_features + 1));

    // Set dummy feature values
    x_with_dummy.column_mut(0).fill(value);

    // Copy original features
    x_with_dummy
        .slice_mut(scirs2_core::ndarray::s![.., 1..])
        .assign(x);

    Ok(x_with_dummy)
}

/// Binarize labels in a one-vs-all fashion
///
/// # Arguments
/// * `y` - Target values (labels)
/// * `neg_label` - Value for negative labels
/// * `pos_label` - Value for positive labels
///
/// # Returns
/// Binarized labels
pub fn label_binarize<T>(y: &Array1<T>, neg_label: i32, pos_label: i32) -> Result<Array2<Float>>
where
    T: std::hash::Hash + Eq + Clone + std::fmt::Debug + Ord + Send + Sync,
{
    let binarizer = LabelBinarizer::<T>::new()
        .neg_label(neg_label)
        .pos_label(pos_label);
    let fitted = binarizer.fit(y, &())?;
    fitted.transform(y)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{arr1, arr2};

    #[test]
    fn test_scale() {
        let x = arr2(&[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]);

        // Scale along features
        let scaled = scale(&x, 0, true, true).unwrap();

        // Check mean is 0
        for j in 0..x.ncols() {
            let col_mean = scaled.column(j).mean().unwrap();
            assert_abs_diff_eq!(col_mean, 0.0, epsilon = 1e-10);
        }

        // Check std is 1
        for j in 0..x.ncols() {
            let col = scaled.column(j);
            let std = col.std(0.0);
            assert_abs_diff_eq!(std, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalize() {
        let x = arr2(&[[4.0, 3.0], [1.0, 2.0]]);

        // L2 normalize along samples
        let normalized = normalize(&x, NormType::L2, 1).unwrap();

        // Check L2 norm is 1 for each row
        for i in 0..x.nrows() {
            let row = normalized.row(i);
            let norm = row.dot(&row).sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_binarize() {
        let x = arr2(&[[0.5, 1.5], [2.5, 3.5]]);

        let binarized = binarize(&x, 2.0).unwrap();

        assert_eq!(binarized[[0, 0]], 0.0);
        assert_eq!(binarized[[0, 1]], 0.0);
        assert_eq!(binarized[[1, 0]], 1.0);
        assert_eq!(binarized[[1, 1]], 1.0);
    }

    #[test]
    fn test_minmax_scale() {
        let x = arr2(&[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]);

        let scaled = minmax_scale(&x, (0.0, 1.0), 0).unwrap();

        // Check min is 0 and max is 1 for each feature
        for j in 0..x.ncols() {
            let col = scaled.column(j);
            let min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            assert_abs_diff_eq!(min, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(max, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_add_dummy_feature() {
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let x_with_dummy = add_dummy_feature(&x, 1.0).unwrap();

        assert_eq!(x_with_dummy.shape(), &[2, 3]);
        assert_eq!(x_with_dummy[[0, 0]], 1.0);
        assert_eq!(x_with_dummy[[1, 0]], 1.0);
        assert_eq!(x_with_dummy[[0, 1]], 1.0);
        assert_eq!(x_with_dummy[[0, 2]], 2.0);
    }

    #[test]
    fn test_label_binarize() {
        let y = arr1(&[0, 1, 2, 1, 0]);

        let binarized = label_binarize(&y, 0, 1).unwrap();

        // Should have shape (n_samples, n_classes)
        assert_eq!(binarized.shape(), &[5, 3]);

        // Check one-hot encoding
        assert_eq!(binarized[[0, 0]], 1.0); // class 0
        assert_eq!(binarized[[1, 1]], 1.0); // class 1
        assert_eq!(binarized[[2, 2]], 1.0); // class 2
    }
}
