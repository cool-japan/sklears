//! Partial dependence plots

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::ArrayView2;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

use crate::types::{PartialDependenceKind, PartialDependenceResult};

///
/// // Mock predictor function
/// let predict_fn = |x: &scirs2_core::ndarray::ArrayView2<f64>| -> Vec<f64> {
///     x.rows().into_iter()
///         .map(|row| row.iter().sum())
///         .collect()
/// };
///
/// let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let features = vec![0]; // Partial dependence for first feature
/// let grid = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]]; // Grid values for first feature
///
/// let result = partial_dependence(
///     &predict_fn,
///     &X.view(),
///     &features,
///     &grid,
///     PartialDependenceKind::Average,
/// ).unwrap();
///
/// assert_eq!(result.values.len(), 5); // 5 grid points
/// ```
pub fn partial_dependence<F>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    features: &[usize],
    grid: &[Vec<Float>],
    kind: PartialDependenceKind,
) -> SklResult<PartialDependenceResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X.dim();

    if features.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Must specify at least one feature".to_string(),
        ));
    }

    if features.len() != grid.len() {
        return Err(SklearsError::InvalidInput(
            "Number of features must match number of grid dimensions".to_string(),
        ));
    }

    for &feature_idx in features {
        if feature_idx >= n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Feature index {feature_idx} is out of bounds"
            )));
        }
    }

    // For simplicity, handle only 1D partial dependence
    if features.len() != 1 {
        return Err(SklearsError::InvalidInput(
            "Only 1D partial dependence is supported in this implementation".to_string(),
        ));
    }

    let feature_idx = features[0];
    let feature_grid = &grid[0];
    let mut pd_values = Vec::new();
    let mut individual_values = Vec::new();

    for &grid_value in feature_grid {
        let mut predictions_at_grid: Vec<Float> = Vec::new();

        // Create modified data with feature set to grid value
        let mut X_modified = X.to_owned();
        for sample_idx in 0..n_samples {
            X_modified[[sample_idx, feature_idx]] = grid_value;
        }

        let predictions = predict_fn(&X_modified.view());
        predictions_at_grid.extend(predictions.iter());

        match kind {
            PartialDependenceKind::Average => {
                let mean_prediction = predictions.iter().sum::<Float>() / n_samples as Float;
                pd_values.push(mean_prediction);
            }
            PartialDependenceKind::Individual => {
                individual_values.push(predictions);
                pd_values.push(0.0); // Placeholder for individual case
            }
        }
    }

    Ok(PartialDependenceResult {
        values: pd_values,
        individual_values,
        grid: feature_grid.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PartialDependenceKind;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_partial_dependence_average() {
        // Simple additive model: prediction = sum of features
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row.iter().sum()).collect()
        };

        let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let features = vec![0]; // Partial dependence for first feature
        let grid = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]]; // Grid values for first feature

        let result = partial_dependence(
            &predict_fn,
            &X.view(),
            &features,
            &grid,
            PartialDependenceKind::Average,
        )
        .unwrap();

        assert_eq!(result.values.len(), 5); // 5 grid points
        assert_eq!(result.grid.len(), 5);

        // The effect should be linear since we replace the feature directly
        for i in 1..result.values.len() {
            let diff = result.values[i] - result.values[i - 1];
            assert!((diff - 1.0).abs() < 1e-10); // Should increase by 1 for each step
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_partial_dependence_individual() {
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row.iter().sum()).collect()
        };

        let X = array![[1.0, 2.0], [4.0, 5.0]];
        let features = vec![0];
        let grid = vec![vec![0.0, 10.0]];

        let result = partial_dependence(
            &predict_fn,
            &X.view(),
            &features,
            &grid,
            PartialDependenceKind::Individual,
        )
        .unwrap();

        assert_eq!(result.individual_values.len(), 2); // 2 grid points
        assert_eq!(result.individual_values[0].len(), 2); // 2 samples
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_partial_dependence_errors() {
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row.iter().sum()).collect()
        };

        let X = array![[1.0, 2.0], [4.0, 5.0]];

        // Empty features
        let result = partial_dependence(
            &predict_fn,
            &X.view(),
            &[],
            &[],
            PartialDependenceKind::Average,
        );
        assert!(result.is_err());

        // Feature index out of bounds
        let features = vec![10];
        let grid = vec![vec![0.0, 1.0]];
        let result = partial_dependence(
            &predict_fn,
            &X.view(),
            &features,
            &grid,
            PartialDependenceKind::Average,
        );
        assert!(result.is_err());

        // Mismatched feature and grid lengths
        let features = vec![0, 1];
        let grid = vec![vec![0.0, 1.0]]; // Only one grid dimension
        let result = partial_dependence(
            &predict_fn,
            &X.view(),
            &features,
            &grid,
            PartialDependenceKind::Average,
        );
        assert!(result.is_err());
    }
}
