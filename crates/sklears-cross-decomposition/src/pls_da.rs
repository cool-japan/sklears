//! PLS Discriminant Analysis

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// PLS Discriminant Analysis for classification
///
/// PLS-DA extends PLS regression to classification by encoding class labels
/// as dummy variables and finding latent variables that are maximally correlated
/// with the class structure.
///
/// # Parameters
///
/// * `n_components` - Number of components to keep
/// * `scale` - Whether to scale the data
/// * `max_iter` - Maximum number of iterations for the algorithm
/// * `tol` - Tolerance for convergence
/// * `copy` - Whether to copy X and Y during fit
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::PLSDA;
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let y = array![0, 1, 0, 1];
///
/// let pls_da = PLSDA::new(1);
/// let fitted = pls_da.fit(&X, &y).unwrap();
/// let predictions = fitted.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PLSDA<State = Untrained> {
    /// Number of components to keep
    pub n_components: usize,
    /// Whether to scale X
    pub scale: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Whether to copy X and Y
    pub copy: bool,

    // Fitted attributes
    x_weights_: Option<Array2<Float>>,
    y_weights_: Option<Array2<Float>>,
    x_loadings_: Option<Array2<Float>>,
    y_loadings_: Option<Array2<Float>>,
    x_scores_: Option<Array2<Float>>,
    y_scores_: Option<Array2<Float>>,
    x_rotations_: Option<Array2<Float>>,
    coef_: Option<Array2<Float>>,
    n_iter_: Option<Vec<usize>>,
    x_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    classes_: Option<Array1<usize>>,
    y_dummy_: Option<Array2<Float>>,
    y_mean_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl PLSDA<Untrained> {
    /// Create a new PLS-DA model
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
            max_iter: 500,
            tol: 1e-6,
            copy: true,
            x_weights_: None,
            y_weights_: None,
            x_loadings_: None,
            y_loadings_: None,
            x_scores_: None,
            y_scores_: None,
            x_rotations_: None,
            coef_: None,
            n_iter_: None,
            x_mean_: None,
            x_std_: None,
            classes_: None,
            y_dummy_: None,
            y_mean_: None,
            _state: PhantomData,
        }
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to copy the data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }
}

impl Estimator for PLSDA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<usize>> for PLSDA<Untrained> {
    type Fitted = PLSDA<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<usize>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Extract unique classes
        let mut classes: Vec<usize> = y.iter().cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Create dummy variables for classes (one-hot encoding)
        let mut y_dummy = Array2::zeros((n_samples, n_classes));
        for (i, &label) in y.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                y_dummy[[i, class_idx]] = 1.0;
            }
        }

        // Center and scale data
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y_dummy.mean_axis(Axis(0)).unwrap();

        let mut x_centered = x - &x_mean.view().insert_axis(Axis(0));
        let mut y_centered = y_dummy.clone() - &y_mean.view().insert_axis(Axis(0));

        let x_std = if self.scale {
            let x_std = x_centered.std_axis(Axis(0), 0.0);

            // Avoid division by zero
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_centered.column_mut(i).mapv_inplace(|v| v / std);
                }
            }

            x_std
        } else {
            Array1::ones(n_features)
        };

        // Check that n_components doesn't exceed the minimum dimension
        let max_components = n_features.min(n_classes).min(n_samples);
        if self.n_components > max_components {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) cannot exceed min(n_features, n_classes, n_samples) ({})",
                self.n_components, max_components
            )));
        }

        // NIPALS algorithm for PLS
        let mut x_weights = Array2::zeros((n_features, self.n_components));
        let mut y_weights = Array2::zeros((n_classes, self.n_components));
        let mut x_loadings = Array2::zeros((n_features, self.n_components));
        let mut y_loadings = Array2::zeros((n_classes, self.n_components));
        let mut x_scores = Array2::zeros((n_samples, self.n_components));
        let mut y_scores = Array2::zeros((n_samples, self.n_components));
        let mut n_iter = Vec::new();

        let mut x_k = x_centered.clone();
        let mut y_k = y_centered.clone();

        for k in 0..self.n_components {
            // Initialize y_score as first column of Y_k (or first non-zero column)
            let mut y_score = if y_k.ncols() > 0 {
                y_k.column(0).to_owned()
            } else {
                Array1::ones(n_samples)
            };

            let mut iter = 0;
            let mut w_old = Array1::zeros(n_features);

            while iter < self.max_iter {
                // 1. Update X weights
                let w = x_k.t().dot(&y_score);
                let w_norm = w.dot(&w).sqrt();
                let w = if w_norm > 0.0 { w / w_norm } else { w };

                // Check convergence
                let diff = (&w - &w_old).mapv(|x| x.abs()).sum();
                if diff < self.tol {
                    break;
                }
                w_old = w.clone();

                // 2. Update X scores
                let x_score = x_k.dot(&w);

                // 3. Update Y weights
                let c = y_k.t().dot(&x_score);
                let c_norm = c.dot(&c).sqrt();
                let c = if c_norm > 0.0 { c / c_norm } else { c };

                // 4. Update Y scores
                y_score = y_k.dot(&c);

                // Store weights
                x_weights.column_mut(k).assign(&w);
                y_weights.column_mut(k).assign(&c);

                iter += 1;
            }

            n_iter.push(iter);

            // Calculate scores
            let x_score = x_k.dot(&x_weights.column(k));
            x_scores.column_mut(k).assign(&x_score);
            y_scores.column_mut(k).assign(&y_score);

            // Calculate loadings
            let x_loading = x_k.t().dot(&x_score) / (x_score.dot(&x_score) + 1e-10);
            let y_loading = y_k.t().dot(&x_score) / (x_score.dot(&x_score) + 1e-10);
            x_loadings.column_mut(k).assign(&x_loading);
            y_loadings.column_mut(k).assign(&y_loading);

            // Deflate X and Y
            let x_score_matrix = x_score.view().insert_axis(Axis(1));
            let x_loading_matrix = x_loading.view().insert_axis(Axis(1));
            let y_loading_matrix = y_loading.view().insert_axis(Axis(1));

            x_k = x_k - x_score_matrix.dot(&x_loading_matrix.t());
            y_k = y_k - x_score_matrix.dot(&y_loading_matrix.t());
        }

        // Calculate rotations and regression coefficients
        let x_rotations = x_weights.clone();
        let coef = x_weights.dot(&y_loadings.t());

        Ok(PLSDA {
            n_components: self.n_components,
            scale: self.scale,
            max_iter: self.max_iter,
            tol: self.tol,
            copy: self.copy,
            x_weights_: Some(x_weights),
            y_weights_: Some(y_weights),
            x_loadings_: Some(x_loadings),
            y_loadings_: Some(y_loadings),
            x_scores_: Some(x_scores),
            y_scores_: Some(y_scores),
            x_rotations_: Some(x_rotations),
            coef_: Some(coef),
            n_iter_: Some(n_iter),
            x_mean_: Some(x_mean),
            x_std_: Some(x_std),
            classes_: Some(classes),
            y_dummy_: Some(y_dummy),
            y_mean_: Some(y_mean),
            _state: PhantomData,
        })
    }
}

impl Predict<Array2<Float>, Array1<usize>> for PLSDA<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<usize>> {
        let x_mean = self.x_mean_.as_ref().unwrap();
        let x_std = self.x_std_.as_ref().unwrap();
        let coef = self.coef_.as_ref().unwrap();
        let y_mean = self.y_mean_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Predict probabilities
        let mut y_pred = x_scaled.dot(coef);
        y_pred += &y_mean.view().insert_axis(Axis(0));

        // Convert to class predictions (argmax)
        let predictions: Vec<usize> = y_pred
            .rows()
            .into_iter()
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

impl Transform<Array2<Float>> for PLSDA<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_mean = self.x_mean_.as_ref().unwrap();
        let x_std = self.x_std_.as_ref().unwrap();
        let x_rotations = self.x_rotations_.as_ref().unwrap();

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform to PLS space
        Ok(x_scaled.dot(x_rotations))
    }
}

impl PLSDA<Trained> {
    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_mean = self.x_mean_.as_ref().unwrap();
        let x_std = self.x_std_.as_ref().unwrap();
        let coef = self.coef_.as_ref().unwrap();
        let y_mean = self.y_mean_.as_ref().unwrap();

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Predict probabilities
        let mut y_pred = x_scaled.dot(coef);
        y_pred += &y_mean.view().insert_axis(Axis(0));

        // Apply softmax to get probabilities
        for mut row in y_pred.rows_mut() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            row.mapv_inplace(|x| (x - max_val).exp());
            let sum: Float = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            }
        }

        Ok(y_pred)
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<usize> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the X weights
    pub fn x_weights(&self) -> &Array2<Float> {
        self.x_weights_.as_ref().unwrap()
    }

    /// Get the Y weights
    pub fn y_weights(&self) -> &Array2<Float> {
        self.y_weights_.as_ref().unwrap()
    }

    /// Get the X loadings
    pub fn x_loadings(&self) -> &Array2<Float> {
        self.x_loadings_.as_ref().unwrap()
    }

    /// Get the Y loadings
    pub fn y_loadings(&self) -> &Array2<Float> {
        self.y_loadings_.as_ref().unwrap()
    }

    /// Get the regression coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the X scores
    pub fn x_scores(&self) -> &Array2<Float> {
        self.x_scores_.as_ref().unwrap()
    }

    /// Get the Y scores
    pub fn y_scores(&self) -> &Array2<Float> {
        self.y_scores_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pls_da_basic() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ];

        let y = array![0, 0, 1, 1, 2, 2];

        let pls_da = PLSDA::new(2);
        let fitted = pls_da.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);

        // Check that all predicted classes are valid
        let classes = fitted.classes();
        for &pred in predictions.iter() {
            assert!(classes.iter().any(|&c| c == pred));
        }
    }

    #[test]
    fn test_pls_da_binary_classification() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [8.0, 9.0], [9.0, 10.0],];

        let y = array![0, 0, 1, 1];

        let pls_da = PLSDA::new(1);
        let fitted = pls_da.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();
        let probabilities = fitted.predict_proba(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(probabilities.shape(), &[4, 2]);

        // Check that probabilities sum to approximately 1
        for row in probabilities.rows() {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pls_da_transform() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let y = array![0, 1, 0, 1];

        let pls_da = PLSDA::new(2);
        let fitted = pls_da.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[4, 2]);
    }

    #[test]
    fn test_pls_da_no_scaling() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![0, 0, 1, 1];

        let pls_da = PLSDA::new(1).scale(false);
        let fitted = pls_da.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_pls_da_single_component() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [10.0, 11.0], [11.0, 12.0],];

        let y = array![0, 0, 1, 1];

        let pls_da = PLSDA::new(1);
        let fitted = pls_da.fit(&x, &y).unwrap();

        // Should be able to separate well-separated classes
        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 0);
        assert_eq!(predictions[2], 1);
        assert_eq!(predictions[3], 1);
    }

    #[test]
    fn test_pls_da_error_cases() {
        let x = array![[1.0, 2.0], [2.0, 3.0]];

        // Test with only one class
        let y_single_class = array![0, 0];
        let pls_da = PLSDA::new(1);
        assert!(pls_da.fit(&x, &y_single_class).is_err());

        // Test with mismatched dimensions
        let y_wrong_size = array![0];
        let pls_da2 = PLSDA::new(1);
        assert!(pls_da2.fit(&x, &y_wrong_size).is_err());
    }

    // Property-based tests
    proptest! {
        #[test]
        fn test_pls_da_prediction_consistency(
            n_samples in 10..30usize,
            n_features in 2..8usize,
            n_classes in 2..5usize,
            n_components in 1..3usize,
            data in prop::collection::vec(-5.0..5.0f64, 0..300)
        ) {
            if data.len() < n_samples * n_features {
                return Ok(());
            }

            let n_components = n_components.min(n_features).min(n_classes);

            // Create random X matrix
            let x_data = &data[0..n_samples * n_features];
            let x = Array2::from_shape_vec((n_samples, n_features), x_data.to_vec())?;

            // Create balanced class labels
            let y: Array1<usize> = Array1::from_shape_fn(n_samples, |i| i % n_classes);

            if let Ok(fitted) = PLSDA::new(n_components).fit(&x, &y) {
                // Test prediction consistency
                let predictions1 = fitted.predict(&x)?;
                let predictions2 = fitted.predict(&x)?;

                // Predictions should be identical
                prop_assert_eq!(predictions1.len(), predictions2.len());
                for (a, b) in predictions1.iter().zip(predictions2.iter()) {
                    prop_assert_eq!(a, b);
                }

                // Check output dimensions
                prop_assert_eq!(predictions1.len(), n_samples);

                // All predictions should be valid class labels
                let classes = fitted.classes();
                for &pred in predictions1.iter() {
                    prop_assert!(classes.iter().any(|&c| c == pred));
                }

                // Test probability consistency
                let probabilities1 = fitted.predict_proba(&x)?;
                let probabilities2 = fitted.predict_proba(&x)?;

                prop_assert_eq!(probabilities1.shape(), &[n_samples, n_classes]);

                for (a, b) in probabilities1.iter().zip(probabilities2.iter()) {
                    prop_assert!((a - b).abs() < 1e-12);
                }

                // Probabilities should sum to approximately 1
                for row in probabilities1.rows() {
                    let sum: Float = row.sum();
                    prop_assert!((sum - 1.0).abs() < 1e-6);
                }

                // All probabilities should be non-negative
                for &prob in probabilities1.iter() {
                    prop_assert!(prob >= 0.0);
                    prop_assert!(prob <= 1.0);
                }
            }
        }

        #[test]
        fn test_pls_da_perfect_separation(
            n_samples_per_class in 5..15usize,
            n_features in 2..6usize,
            separation_factor in 5.0..20.0f64
        ) {
            let n_classes = 2;
            let n_samples = n_samples_per_class * n_classes;

            // Create perfectly separated classes
            let mut x = Array2::zeros((n_samples, n_features));
            let mut y = Array1::zeros(n_samples);

            for class in 0..n_classes {
                let start_idx = class * n_samples_per_class;
                let end_idx = (class + 1) * n_samples_per_class;

                for i in start_idx..end_idx {
                    y[i] = class;
                    for j in 0..n_features {
                        x[[i, j]] = (class as Float) * separation_factor + (j as Float) + 1.0;
                    }
                }
            }

            if let Ok(fitted) = PLSDA::new(1).fit(&x, &y) {
                let predictions = fitted.predict(&x)?;

                // For perfectly separated classes, PLS-DA should predict accurately
                let accuracy = predictions.iter()
                    .zip(y.iter())
                    .map(|(&pred, &actual)| if pred == actual { 1.0 } else { 0.0 })
                    .sum::<Float>() / n_samples as Float;

                prop_assert!(accuracy >= 0.8, "Accuracy too low: {}", accuracy);

                // Check that probabilities reflect the separation
                let probabilities = fitted.predict_proba(&x)?;
                for i in 0..n_samples {
                    let predicted_class = predictions[i];
                    let class_idx = fitted.classes().iter().position(|&c| c == predicted_class).unwrap();
                    let confidence = probabilities[[i, class_idx]];
                    prop_assert!(confidence > 0.5, "Low confidence: {}", confidence);
                }
            }
        }

        #[test]
        fn test_pls_da_multiclass_properties(
            n_samples in 15..30usize,
            n_features in 3..7usize,
            n_classes in 3..6usize,
            data in prop::collection::vec(-3.0..3.0f64, 0..200)
        ) {
            if data.len() < n_samples * n_features {
                return Ok(());
            }

            let x_data = &data[0..n_samples * n_features];
            let x = Array2::from_shape_vec((n_samples, n_features), x_data.to_vec())?;

            // Create balanced multiclass labels
            let y: Array1<usize> = Array1::from_shape_fn(n_samples, |i| i % n_classes);

            let n_components = (n_classes - 1).min(n_features);

            if let Ok(fitted) = PLSDA::new(n_components).fit(&x, &y) {
                // Check that all classes are represented
                let classes = fitted.classes();
                prop_assert_eq!(classes.len(), n_classes);

                let mut class_set = std::collections::HashSet::new();
                for &class in classes.iter() {
                    class_set.insert(class);
                }
                prop_assert_eq!(class_set.len(), n_classes);

                // Test predictions
                let predictions = fitted.predict(&x)?;
                prop_assert_eq!(predictions.len(), n_samples);

                // Test probabilities
                let probabilities = fitted.predict_proba(&x)?;
                prop_assert_eq!(probabilities.shape(), &[n_samples, n_classes]);

                // Test transform
                let transformed = fitted.transform(&x)?;
                prop_assert_eq!(transformed.shape(), &[n_samples, n_components]);

                // Check that all values are finite
                for val in transformed.iter() {
                    prop_assert!(val.is_finite());
                }
            }
        }

        #[test]
        fn test_pls_da_scaling_invariance(
            scale_factor in 1.0..50.0f64
        ) {
            let x = array![
                [1.0, 2.0],
                [2.0, 3.0],
                [8.0, 9.0],
                [9.0, 10.0],
                [1.5, 2.5],
                [8.5, 9.5],
            ];
            let y = array![0, 0, 1, 1, 0, 1];

            // Scale X by the factor
            let x_scaled = &x * scale_factor;

            let pls_da1 = PLSDA::new(1).scale(true);
            let pls_da2 = PLSDA::new(1).scale(true);

            if let (Ok(fitted1), Ok(fitted2)) = (pls_da1.fit(&x, &y), pls_da2.fit(&x_scaled, &y)) {
                let pred1 = fitted1.predict(&x)?;
                let pred2 = fitted2.predict(&x_scaled)?;

                // With scaling enabled, predictions should be identical or very similar
                let accuracy = pred1.iter()
                    .zip(pred2.iter())
                    .map(|(&a, &b)| if a == b { 1.0 } else { 0.0 })
                    .sum::<Float>() / pred1.len() as Float;

                prop_assert!(accuracy >= 0.8);  // Most predictions should be the same
            }
        }

        #[test]
        fn test_pls_da_numerical_stability(
            noise_level in 0.0..0.05f64
        ) {
            let mut x = array![
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [10.0, 11.0, 12.0],
                [11.0, 12.0, 13.0],
                [1.5, 2.5, 3.5],
                [10.5, 11.5, 12.5],
            ];
            let y = array![0, 0, 1, 1, 0, 1];

            // Add small amount of noise
            for val in x.iter_mut() {
                *val += noise_level * ((*val * 12345.0_f64).sin());
            }

            if let Ok(fitted) = PLSDA::new(2).fit(&x, &y) {
                let predictions = fitted.predict(&x)?;
                let probabilities = fitted.predict_proba(&x)?;
                let transform = fitted.transform(&x)?;

                // Check that outputs are finite
                for &pred in predictions.iter() {
                    prop_assert!(pred < 1000);  // Should be reasonable class label
                }

                for val in probabilities.iter() {
                    prop_assert!(val.is_finite());
                    prop_assert!(*val >= 0.0);
                    prop_assert!(*val <= 1.0);
                }

                for val in transform.iter() {
                    prop_assert!(val.is_finite());
                }

                // Check that weights are reasonable
                let x_weights = fitted.x_weights();
                for val in x_weights.iter() {
                    prop_assert!(val.is_finite());
                    prop_assert!(val.abs() < 1000.0);
                }
            }
        }

        #[test]
        fn test_pls_da_component_limitation(
            n_samples in 8..20usize,
            n_features in 3..8usize,
            n_classes in 2..5usize
        ) {
            // Create synthetic data
            let x: Array2<Float> = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                (i as Float + j as Float) / 10.0
            });

            let y: Array1<usize> = Array1::from_shape_fn(n_samples, |i| i % n_classes);

            // Try different numbers of components
            for n_components in 1..=(n_features.min(n_classes)) {
                if let Ok(fitted) = PLSDA::new(n_components).fit(&x, &y) {
                    let transformed = fitted.transform(&x)?;

                    // Check that the number of components is respected
                    prop_assert_eq!(transformed.shape(), &[n_samples, n_components]);

                    // Check that predictions work
                    let predictions = fitted.predict(&x)?;
                    prop_assert_eq!(predictions.len(), n_samples);

                    // Check that probabilities work
                    let probabilities = fitted.predict_proba(&x)?;
                    prop_assert_eq!(probabilities.shape(), &[n_samples, n_classes]);
                }
            }
        }
    }
}
