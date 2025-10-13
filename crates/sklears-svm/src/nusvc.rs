//! Nu Support Vector Classification
//!
//! This module implements Nu-SVC, an alternative formulation of SVM classification
//! that uses a parameter nu instead of C for controlling the regularization.

use crate::{
    kernels::{Kernel, KernelType},
    smo::{SmoConfig, SmoSolver},
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Nu Support Vector Classification Configuration
#[derive(Debug, Clone)]
pub struct NuSVCConfig {
    /// Nu parameter (0 < nu <= 1)
    pub nu: Float,
    /// Kernel function to use
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Random seed
    pub random_state: Option<u64>,
    /// Whether to use probability estimates
    pub probability: bool,
}

impl Default for NuSVCConfig {
    fn default() -> Self {
        Self {
            nu: 0.5,
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 200, // Reduced from 1000 to improve performance
            random_state: None,
            probability: false,
        }
    }
}

/// Nu Support Vector Classification
///
/// Nu-SVC is an alternative formulation of SVM that uses a parameter nu
/// instead of C. The parameter nu controls the fraction of support vectors
/// and roughly corresponds to the fraction of training errors.
#[derive(Debug)]
pub struct NuSVC<State = Untrained> {
    config: NuSVCConfig,
    state: PhantomData<State>,
    // Fitted attributes
    support_vectors_: Option<Array2<Float>>,
    support_: Option<Array1<usize>>,
    dual_coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    classes_: Option<Array1<i32>>,
    n_features_in_: Option<usize>,
    n_support_: Option<Array1<usize>>,
}

impl NuSVC<Untrained> {
    /// Create a new Nu-SVC classifier
    pub fn new() -> Self {
        Self {
            config: NuSVCConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            support_: None,
            dual_coef_: None,
            intercept_: None,
            classes_: None,
            n_features_in_: None,
            n_support_: None,
        }
    }

    /// Set the nu parameter (0 < nu <= 1)
    pub fn nu(mut self, nu: Float) -> Self {
        if nu <= 0.0 || nu > 1.0 {
            panic!("Nu must be in the range (0, 1]");
        }
        self.config.nu = nu;
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Enable probability estimates
    pub fn probability(mut self, probability: bool) -> Self {
        self.config.probability = probability;
        self
    }

    /// Automatically select optimal nu parameter using cross-validation
    pub fn auto_select_nu(
        mut self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        nu_candidates: Option<Vec<Float>>,
        cv_folds: Option<usize>,
    ) -> Result<Self> {
        let nu_values = nu_candidates
            .unwrap_or_else(|| vec![0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let folds = cv_folds.unwrap_or(5);

        let best_nu = self.grid_search_nu(x, y, &nu_values, folds)?;
        self.config.nu = best_nu;
        Ok(self)
    }

    /// Grid search for optimal nu parameter
    fn grid_search_nu(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        nu_candidates: &[Float],
        cv_folds: usize,
    ) -> Result<Float> {
        let mut best_score = Float::NEG_INFINITY;
        let mut best_nu = nu_candidates[0];

        for &nu_val in nu_candidates {
            if nu_val <= 0.0 || nu_val > 1.0 {
                continue;
            }

            let score = self.cross_validate_nu(x, y, nu_val, cv_folds)?;
            if score > best_score {
                best_score = score;
                best_nu = nu_val;
            }
        }

        Ok(best_nu)
    }

    /// Perform cross-validation for a specific nu parameter
    fn cross_validate_nu(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        nu_val: Float,
        cv_folds: usize,
    ) -> Result<Float> {
        let n_samples = x.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle indices for random cross-validation splits
        let mut rng = StdRng::seed_from_u64(42);
        use scirs2_core::random::seq::SliceRandom;
        indices.shuffle(&mut rng);

        let fold_size = n_samples / cv_folds;
        let mut scores = Vec::new();

        for fold in 0..cv_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/validation split
            let val_indices = &indices[start_idx..end_idx];
            let train_indices: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < start_idx || *i >= end_idx)
                .map(|(_, &idx)| idx)
                .collect();

            if train_indices.is_empty() || val_indices.is_empty() {
                continue;
            }

            // Extract train data
            let mut x_train = Array2::zeros((train_indices.len(), x.ncols()));
            let mut y_train = Array1::zeros(train_indices.len());
            for (i, &idx) in train_indices.iter().enumerate() {
                x_train.row_mut(i).assign(&x.row(idx));
                y_train[i] = y[idx];
            }

            // Extract validation data
            let mut x_val = Array2::zeros((val_indices.len(), x.ncols()));
            let mut y_val = Array1::zeros(val_indices.len());
            for (i, &idx) in val_indices.iter().enumerate() {
                x_val.row_mut(i).assign(&x.row(idx));
                y_val[i] = y[idx];
            }

            // Train model with current nu
            let mut config = self.config.clone();
            config.nu = nu_val;
            let model = NuSVC {
                config,
                state: PhantomData,
                support_vectors_: None,
                support_: None,
                dual_coef_: None,
                intercept_: None,
                classes_: None,
                n_features_in_: None,
                n_support_: None,
            };

            let trained_model = model.fit(&x_train, &y_train)?;
            let predictions = trained_model.predict(&x_val)?;

            // Calculate accuracy
            let correct = predictions
                .iter()
                .zip(y_val.iter())
                .filter(|(&pred, &actual)| pred == actual)
                .count();
            let accuracy = correct as Float / val_indices.len() as Float;
            scores.push(accuracy);
        }

        // Return mean cross-validation score
        if scores.is_empty() {
            Ok(0.0)
        } else {
            Ok(scores.iter().sum::<Float>() / scores.len() as Float)
        }
    }
}

impl Default for NuSVC<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<i32>> for NuSVC<Untrained> {
    type Fitted = NuSVC<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} samples", y.len()),
            });
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Get unique classes
        let mut classes: Vec<i32> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes_array = Array1::from_vec(classes);
        let n_classes = classes_array.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Convert nu to C parameter for SMO solver
        // This is a simplified conversion - in practice, the relationship is more complex
        let c = 1.0 / (self.config.nu * n_samples as Float);

        // For binary classification, solve directly
        if n_classes == 2 {
            // Convert labels to -1, +1
            let binary_y = Array1::from_vec(
                y.iter()
                    .map(|&label| if label == classes_array[0] { -1.0 } else { 1.0 })
                    .collect(),
            );

            // Solve using SMO with KernelType directly
            let config = SmoConfig {
                c,
                tol: self.config.tol,
                max_iter: self.config.max_iter,
                ..Default::default()
            };
            let mut solver = SmoSolver::new(config, self.config.kernel.clone());
            let solution = solver.solve(x, &binary_y)?;

            // Extract support vectors
            let support_indices: Vec<usize> = solution
                .alpha
                .iter()
                .enumerate()
                .filter(|(_, &alpha)| alpha.abs() > 1e-8)
                .map(|(i, _)| i)
                .collect();

            let support_vectors = if support_indices.is_empty() {
                Array2::zeros((1, n_features))
            } else {
                let mut sv = Array2::zeros((support_indices.len(), n_features));
                for (i, &idx) in support_indices.iter().enumerate() {
                    sv.row_mut(i).assign(&x.row(idx));
                }
                sv
            };

            let dual_coef =
                Array1::from_vec(support_indices.iter().map(|&i| solution.alpha[i]).collect());

            let support = Array1::from_vec(support_indices);
            let n_support = Array1::from_vec(vec![support.len()]);

            Ok(NuSVC {
                config: self.config,
                state: PhantomData,
                support_vectors_: Some(support_vectors),
                support_: Some(support),
                dual_coef_: Some(dual_coef),
                intercept_: Some(solution.b),
                classes_: Some(classes_array),
                n_features_in_: Some(n_features),
                n_support_: Some(n_support),
            })
        } else {
            // For multiclass, this is a placeholder implementation
            // In practice, you would use One-vs-One or One-vs-Rest
            Err(SklearsError::NotImplemented(
                "Multiclass Nu-SVC not yet implemented".to_string(),
            ))
        }
    }
}

impl Predict<Array2<Float>, Array1<i32>> for NuSVC<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_.unwrap(),
                actual: x.ncols(),
            });
        }

        let decision_values = self.decision_function(x)?;
        let classes = self.classes_.as_ref().unwrap();

        // For binary classification
        if classes.len() == 2 {
            let predictions = decision_values
                .iter()
                .map(|&val| if val >= 0.0 { classes[1] } else { classes[0] })
                .collect();
            Ok(Array1::from_vec(predictions))
        } else {
            Err(SklearsError::NotImplemented(
                "Multiclass prediction not implemented".to_string(),
            ))
        }
    }
}

impl NuSVC<Trained> {
    /// Get the decision function values
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let support_vectors = self.support_vectors_.as_ref().unwrap();
        let dual_coef = self.dual_coef_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();

        let kernel = &self.config.kernel;
        let mut decisions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut score = intercept;
            for (j, &coef) in dual_coef.iter().enumerate() {
                let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                score += coef * k_val;
            }
            decisions[i] = score;
        }

        Ok(decisions)
    }

    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_.as_ref().unwrap()
    }

    /// Get the indices of support vectors
    pub fn support(&self) -> &Array1<usize> {
        self.support_.as_ref().unwrap()
    }

    /// Get the dual coefficients
    pub fn dual_coef(&self) -> &Array1<Float> {
        self.dual_coef_.as_ref().unwrap()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Float {
        self.intercept_.unwrap()
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the number of support vectors for each class
    pub fn n_support(&self) -> &Array1<usize> {
        self.n_support_.as_ref().unwrap()
    }

    /// Get the number of features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::KernelType;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nusvc_creation() {
        let nusvc = NuSVC::new()
            .nu(0.3)
            .kernel(KernelType::Linear)
            .tol(1e-4)
            .max_iter(500)
            .random_state(42);

        assert_eq!(nusvc.config.nu, 0.3);
        assert_eq!(nusvc.config.tol, 1e-4);
        assert_eq!(nusvc.config.max_iter, 500);
        assert_eq!(nusvc.config.random_state, Some(42));
    }

    #[test]
    #[should_panic(expected = "Nu must be in the range (0, 1]")]
    fn test_nusvc_invalid_nu() {
        let _nusvc = NuSVC::new().nu(1.5);
    }

    #[test]
    #[ignore = "Slow test: trains SVM with SMO algorithm. Run with --ignored flag"]
    fn test_nusvc_binary_classification() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let nusvc = NuSVC::new()
            .nu(0.5)
            .kernel(KernelType::Linear)
            .tol(0.1) // Very high tolerance for test speed
            .max_iter(10); // Very low iterations for tests
        let fitted_model = nusvc.fit(&x, &y).unwrap();

        assert_eq!(fitted_model.n_features_in(), 2);
        assert_eq!(fitted_model.classes().len(), 2);

        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // Check that predictions are valid class labels
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    #[ignore = "Slow test: trains SVM with SMO algorithm. Run with --ignored flag"]
    fn test_nusvc_decision_function() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0],];
        let y = array![0, 0, 1, 1];

        let nusvc = NuSVC::new()
            .nu(0.5)
            .kernel(KernelType::Linear)
            .tol(0.1) // Very high tolerance for test speed
            .max_iter(10); // Very low iterations for tests
        let fitted_model = nusvc.fit(&x, &y).unwrap();

        let decision_values = fitted_model.decision_function(&x).unwrap();
        assert_eq!(decision_values.len(), 4);

        // Decision values should be finite
        for &val in decision_values.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_nusvc_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0]; // Wrong length

        let nusvc = NuSVC::new();
        let result = nusvc.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_nusvc_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0, 1];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let nusvc = NuSVC::new();
        let fitted_model = nusvc.fit(&x_train, &y_train).unwrap();
        let result = fitted_model.predict(&x_test);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Feature"));
    }

    #[test]
    fn test_nusvc_single_class() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 0]; // Only one class

        let nusvc = NuSVC::new();
        let result = nusvc.fit(&x, &y);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least 2 classes"));
    }

    #[test]
    fn test_nusvc_empty_data() {
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<i32> = Array1::zeros(0);

        let nusvc = NuSVC::new();
        let result = nusvc.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Empty dataset"));
    }

    #[test]
    #[ignore = "Slow test: performs cross-validation for parameter selection. Run with --ignored flag"]
    fn test_nusvc_auto_parameter_selection() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let nu_candidates = vec![0.1, 0.3, 0.5, 0.7];
        let nusvc = NuSVC::new()
            .kernel(KernelType::Linear)
            .tol(0.1)
            .max_iter(10)
            .random_state(42)
            .auto_select_nu(&x, &y, Some(nu_candidates), Some(3))
            .unwrap();

        // Verify that a nu parameter was selected
        assert!(nusvc.config.nu > 0.0 && nusvc.config.nu <= 1.0);

        // Verify the model can still be trained
        let fitted_model = nusvc.fit(&x, &y).unwrap();
        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 8);
    }
}
