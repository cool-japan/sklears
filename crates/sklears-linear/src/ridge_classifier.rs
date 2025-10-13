//! Ridge Classifier
//!
//! Classifier using Ridge regression. This classifier first converts binary targets to
//! {-1, 1} and then treats the problem as a regression task (multi-output regression in
//! the multiclass case).

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::solve;
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::{Float, Int},
};

use crate::solver::Solver;

/// Configuration for RidgeClassifier
#[derive(Debug, Clone)]
pub struct RidgeClassifierConfig {
    /// Regularization strength; must be a positive float
    pub alpha: Float,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// If True, the regressors X will be normalized before regression
    pub normalize: bool,
    /// Solver to use in the computational routines
    pub solver: Solver,
    /// Maximum number of iterations for iterative solvers
    pub max_iter: Option<usize>,
    /// Precision of the solution
    pub tol: Float,
    /// Random state for shuffling the data
    pub random_state: Option<u64>,
}

impl Default for RidgeClassifierConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            normalize: false,
            solver: Solver::Auto,
            max_iter: None,
            tol: 1e-3,
            random_state: None,
        }
    }
}

/// Ridge Classifier
pub struct RidgeClassifier<State = Untrained> {
    config: RidgeClassifierConfig,
    state: PhantomData<State>,
    coef_: Option<Array2<Float>>,
    intercept_: Option<Array1<Float>>,
    classes_: Option<Array1<Int>>,
    n_features_in_: Option<usize>,
}

impl RidgeClassifier<Untrained> {
    /// Create a new RidgeClassifier with default configuration
    pub fn new() -> Self {
        Self {
            config: RidgeClassifierConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            classes_: None,
            n_features_in_: None,
        }
    }

    /// Set the regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set whether to fit the intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to normalize the features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Set the solver
    pub fn solver(mut self, solver: Solver) -> Self {
        self.config.solver = solver;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }
}

impl Default for RidgeClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RidgeClassifier<Untrained> {
    type Float = Float;
    type Config = RidgeClassifierConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for RidgeClassifier<Trained> {
    type Float = Float;
    type Config = RidgeClassifierConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Convert class labels to regression targets
fn label_binarize(y: &Array1<Int>, classes: &[Int]) -> Array2<Float> {
    let n_samples = y.len();
    let n_classes = classes.len();

    if n_classes == 2 {
        // Binary case: convert to {-1, 1}
        let mut y_bin = Array1::zeros(n_samples);
        for (i, &label) in y.iter().enumerate() {
            if label == classes[1] {
                y_bin[i] = 1.0;
            } else {
                y_bin[i] = -1.0;
            }
        }
        y_bin.insert_axis(Axis(1))
    } else {
        // Multi-class case: one-vs-all encoding
        let mut y_bin = Array2::from_elem((n_samples, n_classes), -1.0);
        for (i, &label) in y.iter().enumerate() {
            for (j, &class) in classes.iter().enumerate() {
                if label == class {
                    y_bin[[i, j]] = 1.0;
                }
            }
        }
        y_bin
    }
}

impl Fit<Array2<Float>, Array1<Int>> for RidgeClassifier<Untrained> {
    type Fitted = RidgeClassifier<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Int>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Get unique classes
        let mut classes: Vec<Int> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "At least two classes are required".to_string(),
            ));
        }

        // Convert labels to regression targets
        let y_bin = label_binarize(y, &classes);

        // Center X and y if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.config.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y_bin.mean_axis(Axis(0)).unwrap();
            let x_centered = x - &x_mean;
            let y_centered = if n_classes == 2 {
                // For binary case, just center the single column
                y_bin - y_mean[0]
            } else {
                // For multi-class, center each column
                &y_bin - &y_mean
            };
            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y_bin.clone(), None, None)
        };

        // Solve the ridge regression problem for each class
        let mut coef = Array2::zeros((n_classes, n_features));

        // Compute X^T X + alpha * I
        let xt_x = x_centered.t().dot(&x_centered);
        let xt_x_reg =
            &xt_x + &(Array2::<Float>::eye(n_features) * self.config.alpha * n_samples as Float);

        if n_classes == 2 {
            // Binary case: solve once
            let xt_y = x_centered.t().dot(&y_centered.column(0));

            match solve(&xt_x_reg.view(), &xt_y.view(), None) {
                Ok(solution) => {
                    coef.row_mut(0).assign(&(-&solution));
                    coef.row_mut(1).assign(&solution);
                }
                Err(_) => {
                    return Err(SklearsError::InvalidInput(
                        "Failed to solve linear system".to_string(),
                    ));
                }
            }
        } else {
            // Multi-class case: solve for each class
            for k in 0..n_classes {
                let xt_y = x_centered.t().dot(&y_centered.column(k));

                match solve(&xt_x_reg.view(), &xt_y.view(), None) {
                    Ok(solution) => {
                        coef.row_mut(k).assign(&solution);
                    }
                    Err(_) => {
                        return Err(SklearsError::InvalidInput(format!(
                            "Failed to solve linear system for class {}",
                            k
                        )));
                    }
                }
            }
        }

        // Compute intercept if needed
        let intercept = if self.config.fit_intercept {
            let x_mean = x_mean.unwrap();
            let y_mean = y_mean.unwrap();

            if n_classes == 2 {
                // Binary case
                let intercept_val = y_mean[0] - x_mean.dot(&coef.row(1));
                Array1::from_vec(vec![-intercept_val, intercept_val])
            } else {
                // Multi-class case
                let mut intercept = Array1::zeros(n_classes);
                for k in 0..n_classes {
                    intercept[k] = y_mean[k] - x_mean.dot(&coef.row(k));
                }
                intercept
            }
        } else {
            Array1::zeros(n_classes)
        };

        Ok(RidgeClassifier {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            classes_: Some(Array1::from_vec(classes)),
            n_features_in_: Some(n_features),
        })
    }
}

impl Predict<Array2<Float>, Array1<Int>> for RidgeClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Int>> {
        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        // Compute decision function
        let scores = x.dot(&coef.t()) + intercept;

        // Predict class with maximum score
        let predictions = scores
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

impl Score<Array2<Float>, Array1<Int>> for RidgeClassifier<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Int>) -> Result<Float> {
        let predictions = self.predict(x)?;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_val)| pred == true_val)
            .count();

        Ok(correct as Float / y.len() as Float)
    }
}

impl RidgeClassifier<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<&Array1<Float>> {
        self.intercept_.as_ref()
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<Int> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the number of features seen during fit
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get decision function (raw scores)
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.as_ref().unwrap();

        Ok(x.dot(&coef.t()) + intercept)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_ridge_classifier_binary() {
        // Simple linearly separable data
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [-3.0, -3.0],
        ];
        let y = array![1, 1, 1, 0, 0, 0];

        let model = RidgeClassifier::new().alpha(1.0).fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();
        let accuracy = model.score(&x, &y).unwrap();

        // Should achieve good classification on this simple data
        assert!(accuracy > 0.8);

        // Check binary class structure
        assert_eq!(model.classes().len(), 2);
        assert_eq!(model.coef().nrows(), 2);
    }

    #[test]
    fn test_ridge_classifier_multiclass() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let model = RidgeClassifier::new().alpha(0.1).fit(&x, &y).unwrap();

        let accuracy = model.score(&x, &y).unwrap();
        assert!(accuracy > 0.8);

        // Check that we have the right number of classes
        assert_eq!(model.classes().len(), 3);
        assert_eq!(model.coef().nrows(), 3);
    }

    #[test]
    fn test_ridge_classifier_no_intercept() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0],];
        let y = array![1, 1, 0, 0];

        let model = RidgeClassifier::new()
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        let intercept = model.intercept().unwrap();
        assert!(intercept.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ridge_classifier_strong_regularization() {
        let x = array![[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0],];
        let y = array![0, 0, 1, 1];

        // With very high alpha, coefficients should be small
        let model = RidgeClassifier::new().alpha(1000.0).fit(&x, &y).unwrap();

        let coef = model.coef();
        assert!(coef.iter().all(|&c| c.abs() < 0.1));
    }

    #[test]
    fn test_ridge_classifier_decision_function() {
        let x = array![[1.0, 1.0], [-1.0, -1.0],];
        let y = array![1, 0];

        let model = RidgeClassifier::new().fit(&x, &y).unwrap();

        let decision = model.decision_function(&x).unwrap();

        // For binary classification, we should have 2 columns
        assert_eq!(decision.ncols(), 2);

        // The predicted class should have the highest score
        let predictions = model.predict(&x).unwrap();
        for (i, &pred) in predictions.iter().enumerate() {
            let scores = decision.row(i);
            let max_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(model.classes()[max_idx], pred);
        }
    }

    #[test]
    fn test_label_binarize() {
        // Test binary case
        let y = array![0, 1, 1, 0];
        let classes = vec![0, 1];
        let y_bin = label_binarize(&y, &classes);

        assert_eq!(y_bin.shape(), &[4, 1]);
        assert_eq!(y_bin[[0, 0]], -1.0);
        assert_eq!(y_bin[[1, 0]], 1.0);

        // Test multi-class case
        let y = array![0, 1, 2, 0];
        let classes = vec![0, 1, 2];
        let y_bin = label_binarize(&y, &classes);

        assert_eq!(y_bin.shape(), &[4, 3]);
        assert_eq!(y_bin[[0, 0]], 1.0);
        assert_eq!(y_bin[[0, 1]], -1.0);
        assert_eq!(y_bin[[2, 2]], 1.0);
    }
}
