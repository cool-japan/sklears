//! Least Squares Support Vector Machines (LS-SVM)
//!
//! This module implements Least Squares Support Vector Machines, which use equality
//! constraints instead of inequality constraints, leading to solving a linear system
//! instead of a quadratic programming problem. This makes LS-SVM computationally
//! more efficient for many problems.

use crate::kernels::{Kernel, KernelType};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::compat::ArrayLinalgExt;
// Removed SVD import - using ArrayLinalgExt for both solve and svd methods
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Least Squares SVM
#[derive(Debug, Clone)]
pub struct LSVMConfig {
    /// Regularization parameter (gamma)
    pub gamma: Float,
    /// Kernel function to use
    pub kernel: KernelType,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Tolerance for numerical stability
    pub tol: Float,
    /// Whether to use regularized kernel matrix (add regularization to diagonal)
    pub regularized_kernel: bool,
}

impl Default for LSVMConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            kernel: KernelType::Rbf { gamma: 1.0 },
            fit_intercept: true,
            tol: 1e-12,
            regularized_kernel: true,
        }
    }
}

/// Least Squares Support Vector Machine for classification and regression
#[derive(Debug)]
pub struct LSSVM<State = Untrained> {
    config: LSVMConfig,
    state: PhantomData<State>,
    // Fitted attributes
    support_vectors_: Option<Array2<Float>>,
    alpha_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    training_labels_: Option<Array1<Float>>,
    n_features_in_: Option<usize>,
}

impl LSSVM<Untrained> {
    /// Create a new LS-SVM
    pub fn new() -> Self {
        Self {
            config: LSVMConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            alpha_: None,
            intercept_: None,
            training_labels_: None,
            n_features_in_: None,
        }
    }

    /// Set the regularization parameter gamma
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the tolerance for numerical stability
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set whether to use regularized kernel matrix
    pub fn regularized_kernel(mut self, regularized: bool) -> Self {
        self.config.regularized_kernel = regularized;
        self
    }
}

impl Default for LSSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for LSSVM<Untrained> {
    type Fitted = LSSVM<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Shape mismatch: X and y must have the same number of samples".to_string(),
            ));
        }

        // Create kernel instance
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>, // Default fallback
        };

        // Compute kernel matrix
        let mut k_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let k_val = kernel.compute(x.row(i), x.row(j));
                k_matrix[[i, j]] = k_val;
            }
        }

        // Add regularization to diagonal if requested
        if self.config.regularized_kernel {
            for i in 0..n_samples {
                k_matrix[[i, i]] += 1.0 / self.config.gamma;
            }
        }

        // Solve the LS-SVM system
        let (alpha, intercept) = if self.config.fit_intercept {
            self.solve_with_intercept(&k_matrix, y)?
        } else {
            let alpha = self.solve_without_intercept(&k_matrix, y)?;
            (alpha, 0.0)
        };

        Ok(LSSVM {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(x.clone()),
            alpha_: Some(alpha),
            intercept_: Some(intercept),
            training_labels_: Some(y.clone()),
            n_features_in_: Some(n_features),
        })
    }
}

impl LSSVM<Untrained> {
    /// Solve LS-SVM system with intercept
    fn solve_with_intercept(
        &self,
        k_matrix: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Float)> {
        let n = k_matrix.nrows();

        // Create augmented system: [K + I/gamma  1; 1^T  0] [alpha; b] = [y; 0]
        let mut a_matrix = Array2::zeros((n + 1, n + 1));
        let mut b_vector = Array1::zeros(n + 1);

        // Fill K matrix part
        for i in 0..n {
            for j in 0..n {
                a_matrix[[i, j]] = k_matrix[[i, j]];
            }
        }

        // Add regularization to diagonal
        if !self.config.regularized_kernel {
            for i in 0..n {
                a_matrix[[i, i]] += 1.0 / self.config.gamma;
            }
        }

        // Add ones vector for intercept
        for i in 0..n {
            a_matrix[[i, n]] = 1.0;
            a_matrix[[n, i]] = 1.0;
        }

        // Fill b vector
        for i in 0..n {
            b_vector[i] = y[i];
        }

        // Solve the linear system using scirs2-linalg
        let solution = a_matrix.solve(&b_vector).map_err(|e| {
            SklearsError::NumericalError(format!("Failed to solve LS-SVM linear system: {}", e))
        })?;

        let alpha = solution.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let intercept = solution[n];

        Ok((alpha, intercept))
    }

    /// Solve LS-SVM system without intercept
    fn solve_without_intercept(
        &self,
        k_matrix: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n = k_matrix.nrows();
        let mut a_matrix = k_matrix.clone();

        // Add regularization to diagonal if not already done
        if !self.config.regularized_kernel {
            for i in 0..n {
                a_matrix[[i, i]] += 1.0 / self.config.gamma;
            }
        }

        // Solve the linear system using scirs2-linalg
        let alpha = a_matrix.solve(y).map_err(|e| {
            SklearsError::NumericalError(format!("Failed to solve LS-SVM linear system: {}", e))
        })?;

        Ok(alpha)
    }
}

impl Predict<Array2<Float>, Array1<Float>> for LSSVM<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(
                "Feature mismatch: X has different number of features than training data"
                    .to_string(),
            ));
        }

        let support_vectors = self.support_vectors_.as_ref().unwrap();
        let alpha = self.alpha_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>, // Default fallback
        };

        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut prediction = if self.config.fit_intercept {
                intercept
            } else {
                0.0
            };

            for j in 0..support_vectors.nrows() {
                let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                prediction += alpha[j] * k_val;
            }

            predictions[i] = prediction;
        }

        Ok(predictions)
    }
}

impl LSSVM<Trained> {
    /// Get the support vectors (training data for LS-SVM)
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_.as_ref().unwrap()
    }

    /// Get the alpha coefficients
    pub fn alpha(&self) -> &Array1<Float> {
        self.alpha_.as_ref().unwrap()
    }

    /// Get the intercept term
    pub fn intercept(&self) -> Float {
        self.intercept_.unwrap()
    }

    /// Get the training labels
    pub fn training_labels(&self) -> &Array1<Float> {
        self.training_labels_.as_ref().unwrap()
    }

    /// Get the number of features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Compute the decision function values
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        self.predict(x)
    }

    /// Get the kernel matrix for the training data
    pub fn kernel_matrix(&self) -> Result<Array2<Float>> {
        let support_vectors = self.support_vectors();
        let n_samples = support_vectors.nrows();
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>, // Default fallback
        };

        let mut k_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let k_val = kernel.compute(support_vectors.row(i), support_vectors.row(j));
                k_matrix[[i, j]] = k_val;
            }
        }

        Ok(k_matrix)
    }
}

/// LS-SVM Classifier wrapper for binary classification
#[derive(Debug)]
pub struct LSVMClassifier<State = Untrained> {
    lssvm: LSSVM<State>,
}

impl LSVMClassifier<Untrained> {
    /// Create a new LS-SVM classifier
    pub fn new() -> Self {
        Self {
            lssvm: LSSVM::new(),
        }
    }

    /// Set the regularization parameter gamma
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.lssvm = self.lssvm.gamma(gamma);
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.lssvm = self.lssvm.kernel(kernel);
        self
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.lssvm = self.lssvm.fit_intercept(fit_intercept);
        self
    }
}

impl Default for LSVMClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<i32>> for LSVMClassifier<Untrained> {
    type Fitted = LSVMClassifier<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        // Convert labels to {-1, +1}
        let unique_classes: Vec<i32> = {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        if unique_classes.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "LS-SVM classifier currently supports only binary classification".to_string(),
            ));
        }

        let y_float = Array1::from_vec(
            y.iter()
                .map(|&label| {
                    if label == unique_classes[0] {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect(),
        );

        let fitted_lssvm = self.lssvm.fit(x, &y_float)?;

        Ok(LSVMClassifier {
            lssvm: fitted_lssvm,
        })
    }
}

impl Predict<Array2<Float>, Array1<i32>> for LSVMClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let decision_values = self.lssvm.predict(x)?;

        let predictions = Array1::from_vec(
            decision_values
                .iter()
                .map(|&val| if val >= 0.0 { 1 } else { 0 })
                .collect(),
        );

        Ok(predictions)
    }
}

impl LSVMClassifier<Trained> {
    /// Get the decision function values
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        self.lssvm.decision_function(x)
    }

    /// Get the underlying LS-SVM model
    pub fn lssvm(&self) -> &LSSVM<Trained> {
        &self.lssvm
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lssvm_creation() {
        let lssvm = LSSVM::new()
            .gamma(2.0)
            .kernel(KernelType::Linear)
            .fit_intercept(false)
            .tol(1e-10)
            .regularized_kernel(false);

        assert_eq!(lssvm.config.gamma, 2.0);
        assert_eq!(lssvm.config.kernel, KernelType::Linear);
        assert!(!lssvm.config.fit_intercept);
        assert_eq!(lssvm.config.tol, 1e-10);
        assert!(!lssvm.config.regularized_kernel);
    }

    #[test]
    fn test_lssvm_classifier_creation() {
        let classifier = LSVMClassifier::new()
            .gamma(1.5)
            .kernel(KernelType::Rbf { gamma: 0.5 })
            .fit_intercept(true);

        assert_eq!(classifier.lssvm.config.gamma, 1.5);
        assert_eq!(
            classifier.lssvm.config.kernel,
            KernelType::Rbf { gamma: 0.5 }
        );
        assert!(classifier.lssvm.config.fit_intercept);
    }

    #[test]
    #[ignore = "Slow test: trains LS-SVM. Run with --ignored flag"]
    fn test_lssvm_regression() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2*x

        let lssvm = LSSVM::new()
            .gamma(1.0)
            .kernel(KernelType::Linear)
            .fit_intercept(true);

        let fitted_model = lssvm.fit(&x, &y).unwrap();

        assert_eq!(fitted_model.n_features_in(), 1);

        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 5);

        // Predictions should be close to the linear relationship
        for (i, &pred) in predictions.iter().enumerate() {
            let expected = 2.0 * (i + 1) as Float;
            assert!(
                (pred - expected).abs() < 1.0,
                "Prediction {} should be close to {}",
                pred,
                expected
            );
        }
    }

    #[test]
    #[ignore = "Slow test: trains LS-SVM classifier. Run with --ignored flag"]
    fn test_lssvm_binary_classification() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let classifier = LSVMClassifier::new()
            .gamma(1.0)
            .kernel(KernelType::Linear)
            .fit_intercept(true);

        let fitted_model = classifier.fit(&x, &y).unwrap();

        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // Check that predictions are valid class labels
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }

        let decision_values = fitted_model.decision_function(&x).unwrap();
        assert_eq!(decision_values.len(), 6);

        // Decision values should be finite
        for &val in decision_values.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_lssvm_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0]; // Wrong length

        let lssvm = LSSVM::new();
        let result = lssvm.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_lssvm_classifier_multiclass_error() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0, 1, 2]; // Three classes

        let classifier = LSVMClassifier::new();
        let result = classifier.fit(&x, &y);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("binary classification"));
    }

    #[test]
    fn test_lssvm_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![1.0, 2.0];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let lssvm = LSSVM::new();
        let fitted_model = lssvm.fit(&x_train, &y_train).unwrap();
        let result = fitted_model.predict(&x_test);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Feature mismatch"));
    }
}
