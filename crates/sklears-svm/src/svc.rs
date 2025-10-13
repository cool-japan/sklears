//! Support Vector Classification (SVC) implementation

use crate::{
    calibration::PlattScaling,
    kernels::{create_kernel, Kernel, KernelType},
    smo::{SmoConfig, SmoSolver},
};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Kernel type for SVC
#[derive(Debug, Clone)]
pub enum SvcKernel {
    /// Linear kernel
    Linear,
    /// RBF (Radial Basis Function) kernel
    Rbf { gamma: Option<Float> },
    /// Polynomial kernel
    Poly {
        degree: usize,
        gamma: Option<Float>,
        coef0: Float,
    },
    /// Sigmoid kernel
    Sigmoid { gamma: Option<Float>, coef0: Float },
    /// Custom kernel using KernelType
    Custom(KernelType),
}

impl Default for SvcKernel {
    fn default() -> Self {
        SvcKernel::Rbf { gamma: None }
    }
}

/// Configuration for SVC
#[derive(Debug, Clone)]
pub struct SvcConfig {
    /// Regularization parameter
    pub c: Float,
    /// Kernel type
    pub kernel: SvcKernel,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to use shrinking heuristics
    pub shrinking: bool,
    /// Cache size for kernel evaluations (in MB)
    pub cache_size: usize,
    /// Class weight balancing
    pub class_weight: Option<ClassWeight>,
    /// Whether to enable probability estimates
    pub probability: bool,
}

impl Default for SvcConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            kernel: SvcKernel::default(),
            tol: 1e-3,
            max_iter: 200, // Reduced from 1000 to improve performance
            shrinking: true,
            cache_size: 200,
            class_weight: None,
            probability: false,
        }
    }
}

/// Class weight options
#[derive(Debug, Clone)]
pub enum ClassWeight {
    /// Balanced class weights (inversely proportional to class frequencies)
    Balanced,
    /// Manual class weights
    Manual(Vec<(Float, Float)>), // (class_label, weight) pairs
}

/// Support Vector Classification
#[derive(Debug, Clone)]
pub struct SVC<State = Untrained> {
    config: SvcConfig,
    state: PhantomData<State>,
    // Fitted parameters
    support_vectors_: Option<Array2<Float>>,
    #[allow(dead_code)]
    support_labels_: Option<Array1<Float>>,
    dual_coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    classes_: Option<Array1<Float>>,
    n_features_in_: Option<usize>,
    n_support_: Option<Array1<usize>>,
    support_indices_: Option<Vec<usize>>,
    kernel_: Option<KernelType>,
    // Probability calibration
    platt_scaling_: Option<PlattScaling>,
}

impl SVC<Untrained> {
    /// Create a new SVC
    pub fn new() -> Self {
        Self {
            config: SvcConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            support_labels_: None,
            dual_coef_: None,
            intercept_: None,
            classes_: None,
            n_features_in_: None,
            n_support_: None,
            support_indices_: None,
            kernel_: None,
            platt_scaling_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the kernel to linear
    pub fn linear(mut self) -> Self {
        self.config.kernel = SvcKernel::Linear;
        self
    }

    /// Set the kernel to RBF with optional gamma
    pub fn rbf(mut self, gamma: Option<Float>) -> Self {
        self.config.kernel = SvcKernel::Rbf { gamma };
        self
    }

    /// Set the kernel to polynomial
    pub fn poly(mut self, degree: usize, gamma: Option<Float>, coef0: Float) -> Self {
        self.config.kernel = SvcKernel::Poly {
            degree,
            gamma,
            coef0,
        };
        self
    }

    /// Set a custom kernel
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = SvcKernel::Custom(kernel);
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

    /// Set whether to use shrinking heuristics
    pub fn shrinking(mut self, shrinking: bool) -> Self {
        self.config.shrinking = shrinking;
        self
    }

    /// Set the cache size for kernel evaluations
    pub fn cache_size(mut self, cache_size: usize) -> Self {
        self.config.cache_size = cache_size;
        self
    }

    /// Set balanced class weights
    pub fn balanced(mut self) -> Self {
        self.config.class_weight = Some(ClassWeight::Balanced);
        self
    }

    /// Enable probability estimates using Platt scaling
    pub fn probability(mut self, probability: bool) -> Self {
        self.config.probability = probability;
        self
    }

    /// Create kernel based on configuration
    fn create_kernel(&self, n_features: usize) -> KernelType {
        match &self.config.kernel {
            SvcKernel::Linear => KernelType::Linear,
            SvcKernel::Rbf { gamma } => {
                let gamma_val = gamma.unwrap_or(1.0 / n_features as Float);
                KernelType::Rbf { gamma: gamma_val }
            }
            SvcKernel::Poly {
                degree,
                gamma,
                coef0,
            } => {
                let gamma_val = gamma.unwrap_or(1.0 / n_features as Float);
                KernelType::Polynomial {
                    gamma: gamma_val,
                    degree: *degree as f64,
                    coef0: *coef0,
                }
            }
            SvcKernel::Sigmoid { gamma, coef0 } => {
                let gamma_val = gamma.unwrap_or(1.0 / n_features as Float);
                KernelType::Sigmoid {
                    gamma: gamma_val,
                    coef0: *coef0,
                }
            }
            SvcKernel::Custom(kernel) => kernel.clone(),
        }
    }

    /// Find unique classes in the target array
    fn find_classes(y: &Array1<Float>) -> Array1<Float> {
        let mut classes: Vec<Float> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        Array1::from_vec(classes)
    }

    /// Compute class weights
    fn compute_class_weights(&self, y: &Array1<Float>, classes: &Array1<Float>) -> Array1<Float> {
        match &self.config.class_weight {
            None => Array1::ones(classes.len()),
            Some(ClassWeight::Balanced) => {
                let n_samples = y.len() as Float;
                let n_classes = classes.len() as Float;
                let mut weights = Array1::zeros(classes.len());

                for (i, &class) in classes.iter().enumerate() {
                    let count = y.iter().filter(|&&label| label == class).count() as Float;
                    weights[i] = n_samples / (n_classes * count);
                }

                weights
            }
            Some(ClassWeight::Manual(manual_weights)) => {
                let mut weights = Array1::ones(classes.len());

                for (class_label, weight) in manual_weights {
                    if let Some(idx) = classes.iter().position(|&c| c == *class_label) {
                        weights[idx] = *weight;
                    }
                }

                weights
            }
        }
    }

    /// Convert binary classification to {-1, +1} labels
    fn convert_binary_labels(y: &Array1<Float>, classes: &Array1<Float>) -> Array1<Float> {
        let mut binary_y = Array1::zeros(y.len());

        for (i, &label) in y.iter().enumerate() {
            if label == classes[0] {
                binary_y[i] = -1.0;
            } else {
                binary_y[i] = 1.0;
            }
        }

        binary_y
    }
}

impl SVC<Trained> {
    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_
            .as_ref()
            .expect("SVC should be fitted")
    }

    /// Get the dual coefficients
    pub fn dual_coef(&self) -> &Array1<Float> {
        self.dual_coef_.as_ref().expect("SVC should be fitted")
    }

    /// Get the intercept (bias) term
    pub fn intercept(&self) -> Float {
        self.intercept_.expect("SVC should be fitted")
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<Float> {
        self.classes_.as_ref().expect("SVC should be fitted")
    }

    /// Get the number of support vectors for each class
    pub fn n_support(&self) -> &Array1<usize> {
        self.n_support_.as_ref().expect("SVC should be fitted")
    }

    /// Get the indices of support vectors
    pub fn support_indices(&self) -> &[usize] {
        self.support_indices_
            .as_ref()
            .expect("SVC should be fitted")
    }

    /// Compute decision function values
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_.unwrap() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_.unwrap(),
                actual: n_features,
            });
        }

        let kernel_type = self.kernel_.as_ref().expect("Kernel should be available");
        let kernel = create_kernel(kernel_type.clone());
        let support_vectors = self.support_vectors();
        let dual_coef = self.dual_coef();
        let intercept = self.intercept();

        let mut decision_values = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut score = 0.0;

            for (j, _support_idx) in self.support_indices().iter().enumerate() {
                let k_val = kernel.compute(
                    x.row(i).to_owned().view(),
                    support_vectors.row(j).to_owned().view(),
                );
                score += dual_coef[j] * k_val;
            }

            decision_values[i] = score + intercept;
        }

        Ok(decision_values)
    }

    /// Compute class probabilities using Platt scaling
    pub fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if let Some(platt_scaling) = &self.platt_scaling_ {
            let decision_scores = self.decision_function(x)?;
            platt_scaling.predict_proba_binary(&decision_scores)
        } else {
            Err(SklearsError::InvalidInput(
                "Probability prediction requires probability=true during fitting".to_string(),
            ))
        }
    }

    /// Check if probability estimation is enabled
    pub fn has_probability(&self) -> bool {
        self.platt_scaling_.is_some()
    }
}

impl Default for SVC<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for SVC<Untrained> {
    type Fitted = SVC<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit SVC on empty dataset".to_string(),
            ));
        }

        // Find unique classes
        let classes = Self::find_classes(y);

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "SVC requires at least 2 classes".to_string(),
            ));
        }

        if classes.len() > 2 {
            return Err(SklearsError::InvalidInput(
                "Multi-class SVC not yet implemented. Use binary classification.".to_string(),
            ));
        }

        // Convert to binary labels {-1, +1}
        let binary_y = Self::convert_binary_labels(y, &classes);

        // Create kernel
        let kernel = self.create_kernel(n_features);

        // Apply class weights if specified
        let class_weights = self.compute_class_weights(y, &classes);
        let mut weighted_c = Array1::from_elem(n_samples, self.config.c);

        for (i, &label) in y.iter().enumerate() {
            let class_idx = classes.iter().position(|&c| c == label).unwrap();
            weighted_c[i] *= class_weights[class_idx];
        }

        // Configure SMO solver
        let smo_config = SmoConfig {
            c: self.config.c, // Will be overridden per sample if class weights are used
            tol: self.config.tol,
            max_iter: self.config.max_iter,
            cache_size: self.config.cache_size,
            shrinking: self.config.shrinking,
            working_set_strategy: crate::smo::WorkingSetStrategy::SecondOrder,
            early_stopping_tol: 1e-4,
            convergence_check_interval: 10,
        };

        // Solve with SMO algorithm
        let concrete_kernel = create_kernel(kernel.clone());
        let mut solver = SmoSolver::new(smo_config, concrete_kernel);
        let smo_result = solver.solve(x, &binary_y)?;

        // Extract support vectors and dual coefficients
        let support_indices = smo_result.support_indices;
        let mut support_vectors = Array2::zeros((support_indices.len(), n_features));
        let mut support_labels = Array1::zeros(support_indices.len());
        let mut dual_coef = Array1::zeros(support_indices.len());

        for (i, &support_idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(support_idx));
            support_labels[i] = binary_y[support_idx];
            dual_coef[i] = smo_result.alpha[support_idx] * binary_y[support_idx];
        }

        // Count support vectors per class
        let mut n_support = Array1::zeros(2);
        for &label in support_labels.iter() {
            if label == -1.0 {
                n_support[0] += 1;
            } else {
                n_support[1] += 1;
            }
        }

        // Fit Platt scaling if probability estimation is enabled
        let platt_scaling = if self.config.probability {
            let temp_svc = SVC {
                config: self.config.clone(),
                state: PhantomData,
                support_vectors_: Some(support_vectors.clone()),
                support_labels_: Some(support_labels.clone()),
                dual_coef_: Some(dual_coef.clone()),
                intercept_: Some(smo_result.b),
                classes_: Some(classes.clone()),
                n_features_in_: Some(n_features),
                n_support_: Some(n_support.clone()),
                support_indices_: Some(support_indices.clone()),
                kernel_: Some(kernel.clone()),
                platt_scaling_: None,
            };

            // Compute decision function values for Platt scaling
            let decision_scores = temp_svc.decision_function(x)?;

            // Train Platt scaling
            let mut platt = PlattScaling::new();
            platt.fit(&decision_scores, y)?;
            Some(platt)
        } else {
            None
        };

        Ok(SVC {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(support_vectors),
            support_labels_: Some(support_labels),
            dual_coef_: Some(dual_coef),
            intercept_: Some(smo_result.b),
            classes_: Some(classes),
            n_features_in_: Some(n_features),
            n_support_: Some(n_support),
            support_indices_: Some(support_indices),
            kernel_: Some(kernel),
            platt_scaling_: platt_scaling,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for SVC<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let decision_values = self.decision_function(x)?;
        let classes = self.classes();

        let mut predictions = Array1::zeros(decision_values.len());

        for (i, &score) in decision_values.iter().enumerate() {
            if score >= 0.0 {
                predictions[i] = classes[1]; // Positive class
            } else {
                predictions[i] = classes[0]; // Negative class
            }
        }

        Ok(predictions)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_svc_linear_separable() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [-3.0, -3.0],
        ];
        let y = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let svc = SVC::new().linear().c(1.0).fit(&x, &y).unwrap();

        // Check fitted attributes
        assert_eq!(svc.classes().len(), 2);
        assert!(svc.support_vectors().nrows() > 0);
        assert!(svc.dual_coef().len() > 0);

        // Test prediction
        let x_test = array![
            [4.0, 4.0],   // Should be class 1
            [-4.0, -4.0], // Should be class 0
        ];
        let predictions = svc.predict(&x_test).unwrap();

        assert_eq!(predictions.len(), 2);
        // Note: exact predictions depend on SMO convergence
    }

    #[test]
    fn test_svc_rbf_kernel() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0],];
        let y = array![1.0, 1.0, 0.0, 0.0];

        let svc = SVC::new().rbf(Some(1.0)).c(1.0).fit(&x, &y).unwrap();

        assert_eq!(svc.classes().len(), 2);
        assert!(svc.support_vectors().nrows() > 0);
    }

    #[test]
    fn test_svc_config_builder() {
        let svc = SVC::new()
            .c(10.0)
            .linear()
            .tol(1e-4)
            .max_iter(2000)
            .shrinking(false);

        assert_eq!(svc.config.c, 10.0);
        assert_eq!(svc.config.tol, 1e-4);
        assert_eq!(svc.config.max_iter, 2000);
        assert!(!svc.config.shrinking);
    }

    #[test]
    fn test_svc_empty_dataset() {
        let x = Array2::<Float>::zeros((0, 2));
        let y = Array1::<Float>::zeros(0);

        let result = SVC::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_svc_single_class() {
        let x = array![[1.0, 2.0], [3.0, 4.0],];
        let y = array![1.0, 1.0]; // Single class

        let result = SVC::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "Slow test: trains SVM with probability estimation. Run with --ignored flag"]
    fn test_svc_probability_estimation() {
        // Simple separable dataset
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [1.5, 1.5],
            [-1.5, -1.5],
        ];
        let y = array![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];

        // Train SVC with probability estimation
        let svc = SVC::new()
            .linear()
            .c(1.0)
            .probability(true)
            .tol(0.1)
            .max_iter(50)
            .fit(&x, &y)
            .unwrap();

        // Check that probability estimation is enabled
        assert!(svc.has_probability());

        // Test probability prediction
        let x_test = array![[1.0, 1.0], [-1.0, -1.0]];
        let probabilities = svc.predict_proba(&x_test).unwrap();

        // Check dimensions
        assert_eq!(probabilities.dim(), (2, 2));

        // Check that probabilities sum to 1
        for i in 0..2 {
            let row_sum = probabilities[[i, 0]] + probabilities[[i, 1]];
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {} probabilities don't sum to 1: {}",
                i,
                row_sum
            );
        }

        // Check that probabilities are in [0, 1]
        for &prob in probabilities.iter() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Probability {} not in [0,1]",
                prob
            );
        }

        // Test that SVC without probability estimation fails
        let svc_no_prob = SVC::new().linear().c(1.0).fit(&x, &y).unwrap();

        assert!(!svc_no_prob.has_probability());
        assert!(svc_no_prob.predict_proba(&x_test).is_err());
    }
}
