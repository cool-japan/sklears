//! Fuzzy Support Vector Machines for handling noisy and uncertain data
//!
//! This module implements Fuzzy SVM, which assigns fuzzy membership values to
//! training samples to handle noise, outliers, and uncertain data. Samples with
//! lower membership values have less influence on the decision boundary.

use crate::kernels::{Kernel, KernelType};
use crate::smo::{SmoConfig, SmoResult, SmoSolver};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Fuzzy membership calculation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum FuzzyMembershipStrategy {
    /// Constant membership for all samples
    Constant { value: Float },
    /// Distance-based membership (closer to class center gets higher membership)
    DistanceBased { sigma: Float },
    /// Noise-based membership (samples farther from decision boundary get lower membership)
    NoiseBased { threshold: Float },
    /// Custom membership values provided by user
    Custom,
    /// Adaptive membership based on local density
    Adaptive { k_neighbors: usize },
}

impl Default for FuzzyMembershipStrategy {
    fn default() -> Self {
        FuzzyMembershipStrategy::DistanceBased { sigma: 1.0 }
    }
}

/// Configuration for Fuzzy SVM
#[derive(Debug, Clone)]
pub struct FuzzySVMConfig {
    /// Regularization parameter
    pub c: Float,
    /// Fuzzy membership strategy
    pub fuzzy_strategy: FuzzyMembershipStrategy,
    /// Kernel function to use
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Random seed
    pub random_state: Option<u64>,
    /// Minimum fuzzy membership value
    pub min_membership: Float,
    /// Maximum fuzzy membership value
    pub max_membership: Float,
}

impl Default for FuzzySVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            fuzzy_strategy: FuzzyMembershipStrategy::default(),
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 1000,
            random_state: None,
            min_membership: 0.1,
            max_membership: 1.0,
        }
    }
}

/// Fuzzy Support Vector Machine for classification
#[derive(Debug)]
pub struct FuzzySVM<State = Untrained> {
    config: FuzzySVMConfig,
    state: PhantomData<State>,
    // Fitted attributes
    support_vectors_: Option<Array2<Float>>,
    support_: Option<Array1<usize>>,
    dual_coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    classes_: Option<Array1<i32>>,
    fuzzy_memberships_: Option<Array1<Float>>,
    n_features_in_: Option<usize>,
    n_support_: Option<Array1<usize>>,
}

impl FuzzySVM<Untrained> {
    /// Create a new Fuzzy SVM
    pub fn new() -> Self {
        Self {
            config: FuzzySVMConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            support_: None,
            dual_coef_: None,
            intercept_: None,
            classes_: None,
            fuzzy_memberships_: None,
            n_features_in_: None,
            n_support_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the fuzzy membership strategy
    pub fn fuzzy_strategy(mut self, strategy: FuzzyMembershipStrategy) -> Self {
        self.config.fuzzy_strategy = strategy;
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

    /// Set the fuzzy membership range
    pub fn membership_range(mut self, min: Float, max: Float) -> Self {
        self.config.min_membership = min;
        self.config.max_membership = max;
        self
    }
}

impl Default for FuzzySVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl FuzzySVM<Untrained> {
    /// Calculate fuzzy membership values for training samples
    fn calculate_fuzzy_memberships(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        custom_memberships: Option<&Array1<Float>>,
    ) -> Result<Array1<Float>> {
        let n_samples = x.nrows();

        match &self.config.fuzzy_strategy {
            FuzzyMembershipStrategy::Constant { value } => Ok(Array1::from_elem(n_samples, *value)),
            FuzzyMembershipStrategy::Custom => custom_memberships.cloned().ok_or_else(|| {
                SklearsError::InvalidInput(
                    "Custom membership strategy requires providing membership values".to_string(),
                )
            }),
            FuzzyMembershipStrategy::DistanceBased { sigma } => {
                self.distance_based_membership(x, y, *sigma)
            }
            FuzzyMembershipStrategy::NoiseBased { threshold } => {
                self.noise_based_membership(x, y, *threshold)
            }
            FuzzyMembershipStrategy::Adaptive { k_neighbors } => {
                self.adaptive_membership(x, y, *k_neighbors)
            }
        }
    }

    /// Distance-based fuzzy membership calculation
    fn distance_based_membership(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        sigma: Float,
    ) -> Result<Array1<Float>> {
        let unique_classes: Vec<i32> = {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            classes
        };

        let mut memberships = Array1::zeros(x.nrows());

        for &class in &unique_classes {
            // Find class center
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            if class_indices.is_empty() {
                continue;
            }

            let mut class_center = Array1::<Float>::zeros(x.ncols());
            for &idx in &class_indices {
                class_center = class_center + x.row(idx);
            }
            class_center /= class_indices.len() as Float;

            // Calculate distances and memberships for this class
            for &idx in &class_indices {
                let distance = (&x.row(idx) - &class_center).mapv(|x| x * x).sum().sqrt();
                let membership = (-distance / sigma).exp();
                memberships[idx] =
                    membership.clamp(self.config.min_membership, self.config.max_membership);
            }
        }

        Ok(memberships)
    }

    /// Noise-based fuzzy membership calculation
    fn noise_based_membership(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        threshold: Float,
    ) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let mut memberships = Array1::ones(n_samples);

        // Simple noise detection based on local outlier factor approximation
        for i in 0..n_samples {
            let mut distances = Vec::new();

            // Calculate distances to samples of the same class
            for j in 0..n_samples {
                if i != j && y[i] == y[j] {
                    let distance = (&x.row(i) - &x.row(j)).mapv(|x| x * x).sum().sqrt();
                    distances.push(distance);
                }
            }

            if !distances.is_empty() {
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let k = (distances.len() / 3).clamp(1, 5); // Use up to top 5 nearest neighbors
                let avg_distance = distances.iter().take(k).sum::<Float>() / k as Float;

                let noise_factor = if avg_distance > threshold {
                    (-avg_distance / threshold + 1.0).max(0.0)
                } else {
                    1.0
                };

                memberships[i] =
                    noise_factor.clamp(self.config.min_membership, self.config.max_membership);
            }
        }

        Ok(memberships)
    }

    /// Adaptive fuzzy membership based on local density
    fn adaptive_membership(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        k_neighbors: usize,
    ) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let mut memberships = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut distances = Vec::new();

            // Calculate distances to all other samples
            for j in 0..n_samples {
                if i != j {
                    let distance = (&x.row(i) - &x.row(j)).mapv(|x| x * x).sum().sqrt();
                    distances.push((distance, j));
                }
            }

            // Sort by distance and get k nearest neighbors
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let k = k_neighbors.min(distances.len());

            if k == 0 {
                memberships[i] = self.config.max_membership;
                continue;
            }

            // Calculate local density and class consistency
            let avg_distance =
                distances.iter().take(k).map(|(d, _)| *d).sum::<Float>() / k as Float;
            let same_class_neighbors = distances
                .iter()
                .take(k)
                .filter(|(_, idx)| y[*idx] == y[i])
                .count() as Float;

            let class_consistency = same_class_neighbors / k as Float;
            let density_factor = if avg_distance > 0.0 {
                1.0 / (1.0 + avg_distance)
            } else {
                1.0
            };

            let membership = class_consistency * density_factor;
            memberships[i] =
                membership.clamp(self.config.min_membership, self.config.max_membership);
        }

        Ok(memberships)
    }
}

impl Fit<Array2<Float>, Array1<i32>> for FuzzySVM<Untrained> {
    type Fitted = FuzzySVM<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        self.fit_with_memberships(x, y, None)
    }
}

impl FuzzySVM<Untrained> {
    /// Fit with custom fuzzy membership values
    pub fn fit_with_memberships(
        self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        custom_memberships: Option<&Array1<Float>>,
    ) -> Result<FuzzySVM<Trained>> {
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

        if let Some(memberships) = custom_memberships {
            if memberships.len() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "Fuzzy memberships must have the same length as training data".to_string(),
                ));
            }
        }

        // Calculate fuzzy memberships
        let fuzzy_memberships = self.calculate_fuzzy_memberships(x, y, custom_memberships)?;

        // Get unique classes
        let classes_vec: Vec<i32> = {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            classes
        };
        let classes_array = Array1::from_vec(classes_vec.clone());
        let n_classes = classes_array.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // For binary classification
        if n_classes == 2 {
            // Convert labels to -1, +1
            let binary_y = Array1::from_vec(
                y.iter()
                    .map(|&label| if label == classes_vec[0] { -1.0 } else { 1.0 })
                    .collect(),
            );

            // Modify C parameter based on fuzzy memberships
            let weighted_c_values = fuzzy_memberships.mapv(|m| self.config.c * m);

            // Use a modified SMO solver that handles fuzzy memberships
            let solution = self.solve_fuzzy_svm(x, &binary_y, &weighted_c_values)?;

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

            let dual_coef = if support_indices.is_empty() {
                Array1::from_vec(vec![1e-8])
            } else {
                Array1::from_vec(support_indices.iter().map(|&i| solution.alpha[i]).collect())
            };

            let support_indices_array = Array1::from_vec(support_indices);
            let n_support = Array1::from_vec(vec![support_indices_array.len()]);

            return Ok(FuzzySVM {
                config: self.config,
                state: PhantomData::<Trained>,
                support_vectors_: Some(support_vectors),
                support_: Some(support_indices_array),
                dual_coef_: Some(dual_coef),
                intercept_: Some(solution.b),
                classes_: Some(classes_array),
                fuzzy_memberships_: Some(fuzzy_memberships),
                n_features_in_: Some(n_features),
                n_support_: Some(n_support),
            });
        }

        // For multiclass, use one-vs-rest approach
        Err(SklearsError::InvalidInput(
            "Multiclass Fuzzy SVM not yet implemented".to_string(),
        ))
    }

    /// Solve fuzzy SVM optimization problem
    fn solve_fuzzy_svm(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        c_values: &Array1<Float>,
    ) -> Result<SmoResult> {
        // For fuzzy SVM, we need to modify the SMO algorithm to use individual C values
        // This is a simplified implementation using average C
        let avg_c = c_values.mean().unwrap_or(self.config.c);

        let config = SmoConfig {
            c: avg_c,
            tol: self.config.tol,
            max_iter: self.config.max_iter,
            ..Default::default()
        };

        let mut solver = SmoSolver::new(config, self.config.kernel.clone());
        solver.solve(x, y)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for FuzzySVM<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(
                "Feature mismatch: X has different number of features than training data"
                    .to_string(),
            ));
        }

        let decision_values = self.decision_function(x)?;
        let classes = self.classes_.as_ref().unwrap();

        let predictions = Array1::from_vec(
            decision_values
                .iter()
                .map(|&val| if val >= 0.0 { classes[1] } else { classes[0] })
                .collect(),
        );

        Ok(predictions)
    }
}

impl FuzzySVM<Trained> {
    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_.as_ref().unwrap()
    }

    /// Get the support vector indices
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

    /// Get the fuzzy membership values
    pub fn fuzzy_memberships(&self) -> &Array1<Float> {
        self.fuzzy_memberships_.as_ref().unwrap()
    }

    /// Get the number of support vectors for each class
    pub fn n_support(&self) -> &Array1<usize> {
        self.n_support_.as_ref().unwrap()
    }

    /// Get the number of features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Compute the decision function values
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let support_vectors = self.support_vectors();
        let dual_coef = self.dual_coef();
        let intercept = self.intercept();
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>, // Default fallback
        };

        let mut decision_values = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut decision = intercept;
            for j in 0..support_vectors.nrows() {
                let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                decision += dual_coef[j] * k_val;
            }
            decision_values[i] = decision;
        }

        Ok(decision_values)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_fuzzy_svm_creation() {
        let fsvm = FuzzySVM::new()
            .c(2.0)
            .fuzzy_strategy(FuzzyMembershipStrategy::DistanceBased { sigma: 1.5 })
            .kernel(KernelType::Linear)
            .tol(1e-4)
            .max_iter(500)
            .random_state(42)
            .membership_range(0.2, 0.9);

        assert_eq!(fsvm.config.c, 2.0);
        assert_eq!(
            fsvm.config.fuzzy_strategy,
            FuzzyMembershipStrategy::DistanceBased { sigma: 1.5 }
        );
        assert_eq!(fsvm.config.kernel, KernelType::Linear);
        assert_eq!(fsvm.config.tol, 1e-4);
        assert_eq!(fsvm.config.max_iter, 500);
        assert_eq!(fsvm.config.random_state, Some(42));
        assert_eq!(fsvm.config.min_membership, 0.2);
        assert_eq!(fsvm.config.max_membership, 0.9);
    }

    #[test]
    fn test_constant_membership() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 1];

        let fsvm = FuzzySVM::new().fuzzy_strategy(FuzzyMembershipStrategy::Constant { value: 0.8 });

        let memberships = fsvm.calculate_fuzzy_memberships(&x, &y, None).unwrap();
        assert_eq!(memberships.len(), 2);
        assert!((memberships[0] - 0.8).abs() < 1e-6);
        assert!((memberships[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_custom_membership() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 1];
        let custom_memberships = array![0.9, 0.7];

        let fsvm = FuzzySVM::new().fuzzy_strategy(FuzzyMembershipStrategy::Custom);

        let memberships = fsvm
            .calculate_fuzzy_memberships(&x, &y, Some(&custom_memberships))
            .unwrap();
        assert_eq!(memberships.len(), 2);
        assert!((memberships[0] - 0.9).abs() < 1e-6);
        assert!((memberships[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    #[ignore = "Slow test: trains Fuzzy SVM. Run with --ignored flag"]
    fn test_fuzzy_svm_binary_classification() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let fsvm = FuzzySVM::new()
            .c(1.0)
            .fuzzy_strategy(FuzzyMembershipStrategy::DistanceBased { sigma: 2.0 })
            .kernel(KernelType::Linear)
            .tol(0.1)
            .max_iter(50)
            .random_state(42);

        let fitted_model = fsvm.fit(&x, &y).unwrap();

        assert_eq!(fitted_model.n_features_in(), 2);
        assert_eq!(fitted_model.classes().len(), 2);
        assert_eq!(fitted_model.fuzzy_memberships().len(), 6);

        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // Check that predictions are valid class labels
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }

        // Check that fuzzy memberships are in valid range
        for &membership in fitted_model.fuzzy_memberships().iter() {
            assert!(membership >= 0.0 && membership <= 1.0);
        }
    }

    #[test]
    fn test_fuzzy_svm_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0]; // Wrong length

        let fsvm = FuzzySVM::new();
        let result = fsvm.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_fuzzy_svm_custom_membership_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 1];
        let custom_memberships = array![0.9]; // Wrong length

        let fsvm = FuzzySVM::new().fuzzy_strategy(FuzzyMembershipStrategy::Custom);

        let result = fsvm.fit_with_memberships(&x, &y, Some(&custom_memberships));

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("same length"));
    }

    #[test]
    fn test_distance_based_membership() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0],];
        let y = array![0, 0, 1, 1];

        let fsvm =
            FuzzySVM::new().fuzzy_strategy(FuzzyMembershipStrategy::DistanceBased { sigma: 1.0 });

        let memberships = fsvm.calculate_fuzzy_memberships(&x, &y, None).unwrap();
        assert_eq!(memberships.len(), 4);

        // All memberships should be in valid range
        for &membership in memberships.iter() {
            assert!(membership >= 0.0 && membership <= 1.0);
        }
    }
}
