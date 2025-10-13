//! Semi-supervised Support Vector Machines
//!
//! This module implements various semi-supervised SVM algorithms that leverage both
//! labeled and unlabeled data to improve classification performance.
//!
//! Algorithms included:
//! - Transductive SVM (TSVM): Extends SVM to use unlabeled data by finding decision
//!   boundaries that pass through low-density regions
//! - Self-Training SVM: Iteratively trains SVM on labeled data, then uses confident
//!   predictions on unlabeled data to expand the training set
//! - Co-Training SVM: Uses two different views of the data to train complementary
//!   classifiers that teach each other

// TODO: Replace with scirs2-linalg
// use nalgebra::{DMatrix, DVector};
#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{thread_rng, SeedableRng};

use crate::kernels::{Kernel, KernelType};
use crate::svc::{SvcKernel, SVC};
use scirs2_core::Rng;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict, Trained};

/// Configuration for semi-supervised SVM algorithms
#[derive(Debug, Clone)]
pub struct SemiSupervisedConfig {
    /// Regularization parameter for supervised loss
    pub c_supervised: f64,
    /// Regularization parameter for unsupervised loss
    pub c_unsupervised: f64,
    /// Kernel type to use
    pub kernel: KernelType,
    /// Tolerance for optimization convergence
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Confidence threshold for self-training
    pub confidence_threshold: f64,
    /// Number of iterations for iterative algorithms
    pub n_iterations: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for SemiSupervisedConfig {
    fn default() -> Self {
        Self {
            c_supervised: 1.0,
            c_unsupervised: 0.1,
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 1000,
            confidence_threshold: 0.9,
            n_iterations: 10,
            random_state: Some(42),
        }
    }
}

/// Result of semi-supervised SVM training
#[derive(Debug, Clone)]
pub struct SemiSupervisedResult {
    /// Final support vectors
    pub support_vectors: DMatrix<f64>,
    /// Dual coefficients
    pub dual_coef: DVector<f64>,
    /// Intercept term
    pub intercept: f64,
    /// Indices of support vectors
    pub support_indices: Vec<usize>,
    /// Predicted labels for unlabeled data
    pub unlabeled_predictions: Vec<f64>,
    /// Confidence scores for unlabeled predictions
    pub confidence_scores: Vec<f64>,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Final objective value
    pub objective_value: f64,
}

/// Transductive Support Vector Machine (TSVM)
///
/// TSVM extends the standard SVM to use unlabeled data by finding decision boundaries
/// that pass through low-density regions of the unlabeled data distribution.
///
/// The algorithm alternates between:
/// 1. Solving the SVM optimization problem with current pseudo-labels
/// 2. Updating pseudo-labels for unlabeled data to maintain low density
///
/// Reference: Joachims, T. (1999). Transductive inference for text classification
/// using support vector machines.
#[derive(Debug, Clone)]
pub struct TransductiveSVM {
    config: SemiSupervisedConfig,
    kernel: Option<KernelType>,
    is_fitted: bool,
}

impl TransductiveSVM {
    /// Create a new TransductiveSVM
    pub fn new(config: SemiSupervisedConfig) -> Self {
        Self {
            config,
            kernel: None,
            is_fitted: false,
        }
    }

    /// Create a new TransductiveSVM with default configuration
    pub fn default() -> Self {
        Self::new(SemiSupervisedConfig::default())
    }

    /// Fit the TSVM model using labeled and unlabeled data
    pub fn fit(
        &mut self,
        x_labeled: &DMatrix<f64>,
        y_labeled: &DVector<f64>,
        x_unlabeled: &DMatrix<f64>,
    ) -> Result<SemiSupervisedResult> {
        // Validate inputs
        if x_labeled.nrows() != y_labeled.len() {
            return Err(SklearsError::InvalidInput(
                "Number of labeled samples must match number of labels".to_string(),
            ));
        }

        if x_labeled.ncols() != x_unlabeled.ncols() {
            return Err(SklearsError::InvalidInput(
                "Labeled and unlabeled data must have same number of features".to_string(),
            ));
        }

        // Initialize kernel
        let kernel = self.config.kernel.clone();
        self.kernel = Some(kernel);

        // Combine labeled and unlabeled data
        let n_labeled = x_labeled.nrows();
        let n_unlabeled = x_unlabeled.nrows();
        let n_total = n_labeled + n_unlabeled;

        let mut x_combined = DMatrix::zeros(n_total, x_labeled.ncols());
        x_combined.rows_mut(0, n_labeled).copy_from(x_labeled);
        x_combined
            .rows_mut(n_labeled, n_unlabeled)
            .copy_from(x_unlabeled);

        // Initialize pseudo-labels for unlabeled data
        let mut rng = StdRng::from_rng(&mut thread_rng());

        let mut y_pseudo = DVector::zeros(n_total);
        y_pseudo.rows_mut(0, n_labeled).copy_from(y_labeled);

        // Random initialization of pseudo-labels
        for i in n_labeled..n_total {
            y_pseudo[i] = if rng.random::<f64>() > 0.5 { 1.0 } else { -1.0 };
        }

        let mut best_objective = f64::INFINITY;
        let mut best_result = None;

        // TSVM alternating optimization
        for iteration in 0..self.config.n_iterations {
            // Step 1: Solve SVM with current pseudo-labels
            let kernel = match &self.config.kernel {
                KernelType::Linear => SvcKernel::Linear,
                KernelType::Rbf { gamma } => SvcKernel::Rbf {
                    gamma: Some(*gamma),
                },
                KernelType::Polynomial {
                    gamma,
                    degree,
                    coef0,
                } => SvcKernel::Poly {
                    degree: *degree as usize,
                    gamma: Some(*gamma), // Default gamma for polynomial kernel
                    coef0: *coef0,
                },
                _ => SvcKernel::Linear, // Default fallback
            };

            let svm = SVC::new()
                .c(self.config.c_supervised)
                .tol(self.config.tol)
                .max_iter(self.config.max_iter);

            // Convert nalgebra to ndarray for SVM
            let x_combined_ndarray = Array2::from_shape_vec(
                (x_combined.nrows(), x_combined.ncols()),
                x_combined.iter().cloned().collect(),
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
            let y_pseudo_ndarray = Array1::from_vec(y_pseudo.iter().cloned().collect());

            let svm_result = svm.fit(&x_combined_ndarray, &y_pseudo_ndarray)?;

            // Step 2: Update pseudo-labels for unlabeled data
            let decision_values = svm_result.decision_function(&x_combined_ndarray)?;

            // Calculate objective value
            let objective = self.calculate_objective(
                &x_combined,
                &y_pseudo,
                decision_values.as_slice().unwrap(),
                n_labeled,
            )?;

            // Update pseudo-labels for unlabeled data
            let mut changed = false;
            for i in n_labeled..n_total {
                let new_label = if decision_values[i] > 0.0 { 1.0 } else { -1.0 };
                if (y_pseudo[i] - new_label).abs() > 1e-10 {
                    y_pseudo[i] = new_label;
                    changed = true;
                }
            }

            // Check for convergence
            if !changed || objective < best_objective {
                best_objective = objective;

                // Calculate confidence scores
                let confidence_scores = decision_values
                    .iter()
                    .skip(n_labeled)
                    .map(|&val| val.abs())
                    .collect();

                // Convert ndarray back to nalgebra for SemiSupervisedResult
                let support_vectors_ndarray = svm_result.support_vectors();
                let dual_coef_ndarray = svm_result.dual_coef();

                let support_vectors_nalgebra = DMatrix::from_vec(
                    support_vectors_ndarray.nrows(),
                    support_vectors_ndarray.ncols(),
                    support_vectors_ndarray.iter().cloned().collect(),
                );
                let dual_coef_nalgebra =
                    DVector::from_vec(dual_coef_ndarray.iter().cloned().collect());

                best_result = Some(SemiSupervisedResult {
                    support_vectors: support_vectors_nalgebra,
                    dual_coef: dual_coef_nalgebra,
                    intercept: svm_result.intercept(),
                    support_indices: svm_result.support_indices().to_vec(),
                    unlabeled_predictions: y_pseudo
                        .rows(n_labeled, n_unlabeled)
                        .iter()
                        .cloned()
                        .collect(),
                    confidence_scores,
                    n_iterations: iteration + 1,
                    objective_value: objective,
                });

                if !changed {
                    break;
                }
            }
        }

        self.is_fitted = true;

        best_result.ok_or_else(|| {
            SklearsError::InvalidInput("TSVM optimization failed to converge".to_string())
        })
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &DMatrix<f64>, result: &SemiSupervisedResult) -> Result<DVector<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let decision_values = self.decision_function(x, result)?;
        Ok(DVector::from_vec(
            decision_values
                .iter()
                .map(|&val| if val > 0.0 { 1.0 } else { -1.0 })
                .collect(),
        ))
    }

    /// Calculate decision function values
    fn decision_function(
        &self,
        x: &DMatrix<f64>,
        result: &SemiSupervisedResult,
    ) -> Result<Vec<f64>> {
        let kernel = self.kernel.as_ref().unwrap();
        let mut decision_values = Vec::with_capacity(x.nrows());

        for i in 0..x.nrows() {
            let mut sum = 0.0;
            for (j, &support_idx) in result.support_indices.iter().enumerate() {
                // Convert nalgebra rows to ndarray for kernel computation
                let x_row_vec: Vec<f64> = x.row(i).iter().cloned().collect();
                let support_row_vec: Vec<f64> = result
                    .support_vectors
                    .row(support_idx)
                    .iter()
                    .cloned()
                    .collect();
                let x_row_ndarray = Array1::from_vec(x_row_vec);
                let support_row_ndarray = Array1::from_vec(support_row_vec);

                let kernel_val = kernel.compute(x_row_ndarray.view(), support_row_ndarray.view());
                sum += result.dual_coef[j] * kernel_val;
            }
            decision_values.push(sum + result.intercept);
        }

        Ok(decision_values)
    }

    /// Calculate the TSVM objective function
    fn calculate_objective(
        &self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        decision_values: &[f64],
        n_labeled: usize,
    ) -> Result<f64> {
        let mut objective = 0.0;

        // Supervised loss (hinge loss on labeled data)
        for i in 0..n_labeled {
            let margin = y[i] * decision_values[i];
            if margin < 1.0 {
                objective += self.config.c_supervised * (1.0 - margin);
            }
        }

        // Unsupervised loss (encourage low-density separation)
        for i in n_labeled..x.nrows() {
            let margin = y[i] * decision_values[i];
            if margin < 1.0 {
                objective += self.config.c_unsupervised * (1.0 - margin);
            }
        }

        Ok(objective)
    }
}

/// Self-Training SVM
///
/// Self-training is a semi-supervised learning approach where a classifier is initially
/// trained on labeled data, then iteratively retrained by adding confidently predicted
/// unlabeled examples to the training set.
#[derive(Debug, Clone)]
pub struct SelfTrainingSVM {
    config: SemiSupervisedConfig,
    base_classifier: Option<SVC<Trained>>,
    is_fitted: bool,
}

impl SelfTrainingSVM {
    /// Create a new SelfTrainingSVM
    pub fn new(config: SemiSupervisedConfig) -> Self {
        Self {
            config,
            base_classifier: None,
            is_fitted: false,
        }
    }

    /// Create a new SelfTrainingSVM with default configuration
    pub fn default() -> Self {
        Self::new(SemiSupervisedConfig::default())
    }

    /// Fit the Self-Training SVM model
    pub fn fit(
        &mut self,
        x_labeled: &DMatrix<f64>,
        y_labeled: &DVector<f64>,
        x_unlabeled: &DMatrix<f64>,
    ) -> Result<SemiSupervisedResult> {
        // Validate inputs
        if x_labeled.nrows() != y_labeled.len() {
            return Err(SklearsError::InvalidInput(
                "Number of labeled samples must match number of labels".to_string(),
            ));
        }

        if x_labeled.ncols() != x_unlabeled.ncols() {
            return Err(SklearsError::InvalidInput(
                "Labeled and unlabeled data must have same number of features".to_string(),
            ));
        }

        // Initialize training data
        let mut x_train = x_labeled.clone();
        let mut y_train = y_labeled.clone();
        let mut x_unlabeled_remaining = x_unlabeled.clone();
        let mut unlabeled_indices: Vec<usize> = (0..x_unlabeled.nrows()).collect();

        let mut iteration = 0;

        while iteration < self.config.n_iterations && !x_unlabeled_remaining.is_empty() {
            // Train SVM on current labeled data
            let kernel = match &self.config.kernel {
                KernelType::Linear => SvcKernel::Linear,
                KernelType::Rbf { gamma } => SvcKernel::Rbf {
                    gamma: Some(*gamma),
                },
                KernelType::Polynomial {
                    gamma,
                    degree,
                    coef0,
                } => SvcKernel::Poly {
                    degree: *degree as usize,
                    gamma: Some(*gamma),
                    coef0: *coef0,
                },
                _ => SvcKernel::Linear,
            };

            let svm = SVC::new()
                .c(self.config.c_supervised)
                .tol(self.config.tol)
                .max_iter(self.config.max_iter);

            // Convert nalgebra to ndarray for SVM
            let x_train_ndarray = Array2::from_shape_vec(
                (x_train.nrows(), x_train.ncols()),
                x_train.iter().cloned().collect(),
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
            let y_train_ndarray = Array1::from_vec(y_train.iter().cloned().collect());

            let svm_result = svm.fit(&x_train_ndarray, &y_train_ndarray)?;

            // Predict on unlabeled data
            let x_unlabeled_ndarray = Array2::from_shape_vec(
                (x_unlabeled_remaining.nrows(), x_unlabeled_remaining.ncols()),
                x_unlabeled_remaining.iter().cloned().collect(),
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;

            let predictions = svm_result.predict(&x_unlabeled_ndarray)?;
            let decision_values = svm_result.decision_function(&x_unlabeled_ndarray)?;

            // Calculate confidence scores
            let confidence_scores: Vec<f64> =
                decision_values.iter().map(|&val| val.abs()).collect();

            // Find most confident predictions
            let mut confident_indices = Vec::new();
            for (i, &confidence) in confidence_scores.iter().enumerate() {
                if confidence > self.config.confidence_threshold {
                    confident_indices.push(i);
                }
            }

            if confident_indices.is_empty() {
                break; // No confident predictions
            }

            // Add confident predictions to training set
            if !confident_indices.is_empty() {
                // Collect the confident samples
                let mut confident_x = DMatrix::zeros(confident_indices.len(), x_train.ncols());
                let mut confident_y = DVector::zeros(confident_indices.len());

                for (i, &idx) in confident_indices.iter().enumerate() {
                    confident_x
                        .row_mut(i)
                        .copy_from(&x_unlabeled_remaining.row(idx));
                    confident_y[i] = predictions[idx];
                }

                // Concatenate with existing training data
                let mut new_x_train =
                    DMatrix::zeros(x_train.nrows() + confident_indices.len(), x_train.ncols());
                new_x_train.rows_mut(0, x_train.nrows()).copy_from(&x_train);
                new_x_train
                    .rows_mut(x_train.nrows(), confident_indices.len())
                    .copy_from(&confident_x);

                let mut new_y_train = DVector::zeros(y_train.len() + confident_indices.len());
                new_y_train.rows_mut(0, y_train.len()).copy_from(&y_train);
                new_y_train
                    .rows_mut(y_train.len(), confident_indices.len())
                    .copy_from(&confident_y);

                x_train = new_x_train;
                y_train = new_y_train;
            }

            // Remove confident samples from unlabeled set
            confident_indices.sort_by(|a, b| b.cmp(a)); // Sort in descending order
            for &idx in &confident_indices {
                x_unlabeled_remaining = x_unlabeled_remaining.remove_row(idx);
                unlabeled_indices.remove(idx);
            }

            iteration += 1;
        }

        // Final training on all labeled data
        let kernel = match &self.config.kernel {
            KernelType::Linear => SvcKernel::Linear,
            KernelType::Rbf { gamma } => SvcKernel::Rbf {
                gamma: Some(*gamma),
            },
            KernelType::Polynomial {
                gamma,
                degree,
                coef0,
            } => SvcKernel::Poly {
                degree: *degree as usize,
                gamma: Some(*gamma),
                coef0: *coef0,
            },
            _ => SvcKernel::Linear,
        };

        let final_svm = SVC::new()
            .c(self.config.c_supervised)
            .tol(self.config.tol)
            .max_iter(self.config.max_iter);

        // Convert nalgebra to ndarray for final SVM training
        let x_train_final_ndarray = Array2::from_shape_vec(
            (x_train.nrows(), x_train.ncols()),
            x_train.iter().cloned().collect(),
        )
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}")))?;
        let y_train_final_ndarray = Array1::from_vec(y_train.iter().cloned().collect());

        let final_result = final_svm.fit(&x_train_final_ndarray, &y_train_final_ndarray)?;
        self.base_classifier = Some(final_result);
        self.is_fitted = true;

        // Predict on all unlabeled data
        let x_unlabeled_ndarray = Array2::from_shape_vec(
            (x_unlabeled.nrows(), x_unlabeled.ncols()),
            x_unlabeled.iter().cloned().collect(),
        )
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}")))?;

        let final_predictions = self
            .base_classifier
            .as_ref()
            .unwrap()
            .predict(&x_unlabeled_ndarray)?;
        let final_decision_values = self
            .base_classifier
            .as_ref()
            .unwrap()
            .decision_function(&x_unlabeled_ndarray)?;
        let final_confidence_scores: Vec<f64> =
            final_decision_values.iter().map(|&val| val.abs()).collect();

        // Convert ndarray back to nalgebra for result
        let classifier = self.base_classifier.as_ref().unwrap();
        let support_vectors_ndarray = classifier.support_vectors();
        let dual_coef_ndarray = classifier.dual_coef();

        let support_vectors_nalgebra = DMatrix::from_vec(
            support_vectors_ndarray.nrows(),
            support_vectors_ndarray.ncols(),
            support_vectors_ndarray.iter().cloned().collect(),
        );
        let dual_coef_nalgebra = DVector::from_vec(dual_coef_ndarray.iter().cloned().collect());

        Ok(SemiSupervisedResult {
            support_vectors: support_vectors_nalgebra,
            dual_coef: dual_coef_nalgebra,
            intercept: classifier.intercept(),
            support_indices: classifier.support_indices().to_vec(),
            unlabeled_predictions: final_predictions.iter().cloned().collect(),
            confidence_scores: final_confidence_scores,
            n_iterations: iteration,
            objective_value: 0.0, // Not applicable for self-training
        })
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &DMatrix<f64>) -> Result<DVector<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        // Convert nalgebra to ndarray for prediction
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;

        let predictions_ndarray = self.base_classifier.as_ref().unwrap().predict(&x_ndarray)?;

        // Convert ndarray back to nalgebra
        let predictions_nalgebra = DVector::from_vec(predictions_ndarray.iter().cloned().collect());

        Ok(predictions_nalgebra)
    }

    /// Calculate decision function values
    pub fn decision_function(&self, x: &DMatrix<f64>) -> Result<DVector<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        // Convert nalgebra to ndarray for decision function
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;

        let decision_values_ndarray = self
            .base_classifier
            .as_ref()
            .unwrap()
            .decision_function(&x_ndarray)?;

        // Convert ndarray back to nalgebra
        let decision_values_nalgebra =
            DVector::from_vec(decision_values_ndarray.iter().cloned().collect());

        Ok(decision_values_nalgebra)
    }
}

/// Co-Training SVM
///
/// Co-training uses two different views of the data to train complementary classifiers
/// that teach each other by labeling examples for each other.
#[derive(Debug, Clone)]
pub struct CoTrainingSVM {
    config: SemiSupervisedConfig,
    classifier1: Option<SVC<Trained>>,
    classifier2: Option<SVC<Trained>>,
    feature_split: usize,
    is_fitted: bool,
}

impl CoTrainingSVM {
    /// Create a new CoTrainingSVM
    pub fn new(config: SemiSupervisedConfig, feature_split: usize) -> Self {
        Self {
            config,
            classifier1: None,
            classifier2: None,
            feature_split,
            is_fitted: false,
        }
    }

    /// Fit the Co-Training SVM model
    pub fn fit(
        &mut self,
        x_labeled: &DMatrix<f64>,
        y_labeled: &DVector<f64>,
        x_unlabeled: &DMatrix<f64>,
    ) -> Result<SemiSupervisedResult> {
        // Validate inputs
        if x_labeled.nrows() != y_labeled.len() {
            return Err(SklearsError::InvalidInput(
                "Number of labeled samples must match number of labels".to_string(),
            ));
        }

        if x_labeled.ncols() != x_unlabeled.ncols() {
            return Err(SklearsError::InvalidInput(
                "Labeled and unlabeled data must have same number of features".to_string(),
            ));
        }

        if self.feature_split >= x_labeled.ncols() {
            return Err(SklearsError::InvalidInput(
                "Feature split must be less than number of features".to_string(),
            ));
        }

        // Split features into two views
        let x_labeled_view1 = x_labeled.columns(0, self.feature_split).into_owned();
        let x_labeled_view2 = x_labeled
            .columns(self.feature_split, x_labeled.ncols() - self.feature_split)
            .into_owned();

        let x_unlabeled_view1 = x_unlabeled.columns(0, self.feature_split).into_owned();
        let x_unlabeled_view2 = x_unlabeled
            .columns(self.feature_split, x_unlabeled.ncols() - self.feature_split)
            .into_owned();

        // Initialize training data
        let mut x_train1 = x_labeled_view1.clone();
        let mut y_train1 = y_labeled.clone();
        let mut x_train2 = x_labeled_view2.clone();
        let mut y_train2 = y_labeled.clone();

        let mut x_unlabeled_remaining1 = x_unlabeled_view1.clone();
        let mut x_unlabeled_remaining2 = x_unlabeled_view2.clone();

        let mut iteration = 0;

        while iteration < self.config.n_iterations && !x_unlabeled_remaining1.is_empty() {
            // Train both classifiers
            let kernel = match &self.config.kernel {
                KernelType::Linear => SvcKernel::Linear,
                KernelType::Rbf { gamma } => SvcKernel::Rbf {
                    gamma: Some(*gamma),
                },
                KernelType::Polynomial {
                    gamma,
                    degree,
                    coef0,
                } => SvcKernel::Poly {
                    degree: *degree as usize,
                    gamma: Some(*gamma),
                    coef0: *coef0,
                },
                _ => SvcKernel::Linear,
            };

            let svm1 = SVC::new()
                .c(self.config.c_supervised)
                .tol(self.config.tol)
                .max_iter(self.config.max_iter);

            let svm2 = SVC::new()
                .c(self.config.c_supervised)
                .tol(self.config.tol)
                .max_iter(self.config.max_iter);

            // Convert nalgebra to ndarray for SVM training
            let x_train1_ndarray = Array2::from_shape_vec(
                (x_train1.nrows(), x_train1.ncols()),
                x_train1.iter().cloned().collect(),
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
            let y_train1_ndarray = Array1::from_vec(y_train1.iter().cloned().collect());

            let x_train2_ndarray = Array2::from_shape_vec(
                (x_train2.nrows(), x_train2.ncols()),
                x_train2.iter().cloned().collect(),
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
            let y_train2_ndarray = Array1::from_vec(y_train2.iter().cloned().collect());

            let svm1_result = svm1.fit(&x_train1_ndarray, &y_train1_ndarray)?;
            let svm2_result = svm2.fit(&x_train2_ndarray, &y_train2_ndarray)?;

            // Get predictions from both classifiers
            let x_unlabeled1_ndarray = Array2::from_shape_vec(
                (
                    x_unlabeled_remaining1.nrows(),
                    x_unlabeled_remaining1.ncols(),
                ),
                x_unlabeled_remaining1.iter().cloned().collect(),
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;
            let x_unlabeled2_ndarray = Array2::from_shape_vec(
                (
                    x_unlabeled_remaining2.nrows(),
                    x_unlabeled_remaining2.ncols(),
                ),
                x_unlabeled_remaining2.iter().cloned().collect(),
            )
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;

            let predictions1 = svm1_result.predict(&x_unlabeled1_ndarray)?;
            let predictions2 = svm2_result.predict(&x_unlabeled2_ndarray)?;

            let decision_values1 = svm1_result.decision_function(&x_unlabeled1_ndarray)?;
            let decision_values2 = svm2_result.decision_function(&x_unlabeled2_ndarray)?;

            // Calculate confidence scores
            let confidence_scores1: Vec<f64> =
                decision_values1.iter().map(|&val| val.abs()).collect();
            let confidence_scores2: Vec<f64> =
                decision_values2.iter().map(|&val| val.abs()).collect();

            // Find confident predictions from both classifiers
            let mut confident_indices1 = Vec::new();
            let mut confident_indices2 = Vec::new();

            for (i, &confidence) in confidence_scores1.iter().enumerate() {
                if confidence > self.config.confidence_threshold {
                    confident_indices1.push(i);
                }
            }

            for (i, &confidence) in confidence_scores2.iter().enumerate() {
                if confidence > self.config.confidence_threshold {
                    confident_indices2.push(i);
                }
            }

            if confident_indices1.is_empty() && confident_indices2.is_empty() {
                break; // No confident predictions
            }

            // Add confident predictions from classifier 1 to training set of classifier 2
            if !confident_indices1.is_empty() {
                let mut confident_x2 = DMatrix::zeros(confident_indices1.len(), x_train2.ncols());
                let mut confident_y2 = DVector::zeros(confident_indices1.len());

                for (i, &idx) in confident_indices1.iter().enumerate() {
                    confident_x2
                        .row_mut(i)
                        .copy_from(&x_unlabeled_remaining2.row(idx));
                    confident_y2[i] = predictions1[idx];
                }

                let mut new_x_train2 = DMatrix::zeros(
                    x_train2.nrows() + confident_indices1.len(),
                    x_train2.ncols(),
                );
                new_x_train2
                    .rows_mut(0, x_train2.nrows())
                    .copy_from(&x_train2);
                new_x_train2
                    .rows_mut(x_train2.nrows(), confident_indices1.len())
                    .copy_from(&confident_x2);

                let mut new_y_train2 = DVector::zeros(y_train2.len() + confident_indices1.len());
                new_y_train2
                    .rows_mut(0, y_train2.len())
                    .copy_from(&y_train2);
                new_y_train2
                    .rows_mut(y_train2.len(), confident_indices1.len())
                    .copy_from(&confident_y2);

                x_train2 = new_x_train2;
                y_train2 = new_y_train2;
            }

            // Add confident predictions from classifier 2 to training set of classifier 1
            if !confident_indices2.is_empty() {
                let mut confident_x1 = DMatrix::zeros(confident_indices2.len(), x_train1.ncols());
                let mut confident_y1 = DVector::zeros(confident_indices2.len());

                for (i, &idx) in confident_indices2.iter().enumerate() {
                    confident_x1
                        .row_mut(i)
                        .copy_from(&x_unlabeled_remaining1.row(idx));
                    confident_y1[i] = predictions2[idx];
                }

                let mut new_x_train1 = DMatrix::zeros(
                    x_train1.nrows() + confident_indices2.len(),
                    x_train1.ncols(),
                );
                new_x_train1
                    .rows_mut(0, x_train1.nrows())
                    .copy_from(&x_train1);
                new_x_train1
                    .rows_mut(x_train1.nrows(), confident_indices2.len())
                    .copy_from(&confident_x1);

                let mut new_y_train1 = DVector::zeros(y_train1.len() + confident_indices2.len());
                new_y_train1
                    .rows_mut(0, y_train1.len())
                    .copy_from(&y_train1);
                new_y_train1
                    .rows_mut(y_train1.len(), confident_indices2.len())
                    .copy_from(&confident_y1);

                x_train1 = new_x_train1;
                y_train1 = new_y_train1;
            }

            // Remove confident samples from unlabeled set
            let mut all_confident_indices = confident_indices1;
            all_confident_indices.extend(confident_indices2);
            all_confident_indices.sort_by(|a, b| b.cmp(a)); // Sort in descending order
            all_confident_indices.dedup();

            for &idx in &all_confident_indices {
                if idx < x_unlabeled_remaining1.nrows() {
                    x_unlabeled_remaining1 = x_unlabeled_remaining1.remove_row(idx);
                    x_unlabeled_remaining2 = x_unlabeled_remaining2.remove_row(idx);
                }
            }

            iteration += 1;
        }

        // Final training on combined views
        let mut x_train_combined = DMatrix::zeros(x_train1.nrows(), x_labeled.ncols());
        x_train_combined
            .columns_mut(0, self.feature_split)
            .copy_from(&x_train1);
        x_train_combined
            .columns_mut(self.feature_split, x_labeled.ncols() - self.feature_split)
            .copy_from(&x_train2);

        let kernel = match &self.config.kernel {
            KernelType::Linear => SvcKernel::Linear,
            KernelType::Rbf { gamma } => SvcKernel::Rbf {
                gamma: Some(*gamma),
            },
            KernelType::Polynomial {
                gamma,
                degree,
                coef0,
            } => SvcKernel::Poly {
                degree: *degree as usize,
                gamma: Some(*gamma),
                coef0: *coef0,
            },
            _ => SvcKernel::Linear,
        };

        let final_svm = SVC::new()
            .c(self.config.c_supervised)
            .tol(self.config.tol)
            .max_iter(self.config.max_iter);

        // Convert nalgebra to ndarray for final SVM training
        let x_train_combined_ndarray = Array2::from_shape_vec(
            (x_train_combined.nrows(), x_train_combined.ncols()),
            x_train_combined.iter().cloned().collect(),
        )
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}")))?;
        let y_train1_ndarray = Array1::from_vec(y_train1.iter().cloned().collect());

        let final_result = final_svm.fit(&x_train_combined_ndarray, &y_train1_ndarray)?;
        self.classifier1 = Some(final_result);
        self.is_fitted = true;

        // Predict on all unlabeled data
        let x_unlabeled_ndarray = Array2::from_shape_vec(
            (x_unlabeled.nrows(), x_unlabeled.ncols()),
            x_unlabeled.iter().cloned().collect(),
        )
        .map_err(|e| SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}")))?;

        let final_predictions = self
            .classifier1
            .as_ref()
            .unwrap()
            .predict(&x_unlabeled_ndarray)?;
        let final_decision_values = self
            .classifier1
            .as_ref()
            .unwrap()
            .decision_function(&x_unlabeled_ndarray)?;
        let final_confidence_scores: Vec<f64> =
            final_decision_values.iter().map(|&val| val.abs()).collect();

        // Convert ndarray back to nalgebra for result
        let classifier = self.classifier1.as_ref().unwrap();
        let support_vectors_ndarray = classifier.support_vectors();
        let dual_coef_ndarray = classifier.dual_coef();

        let support_vectors_nalgebra = DMatrix::from_vec(
            support_vectors_ndarray.nrows(),
            support_vectors_ndarray.ncols(),
            support_vectors_ndarray.iter().cloned().collect(),
        );
        let dual_coef_nalgebra = DVector::from_vec(dual_coef_ndarray.iter().cloned().collect());

        Ok(SemiSupervisedResult {
            support_vectors: support_vectors_nalgebra,
            dual_coef: dual_coef_nalgebra,
            intercept: classifier.intercept(),
            support_indices: classifier.support_indices().to_vec(),
            unlabeled_predictions: final_predictions.iter().cloned().collect(),
            confidence_scores: final_confidence_scores,
            n_iterations: iteration,
            objective_value: 0.0, // Not applicable for co-training
        })
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &DMatrix<f64>) -> Result<DVector<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        // Convert nalgebra to ndarray for prediction
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;

        let predictions_ndarray = self.classifier1.as_ref().unwrap().predict(&x_ndarray)?;

        // Convert ndarray back to nalgebra
        let predictions_nalgebra = DVector::from_vec(predictions_ndarray.iter().cloned().collect());

        Ok(predictions_nalgebra)
    }

    /// Calculate decision function values
    pub fn decision_function(&self, x: &DMatrix<f64>) -> Result<DVector<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        // Convert nalgebra to ndarray for decision function
        let x_ndarray = Array2::from_shape_vec((x.nrows(), x.ncols()), x.iter().cloned().collect())
            .map_err(|e| {
                SklearsError::InvalidInput(format!("Failed to convert to ndarray: {e}"))
            })?;

        let decision_values_ndarray = self
            .classifier1
            .as_ref()
            .unwrap()
            .decision_function(&x_ndarray)?;

        // Convert ndarray back to nalgebra
        let decision_values_nalgebra =
            DVector::from_vec(decision_values_ndarray.iter().cloned().collect());

        Ok(decision_values_nalgebra)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn create_test_data() -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
        let x_labeled = DMatrix::from_row_slice(4, 2, &[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0]);
        let y_labeled = DVector::from_vec(vec![1.0, 1.0, -1.0, -1.0]);
        let x_unlabeled = DMatrix::from_row_slice(4, 2, &[1.5, 2.5, 2.5, 3.5, 3.5, 2.5, 4.5, 3.5]);
        (x_labeled, y_labeled, x_unlabeled)
    }

    #[test]
    fn test_transductive_svm_basic() {
        let (x_labeled, y_labeled, x_unlabeled) = create_test_data();
        let mut tsvm = TransductiveSVM::default();
        let result = tsvm.fit(&x_labeled, &y_labeled, &x_unlabeled);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.unlabeled_predictions.len(), x_unlabeled.nrows());
        assert_eq!(result.confidence_scores.len(), x_unlabeled.nrows());
    }

    #[test]
    fn test_self_training_svm_basic() {
        let (x_labeled, y_labeled, x_unlabeled) = create_test_data();
        let mut stsvm = SelfTrainingSVM::default();
        let result = stsvm.fit(&x_labeled, &y_labeled, &x_unlabeled);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.unlabeled_predictions.len(), x_unlabeled.nrows());
        assert_eq!(result.confidence_scores.len(), x_unlabeled.nrows());
    }

    #[test]
    fn test_cotraining_svm_basic() {
        let (x_labeled, y_labeled, x_unlabeled) = create_test_data();
        let mut ctsvm = CoTrainingSVM::new(SemiSupervisedConfig::default(), 1);
        let result = ctsvm.fit(&x_labeled, &y_labeled, &x_unlabeled);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.unlabeled_predictions.len(), x_unlabeled.nrows());
        assert_eq!(result.confidence_scores.len(), x_unlabeled.nrows());
    }

    #[test]
    fn test_tsvm_prediction() {
        let (x_labeled, y_labeled, x_unlabeled) = create_test_data();
        let mut tsvm = TransductiveSVM::default();
        let result = tsvm.fit(&x_labeled, &y_labeled, &x_unlabeled).unwrap();

        let predictions = tsvm.predict(&x_unlabeled, &result);
        assert!(predictions.is_ok());

        let predictions = predictions.unwrap();
        assert_eq!(predictions.len(), x_unlabeled.nrows());
        assert!(predictions.iter().all(|&p| p == 1.0 || p == -1.0));
    }

    #[test]
    fn test_semi_supervised_config() {
        let config = SemiSupervisedConfig {
            c_supervised: 0.5,
            c_unsupervised: 0.05,
            kernel: KernelType::Linear,
            tol: 1e-4,
            max_iter: 500,
            confidence_threshold: 0.95,
            n_iterations: 5,
            random_state: Some(123),
        };

        let mut tsvm = TransductiveSVM::new(config.clone());
        assert_eq!(tsvm.config.c_supervised, 0.5);
        assert_eq!(tsvm.config.c_unsupervised, 0.05);
        assert_eq!(tsvm.config.confidence_threshold, 0.95);
    }
}
