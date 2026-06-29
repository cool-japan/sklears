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

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::{essentials::Uniform, seeded_rng};

use crate::kernels::{Kernel, KernelType};
use crate::svc::{SvcKernel, SVC};
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
    pub support_vectors: Array2<f64>,
    /// Dual coefficients
    pub dual_coef: Array1<f64>,
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

/// Translate a [`KernelType`] into the matching [`SvcKernel`] builder variant.
fn svc_kernel_for(kernel: &KernelType) -> SvcKernel {
    match kernel {
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
        KernelType::Sigmoid { gamma, coef0 } => SvcKernel::Sigmoid {
            gamma: Some(*gamma),
            coef0: *coef0,
        },
        other => SvcKernel::Custom(other.clone()),
    }
}

/// Vertically stack two row-major matrices with the same number of columns.
fn vstack(top: &Array2<f64>, bottom: &Array2<f64>) -> Array2<f64> {
    let n_cols = top.ncols().max(bottom.ncols());
    let mut out = Array2::zeros((top.nrows() + bottom.nrows(), n_cols));
    if top.nrows() > 0 {
        out.slice_mut(s![..top.nrows(), ..]).assign(top);
    }
    if bottom.nrows() > 0 {
        out.slice_mut(s![top.nrows().., ..]).assign(bottom);
    }
    out
}

/// Concatenate two vectors.
fn vconcat(head: &Array1<f64>, tail: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::zeros(head.len() + tail.len());
    if !head.is_empty() {
        out.slice_mut(s![..head.len()]).assign(head);
    }
    if !tail.is_empty() {
        out.slice_mut(s![head.len()..]).assign(tail);
    }
    out
}

/// Return a copy of `matrix` with the rows in `to_remove` deleted.
///
/// `to_remove` does not need to be sorted or unique.
fn remove_rows(matrix: &Array2<f64>, to_remove: &[usize]) -> Array2<f64> {
    let keep: Vec<usize> = (0..matrix.nrows())
        .filter(|i| !to_remove.contains(i))
        .collect();
    matrix.select(Axis(0), &keep)
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

    /// Fit the TSVM model using labeled and unlabeled data
    pub fn fit(
        &mut self,
        x_labeled: &Array2<f64>,
        y_labeled: &Array1<f64>,
        x_unlabeled: &Array2<f64>,
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
        self.kernel = Some(self.config.kernel.clone());

        // Combine labeled and unlabeled data
        let n_labeled = x_labeled.nrows();
        let n_unlabeled = x_unlabeled.nrows();
        let n_total = n_labeled + n_unlabeled;

        let x_combined = vstack(x_labeled, x_unlabeled);

        // Initialize pseudo-labels for unlabeled data. Use a deterministic seed
        // when configured so that fits are reproducible, otherwise draw a fresh
        // seed from the global RNG.
        let seed = self
            .config
            .random_state
            .unwrap_or_else(scirs2_core::random::random::<u64>);
        let mut rng = seeded_rng(seed);
        let unit = Uniform::new(0.0, 1.0)
            .map_err(|e| SklearsError::InvalidInput(format!("invalid distribution: {e}")))?;

        let mut y_pseudo = Array1::zeros(n_total);
        y_pseudo.slice_mut(s![..n_labeled]).assign(y_labeled);

        // Random initialization of pseudo-labels
        for i in n_labeled..n_total {
            y_pseudo[i] = if rng.sample(unit) > 0.5 { 1.0 } else { -1.0 };
        }

        let mut best_objective = f64::INFINITY;
        let mut best_result = None;

        // TSVM alternating optimization
        for iteration in 0..self.config.n_iterations {
            // Step 1: Solve SVM with current pseudo-labels
            let svm = SVC::new()
                .c(self.config.c_supervised)
                .tol(self.config.tol)
                .max_iter(self.config.max_iter)
                .svc_kernel(svc_kernel_for(&self.config.kernel));

            let svm_result = svm.fit(&x_combined, &y_pseudo)?;

            // Step 2: Update pseudo-labels for unlabeled data
            let decision_values = svm_result.decision_function(&x_combined)?;

            // Calculate objective value
            let objective = self.calculate_objective(
                x_combined.nrows(),
                &y_pseudo,
                decision_values.as_slice().ok_or_else(|| {
                    SklearsError::InvalidInput("non-contiguous decision array".to_string())
                })?,
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

                best_result = Some(SemiSupervisedResult {
                    support_vectors: svm_result.support_vectors().clone(),
                    dual_coef: svm_result.dual_coef().clone(),
                    intercept: svm_result.intercept(),
                    support_indices: svm_result.support_indices().to_vec(),
                    unlabeled_predictions: y_pseudo
                        .slice(s![n_labeled..])
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
    pub fn predict(&self, x: &Array2<f64>, result: &SemiSupervisedResult) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "prediction".to_string(),
            });
        }

        let decision_values = self.decision_function(x, result)?;
        Ok(decision_values.mapv(|val| if val > 0.0 { 1.0 } else { -1.0 }))
    }

    /// Calculate decision function values
    fn decision_function(
        &self,
        x: &Array2<f64>,
        result: &SemiSupervisedResult,
    ) -> Result<Array1<f64>> {
        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "decision_function".to_string(),
            })?;
        let concrete_kernel = crate::kernels::create_kernel(kernel.clone())?;

        // `support_vectors` is already compacted to one row per support vector,
        // and `dual_coef` is aligned with those rows. We therefore index both by
        // the same enumerate position rather than by the original-sample index
        // stored in `support_indices`.
        let n_support = result.support_vectors.nrows();
        let mut decision_values = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut sum = 0.0;
            for j in 0..n_support {
                let kernel_val = concrete_kernel.compute(x.row(i), result.support_vectors.row(j));
                sum += result.dual_coef[j] * kernel_val;
            }
            decision_values[i] = sum + result.intercept;
        }

        Ok(decision_values)
    }

    /// Calculate the TSVM objective function
    fn calculate_objective(
        &self,
        n_samples: usize,
        y: &Array1<f64>,
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
        for i in n_labeled..n_samples {
            let margin = y[i] * decision_values[i];
            if margin < 1.0 {
                objective += self.config.c_unsupervised * (1.0 - margin);
            }
        }

        Ok(objective)
    }
}

impl Default for TransductiveSVM {
    fn default() -> Self {
        Self::new(SemiSupervisedConfig::default())
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

    fn train_svm(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<SVC<Trained>> {
        SVC::new()
            .c(self.config.c_supervised)
            .tol(self.config.tol)
            .max_iter(self.config.max_iter)
            .svc_kernel(svc_kernel_for(&self.config.kernel))
            .fit(x, y)
    }

    /// Fit the Self-Training SVM model
    pub fn fit(
        &mut self,
        x_labeled: &Array2<f64>,
        y_labeled: &Array1<f64>,
        x_unlabeled: &Array2<f64>,
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

        let mut iteration = 0;

        while iteration < self.config.n_iterations && !x_unlabeled_remaining.is_empty() {
            // Train SVM on the current labeled data.
            let svm_result = self.train_svm(&x_train, &y_train)?;

            // Predict on the remaining unlabeled data.
            let predictions = svm_result.predict(&x_unlabeled_remaining)?;
            let decision_values = svm_result.decision_function(&x_unlabeled_remaining)?;

            // Calculate confidence scores.
            let confidence_scores: Vec<f64> =
                decision_values.iter().map(|&val| val.abs()).collect();

            // Find the most confident predictions.
            let confident_indices: Vec<usize> = confidence_scores
                .iter()
                .enumerate()
                .filter(|(_, &confidence)| confidence > self.config.confidence_threshold)
                .map(|(i, _)| i)
                .collect();

            if confident_indices.is_empty() {
                break; // No confident predictions
            }

            // Collect the confident samples.
            let mut confident_x = Array2::zeros((confident_indices.len(), x_train.ncols()));
            let mut confident_y = Array1::zeros(confident_indices.len());
            for (i, &idx) in confident_indices.iter().enumerate() {
                confident_x
                    .row_mut(i)
                    .assign(&x_unlabeled_remaining.row(idx));
                confident_y[i] = predictions[idx];
            }

            // Add the confident predictions to the training set.
            x_train = vstack(&x_train, &confident_x);
            y_train = vconcat(&y_train, &confident_y);

            // Remove the confident samples from the unlabeled set.
            x_unlabeled_remaining = remove_rows(&x_unlabeled_remaining, &confident_indices);

            iteration += 1;
        }

        // Final training on all accumulated labeled data.
        let final_result = self.train_svm(&x_train, &y_train)?;
        self.base_classifier = Some(final_result);
        self.is_fitted = true;

        // Predict on all original unlabeled data.
        let classifier = self
            .base_classifier
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "self-training final prediction".to_string(),
            })?;

        let final_predictions = classifier.predict(x_unlabeled)?;
        let final_decision_values = classifier.decision_function(x_unlabeled)?;
        let final_confidence_scores: Vec<f64> =
            final_decision_values.iter().map(|&val| val.abs()).collect();

        Ok(SemiSupervisedResult {
            support_vectors: classifier.support_vectors().clone(),
            dual_coef: classifier.dual_coef().clone(),
            intercept: classifier.intercept(),
            support_indices: classifier.support_indices().to_vec(),
            unlabeled_predictions: final_predictions.iter().cloned().collect(),
            confidence_scores: final_confidence_scores,
            n_iterations: iteration,
            objective_value: 0.0, // Not applicable for self-training
        })
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let classifier = self
            .base_classifier
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "prediction".to_string(),
            })?;
        classifier.predict(x)
    }

    /// Calculate decision function values
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let classifier = self
            .base_classifier
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "decision_function".to_string(),
            })?;
        classifier.decision_function(x)
    }
}

impl Default for SelfTrainingSVM {
    fn default() -> Self {
        Self::new(SemiSupervisedConfig::default())
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

    fn train_svm(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<SVC<Trained>> {
        SVC::new()
            .c(self.config.c_supervised)
            .tol(self.config.tol)
            .max_iter(self.config.max_iter)
            .svc_kernel(svc_kernel_for(&self.config.kernel))
            .fit(x, y)
    }

    fn slice_columns(x: &Array2<f64>, start: usize, len: usize) -> Array2<f64> {
        x.slice(s![.., start..start + len]).to_owned()
    }

    /// Fit the Co-Training SVM model
    pub fn fit(
        &mut self,
        x_labeled: &Array2<f64>,
        y_labeled: &Array1<f64>,
        x_unlabeled: &Array2<f64>,
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

        if self.feature_split == 0 || self.feature_split >= x_labeled.ncols() {
            return Err(SklearsError::InvalidInput(
                "Feature split must be in (0, n_features)".to_string(),
            ));
        }

        let split = self.feature_split;

        // Maintain a single shared labeled pool of full-feature rows. Both views
        // are derived from this pool every iteration, which keeps the two
        // classifiers' training sets perfectly row-aligned even as different
        // samples are added on behalf of each view. This is the mathematically
        // correct co-training formulation: a sample confidently labelled by one
        // view contributes its complete feature vector (both views) and the
        // pseudo-label to the shared pool used by the other view.
        let mut x_pool = x_labeled.clone();
        let mut y_pool = y_labeled.clone();

        // Remaining unlabeled data, kept as full-feature rows.
        let mut x_unlabeled_remaining = x_unlabeled.clone();

        let mut iteration = 0;

        while iteration < self.config.n_iterations && !x_unlabeled_remaining.is_empty() {
            // Build the two column-views of the shared labeled pool.
            let x_train1 = Self::slice_columns(&x_pool, 0, split);
            let x_train2 = Self::slice_columns(&x_pool, split, x_pool.ncols() - split);

            // Build the two column-views of the remaining unlabeled data.
            let x_unlabeled_view1 = Self::slice_columns(&x_unlabeled_remaining, 0, split);
            let x_unlabeled_view2 = Self::slice_columns(
                &x_unlabeled_remaining,
                split,
                x_unlabeled_remaining.ncols() - split,
            );

            // Train both classifiers on their respective views.
            let svm1_result = self.train_svm(&x_train1, &y_pool)?;
            let svm2_result = self.train_svm(&x_train2, &y_pool)?;
            self.classifier1 = Some(svm1_result.clone());
            self.classifier2 = Some(svm2_result.clone());

            // Get predictions from both classifiers.
            let predictions1 = svm1_result.predict(&x_unlabeled_view1)?;
            let predictions2 = svm2_result.predict(&x_unlabeled_view2)?;

            let decision_values1 = svm1_result.decision_function(&x_unlabeled_view1)?;
            let decision_values2 = svm2_result.decision_function(&x_unlabeled_view2)?;

            // Collect (sample_index, pseudo_label) pairs for confident samples
            // from each view. Classifier 1's confident samples teach classifier 2
            // and vice versa; both contribute to the shared pool.
            let mut newly_labeled: Vec<(usize, f64)> = Vec::new();
            for (i, &val) in decision_values1.iter().enumerate() {
                if val.abs() > self.config.confidence_threshold {
                    newly_labeled.push((i, predictions1[i]));
                }
            }
            for (i, &val) in decision_values2.iter().enumerate() {
                if val.abs() > self.config.confidence_threshold {
                    newly_labeled.push((i, predictions2[i]));
                }
            }

            if newly_labeled.is_empty() {
                break; // No confident predictions
            }

            // De-duplicate by sample index (a sample confident in both views is
            // added once, using the first view's label).
            newly_labeled.sort_by_key(|(idx, _)| *idx);
            newly_labeled.dedup_by_key(|(idx, _)| *idx);

            // Append the confidently labelled full-feature rows to the pool.
            let n_new = newly_labeled.len();
            let mut new_x = Array2::zeros((n_new, x_pool.ncols()));
            let mut new_y = Array1::zeros(n_new);
            for (row, (idx, label)) in newly_labeled.iter().enumerate() {
                new_x.row_mut(row).assign(&x_unlabeled_remaining.row(*idx));
                new_y[row] = *label;
            }
            x_pool = vstack(&x_pool, &new_x);
            y_pool = vconcat(&y_pool, &new_y);

            // Remove the newly labelled samples from the unlabeled set.
            let removed: Vec<usize> = newly_labeled.iter().map(|(idx, _)| *idx).collect();
            x_unlabeled_remaining = remove_rows(&x_unlabeled_remaining, &removed);

            iteration += 1;
        }

        // Final training on the combined views of the full shared pool.
        let final_result = self.train_svm(&x_pool, &y_pool)?;
        self.classifier1 = Some(final_result);
        self.is_fitted = true;

        // Predict on all original unlabeled data.
        let classifier = self
            .classifier1
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "co-training final prediction".to_string(),
            })?;

        let final_predictions = classifier.predict(x_unlabeled)?;
        let final_decision_values = classifier.decision_function(x_unlabeled)?;
        let final_confidence_scores: Vec<f64> =
            final_decision_values.iter().map(|&val| val.abs()).collect();

        Ok(SemiSupervisedResult {
            support_vectors: classifier.support_vectors().clone(),
            dual_coef: classifier.dual_coef().clone(),
            intercept: classifier.intercept(),
            support_indices: classifier.support_indices().to_vec(),
            unlabeled_predictions: final_predictions.iter().cloned().collect(),
            confidence_scores: final_confidence_scores,
            n_iterations: iteration,
            objective_value: 0.0, // Not applicable for co-training
        })
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let classifier = self
            .classifier1
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "prediction".to_string(),
            })?;
        classifier.predict(x)
    }

    /// Calculate decision function values
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let classifier = self
            .classifier1
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "decision_function".to_string(),
            })?;
        classifier.decision_function(x)
    }

    /// Whether the second co-training classifier is available for inspection.
    pub fn has_second_view(&self) -> bool {
        self.classifier2.is_some()
    }
}

/// Build an `Array2<f64>` from a row-major slice (mirrors the old test helper).
#[cfg(test)]
fn matrix_from_rows(rows: usize, cols: usize, data: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((rows, cols), data.to_vec()).expect("array shape mismatch")
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let x_labeled = matrix_from_rows(4, 2, &[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0]);
        let y_labeled = Array1::from_vec(vec![1.0, 1.0, -1.0, -1.0]);
        let x_unlabeled = matrix_from_rows(4, 2, &[1.5, 2.5, 2.5, 3.5, 3.5, 2.5, 4.5, 3.5]);
        (x_labeled, y_labeled, x_unlabeled)
    }

    #[test]
    fn test_transductive_svm_basic() {
        let (x_labeled, y_labeled, x_unlabeled) = create_test_data();
        let mut tsvm = TransductiveSVM::default();
        let result = tsvm.fit(&x_labeled, &y_labeled, &x_unlabeled);
        assert!(result.is_ok());

        let result = result.expect("operation should succeed");
        assert_eq!(result.unlabeled_predictions.len(), x_unlabeled.nrows());
        assert_eq!(result.confidence_scores.len(), x_unlabeled.nrows());
    }

    #[test]
    fn test_self_training_svm_basic() {
        let (x_labeled, y_labeled, x_unlabeled) = create_test_data();
        let mut stsvm = SelfTrainingSVM::default();
        let result = stsvm.fit(&x_labeled, &y_labeled, &x_unlabeled);
        assert!(result.is_ok());

        let result = result.expect("operation should succeed");
        assert_eq!(result.unlabeled_predictions.len(), x_unlabeled.nrows());
        assert_eq!(result.confidence_scores.len(), x_unlabeled.nrows());
    }

    #[test]
    fn test_cotraining_svm_basic() {
        let (x_labeled, y_labeled, x_unlabeled) = create_test_data();
        let mut ctsvm = CoTrainingSVM::new(SemiSupervisedConfig::default(), 1);
        let result = ctsvm.fit(&x_labeled, &y_labeled, &x_unlabeled);
        assert!(result.is_ok());

        let result = result.expect("operation should succeed");
        assert_eq!(result.unlabeled_predictions.len(), x_unlabeled.nrows());
        assert_eq!(result.confidence_scores.len(), x_unlabeled.nrows());
    }

    #[test]
    fn test_tsvm_prediction() {
        let (x_labeled, y_labeled, x_unlabeled) = create_test_data();
        let mut tsvm = TransductiveSVM::default();
        let result = tsvm
            .fit(&x_labeled, &y_labeled, &x_unlabeled)
            .expect("model fitting should succeed");

        let predictions = tsvm.predict(&x_unlabeled, &result);
        assert!(predictions.is_ok());

        let predictions = predictions.expect("operation should succeed");
        assert_eq!(predictions.len(), x_unlabeled.nrows());
        assert!(predictions.iter().all(|&p| p == 1.0 || p == -1.0));
    }

    #[test]
    fn test_predict_before_fit_errors() {
        let stsvm = SelfTrainingSVM::default();
        let x = matrix_from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(
            stsvm.predict(&x),
            Err(SklearsError::NotFitted { .. })
        ));
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

        let tsvm = TransductiveSVM::new(config.clone());
        assert_eq!(tsvm.config.c_supervised, 0.5);
        assert_eq!(tsvm.config.c_unsupervised, 0.05);
        assert_eq!(tsvm.config.confidence_threshold, 0.95);
    }
}
