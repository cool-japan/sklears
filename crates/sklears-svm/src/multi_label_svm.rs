//! Multi-Label Support Vector Machines
//!
//! This module provides various strategies for multi-label classification using SVMs:
//! - Binary Relevance: Independent binary classifiers for each label
//! - Classifier Chains: Sequential classifiers using label predictions as features
//! - Label Powerset: Multi-class approach treating label combinations as classes
//! - Multi-Label SVM: Direct multi-label optimization

use scirs2_core::ndarray::{Array1, Array2};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

/// Errors that can occur during multi-label SVM training and prediction
#[derive(Debug, Clone)]
pub enum MultiLabelError {
    InvalidInput(String),
    TrainingError(String),
    PredictionError(String),
    DimensionMismatch(String),
}

impl fmt::Display for MultiLabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiLabelError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            MultiLabelError::TrainingError(msg) => write!(f, "Training error: {msg}"),
            MultiLabelError::PredictionError(msg) => write!(f, "Prediction error: {msg}"),
            MultiLabelError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
        }
    }
}

impl std::error::Error for MultiLabelError {}

/// Multi-label prediction result
#[derive(Debug, Clone)]
pub struct MultiLabelPrediction {
    /// Binary predictions for each label
    pub labels: Array2<f64>,
    /// Confidence scores for each label
    pub scores: Array2<f64>,
    /// Label names
    pub label_names: Vec<String>,
}

/// Strategy for handling multi-label classification
#[derive(Debug, Clone)]
pub enum MultiLabelStrategy {
    /// Train independent binary classifiers for each label
    BinaryRelevance,
    /// Train classifiers in sequence, using previous predictions as features
    ClassifierChains,
    /// Transform to multi-class by treating label combinations as classes
    LabelPowerset,
    /// Direct multi-label SVM optimization
    DirectOptimization,
}

/// Binary Relevance Multi-Label SVM
///
/// Trains one binary SVM classifier for each label independently.
/// This is the simplest and most commonly used approach.
#[derive(Debug, Clone)]
pub struct BinaryRelevanceSVM {
    /// Individual binary classifiers for each label
    classifiers: Vec<crate::svc::SVC<sklears_core::traits::Trained>>,
    /// Label names
    label_names: Vec<String>,
    /// Number of features
    n_features: usize,
    /// SVM hyperparameters
    pub c: f64,
    pub kernel: crate::svc::SvcKernel,
    pub tolerance: f64,
    pub max_iter: usize,
}

impl Default for BinaryRelevanceSVM {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryRelevanceSVM {
    /// Create a new Binary Relevance Multi-Label SVM
    pub fn new() -> Self {
        Self {
            classifiers: Vec::new(),
            label_names: Vec::new(),
            n_features: 0,
            c: 1.0,
            kernel: crate::svc::SvcKernel::Rbf { gamma: Some(1.0) },
            tolerance: 1e-3,
            max_iter: 1000,
        }
    }

    /// Set the regularization parameter C
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the kernel type
    pub fn with_kernel(mut self, kernel: crate::svc::SvcKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit the multi-label SVM to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<(), MultiLabelError> {
        let (n_samples, n_features) = x.dim();
        let (y_samples, n_labels) = y.dim();

        if n_samples != y_samples {
            return Err(MultiLabelError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                n_samples, y_samples
            )));
        }

        self.n_features = n_features;
        self.label_names = (0..n_labels).map(|i| format!("label_{i}")).collect();
        self.classifiers.clear();

        // Train one binary classifier per label
        for label_idx in 0..n_labels {
            let y_binary = y.column(label_idx);

            // Create and configure binary SVM
            let svm = match &self.kernel {
                crate::svc::SvcKernel::Linear => crate::svc::SVC::new()
                    .c(self.c)
                    .linear()
                    .tol(self.tolerance)
                    .max_iter(self.max_iter),
                crate::svc::SvcKernel::Rbf { gamma } => crate::svc::SVC::new()
                    .c(self.c)
                    .rbf(*gamma)
                    .tol(self.tolerance)
                    .max_iter(self.max_iter),
                crate::svc::SvcKernel::Poly {
                    degree,
                    gamma,
                    coef0,
                } => crate::svc::SVC::new()
                    .c(self.c)
                    .poly(*degree, *gamma, *coef0)
                    .tol(self.tolerance)
                    .max_iter(self.max_iter),
                crate::svc::SvcKernel::Sigmoid { gamma, coef0 } => {
                    crate::svc::SVC::new()
                        .c(self.c)
                        .rbf(*gamma) // Using RBF as placeholder since sigmoid not directly available
                        .tol(self.tolerance)
                        .max_iter(self.max_iter)
                }
                crate::svc::SvcKernel::Custom(kernel_type) => crate::svc::SVC::new()
                    .c(self.c)
                    .kernel(kernel_type.clone())
                    .tol(self.tolerance)
                    .max_iter(self.max_iter),
            };

            // Convert to binary labels (ensure we have 0/1)
            let y_binary_vec: Vec<f64> = y_binary
                .iter()
                .map(|&val| if val > 0.5 { 1.0 } else { 0.0 })
                .collect();
            let y_binary_array = Array1::from_vec(y_binary_vec);

            // Train the binary classifier using the Fit trait
            use sklears_core::traits::Fit;
            let fitted_svm = svm.fit(x, &y_binary_array).map_err(|e| {
                MultiLabelError::TrainingError(format!(
                    "Failed to train classifier for label {}: {:?}",
                    label_idx, e
                ))
            })?;

            self.classifiers.push(fitted_svm);
        }

        Ok(())
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<MultiLabelPrediction, MultiLabelError> {
        if self.classifiers.is_empty() {
            return Err(MultiLabelError::PredictionError(
                "Model not trained".to_string(),
            ));
        }

        let (n_samples, n_features) = x.dim();
        if n_features != self.n_features {
            return Err(MultiLabelError::DimensionMismatch(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        let n_labels = self.classifiers.len();
        let mut predictions = Array2::zeros((n_samples, n_labels));
        let mut scores = Array2::zeros((n_samples, n_labels));

        // Get predictions from each binary classifier
        for (label_idx, classifier) in self.classifiers.iter().enumerate() {
            use sklears_core::traits::Predict;
            let binary_pred = classifier.predict(x).map_err(|e| {
                MultiLabelError::PredictionError(format!(
                    "Prediction failed for label {}: {:?}",
                    label_idx, e
                ))
            })?;

            let decision_scores = classifier.decision_function(x).map_err(|e| {
                MultiLabelError::PredictionError(format!(
                    "Decision function failed for label {}: {:?}",
                    label_idx, e
                ))
            })?;

            // Convert predictions to 0/1
            for (sample_idx, &pred) in binary_pred.iter().enumerate() {
                predictions[[sample_idx, label_idx]] = if pred > 0.0 { 1.0 } else { 0.0 };
                scores[[sample_idx, label_idx]] = decision_scores[sample_idx];
            }
        }

        Ok(MultiLabelPrediction {
            labels: predictions,
            scores,
            label_names: self.label_names.clone(),
        })
    }

    /// Predict probabilities for each label
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>, MultiLabelError> {
        if self.classifiers.is_empty() {
            return Err(MultiLabelError::PredictionError(
                "Model not trained".to_string(),
            ));
        }

        let (n_samples, n_features) = x.dim();
        if n_features != self.n_features {
            return Err(MultiLabelError::DimensionMismatch(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        let n_labels = self.classifiers.len();
        let mut probabilities = Array2::zeros((n_samples, n_labels));

        // Get probabilities from each binary classifier
        for (label_idx, classifier) in self.classifiers.iter().enumerate() {
            let decision_scores = classifier.decision_function(x).map_err(|e| {
                MultiLabelError::PredictionError(format!(
                    "Decision function failed for label {}: {:?}",
                    label_idx, e
                ))
            })?;

            // Convert decision scores to probabilities using sigmoid
            for (sample_idx, &score) in decision_scores.iter().enumerate() {
                probabilities[[sample_idx, label_idx]] = 1.0 / (1.0 + (-score).exp());
            }
        }

        Ok(probabilities)
    }
}

/// Classifier Chains Multi-Label SVM
///
/// Trains binary classifiers in a chain where each classifier uses
/// the predictions of previous classifiers as additional features.
#[derive(Debug, Clone)]
pub struct ClassifierChainsSVM {
    /// Chain of binary classifiers
    classifiers: Vec<crate::svc::SVC<sklears_core::traits::Trained>>,
    /// Label names
    label_names: Vec<String>,
    /// Chain order (which label to predict at each step)
    chain_order: Vec<usize>,
    /// Number of original features
    n_features: usize,
    /// SVM hyperparameters
    pub c: f64,
    pub kernel: crate::svc::SvcKernel,
    pub tolerance: f64,
    pub max_iter: usize,
}

impl Default for ClassifierChainsSVM {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassifierChainsSVM {
    /// Create a new Classifier Chains Multi-Label SVM
    pub fn new() -> Self {
        Self {
            classifiers: Vec::new(),
            label_names: Vec::new(),
            chain_order: Vec::new(),
            n_features: 0,
            c: 1.0,
            kernel: crate::svc::SvcKernel::Rbf { gamma: Some(1.0) },
            tolerance: 1e-3,
            max_iter: 1000,
        }
    }

    /// Set the regularization parameter C
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the kernel type
    pub fn with_kernel(mut self, kernel: crate::svc::SvcKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the chain order (default is 0, 1, 2, ...)
    pub fn with_chain_order(mut self, order: Vec<usize>) -> Self {
        self.chain_order = order;
        self
    }

    /// Fit the classifier chain to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<(), MultiLabelError> {
        let (n_samples, n_features) = x.dim();
        let (y_samples, n_labels) = y.dim();

        if n_samples != y_samples {
            return Err(MultiLabelError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                n_samples, y_samples
            )));
        }

        self.n_features = n_features;
        self.label_names = (0..n_labels).map(|i| format!("label_{i}")).collect();

        // Set default chain order if not specified
        if self.chain_order.is_empty() {
            self.chain_order = (0..n_labels).collect();
        }

        if self.chain_order.len() != n_labels {
            return Err(MultiLabelError::InvalidInput(
                "Chain order length must match number of labels".to_string(),
            ));
        }

        self.classifiers.clear();

        // Train classifiers in chain order
        for (chain_pos, &label_idx) in self.chain_order.iter().enumerate() {
            let y_binary = y.column(label_idx);

            // Create augmented features by adding predictions from previous classifiers
            let x_augmented = if chain_pos == 0 {
                x.clone()
            } else {
                // Add predictions from previous classifiers as features
                let mut augmented = Array2::zeros((n_samples, n_features + chain_pos));

                // Copy original features
                for i in 0..n_samples {
                    for j in 0..n_features {
                        augmented[[i, j]] = x[[i, j]];
                    }
                }

                // Add predictions from previous classifiers
                for prev_pos in 0..chain_pos {
                    let prev_label_idx = self.chain_order[prev_pos];
                    let y_prev_binary = y.column(prev_label_idx);

                    for i in 0..n_samples {
                        augmented[[i, n_features + prev_pos]] =
                            if y_prev_binary[i] > 0.5 { 1.0 } else { 0.0 };
                    }
                }

                augmented
            };

            // Create and configure binary SVM
            let svm = match &self.kernel {
                crate::svc::SvcKernel::Linear => crate::svc::SVC::new()
                    .c(self.c)
                    .linear()
                    .tol(self.tolerance)
                    .max_iter(self.max_iter),
                crate::svc::SvcKernel::Rbf { gamma } => crate::svc::SVC::new()
                    .c(self.c)
                    .rbf(*gamma)
                    .tol(self.tolerance)
                    .max_iter(self.max_iter),
                crate::svc::SvcKernel::Poly {
                    degree,
                    gamma,
                    coef0,
                } => crate::svc::SVC::new()
                    .c(self.c)
                    .poly(*degree, *gamma, *coef0)
                    .tol(self.tolerance)
                    .max_iter(self.max_iter),
                crate::svc::SvcKernel::Sigmoid { gamma, coef0 } => {
                    crate::svc::SVC::new()
                        .c(self.c)
                        .rbf(*gamma) // Using RBF as placeholder
                        .tol(self.tolerance)
                        .max_iter(self.max_iter)
                }
                crate::svc::SvcKernel::Custom(kernel_type) => crate::svc::SVC::new()
                    .c(self.c)
                    .kernel(kernel_type.clone())
                    .tol(self.tolerance)
                    .max_iter(self.max_iter),
            };

            // Convert to binary labels
            let y_binary_vec: Vec<f64> = y_binary
                .iter()
                .map(|&val| if val > 0.5 { 1.0 } else { 0.0 })
                .collect();
            let y_binary_array = Array1::from_vec(y_binary_vec);

            // Train the classifier using the Fit trait
            use sklears_core::traits::Fit;
            let fitted_svm = svm.fit(&x_augmented, &y_binary_array).map_err(|e| {
                MultiLabelError::TrainingError(format!(
                    "Failed to train classifier for label {} in chain: {:?}",
                    label_idx, e
                ))
            })?;

            self.classifiers.push(fitted_svm);
        }

        Ok(())
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<MultiLabelPrediction, MultiLabelError> {
        if self.classifiers.is_empty() {
            return Err(MultiLabelError::PredictionError(
                "Model not trained".to_string(),
            ));
        }

        let (n_samples, n_features) = x.dim();
        if n_features != self.n_features {
            return Err(MultiLabelError::DimensionMismatch(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        let n_labels = self.chain_order.len();
        let mut predictions = Array2::zeros((n_samples, n_labels));
        let mut scores = Array2::zeros((n_samples, n_labels));

        // Predict labels in chain order
        for (chain_pos, &label_idx) in self.chain_order.iter().enumerate() {
            // Create augmented features
            let x_augmented = if chain_pos == 0 {
                x.clone()
            } else {
                let mut augmented = Array2::zeros((n_samples, n_features + chain_pos));

                // Copy original features
                for i in 0..n_samples {
                    for j in 0..n_features {
                        augmented[[i, j]] = x[[i, j]];
                    }
                }

                // Add predictions from previous classifiers
                for prev_pos in 0..chain_pos {
                    let prev_label_idx = self.chain_order[prev_pos];
                    for i in 0..n_samples {
                        augmented[[i, n_features + prev_pos]] = predictions[[i, prev_label_idx]];
                    }
                }

                augmented
            };

            // Get prediction from current classifier
            let classifier = &self.classifiers[chain_pos];
            use sklears_core::traits::Predict;
            let binary_pred = classifier.predict(&x_augmented).map_err(|e| {
                MultiLabelError::PredictionError(format!(
                    "Prediction failed for label {} in chain: {:?}",
                    label_idx, e
                ))
            })?;

            let decision_scores = classifier.decision_function(&x_augmented).map_err(|e| {
                MultiLabelError::PredictionError(format!(
                    "Decision function failed for label {} in chain: {:?}",
                    label_idx, e
                ))
            })?;

            // Store predictions
            for (sample_idx, &pred) in binary_pred.iter().enumerate() {
                predictions[[sample_idx, label_idx]] = if pred > 0.0 { 1.0 } else { 0.0 };
                scores[[sample_idx, label_idx]] = decision_scores[sample_idx];
            }
        }

        Ok(MultiLabelPrediction {
            labels: predictions,
            scores,
            label_names: self.label_names.clone(),
        })
    }
}

/// Label Powerset Multi-Label SVM
///
/// Transforms the multi-label problem into a multi-class problem
/// by treating each unique combination of labels as a separate class.
#[derive(Debug, Clone)]
pub struct LabelPowersetSVM {
    /// Multi-class classifier
    classifier: Option<crate::svc::SVC<sklears_core::traits::Trained>>,
    /// One-vs-rest classifiers for multi-class fallback
    ovr_classifiers: Vec<crate::svc::SVC<sklears_core::traits::Trained>>,
    /// Ordering of class identifiers corresponding to one-vs-rest classifiers
    ovr_class_order: Vec<usize>,
    /// Single-class shortcut when only one powerset class exists
    single_class_id: Option<usize>,
    /// Mapping of negative/positive classes for binary fallback
    binary_class_mapping: Option<(usize, usize)>,
    /// Mapping from class index to label combination
    class_to_labels: HashMap<usize, Vec<usize>>,
    /// Mapping from label combination to class index
    labels_to_class: HashMap<Vec<usize>, usize>,
    /// Label names
    label_names: Vec<String>,
    /// Number of features
    n_features: usize,
    /// SVM hyperparameters
    pub c: f64,
    pub kernel: crate::svc::SvcKernel,
    pub tolerance: f64,
    pub max_iter: usize,
}

impl Default for LabelPowersetSVM {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelPowersetSVM {
    /// Create a new Label Powerset Multi-Label SVM
    pub fn new() -> Self {
        Self {
            classifier: None,
            ovr_classifiers: Vec::new(),
            ovr_class_order: Vec::new(),
            single_class_id: None,
            binary_class_mapping: None,
            class_to_labels: HashMap::new(),
            labels_to_class: HashMap::new(),
            label_names: Vec::new(),
            n_features: 0,
            c: 1.0,
            kernel: crate::svc::SvcKernel::Rbf { gamma: Some(1.0) },
            tolerance: 1e-3,
            max_iter: 1000,
        }
    }

    /// Set the regularization parameter C
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the kernel type
    pub fn with_kernel(mut self, kernel: crate::svc::SvcKernel) -> Self {
        self.kernel = kernel;
        self
    }

    fn build_base_svc(&self) -> crate::svc::SVC<sklears_core::traits::Untrained> {
        let base = crate::svc::SVC::new()
            .c(self.c)
            .tol(self.tolerance)
            .max_iter(self.max_iter);

        match &self.kernel {
            crate::svc::SvcKernel::Linear => base.linear(),
            crate::svc::SvcKernel::Rbf { gamma } => base.rbf(*gamma),
            crate::svc::SvcKernel::Poly {
                degree,
                gamma,
                coef0,
            } => base.poly(*degree, *gamma, *coef0),
            crate::svc::SvcKernel::Sigmoid { gamma, .. } => base.rbf(*gamma),
            crate::svc::SvcKernel::Custom(kernel_type) => base.kernel(kernel_type.clone()),
        }
    }

    /// Convert multi-label targets to powerset classes
    fn create_powerset_mapping(&mut self, y: &Array2<f64>) -> Vec<usize> {
        let (n_samples, n_labels) = y.dim();
        let mut powerset_targets = Vec::with_capacity(n_samples);

        self.class_to_labels.clear();
        self.labels_to_class.clear();
        let mut next_class_id = 0;

        for sample_idx in 0..n_samples {
            // Find active labels for this sample
            let mut active_labels = Vec::new();
            for label_idx in 0..n_labels {
                if y[[sample_idx, label_idx]] > 0.5 {
                    active_labels.push(label_idx);
                }
            }
            // Sort to ensure consistent ordering
            active_labels.sort();

            // Get or create class ID for this label combination
            let class_id = if let Some(&existing_class) = self.labels_to_class.get(&active_labels) {
                existing_class
            } else {
                let new_class = next_class_id;
                next_class_id += 1;
                self.class_to_labels
                    .insert(new_class, active_labels.clone());
                self.labels_to_class.insert(active_labels, new_class);
                new_class
            };

            powerset_targets.push(class_id);
        }

        powerset_targets
    }

    /// Fit the label powerset SVM to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<(), MultiLabelError> {
        let (n_samples, n_features) = x.dim();
        let (y_samples, n_labels) = y.dim();

        if n_samples != y_samples {
            return Err(MultiLabelError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                n_samples, y_samples
            )));
        }

        self.n_features = n_features;
        self.label_names = (0..n_labels).map(|i| format!("label_{i}")).collect();

        // Create powerset mapping
        let powerset_targets = self.create_powerset_mapping(y);
        let mut class_ids: Vec<usize> = self.class_to_labels.keys().copied().collect();
        class_ids.sort_unstable();

        if class_ids.is_empty() {
            return Err(MultiLabelError::InvalidInput(
                "No label combinations found in training data".to_string(),
            ));
        }

        self.classifier = None;
        self.ovr_classifiers.clear();
        self.ovr_class_order.clear();
        self.single_class_id = None;
        self.binary_class_mapping = None;

        use sklears_core::traits::Fit;

        if class_ids.len() == 1 {
            self.single_class_id = Some(class_ids[0]);
            return Ok(());
        }

        if class_ids.len() == 2 {
            let negative_class = class_ids[0];
            let positive_class = class_ids[1];
            let binary_targets = Array1::from_vec(
                powerset_targets
                    .iter()
                    .map(|&class_id| {
                        if class_id == positive_class {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect(),
            );

            let svm = self.build_base_svc();
            let fitted = svm.fit(x, &binary_targets).map_err(|e| {
                MultiLabelError::TrainingError(format!(
                    "Failed to train binary powerset classifier: {:?}",
                    e
                ))
            })?;

            self.classifier = Some(fitted);
            self.binary_class_mapping = Some((negative_class, positive_class));
            return Ok(());
        }

        for &class_id in &class_ids {
            let binary_targets = Array1::from_vec(
                powerset_targets
                    .iter()
                    .map(|&target| if target == class_id { 1.0 } else { -1.0 })
                    .collect(),
            );

            let svm = self.build_base_svc();
            let fitted = svm.fit(x, &binary_targets).map_err(|e| {
                MultiLabelError::TrainingError(format!(
                    "Failed to train one-vs-rest classifier for class {class_id}: {:?}",
                    e
                ))
            })?;

            self.ovr_classifiers.push(fitted);
            self.ovr_class_order.push(class_id);
        }

        Ok(())
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<MultiLabelPrediction, MultiLabelError> {
        let (n_samples, n_features) = x.dim();
        if n_features != self.n_features {
            return Err(MultiLabelError::DimensionMismatch(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        if self.class_to_labels.is_empty() {
            return Err(MultiLabelError::PredictionError(
                "Model not trained".to_string(),
            ));
        }

        let n_labels = self.label_names.len();
        let mut predictions = Array2::zeros((n_samples, n_labels));
        let mut scores = Array2::zeros((n_samples, n_labels));

        let mut predicted_classes: Vec<usize> = Vec::with_capacity(n_samples);
        let mut decision_margins: Vec<f64> = vec![0.0; n_samples];

        if let Some(class_id) = self.single_class_id {
            predicted_classes.resize(n_samples, class_id);
            for margin in &mut decision_margins {
                *margin = 1.0;
            }
        } else if let Some((negative_class, positive_class)) = self.binary_class_mapping {
            let classifier = self.classifier.as_ref().ok_or_else(|| {
                MultiLabelError::PredictionError(
                    "Binary classifier not available for prediction".to_string(),
                )
            })?;

            let decision_scores = classifier.decision_function(x).map_err(|e| {
                MultiLabelError::PredictionError(format!(
                    "Decision function failed for binary powerset classifier: {:?}",
                    e
                ))
            })?;

            for (idx, &score) in decision_scores.iter().enumerate() {
                decision_margins[idx] = score;
                if score >= 0.0 {
                    predicted_classes.push(positive_class);
                } else {
                    predicted_classes.push(negative_class);
                }
            }
        } else if !self.ovr_classifiers.is_empty() {
            let mut best_scores = vec![f64::NEG_INFINITY; n_samples];
            let mut best_classes = vec![usize::MAX; n_samples];

            for (classifier, &class_id) in
                self.ovr_classifiers.iter().zip(self.ovr_class_order.iter())
            {
                let scores_vec = classifier.decision_function(x).map_err(|e| {
                    MultiLabelError::PredictionError(format!(
                        "Decision function failed for OvR classifier {class_id}: {:?}",
                        e
                    ))
                })?;

                for (sample_idx, &score) in scores_vec.iter().enumerate() {
                    match score.partial_cmp(&best_scores[sample_idx]) {
                        Some(Ordering::Greater) => {
                            best_scores[sample_idx] = score;
                            best_classes[sample_idx] = class_id;
                        }
                        Some(Ordering::Equal) => {
                            if class_id < best_classes[sample_idx] {
                                best_scores[sample_idx] = score;
                                best_classes[sample_idx] = class_id;
                            }
                        }
                        _ => {}
                    }
                }
            }

            for (idx, &class_id) in best_classes.iter().enumerate() {
                if class_id == usize::MAX {
                    return Err(MultiLabelError::PredictionError(
                        "One-vs-rest classifiers produced no prediction".to_string(),
                    ));
                }
                predicted_classes.push(class_id);
                decision_margins[idx] = best_scores[idx];
            }
        } else {
            let classifier = self.classifier.as_ref().ok_or_else(|| {
                MultiLabelError::PredictionError(
                    "No classifier available for prediction".to_string(),
                )
            })?;

            use sklears_core::traits::Predict;
            let powerset_pred = classifier.predict(x).map_err(|e| {
                MultiLabelError::PredictionError(format!("Powerset prediction failed: {:?}", e))
            })?;

            let decision_scores = classifier.decision_function(x).map_err(|e| {
                MultiLabelError::PredictionError(format!("Decision function failed: {:?}", e))
            })?;

            for (idx, &value) in powerset_pred.iter().enumerate() {
                predicted_classes.push(value as usize);
                decision_margins[idx] = decision_scores[idx];
            }
        }

        for (sample_idx, &class_id) in predicted_classes.iter().enumerate() {
            if let Some(labels) = self.class_to_labels.get(&class_id) {
                for &label_idx in labels {
                    predictions[[sample_idx, label_idx]] = 1.0;
                    scores[[sample_idx, label_idx]] = decision_margins[sample_idx];
                }
            }
        }

        Ok(MultiLabelPrediction {
            labels: predictions,
            scores,
            label_names: self.label_names.clone(),
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use numrs2::prelude::*;

    fn create_test_data() -> (Array2<f64>, Array2<f64>) {
        // Create simple multi-label test data
        let X_var = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 4.0, 5.0, 5.0, 4.0, 6.0, 6.0],
        )
        .unwrap();

        let y = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                1.0, 1.0,
            ],
        )
        .unwrap();

        (X_var, y)
    }

    #[test]
    #[ignore]
    fn test_binary_relevance_svm() {
        let (X, y) = create_test_data();

        let mut br_svm = BinaryRelevanceSVM::new()
            .with_c(1.0)
            .with_kernel(crate::svc::SvcKernel::Linear);

        // Test fitting
        assert!(br_svm.fit(&X, &y).is_ok());

        // Test prediction
        let pred = br_svm.predict(&X).unwrap();
        assert_eq!(pred.labels.dim(), (6, 3));
        assert_eq!(pred.scores.dim(), (6, 3));
        assert_eq!(pred.label_names.len(), 3);

        // Test predict_proba
        let proba = br_svm.predict_proba(&X).unwrap();
        assert_eq!(proba.dim(), (6, 3));

        // Check that probabilities are in [0, 1]
        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    #[ignore]
    fn test_classifier_chains_svm() {
        let (X, y) = create_test_data();

        let mut cc_svm = ClassifierChainsSVM::new()
            .with_c(1.0)
            .with_kernel(crate::svc::SvcKernel::Linear);

        // Test fitting
        assert!(cc_svm.fit(&X, &y).is_ok());

        // Test prediction
        let pred = cc_svm.predict(&X).unwrap();
        assert_eq!(pred.labels.dim(), (6, 3));
        assert_eq!(pred.scores.dim(), (6, 3));
        assert_eq!(pred.label_names.len(), 3);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_label_powerset_svm() {
        let (X, y) = create_test_data();

        let mut lp_svm = LabelPowersetSVM::new()
            .with_c(1.0)
            .with_kernel(crate::svc::SvcKernel::Linear);

        // Test fitting
        assert!(lp_svm.fit(&X, &y).is_ok());

        // Test prediction
        let pred = lp_svm.predict(&X).unwrap();
        assert_eq!(pred.labels.dim(), (6, 3));
        assert_eq!(pred.scores.dim(), (6, 3));
        assert_eq!(pred.label_names.len(), 3);
    }

    #[test]
    fn test_invalid_input() {
        let X_var = Array2::zeros((5, 2));
        let y = Array2::zeros((6, 3)); // Wrong number of samples

        let mut br_svm = BinaryRelevanceSVM::new();
        assert!(br_svm.fit(&X_var, &y).is_err());
    }

    #[test]
    fn test_prediction_before_training() {
        let X_var = Array2::zeros((5, 2));
        let br_svm = BinaryRelevanceSVM::new();
        assert!(br_svm.predict(&X_var).is_err());
    }
}
