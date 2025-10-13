//! Multi-class SVM classification using One-vs-Rest and One-vs-One strategies

use crate::svc::{SvcConfig, SVC};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Rng;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Multi-class strategy for SVM
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MultiClassStrategy {
    /// One-vs-Rest (OvR) strategy
    #[default]
    OneVsRest,
    /// One-vs-One (OvO) strategy with majority voting
    OneVsOne,
    /// One-vs-One (OvO) strategy with decision-based voting
    OneVsOneDecision,
    /// Error-Correcting Output Codes (ECOC) strategy
    Ecoc,
    /// Hierarchical classification using binary tree
    HierarchicalTree,
}

/// Node in a hierarchical classification tree
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// Leaf node containing a class label
    Leaf(Float),
    /// Internal node with classifier and children
    Internal {
        classifier: SVC<Trained>,
        left_classes: Vec<Float>,
        right_classes: Vec<Float>,
        left_child: Box<TreeNode>,
        right_child: Box<TreeNode>,
    },
}

/// Hierarchical tree for multi-class classification
#[derive(Debug, Clone)]
pub struct HierarchicalTree {
    root: TreeNode,
}

impl HierarchicalTree {
    /// Create a new hierarchical tree with the given root
    pub fn new(root: TreeNode) -> Self {
        Self { root }
    }

    /// Predict class for a single sample
    pub fn predict_sample(&self, x: &Array2<Float>) -> Result<Float> {
        self.predict_node(&self.root, x)
    }

    /// Recursively predict using tree nodes
    fn predict_node(&self, node: &TreeNode, x: &Array2<Float>) -> Result<Float> {
        match node {
            TreeNode::Leaf(class) => Ok(*class),
            TreeNode::Internal {
                classifier,
                left_classes: _,
                right_classes: _,
                left_child,
                right_child,
            } => {
                let decision = classifier.decision_function(x)?[0];
                if decision <= 0.0 {
                    self.predict_node(left_child, x)
                } else {
                    self.predict_node(right_child, x)
                }
            }
        }
    }

    /// Get decision path for a sample (for interpretability)
    pub fn decision_path(&self, x: &Array2<Float>) -> Result<Vec<Float>> {
        let mut path = Vec::new();
        self.collect_decision_path(&self.root, x, &mut path)?;
        Ok(path)
    }

    fn collect_decision_path(
        &self,
        node: &TreeNode,
        x: &Array2<Float>,
        path: &mut Vec<Float>,
    ) -> Result<()> {
        match node {
            TreeNode::Leaf(_) => Ok(()),
            TreeNode::Internal {
                classifier,
                left_classes: _,
                right_classes: _,
                left_child,
                right_child,
            } => {
                let decision = classifier.decision_function(x)?[0];
                path.push(decision);
                if decision <= 0.0 {
                    self.collect_decision_path(left_child, x, path)
                } else {
                    self.collect_decision_path(right_child, x, path)
                }
            }
        }
    }
}

/// Multi-class Support Vector Classification
#[derive(Debug, Clone)]
pub struct MultiClassSVC<State = Untrained> {
    config: SvcConfig,
    strategy: MultiClassStrategy,
    state: PhantomData<State>,
    // Fitted parameters
    estimators_: Option<Vec<SVC<Trained>>>,
    classes_: Option<Array1<Float>>,
    n_features_in_: Option<usize>,
    class_pairs_: Option<Vec<(Float, Float)>>, // For OvO strategy
    codebook_: Option<Array2<Float>>,          // For ECOC strategy (classes x bits)
    hierarchy_tree_: Option<HierarchicalTree>, // For hierarchical strategy
}

impl MultiClassSVC<Untrained> {
    /// Create a new multi-class SVC
    pub fn new() -> Self {
        Self {
            config: SvcConfig::default(),
            strategy: MultiClassStrategy::default(),
            state: PhantomData,
            estimators_: None,
            classes_: None,
            n_features_in_: None,
            class_pairs_: None,
            codebook_: None,
            hierarchy_tree_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the kernel to linear
    pub fn linear(mut self) -> Self {
        self.config.kernel = crate::svc::SvcKernel::Linear;
        self
    }

    /// Set the kernel to RBF with optional gamma
    pub fn rbf(mut self, gamma: Option<Float>) -> Self {
        self.config.kernel = crate::svc::SvcKernel::Rbf { gamma };
        self
    }

    /// Set the kernel to polynomial
    pub fn poly(mut self, degree: usize, gamma: Option<Float>, coef0: Float) -> Self {
        self.config.kernel = crate::svc::SvcKernel::Poly {
            degree,
            gamma,
            coef0,
        };
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

    /// Set the multi-class strategy
    pub fn strategy(mut self, strategy: MultiClassStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set One-vs-Rest strategy
    pub fn one_vs_rest(mut self) -> Self {
        self.strategy = MultiClassStrategy::OneVsRest;
        self
    }

    /// Set One-vs-One strategy
    pub fn one_vs_one(mut self) -> Self {
        self.strategy = MultiClassStrategy::OneVsOne;
        self
    }

    /// Set One-vs-One strategy with decision-based voting
    pub fn one_vs_one_decision(mut self) -> Self {
        self.strategy = MultiClassStrategy::OneVsOneDecision;
        self
    }

    /// Set Error-Correcting Output Codes (ECOC) strategy
    pub fn ecoc(mut self) -> Self {
        self.strategy = MultiClassStrategy::Ecoc;
        self
    }

    /// Set hierarchical tree strategy
    pub fn hierarchical_tree(mut self) -> Self {
        self.strategy = MultiClassStrategy::HierarchicalTree;
        self
    }

    /// Set balanced class weights
    pub fn balanced(mut self) -> Self {
        self.config.class_weight = Some(crate::svc::ClassWeight::Balanced);
        self
    }

    /// Find unique classes in the target array
    fn find_classes(y: &Array1<Float>) -> Array1<Float> {
        let mut classes: Vec<Float> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        Array1::from_vec(classes)
    }

    /// Generate class pairs for One-vs-One strategy
    fn generate_class_pairs(classes: &Array1<Float>) -> Vec<(Float, Float)> {
        let mut pairs = Vec::new();
        for (i, &class_a) in classes.iter().enumerate() {
            for &class_b in classes.iter().skip(i + 1) {
                pairs.push((class_a, class_b));
            }
        }
        pairs
    }

    /// Create binary labels for One-vs-Rest
    fn create_ovr_labels(y: &Array1<Float>, positive_class: Float) -> Array1<Float> {
        y.mapv(|label| if label == positive_class { 1.0 } else { 0.0 })
    }

    /// Extract samples for One-vs-One classification
    fn extract_ovo_samples(
        x: &Array2<Float>,
        y: &Array1<Float>,
        class_a: Float,
        class_b: Float,
    ) -> (Array2<Float>, Array1<Float>, Vec<usize>) {
        let mut indices = Vec::new();
        let mut labels = Vec::new();

        for (i, &label) in y.iter().enumerate() {
            if label == class_a || label == class_b {
                indices.push(i);
                labels.push(if label == class_a { 0.0 } else { 1.0 });
            }
        }

        let n_samples = indices.len();
        let n_features = x.ncols();
        let mut x_binary = Array2::zeros((n_samples, n_features));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            x_binary.row_mut(new_idx).assign(&x.row(old_idx));
        }

        (x_binary, Array1::from_vec(labels), indices)
    }

    /// Generate ECOC codebook using random codes with improved discrimination
    fn generate_ecoc_codebook(n_classes: usize, n_bits: Option<usize>) -> Array2<Float> {
        // Use at least ceil(log2(n_classes)) bits, but more for error correction
        let n_bits = n_bits.unwrap_or_else(|| {
            let min_bits = (n_classes as f64).log2().ceil() as usize;
            (min_bits * 2).max(6) // Use more bits for better error correction
        });

        let mut rng = scirs2_core::random::thread_rng();
        let mut attempts = 0;
        const MAX_ATTEMPTS: usize = 100;

        loop {
            let mut codebook = Array2::zeros((n_classes, n_bits));

            // Generate random binary codes with balanced bits
            for j in 0..n_bits {
                // Ensure each bit has roughly balanced +1/-1 values
                let mut bit_values: Vec<Float> = (0..n_classes)
                    .map(|i| if i < n_classes / 2 { 1.0 } else { -1.0 })
                    .collect();

                // Shuffle to randomize assignment
                for i in 0..n_classes {
                    let swap_idx = rng.gen_range(0..n_classes);
                    bit_values.swap(i, swap_idx);
                }

                for i in 0..n_classes {
                    codebook[[i, j]] = bit_values[i];
                }
            }

            // Check if codebook is valid (no identical or complement codes)
            let mut valid = true;
            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    let mut similarity = 0;
                    let mut complement_similarity = 0;

                    for k in 0..n_bits {
                        if codebook[[i, k]] == codebook[[j, k]] {
                            similarity += 1;
                        }
                        if codebook[[i, k]] == -codebook[[j, k]] {
                            complement_similarity += 1;
                        }
                    }

                    // Require minimum Hamming distance
                    let min_distance = (n_bits / 3).max(1);
                    if n_bits - similarity < min_distance
                        || n_bits - complement_similarity < min_distance
                    {
                        valid = false;
                        break;
                    }
                }
                if !valid {
                    break;
                }
            }

            if valid {
                return codebook;
            }

            attempts += 1;
            if attempts >= MAX_ATTEMPTS {
                // Fallback to simple random generation if we can't find good codes
                let mut codebook = Array2::zeros((n_classes, n_bits));
                for i in 0..n_classes {
                    for j in 0..n_bits {
                        codebook[[i, j]] = if rng.random::<bool>() { 1.0 } else { -1.0 };
                    }
                }
                return codebook;
            }
        }
    }

    /// Create binary labels for ECOC bit classifier
    fn create_ecoc_labels(
        y: &Array1<Float>,
        classes: &Array1<Float>,
        bit_idx: usize,
        codebook: &Array2<Float>,
    ) -> Array1<Float> {
        y.mapv(|label| {
            let class_idx = classes.iter().position(|&c| c == label).unwrap();
            codebook[[class_idx, bit_idx]]
        })
    }

    /// Build hierarchical tree using clustering-based approach
    fn build_hierarchical_tree(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        classes: &[Float],
    ) -> Result<TreeNode> {
        // Base case: single class
        if classes.len() == 1 {
            return Ok(TreeNode::Leaf(classes[0]));
        }

        // Base case: two classes - create binary classifier
        if classes.len() == 2 {
            let left_class = classes[0];
            let right_class = classes[1];

            // Extract samples for binary classification
            let (x_binary, y_binary, _) = Self::extract_ovo_samples(x, y, left_class, right_class);

            // Create and train binary classifier
            let svc = SVC::new()
                .c(self.config.c)
                .tol(self.config.tol)
                .max_iter(self.config.max_iter);

            let fitted_svc = match &self.config.kernel {
                crate::svc::SvcKernel::Linear => svc.linear(),
                crate::svc::SvcKernel::Rbf { gamma } => svc.rbf(*gamma),
                crate::svc::SvcKernel::Poly {
                    degree,
                    gamma,
                    coef0,
                } => svc.poly(*degree, *gamma, *coef0),
                crate::svc::SvcKernel::Sigmoid { gamma, coef0: _ } => svc.rbf(*gamma),
                crate::svc::SvcKernel::Custom(kernel) => svc.kernel(kernel.clone()),
            }
            .fit(&x_binary, &y_binary)?;

            return Ok(TreeNode::Internal {
                classifier: fitted_svc,
                left_classes: vec![left_class],
                right_classes: vec![right_class],
                left_child: Box::new(TreeNode::Leaf(left_class)),
                right_child: Box::new(TreeNode::Leaf(right_class)),
            });
        }

        // Recursive case: split classes into two groups
        let (left_classes, right_classes) = self.split_classes(classes);

        // Create binary labels for the split
        let mut binary_y = Array1::zeros(y.len());
        for (i, &label) in y.iter().enumerate() {
            if left_classes.contains(&label) {
                binary_y[i] = 0.0;
            } else if right_classes.contains(&label) {
                binary_y[i] = 1.0;
            }
        }

        // Filter samples that belong to the current classes
        let mut relevant_indices = Vec::new();
        for (i, &label) in y.iter().enumerate() {
            if classes.contains(&label) {
                relevant_indices.push(i);
            }
        }

        if relevant_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No samples found for hierarchical tree node".to_string(),
            ));
        }

        // Extract relevant samples
        let n_relevant = relevant_indices.len();
        let n_features = x.ncols();
        let mut x_relevant = Array2::zeros((n_relevant, n_features));
        let mut y_relevant = Array1::zeros(n_relevant);

        for (new_idx, &old_idx) in relevant_indices.iter().enumerate() {
            x_relevant.row_mut(new_idx).assign(&x.row(old_idx));
            y_relevant[new_idx] = binary_y[old_idx];
        }

        // Train binary classifier for this split
        let svc = SVC::new()
            .c(self.config.c)
            .tol(self.config.tol)
            .max_iter(self.config.max_iter);

        let fitted_svc = match &self.config.kernel {
            crate::svc::SvcKernel::Linear => svc.linear(),
            crate::svc::SvcKernel::Rbf { gamma } => svc.rbf(*gamma),
            crate::svc::SvcKernel::Poly {
                degree,
                gamma,
                coef0,
            } => svc.poly(*degree, *gamma, *coef0),
            crate::svc::SvcKernel::Sigmoid { gamma, coef0: _ } => svc.rbf(*gamma),
            crate::svc::SvcKernel::Custom(kernel) => svc.kernel(kernel.clone()),
        }
        .fit(&x_relevant, &y_relevant)?;

        // Recursively build child trees
        let left_child = Box::new(self.build_hierarchical_tree(x, y, &left_classes)?);
        let right_child = Box::new(self.build_hierarchical_tree(x, y, &right_classes)?);

        Ok(TreeNode::Internal {
            classifier: fitted_svc,
            left_classes,
            right_classes,
            left_child,
            right_child,
        })
    }

    /// Split classes into two roughly equal groups
    /// For simplicity, we use a round-robin approach
    /// In practice, you might use clustering or other sophisticated methods
    fn split_classes(&self, classes: &[Float]) -> (Vec<Float>, Vec<Float>) {
        let mut left_classes = Vec::new();
        let mut right_classes = Vec::new();

        for (i, &class) in classes.iter().enumerate() {
            if i % 2 == 0 {
                left_classes.push(class);
            } else {
                right_classes.push(class);
            }
        }

        // Ensure both groups have at least one class
        if left_classes.is_empty() {
            left_classes.push(right_classes.pop().unwrap());
        } else if right_classes.is_empty() {
            right_classes.push(left_classes.pop().unwrap());
        }

        (left_classes, right_classes)
    }
}

impl MultiClassSVC<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<Float> {
        self.classes_
            .as_ref()
            .expect("MultiClassSVC should be fitted")
    }

    /// Get the number of estimators
    pub fn n_estimators(&self) -> usize {
        self.estimators_
            .as_ref()
            .expect("MultiClassSVC should be fitted")
            .len()
    }

    /// Get a reference to the binary estimators
    pub fn estimators(&self) -> &[SVC<Trained>] {
        self.estimators_
            .as_ref()
            .expect("MultiClassSVC should be fitted")
    }

    /// Compute decision function values for each binary classifier
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature mismatch: expected {} features, got {}",
                self.n_features_in_.unwrap(),
                n_features
            )));
        }

        let estimators = self.estimators();
        let n_estimators = estimators.len();
        let mut decision_scores = Array2::zeros((n_samples, n_estimators));

        for (i, estimator) in estimators.iter().enumerate() {
            let scores = estimator.decision_function(x)?;
            decision_scores.column_mut(i).assign(&scores);
        }

        Ok(decision_scores)
    }
}

impl<State> MultiClassSVC<State> {
    /// Compute Hamming distance between predicted code and class codewords
    fn ecoc_hamming_distance(predicted_code: &Array1<Float>, class_code: &Array1<Float>) -> usize {
        predicted_code
            .iter()
            .zip(class_code.iter())
            .map(|(&pred, &class)| {
                if (pred > 0.0 && class > 0.0) || (pred <= 0.0 && class <= 0.0) {
                    0
                } else {
                    1
                }
            })
            .sum()
    }
}

impl Default for MultiClassSVC<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for MultiClassSVC<Untrained> {
    type Fitted = MultiClassSVC<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit MultiClassSVC on empty dataset".to_string(),
            ));
        }

        // Find unique classes
        let classes = Self::find_classes(y);

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "MultiClassSVC requires at least 2 classes".to_string(),
            ));
        }

        // If binary classification, use single SVC
        if classes.len() == 2 {
            let svc = SVC::new()
                .c(self.config.c)
                .tol(self.config.tol)
                .max_iter(self.config.max_iter);

            let fitted_svc = match &self.config.kernel {
                crate::svc::SvcKernel::Linear => svc.linear(),
                crate::svc::SvcKernel::Rbf { gamma } => svc.rbf(*gamma),
                crate::svc::SvcKernel::Poly {
                    degree,
                    gamma,
                    coef0,
                } => svc.poly(*degree, *gamma, *coef0),
                crate::svc::SvcKernel::Sigmoid { gamma, coef0: _ } => {
                    // For now, use RBF as sigmoid is not implemented in SVC
                    svc.rbf(*gamma)
                }
                crate::svc::SvcKernel::Custom(kernel) => svc.kernel(kernel.clone()),
            }
            .fit(x, y)?;

            return Ok(MultiClassSVC {
                config: self.config,
                strategy: self.strategy,
                state: PhantomData,
                estimators_: Some(vec![fitted_svc]),
                classes_: Some(classes),
                n_features_in_: Some(n_features),
                class_pairs_: None,
                codebook_: None,
                hierarchy_tree_: None,
            });
        }

        // Multi-class case
        let mut estimators = Vec::new();
        let mut class_pairs = None;
        let mut codebook = None;
        let mut hierarchy_tree = None;

        match self.strategy {
            MultiClassStrategy::OneVsRest => {
                // Train one binary classifier per class
                for &positive_class in classes.iter() {
                    let binary_y = Self::create_ovr_labels(y, positive_class);

                    let svc = SVC::new()
                        .c(self.config.c)
                        .tol(self.config.tol)
                        .max_iter(self.config.max_iter);

                    let fitted_svc = match &self.config.kernel {
                        crate::svc::SvcKernel::Linear => svc.linear(),
                        crate::svc::SvcKernel::Rbf { gamma } => svc.rbf(*gamma),
                        crate::svc::SvcKernel::Poly {
                            degree,
                            gamma,
                            coef0,
                        } => svc.poly(*degree, *gamma, *coef0),
                        crate::svc::SvcKernel::Sigmoid { gamma, coef0: _ } => {
                            // For now, use RBF as sigmoid is not implemented in SVC
                            svc.rbf(*gamma)
                        }
                        crate::svc::SvcKernel::Custom(kernel) => svc.kernel(kernel.clone()),
                    }
                    .fit(x, &binary_y)?;

                    estimators.push(fitted_svc);
                }
            }
            MultiClassStrategy::OneVsOne | MultiClassStrategy::OneVsOneDecision => {
                // Train one binary classifier for each pair of classes
                let pairs = Self::generate_class_pairs(&classes);

                for &(class_a, class_b) in &pairs {
                    let (x_binary, y_binary, _) = Self::extract_ovo_samples(x, y, class_a, class_b);

                    let svc = SVC::new()
                        .c(self.config.c)
                        .tol(self.config.tol)
                        .max_iter(self.config.max_iter);

                    let fitted_svc = match &self.config.kernel {
                        crate::svc::SvcKernel::Linear => svc.linear(),
                        crate::svc::SvcKernel::Rbf { gamma } => svc.rbf(*gamma),
                        crate::svc::SvcKernel::Poly {
                            degree,
                            gamma,
                            coef0,
                        } => svc.poly(*degree, *gamma, *coef0),
                        crate::svc::SvcKernel::Sigmoid { gamma, coef0: _ } => {
                            // For now, use RBF as sigmoid is not implemented in SVC
                            svc.rbf(*gamma)
                        }
                        crate::svc::SvcKernel::Custom(kernel) => svc.kernel(kernel.clone()),
                    }
                    .fit(&x_binary, &y_binary)?;

                    estimators.push(fitted_svc);
                }

                class_pairs = Some(pairs);
            }
            MultiClassStrategy::Ecoc => {
                // Generate ECOC codebook
                let generated_codebook = Self::generate_ecoc_codebook(classes.len(), None);
                let n_bits = generated_codebook.ncols();

                // Train one binary classifier for each bit
                for bit_idx in 0..n_bits {
                    let binary_y =
                        Self::create_ecoc_labels(y, &classes, bit_idx, &generated_codebook);

                    // Skip bits that have only one class (all same values)
                    let has_positive = binary_y.iter().any(|&x| x > 0.0);
                    let has_negative = binary_y.iter().any(|&x| x <= 0.0);
                    if !has_positive || !has_negative {
                        continue; // Skip this bit as it doesn't provide discriminative information
                    }

                    let svc = SVC::new()
                        .c(self.config.c)
                        .tol(self.config.tol)
                        .max_iter(self.config.max_iter);

                    let fitted_svc = match &self.config.kernel {
                        crate::svc::SvcKernel::Linear => svc.linear(),
                        crate::svc::SvcKernel::Rbf { gamma } => svc.rbf(*gamma),
                        crate::svc::SvcKernel::Poly {
                            degree,
                            gamma,
                            coef0,
                        } => svc.poly(*degree, *gamma, *coef0),
                        crate::svc::SvcKernel::Sigmoid { gamma, coef0: _ } => {
                            // For now, use RBF as sigmoid is not implemented in SVC
                            svc.rbf(*gamma)
                        }
                        crate::svc::SvcKernel::Custom(kernel) => svc.kernel(kernel.clone()),
                    }
                    .fit(x, &binary_y)?;

                    estimators.push(fitted_svc);
                }

                codebook = Some(generated_codebook);
            }
            MultiClassStrategy::HierarchicalTree => {
                // Build hierarchical tree
                let root = self.build_hierarchical_tree(x, y, &classes.to_vec())?;
                hierarchy_tree = Some(HierarchicalTree::new(root));

                // For hierarchical tree, we don't need separate estimators since they're embedded in the tree
                // But we'll keep the estimators empty for consistency with the interface
            }
        }

        Ok(MultiClassSVC {
            config: self.config,
            strategy: self.strategy,
            state: PhantomData,
            estimators_: Some(estimators),
            classes_: Some(classes),
            n_features_in_: Some(n_features),
            class_pairs_: class_pairs,
            codebook_: codebook,
            hierarchy_tree_: hierarchy_tree,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for MultiClassSVC<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature mismatch: expected {} features, got {}",
                self.n_features_in_.unwrap(),
                n_features
            )));
        }

        let classes = self.classes();
        let estimators = self.estimators();

        // Binary classification case
        if classes.len() == 2 {
            return estimators[0].predict(x);
        }

        let mut predictions = Array1::zeros(n_samples);

        match self.strategy {
            MultiClassStrategy::OneVsRest => {
                // Get decision scores from all binary classifiers
                let decision_scores = self.decision_function(x)?;

                // Predict class with highest decision score
                for i in 0..n_samples {
                    let mut max_score = Float::NEG_INFINITY;
                    let mut best_class = classes[0];

                    for (j, &class) in classes.iter().enumerate() {
                        let score = decision_scores[[i, j]];
                        if score > max_score {
                            max_score = score;
                            best_class = class;
                        }
                    }

                    predictions[i] = best_class;
                }
            }
            MultiClassStrategy::OneVsOne => {
                let pairs = self.class_pairs_.as_ref().unwrap();

                // Vote-based prediction with improved tie handling
                for i in 0..n_samples {
                    let mut votes = vec![0usize; classes.len()];
                    let mut decision_sums = vec![0.0; classes.len()];

                    // Get votes from each binary classifier
                    for (j, &(class_a, class_b)) in pairs.iter().enumerate() {
                        let sample_view = x.row(i);
                        let sample =
                            Array2::from_shape_vec((1, n_features), sample_view.to_vec()).unwrap();
                        let prediction = estimators[j].predict(&sample)?;
                        let decision_score = estimators[j].decision_function(&sample)?[0];

                        let predicted_class = if prediction[0] == 0.0 {
                            class_a
                        } else {
                            class_b
                        };

                        // Find class indices for decision scores
                        let idx_a = classes.iter().position(|&c| c == class_a).unwrap();
                        let idx_b = classes.iter().position(|&c| c == class_b).unwrap();

                        // Accumulate decision scores
                        if decision_score > 0.0 {
                            decision_sums[idx_b] += decision_score;
                        } else {
                            decision_sums[idx_a] += -decision_score;
                        }

                        // Count votes
                        for (class_idx, &class) in classes.iter().enumerate() {
                            if class == predicted_class {
                                votes[class_idx] += 1;
                                break;
                            }
                        }
                    }

                    // Find class with most votes, use decision scores for tie-breaking
                    let max_votes = *votes.iter().max().unwrap();
                    let tied_classes: Vec<usize> = votes
                        .iter()
                        .enumerate()
                        .filter(|&(_, &count)| count == max_votes)
                        .map(|(idx, _)| idx)
                        .collect();

                    let best_class_idx = if tied_classes.len() == 1 {
                        tied_classes[0]
                    } else {
                        // Break tie using decision scores
                        tied_classes
                            .iter()
                            .max_by(|&&a, &&b| {
                                decision_sums[a].partial_cmp(&decision_sums[b]).unwrap()
                            })
                            .copied()
                            .unwrap_or(0)
                    };

                    predictions[i] = classes[best_class_idx];
                }
            }
            MultiClassStrategy::OneVsOneDecision => {
                let pairs = self.class_pairs_.as_ref().unwrap();

                // Decision-based prediction
                for i in 0..n_samples {
                    let mut decision_sums = vec![0.0; classes.len()];

                    // Accumulate decision scores for each class
                    for (j, &(class_a, class_b)) in pairs.iter().enumerate() {
                        let sample_view = x.row(i);
                        let sample =
                            Array2::from_shape_vec((1, n_features), sample_view.to_vec()).unwrap();
                        let decision_score = estimators[j].decision_function(&sample)?[0];

                        // Find class indices
                        let idx_a = classes.iter().position(|&c| c == class_a).unwrap();
                        let idx_b = classes.iter().position(|&c| c == class_b).unwrap();

                        // Positive score favors class_b, negative favors class_a
                        if decision_score > 0.0 {
                            decision_sums[idx_b] += decision_score;
                        } else {
                            decision_sums[idx_a] += -decision_score;
                        }
                    }

                    // Find class with highest decision sum
                    let best_class_idx = decision_sums
                        .iter()
                        .enumerate()
                        .max_by(|&(_, &a), &(_, &b)| a.partial_cmp(&b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    predictions[i] = classes[best_class_idx];
                }
            }
            MultiClassStrategy::Ecoc => {
                let codebook = self.codebook_.as_ref().unwrap();
                let n_bits = codebook.ncols();

                // Predict using ECOC
                for i in 0..n_samples {
                    let mut predicted_code = Array1::zeros(n_bits);

                    // Get prediction from each bit classifier
                    for bit_idx in 0..n_bits {
                        let sample_view = x.row(i);
                        let sample =
                            Array2::from_shape_vec((1, n_features), sample_view.to_vec()).unwrap();
                        let decision_score = estimators[bit_idx].decision_function(&sample)?[0];
                        predicted_code[bit_idx] = if decision_score > 0.0 { 1.0 } else { -1.0 };
                    }

                    // Find closest codeword using Hamming distance
                    let mut min_distance = usize::MAX;
                    let mut best_class_idx = 0;

                    for (class_idx, &_class) in classes.iter().enumerate() {
                        let class_code = codebook.row(class_idx);
                        let distance =
                            Self::ecoc_hamming_distance(&predicted_code, &class_code.to_owned());

                        if distance < min_distance {
                            min_distance = distance;
                            best_class_idx = class_idx;
                        }
                    }

                    predictions[i] = classes[best_class_idx];
                }
            }
            MultiClassStrategy::HierarchicalTree => {
                let tree = self.hierarchy_tree_.as_ref().unwrap();

                for i in 0..n_samples {
                    let sample_view = x.row(i);
                    let sample =
                        Array2::from_shape_vec((1, n_features), sample_view.to_vec()).unwrap();
                    predictions[i] = tree.predict_sample(&sample)?;
                }
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
    #[ignore = "Slow test: trains multiple SVM classifiers. Run with --ignored flag"]
    fn test_multiclass_svc_ovr() {
        // Create a simple 3-class dataset
        let x = array![
            [1.0, 1.0], // Class 0
            [1.1, 1.1], // Class 0
            [5.0, 5.0], // Class 1
            [5.1, 5.1], // Class 1
            [9.0, 9.0], // Class 2
            [9.1, 9.1], // Class 2
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let svc = MultiClassSVC::new()
            .linear()
            .c(1.0)
            .tol(0.1) // Very high tolerance for test speed
            .max_iter(10) // Very low iterations for tests
            .one_vs_rest()
            .fit(&x, &y)
            .unwrap();

        // Check fitted attributes
        assert_eq!(svc.classes().len(), 3);
        assert_eq!(svc.n_estimators(), 3); // One classifier per class

        // Test prediction
        let x_test = array![
            [1.0, 1.0], // Should be class 0
            [5.0, 5.0], // Should be class 1
            [9.0, 9.0], // Should be class 2
        ];
        let predictions = svc.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    #[ignore = "Slow test: trains multiple SVM classifiers. Run with --ignored flag"]
    fn test_multiclass_svc_ovo() {
        // Create a simple 3-class dataset
        let x = array![
            [1.0, 1.0], // Class 0
            [1.1, 1.1], // Class 0
            [5.0, 5.0], // Class 1
            [5.1, 5.1], // Class 1
            [9.0, 9.0], // Class 2
            [9.1, 9.1], // Class 2
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let svc = MultiClassSVC::new()
            .linear()
            .c(1.0)
            .tol(0.1) // Very high tolerance for test speed
            .max_iter(10) // Very low iterations for tests
            .one_vs_one()
            .fit(&x, &y)
            .unwrap();

        // Check fitted attributes
        assert_eq!(svc.classes().len(), 3);
        assert_eq!(svc.n_estimators(), 3); // One classifier per pair: (0,1), (0,2), (1,2)

        // Test prediction
        let x_test = array![
            [1.0, 1.0], // Should be class 0
            [5.0, 5.0], // Should be class 1
            [9.0, 9.0], // Should be class 2
        ];
        let predictions = svc.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_multiclass_svc_binary_fallback() {
        // Test that binary classification works correctly
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0],];
        let y = array![1.0, 1.0, 0.0, 0.0];

        let svc = MultiClassSVC::new().linear().c(1.0).fit(&x, &y).unwrap();

        assert_eq!(svc.classes().len(), 2);
        assert_eq!(svc.n_estimators(), 1); // Single binary classifier

        let x_test = array![[1.5, 1.5], [-1.5, -1.5]];
        let predictions = svc.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_generate_class_pairs() {
        let classes = array![0.0, 1.0, 2.0, 3.0];
        let pairs = MultiClassSVC::generate_class_pairs(&classes);

        let expected = vec![
            (0.0, 1.0),
            (0.0, 2.0),
            (0.0, 3.0),
            (1.0, 2.0),
            (1.0, 3.0),
            (2.0, 3.0),
        ];

        assert_eq!(pairs, expected);
        assert_eq!(pairs.len(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_create_ovr_labels() {
        let y = array![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let binary_y = MultiClassSVC::create_ovr_labels(&y, 1.0);
        let expected = array![0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(binary_y, expected);
    }

    #[test]
    #[ignore = "Slow test: trains multiple SVM classifiers. Run with --ignored flag"]
    fn test_multiclass_svc_ecoc() {
        // Create a simple 3-class dataset
        let x = array![
            [1.0, 1.0], // Class 0
            [1.1, 1.1], // Class 0
            [5.0, 5.0], // Class 1
            [5.1, 5.1], // Class 1
            [9.0, 9.0], // Class 2
            [9.1, 9.1], // Class 2
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let svc = MultiClassSVC::new()
            .linear()
            .c(1.0)
            .tol(0.1) // Very high tolerance for test speed
            .max_iter(10) // Very low iterations for tests
            .ecoc()
            .fit(&x, &y)
            .unwrap();

        // Check fitted attributes
        assert_eq!(svc.classes().len(), 3);
        assert!(svc.n_estimators() >= 4); // At least ceil(log2(3)) + 2 = 4 bits

        // Test prediction
        let x_test = array![
            [1.0, 1.0], // Should be class 0
            [5.0, 5.0], // Should be class 1
            [9.0, 9.0], // Should be class 2
        ];
        let predictions = svc.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_ecoc_codebook_generation() {
        let codebook = MultiClassSVC::generate_ecoc_codebook(4, Some(6));
        assert_eq!(codebook.nrows(), 4); // 4 classes
        assert_eq!(codebook.ncols(), 6); // 6 bits

        // Check that all values are either 1.0 or -1.0
        for &val in codebook.iter() {
            assert!(val == 1.0 || val == -1.0);
        }
    }

    #[test]
    fn test_ecoc_hamming_distance() {
        let code1 = array![1.0, -1.0, 1.0, -1.0];
        let code2 = array![1.0, 1.0, 1.0, -1.0];
        let distance = MultiClassSVC::<Untrained>::ecoc_hamming_distance(&code1, &code2);
        assert_eq!(distance, 1); // Only second bit differs

        let code3 = array![-1.0, 1.0, -1.0, 1.0];
        let distance2 = MultiClassSVC::<Untrained>::ecoc_hamming_distance(&code1, &code3);
        assert_eq!(distance2, 4); // All bits differ
    }

    #[test]
    #[ignore = "Slow test: trains multiple SVM classifiers. Run with --ignored flag"]
    fn test_multiclass_svc_hierarchical_tree() {
        // Create a simple 4-class dataset
        let x = array![
            [1.0, 1.0],   // Class 0
            [1.1, 1.1],   // Class 0
            [5.0, 5.0],   // Class 1
            [5.1, 5.1],   // Class 1
            [9.0, 9.0],   // Class 2
            [9.1, 9.1],   // Class 2
            [13.0, 13.0], // Class 3
            [13.1, 13.1], // Class 3
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];

        let svc = MultiClassSVC::new()
            .linear()
            .c(1.0)
            .tol(0.1) // Very high tolerance for test speed
            .max_iter(10) // Very low iterations for tests
            .hierarchical_tree()
            .fit(&x, &y)
            .unwrap();

        // Check fitted attributes
        assert_eq!(svc.classes().len(), 4);
        // For hierarchical tree, the estimators may be empty since they're embedded in the tree

        // Test prediction
        let x_test = array![
            [1.0, 1.0],   // Should be class 0
            [5.0, 5.0],   // Should be class 1
            [9.0, 9.0],   // Should be class 2
            [13.0, 13.0], // Should be class 3
        ];
        let predictions = svc.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 4);

        // Check that all predictions are valid classes
        for &pred in predictions.iter() {
            assert!(pred >= 0.0 && pred <= 3.0);
        }
    }

    #[test]
    fn test_hierarchical_tree_decision_path() {
        // Test the decision path functionality
        let tree = HierarchicalTree::new(TreeNode::Leaf(1.0));

        let sample = array![[1.0, 2.0]];
        let path = tree.decision_path(&sample).unwrap();

        // For a leaf node, path should be empty
        assert_eq!(path.len(), 0);
    }

    #[test]
    fn test_split_classes_method() {
        let svc = MultiClassSVC::new();

        // Test even number of classes
        let classes = [0.0, 1.0, 2.0, 3.0];
        let (left, right) = svc.split_classes(&classes);
        assert_eq!(left.len() + right.len(), 4);
        assert!(!left.is_empty());
        assert!(!right.is_empty());

        // Test odd number of classes
        let classes = [0.0, 1.0, 2.0];
        let (left, right) = svc.split_classes(&classes);
        assert_eq!(left.len() + right.len(), 3);
        assert!(!left.is_empty());
        assert!(!right.is_empty());

        // Test single class
        let classes = [0.0];
        let (left, right) = svc.split_classes(&classes);
        assert_eq!(left.len() + right.len(), 1);
        assert!(!left.is_empty() || !right.is_empty()); // At least one should be non-empty
    }
}
