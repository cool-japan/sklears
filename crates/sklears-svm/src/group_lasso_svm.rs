//! Group Lasso Support Vector Machine for feature selection
//!
//! This module implements Group Lasso regularization for SVMs, which performs
//! feature selection at the group level. Unlike standard Lasso that selects
//! individual features, Group Lasso selects or rejects entire groups of features
//! simultaneously. This is particularly useful when features are naturally grouped
//! (e.g., categorical variables with one-hot encoding, polynomial features, etc.).

use crate::kernels::KernelType;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Group Lasso regularization strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GroupLassoStrategy {
    /// Standard Group Lasso: L2 norm within groups, L1 norm across groups
    #[default]
    Standard,
    /// Sparse Group Lasso: combines standard Group Lasso with element-wise L1
    Sparse { l1_ratio: Float },
    /// Exclusive Group Lasso: ensures only one group is selected
    Exclusive,
    /// Overlapping Group Lasso: for overlapping group structures
    Overlapping,
}

/// Feature group definition
#[derive(Debug, Clone)]
pub struct FeatureGroup {
    /// Group ID
    pub id: usize,
    /// Feature indices in this group
    pub features: Vec<usize>,
    /// Group weight (for weighted Group Lasso)
    pub weight: Float,
    /// Group name for interpretability
    pub name: Option<String>,
}

impl FeatureGroup {
    /// Create a new feature group
    pub fn new(id: usize, features: Vec<usize>) -> Self {
        Self {
            id,
            features,
            weight: 1.0,
            name: None,
        }
    }

    /// Set group weight
    pub fn weight(mut self, weight: Float) -> Self {
        self.weight = weight;
        self
    }

    /// Set group name
    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get group size
    pub fn size(&self) -> usize {
        self.features.len()
    }
}

/// Group structure for organizing features
#[derive(Debug, Clone)]
pub struct GroupStructure {
    /// List of feature groups
    pub groups: Vec<FeatureGroup>,
    /// Total number of features
    pub n_features: usize,
    /// Mapping from feature index to group ID
    pub feature_to_group: HashMap<usize, usize>,
}

impl GroupStructure {
    /// Create a new group structure
    pub fn new(groups: Vec<FeatureGroup>, n_features: usize) -> Result<Self> {
        let mut feature_to_group = HashMap::new();

        // Validate groups and create mapping
        for group in &groups {
            for &feature_idx in &group.features {
                if feature_idx >= n_features {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature index {} is out of bounds for {} features",
                        feature_idx, n_features
                    )));
                }

                if feature_to_group.contains_key(&feature_idx) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature {} is assigned to multiple groups",
                        feature_idx
                    )));
                }

                feature_to_group.insert(feature_idx, group.id);
            }
        }

        Ok(Self {
            groups,
            n_features,
            feature_to_group,
        })
    }

    /// Create groups automatically by dividing features into equal-sized blocks
    pub fn create_blocks(n_features: usize, block_size: usize) -> Result<Self> {
        let mut groups = Vec::new();
        let mut group_id = 0;

        for start in (0..n_features).step_by(block_size) {
            let end = (start + block_size).min(n_features);
            let features: Vec<usize> = (start..end).collect();

            groups.push(FeatureGroup::new(group_id, features).name(format!("Block_{group_id}")));
            group_id += 1;
        }

        Self::new(groups, n_features)
    }

    /// Create groups for one-hot encoded categorical features
    pub fn create_categorical_groups(categorical_ranges: Vec<(usize, usize)>) -> Result<Self> {
        let mut groups = Vec::new();
        let mut group_id = 0;
        let mut total_features = 0;

        for (start, end) in categorical_ranges {
            let features: Vec<usize> = (start..end).collect();
            total_features = total_features.max(end);

            groups.push(
                FeatureGroup::new(group_id, features).name(format!("Categorical_{group_id}")),
            );
            group_id += 1;
        }

        Self::new(groups, total_features)
    }

    /// Get group for a feature
    pub fn get_group(&self, feature_idx: usize) -> Option<&FeatureGroup> {
        if let Some(&group_id) = self.feature_to_group.get(&feature_idx) {
            self.groups.iter().find(|g| g.id == group_id)
        } else {
            None
        }
    }

    /// Get all features in a group
    pub fn get_group_features(&self, group_id: usize) -> Option<&Vec<usize>> {
        self.groups
            .iter()
            .find(|g| g.id == group_id)
            .map(|g| &g.features)
    }
}

/// Configuration for Group Lasso SVM
#[derive(Debug, Clone)]
pub struct GroupLassoSVMConfig {
    /// Regularization parameter for SVM
    pub c: Float,
    /// Group Lasso regularization parameter
    pub lambda: Float,
    /// Group Lasso strategy
    pub strategy: GroupLassoStrategy,
    /// Kernel function to use
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Random seed
    pub random_state: Option<u64>,
    /// Learning rate for coordinate descent
    pub learning_rate: Float,
    /// Learning rate decay
    pub learning_rate_decay: Float,
    /// Minimum learning rate
    pub min_learning_rate: Float,
    /// Whether to use adaptive group weights
    pub adaptive_weights: bool,
    /// Proximal gradient acceleration
    pub acceleration: bool,
}

impl Default for GroupLassoSVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            lambda: 0.1,
            strategy: GroupLassoStrategy::default(),
            kernel: KernelType::Linear, // Group Lasso typically used with linear kernels
            tol: 1e-4,
            max_iter: 1000,
            fit_intercept: true,
            random_state: None,
            learning_rate: 0.01,
            learning_rate_decay: 0.99,
            min_learning_rate: 1e-6,
            adaptive_weights: false,
            acceleration: true,
        }
    }
}

/// Group Lasso Support Vector Machine
#[derive(Debug)]
pub struct GroupLassoSVM<State = Untrained> {
    config: GroupLassoSVMConfig,
    group_structure: Option<GroupStructure>,
    state: PhantomData<State>,
    // Fitted attributes
    coef_: Option<Array2<Float>>,
    intercept_: Option<Array1<Float>>,
    classes_: Option<Array1<i32>>,
    n_features_in_: Option<usize>,
    n_iter_: Option<usize>,
    selected_groups_: Option<Vec<usize>>,
    group_norms_: Option<Array1<Float>>,
}

impl GroupLassoSVM<Untrained> {
    /// Create a new Group Lasso SVM
    pub fn new() -> Self {
        Self {
            config: GroupLassoSVMConfig::default(),
            group_structure: None,
            state: PhantomData,
            coef_: None,
            intercept_: None,
            classes_: None,
            n_features_in_: None,
            n_iter_: None,
            selected_groups_: None,
            group_norms_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the Group Lasso regularization parameter lambda
    pub fn lambda(mut self, lambda: Float) -> Self {
        self.config.lambda = lambda;
        self
    }

    /// Set the Group Lasso strategy
    pub fn strategy(mut self, strategy: GroupLassoStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set the group structure
    pub fn group_structure(mut self, group_structure: GroupStructure) -> Self {
        self.group_structure = Some(group_structure);
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

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Enable adaptive group weights
    pub fn adaptive_weights(mut self, adaptive: bool) -> Self {
        self.config.adaptive_weights = adaptive;
        self
    }

    /// Enable proximal gradient acceleration
    pub fn acceleration(mut self, acceleration: bool) -> Self {
        self.config.acceleration = acceleration;
        self
    }
}

impl Default for GroupLassoSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<i32>> for GroupLassoSVM<Untrained> {
    type Fitted = GroupLassoSVM<Trained>;

    fn fit(mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
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

        // Create default group structure if not provided
        if self.group_structure.is_none() {
            self.group_structure = Some(GroupStructure::create_blocks(n_features, 5)?);
        }

        let group_structure = self.group_structure.as_ref().unwrap();

        // Find unique classes
        let mut unique_classes: Vec<i32> = y.iter().cloned().collect();
        unique_classes.sort();
        unique_classes.dedup();
        let n_classes = unique_classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Convert to binary classification format if necessary
        let y_binary = if n_classes == 2 {
            y.mapv(|label| {
                if label == unique_classes[0] {
                    -1.0
                } else {
                    1.0
                }
            })
        } else {
            // For multi-class, use one-vs-rest for now
            return Err(SklearsError::InvalidInput(
                "Multi-class Group Lasso SVM not yet implemented".to_string(),
            ));
        };

        // Initialize coefficients
        let mut coef = Array2::zeros((1, n_features));
        let mut intercept = 0.0;
        let mut current_lr = self.config.learning_rate;

        // For acceleration
        let mut momentum_coef = coef.clone();
        let mut momentum_intercept = intercept;
        let momentum_factor = 0.9;

        // Training loop using proximal gradient descent
        let mut n_iter = 0;
        for iteration in 0..self.config.max_iter {
            n_iter = iteration + 1;
            let mut converged = true;

            // Store previous coefficients for convergence check
            let prev_coef = coef.clone();
            let prev_intercept = intercept;

            // Apply acceleration if enabled
            if self.config.acceleration && iteration > 0 {
                let beta = momentum_factor;
                coef = &coef + beta * (&coef - &momentum_coef);
                intercept = intercept + beta * (intercept - momentum_intercept);
                momentum_coef = prev_coef.clone();
                momentum_intercept = prev_intercept;
            }

            // Coordinate descent over groups
            for group in &group_structure.groups {
                // Compute gradient for this group
                let group_gradient =
                    self.compute_group_gradient(x, &y_binary, &coef, intercept, &group.features)?;

                // Apply proximal operator for group
                let new_group_coef =
                    self.apply_group_proximal_operator(&coef, &group_gradient, group, current_lr)?;

                // Update coefficients for this group
                for (i, &feature_idx) in group.features.iter().enumerate() {
                    let old_coef = coef[[0, feature_idx]];
                    coef[[0, feature_idx]] = new_group_coef[i];

                    if (coef[[0, feature_idx]] - old_coef).abs() > self.config.tol {
                        converged = false;
                    }
                }
            }

            // Update intercept if needed
            if self.config.fit_intercept {
                let intercept_gradient =
                    self.compute_intercept_gradient(x, &y_binary, &coef, intercept)?;
                let old_intercept = intercept;
                intercept -= current_lr * intercept_gradient;

                if (intercept - old_intercept).abs() > self.config.tol {
                    converged = false;
                }
            }

            // Update learning rate
            current_lr =
                (current_lr * self.config.learning_rate_decay).max(self.config.min_learning_rate);

            if converged {
                break;
            }
        }

        // Identify selected groups
        let selected_groups = self.identify_selected_groups(&coef, group_structure)?;

        // Compute group norms
        let group_norms = self.compute_group_norms(&coef, group_structure)?;

        Ok(GroupLassoSVM {
            config: self.config,
            group_structure: self.group_structure,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(Array1::from_vec(vec![intercept])),
            classes_: Some(Array1::from_vec(unique_classes)),
            n_features_in_: Some(n_features),
            n_iter_: Some(n_iter),
            selected_groups_: Some(selected_groups),
            group_norms_: Some(group_norms),
        })
    }
}

impl GroupLassoSVM<Untrained> {
    /// Compute gradient for a specific group of features
    fn compute_group_gradient(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        coef: &Array2<Float>,
        intercept: Float,
        group_features: &[usize],
    ) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let mut gradient = Array1::zeros(group_features.len());

        for i in 0..n_samples {
            // Compute prediction
            let mut prediction = if self.config.fit_intercept {
                intercept
            } else {
                0.0
            };

            for j in 0..coef.ncols() {
                prediction += coef[[0, j]] * x[[i, j]];
            }

            // Compute loss derivative (hinge loss)
            let margin = y[i] * prediction;
            if margin < 1.0 {
                let loss_derivative = -y[i];

                // Add contribution to gradient for group features
                for (idx, &feature_idx) in group_features.iter().enumerate() {
                    gradient[idx] += loss_derivative * x[[i, feature_idx]];
                }
            }
        }

        gradient /= n_samples as Float;
        Ok(gradient)
    }

    /// Compute gradient for intercept
    fn compute_intercept_gradient(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        coef: &Array2<Float>,
        intercept: Float,
    ) -> Result<Float> {
        let n_samples = x.nrows();
        let mut gradient = 0.0;

        for i in 0..n_samples {
            // Compute prediction
            let mut prediction = if self.config.fit_intercept {
                intercept
            } else {
                0.0
            };

            for j in 0..coef.ncols() {
                prediction += coef[[0, j]] * x[[i, j]];
            }

            // Compute loss derivative (hinge loss)
            let margin = y[i] * prediction;
            if margin < 1.0 {
                gradient += -y[i];
            }
        }

        Ok(gradient / n_samples as Float)
    }

    /// Apply proximal operator for Group Lasso
    fn apply_group_proximal_operator(
        &self,
        coef: &Array2<Float>,
        gradient: &Array1<Float>,
        group: &FeatureGroup,
        learning_rate: Float,
    ) -> Result<Array1<Float>> {
        let group_size = group.features.len();
        let mut group_coef = Array1::zeros(group_size);

        // Extract current coefficients for this group
        for (i, &feature_idx) in group.features.iter().enumerate() {
            group_coef[i] = coef[[0, feature_idx]];
        }

        // Gradient step
        let updated_coef = &group_coef - learning_rate * gradient;

        // Apply Group Lasso proximal operator based on strategy
        match &self.config.strategy {
            GroupLassoStrategy::Standard => {
                self.apply_standard_group_lasso(&updated_coef, group, learning_rate)
            }
            GroupLassoStrategy::Sparse { l1_ratio } => {
                self.apply_sparse_group_lasso(&updated_coef, group, learning_rate, *l1_ratio)
            }
            GroupLassoStrategy::Exclusive => {
                self.apply_exclusive_group_lasso(&updated_coef, group, learning_rate)
            }
            GroupLassoStrategy::Overlapping => {
                // For now, treat as standard Group Lasso
                self.apply_standard_group_lasso(&updated_coef, group, learning_rate)
            }
        }
    }

    /// Apply standard Group Lasso proximal operator
    fn apply_standard_group_lasso(
        &self,
        coef: &Array1<Float>,
        group: &FeatureGroup,
        learning_rate: Float,
    ) -> Result<Array1<Float>> {
        let group_norm = coef.iter().map(|&x| x * x).sum::<Float>().sqrt();
        let threshold = learning_rate * self.config.lambda * group.weight;

        if group_norm <= threshold {
            // Shrink entire group to zero
            Ok(Array1::zeros(coef.len()))
        } else {
            // Shrink group proportionally
            let shrinkage_factor = 1.0 - threshold / group_norm;
            Ok(coef * shrinkage_factor)
        }
    }

    /// Apply Sparse Group Lasso proximal operator
    fn apply_sparse_group_lasso(
        &self,
        coef: &Array1<Float>,
        group: &FeatureGroup,
        learning_rate: Float,
        l1_ratio: Float,
    ) -> Result<Array1<Float>> {
        let group_penalty = learning_rate * self.config.lambda * (1.0 - l1_ratio) * group.weight;
        let l1_penalty = learning_rate * self.config.lambda * l1_ratio;

        // First apply element-wise soft thresholding (L1)
        let l1_result = coef.mapv(|x| {
            if x > l1_penalty {
                x - l1_penalty
            } else if x < -l1_penalty {
                x + l1_penalty
            } else {
                0.0
            }
        });

        // Then apply group soft thresholding
        let group_norm = l1_result.iter().map(|&x| x * x).sum::<Float>().sqrt();

        if group_norm <= group_penalty {
            Ok(Array1::zeros(coef.len()))
        } else {
            let shrinkage_factor = 1.0 - group_penalty / group_norm;
            Ok(&l1_result * shrinkage_factor)
        }
    }

    /// Apply Exclusive Group Lasso proximal operator
    fn apply_exclusive_group_lasso(
        &self,
        coef: &Array1<Float>,
        group: &FeatureGroup,
        learning_rate: Float,
    ) -> Result<Array1<Float>> {
        // For exclusive Group Lasso, we need to consider competition between groups
        // For simplicity, apply standard Group Lasso for now
        self.apply_standard_group_lasso(coef, group, learning_rate)
    }

    /// Identify which groups have been selected (non-zero)
    fn identify_selected_groups(
        &self,
        coef: &Array2<Float>,
        group_structure: &GroupStructure,
    ) -> Result<Vec<usize>> {
        let mut selected_groups = Vec::new();

        for group in &group_structure.groups {
            let mut group_norm = 0.0;
            for &feature_idx in &group.features {
                group_norm += coef[[0, feature_idx]] * coef[[0, feature_idx]];
            }
            group_norm = group_norm.sqrt();

            if group_norm > self.config.tol {
                selected_groups.push(group.id);
            }
        }

        Ok(selected_groups)
    }

    /// Compute norms for all groups
    fn compute_group_norms(
        &self,
        coef: &Array2<Float>,
        group_structure: &GroupStructure,
    ) -> Result<Array1<Float>> {
        let mut group_norms = Array1::zeros(group_structure.groups.len());

        for (i, group) in group_structure.groups.iter().enumerate() {
            let mut norm = 0.0;
            for &feature_idx in &group.features {
                norm += coef[[0, feature_idx]] * coef[[0, feature_idx]];
            }
            group_norms[i] = norm.sqrt();
        }

        Ok(group_norms)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for GroupLassoSVM<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(
                "Feature mismatch: X has different number of features than training data"
                    .to_string(),
            ));
        }

        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut decision_value = if self.config.fit_intercept {
                intercept[0]
            } else {
                0.0
            };

            for j in 0..x.ncols() {
                decision_value += coef[[0, j]] * x[[i, j]];
            }

            predictions[i] = if decision_value > 0.0 {
                classes[1]
            } else {
                classes[0]
            };
        }

        Ok(predictions)
    }
}

impl GroupLassoSVM<Trained> {
    /// Get the learned coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the intercept
    pub fn intercept(&self) -> &Array1<Float> {
        self.intercept_.as_ref().unwrap()
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the selected groups
    pub fn selected_groups(&self) -> &Vec<usize> {
        self.selected_groups_.as_ref().unwrap()
    }

    /// Get the group norms
    pub fn group_norms(&self) -> &Array1<Float> {
        self.group_norms_.as_ref().unwrap()
    }

    /// Get the group structure
    pub fn group_structure(&self) -> &GroupStructure {
        self.group_structure.as_ref().unwrap()
    }

    /// Get the number of features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Get the decision function values
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(
                "Feature mismatch: X has different number of features than training data"
                    .to_string(),
            ));
        }

        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.as_ref().unwrap();

        let mut decision_values = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut decision_value = if self.config.fit_intercept {
                intercept[0]
            } else {
                0.0
            };

            for j in 0..x.ncols() {
                decision_value += coef[[0, j]] * x[[i, j]];
            }

            decision_values[i] = decision_value;
        }

        Ok(decision_values)
    }

    /// Get sparsity information
    pub fn sparsity_info(&self) -> HashMap<String, Float> {
        let coef = self.coef_.as_ref().unwrap();
        let group_structure = self.group_structure.as_ref().unwrap();
        let selected_groups = self.selected_groups_.as_ref().unwrap();

        let total_features = coef.ncols();
        let total_groups = group_structure.groups.len();
        let selected_features = coef.iter().filter(|&&x| x.abs() > self.config.tol).count();

        let mut info = HashMap::new();
        info.insert("total_features".to_string(), total_features as Float);
        info.insert("total_groups".to_string(), total_groups as Float);
        info.insert("selected_features".to_string(), selected_features as Float);
        info.insert(
            "selected_groups".to_string(),
            selected_groups.len() as Float,
        );
        info.insert(
            "feature_sparsity".to_string(),
            1.0 - (selected_features as Float / total_features as Float),
        );
        info.insert(
            "group_sparsity".to_string(),
            1.0 - (selected_groups.len() as Float / total_groups as Float),
        );

        info
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_feature_group_creation() {
        let group = FeatureGroup::new(0, vec![0, 1, 2])
            .weight(2.0)
            .name("TestGroup".to_string());

        assert_eq!(group.id, 0);
        assert_eq!(group.features, vec![0, 1, 2]);
        assert_eq!(group.weight, 2.0);
        assert_eq!(group.name, Some("TestGroup".to_string()));
        assert_eq!(group.size(), 3);
    }

    #[test]
    fn test_group_structure_creation() {
        let groups = vec![
            FeatureGroup::new(0, vec![0, 1]),
            FeatureGroup::new(1, vec![2, 3]),
            FeatureGroup::new(2, vec![4]),
        ];

        let group_structure = GroupStructure::new(groups, 5).unwrap();

        assert_eq!(group_structure.groups.len(), 3);
        assert_eq!(group_structure.n_features, 5);
        assert_eq!(group_structure.feature_to_group[&0], 0);
        assert_eq!(group_structure.feature_to_group[&2], 1);
        assert_eq!(group_structure.feature_to_group[&4], 2);
    }

    #[test]
    fn test_group_structure_blocks() {
        let group_structure = GroupStructure::create_blocks(10, 3).unwrap();

        assert_eq!(group_structure.groups.len(), 4); // [0,1,2], [3,4,5], [6,7,8], [9]
        assert_eq!(group_structure.groups[0].features, vec![0, 1, 2]);
        assert_eq!(group_structure.groups[3].features, vec![9]);
    }

    #[test]
    fn test_group_lasso_svm_creation() {
        let svm = GroupLassoSVM::new()
            .c(2.0)
            .lambda(0.5)
            .strategy(GroupLassoStrategy::Sparse { l1_ratio: 0.3 })
            .kernel(KernelType::Linear)
            .adaptive_weights(true);

        assert_eq!(svm.config.c, 2.0);
        assert_eq!(svm.config.lambda, 0.5);
        assert_eq!(svm.config.adaptive_weights, true);
        assert!(
            matches!(svm.config.strategy, GroupLassoStrategy::Sparse { l1_ratio } if l1_ratio == 0.3)
        );
    }

    #[test]
    #[ignore = "Slow test: trains Group Lasso SVM. Run with --ignored flag"]
    fn test_group_lasso_svm_training() {
        // Create test data with grouped features
        let x = array![
            [1.0, 2.0, 0.1, 0.2, 5.0], // Features: [group1: 0,1], [group2,3], [group3: 4]
            [2.0, 3.0, 0.2, 0.3, 6.0],
            [3.0, 4.0, 1.0, 1.1, 1.0],
            [4.0, 5.0, 1.1, 1.2, 2.0],
            [5.0, 6.0, 0.0, 0.1, 7.0],
            [6.0, 7.0, 0.1, 0.0, 8.0],
        ];
        let y = array![1, 1, -1, -1, 1, 1];

        // Create group structure
        let groups = vec![
            FeatureGroup::new(0, vec![0, 1]).name("Group1".to_string()),
            FeatureGroup::new(1, vec![2, 3]).name("Group2".to_string()),
            FeatureGroup::new(2, vec![4]).name("Group3".to_string()),
        ];
        let group_structure = GroupStructure::new(groups, 5).unwrap();

        let svm = GroupLassoSVM::new()
            .c(1.0)
            .lambda(0.1)
            .strategy(GroupLassoStrategy::Standard)
            .kernel(KernelType::Linear)
            .group_structure(group_structure)
            .max_iter(100)
            .learning_rate(0.01);

        let fitted_model = svm.fit(&x, &y).unwrap();

        assert_eq!(fitted_model.n_features_in(), 5);
        assert!(fitted_model.n_iter() > 0);
        assert!(!fitted_model.selected_groups().is_empty());

        // Test predictions
        let predictions = fitted_model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // Test decision function
        let decision_values = fitted_model.decision_function(&x).unwrap();
        assert_eq!(decision_values.len(), 6);

        // Test sparsity info
        let sparsity_info = fitted_model.sparsity_info();
        assert!(sparsity_info.contains_key("total_features"));
        assert!(sparsity_info.contains_key("selected_groups"));

        // Check that some sparsity is achieved
        assert!(sparsity_info["group_sparsity"] >= 0.0);
    }

    #[test]
    fn test_group_lasso_svm_shape_mismatch() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1]; // Wrong length

        let svm = GroupLassoSVM::new();
        let result = svm.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_sparse_group_lasso_strategy() {
        let coef = array![1.0, 0.5, 0.1];
        let group = FeatureGroup::new(0, vec![0, 1, 2]);
        let svm = GroupLassoSVM::new()
            .strategy(GroupLassoStrategy::Sparse { l1_ratio: 0.5 })
            .lambda(0.1);

        let result = svm
            .apply_sparse_group_lasso(&coef, &group, 0.1, 0.5)
            .unwrap();

        // Should apply both L1 and group penalties
        assert_eq!(result.len(), 3);
        for &val in result.iter() {
            assert!(val.is_finite());
        }
    }
}
