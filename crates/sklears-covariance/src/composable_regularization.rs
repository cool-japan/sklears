//! Composable Regularization Strategies
//!
//! This module provides a framework for composing and combining different regularization
//! strategies for covariance estimation. It allows users to easily mix and match
//! regularization techniques and create custom combinations.

use scirs2_core::ndarray::Array2;
use sklears_core::error::SklearsError;
use std::collections::HashMap;
use std::fmt::Debug;

/// Trait for regularization strategies
pub trait RegularizationStrategy: Debug + Send + Sync {
    /// Apply regularization to a covariance matrix
    fn apply(&self, covariance: &Array2<f64>, lambda: f64) -> Result<Array2<f64>, SklearsError>;

    /// Get the name of the regularization strategy
    fn name(&self) -> &'static str;

    /// Get hyperparameters specific to this strategy
    fn hyperparameters(&self) -> HashMap<String, f64>;

    /// Set hyperparameters for this strategy
    fn set_hyperparameters(&mut self, params: HashMap<String, f64>) -> Result<(), SklearsError>;

    /// Validate that the strategy can be applied to the given matrix dimensions
    fn validate(&self, n_features: usize) -> Result<(), SklearsError>;

    /// Compute the regularization penalty for the given covariance matrix
    fn penalty(&self, covariance: &Array2<f64>, lambda: f64) -> Result<f64, SklearsError>;

    /// Clone the strategy into a boxed trait object
    fn clone_box(&self) -> Box<dyn RegularizationStrategy>;
}

/// L1 regularization (Lasso)
#[derive(Debug, Clone)]
pub struct L1Regularization {
    /// Whether to regularize diagonal elements
    pub regularize_diagonal: bool,
    /// Weights for different matrix elements
    pub element_weights: Option<Array2<f64>>,
}

impl Default for L1Regularization {
    fn default() -> Self {
        Self::new()
    }
}

impl L1Regularization {
    /// Create a new L1 regularization strategy
    pub fn new() -> Self {
        L1Regularization {
            regularize_diagonal: false,
            element_weights: None,
        }
    }

    /// Set whether to regularize diagonal elements
    pub fn regularize_diagonal(mut self, regularize: bool) -> Self {
        self.regularize_diagonal = regularize;
        self
    }

    /// Set element-wise weights
    pub fn element_weights(mut self, weights: Array2<f64>) -> Self {
        self.element_weights = Some(weights);
        self
    }
}

impl RegularizationStrategy for L1Regularization {
    fn apply(&self, covariance: &Array2<f64>, lambda: f64) -> Result<Array2<f64>, SklearsError> {
        let mut regularized = covariance.clone();
        let (n, m) = covariance.dim();

        if n != m {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        for i in 0..n {
            for j in 0..m {
                if i == j && !self.regularize_diagonal {
                    continue;
                }

                let weight = self
                    .element_weights
                    .as_ref()
                    .map(|w| w[[i, j]])
                    .unwrap_or(1.0);

                let threshold = lambda * weight;
                let value = covariance[[i, j]];

                // Soft thresholding
                regularized[[i, j]] = if value > threshold {
                    value - threshold
                } else if value < -threshold {
                    value + threshold
                } else {
                    0.0
                };
            }
        }

        Ok(regularized)
    }

    fn name(&self) -> &'static str {
        "L1"
    }

    fn hyperparameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert(
            "regularize_diagonal".to_string(),
            if self.regularize_diagonal { 1.0 } else { 0.0 },
        );
        params
    }

    fn set_hyperparameters(&mut self, params: HashMap<String, f64>) -> Result<(), SklearsError> {
        if let Some(&value) = params.get("regularize_diagonal") {
            self.regularize_diagonal = value > 0.5;
        }
        Ok(())
    }

    fn validate(&self, n_features: usize) -> Result<(), SklearsError> {
        if let Some(ref weights) = self.element_weights {
            if weights.dim() != (n_features, n_features) {
                return Err(SklearsError::InvalidInput(
                    "Element weights dimension mismatch".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn penalty(&self, covariance: &Array2<f64>, lambda: f64) -> Result<f64, SklearsError> {
        let mut penalty = 0.0;
        let (n, m) = covariance.dim();

        for i in 0..n {
            for j in 0..m {
                if i == j && !self.regularize_diagonal {
                    continue;
                }

                let weight = self
                    .element_weights
                    .as_ref()
                    .map(|w| w[[i, j]])
                    .unwrap_or(1.0);

                penalty += lambda * weight * covariance[[i, j]].abs();
            }
        }

        Ok(penalty)
    }

    fn clone_box(&self) -> Box<dyn RegularizationStrategy> {
        Box::new(self.clone())
    }
}

/// L2 regularization (Ridge)
#[derive(Debug, Clone)]
pub struct L2Regularization {
    /// Whether to regularize diagonal elements
    pub regularize_diagonal: bool,
    /// Weights for different matrix elements
    pub element_weights: Option<Array2<f64>>,
}

impl Default for L2Regularization {
    fn default() -> Self {
        Self::new()
    }
}

impl L2Regularization {
    /// Create a new L2 regularization strategy
    pub fn new() -> Self {
        L2Regularization {
            regularize_diagonal: true,
            element_weights: None,
        }
    }

    /// Set whether to regularize diagonal elements
    pub fn regularize_diagonal(mut self, regularize: bool) -> Self {
        self.regularize_diagonal = regularize;
        self
    }

    /// Set element-wise weights
    pub fn element_weights(mut self, weights: Array2<f64>) -> Self {
        self.element_weights = Some(weights);
        self
    }
}

impl RegularizationStrategy for L2Regularization {
    fn apply(&self, covariance: &Array2<f64>, lambda: f64) -> Result<Array2<f64>, SklearsError> {
        let mut regularized = covariance.clone();
        let (n, m) = covariance.dim();

        if n != m {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        // Add ridge regularization
        for i in 0..n {
            if self.regularize_diagonal {
                let weight = self
                    .element_weights
                    .as_ref()
                    .map(|w| w[[i, i]])
                    .unwrap_or(1.0);
                regularized[[i, i]] += lambda * weight;
            }
        }

        Ok(regularized)
    }

    fn name(&self) -> &'static str {
        "L2"
    }

    fn hyperparameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert(
            "regularize_diagonal".to_string(),
            if self.regularize_diagonal { 1.0 } else { 0.0 },
        );
        params
    }

    fn set_hyperparameters(&mut self, params: HashMap<String, f64>) -> Result<(), SklearsError> {
        if let Some(&value) = params.get("regularize_diagonal") {
            self.regularize_diagonal = value > 0.5;
        }
        Ok(())
    }

    fn validate(&self, n_features: usize) -> Result<(), SklearsError> {
        if let Some(ref weights) = self.element_weights {
            if weights.dim() != (n_features, n_features) {
                return Err(SklearsError::InvalidInput(
                    "Element weights dimension mismatch".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn penalty(&self, covariance: &Array2<f64>, lambda: f64) -> Result<f64, SklearsError> {
        let mut penalty = 0.0;
        let (n, _) = covariance.dim();

        for i in 0..n {
            if self.regularize_diagonal {
                let weight = self
                    .element_weights
                    .as_ref()
                    .map(|w| w[[i, i]])
                    .unwrap_or(1.0);
                penalty += lambda * weight * covariance[[i, i]].powi(2);
            }
        }

        Ok(penalty)
    }

    fn clone_box(&self) -> Box<dyn RegularizationStrategy> {
        Box::new(self.clone())
    }
}

/// Nuclear norm regularization
#[derive(Debug, Clone)]
pub struct NuclearNormRegularization {
    /// Rank constraint (if any)
    pub rank_constraint: Option<usize>,
    /// Tolerance for singular value thresholding
    pub svd_tolerance: f64,
}

impl Default for NuclearNormRegularization {
    fn default() -> Self {
        Self::new()
    }
}

impl NuclearNormRegularization {
    /// Create a new nuclear norm regularization strategy
    pub fn new() -> Self {
        NuclearNormRegularization {
            rank_constraint: None,
            svd_tolerance: 1e-12,
        }
    }

    /// Set rank constraint
    pub fn rank_constraint(mut self, rank: Option<usize>) -> Self {
        self.rank_constraint = rank;
        self
    }

    /// Set SVD tolerance
    pub fn svd_tolerance(mut self, tolerance: f64) -> Self {
        self.svd_tolerance = tolerance;
        self
    }
}

impl RegularizationStrategy for NuclearNormRegularization {
    fn apply(&self, covariance: &Array2<f64>, lambda: f64) -> Result<Array2<f64>, SklearsError> {
        let (n, m) = covariance.dim();

        if n != m {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        // Simplified nuclear norm regularization using SVD
        // In practice, this would use proper SVD decomposition
        let mut regularized = covariance.clone();

        // Simulate singular value thresholding
        let threshold = lambda;
        for i in 0..n {
            for j in 0..m {
                if covariance[[i, j]].abs() < threshold {
                    regularized[[i, j]] = 0.0;
                }
            }
        }

        Ok(regularized)
    }

    fn name(&self) -> &'static str {
        "NuclearNorm"
    }

    fn hyperparameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("svd_tolerance".to_string(), self.svd_tolerance);
        if let Some(rank) = self.rank_constraint {
            params.insert("rank_constraint".to_string(), rank as f64);
        }
        params
    }

    fn set_hyperparameters(&mut self, params: HashMap<String, f64>) -> Result<(), SklearsError> {
        if let Some(&value) = params.get("svd_tolerance") {
            self.svd_tolerance = value;
        }
        if let Some(&value) = params.get("rank_constraint") {
            self.rank_constraint = Some(value as usize);
        }
        Ok(())
    }

    fn validate(&self, n_features: usize) -> Result<(), SklearsError> {
        if let Some(rank) = self.rank_constraint {
            if rank > n_features {
                return Err(SklearsError::InvalidInput(
                    "Rank constraint exceeds matrix dimension".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn penalty(&self, covariance: &Array2<f64>, lambda: f64) -> Result<f64, SklearsError> {
        // Simplified nuclear norm calculation (sum of absolute values as approximation)
        let nuclear_norm = covariance.iter().map(|&x| x.abs()).sum::<f64>();
        Ok(lambda * nuclear_norm)
    }

    fn clone_box(&self) -> Box<dyn RegularizationStrategy> {
        Box::new(self.clone())
    }
}

/// Group Lasso regularization
#[derive(Debug, Clone)]
pub struct GroupLassoRegularization {
    /// Group assignments for variables
    pub groups: Vec<usize>,
    /// Group weights
    pub group_weights: Option<HashMap<usize, f64>>,
}

impl GroupLassoRegularization {
    /// Create a new group lasso regularization strategy
    pub fn new(groups: Vec<usize>) -> Self {
        GroupLassoRegularization {
            groups,
            group_weights: None,
        }
    }

    /// Set group weights
    pub fn group_weights(mut self, weights: HashMap<usize, f64>) -> Self {
        self.group_weights = Some(weights);
        self
    }
}

impl RegularizationStrategy for GroupLassoRegularization {
    fn apply(&self, covariance: &Array2<f64>, lambda: f64) -> Result<Array2<f64>, SklearsError> {
        let mut regularized = covariance.clone();
        let (n, m) = covariance.dim();

        if n != m {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        if self.groups.len() != n {
            return Err(SklearsError::InvalidInput(
                "Groups length must match matrix dimension".to_string(),
            ));
        }

        // Group-wise soft thresholding
        let unique_groups: std::collections::HashSet<_> = self.groups.iter().collect();

        for &group_id in &unique_groups {
            let group_indices: Vec<usize> = self
                .groups
                .iter()
                .enumerate()
                .filter(|(_, &g)| g == *group_id)
                .map(|(i, _)| i)
                .collect();

            let group_weight = self
                .group_weights
                .as_ref()
                .and_then(|w| w.get(group_id))
                .copied()
                .unwrap_or(1.0);

            // Calculate group norm
            let mut group_norm = 0.0;
            for &i in &group_indices {
                for &j in &group_indices {
                    group_norm += covariance[[i, j]].powi(2);
                }
            }
            group_norm = group_norm.sqrt();

            let threshold = lambda * group_weight;
            let shrinkage_factor = if group_norm > threshold {
                1.0 - threshold / group_norm
            } else {
                0.0
            };

            // Apply shrinkage to group
            for &i in &group_indices {
                for &j in &group_indices {
                    regularized[[i, j]] *= shrinkage_factor;
                }
            }
        }

        Ok(regularized)
    }

    fn name(&self) -> &'static str {
        "GroupLasso"
    }

    fn hyperparameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert(
            "n_groups".to_string(),
            (*self.groups.iter().max().unwrap_or(&0) + 1) as f64,
        );
        params
    }

    fn set_hyperparameters(&mut self, _params: HashMap<String, f64>) -> Result<(), SklearsError> {
        // Group structure is typically not changed via hyperparameters
        Ok(())
    }

    fn validate(&self, n_features: usize) -> Result<(), SklearsError> {
        if self.groups.len() != n_features {
            return Err(SklearsError::InvalidInput(
                "Groups length must match number of features".to_string(),
            ));
        }
        Ok(())
    }

    fn penalty(&self, covariance: &Array2<f64>, lambda: f64) -> Result<f64, SklearsError> {
        let unique_groups: std::collections::HashSet<_> = self.groups.iter().collect();
        let mut penalty = 0.0;

        for &group_id in &unique_groups {
            let group_indices: Vec<usize> = self
                .groups
                .iter()
                .enumerate()
                .filter(|(_, &g)| g == *group_id)
                .map(|(i, _)| i)
                .collect();

            let group_weight = self
                .group_weights
                .as_ref()
                .and_then(|w| w.get(group_id))
                .copied()
                .unwrap_or(1.0);

            // Calculate group norm
            let mut group_norm = 0.0;
            for &i in &group_indices {
                for &j in &group_indices {
                    group_norm += covariance[[i, j]].powi(2);
                }
            }
            group_norm = group_norm.sqrt();

            penalty += lambda * group_weight * group_norm;
        }

        Ok(penalty)
    }

    fn clone_box(&self) -> Box<dyn RegularizationStrategy> {
        Box::new(self.clone())
    }
}

/// Composite regularization strategy
#[derive(Debug)]
pub struct CompositeRegularization {
    /// List of regularization strategies with their weights
    pub strategies: Vec<(Box<dyn RegularizationStrategy>, f64)>,
    /// Combination method
    pub combination_method: CombinationMethod,
}

impl Clone for CompositeRegularization {
    fn clone(&self) -> Self {
        let strategies = self
            .strategies
            .iter()
            .map(|(strategy, weight)| (strategy.clone_box(), *weight))
            .collect();
        CompositeRegularization {
            strategies,
            combination_method: self.combination_method,
        }
    }
}

/// Methods for combining regularization strategies
#[derive(Debug, Clone, Copy)]
pub enum CombinationMethod {
    /// Weighted sum of regularizations
    WeightedSum,
    /// Sequential application
    Sequential,
    /// Alternating application
    Alternating,
    /// Multiplicative combination
    Multiplicative,
}

impl CompositeRegularization {
    /// Create a new composite regularization strategy
    pub fn new() -> Self {
        CompositeRegularization {
            strategies: Vec::new(),
            combination_method: CombinationMethod::WeightedSum,
        }
    }

    /// Add a regularization strategy with weight
    pub fn add_strategy<T: RegularizationStrategy + 'static>(
        mut self,
        strategy: T,
        weight: f64,
    ) -> Self {
        self.strategies.push((Box::new(strategy), weight));
        self
    }

    /// Set combination method
    pub fn combination_method(mut self, method: CombinationMethod) -> Self {
        self.combination_method = method;
        self
    }
}

impl Default for CompositeRegularization {
    fn default() -> Self {
        Self::new()
    }
}

impl RegularizationStrategy for CompositeRegularization {
    fn apply(&self, covariance: &Array2<f64>, lambda: f64) -> Result<Array2<f64>, SklearsError> {
        if self.strategies.is_empty() {
            return Ok(covariance.clone());
        }

        match self.combination_method {
            CombinationMethod::WeightedSum => {
                let mut result = covariance.clone();
                let mut total_weight = 0.0;

                for (strategy, weight) in &self.strategies {
                    let regularized = strategy.apply(covariance, lambda * weight)?;
                    let effective_weight = weight / self.strategies.len() as f64;

                    if total_weight == 0.0 {
                        result = regularized * effective_weight;
                    } else {
                        result = result + regularized * effective_weight;
                    }
                    total_weight += effective_weight;
                }

                if total_weight > 0.0 {
                    result = result / total_weight;
                }

                Ok(result)
            }
            CombinationMethod::Sequential => {
                let mut result = covariance.clone();

                for (strategy, weight) in &self.strategies {
                    result = strategy.apply(&result, lambda * weight)?;
                }

                Ok(result)
            }
            CombinationMethod::Alternating => {
                // For simplicity, use sequential for now
                self.apply(covariance, lambda)
            }
            CombinationMethod::Multiplicative => {
                let mut result = Array2::ones(covariance.dim());

                for (strategy, weight) in &self.strategies {
                    let regularized = strategy.apply(covariance, lambda * weight)?;
                    result = result * regularized;
                }

                Ok(result)
            }
        }
    }

    fn name(&self) -> &'static str {
        "Composite"
    }

    fn hyperparameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_strategies".to_string(), self.strategies.len() as f64);

        for (i, (strategy, weight)) in self.strategies.iter().enumerate() {
            params.insert(format!("strategy_{}_weight", i), *weight);
            for (key, value) in strategy.hyperparameters() {
                params.insert(format!("strategy_{}_{}", i, key), value);
            }
        }

        params
    }

    fn set_hyperparameters(&mut self, params: HashMap<String, f64>) -> Result<(), SklearsError> {
        for (i, (strategy, weight)) in self.strategies.iter_mut().enumerate() {
            if let Some(&new_weight) = params.get(&format!("strategy_{}_weight", i)) {
                *weight = new_weight;
            }

            // Extract strategy-specific parameters
            let strategy_params: HashMap<String, f64> = params
                .iter()
                .filter_map(|(key, &value)| {
                    let prefix = format!("strategy_{}_", i);
                    if key.starts_with(&prefix) && !key.ends_with("_weight") {
                        let param_name = key.strip_prefix(&prefix).unwrap();
                        Some((param_name.to_string(), value))
                    } else {
                        None
                    }
                })
                .collect();

            if !strategy_params.is_empty() {
                strategy.set_hyperparameters(strategy_params)?;
            }
        }

        Ok(())
    }

    fn validate(&self, n_features: usize) -> Result<(), SklearsError> {
        for (strategy, _) in &self.strategies {
            strategy.validate(n_features)?;
        }
        Ok(())
    }

    fn penalty(&self, covariance: &Array2<f64>, lambda: f64) -> Result<f64, SklearsError> {
        let mut total_penalty = 0.0;

        for (strategy, weight) in &self.strategies {
            total_penalty += strategy.penalty(covariance, lambda * weight)?;
        }

        Ok(total_penalty)
    }

    fn clone_box(&self) -> Box<dyn RegularizationStrategy> {
        Box::new(self.clone())
    }
}

/// Regularization factory for creating common combinations
pub struct RegularizationFactory;

impl RegularizationFactory {
    /// Create Elastic Net regularization (L1 + L2)
    pub fn elastic_net(l1_ratio: f64) -> CompositeRegularization {
        let l1_weight = l1_ratio;
        let l2_weight = 1.0 - l1_ratio;

        CompositeRegularization::new()
            .add_strategy(L1Regularization::new(), l1_weight)
            .add_strategy(L2Regularization::new(), l2_weight)
            .combination_method(CombinationMethod::WeightedSum)
    }

    /// Create sparse low-rank regularization (L1 + Nuclear Norm)
    pub fn sparse_low_rank(sparsity_ratio: f64) -> CompositeRegularization {
        let l1_weight = sparsity_ratio;
        let nuclear_weight = 1.0 - sparsity_ratio;

        CompositeRegularization::new()
            .add_strategy(L1Regularization::new(), l1_weight)
            .add_strategy(NuclearNormRegularization::new(), nuclear_weight)
            .combination_method(CombinationMethod::WeightedSum)
    }

    /// Create group-sparse regularization
    pub fn group_sparse(groups: Vec<usize>, group_ratio: f64) -> CompositeRegularization {
        let group_weight = group_ratio;
        let l1_weight = 1.0 - group_ratio;

        CompositeRegularization::new()
            .add_strategy(GroupLassoRegularization::new(groups), group_weight)
            .add_strategy(L1Regularization::new(), l1_weight)
            .combination_method(CombinationMethod::WeightedSum)
    }

    /// Create adaptive regularization that changes weights based on data
    pub fn adaptive(
        base_strategy: Box<dyn RegularizationStrategy>,
        covariance: &Array2<f64>,
    ) -> Result<CompositeRegularization, SklearsError> {
        // Simplified adaptive weights based on matrix properties
        let trace = covariance.diag().sum();
        let frobenius_norm = covariance.iter().map(|&x| x * x).sum::<f64>().sqrt();

        let adaptive_weight = if frobenius_norm > trace {
            0.8 // More regularization for ill-conditioned matrices
        } else {
            0.2 // Less regularization for well-conditioned matrices
        };

        let mut composite = CompositeRegularization::new();
        composite.strategies.push((base_strategy, adaptive_weight));

        Ok(composite)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_l1_regularization() {
        let covariance = array![[2.0, 1.5, 0.5], [1.5, 3.0, 0.8], [0.5, 0.8, 1.5]];

        let l1_reg = L1Regularization::new();
        let result = l1_reg.apply(&covariance, 0.5).unwrap();

        // Check that small values are thresholded to zero
        assert_eq!(result[[0, 2]], 0.0);
        assert_eq!(result[[2, 0]], 0.0);

        // Check that larger values are shrunk
        assert!(result[[0, 1]] < covariance[[0, 1]]);
        assert!(result[[1, 0]] < covariance[[1, 0]]);
    }

    #[test]
    fn test_l2_regularization() {
        let covariance = array![[2.0, 1.5, 0.5], [1.5, 3.0, 0.8], [0.5, 0.8, 1.5]];

        let l2_reg = L2Regularization::new();
        let result = l2_reg.apply(&covariance, 0.5).unwrap();

        // Check that diagonal elements are increased (ridge regularization)
        assert!(result[[0, 0]] > covariance[[0, 0]]);
        assert!(result[[1, 1]] > covariance[[1, 1]]);
        assert!(result[[2, 2]] > covariance[[2, 2]]);

        // Check that off-diagonal elements are unchanged
        assert_eq!(result[[0, 1]], covariance[[0, 1]]);
        assert_eq!(result[[1, 0]], covariance[[1, 0]]);
    }

    #[test]
    fn test_group_lasso_regularization() {
        let covariance = array![[2.0, 1.5, 0.5], [1.5, 3.0, 0.8], [0.5, 0.8, 1.5]];

        let groups = vec![0, 0, 1]; // First two variables in group 0, third in group 1
        let group_lasso = GroupLassoRegularization::new(groups);
        let result = group_lasso.apply(&covariance, 0.1).unwrap();

        // Result should have proper dimensions
        assert_eq!(result.dim(), (3, 3));
    }

    #[test]
    fn test_composite_regularization() {
        let covariance = array![[2.0, 1.5, 0.5], [1.5, 3.0, 0.8], [0.5, 0.8, 1.5]];

        let composite = CompositeRegularization::new()
            .add_strategy(L1Regularization::new(), 0.5)
            .add_strategy(L2Regularization::new(), 0.5)
            .combination_method(CombinationMethod::WeightedSum);

        let result = composite.apply(&covariance, 0.2).unwrap();

        // Result should have proper dimensions
        assert_eq!(result.dim(), (3, 3));

        // Should combine effects of both regularizations
        assert!(result[[0, 0]] != covariance[[0, 0]]);
    }

    #[test]
    fn test_elastic_net_factory() {
        let elastic_net = RegularizationFactory::elastic_net(0.5);

        assert_eq!(elastic_net.strategies.len(), 2);
        assert_eq!(elastic_net.strategies[0].1, 0.5); // L1 weight
        assert_eq!(elastic_net.strategies[1].1, 0.5); // L2 weight
    }

    #[test]
    fn test_hyperparameter_management() {
        let mut l1_reg = L1Regularization::new();

        let initial_params = l1_reg.hyperparameters();
        assert_eq!(initial_params["regularize_diagonal"], 0.0);

        let mut new_params = HashMap::new();
        new_params.insert("regularize_diagonal".to_string(), 1.0);

        l1_reg.set_hyperparameters(new_params).unwrap();

        let updated_params = l1_reg.hyperparameters();
        assert_eq!(updated_params["regularize_diagonal"], 1.0);
        assert!(l1_reg.regularize_diagonal);
    }
}
