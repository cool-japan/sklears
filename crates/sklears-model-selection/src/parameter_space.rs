//! Enhanced parameter space definitions with categorical parameter handling
//!
//! This module provides advanced parameter space definitions including conditional parameters,
//! constraints, and dependency handling for sophisticated hyperparameter optimization.

use crate::grid_search::{ParameterSet, ParameterValue};
use scirs2_core::rand_prelude::IndexedRandom;
use scirs2_core::random::prelude::*;
use sklears_core::error::{Result, SklearsError};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Categorical parameter definition with enhanced features
#[derive(Debug, Clone)]
pub struct CategoricalParameter {
    /// Name of the parameter
    pub name: String,
    /// Possible values
    pub values: Vec<ParameterValue>,
    /// Whether the categories have an ordering
    pub ordered: bool,
    /// Default value if any
    pub default: Option<ParameterValue>,
    /// Description of the parameter
    pub description: Option<String>,
}

impl CategoricalParameter {
    /// Create a new categorical parameter
    pub fn new(name: String, values: Vec<ParameterValue>) -> Self {
        Self {
            name,
            values,
            ordered: false,
            default: None,
            description: None,
        }
    }

    /// Create an ordered categorical parameter
    pub fn ordered(name: String, values: Vec<ParameterValue>) -> Self {
        Self {
            name,
            values,
            ordered: true,
            default: None,
            description: None,
        }
    }

    /// Set default value
    pub fn with_default(mut self, default: ParameterValue) -> Self {
        self.default = Some(default);
        self
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Sample a random value from this parameter
    pub fn sample(&self, rng: &mut impl Rng) -> ParameterValue {
        self.values.choose(rng).unwrap().clone()
    }

    /// Get the index of a value (useful for ordered categories)
    pub fn get_index(&self, value: &ParameterValue) -> Option<usize> {
        self.values.iter().position(|v| v == value)
    }

    /// Get neighboring values for ordered categories
    pub fn get_neighbors(&self, value: &ParameterValue) -> Vec<ParameterValue> {
        if !self.ordered {
            return vec![];
        }

        if let Some(idx) = self.get_index(value) {
            let mut neighbors = Vec::new();
            if idx > 0 {
                neighbors.push(self.values[idx - 1].clone());
            }
            if idx + 1 < self.values.len() {
                neighbors.push(self.values[idx + 1].clone());
            }
            neighbors
        } else {
            vec![]
        }
    }
}

/// Parameter constraint type
#[derive(Clone)]
pub enum ParameterConstraint {
    /// Equality constraint: param1 == value when param2 == condition
    Equality {
        param: String,

        value: ParameterValue,

        condition_param: String,

        condition_value: ParameterValue,
    },
    /// Inequality constraint: param1 != value when param2 == condition
    Inequality {
        param: String,
        value: ParameterValue,
        condition_param: String,
        condition_value: ParameterValue,
    },
    /// Range constraint: param1 in range when param2 == condition
    Range {
        param: String,
        min_value: ParameterValue,
        max_value: ParameterValue,
        condition_param: String,
        condition_value: ParameterValue,
    },
    /// Mutual exclusion: if param1 == value1, then param2 != value2
    MutualExclusion {
        param1: String,
        value1: ParameterValue,
        param2: String,
        value2: ParameterValue,
    },
    /// Custom constraint function
    Custom {
        name: String,
        constraint_fn: Arc<dyn Fn(&ParameterSet) -> bool + Send + Sync>,
    },
}

impl std::fmt::Debug for ParameterConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParameterConstraint::Equality {
                param,
                value,
                condition_param,
                condition_value,
            } => {
                write!(f, "Equality {{ param: {:?}, value: {:?}, condition_param: {:?}, condition_value: {:?} }}", 
                       param, value, condition_param, condition_value)
            }
            ParameterConstraint::Inequality {
                param,
                value,
                condition_param,
                condition_value,
            } => {
                write!(f, "Inequality {{ param: {:?}, value: {:?}, condition_param: {:?}, condition_value: {:?} }}", 
                       param, value, condition_param, condition_value)
            }
            ParameterConstraint::Range {
                param,
                min_value,
                max_value,
                condition_param,
                condition_value,
            } => {
                write!(f, "Range {{ param: {:?}, min_value: {:?}, max_value: {:?}, condition_param: {:?}, condition_value: {:?} }}", 
                       param, min_value, max_value, condition_param, condition_value)
            }
            ParameterConstraint::MutualExclusion {
                param1,
                value1,
                param2,
                value2,
            } => {
                write!(
                    f,
                    "MutualExclusion {{ param1: {:?}, value1: {:?}, param2: {:?}, value2: {:?} }}",
                    param1, value1, param2, value2
                )
            }
            ParameterConstraint::Custom { name, .. } => {
                write!(f, "Custom {{ name: {:?}, constraint_fn: <closure> }}", name)
            }
        }
    }
}

impl ParameterConstraint {
    /// Check if a parameter set satisfies this constraint
    pub fn is_satisfied(&self, params: &ParameterSet) -> bool {
        match self {
            ParameterConstraint::Equality {
                param,
                value,
                condition_param,
                condition_value,
            } => {
                if let (Some(param_val), Some(condition_val)) =
                    (params.get(param), params.get(condition_param))
                {
                    if condition_val == condition_value {
                        param_val == value
                    } else {
                        true // Constraint doesn't apply
                    }
                } else {
                    false // Missing parameters
                }
            }
            ParameterConstraint::Inequality {
                param,
                value,
                condition_param,
                condition_value,
            } => {
                if let (Some(param_val), Some(condition_val)) =
                    (params.get(param), params.get(condition_param))
                {
                    if condition_val == condition_value {
                        param_val != value
                    } else {
                        true // Constraint doesn't apply
                    }
                } else {
                    false // Missing parameters
                }
            }
            ParameterConstraint::Range {
                param,
                min_value,
                max_value,
                condition_param,
                condition_value,
            } => {
                if let (Some(param_val), Some(condition_val)) =
                    (params.get(param), params.get(condition_param))
                {
                    if condition_val == condition_value {
                        self.is_in_range(param_val, min_value, max_value)
                    } else {
                        true // Constraint doesn't apply
                    }
                } else {
                    false // Missing parameters
                }
            }
            ParameterConstraint::MutualExclusion {
                param1,
                value1,
                param2,
                value2,
            } => {
                if let (Some(param1_val), Some(param2_val)) =
                    (params.get(param1), params.get(param2))
                {
                    !(param1_val == value1 && param2_val == value2)
                } else {
                    true // Missing parameters - constraint satisfied
                }
            }
            ParameterConstraint::Custom { constraint_fn, .. } => constraint_fn(params),
        }
    }

    fn is_in_range(
        &self,
        value: &ParameterValue,
        min_value: &ParameterValue,
        max_value: &ParameterValue,
    ) -> bool {
        match (value, min_value, max_value) {
            (ParameterValue::Int(v), ParameterValue::Int(min), ParameterValue::Int(max)) => {
                v >= min && v <= max
            }
            (ParameterValue::Float(v), ParameterValue::Float(min), ParameterValue::Float(max)) => {
                v >= min && v <= max
            }
            _ => false, // Type mismatch
        }
    }
}

/// Conditional parameter definition
#[derive(Debug, Clone)]
pub struct ConditionalParameter {
    /// Base parameter
    pub parameter: CategoricalParameter,
    /// Conditions under which this parameter is active
    pub conditions: Vec<(String, ParameterValue)>,
    /// Whether all conditions must be met (AND) or any (OR)
    pub require_all_conditions: bool,
}

impl ConditionalParameter {
    /// Create a new conditional parameter
    pub fn new(parameter: CategoricalParameter, conditions: Vec<(String, ParameterValue)>) -> Self {
        Self {
            parameter,
            conditions,
            require_all_conditions: true,
        }
    }

    /// Set whether all conditions must be met
    pub fn require_all_conditions(mut self, require_all: bool) -> Self {
        self.require_all_conditions = require_all;
        self
    }

    /// Check if this parameter is active given the current parameter set
    pub fn is_active(&self, params: &ParameterSet) -> bool {
        if self.conditions.is_empty() {
            return true;
        }

        let satisfied_conditions = self
            .conditions
            .iter()
            .filter(|(param_name, expected_value)| {
                params
                    .get(param_name)
                    .map(|value| value == expected_value)
                    .unwrap_or(false)
            })
            .count();

        if self.require_all_conditions {
            satisfied_conditions == self.conditions.len()
        } else {
            satisfied_conditions > 0
        }
    }

    /// Sample from this parameter if it's active
    pub fn sample_if_active(
        &self,
        params: &ParameterSet,
        rng: &mut impl Rng,
    ) -> Option<ParameterValue> {
        if self.is_active(params) {
            Some(self.parameter.sample(rng))
        } else {
            None
        }
    }
}

/// Enhanced parameter space with categorical parameter support
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    /// Base categorical parameters
    pub categorical_params: HashMap<String, CategoricalParameter>,
    /// Conditional parameters
    pub conditional_params: HashMap<String, ConditionalParameter>,
    /// Parameter constraints
    pub constraints: Vec<ParameterConstraint>,
    /// Parameter dependencies (which parameters depend on which)
    pub dependencies: HashMap<String, HashSet<String>>,
}

impl ParameterSpace {
    /// Create a new parameter space
    pub fn new() -> Self {
        Self {
            categorical_params: HashMap::new(),
            conditional_params: HashMap::new(),
            constraints: Vec::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Add a categorical parameter
    pub fn add_categorical_parameter(&mut self, param: CategoricalParameter) {
        self.categorical_params.insert(param.name.clone(), param);
    }

    /// Add a conditional parameter
    pub fn add_conditional_parameter(&mut self, param: ConditionalParameter) {
        // Track dependencies
        for (dep_param, _) in &param.conditions {
            self.dependencies
                .entry(param.parameter.name.clone())
                .or_default()
                .insert(dep_param.clone());
        }
        self.conditional_params
            .insert(param.parameter.name.clone(), param);
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: ParameterConstraint) {
        self.constraints.push(constraint);
    }

    /// Sample a valid parameter set from this space
    pub fn sample(&self, rng: &mut impl Rng) -> Result<ParameterSet> {
        let mut params = ParameterSet::new();
        let mut attempts = 0;
        const MAX_ATTEMPTS: usize = 1000;

        while attempts < MAX_ATTEMPTS {
            params.clear();

            // Sample base categorical parameters first
            for (name, param) in &self.categorical_params {
                params.insert(name.clone(), param.sample(rng));
            }

            // Sample conditional parameters
            for (name, conditional_param) in &self.conditional_params {
                if let Some(value) = conditional_param.sample_if_active(&params, rng) {
                    params.insert(name.clone(), value);
                }
            }

            // Check constraints
            if self.is_valid_parameter_set(&params) {
                return Ok(params);
            }

            attempts += 1;
        }

        Err(SklearsError::InvalidInput(format!(
            "Failed to sample valid parameter set after {} attempts",
            MAX_ATTEMPTS
        )))
    }

    /// Check if a parameter set is valid according to all constraints
    pub fn is_valid_parameter_set(&self, params: &ParameterSet) -> bool {
        self.constraints
            .iter()
            .all(|constraint| constraint.is_satisfied(params))
    }

    /// Get all possible parameter names
    pub fn get_parameter_names(&self) -> HashSet<String> {
        let mut names = HashSet::new();
        names.extend(self.categorical_params.keys().cloned());
        names.extend(self.conditional_params.keys().cloned());
        names
    }

    /// Get parameters that depend on a given parameter
    pub fn get_dependent_parameters(&self, param_name: &str) -> HashSet<String> {
        self.dependencies
            .iter()
            .filter_map(|(dependent, dependencies)| {
                if dependencies.contains(param_name) {
                    Some(dependent.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the dependencies of a parameter
    pub fn get_parameter_dependencies(&self, param_name: &str) -> HashSet<String> {
        self.dependencies
            .get(param_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Generate a smart sample that respects parameter importance
    pub fn sample_with_importance(
        &self,
        rng: &mut impl Rng,
        importance_weights: &HashMap<String, f64>,
    ) -> Result<ParameterSet> {
        let mut params = ParameterSet::new();
        let mut attempts = 0;
        const MAX_ATTEMPTS: usize = 1000;

        while attempts < MAX_ATTEMPTS {
            params.clear();

            // Sort parameters by importance (higher importance first)
            let mut sorted_params: Vec<_> = self.categorical_params.keys().collect();
            sorted_params.sort_by(|a, b| {
                let weight_a = importance_weights.get(*a).unwrap_or(&1.0);
                let weight_b = importance_weights.get(*b).unwrap_or(&1.0);
                weight_b.partial_cmp(weight_a).unwrap()
            });

            // Sample important parameters first
            for name in sorted_params {
                if let Some(param) = self.categorical_params.get(name) {
                    params.insert(name.clone(), param.sample(rng));
                }
            }

            // Sample conditional parameters
            for (name, conditional_param) in &self.conditional_params {
                if let Some(value) = conditional_param.sample_if_active(&params, rng) {
                    params.insert(name.clone(), value);
                }
            }

            // Check constraints
            if self.is_valid_parameter_set(&params) {
                return Ok(params);
            }

            attempts += 1;
        }

        Err(SklearsError::InvalidInput(format!(
            "Failed to sample valid parameter set after {} attempts",
            MAX_ATTEMPTS
        )))
    }

    /// Convenience method to add a float parameter with min/max range
    pub fn add_float_param(&mut self, name: &str, min: f64, max: f64) {
        // Create a reasonable set of values across the range
        let mut values = Vec::new();
        let n_values = 10; // Default number of values to sample
        for i in 0..n_values {
            let ratio = i as f64 / (n_values - 1) as f64;
            let value = min + ratio * (max - min);
            values.push(ParameterValue::Float(value));
        }

        let param = CategoricalParameter::new(name.to_string(), values);
        self.add_categorical_parameter(param);
    }

    /// Convenience method to add an integer parameter with min/max range
    pub fn add_int_param(&mut self, name: &str, min: i64, max: i64) {
        let mut values = Vec::new();
        let range = max - min + 1;
        let n_values = if range <= 20 {
            // If small range, include all values
            range as usize
        } else {
            // If large range, sample 10 values
            10
        };

        for i in 0..n_values {
            let value = if range <= 20 {
                min + i as i64
            } else {
                let ratio = i as f64 / (n_values - 1) as f64;
                min + (ratio * (max - min) as f64) as i64
            };
            values.push(ParameterValue::Int(value));
        }

        let param = CategoricalParameter::new(name.to_string(), values);
        self.add_categorical_parameter(param);
    }

    /// Convenience method to add a categorical parameter from string slice
    pub fn add_categorical_param(&mut self, name: &str, values: Vec<&str>) {
        let param_values = values
            .into_iter()
            .map(|s| ParameterValue::String(s.to_string()))
            .collect();

        let param = CategoricalParameter::new(name.to_string(), param_values);
        self.add_categorical_parameter(param);
    }

    /// Convenience method to add a boolean parameter
    pub fn add_boolean_param(&mut self, name: &str) {
        let values = vec![ParameterValue::Bool(false), ParameterValue::Bool(true)];
        let param = CategoricalParameter::new(name.to_string(), values);
        self.add_categorical_parameter(param);
    }

    /// Auto-detect parameter ranges from a dataset of parameter sets
    pub fn auto_detect_ranges(parameter_sets: &[ParameterSet]) -> Result<Self> {
        let mut space = ParameterSpace::new();

        if parameter_sets.is_empty() {
            return Ok(space);
        }

        // Collect all parameter names
        let mut all_param_names = HashSet::new();
        for param_set in parameter_sets {
            all_param_names.extend(param_set.keys().cloned());
        }

        // For each parameter, detect its range/categories
        for param_name in all_param_names {
            let mut values = HashSet::new();
            for param_set in parameter_sets {
                if let Some(value) = param_set.get(&param_name) {
                    values.insert(value.clone());
                }
            }

            if !values.is_empty() {
                let values_vec: Vec<ParameterValue> = values.into_iter().collect();
                let categorical_param = CategoricalParameter::new(param_name, values_vec);
                space.add_categorical_parameter(categorical_param);
            }
        }

        Ok(space)
    }
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameter importance analyzer
#[derive(Debug)]
pub struct ParameterImportanceAnalyzer {
    /// Historical parameter evaluations
    evaluations: Vec<(ParameterSet, f64)>,
}

impl ParameterImportanceAnalyzer {
    /// Create a new importance analyzer
    pub fn new() -> Self {
        Self {
            evaluations: Vec::new(),
        }
    }

    /// Add an evaluation result
    pub fn add_evaluation(&mut self, params: ParameterSet, score: f64) {
        self.evaluations.push((params, score));
    }

    /// Calculate parameter importance using variance-based analysis
    pub fn calculate_importance(&self) -> HashMap<String, f64> {
        let mut importance = HashMap::new();

        if self.evaluations.len() < 2 {
            return importance;
        }

        // Get all parameter names
        let mut all_params = HashSet::new();
        for (params, _) in &self.evaluations {
            all_params.extend(params.keys().cloned());
        }

        // Calculate variance for each parameter
        for param_name in all_params {
            let variance = self.calculate_parameter_variance(&param_name);
            importance.insert(param_name, variance);
        }

        // Normalize importances
        let max_importance = importance.values().fold(0.0f64, |a, &b| a.max(b));
        if max_importance > 0.0 {
            for value in importance.values_mut() {
                *value /= max_importance;
            }
        }

        importance
    }

    fn calculate_parameter_variance(&self, param_name: &str) -> f64 {
        // Group evaluations by parameter value
        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();

        for (params, score) in &self.evaluations {
            if let Some(param_value) = params.get(param_name) {
                let key = format!("{:?}", param_value);
                groups.entry(key).or_default().push(*score);
            }
        }

        if groups.len() < 2 {
            return 0.0;
        }

        // Calculate within-group and between-group variance
        let mut total_variance = 0.0;
        let total_count = self.evaluations.len();

        for scores in groups.values() {
            if scores.len() > 1 {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                let variance = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (scores.len() - 1) as f64;
                total_variance += variance * (scores.len() as f64 / total_count as f64);
            }
        }

        total_variance
    }
}

impl Default for ParameterImportanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorical_parameter() {
        let param = CategoricalParameter::new(
            "algorithm".to_string(),
            vec!["svm".into(), "random_forest".into(), "neural_net".into()],
        );

        assert_eq!(param.values.len(), 3);
        assert!(!param.ordered);
    }

    #[test]
    fn test_ordered_categorical_parameter() {
        let param = CategoricalParameter::ordered(
            "complexity".to_string(),
            vec!["low".into(), "medium".into(), "high".into()],
        );

        assert!(param.ordered);
        let neighbors = param.get_neighbors(&"medium".into());
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_parameter_constraint() {
        let constraint = ParameterConstraint::Equality {
            param: "kernel".to_string(),
            value: "rbf".into(),
            condition_param: "algorithm".to_string(),
            condition_value: "svm".into(),
        };

        let mut params = ParameterSet::new();
        params.insert("algorithm".to_string(), "svm".into());
        params.insert("kernel".to_string(), "rbf".into());

        assert!(constraint.is_satisfied(&params));

        params.insert("kernel".to_string(), "linear".into());
        assert!(!constraint.is_satisfied(&params));
    }

    #[test]
    fn test_conditional_parameter() {
        let base_param = CategoricalParameter::new(
            "kernel".to_string(),
            vec!["linear".into(), "rbf".into(), "poly".into()],
        );

        let conditional_param =
            ConditionalParameter::new(base_param, vec![("algorithm".to_string(), "svm".into())]);

        let mut params = ParameterSet::new();
        params.insert("algorithm".to_string(), "svm".into());
        assert!(conditional_param.is_active(&params));

        params.insert("algorithm".to_string(), "random_forest".into());
        assert!(!conditional_param.is_active(&params));
    }

    #[test]
    fn test_parameter_space_sampling() {
        let mut space = ParameterSpace::new();

        let algorithm_param = CategoricalParameter::new(
            "algorithm".to_string(),
            vec!["svm".into(), "random_forest".into()],
        );
        space.add_categorical_parameter(algorithm_param);

        let kernel_param =
            CategoricalParameter::new("kernel".to_string(), vec!["linear".into(), "rbf".into()]);
        let conditional_kernel =
            ConditionalParameter::new(kernel_param, vec![("algorithm".to_string(), "svm".into())]);
        space.add_conditional_parameter(conditional_kernel);

        let mut rng = scirs2_core::random::thread_rng();
        let params = space.sample(&mut rng).unwrap();

        assert!(params.contains_key("algorithm"));

        if params.get("algorithm").unwrap() == &"svm".into() {
            assert!(params.contains_key("kernel"));
        }
    }
}
