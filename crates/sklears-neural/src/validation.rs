//! Hyperparameter validation and configuration management for neural networks.
//!
//! This module provides comprehensive validation of neural network hyperparameters,
//! configuration templates, and automatic parameter tuning support.

use crate::NeuralResult;
use sklears_core::error::SklearsError;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde_json;

/// Range constraint for numeric parameters
#[derive(Debug, Clone, PartialEq)]
pub enum RangeConstraint<T> {
    /// Parameter must be greater than value
    GreaterThan(T),
    /// Parameter must be greater than or equal to value
    GreaterEqualThan(T),
    /// Parameter must be less than value
    LessThan(T),
    /// Parameter must be less than or equal to value
    LessEqualThan(T),
    /// Parameter must be within inclusive range
    Range(T, T),
    /// Parameter must be one of specific values
    OneOf(Vec<T>),
    /// Parameter must be positive
    Positive,
    /// Parameter must be non-negative
    NonNegative,
    /// No constraint
    Any,
}

impl RangeConstraint<f64> {
    /// Validate that a value satisfies the constraint (f64 version)
    pub fn validate_f64(&self, value: f64, param_name: &str) -> NeuralResult<()> {
        match self {
            RangeConstraint::GreaterThan(threshold) => {
                if value <= *threshold {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be greater than {}", value, threshold),
                    });
                }
            }
            RangeConstraint::GreaterEqualThan(threshold) => {
                if value < *threshold {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!(
                            "Value {} must be greater than or equal to {}",
                            value, threshold
                        ),
                    });
                }
            }
            RangeConstraint::LessThan(threshold) => {
                if value >= *threshold {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be less than {}", value, threshold),
                    });
                }
            }
            RangeConstraint::LessEqualThan(threshold) => {
                if value > *threshold {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!(
                            "Value {} must be less than or equal to {}",
                            value, threshold
                        ),
                    });
                }
            }
            RangeConstraint::Range(min_val, max_val) => {
                if value < *min_val || value > *max_val {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!(
                            "Value {} must be between {} and {}",
                            value, min_val, max_val
                        ),
                    });
                }
            }
            RangeConstraint::OneOf(valid_values) => {
                if !valid_values.contains(&value) {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be one of: {:?}", value, valid_values),
                    });
                }
            }
            RangeConstraint::Positive => {
                if value <= 0.0 {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be positive", value),
                    });
                }
            }
            RangeConstraint::NonNegative => {
                if value < 0.0 {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be non-negative", value),
                    });
                }
            }
            RangeConstraint::Any => {
                // No constraint
            }
        }
        Ok(())
    }
}

impl RangeConstraint<i64> {
    /// Validate that a value satisfies the constraint (i64 version)
    pub fn validate_i64(&self, value: i64, param_name: &str) -> NeuralResult<()> {
        match self {
            RangeConstraint::GreaterThan(threshold) => {
                if value <= *threshold {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be greater than {}", value, threshold),
                    });
                }
            }
            RangeConstraint::GreaterEqualThan(threshold) => {
                if value < *threshold {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!(
                            "Value {} must be greater than or equal to {}",
                            value, threshold
                        ),
                    });
                }
            }
            RangeConstraint::LessThan(threshold) => {
                if value >= *threshold {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be less than {}", value, threshold),
                    });
                }
            }
            RangeConstraint::LessEqualThan(threshold) => {
                if value > *threshold {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!(
                            "Value {} must be less than or equal to {}",
                            value, threshold
                        ),
                    });
                }
            }
            RangeConstraint::Range(min_val, max_val) => {
                if value < *min_val || value > *max_val {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!(
                            "Value {} must be between {} and {}",
                            value, min_val, max_val
                        ),
                    });
                }
            }
            RangeConstraint::OneOf(valid_values) => {
                if !valid_values.contains(&value) {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be one of: {:?}", value, valid_values),
                    });
                }
            }
            RangeConstraint::Positive => {
                if value <= 0 {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be positive", value),
                    });
                }
            }
            RangeConstraint::NonNegative => {
                if value < 0 {
                    return Err(SklearsError::InvalidParameter {
                        name: param_name.to_string(),
                        reason: format!("Value {} must be non-negative", value),
                    });
                }
            }
            RangeConstraint::Any => {
                // No constraint
            }
        }
        Ok(())
    }
}

/// Parameter validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Parameter name
    pub name: String,
    /// Description of the parameter
    pub description: String,
    /// Whether the parameter is required
    pub required: bool,
    /// Numeric constraints (for numeric parameters)
    pub numeric_constraint: Option<RangeConstraint<f64>>,
    /// Integer constraints (for integer parameters)
    pub integer_constraint: Option<RangeConstraint<i64>>,
    /// String constraints (for string parameters)
    pub string_constraint: Option<Vec<String>>,
    /// Custom validation function
    #[cfg(feature = "serde")]
    pub custom_validator: Option<fn(&serde_json::Value) -> NeuralResult<()>>,
    /// Default value (if not required)
    #[cfg(feature = "serde")]
    pub default_value: Option<serde_json::Value>,
}

impl ValidationRule {
    /// Create a new validation rule (basic version)
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            required: false,
            numeric_constraint: None,
            integer_constraint: None,
            string_constraint: None,
            #[cfg(feature = "serde")]
            custom_validator: None,
            #[cfg(feature = "serde")]
            default_value: None,
        }
    }

    /// Mark parameter as required
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Add numeric constraint
    pub fn with_numeric_constraint(mut self, constraint: RangeConstraint<f64>) -> Self {
        self.numeric_constraint = Some(constraint);
        self
    }

    /// Add integer constraint
    pub fn with_integer_constraint(mut self, constraint: RangeConstraint<i64>) -> Self {
        self.integer_constraint = Some(constraint);
        self
    }

    /// Add string constraint (allowed values)
    pub fn with_string_constraint(mut self, allowed_values: Vec<String>) -> Self {
        self.string_constraint = Some(allowed_values);
        self
    }
}

#[cfg(feature = "serde")]
impl ValidationRule {
    /// Add custom validator
    pub fn with_custom_validator(
        mut self,
        validator: fn(&serde_json::Value) -> NeuralResult<()>,
    ) -> Self {
        self.custom_validator = Some(validator);
        self
    }

    /// Set default value
    pub fn with_default(mut self, default_value: serde_json::Value) -> Self {
        self.default_value = Some(default_value);
        self.required = false; // Can't be required if has default
        self
    }

    /// Validate a parameter value
    pub fn validate(&self, value: Option<&serde_json::Value>) -> NeuralResult<()> {
        match value {
            Some(val) => {
                // Validate numeric constraints
                if let Some(ref constraint) = self.numeric_constraint {
                    if val.is_null() {
                        // Allow null values for optional parameters
                        if self.required {
                            return Err(SklearsError::InvalidParameter {
                                name: self.name.clone(),
                                reason: "Required parameter cannot be null".to_string(),
                            });
                        }
                    } else if let Some(num_val) = val.as_f64() {
                        constraint.validate_f64(num_val, &self.name)?;
                    } else {
                        return Err(SklearsError::InvalidParameter {
                            name: self.name.clone(),
                            reason: "Expected numeric value".to_string(),
                        });
                    }
                }

                // Validate integer constraints
                if let Some(ref constraint) = self.integer_constraint {
                    if val.is_null() {
                        // Allow null values for optional parameters
                        if self.required {
                            return Err(SklearsError::InvalidParameter {
                                name: self.name.clone(),
                                reason: "Required parameter cannot be null".to_string(),
                            });
                        }
                    } else if let Some(int_val) = val.as_i64() {
                        constraint.validate_i64(int_val, &self.name)?;
                    } else {
                        return Err(SklearsError::InvalidParameter {
                            name: self.name.clone(),
                            reason: "Expected integer value".to_string(),
                        });
                    }
                }

                // Validate string constraints
                if let Some(ref allowed_values) = self.string_constraint {
                    if let Some(str_val) = val.as_str() {
                        if !allowed_values.contains(&str_val.to_string()) {
                            return Err(SklearsError::InvalidParameter {
                                name: self.name.clone(),
                                reason: format!(
                                    "Value '{}' must be one of: {:?}",
                                    str_val, allowed_values
                                ),
                            });
                        }
                    } else {
                        return Err(SklearsError::InvalidParameter {
                            name: self.name.clone(),
                            reason: "Expected string value".to_string(),
                        });
                    }
                }

                // Run custom validator
                if let Some(validator) = self.custom_validator {
                    validator(val)?;
                }
            }
            None => {
                if self.required {
                    return Err(SklearsError::InvalidParameter {
                        name: self.name.clone(),
                        reason: "Required parameter is missing".to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Hyperparameter validator for neural networks
pub struct HyperparameterValidator {
    /// Validation rules for each parameter
    rules: HashMap<String, ValidationRule>,
    /// Model type this validator is for
    model_type: String,
}

impl HyperparameterValidator {
    /// Create a new hyperparameter validator
    pub fn new(model_type: String) -> Self {
        Self {
            rules: HashMap::new(),
            model_type,
        }
    }

    /// Add a validation rule
    pub fn add_rule(mut self, rule: ValidationRule) -> Self {
        self.rules.insert(rule.name.clone(), rule);
        self
    }

    /// Add multiple validation rules
    pub fn add_rules(mut self, rules: Vec<ValidationRule>) -> Self {
        for rule in rules {
            self.rules.insert(rule.name.clone(), rule);
        }
        self
    }
}

#[cfg(feature = "serde")]
impl HyperparameterValidator {
    /// Validate hyperparameters
    pub fn validate(&self, params: &HashMap<String, serde_json::Value>) -> NeuralResult<()> {
        // Check all rules
        for rule in self.rules.values() {
            let param_value = params.get(&rule.name);
            rule.validate(param_value)?;
        }

        // Check for unknown parameters
        for param_name in params.keys() {
            if !self.rules.contains_key(param_name) {
                log::warn!(
                    "Unknown parameter '{}' for model type '{}'",
                    param_name,
                    self.model_type
                );
            }
        }

        Ok(())
    }

    /// Get parameter with default value if missing
    pub fn get_parameter_with_default(
        &self,
        params: &HashMap<String, serde_json::Value>,
        param_name: &str,
    ) -> NeuralResult<Option<serde_json::Value>> {
        if let Some(value) = params.get(param_name) {
            Ok(Some(value.clone()))
        } else if let Some(rule) = self.rules.get(param_name) {
            Ok(rule.default_value.clone())
        } else {
            Ok(None)
        }
    }

    /// Fill in missing parameters with default values
    pub fn apply_defaults(
        &self,
        params: &mut HashMap<String, serde_json::Value>,
    ) -> NeuralResult<()> {
        for rule in self.rules.values() {
            if !params.contains_key(&rule.name) {
                if let Some(ref default_value) = rule.default_value {
                    params.insert(rule.name.clone(), default_value.clone());
                }
            }
        }
        Ok(())
    }

    /// Get validation summary
    pub fn get_validation_summary(&self) -> ValidationSummary {
        let mut required_params = Vec::new();
        let mut optional_params = Vec::new();

        for rule in self.rules.values() {
            let param_info = ParameterInfo {
                name: rule.name.clone(),
                description: rule.description.clone(),
                required: rule.required,
                default_value: rule.default_value.clone(),
                constraints: self.get_constraint_description(rule),
            };

            if rule.required {
                required_params.push(param_info);
            } else {
                optional_params.push(param_info);
            }
        }

        ValidationSummary {
            model_type: self.model_type.clone(),
            required_params,
            optional_params,
        }
    }

    fn get_constraint_description(&self, rule: &ValidationRule) -> Vec<String> {
        let mut constraints = Vec::new();

        if let Some(ref numeric_constraint) = rule.numeric_constraint {
            constraints.push(format!("Numeric: {:?}", numeric_constraint));
        }

        if let Some(ref integer_constraint) = rule.integer_constraint {
            constraints.push(format!("Integer: {:?}", integer_constraint));
        }

        if let Some(ref string_constraint) = rule.string_constraint {
            constraints.push(format!("String options: {:?}", string_constraint));
        }

        if rule.custom_validator.is_some() {
            constraints.push("Custom validation".to_string());
        }

        constraints
    }
}

/// Parameter information for documentation
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub name: String,
    pub description: String,
    pub required: bool,
    #[cfg(feature = "serde")]
    pub default_value: Option<serde_json::Value>,
    pub constraints: Vec<String>,
}

/// Validation summary for documentation
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    pub model_type: String,
    pub required_params: Vec<ParameterInfo>,
    pub optional_params: Vec<ParameterInfo>,
}

/// Configuration templates for common neural network architectures
pub struct ConfigurationTemplates;

#[cfg(feature = "serde")]
impl ConfigurationTemplates {
    /// Get MLP classifier template validator
    pub fn mlp_classifier() -> HyperparameterValidator {
        HyperparameterValidator::new("MLPClassifier".to_string()).add_rules(vec![
            ValidationRule::new(
                "hidden_layer_sizes".to_string(),
                "Number of neurons in each hidden layer".to_string(),
            )
            .with_default(serde_json::json!([100])),
            ValidationRule::new(
                "activation".to_string(),
                "Activation function for hidden layers".to_string(),
            )
            .with_string_constraint(vec![
                "relu".to_string(),
                "tanh".to_string(),
                "sigmoid".to_string(),
                "elu".to_string(),
                "gelu".to_string(),
                "swish".to_string(),
                "leaky_relu".to_string(),
                "mish".to_string(),
            ])
            .with_default(serde_json::json!("relu")),
            ValidationRule::new(
                "learning_rate".to_string(),
                "Initial learning rate".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::Range(1e-6, 1.0))
            .with_default(serde_json::json!(0.001)),
            ValidationRule::new(
                "max_iter".to_string(),
                "Maximum number of training iterations".to_string(),
            )
            .with_integer_constraint(RangeConstraint::Positive)
            .with_default(serde_json::json!(200)),
            ValidationRule::new(
                "batch_size".to_string(),
                "Size of minibatches for training".to_string(),
            )
            .with_integer_constraint(RangeConstraint::Positive)
            .with_default(serde_json::json!(32)),
            ValidationRule::new("solver".to_string(), "Optimization algorithm".to_string())
                .with_string_constraint(vec![
                    "sgd".to_string(),
                    "adam".to_string(),
                    "adamw".to_string(),
                    "rmsprop".to_string(),
                    "nadam".to_string(),
                    "lamb".to_string(),
                    "lars".to_string(),
                ])
                .with_default(serde_json::json!("adam")),
            ValidationRule::new(
                "alpha".to_string(),
                "L2 regularization parameter".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::NonNegative)
            .with_default(serde_json::json!(0.0001)),
            ValidationRule::new(
                "random_state".to_string(),
                "Random seed for reproducibility".to_string(),
            )
            .with_integer_constraint(RangeConstraint::NonNegative)
            .with_default(serde_json::json!(null)),
            ValidationRule::new(
                "tol".to_string(),
                "Tolerance for optimization convergence".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::Positive)
            .with_default(serde_json::json!(1e-4)),
            ValidationRule::new(
                "momentum".to_string(),
                "Momentum for SGD optimizer".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::Range(0.0, 1.0))
            .with_default(serde_json::json!(0.9)),
            ValidationRule::new(
                "beta_1".to_string(),
                "Beta1 parameter for Adam optimizer".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::Range(0.0, 1.0))
            .with_default(serde_json::json!(0.9)),
            ValidationRule::new(
                "beta_2".to_string(),
                "Beta2 parameter for Adam optimizer".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::Range(0.0, 1.0))
            .with_default(serde_json::json!(0.999)),
            ValidationRule::new(
                "epsilon".to_string(),
                "Epsilon parameter for Adam optimizer".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::Positive)
            .with_default(serde_json::json!(1e-8)),
            ValidationRule::new(
                "early_stopping".to_string(),
                "Whether to use early stopping".to_string(),
            )
            .with_default(serde_json::json!(false)),
            ValidationRule::new(
                "validation_fraction".to_string(),
                "Fraction of training data to use for validation".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::Range(0.0, 1.0))
            .with_default(serde_json::json!(0.1)),
            ValidationRule::new(
                "n_iter_no_change".to_string(),
                "Maximum number of epochs without improvement for early stopping".to_string(),
            )
            .with_integer_constraint(RangeConstraint::Positive)
            .with_default(serde_json::json!(10)),
        ])
    }

    /// Get MLP regressor template validator
    pub fn mlp_regressor() -> HyperparameterValidator {
        let mut validator = Self::mlp_classifier();
        validator.model_type = "MLPRegressor".to_string();
        validator
    }

    /// Get CNN classifier template validator
    pub fn cnn_classifier() -> HyperparameterValidator {
        HyperparameterValidator::new("CNNClassifier".to_string()).add_rules(vec![
            ValidationRule::new(
                "conv_layers".to_string(),
                "Configuration for convolutional layers".to_string(),
            )
            .required(),
            ValidationRule::new(
                "pool_size".to_string(),
                "Pooling layer kernel size".to_string(),
            )
            .with_integer_constraint(RangeConstraint::Positive)
            .with_default(serde_json::json!(2)),
            ValidationRule::new(
                "kernel_size".to_string(),
                "Convolutional kernel size".to_string(),
            )
            .with_integer_constraint(RangeConstraint::Positive)
            .with_default(serde_json::json!(3)),
            ValidationRule::new("stride".to_string(), "Convolutional stride".to_string())
                .with_integer_constraint(RangeConstraint::Positive)
                .with_default(serde_json::json!(1)),
            ValidationRule::new(
                "padding".to_string(),
                "Padding type for convolution".to_string(),
            )
            .with_string_constraint(vec!["valid".to_string(), "same".to_string()])
            .with_default(serde_json::json!("valid")),
            ValidationRule::new(
                "dropout_rate".to_string(),
                "Dropout rate for regularization".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::Range(0.0, 1.0))
            .with_default(serde_json::json!(0.0)),
        ])
    }

    /// Get LSTM classifier template validator
    pub fn lstm_classifier() -> HyperparameterValidator {
        HyperparameterValidator::new("LSTMClassifier".to_string()).add_rules(vec![
            ValidationRule::new(
                "hidden_size".to_string(),
                "Number of features in hidden state".to_string(),
            )
            .with_integer_constraint(RangeConstraint::Positive)
            .with_default(serde_json::json!(128)),
            ValidationRule::new(
                "num_layers".to_string(),
                "Number of recurrent layers".to_string(),
            )
            .with_integer_constraint(RangeConstraint::Positive)
            .with_default(serde_json::json!(1)),
            ValidationRule::new(
                "bidirectional".to_string(),
                "Whether to use bidirectional LSTM".to_string(),
            )
            .with_default(serde_json::json!(false)),
            ValidationRule::new(
                "sequence_length".to_string(),
                "Input sequence length".to_string(),
            )
            .with_integer_constraint(RangeConstraint::Positive)
            .required(),
            ValidationRule::new(
                "dropout_rate".to_string(),
                "Dropout rate between LSTM layers".to_string(),
            )
            .with_numeric_constraint(RangeConstraint::Range(0.0, 1.0))
            .with_default(serde_json::json!(0.0)),
        ])
    }
}

/// Parameter tuning suggestions based on validation results
pub struct ParameterTuner;

#[cfg(feature = "serde")]
impl ParameterTuner {
    /// Suggest parameter ranges for hyperparameter optimization
    pub fn suggest_ranges(
        validator: &HyperparameterValidator,
        base_params: &HashMap<String, serde_json::Value>,
    ) -> HashMap<String, ParameterRange> {
        let mut suggestions = HashMap::new();

        for rule in validator.rules.values() {
            if let Some(range) = Self::suggest_range_for_rule(rule, base_params.get(&rule.name)) {
                suggestions.insert(rule.name.clone(), range);
            }
        }

        suggestions
    }

    fn suggest_range_for_rule(
        rule: &ValidationRule,
        current_value: Option<&serde_json::Value>,
    ) -> Option<ParameterRange> {
        match rule.name.as_str() {
            "learning_rate" => Some(ParameterRange::LogUniform(1e-6, 1e-1)),
            "batch_size" => Some(ParameterRange::Choice(vec![
                serde_json::json!(16),
                serde_json::json!(32),
                serde_json::json!(64),
                serde_json::json!(128),
                serde_json::json!(256),
            ])),
            "hidden_layer_sizes" => Some(ParameterRange::Choice(vec![
                serde_json::json!([50]),
                serde_json::json!([100]),
                serde_json::json!([100, 50]),
                serde_json::json!([200, 100]),
                serde_json::json!([300, 200, 100]),
            ])),
            "alpha" => Some(ParameterRange::LogUniform(1e-6, 1e-1)),
            "momentum" => Some(ParameterRange::Uniform(0.0, 1.0)),
            "beta_1" => Some(ParameterRange::Uniform(0.8, 0.999)),
            "beta_2" => Some(ParameterRange::Uniform(0.9, 0.9999)),
            "dropout_rate" => Some(ParameterRange::Uniform(0.0, 0.5)),
            _ => None,
        }
    }
}

/// Parameter range for hyperparameter optimization
#[derive(Debug, Clone)]
pub enum ParameterRange {
    /// Uniform distribution over range
    Uniform(f64, f64),
    /// Log-uniform distribution over range
    LogUniform(f64, f64),
    /// Discrete choices
    #[cfg(feature = "serde")]
    Choice(Vec<serde_json::Value>),
    /// Integer range
    IntRange(i64, i64),
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_range_constraint_validation() {
        let constraint = RangeConstraint::Range(0.0, 1.0);
        assert!(constraint.validate_f64(0.5, "test_param").is_ok());
        assert!(constraint.validate_f64(-0.1, "test_param").is_err());
        assert!(constraint.validate_f64(1.1, "test_param").is_err());

        let positive_constraint = RangeConstraint::Positive;
        assert!(positive_constraint.validate_f64(1.0, "test_param").is_ok());
        assert!(positive_constraint.validate_f64(0.0, "test_param").is_err());
        assert!(positive_constraint
            .validate_f64(-1.0, "test_param")
            .is_err());
    }

    #[test]
    fn test_validation_rule() {
        let rule = ValidationRule::new(
            "learning_rate".to_string(),
            "Learning rate parameter".to_string(),
        )
        .with_numeric_constraint(RangeConstraint::Range(1e-6, 1.0))
        .with_default(json!(0.001));

        // Valid value
        assert!(rule.validate(Some(&json!(0.01))).is_ok());

        // Invalid value (too high)
        assert!(rule.validate(Some(&json!(2.0))).is_err());

        // Missing value with default
        assert!(rule.validate(None).is_ok());

        // Non-numeric value
        assert!(rule.validate(Some(&json!("invalid"))).is_err());
    }

    #[test]
    fn test_hyperparameter_validator() {
        let validator = HyperparameterValidator::new("TestModel".to_string())
            .add_rule(
                ValidationRule::new("learning_rate".to_string(), "Learning rate".to_string())
                    .with_numeric_constraint(RangeConstraint::Positive)
                    .required(),
            )
            .add_rule(
                ValidationRule::new("batch_size".to_string(), "Batch size".to_string())
                    .with_integer_constraint(RangeConstraint::Positive)
                    .with_default(json!(32)),
            );

        let mut valid_params = HashMap::new();
        valid_params.insert("learning_rate".to_string(), json!(0.01));
        assert!(validator.validate(&valid_params).is_ok());

        let invalid_params = HashMap::new(); // Missing required parameter
        assert!(validator.validate(&invalid_params).is_err());

        let mut params_with_defaults = HashMap::new();
        params_with_defaults.insert("learning_rate".to_string(), json!(0.01));
        let mut params_with_defaults_applied = params_with_defaults.clone();
        validator
            .apply_defaults(&mut params_with_defaults_applied)
            .unwrap();
        assert!(params_with_defaults_applied.contains_key("batch_size"));
    }

    #[test]
    fn test_mlp_classifier_template() {
        let validator = ConfigurationTemplates::mlp_classifier();

        let mut params = HashMap::new();
        validator.apply_defaults(&mut params).unwrap();

        // Should have all default values
        assert!(params.contains_key("hidden_layer_sizes"));
        assert!(params.contains_key("activation"));
        assert!(params.contains_key("learning_rate"));

        // Should validate successfully
        assert!(validator.validate(&params).is_ok());

        // Test invalid activation
        params.insert("activation".to_string(), json!("invalid_activation"));
        assert!(validator.validate(&params).is_err());
    }

    #[test]
    fn test_parameter_tuner() {
        let validator = ConfigurationTemplates::mlp_classifier();
        let params = HashMap::new();

        let suggestions = ParameterTuner::suggest_ranges(&validator, &params);

        assert!(suggestions.contains_key("learning_rate"));
        assert!(suggestions.contains_key("batch_size"));
        assert!(suggestions.contains_key("hidden_layer_sizes"));

        if let Some(ParameterRange::LogUniform(min, max)) = suggestions.get("learning_rate") {
            assert!(min < max);
            assert!(*min > 0.0);
        } else {
            panic!("Expected LogUniform range for learning_rate");
        }
    }

    #[test]
    fn test_validation_summary() {
        let validator = ValidationRule::new("test_param".to_string(), "Test parameter".to_string())
            .required()
            .with_numeric_constraint(RangeConstraint::Positive);

        let validator = HyperparameterValidator::new("TestModel".to_string()).add_rule(validator);

        let summary = validator.get_validation_summary();
        assert_eq!(summary.model_type, "TestModel");
        assert_eq!(summary.required_params.len(), 1);
        assert_eq!(summary.optional_params.len(), 0);
        assert_eq!(summary.required_params[0].name, "test_param");
    }
}
