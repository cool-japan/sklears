/// Comprehensive validation framework for machine learning parameters and data
///
/// This module provides a robust validation system with custom derive macros
/// for automatic parameter validation in ML algorithms.
use crate::error::{Result, SklearsError};
use crate::types::{Array1, Array2, FloatBounds, Numeric};
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

/// Type alias for validation guard function
pub type ValidationGuardFn = Box<dyn Fn(&dyn std::any::Any) -> Result<bool> + Send + Sync>;

/// Type alias for validation destructuring function
pub type ValidationDestructureFn =
    Box<dyn Fn(&dyn std::any::Any) -> Result<ValidationResult> + Send + Sync>;

/// Core validation trait that can be derived for automatic parameter validation
pub trait Validate {
    /// Validate all parameters and return an error if any validation fails
    fn validate(&self) -> Result<()>;

    /// Validate and provide detailed error information  
    fn validate_with_context(&self, context: &str) -> Result<()> {
        self.validate()
            .map_err(|e| SklearsError::Other(format!("{context}: {e}")))
    }
}

/// Validation attributes for ML parameter constraints
#[derive(Debug, Clone)]
pub enum ValidationRule {
    /// Value must be positive (> 0)
    Positive,
    /// Value must be non-negative (>= 0)
    NonNegative,
    /// Value must be finite (not NaN or infinity)
    Finite,
    /// Value must be in a specific range [min, max]
    Range { min: f64, max: f64 },
    /// Value must be one of the specified options
    OneOf(Vec<String>),
    /// Array must have minimum number of elements
    MinLength(usize),
    /// Array must have maximum number of elements
    MaxLength(usize),
    /// Array elements must be unique
    UniqueElements,
    /// Custom validation function
    Custom(fn(&dyn std::any::Any) -> Result<()>),
    /// Pattern guard validation with custom matching
    PatternGuard(PatternGuardRule),
}

/// Pattern guard rule for advanced validation with custom matching
pub struct PatternGuardRule {
    /// Name of the pattern for error reporting
    pub pattern_name: String,
    /// Function that performs the pattern matching and validation
    pub guard_fn: ValidationGuardFn,
    /// Error message when the pattern doesn't match
    pub error_message: String,
    /// Optional structured destructuring validator
    pub destructure_fn: Option<ValidationDestructureFn>,
}

impl std::fmt::Debug for PatternGuardRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatternGuardRule")
            .field("pattern_name", &self.pattern_name)
            .field("guard_fn", &"<function>")
            .field("error_message", &self.error_message)
            .field(
                "destructure_fn",
                &self.destructure_fn.as_ref().map(|_| "<function>"),
            )
            .finish()
    }
}

impl Clone for PatternGuardRule {
    fn clone(&self) -> Self {
        // Create a simple clone that preserves the pattern name and error message
        // but uses a default guard function
        Self {
            pattern_name: self.pattern_name.clone(),
            guard_fn: Box::new(|_| Ok(true)), // Default to always pass
            error_message: self.error_message.clone(),
            destructure_fn: None,
        }
    }
}

/// Result of pattern matching and validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the pattern matched
    pub matched: bool,
    /// Additional context from pattern matching
    pub context: std::collections::HashMap<String, String>,
    /// Any warnings generated during validation
    pub warnings: Vec<String>,
}

/// Macro for creating pattern guards with custom validation logic
#[macro_export]
macro_rules! pattern_guard {
    // Pattern guard for numeric types with range validation
    (numeric_range, $min:expr, $max:expr) => {
        $crate::validation::PatternGuardRule {
            pattern_name: "numeric_range".to_string(),
            guard_fn: Box::new(move |value| {
                if let Some(val) = value.downcast_ref::<f64>() {
                    Ok(*val >= $min && *val <= $max)
                } else if let Some(val) = value.downcast_ref::<f32>() {
                    Ok(*val >= $min as f32 && *val <= $max as f32)
                } else if let Some(val) = value.downcast_ref::<i32>() {
                    Ok(*val >= $min as i32 && *val <= $max as i32)
                } else if let Some(val) = value.downcast_ref::<usize>() {
                    Ok(*val >= $min as usize && *val <= $max as usize)
                } else {
                    Ok(false)
                }
            }),
            error_message: format!("Value must be in range [{}, {}]", $min, $max),
            destructure_fn: None,
        }
    };

    // Pattern guard for array shape validation
    (array_shape, $expected_shape:expr) => {
        $crate::validation::PatternGuardRule {
            pattern_name: "array_shape".to_string(),
            guard_fn: Box::new(move |value| {
                // This would need proper array type checking in real implementation
                // For now, just return true as placeholder
                Ok(true)
            }),
            error_message: format!("Array shape must match {:?}", $expected_shape),
            destructure_fn: None,
        }
    };

    // Pattern guard for string enum validation
    (string_enum, $valid_options:expr) => {
        $crate::validation::PatternGuardRule {
            pattern_name: "string_enum".to_string(),
            guard_fn: Box::new(move |value| {
                if let Some(val) = value.downcast_ref::<String>() {
                    Ok($valid_options.contains(&val.as_str()))
                } else if let Some(val) = value.downcast_ref::<&str>() {
                    Ok($valid_options.contains(val))
                } else {
                    Ok(false)
                }
            }),
            error_message: format!("Value must be one of {:?}", $valid_options),
            destructure_fn: None,
        }
    };

    // Pattern guard with custom function and error message
    ($pattern_name:literal, $guard:expr, $error_msg:literal) => {
        $crate::validation::PatternGuardRule {
            pattern_name: $pattern_name.to_string(),
            guard_fn: Box::new($guard),
            error_message: $error_msg.to_string(),
            destructure_fn: None,
        }
    };

    // Pattern guard with destructuring validation
    ($pattern_name:literal, $guard_fn:expr, $destructure_fn:expr) => {
        $crate::validation::PatternGuardRule {
            pattern_name: $pattern_name.to_string(),
            guard_fn: Box::new($guard_fn),
            error_message: format!("Pattern '{}' validation failed", $pattern_name),
            destructure_fn: Some(Box::new($destructure_fn)),
        }
    };
}

/// Trait for types that can be pattern matched and validated
pub trait PatternValidate {
    /// Apply pattern guard validation
    fn validate_with_pattern(&self, guard: &PatternGuardRule) -> Result<ValidationResult>;

    /// Check if value matches a specific pattern
    fn matches_pattern(&self, pattern_name: &str) -> bool;

    /// Extract structured data using pattern destructuring
    fn destructure(&self, pattern: &str) -> Result<std::collections::HashMap<String, String>>;
}

/// Implementation of PatternValidate for f64
impl PatternValidate for f64 {
    fn validate_with_pattern(&self, guard: &PatternGuardRule) -> Result<ValidationResult> {
        let value_any = self as &dyn std::any::Any;
        let matched = (guard.guard_fn)(value_any)?;

        let mut context = std::collections::HashMap::new();
        context.insert("value".to_string(), self.to_string());
        context.insert("type".to_string(), "f64".to_string());

        if let Some(destructure_fn) = &guard.destructure_fn {
            let destructure_result = destructure_fn(value_any)?;
            Ok(ValidationResult {
                matched: matched && destructure_result.matched,
                context: destructure_result.context,
                warnings: destructure_result.warnings,
            })
        } else {
            Ok(ValidationResult {
                matched,
                context,
                warnings: Vec::new(),
            })
        }
    }

    fn matches_pattern(&self, pattern_name: &str) -> bool {
        match pattern_name {
            "finite" => self.is_finite(),
            "positive" => *self > 0.0,
            "non_negative" => *self >= 0.0,
            "probability" => *self >= 0.0 && *self <= 1.0,
            _ => false,
        }
    }

    fn destructure(&self, pattern: &str) -> Result<std::collections::HashMap<String, String>> {
        let mut result = std::collections::HashMap::new();
        match pattern {
            "range_info" => {
                result.insert("value".to_string(), self.to_string());
                result.insert("is_finite".to_string(), self.is_finite().to_string());
                result.insert("is_positive".to_string(), (*self > 0.0).to_string());
                result.insert(
                    "sign".to_string(),
                    if *self >= 0.0 {
                        "positive".to_string()
                    } else {
                        "negative".to_string()
                    },
                );
            }
            _ => {
                result.insert("value".to_string(), self.to_string());
            }
        }
        Ok(result)
    }
}

/// Implementation of PatternValidate for usize
impl PatternValidate for usize {
    fn validate_with_pattern(&self, guard: &PatternGuardRule) -> Result<ValidationResult> {
        let value_any = self as &dyn std::any::Any;
        let matched = (guard.guard_fn)(value_any)?;

        let mut context = std::collections::HashMap::new();
        context.insert("value".to_string(), self.to_string());
        context.insert("type".to_string(), "usize".to_string());

        Ok(ValidationResult {
            matched,
            context,
            warnings: Vec::new(),
        })
    }

    fn matches_pattern(&self, pattern_name: &str) -> bool {
        match pattern_name {
            "positive" => *self > 0,
            "non_negative" => true, // usize is always non-negative
            "power_of_two" => self.is_power_of_two(),
            _ => false,
        }
    }

    fn destructure(&self, pattern: &str) -> Result<std::collections::HashMap<String, String>> {
        let mut result = std::collections::HashMap::new();
        match pattern {
            "number_info" => {
                result.insert("value".to_string(), self.to_string());
                result.insert("is_positive".to_string(), (*self > 0).to_string());
                result.insert(
                    "is_power_of_two".to_string(),
                    self.is_power_of_two().to_string(),
                );
            }
            _ => {
                result.insert("value".to_string(), self.to_string());
            }
        }
        Ok(result)
    }
}

/// Implementation of PatternValidate for String
impl PatternValidate for String {
    fn validate_with_pattern(&self, guard: &PatternGuardRule) -> Result<ValidationResult> {
        let value_any = self as &dyn std::any::Any;
        let matched = (guard.guard_fn)(value_any)?;

        let mut context = std::collections::HashMap::new();
        context.insert("value".to_string(), self.clone());
        context.insert("type".to_string(), "String".to_string());
        context.insert("length".to_string(), self.len().to_string());

        Ok(ValidationResult {
            matched,
            context,
            warnings: Vec::new(),
        })
    }

    fn matches_pattern(&self, pattern_name: &str) -> bool {
        match pattern_name {
            "non_empty" => !self.is_empty(),
            "alphanumeric" => self.chars().all(|c| c.is_alphanumeric()),
            "lowercase" => self.chars().all(|c| !c.is_alphabetic() || c.is_lowercase()),
            "uppercase" => self.chars().all(|c| !c.is_alphabetic() || c.is_uppercase()),
            _ => false,
        }
    }

    fn destructure(&self, pattern: &str) -> Result<std::collections::HashMap<String, String>> {
        let mut result = std::collections::HashMap::new();
        match pattern {
            "string_info" => {
                result.insert("value".to_string(), self.clone());
                result.insert("length".to_string(), self.len().to_string());
                result.insert("is_empty".to_string(), self.is_empty().to_string());
                result.insert(
                    "is_alphanumeric".to_string(),
                    self.chars().all(|c| c.is_alphanumeric()).to_string(),
                );
            }
            _ => {
                result.insert("value".to_string(), self.clone());
            }
        }
        Ok(result)
    }
}

/// Default implementations for common pattern validation scenarios
pub mod pattern_guards {
    use super::*;

    /// Pattern guard for ML model hyperparameters
    pub fn hyperparameter_pattern<T: FloatBounds + std::fmt::Debug>(
        min_val: T,
        max_val: T,
        finite_required: bool,
    ) -> PatternGuardRule {
        PatternGuardRule {
            pattern_name: "hyperparameter_bounds".to_string(),
            guard_fn: Box::new(|_value| {
                // In real implementation, would need proper type casting
                Ok(true) // Placeholder
            }),
            error_message: format!(
                "Hyperparameter must be in range [{}, {}]{}",
                min_val,
                max_val,
                if finite_required { " and finite" } else { "" }
            ),
            destructure_fn: None,
        }
    }

    /// Pattern guard for array shape validation
    pub fn array_shape_pattern(expected_dims: &[usize]) -> PatternGuardRule {
        let dims_str = expected_dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        PatternGuardRule {
            pattern_name: "array_shape".to_string(),
            guard_fn: Box::new(|_value| {
                // Would validate array shape in real implementation
                Ok(true)
            }),
            error_message: format!("Array shape must match [{dims_str}]"),
            destructure_fn: None, // Remove capturing closure for now
        }
    }

    /// Pattern guard for ML algorithm configuration
    pub fn algorithm_config_pattern(required_fields: &[&str]) -> PatternGuardRule {
        let fields_str = required_fields.join(", ");

        PatternGuardRule {
            pattern_name: "algorithm_config".to_string(),
            guard_fn: Box::new(|_value| {
                // Would validate configuration completeness
                Ok(true)
            }),
            error_message: format!("Configuration must contain fields: {fields_str}"),
            destructure_fn: None, // Remove capturing closure for now
        }
    }

    /// Pattern guard for data type consistency
    pub fn data_type_pattern(expected_types: &[&str]) -> PatternGuardRule {
        let types_str = expected_types.join(" | ");

        PatternGuardRule {
            pattern_name: "data_type_consistency".to_string(),
            guard_fn: Box::new(|_value| {
                // Would validate data type consistency
                Ok(true)
            }),
            error_message: format!("Data type must be one of: {types_str}"),
            destructure_fn: None,
        }
    }
}

/// Container for multiple validation rules
#[derive(Debug, Clone)]
pub struct ValidationRules {
    pub rules: Vec<ValidationRule>,
    pub field_name: String,
}

impl ValidationRules {
    /// Create a new validation rules container
    pub fn new(field_name: &str) -> Self {
        Self {
            rules: Vec::new(),
            field_name: field_name.to_string(),
        }
    }

    /// Add a validation rule
    pub fn add_rule(mut self, rule: ValidationRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Validate a numeric value against all rules
    pub fn validate_numeric<T>(&self, value: &T) -> Result<()>
    where
        T: Numeric + PartialOrd + Debug + Copy + NumCast,
    {
        for rule in &self.rules {
            match rule {
                ValidationRule::Positive => {
                    if *value <= T::zero() {
                        return Err(SklearsError::InvalidParameter {
                            name: self.field_name.clone(),
                            reason: "must be positive".to_string(),
                        });
                    }
                }
                ValidationRule::NonNegative => {
                    if *value < T::zero() {
                        return Err(SklearsError::InvalidParameter {
                            name: self.field_name.clone(),
                            reason: "must be non-negative".to_string(),
                        });
                    }
                }
                ValidationRule::Finite => {
                    if let Some(float_val) = NumCast::from(*value) {
                        let f: f64 = float_val;
                        if !f.is_finite() {
                            return Err(SklearsError::InvalidParameter {
                                name: self.field_name.clone(),
                                reason: "must be finite".to_string(),
                            });
                        }
                    }
                }
                ValidationRule::Range { min, max } => {
                    if let Some(float_val) = NumCast::from(*value) {
                        let f: f64 = float_val;
                        if f < *min || f > *max {
                            return Err(SklearsError::InvalidParameter {
                                name: self.field_name.clone(),
                                reason: format!("must be in range [{min}, {max}]"),
                            });
                        }
                    }
                }
                ValidationRule::PatternGuard(_pattern_guard) => {
                    // TODO: Fix lifetime issues with pattern guard validation
                    // let value_any = &value as &dyn std::any::Any;
                    // let result = (pattern_guard.guard_fn)(value_any)?;
                    // if !result {
                    //     return Err(SklearsError::InvalidParameter {
                    //         name: self.field_name.clone(),
                    //         reason: pattern_guard.error_message.clone(),
                    //     });
                    // }
                }
                _ => {
                    // Skip rules that don't apply to numeric values
                }
            }
        }
        Ok(())
    }

    /// Validate a string value against all rules
    pub fn validate_string(&self, value: &str) -> Result<()> {
        for rule in &self.rules {
            match rule {
                ValidationRule::OneOf(options) => {
                    if !options.contains(&value.to_string()) {
                        return Err(SklearsError::InvalidParameter {
                            name: self.field_name.clone(),
                            reason: format!("must be one of {options:?}"),
                        });
                    }
                }
                ValidationRule::PatternGuard(_pattern_guard) => {
                    // TODO: Fix lifetime issues with pattern guard validation
                    // let value_any = &value as &dyn std::any::Any;
                    // let result = (pattern_guard.guard_fn)(value_any)?;
                    // if !result {
                    //     return Err(SklearsError::InvalidParameter {
                    //         name: self.field_name.clone(),
                    //         reason: pattern_guard.error_message.clone(),
                    //     });
                    // }
                }
                _ => {
                    // Skip rules that don't apply to string values
                }
            }
        }
        Ok(())
    }

    /// Validate an array/vector against all rules
    pub fn validate_array<T>(&self, value: &[T]) -> Result<()> {
        for rule in &self.rules {
            match rule {
                ValidationRule::MinLength(min_len) => {
                    if value.len() < *min_len {
                        return Err(SklearsError::InvalidParameter {
                            name: self.field_name.clone(),
                            reason: format!("must have at least {min_len} elements"),
                        });
                    }
                }
                ValidationRule::MaxLength(max_len) => {
                    if value.len() > *max_len {
                        return Err(SklearsError::InvalidParameter {
                            name: self.field_name.clone(),
                            reason: format!("must have at most {max_len} elements"),
                        });
                    }
                }
                ValidationRule::PatternGuard(_pattern_guard) => {
                    // TODO: Fix lifetime issues with pattern guard validation
                    // let value_any = &value as &dyn std::any::Any;
                    // let result = (pattern_guard.guard_fn)(value_any)?;
                    // if !result {
                    //     return Err(SklearsError::InvalidParameter {
                    //         name: self.field_name.clone(),
                    //         reason: pattern_guard.error_message.clone(),
                    //     });
                    // }
                }
                _ => {
                    // Skip rules that don't apply to arrays
                }
            }
        }
        Ok(())
    }

    /// Validate an unsigned integer value (usize) against all rules
    pub fn validate_usize(&self, value: &usize) -> Result<()> {
        for rule in &self.rules {
            match rule {
                ValidationRule::Positive => {
                    if *value == 0 {
                        return Err(SklearsError::InvalidParameter {
                            name: self.field_name.clone(),
                            reason: "must be positive".to_string(),
                        });
                    }
                }
                ValidationRule::NonNegative => {
                    // usize is always non-negative, so this always passes
                }
                ValidationRule::Range { min, max } => {
                    let val = *value as f64;
                    if val < *min || val > *max {
                        return Err(SklearsError::InvalidParameter {
                            name: self.field_name.clone(),
                            reason: format!("must be in range [{min}, {max}]"),
                        });
                    }
                }
                _ => {
                    // Skip rules that don't apply to usize values
                }
            }
        }
        Ok(())
    }
}

/// ML-specific validation functions
pub mod ml {
    use super::*;

    /// Validate learning rate (must be positive and typically < 1.0)
    pub fn validate_learning_rate<T: FloatBounds>(lr: T) -> Result<()> {
        if lr <= T::zero() {
            return Err(SklearsError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        if !Float::is_finite(lr) {
            return Err(SklearsError::InvalidParameter {
                name: "learning_rate".to_string(),
                reason: "must be finite".to_string(),
            });
        }

        // Warning for unusually high learning rates
        if lr > T::one() {
            log::warn!("Learning rate {lr} is unusually high, consider using a smaller value");
        }

        Ok(())
    }

    /// Validate regularization parameter (must be non-negative)
    pub fn validate_regularization<T: FloatBounds>(reg: T) -> Result<()> {
        if reg < T::zero() {
            return Err(SklearsError::InvalidParameter {
                name: "regularization".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }

        if !Float::is_finite(reg) {
            return Err(SklearsError::InvalidParameter {
                name: "regularization".to_string(),
                reason: "must be finite".to_string(),
            });
        }

        Ok(())
    }

    /// Validate number of clusters (must be positive integer)
    pub fn validate_n_clusters(n_clusters: usize, n_samples: usize) -> Result<()> {
        if n_clusters == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_clusters".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        if n_clusters > n_samples {
            return Err(SklearsError::InvalidParameter {
                name: "n_clusters".to_string(),
                reason: format!("cannot exceed number of samples ({n_samples})"),
            });
        }

        Ok(())
    }

    /// Validate number of neighbors for KNN (must be positive and <= n_samples)
    pub fn validate_n_neighbors(n_neighbors: usize, n_samples: usize) -> Result<()> {
        if n_neighbors == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_neighbors".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        if n_neighbors > n_samples {
            return Err(SklearsError::InvalidParameter {
                name: "n_neighbors".to_string(),
                reason: format!("cannot exceed number of samples ({n_samples})"),
            });
        }

        Ok(())
    }

    /// Validate tolerance parameter (must be positive and small)
    pub fn validate_tolerance<T: FloatBounds>(tol: T) -> Result<()> {
        if tol <= T::zero() {
            return Err(SklearsError::InvalidParameter {
                name: "tolerance".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        if !Float::is_finite(tol) {
            return Err(SklearsError::InvalidParameter {
                name: "tolerance".to_string(),
                reason: "must be finite".to_string(),
            });
        }

        // Warning for very large tolerances
        if tol > T::from(0.1).unwrap_or(T::one()) {
            log::warn!("Tolerance {tol} is very large, algorithm may converge prematurely");
        }

        Ok(())
    }

    /// Validate max iterations (must be positive)
    pub fn validate_max_iter(max_iter: usize) -> Result<()> {
        if max_iter == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "max_iter".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        Ok(())
    }

    /// Validate probability values (must be in [0, 1])
    pub fn validate_probability<T: FloatBounds>(prob: T) -> Result<()> {
        if prob < T::zero() || prob > T::one() {
            return Err(SklearsError::InvalidParameter {
                name: "probability".to_string(),
                reason: "must be in range [0, 1]".to_string(),
            });
        }

        if !Float::is_finite(prob) {
            return Err(SklearsError::InvalidParameter {
                name: "probability".to_string(),
                reason: "must be finite".to_string(),
            });
        }

        Ok(())
    }

    /// Validate data shapes for supervised learning
    pub fn validate_supervised_data<T, U>(x: &Array2<T>, y: &Array1<U>) -> Result<()> {
        if x.is_empty() {
            return Err(SklearsError::InvalidData {
                reason: "X cannot be empty".to_string(),
            });
        }

        if y.is_empty() {
            return Err(SklearsError::InvalidData {
                reason: "y cannot be empty".to_string(),
            });
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, y.shape[0]={}", x.nrows(), y.len()),
            });
        }

        Ok(())
    }

    /// Validate data for unsupervised learning
    pub fn validate_unsupervised_data<T>(x: &Array2<T>) -> Result<()> {
        if x.is_empty() {
            return Err(SklearsError::InvalidData {
                reason: "X cannot be empty".to_string(),
            });
        }

        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidData {
                reason: "X must have positive dimensions".to_string(),
            });
        }

        Ok(())
    }

    /// Validate feature consistency between training and prediction
    pub fn validate_feature_consistency<T, U>(
        x_train: &Array2<T>,
        x_test: &Array2<U>,
        _model_name: &str,
    ) -> Result<()> {
        if x_train.ncols() != x_test.ncols() {
            return Err(SklearsError::FeatureMismatch {
                expected: x_train.ncols(),
                actual: x_test.ncols(),
            });
        }

        Ok(())
    }
}

/// Proc macro helper functions for derive implementation
pub mod derive_helpers {
    /// Generate validation code for a field with validation attributes
    pub fn generate_field_validation(
        field_name: &str,
        _field_type: &str,
        validation_attrs: &[String],
    ) -> String {
        let mut validations = Vec::new();

        for attr in validation_attrs {
            match attr.as_str() {
                "positive" => {
                    validations.push(format!(
                        "ValidationRules::new(\"{field_name}\").add_rule(ValidationRule::Positive).validate_numeric(&self.{field_name})?;"
                    ));
                }
                "non_negative" => {
                    validations.push(format!(
                        "ValidationRules::new(\"{field_name}\").add_rule(ValidationRule::NonNegative).validate_numeric(&self.{field_name})?;"
                    ));
                }
                "finite" => {
                    validations.push(format!(
                        "ValidationRules::new(\"{field_name}\").add_rule(ValidationRule::Finite).validate_numeric(&self.{field_name})?;"
                    ));
                }
                _ if attr.starts_with("range(") => {
                    // Parse range(min, max) format
                    let range_str = attr
                        .strip_prefix("range(")
                        .unwrap()
                        .strip_suffix(")")
                        .unwrap();
                    let parts: Vec<&str> = range_str.split(',').map(|s| s.trim()).collect();
                    if parts.len() == 2 {
                        let min_val = parts[0];
                        let max_val = parts[1];
                        validations.push(format!(
                            "ValidationRules::new(\"{field_name}\").add_rule(ValidationRule::Range {{ min: {min_val}, max: {max_val} }}).validate_numeric(&self.{field_name})?;"
                        ));
                    }
                }
                _ => {}
            }
        }

        validations.join("\n")
    }
}

/// Configuration validation for complete ML algorithms
pub trait ConfigValidation {
    /// Validate the entire configuration
    fn validate_config(&self) -> Result<()>;

    /// Get validation warnings (non-fatal issues)
    fn get_warnings(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Validation context for providing better error messages
#[derive(Debug, Clone)]
pub struct ValidationContext {
    pub algorithm: String,
    pub operation: String,
    pub data_info: Option<DataInfo>,
}

/// Information about the data being validated
#[derive(Debug, Clone)]
pub struct DataInfo {
    pub n_samples: usize,
    pub n_features: usize,
    pub data_type: String,
}

impl ValidationContext {
    /// Create a new validation context
    pub fn new(algorithm: &str, operation: &str) -> Self {
        Self {
            algorithm: algorithm.to_string(),
            operation: operation.to_string(),
            data_info: None,
        }
    }

    /// Add data information to the context
    pub fn with_data_info(mut self, n_samples: usize, n_features: usize, data_type: &str) -> Self {
        self.data_info = Some(DataInfo {
            n_samples,
            n_features,
            data_type: data_type.to_string(),
        });
        self
    }

    /// Format error with context information
    pub fn format_error(&self, error: &SklearsError) -> String {
        let mut msg = format!(
            "Error in {} during {}: {error}",
            self.algorithm, self.operation
        );

        if let Some(data_info) = &self.data_info {
            msg.push_str(&format!(
                " (data: {} samples, {} features, type: {})",
                data_info.n_samples, data_info.n_features, data_info.data_type
            ));
        }

        msg
    }
}

/// Structured destructuring for complex data types
pub mod structured_destructuring {
    use super::*;

    /// Trait for types that support structured destructuring
    pub trait StructuredDestructure {
        /// Destructure into named components
        fn destructure_into_components(
            &self,
        ) -> Result<std::collections::HashMap<String, Box<dyn std::any::Any>>>;

        /// Extract specific fields by path (e.g., "user.address.city")
        fn extract_field(&self, field_path: &str) -> Result<Box<dyn std::any::Any>>;

        /// Validate structure matches expected schema
        fn validate_structure(&self, schema: &StructuralSchema) -> Result<ValidationResult>;
    }

    /// Schema for validating complex data structures
    #[derive(Debug, Clone, Default)]
    pub struct StructuralSchema {
        pub required_fields: Vec<String>,
        pub optional_fields: Vec<String>,
        pub field_types: std::collections::HashMap<String, String>,
        pub nested_schemas: std::collections::HashMap<String, StructuralSchema>,
    }

    impl StructuralSchema {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn require_field(mut self, field_name: &str, field_type: &str) -> Self {
            self.required_fields.push(field_name.to_string());
            self.field_types
                .insert(field_name.to_string(), field_type.to_string());
            self
        }

        pub fn optional_field(mut self, field_name: &str, field_type: &str) -> Self {
            self.optional_fields.push(field_name.to_string());
            self.field_types
                .insert(field_name.to_string(), field_type.to_string());
            self
        }

        pub fn nested_schema(mut self, field_name: &str, schema: StructuralSchema) -> Self {
            self.nested_schemas.insert(field_name.to_string(), schema);
            self
        }
    }

    /// Configuration for ML algorithms with structured validation
    #[derive(Debug, Clone)]
    pub struct AlgorithmConfig {
        pub algorithm_name: String,
        pub hyperparameters: std::collections::HashMap<String, ConfigValue>,
        pub metadata: std::collections::HashMap<String, String>,
    }

    /// Values that can be stored in configuration
    #[derive(Debug, Clone)]
    pub enum ConfigValue {
        Float(f64),
        Integer(i64),
        String(String),
        Boolean(bool),
        Array(Vec<ConfigValue>),
        Object(std::collections::HashMap<String, ConfigValue>),
    }

    impl StructuredDestructure for AlgorithmConfig {
        fn destructure_into_components(
            &self,
        ) -> Result<std::collections::HashMap<String, Box<dyn std::any::Any>>> {
            let mut components = std::collections::HashMap::new();

            components.insert(
                "algorithm_name".to_string(),
                Box::new(self.algorithm_name.clone()) as Box<dyn std::any::Any>,
            );
            components.insert(
                "hyperparameters".to_string(),
                Box::new(self.hyperparameters.clone()) as Box<dyn std::any::Any>,
            );
            components.insert(
                "metadata".to_string(),
                Box::new(self.metadata.clone()) as Box<dyn std::any::Any>,
            );

            Ok(components)
        }

        fn extract_field(&self, field_path: &str) -> Result<Box<dyn std::any::Any>> {
            let parts: Vec<&str> = field_path.split('.').collect();

            match parts.first() {
                Some(&"algorithm_name") => Ok(Box::new(self.algorithm_name.clone())),
                Some(&"hyperparameters") => {
                    if parts.len() > 1 {
                        if let Some(param_value) = self.hyperparameters.get(parts[1]) {
                            Ok(Box::new(param_value.clone()))
                        } else {
                            Err(SklearsError::InvalidParameter {
                                name: field_path.to_string(),
                                reason: format!("Hyperparameter '{}' not found", parts[1]),
                            })
                        }
                    } else {
                        Ok(Box::new(self.hyperparameters.clone()))
                    }
                }
                Some(&"metadata") => {
                    if parts.len() > 1 {
                        if let Some(meta_value) = self.metadata.get(parts[1]) {
                            Ok(Box::new(meta_value.clone()))
                        } else {
                            Err(SklearsError::InvalidParameter {
                                name: field_path.to_string(),
                                reason: format!("Metadata '{}' not found", parts[1]),
                            })
                        }
                    } else {
                        Ok(Box::new(self.metadata.clone()))
                    }
                }
                _ => Err(SklearsError::InvalidParameter {
                    name: field_path.to_string(),
                    reason: "Invalid field path".to_string(),
                }),
            }
        }

        fn validate_structure(&self, schema: &StructuralSchema) -> Result<ValidationResult> {
            let mut warnings = Vec::new();
            let mut context = std::collections::HashMap::new();

            // Check required fields
            for required_field in &schema.required_fields {
                match required_field.as_str() {
                    "algorithm_name" => {
                        if self.algorithm_name.is_empty() {
                            return Err(SklearsError::InvalidParameter {
                                name: "algorithm_name".to_string(),
                                reason: "Required field cannot be empty".to_string(),
                            });
                        }
                        context.insert("algorithm_name".to_string(), "present".to_string());
                    }
                    "hyperparameters" => {
                        context.insert(
                            "hyperparameters_count".to_string(),
                            self.hyperparameters.len().to_string(),
                        );
                    }
                    _ => {
                        warnings.push(format!("Unknown required field: {required_field}"));
                    }
                }
            }

            Ok(ValidationResult {
                matched: true,
                context,
                warnings,
            })
        }
    }

    /// Pattern matching for complex validation scenarios
    pub fn create_complex_pattern_guard<T>(
        pattern_name: &str,
        validator: impl Fn(&T) -> Result<bool> + Send + Sync + 'static,
        error_message: &str,
    ) -> PatternGuardRule
    where
        T: 'static,
    {
        PatternGuardRule {
            pattern_name: pattern_name.to_string(),
            guard_fn: Box::new(move |value| {
                if let Some(typed_value) = value.downcast_ref::<T>() {
                    validator(typed_value)
                } else {
                    Ok(false)
                }
            }),
            error_message: error_message.to_string(),
            destructure_fn: None,
        }
    }
}

/// Macro for easy destructuring of complex types
#[macro_export]
macro_rules! destructure {
    // Basic field extraction
    ($obj:expr, $field:literal) => {
        $obj.extract_field($field)
    };

    // Multiple field extraction
    ($obj:expr, { $($field:literal),* }) => {
        {
            let mut results = std::collections::HashMap::new();
            $(
                if let Ok(value) = $obj.extract_field($field) {
                    results.insert($field.to_string(), value);
                }
            )*
            results
        }
    };

    // Destructuring with validation
    ($obj:expr, validate: $schema:expr) => {
        $obj.validate_structure(&$schema)
    };
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_rules_numeric() {
        let rules = ValidationRules::new("test_param")
            .add_rule(ValidationRule::Positive)
            .add_rule(ValidationRule::Finite);

        // Valid value
        assert!(rules.validate_numeric(&1.5f64).is_ok());

        // Invalid: non-positive
        assert!(rules.validate_numeric(&0.0f64).is_err());
        assert!(rules.validate_numeric(&-1.0f64).is_err());

        // Invalid: non-finite
        assert!(rules.validate_numeric(&f64::NAN).is_err());
        assert!(rules.validate_numeric(&f64::INFINITY).is_err());
    }

    #[test]
    fn test_validation_rules_range() {
        let rules = ValidationRules::new("test_param")
            .add_rule(ValidationRule::Range { min: 0.0, max: 1.0 });

        // Valid values
        assert!(rules.validate_numeric(&0.5f64).is_ok());
        assert!(rules.validate_numeric(&0.0f64).is_ok());
        assert!(rules.validate_numeric(&1.0f64).is_ok());

        // Invalid values
        assert!(rules.validate_numeric(&-0.1f64).is_err());
        assert!(rules.validate_numeric(&1.1f64).is_err());
    }

    #[test]
    fn test_validation_rules_string() {
        let rules = ValidationRules::new("test_param").add_rule(ValidationRule::OneOf(vec![
            "option1".to_string(),
            "option2".to_string(),
        ]));

        // Valid values
        assert!(rules.validate_string("option1").is_ok());
        assert!(rules.validate_string("option2").is_ok());

        // Invalid value
        assert!(rules.validate_string("option3").is_err());
    }

    #[test]
    fn test_validation_rules_array() {
        let rules = ValidationRules::new("test_param")
            .add_rule(ValidationRule::MinLength(2))
            .add_rule(ValidationRule::MaxLength(5));

        // Valid arrays
        assert!(rules.validate_array(&[1, 2]).is_ok());
        assert!(rules.validate_array(&[1, 2, 3, 4, 5]).is_ok());

        // Invalid: too short
        assert!(rules.validate_array(&[1]).is_err());

        // Invalid: too long
        assert!(rules.validate_array(&[1, 2, 3, 4, 5, 6]).is_err());
    }

    #[test]
    fn test_ml_validation_learning_rate() {
        // Valid learning rates
        assert!(ml::validate_learning_rate(0.01f64).is_ok());
        assert!(ml::validate_learning_rate(0.1f64).is_ok());

        // Invalid: non-positive
        assert!(ml::validate_learning_rate(0.0f64).is_err());
        assert!(ml::validate_learning_rate(-0.1f64).is_err());

        // Invalid: non-finite
        assert!(ml::validate_learning_rate(f64::NAN).is_err());
    }

    #[test]
    fn test_ml_validation_n_clusters() {
        // Valid
        assert!(ml::validate_n_clusters(3, 10).is_ok());
        assert!(ml::validate_n_clusters(10, 10).is_ok());

        // Invalid: zero clusters
        assert!(ml::validate_n_clusters(0, 10).is_err());

        // Invalid: more clusters than samples
        assert!(ml::validate_n_clusters(15, 10).is_err());
    }

    #[test]
    fn test_ml_validation_probability() {
        // Valid probabilities
        assert!(ml::validate_probability(0.0f64).is_ok());
        assert!(ml::validate_probability(0.5f64).is_ok());
        assert!(ml::validate_probability(1.0f64).is_ok());

        // Invalid: out of range
        assert!(ml::validate_probability(-0.1f64).is_err());
        assert!(ml::validate_probability(1.1f64).is_err());

        // Invalid: non-finite
        assert!(ml::validate_probability(f64::NAN).is_err());
    }

    #[test]
    fn test_validation_context() {
        let context = ValidationContext::new("KMeans", "fit").with_data_info(100, 5, "float64");

        let error = SklearsError::InvalidParameter {
            name: "n_clusters".to_string(),
            reason: "must be positive".to_string(),
        };

        let formatted = context.format_error(&error);
        assert!(formatted.contains("KMeans"));
        assert!(formatted.contains("fit"));
        assert!(formatted.contains("100 samples"));
        assert!(formatted.contains("5 features"));
    }
}
