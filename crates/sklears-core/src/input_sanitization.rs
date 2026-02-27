/// Input sanitization for untrusted data
///
/// This module provides comprehensive input sanitization and validation for data
/// coming from untrusted sources to prevent security vulnerabilities and ensure
/// data integrity in machine learning workflows.
use crate::error::{Result, SklearsError};
use crate::types::{Array1, Array2, FloatBounds};
use std::collections::HashMap;

/// Trait for sanitizing input data
pub trait Sanitize {
    /// Sanitize the input and return a cleaned version
    fn sanitize(self) -> Result<Self>
    where
        Self: Sized;

    /// Check if input is safe without modifying it
    fn is_safe(&self) -> bool;

    /// Get detailed information about safety issues
    fn safety_issues(&self) -> Vec<SafetyIssue>;
}

/// Types of safety issues that can be found in input data
#[derive(Debug, Clone, PartialEq)]
pub enum SafetyIssue {
    /// Contains NaN values
    ContainsNaN {
        count: usize,
        locations: Vec<String>,
    },
    /// Contains infinite values
    ContainsInfinity {
        count: usize,
        locations: Vec<String>,
    },
    /// Values outside acceptable range
    OutOfRange {
        min_allowed: f64,
        max_allowed: f64,
        violations: usize,
    },
    /// Array shape is invalid
    InvalidShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Empty data where data is required
    EmptyData,
    /// Suspicious patterns that might indicate attacks
    SuspiciousPattern {
        pattern: String,
        description: String,
    },
    /// String contains potentially dangerous characters
    UnsafeCharacters { characters: Vec<char> },
    /// Data size exceeds limits
    ExceedsLimits { size: usize, limit: usize },
}

/// Configuration for input sanitization
#[derive(Debug, Clone)]
pub struct SanitizationConfig {
    /// Whether to remove NaN values
    pub remove_nan: bool,
    /// Whether to remove infinite values  
    pub remove_infinity: bool,
    /// Whether to clamp values to valid ranges
    pub clamp_values: bool,
    /// Valid range for numeric values
    pub valid_range: Option<(f64, f64)>,
    /// Maximum allowed array size
    pub max_array_size: Option<usize>,
    /// Maximum string length
    pub max_string_length: Option<usize>,
    /// Characters that are not allowed in strings
    pub forbidden_chars: Vec<char>,
    /// Whether to perform deep validation
    pub deep_validation: bool,
}

impl Default for SanitizationConfig {
    fn default() -> Self {
        Self {
            remove_nan: true,
            remove_infinity: true,
            clamp_values: false,
            valid_range: None,
            max_array_size: Some(1_000_000), // 1M elements
            max_string_length: Some(1000),
            forbidden_chars: vec!['\0', '\x01', '\x02', '\x03'],
            deep_validation: true,
        }
    }
}

/// Input sanitizer with configurable policies
#[allow(dead_code)]
pub struct InputSanitizer {
    config: SanitizationConfig,
    validation_cache: std::sync::Mutex<HashMap<String, bool>>,
}

impl InputSanitizer {
    /// Create a new input sanitizer with default configuration
    pub fn new() -> Self {
        Self {
            config: SanitizationConfig::default(),
            validation_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Create a new input sanitizer with custom configuration
    pub fn with_config(config: SanitizationConfig) -> Self {
        Self {
            config,
            validation_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Sanitize a 2D array
    pub fn sanitize_array2<T>(&self, array: Array2<T>) -> Result<Array2<T>>
    where
        T: FloatBounds + Copy,
    {
        // Check size limits
        if let Some(max_size) = self.config.max_array_size {
            if array.len() > max_size {
                return Err(SklearsError::InvalidData {
                    reason: format!("Array size {} exceeds limit {max_size}", array.len()),
                });
            }
        }

        let mut sanitized = array.clone();
        let mut removed_count = 0;

        // Check for NaN and infinity values
        for element in sanitized.iter_mut() {
            if self.config.remove_nan && element.is_nan() {
                *element = T::zero();
                removed_count += 1;
            } else if self.config.remove_infinity && element.is_infinite() {
                *element = if element.is_sign_positive() {
                    T::from(1e10).unwrap_or(T::one())
                } else {
                    T::from(-1e10).unwrap_or(-T::one())
                };
                removed_count += 1;
            }

            // Clamp values if configured
            if let Some((min_val, max_val)) = self.config.valid_range {
                if self.config.clamp_values {
                    let val = element.to_f64().unwrap_or(0.0);
                    if val < min_val {
                        *element = T::from(min_val).unwrap_or(T::zero());
                    } else if val > max_val {
                        *element = T::from(max_val).unwrap_or(T::one());
                    }
                }
            }
        }

        if removed_count > 0 {
            log::warn!("Sanitized {removed_count} problematic values in array");
        }

        Ok(sanitized)
    }

    /// Sanitize a 1D array
    pub fn sanitize_array1<T>(&self, array: Array1<T>) -> Result<Array1<T>>
    where
        T: FloatBounds + Copy,
    {
        // Check size limits
        if let Some(max_size) = self.config.max_array_size {
            if array.len() > max_size {
                return Err(SklearsError::InvalidData {
                    reason: format!("Array size {} exceeds limit {max_size}", array.len()),
                });
            }
        }

        let mut sanitized = array.clone();
        let mut removed_count = 0;

        // Check for NaN and infinity values
        for element in sanitized.iter_mut() {
            if self.config.remove_nan && element.is_nan() {
                *element = T::zero();
                removed_count += 1;
            } else if self.config.remove_infinity && element.is_infinite() {
                *element = if element.is_sign_positive() {
                    T::from(1e10).unwrap_or(T::one())
                } else {
                    T::from(-1e10).unwrap_or(-T::one())
                };
                removed_count += 1;
            }
        }

        if removed_count > 0 {
            log::warn!("Sanitized {removed_count} problematic values in 1D array");
        }

        Ok(sanitized)
    }

    /// Sanitize a string input
    pub fn sanitize_string(&self, input: String) -> Result<String> {
        // Check length limits
        if let Some(max_len) = self.config.max_string_length {
            if input.len() > max_len {
                return Err(SklearsError::InvalidData {
                    reason: format!("String length {} exceeds limit {}", input.len(), max_len),
                });
            }
        }

        // Remove forbidden characters
        let sanitized = input
            .chars()
            .filter(|c| !self.config.forbidden_chars.contains(c))
            .collect::<String>();

        // Check for suspicious patterns
        if self.config.deep_validation {
            self.check_suspicious_patterns(&sanitized)?;
        }

        Ok(sanitized)
    }

    /// Check for suspicious patterns in strings
    fn check_suspicious_patterns(&self, input: &str) -> Result<()> {
        // Check for potential SQL injection patterns
        let sql_patterns = [
            "DROP TABLE",
            "DELETE FROM",
            "INSERT INTO",
            "UPDATE SET",
            "UNION SELECT",
        ];
        for pattern in &sql_patterns {
            if input.to_uppercase().contains(pattern) {
                return Err(SklearsError::InvalidData {
                    reason: format!("Potentially dangerous SQL pattern detected: {pattern}"),
                });
            }
        }

        // Check for script injection patterns
        let script_patterns = ["<script", "javascript:", "onload=", "onerror="];
        for pattern in &script_patterns {
            if input.to_lowercase().contains(pattern) {
                return Err(SklearsError::InvalidData {
                    reason: format!("Potentially dangerous script pattern detected: {pattern}"),
                });
            }
        }

        // Check for path traversal patterns
        if input.contains("../") || input.contains("..\\") {
            return Err(SklearsError::InvalidData {
                reason: "Path traversal pattern detected".to_string(),
            });
        }

        Ok(())
    }

    /// Validate numeric input ranges
    pub fn validate_range<T>(&self, value: T, min: T, max: T) -> Result<()>
    where
        T: PartialOrd + std::fmt::Display,
    {
        if value < min || value > max {
            return Err(SklearsError::InvalidParameter {
                name: "value".to_string(),
                reason: format!("Value {value} is outside valid range [{min}, {max}]"),
            });
        }
        Ok(())
    }

    /// Comprehensive input validation
    pub fn validate_ml_input<T>(
        &self,
        features: &Array2<T>,
        targets: Option<&Array1<T>>,
    ) -> Result<()>
    where
        T: FloatBounds + std::fmt::Display,
    {
        // Check if features array is empty
        if features.is_empty() {
            return Err(SklearsError::InvalidData {
                reason: "Feature array cannot be empty".to_string(),
            });
        }

        // Check for invalid dimensions
        if features.nrows() == 0 || features.ncols() == 0 {
            return Err(SklearsError::InvalidData {
                reason: "Feature array must have positive dimensions".to_string(),
            });
        }

        // Check targets if provided
        if let Some(targets) = targets {
            if targets.len() != features.nrows() {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("{} target values", features.nrows()),
                    actual: format!("{} target values", targets.len()),
                });
            }

            // Check for problematic values in targets
            for (i, &value) in targets.iter().enumerate() {
                if value.is_nan() {
                    return Err(SklearsError::InvalidData {
                        reason: format!("NaN value found in targets at index {i}"),
                    });
                }
                if value.is_infinite() {
                    return Err(SklearsError::InvalidData {
                        reason: format!("Infinite value found in targets at index {i}"),
                    });
                }
            }
        }

        // Check for problematic values in features
        let mut nan_count = 0;
        let mut inf_count = 0;

        for (i, row) in features.outer_iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if value.is_nan() {
                    nan_count += 1;
                    if !self.config.remove_nan {
                        return Err(SklearsError::InvalidData {
                            reason: format!("NaN value found in features at position ({i}, {j})"),
                        });
                    }
                }
                if value.is_infinite() {
                    inf_count += 1;
                    if !self.config.remove_infinity {
                        return Err(SklearsError::InvalidData {
                            reason: format!(
                                "Infinite value found in features at position ({i}, {j})"
                            ),
                        });
                    }
                }
            }
        }

        if nan_count > 0 || inf_count > 0 {
            log::warn!("Found {nan_count} NaN and {inf_count} infinite values in features");
        }

        Ok(())
    }
}

impl Default for InputSanitizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementations of Sanitize trait for common types
impl<T> Sanitize for Array2<T>
where
    T: FloatBounds + Copy,
{
    fn sanitize(self) -> Result<Self> {
        let sanitizer = InputSanitizer::new();
        sanitizer.sanitize_array2(self)
    }

    fn is_safe(&self) -> bool {
        self.safety_issues().is_empty()
    }

    fn safety_issues(&self) -> Vec<SafetyIssue> {
        let mut issues = Vec::new();

        // Check for empty data
        if self.is_empty() {
            issues.push(SafetyIssue::EmptyData);
            return issues;
        }

        // Check for NaN and infinity values
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut nan_locations = Vec::new();
        let mut inf_locations = Vec::new();

        for (i, row) in self.outer_iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if value.is_nan() {
                    nan_count += 1;
                    nan_locations.push(format!("({i}, {j})"));
                }
                if value.is_infinite() {
                    inf_count += 1;
                    inf_locations.push(format!("({i}, {j})"));
                }
            }
        }

        if nan_count > 0 {
            issues.push(SafetyIssue::ContainsNaN {
                count: nan_count,
                locations: nan_locations,
            });
        }

        if inf_count > 0 {
            issues.push(SafetyIssue::ContainsInfinity {
                count: inf_count,
                locations: inf_locations,
            });
        }

        // Check size limits
        if self.len() > 1_000_000 {
            issues.push(SafetyIssue::ExceedsLimits {
                size: self.len(),
                limit: 1_000_000,
            });
        }

        issues
    }
}

impl<T> Sanitize for Array1<T>
where
    T: FloatBounds + Copy,
{
    fn sanitize(self) -> Result<Self> {
        let sanitizer = InputSanitizer::new();
        sanitizer.sanitize_array1(self)
    }

    fn is_safe(&self) -> bool {
        self.safety_issues().is_empty()
    }

    fn safety_issues(&self) -> Vec<SafetyIssue> {
        let mut issues = Vec::new();

        // Check for empty data
        if self.is_empty() {
            issues.push(SafetyIssue::EmptyData);
            return issues;
        }

        // Check for NaN and infinity values
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut nan_locations = Vec::new();
        let mut inf_locations = Vec::new();

        for (i, &value) in self.iter().enumerate() {
            if value.is_nan() {
                nan_count += 1;
                nan_locations.push(format!("[{i}]"));
            }
            if value.is_infinite() {
                inf_count += 1;
                inf_locations.push(format!("[{i}]"));
            }
        }

        if nan_count > 0 {
            issues.push(SafetyIssue::ContainsNaN {
                count: nan_count,
                locations: nan_locations,
            });
        }

        if inf_count > 0 {
            issues.push(SafetyIssue::ContainsInfinity {
                count: inf_count,
                locations: inf_locations,
            });
        }

        issues
    }
}

impl Sanitize for String {
    fn sanitize(self) -> Result<Self> {
        let sanitizer = InputSanitizer::new();
        sanitizer.sanitize_string(self)
    }

    fn is_safe(&self) -> bool {
        self.safety_issues().is_empty()
    }

    fn safety_issues(&self) -> Vec<SafetyIssue> {
        let mut issues = Vec::new();

        // Check length
        if self.len() > 1000 {
            issues.push(SafetyIssue::ExceedsLimits {
                size: self.len(),
                limit: 1000,
            });
        }

        // Check for forbidden characters
        let forbidden_chars = ['\0', '\x01', '\x02', '\x03'];
        let found_chars: Vec<char> = self
            .chars()
            .filter(|c| forbidden_chars.contains(c))
            .collect();

        if !found_chars.is_empty() {
            issues.push(SafetyIssue::UnsafeCharacters {
                characters: found_chars,
            });
        }

        // Check for suspicious patterns
        let dangerous_patterns = [
            ("SQL_INJECTION", "DROP TABLE"),
            ("SCRIPT_INJECTION", "<script"),
            ("PATH_TRAVERSAL", "../"),
        ];

        for (pattern_type, pattern) in &dangerous_patterns {
            if self.to_lowercase().contains(&pattern.to_lowercase()) {
                issues.push(SafetyIssue::SuspiciousPattern {
                    pattern: pattern_type.to_string(),
                    description: format!("Contains potentially dangerous pattern: {pattern}"),
                });
            }
        }

        issues
    }
}

/// Convenience functions for quick sanitization
/// Sanitize machine learning input data
pub fn sanitize_ml_data<T>(
    features: Array2<T>,
    targets: Option<Array1<T>>,
) -> Result<(Array2<T>, Option<Array1<T>>)>
where
    T: FloatBounds + Copy,
{
    let sanitizer = InputSanitizer::new();

    // Validate first
    sanitizer.validate_ml_input(&features, targets.as_ref())?;

    // Sanitize features
    let clean_features = sanitizer.sanitize_array2(features)?;

    // Sanitize targets if provided
    let clean_targets = if let Some(targets) = targets {
        Some(sanitizer.sanitize_array1(targets)?)
    } else {
        None
    };

    Ok((clean_features, clean_targets))
}

/// Quick safety check for ML data
pub fn is_ml_data_safe<T>(features: &Array2<T>, targets: Option<&Array1<T>>) -> bool
where
    T: FloatBounds + Copy,
{
    features.is_safe() && targets.map_or(true, |t| t.is_safe())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Array2;

    #[test]
    fn test_array_sanitization() {
        let mut array: Array2<f64> = Array2::zeros((2, 3));
        array[[0, 0]] = f64::NAN;
        array[[1, 1]] = f64::INFINITY;

        assert!(!array.is_safe());
        let issues = array.safety_issues();
        assert!(!issues.is_empty());

        let sanitized = array.sanitize().expect("sanitize should succeed");
        assert!(sanitized.is_safe());
    }

    #[test]
    fn test_string_sanitization() {
        let dangerous_string = "Hello\0World<script>alert('xss')</script>".to_string();

        assert!(!dangerous_string.is_safe());
        let issues = dangerous_string.safety_issues();
        assert!(!issues.is_empty());

        // This should fail due to dangerous patterns
        assert!(dangerous_string.sanitize().is_err());

        // Test a string with only forbidden characters (no dangerous patterns)
        let string_with_forbidden_chars = "Hello\0World".to_string();
        let sanitized = string_with_forbidden_chars
            .sanitize()
            .expect("sanitize should succeed");
        assert!(!sanitized.contains('\0'));
    }

    #[test]
    fn test_ml_data_validation() {
        let features: Array2<f64> = Array2::zeros((100, 5));
        let targets: Array1<f64> = Array1::zeros(100);

        let sanitizer = InputSanitizer::new();
        assert!(sanitizer
            .validate_ml_input(&features, Some(&targets))
            .is_ok());

        // Test mismatched dimensions
        let bad_targets: Array1<f64> = Array1::zeros(50);
        assert!(sanitizer
            .validate_ml_input(&features, Some(&bad_targets))
            .is_err());
    }

    #[test]
    fn test_sanitization_config() {
        let config = SanitizationConfig {
            max_string_length: Some(10),
            ..Default::default()
        };

        let sanitizer = InputSanitizer::with_config(config);
        let long_string = "This is a very long string that exceeds the limit".to_string();

        assert!(sanitizer.sanitize_string(long_string).is_err());
    }

    #[test]
    fn test_range_validation() {
        let sanitizer = InputSanitizer::new();

        assert!(sanitizer.validate_range(5.0, 0.0, 10.0).is_ok());
        assert!(sanitizer.validate_range(-1.0, 0.0, 10.0).is_err());
        assert!(sanitizer.validate_range(15.0, 0.0, 10.0).is_err());
    }
}
