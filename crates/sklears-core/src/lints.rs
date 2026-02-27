/// Custom lints for ML-specific patterns and best practices
///
/// This module provides custom linting rules that help enforce ML-specific
/// coding patterns and catch common mistakes in machine learning code.
///
/// # Lint Categories
///
/// - **Data Validation**: Ensure proper input validation
/// - **Memory Safety**: Check for potential memory issues in ML workloads
/// - **Performance**: Identify performance anti-patterns
use std::collections::HashMap;
/// - **API Usage**: Enforce proper API usage patterns
/// - **Numerical Stability**: Catch numerical stability issues
///
/// # Usage
///
/// These lints can be enabled with the `custom_lints` feature flag:
///
/// ```toml
/// [dependencies]
/// sklears-core = { version = "0.1", features = ["custom_lints"] }
/// ```
///
/// Individual lints can be configured in your Cargo.toml:
///
/// ```toml
/// [lints.rust]
/// sklears_data_validation = "warn"
/// sklears_memory_safety = "deny"
/// ```
/// Trait for defining custom lint rules
pub trait LintRule {
    /// Name of the lint rule
    fn name(&self) -> &'static str;

    /// Description of what the lint checks for
    fn description(&self) -> &'static str;

    /// Severity level of the lint
    fn severity(&self) -> LintSeverity;

    /// Category of the lint
    fn category(&self) -> LintCategory;

    /// Example of code that would trigger this lint
    fn example_violation(&self) -> &'static str;

    /// Example of how to fix the violation
    fn example_fix(&self) -> &'static str;

    /// Additional help text
    fn help_text(&self) -> Option<&'static str> {
        None
    }
}

/// Severity levels for lints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintSeverity {
    /// Allow the pattern (information only)
    Allow,
    /// Warn about the pattern
    Warn,
    /// Deny the pattern (error)
    Deny,
    /// Forbid the pattern (hard error)
    Forbid,
}

/// Categories of ML-specific lints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LintCategory {
    /// Data validation and preprocessing
    DataValidation,
    /// Memory safety and management
    MemorySafety,
    /// Performance optimization
    Performance,
    /// API usage patterns
    ApiUsage,
    /// Numerical stability
    NumericalStability,
    /// Model lifecycle management
    ModelLifecycle,
    /// Feature engineering
    FeatureEngineering,
    /// Testing and validation
    Testing,
}

// =============================================================================
// Specific Lint Rules
// =============================================================================

/// Lint for missing data validation
pub struct DataValidationLint;

impl LintRule for DataValidationLint {
    fn name(&self) -> &'static str {
        "sklears_data_validation"
    }

    fn description(&self) -> &'static str {
        "Checks for missing input data validation in ML algorithms"
    }

    fn severity(&self) -> LintSeverity {
        LintSeverity::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::DataValidation
    }

    fn example_violation(&self) -> &'static str {
        r#"
fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    // Missing validation - should check shapes, NaN values, etc.
    self.train_model(x, y)
}
        "#
    }

    fn example_fix(&self) -> &'static str {
        r#"
fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    // Validate input data
    if x.nrows() != y.len() {
        return Err(SklearsError::InvalidInput("Shape mismatch".to_string()));
    }
    if x.iter().any(|v| v.is_nan()) {
        return Err(SklearsError::InvalidInput("NaN values found".to_string()));
    }
    
    self.train_model(x, y)
}
        "#
    }

    fn help_text(&self) -> Option<&'static str> {
        Some(
            "Always validate input data before processing. Check for:\n\
              - Shape compatibility\n\
              - NaN or infinite values\n\
              - Empty datasets\n\
              - Data type consistency",
        )
    }
}

/// Lint for potential memory leaks in ML workloads
pub struct MemoryLeakLint;

impl LintRule for MemoryLeakLint {
    fn name(&self) -> &'static str {
        "sklears_memory_leak"
    }

    fn description(&self) -> &'static str {
        "Detects potential memory leaks in iterative ML algorithms"
    }

    fn severity(&self) -> LintSeverity {
        LintSeverity::Deny
    }

    fn category(&self) -> LintCategory {
        LintCategory::MemorySafety
    }

    fn example_violation(&self) -> &'static str {
        r#"
fn train_epochs(&mut self, data: &Dataset) -> Result<()> {
    for epoch in 0..self.max_epochs {
        let mut gradients = Vec::new();
        
        for batch in data.batches() {
            gradients.push(self.compute_gradients(batch));
            // Memory grows unbounded - gradients accumulate
        }
        
        self.apply_gradients(&gradients);
    }
    Ok(())
}
        "#
    }

    fn example_fix(&self) -> &'static str {
        r#"
fn train_epochs(&mut self, data: &Dataset) -> Result<()> {
    for epoch in 0..self.max_epochs {
        let mut accumulated_gradients = self.zero_gradients();
        
        for batch in data.batches() {
            let gradients = self.compute_gradients(batch);
            self.accumulate_gradients(&mut accumulated_gradients, &gradients);
            // Process gradients incrementally instead of storing all
        }
        
        self.apply_gradients(&accumulated_gradients);
    }
    Ok(())
}
        "#
    }

    fn help_text(&self) -> Option<&'static str> {
        Some(
            "In iterative algorithms, avoid accumulating large amounts of data.\n\
              Use streaming or incremental processing instead.",
        )
    }
}

/// Lint for inefficient array operations
pub struct ArrayPerformanceLint;

impl LintRule for ArrayPerformanceLint {
    fn name(&self) -> &'static str {
        "sklears_array_performance"
    }

    fn description(&self) -> &'static str {
        "Identifies inefficient array operations that could be optimized"
    }

    fn severity(&self) -> LintSeverity {
        LintSeverity::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Performance
    }

    fn example_violation(&self) -> &'static str {
        r#"
fn dot_product(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    // Inefficient: manual loop instead of BLAS
    let mut result = 0.0;
    for i in 0..a.len() {
        result += a[i] * b[i];
    }
    result
}
        "#
    }

    fn example_fix(&self) -> &'static str {
        r#"
fn dot_product(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    // Efficient: use optimized BLAS operations
    a.dot(b)
}
        "#
    }

    fn help_text(&self) -> Option<&'static str> {
        Some(
            "Use optimized BLAS operations instead of manual loops for:\n\
              - Matrix multiplication\n\
              - Vector operations\n\
              - Element-wise operations",
        )
    }
}

/// Lint for improper API usage patterns
pub struct ApiUsageLint;

impl LintRule for ApiUsageLint {
    fn name(&self) -> &'static str {
        "sklears_api_usage"
    }

    fn description(&self) -> &'static str {
        "Checks for improper usage of sklears APIs"
    }

    fn severity(&self) -> LintSeverity {
        LintSeverity::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::ApiUsage
    }

    fn example_violation(&self) -> &'static str {
        r#"
// Using trained model before fitting
let model = LinearRegression::new();
let predictions = model.predict(&test_data)?; // Error: not fitted
        "#
    }

    fn example_fix(&self) -> &'static str {
        r#"
// Proper model lifecycle
let model = LinearRegression::new();
let fitted_model = model.fit(&train_x, &train_y)?;
let predictions = fitted_model.predict(&test_data)?;
        "#
    }

    fn help_text(&self) -> Option<&'static str> {
        Some(
            "Follow the proper ML model lifecycle:\n\
              1. Create model\n\
              2. Fit on training data\n\
              3. Use fitted model for prediction",
        )
    }
}

/// Lint for numerical stability issues
pub struct NumericalStabilityLint;

impl LintRule for NumericalStabilityLint {
    fn name(&self) -> &'static str {
        "sklears_numerical_stability"
    }

    fn description(&self) -> &'static str {
        "Detects patterns that may cause numerical instability"
    }

    fn severity(&self) -> LintSeverity {
        LintSeverity::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::NumericalStability
    }

    fn example_violation(&self) -> &'static str {
        r#"
fn log_softmax(&self, x: &Array1<f64>) -> Array1<f64> {
    let exp_x: Array1<f64> = x.mapv(|v| v.exp());
    let sum_exp = exp_x.sum();
    exp_x.mapv(|v| (v / sum_exp).ln()) // Numerically unstable
}
        "#
    }

    fn example_fix(&self) -> &'static str {
        r#"
fn log_softmax(&self, x: &Array1<f64>) -> Array1<f64> {
    let max_x = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let shifted = x.mapv(|v| v - max_x);
    let log_sum_exp = shifted.mapv(|v| v.exp()).sum().ln();
    shifted.mapv(|v| v - log_sum_exp) // Numerically stable
}
        "#
    }

    fn help_text(&self) -> Option<&'static str> {
        Some(
            "Use numerically stable algorithms:\n\
              - Subtract max before exp() operations\n\
              - Use log-space arithmetic when possible\n\
              - Check for overflow/underflow conditions",
        )
    }
}

/// Lint for missing model validation
pub struct ModelValidationLint;

impl LintRule for ModelValidationLint {
    fn name(&self) -> &'static str {
        "sklears_model_validation"
    }

    fn description(&self) -> &'static str {
        "Ensures proper model validation and testing practices"
    }

    fn severity(&self) -> LintSeverity {
        LintSeverity::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Testing
    }

    fn example_violation(&self) -> &'static str {
        r#"
fn train_model(&mut self, data: &Dataset) -> Result<()> {
    self.fit(&data.features, &data.targets)?;
    // Missing: no validation or testing
    Ok(())
}
        "#
    }

    fn example_fix(&self) -> &'static str {
        r#"
fn train_model(&mut self, data: &Dataset) -> Result<()> {
    let (train, test) = data.train_test_split(0.8)?;
    
    self.fit(&train.features, &train.targets)?;
    
    // Validate model performance
    let predictions = self.predict(&test.features)?;
    let score = self.score(&test.features, &test.targets)?;
    
    if score < self.min_acceptable_score {
        return Err(SklearsError::InvalidOperation(
            "Model performance below threshold".to_string()
        ));
    }
    
    Ok(())
}
        "#
    }

    fn help_text(&self) -> Option<&'static str> {
        Some(
            "Always validate model performance:\n\
              - Use train/validation/test splits\n\
              - Implement cross-validation\n\
              - Monitor for overfitting",
        )
    }
}

// =============================================================================
// Lint Registry and Management
// =============================================================================

/// Registry of all available lints
pub struct LintRegistry {
    rules: HashMap<&'static str, Box<dyn LintRule>>,
    enabled_rules: HashMap<&'static str, LintSeverity>,
}

impl LintRegistry {
    /// Create a new lint registry with default rules
    pub fn new() -> Self {
        let mut registry = Self {
            rules: HashMap::new(),
            enabled_rules: HashMap::new(),
        };

        // Register default lint rules
        registry.register(Box::new(DataValidationLint));
        registry.register(Box::new(MemoryLeakLint));
        registry.register(Box::new(ArrayPerformanceLint));
        registry.register(Box::new(ApiUsageLint));
        registry.register(Box::new(NumericalStabilityLint));
        registry.register(Box::new(ModelValidationLint));

        registry
    }

    /// Register a new lint rule
    pub fn register(&mut self, rule: Box<dyn LintRule>) {
        let name = rule.name();
        let severity = rule.severity();
        self.enabled_rules.insert(name, severity);
        self.rules.insert(name, rule);
    }

    /// Enable a lint rule with specified severity
    pub fn enable_rule(&mut self, name: &str, severity: LintSeverity) -> Result<(), String> {
        if let Some(rule) = self.rules.get(name) {
            let static_name = rule.name(); // Get the static name from the rule
            self.enabled_rules.insert(static_name, severity);
            Ok(())
        } else {
            Err(format!("Unknown lint rule: {name}"))
        }
    }

    /// Disable a lint rule
    pub fn disable_rule(&mut self, name: &str) {
        if let Some(rule) = self.rules.get(name) {
            let static_name = rule.name(); // Get the static name from the rule
            self.enabled_rules.remove(static_name);
        }
    }

    /// Get all available lint rules
    pub fn available_rules(&self) -> Vec<&str> {
        self.rules.keys().copied().collect()
    }

    /// Get enabled lint rules
    pub fn enabled_rules(&self) -> &HashMap<&'static str, LintSeverity> {
        &self.enabled_rules
    }

    /// Get lint rule by name
    pub fn get_rule(&self, name: &str) -> Option<&dyn LintRule> {
        self.rules.get(name).map(|r| r.as_ref())
    }

    /// Get lint rules by category
    pub fn rules_by_category(&self, category: LintCategory) -> Vec<&dyn LintRule> {
        self.rules
            .values()
            .filter(|rule| rule.category() == category)
            .map(|rule| rule.as_ref())
            .collect()
    }

    /// Generate lint configuration for Cargo.toml
    pub fn generate_cargo_config(&self) -> String {
        let mut config = String::new();
        config.push_str("[lints.rust]\n");

        for (name, severity) in &self.enabled_rules {
            let severity_str = match severity {
                LintSeverity::Allow => "allow",
                LintSeverity::Warn => "warn",
                LintSeverity::Deny => "deny",
                LintSeverity::Forbid => "forbid",
            };
            config.push_str(&format!("{name} = \"{severity_str}\"\n"));
        }

        config
    }

    /// Generate lint documentation
    pub fn generate_documentation(&self) -> String {
        let mut doc = String::new();
        doc.push_str("# SKLears Custom Lints\n\n");

        // Group by category
        let mut categories: HashMap<LintCategory, Vec<&dyn LintRule>> = HashMap::new();
        for rule in self.rules.values() {
            categories
                .entry(rule.category())
                .or_default()
                .push(rule.as_ref());
        }

        for (category, rules) in categories {
            doc.push_str(&format!("## {category:?} Lints\n\n"));

            for rule in rules {
                doc.push_str(&format!("### {}\n\n", rule.name()));
                doc.push_str(&format!("**Description**: {}\n\n", rule.description()));
                doc.push_str(&format!("**Severity**: {:?}\n\n", rule.severity()));

                if let Some(help) = rule.help_text() {
                    doc.push_str(&format!("**Help**: {help}\n\n"));
                }

                doc.push_str("**Example violation**:\n");
                doc.push_str("```rust\n");
                doc.push_str(rule.example_violation());
                doc.push_str("\n```\n\n");

                doc.push_str("**Example fix**:\n");
                doc.push_str("```rust\n");
                doc.push_str(rule.example_fix());
                doc.push_str("\n```\n\n");
            }
        }

        doc
    }
}

impl Default for LintRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for lint enforcement
#[derive(Debug, Clone)]
pub struct LintConfig {
    /// Whether to enable custom lints
    pub enabled: bool,
    /// Default severity for new lints
    pub default_severity: LintSeverity,
    /// Whether to fail build on lint violations
    pub fail_on_violations: bool,
    /// Maximum number of violations before failing
    pub max_violations: Option<usize>,
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_severity: LintSeverity::Warn,
            fail_on_violations: false,
            max_violations: Some(100),
        }
    }
}

// =============================================================================
// Lint Utilities
// =============================================================================

/// Utility functions for working with lints
pub mod utils {
    use super::*;

    /// Check if a lint should be applied based on configuration
    pub fn should_apply_lint(
        rule_name: &str,
        config: &LintConfig,
        registry: &LintRegistry,
    ) -> bool {
        if !config.enabled {
            return false;
        }

        registry.enabled_rules().contains_key(rule_name)
    }

    /// Format a lint violation message
    pub fn format_violation(rule: &dyn LintRule, location: &str, message: &str) -> String {
        format!(
            "[{}] {}: {} ({})",
            rule.name(),
            location,
            message,
            rule.description()
        )
    }

    /// Generate a quick-fix suggestion
    pub fn suggest_fix(rule: &dyn LintRule) -> String {
        let mut suggestion = String::new();
        suggestion.push_str("Suggested fix:\n");
        suggestion.push_str(rule.example_fix());

        if let Some(help) = rule.help_text() {
            suggestion.push_str("\n\nAdditional help:\n");
            suggestion.push_str(help);
        }

        suggestion
    }
}

// =============================================================================
// Tests
// =============================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_registry() {
        let mut registry = LintRegistry::new();

        // Test that default rules are registered
        assert!(!registry.available_rules().is_empty());
        assert!(registry
            .available_rules()
            .contains(&"sklears_data_validation"));

        // Test enabling/disabling rules
        assert!(registry
            .enable_rule("sklears_data_validation", LintSeverity::Deny)
            .is_ok());
        assert!(registry
            .enable_rule("nonexistent_rule", LintSeverity::Warn)
            .is_err());

        registry.disable_rule("sklears_data_validation");
        assert!(!registry
            .enabled_rules()
            .contains_key("sklears_data_validation"));
    }

    #[test]
    fn test_lint_rules() {
        let data_lint = DataValidationLint;
        assert_eq!(data_lint.name(), "sklears_data_validation");
        assert_eq!(data_lint.category(), LintCategory::DataValidation);
        assert_eq!(data_lint.severity(), LintSeverity::Warn);
        assert!(!data_lint.example_violation().is_empty());
        assert!(!data_lint.example_fix().is_empty());
    }

    #[test]
    fn test_rules_by_category() {
        let registry = LintRegistry::new();
        let data_rules = registry.rules_by_category(LintCategory::DataValidation);
        assert!(!data_rules.is_empty());

        for rule in data_rules {
            assert_eq!(rule.category(), LintCategory::DataValidation);
        }
    }

    #[test]
    fn test_cargo_config_generation() {
        let mut registry = LintRegistry::new();
        registry
            .enable_rule("sklears_data_validation", LintSeverity::Warn)
            .expect("expected valid value");
        registry
            .enable_rule("sklears_memory_leak", LintSeverity::Deny)
            .expect("expected valid value");

        let config = registry.generate_cargo_config();
        assert!(config.contains("sklears_data_validation = \"warn\""));
        assert!(config.contains("sklears_memory_leak = \"deny\""));
    }

    #[test]
    fn test_documentation_generation() {
        let registry = LintRegistry::new();
        let doc = registry.generate_documentation();

        assert!(doc.contains("# SKLears Custom Lints"));
        assert!(doc.contains("sklears_data_validation"));
        assert!(doc.contains("Example violation"));
        assert!(doc.contains("Example fix"));
    }

    #[test]
    fn test_lint_config() {
        let config = LintConfig::default();
        assert!(config.enabled);
        assert_eq!(config.default_severity, LintSeverity::Warn);
    }

    #[test]
    fn test_lint_utils() {
        let registry = LintRegistry::new();
        let config = LintConfig::default();

        // Test lint application check
        assert!(utils::should_apply_lint(
            "sklears_data_validation",
            &config,
            &registry
        ));

        let disabled_config = LintConfig {
            enabled: false,
            ..Default::default()
        };
        assert!(!utils::should_apply_lint(
            "sklears_data_validation",
            &disabled_config,
            &registry
        ));

        // Test formatting
        let rule = DataValidationLint;
        let message = utils::format_violation(&rule, "src/main.rs:42", "Missing validation");
        assert!(message.contains("sklears_data_validation"));
        assert!(message.contains("src/main.rs:42"));

        // Test fix suggestion
        let suggestion = utils::suggest_fix(&rule);
        assert!(suggestion.contains("Suggested fix"));
    }
}
