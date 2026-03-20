//! API Generator Configuration Module
//!
//! This module provides configuration structures and settings for the API reference generator,
//! including output formats, validation options, and generation parameters.

use serde::{Deserialize, Serialize};

/// Configuration for the API reference generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Output format for the generated documentation
    pub output_format: OutputFormat,
    /// Whether to include code examples in the output
    pub include_examples: bool,
    /// Whether to validate examples by compilation
    pub validate_examples: bool,
    /// Whether to extract type information
    pub include_type_info: bool,
    /// Whether to generate cross-references
    pub include_cross_refs: bool,
    /// Maximum depth for trait hierarchy analysis
    pub max_hierarchy_depth: usize,
    /// Whether to include private items
    pub include_private: bool,
    /// Custom CSS/styling for HTML output
    pub custom_styling: Option<String>,
}

impl GeneratorConfig {
    /// Create a new generator configuration with default settings
    pub fn new() -> Self {
        Self {
            output_format: OutputFormat::Json,
            include_examples: true,
            validate_examples: false, // Disabled by default for performance
            include_type_info: true,
            include_cross_refs: true,
            max_hierarchy_depth: 5,
            include_private: false,
            custom_styling: None,
        }
    }

    /// Set the output format
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Enable or disable example validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_examples = validate;
        self
    }

    /// Enable or disable example inclusion
    pub fn with_examples(mut self, include: bool) -> Self {
        self.include_examples = include;
        self
    }

    /// Set maximum hierarchy depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_hierarchy_depth = depth;
        self
    }

    /// Include private items in documentation
    pub fn with_private_items(mut self, include: bool) -> Self {
        self.include_private = include;
        self
    }

    /// Set custom CSS styling for HTML output
    pub fn with_custom_styling(mut self, css: Option<String>) -> Self {
        self.custom_styling = css;
        self
    }

    /// Enable or disable type information extraction
    pub fn with_type_info(mut self, include: bool) -> Self {
        self.include_type_info = include;
        self
    }

    /// Enable or disable cross-reference generation
    pub fn with_cross_refs(mut self, include: bool) -> Self {
        self.include_cross_refs = include;
        self
    }
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported output formats for API documentation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutputFormat {
    /// JSON format for programmatic consumption
    #[default]
    Json,
    /// HTML format for web display
    Html,
    /// Markdown format for text display
    Markdown,
    /// Interactive WebAssembly playground
    Interactive,
    /// OpenAPI specification
    OpenApi,
}

impl OutputFormat {
    /// Get the file extension for this format
    pub fn file_extension(&self) -> &'static str {
        match self {
            OutputFormat::Json => "json",
            OutputFormat::Html => "html",
            OutputFormat::Markdown => "md",
            OutputFormat::Interactive => "html",
            OutputFormat::OpenApi => "yaml",
        }
    }

    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            OutputFormat::Json => "application/json",
            OutputFormat::Html => "text/html",
            OutputFormat::Markdown => "text/markdown",
            OutputFormat::Interactive => "text/html",
            OutputFormat::OpenApi => "application/x-yaml",
        }
    }

    /// Check if this format supports interactive features
    pub fn supports_interactive(&self) -> bool {
        matches!(self, OutputFormat::Html | OutputFormat::Interactive)
    }

    /// Check if this format supports syntax highlighting
    pub fn supports_syntax_highlighting(&self) -> bool {
        matches!(
            self,
            OutputFormat::Html | OutputFormat::Interactive | OutputFormat::Markdown
        )
    }
}

/// Configuration for playground generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundConfig {
    /// Enable WebAssembly compilation
    pub wasm_enabled: bool,
    /// Enable real-time code execution
    pub real_time_execution: bool,
    /// Enable syntax highlighting
    pub syntax_highlighting: bool,
    /// Enable auto-completion
    pub auto_completion: bool,
    /// Enable error highlighting
    pub error_highlighting: bool,
    /// Available crates for import
    pub available_crates: Vec<String>,
    /// Number of examples available
    pub example_count: usize,
    /// Number of traits documented
    pub trait_count: usize,
    /// Number of types documented
    pub type_count: usize,
}

impl PlaygroundConfig {
    /// Create a new playground configuration with default settings
    pub fn new() -> Self {
        Self {
            wasm_enabled: true,
            real_time_execution: true,
            syntax_highlighting: true,
            auto_completion: true,
            error_highlighting: true,
            available_crates: vec!["sklears-core".to_string()],
            example_count: 0,
            trait_count: 0,
            type_count: 0,
        }
    }

    /// Enable or disable WebAssembly compilation
    pub fn with_wasm(mut self, enabled: bool) -> Self {
        self.wasm_enabled = enabled;
        self
    }

    /// Enable or disable real-time execution
    pub fn with_real_time_execution(mut self, enabled: bool) -> Self {
        self.real_time_execution = enabled;
        self
    }

    /// Enable or disable syntax highlighting
    pub fn with_syntax_highlighting(mut self, enabled: bool) -> Self {
        self.syntax_highlighting = enabled;
        self
    }

    /// Enable or disable auto-completion
    pub fn with_auto_completion(mut self, enabled: bool) -> Self {
        self.auto_completion = enabled;
        self
    }

    /// Enable or disable error highlighting
    pub fn with_error_highlighting(mut self, enabled: bool) -> Self {
        self.error_highlighting = enabled;
        self
    }

    /// Set available crates for import
    pub fn with_available_crates(mut self, crates: Vec<String>) -> Self {
        self.available_crates = crates;
        self
    }

    /// Update statistics
    pub fn with_stats(
        mut self,
        example_count: usize,
        trait_count: usize,
        type_count: usize,
    ) -> Self {
        self.example_count = example_count;
        self.trait_count = trait_count;
        self.type_count = type_count;
        self
    }
}

impl Default for PlaygroundConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation configuration for code examples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable compilation validation
    pub compile_check: bool,
    /// Enable runtime validation
    pub runtime_check: bool,
    /// Timeout for validation (in milliseconds)
    pub timeout_ms: u64,
    /// Maximum memory usage for validation (in bytes)
    pub max_memory: usize,
    /// Enable linting during validation
    pub lint_check: bool,
    /// Additional validation flags
    pub extra_flags: Vec<String>,
}

impl ValidationConfig {
    /// Create a new validation configuration with default settings
    pub fn new() -> Self {
        Self {
            compile_check: true,
            runtime_check: false,
            timeout_ms: 5000,
            max_memory: 1024 * 1024 * 50, // 50MB
            lint_check: true,
            extra_flags: Vec::new(),
        }
    }

    /// Enable or disable compilation checking
    pub fn with_compile_check(mut self, enabled: bool) -> Self {
        self.compile_check = enabled;
        self
    }

    /// Enable or disable runtime checking
    pub fn with_runtime_check(mut self, enabled: bool) -> Self {
        self.runtime_check = enabled;
        self
    }

    /// Set validation timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set maximum memory usage
    pub fn with_max_memory(mut self, max_memory: usize) -> Self {
        self.max_memory = max_memory;
        self
    }

    /// Enable or disable linting
    pub fn with_lint_check(mut self, enabled: bool) -> Self {
        self.lint_check = enabled;
        self
    }

    /// Add extra validation flags
    pub fn with_extra_flags(mut self, flags: Vec<String>) -> Self {
        self.extra_flags = flags;
        self
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Theme configuration for output styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    /// Primary theme color
    pub primary_color: String,
    /// Secondary theme color
    pub secondary_color: String,
    /// Background color
    pub background_color: String,
    /// Text color
    pub text_color: String,
    /// Code block background color
    pub code_background: String,
    /// Code text color
    pub code_text_color: String,
    /// Font family for regular text
    pub font_family: String,
    /// Font family for code
    pub code_font_family: String,
    /// Enable dark mode
    pub dark_mode: bool,
}

impl ThemeConfig {
    /// Create a new theme configuration with default light theme
    pub fn light_theme() -> Self {
        Self {
            primary_color: "#0066cc".to_string(),
            secondary_color: "#ff6b35".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#333333".to_string(),
            code_background: "#f5f5f5".to_string(),
            code_text_color: "#333333".to_string(),
            font_family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif".to_string(),
            code_font_family: "'Monaco', 'Menlo', 'Ubuntu Mono', monospace".to_string(),
            dark_mode: false,
        }
    }

    /// Create a new theme configuration with default dark theme
    pub fn dark_theme() -> Self {
        Self {
            primary_color: "#4da6ff".to_string(),
            secondary_color: "#ff8c66".to_string(),
            background_color: "#1e1e1e".to_string(),
            text_color: "#d4d4d4".to_string(),
            code_background: "#2d2d30".to_string(),
            code_text_color: "#d4d4d4".to_string(),
            font_family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif".to_string(),
            code_font_family: "'Monaco', 'Menlo', 'Ubuntu Mono', monospace".to_string(),
            dark_mode: true,
        }
    }

    /// Generate CSS variables for this theme
    pub fn to_css_variables(&self) -> String {
        format!(
            r#":root {{
    --primary-color: {};
    --secondary-color: {};
    --background-color: {};
    --text-color: {};
    --code-background: {};
    --code-text-color: {};
    --font-family: {};
    --code-font-family: {};
}}"#,
            self.primary_color,
            self.secondary_color,
            self.background_color,
            self.text_color,
            self.code_background,
            self.code_text_color,
            self.font_family,
            self.code_font_family
        )
    }
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self::light_theme()
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_config_creation() {
        let config = GeneratorConfig::new();
        assert_eq!(config.output_format, OutputFormat::Json);
        assert!(config.include_examples);
        assert!(!config.validate_examples);
        assert!(config.include_type_info);
        assert!(config.include_cross_refs);
        assert_eq!(config.max_hierarchy_depth, 5);
        assert!(!config.include_private);
        assert!(config.custom_styling.is_none());
    }

    #[test]
    fn test_generator_config_builder() {
        let config = GeneratorConfig::new()
            .with_output_format(OutputFormat::Html)
            .with_validation(true)
            .with_examples(false)
            .with_max_depth(10)
            .with_private_items(true);

        assert_eq!(config.output_format, OutputFormat::Html);
        assert!(config.validate_examples);
        assert!(!config.include_examples);
        assert_eq!(config.max_hierarchy_depth, 10);
        assert!(config.include_private);
    }

    #[test]
    fn test_output_format_properties() {
        assert_eq!(OutputFormat::Json.file_extension(), "json");
        assert_eq!(OutputFormat::Html.file_extension(), "html");
        assert_eq!(OutputFormat::Markdown.file_extension(), "md");
        assert_eq!(OutputFormat::Interactive.file_extension(), "html");
        assert_eq!(OutputFormat::OpenApi.file_extension(), "yaml");

        assert_eq!(OutputFormat::Json.mime_type(), "application/json");
        assert_eq!(OutputFormat::Html.mime_type(), "text/html");
        assert_eq!(OutputFormat::Markdown.mime_type(), "text/markdown");

        assert!(!OutputFormat::Json.supports_interactive());
        assert!(OutputFormat::Html.supports_interactive());
        assert!(OutputFormat::Interactive.supports_interactive());

        assert!(!OutputFormat::Json.supports_syntax_highlighting());
        assert!(OutputFormat::Html.supports_syntax_highlighting());
        assert!(OutputFormat::Markdown.supports_syntax_highlighting());
    }

    #[test]
    fn test_playground_config() {
        let config = PlaygroundConfig::new()
            .with_wasm(false)
            .with_real_time_execution(false)
            .with_available_crates(vec!["sklears-core".to_string(), "std".to_string()])
            .with_stats(10, 5, 8);

        assert!(!config.wasm_enabled);
        assert!(!config.real_time_execution);
        assert!(config.syntax_highlighting);
        assert_eq!(config.available_crates.len(), 2);
        assert_eq!(config.example_count, 10);
        assert_eq!(config.trait_count, 5);
        assert_eq!(config.type_count, 8);
    }

    #[test]
    fn test_validation_config() {
        let config = ValidationConfig::new()
            .with_compile_check(false)
            .with_runtime_check(true)
            .with_timeout(10000)
            .with_max_memory(1024 * 1024 * 100);

        assert!(!config.compile_check);
        assert!(config.runtime_check);
        assert_eq!(config.timeout_ms, 10000);
        assert_eq!(config.max_memory, 1024 * 1024 * 100);
        assert!(config.lint_check);
    }

    #[test]
    fn test_theme_config_light() {
        let theme = ThemeConfig::light_theme();
        assert_eq!(theme.primary_color, "#0066cc");
        assert_eq!(theme.background_color, "#ffffff");
        assert!(!theme.dark_mode);

        let css = theme.to_css_variables();
        assert!(css.contains("--primary-color: #0066cc"));
        assert!(css.contains("--background-color: #ffffff"));
    }

    #[test]
    fn test_theme_config_dark() {
        let theme = ThemeConfig::dark_theme();
        assert_eq!(theme.primary_color, "#4da6ff");
        assert_eq!(theme.background_color, "#1e1e1e");
        assert!(theme.dark_mode);

        let css = theme.to_css_variables();
        assert!(css.contains("--primary-color: #4da6ff"));
        assert!(css.contains("--background-color: #1e1e1e"));
    }

    #[test]
    fn test_config_serialization() {
        let config = GeneratorConfig::new().with_output_format(OutputFormat::Html);
        let serialized = serde_json::to_string(&config).unwrap_or_default();
        let deserialized: GeneratorConfig =
            serde_json::from_str(&serialized).expect("valid JSON operation");

        assert_eq!(config.output_format, deserialized.output_format);
        assert_eq!(config.include_examples, deserialized.include_examples);
    }

    #[test]
    fn test_default_implementations() {
        let _: GeneratorConfig = Default::default();
        let _: OutputFormat = Default::default();
        let _: PlaygroundConfig = Default::default();
        let _: ValidationConfig = Default::default();
        let _: ThemeConfig = Default::default();
    }
}
