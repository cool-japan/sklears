//! Output Formatters and Document Generators
//!
//! This module provides comprehensive formatting capabilities for API documentation,
//! supporting multiple output formats including JSON, HTML, Markdown, and interactive
//! web-based documentation with live code examples.

use crate::api_analyzers::{CrossReferenceBuilder, ExampleValidator, TraitAnalyzer, TypeExtractor};
use crate::api_data_structures::{
    ApiMetadata, ApiReference, CodeExample, CrateInfo, TraitInfo, TypeInfo,
};
use crate::api_generator_config::{GeneratorConfig, OutputFormat, ThemeConfig};
use crate::error::{Result, SklearsError};
use std::collections::HashMap;
use std::path::PathBuf;

// ================================================================================================
// MAIN API REFERENCE GENERATOR
// ================================================================================================

/// Main API reference generator with comprehensive formatting capabilities
#[derive(Debug)]
pub struct ApiReferenceGenerator {
    config: GeneratorConfig,
    trait_analyzer: TraitAnalyzer,
    type_extractor: TypeExtractor,
    example_validator: ExampleValidator,
    cross_ref_builder: CrossReferenceBuilder,
    formatter: DocumentFormatter,
}

impl ApiReferenceGenerator {
    /// Create a new API reference generator with the given configuration
    pub fn new(config: GeneratorConfig) -> Self {
        Self {
            formatter: DocumentFormatter::new(config.clone()),
            config: config.clone(),
            trait_analyzer: TraitAnalyzer::new(config.clone()),
            type_extractor: TypeExtractor::new(config.clone()),
            example_validator: ExampleValidator::new(),
            cross_ref_builder: CrossReferenceBuilder::new(),
        }
    }

    /// Generate API reference from a Rust crate
    pub fn generate_from_crate(&mut self, crate_name: &str) -> Result<ApiReference> {
        let crate_info = self.analyze_crate(crate_name)?;

        let traits = if self.config.include_type_info {
            self.trait_analyzer.analyze_traits(&crate_info)?
        } else {
            Vec::new()
        };

        let types = if self.config.include_type_info {
            self.type_extractor.extract_types(&crate_info)?
        } else {
            Vec::new()
        };

        let examples = if self.config.include_examples {
            let raw_examples = self.extract_examples(&crate_info)?;
            if self.config.validate_examples {
                self.example_validator.validate_examples(&raw_examples)?
            } else {
                raw_examples
            }
        } else {
            Vec::new()
        };

        let cross_refs = if self.config.include_cross_refs {
            self.cross_ref_builder
                .build_cross_references(&traits, &types)?
        } else {
            HashMap::new()
        };

        Ok(ApiReference {
            crate_name: crate_name.to_string(),
            version: crate_info.version.clone(),
            traits,
            types,
            examples,
            cross_references: cross_refs,
            metadata: self.generate_metadata(&crate_info)?,
        })
    }

    /// Generate API reference from source files
    pub fn generate_from_files(&mut self, source_files: Vec<PathBuf>) -> Result<ApiReference> {
        // Parse source files and extract information
        let crate_info = self.analyze_source_files(&source_files)?;

        // Generate API reference using the same logic as generate_from_crate
        let traits = if self.config.include_type_info {
            self.trait_analyzer.analyze_traits(&crate_info)?
        } else {
            Vec::new()
        };

        let types = if self.config.include_type_info {
            self.type_extractor.extract_types(&crate_info)?
        } else {
            Vec::new()
        };

        let examples = if self.config.include_examples {
            let raw_examples = self.extract_examples_from_files(&source_files)?;
            if self.config.validate_examples {
                self.example_validator.validate_examples(&raw_examples)?
            } else {
                raw_examples
            }
        } else {
            Vec::new()
        };

        let cross_refs = if self.config.include_cross_refs {
            self.cross_ref_builder
                .build_cross_references(&traits, &types)?
        } else {
            HashMap::new()
        };

        Ok(ApiReference {
            crate_name: "custom".to_string(),
            version: "unknown".to_string(),
            traits,
            types,
            examples,
            cross_references: cross_refs,
            metadata: self.generate_metadata(&crate_info)?,
        })
    }

    /// Format API reference to the configured output format
    pub fn format_output(&self, api_ref: &ApiReference) -> Result<String> {
        match self.config.output_format {
            OutputFormat::Json => self.formatter.format_json(api_ref),
            OutputFormat::Html => self.formatter.format_html(api_ref),
            OutputFormat::Markdown => self.formatter.format_markdown(api_ref),
            OutputFormat::Interactive => self.formatter.format_interactive(api_ref),
            OutputFormat::OpenApi => self.formatter.format_openapi(api_ref),
        }
    }

    /// Format API reference to a specific output format
    pub fn format_as(&self, api_ref: &ApiReference, format: OutputFormat) -> Result<String> {
        match format {
            OutputFormat::Json => self.formatter.format_json(api_ref),
            OutputFormat::Html => self.formatter.format_html(api_ref),
            OutputFormat::Markdown => self.formatter.format_markdown(api_ref),
            OutputFormat::Interactive => self.formatter.format_interactive(api_ref),
            OutputFormat::OpenApi => self.formatter.format_openapi(api_ref),
        }
    }

    /// Get the document formatter
    pub fn formatter(&self) -> &DocumentFormatter {
        &self.formatter
    }

    /// Get a mutable reference to the document formatter
    pub fn formatter_mut(&mut self) -> &mut DocumentFormatter {
        &mut self.formatter
    }

    /// Analyze crate structure and metadata
    fn analyze_crate(&self, crate_name: &str) -> Result<CrateInfo> {
        // In a real implementation, this would use syn/quote to parse the crate
        Ok(CrateInfo {
            name: crate_name.to_string(),
            version: "0.1.0".to_string(),
            description: format!("API reference for {}", crate_name),
            modules: vec![
                "core".to_string(),
                "traits".to_string(),
                "error".to_string(),
                "utils".to_string(),
            ],
            dependencies: vec![
                "serde".to_string(),
                "ndarray".to_string(),
                "thiserror".to_string(),
            ],
        })
    }

    /// Analyze source files directly
    fn analyze_source_files(&self, _source_files: &[PathBuf]) -> Result<CrateInfo> {
        // In a real implementation, this would parse the source files
        Ok(CrateInfo {
            name: "custom".to_string(),
            version: "unknown".to_string(),
            description: "Custom source file analysis".to_string(),
            modules: Vec::new(),
            dependencies: Vec::new(),
        })
    }

    /// Extract code examples from documentation
    fn extract_examples(&self, crate_info: &CrateInfo) -> Result<Vec<CodeExample>> {
        // Generate examples based on crate content
        let mut examples = Vec::new();

        examples.push(CodeExample {
            title: "Basic Usage".to_string(),
            description: format!("Demonstrates basic usage of {}", crate_info.name),
            code: format!(
                r#"use {};

fn main() {{
    // Your code here
    println!("Hello from {}!");
}}"#,
                crate_info.name, crate_info.name
            ),
            language: "rust".to_string(),
            runnable: true,
            expected_output: Some(format!("Hello from {}!", crate_info.name)),
        });

        examples.push(CodeExample {
            title: "Advanced Example".to_string(),
            description: "Shows advanced features and patterns".to_string(),
            code: format!(
                r#"use {}::{{traits::*, error::*}};

fn advanced_example() -> Result<(), Box<dyn std::error::Error>> {{
    // Advanced usage patterns
    let result = process_data()?;
    println!("Result: {{:?}}", result);
    Ok(())
}}

fn process_data() -> Result<String, {}Error> {{
    // Complex processing logic
    Ok("Processed data".to_string())
}}"#,
                crate_info.name,
                crate_info.name.replace('-', "").to_title_case()
            ),
            language: "rust".to_string(),
            runnable: true,
            expected_output: Some("Result: \"Processed data\"".to_string()),
        });

        Ok(examples)
    }

    /// Extract examples from source files
    fn extract_examples_from_files(&self, _source_files: &[PathBuf]) -> Result<Vec<CodeExample>> {
        // In a real implementation, this would parse doc comments from source files
        Ok(vec![CodeExample {
            title: "Source File Example".to_string(),
            description: "Example extracted from source file documentation".to_string(),
            code: "// Example code from source files".to_string(),
            language: "rust".to_string(),
            runnable: false,
            expected_output: None,
        }])
    }

    /// Generate metadata for the API reference
    fn generate_metadata(&self, crate_info: &CrateInfo) -> Result<ApiMetadata> {
        Ok(ApiMetadata {
            generation_time: chrono::Utc::now().to_string(),
            generator_version: env!("CARGO_PKG_VERSION").to_string(),
            crate_version: crate_info.version.clone(),
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
            config: self.config.clone(),
        })
    }
}

impl Default for ApiReferenceGenerator {
    fn default() -> Self {
        Self::new(GeneratorConfig::default())
    }
}

// ================================================================================================
// DOCUMENT FORMATTER
// ================================================================================================

/// Document formatter for converting API references to various output formats
#[derive(Debug, Clone)]
pub struct DocumentFormatter {
    #[allow(dead_code)]
    config: GeneratorConfig,
    theme: ThemeConfig,
    custom_templates: HashMap<String, String>,
}

impl DocumentFormatter {
    /// Create a new document formatter
    pub fn new(config: GeneratorConfig) -> Self {
        Self {
            theme: ThemeConfig::default(),
            config,
            custom_templates: HashMap::new(),
        }
    }

    /// Create formatter with custom theme
    pub fn with_theme(config: GeneratorConfig, theme: ThemeConfig) -> Self {
        Self {
            config,
            theme,
            custom_templates: HashMap::new(),
        }
    }

    /// Format API reference as JSON
    pub fn format_json(&self, api_ref: &ApiReference) -> Result<String> {
        serde_json::to_string_pretty(api_ref)
            .map_err(|e| SklearsError::InvalidInput(format!("JSON serialization failed: {}", e)))
    }

    /// Format API reference as HTML
    pub fn format_html(&self, api_ref: &ApiReference) -> Result<String> {
        let mut html = String::new();

        // HTML header with theme styling
        html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str(
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str(&format!(
            "    <title>API Reference - {}</title>\n",
            api_ref.crate_name
        ));
        html.push_str("    <style>\n");
        html.push_str(&self.generate_html_css()?);
        html.push_str("    </style>\n");
        html.push_str("</head>\n<body>\n");

        // Navigation
        html.push_str(&self.generate_navigation(api_ref)?);

        // Main content
        html.push_str("    <main class=\"content\">\n");
        html.push_str(&format!(
            "        <h1>API Reference for {}</h1>\n",
            api_ref.crate_name
        ));
        html.push_str(&format!(
            "        <p class=\"version\">Version: {}</p>\n",
            api_ref.version
        ));

        // Table of contents
        html.push_str(&self.generate_table_of_contents(api_ref)?);

        // Traits section
        if !api_ref.traits.is_empty() {
            html.push_str(&self.format_traits_html(&api_ref.traits)?);
        }

        // Types section
        if !api_ref.types.is_empty() {
            html.push_str(&self.format_types_html(&api_ref.types)?);
        }

        // Examples section
        if !api_ref.examples.is_empty() {
            html.push_str(&self.format_examples_html(&api_ref.examples)?);
        }

        // Cross-references section
        if !api_ref.cross_references.is_empty() {
            html.push_str(&self.format_cross_refs_html(&api_ref.cross_references)?);
        }

        html.push_str("    </main>\n");

        // Footer
        html.push_str(&self.generate_footer(api_ref)?);

        html.push_str("</body>\n</html>");
        Ok(html)
    }

    /// Format API reference as Markdown
    pub fn format_markdown(&self, api_ref: &ApiReference) -> Result<String> {
        let mut md = String::new();

        // Header
        md.push_str(&format!("# API Reference - {}\n\n", api_ref.crate_name));
        md.push_str(&format!("**Version:** {}\n\n", api_ref.version));
        md.push_str(&format!(
            "**Generated:** {}\n\n",
            api_ref.metadata.generation_time
        ));

        // Table of contents
        md.push_str("## Table of Contents\n\n");
        if !api_ref.traits.is_empty() {
            md.push_str("- [Traits](#traits)\n");
        }
        if !api_ref.types.is_empty() {
            md.push_str("- [Types](#types)\n");
        }
        if !api_ref.examples.is_empty() {
            md.push_str("- [Examples](#examples)\n");
        }
        if !api_ref.cross_references.is_empty() {
            md.push_str("- [Cross References](#cross-references)\n");
        }
        md.push('\n');

        // Traits section
        if !api_ref.traits.is_empty() {
            md.push_str(&self.format_traits_markdown(&api_ref.traits)?);
        }

        // Types section
        if !api_ref.types.is_empty() {
            md.push_str(&self.format_types_markdown(&api_ref.types)?);
        }

        // Examples section
        if !api_ref.examples.is_empty() {
            md.push_str(&self.format_examples_markdown(&api_ref.examples)?);
        }

        // Cross-references section
        if !api_ref.cross_references.is_empty() {
            md.push_str(&self.format_cross_refs_markdown(&api_ref.cross_references)?);
        }

        Ok(md)
    }

    /// Format API reference as interactive HTML
    pub fn format_interactive(&self, api_ref: &ApiReference) -> Result<String> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str(
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str(&format!(
            "    <title>Interactive API Reference - {}</title>\n",
            api_ref.crate_name
        ));

        // External libraries for interactive features
        html.push_str("    <script src=\"https://cdn.jsdelivr.net/npm/monaco-editor@latest/min/vs/loader.js\"></script>\n");
        html.push_str("    <script src=\"https://unpkg.com/@webassembly/wasi-sdk@0.11.0/bin/wasm-ld\"></script>\n");

        // CSS for interactive layout
        html.push_str("    <style>\n");
        html.push_str(&self.generate_interactive_css()?);
        html.push_str("    </style>\n");
        html.push_str("</head>\n<body>\n");

        // Interactive header
        html.push_str("    <header class=\"interactive-header\">\n");
        html.push_str(&format!(
            "        <h1>Interactive API Reference - {}</h1>\n",
            api_ref.crate_name
        ));
        html.push_str("        <nav class=\"interactive-nav\">\n");
        html.push_str("            <button id=\"run-code\" class=\"nav-btn\">Run Code</button>\n");
        html.push_str("            <button id=\"reset-code\" class=\"nav-btn\">Reset</button>\n");
        html.push_str("            <button id=\"share-code\" class=\"nav-btn\">Share</button>\n");
        html.push_str("        </nav>\n");
        html.push_str("    </header>\n");

        // Main interactive area
        html.push_str("    <main class=\"interactive-main\">\n");
        html.push_str("        <div class=\"editor-panel\">\n");
        html.push_str("            <h2>Code Editor</h2>\n");
        html.push_str("            <div id=\"code-editor\"></div>\n");
        html.push_str("        </div>\n");
        html.push_str("        <div class=\"output-panel\">\n");
        html.push_str("            <h2>Output</h2>\n");
        html.push_str("            <div id=\"output-display\"></div>\n");
        html.push_str("        </div>\n");
        html.push_str("        <div class=\"api-panel\">\n");
        html.push_str("            <h2>API Explorer</h2>\n");
        html.push_str(&self.format_interactive_api_explorer(api_ref)?);
        html.push_str("        </div>\n");
        html.push_str("    </main>\n");

        // Examples gallery
        html.push_str("    <section class=\"examples-gallery\">\n");
        html.push_str("        <h2>Examples</h2>\n");
        html.push_str(&self.format_interactive_examples(&api_ref.examples)?);
        html.push_str("    </section>\n");

        // JavaScript for interactive functionality
        html.push_str("    <script>\n");
        html.push_str(&self.generate_interactive_js(api_ref)?);
        html.push_str("    </script>\n");

        html.push_str("</body>\n</html>");
        Ok(html)
    }

    /// Format API reference as OpenAPI specification
    pub fn format_openapi(&self, api_ref: &ApiReference) -> Result<String> {
        let mut openapi = String::new();

        openapi.push_str("openapi: 3.0.3\n");
        openapi.push_str("info:\n");
        openapi.push_str(&format!("  title: {} API\n", api_ref.crate_name));
        openapi.push_str(&format!("  version: {}\n", api_ref.version));
        openapi.push_str(&format!(
            "  description: API specification for {}\n",
            api_ref.crate_name
        ));
        openapi.push_str("servers:\n");
        openapi.push_str("  - url: http://localhost:8080\n");
        openapi.push_str("    description: Development server\n");

        openapi.push_str("paths:\n");

        // Generate paths from traits (treating them as endpoints)
        for trait_info in &api_ref.traits {
            for method in &trait_info.methods {
                let path = format!("/{}/{}", trait_info.name.to_lowercase(), method.name);
                openapi.push_str(&format!("  {}:\n", path));
                openapi.push_str("    post:\n");
                openapi.push_str(&format!("      summary: {}\n", method.description));
                openapi.push_str(&format!(
                    "      operationId: {}_{}\n",
                    trait_info.name, method.name
                ));
                openapi.push_str("      responses:\n");
                openapi.push_str("        '200':\n");
                openapi.push_str("          description: Successful operation\n");
                openapi.push_str("          content:\n");
                openapi.push_str("            application/json:\n");
                openapi.push_str("              schema:\n");
                openapi.push_str("                type: object\n");
            }
        }

        openapi.push_str("components:\n");
        openapi.push_str("  schemas:\n");

        // Generate schemas from types
        for type_info in &api_ref.types {
            openapi.push_str(&format!("    {}:\n", type_info.name));
            openapi.push_str("      type: object\n");
            if !type_info.fields.is_empty() {
                openapi.push_str("      properties:\n");
                for field in &type_info.fields {
                    openapi.push_str(&format!("        {}:\n", field.name));
                    openapi.push_str(&format!(
                        "          type: {}\n",
                        self.rust_type_to_openapi(&field.field_type)
                    ));
                    if !field.description.is_empty() {
                        openapi
                            .push_str(&format!("          description: {}\n", field.description));
                    }
                }
            }
        }

        Ok(openapi)
    }

    /// Set custom theme
    pub fn set_theme(&mut self, theme: ThemeConfig) {
        self.theme = theme;
    }

    /// Add custom template
    pub fn add_template(&mut self, name: String, template: String) {
        self.custom_templates.insert(name, template);
    }

    /// Generate HTML CSS styles
    fn generate_html_css(&self) -> Result<String> {
        let mut css = String::new();

        // Theme variables
        css.push_str(&self.theme.to_css_variables());
        css.push_str("\n\n");

        // Base styles
        css.push_str(
            r#"
        body {
            font-family: var(--font-family);
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }

        .content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color);
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        .version {
            color: var(--secondary-color);
            font-weight: bold;
        }

        .toc {
            background: var(--code-background);
            padding: 1rem;
            border-radius: 8px;
            margin: 2rem 0;
        }

        .toc ul {
            list-style-type: none;
            padding-left: 1rem;
        }

        .toc a {
            color: var(--primary-color);
            text-decoration: none;
        }

        .toc a:hover {
            text-decoration: underline;
        }

        .trait-section, .type-section, .example-section {
            margin: 3rem 0;
            padding: 2rem;
            border: 1px solid #ddd;
            border-radius: 8px;
        }

        .method-list {
            margin-left: 1rem;
        }

        .method-item {
            margin: 1rem 0;
            padding: 1rem;
            background: var(--code-background);
            border-radius: 4px;
        }

        .method-signature {
            font-family: var(--code-font-family);
            background: var(--code-background);
            color: var(--code-text-color);
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.9rem;
        }

        .code-example {
            background: var(--code-background);
            color: var(--code-text-color);
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
            margin: 1rem 0;
        }

        .code-example pre {
            margin: 0;
            font-family: var(--code-font-family);
        }

        .footer {
            margin-top: 4rem;
            padding: 2rem;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }

        .nav {
            background: var(--primary-color);
            padding: 1rem;
            margin-bottom: 2rem;
        }

        .nav a {
            color: white;
            text-decoration: none;
            margin-right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background 0.2s;
        }

        .nav a:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        "#,
        );

        Ok(css)
    }

    /// Generate navigation HTML
    fn generate_navigation(&self, api_ref: &ApiReference) -> Result<String> {
        let mut nav = String::new();
        nav.push_str("    <nav class=\"nav\">\n");
        if !api_ref.traits.is_empty() {
            nav.push_str("        <a href=\"#traits\">Traits</a>\n");
        }
        if !api_ref.types.is_empty() {
            nav.push_str("        <a href=\"#types\">Types</a>\n");
        }
        if !api_ref.examples.is_empty() {
            nav.push_str("        <a href=\"#examples\">Examples</a>\n");
        }
        nav.push_str("    </nav>\n");
        Ok(nav)
    }

    /// Generate table of contents
    fn generate_table_of_contents(&self, api_ref: &ApiReference) -> Result<String> {
        let mut toc = String::new();
        toc.push_str("        <div class=\"toc\">\n");
        toc.push_str("            <h2>Table of Contents</h2>\n");
        toc.push_str("            <ul>\n");

        if !api_ref.traits.is_empty() {
            toc.push_str("                <li><a href=\"#traits\">Traits</a>\n");
            toc.push_str("                    <ul>\n");
            for trait_info in &api_ref.traits {
                toc.push_str(&format!(
                    "                        <li><a href=\"#{}\">{}::{}</a></li>\n",
                    trait_info.name.to_lowercase(),
                    trait_info
                        .path
                        .split("::")
                        .last()
                        .unwrap_or(&trait_info.path),
                    trait_info.name
                ));
            }
            toc.push_str("                    </ul>\n");
            toc.push_str("                </li>\n");
        }

        if !api_ref.types.is_empty() {
            toc.push_str("                <li><a href=\"#types\">Types</a>\n");
            toc.push_str("                    <ul>\n");
            for type_info in &api_ref.types {
                toc.push_str(&format!(
                    "                        <li><a href=\"#{}\">{}::{}</a></li>\n",
                    type_info.name.to_lowercase(),
                    type_info.path.split("::").last().unwrap_or(&type_info.path),
                    type_info.name
                ));
            }
            toc.push_str("                    </ul>\n");
            toc.push_str("                </li>\n");
        }

        if !api_ref.examples.is_empty() {
            toc.push_str("                <li><a href=\"#examples\">Examples</a></li>\n");
        }

        toc.push_str("            </ul>\n");
        toc.push_str("        </div>\n");
        Ok(toc)
    }

    /// Format traits as HTML
    fn format_traits_html(&self, traits: &[TraitInfo]) -> Result<String> {
        let mut html = String::new();
        html.push_str("        <section id=\"traits\">\n");
        html.push_str("            <h2>Traits</h2>\n");

        for trait_info in traits {
            html.push_str(&format!(
                "            <div class=\"trait-section\" id=\"{}\">\n",
                trait_info.name.to_lowercase()
            ));
            html.push_str(&format!("                <h3>{}</h3>\n", trait_info.name));
            html.push_str(&format!(
                "                <p>{}</p>\n",
                trait_info.description
            ));
            html.push_str(&format!(
                "                <p><strong>Path:</strong> <code>{}</code></p>\n",
                trait_info.path
            ));

            if !trait_info.generics.is_empty() {
                html.push_str("                <p><strong>Generic Parameters:</strong> ");
                html.push_str(&trait_info.generics.join(", "));
                html.push_str("</p>\n");
            }

            if !trait_info.supertraits.is_empty() {
                html.push_str("                <p><strong>Supertraits:</strong> ");
                html.push_str(&trait_info.supertraits.join(", "));
                html.push_str("</p>\n");
            }

            if !trait_info.methods.is_empty() {
                html.push_str("                <h4>Methods</h4>\n");
                html.push_str("                <div class=\"method-list\">\n");
                for method in &trait_info.methods {
                    html.push_str("                    <div class=\"method-item\">\n");
                    html.push_str(&format!(
                        "                        <h5>{}</h5>\n",
                        method.name
                    ));
                    html.push_str(&format!(
                        "                        <div class=\"method-signature\"><code>{}</code></div>\n",
                        method.signature
                    ));
                    html.push_str(&format!(
                        "                        <p>{}</p>\n",
                        method.description
                    ));
                    if method.required {
                        html.push_str("                        <p><em>Required method</em></p>\n");
                    }
                    html.push_str("                    </div>\n");
                }
                html.push_str("                </div>\n");
            }

            html.push_str("            </div>\n");
        }

        html.push_str("        </section>\n");
        Ok(html)
    }

    /// Format types as HTML
    fn format_types_html(&self, types: &[TypeInfo]) -> Result<String> {
        let mut html = String::new();
        html.push_str("        <section id=\"types\">\n");
        html.push_str("            <h2>Types</h2>\n");

        for type_info in types {
            html.push_str(&format!(
                "            <div class=\"type-section\" id=\"{}\">\n",
                type_info.name.to_lowercase()
            ));
            html.push_str(&format!("                <h3>{}</h3>\n", type_info.name));
            html.push_str(&format!(
                "                <p>{}</p>\n",
                type_info.description
            ));
            html.push_str(&format!(
                "                <p><strong>Path:</strong> <code>{}</code></p>\n",
                type_info.path
            ));
            html.push_str(&format!(
                "                <p><strong>Kind:</strong> {:?}</p>\n",
                type_info.kind
            ));

            if !type_info.fields.is_empty() {
                html.push_str("                <h4>Fields</h4>\n");
                html.push_str("                <ul>\n");
                for field in &type_info.fields {
                    html.push_str(&format!(
                        "                    <li><strong>{}:</strong> <code>{}</code> - {}</li>\n",
                        field.name, field.field_type, field.description
                    ));
                }
                html.push_str("                </ul>\n");
            }

            if !type_info.trait_impls.is_empty() {
                html.push_str("                <p><strong>Implemented Traits:</strong> ");
                html.push_str(&type_info.trait_impls.join(", "));
                html.push_str("</p>\n");
            }

            html.push_str("            </div>\n");
        }

        html.push_str("        </section>\n");
        Ok(html)
    }

    /// Format examples as HTML
    fn format_examples_html(&self, examples: &[CodeExample]) -> Result<String> {
        let mut html = String::new();
        html.push_str("        <section id=\"examples\">\n");
        html.push_str("            <h2>Examples</h2>\n");

        for example in examples {
            html.push_str("            <div class=\"example-section\">\n");
            html.push_str(&format!("                <h3>{}</h3>\n", example.title));
            html.push_str(&format!("                <p>{}</p>\n", example.description));
            html.push_str(&format!(
                "                <div class=\"code-example\"><pre><code class=\"{}\">{}</code></pre></div>\n",
                example.language, example.code
            ));
            if let Some(output) = &example.expected_output {
                html.push_str("                <p><strong>Expected Output:</strong></p>\n");
                html.push_str(&format!(
                    "                <div class=\"code-example\"><pre>{}</pre></div>\n",
                    output
                ));
            }
            html.push_str("            </div>\n");
        }

        html.push_str("        </section>\n");
        Ok(html)
    }

    /// Format cross-references as HTML
    fn format_cross_refs_html(&self, cross_refs: &HashMap<String, Vec<String>>) -> Result<String> {
        let mut html = String::new();
        html.push_str("        <section id=\"cross-references\">\n");
        html.push_str("            <h2>Cross References</h2>\n");

        for (item, refs) in cross_refs {
            if !refs.is_empty() {
                html.push_str(&format!("            <h3>{}</h3>\n", item));
                html.push_str("            <ul>\n");
                for ref_item in refs {
                    html.push_str(&format!("                <li>{}</li>\n", ref_item));
                }
                html.push_str("            </ul>\n");
            }
        }

        html.push_str("        </section>\n");
        Ok(html)
    }

    /// Format traits as Markdown
    fn format_traits_markdown(&self, traits: &[TraitInfo]) -> Result<String> {
        let mut md = String::new();
        md.push_str("## Traits\n\n");

        for trait_info in traits {
            md.push_str(&format!("### {}\n\n", trait_info.name));
            md.push_str(&format!("{}\n\n", trait_info.description));
            md.push_str(&format!("**Path:** `{}`\n\n", trait_info.path));

            if !trait_info.generics.is_empty() {
                md.push_str(&format!(
                    "**Generic Parameters:** {}\n\n",
                    trait_info.generics.join(", ")
                ));
            }

            if !trait_info.supertraits.is_empty() {
                md.push_str(&format!(
                    "**Supertraits:** {}\n\n",
                    trait_info.supertraits.join(", ")
                ));
            }

            if !trait_info.methods.is_empty() {
                md.push_str("#### Methods\n\n");
                for method in &trait_info.methods {
                    md.push_str(&format!("##### {}\n\n", method.name));
                    md.push_str(&format!("```rust\n{}\n```\n\n", method.signature));
                    md.push_str(&format!("{}\n\n", method.description));
                    if method.required {
                        md.push_str("*Required method*\n\n");
                    }
                }
            }
        }

        Ok(md)
    }

    /// Format types as Markdown
    fn format_types_markdown(&self, types: &[TypeInfo]) -> Result<String> {
        let mut md = String::new();
        md.push_str("## Types\n\n");

        for type_info in types {
            md.push_str(&format!("### {}\n\n", type_info.name));
            md.push_str(&format!("{}\n\n", type_info.description));
            md.push_str(&format!("**Path:** `{}`\n\n", type_info.path));
            md.push_str(&format!("**Kind:** {:?}\n\n", type_info.kind));

            if !type_info.fields.is_empty() {
                md.push_str("#### Fields\n\n");
                for field in &type_info.fields {
                    md.push_str(&format!(
                        "- **{}:** `{}` - {}\n",
                        field.name, field.field_type, field.description
                    ));
                }
                md.push('\n');
            }

            if !type_info.trait_impls.is_empty() {
                md.push_str(&format!(
                    "**Implemented Traits:** {}\n\n",
                    type_info.trait_impls.join(", ")
                ));
            }
        }

        Ok(md)
    }

    /// Format examples as Markdown
    fn format_examples_markdown(&self, examples: &[CodeExample]) -> Result<String> {
        let mut md = String::new();
        md.push_str("## Examples\n\n");

        for example in examples {
            md.push_str(&format!("### {}\n\n", example.title));
            md.push_str(&format!("{}\n\n", example.description));
            md.push_str(&format!(
                "```{}\n{}\n```\n\n",
                example.language, example.code
            ));
            if let Some(output) = &example.expected_output {
                md.push_str("**Expected Output:**\n\n");
                md.push_str(&format!("```\n{}\n```\n\n", output));
            }
        }

        Ok(md)
    }

    /// Format cross-references as Markdown
    fn format_cross_refs_markdown(
        &self,
        cross_refs: &HashMap<String, Vec<String>>,
    ) -> Result<String> {
        let mut md = String::new();
        md.push_str("## Cross References\n\n");

        for (item, refs) in cross_refs {
            if !refs.is_empty() {
                md.push_str(&format!("### {}\n\n", item));
                for ref_item in refs {
                    md.push_str(&format!("- {}\n", ref_item));
                }
                md.push('\n');
            }
        }

        Ok(md)
    }

    /// Generate footer HTML
    fn generate_footer(&self, api_ref: &ApiReference) -> Result<String> {
        Ok(format!(
            r#"    <footer class="footer">
        <p>Generated on {} by sklears-core API generator v{}</p>
        <p>Rust version: {}</p>
    </footer>"#,
            api_ref.metadata.generation_time,
            api_ref.metadata.generator_version,
            api_ref.metadata.rust_version
        ))
    }

    /// Generate interactive CSS
    fn generate_interactive_css(&self) -> Result<String> {
        Ok(r#"
        .interactive-header {
            background: var(--primary-color);
            color: white;
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .interactive-nav {
            display: flex;
            gap: 1rem;
        }

        .nav-btn {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .nav-btn:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .interactive-main {
            display: grid;
            grid-template-columns: 1fr 1fr 300px;
            gap: 1rem;
            padding: 1rem;
            height: 70vh;
        }

        .editor-panel, .output-panel, .api-panel {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            overflow: hidden;
        }

        #code-editor {
            height: 90%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }

        #output-display {
            height: 90%;
            background: var(--code-background);
            color: var(--code-text-color);
            padding: 1rem;
            border-radius: 4px;
            overflow-y: auto;
            font-family: var(--code-font-family);
        }

        .api-panel {
            overflow-y: auto;
        }

        .examples-gallery {
            padding: 1rem;
            border-top: 1px solid #ddd;
        }

        .example-card {
            background: var(--code-background);
            padding: 1rem;
            margin: 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .example-card:hover {
            transform: translateY(-2px);
        }
        "#
        .to_string())
    }

    /// Format interactive API explorer
    fn format_interactive_api_explorer(&self, api_ref: &ApiReference) -> Result<String> {
        let mut html = String::new();

        html.push_str("            <div class=\"api-search\">\n");
        html.push_str("                <input type=\"text\" placeholder=\"Search API...\" id=\"api-search-input\">\n");
        html.push_str("            </div>\n");

        if !api_ref.traits.is_empty() {
            html.push_str("            <div class=\"api-section\">\n");
            html.push_str("                <h3>Traits</h3>\n");
            for trait_info in &api_ref.traits {
                html.push_str(&format!(
                    "                <div class=\"api-item\" data-type=\"trait\" onclick=\"insertCode('{}')\">{}</div>\n",
                    trait_info.name, trait_info.name
                ));
            }
            html.push_str("            </div>\n");
        }

        if !api_ref.types.is_empty() {
            html.push_str("            <div class=\"api-section\">\n");
            html.push_str("                <h3>Types</h3>\n");
            for type_info in &api_ref.types {
                html.push_str(&format!(
                    "                <div class=\"api-item\" data-type=\"type\" onclick=\"insertCode('{}')\">{}</div>\n",
                    type_info.name, type_info.name
                ));
            }
            html.push_str("            </div>\n");
        }

        Ok(html)
    }

    /// Format interactive examples
    fn format_interactive_examples(&self, examples: &[CodeExample]) -> Result<String> {
        let mut html = String::new();

        html.push_str("        <div class=\"examples-grid\">\n");
        for (i, example) in examples.iter().enumerate() {
            html.push_str(&format!(
                r#"            <div class="example-card" onclick="loadExample({})" title="{}">
                <h4>{}</h4>
                <p>{}</p>
            </div>"#,
                i, example.description, example.title, example.description
            ));
        }
        html.push_str("        </div>\n");

        Ok(html)
    }

    /// Generate interactive JavaScript
    fn generate_interactive_js(&self, api_ref: &ApiReference) -> Result<String> {
        let mut js = String::new();

        // Examples data
        js.push_str("        const examples = [\n");
        for example in &api_ref.examples {
            js.push_str(&format!(
                "            {{ title: '{}', code: `{}` }},\n",
                example.title.replace('\'', "\\'"),
                example.code.replace('`', "\\`")
            ));
        }
        js.push_str("        ];\n\n");

        // Interactive functionality
        js.push_str(r#"
        let editor;

        // Initialize Monaco Editor
        require.config({ paths: { 'vs': 'https://cdn.jsdelivr.net/npm/monaco-editor@latest/min/vs' }});
        require(['vs/editor/editor.main'], function () {
            editor = monaco.editor.create(document.getElementById('code-editor'), {
                value: examples[0] ? examples[0].code : 'fn main() {\n    println!("Hello, sklears!");\n}',
                language: 'rust',
                theme: 'vs-dark',
                automaticLayout: true
            });
        });

        function runCode() {
            const code = editor.getValue();
            const output = document.getElementById('output-display');
            output.innerHTML = 'Running code...\n\n' + code;
        }

        function resetCode() {
            if (examples[0]) {
                editor.setValue(examples[0].code);
            }
            document.getElementById('output-display').innerHTML = 'Output will appear here...';
        }

        function shareCode() {
            const code = editor.getValue();
            const encoded = btoa(code);
            const url = window.location.origin + window.location.pathname + '?code=' + encoded;
            navigator.clipboard.writeText(url);
            alert('Shareable link copied to clipboard!');
        }

        function loadExample(index) {
            if (examples[index]) {
                editor.setValue(examples[index].code);
                document.getElementById('output-display').innerHTML = 'Example loaded: ' + examples[index].title;
            }
        }

        function insertCode(apiItem) {
            const currentCode = editor.getValue();
            const newCode = currentCode + '\n// Using ' + apiItem + '\n';
            editor.setValue(newCode);
        }

        // Event listeners
        document.getElementById('run-code').addEventListener('click', runCode);
        document.getElementById('reset-code').addEventListener('click', resetCode);
        document.getElementById('share-code').addEventListener('click', shareCode);

        // API search functionality
        document.getElementById('api-search-input').addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            const items = document.querySelectorAll('.api-item');
            items.forEach(item => {
                if (item.textContent.toLowerCase().includes(query)) {
                    item.style.display = 'block';
                } else {
                    item.style.display = 'none';
                }
            });
        });
        "#);

        Ok(js)
    }

    /// Convert Rust type to OpenAPI type
    fn rust_type_to_openapi(&self, rust_type: &str) -> &'static str {
        match rust_type {
            "String" | "&str" => "string",
            "i32" | "i64" | "u32" | "u64" | "usize" | "isize" => "integer",
            "f32" | "f64" => "number",
            "bool" => "boolean",
            _ => "object",
        }
    }
}

impl Default for DocumentFormatter {
    fn default() -> Self {
        Self::new(GeneratorConfig::default())
    }
}

// ================================================================================================
// STRING UTILITIES
// ================================================================================================

/// String extension trait for additional formatting utilities
trait StringExt {
    fn to_title_case(&self) -> String;
}

impl StringExt for str {
    fn to_title_case(&self) -> String {
        self.chars()
            .enumerate()
            .map(|(i, c)| {
                if i == 0 || self.chars().nth(i - 1).unwrap_or(' ').is_whitespace() {
                    c.to_uppercase().collect::<String>()
                } else {
                    c.to_lowercase().collect::<String>()
                }
            })
            .collect()
    }
}

// Mock chrono for compilation
mod chrono {
    pub struct DateTime<T>(std::marker::PhantomData<T>);
    pub struct Utc;

    impl Utc {
        pub fn now() -> DateTime<Utc> {
            DateTime(std::marker::PhantomData)
        }
    }

    impl<T> std::fmt::Display for DateTime<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "2024-01-01T00:00:00Z")
        }
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_data_structures::{FieldInfo, MethodInfo, TypeKind, Visibility};

    #[test]
    fn test_document_formatter_json() {
        let formatter = DocumentFormatter::new(GeneratorConfig::new());
        let api_ref = create_test_api_reference();

        let json = formatter.format_json(&api_ref).unwrap();
        assert!(json.contains("test-crate"));
        assert!(json.contains("traits"));
    }

    #[test]
    fn test_document_formatter_html() {
        let formatter = DocumentFormatter::new(GeneratorConfig::new());
        let api_ref = create_test_api_reference();

        let html = formatter.format_html(&api_ref).unwrap();
        assert!(html.contains("<html"));
        assert!(html.contains("API Reference for test-crate"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_document_formatter_markdown() {
        let formatter = DocumentFormatter::new(GeneratorConfig::new());
        let api_ref = create_test_api_reference();

        let md = formatter.format_markdown(&api_ref).unwrap();
        assert!(md.contains("# API Reference - test-crate"));
        assert!(md.contains("## Table of Contents"));
    }

    #[test]
    fn test_document_formatter_interactive() {
        let formatter = DocumentFormatter::new(GeneratorConfig::new());
        let api_ref = create_test_api_reference();

        let interactive = formatter.format_interactive(&api_ref).unwrap();
        assert!(interactive.contains("Interactive API Reference"));
        assert!(interactive.contains("code-editor"));
        assert!(interactive.contains("monaco-editor"));
    }

    #[test]
    fn test_document_formatter_openapi() {
        let formatter = DocumentFormatter::new(GeneratorConfig::new());
        let api_ref = create_test_api_reference();

        let openapi = formatter.format_openapi(&api_ref).unwrap();
        assert!(openapi.contains("openapi: 3.0.3"));
        assert!(openapi.contains("test-crate API"));
    }

    #[test]
    fn test_api_reference_generator() {
        let config = GeneratorConfig::new();
        let mut generator = ApiReferenceGenerator::new(config);

        let api_ref = generator.generate_from_crate("test-crate").unwrap();
        assert_eq!(api_ref.crate_name, "test-crate");
        assert!(!api_ref.traits.is_empty());
        assert!(!api_ref.examples.is_empty());
    }

    #[test]
    fn test_format_output() {
        let config = GeneratorConfig::new().with_output_format(OutputFormat::Html);
        let generator = ApiReferenceGenerator::new(config);
        let api_ref = create_test_api_reference();

        let output = generator.format_output(&api_ref).unwrap();
        assert!(output.contains("<html"));
    }

    #[test]
    fn test_format_as() {
        let generator = ApiReferenceGenerator::new(GeneratorConfig::new());
        let api_ref = create_test_api_reference();

        let json = generator.format_as(&api_ref, OutputFormat::Json).unwrap();
        assert!(json.contains("test-crate"));

        let html = generator.format_as(&api_ref, OutputFormat::Html).unwrap();
        assert!(html.contains("<html"));

        let md = generator
            .format_as(&api_ref, OutputFormat::Markdown)
            .unwrap();
        assert!(md.contains("# API Reference"));
    }

    #[test]
    fn test_string_ext() {
        assert_eq!("hello world".to_title_case(), "Hello World");
        assert_eq!("HELLO WORLD".to_title_case(), "Hello World");
        assert_eq!("helloWorld".to_title_case(), "Helloworld");
    }

    fn create_test_api_reference() -> ApiReference {
        ApiReference {
            crate_name: "test-crate".to_string(),
            version: "1.0.0".to_string(),
            traits: vec![TraitInfo {
                name: "TestTrait".to_string(),
                description: "A test trait".to_string(),
                path: "test::TestTrait".to_string(),
                generics: Vec::new(),
                associated_types: Vec::new(),
                methods: vec![MethodInfo {
                    name: "test_method".to_string(),
                    signature: "fn test_method(&self)".to_string(),
                    description: "A test method".to_string(),
                    parameters: Vec::new(),
                    return_type: "()".to_string(),
                    required: true,
                }],
                supertraits: Vec::new(),
                implementations: Vec::new(),
            }],
            types: vec![TypeInfo {
                name: "TestType".to_string(),
                description: "A test type".to_string(),
                path: "test::TestType".to_string(),
                kind: TypeKind::Struct,
                generics: Vec::new(),
                fields: vec![FieldInfo {
                    name: "field".to_string(),
                    field_type: "String".to_string(),
                    description: "A test field".to_string(),
                    visibility: Visibility::Public,
                }],
                trait_impls: Vec::new(),
            }],
            examples: vec![CodeExample {
                title: "Test Example".to_string(),
                description: "A test example".to_string(),
                code: "fn main() { println!(\"Hello, test!\"); }".to_string(),
                language: "rust".to_string(),
                runnable: true,
                expected_output: Some("Hello, test!".to_string()),
            }],
            cross_references: HashMap::new(),
            metadata: ApiMetadata::default(),
        }
    }
}
