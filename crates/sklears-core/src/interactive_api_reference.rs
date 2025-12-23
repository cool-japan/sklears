//! Interactive API Reference Generator
//!
//! This module provides an interactive, searchable API reference with live examples,
//! type information, and cross-references.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Interactive API reference generator
///
/// Generates comprehensive, interactive API documentation with search,
/// filtering, and live code examples.
#[derive(Debug, Clone)]
pub struct InteractiveAPIReference {
    /// All documented traits
    pub traits: HashMap<String, DocumentedTrait>,
    /// All documented types
    pub types: HashMap<String, DocumentedType>,
    /// All documented functions
    pub functions: HashMap<String, DocumentedFunction>,
    /// Search index for fast lookups
    pub search_index: SearchIndex,
    /// Configuration
    pub config: APIReferenceConfig,
}

/// Configuration for API reference generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIReferenceConfig {
    /// Enable live code examples
    pub enable_live_examples: bool,
    /// Enable type highlighting
    pub enable_type_highlighting: bool,
    /// Enable cross-references
    pub enable_cross_references: bool,
    /// Theme for syntax highlighting
    pub syntax_theme: String,
}

impl Default for APIReferenceConfig {
    fn default() -> Self {
        Self {
            enable_live_examples: true,
            enable_type_highlighting: true,
            enable_cross_references: true,
            syntax_theme: "monokai".to_string(),
        }
    }
}

/// Documented trait with enhanced information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentedTrait {
    /// Trait name
    pub name: String,
    /// Full qualified name
    pub full_path: String,
    /// Description
    pub description: String,
    /// Associated types
    pub associated_types: Vec<AssociatedType>,
    /// Required methods
    pub required_methods: Vec<DocumentedMethod>,
    /// Provided methods
    pub provided_methods: Vec<DocumentedMethod>,
    /// Implementors
    pub implementors: Vec<String>,
    /// Examples
    pub examples: Vec<InteractiveExample>,
    /// Related traits
    pub related_traits: Vec<String>,
    /// Since version
    pub since: String,
}

/// Associated type in a trait
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociatedType {
    /// Type name
    pub name: String,
    /// Type bounds
    pub bounds: Vec<String>,
    /// Description
    pub description: String,
    /// Default type (if any)
    pub default: Option<String>,
}

/// Documented method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentedMethod {
    /// Method name
    pub name: String,
    /// Method signature
    pub signature: String,
    /// Description
    pub description: String,
    /// Parameters
    pub parameters: Vec<Parameter>,
    /// Return type
    pub return_type: String,
    /// Examples
    pub examples: Vec<String>,
    /// Safety notes (for unsafe methods)
    pub safety: Option<String>,
}

/// Method/function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Description
    pub description: String,
    /// Default value (if optional)
    pub default: Option<String>,
}

/// Documented type (struct, enum, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentedType {
    /// Type name
    pub name: String,
    /// Full path
    pub full_path: String,
    /// Type kind
    pub kind: TypeKind,
    /// Description
    pub description: String,
    /// Fields (for structs)
    pub fields: Vec<Field>,
    /// Variants (for enums)
    pub variants: Vec<Variant>,
    /// Implemented traits
    pub traits: Vec<String>,
    /// Methods
    pub methods: Vec<DocumentedMethod>,
    /// Examples
    pub examples: Vec<InteractiveExample>,
}

/// Kind of type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeKind {
    Struct,
    Enum,
    Union,
    TypeAlias,
    Trait,
}

/// Struct field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: String,
    /// Description
    pub description: String,
    /// Visibility
    pub visibility: Visibility,
}

/// Enum variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Variant name
    pub name: String,
    /// Variant fields (if any)
    pub fields: Vec<Field>,
    /// Description
    pub description: String,
}

/// Visibility level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Crate,
    Module,
    Private,
}

/// Documented function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentedFunction {
    /// Function name
    pub name: String,
    /// Full path
    pub full_path: String,
    /// Function signature
    pub signature: String,
    /// Description
    pub description: String,
    /// Parameters
    pub parameters: Vec<Parameter>,
    /// Return type
    pub return_type: String,
    /// Examples
    pub examples: Vec<InteractiveExample>,
    /// Async function
    pub is_async: bool,
}

/// Interactive code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveExample {
    /// Example title
    pub title: String,
    /// Source code
    pub code: String,
    /// Expected output
    pub expected_output: Option<String>,
    /// Whether example is runnable
    pub runnable: bool,
    /// Language (usually "rust")
    pub language: String,
}

/// Search index for API reference
#[derive(Debug, Clone)]
pub struct SearchIndex {
    /// Trait index
    pub trait_index: HashMap<String, Vec<String>>,
    /// Type index
    pub type_index: HashMap<String, Vec<String>>,
    /// Function index
    pub function_index: HashMap<String, Vec<String>>,
}

impl InteractiveAPIReference {
    /// Create a new API reference
    pub fn new() -> Self {
        Self {
            traits: HashMap::new(),
            types: HashMap::new(),
            functions: HashMap::new(),
            search_index: SearchIndex {
                trait_index: HashMap::new(),
                type_index: HashMap::new(),
                function_index: HashMap::new(),
            },
            config: APIReferenceConfig::default(),
        }
    }

    /// Add a documented trait
    pub fn add_trait(&mut self, trait_doc: DocumentedTrait) {
        // Index for search
        self.index_trait(&trait_doc);
        self.traits.insert(trait_doc.name.clone(), trait_doc);
    }

    /// Add a documented type
    pub fn add_type(&mut self, type_doc: DocumentedType) {
        self.index_type(&type_doc);
        self.types.insert(type_doc.name.clone(), type_doc);
    }

    /// Add a documented function
    pub fn add_function(&mut self, func_doc: DocumentedFunction) {
        self.index_function(&func_doc);
        self.functions.insert(func_doc.name.clone(), func_doc);
    }

    /// Search the API reference
    pub fn search(&self, query: &str) -> SearchResults {
        let query_lower = query.to_lowercase();
        let mut results = SearchResults {
            traits: Vec::new(),
            types: Vec::new(),
            functions: Vec::new(),
        };

        // Search traits
        for (name, trait_doc) in &self.traits {
            if name.to_lowercase().contains(&query_lower)
                || trait_doc.description.to_lowercase().contains(&query_lower)
            {
                results.traits.push(trait_doc.clone());
            }
        }

        // Search types
        for (name, type_doc) in &self.types {
            if name.to_lowercase().contains(&query_lower)
                || type_doc.description.to_lowercase().contains(&query_lower)
            {
                results.types.push(type_doc.clone());
            }
        }

        // Search functions
        for (name, func_doc) in &self.functions {
            if name.to_lowercase().contains(&query_lower)
                || func_doc.description.to_lowercase().contains(&query_lower)
            {
                results.functions.push(func_doc.clone());
            }
        }

        results
    }

    /// Generate HTML documentation
    pub fn generate_html(&self) -> String {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>sklears API Reference</title>\n");
        html.push_str("<style>\n");
        html.push_str(self.generate_css());
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        // Header
        html.push_str("<h1>sklears API Reference</h1>\n");

        // Search box
        html.push_str("<div class='search-box'>\n");
        html.push_str("<input type='text' id='search' placeholder='Search API...'>\n");
        html.push_str("</div>\n");

        // Navigation
        html.push_str("<nav>\n");
        html.push_str("<h2>Categories</h2>\n");
        html.push_str("<ul>\n");
        html.push_str("<li><a href='#traits'>Traits</a></li>\n");
        html.push_str("<li><a href='#types'>Types</a></li>\n");
        html.push_str("<li><a href='#functions'>Functions</a></li>\n");
        html.push_str("</ul>\n");
        html.push_str("</nav>\n");

        // Content
        html.push_str("<main>\n");

        // Traits section
        html.push_str("<section id='traits'>\n");
        html.push_str("<h2>Traits</h2>\n");
        for trait_doc in self.traits.values() {
            html.push_str(&self.generate_trait_html(trait_doc));
        }
        html.push_str("</section>\n");

        // Types section
        html.push_str("<section id='types'>\n");
        html.push_str("<h2>Types</h2>\n");
        for type_doc in self.types.values() {
            html.push_str(&self.generate_type_html(type_doc));
        }
        html.push_str("</section>\n");

        html.push_str("</main>\n");
        html.push_str("</body>\n</html>");

        html
    }

    /// Generate CSS for HTML documentation
    fn generate_css(&self) -> &str {
        r#"
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007acc;
            padding-bottom: 10px;
        }
        .search-box {
            margin: 20px 0;
        }
        #search {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        nav {
            background: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        nav ul {
            list-style: none;
            padding: 0;
        }
        nav li {
            margin: 10px 0;
        }
        nav a {
            color: #007acc;
            text-decoration: none;
            font-weight: bold;
        }
        section {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .trait, .type, .function {
            border-left: 4px solid #007acc;
            padding-left: 20px;
            margin: 20px 0;
        }
        code {
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        "#
    }

    /// Generate HTML for a trait
    fn generate_trait_html(&self, trait_doc: &DocumentedTrait) -> String {
        let mut html = String::new();
        html.push_str("<div class='trait'>\n");
        html.push_str(&format!("<h3>{}</h3>\n", trait_doc.name));
        html.push_str(&format!("<p>{}</p>\n", trait_doc.description));

        if !trait_doc.required_methods.is_empty() {
            html.push_str("<h4>Required Methods</h4>\n");
            html.push_str("<ul>\n");
            for method in &trait_doc.required_methods {
                html.push_str(&format!(
                    "<li><code>{}</code> - {}</li>\n",
                    method.signature, method.description
                ));
            }
            html.push_str("</ul>\n");
        }

        html.push_str("</div>\n");
        html
    }

    /// Generate HTML for a type
    fn generate_type_html(&self, type_doc: &DocumentedType) -> String {
        let mut html = String::new();
        html.push_str("<div class='type'>\n");
        html.push_str(&format!("<h3>{}</h3>\n", type_doc.name));
        html.push_str(&format!("<p>{}</p>\n", type_doc.description));
        html.push_str("</div>\n");
        html
    }

    /// Index a trait for search
    fn index_trait(&mut self, trait_doc: &DocumentedTrait) {
        let keywords: Vec<String> = trait_doc
            .name
            .split('_')
            .map(|s| s.to_lowercase())
            .collect();

        for keyword in keywords {
            self.search_index
                .trait_index
                .entry(keyword)
                .or_default()
                .push(trait_doc.name.clone());
        }
    }

    /// Index a type for search
    fn index_type(&mut self, type_doc: &DocumentedType) {
        let keywords: Vec<String> = type_doc.name.split('_').map(|s| s.to_lowercase()).collect();

        for keyword in keywords {
            self.search_index
                .type_index
                .entry(keyword)
                .or_default()
                .push(type_doc.name.clone());
        }
    }

    /// Index a function for search
    fn index_function(&mut self, func_doc: &DocumentedFunction) {
        let keywords: Vec<String> = func_doc.name.split('_').map(|s| s.to_lowercase()).collect();

        for keyword in keywords {
            self.search_index
                .function_index
                .entry(keyword)
                .or_default()
                .push(func_doc.name.clone());
        }
    }
}

impl Default for InteractiveAPIReference {
    fn default() -> Self {
        Self::new()
    }
}

/// Search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// Matching traits
    pub traits: Vec<DocumentedTrait>,
    /// Matching types
    pub types: Vec<DocumentedType>,
    /// Matching functions
    pub functions: Vec<DocumentedFunction>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_reference_creation() {
        let api_ref = InteractiveAPIReference::new();
        assert_eq!(api_ref.traits.len(), 0);
        assert_eq!(api_ref.types.len(), 0);
    }

    #[test]
    fn test_add_trait() {
        let mut api_ref = InteractiveAPIReference::new();

        let trait_doc = DocumentedTrait {
            name: "Estimator".to_string(),
            full_path: "sklears::traits::Estimator".to_string(),
            description: "Base trait for all estimators".to_string(),
            associated_types: vec![],
            required_methods: vec![],
            provided_methods: vec![],
            implementors: vec![],
            examples: vec![],
            related_traits: vec![],
            since: "0.1.0".to_string(),
        };

        api_ref.add_trait(trait_doc);
        assert_eq!(api_ref.traits.len(), 1);
    }

    #[test]
    fn test_search() {
        let mut api_ref = InteractiveAPIReference::new();

        let trait_doc = DocumentedTrait {
            name: "Estimator".to_string(),
            full_path: "sklears::traits::Estimator".to_string(),
            description: "Base trait for all estimators".to_string(),
            associated_types: vec![],
            required_methods: vec![],
            provided_methods: vec![],
            implementors: vec![],
            examples: vec![],
            related_traits: vec![],
            since: "0.1.0".to_string(),
        };

        api_ref.add_trait(trait_doc);

        let results = api_ref.search("estimator");
        assert_eq!(results.traits.len(), 1);
    }

    #[test]
    fn test_html_generation() {
        let api_ref = InteractiveAPIReference::new();
        let html = api_ref.generate_html();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("sklears API Reference"));
    }

    #[test]
    fn test_interactive_example() {
        let example = InteractiveExample {
            title: "Basic Usage".to_string(),
            code: "let x = 5;".to_string(),
            expected_output: Some("5".to_string()),
            runnable: true,
            language: "rust".to_string(),
        };

        assert_eq!(example.language, "rust");
        assert!(example.runnable);
    }

    #[test]
    fn test_documented_method() {
        let method = DocumentedMethod {
            name: "fit".to_string(),
            signature: "fn fit(&mut self, X: &Array2<f64>)".to_string(),
            description: "Fit the model".to_string(),
            parameters: vec![],
            return_type: "Result<()>".to_string(),
            examples: vec![],
            safety: None,
        };

        assert_eq!(method.name, "fit");
    }

    #[test]
    fn test_type_kind() {
        assert_eq!(TypeKind::Struct, TypeKind::Struct);
        assert_ne!(TypeKind::Struct, TypeKind::Enum);
    }

    #[test]
    fn test_visibility() {
        assert_eq!(Visibility::Public, Visibility::Public);
        assert_ne!(Visibility::Public, Visibility::Private);
    }

    #[test]
    fn test_search_results() {
        let results = SearchResults {
            traits: vec![],
            types: vec![],
            functions: vec![],
        };

        assert_eq!(results.traits.len(), 0);
    }

    #[test]
    fn test_config_default() {
        let config = APIReferenceConfig::default();
        assert!(config.enable_live_examples);
        assert!(config.enable_cross_references);
    }
}
