//! Rich API-reference types for the trait graph visualization subsystem.
//!
//! [`crate::api_data_structures`] defines a deliberately simple `TraitInfo`
//! shape (`description`/`path`/`methods: Vec<MethodInfo>` as plain strings)
//! that backs the modularized API reference generator
//! (`api_analyzers`/`api_formatters`/`api_generator_config`/...).
//!
//! The trait graph visualization system
//! (`crate::trait_explorer::graph_visualization`) needs a richer shape that
//! carries structured documentation, visibility, source location, and
//! feature-flag metadata for each trait, associated type, and method so that
//! it can render meaningful graphs (node coloring by stability, tooltips with
//! signatures, filtering by feature flag, etc.). This module provides that
//! richer shape.
//!
//! The two `TraitInfo` shapes are intentionally independent types: this
//! module does not modify or replace [`crate::api_data_structures::TraitInfo`],
//! which remains in use by the already-enabled modularized API reference
//! system.

use crate::api_data_structures::ParameterInfo;

/// Visibility levels for API items.
///
/// Re-exported from [`crate::api_data_structures`] so callers that only deal
/// with the graph-visualization API surface can refer to
/// `crate::api_reference_generator::Visibility` without an additional
/// dependency on [`crate::api_data_structures`].
pub use crate::api_data_structures::Visibility;

/// Rich description of a Rust trait definition.
///
/// This is the shape consumed by
/// [`crate::trait_explorer::graph_visualization::graph_generator::TraitGraphGenerator`]
/// when building a [`crate::trait_explorer::graph_visualization::TraitGraph`]
/// from source-level trait information.
#[derive(Debug, Clone)]
pub struct TraitInfo {
    /// Name of the trait (e.g. `"Estimator"`).
    pub name: String,
    /// Rustdoc documentation attached to the trait, if any.
    pub docs: Option<String>,
    /// Module path the trait is declared in (e.g. `"sklears_core::traits"`).
    pub module_path: Option<String>,
    /// Visibility of the trait declaration.
    pub visibility: Visibility,
    /// Generic parameters declared on the trait, e.g. `["T: Clone + Send"]`.
    pub generics: Vec<String>,
    /// Names of the supertraits this trait requires.
    pub supertraits: Vec<String>,
    /// Associated types declared by the trait.
    pub associated_types: Vec<AssociatedType>,
    /// Methods declared by the trait (required and provided).
    pub methods: Vec<MethodInfo>,
    /// Source file the trait is declared in, if known.
    pub source_file: Option<String>,
    /// Line number of the trait declaration within `source_file`, if known.
    pub source_line: Option<u32>,
    /// Cargo feature flags that must be enabled for this trait to be
    /// available (e.g. `["experimental"]`).
    pub feature_flags: Vec<String>,
}

/// An associated type declared by a trait.
#[derive(Debug, Clone)]
pub struct AssociatedType {
    /// Name of the associated type (e.g. `"Output"`).
    pub name: String,
    /// Trait bounds placed on the associated type (e.g. `["Clone", "Send"]`).
    pub bounds: Vec<String>,
    /// Default type, if the associated type declares one.
    pub default: Option<String>,
}

/// A method declared by a trait.
#[derive(Debug, Clone)]
pub struct MethodInfo {
    /// Name of the method.
    pub name: String,
    /// Full method signature as written in source, e.g.
    /// `"fn fit(&mut self, x: &Array2<f64>) -> Result<()>"`.
    pub signature: String,
    /// Rustdoc documentation attached to the method, if any.
    pub docs: Option<String>,
    /// Whether the method has no default implementation (must be implemented
    /// by every conforming type).
    pub is_required: bool,
    /// Whether the method is declared `async`.
    pub is_async: bool,
    /// Whether the method is declared `unsafe`.
    pub is_unsafe: bool,
    /// Generic parameters declared on the method itself (independent of the
    /// trait's own generics).
    pub generics: Vec<String>,
    /// Return type of the method, if it returns something other than `()`.
    pub return_type: Option<String>,
    /// Parameters accepted by the method (excluding `self`).
    pub arguments: Vec<ParameterInfo>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_trait_info() -> TraitInfo {
        TraitInfo {
            name: "Estimator".to_string(),
            docs: Some("Base trait for ML estimators.".to_string()),
            module_path: Some("sklears_core::traits".to_string()),
            visibility: Visibility::Public,
            generics: vec!["T: Clone".to_string()],
            supertraits: vec!["Send".to_string(), "Sync".to_string()],
            associated_types: vec![AssociatedType {
                name: "Config".to_string(),
                bounds: vec!["Default".to_string()],
                default: None,
            }],
            methods: vec![MethodInfo {
                name: "fit".to_string(),
                signature: "fn fit(&mut self, x: &Array2<f64>) -> Result<()>".to_string(),
                docs: Some("Fit the estimator to training data.".to_string()),
                is_required: true,
                is_async: false,
                is_unsafe: false,
                generics: Vec::new(),
                return_type: Some("Result<()>".to_string()),
                arguments: vec![ParameterInfo {
                    name: "x".to_string(),
                    param_type: "&Array2<f64>".to_string(),
                    description: "training features".to_string(),
                    optional: false,
                }],
            }],
            source_file: Some("src/traits.rs".to_string()),
            source_line: Some(10),
            feature_flags: Vec::new(),
        }
    }

    #[test]
    fn test_trait_info_construction() {
        let trait_info = sample_trait_info();
        assert_eq!(trait_info.name, "Estimator");
        assert_eq!(trait_info.supertraits.len(), 2);
        assert_eq!(trait_info.associated_types.len(), 1);
        assert_eq!(trait_info.methods.len(), 1);
        assert!(matches!(trait_info.visibility, Visibility::Public));
    }

    #[test]
    fn test_associated_type_construction() {
        let assoc = AssociatedType {
            name: "Item".to_string(),
            bounds: vec!["Clone".to_string(), "Debug".to_string()],
            default: Some("()".to_string()),
        };
        assert_eq!(assoc.name, "Item");
        assert_eq!(assoc.bounds.len(), 2);
        assert_eq!(assoc.default.as_deref(), Some("()"));
    }

    #[test]
    fn test_method_info_construction() {
        let trait_info = sample_trait_info();
        let method = &trait_info.methods[0];
        assert_eq!(method.name, "fit");
        assert!(method.is_required);
        assert!(!method.is_async);
        assert!(!method.is_unsafe);
        assert_eq!(method.arguments.len(), 1);
        assert_eq!(method.arguments[0].name, "x");
    }

    #[test]
    fn test_trait_info_clone() {
        let trait_info = sample_trait_info();
        let cloned = trait_info.clone();
        assert_eq!(trait_info.name, cloned.name);
        assert_eq!(trait_info.methods.len(), cloned.methods.len());
    }

    #[test]
    fn test_visibility_reexport_matches_api_data_structures() {
        // The re-exported `Visibility` must be the exact same type as
        // `crate::api_data_structures::Visibility`, not a copy.
        let v: Visibility = crate::api_data_structures::Visibility::Private;
        assert!(matches!(v, Visibility::Private));
    }
}
