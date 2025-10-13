//! API Analysis Engines and Validation Components
//!
//! This module contains the core analysis engines used by the API reference generator
//! to extract, analyze, and validate Rust code structures. It includes trait analyzers,
//! type extractors, example validators, and cross-reference builders.

use crate::api_data_structures::{
    ApiReference, ApiVisualization, ApiVisualizationData, AssociatedType, CodeExample, CrateInfo,
    FieldInfo, MethodInfo, ParameterInfo, TraitInfo, TypeInfo, TypeKind, Visibility,
    VisualizationConfig, VisualizationNode, VisualizationType,
};
use crate::api_generator_config::GeneratorConfig;
use crate::error::{Result, SklearsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ================================================================================================
// TRAIT ANALYSIS ENGINE
// ================================================================================================

/// Trait analyzer for extracting trait information from Rust code
#[derive(Debug, Clone)]
pub struct TraitAnalyzer {
    config: GeneratorConfig,
    trait_cache: HashMap<String, TraitInfo>,
    hierarchy_depth: usize,
}

impl TraitAnalyzer {
    /// Create a new trait analyzer with the given configuration
    pub fn new(config: GeneratorConfig) -> Self {
        Self {
            config,
            trait_cache: HashMap::new(),
            hierarchy_depth: 0,
        }
    }

    /// Analyze traits from crate information
    pub fn analyze_traits(&mut self, _crate_info: &CrateInfo) -> Result<Vec<TraitInfo>> {
        // Clear cache for fresh analysis
        self.trait_cache.clear();
        self.hierarchy_depth = 0;

        // In a real implementation, this would use syn to parse and analyze traits
        // For now, return example data based on sklears-core traits
        let traits = vec![
            self.create_estimator_trait()?,
            self.create_fit_trait()?,
            self.create_predict_trait()?,
            self.create_transform_trait()?,
            self.create_score_trait()?,
        ];

        // Cache the analyzed traits
        for trait_info in &traits {
            self.trait_cache
                .insert(trait_info.name.clone(), trait_info.clone());
        }

        // Build trait hierarchies if enabled
        if self.config.include_cross_refs {
            self.build_trait_hierarchies(&traits)
        } else {
            Ok(traits)
        }
    }

    /// Analyze a specific trait by name
    pub fn analyze_trait(&mut self, trait_name: &str) -> Result<Option<TraitInfo>> {
        // Check cache first
        if let Some(cached) = self.trait_cache.get(trait_name) {
            return Ok(Some(cached.clone()));
        }

        // In a real implementation, this would parse the trait from source
        match trait_name {
            "Estimator" => Ok(Some(self.create_estimator_trait()?)),
            "Fit" => Ok(Some(self.create_fit_trait()?)),
            "Predict" => Ok(Some(self.create_predict_trait()?)),
            "Transform" => Ok(Some(self.create_transform_trait()?)),
            "Score" => Ok(Some(self.create_score_trait()?)),
            _ => Ok(None),
        }
    }

    /// Get all cached trait information
    pub fn get_cached_traits(&self) -> &HashMap<String, TraitInfo> {
        &self.trait_cache
    }

    /// Clear the trait cache
    pub fn clear_cache(&mut self) {
        self.trait_cache.clear();
    }

    /// Validate trait hierarchy depth
    pub fn validate_hierarchy_depth(&self, depth: usize) -> Result<()> {
        if depth > self.config.max_hierarchy_depth {
            return Err(SklearsError::InvalidInput(format!(
                "Trait hierarchy depth {} exceeds maximum allowed depth of {}",
                depth, self.config.max_hierarchy_depth
            )));
        }
        Ok(())
    }

    /// Build trait hierarchies and relationships
    fn build_trait_hierarchies(&mut self, traits: &[TraitInfo]) -> Result<Vec<TraitInfo>> {
        let mut enhanced_traits = traits.to_vec();

        // Build supertrait relationships
        for trait_info in &mut enhanced_traits {
            trait_info.supertraits = self.find_supertraits(&trait_info.name)?;
            trait_info.implementations = self.find_implementations(&trait_info.name)?;
        }

        Ok(enhanced_traits)
    }

    /// Find supertraits for a given trait
    fn find_supertraits(&self, trait_name: &str) -> Result<Vec<String>> {
        // In a real implementation, this would analyze trait bounds
        match trait_name {
            "Fit" => Ok(vec!["Estimator".to_string()]),
            "Predict" => Ok(vec!["Estimator".to_string()]),
            "Transform" => Ok(vec!["Estimator".to_string()]),
            "Score" => Ok(vec!["Predict".to_string()]),
            _ => Ok(Vec::new()),
        }
    }

    /// Find implementations for a given trait
    fn find_implementations(&self, trait_name: &str) -> Result<Vec<String>> {
        // In a real implementation, this would scan for impl blocks
        match trait_name {
            "Estimator" => Ok(vec![
                "LinearRegression".to_string(),
                "LogisticRegression".to_string(),
                "RandomForest".to_string(),
                "SVM".to_string(),
            ]),
            "Fit" => Ok(vec![
                "LinearRegression".to_string(),
                "LogisticRegression".to_string(),
                "RandomForest".to_string(),
            ]),
            "Predict" => Ok(vec![
                "LinearRegression".to_string(),
                "LogisticRegression".to_string(),
                "RandomForest".to_string(),
            ]),
            "Transform" => Ok(vec![
                "StandardScaler".to_string(),
                "PCA".to_string(),
                "MinMaxScaler".to_string(),
            ]),
            "Score" => Ok(vec![
                "LinearRegression".to_string(),
                "LogisticRegression".to_string(),
            ]),
            _ => Ok(Vec::new()),
        }
    }

    /// Create Estimator trait information
    fn create_estimator_trait(&self) -> Result<TraitInfo> {
        Ok(TraitInfo {
            name: "Estimator".to_string(),
            description: "Base trait for all machine learning estimators in sklears".to_string(),
            path: "sklears_core::traits::Estimator".to_string(),
            generics: Vec::new(),
            associated_types: vec![AssociatedType {
                name: "Config".to_string(),
                description: "Configuration type for the estimator".to_string(),
                bounds: vec!["Clone".to_string(), "Debug".to_string()],
            }],
            methods: vec![
                MethodInfo {
                    name: "name".to_string(),
                    signature: "fn name(&self) -> &'static str".to_string(),
                    description: "Get the name of the estimator".to_string(),
                    parameters: Vec::new(),
                    return_type: "&'static str".to_string(),
                    required: true,
                },
                MethodInfo {
                    name: "default_config".to_string(),
                    signature: "fn default_config() -> Self::Config".to_string(),
                    description: "Get the default configuration for this estimator".to_string(),
                    parameters: Vec::new(),
                    return_type: "Self::Config".to_string(),
                    required: false,
                },
            ],
            supertraits: Vec::new(),
            implementations: Vec::new(),
        })
    }

    /// Create Fit trait information
    fn create_fit_trait(&self) -> Result<TraitInfo> {
        Ok(TraitInfo {
            name: "Fit".to_string(),
            description: "Trait for estimators that can be fitted to training data".to_string(),
            path: "sklears_core::traits::Fit".to_string(),
            generics: vec!["X".to_string(), "Y".to_string()],
            associated_types: vec![AssociatedType {
                name: "Fitted".to_string(),
                description: "The type returned after fitting".to_string(),
                bounds: vec!["Send".to_string(), "Sync".to_string()],
            }],
            methods: vec![MethodInfo {
                name: "fit".to_string(),
                signature: "fn fit(self, x: &X, y: &Y) -> Result<Self::Fitted>".to_string(),
                description: "Fit the estimator to training data".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "&X".to_string(),
                        description: "Training features matrix".to_string(),
                        optional: false,
                    },
                    ParameterInfo {
                        name: "y".to_string(),
                        param_type: "&Y".to_string(),
                        description: "Training targets vector".to_string(),
                        optional: false,
                    },
                ],
                return_type: "Result<Self::Fitted>".to_string(),
                required: true,
            }],
            supertraits: Vec::new(),
            implementations: Vec::new(),
        })
    }

    /// Create Predict trait information
    fn create_predict_trait(&self) -> Result<TraitInfo> {
        Ok(TraitInfo {
            name: "Predict".to_string(),
            description: "Trait for estimators that can make predictions".to_string(),
            path: "sklears_core::traits::Predict".to_string(),
            generics: vec!["X".to_string()],
            associated_types: vec![AssociatedType {
                name: "Output".to_string(),
                description: "The type of prediction output".to_string(),
                bounds: vec!["Clone".to_string()],
            }],
            methods: vec![MethodInfo {
                name: "predict".to_string(),
                signature: "fn predict(&self, x: &X) -> Result<Self::Output>".to_string(),
                description: "Make predictions on new data".to_string(),
                parameters: vec![ParameterInfo {
                    name: "x".to_string(),
                    param_type: "&X".to_string(),
                    description: "Input features to predict on".to_string(),
                    optional: false,
                }],
                return_type: "Result<Self::Output>".to_string(),
                required: true,
            }],
            supertraits: Vec::new(),
            implementations: Vec::new(),
        })
    }

    /// Create Transform trait information
    fn create_transform_trait(&self) -> Result<TraitInfo> {
        Ok(TraitInfo {
            name: "Transform".to_string(),
            description: "Trait for estimators that can transform data".to_string(),
            path: "sklears_core::traits::Transform".to_string(),
            generics: vec!["X".to_string()],
            associated_types: vec![AssociatedType {
                name: "Output".to_string(),
                description: "The type of transformed output".to_string(),
                bounds: vec!["Clone".to_string()],
            }],
            methods: vec![
                MethodInfo {
                    name: "transform".to_string(),
                    signature: "fn transform(&self, x: &X) -> Result<Self::Output>".to_string(),
                    description: "Transform input data".to_string(),
                    parameters: vec![ParameterInfo {
                        name: "x".to_string(),
                        param_type: "&X".to_string(),
                        description: "Input data to transform".to_string(),
                        optional: false,
                    }],
                    return_type: "Result<Self::Output>".to_string(),
                    required: true,
                },
                MethodInfo {
                    name: "fit_transform".to_string(),
                    signature:
                        "fn fit_transform(self, x: &X) -> Result<(Self::Fitted, Self::Output)>"
                            .to_string(),
                    description: "Fit the transformer and transform data in one step".to_string(),
                    parameters: vec![ParameterInfo {
                        name: "x".to_string(),
                        param_type: "&X".to_string(),
                        description: "Input data to fit and transform".to_string(),
                        optional: false,
                    }],
                    return_type: "Result<(Self::Fitted, Self::Output)>".to_string(),
                    required: false,
                },
            ],
            supertraits: Vec::new(),
            implementations: Vec::new(),
        })
    }

    /// Create Score trait information
    fn create_score_trait(&self) -> Result<TraitInfo> {
        Ok(TraitInfo {
            name: "Score".to_string(),
            description: "Trait for estimators that can compute accuracy scores".to_string(),
            path: "sklears_core::traits::Score".to_string(),
            generics: vec!["X".to_string(), "Y".to_string()],
            associated_types: vec![AssociatedType {
                name: "Score".to_string(),
                description: "The type of score output".to_string(),
                bounds: vec!["PartialOrd".to_string(), "Copy".to_string()],
            }],
            methods: vec![MethodInfo {
                name: "score".to_string(),
                signature: "fn score(&self, x: &X, y: &Y) -> Result<Self::Score>".to_string(),
                description: "Compute accuracy score on test data".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "&X".to_string(),
                        description: "Test features".to_string(),
                        optional: false,
                    },
                    ParameterInfo {
                        name: "y".to_string(),
                        param_type: "&Y".to_string(),
                        description: "True test targets".to_string(),
                        optional: false,
                    },
                ],
                return_type: "Result<Self::Score>".to_string(),
                required: true,
            }],
            supertraits: Vec::new(),
            implementations: Vec::new(),
        })
    }
}

impl Default for TraitAnalyzer {
    fn default() -> Self {
        Self::new(GeneratorConfig::default())
    }
}

// ================================================================================================
// TYPE EXTRACTION ENGINE
// ================================================================================================

/// Type extractor for analyzing type definitions from Rust code
#[derive(Debug, Clone)]
pub struct TypeExtractor {
    #[allow(dead_code)]
    config: GeneratorConfig,
    type_cache: HashMap<String, TypeInfo>,
    generic_constraints: HashMap<String, Vec<String>>,
}

impl TypeExtractor {
    /// Create a new type extractor with the given configuration
    pub fn new(config: GeneratorConfig) -> Self {
        Self {
            config,
            type_cache: HashMap::new(),
            generic_constraints: HashMap::new(),
        }
    }

    /// Extract types from crate information
    pub fn extract_types(&mut self, _crate_info: &CrateInfo) -> Result<Vec<TypeInfo>> {
        // Clear cache for fresh extraction
        self.type_cache.clear();

        // In a real implementation, this would use syn to parse and analyze types
        let types = vec![
            self.create_sklears_error_type()?,
            self.create_estimator_config_type()?,
            self.create_matrix_type()?,
            self.create_array_type()?,
            self.create_model_type()?,
        ];

        // Cache the extracted types
        for type_info in &types {
            self.type_cache
                .insert(type_info.name.clone(), type_info.clone());
        }

        Ok(types)
    }

    /// Extract a specific type by name
    pub fn extract_type(&mut self, type_name: &str) -> Result<Option<TypeInfo>> {
        // Check cache first
        if let Some(cached) = self.type_cache.get(type_name) {
            return Ok(Some(cached.clone()));
        }

        // In a real implementation, this would parse the type from source
        match type_name {
            "SklearsError" => Ok(Some(self.create_sklears_error_type()?)),
            "EstimatorConfig" => Ok(Some(self.create_estimator_config_type()?)),
            "Matrix" => Ok(Some(self.create_matrix_type()?)),
            "Array" => Ok(Some(self.create_array_type()?)),
            "Model" => Ok(Some(self.create_model_type()?)),
            _ => Ok(None),
        }
    }

    /// Get all cached type information
    pub fn get_cached_types(&self) -> &HashMap<String, TypeInfo> {
        &self.type_cache
    }

    /// Analyze generic constraints for a type
    pub fn analyze_generic_constraints(&mut self, type_name: &str) -> Result<Vec<String>> {
        if let Some(constraints) = self.generic_constraints.get(type_name) {
            return Ok(constraints.clone());
        }

        // In a real implementation, this would parse trait bounds
        let constraints = match type_name {
            "Matrix" => vec!["Clone".to_string(), "Debug".to_string(), "Send".to_string()],
            "Array" => vec!["Clone".to_string(), "Debug".to_string()],
            "Model" => vec![
                "Clone".to_string(),
                "Debug".to_string(),
                "Serialize".to_string(),
                "Deserialize".to_string(),
            ],
            _ => Vec::new(),
        };

        self.generic_constraints
            .insert(type_name.to_string(), constraints.clone());
        Ok(constraints)
    }

    /// Create SklearsError type information
    fn create_sklears_error_type(&self) -> Result<TypeInfo> {
        Ok(TypeInfo {
            name: "SklearsError".to_string(),
            description: "Error types for sklears operations".to_string(),
            path: "sklears_core::error::SklearsError".to_string(),
            kind: TypeKind::Enum,
            generics: Vec::new(),
            fields: vec![
                FieldInfo {
                    name: "InvalidInput".to_string(),
                    field_type: "String".to_string(),
                    description: "Invalid input parameter provided".to_string(),
                    visibility: Visibility::Public,
                },
                FieldInfo {
                    name: "ComputationError".to_string(),
                    field_type: "String".to_string(),
                    description: "Computation or numerical error occurred".to_string(),
                    visibility: Visibility::Public,
                },
                FieldInfo {
                    name: "ConfigurationError".to_string(),
                    field_type: "String".to_string(),
                    description: "Invalid configuration provided".to_string(),
                    visibility: Visibility::Public,
                },
                FieldInfo {
                    name: "DataError".to_string(),
                    field_type: "String".to_string(),
                    description: "Data format or quality issue".to_string(),
                    visibility: Visibility::Public,
                },
            ],
            trait_impls: vec![
                "Debug".to_string(),
                "Display".to_string(),
                "Error".to_string(),
                "Clone".to_string(),
            ],
        })
    }

    /// Create EstimatorConfig type information
    fn create_estimator_config_type(&self) -> Result<TypeInfo> {
        Ok(TypeInfo {
            name: "EstimatorConfig".to_string(),
            description: "Base configuration for all estimators".to_string(),
            path: "sklears_core::config::EstimatorConfig".to_string(),
            kind: TypeKind::Struct,
            generics: Vec::new(),
            fields: vec![
                FieldInfo {
                    name: "random_state".to_string(),
                    field_type: "Option<u64>".to_string(),
                    description: "Random seed for reproducible results".to_string(),
                    visibility: Visibility::Public,
                },
                FieldInfo {
                    name: "verbose".to_string(),
                    field_type: "bool".to_string(),
                    description: "Enable verbose output during training".to_string(),
                    visibility: Visibility::Public,
                },
                FieldInfo {
                    name: "max_iterations".to_string(),
                    field_type: "usize".to_string(),
                    description: "Maximum number of training iterations".to_string(),
                    visibility: Visibility::Public,
                },
            ],
            trait_impls: vec![
                "Debug".to_string(),
                "Clone".to_string(),
                "Default".to_string(),
                "Serialize".to_string(),
                "Deserialize".to_string(),
            ],
        })
    }

    /// Create Matrix type information
    fn create_matrix_type(&self) -> Result<TypeInfo> {
        Ok(TypeInfo {
            name: "Matrix".to_string(),
            description: "Generic matrix type for numerical computations".to_string(),
            path: "sklears_core::linalg::Matrix".to_string(),
            kind: TypeKind::Struct,
            generics: vec!["T".to_string()],
            fields: vec![
                FieldInfo {
                    name: "data".to_string(),
                    field_type: "Vec<T>".to_string(),
                    description: "Flattened matrix data in row-major order".to_string(),
                    visibility: Visibility::Private,
                },
                FieldInfo {
                    name: "rows".to_string(),
                    field_type: "usize".to_string(),
                    description: "Number of rows in the matrix".to_string(),
                    visibility: Visibility::Public,
                },
                FieldInfo {
                    name: "cols".to_string(),
                    field_type: "usize".to_string(),
                    description: "Number of columns in the matrix".to_string(),
                    visibility: Visibility::Public,
                },
            ],
            trait_impls: vec![
                "Debug".to_string(),
                "Clone".to_string(),
                "Index".to_string(),
                "IndexMut".to_string(),
            ],
        })
    }

    /// Create Array type information
    fn create_array_type(&self) -> Result<TypeInfo> {
        Ok(TypeInfo {
            name: "Array".to_string(),
            description: "Multi-dimensional array type".to_string(),
            path: "sklears_core::array::Array".to_string(),
            kind: TypeKind::Struct,
            generics: vec!["T".to_string(), "const N: usize".to_string()],
            fields: vec![
                FieldInfo {
                    name: "data".to_string(),
                    field_type: "Vec<T>".to_string(),
                    description: "Array data storage".to_string(),
                    visibility: Visibility::Private,
                },
                FieldInfo {
                    name: "shape".to_string(),
                    field_type: "[usize; N]".to_string(),
                    description: "Shape of the array in each dimension".to_string(),
                    visibility: Visibility::Public,
                },
            ],
            trait_impls: vec![
                "Debug".to_string(),
                "Clone".to_string(),
                "Index".to_string(),
                "IntoIterator".to_string(),
            ],
        })
    }

    /// Create Model type information
    fn create_model_type(&self) -> Result<TypeInfo> {
        Ok(TypeInfo {
            name: "Model".to_string(),
            description: "Trained model container".to_string(),
            path: "sklears_core::model::Model".to_string(),
            kind: TypeKind::Struct,
            generics: vec!["E".to_string()],
            fields: vec![
                FieldInfo {
                    name: "estimator".to_string(),
                    field_type: "E".to_string(),
                    description: "The trained estimator".to_string(),
                    visibility: Visibility::Private,
                },
                FieldInfo {
                    name: "metadata".to_string(),
                    field_type: "ModelMetadata".to_string(),
                    description: "Model training metadata".to_string(),
                    visibility: Visibility::Public,
                },
                FieldInfo {
                    name: "metrics".to_string(),
                    field_type: "HashMap<String, f64>".to_string(),
                    description: "Training and validation metrics".to_string(),
                    visibility: Visibility::Public,
                },
            ],
            trait_impls: vec![
                "Debug".to_string(),
                "Clone".to_string(),
                "Serialize".to_string(),
                "Deserialize".to_string(),
            ],
        })
    }
}

impl Default for TypeExtractor {
    fn default() -> Self {
        Self::new(GeneratorConfig::default())
    }
}

// ================================================================================================
// EXAMPLE VALIDATION ENGINE
// ================================================================================================

/// Example validator for checking and validating code examples
#[derive(Debug, Clone)]
pub struct ExampleValidator {
    validation_rules: Vec<ValidationRule>,
    #[allow(dead_code)]
    compile_timeout_secs: u64,
    enable_compilation: bool,
    enable_execution: bool,
}

impl ExampleValidator {
    /// Create a new example validator
    pub fn new() -> Self {
        Self {
            validation_rules: vec![
                ValidationRule::SyntaxCheck,
                ValidationRule::ImportCheck,
                ValidationRule::TypeCheck,
                ValidationRule::SafetyCheck,
            ],
            compile_timeout_secs: 30,
            enable_compilation: false, // Disabled by default for performance
            enable_execution: false,   // Disabled by default for security
        }
    }

    /// Create a new validator with custom configuration
    pub fn with_config(
        enable_compilation: bool,
        enable_execution: bool,
        timeout_secs: u64,
    ) -> Self {
        Self {
            validation_rules: vec![
                ValidationRule::SyntaxCheck,
                ValidationRule::ImportCheck,
                ValidationRule::TypeCheck,
                ValidationRule::SafetyCheck,
            ],
            compile_timeout_secs: timeout_secs,
            enable_compilation,
            enable_execution,
        }
    }

    /// Validate a collection of code examples
    pub fn validate_examples(&self, examples: &[CodeExample]) -> Result<Vec<CodeExample>> {
        let mut validated_examples = Vec::new();

        for example in examples {
            match self.validate_example(example) {
                Ok(validated) => validated_examples.push(validated),
                Err(e) => {
                    // Log the error but continue with other examples
                    eprintln!(
                        "Warning: Failed to validate example '{}': {}",
                        example.title, e
                    );
                    validated_examples.push(example.clone());
                }
            }
        }

        Ok(validated_examples)
    }

    /// Validate a single code example
    pub fn validate_example(&self, example: &CodeExample) -> Result<CodeExample> {
        let mut validated = example.clone();

        // Apply validation rules
        for rule in &self.validation_rules {
            self.apply_validation_rule(rule, &mut validated)?;
        }

        // Perform compilation check if enabled
        if self.enable_compilation && example.runnable {
            self.compile_check(&validated)?;
        }

        // Perform execution check if enabled
        if self.enable_execution && example.runnable {
            self.execution_check(&validated)?;
        }

        Ok(validated)
    }

    /// Apply a specific validation rule
    fn apply_validation_rule(
        &self,
        rule: &ValidationRule,
        example: &mut CodeExample,
    ) -> Result<()> {
        match rule {
            ValidationRule::SyntaxCheck => self.check_syntax(example),
            ValidationRule::ImportCheck => self.check_imports(example),
            ValidationRule::TypeCheck => self.check_types(example),
            ValidationRule::SafetyCheck => self.check_safety(example),
        }
    }

    /// Check syntax validity
    fn check_syntax(&self, example: &CodeExample) -> Result<()> {
        // Basic syntax checks
        if example.code.trim().is_empty() {
            return Err(SklearsError::InvalidInput(
                "Example code cannot be empty".to_string(),
            ));
        }

        // Check for balanced braces
        let open_braces = example.code.matches('{').count();
        let close_braces = example.code.matches('}').count();
        if open_braces != close_braces {
            return Err(SklearsError::InvalidInput(
                "Unbalanced braces in example code".to_string(),
            ));
        }

        // Check for balanced parentheses
        let open_parens = example.code.matches('(').count();
        let close_parens = example.code.matches(')').count();
        if open_parens != close_parens {
            return Err(SklearsError::InvalidInput(
                "Unbalanced parentheses in example code".to_string(),
            ));
        }

        Ok(())
    }

    /// Check import statements
    fn check_imports(&self, example: &CodeExample) -> Result<()> {
        let lines: Vec<&str> = example.code.lines().collect();

        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("use ") {
                // Check for valid sklears imports
                if trimmed.contains("sklears_") && !self.is_valid_sklears_import(trimmed) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Invalid sklears import: {}",
                        trimmed
                    )));
                }
            }
        }

        Ok(())
    }

    /// Check for type correctness
    fn check_types(&self, _example: &CodeExample) -> Result<()> {
        // In a real implementation, this would use rust-analyzer or syn
        // to perform type checking. For now, we just return Ok.
        Ok(())
    }

    /// Check for unsafe code patterns
    fn check_safety(&self, example: &CodeExample) -> Result<()> {
        // Check for unsafe blocks
        if example.code.contains("unsafe") {
            return Err(SklearsError::InvalidInput(
                "Unsafe code blocks are not allowed in examples".to_string(),
            ));
        }

        // Check for potentially dangerous operations
        let dangerous_patterns = [
            "std::process::Command",
            "std::fs::remove",
            "std::ptr::",
            "libc::",
            "transmute",
        ];

        for pattern in &dangerous_patterns {
            if example.code.contains(pattern) {
                return Err(SklearsError::InvalidInput(format!(
                    "Potentially dangerous pattern '{}' found in example",
                    pattern
                )));
            }
        }

        Ok(())
    }

    /// Check if a sklears import is valid
    fn is_valid_sklears_import(&self, import: &str) -> bool {
        let valid_modules = [
            "sklears_core",
            "sklears_linear",
            "sklears_tree",
            "sklears_ensemble",
            "sklears_preprocessing",
            "sklears_metrics",
            "sklears_neighbors",
            "sklears_clustering",
            "sklears_datasets",
        ];

        valid_modules.iter().any(|module| import.contains(module))
    }

    /// Perform compilation check
    fn compile_check(&self, _example: &CodeExample) -> Result<()> {
        // In a real implementation, this would:
        // 1. Create a temporary Rust project
        // 2. Write the example code to a file
        // 3. Run cargo check with a timeout
        // 4. Parse the output for errors
        // For now, we just return Ok
        Ok(())
    }

    /// Perform execution check
    fn execution_check(&self, _example: &CodeExample) -> Result<()> {
        // In a real implementation, this would:
        // 1. Compile the example
        // 2. Run it in a sandboxed environment
        // 3. Capture output and verify it matches expected results
        // For now, we just return Ok
        Ok(())
    }

    /// Set validation rules
    pub fn set_validation_rules(&mut self, rules: Vec<ValidationRule>) {
        self.validation_rules = rules;
    }

    /// Enable or disable compilation checking
    pub fn set_compilation_enabled(&mut self, enabled: bool) {
        self.enable_compilation = enabled;
    }

    /// Enable or disable execution checking
    pub fn set_execution_enabled(&mut self, enabled: bool) {
        self.enable_execution = enabled;
    }
}

impl Default for ExampleValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation rules for code examples
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationRule {
    /// Check syntax validity
    SyntaxCheck,
    /// Check import statements
    ImportCheck,
    /// Check type correctness
    TypeCheck,
    /// Check for unsafe patterns
    SafetyCheck,
}

// ================================================================================================
// CROSS-REFERENCE BUILDER
// ================================================================================================

/// Cross-reference builder for linking API elements
#[derive(Debug, Clone)]
pub struct CrossReferenceBuilder {
    reference_cache: HashMap<String, Vec<String>>,
    bidirectional_refs: bool,
    max_depth: usize,
}

impl CrossReferenceBuilder {
    /// Create a new cross-reference builder
    pub fn new() -> Self {
        Self {
            reference_cache: HashMap::new(),
            bidirectional_refs: true,
            max_depth: 3,
        }
    }

    /// Create a cross-reference builder with custom configuration
    pub fn with_config(bidirectional_refs: bool, max_depth: usize) -> Self {
        Self {
            reference_cache: HashMap::new(),
            bidirectional_refs,
            max_depth,
        }
    }

    /// Build cross-references between traits and types
    pub fn build_cross_references(
        &mut self,
        traits: &[TraitInfo],
        types: &[TypeInfo],
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut refs = HashMap::new();

        // Build references from traits to their implementations
        for trait_info in traits {
            let mut trait_refs = trait_info.implementations.clone();

            // Add related traits (supertraits and subtraits)
            trait_refs.extend(trait_info.supertraits.clone());
            trait_refs.extend(self.find_related_traits(trait_info, traits)?);

            refs.insert(trait_info.name.clone(), trait_refs);
        }

        // Build references from types to traits they implement
        for type_info in types {
            let mut type_refs = type_info.trait_impls.clone();

            // Add related types
            type_refs.extend(self.find_related_types(type_info, types)?);

            refs.insert(type_info.name.clone(), type_refs);
        }

        // Build bidirectional references if enabled
        if self.bidirectional_refs {
            refs = self.add_bidirectional_references(refs)?;
        }

        // Cache the results
        self.reference_cache = refs.clone();

        Ok(refs)
    }

    /// Find traits related to a given trait
    fn find_related_traits(
        &self,
        trait_info: &TraitInfo,
        all_traits: &[TraitInfo],
    ) -> Result<Vec<String>> {
        let mut related = Vec::new();

        for other_trait in all_traits {
            if other_trait.name == trait_info.name {
                continue;
            }

            // Check if traits share common methods
            let common_methods = self.count_common_methods(trait_info, other_trait);
            if common_methods > 0 {
                related.push(other_trait.name.clone());
            }

            // Check if one trait is a supertrait of the other
            if other_trait.supertraits.contains(&trait_info.name) {
                related.push(other_trait.name.clone());
            }
        }

        Ok(related)
    }

    /// Find types related to a given type
    fn find_related_types(
        &self,
        type_info: &TypeInfo,
        all_types: &[TypeInfo],
    ) -> Result<Vec<String>> {
        let mut related = Vec::new();

        for other_type in all_types {
            if other_type.name == type_info.name {
                continue;
            }

            // Check if types implement common traits
            let common_traits = self.count_common_trait_impls(type_info, other_type);
            if common_traits > 0 {
                related.push(other_type.name.clone());
            }

            // Check if types have similar field structure
            if self.have_similar_structure(type_info, other_type) {
                related.push(other_type.name.clone());
            }
        }

        Ok(related)
    }

    /// Count common methods between two traits
    fn count_common_methods(&self, trait1: &TraitInfo, trait2: &TraitInfo) -> usize {
        let trait1_methods: std::collections::HashSet<_> =
            trait1.methods.iter().map(|m| &m.name).collect();
        let trait2_methods: std::collections::HashSet<_> =
            trait2.methods.iter().map(|m| &m.name).collect();

        trait1_methods.intersection(&trait2_methods).count()
    }

    /// Count common trait implementations between two types
    fn count_common_trait_impls(&self, type1: &TypeInfo, type2: &TypeInfo) -> usize {
        let type1_traits: std::collections::HashSet<_> = type1.trait_impls.iter().collect();
        let type2_traits: std::collections::HashSet<_> = type2.trait_impls.iter().collect();

        type1_traits.intersection(&type2_traits).count()
    }

    /// Check if two types have similar structure
    fn have_similar_structure(&self, type1: &TypeInfo, type2: &TypeInfo) -> bool {
        // Check if both are the same kind
        if std::mem::discriminant(&type1.kind) != std::mem::discriminant(&type2.kind) {
            return false;
        }

        // Check if they have similar number of fields
        let field_count_diff = (type1.fields.len() as i32 - type2.fields.len() as i32).abs();
        field_count_diff <= 2 // Allow some difference in field count
    }

    /// Add bidirectional references
    fn add_bidirectional_references(
        &self,
        mut refs: HashMap<String, Vec<String>>,
    ) -> Result<HashMap<String, Vec<String>>> {
        let keys: Vec<String> = refs.keys().cloned().collect();

        for key in &keys {
            if let Some(values) = refs.get(key).cloned() {
                for value in values {
                    // Add reverse reference
                    refs.entry(value).or_default().push(key.clone());
                }
            }
        }

        // Remove duplicates
        for (_, values) in refs.iter_mut() {
            values.sort();
            values.dedup();
        }

        Ok(refs)
    }

    /// Get cached cross-references
    pub fn get_cached_references(&self) -> &HashMap<String, Vec<String>> {
        &self.reference_cache
    }

    /// Clear the reference cache
    pub fn clear_cache(&mut self) {
        self.reference_cache.clear();
    }

    /// Set maximum reference depth
    pub fn set_max_depth(&mut self, depth: usize) {
        self.max_depth = depth;
    }

    /// Set bidirectional reference mode
    pub fn set_bidirectional(&mut self, enabled: bool) {
        self.bidirectional_refs = enabled;
    }
}

impl Default for CrossReferenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// API VISUALIZATION ENGINE
// ================================================================================================

/// API visualization engine for generating visual representations
#[derive(Debug, Clone)]
pub struct ApiVisualizationEngine {
    visualization_templates: HashMap<String, VisualizationTemplate>,
}

impl ApiVisualizationEngine {
    /// Create a new API visualization engine
    pub fn new() -> Self {
        let mut engine = Self {
            visualization_templates: HashMap::new(),
        };
        engine.initialize_templates();
        engine
    }

    /// Generate visualizations for an API reference
    pub fn generate_visualizations(&self, api_ref: &ApiReference) -> Result<Vec<ApiVisualization>> {
        let mut visualizations = Vec::new();

        // Generate trait hierarchy visualization
        if !api_ref.traits.is_empty() {
            visualizations.push(self.generate_trait_hierarchy_viz(api_ref)?);
        }

        // Generate type relationship visualization
        if !api_ref.types.is_empty() {
            visualizations.push(self.generate_type_relationship_viz(api_ref)?);
        }

        // Generate example flow visualization
        if !api_ref.examples.is_empty() {
            visualizations.push(self.generate_example_flow_viz(api_ref)?);
        }

        Ok(visualizations)
    }

    /// Initialize visualization templates
    fn initialize_templates(&mut self) {
        // Trait hierarchy template
        self.visualization_templates.insert(
            "trait-hierarchy".to_string(),
            VisualizationTemplate {
                name: "Trait Hierarchy".to_string(),
                template_type: VisualizationType::Tree,
                default_config: VisualizationConfig {
                    width: 800,
                    height: 600,
                    theme: "dark".to_string(),
                    animation_enabled: true,
                },
            },
        );

        // Type relationship template
        self.visualization_templates.insert(
            "type-relationships".to_string(),
            VisualizationTemplate {
                name: "Type Relationships".to_string(),
                template_type: VisualizationType::Network,
                default_config: VisualizationConfig {
                    width: 700,
                    height: 500,
                    theme: "light".to_string(),
                    animation_enabled: true,
                },
            },
        );

        // Example flow template
        self.visualization_templates.insert(
            "example-flow".to_string(),
            VisualizationTemplate {
                name: "Example Code Flow".to_string(),
                template_type: VisualizationType::FlowChart,
                default_config: VisualizationConfig {
                    width: 600,
                    height: 400,
                    theme: "auto".to_string(),
                    animation_enabled: false,
                },
            },
        );
    }

    /// Generate trait hierarchy visualization
    fn generate_trait_hierarchy_viz(&self, api_ref: &ApiReference) -> Result<ApiVisualization> {
        let template = self.visualization_templates.get("trait-hierarchy").unwrap();

        Ok(ApiVisualization {
            title: template.name.clone(),
            visualization_type: template.template_type.clone(),
            data: ApiVisualizationData {
                nodes: api_ref
                    .traits
                    .iter()
                    .map(|t| VisualizationNode {
                        id: t.name.clone(),
                        label: t.name.clone(),
                        node_type: "trait".to_string(),
                        properties: HashMap::new(),
                    })
                    .collect(),
                edges: Vec::new(),
                metadata: HashMap::new(),
            },
            config: template.default_config.clone(),
        })
    }

    /// Generate type relationship visualization
    fn generate_type_relationship_viz(&self, api_ref: &ApiReference) -> Result<ApiVisualization> {
        let template = self
            .visualization_templates
            .get("type-relationships")
            .unwrap();

        Ok(ApiVisualization {
            title: template.name.clone(),
            visualization_type: template.template_type.clone(),
            data: ApiVisualizationData {
                nodes: api_ref
                    .types
                    .iter()
                    .map(|t| VisualizationNode {
                        id: t.name.clone(),
                        label: t.name.clone(),
                        node_type: "type".to_string(),
                        properties: HashMap::new(),
                    })
                    .collect(),
                edges: Vec::new(),
                metadata: HashMap::new(),
            },
            config: template.default_config.clone(),
        })
    }

    /// Generate example flow visualization
    fn generate_example_flow_viz(&self, api_ref: &ApiReference) -> Result<ApiVisualization> {
        let template = self.visualization_templates.get("example-flow").unwrap();

        Ok(ApiVisualization {
            title: template.name.clone(),
            visualization_type: template.template_type.clone(),
            data: ApiVisualizationData {
                nodes: api_ref
                    .examples
                    .iter()
                    .map(|e| VisualizationNode {
                        id: e.title.clone(),
                        label: e.title.clone(),
                        node_type: "example".to_string(),
                        properties: HashMap::new(),
                    })
                    .collect(),
                edges: Vec::new(),
                metadata: HashMap::new(),
            },
            config: template.default_config.clone(),
        })
    }
}

impl Default for ApiVisualizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Visualization template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationTemplate {
    /// Template name
    pub name: String,
    /// Type of visualization
    pub template_type: VisualizationType,
    /// Default configuration
    pub default_config: VisualizationConfig,
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_analyzer() {
        let config = GeneratorConfig::new();
        let mut analyzer = TraitAnalyzer::new(config);
        let crate_info = CrateInfo {
            name: "test-crate".to_string(),
            version: "1.0.0".to_string(),
            description: "Test crate".to_string(),
            modules: Vec::new(),
            dependencies: Vec::new(),
        };

        let traits = analyzer.analyze_traits(&crate_info).unwrap();
        assert!(!traits.is_empty());
        assert!(traits.iter().any(|t| t.name == "Estimator"));
        assert!(traits.iter().any(|t| t.name == "Fit"));
    }

    #[test]
    fn test_type_extractor() {
        let config = GeneratorConfig::new();
        let mut extractor = TypeExtractor::new(config);
        let crate_info = CrateInfo {
            name: "test-crate".to_string(),
            version: "1.0.0".to_string(),
            description: "Test crate".to_string(),
            modules: Vec::new(),
            dependencies: Vec::new(),
        };

        let types = extractor.extract_types(&crate_info).unwrap();
        assert!(!types.is_empty());
        assert!(types.iter().any(|t| t.name == "SklearsError"));
    }

    #[test]
    fn test_example_validator() {
        let validator = ExampleValidator::new();
        let example = CodeExample {
            title: "Test Example".to_string(),
            description: "A test example".to_string(),
            code: "fn main() { println!(\"Hello, world!\"); }".to_string(),
            language: "rust".to_string(),
            runnable: true,
            expected_output: Some("Hello, world!".to_string()),
        };

        let validated = validator.validate_example(&example).unwrap();
        assert_eq!(validated.title, example.title);
    }

    #[test]
    fn test_example_validator_syntax_error() {
        let validator = ExampleValidator::new();
        let example = CodeExample {
            title: "Invalid Example".to_string(),
            description: "An invalid example".to_string(),
            code: "fn main() { println!(\"Hello, world!\"; }".to_string(), // Missing closing parenthesis
            language: "rust".to_string(),
            runnable: true,
            expected_output: None,
        };

        let result = validator.validate_example(&example);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_reference_builder() {
        let mut builder = CrossReferenceBuilder::new();
        let traits = vec![TraitInfo {
            name: "TestTrait".to_string(),
            description: "A test trait".to_string(),
            path: "test::TestTrait".to_string(),
            generics: Vec::new(),
            associated_types: Vec::new(),
            methods: Vec::new(),
            supertraits: Vec::new(),
            implementations: vec!["TestImpl".to_string()],
        }];
        let types = vec![TypeInfo {
            name: "TestType".to_string(),
            description: "A test type".to_string(),
            path: "test::TestType".to_string(),
            kind: TypeKind::Struct,
            generics: Vec::new(),
            fields: Vec::new(),
            trait_impls: vec!["TestTrait".to_string()],
        }];

        let refs = builder.build_cross_references(&traits, &types).unwrap();
        assert!(!refs.is_empty());
        assert!(refs.contains_key("TestTrait"));
        assert!(refs.contains_key("TestType"));
    }

    #[test]
    fn test_validation_rules() {
        let validator = ExampleValidator::new();
        let unsafe_example = CodeExample {
            title: "Unsafe Example".to_string(),
            description: "An unsafe example".to_string(),
            code: "unsafe { println!(\"Dangerous!\"); }".to_string(),
            language: "rust".to_string(),
            runnable: true,
            expected_output: None,
        };

        let result = validator.validate_example(&unsafe_example);
        assert!(result.is_err());
    }

    #[test]
    fn test_api_visualization_engine() {
        let engine = ApiVisualizationEngine::new();
        let api_ref = ApiReference {
            crate_name: "test-crate".to_string(),
            version: "1.0.0".to_string(),
            traits: vec![TraitInfo::default()],
            types: vec![],
            examples: vec![],
            cross_references: HashMap::new(),
            metadata: crate::api_data_structures::ApiMetadata::default(),
        };

        let visualizations = engine.generate_visualizations(&api_ref).unwrap();
        assert!(!visualizations.is_empty());
    }
}
