//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub(crate) mod base64 {
    pub fn encode(data: &str) -> String {
        data.chars()
            .map(|c| (c as u8).to_string())
            .collect::<Vec<_>>()
            .join("")
    }
}
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use crate::api_data_structures::{ApiReference, CodeExample};
    use crate::interactive_playground::types::*;
    use std::collections::HashMap;
    #[test]
    fn test_interactive_doc_generator() {
        let generator = InteractiveDocGenerator::new();
        assert!(generator.code_runner.execution_timeout.as_secs() > 0);
    }
    #[test]
    fn test_live_code_runner() {
        let runner = LiveCodeRunner::new();
        let example = CodeExample {
            title: "Test".to_string(),
            description: "Test example".to_string(),
            code: "println!(\"Hello, world!\");".to_string(),
            language: "rust".to_string(),
            runnable: true,
            expected_output: Some("Hello, world!".to_string()),
        };
        let result = runner.execute_example(&example).unwrap();
        assert_eq!(result.exit_code, 0);
    }
    #[test]
    fn test_code_validation() {
        let runner = LiveCodeRunner::new();
        assert!(runner.validate_code_safety("println!(\"Hello\");").is_ok());
        assert!(runner
            .validate_code_safety("unsafe { /* dangerous */ }")
            .is_err());
    }
    #[test]
    fn test_wasm_playground_manager() {
        let manager = WasmPlaygroundManager::new();
        assert!(manager.wasm_enabled);
    }
    #[test]
    fn test_ui_component_builder() {
        let builder = UIComponentBuilder::new();
        let api_ref = create_test_api_reference();
        let components = builder.generate_ui_components(&api_ref).unwrap();
        assert!(!components.is_empty());
        assert!(components.iter().any(|c| c.name == "code-editor"));
    }
    #[test]
    fn test_api_search_engine() {
        let mut search_engine = ApiSearchEngine::new();
        let api_ref = create_test_api_reference();
        let _index = search_engine.build_search_index(&api_ref).unwrap();
        assert!(!search_engine.indexed_items.is_empty());
    }
    #[test]
    fn test_signature_conversion() {
        let manager = WasmPlaygroundManager::new();
        let rust_sig = "fn test(&self, x: usize) -> Result<String>";
        let js_sig = manager.convert_to_js_signature(rust_sig).unwrap();
        assert!(js_sig.contains("Promise"));
        assert!(js_sig.contains("number"));
        assert!(js_sig.contains("string"));
    }
    fn create_test_api_reference() -> ApiReference {
        use crate::api_data_structures::*;
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
                fields: Vec::new(),
                trait_impls: Vec::new(),
            }],
            examples: vec![CodeExample {
                title: "Test Example".to_string(),
                description: "A test example".to_string(),
                code: "println!(\"Hello, test!\");".to_string(),
                language: "rust".to_string(),
                runnable: true,
                expected_output: Some("Hello, test!".to_string()),
            }],
            cross_references: HashMap::new(),
            metadata: ApiMetadata::default(),
        }
    }
}
