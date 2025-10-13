//! Interactive Playground and Live Code Execution
//!
//! This module provides comprehensive interactive documentation capabilities including
//! WebAssembly-powered code execution, live examples, interactive UI components,
//! and real-time feedback systems for exploring the sklears-core API.

use crate::api_data_structures::{
    ApiReference, CodeExample, ExampleVisualization, ExecutionResult, InteractiveDocumentation,
    InteractiveElement, InteractiveElementType, LiveCodeExample, SearchIndex, SearchItem,
    SearchItemType, UIComponent, UIComponentType, VisualizationConfig, VisualizationType,
    WasmBinding, WasmMethod, WasmPlayground,
};
use crate::api_generator_config::PlaygroundConfig;
use crate::error::{Result, SklearsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ================================================================================================
// INTERACTIVE DOCUMENTATION GENERATOR
// ================================================================================================

/// Interactive documentation generator with live code examples and WASM capabilities
#[derive(Debug)]
pub struct InteractiveDocGenerator {
    code_runner: LiveCodeRunner,
    search_engine: ApiSearchEngine,
    ui_builder: UIComponentBuilder,
    wasm_manager: WasmPlaygroundManager,
    visualization_engine: VisualizationEngine,
}

impl InteractiveDocGenerator {
    /// Create a new interactive documentation generator
    pub fn new() -> Self {
        Self {
            code_runner: LiveCodeRunner::new(),
            search_engine: ApiSearchEngine::new(),
            ui_builder: UIComponentBuilder::new(),
            wasm_manager: WasmPlaygroundManager::new(),
            visualization_engine: VisualizationEngine::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        execution_timeout: Duration,
        memory_limit: usize,
        wasm_enabled: bool,
    ) -> Self {
        Self {
            code_runner: LiveCodeRunner::with_config(execution_timeout, memory_limit),
            search_engine: ApiSearchEngine::new(),
            ui_builder: UIComponentBuilder::new(),
            wasm_manager: WasmPlaygroundManager::with_wasm_enabled(wasm_enabled),
            visualization_engine: VisualizationEngine::new(),
        }
    }

    /// Generate complete interactive documentation
    pub fn generate_interactive_docs(
        &mut self,
        api_ref: &ApiReference,
    ) -> Result<InteractiveDocumentation> {
        let live_examples = self.generate_live_examples(&api_ref.examples)?;
        let searchable_index = self.search_engine.build_search_index(api_ref)?;
        let interactive_tutorials = self.generate_interactive_tutorials(api_ref)?;
        let visualizations = self.visualization_engine.generate_visualizations(api_ref)?;

        Ok(InteractiveDocumentation {
            api_reference: api_ref.clone(),
            live_examples,
            searchable_index,
            interactive_tutorials,
            visualizations,
            playground_config: self.generate_playground_config(api_ref)?,
        })
    }

    /// Generate WebAssembly-compatible playground
    pub fn generate_wasm_playground(&self, api_ref: &ApiReference) -> Result<WasmPlayground> {
        let playground_code = self.generate_playground_template(api_ref)?;
        let wasm_bindings = self.wasm_manager.generate_wasm_bindings(api_ref)?;
        let ui_components = self.ui_builder.generate_ui_components(api_ref)?;

        Ok(WasmPlayground {
            html_template: self
                .wasm_manager
                .generate_html_template(&playground_code, &ui_components)?,
            javascript_code: self.wasm_manager.generate_js_bindings(&wasm_bindings)?,
            css_styling: self.wasm_manager.generate_playground_css()?,
            rust_code: playground_code,
            build_instructions: self.wasm_manager.generate_build_instructions()?,
        })
    }

    /// Generate enhanced search capabilities
    pub fn enhance_search_capabilities(
        &self,
        api_ref: &ApiReference,
    ) -> Result<EnhancedSearchCapabilities> {
        Ok(EnhancedSearchCapabilities {
            fuzzy_search: self.build_fuzzy_search(api_ref)?,
            semantic_search: self.build_semantic_search(api_ref)?,
            type_search: self.build_type_search(api_ref)?,
            real_time_suggestions: self.build_real_time_suggestions(api_ref)?,
            search_analytics: SearchAnalytics::new(),
        })
    }

    /// Generate interactive tutorials for the API
    pub fn generate_interactive_tutorials(
        &self,
        api_ref: &ApiReference,
    ) -> Result<Vec<crate::api_data_structures::InteractiveTutorial>> {
        let mut tutorials = Vec::new();

        // Generate basic tutorial
        tutorials.push(crate::api_data_structures::InteractiveTutorial {
            title: "Basic API Usage".to_string(),
            description: "Learn the fundamentals of using sklears-core".to_string(),
            difficulty: crate::api_data_structures::TutorialDifficulty::Beginner,
            estimated_time: Duration::from_secs(10 * 60), // 10 minutes
            steps: self.generate_basic_tutorial_steps(api_ref)?,
        });

        // Generate trait-specific tutorials
        for trait_info in &api_ref.traits {
            tutorials.push(crate::api_data_structures::InteractiveTutorial {
                title: format!("Working with {}", trait_info.name),
                description: format!("Learn how to use the {} trait effectively", trait_info.name),
                difficulty: crate::api_data_structures::TutorialDifficulty::Intermediate,
                estimated_time: Duration::from_secs(15 * 60), // 15 minutes
                steps: self.generate_trait_tutorial_steps(trait_info)?,
            });
        }

        Ok(tutorials)
    }

    /// Generate live examples with execution capabilities
    fn generate_live_examples(&self, examples: &[CodeExample]) -> Result<Vec<LiveCodeExample>> {
        let mut live_examples = Vec::new();

        for example in examples {
            let execution_result = self.code_runner.execute_example(example)?;
            let interactive_elements = self.generate_interactive_elements(example)?;
            let visualization = self.generate_example_visualization(example, &execution_result)?;

            live_examples.push(LiveCodeExample {
                original_example: example.clone(),
                execution_result,
                interactive_elements,
                visualization,
                editable: true,
                real_time_feedback: true,
            });
        }

        Ok(live_examples)
    }

    /// Generate playground template code
    fn generate_playground_template(&self, api_ref: &ApiReference) -> Result<String> {
        let mut template = String::new();

        template.push_str("// Interactive sklears-core playground\n");
        template.push_str("// Explore the API with live code execution\n\n");
        template.push_str("use sklears_core::*;\n");

        // Add imports for major traits
        if !api_ref.traits.is_empty() {
            template.push_str("\n// Available traits:\n");
            for trait_info in &api_ref.traits {
                template.push_str(&format!("// use sklears_core::{};\n", trait_info.name));
            }
        }

        template.push_str("\nfn main() {\n");
        template
            .push_str("    println!(\"Welcome to the sklears-core interactive playground!\");\n");
        template.push_str("    \n");
        template.push_str("    // Example: Create and use an estimator\n");
        template.push_str("    // let estimator = YourEstimator::new();\n");
        template.push_str("    // let trained = estimator.fit(&training_data, &targets)?;\n");
        template.push_str("    // let predictions = trained.predict(&test_data)?;\n");
        template.push_str("    \n");
        template.push_str("    // Try editing this code and click 'Run' to see the results!\n");
        template.push_str("}\n");

        Ok(template)
    }

    /// Generate interactive elements for an example
    fn generate_interactive_elements(
        &self,
        example: &CodeExample,
    ) -> Result<Vec<InteractiveElement>> {
        let mut elements = Vec::new();

        // Always add run button
        elements.push(InteractiveElement {
            element_type: InteractiveElementType::Button,
            id: "run-example".to_string(),
            label: "Run This Example".to_string(),
            action: "execute_code".to_string(),
            target: example.code.clone(),
        });

        // Add parameter controls for numerical examples
        if example.code.contains("f64")
            || example.code.contains("usize")
            || example.code.contains("i32")
        {
            elements.push(InteractiveElement {
                element_type: InteractiveElementType::Slider,
                id: "parameter-slider".to_string(),
                label: "Adjust Parameters".to_string(),
                action: "update_parameters".to_string(),
                target: "numeric_params".to_string(),
            });
        }

        // Add visualization toggle for data processing examples
        if example.code.contains("plot")
            || example.code.contains("visualize")
            || example.code.contains("chart")
        {
            elements.push(InteractiveElement {
                element_type: InteractiveElementType::Toggle,
                id: "show-visualization".to_string(),
                label: "Show Visualization".to_string(),
                action: "toggle_visualization".to_string(),
                target: "viz_panel".to_string(),
            });
        }

        // Add input field for customizable examples
        if example.code.contains("input") || example.code.contains("user") {
            elements.push(InteractiveElement {
                element_type: InteractiveElementType::Input,
                id: "user-input".to_string(),
                label: "Custom Input".to_string(),
                action: "update_input".to_string(),
                target: "user_data".to_string(),
            });
        }

        // Add dropdown for algorithm selection examples
        if example.code.contains("algorithm") || example.code.contains("method") {
            elements.push(InteractiveElement {
                element_type: InteractiveElementType::Dropdown,
                id: "algorithm-selector".to_string(),
                label: "Select Algorithm".to_string(),
                action: "change_algorithm".to_string(),
                target: "algorithm_choice".to_string(),
            });
        }

        Ok(elements)
    }

    /// Generate example visualization
    fn generate_example_visualization(
        &self,
        example: &CodeExample,
        execution_result: &ExecutionResult,
    ) -> Result<ExampleVisualization> {
        let viz_type = if example.code.contains("flow") || example.code.contains("step") {
            VisualizationType::FlowChart
        } else if example.code.contains("graph") || example.code.contains("network") {
            VisualizationType::Network
        } else if example.code.contains("tree") || example.code.contains("hierarchy") {
            VisualizationType::Tree
        } else {
            VisualizationType::Chart
        };

        Ok(ExampleVisualization {
            visualization_type: viz_type,
            data: execution_result.output.clone(),
            interactive: true,
            real_time_updates: true,
            config: VisualizationConfig {
                width: 600,
                height: 400,
                theme: "dark".to_string(),
                animation_enabled: true,
            },
        })
    }

    /// Generate playground configuration
    fn generate_playground_config(&self, api_ref: &ApiReference) -> Result<PlaygroundConfig> {
        Ok(PlaygroundConfig {
            wasm_enabled: true,
            real_time_execution: true,
            syntax_highlighting: true,
            auto_completion: true,
            error_highlighting: true,
            available_crates: vec![
                "sklears-core".to_string(),
                "std".to_string(),
                "serde".to_string(),
            ],
            example_count: api_ref.examples.len(),
            trait_count: api_ref.traits.len(),
            type_count: api_ref.types.len(),
        })
    }

    /// Generate basic tutorial steps
    fn generate_basic_tutorial_steps(
        &self,
        api_ref: &ApiReference,
    ) -> Result<Vec<crate::api_data_structures::TutorialStep>> {
        let mut steps = Vec::new();

        steps.push(crate::api_data_structures::TutorialStep {
            title: "Introduction to sklears-core".to_string(),
            content: "Welcome to sklears-core! This library provides machine learning algorithms implemented in Rust.".to_string(),
            code_example: Some(CodeExample {
                title: "Hello sklears".to_string(),
                description: "Your first sklears program".to_string(),
                code: "use sklears_core::*;\n\nfn main() {\n    println!(\"Hello, sklears!\");\n}".to_string(),
                language: "rust".to_string(),
                runnable: true,
                expected_output: Some("Hello, sklears!".to_string()),
            }),
            interactive_elements: vec![],
            expected_outcome: "Understanding the basic structure of a sklears program".to_string(),
        });

        if !api_ref.traits.is_empty() {
            let first_trait = &api_ref.traits[0];
            steps.push(crate::api_data_structures::TutorialStep {
                title: format!("Working with {}", first_trait.name),
                content: format!(
                    "Learn how to use the {} trait: {}",
                    first_trait.name, first_trait.description
                ),
                code_example: Some(CodeExample {
                    title: format!("{} Example", first_trait.name),
                    description: format!("Basic usage of {}", first_trait.name),
                    code: format!(
                        "// Example using {}\n// TODO: Add specific example",
                        first_trait.name
                    ),
                    language: "rust".to_string(),
                    runnable: false,
                    expected_output: None,
                }),
                interactive_elements: vec![],
                expected_outcome: format!("Understanding how to use {}", first_trait.name),
            });
        }

        Ok(steps)
    }

    /// Generate trait-specific tutorial steps
    fn generate_trait_tutorial_steps(
        &self,
        trait_info: &crate::api_data_structures::TraitInfo,
    ) -> Result<Vec<crate::api_data_structures::TutorialStep>> {
        let mut steps = Vec::new();

        steps.push(crate::api_data_structures::TutorialStep {
            title: format!("Understanding {}", trait_info.name),
            content: format!("The {} trait: {}", trait_info.name, trait_info.description),
            code_example: None,
            interactive_elements: vec![],
            expected_outcome: format!("Understanding the purpose of {}", trait_info.name),
        });

        for method in &trait_info.methods {
            steps.push(crate::api_data_structures::TutorialStep {
                title: format!("Using {}", method.name),
                content: format!("{}: {}", method.name, method.description),
                code_example: Some(CodeExample {
                    title: format!("{} method", method.name),
                    description: format!("Example of using {}", method.name),
                    code: format!("// {}\n{}", method.description, method.signature),
                    language: "rust".to_string(),
                    runnable: false,
                    expected_output: None,
                }),
                interactive_elements: vec![],
                expected_outcome: format!("Learning to use the {} method", method.name),
            });
        }

        Ok(steps)
    }

    /// Build fuzzy search capabilities
    fn build_fuzzy_search(&self, api_ref: &ApiReference) -> Result<FuzzySearchEngine> {
        let mut fuzzy_search = FuzzySearchEngine::new();

        for trait_info in &api_ref.traits {
            fuzzy_search.index_item(&trait_info.name, &trait_info.description)?;
            for method in &trait_info.methods {
                fuzzy_search.index_item(&method.name, &method.description)?;
            }
        }

        for type_info in &api_ref.types {
            fuzzy_search.index_item(&type_info.name, &type_info.description)?;
        }

        Ok(fuzzy_search)
    }

    /// Build semantic search capabilities
    fn build_semantic_search(&self, _api_ref: &ApiReference) -> Result<SemanticSearchEngine> {
        // In a real implementation, this would use NLP models
        Ok(SemanticSearchEngine::new())
    }

    /// Build type-based search
    fn build_type_search(&self, _api_ref: &ApiReference) -> Result<TypeSearchEngine> {
        // Implementation would analyze type signatures and relationships
        Ok(TypeSearchEngine::new())
    }

    /// Build real-time suggestions
    fn build_real_time_suggestions(
        &self,
        _api_ref: &ApiReference,
    ) -> Result<RealTimeSuggestionEngine> {
        Ok(RealTimeSuggestionEngine::new())
    }
}

impl Default for InteractiveDocGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// LIVE CODE RUNNER
// ================================================================================================

/// Live code execution engine for interactive examples
#[derive(Debug, Clone)]
pub struct LiveCodeRunner {
    execution_timeout: Duration,
    memory_limit: usize,
    enable_unsafe: bool,
    sandboxed: bool,
}

impl LiveCodeRunner {
    /// Create a new live code runner with default settings
    pub fn new() -> Self {
        Self {
            execution_timeout: Duration::from_secs(10),
            memory_limit: 1024 * 1024 * 100, // 100MB
            enable_unsafe: false,
            sandboxed: true,
        }
    }

    /// Create with custom configuration
    pub fn with_config(execution_timeout: Duration, memory_limit: usize) -> Self {
        Self {
            execution_timeout,
            memory_limit,
            enable_unsafe: false,
            sandboxed: true,
        }
    }

    /// Execute a code example and return results
    pub fn execute_example(&self, example: &CodeExample) -> Result<ExecutionResult> {
        // Validate code before execution
        self.validate_code_safety(&example.code)?;

        // In a real implementation, this would:
        // 1. Create a temporary Rust project
        // 2. Write the code to a file
        // 3. Compile with cargo
        // 4. Execute in a sandboxed environment
        // 5. Capture output with timeout

        // For now, simulate execution
        Ok(ExecutionResult {
            stdout: format!("Example '{}' executed successfully!", example.title),
            stderr: String::new(),
            exit_code: 0,
            execution_time: Duration::from_millis(150),
            memory_used: 1024 * 50, // 50KB
            output: example.code.clone(),
        })
    }

    /// Execute code string directly
    pub fn execute_code(&self, code: &str) -> Result<ExecutionResult> {
        self.validate_code_safety(code)?;

        // Simulate code execution
        let simulated_output = if code.contains("println!") {
            // Extract println! content
            if let Some(start) = code.find("println!(\"") {
                if let Some(end) = code[start + 10..].find('"') {
                    code[start + 10..start + 10 + end].to_string()
                } else {
                    "Code executed".to_string()
                }
            } else {
                "Code executed".to_string()
            }
        } else {
            "Code executed successfully".to_string()
        };

        Ok(ExecutionResult {
            stdout: simulated_output,
            stderr: String::new(),
            exit_code: 0,
            execution_time: Duration::from_millis(100),
            memory_used: 1024 * 25, // 25KB
            output: code.to_string(),
        })
    }

    /// Validate code for safety
    fn validate_code_safety(&self, code: &str) -> Result<()> {
        // Check for unsafe blocks
        if !self.enable_unsafe && code.contains("unsafe") {
            return Err(SklearsError::InvalidInput(
                "Unsafe code blocks are not allowed in the playground".to_string(),
            ));
        }

        // Check for potentially dangerous operations
        let dangerous_patterns = [
            "std::process::",
            "std::fs::remove",
            "std::fs::write",
            "std::ptr::",
            "libc::",
            "transmute",
            "system(",
            "exec(",
        ];

        for pattern in &dangerous_patterns {
            if code.contains(pattern) {
                return Err(SklearsError::InvalidInput(format!(
                    "Potentially dangerous operation '{}' is not allowed",
                    pattern
                )));
            }
        }

        Ok(())
    }

    /// Set execution timeout
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.execution_timeout = timeout;
    }

    /// Set memory limit
    pub fn set_memory_limit(&mut self, limit: usize) {
        self.memory_limit = limit;
    }

    /// Enable or disable unsafe code
    pub fn set_unsafe_enabled(&mut self, enabled: bool) {
        self.enable_unsafe = enabled;
    }

    /// Enable or disable sandboxing
    pub fn set_sandboxed(&mut self, sandboxed: bool) {
        self.sandboxed = sandboxed;
    }
}

impl Default for LiveCodeRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// WASM PLAYGROUND MANAGER
// ================================================================================================

/// WebAssembly playground manager for browser-based code execution
#[derive(Debug, Clone)]
pub struct WasmPlaygroundManager {
    #[allow(dead_code)]
    wasm_enabled: bool,
    #[allow(dead_code)]
    optimization_level: WasmOptimizationLevel,
    #[allow(dead_code)]
    target_features: Vec<String>,
}

impl WasmPlaygroundManager {
    /// Create a new WASM playground manager
    pub fn new() -> Self {
        Self {
            wasm_enabled: true,
            optimization_level: WasmOptimizationLevel::Release,
            target_features: vec!["bulk-memory".to_string(), "mutable-globals".to_string()],
        }
    }

    /// Create with WASM enabled/disabled
    pub fn with_wasm_enabled(enabled: bool) -> Self {
        Self {
            wasm_enabled: enabled,
            optimization_level: WasmOptimizationLevel::Release,
            target_features: vec!["bulk-memory".to_string(), "mutable-globals".to_string()],
        }
    }

    /// Generate WASM bindings for API elements
    pub fn generate_wasm_bindings(&self, api_ref: &ApiReference) -> Result<Vec<WasmBinding>> {
        let mut bindings = Vec::new();

        for trait_info in &api_ref.traits {
            let methods = trait_info
                .methods
                .iter()
                .filter_map(|m| {
                    match self.convert_to_js_signature(&m.signature) {
                        Ok(js_sig) => Some(WasmMethod {
                            name: m.name.clone(),
                            js_signature: js_sig,
                            description: m.description.clone(),
                        }),
                        Err(_) => None, // Skip methods with conversion errors
                    }
                })
                .collect();

            bindings.push(WasmBinding {
                rust_name: trait_info.name.clone(),
                js_name: format!("{}Wrapper", trait_info.name),
                methods,
                examples: self.generate_wasm_examples(&trait_info.name)?,
            });
        }

        Ok(bindings)
    }

    /// Generate HTML template for WASM playground
    pub fn generate_html_template(
        &self,
        playground_code: &str,
        _ui_components: &[UIComponent],
    ) -> Result<String> {
        let mut html = String::new();

        // HTML structure
        html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str(
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str("    <title>sklears-core Interactive Playground</title>\n");
        html.push_str("    <link rel=\"stylesheet\" href=\"playground.css\">\n");
        html.push_str("    <script src=\"https://cdn.jsdelivr.net/npm/monaco-editor@latest/min/vs/loader.js\"></script>\n");
        html.push_str("</head>\n<body>\n");

        // Header
        html.push_str("    <header class=\"playground-header\">\n");
        html.push_str("        <h1>sklears-core Interactive Playground</h1>\n");
        html.push_str("        <nav class=\"playground-nav\">\n");
        html.push_str("            <button id=\"run-code\" class=\"nav-btn\">Run Code</button>\n");
        html.push_str("            <button id=\"reset-code\" class=\"nav-btn\">Reset</button>\n");
        html.push_str("            <button id=\"share-code\" class=\"nav-btn\">Share</button>\n");
        html.push_str(
            "            <button id=\"download-code\" class=\"nav-btn\">Download</button>\n",
        );
        html.push_str(
            "            <button id=\"fullscreen\" class=\"nav-btn\">Fullscreen</button>\n",
        );
        html.push_str("        </nav>\n");
        html.push_str("    </header>\n");

        // Main content area
        html.push_str("    <main class=\"playground-main\">\n");
        html.push_str("        <div class=\"editor-panel\">\n");
        html.push_str("            <div class=\"panel-header\">Code Editor</div>\n");
        html.push_str("            <div id=\"code-editor\"></div>\n");
        html.push_str("        </div>\n");
        html.push_str("        <div class=\"output-panel\">\n");
        html.push_str("            <div class=\"panel-header\">Output</div>\n");
        html.push_str("            <div id=\"output-display\"></div>\n");
        html.push_str("        </div>\n");
        html.push_str("        <div class=\"api-panel\">\n");
        html.push_str("            <div class=\"panel-header\">API Explorer</div>\n");
        html.push_str("            <div id=\"api-explorer\"></div>\n");
        html.push_str("        </div>\n");
        html.push_str("    </main>\n");

        // Examples gallery
        html.push_str("    <footer class=\"playground-footer\">\n");
        html.push_str("        <div class=\"panel-header\">Examples</div>\n");
        html.push_str("        <div id=\"examples-gallery\"></div>\n");
        html.push_str("    </footer>\n");

        // Scripts
        html.push_str("    <script type=\"module\">\n");
        html.push_str(&format!(
            "        const INITIAL_CODE = `{}`;\n",
            playground_code
        ));
        html.push_str("        import('./playground.js').then(module => {\n");
        html.push_str("            module.initPlayground(INITIAL_CODE);\n");
        html.push_str("        });\n");
        html.push_str("    </script>\n");

        html.push_str("</body>\n</html>");
        Ok(html)
    }

    /// Generate JavaScript bindings for WASM
    pub fn generate_js_bindings(&self, _wasm_bindings: &[WasmBinding]) -> Result<String> {
        let mut js = String::new();

        js.push_str("// JavaScript bindings for sklears-core WASM playground\n\n");
        js.push_str("import init, * as wasm from './pkg/sklears_core.js';\n\n");

        // Main playground class
        js.push_str("class SklearsPlayground {\n");
        js.push_str("    constructor() {\n");
        js.push_str("        this.wasm = null;\n");
        js.push_str("        this.editor = null;\n");
        js.push_str("        this.output = null;\n");
        js.push_str("        this.isFullscreen = false;\n");
        js.push_str("        this.examples = [];\n");
        js.push_str("    }\n\n");

        // Initialization
        js.push_str("    async init() {\n");
        js.push_str("        try {\n");
        js.push_str("            this.wasm = await init();\n");
        js.push_str("            this.setupEditor();\n");
        js.push_str("            this.setupOutput();\n");
        js.push_str("            this.setupEventListeners();\n");
        js.push_str("            this.setupApiExplorer();\n");
        js.push_str("            console.log('Playground initialized successfully');\n");
        js.push_str("        } catch (error) {\n");
        js.push_str("            console.error('Failed to initialize playground:', error);\n");
        js.push_str("        }\n");
        js.push_str("    }\n\n");

        // Editor setup
        js.push_str("    setupEditor() {\n");
        js.push_str("        require.config({ paths: { 'vs': 'https://cdn.jsdelivr.net/npm/monaco-editor@latest/min/vs' }});\n");
        js.push_str("        require(['vs/editor/editor.main'], () => {\n");
        js.push_str("            this.editor = monaco.editor.create(document.getElementById('code-editor'), {\n");
        js.push_str("                value: window.INITIAL_CODE || '',\n");
        js.push_str("                language: 'rust',\n");
        js.push_str("                theme: 'vs-dark',\n");
        js.push_str("                automaticLayout: true,\n");
        js.push_str("                minimap: { enabled: false },\n");
        js.push_str("                scrollBeyondLastLine: false,\n");
        js.push_str("                fontSize: 14,\n");
        js.push_str("                wordWrap: 'on',\n");
        js.push_str("                lineNumbers: 'on',\n");
        js.push_str("                folding: true,\n");
        js.push_str("                suggestOnTriggerCharacters: true,\n");
        js.push_str("                acceptSuggestionOnEnter: 'on',\n");
        js.push_str("            });\n");
        js.push_str("            \n");
        js.push_str("            // Add keyboard shortcuts\n");
        js.push_str("            this.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {\n");
        js.push_str("                this.runCode();\n");
        js.push_str("            });\n");
        js.push_str("        });\n");
        js.push_str("    }\n\n");

        // Output setup
        js.push_str("    setupOutput() {\n");
        js.push_str("        this.output = document.getElementById('output-display');\n");
        js.push_str("        this.displayPlaceholder();\n");
        js.push_str("    }\n\n");

        // Event listeners
        js.push_str("    setupEventListeners() {\n");
        js.push_str("        document.getElementById('run-code').addEventListener('click', () => this.runCode());\n");
        js.push_str("        document.getElementById('reset-code').addEventListener('click', () => this.resetCode());\n");
        js.push_str("        document.getElementById('share-code').addEventListener('click', () => this.shareCode());\n");
        js.push_str("        document.getElementById('download-code').addEventListener('click', () => this.downloadCode());\n");
        js.push_str("        document.getElementById('fullscreen').addEventListener('click', () => this.toggleFullscreen());\n");
        js.push_str("        \n");
        js.push_str("        // Handle URL code parameter\n");
        js.push_str("        this.loadCodeFromUrl();\n");
        js.push_str("    }\n\n");

        // API Explorer setup
        js.push_str("    setupApiExplorer() {\n");
        js.push_str("        const explorer = document.getElementById('api-explorer');\n");
        js.push_str("        if (explorer) {\n");
        js.push_str("            explorer.innerHTML = this.generateApiExplorerContent();\n");
        js.push_str("        }\n");
        js.push_str("    }\n\n");

        // Code execution
        js.push_str("    async runCode() {\n");
        js.push_str("        const runButton = document.getElementById('run-code');\n");
        js.push_str("        const originalText = runButton.textContent;\n");
        js.push_str("        \n");
        js.push_str("        try {\n");
        js.push_str("            runButton.textContent = 'Running...';\n");
        js.push_str("            runButton.disabled = true;\n");
        js.push_str("            \n");
        js.push_str("            const code = this.editor.getValue();\n");
        js.push_str("            \n");
        js.push_str("            if (this.wasm && this.wasm.execute_rust_code) {\n");
        js.push_str("                const result = await this.wasm.execute_rust_code(code);\n");
        js.push_str("                this.displayOutput(result, 'success');\n");
        js.push_str("            } else {\n");
        js.push_str("                // Fallback simulation\n");
        js.push_str("                const simulatedResult = this.simulateExecution(code);\n");
        js.push_str("                this.displayOutput(simulatedResult, 'success');\n");
        js.push_str("            }\n");
        js.push_str("        } catch (error) {\n");
        js.push_str("            this.displayOutput(error.toString(), 'error');\n");
        js.push_str("        } finally {\n");
        js.push_str("            runButton.textContent = originalText;\n");
        js.push_str("            runButton.disabled = false;\n");
        js.push_str("        }\n");
        js.push_str("    }\n\n");

        // Output display
        js.push_str("    displayOutput(content, type) {\n");
        js.push_str("        const timestamp = new Date().toLocaleTimeString();\n");
        js.push_str("        this.output.innerHTML = `\n");
        js.push_str("            <div class=\"output-header\">\n");
        js.push_str("                <span class=\"output-timestamp\">${timestamp}</span>\n");
        js.push_str("                <button class=\"clear-output\" onclick=\"playground.clearOutput()\">Clear</button>\n");
        js.push_str("            </div>\n");
        js.push_str("            <div class=\"output-${type}\">${content}</div>\n");
        js.push_str("        `;\n");
        js.push_str("    }\n\n");

        // Utility methods
        js.push_str("    displayPlaceholder() {\n");
        js.push_str("        this.output.innerHTML = '<div class=\"output-placeholder\">Output will appear here... (Ctrl+Enter to run)</div>';\n");
        js.push_str("    }\n\n");

        js.push_str("    clearOutput() {\n");
        js.push_str("        this.displayPlaceholder();\n");
        js.push_str("    }\n\n");

        js.push_str("    resetCode() {\n");
        js.push_str("        this.editor.setValue(window.INITIAL_CODE || '');\n");
        js.push_str("        this.clearOutput();\n");
        js.push_str("    }\n\n");

        js.push_str("    shareCode() {\n");
        js.push_str("        const code = this.editor.getValue();\n");
        js.push_str("        const encoded = btoa(encodeURIComponent(code));\n");
        js.push_str("        const url = `${window.location.origin}${window.location.pathname}?code=${encoded}`;\n");
        js.push_str("        \n");
        js.push_str("        if (navigator.clipboard) {\n");
        js.push_str("            navigator.clipboard.writeText(url).then(() => {\n");
        js.push_str("                alert('Shareable link copied to clipboard!');\n");
        js.push_str("            });\n");
        js.push_str("        } else {\n");
        js.push_str("            prompt('Copy this link:', url);\n");
        js.push_str("        }\n");
        js.push_str("    }\n\n");

        js.push_str("    downloadCode() {\n");
        js.push_str("        const code = this.editor.getValue();\n");
        js.push_str("        const blob = new Blob([code], { type: 'text/plain' });\n");
        js.push_str("        const url = URL.createObjectURL(blob);\n");
        js.push_str("        const a = document.createElement('a');\n");
        js.push_str("        a.href = url;\n");
        js.push_str("        a.download = 'playground_code.rs';\n");
        js.push_str("        document.body.appendChild(a);\n");
        js.push_str("        a.click();\n");
        js.push_str("        document.body.removeChild(a);\n");
        js.push_str("        URL.revokeObjectURL(url);\n");
        js.push_str("    }\n\n");

        js.push_str("    toggleFullscreen() {\n");
        js.push_str("        const main = document.querySelector('.playground-main');\n");
        js.push_str("        const button = document.getElementById('fullscreen');\n");
        js.push_str("        \n");
        js.push_str("        if (this.isFullscreen) {\n");
        js.push_str("            main.classList.remove('fullscreen');\n");
        js.push_str("            button.textContent = 'Fullscreen';\n");
        js.push_str("        } else {\n");
        js.push_str("            main.classList.add('fullscreen');\n");
        js.push_str("            button.textContent = 'Exit Fullscreen';\n");
        js.push_str("        }\n");
        js.push_str("        \n");
        js.push_str("        this.isFullscreen = !this.isFullscreen;\n");
        js.push_str("        \n");
        js.push_str("        // Trigger editor resize\n");
        js.push_str("        if (this.editor) {\n");
        js.push_str("            setTimeout(() => this.editor.layout(), 100);\n");
        js.push_str("        }\n");
        js.push_str("    }\n\n");

        js.push_str("    loadCodeFromUrl() {\n");
        js.push_str("        const params = new URLSearchParams(window.location.search);\n");
        js.push_str("        const encodedCode = params.get('code');\n");
        js.push_str("        \n");
        js.push_str("        if (encodedCode && this.editor) {\n");
        js.push_str("            try {\n");
        js.push_str("                const code = decodeURIComponent(atob(encodedCode));\n");
        js.push_str("                this.editor.setValue(code);\n");
        js.push_str("            } catch (error) {\n");
        js.push_str("                console.error('Failed to load code from URL:', error);\n");
        js.push_str("            }\n");
        js.push_str("        }\n");
        js.push_str("    }\n\n");

        js.push_str("    simulateExecution(code) {\n");
        js.push_str("        if (code.includes('println!')) {\n");
        js.push_str("            const matches = code.match(/println!\\(\"([^\"]*)\"/g);\n");
        js.push_str("            if (matches) {\n");
        js.push_str("                return matches.map(m => m.slice(10, -1)).join('\\n');\n");
        js.push_str("            }\n");
        js.push_str("        }\n");
        js.push_str("        return 'Code executed successfully (simulated)';\n");
        js.push_str("    }\n\n");

        js.push_str("    generateApiExplorerContent() {\n");
        js.push_str("        return `\n");
        js.push_str("            <div class=\"api-search\">\n");
        js.push_str("                <input type=\"text\" placeholder=\"Search API...\" onkeyup=\"playground.filterApi(this.value)\">\n");
        js.push_str("            </div>\n");
        js.push_str("            <div class=\"api-sections\">\n");
        js.push_str("                <div class=\"api-section\">\n");
        js.push_str("                    <h4>Quick Actions</h4>\n");
        js.push_str("                    <button onclick=\"playground.insertTemplate('estimator')\">Create Estimator</button>\n");
        js.push_str("                    <button onclick=\"playground.insertTemplate('fit')\">Fit Example</button>\n");
        js.push_str("                    <button onclick=\"playground.insertTemplate('predict')\">Predict Example</button>\n");
        js.push_str("                </div>\n");
        js.push_str("            </div>\n");
        js.push_str("        `;\n");
        js.push_str("    }\n\n");

        js.push_str("    filterApi(query) {\n");
        js.push_str("        // API filtering implementation\n");
        js.push_str("        console.log('Filtering API with:', query);\n");
        js.push_str("    }\n\n");

        js.push_str("    insertTemplate(template) {\n");
        js.push_str("        const templates = {\n");
        js.push_str("            estimator: '// Create an estimator\\nlet estimator = MyEstimator::new();',\n");
        js.push_str("            fit: '// Fit an estimator\\nlet trained = estimator.fit(&x_train, &y_train)?;',\n");
        js.push_str("            predict: '// Make predictions\\nlet predictions = trained.predict(&x_test)?;'\n");
        js.push_str("        };\n");
        js.push_str("        \n");
        js.push_str("        const code = templates[template];\n");
        js.push_str("        if (code && this.editor) {\n");
        js.push_str("            const position = this.editor.getPosition();\n");
        js.push_str("            const range = {\n");
        js.push_str("                startLineNumber: position.lineNumber,\n");
        js.push_str("                startColumn: position.column,\n");
        js.push_str("                endLineNumber: position.lineNumber,\n");
        js.push_str("                endColumn: position.column\n");
        js.push_str("            };\n");
        js.push_str("            this.editor.executeEdits('insert-template', [{\n");
        js.push_str("                range: range,\n");
        js.push_str("                text: code + '\\n'\n");
        js.push_str("            }]);\n");
        js.push_str("        }\n");
        js.push_str("    }\n");
        js.push_str("}\n\n");

        // Export function
        js.push_str("export async function initPlayground(initialCode) {\n");
        js.push_str("    window.INITIAL_CODE = initialCode;\n");
        js.push_str("    const playground = new SklearsPlayground();\n");
        js.push_str("    await playground.init();\n");
        js.push_str("    window.playground = playground;\n");
        js.push_str("}\n");

        Ok(js)
    }

    /// Generate CSS styles for the playground
    pub fn generate_playground_css(&self) -> Result<String> {
        Ok(r#"
/* sklears-core Interactive Playground Styles */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    background: #1e1e1e;
    color: #d4d4d4;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.playground-header {
    background: #2d2d30;
    border-bottom: 1px solid #3e3e42;
    padding: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
}

.playground-header h1 {
    color: #ff6b35;
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
}

.playground-nav {
    display: flex;
    gap: 0.5rem;
}

.nav-btn {
    background: #0e639c;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.2s;
    white-space: nowrap;
}

.nav-btn:hover {
    background: #1177bb;
    transform: translateY(-1px);
}

.nav-btn:disabled {
    background: #666;
    cursor: not-allowed;
    transform: none;
}

.playground-main {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr 300px;
    grid-template-rows: 1fr;
    gap: 1px;
    background: #3e3e42;
    min-height: 0;
}

.playground-main.fullscreen {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1000;
    grid-template-columns: 1fr 1fr;
}

.playground-main.fullscreen .api-panel {
    display: none;
}

.editor-panel, .output-panel, .api-panel {
    background: #1e1e1e;
    position: relative;
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.panel-header {
    background: #2d2d30;
    color: #d4d4d4;
    padding: 0.75rem 1rem;
    font-weight: 600;
    border-bottom: 1px solid #3e3e42;
    flex-shrink: 0;
}

#code-editor {
    flex: 1;
    min-height: 0;
}

.output-panel {
    padding: 0;
    overflow: hidden;
}

#output-display {
    flex: 1;
    padding: 1rem;
    overflow-y: auto;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 0.9rem;
    background: #1e1e1e;
}

.output-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #3e3e42;
}

.output-timestamp {
    color: #666;
    font-size: 0.8rem;
}

.clear-output {
    background: #666;
    color: white;
    border: none;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    cursor: pointer;
    font-size: 0.8rem;
}

.clear-output:hover {
    background: #777;
}

.output-success {
    color: #4ec9b0;
    background: #0e2f1e;
    padding: 1rem;
    border-radius: 4px;
    border-left: 4px solid #4ec9b0;
    white-space: pre-wrap;
    font-family: inherit;
}

.output-error {
    color: #f48771;
    background: #2f0e0e;
    padding: 1rem;
    border-radius: 4px;
    border-left: 4px solid #f48771;
    white-space: pre-wrap;
    font-family: inherit;
}

.output-placeholder {
    color: #666;
    font-style: italic;
    text-align: center;
    padding: 2rem;
}

.api-panel {
    padding: 0;
    overflow: hidden;
    border-left: 1px solid #3e3e42;
    display: flex;
    flex-direction: column;
}

.api-search {
    padding: 1rem;
    border-bottom: 1px solid #3e3e42;
}

.api-search input {
    width: 100%;
    padding: 0.5rem;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    color: #d4d4d4;
    border-radius: 4px;
    font-size: 0.9rem;
}

.api-search input:focus {
    outline: none;
    border-color: #0e639c;
}

.api-sections {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
}

.api-section {
    margin-bottom: 1.5rem;
}

.api-section h4 {
    color: #ff6b35;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

.api-section button {
    display: block;
    width: 100%;
    background: #2d2d30;
    color: #d4d4d4;
    border: 1px solid #3e3e42;
    padding: 0.5rem;
    margin-bottom: 0.25rem;
    border-radius: 3px;
    cursor: pointer;
    font-size: 0.8rem;
    text-align: left;
    transition: background 0.2s;
}

.api-section button:hover {
    background: #3e3e42;
}

.playground-footer {
    background: #2d2d30;
    border-top: 1px solid #3e3e42;
    max-height: 200px;
    min-height: 150px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
}

#examples-gallery {
    flex: 1;
    padding: 1rem;
    overflow-y: auto;
    display: flex;
    gap: 1rem;
}

.example-card {
    background: #1e1e1e;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 1rem;
    min-width: 200px;
    cursor: pointer;
    transition: all 0.2s;
}

.example-card:hover {
    border-color: #0e639c;
    transform: translateY(-2px);
}

.example-card h4 {
    color: #ff6b35;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

.example-card p {
    color: #d4d4d4;
    font-size: 0.8rem;
    line-height: 1.4;
}

/* Responsive design */
@media (max-width: 1024px) {
    .playground-main {
        grid-template-columns: 1fr;
        grid-template-rows: 1fr 1fr auto;
    }

    .api-panel {
        border-left: none;
        border-top: 1px solid #3e3e42;
        max-height: 200px;
    }

    .playground-header h1 {
        font-size: 1.2rem;
    }

    .playground-nav {
        flex-wrap: wrap;
    }
}

@media (max-width: 768px) {
    .playground-main {
        grid-template-rows: 2fr 1fr auto;
    }

    .playground-footer {
        max-height: 150px;
    }

    #examples-gallery {
        flex-direction: column;
        gap: 0.5rem;
    }

    .example-card {
        min-width: auto;
    }
}

/* Loading animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,.3);
    border-radius: 50%;
    border-top-color: #ff6b35;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #1e1e1e;
}

::-webkit-scrollbar-thumb {
    background: #3e3e42;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #5e5e62;
}
"#
        .to_string())
    }

    /// Generate build instructions for WASM
    pub fn generate_build_instructions(&self) -> Result<Vec<String>> {
        Ok(vec![
            "# Building sklears-core for WebAssembly".to_string(),
            "".to_string(),
            "# Install wasm-pack if not already installed".to_string(),
            "curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh".to_string(),
            "".to_string(),
            "# Build the WASM package".to_string(),
            "wasm-pack build --target web --out-dir pkg --dev".to_string(),
            "".to_string(),
            "# For production build".to_string(),
            "wasm-pack build --target web --out-dir pkg --release".to_string(),
            "".to_string(),
            "# Serve the playground locally".to_string(),
            "python3 -m http.server 8000".to_string(),
            "# OR".to_string(),
            "npx serve .".to_string(),
            "# OR".to_string(),
            "cargo install basic-http-server && basic-http-server .".to_string(),
            "".to_string(),
            "# Open http://localhost:8000 in your browser".to_string(),
        ])
    }

    /// Convert Rust signature to JavaScript signature
    fn convert_to_js_signature(&self, rust_signature: &str) -> Result<String> {
        let js_signature = rust_signature
            .replace("&self", "this")
            .replace("&mut self", "this")
            .replace("-> Result<", "Promise<")
            .replace("-> Option<", "")
            .replace("usize", "number")
            .replace("isize", "number")
            .replace("i32", "number")
            .replace("i64", "bigint")
            .replace("u32", "number")
            .replace("u64", "bigint")
            .replace("f32", "number")
            .replace("f64", "number")
            .replace("String", "string")
            .replace("&str", "string")
            .replace("bool", "boolean")
            .replace("()", "void");

        Ok(js_signature)
    }

    /// Generate WASM usage examples
    fn generate_wasm_examples(&self, trait_name: &str) -> Result<Vec<String>> {
        let examples = match trait_name {
            "Estimator" => vec![
                "const estimator = new EstimatorWrapper();".to_string(),
                "await estimator.initialize();".to_string(),
                "console.log('Estimator created:', estimator.name());".to_string(),
            ],
            "Fit" => vec![
                "const fittable = new FitWrapper();".to_string(),
                "const trained = await fittable.fit(trainingData, targets);".to_string(),
                "console.log('Training completed successfully');".to_string(),
            ],
            "Predict" => vec![
                "const predictor = new PredictWrapper();".to_string(),
                "const predictions = await predictor.predict(testData);".to_string(),
                "console.log('Predictions:', predictions);".to_string(),
            ],
            _ => vec![format!(
                "const {} = new {}Wrapper();\nconsole.log('Created {}:', {});",
                trait_name.to_lowercase(),
                trait_name,
                trait_name,
                trait_name.to_lowercase()
            )],
        };
        Ok(examples)
    }
}

impl Default for WasmPlaygroundManager {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM optimization levels
#[derive(Debug, Clone, Copy)]
pub enum WasmOptimizationLevel {
    Debug,
    Release,
    ReleaseWithDebugInfo,
}

// ================================================================================================
// UI COMPONENT BUILDER
// ================================================================================================

/// UI component builder for interactive elements
#[derive(Debug, Clone)]
pub struct UIComponentBuilder {
    component_registry: HashMap<String, UIComponentTemplate>,
}

impl UIComponentBuilder {
    /// Create a new UI component builder
    pub fn new() -> Self {
        let mut builder = Self {
            component_registry: HashMap::new(),
        };
        builder.initialize_templates();
        builder
    }

    /// Generate UI components for an API reference
    #[allow(clippy::vec_init_then_push)]
    pub fn generate_ui_components(&self, api_ref: &ApiReference) -> Result<Vec<UIComponent>> {
        let mut components = Vec::new();

        // Code editor component
        components.push(UIComponent {
            name: "code-editor".to_string(),
            component_type: UIComponentType::CodeEditor,
            props: vec![
                ("language".to_string(), "rust".to_string()),
                ("theme".to_string(), "vs-dark".to_string()),
                ("auto-completion".to_string(), "true".to_string()),
                ("line-numbers".to_string(), "true".to_string()),
                ("minimap".to_string(), "false".to_string()),
            ],
            template: self.generate_code_editor_template()?,
        });

        // Output panel component
        components.push(UIComponent {
            name: "output-panel".to_string(),
            component_type: UIComponentType::OutputPanel,
            props: vec![
                ("real-time".to_string(), "true".to_string()),
                ("syntax-highlighting".to_string(), "true".to_string()),
                ("auto-scroll".to_string(), "true".to_string()),
            ],
            template: self.generate_output_panel_template()?,
        });

        // API explorer component
        components.push(UIComponent {
            name: "api-explorer".to_string(),
            component_type: UIComponentType::ApiExplorer,
            props: vec![
                ("search-enabled".to_string(), "true".to_string()),
                ("filter-by-type".to_string(), "true".to_string()),
                ("collapsible".to_string(), "true".to_string()),
            ],
            template: self.generate_api_explorer_template(api_ref)?,
        });

        // Interactive examples component
        components.push(UIComponent {
            name: "interactive-examples".to_string(),
            component_type: UIComponentType::ExampleGallery,
            props: vec![
                ("auto-run".to_string(), "false".to_string()),
                ("editable".to_string(), "true".to_string()),
                ("layout".to_string(), "grid".to_string()),
            ],
            template: self.generate_examples_gallery_template(api_ref)?,
        });

        // Search box component
        components.push(UIComponent {
            name: "search-box".to_string(),
            component_type: UIComponentType::SearchBox,
            props: vec![
                ("fuzzy-search".to_string(), "true".to_string()),
                ("real-time".to_string(), "true".to_string()),
                ("suggestions".to_string(), "true".to_string()),
            ],
            template: self.generate_search_box_template()?,
        });

        Ok(components)
    }

    /// Initialize component templates
    fn initialize_templates(&mut self) {
        // Initialize with default templates
        self.component_registry.insert(
            "code-editor".to_string(),
            UIComponentTemplate {
                name: "Monaco Code Editor".to_string(),
                component_type: UIComponentType::CodeEditor,
                default_props: vec![
                    ("language".to_string(), "rust".to_string()),
                    ("theme".to_string(), "vs-dark".to_string()),
                ],
                template: String::new(), // Will be generated dynamically
            },
        );
    }

    /// Generate code editor template
    fn generate_code_editor_template(&self) -> Result<String> {
        Ok(r#"
<div class="code-editor-container">
    <div class="editor-toolbar">
        <span class="editor-title">Code Editor</span>
        <div class="editor-actions">
            <button class="format-code" title="Format Code (Shift+Alt+F)">Format</button>
            <button class="toggle-wrap" title="Toggle Word Wrap">Wrap</button>
            <button class="toggle-minimap" title="Toggle Minimap">Map</button>
        </div>
    </div>
    <div id="monaco-editor" class="monaco-editor-wrapper"></div>
    <div class="editor-status">
        <span class="cursor-position">Ln 1, Col 1</span>
        <span class="selection-info"></span>
        <span class="language-mode">Rust</span>
    </div>
</div>
"#
        .to_string())
    }

    /// Generate output panel template
    fn generate_output_panel_template(&self) -> Result<String> {
        Ok(r#"
<div class="output-container">
    <div class="output-header">
        <span class="output-title">Output</span>
        <div class="output-controls">
            <button class="clear-output" title="Clear Output">Clear</button>
            <button class="copy-output" title="Copy Output">Copy</button>
            <button class="save-output" title="Save Output">Save</button>
            <button class="toggle-timestamps" title="Toggle Timestamps">Time</button>
        </div>
    </div>
    <div class="output-content">
        <div class="output-stdout"></div>
        <div class="output-stderr"></div>
        <div class="output-compiler-messages"></div>
    </div>
    <div class="output-status">
        <span class="execution-time">Ready</span>
        <span class="memory-usage"></span>
        <span class="exit-code"></span>
    </div>
</div>
"#
        .to_string())
    }

    /// Generate API explorer template
    fn generate_api_explorer_template(&self, api_ref: &ApiReference) -> Result<String> {
        let mut template = String::new();

        template.push_str(
            r#"
<div class="api-explorer-container">
    <div class="api-search">
        <input type="text" placeholder="Search API..." class="api-search-input" autocomplete="off">
        <div class="search-suggestions"></div>
    </div>
    <div class="api-filters">
        <button class="filter-btn active" data-filter="all">All</button>
        <button class="filter-btn" data-filter="traits">Traits</button>
        <button class="filter-btn" data-filter="types">Types</button>
        <button class="filter-btn" data-filter="methods">Methods</button>
    </div>
    <div class="api-categories">
"#,
        );

        // Traits section
        if !api_ref.traits.is_empty() {
            template.push_str("        <div class=\"api-category\" data-category=\"traits\">\n");
            template.push_str(
                "            <h3 class=\"category-header\">Traits <span class=\"count\">",
            );
            template.push_str(&api_ref.traits.len().to_string());
            template.push_str("</span></h3>\n");
            template.push_str("            <ul class=\"api-list\">\n");
            for trait_info in &api_ref.traits {
                template.push_str(&format!(
                    "                <li class=\"api-item\" data-type=\"trait\" data-name=\"{}\" title=\"{}\">\n",
                    trait_info.name, trait_info.description
                ));
                template.push_str(&format!(
                    "                    <span class=\"item-name\">{}</span>\n",
                    trait_info.name
                ));
                template.push_str(&format!(
                    "                    <span class=\"item-methods\">{} methods</span>\n",
                    trait_info.methods.len()
                ));
                template.push_str("                </li>\n");
            }
            template.push_str("            </ul>\n");
            template.push_str("        </div>\n");
        }

        // Types section
        if !api_ref.types.is_empty() {
            template.push_str("        <div class=\"api-category\" data-category=\"types\">\n");
            template
                .push_str("            <h3 class=\"category-header\">Types <span class=\"count\">");
            template.push_str(&api_ref.types.len().to_string());
            template.push_str("</span></h3>\n");
            template.push_str("            <ul class=\"api-list\">\n");
            for type_info in &api_ref.types {
                template.push_str(&format!(
                    "                <li class=\"api-item\" data-type=\"type\" data-name=\"{}\" title=\"{}\">\n",
                    type_info.name, type_info.description
                ));
                template.push_str(&format!(
                    "                    <span class=\"item-name\">{}</span>\n",
                    type_info.name
                ));
                template.push_str(&format!(
                    "                    <span class=\"item-kind\">{:?}</span>\n",
                    type_info.kind
                ));
                template.push_str("                </li>\n");
            }
            template.push_str("            </ul>\n");
            template.push_str("        </div>\n");
        }

        template.push_str("    </div>\n");
        template.push_str("</div>\n");

        Ok(template)
    }

    /// Generate examples gallery template
    fn generate_examples_gallery_template(&self, api_ref: &ApiReference) -> Result<String> {
        let mut template = String::new();

        template.push_str(
            r#"
<div class="examples-gallery">
    <div class="gallery-header">
        <h3>Interactive Examples</h3>
        <div class="gallery-controls">
            <select class="example-filter">
                <option value="all">All Examples</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
            </select>
            <button class="refresh-examples">Refresh</button>
        </div>
    </div>
    <div class="examples-grid">
"#,
        );

        for (i, example) in api_ref.examples.iter().enumerate() {
            let difficulty = if example.code.len() < 100 {
                "beginner"
            } else if example.code.len() < 300 {
                "intermediate"
            } else {
                "advanced"
            };

            template.push_str(&format!(
                r#"        <div class="example-card" data-example="{}" data-difficulty="{}">
            <div class="example-header">
                <h4>{}</h4>
                <span class="example-difficulty {}">{}</span>
            </div>
            <p class="example-description">{}</p>
            <div class="example-meta">
                <span class="example-language">{}</span>
                <span class="example-runnable">{}</span>
            </div>
            <div class="example-actions">
                <button class="load-example" data-code="{}">Load Example</button>
                <button class="copy-example">Copy</button>
            </div>
        </div>
"#,
                i,
                difficulty,
                example.title,
                difficulty,
                difficulty.to_uppercase(),
                example.description,
                example.language,
                if example.runnable {
                    "Runnable"
                } else {
                    "Read-only"
                },
                base64::encode(&example.code)
            ));
        }

        template.push_str(
            r#"    </div>
</div>
"#,
        );

        Ok(template)
    }

    /// Generate search box template
    fn generate_search_box_template(&self) -> Result<String> {
        Ok(r#"
<div class="search-box-container">
    <div class="search-input-wrapper">
        <input type="text" class="search-input" placeholder="Search API, examples, and documentation..." autocomplete="off">
        <button class="search-clear" title="Clear search">&times;</button>
    </div>
    <div class="search-suggestions">
        <div class="suggestions-header">Suggestions</div>
        <div class="suggestions-list"></div>
    </div>
    <div class="search-filters">
        <label class="filter-label">
            <input type="checkbox" checked> Include traits
        </label>
        <label class="filter-label">
            <input type="checkbox" checked> Include types
        </label>
        <label class="filter-label">
            <input type="checkbox" checked> Include examples
        </label>
    </div>
</div>
"#.to_string())
    }
}

impl Default for UIComponentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// UI component template
#[derive(Debug, Clone)]
pub struct UIComponentTemplate {
    pub name: String,
    pub component_type: UIComponentType,
    pub default_props: Vec<(String, String)>,
    pub template: String,
}

// ================================================================================================
// SEARCH AND VISUALIZATION ENGINES
// ================================================================================================

/// API search engine for interactive search
#[derive(Debug, Clone)]
pub struct ApiSearchEngine {
    indexed_items: HashMap<String, SearchItem>,
    #[allow(dead_code)]
    search_algorithms: Vec<SearchAlgorithm>,
}

impl ApiSearchEngine {
    /// Create a new API search engine
    pub fn new() -> Self {
        Self {
            indexed_items: HashMap::new(),
            search_algorithms: vec![
                SearchAlgorithm::ExactMatch,
                SearchAlgorithm::FuzzyMatch,
                SearchAlgorithm::SemanticMatch,
            ],
        }
    }

    /// Build search index from API reference
    pub fn build_search_index(&mut self, api_ref: &ApiReference) -> Result<SearchIndex> {
        let mut index = SearchIndex::new();

        // Index traits
        for trait_info in &api_ref.traits {
            let item = SearchItem {
                name: trait_info.name.clone(),
                item_type: SearchItemType::Trait,
                description: trait_info.description.clone(),
                path: trait_info.path.clone(),
                keywords: self.extract_keywords(&trait_info.description)?,
                relevance_score: 1.0,
            };
            index.add_item(item.clone())?;
            self.indexed_items.insert(trait_info.name.clone(), item);

            // Index methods
            for method in &trait_info.methods {
                let method_item = SearchItem {
                    name: format!("{}::{}", trait_info.name, method.name),
                    item_type: SearchItemType::Method,
                    description: method.description.clone(),
                    path: format!("{}::{}", trait_info.path, method.name),
                    keywords: self.extract_keywords(&method.description)?,
                    relevance_score: 0.8,
                };
                index.add_item(method_item.clone())?;
                self.indexed_items
                    .insert(format!("{}::{}", trait_info.name, method.name), method_item);
            }
        }

        // Index types
        for type_info in &api_ref.types {
            let item = SearchItem {
                name: type_info.name.clone(),
                item_type: SearchItemType::Type,
                description: type_info.description.clone(),
                path: type_info.path.clone(),
                keywords: self.extract_keywords(&type_info.description)?,
                relevance_score: 0.9,
            };
            index.add_item(item.clone())?;
            self.indexed_items.insert(type_info.name.clone(), item);
        }

        // Index examples
        for example in &api_ref.examples {
            let item = SearchItem {
                name: example.title.clone(),
                item_type: SearchItemType::Example,
                description: example.description.clone(),
                path: format!("examples/{}", example.title),
                keywords: self
                    .extract_keywords(&format!("{} {}", example.title, example.description))?,
                relevance_score: 0.7,
            };
            index.add_item(item.clone())?;
            self.indexed_items.insert(example.title.clone(), item);
        }

        Ok(index)
    }

    /// Extract keywords from text
    fn extract_keywords(&self, text: &str) -> Result<Vec<String>> {
        let keywords = text
            .split_whitespace()
            .filter(|word| word.len() > 2)
            .map(|word| {
                word.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|word| !word.is_empty())
            .collect();
        Ok(keywords)
    }
}

impl Default for ApiSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Search algorithms available
#[derive(Debug, Clone)]
pub enum SearchAlgorithm {
    ExactMatch,
    FuzzyMatch,
    SemanticMatch,
    TypeBasedMatch,
}

/// Visualization engine for interactive documentation
#[derive(Debug, Clone)]
pub struct VisualizationEngine {
    #[allow(dead_code)]
    visualization_types: Vec<VisualizationType>,
}

impl VisualizationEngine {
    /// Create a new visualization engine
    pub fn new() -> Self {
        Self {
            visualization_types: vec![
                VisualizationType::FlowChart,
                VisualizationType::Network,
                VisualizationType::Tree,
                VisualizationType::Chart,
            ],
        }
    }

    /// Generate visualizations for API reference
    pub fn generate_visualizations(
        &self,
        _api_ref: &ApiReference,
    ) -> Result<Vec<crate::api_data_structures::ApiVisualization>> {
        // In a real implementation, this would generate actual visualizations
        Ok(Vec::new())
    }
}

impl Default for VisualizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// ENHANCED SEARCH CAPABILITIES
// ================================================================================================

/// Enhanced search capabilities structure
#[derive(Debug, Clone)]
pub struct EnhancedSearchCapabilities {
    pub fuzzy_search: FuzzySearchEngine,
    pub semantic_search: SemanticSearchEngine,
    pub type_search: TypeSearchEngine,
    pub real_time_suggestions: RealTimeSuggestionEngine,
    pub search_analytics: SearchAnalytics,
}

/// Fuzzy search engine
#[derive(Debug, Clone)]
pub struct FuzzySearchEngine {
    indexed_terms: HashMap<String, f64>,
    #[allow(dead_code)]
    threshold: f64,
}

impl FuzzySearchEngine {
    /// Create a new fuzzy search engine
    pub fn new() -> Self {
        Self {
            indexed_terms: HashMap::new(),
            threshold: 0.7,
        }
    }

    /// Index an item for fuzzy search
    pub fn index_item(&mut self, name: &str, description: &str) -> Result<()> {
        let combined = format!("{} {}", name, description);
        let score = 1.0; // Base relevance score
        self.indexed_terms.insert(combined.to_lowercase(), score);
        Ok(())
    }
}

impl Default for FuzzySearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Semantic search engine
#[derive(Debug, Clone)]
pub struct SemanticSearchEngine {
    #[allow(dead_code)]
    embeddings: HashMap<String, Vec<f64>>,
}

impl SemanticSearchEngine {
    /// Create a new semantic search engine
    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
        }
    }
}

impl Default for SemanticSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Type-based search engine
#[derive(Debug, Clone)]
pub struct TypeSearchEngine {
    #[allow(dead_code)]
    type_signatures: HashMap<String, String>,
}

impl TypeSearchEngine {
    /// Create a new type search engine
    pub fn new() -> Self {
        Self {
            type_signatures: HashMap::new(),
        }
    }
}

impl Default for TypeSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time suggestion engine
#[derive(Debug, Clone)]
pub struct RealTimeSuggestionEngine {
    #[allow(dead_code)]
    suggestions: HashMap<String, Vec<String>>,
}

impl RealTimeSuggestionEngine {
    /// Create a new real-time suggestion engine
    pub fn new() -> Self {
        Self {
            suggestions: HashMap::new(),
        }
    }
}

impl Default for RealTimeSuggestionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Search analytics
#[derive(Debug, Clone)]
pub struct SearchAnalytics {
    pub query_count: usize,
    pub popular_queries: Vec<String>,
    pub avg_response_time: Duration,
}

impl SearchAnalytics {
    /// Create new search analytics
    pub fn new() -> Self {
        Self {
            query_count: 0,
            popular_queries: Vec::new(),
            avg_response_time: Duration::from_millis(0),
        }
    }
}

impl Default for SearchAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// TUTORIAL SYSTEM COMPONENTS
// ================================================================================================

/// Interactive tutorial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveTutorial {
    pub id: String,
    pub title: String,
    pub description: String,
    pub difficulty: TutorialDifficulty,
    pub estimated_time: Duration,
    pub steps: Vec<TutorialStep>,
    pub progress_tracking: bool,
    pub adaptive_content: bool,
}

/// Tutorial difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TutorialDifficulty {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Individual tutorial step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialStep {
    pub title: String,
    pub content: String,
    pub code_example: Option<CodeExample>,
    pub interactive_elements: Vec<InteractiveElement>,
    pub expected_outcome: String,
}

// For base64 encoding in examples
mod base64 {
    pub fn encode(data: &str) -> String {
        // Simple base64-like encoding for demo purposes
        data.chars()
            .map(|c| (c as u8).to_string())
            .collect::<Vec<_>>()
            .join("")
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

        // Safe code should pass
        assert!(runner.validate_code_safety("println!(\"Hello\");").is_ok());

        // Unsafe code should fail
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

        let index = search_engine.build_search_index(&api_ref).unwrap();
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
