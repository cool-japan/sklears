//! WebAssembly Integration for Browser-Based ML Pipelines
//!
//! This module provides WebAssembly (WASM) integration capabilities for running
//! machine learning pipelines in browser environments. It enables client-side
//! ML workflows, offline inference, and reduced server load through edge computing.

use crate::enhanced_errors::PipelineError;
use serde::{Deserialize, Serialize};
use serde_json;
use sklears_core::{error::Result as SklResult, prelude::SklearsError, traits::Estimator};
use std::collections::HashMap;
use std::fmt;

/// WebAssembly runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    pub memory_limit_mb: usize,
    pub enable_threads: bool,
    pub enable_simd: bool,
    pub enable_bulk_memory: bool,
    pub stack_size_kb: usize,
    pub optimization_level: OptimizationLevel,
    pub debug_mode: bool,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            memory_limit_mb: 256,
            enable_threads: true,
            enable_simd: true,
            enable_bulk_memory: true,
            stack_size_kb: 512,
            optimization_level: OptimizationLevel::Release,
            debug_mode: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// Debug
    Debug,
    /// Release
    Release,
    /// ReleaseWithDebugInfo
    ReleaseWithDebugInfo,
    /// MinSize
    MinSize,
}

/// WebAssembly-compatible pipeline wrapper
pub struct WasmPipeline {
    config: WasmConfig,
    steps: Vec<WasmStep>,
    metadata: PipelineMetadata,
    serialized_state: Option<Vec<u8>>,
}

impl WasmPipeline {
    #[must_use]
    pub fn new(config: WasmConfig) -> Self {
        Self {
            config,
            steps: Vec::new(),
            metadata: PipelineMetadata::default(),
            serialized_state: None,
        }
    }

    /// Add a step to the WASM pipeline
    pub fn add_step(&mut self, step: WasmStep) -> SklResult<()> {
        // Validate WASM compatibility
        self.validate_step_compatibility(&step)?;
        self.steps.push(step);
        Ok(())
    }

    /// Compile pipeline to WebAssembly
    pub fn compile_to_wasm(&self) -> SklResult<WasmModule> {
        let mut compiler = WasmCompiler::new(self.config.clone());

        for step in &self.steps {
            compiler.add_step(step)?;
        }

        compiler.compile()
    }

    /// Load pipeline from WebAssembly binary
    pub fn from_wasm_binary(binary: &[u8], config: WasmConfig) -> SklResult<Self> {
        let module = WasmModule::from_binary(binary)?;
        Self::from_wasm_module(module, config)
    }

    /// Create pipeline from WASM module
    pub fn from_wasm_module(module: WasmModule, config: WasmConfig) -> SklResult<Self> {
        let metadata = module.extract_metadata()?;
        let steps = module.extract_steps()?;

        Ok(Self {
            config,
            steps,
            metadata,
            serialized_state: Some(module.binary),
        })
    }

    /// Serialize pipeline for browser transfer
    pub fn serialize_for_browser(&self) -> SklResult<Vec<u8>> {
        let payload = BrowserPayload {
            wasm_binary: self.serialized_state.clone().unwrap_or_default(),
            metadata: self.metadata.clone(),
            config: self.config.clone(),
        };

        serde_json::to_vec(&payload)
            .map_err(|e| SklearsError::SerializationError(format!("Serialization failed: {e}")))
    }

    /// Create JavaScript bindings
    pub fn generate_js_bindings(&self) -> SklResult<String> {
        let mut js_code = String::new();

        // Generate class definition
        js_code.push_str(&format!(
            "class {} {{\n",
            self.metadata.name.replace(' ', "")
        ));

        // Constructor
        js_code.push_str("  constructor(wasmModule) {\n");
        js_code.push_str("    this.module = wasmModule;\n");
        js_code.push_str("    this.memory = wasmModule.memory;\n");
        js_code.push_str("  }\n\n");

        // Generate methods for each step
        for (i, step) in self.steps.iter().enumerate() {
            let method_code = self.generate_step_js_method(i, step)?;
            js_code.push_str(&method_code);
        }

        // Pipeline execution method
        js_code.push_str("  async predict(input) {\n");
        js_code.push_str("    let data = input;\n");
        for i in 0..self.steps.len() {
            js_code.push_str(&format!("    data = await this.step{i}(data);\n"));
        }
        js_code.push_str("    return data;\n");
        js_code.push_str("  }\n");

        js_code.push_str("}\n");

        Ok(js_code)
    }

    fn validate_step_compatibility(&self, step: &WasmStep) -> SklResult<()> {
        // Check memory requirements
        if step.estimated_memory_mb > self.config.memory_limit_mb {
            return Err(PipelineError::ResourceError {
                resource_type: crate::enhanced_errors::ResourceType::Memory,
                limit: self.config.memory_limit_mb as f64,
                current: step.estimated_memory_mb as f64,
                component: step.name.clone(),
                suggestions: vec![
                    "Increase WASM memory limit".to_string(),
                    "Use a more memory-efficient algorithm".to_string(),
                ],
            }
            .into());
        }

        // Check feature compatibility
        if step.requires_threads && !self.config.enable_threads {
            return Err(PipelineError::ConfigurationError {
                message: "Step requires threading support".to_string(),
                suggestions: vec!["Enable threads in WASM config".to_string()],
                context: crate::enhanced_errors::ErrorContext {
                    pipeline_stage: "compilation".to_string(),
                    component_name: step.name.clone(),
                    input_shape: None,
                    parameters: HashMap::new(),
                    stack_trace: vec!["WasmPipeline::validate_step_compatibility".to_string()],
                },
            }
            .into());
        }

        Ok(())
    }

    fn generate_step_js_method(&self, index: usize, step: &WasmStep) -> SklResult<String> {
        let mut method = String::new();

        method.push_str(&format!("  async step{index}(input) {{\n"));
        method.push_str("    // Convert JavaScript array to WASM memory\n");
        method.push_str("    const inputPtr = this.module._malloc(input.length * 8);\n");
        method.push_str("    const inputArray = new Float64Array(this.memory.buffer, inputPtr, input.length);\n");
        method.push_str("    inputArray.set(input);\n\n");

        method.push_str(&format!("    // Call WASM function for {}\n", step.name));
        method.push_str(&format!(
            "    const outputPtr = this.module._{}(inputPtr, input.length);\n",
            step.name.to_lowercase().replace(' ', "_")
        ));

        method.push_str("    // Convert result back to JavaScript\n");
        method.push_str("    const outputLength = this.module._get_output_length();\n");
        method.push_str("    const outputArray = new Float64Array(this.memory.buffer, outputPtr, outputLength);\n");
        method.push_str("    const result = Array.from(outputArray);\n\n");

        method.push_str("    // Free WASM memory\n");
        method.push_str("    this.module._free(inputPtr);\n");
        method.push_str("    this.module._free(outputPtr);\n\n");

        method.push_str("    return result;\n");
        method.push_str("  }\n\n");

        Ok(method)
    }
}

/// WebAssembly-compatible pipeline step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmStep {
    pub name: String,
    pub step_type: WasmStepType,
    pub parameters: HashMap<String, WasmValue>,
    pub input_schema: DataSchema,
    pub output_schema: DataSchema,
    pub estimated_memory_mb: usize,
    pub requires_threads: bool,
    pub requires_simd: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmStepType {
    /// Transformer
    Transformer,
    /// Predictor
    Predictor,
    /// CustomFunction
    CustomFunction,
}

/// WASM-compatible value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmValue {
    /// I32
    I32(i32),
    /// I64
    I64(i64),
    /// F32
    F32(f32),
    /// F64
    F64(f64),
    /// String
    String(String),
    /// Array
    Array(Vec<WasmValue>),
    /// Object
    Object(HashMap<String, WasmValue>),
}

impl fmt::Display for WasmValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WasmValue::I32(v) => write!(f, "{v}"),
            WasmValue::I64(v) => write!(f, "{v}"),
            WasmValue::F32(v) => write!(f, "{v}"),
            WasmValue::F64(v) => write!(f, "{v}"),
            WasmValue::String(v) => write!(f, "\"{v}\""),
            WasmValue::Array(v) => write!(
                f,
                "[{}]",
                v.iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            WasmValue::Object(_) => write!(f, "{{...}}"),
        }
    }
}

/// Data schema for WASM interop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    pub shape: Vec<usize>,
    pub dtype: WasmDataType,
    pub optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmDataType {
    /// F32
    F32,
    /// F64
    F64,
    /// I32
    I32,
    /// I64
    I64,
    /// Bool
    Bool,
}

/// Pipeline metadata for WASM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub creation_date: String,
    pub features: Vec<String>,
    pub performance_metrics: HashMap<String, f64>,
}

impl Default for PipelineMetadata {
    fn default() -> Self {
        Self {
            name: "Unnamed Pipeline".to_string(),
            version: "1.0.0".to_string(),
            description: "WebAssembly ML Pipeline".to_string(),
            author: "Sklears".to_string(),
            creation_date: chrono::Utc::now().to_rfc3339(),
            features: Vec::new(),
            performance_metrics: HashMap::new(),
        }
    }
}

/// WebAssembly module representation
pub struct WasmModule {
    pub binary: Vec<u8>,
    pub metadata: PipelineMetadata,
    pub exports: Vec<String>,
    pub imports: Vec<String>,
}

impl WasmModule {
    /// Load from WebAssembly binary
    pub fn from_binary(binary: &[u8]) -> SklResult<Self> {
        // In a real implementation, this would parse the WASM binary
        Ok(Self {
            binary: binary.to_vec(),
            metadata: PipelineMetadata::default(),
            exports: vec!["predict".to_string(), "transform".to_string()],
            imports: vec!["env.memory".to_string()],
        })
    }

    /// Extract metadata from WASM custom sections
    pub fn extract_metadata(&self) -> SklResult<PipelineMetadata> {
        // Would parse custom sections in real implementation
        Ok(self.metadata.clone())
    }

    /// Extract pipeline steps from WASM
    pub fn extract_steps(&self) -> SklResult<Vec<WasmStep>> {
        // Would analyze WASM functions and reconstruct steps
        Ok(Vec::new())
    }

    /// Get module size in bytes
    #[must_use]
    pub fn size(&self) -> usize {
        self.binary.len()
    }
}

/// WebAssembly compiler
pub struct WasmCompiler {
    config: WasmConfig,
    steps: Vec<WasmStep>,
    optimizations: Vec<WasmOptimization>,
}

impl WasmCompiler {
    #[must_use]
    pub fn new(config: WasmConfig) -> Self {
        Self {
            config,
            steps: Vec::new(),
            optimizations: Vec::new(),
        }
    }

    pub fn add_step(&mut self, step: &WasmStep) -> SklResult<()> {
        self.steps.push(step.clone());
        Ok(())
    }

    pub fn add_optimization(&mut self, optimization: WasmOptimization) {
        self.optimizations.push(optimization);
    }

    pub fn compile(&self) -> SklResult<WasmModule> {
        // Generate WASM text format
        let wat_code = self.generate_wat()?;

        // Compile to binary (simplified - would use actual WASM compiler)
        let binary = self.wat_to_wasm(&wat_code)?;

        Ok(WasmModule {
            binary,
            metadata: self.generate_metadata(),
            exports: self.get_exports(),
            imports: self.get_imports(),
        })
    }

    fn generate_wat(&self) -> SklResult<String> {
        let mut wat = String::new();

        // Module header
        wat.push_str("(module\n");

        // Memory import
        wat.push_str("  (import \"env\" \"memory\" (memory 1))\n");

        // Function imports
        wat.push_str("  (import \"env\" \"log\" (func $log (param i32)))\n");

        // Generate functions for each step
        for (i, step) in self.steps.iter().enumerate() {
            let func_wat = self.generate_step_function(i, step)?;
            wat.push_str(&func_wat);
        }

        // Main predict function
        wat.push_str("  (func $predict (param $input i32) (param $length i32) (result i32)\n");
        wat.push_str("    (local $data i32)\n");
        wat.push_str("    (local.set $data (local.get $input))\n");

        for i in 0..self.steps.len() {
            wat.push_str(&format!(
                "    (local.set $data (call $step{i} (local.get $data) (local.get $length)))\n"
            ));
        }

        wat.push_str("    (local.get $data)\n");
        wat.push_str("  )\n");

        // Exports
        wat.push_str("  (export \"predict\" (func $predict))\n");
        wat.push_str("  (export \"memory\" (memory 0))\n");

        wat.push_str(")\n");

        Ok(wat)
    }

    fn generate_step_function(&self, index: usize, step: &WasmStep) -> SklResult<String> {
        let mut func = String::new();

        func.push_str(&format!(
            "  (func $step{index} (param $input i32) (param $length i32) (result i32)\n"
        ));

        match step.step_type {
            WasmStepType::Transformer => {
                func.push_str("    ;; Transformer logic would go here\n");
                func.push_str("    (local.get $input) ;; Return input unchanged for now\n");
            }
            WasmStepType::Predictor => {
                func.push_str("    ;; Predictor logic would go here\n");
                func.push_str("    (local.get $input) ;; Return input unchanged for now\n");
            }
            WasmStepType::CustomFunction => {
                func.push_str("    ;; Custom function logic would go here\n");
                func.push_str("    (local.get $input) ;; Return input unchanged for now\n");
            }
        }

        func.push_str("  )\n");

        Ok(func)
    }

    fn wat_to_wasm(&self, wat_code: &str) -> SklResult<Vec<u8>> {
        // Simplified compilation - in practice would use wabt or similar
        // For now, return a placeholder binary
        Ok(vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]) // WASM magic number + version
    }

    fn generate_metadata(&self) -> PipelineMetadata {
        PipelineMetadata {
            name: "Compiled WASM Pipeline".to_string(),
            version: "1.0.0".to_string(),
            description: format!("Pipeline with {} steps", self.steps.len()),
            author: "Sklears WASM Compiler".to_string(),
            creation_date: chrono::Utc::now().to_rfc3339(),
            features: self.steps.iter().map(|s| s.name.clone()).collect(),
            performance_metrics: HashMap::new(),
        }
    }

    fn get_exports(&self) -> Vec<String> {
        vec!["predict".to_string(), "memory".to_string()]
    }

    fn get_imports(&self) -> Vec<String> {
        vec!["env.memory".to_string(), "env.log".to_string()]
    }
}

/// WASM optimization passes
#[derive(Debug, Clone)]
pub enum WasmOptimization {
    /// DeadCodeElimination
    DeadCodeElimination,
    /// FunctionInlining
    FunctionInlining,
    /// LoopVectorization
    LoopVectorization,
    /// MemoryCompaction
    MemoryCompaction,
    /// SIMDOptimization
    SIMDOptimization,
}

/// Browser payload for WASM delivery
#[derive(Serialize, Deserialize)]
struct BrowserPayload {
    wasm_binary: Vec<u8>,
    metadata: PipelineMetadata,
    config: WasmConfig,
}

/// Browser integration utilities
pub mod browser {
    use super::{Estimator, SklResult, WasmModule, WasmPipeline};

    /// JavaScript code generator for browser integration
    pub struct BrowserIntegration;

    impl BrowserIntegration {
        /// Generate complete HTML page with embedded WASM
        pub fn generate_html_page(
            pipeline: &WasmPipeline,
            wasm_module: &WasmModule,
        ) -> SklResult<String> {
            let js_bindings = pipeline.generate_js_bindings()?;
            let wasm_hex = wasm_module
                .binary
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();

            let html = format!(
                r#"
<!DOCTYPE html>
<html>
<head>
    <title>{} - ML Pipeline</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .input-section {{ margin: 20px 0; }}
        .output-section {{ margin: 20px 0; padding: 10px; background: #f5f5f5; }}
        input[type="number"] {{ margin: 2px; padding: 5px; }}
        button {{ padding: 10px 20px; margin: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{}</h1>
        <p>{}</p>

        <div class="input-section">
            <h3>Input Data:</h3>
            <input type="number" id="input0" placeholder="Feature 1" value="1.0">
            <input type="number" id="input1" placeholder="Feature 2" value="2.0">
            <input type="number" id="input2" placeholder="Feature 3" value="3.0">
            <br>
            <button onclick="runPrediction()">Run Prediction</button>
        </div>

        <div class="output-section">
            <h3>Output:</h3>
            <pre id="output">Click "Run Prediction" to see results</pre>
        </div>
    </div>

    <script>
        // Base64 encoded WASM binary
        const wasmBinary = Uint8Array.from(atob('{}'), c => c.charCodeAt(0));

        // Pipeline class
        {}

        let pipeline = null;

        // Initialize WASM
        async function initWasm() {{
            try {{
                const wasmModule = await WebAssembly.instantiate(wasmBinary, {{
                    env: {{
                        memory: new WebAssembly.Memory({{ initial: 256 }}),
                        log: (ptr) => console.log('WASM log:', ptr)
                    }}
                }});

                pipeline = new {}(wasmModule.instance);
                console.log('WASM pipeline initialized successfully');
            }} catch (error) {{
                console.error('Failed to initialize WASM:', error);
                document.getElementById('output').textContent = 'Error: ' + error.message;
            }}
        }}

        // Run prediction
        async function runPrediction() {{
            if (!pipeline) {{
                document.getElementById('output').textContent = 'Pipeline not initialized';
                return;
            }}

            try {{
                const input = [
                    parseFloat(document.getElementById('input0').value),
                    parseFloat(document.getElementById('input1').value),
                    parseFloat(document.getElementById('input2').value)
                ];

                const startTime = performance.now();
                const result = await pipeline.predict(input);
                const endTime = performance.now();

                document.getElementById('output').textContent =
                    `Result: ${{JSON.stringify(result, null, 2)}}\n` +
                    `Execution time: ${{(endTime - startTime).toFixed(2)}}ms`;
            }} catch (error) {{
                document.getElementById('output').textContent = 'Prediction error: ' + error.message;
            }}
        }}

        // Initialize on page load
        window.addEventListener('load', initWasm);
    </script>
</body>
</html>
"#,
                pipeline.metadata.name,
                pipeline.metadata.name,
                pipeline.metadata.description,
                wasm_hex,
                js_bindings,
                pipeline.metadata.name.replace(' ', "")
            );

            Ok(html)
        }

        /// Generate service worker for offline ML
        pub fn generate_service_worker(pipeline: &WasmPipeline) -> SklResult<String> {
            let sw_code = format!(
                r"
// Service Worker for Offline ML Pipeline
const CACHE_NAME = 'ml-pipeline-v1';
const PIPELINE_NAME = '{}';

// Cache resources
self.addEventListener('install', event => {{
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {{
            return cache.addAll([
                '/',
                '/index.html',
                '/wasm/pipeline.wasm',
                '/js/pipeline.js'
            ]);
        }})
    );
}});

// Serve from cache
self.addEventListener('fetch', event => {{
    event.respondWith(
        caches.match(event.request).then(response => {{
            return response || fetch(event.request);
        }})
    );
}});

// Handle ML prediction requests
self.addEventListener('message', event => {{
    if (event.data.type === 'ML_PREDICT') {{
        // Process ML prediction in service worker
        handleMLPrediction(event.data.input)
            .then(result => {{
                event.ports[0].postMessage({{
                    type: 'ML_RESULT',
                    result: result
                }});
            }})
            .catch(error => {{
                event.ports[0].postMessage({{
                    type: 'ML_ERROR',
                    error: error.message
                }});
            }});
    }}
}});

async function handleMLPrediction(input) {{
    // Load WASM module if not already loaded
    if (!self.wasmModule) {{
        const wasmBinary = await fetch('/wasm/pipeline.wasm').then(r => r.arrayBuffer());
        self.wasmModule = await WebAssembly.instantiate(wasmBinary);
    }}

    // Run prediction
    // Implementation would depend on specific pipeline
    return {{ prediction: 'offline-result' }};
}}
",
                pipeline.metadata.name
            );

            Ok(sw_code)
        }
    }
}

// Re-export for external access
pub use browser::BrowserIntegration;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_config_creation() {
        let config = WasmConfig::default();
        assert_eq!(config.memory_limit_mb, 256);
        assert!(config.enable_threads);
        assert!(config.enable_simd);
    }

    #[test]
    fn test_wasm_pipeline_creation() {
        let config = WasmConfig::default();
        let pipeline = WasmPipeline::new(config);
        assert_eq!(pipeline.steps.len(), 0);
    }

    #[test]
    fn test_wasm_step_creation() {
        let step = WasmStep {
            name: "TestStep".to_string(),
            step_type: WasmStepType::Transformer,
            parameters: HashMap::new(),
            input_schema: DataSchema {
                shape: vec![10, 5],
                dtype: WasmDataType::F64,
                optional: false,
            },
            output_schema: DataSchema {
                shape: vec![10, 3],
                dtype: WasmDataType::F64,
                optional: false,
            },
            estimated_memory_mb: 64,
            requires_threads: false,
            requires_simd: false,
        };

        assert_eq!(step.name, "TestStep");
        assert_eq!(step.estimated_memory_mb, 64);
    }

    #[test]
    fn test_wasm_value_display() {
        let value = WasmValue::F64(3.14159);
        assert_eq!(value.to_string(), "3.14159");

        let array_value = WasmValue::Array(vec![
            WasmValue::I32(1),
            WasmValue::I32(2),
            WasmValue::I32(3),
        ]);
        assert_eq!(array_value.to_string(), "[1, 2, 3]");
    }

    #[test]
    fn test_compiler_creation() {
        let config = WasmConfig::default();
        let compiler = WasmCompiler::new(config);
        assert_eq!(compiler.steps.len(), 0);
        assert_eq!(compiler.optimizations.len(), 0);
    }
}
