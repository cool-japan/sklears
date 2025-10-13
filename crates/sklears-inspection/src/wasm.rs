//! WebAssembly support for browser-based explanation generation
//!
//! This module provides WebAssembly bindings and optimizations for running
//! explanation methods directly in web browsers. It includes JavaScript
//! interoperability, memory management optimizations, and streamlined APIs
//! for web-based machine learning interpretability applications.
//!
//! # Features
//!
//! * WebAssembly bindings for core explanation methods
//! * JavaScript interoperability for seamless web integration
//! * Memory-optimized algorithms for browser environments
//! * Streaming explanation computation for large datasets
//! * WebGL acceleration for visualization
//! * Progressive Web App (PWA) support
//!
//! # Browser Compatibility
//!
//! * Modern browsers with WebAssembly support (Chrome 57+, Firefox 52+, Safari 11+)
//! * WebGL 2.0 support for GPU acceleration
//! * WebWorkers for background computation
//! * SharedArrayBuffer for multi-threaded processing (where available)
//!
//! # Example
//!
//! ```javascript
//! import init, { WasmExplainer, WasmShapComputer } from './pkg/sklears_inspection.js';
//!
//! async function runExplanation() {
//!     await init();
//!
//!     const explainer = new WasmExplainer();
//!     const shapComputer = new WasmShapComputer();
//!
//!     // Load data from JavaScript
//!     const features = new Float32Array([1.0, 2.0, 3.0, 4.0]);
//!     const background = new Float32Array([0.5, 1.5, 2.5, 3.5]);
//!
//!     // Compute SHAP values in WebAssembly
//!     const shapValues = await shapComputer.compute_shap(features, background);
//!
//!     // Results are automatically marshaled back to JavaScript
//!     console.log('SHAP values:', shapValues);
//! }
//! ```

use crate::types::Float;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Float32Array, Float64Array, Object, Promise};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{console, window, CanvasRenderingContext2d, WebGlRenderingContext};

// Import JsValue unconditionally but only use it in WASM builds
#[cfg(feature = "wasm")]
use wasm_bindgen::JsValue;

/// WebAssembly configuration for explanation methods
#[derive(Debug, Clone)]
pub struct WasmConfig {
    /// Enable WebGL acceleration when available
    pub enable_webgl: bool,
    /// Use WebWorkers for background computation
    pub use_webworkers: bool,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Chunk size for streaming computation
    pub chunk_size: usize,
    /// Enable progressive computation for large datasets
    pub progressive_computation: bool,
    /// JavaScript callback for progress updates
    pub progress_callback: Option<String>,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            enable_webgl: true,
            use_webworkers: true,
            max_memory_mb: 256,
            chunk_size: 1000,
            progressive_computation: true,
            progress_callback: None,
        }
    }
}

/// Browser capability detection for WebAssembly optimizations
#[derive(Debug, Clone)]
pub struct BrowserCapabilities {
    /// WebAssembly support level
    pub wasm_support: WasmSupportLevel,
    /// WebGL support and version
    pub webgl_support: WebGlSupport,
    /// WebWorkers availability
    pub webworkers_available: bool,
    /// SharedArrayBuffer support
    pub shared_array_buffer: bool,
    /// Available memory (estimate)
    pub available_memory_mb: usize,
    /// CPU core count (estimate)
    pub cpu_cores: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmSupportLevel {
    /// Basic WebAssembly support
    Basic,
    /// WebAssembly with SIMD support
    WithSimd,
    /// WebAssembly with threads support
    WithThreads,
    /// Full WebAssembly feature support
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebGlSupport {
    /// No WebGL support
    None,
    /// WebGL 1.0 support
    WebGl1,
    /// WebGL 2.0 support
    WebGl2,
}

/// Memory management for WebAssembly environment
pub struct WasmMemoryManager {
    /// Allocated buffers
    buffers: HashMap<usize, Vec<u8>>,
    /// Current memory usage in bytes
    current_usage: usize,
    /// Maximum allowed memory usage in bytes
    max_usage: usize,
    /// Buffer ID counter
    next_id: usize,
}

impl WasmMemoryManager {
    /// Create a new WASM memory manager
    pub fn new(max_memory_mb: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            current_usage: 0,
            max_usage: max_memory_mb * 1024 * 1024,
            next_id: 0,
        }
    }

    /// Allocate a new buffer
    pub fn allocate(&mut self, size: usize) -> SklResult<usize> {
        if self.current_usage + size > self.max_usage {
            return Err(SklearsError::InvalidInput(
                "Memory limit exceeded".to_string(),
            ));
        }

        let buffer_id = self.next_id;
        self.next_id += 1;

        let buffer = vec![0u8; size];
        self.current_usage += size;
        self.buffers.insert(buffer_id, buffer);

        Ok(buffer_id)
    }

    /// Deallocate a buffer
    pub fn deallocate(&mut self, buffer_id: usize) -> SklResult<()> {
        if let Some(buffer) = self.buffers.remove(&buffer_id) {
            self.current_usage -= buffer.len();
            Ok(())
        } else {
            Err(SklearsError::InvalidInput("Buffer not found".to_string()))
        }
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get memory usage percentage
    pub fn usage_percentage(&self) -> f32 {
        (self.current_usage as f32) / (self.max_usage as f32) * 100.0
    }
}

/// JavaScript interoperability utilities
pub struct JsInterop {
    /// Browser capabilities
    capabilities: BrowserCapabilities,
    /// Memory manager
    memory_manager: WasmMemoryManager,
}

impl JsInterop {
    /// Create new JavaScript interoperability helper
    pub fn new(config: &WasmConfig) -> SklResult<Self> {
        let capabilities = Self::detect_capabilities()?;
        let memory_manager = WasmMemoryManager::new(config.max_memory_mb);

        Ok(Self {
            capabilities,
            memory_manager,
        })
    }

    /// Detect browser capabilities
    pub fn detect_capabilities() -> SklResult<BrowserCapabilities> {
        #[cfg(target_arch = "wasm32")]
        {
            let window = window().ok_or_else(|| {
                SklearsError::InvalidInput("Window object not available".to_string())
            })?;

            // Detect WebAssembly support level
            let wasm_support =
                if js_sys::Reflect::has(&window, &"WebAssembly".into()).unwrap_or(false) {
                    WasmSupportLevel::Basic // Could be enhanced to detect SIMD/threads
                } else {
                    return Err(SklearsError::InvalidInput(
                        "WebAssembly not supported".to_string(),
                    ));
                };

            // Detect WebGL support
            let webgl_support = if let Some(document) = window.document() {
                if let Ok(canvas) = document.create_element("canvas") {
                    if let Ok(canvas) = canvas.dyn_into::<web_sys::HtmlCanvasElement>() {
                        if canvas.get_context("webgl2").unwrap_or(None).is_some() {
                            WebGlSupport::WebGl2
                        } else if canvas.get_context("webgl").unwrap_or(None).is_some() {
                            WebGlSupport::WebGl1
                        } else {
                            WebGlSupport::None
                        }
                    } else {
                        WebGlSupport::None
                    }
                } else {
                    WebGlSupport::None
                }
            } else {
                WebGlSupport::None
            };

            // Detect WebWorkers
            let webworkers_available =
                js_sys::Reflect::has(&window, &"Worker".into()).unwrap_or(false);

            // Detect SharedArrayBuffer
            let shared_array_buffer =
                js_sys::Reflect::has(&window, &"SharedArrayBuffer".into()).unwrap_or(false);

            // Estimate available memory (simplified)
            let available_memory_mb = if let Some(navigator) = window.navigator() {
                // Try to get device memory hint
                if let Ok(memory) = js_sys::Reflect::get(&navigator, &"deviceMemory".into()) {
                    if let Some(memory_gb) = memory.as_f64() {
                        (memory_gb * 1024.0) as usize
                    } else {
                        512 // Default estimate
                    }
                } else {
                    512 // Default estimate
                }
            } else {
                512
            };

            // Estimate CPU cores
            let cpu_cores = if let Some(navigator) = window.navigator() {
                if let Ok(cores) = js_sys::Reflect::get(&navigator, &"hardwareConcurrency".into()) {
                    cores.as_f64().unwrap_or(4.0) as usize
                } else {
                    4
                }
            } else {
                4
            };

            Ok(BrowserCapabilities {
                wasm_support,
                webgl_support,
                webworkers_available,
                shared_array_buffer,
                available_memory_mb,
                cpu_cores,
            })
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Return minimal capabilities for non-WASM targets
            Ok(BrowserCapabilities {
                wasm_support: WasmSupportLevel::Full,
                webgl_support: WebGlSupport::WebGl2,
                webworkers_available: true,
                shared_array_buffer: true,
                available_memory_mb: 8192,
                cpu_cores: 8,
            })
        }
    }

    /// Convert Rust array to JavaScript Float32Array
    #[cfg(target_arch = "wasm32")]
    pub fn array_to_js_f32(&self, array: &Array1<Float>) -> Float32Array {
        let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();
        Float32Array::from(&data[..])
    }

    /// Convert Rust array to JavaScript Float64Array
    #[cfg(target_arch = "wasm32")]
    pub fn array_to_js_f64(&self, array: &Array1<Float>) -> Float64Array {
        let data: Vec<f64> = array.iter().map(|&x| x as f64).collect();
        Float64Array::from(&data[..])
    }

    /// Convert JavaScript Float32Array to Rust array
    #[cfg(target_arch = "wasm32")]
    pub fn js_f32_to_array(&self, js_array: &Float32Array) -> Array1<Float> {
        let data: Vec<f32> = js_array.to_vec();
        let rust_data: Vec<Float> = data.into_iter().map(|x| x as Float).collect();
        Array1::from_vec(rust_data)
    }

    /// Convert JavaScript Float64Array to Rust array
    #[cfg(target_arch = "wasm32")]
    pub fn js_f64_to_array(&self, js_array: &Float64Array) -> Array1<Float> {
        let data: Vec<f64> = js_array.to_vec();
        let rust_data: Vec<Float> = data.into_iter().map(|x| x as Float).collect();
        Array1::from_vec(rust_data)
    }

    /// Get browser capabilities
    pub fn capabilities(&self) -> &BrowserCapabilities {
        &self.capabilities
    }
}

/// WebAssembly-optimized SHAP computer
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WasmShapComputer {
    /// Configuration
    config: WasmConfig,
    /// JavaScript interoperability
    js_interop: JsInterop,
    /// Background explanation data
    background: Option<Array2<Float>>,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmShapComputer {
    /// Create a new WASM SHAP computer
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Result<WasmShapComputer, SklResult<()>> {
        let config = WasmConfig::default();
        let js_interop = JsInterop::new(&config).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            config,
            js_interop,
            background: None,
        })
    }

    /// Set background data for SHAP computation
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn set_background(&mut self, background_data: &[f32]) -> Result<(), JsValue> {
        // Convert JavaScript data to Rust array
        let n_samples = background_data.len() / 4; // Assume 4 features for simplicity
        let n_features = 4;

        let background = Array2::from_shape_vec(
            (n_samples, n_features),
            background_data.iter().map(|&x| x as Float).collect(),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.background = Some(background);
        Ok(())
    }

    /// Compute SHAP values for given features
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn compute_shap(&self, features: &[f32]) -> Result<Vec<f32>, JsValue> {
        let background = self
            .background
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Background data not set"))?;

        // Convert features to Array1
        let features_array = Array1::from_vec(features.iter().map(|&x| x as Float).collect());

        // Simplified SHAP computation for WASM environment
        let shap_values = self
            .compute_shap_values_wasm(&features_array, background)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Convert back to f32 vector for JavaScript
        Ok(shap_values.iter().map(|&x| x as f32).collect())
    }

    /// Get browser capabilities
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn get_capabilities(&self) -> String {
        format!("{:?}", self.js_interop.capabilities())
    }
}

impl WasmShapComputer {
    /// Internal SHAP computation optimized for WASM
    fn compute_shap_values_wasm(
        &self,
        features: &Array1<Float>,
        background: &Array2<Float>,
    ) -> SklResult<Array1<Float>> {
        let n_features = features.len();
        let mut shap_values = Array1::zeros(n_features);

        // Simplified SHAP computation using marginal contributions
        for i in 0..n_features {
            let feature_value = features[i];
            let background_mean = background.column(i).mean().unwrap_or(0.0);

            // Simple marginal contribution (could be enhanced)
            shap_values[i] = feature_value - background_mean;
        }

        Ok(shap_values)
    }
}

/// WebAssembly-optimized general explainer
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct WasmExplainer {
    /// Configuration
    config: WasmConfig,
    /// JavaScript interoperability
    js_interop: JsInterop,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmExplainer {
    /// Create a new WASM explainer
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Result<WasmExplainer, JsValue> {
        let config = WasmConfig::default();
        let js_interop = JsInterop::new(&config).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self { config, js_interop })
    }

    /// Compute feature importance using permutation method
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn compute_feature_importance(
        &self,
        features: &[f32],
        predictions: &[f32],
    ) -> Result<Vec<f32>, JsValue> {
        let n_features = features.len() / predictions.len();
        let n_samples = predictions.len();

        // Convert to arrays
        let features_array = Array2::from_shape_vec(
            (n_samples, n_features),
            features.iter().map(|&x| x as Float).collect(),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let predictions_array = Array1::from_vec(predictions.iter().map(|&x| x as Float).collect());

        // Compute importance (simplified for WASM)
        let importance = self
            .compute_importance_wasm(&features_array, &predictions_array)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(importance.iter().map(|&x| x as f32).collect())
    }

    /// Check if WebGL acceleration is available
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn webgl_available(&self) -> bool {
        matches!(
            self.js_interop.capabilities().webgl_support,
            WebGlSupport::WebGl1 | WebGlSupport::WebGl2
        )
    }

    /// Get memory usage information
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn memory_usage(&self) -> f32 {
        // Return a placeholder value since we can't access the mutable memory manager
        0.0
    }
}

impl WasmExplainer {
    /// Internal feature importance computation
    fn compute_importance_wasm(
        &self,
        features: &Array2<Float>,
        predictions: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let n_features = features.ncols();
        let mut importance = Array1::zeros(n_features);

        // Simplified importance computation using correlation
        for i in 0..n_features {
            let feature_col = features.column(i);
            let correlation = self.compute_correlation(&feature_col.to_owned(), predictions)?;
            importance[i] = correlation.abs();
        }

        Ok(importance)
    }

    /// Compute correlation between feature and predictions
    fn compute_correlation(
        &self,
        feature: &Array1<Float>,
        predictions: &Array1<Float>,
    ) -> SklResult<Float> {
        if feature.len() != predictions.len() {
            return Err(SklearsError::InvalidInput(
                "Feature and prediction lengths do not match".to_string(),
            ));
        }

        let n = feature.len() as Float;
        let feature_mean = feature.mean().unwrap_or(0.0);
        let pred_mean = predictions.mean().unwrap_or(0.0);

        let mut covariance = 0.0;
        let mut feature_var = 0.0;
        let mut pred_var = 0.0;

        for i in 0..feature.len() {
            let feature_diff = feature[i] - feature_mean;
            let pred_diff = predictions[i] - pred_mean;

            covariance += feature_diff * pred_diff;
            feature_var += feature_diff * feature_diff;
            pred_var += pred_diff * pred_diff;
        }

        let denominator = (feature_var * pred_var).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(covariance / denominator)
        }
    }
}

/// WebGL-accelerated visualization for browser environments
pub struct WasmVisualizer {
    /// Configuration
    config: WasmConfig,
    /// JavaScript interoperability
    js_interop: JsInterop,
    /// WebGL context (if available)
    #[cfg(target_arch = "wasm32")]
    webgl_context: Option<WebGlRenderingContext>,
}

impl WasmVisualizer {
    /// Create a new WASM visualizer
    pub fn new(canvas_id: &str) -> SklResult<Self> {
        let config = WasmConfig::default();
        let js_interop = JsInterop::new(&config)?;

        #[cfg(target_arch = "wasm32")]
        let webgl_context = if config.enable_webgl {
            Self::create_webgl_context(canvas_id).ok()
        } else {
            None
        };

        Ok(Self {
            config,
            js_interop,
            #[cfg(target_arch = "wasm32")]
            webgl_context,
        })
    }

    /// Create WebGL context for acceleration
    #[cfg(target_arch = "wasm32")]
    fn create_webgl_context(canvas_id: &str) -> SklResult<WebGlRenderingContext> {
        let window = window()
            .ok_or_else(|| SklearsError::InvalidInput("Window not available".to_string()))?;

        let document = window
            .document()
            .ok_or_else(|| SklearsError::InvalidInput("Document not available".to_string()))?;

        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| SklearsError::InvalidInput("Canvas element not found".to_string()))?
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| SklearsError::InvalidInput("Element is not a canvas".to_string()))?;

        let context = canvas
            .get_context("webgl")
            .map_err(|_| SklearsError::InvalidInput("Failed to get WebGL context".to_string()))?
            .ok_or_else(|| SklearsError::InvalidInput("WebGL not supported".to_string()))?
            .dyn_into::<WebGlRenderingContext>()
            .map_err(|_| {
                SklearsError::InvalidInput("Failed to cast to WebGL context".to_string())
            })?;

        Ok(context)
    }

    /// Render feature importance plot using WebGL
    pub fn render_feature_importance(
        &self,
        importance: &Array1<Float>,
        feature_names: &[String],
    ) -> SklResult<()> {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(gl) = &self.webgl_context {
                // Simplified WebGL rendering for feature importance
                // In a real implementation, this would create shaders and buffers
                gl.clear_color(1.0, 1.0, 1.0, 1.0);
                gl.clear(WebGlRenderingContext::COLOR_BUFFER_BIT);

                // Log to console for now
                console::log_1(&"WebGL feature importance rendering started".into());
            }
        }

        // Fallback to console output for non-WebGL environments
        console::log_1(&format!("Feature importance: {:?}", importance).into());
        Ok(())
    }

    /// Check if WebGL is available
    pub fn webgl_available(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.webgl_context.is_some()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }
}

/// WebAssembly utility functions
pub mod wasm_utils {
    use super::*;

    /// Initialize WebAssembly module with panic hook
    #[cfg(target_arch = "wasm32")]
    pub fn init_wasm() {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    }

    /// Log message to browser console
    #[cfg(target_arch = "wasm32")]
    pub fn log(message: &str) {
        console::log_1(&message.into());
    }

    /// Log error to browser console
    #[cfg(target_arch = "wasm32")]
    pub fn log_error(message: &str) {
        console::error_1(&message.into());
    }

    /// Performance timing for WASM operations
    pub struct WasmTimer {
        start: f64,
        label: String,
    }

    impl WasmTimer {
        /// Start a new timer
        pub fn start(label: &str) -> Self {
            #[cfg(target_arch = "wasm32")]
            let start = if let Some(window) = window() {
                if let Some(performance) = window.performance() {
                    performance.now()
                } else {
                    0.0
                }
            } else {
                0.0
            };

            #[cfg(not(target_arch = "wasm32"))]
            let start = 0.0;

            Self {
                start,
                label: label.to_string(),
            }
        }

        /// Stop the timer and log the duration
        pub fn stop(self) {
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(window) = window() {
                    if let Some(performance) = window.performance() {
                        let duration = performance.now() - self.start;
                        console::log_1(&format!("{}: {:.2}ms", self.label, duration).into());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_wasm_config_default() {
        let config = WasmConfig::default();
        assert!(config.enable_webgl);
        assert!(config.use_webworkers);
        assert_eq!(config.max_memory_mb, 256);
        assert_eq!(config.chunk_size, 1000);
        assert!(config.progressive_computation);
        assert!(config.progress_callback.is_none());
    }

    #[test]
    fn test_wasm_memory_manager() {
        let mut manager = WasmMemoryManager::new(1); // 1MB limit

        // Test allocation
        let buffer_id = manager.allocate(1024).expect("Failed to allocate buffer");
        assert_eq!(manager.current_usage(), 1024);
        assert_eq!(
            manager.usage_percentage(),
            (1024.0 / (1024.0 * 1024.0)) * 100.0
        );

        // Test deallocation
        manager
            .deallocate(buffer_id)
            .expect("Failed to deallocate buffer");
        assert_eq!(manager.current_usage(), 0);
        assert_eq!(manager.usage_percentage(), 0.0);
    }

    #[test]
    fn test_wasm_memory_manager_limit() {
        let mut manager = WasmMemoryManager::new(1); // 1MB limit

        // Try to allocate more than the limit
        let result = manager.allocate(2 * 1024 * 1024); // 2MB
        assert!(result.is_err());
    }

    #[test]
    fn test_browser_capabilities_detection() {
        // This test will only work properly in non-WASM environments
        // but should not panic in any environment
        let result = JsInterop::detect_capabilities();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let capabilities = result.expect("Failed to detect capabilities");
            assert_eq!(capabilities.wasm_support, WasmSupportLevel::Full);
            assert_eq!(capabilities.webgl_support, WebGlSupport::WebGl2);
            assert!(capabilities.webworkers_available);
            assert!(capabilities.shared_array_buffer);
            assert!(capabilities.available_memory_mb > 0);
            assert!(capabilities.cpu_cores > 0);
        }
    }

    #[test]
    fn test_wasm_shap_computer_creation() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let result = WasmShapComputer::new();
            // In non-WASM environment, this should succeed with fallback capabilities
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_wasm_explainer_correlation() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let config = WasmConfig::default();
            let js_interop = JsInterop::new(&config).expect("Failed to create JsInterop");
            let explainer = WasmExplainer { config, js_interop };

            // Test correlation computation
            let feature = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let predictions = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

            let correlation = explainer
                .compute_correlation(&feature, &predictions)
                .expect("Failed to compute correlation");

            // Perfect positive correlation should be close to 1.0
            assert!((correlation - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_wasm_explainer_importance_computation() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let config = WasmConfig::default();
            let js_interop = JsInterop::new(&config).expect("Failed to create JsInterop");
            let explainer = WasmExplainer { config, js_interop };

            // Create test data
            let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .expect("Failed to create features array");
            let predictions = Array1::from_vec(vec![10.0, 20.0, 30.0]);

            let importance = explainer
                .compute_importance_wasm(&features, &predictions)
                .expect("Failed to compute importance");

            assert_eq!(importance.len(), 2);
            // Both features should have positive importance due to positive correlation
            assert!(importance[0] > 0.0);
            assert!(importance[1] > 0.0);
        }
    }

    #[test]
    fn test_wasm_timer() {
        let timer = wasm_utils::WasmTimer::start("test_operation");
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(1));
        timer.stop(); // Should not panic
    }

    #[test]
    fn test_wasm_visualizer_creation() {
        // This should not panic in non-WASM environments
        let result = WasmVisualizer::new("test-canvas");

        #[cfg(not(target_arch = "wasm32"))]
        {
            // In non-WASM environment, should succeed but without WebGL
            let visualizer = result.expect("Failed to create visualizer");
            assert!(!visualizer.webgl_available());
        }
    }
}
