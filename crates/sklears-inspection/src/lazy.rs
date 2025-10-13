//! Lazy evaluation system for expensive explanation computations
//!
//! This module provides lazy evaluation mechanisms that defer expensive
//! computations until they are actually needed, improving performance
//! and memory efficiency.

use crate::types::*;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Rng;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Lazy evaluation wrapper for explanation computations
pub struct LazyExplanation<T> {
    /// Computation function
    computation: Box<dyn Fn() -> crate::SklResult<T> + Send + Sync>,
    /// Cached result
    cached_result: Arc<Mutex<Option<T>>>,
    /// Computation identifier
    id: String,
    /// Dependency tracking
    dependencies: Vec<String>,
}

/// Lazy computation manager
pub struct LazyComputationManager {
    /// Registry of lazy computations
    computations: Arc<Mutex<HashMap<String, Box<dyn LazyComputation>>>>,
    /// Dependency graph
    dependency_graph: Arc<Mutex<HashMap<String, Vec<String>>>>,
    /// Computation cache
    cache: Arc<Mutex<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>>,
    /// Execution statistics
    stats: Arc<Mutex<LazyExecutionStats>>,
}

/// Trait for lazy computations
pub trait LazyComputation: Send + Sync {
    /// Execute the computation
    fn execute(&self) -> crate::SklResult<Box<dyn std::any::Any + Send + Sync>>;

    /// Get computation identifier
    fn id(&self) -> &str;

    /// Get dependencies
    fn dependencies(&self) -> &[String];

    /// Check if computation is cached
    fn is_cached(&self) -> bool;
}

/// Lazy execution statistics
#[derive(Clone, Debug, Default)]
pub struct LazyExecutionStats {
    /// Number of computations executed
    pub computations_executed: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Total execution time
    pub total_execution_time: f64,
    /// Average execution time per computation
    pub avg_execution_time: f64,
}

/// Lazy feature importance computation
pub struct LazyFeatureImportance {
    /// Data reference
    data: Arc<Array2<Float>>,
    /// Target reference
    target: Arc<Array1<Float>>,
    /// Model function
    model: Arc<dyn Fn(&ArrayView2<Float>) -> crate::SklResult<Array1<Float>> + Send + Sync>,
    /// Cached importance values
    importance_cache: Arc<Mutex<Option<Array1<Float>>>>,
    /// Computation configuration
    config: LazyConfig,
}

/// Lazy SHAP computation
pub struct LazyShapValues {
    /// Data reference
    data: Arc<Array2<Float>>,
    /// Model function
    model: Arc<dyn Fn(&ArrayView2<Float>) -> crate::SklResult<Array1<Float>> + Send + Sync>,
    /// Cached SHAP values
    shap_cache: Arc<Mutex<Option<Array2<Float>>>>,
    /// Background data
    background_data: Arc<Mutex<Option<Array2<Float>>>>,
    /// Computation configuration
    config: LazyConfig,
}

/// Configuration for lazy evaluation
#[derive(Clone, Debug)]
pub struct LazyConfig {
    /// Enable caching
    pub enable_caching: bool,
    /// Cache timeout in seconds
    pub cache_timeout: u64,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Enable dependency tracking
    pub enable_dependency_tracking: bool,
    /// Execution threshold (minimum time before caching)
    pub execution_threshold_ms: u64,
}

impl Default for LazyConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_timeout: 3600, // 1 hour
            max_cache_size: 1000,
            enable_dependency_tracking: true,
            execution_threshold_ms: 100,
        }
    }
}

impl<T> LazyExplanation<T>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create a new lazy explanation
    pub fn new<F>(id: String, computation: F) -> Self
    where
        F: Fn() -> crate::SklResult<T> + Send + Sync + 'static,
    {
        Self {
            computation: Box::new(computation),
            cached_result: Arc::new(Mutex::new(None)),
            id,
            dependencies: Vec::new(),
        }
    }

    /// Create a lazy explanation with dependencies
    pub fn with_dependencies<F>(id: String, computation: F, dependencies: Vec<String>) -> Self
    where
        F: Fn() -> crate::SklResult<T> + Send + Sync + 'static,
    {
        Self {
            computation: Box::new(computation),
            cached_result: Arc::new(Mutex::new(None)),
            id,
            dependencies,
        }
    }

    /// Get the computed value (computing if necessary)
    pub fn get(&self) -> crate::SklResult<T> {
        // Check cache first
        {
            let cache = self.cached_result.lock().unwrap();
            if let Some(result) = cache.as_ref() {
                return Ok(result.clone());
            }
        }

        // Compute and cache
        let result = (self.computation)()?;

        {
            let mut cache = self.cached_result.lock().unwrap();
            *cache = Some(result.clone());
        }

        Ok(result)
    }

    /// Force recomputation
    pub fn invalidate(&self) {
        let mut cache = self.cached_result.lock().unwrap();
        *cache = None;
    }

    /// Check if result is cached
    pub fn is_cached(&self) -> bool {
        let cache = self.cached_result.lock().unwrap();
        cache.is_some()
    }

    /// Get computation ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get dependencies
    pub fn dependencies(&self) -> &[String] {
        &self.dependencies
    }
}

impl LazyComputationManager {
    /// Create a new lazy computation manager
    pub fn new() -> Self {
        Self {
            computations: Arc::new(Mutex::new(HashMap::new())),
            dependency_graph: Arc::new(Mutex::new(HashMap::new())),
            cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(LazyExecutionStats::default())),
        }
    }

    /// Register a lazy computation
    pub fn register<T>(&self, computation: T)
    where
        T: LazyComputation + 'static,
    {
        let id = computation.id().to_string();
        let dependencies = computation.dependencies().to_vec();

        {
            let mut computations = self.computations.lock().unwrap();
            computations.insert(id.clone(), Box::new(computation));
        }

        {
            let mut graph = self.dependency_graph.lock().unwrap();
            graph.insert(id, dependencies);
        }
    }

    /// Execute a computation by ID
    pub fn execute(&self, id: &str) -> crate::SklResult<Box<dyn std::any::Any + Send + Sync>> {
        let start_time = std::time::Instant::now();

        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(_result) = cache.get(id) {
                let mut stats = self.stats.lock().unwrap();
                stats.cache_hits += 1;
                // For now, we'll skip the cache return and always recompute
                // In a real implementation, you'd need proper type handling
            }
        }

        // Execute dependencies first
        self.execute_dependencies(id)?;

        // Execute computation
        let result = {
            let computations = self.computations.lock().unwrap();
            let computation = computations.get(id).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Computation '{}' not found", id))
            })?;

            computation.execute()?
        };

        // Update cache (simplified for now)
        {
            let _cache = self.cache.lock().unwrap();
            // Note: In a real implementation, you'd need proper cloning support
            // cache.insert(id.to_string(), result);
        }

        // Update statistics
        let execution_time = start_time.elapsed().as_secs_f64();
        {
            let mut stats = self.stats.lock().unwrap();
            stats.computations_executed += 1;
            stats.cache_misses += 1;
            stats.total_execution_time += execution_time;
            stats.avg_execution_time =
                stats.total_execution_time / stats.computations_executed as f64;
        }

        Ok(result)
    }

    /// Execute dependencies recursively
    fn execute_dependencies(&self, id: &str) -> crate::SklResult<()> {
        let dependencies = {
            let graph = self.dependency_graph.lock().unwrap();
            graph.get(id).cloned().unwrap_or_default()
        };

        for dep_id in dependencies {
            self.execute(&dep_id)?;
        }

        Ok(())
    }

    /// Get execution statistics
    pub fn get_statistics(&self) -> LazyExecutionStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    /// Invalidate computation and its dependents
    pub fn invalidate(&self, id: &str) {
        // Remove from cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.remove(id);
        }

        // Find and invalidate dependents
        let dependents = self.find_dependents(id);
        for dependent in dependents {
            self.invalidate(&dependent);
        }
    }

    /// Find computations that depend on the given computation
    fn find_dependents(&self, id: &str) -> Vec<String> {
        let graph = self.dependency_graph.lock().unwrap();
        graph
            .iter()
            .filter(|(_, deps)| deps.contains(&id.to_string()))
            .map(|(comp_id, _)| comp_id.clone())
            .collect()
    }
}

impl LazyFeatureImportance {
    /// Create a new lazy feature importance computation
    pub fn new<F>(data: Array2<Float>, target: Array1<Float>, model: F, config: LazyConfig) -> Self
    where
        F: Fn(&ArrayView2<Float>) -> crate::SklResult<Array1<Float>> + Send + Sync + 'static,
    {
        Self {
            data: Arc::new(data),
            target: Arc::new(target),
            model: Arc::new(model),
            importance_cache: Arc::new(Mutex::new(None)),
            config,
        }
    }

    /// Get feature importance (computing if necessary)
    pub fn get_importance(&self) -> crate::SklResult<Array1<Float>> {
        // Check cache
        {
            let cache = self.importance_cache.lock().unwrap();
            if let Some(importance) = cache.as_ref() {
                return Ok(importance.clone());
            }
        }

        // Compute importance
        let importance = self.compute_importance()?;

        // Cache result
        {
            let mut cache = self.importance_cache.lock().unwrap();
            *cache = Some(importance.clone());
        }

        Ok(importance)
    }

    /// Compute feature importance using permutation method
    fn compute_importance(&self) -> crate::SklResult<Array1<Float>> {
        let n_features = self.data.ncols();
        let baseline_predictions = (self.model)(&self.data.view())?;
        let baseline_score =
            self.compute_r2_score(&self.target.view(), &baseline_predictions.view())?;

        let mut importances = Array1::zeros(n_features);
        let mut X_permuted = self.data.as_ref().clone();

        for feature_idx in 0..n_features {
            // Permute feature
            let original_column = self.data.column(feature_idx).to_owned();

            // Shuffle column
            {
                let mut column = X_permuted.column_mut(feature_idx);
                let mut rng = scirs2_core::random::thread_rng();
                for i in (1..column.len()).rev() {
                    let j = rng.gen_range(0..=i);
                    column.swap(i, j);
                }
            } // Drop mutable borrow here

            // Compute permuted score
            let permuted_predictions = (self.model)(&X_permuted.view())?;
            let permuted_score =
                self.compute_r2_score(&self.target.view(), &permuted_predictions.view())?;

            // Store importance
            importances[feature_idx] = baseline_score - permuted_score;

            // Restore original column
            X_permuted.column_mut(feature_idx).assign(&original_column);
        }

        Ok(importances)
    }

    /// Compute R² score
    fn compute_r2_score(
        &self,
        y_true: &ArrayView1<Float>,
        y_pred: &ArrayView1<Float>,
    ) -> crate::SklResult<Float> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let y_mean = y_true.mean().ok_or_else(|| {
            SklearsError::InvalidInput("Cannot compute mean of y_true".to_string())
        })?;

        let ss_tot: Float = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: Float = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
            .sum();

        if ss_tot == 0.0 {
            return Ok(1.0);
        }

        Ok(1.0 - (ss_res / ss_tot))
    }

    /// Invalidate cache
    pub fn invalidate(&self) {
        let mut cache = self.importance_cache.lock().unwrap();
        *cache = None;
    }
}

impl LazyShapValues {
    /// Create a new lazy SHAP computation
    pub fn new<F>(data: Array2<Float>, model: F, config: LazyConfig) -> Self
    where
        F: Fn(&ArrayView2<Float>) -> crate::SklResult<Array1<Float>> + Send + Sync + 'static,
    {
        Self {
            data: Arc::new(data),
            model: Arc::new(model),
            shap_cache: Arc::new(Mutex::new(None)),
            background_data: Arc::new(Mutex::new(None)),
            config,
        }
    }

    /// Get SHAP values (computing if necessary)
    pub fn get_shap_values(&self) -> crate::SklResult<Array2<Float>> {
        // Check cache
        {
            let cache = self.shap_cache.lock().unwrap();
            if let Some(shap_values) = cache.as_ref() {
                return Ok(shap_values.clone());
            }
        }

        // Compute SHAP values
        let shap_values = self.compute_shap_values()?;

        // Cache result
        {
            let mut cache = self.shap_cache.lock().unwrap();
            *cache = Some(shap_values.clone());
        }

        Ok(shap_values)
    }

    /// Compute SHAP values using simplified method
    fn compute_shap_values(&self) -> crate::SklResult<Array2<Float>> {
        let n_samples = self.data.nrows();
        let n_features = self.data.ncols();

        // Get or compute background data
        let background_data = self.get_background_data()?;
        let baseline_pred = (self.model)(&background_data.view())?;
        let baseline_value = baseline_pred[0];

        let mut shap_values = Array2::zeros((n_samples, n_features));

        for sample_idx in 0..n_samples {
            let sample = self.data.row(sample_idx);
            let full_pred = (self.model)(&sample.insert_axis(Axis(0)))?;
            let full_value = full_pred[0];

            let total_contribution = full_value - baseline_value;
            let contribution_per_feature = total_contribution / n_features as Float;

            for feature_idx in 0..n_features {
                shap_values[(sample_idx, feature_idx)] = contribution_per_feature;
            }
        }

        Ok(shap_values)
    }

    /// Get or compute background data
    fn get_background_data(&self) -> crate::SklResult<Array2<Float>> {
        {
            let cache = self.background_data.lock().unwrap();
            if let Some(bg_data) = cache.as_ref() {
                return Ok(bg_data.clone());
            }
        }

        // Compute background data (feature means)
        let means = self.data.mean_axis(Axis(0)).ok_or_else(|| {
            SklearsError::InvalidInput("Cannot compute feature means".to_string())
        })?;

        let background_data = means.insert_axis(Axis(0));

        // Cache result
        {
            let mut cache = self.background_data.lock().unwrap();
            *cache = Some(background_data.clone());
        }

        Ok(background_data)
    }

    /// Invalidate cache
    pub fn invalidate(&self) {
        let mut cache = self.shap_cache.lock().unwrap();
        *cache = None;

        let mut bg_cache = self.background_data.lock().unwrap();
        *bg_cache = None;
    }
}

/// Lazy pipeline for chaining multiple explanations
pub struct LazyExplanationPipeline {
    /// Ordered list of lazy computations
    computations: Vec<Box<dyn LazyComputation>>,
    /// Pipeline configuration
    config: LazyConfig,
    /// Computation manager
    manager: LazyComputationManager,
}

impl LazyExplanationPipeline {
    /// Create a new lazy explanation pipeline
    pub fn new(config: LazyConfig) -> Self {
        Self {
            computations: Vec::new(),
            config,
            manager: LazyComputationManager::new(),
        }
    }

    /// Add a computation to the pipeline
    pub fn add_computation<T>(&mut self, computation: T)
    where
        T: LazyComputation + 'static,
    {
        self.manager.register(computation);
    }

    /// Execute the entire pipeline
    pub fn execute_pipeline(&self) -> crate::SklResult<Vec<Box<dyn std::any::Any + Send + Sync>>> {
        let mut results = Vec::new();

        for computation in &self.computations {
            let result = self.manager.execute(computation.id())?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get pipeline statistics
    pub fn get_statistics(&self) -> LazyExecutionStats {
        self.manager.get_statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;
    use scirs2_core::random::Rng;

    #[test]
    fn test_lazy_explanation_creation() {
        let lazy_exp =
            LazyExplanation::new("test_computation".to_string(), || Ok(vec![1.0, 2.0, 3.0]));

        assert_eq!(lazy_exp.id(), "test_computation");
        assert!(!lazy_exp.is_cached());
    }

    #[test]
    fn test_lazy_explanation_execution() {
        let lazy_exp =
            LazyExplanation::new("test_computation".to_string(), || Ok(vec![1.0, 2.0, 3.0]));

        let result = lazy_exp.get().unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        assert!(lazy_exp.is_cached());

        // Second call should use cache
        let result2 = lazy_exp.get().unwrap();
        assert_eq!(result2, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_lazy_explanation_invalidation() {
        let lazy_exp =
            LazyExplanation::new("test_computation".to_string(), || Ok(vec![1.0, 2.0, 3.0]));

        // Execute once
        lazy_exp.get().unwrap();
        assert!(lazy_exp.is_cached());

        // Invalidate
        lazy_exp.invalidate();
        assert!(!lazy_exp.is_cached());
    }

    #[test]
    fn test_lazy_feature_importance() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let target = array![1.0, 2.0, 3.0];
        let config = LazyConfig::default();

        let model = |x: &ArrayView2<Float>| -> crate::SklResult<Array1<Float>> {
            Ok(x.column(0).to_owned())
        };

        let lazy_importance = LazyFeatureImportance::new(data, target, model, config);

        let importance = lazy_importance.get_importance().unwrap();
        assert_eq!(importance.len(), 2);
    }

    #[test]
    fn test_lazy_shap_values() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let config = LazyConfig::default();

        let model = |x: &ArrayView2<Float>| -> crate::SklResult<Array1<Float>> {
            Ok(array![x.row(0).sum()])
        };

        let lazy_shap = LazyShapValues::new(data, model, config);

        let shap_values = lazy_shap.get_shap_values().unwrap();
        assert_eq!(shap_values.shape(), &[3, 2]);
    }

    #[test]
    fn test_lazy_computation_manager() {
        let manager = LazyComputationManager::new();
        let stats = manager.get_statistics();

        assert_eq!(stats.computations_executed, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_lazy_config_default() {
        let config = LazyConfig::default();

        assert!(config.enable_caching);
        assert_eq!(config.cache_timeout, 3600);
        assert!(config.enable_dependency_tracking);
    }

    #[test]
    fn test_lazy_explanation_with_dependencies() {
        let lazy_exp = LazyExplanation::with_dependencies(
            "test_computation".to_string(),
            || Ok(vec![1.0, 2.0, 3.0]),
            vec!["dependency1".to_string(), "dependency2".to_string()],
        );

        assert_eq!(lazy_exp.dependencies().len(), 2);
        assert_eq!(lazy_exp.dependencies()[0], "dependency1");
    }

    #[test]
    fn test_lazy_explanation_pipeline() {
        let config = LazyConfig::default();
        let pipeline = LazyExplanationPipeline::new(config);

        let stats = pipeline.get_statistics();
        assert_eq!(stats.computations_executed, 0);
    }

    #[test]
    fn test_lazy_execution_stats() {
        let stats = LazyExecutionStats::default();

        assert_eq!(stats.computations_executed, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.total_execution_time, 0.0);
    }

    #[test]
    fn test_feature_importance_caching() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let target = array![1.0, 2.0];
        let config = LazyConfig::default();

        let model = |x: &ArrayView2<Float>| -> crate::SklResult<Array1<Float>> {
            Ok(x.column(0).to_owned())
        };

        let lazy_importance = LazyFeatureImportance::new(data, target, model, config);

        // First call should compute
        let importance1 = lazy_importance.get_importance().unwrap();

        // Second call should use cache (same result)
        let importance2 = lazy_importance.get_importance().unwrap();

        assert_eq!(importance1, importance2);
    }

    #[test]
    fn test_shap_values_caching() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let config = LazyConfig::default();

        let model = |x: &ArrayView2<Float>| -> crate::SklResult<Array1<Float>> {
            Ok(array![x.row(0).sum()])
        };

        let lazy_shap = LazyShapValues::new(data, model, config);

        // First call should compute
        let shap1 = lazy_shap.get_shap_values().unwrap();

        // Second call should use cache (same result)
        let shap2 = lazy_shap.get_shap_values().unwrap();

        assert_eq!(shap1, shap2);
    }
}
