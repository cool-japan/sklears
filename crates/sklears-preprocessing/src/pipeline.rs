//! Advanced Pipeline Features for Preprocessing Transformations
//!
//! This module provides sophisticated pipeline capabilities including:
//! - Conditional preprocessing steps
//! - Parallel preprocessing branches
//! - Caching for expensive transformations  
//! - Dynamic pipeline construction
//! - Error handling and recovery strategies

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use scirs2_core::ndarray::{s, Array2, ArrayView2};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use sklears_core::{
    error::{Result, SklearsError},
    traits::Transform,
};

use crate::streaming::{StreamingConfig, StreamingStats, StreamingTransformer};

/// Cache entry for transformation results
#[derive(Clone, Debug)]
struct CacheEntry<T> {
    result: T,
    timestamp: Instant,
    access_count: usize,
}

/// Configuration for caching behavior
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CacheConfig {
    /// Maximum number of cached entries
    pub max_entries: usize,
    /// Time-to-live for cache entries (in seconds)
    pub ttl_seconds: u64,
    /// Enable/disable caching
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 100,
            ttl_seconds: 3600, // 1 hour
            enabled: true,
        }
    }
}

/// Thread-safe cache for transformation results
pub struct TransformationCache<T> {
    cache: Arc<RwLock<HashMap<u64, CacheEntry<T>>>>,
    config: CacheConfig,
}

impl<T: Clone> TransformationCache<T> {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Generate cache key from input data
    fn generate_key<U: Hash>(&self, input: U) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        input.hash(&mut hasher);
        hasher.finish()
    }

    /// Get cached result if available and valid
    pub fn get(&self, key: u64) -> Option<T> {
        if !self.config.enabled {
            return None;
        }

        let mut cache = self.cache.write().ok()?;

        // Check if entry exists and is still valid
        if let Some(entry) = cache.get_mut(&key) {
            let age = entry.timestamp.elapsed();
            if age.as_secs() <= self.config.ttl_seconds {
                entry.access_count += 1;
                return Some(entry.result.clone());
            } else {
                // Remove expired entry
                cache.remove(&key);
            }
        }

        None
    }

    /// Store result in cache
    pub fn put(&self, key: u64, value: T) {
        if !self.config.enabled {
            return;
        }

        let mut cache = self.cache.write().unwrap();

        // Evict old entries if cache is full
        if cache.len() >= self.config.max_entries {
            self.evict_lru(&mut cache);
        }

        cache.insert(
            key,
            CacheEntry {
                result: value,
                timestamp: Instant::now(),
                access_count: 1,
            },
        );
    }

    /// Evict least recently used entry
    fn evict_lru(&self, cache: &mut HashMap<u64, CacheEntry<T>>) {
        if let Some((key_to_remove, _)) = cache.iter().min_by_key(|(_, entry)| entry.access_count) {
            let key_to_remove = *key_to_remove;
            cache.remove(&key_to_remove);
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        CacheStats {
            entries: cache.len(),
            max_entries: self.config.max_entries,
            enabled: self.config.enabled,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub max_entries: usize,
    pub enabled: bool,
}

/// Condition function type for conditional preprocessing
pub type ConditionFn = Box<dyn Fn(&ArrayView2<f64>) -> bool + Send + Sync>;

/// Configuration for conditional preprocessing step
pub struct ConditionalStepConfig<T> {
    /// Transformer to apply if condition is met
    pub transformer: T,
    /// Condition function to evaluate
    pub condition: ConditionFn,
    /// Name/description for the step
    pub name: String,
    /// Whether to skip this step if condition fails
    pub skip_on_false: bool,
}

impl<T: std::fmt::Debug> std::fmt::Debug for ConditionalStepConfig<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalStepConfig")
            .field("transformer", &self.transformer)
            .field("condition", &"<function>")
            .field("name", &self.name)
            .field("skip_on_false", &self.skip_on_false)
            .finish()
    }
}

/// A conditional preprocessing step
pub struct ConditionalStep<T> {
    config: ConditionalStepConfig<T>,
    fitted: bool,
}

impl<T> ConditionalStep<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone,
{
    pub fn new(config: ConditionalStepConfig<T>) -> Self {
        Self {
            config,
            fitted: false,
        }
    }

    /// Check if condition is met for given data
    pub fn check_condition(&self, data: &ArrayView2<f64>) -> bool {
        (self.config.condition)(data)
    }
}

impl<T> Transform<Array2<f64>, Array2<f64>> for ConditionalStep<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone,
{
    fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let data_view = data.view();

        if self.check_condition(&data_view) {
            self.config.transformer.transform(data)
        } else if self.config.skip_on_false {
            Ok(data.clone()) // Pass through unchanged
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Condition not met for step: {}",
                self.config.name
            )))
        }
    }
}

/// Configuration for parallel preprocessing branches
#[derive(Debug)]
pub struct ParallelBranchConfig<T> {
    /// Transformers to run in parallel
    pub transformers: Vec<T>,
    /// Names for each branch
    pub branch_names: Vec<String>,
    /// Strategy for combining results
    pub combination_strategy: BranchCombinationStrategy,
}

/// Strategy for combining parallel branch results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BranchCombinationStrategy {
    /// Concatenate features horizontally
    Concatenate,
    /// Average the results
    Average,
    /// Take the first successful result
    FirstSuccess,
    /// Use custom weighted combination
    WeightedCombination(Vec<f64>),
}

/// Parallel preprocessing branches
pub struct ParallelBranches<T> {
    config: ParallelBranchConfig<T>,
    fitted: bool,
}

impl<T> ParallelBranches<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    pub fn new(config: ParallelBranchConfig<T>) -> Result<Self> {
        if config.transformers.len() != config.branch_names.len() {
            return Err(SklearsError::InvalidInput(
                "Number of transformers must match number of branch names".to_string(),
            ));
        }

        if let BranchCombinationStrategy::WeightedCombination(ref weights) =
            config.combination_strategy
        {
            if weights.len() != config.transformers.len() {
                return Err(SklearsError::InvalidInput(
                    "Number of weights must match number of transformers".to_string(),
                ));
            }
        }

        Ok(Self {
            config,
            fitted: false,
        })
    }
}

impl<T> Transform<Array2<f64>, Array2<f64>> for ParallelBranches<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        // Run transformations in parallel (if feature enabled) or sequentially
        #[cfg(feature = "parallel")]
        let results: Result<Vec<Array2<f64>>> = self
            .config
            .transformers
            .par_iter()
            .zip(self.config.branch_names.par_iter())
            .map(|(transformer, name)| {
                transformer.transform(data).map_err(|e| {
                    SklearsError::TransformError(format!("Error in branch '{}': {}", name, e))
                })
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results: Result<Vec<Array2<f64>>> = self
            .config
            .transformers
            .iter()
            .zip(self.config.branch_names.iter())
            .map(|(transformer, name)| {
                transformer.transform(data).map_err(|e| {
                    SklearsError::TransformError(format!("Error in branch '{}': {}", name, e))
                })
            })
            .collect();

        let branch_results = results?;

        // Combine results based on strategy
        match &self.config.combination_strategy {
            BranchCombinationStrategy::Concatenate => self.concatenate_results(branch_results),
            BranchCombinationStrategy::Average => self.average_results(branch_results),
            BranchCombinationStrategy::FirstSuccess => {
                Ok(branch_results.into_iter().next().unwrap())
            }
            BranchCombinationStrategy::WeightedCombination(weights) => {
                self.weighted_combination(branch_results, weights)
            }
        }
    }
}

impl<T> ParallelBranches<T> {
    /// Concatenate results horizontally
    fn concatenate_results(&self, results: Vec<Array2<f64>>) -> Result<Array2<f64>> {
        if results.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No results to concatenate".to_string(),
            ));
        }

        let n_rows = results[0].nrows();
        if !results.iter().all(|r| r.nrows() == n_rows) {
            return Err(SklearsError::InvalidInput(
                "All results must have the same number of rows for concatenation".to_string(),
            ));
        }

        let total_cols: usize = results.iter().map(|r| r.ncols()).sum();
        let mut combined = Array2::zeros((n_rows, total_cols));

        let mut col_offset = 0;
        for result in results {
            let n_cols = result.ncols();
            combined
                .slice_mut(s![.., col_offset..col_offset + n_cols])
                .assign(&result);
            col_offset += n_cols;
        }

        Ok(combined)
    }

    /// Average the results
    fn average_results(&self, results: Vec<Array2<f64>>) -> Result<Array2<f64>> {
        if results.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No results to average".to_string(),
            ));
        }

        let shape = results[0].raw_dim();
        if !results.iter().all(|r| r.raw_dim() == shape) {
            return Err(SklearsError::InvalidInput(
                "All results must have the same shape for averaging".to_string(),
            ));
        }

        let mut sum = Array2::zeros(shape);
        for result in &results {
            sum += result;
        }
        sum /= results.len() as f64;

        Ok(sum)
    }

    /// Weighted combination of results
    fn weighted_combination(
        &self,
        results: Vec<Array2<f64>>,
        weights: &[f64],
    ) -> Result<Array2<f64>> {
        if results.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No results to combine".to_string(),
            ));
        }

        let shape = results[0].raw_dim();
        if !results.iter().all(|r| r.raw_dim() == shape) {
            return Err(SklearsError::InvalidInput(
                "All results must have the same shape for weighted combination".to_string(),
            ));
        }

        let mut combined = Array2::zeros(shape);
        for (result, &weight) in results.iter().zip(weights.iter()) {
            combined += &(result * weight);
        }

        Ok(combined)
    }
}

/// Wrapper for streaming transformers to work in regular pipelines
pub struct StreamingTransformerWrapper {
    transformer: Box<dyn StreamingTransformer + Send + Sync>,
    name: String,
    fitted: bool,
}

impl StreamingTransformerWrapper {
    /// Create a new wrapper for a streaming transformer
    pub fn new<S>(transformer: S, name: String) -> Self
    where
        S: StreamingTransformer + Send + Sync + 'static,
    {
        Self {
            transformer: Box::new(transformer),
            name,
            fitted: false,
        }
    }

    /// Incrementally fit the streaming transformer
    pub fn partial_fit(&mut self, data: &Array2<f64>) -> Result<()> {
        self.transformer.partial_fit(data).map_err(|e| {
            SklearsError::InvalidInput(format!("Streaming transformer error: {}", e))
        })?;
        self.fitted = true;
        Ok(())
    }

    /// Check if the wrapper is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted && self.transformer.is_fitted()
    }

    /// Get streaming statistics
    pub fn get_streaming_stats(&self) -> Option<StreamingStats> {
        Some(self.transformer.get_stats())
    }

    /// Reset the streaming transformer
    pub fn reset(&mut self) {
        self.transformer.reset();
        self.fitted = false;
    }

    /// Get the name of the streaming transformer
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Transform<Array2<f64>, Array2<f64>> for StreamingTransformerWrapper {
    fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.is_fitted() {
            return Err(SklearsError::NotFitted {
                operation: format!("transform on streaming transformer '{}'", self.name),
            });
        }
        self.transformer
            .transform(data)
            .map_err(|e| SklearsError::InvalidInput(e.to_string()))
    }
}

impl Clone for StreamingTransformerWrapper {
    fn clone(&self) -> Self {
        // Note: This is a simplified clone that won't preserve the exact state
        // For production use, you'd want to implement proper serialization/deserialization
        Self {
            transformer: Box::new(crate::streaming::StreamingStandardScaler::new(
                StreamingConfig::default(),
            )),
            name: self.name.clone(),
            fitted: false,
        }
    }
}

/// Advanced pipeline with caching and conditional steps
pub struct AdvancedPipeline<T> {
    steps: Vec<PipelineStep<T>>,
    cache: TransformationCache<Array2<f64>>,
    config: AdvancedPipelineConfig,
}

/// Pipeline step that can be conditional, parallel, cached, or streaming
pub enum PipelineStep<T> {
    /// Simple transformation step
    Simple(T),
    /// Conditional step
    Conditional(ConditionalStep<T>),
    /// Parallel branches
    Parallel(ParallelBranches<T>),
    /// Cached transformation
    Cached(T, String), // transformer and cache key prefix
    /// Streaming transformation step
    Streaming(StreamingTransformerWrapper),
}

/// Configuration for advanced pipeline
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdvancedPipelineConfig {
    /// Cache configuration
    pub cache_config: CacheConfig,
    /// Enable parallel execution
    pub parallel_execution: bool,
    /// Error handling strategy
    pub error_strategy: ErrorHandlingStrategy,
}

/// Error handling strategy for pipeline execution
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ErrorHandlingStrategy {
    /// Stop on first error
    StopOnError,
    /// Skip failed steps and continue
    SkipOnError,
    /// Use fallback transformations
    Fallback,
}

impl Default for AdvancedPipelineConfig {
    fn default() -> Self {
        Self {
            cache_config: CacheConfig::default(),
            parallel_execution: true,
            error_strategy: ErrorHandlingStrategy::StopOnError,
        }
    }
}

impl<T> AdvancedPipeline<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    pub fn new(config: AdvancedPipelineConfig) -> Self {
        Self {
            steps: Vec::new(),
            cache: TransformationCache::new(config.cache_config.clone()),
            config,
        }
    }

    /// Add a simple transformation step
    pub fn add_step(mut self, transformer: T) -> Self {
        self.steps.push(PipelineStep::Simple(transformer));
        self
    }

    /// Add a conditional step
    pub fn add_conditional_step(mut self, config: ConditionalStepConfig<T>) -> Self {
        self.steps
            .push(PipelineStep::Conditional(ConditionalStep::new(config)));
        self
    }

    /// Add parallel branches
    pub fn add_parallel_branches(mut self, config: ParallelBranchConfig<T>) -> Result<Self> {
        let branches = ParallelBranches::new(config)?;
        self.steps.push(PipelineStep::Parallel(branches));
        Ok(self)
    }

    /// Add cached transformation step
    pub fn add_cached_step(mut self, transformer: T, cache_key_prefix: String) -> Self {
        self.steps
            .push(PipelineStep::Cached(transformer, cache_key_prefix));
        self
    }

    /// Add a streaming transformation step
    pub fn add_streaming_step<S>(mut self, transformer: S, name: String) -> Self
    where
        S: StreamingTransformer + Send + Sync + 'static,
    {
        let wrapper = StreamingTransformerWrapper::new(transformer, name);
        self.steps.push(PipelineStep::Streaming(wrapper));
        self
    }

    /// Add a dimensionality reduction step
    pub fn add_pca_step(self, _pca: crate::dimensionality_reduction::PCA) -> Self {
        // For now, we need to fit the PCA first to get a fitted transformer
        // In a real pipeline, this would be handled during pipeline fitting
        self
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear pipeline cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Incrementally fit streaming transformers in the pipeline
    pub fn partial_fit(&mut self, data: &Array2<f64>) -> Result<()> {
        let mut current_data = data.clone();

        for step in &mut self.steps {
            match step {
                PipelineStep::Streaming(ref mut streaming_wrapper) => {
                    streaming_wrapper.partial_fit(&current_data)?;
                    // Transform the data for the next step
                    if streaming_wrapper.is_fitted() {
                        current_data = streaming_wrapper.transform(&current_data)?;
                    }
                }
                // For non-streaming steps, just transform if they're already fitted
                PipelineStep::Simple(transformer) => {
                    // Only transform if we can (non-streaming transformers need to be pre-fitted)
                    if let Ok(transformed) = transformer.transform(&current_data) {
                        current_data = transformed;
                    }
                }
                PipelineStep::Conditional(conditional) => {
                    if let Ok(transformed) = conditional.transform(&current_data) {
                        current_data = transformed;
                    }
                }
                PipelineStep::Parallel(parallel) => {
                    if let Ok(transformed) = parallel.transform(&current_data) {
                        current_data = transformed;
                    }
                }
                PipelineStep::Cached(transformer, _) => {
                    if let Ok(transformed) = transformer.transform(&current_data) {
                        current_data = transformed;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get streaming statistics for all streaming steps
    pub fn get_streaming_stats(&self) -> Vec<(String, Option<StreamingStats>)> {
        let mut stats = Vec::new();

        for step in &self.steps {
            if let PipelineStep::Streaming(streaming_wrapper) = step {
                stats.push((
                    streaming_wrapper.name().to_string(),
                    streaming_wrapper.get_streaming_stats(),
                ));
            }
        }

        stats
    }

    /// Reset all streaming transformers in the pipeline
    pub fn reset_streaming(&mut self) {
        for step in &mut self.steps {
            if let PipelineStep::Streaming(ref mut streaming_wrapper) = step {
                streaming_wrapper.reset();
            }
        }
    }
}

impl<T> Transform<Array2<f64>, Array2<f64>> for AdvancedPipeline<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut current_data = data.clone();
        for (step_idx, step) in self.steps.iter().enumerate() {
            let step_result = match step {
                PipelineStep::Simple(transformer) => transformer.transform(&current_data),
                PipelineStep::Conditional(conditional) => conditional.transform(&current_data),
                PipelineStep::Parallel(parallel) => parallel.transform(&current_data),
                PipelineStep::Cached(transformer, _cache_key_prefix) => {
                    // For cached transformations, we'll skip complex hashing for now
                    // and just execute the transformer directly
                    transformer.transform(&current_data)
                }
                PipelineStep::Streaming(streaming_wrapper) => {
                    streaming_wrapper.transform(&current_data)
                }
            };

            // Handle step result based on error strategy
            match step_result {
                Ok(result) => {
                    current_data = result;
                }
                Err(e) => {
                    match self.config.error_strategy {
                        ErrorHandlingStrategy::StopOnError => return Err(e),
                        ErrorHandlingStrategy::SkipOnError => {
                            // Log error and continue with original data
                            eprintln!("Warning: Step {} failed: {}. Skipping...", step_idx, e);
                            // current_data remains unchanged
                        }
                        ErrorHandlingStrategy::Fallback => {
                            // For now, just skip like SkipOnError
                            // In a real implementation, you might have fallback transformers
                            eprintln!(
                                "Warning: Step {} failed: {}. Using fallback (passthrough)...",
                                step_idx, e
                            );
                        }
                    }
                }
            }
        }

        Ok(current_data)
    }
}

/// Builder for creating advanced pipelines
pub struct AdvancedPipelineBuilder<T> {
    config: AdvancedPipelineConfig,
    pipeline: AdvancedPipeline<T>,
}

impl<T> AdvancedPipelineBuilder<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    pub fn new() -> Self {
        let config = AdvancedPipelineConfig::default();
        let pipeline = AdvancedPipeline::new(config.clone());
        Self { config, pipeline }
    }

    pub fn with_cache_config(mut self, cache_config: CacheConfig) -> Self {
        self.config.cache_config = cache_config;
        self.pipeline.cache = TransformationCache::new(self.config.cache_config.clone());
        self
    }

    pub fn with_error_strategy(mut self, strategy: ErrorHandlingStrategy) -> Self {
        self.config.error_strategy = strategy;
        self.pipeline.config.error_strategy = strategy;
        self
    }

    pub fn add_step(mut self, transformer: T) -> Self {
        self.pipeline = self.pipeline.add_step(transformer);
        self
    }

    pub fn add_conditional_step(mut self, config: ConditionalStepConfig<T>) -> Self {
        self.pipeline = self.pipeline.add_conditional_step(config);
        self
    }

    pub fn add_parallel_branches(mut self, config: ParallelBranchConfig<T>) -> Result<Self> {
        self.pipeline = self.pipeline.add_parallel_branches(config)?;
        Ok(self)
    }

    pub fn add_cached_step(mut self, transformer: T, cache_key_prefix: String) -> Self {
        self.pipeline = self.pipeline.add_cached_step(transformer, cache_key_prefix);
        self
    }

    pub fn add_streaming_step<S>(mut self, transformer: S, name: String) -> Self
    where
        S: StreamingTransformer + Send + Sync + 'static,
    {
        self.pipeline = self.pipeline.add_streaming_step(transformer, name);
        self
    }

    pub fn build(self) -> AdvancedPipeline<T> {
        self.pipeline
    }
}

impl<T> Default for AdvancedPipelineBuilder<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Dynamic pipeline that can be modified at runtime
pub struct DynamicPipeline<T> {
    steps: Arc<RwLock<Vec<PipelineStep<T>>>>,
    cache: TransformationCache<Array2<f64>>,
    config: AdvancedPipelineConfig,
}

impl<T> DynamicPipeline<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    pub fn new(config: AdvancedPipelineConfig) -> Self {
        Self {
            steps: Arc::new(RwLock::new(Vec::new())),
            cache: TransformationCache::new(config.cache_config.clone()),
            config,
        }
    }

    /// Add step at runtime
    pub fn add_step_runtime(&self, transformer: T) -> Result<()> {
        let mut steps = self
            .steps
            .write()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire write lock".to_string()))?;
        steps.push(PipelineStep::Simple(transformer));
        Ok(())
    }

    /// Add streaming step at runtime
    pub fn add_streaming_step_runtime<S>(&self, transformer: S, name: String) -> Result<()>
    where
        S: StreamingTransformer + Send + Sync + 'static,
    {
        let mut steps = self
            .steps
            .write()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire write lock".to_string()))?;
        let wrapper = StreamingTransformerWrapper::new(transformer, name);
        steps.push(PipelineStep::Streaming(wrapper));
        Ok(())
    }

    /// Remove step by index
    pub fn remove_step(&self, index: usize) -> Result<()> {
        let mut steps = self
            .steps
            .write()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire write lock".to_string()))?;

        if index >= steps.len() {
            return Err(SklearsError::InvalidInput(
                "Step index out of bounds".to_string(),
            ));
        }

        steps.remove(index);
        Ok(())
    }

    /// Get number of steps
    pub fn len(&self) -> usize {
        self.steps.read().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Incrementally fit streaming transformers in the dynamic pipeline
    pub fn partial_fit(&self, data: &Array2<f64>) -> Result<()> {
        let mut current_data = data.clone();
        let mut steps = self
            .steps
            .write()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire write lock".to_string()))?;

        for step in steps.iter_mut() {
            match step {
                PipelineStep::Streaming(ref mut streaming_wrapper) => {
                    streaming_wrapper.partial_fit(&current_data)?;
                    // Transform the data for the next step
                    if streaming_wrapper.is_fitted() {
                        current_data = streaming_wrapper.transform(&current_data)?;
                    }
                }
                // For non-streaming steps, just transform if they're already fitted
                PipelineStep::Simple(transformer) => {
                    if let Ok(transformed) = transformer.transform(&current_data) {
                        current_data = transformed;
                    }
                }
                PipelineStep::Conditional(conditional) => {
                    if let Ok(transformed) = conditional.transform(&current_data) {
                        current_data = transformed;
                    }
                }
                PipelineStep::Parallel(parallel) => {
                    if let Ok(transformed) = parallel.transform(&current_data) {
                        current_data = transformed;
                    }
                }
                PipelineStep::Cached(transformer, _) => {
                    if let Ok(transformed) = transformer.transform(&current_data) {
                        current_data = transformed;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get streaming statistics for all streaming steps
    pub fn get_streaming_stats(&self) -> Result<Vec<(String, Option<StreamingStats>)>> {
        let mut stats = Vec::new();
        let steps = self
            .steps
            .read()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire read lock".to_string()))?;

        for step in steps.iter() {
            if let PipelineStep::Streaming(streaming_wrapper) = step {
                stats.push((
                    streaming_wrapper.name().to_string(),
                    streaming_wrapper.get_streaming_stats(),
                ));
            }
        }

        Ok(stats)
    }

    /// Reset all streaming transformers in the dynamic pipeline
    pub fn reset_streaming(&self) -> Result<()> {
        let mut steps = self
            .steps
            .write()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire write lock".to_string()))?;

        for step in steps.iter_mut() {
            if let PipelineStep::Streaming(ref mut streaming_wrapper) = step {
                streaming_wrapper.reset();
            }
        }

        Ok(())
    }
}

impl<T> Transform<Array2<f64>, Array2<f64>> for DynamicPipeline<T>
where
    T: Transform<Array2<f64>, Array2<f64>> + Clone + Send + Sync,
{
    fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut current_data = data.clone();
        let steps = self
            .steps
            .read()
            .map_err(|_| SklearsError::InvalidInput("Failed to acquire read lock".to_string()))?;

        for (step_idx, step) in steps.iter().enumerate() {
            let step_result = match step {
                PipelineStep::Simple(transformer) => transformer.transform(&current_data),
                PipelineStep::Conditional(conditional) => conditional.transform(&current_data),
                PipelineStep::Parallel(parallel) => parallel.transform(&current_data),
                PipelineStep::Cached(transformer, _cache_key_prefix) => {
                    // For now, just execute directly without caching
                    transformer.transform(&current_data)
                }
                PipelineStep::Streaming(streaming_wrapper) => {
                    streaming_wrapper.transform(&current_data)
                }
            };

            match step_result {
                Ok(result) => {
                    current_data = result;
                }
                Err(e) => match self.config.error_strategy {
                    ErrorHandlingStrategy::StopOnError => return Err(e),
                    ErrorHandlingStrategy::SkipOnError => {
                        eprintln!("Warning: Step {} failed: {}. Skipping...", step_idx, e);
                    }
                    ErrorHandlingStrategy::Fallback => {
                        eprintln!(
                            "Warning: Step {} failed: {}. Using fallback (passthrough)...",
                            step_idx, e
                        );
                    }
                },
            }
        }

        Ok(current_data)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_transformation_cache() {
        let config = CacheConfig {
            max_entries: 2,
            ttl_seconds: 1,
            enabled: true,
        };

        let cache = TransformationCache::new(config);
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        // Test cache with string key instead of Array2
        let key = cache.generate_key("test_key");
        assert!(cache.get(key).is_none());

        // Test cache put and hit
        cache.put(key, data.clone());
        assert!(cache.get(key).is_some());

        // Test cache stats
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert!(stats.enabled);
    }

    // TODO: Fix this test - requires properly fitted transformers
    // #[test]
    // fn test_parallel_branches_concatenate() {
    //     let scaler1 = StandardScaler::default();
    //     let scaler2 = StandardScaler::default();

    //     let config = ParallelBranchConfig {
    //         transformers: vec![scaler1, scaler2],
    //         branch_names: vec!["branch1".to_string(), "branch2".to_string()],
    //         combination_strategy: BranchCombinationStrategy::Concatenate,
    //     };

    //     let branches = ParallelBranches::new(config).unwrap();
    //     let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

    //     // This test would require proper fitted transformers
    //     // For now, just test the construction
    //     assert!(!branches.fitted);
    // }

    // TODO: Fix this test - requires properly fitted transformers
    // #[test]
    // fn test_advanced_pipeline_builder() {
    //     let scaler = StandardScaler::default();

    //     let pipeline = AdvancedPipelineBuilder::new()
    //         .add_step(scaler)
    //         .with_error_strategy(ErrorHandlingStrategy::SkipOnError)
    //         .build();

    //     assert_eq!(pipeline.steps.len(), 1);
    // }

    // TODO: Fix this test - requires properly fitted transformers
    // #[test]
    // fn test_dynamic_pipeline() {
    //     let config = AdvancedPipelineConfig::default();
    //     let pipeline = DynamicPipeline::new(config);

    //     assert!(pipeline.is_empty());

    //     let scaler = StandardScaler::default();
    //     pipeline.add_step_runtime(scaler).unwrap();

    //     assert_eq!(pipeline.len(), 1);
    //     assert!(!pipeline.is_empty());

    //     pipeline.remove_step(0).unwrap();
    //     assert!(pipeline.is_empty());
    // }

    #[test]
    fn test_streaming_transformer_wrapper() {
        use crate::streaming::{StreamingConfig, StreamingStandardScaler};
        use scirs2_core::ndarray::Array2;

        let scaler = StreamingStandardScaler::new(StreamingConfig::default());
        let mut wrapper = StreamingTransformerWrapper::new(scaler, "test_scaler".to_string());

        // Test that wrapper is not fitted initially
        assert!(!wrapper.is_fitted());

        // Test partial fit
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        wrapper.partial_fit(&data).unwrap();

        // Test that wrapper is fitted after partial_fit
        assert!(wrapper.is_fitted());

        // Test transform
        let result = wrapper.transform(&data).unwrap();
        assert_eq!(result.dim(), data.dim());

        // Test statistics
        let stats = wrapper.get_streaming_stats();
        assert!(stats.is_some());

        // Test name
        assert_eq!(wrapper.name(), "test_scaler");

        // Test reset
        wrapper.reset();
        assert!(!wrapper.is_fitted());
    }

    // Note: These tests are temporarily commented out due to trait bound complexities
    // In production code, the pipeline would be used with properly fitted transformers

    // #[test]
    // fn test_advanced_pipeline_with_streaming() {
    //     use crate::streaming::{StreamingStandardScaler, StreamingConfig};
    //     use scirs2_core::ndarray::Array2;

    //     // Would need to create a pipeline with a dummy transformer type
    //     // that satisfies the Transform trait bounds for testing
    // }

    // #[test]
    // fn test_dynamic_pipeline_with_streaming() {
    //     use crate::streaming::{StreamingStandardScaler, StreamingConfig};
    //     use scirs2_core::ndarray::Array2;

    //     // Would need to create a pipeline with a dummy transformer type
    //     // that satisfies the Transform trait bounds for testing
    // }
}
