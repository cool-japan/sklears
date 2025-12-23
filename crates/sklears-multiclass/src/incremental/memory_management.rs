//! Memory Management for Incremental Learning
//!
//! Provides efficient memory management strategies for online learning algorithms,
//! including adaptive memory allocation, data buffering, and memory-efficient updates.

use sklears_core::error::Result as SklResult;
use sklears_core::prelude::{Array1, FloatBounds};
use std::collections::{HashMap, VecDeque};

/// Configuration for memory management
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum memory budget in bytes
    pub max_memory_bytes: Option<usize>,
    /// Buffer size for batching updates
    pub buffer_size: usize,
    /// Whether to use compression for stored data
    pub use_compression: bool,
    /// Forgetting factor for old data (0.0 to 1.0)
    pub forgetting_factor: f64,
    /// Maximum number of stored examples per class
    pub max_examples_per_class: Option<usize>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: None,
            buffer_size: 1000,
            use_compression: false,
            forgetting_factor: 0.95,
            max_examples_per_class: Some(1000),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_bytes: usize,
    /// Number of stored examples
    pub num_examples: usize,
    /// Number of buffer flushes
    pub buffer_flushes: usize,
    /// Number of memory cleanups performed
    pub cleanup_count: usize,
}

/// Trait for memory-efficient data storage
pub trait MemoryEfficient<T: FloatBounds> {
    /// Add a new example to memory
    fn add_example(&mut self, x: Array1<T>, y: usize) -> SklResult<()>;

    /// Get stored examples for a specific class
    fn get_examples(&self, class: usize) -> Option<&Vec<Array1<T>>>;

    /// Remove old examples to free memory
    fn cleanup(&mut self) -> SklResult<usize>;

    /// Get current memory statistics
    fn memory_stats(&self) -> MemoryStats;

    /// Force flush any pending operations
    fn flush(&mut self) -> SklResult<()>;
}

/// Adaptive buffer for efficient memory updates
#[derive(Debug)]
pub struct AdaptiveBuffer<T: FloatBounds> {
    config: MemoryConfig,
    buffer: VecDeque<(Array1<T>, usize)>,
    class_examples: HashMap<usize, Vec<Array1<T>>>,
    stats: MemoryStats,
    total_examples: usize,
}

impl<T: FloatBounds> AdaptiveBuffer<T> {
    /// Create a new adaptive buffer
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            class_examples: HashMap::new(),
            stats: MemoryStats::default(),
            total_examples: 0,
        }
    }

    /// Estimate memory usage of an array
    fn estimate_memory_usage(array: &Array1<T>) -> usize {
        array.len() * std::mem::size_of::<T>()
    }

    /// Check if we need to perform memory cleanup
    fn needs_cleanup(&self) -> bool {
        if let Some(max_bytes) = self.config.max_memory_bytes {
            return self.stats.current_bytes > max_bytes;
        }

        if let Some(max_per_class) = self.config.max_examples_per_class {
            return self
                .class_examples
                .values()
                .any(|examples| examples.len() > max_per_class);
        }

        false
    }

    /// Perform memory cleanup using forgetting factor
    fn perform_cleanup(&mut self) -> SklResult<usize> {
        let mut removed_count = 0;

        for (_, examples) in self.class_examples.iter_mut() {
            if let Some(max_per_class) = self.config.max_examples_per_class {
                while examples.len() > max_per_class {
                    let removed = examples.remove(0);
                    self.stats.current_bytes -= Self::estimate_memory_usage(&removed);
                    removed_count += 1;
                }
            }
        }

        // Apply forgetting factor by removing oldest examples
        let target_size = (self.total_examples as f64 * self.config.forgetting_factor) as usize;
        while self.total_examples > target_size {
            let mut removed = false;
            for (_, examples) in self.class_examples.iter_mut() {
                if !examples.is_empty() {
                    let removed_example = examples.remove(0);
                    self.stats.current_bytes -= Self::estimate_memory_usage(&removed_example);
                    removed_count += 1;
                    self.total_examples -= 1;
                    removed = true;
                    break;
                }
            }
            if !removed {
                break;
            }
        }

        self.stats.cleanup_count += 1;
        Ok(removed_count)
    }

    /// Process buffered examples
    fn process_buffer(&mut self) -> SklResult<()> {
        while let Some((x, y)) = self.buffer.pop_front() {
            let memory_usage = Self::estimate_memory_usage(&x);

            let examples = self.class_examples.entry(y).or_default();
            examples.push(x);

            self.stats.current_bytes += memory_usage;
            self.stats.num_examples += 1;
            self.total_examples += 1;
        }

        self.stats.buffer_flushes += 1;
        Ok(())
    }
}

impl<T: FloatBounds> MemoryEfficient<T> for AdaptiveBuffer<T> {
    fn add_example(&mut self, x: Array1<T>, y: usize) -> SklResult<()> {
        self.buffer.push_back((x, y));

        // Process buffer when it reaches capacity
        if self.buffer.len() >= self.config.buffer_size {
            self.flush()?;
        }

        // Perform cleanup if needed
        if self.needs_cleanup() {
            self.cleanup()?;
        }

        Ok(())
    }

    fn get_examples(&self, class: usize) -> Option<&Vec<Array1<T>>> {
        self.class_examples.get(&class)
    }

    fn cleanup(&mut self) -> SklResult<usize> {
        self.perform_cleanup()
    }

    fn memory_stats(&self) -> MemoryStats {
        self.stats.clone()
    }

    fn flush(&mut self) -> SklResult<()> {
        self.process_buffer()
    }
}

/// Memory manager for online learning algorithms
#[derive(Debug)]
pub struct MemoryManager<T: FloatBounds> {
    buffer: AdaptiveBuffer<T>,
    config: MemoryConfig,
    class_weights: HashMap<usize, f64>,
}

impl<T: FloatBounds> MemoryManager<T> {
    /// Create a new memory manager
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            buffer: AdaptiveBuffer::new(config.clone()),
            config,
            class_weights: HashMap::new(),
        }
    }

    /// Update class weights based on memory usage
    pub fn update_class_weights(&mut self) {
        let total_examples = self.buffer.stats.num_examples;
        if total_examples == 0 {
            return;
        }

        for (class, examples) in &self.buffer.class_examples {
            let class_count = examples.len();
            let weight = class_count as f64 / total_examples as f64;
            self.class_weights.insert(*class, weight);
        }
    }

    /// Get weight for a specific class
    pub fn class_weight(&self, class: usize) -> f64 {
        self.class_weights.get(&class).copied().unwrap_or(1.0)
    }

    /// Get all class weights
    pub fn class_weights(&self) -> &HashMap<usize, f64> {
        &self.class_weights
    }

    /// Sample examples for training (with memory efficiency)
    pub fn sample_examples(&self, class: usize, max_samples: usize) -> Vec<Array1<T>> {
        if let Some(examples) = self.buffer.get_examples(class) {
            if examples.len() <= max_samples {
                examples.clone()
            } else {
                // Sample evenly across the available examples
                let step = examples.len() / max_samples;
                examples
                    .iter()
                    .step_by(step.max(1))
                    .take(max_samples)
                    .cloned()
                    .collect()
            }
        } else {
            Vec::new()
        }
    }

    /// Get memory usage summary
    pub fn memory_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();
        let stats = self.buffer.memory_stats();

        summary.insert("total_bytes".to_string(), stats.current_bytes);
        summary.insert("total_examples".to_string(), stats.num_examples);
        summary.insert("buffer_flushes".to_string(), stats.buffer_flushes);
        summary.insert("cleanup_count".to_string(), stats.cleanup_count);
        summary.insert("num_classes".to_string(), self.buffer.class_examples.len());

        summary
    }
}

impl<T: FloatBounds> MemoryEfficient<T> for MemoryManager<T> {
    fn add_example(&mut self, x: Array1<T>, y: usize) -> SklResult<()> {
        let result = self.buffer.add_example(x, y);
        self.update_class_weights();
        result
    }

    fn get_examples(&self, class: usize) -> Option<&Vec<Array1<T>>> {
        self.buffer.get_examples(class)
    }

    fn cleanup(&mut self) -> SklResult<usize> {
        let result = self.buffer.cleanup();
        self.update_class_weights();
        result
    }

    fn memory_stats(&self) -> MemoryStats {
        self.buffer.memory_stats()
    }

    fn flush(&mut self) -> SklResult<()> {
        let result = self.buffer.flush();
        self.update_class_weights();
        result
    }
}

/// Builder for memory configuration
pub struct MemoryConfigBuilder {
    config: MemoryConfig,
}

impl MemoryConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: MemoryConfig::default(),
        }
    }

    /// Set maximum memory budget
    pub fn max_memory_bytes(mut self, bytes: usize) -> Self {
        self.config.max_memory_bytes = Some(bytes);
        self
    }

    /// Set buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Enable compression
    pub fn use_compression(mut self) -> Self {
        self.config.use_compression = true;
        self
    }

    /// Set forgetting factor
    pub fn forgetting_factor(mut self, factor: f64) -> Self {
        self.config.forgetting_factor = factor;
        self
    }

    /// Set maximum examples per class
    pub fn max_examples_per_class(mut self, max: usize) -> Self {
        self.config.max_examples_per_class = Some(max);
        self
    }

    /// Build the configuration
    pub fn build(self) -> MemoryConfig {
        self.config
    }
}

impl Default for MemoryConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_memory_manager_creation() {
        let manager: MemoryManager<f64> = MemoryManager::new(MemoryConfig::default());
        let stats = manager.memory_stats();
        assert_eq!(stats.num_examples, 0);
    }

    #[test]
    fn test_add_example() {
        let mut manager: MemoryManager<f64> = MemoryManager::new(MemoryConfig::default());
        let x = array![1.0, 2.0, 3.0];

        assert!(manager.add_example(x, 0).is_ok());

        manager.flush().unwrap();
        let stats = manager.memory_stats();
        assert_eq!(stats.num_examples, 1);
    }

    #[test]
    fn test_class_weights() {
        let mut manager: MemoryManager<f64> = MemoryManager::new(MemoryConfig::default());

        // Add examples for different classes
        manager.add_example(array![1.0, 2.0], 0).unwrap();
        manager.add_example(array![3.0, 4.0], 1).unwrap();
        manager.add_example(array![5.0, 6.0], 0).unwrap();

        manager.flush().unwrap();

        // Check class weights
        let weight_0 = manager.class_weight(0);
        let weight_1 = manager.class_weight(1);

        assert!(weight_0 > weight_1); // Class 0 has more examples
    }

    #[test]
    fn test_memory_config_builder() {
        let config = MemoryConfigBuilder::new()
            .max_memory_bytes(1000000)
            .buffer_size(500)
            .use_compression()
            .forgetting_factor(0.9)
            .max_examples_per_class(100)
            .build();

        assert_eq!(config.max_memory_bytes, Some(1000000));
        assert_eq!(config.buffer_size, 500);
        assert!(config.use_compression);
        assert_eq!(config.forgetting_factor, 0.9);
        assert_eq!(config.max_examples_per_class, Some(100));
    }

    #[test]
    fn test_sample_examples() {
        let mut manager: MemoryManager<f64> = MemoryManager::new(MemoryConfig::default());

        // Add multiple examples for class 0
        for i in 0..10 {
            let x = array![i as f64, (i + 1) as f64];
            manager.add_example(x, 0).unwrap();
        }

        manager.flush().unwrap();

        // Sample 5 examples
        let samples = manager.sample_examples(0, 5);
        assert_eq!(samples.len(), 5);
    }
}
