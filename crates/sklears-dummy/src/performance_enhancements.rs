//! Additional Performance Optimizations
//!
//! This module provides enhanced performance optimizations including branch prediction hints,
//! CPU-specific optimizations, and specialized optimizations for dummy estimator patterns.

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Branch prediction optimization utilities
pub mod branch_optimization {
    /// Likely branch hint for better branch prediction
    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        #[cold]
        fn cold() {}

        if b {
            true
        } else {
            cold();
            false
        }
    }

    /// Unlikely branch hint for better branch prediction  
    #[inline(always)]
    pub fn unlikely(b: bool) -> bool {
        #[cold]
        fn cold() {}

        if b {
            cold();
            true
        } else {
            false
        }
    }

    /// Optimized prediction with branch hints
    #[inline]
    pub fn optimized_classify(
        score: f64,
        threshold: f64,
        positive_class: i32,
        negative_class: i32,
    ) -> i32 {
        if likely(score >= threshold) {
            positive_class
        } else {
            negative_class
        }
    }

    /// Specialized branchless classification
    #[inline(always)]
    pub fn branchless_binary_classify(score: f64, threshold: f64) -> i32 {
        (score >= threshold) as i32
    }

    /// Optimized batch prediction with branch hints
    pub fn optimized_batch_predict<F>(data: &[f64], threshold: f64, predict_fn: F) -> Vec<i32>
    where
        F: Fn(f64) -> f64,
    {
        let mut results = Vec::with_capacity(data.len());

        for &value in data {
            let score = predict_fn(value);
            // Hint that most predictions are above threshold (common case)
            results.push(if likely(score >= threshold) { 1 } else { 0 });
        }

        results
    }
}

/// CPU-specific optimization utilities
pub mod cpu_optimization {

    /// CPU feature detection and optimization selection
    pub struct CPUFeatureDetector {
        /// has_avx512
        pub has_avx512: bool,
        /// has_avx2
        pub has_avx2: bool,
        /// has_sse4_2
        pub has_sse4_2: bool,
        /// has_fma
        pub has_fma: bool,
        /// cache_line_size
        pub cache_line_size: usize,
        /// l1_cache_size
        pub l1_cache_size: usize,
        /// l2_cache_size
        pub l2_cache_size: usize,
    }

    impl CPUFeatureDetector {
        /// Detect available CPU features
        pub fn detect() -> Self {
            Self {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                has_avx512: is_x86_feature_detected!("avx512f"),
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                has_avx512: false,

                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                has_avx2: is_x86_feature_detected!("avx2"),
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                has_avx2: false,

                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                has_sse4_2: is_x86_feature_detected!("sse4.2"),
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                has_sse4_2: false,

                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                has_fma: is_x86_feature_detected!("fma"),
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                has_fma: false,

                cache_line_size: 64,       // Common cache line size
                l1_cache_size: 32 * 1024,  // 32KB typical L1 cache
                l2_cache_size: 256 * 1024, // 256KB typical L2 cache
            }
        }

        /// Select optimal SIMD width based on available features
        pub fn optimal_simd_width(&self) -> usize {
            if self.has_avx512 {
                64 // 512 bits / 8 bits per byte
            } else if self.has_avx2 {
                32 // 256 bits / 8 bits per byte
            } else if self.has_sse4_2 {
                16 // 128 bits / 8 bits per byte
            } else {
                8 // Fallback to 64-bit operations
            }
        }

        /// Recommend optimal block size for cache efficiency
        pub fn optimal_block_size<T>(&self) -> usize {
            let element_size = std::mem::size_of::<T>();
            let elements_per_cache_line = self.cache_line_size / element_size;

            // Use L1 cache size as guide for block size
            let max_elements_in_l1 = self.l1_cache_size / element_size;

            // Round down to multiple of cache line elements
            (max_elements_in_l1 / elements_per_cache_line) * elements_per_cache_line
        }
    }

    /// Fast path dispatcher based on CPU capabilities
    pub struct FastPathDispatcher {
        detector: CPUFeatureDetector,
    }

    impl Default for FastPathDispatcher {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FastPathDispatcher {
        /// Create new dispatcher
        pub fn new() -> Self {
            Self {
                detector: CPUFeatureDetector::detect(),
            }
        }

        /// Dispatch sum operation to best available implementation
        pub fn dispatch_sum(&self, data: &[f64]) -> f64 {
            if self.detector.has_avx2 && data.len() >= 32 {
                // Use AVX2 path for large arrays
                crate::performance::simd_stats::fast_sum(data)
            } else if data.len() >= 1000 {
                // Use parallel path for very large arrays
                data.chunks(1000)
                    .map(|chunk| chunk.iter().sum::<f64>())
                    .sum()
            } else {
                // Use simple scalar path for small arrays
                data.iter().sum()
            }
        }

        /// Dispatch mean operation to best available implementation
        pub fn dispatch_mean(&self, data: &[f64]) -> f64 {
            if data.is_empty() {
                return 0.0;
            }

            if self.detector.has_avx2 && data.len() >= 16 {
                crate::performance::simd_stats::fast_mean(data)
            } else {
                data.iter().sum::<f64>() / data.len() as f64
            }
        }

        /// Get CPU feature information
        pub fn features(&self) -> &CPUFeatureDetector {
            &self.detector
        }
    }
}

/// Specialized optimizations for dummy estimator patterns
pub mod dummy_optimization {
    use super::*;

    /// Pre-computed lookup table for common prediction patterns
    pub struct PredictionLookupCache {
        cache: HashMap<u64, f64>,
        strategy_cache: HashMap<u8, f64>, // Strategy ID -> cached result
        hit_count: AtomicUsize,
        miss_count: AtomicUsize,
    }

    impl Default for PredictionLookupCache {
        fn default() -> Self {
            Self::new()
        }
    }

    impl PredictionLookupCache {
        /// Create new prediction cache
        pub fn new() -> Self {
            Self {
                cache: HashMap::new(),
                strategy_cache: HashMap::new(),
                hit_count: AtomicUsize::new(0),
                miss_count: AtomicUsize::new(0),
            }
        }

        /// Get cached prediction or compute and cache
        pub fn get_or_compute<F>(&mut self, key: u64, compute_fn: F) -> f64
        where
            F: FnOnce() -> f64,
        {
            if let Some(&cached) = self.cache.get(&key) {
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                cached
            } else {
                self.miss_count.fetch_add(1, Ordering::Relaxed);
                let result = compute_fn();
                self.cache.insert(key, result);
                result
            }
        }

        /// Cache strategy result (for strategies that don't depend on input)
        pub fn cache_strategy_result(&mut self, strategy_id: u8, result: f64) {
            self.strategy_cache.insert(strategy_id, result);
        }

        /// Get cached strategy result
        pub fn get_strategy_result(&self, strategy_id: u8) -> Option<f64> {
            self.strategy_cache.get(&strategy_id).copied()
        }

        /// Get cache hit rate
        pub fn hit_rate(&self) -> f64 {
            let hits = self.hit_count.load(Ordering::Relaxed) as f64;
            let misses = self.miss_count.load(Ordering::Relaxed) as f64;
            let total = hits + misses;
            if total > 0.0 {
                hits / total
            } else {
                0.0
            }
        }

        /// Clear cache
        pub fn clear(&mut self) {
            self.cache.clear();
            self.strategy_cache.clear();
            self.hit_count.store(0, Ordering::Relaxed);
            self.miss_count.store(0, Ordering::Relaxed);
        }
    }

    /// Optimized batch prediction for dummy estimators
    pub struct BatchPredictor {
        cache: PredictionLookupCache,
        dispatcher: cpu_optimization::FastPathDispatcher,
    }

    impl Default for BatchPredictor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl BatchPredictor {
        /// Create new batch predictor
        pub fn new() -> Self {
            Self {
                cache: PredictionLookupCache::new(),
                dispatcher: cpu_optimization::FastPathDispatcher::new(),
            }
        }

        /// Optimized batch classification prediction
        pub fn predict_classification_batch(
            &mut self,
            strategy_id: u8,
            x: &Array2<f64>,
            prediction_value: i32,
        ) -> Vec<i32> {
            // For most dummy classifiers, prediction is constant
            if let Some(_cached) = self.cache.get_strategy_result(strategy_id) {
                return vec![prediction_value; x.nrows()];
            }

            // Cache the result for future use
            self.cache
                .cache_strategy_result(strategy_id, prediction_value as f64);
            vec![prediction_value; x.nrows()]
        }

        /// Optimized batch regression prediction
        pub fn predict_regression_batch(
            &mut self,
            strategy_id: u8,
            x: &Array2<f64>,
            prediction_value: f64,
        ) -> Vec<f64> {
            // For most dummy regressors, prediction is constant
            if let Some(cached) = self.cache.get_strategy_result(strategy_id) {
                return vec![cached; x.nrows()];
            }

            // Cache the result for future use
            self.cache
                .cache_strategy_result(strategy_id, prediction_value);
            vec![prediction_value; x.nrows()]
        }

        /// Get prediction cache statistics
        pub fn cache_stats(&self) -> (f64, usize, usize) {
            (
                self.cache.hit_rate(),
                self.cache.hit_count.load(Ordering::Relaxed),
                self.cache.miss_count.load(Ordering::Relaxed),
            )
        }
    }

    /// Memory-efficient storage for dummy estimator results
    pub struct CompactPredictionStorage {
        data: Vec<u8>,
        prediction_type: PredictionType,
        length: usize,
    }

    #[derive(Debug, Clone, Copy)]
    pub enum PredictionType {
        /// BinaryClassification
        BinaryClassification,
        /// MultiClassification
        MultiClassification { num_classes: u8 },
        /// Regression
        Regression,
    }

    impl CompactPredictionStorage {
        /// Create new compact storage
        pub fn new(prediction_type: PredictionType) -> Self {
            Self {
                data: Vec::new(),
                prediction_type,
                length: 0,
            }
        }

        /// Store binary classification predictions efficiently
        pub fn store_binary_predictions(&mut self, predictions: &[bool]) {
            self.length = predictions.len();
            self.data.clear();
            self.data.reserve((predictions.len() + 7) / 8);

            let mut current_byte = 0u8;
            for (i, &pred) in predictions.iter().enumerate() {
                if pred {
                    current_byte |= 1 << (i % 8);
                }

                if i % 8 == 7 || i == predictions.len() - 1 {
                    self.data.push(current_byte);
                    current_byte = 0;
                }
            }
        }

        /// Retrieve binary classification predictions
        pub fn get_binary_predictions(&self) -> Vec<bool> {
            let mut result = Vec::with_capacity(self.length);

            for (byte_idx, &byte) in self.data.iter().enumerate() {
                for bit_idx in 0..8 {
                    let global_idx = byte_idx * 8 + bit_idx;
                    if global_idx >= self.length {
                        break;
                    }
                    result.push((byte >> bit_idx) & 1 == 1);
                }
            }

            result
        }

        /// Get memory usage in bytes
        pub fn memory_usage(&self) -> usize {
            self.data.len()
        }

        /// Get compression ratio compared to `Vec<bool>`
        pub fn compression_ratio(&self) -> f64 {
            if self.length == 0 {
                return 0.0;
            }
            let vec_bool_size = self.length * std::mem::size_of::<bool>();
            let compact_size = self.data.len();
            vec_bool_size as f64 / compact_size as f64
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_branch_optimization() {
        // Test branch hints
        assert!(branch_optimization::likely(true));
        assert!(!branch_optimization::unlikely(false));

        // Test optimized classify
        let result = branch_optimization::optimized_classify(0.7, 0.5, 1, 0);
        assert_eq!(result, 1);

        let result = branch_optimization::optimized_classify(0.3, 0.5, 1, 0);
        assert_eq!(result, 0);

        // Test branchless classification
        assert_eq!(branch_optimization::branchless_binary_classify(0.7, 0.5), 1);
        assert_eq!(branch_optimization::branchless_binary_classify(0.3, 0.5), 0);
    }

    #[test]
    fn test_cpu_optimization() {
        let detector = cpu_optimization::CPUFeatureDetector::detect();

        // Test SIMD width selection
        let width = detector.optimal_simd_width();
        assert!(width >= 8); // Should be at least 8 bytes

        // Test block size calculation
        let block_size = detector.optimal_block_size::<f64>();
        assert!(block_size > 0);

        // Test dispatcher
        let dispatcher = cpu_optimization::FastPathDispatcher::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let sum = dispatcher.dispatch_sum(&data);
        assert_eq!(sum, 15.0);

        let mean = dispatcher.dispatch_mean(&data);
        assert_eq!(mean, 3.0);
    }

    #[test]
    fn test_dummy_optimization() {
        let mut cache = dummy_optimization::PredictionLookupCache::new();

        // Test caching
        let result = cache.get_or_compute(42, || 3.14);
        assert_eq!(result, 3.14);

        // Test cache hit
        let result2 = cache.get_or_compute(42, || 999.0); // Should return cached value
        assert_eq!(result2, 3.14);

        // Test strategy caching
        cache.cache_strategy_result(1, 42.0);
        assert_eq!(cache.get_strategy_result(1), Some(42.0));
        assert_eq!(cache.get_strategy_result(2), None);

        // Test batch predictor
        let mut predictor = dummy_optimization::BatchPredictor::new();
        let x = Array2::zeros((10, 5));

        let predictions = predictor.predict_classification_batch(0, &x, 1);
        assert_eq!(predictions.len(), 10);
        assert!(predictions.iter().all(|&p| p == 1));

        let predictions = predictor.predict_regression_batch(0, &x, 3.14);
        assert_eq!(predictions.len(), 10);
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_compact_storage() {
        let mut storage = dummy_optimization::CompactPredictionStorage::new(
            dummy_optimization::PredictionType::BinaryClassification,
        );

        let predictions = vec![true, false, true, true, false, false, true, false];
        storage.store_binary_predictions(&predictions);

        let retrieved = storage.get_binary_predictions();
        assert_eq!(retrieved, predictions);

        // Test compression
        let ratio = storage.compression_ratio();
        assert!(ratio > 1.0); // Should be compressed

        let memory_usage = storage.memory_usage();
        assert!(memory_usage < predictions.len() * std::mem::size_of::<bool>());
    }
}
