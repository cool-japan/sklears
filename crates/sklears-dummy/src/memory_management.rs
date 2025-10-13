//! Advanced Memory Management for Dummy Estimators
//!
//! This module provides advanced memory management features including efficient storage,
//! memory pooling, streaming algorithms, memory-mapped access, and reference counting.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock, Weak};

/// Advanced memory pool with size-based allocation strategies
pub mod advanced_pooling {
    use super::*;

    /// Pool statistics for monitoring memory usage
    #[derive(Debug, Clone)]
    pub struct PoolStatistics {
        /// total_allocations
        pub total_allocations: usize,
        /// pool_hits
        pub pool_hits: usize,
        /// pool_misses
        pub pool_misses: usize,
        /// current_pool_size
        pub current_pool_size: usize,
        /// peak_pool_size
        pub peak_pool_size: usize,
        /// total_memory_saved
        pub total_memory_saved: usize,
    }

    /// Size-aware memory pool that tracks allocations by size classes
    pub struct SizeClassMemoryPool<T> {
        pools: HashMap<usize, Vec<Vec<T>>>,
        max_pool_size: usize,
        size_classes: Vec<usize>,
        stats: Arc<Mutex<PoolStatistics>>,
    }

    impl<T> SizeClassMemoryPool<T> {
        /// Create new size-class memory pool
        pub fn new(max_pool_size: usize) -> Self {
            // Common size classes: powers of 2 and some intermediate values
            let size_classes = vec![
                8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
            ];

            Self {
                pools: HashMap::new(),
                max_pool_size,
                size_classes,
                stats: Arc::new(Mutex::new(PoolStatistics {
                    total_allocations: 0,
                    pool_hits: 0,
                    pool_misses: 0,
                    current_pool_size: 0,
                    peak_pool_size: 0,
                    total_memory_saved: 0,
                })),
            }
        }

        /// Get the best size class for a requested size
        fn get_size_class(&self, size: usize) -> usize {
            self.size_classes
                .iter()
                .find(|&&class_size| class_size >= size)
                .copied()
                .unwrap_or(size.next_power_of_two())
        }

        /// Get a vector from the pool or allocate new one
        pub fn get(&mut self, size: usize) -> Vec<T> {
            let size_class = self.get_size_class(size);

            if let Some(pool) = self.pools.get_mut(&size_class) {
                if let Some(mut vec) = pool.pop() {
                    vec.clear();
                    vec.reserve(size);

                    // Update statistics
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.pool_hits += 1;
                        stats.current_pool_size -= 1;
                        stats.total_memory_saved += size_class * std::mem::size_of::<T>();
                    }

                    return vec;
                }
            }

            // Pool miss - allocate new vector
            if let Ok(mut stats) = self.stats.lock() {
                stats.pool_misses += 1;
                stats.total_allocations += 1;
            }

            Vec::with_capacity(size)
        }

        /// Return a vector to the pool
        pub fn return_vec(&mut self, mut vec: Vec<T>) {
            let capacity = vec.capacity();
            let size_class = self.get_size_class(capacity);

            let pool = self.pools.entry(size_class).or_default();

            if pool.len() < self.max_pool_size {
                vec.clear();
                pool.push(vec);

                // Update statistics
                if let Ok(mut stats) = self.stats.lock() {
                    stats.current_pool_size += 1;
                    stats.peak_pool_size = stats.peak_pool_size.max(stats.current_pool_size);
                }
            }
        }

        /// Get pool statistics
        pub fn statistics(&self) -> PoolStatistics {
            self.stats.lock().unwrap().clone()
        }

        /// Get pool efficiency (hit rate)
        pub fn efficiency(&self) -> f64 {
            if let Ok(stats) = self.stats.lock() {
                let total_requests = stats.pool_hits + stats.pool_misses;
                if total_requests > 0 {
                    stats.pool_hits as f64 / total_requests as f64
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }

        /// Clear all pools and reset statistics
        pub fn clear(&mut self) {
            self.pools.clear();
            if let Ok(mut stats) = self.stats.lock() {
                *stats = PoolStatistics {
                    total_allocations: 0,
                    pool_hits: 0,
                    pool_misses: 0,
                    current_pool_size: 0,
                    peak_pool_size: 0,
                    total_memory_saved: 0,
                };
            }
        }
    }

    /// Thread-safe memory pool for concurrent access
    pub struct ThreadSafeMemoryPool<T> {
        inner: Arc<Mutex<SizeClassMemoryPool<T>>>,
    }

    impl<T> ThreadSafeMemoryPool<T> {
        /// Create new thread-safe memory pool
        pub fn new(max_pool_size: usize) -> Self {
            Self {
                inner: Arc::new(Mutex::new(SizeClassMemoryPool::new(max_pool_size))),
            }
        }

        /// Get a vector from the pool
        pub fn get(&self, size: usize) -> Vec<T> {
            self.inner.lock().unwrap().get(size)
        }

        /// Return a vector to the pool
        pub fn return_vec(&self, vec: Vec<T>) {
            self.inner.lock().unwrap().return_vec(vec);
        }

        /// Get pool statistics
        pub fn statistics(&self) -> PoolStatistics {
            self.inner.lock().unwrap().statistics()
        }

        /// Get pool efficiency
        pub fn efficiency(&self) -> f64 {
            self.inner.lock().unwrap().efficiency()
        }
    }

    /// Memory-mapped storage for large datasets
    pub struct MemoryMappedStorage {
        data: Vec<u8>,
        element_size: usize,
        length: usize,
    }

    impl MemoryMappedStorage {
        /// Create new memory-mapped storage
        pub fn new(element_size: usize, initial_capacity: usize) -> Self {
            Self {
                data: Vec::with_capacity(initial_capacity * element_size),
                element_size,
                length: 0,
            }
        }

        /// Append element to storage
        pub fn push<T>(&mut self, element: &T) -> Result<usize, &'static str>
        where
            T: Copy,
        {
            if std::mem::size_of::<T>() != self.element_size {
                return Err("Element size mismatch");
            }

            let bytes = unsafe {
                std::slice::from_raw_parts(element as *const T as *const u8, self.element_size)
            };

            let index = self.length;
            self.data.extend_from_slice(bytes);
            self.length += 1;
            Ok(index)
        }

        /// Get element by index
        pub fn get<T>(&self, index: usize) -> Option<T>
        where
            T: Copy,
        {
            if index >= self.length || std::mem::size_of::<T>() != self.element_size {
                return None;
            }

            let start = index * self.element_size;
            let end = start + self.element_size;
            let bytes = &self.data[start..end];

            unsafe {
                let ptr = bytes.as_ptr() as *const T;
                Some(*ptr)
            }
        }

        /// Get number of elements
        pub fn len(&self) -> usize {
            self.length
        }

        /// Check if empty
        pub fn is_empty(&self) -> bool {
            self.length == 0
        }

        /// Get memory usage in bytes
        pub fn memory_usage(&self) -> usize {
            self.data.len()
        }

        /// Shrink storage to fit current data
        pub fn shrink_to_fit(&mut self) {
            self.data.shrink_to_fit();
        }
    }
}

/// Reference counting for shared data structures
pub mod reference_counting {
    use super::*;

    /// Reference-counted prediction cache
    pub struct SharedPredictionCache {
        inner: Arc<RwLock<HashMap<u64, f64>>>,
        stats: Arc<RwLock<CacheStats>>,
    }

    #[derive(Debug, Clone, Default)]
    pub struct CacheStats {
        /// hits
        pub hits: usize,
        /// misses
        pub misses: usize,
        /// total_size
        pub total_size: usize,
        /// max_size
        pub max_size: usize,
    }

    impl Default for SharedPredictionCache {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SharedPredictionCache {
        /// Create new shared prediction cache
        pub fn new() -> Self {
            Self {
                inner: Arc::new(RwLock::new(HashMap::new())),
                stats: Arc::new(RwLock::new(CacheStats::default())),
            }
        }

        /// Get value from cache
        pub fn get(&self, key: u64) -> Option<f64> {
            if let Ok(cache) = self.inner.read() {
                let result = cache.get(&key).copied();

                if let Ok(mut stats) = self.stats.write() {
                    if result.is_some() {
                        stats.hits += 1;
                    } else {
                        stats.misses += 1;
                    }
                }

                result
            } else {
                None
            }
        }

        /// Insert value into cache
        pub fn insert(&self, key: u64, value: f64) {
            if let Ok(mut cache) = self.inner.write() {
                cache.insert(key, value);

                if let Ok(mut stats) = self.stats.write() {
                    stats.total_size = cache.len();
                    stats.max_size = stats.max_size.max(stats.total_size);
                }
            }
        }

        /// Get cache statistics
        pub fn stats(&self) -> CacheStats {
            self.stats.read().unwrap().clone()
        }

        /// Get cache hit rate
        pub fn hit_rate(&self) -> f64 {
            let stats = self.stats();
            let total = stats.hits + stats.misses;
            if total > 0 {
                stats.hits as f64 / total as f64
            } else {
                0.0
            }
        }

        /// Clear cache
        pub fn clear(&self) {
            if let Ok(mut cache) = self.inner.write() {
                cache.clear();
            }
            if let Ok(mut stats) = self.stats.write() {
                *stats = CacheStats::default();
            }
        }

        /// Create a weak reference to this cache
        pub fn downgrade(&self) -> WeakPredictionCache {
            WeakPredictionCache {
                inner: Arc::downgrade(&self.inner),
                stats: Arc::downgrade(&self.stats),
            }
        }
    }

    impl Clone for SharedPredictionCache {
        fn clone(&self) -> Self {
            Self {
                inner: Arc::clone(&self.inner),
                stats: Arc::clone(&self.stats),
            }
        }
    }

    /// Weak reference to prediction cache
    pub struct WeakPredictionCache {
        inner: Weak<RwLock<HashMap<u64, f64>>>,
        stats: Weak<RwLock<CacheStats>>,
    }

    impl WeakPredictionCache {
        /// Upgrade to strong reference if still alive
        pub fn upgrade(&self) -> Option<SharedPredictionCache> {
            if let (Some(inner), Some(stats)) = (self.inner.upgrade(), self.stats.upgrade()) {
                Some(SharedPredictionCache { inner, stats })
            } else {
                None
            }
        }
    }

    /// Reference-counted model storage
    pub struct SharedModelStorage<T> {
        models: Arc<RwLock<HashMap<String, T>>>,
        access_count: Arc<RwLock<HashMap<String, usize>>>,
    }

    impl<T: Clone> Default for SharedModelStorage<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T: Clone> SharedModelStorage<T> {
        /// Create new shared model storage
        pub fn new() -> Self {
            Self {
                models: Arc::new(RwLock::new(HashMap::new())),
                access_count: Arc::new(RwLock::new(HashMap::new())),
            }
        }

        /// Store a model
        pub fn store(&self, name: String, model: T) {
            if let Ok(mut models) = self.models.write() {
                models.insert(name.clone(), model);
            }
            if let Ok(mut counts) = self.access_count.write() {
                counts.insert(name, 0);
            }
        }

        /// Get a model by name
        pub fn get(&self, name: &str) -> Option<T> {
            let result = if let Ok(models) = self.models.read() {
                models.get(name).cloned()
            } else {
                None
            };

            if result.is_some() {
                // Increment access count
                if let Ok(mut counts) = self.access_count.write() {
                    *counts.entry(name.to_string()).or_insert(0) += 1;
                }
            }

            result
        }

        /// Get access count for a model
        pub fn access_count(&self, name: &str) -> usize {
            self.access_count
                .read()
                .unwrap()
                .get(name)
                .copied()
                .unwrap_or(0)
        }

        /// Get most accessed models
        pub fn most_accessed(&self, limit: usize) -> Vec<(String, usize)> {
            if let Ok(counts) = self.access_count.read() {
                let mut items: Vec<_> = counts
                    .iter()
                    .map(|(name, &count)| (name.clone(), count))
                    .collect();
                items.sort_by(|a, b| b.1.cmp(&a.1));
                items.truncate(limit);
                items
            } else {
                Vec::new()
            }
        }

        /// Remove unused models (access count = 0)
        pub fn cleanup_unused(&self) {
            if let (Ok(mut models), Ok(mut counts)) =
                (self.models.write(), self.access_count.write())
            {
                let unused: Vec<_> = counts
                    .iter()
                    .filter(|(_, &count)| count == 0)
                    .map(|(name, _)| name.clone())
                    .collect();

                for name in unused {
                    models.remove(&name);
                    counts.remove(&name);
                }
            }
        }
    }

    impl<T: Clone> Clone for SharedModelStorage<T> {
        fn clone(&self) -> Self {
            Self {
                models: Arc::clone(&self.models),
                access_count: Arc::clone(&self.access_count),
            }
        }
    }
}

/// Streaming algorithms for constant memory usage
pub mod streaming_algorithms {

    /// Streaming quantile estimation using P² algorithm
    pub struct StreamingQuantileEstimator {
        quantile: f64,
        marker_positions: [f64; 5],
        marker_heights: [f64; 5],
        positions: [i32; 5],
        increments: [f64; 5],
        count: usize,
    }

    impl StreamingQuantileEstimator {
        /// Create new streaming quantile estimator
        pub fn new(quantile: f64) -> Self {
            let p = quantile;
            Self {
                quantile,
                marker_positions: [0.0, 0.0, 0.0, 0.0, 0.0],
                marker_heights: [0.0, 0.0, 0.0, 0.0, 0.0],
                positions: [1, 2, 3, 4, 5],
                increments: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
                count: 0,
            }
        }

        /// Update with new value
        pub fn update(&mut self, value: f64) {
            self.count += 1;

            if self.count <= 5 {
                // Initialize first 5 values
                self.marker_heights[self.count - 1] = value;
                if self.count == 5 {
                    self.marker_heights
                        .sort_by(|a, b| a.partial_cmp(b).unwrap());
                    for i in 0..5 {
                        self.marker_positions[i] = i as f64 + 1.0;
                    }
                }
                return;
            }

            // Find cell k
            let mut k = 0;
            if value < self.marker_heights[0] {
                self.marker_heights[0] = value;
                k = 1;
            } else if value >= self.marker_heights[4] {
                self.marker_heights[4] = value;
                k = 4;
            } else {
                for i in 1..4 {
                    if value < self.marker_heights[i] {
                        k = i;
                        break;
                    }
                }
                k = 4;
            }

            // Increment positions
            for i in k..5 {
                self.positions[i] += 1;
            }

            // Update desired positions
            for i in 0..5 {
                self.marker_positions[i] += self.increments[i];
            }

            // Adjust heights
            for i in 1..4 {
                let d = self.marker_positions[i] - self.positions[i] as f64;
                if (d >= 1.0 && self.positions[i + 1] - self.positions[i] > 1)
                    || (d <= -1.0 && self.positions[i - 1] - self.positions[i] < -1)
                {
                    let d_sign = d.signum() as i32;
                    let new_height = self.parabolic_prediction(i, d_sign);

                    if self.marker_heights[i - 1] < new_height
                        && new_height < self.marker_heights[i + 1]
                    {
                        self.marker_heights[i] = new_height;
                    } else {
                        self.marker_heights[i] = self.linear_prediction(i, d_sign);
                    }
                    self.positions[i] += d_sign;
                }
            }
        }

        /// Get current quantile estimate
        pub fn quantile(&self) -> f64 {
            if self.count < 5 {
                let mut sorted = self.marker_heights[..self.count].to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let index = (self.quantile * (self.count - 1) as f64) as usize;
                sorted[index]
            } else {
                self.marker_heights[2]
            }
        }

        fn parabolic_prediction(&self, i: usize, d: i32) -> f64 {
            let qi = self.marker_heights[i];
            let qi_1 = self.marker_heights[i - 1];
            let qi1 = self.marker_heights[i + 1];
            let ni = self.positions[i] as f64;
            let ni_1 = self.positions[i - 1] as f64;
            let ni1 = self.positions[i + 1] as f64;

            qi + (d as f64 / (ni1 - ni_1))
                * ((ni - ni_1 + d as f64) * (qi1 - qi) / (ni1 - ni)
                    + (ni1 - ni - d as f64) * (qi - qi_1) / (ni - ni_1))
        }

        fn linear_prediction(&self, i: usize, d: i32) -> f64 {
            let qi = self.marker_heights[i];
            let ni = self.positions[i] as f64;

            if d == 1 {
                let qi1 = self.marker_heights[i + 1];
                let ni1 = self.positions[i + 1] as f64;
                qi + (qi1 - qi) / (ni1 - ni)
            } else {
                let qi_1 = self.marker_heights[i - 1];
                let ni_1 = self.positions[i - 1] as f64;
                qi - (qi_1 - qi) / (ni - ni_1)
            }
        }
    }

    /// Streaming histogram with adaptive binning
    pub struct StreamingHistogram {
        bins: Vec<(f64, f64, usize)>, // (min, max, count)
        max_bins: usize,
        total_count: usize,
    }

    impl StreamingHistogram {
        /// Create new streaming histogram
        pub fn new(max_bins: usize) -> Self {
            Self {
                bins: Vec::new(),
                max_bins,
                total_count: 0,
            }
        }

        /// Update histogram with new value
        pub fn update(&mut self, value: f64) {
            self.total_count += 1;

            // Find appropriate bin
            let mut bin_index = None;
            for (i, (min, max, _)) in self.bins.iter().enumerate() {
                if value >= *min && value <= *max {
                    bin_index = Some(i);
                    break;
                }
            }

            if let Some(i) = bin_index {
                // Update existing bin
                self.bins[i].2 += 1;
            } else {
                // Create new bin
                self.bins.push((value, value, 1));

                // Merge bins if we exceed maximum
                if self.bins.len() > self.max_bins {
                    self.merge_closest_bins();
                }
            }
        }

        /// Get histogram bins
        pub fn bins(&self) -> &[(f64, f64, usize)] {
            &self.bins
        }

        /// Get total count
        pub fn total_count(&self) -> usize {
            self.total_count
        }

        /// Estimate quantile from histogram
        pub fn quantile(&self, q: f64) -> f64 {
            let target_count = (q * self.total_count as f64) as usize;
            let mut cumulative = 0;

            for (min, max, count) in &self.bins {
                cumulative += count;
                if cumulative >= target_count {
                    // Linear interpolation within bin
                    let bin_progress = (target_count - (cumulative - count)) as f64 / *count as f64;
                    return min + bin_progress * (max - min);
                }
            }

            // Return maximum value if not found
            self.bins.last().map(|(_, max, _)| *max).unwrap_or(0.0)
        }

        fn merge_closest_bins(&mut self) {
            if self.bins.len() < 2 {
                return;
            }

            // Find two closest bins to merge
            let mut min_distance = f64::INFINITY;
            let mut merge_index = 0;

            for i in 0..self.bins.len() - 1 {
                let distance = self.bins[i + 1].0 - self.bins[i].1; // Gap between bins
                if distance < min_distance {
                    min_distance = distance;
                    merge_index = i;
                }
            }

            // Merge bins
            let bin1 = self.bins[merge_index];
            let bin2 = self.bins[merge_index + 1];
            let merged_bin = (
                bin1.0.min(bin2.0), // min
                bin1.1.max(bin2.1), // max
                bin1.2 + bin2.2,    // count
            );

            self.bins[merge_index] = merged_bin;
            self.bins.remove(merge_index + 1);
        }
    }

    /// Streaming correlation coefficient estimation
    pub struct StreamingCorrelation {
        count: usize,
        sum_x: f64,
        sum_y: f64,
        sum_xx: f64,
        sum_yy: f64,
        sum_xy: f64,
    }

    impl Default for StreamingCorrelation {
        fn default() -> Self {
            Self::new()
        }
    }

    impl StreamingCorrelation {
        /// Create new streaming correlation estimator
        pub fn new() -> Self {
            Self {
                count: 0,
                sum_x: 0.0,
                sum_y: 0.0,
                sum_xx: 0.0,
                sum_yy: 0.0,
                sum_xy: 0.0,
            }
        }

        /// Update with new (x, y) pair
        pub fn update(&mut self, x: f64, y: f64) {
            self.count += 1;
            self.sum_x += x;
            self.sum_y += y;
            self.sum_xx += x * x;
            self.sum_yy += y * y;
            self.sum_xy += x * y;
        }

        /// Get current correlation coefficient
        pub fn correlation(&self) -> f64 {
            if self.count < 2 {
                return 0.0;
            }

            let n = self.count as f64;
            let numerator = n * self.sum_xy - self.sum_x * self.sum_y;
            let denominator = ((n * self.sum_xx - self.sum_x * self.sum_x)
                * (n * self.sum_yy - self.sum_y * self.sum_y))
                .sqrt();

            if denominator == 0.0 {
                0.0
            } else {
                numerator / denominator
            }
        }

        /// Get sample count
        pub fn count(&self) -> usize {
            self.count
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_memory_pool() {
        let mut pool = advanced_pooling::SizeClassMemoryPool::<i32>::new(10);

        // Test getting and returning vectors
        let vec1 = pool.get(100);
        assert!(vec1.capacity() >= 100);

        let vec2 = pool.get(50);
        assert!(vec2.capacity() >= 50);

        pool.return_vec(vec1);
        pool.return_vec(vec2);

        // Test pool reuse
        let vec3 = pool.get(100);
        let stats = pool.statistics();
        assert!(stats.pool_hits > 0);

        // Test efficiency
        let efficiency = pool.efficiency();
        assert!(efficiency > 0.0);
    }

    #[test]
    fn test_thread_safe_memory_pool() {
        let pool = advanced_pooling::ThreadSafeMemoryPool::<f64>::new(5);

        let vec1 = pool.get(50);
        assert!(vec1.capacity() >= 50);

        pool.return_vec(vec1);

        let vec2 = pool.get(40);
        assert!(vec2.capacity() >= 40);

        let efficiency = pool.efficiency();
        assert!(efficiency >= 0.0);
    }

    #[test]
    fn test_memory_mapped_storage() {
        let mut storage =
            advanced_pooling::MemoryMappedStorage::new(std::mem::size_of::<f64>(), 10);

        // Store some values
        let idx1 = storage.push(&3.14f64).unwrap();
        let idx2 = storage.push(&2.71f64).unwrap();

        assert_eq!(storage.len(), 2);
        assert_eq!(storage.get::<f64>(idx1), Some(3.14));
        assert_eq!(storage.get::<f64>(idx2), Some(2.71));

        let memory_usage = storage.memory_usage();
        assert!(memory_usage > 0);
    }

    #[test]
    fn test_shared_prediction_cache() {
        let cache = reference_counting::SharedPredictionCache::new();

        // Test cache operations
        cache.insert(42, 3.14);
        assert_eq!(cache.get(42), Some(3.14));
        assert_eq!(cache.get(99), None);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        let hit_rate = cache.hit_rate();
        assert_eq!(hit_rate, 0.5);

        // Test cloning
        let cache2 = cache.clone();
        assert_eq!(cache2.get(42), Some(3.14));
    }

    #[test]
    fn test_shared_model_storage() {
        let storage = reference_counting::SharedModelStorage::<String>::new();

        storage.store("model1".to_string(), "data1".to_string());
        storage.store("model2".to_string(), "data2".to_string());

        assert_eq!(storage.get("model1"), Some("data1".to_string()));
        assert_eq!(storage.get("model1"), Some("data1".to_string())); // Second access
        assert_eq!(storage.get("model2"), Some("data2".to_string()));

        assert_eq!(storage.access_count("model1"), 2);
        assert_eq!(storage.access_count("model2"), 1);

        let most_accessed = storage.most_accessed(1);
        assert_eq!(most_accessed[0].0, "model1");
        assert_eq!(most_accessed[0].1, 2);
    }

    #[test]
    fn test_streaming_quantile_estimator() {
        let mut estimator = streaming_algorithms::StreamingQuantileEstimator::new(0.5); // Median

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        for value in data {
            estimator.update(value);
        }

        let median = estimator.quantile();
        assert!((median - 5.5).abs() < 1.0); // Should be approximately 5.5
    }

    #[test]
    fn test_streaming_histogram() {
        let mut histogram = streaming_algorithms::StreamingHistogram::new(5);

        for i in 1..=100 {
            histogram.update(i as f64);
        }

        assert_eq!(histogram.total_count(), 100);

        let median = histogram.quantile(0.5);
        assert!((median - 50.0).abs() < 10.0); // Should be approximately 50

        let bins = histogram.bins();
        assert!(bins.len() <= 5);
    }

    #[test]
    fn test_streaming_correlation() {
        let mut corr = streaming_algorithms::StreamingCorrelation::new();

        // Perfect positive correlation
        for i in 1..=10 {
            corr.update(i as f64, (i * 2) as f64);
        }

        let correlation = corr.correlation();
        assert!((correlation - 1.0).abs() < 1e-10);
        assert_eq!(corr.count(), 10);
    }
}
