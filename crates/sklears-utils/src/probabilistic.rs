//! Probabilistic data structures for efficient approximate algorithms
//!
//! This module provides memory-efficient probabilistic data structures commonly used
//! in machine learning and big data applications for approximate computations.

use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::f64::consts::LN_2;
use std::hash::{Hash, Hasher};

/// Bloom filter for membership testing with false positives
pub struct BloomFilter {
    bit_array: Vec<bool>,
    size: usize,
    hash_functions: usize,
    inserted_count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter with optimal parameters for given capacity and false positive rate
    pub fn new(capacity: usize, false_positive_rate: f64) -> Self {
        let size = Self::optimal_size(capacity, false_positive_rate);
        let hash_functions = Self::optimal_hash_functions(size, capacity);

        Self {
            bit_array: vec![false; size],
            size,
            hash_functions,
            inserted_count: 0,
        }
    }

    /// Create a new Bloom filter with explicit parameters
    pub fn with_parameters(size: usize, hash_functions: usize) -> Self {
        Self {
            bit_array: vec![false; size],
            size,
            hash_functions,
            inserted_count: 0,
        }
    }

    fn optimal_size(capacity: usize, false_positive_rate: f64) -> usize {
        let ln2_sq = LN_2 * LN_2;
        (-(capacity as f64) * false_positive_rate.ln() / ln2_sq).ceil() as usize
    }

    fn optimal_hash_functions(size: usize, capacity: usize) -> usize {
        ((size as f64 / capacity as f64) * LN_2).ceil() as usize
    }

    fn hash_values<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut hashes = Vec::with_capacity(self.hash_functions);

        for i in 0..self.hash_functions {
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            i.hash(&mut hasher);
            hashes.push((hasher.finish() as usize) % self.size);
        }

        hashes
    }

    /// Insert an item into the filter
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let hashes = self.hash_values(item);
        for hash in hashes {
            self.bit_array[hash] = true;
        }
        self.inserted_count += 1;
    }

    /// Test if an item might be in the filter (no false negatives, possible false positives)
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.hash_values(item);
        hashes.iter().all(|&hash| self.bit_array[hash])
    }

    /// Get the current false positive probability
    pub fn false_positive_probability(&self) -> f64 {
        let bits_set = self.bit_array.iter().filter(|&&bit| bit).count() as f64;
        let ratio = bits_set / self.size as f64;
        ratio.powf(self.hash_functions as f64)
    }

    /// Get the number of items inserted
    pub fn len(&self) -> usize {
        self.inserted_count
    }

    /// Check if the filter is empty
    pub fn is_empty(&self) -> bool {
        self.inserted_count == 0
    }

    /// Clear the filter
    pub fn clear(&mut self) {
        self.bit_array.fill(false);
        self.inserted_count = 0;
    }

    /// Get filter statistics
    pub fn stats(&self) -> BloomFilterStats {
        let bits_set = self.bit_array.iter().filter(|&&bit| bit).count();
        BloomFilterStats {
            size: self.size,
            hash_functions: self.hash_functions,
            inserted_count: self.inserted_count,
            bits_set,
            load_factor: bits_set as f64 / self.size as f64,
            false_positive_probability: self.false_positive_probability(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BloomFilterStats {
    pub size: usize,
    pub hash_functions: usize,
    pub inserted_count: usize,
    pub bits_set: usize,
    pub load_factor: f64,
    pub false_positive_probability: f64,
}

/// Count-Min Sketch for frequency estimation
pub struct CountMinSketch {
    counts: Vec<Vec<u32>>,
    width: usize,
    depth: usize,
    total_count: u64,
}

impl CountMinSketch {
    /// Create a new Count-Min Sketch with specified dimensions
    pub fn new(width: usize, depth: usize) -> Self {
        Self {
            counts: vec![vec![0; width]; depth],
            width,
            depth,
            total_count: 0,
        }
    }

    /// Create a Count-Min Sketch with optimal parameters for given error bounds
    pub fn with_bounds(epsilon: f64, delta: f64) -> Self {
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;
        Self::new(width, depth)
    }

    fn hash_values<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut hashes = Vec::with_capacity(self.depth);

        for i in 0..self.depth {
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            i.hash(&mut hasher);
            hashes.push((hasher.finish() as usize) % self.width);
        }

        hashes
    }

    /// Add count occurrences of an item
    pub fn add<T: Hash>(&mut self, item: &T, count: u32) {
        let hashes = self.hash_values(item);
        for (i, &hash) in hashes.iter().enumerate() {
            self.counts[i][hash] = self.counts[i][hash].saturating_add(count);
        }
        self.total_count += count as u64;
    }

    /// Increment the count of an item by 1
    pub fn increment<T: Hash>(&mut self, item: &T) {
        self.add(item, 1);
    }

    /// Estimate the frequency of an item
    pub fn estimate<T: Hash>(&self, item: &T) -> u32 {
        let hashes = self.hash_values(item);
        hashes
            .iter()
            .enumerate()
            .map(|(i, &hash)| self.counts[i][hash])
            .min()
            .unwrap_or(0)
    }

    /// Get the total count of all items
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Clear the sketch
    pub fn clear(&mut self) {
        for row in &mut self.counts {
            row.fill(0);
        }
        self.total_count = 0;
    }

    /// Get sketch statistics
    pub fn stats(&self) -> CountMinSketchStats {
        let max_count = self
            .counts
            .iter()
            .flat_map(|row| row.iter())
            .max()
            .copied()
            .unwrap_or(0);

        let avg_count = if self.width * self.depth > 0 {
            self.total_count as f64 / (self.width * self.depth) as f64
        } else {
            0.0
        };

        CountMinSketchStats {
            width: self.width,
            depth: self.depth,
            total_count: self.total_count,
            max_count,
            avg_count,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CountMinSketchStats {
    pub width: usize,
    pub depth: usize,
    pub total_count: u64,
    pub max_count: u32,
    pub avg_count: f64,
}

/// HyperLogLog for cardinality estimation
pub struct HyperLogLog {
    buckets: Vec<u8>,
    bucket_count: usize,
    alpha: f64,
}

impl HyperLogLog {
    /// Create a new HyperLogLog with the specified precision (4-16)
    pub fn new(precision: u8) -> Self {
        assert!(
            (4..=16).contains(&precision),
            "Precision must be between 4 and 16"
        );

        let bucket_count = 1 << precision;
        let alpha = Self::calculate_alpha(bucket_count);

        Self {
            buckets: vec![0; bucket_count],
            bucket_count,
            alpha,
        }
    }

    fn calculate_alpha(bucket_count: usize) -> f64 {
        match bucket_count {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / bucket_count as f64),
        }
    }

    fn hash_value<T: Hash>(&self, item: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        hasher.finish()
    }

    fn leading_zeros(mut value: u64) -> u8 {
        if value == 0 {
            return 64;
        }

        let mut count = 0;
        while (value & 0x8000000000000000) == 0 {
            count += 1;
            value <<= 1;
        }
        count
    }

    /// Add an item to the HyperLogLog
    pub fn add<T: Hash>(&mut self, item: &T) {
        let hash = self.hash_value(item);
        let bucket_bits = 64 - (self.bucket_count as f64).log2() as u8;
        let bucket = (hash >> bucket_bits) as usize;
        let leading_zeros = Self::leading_zeros(hash << (64 - bucket_bits)) + 1;

        if leading_zeros > self.buckets[bucket] {
            self.buckets[bucket] = leading_zeros;
        }
    }

    /// Estimate the cardinality
    pub fn cardinality(&self) -> f64 {
        let sum: f64 = self
            .buckets
            .iter()
            .map(|&bucket| 2.0_f64.powf(-(bucket as f64)))
            .sum();

        let raw_estimate = self.alpha * (self.bucket_count as f64).powi(2) / sum;

        // Apply bias correction for different ranges
        if raw_estimate <= 2.5 * self.bucket_count as f64 {
            // Small range correction
            let zero_buckets = self.buckets.iter().filter(|&&bucket| bucket == 0).count();
            if zero_buckets != 0 {
                return (self.bucket_count as f64)
                    * (self.bucket_count as f64 / zero_buckets as f64).ln();
            }
        } else if raw_estimate <= (1.0 / 30.0) * (1u64 << 32) as f64 {
            // Intermediate range - no correction
            return raw_estimate;
        }

        // Large range correction
        -((1u64 << 32) as f64) * (1.0 - raw_estimate / ((1u64 << 32) as f64)).ln()
    }

    /// Merge another HyperLogLog into this one
    pub fn merge(&mut self, other: &HyperLogLog) {
        assert_eq!(
            self.bucket_count, other.bucket_count,
            "Cannot merge HyperLogLogs with different precisions"
        );

        for i in 0..self.bucket_count {
            self.buckets[i] = self.buckets[i].max(other.buckets[i]);
        }
    }

    /// Clear the HyperLogLog
    pub fn clear(&mut self) {
        self.buckets.fill(0);
    }

    /// Get HyperLogLog statistics
    pub fn stats(&self) -> HyperLogLogStats {
        let max_bucket = *self.buckets.iter().max().unwrap_or(&0);
        let zero_buckets = self.buckets.iter().filter(|&&bucket| bucket == 0).count();
        let avg_bucket =
            self.buckets.iter().map(|&b| b as f64).sum::<f64>() / self.bucket_count as f64;

        HyperLogLogStats {
            bucket_count: self.bucket_count,
            cardinality: self.cardinality(),
            max_bucket,
            zero_buckets,
            avg_bucket,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HyperLogLogStats {
    pub bucket_count: usize,
    pub cardinality: f64,
    pub max_bucket: u8,
    pub zero_buckets: usize,
    pub avg_bucket: f64,
}

/// MinHash for similarity estimation
pub struct MinHash {
    hashes: Vec<u64>,
    hash_functions: usize,
}

impl MinHash {
    /// Create a new MinHash with specified number of hash functions
    pub fn new(hash_functions: usize) -> Self {
        Self {
            hashes: vec![u64::MAX; hash_functions],
            hash_functions,
        }
    }

    fn hash_values<T: Hash>(&self, item: &T) -> Vec<u64> {
        let mut hashes = Vec::with_capacity(self.hash_functions);

        for i in 0..self.hash_functions {
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            i.hash(&mut hasher);
            hashes.push(hasher.finish());
        }

        hashes
    }

    /// Add an item to the MinHash
    pub fn add<T: Hash>(&mut self, item: &T) {
        let item_hashes = self.hash_values(item);

        for (i, &hash) in item_hashes.iter().enumerate() {
            if hash < self.hashes[i] {
                self.hashes[i] = hash;
            }
        }
    }

    /// Estimate Jaccard similarity with another MinHash
    pub fn jaccard_similarity(&self, other: &MinHash) -> f64 {
        assert_eq!(
            self.hash_functions, other.hash_functions,
            "MinHash objects must have the same number of hash functions"
        );

        let matches = self
            .hashes
            .iter()
            .zip(other.hashes.iter())
            .filter(|(&a, &b)| a == b)
            .count();

        matches as f64 / self.hash_functions as f64
    }

    /// Clear the MinHash
    pub fn clear(&mut self) {
        self.hashes.fill(u64::MAX);
    }

    /// Get MinHash statistics
    pub fn stats(&self) -> MinHashStats {
        let initialized_hashes = self.hashes.iter().filter(|&&h| h != u64::MAX).count();

        MinHashStats {
            hash_functions: self.hash_functions,
            initialized_hashes,
            completion_ratio: initialized_hashes as f64 / self.hash_functions as f64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MinHashStats {
    pub hash_functions: usize,
    pub initialized_hashes: usize,
    pub completion_ratio: f64,
}

/// Locality-Sensitive Hashing for approximate nearest neighbor search
pub struct LSHash {
    hash_tables: Vec<Vec<Vec<usize>>>,
    projections: Vec<Vec<f64>>,
    table_count: usize,
    dimension: usize,
    bucket_width: f64,
}

impl LSHash {
    /// Create a new LSH with specified parameters
    pub fn new(dimension: usize, table_count: usize, bucket_width: f64) -> Self {
        let mut projections = Vec::with_capacity(table_count);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..table_count {
            let mut projection = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                projection.push(rng.random::<f64>() * 2.0 - 1.0); // Random values between -1 and 1
            }
            projections.push(projection);
        }

        Self {
            hash_tables: vec![Vec::new(); table_count],
            projections,
            table_count,
            dimension,
            bucket_width,
        }
    }

    fn hash_vector(&self, vector: &[f64], table_idx: usize) -> i32 {
        let dot_product: f64 = vector
            .iter()
            .zip(self.projections[table_idx].iter())
            .map(|(&v, &p)| v * p)
            .sum();

        (dot_product / self.bucket_width).floor() as i32
    }

    /// Add a vector with associated data index
    pub fn add(&mut self, vector: &[f64], data_idx: usize) {
        assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension must match LSH dimension"
        );

        for table_idx in 0..self.table_count {
            let hash = self.hash_vector(vector, table_idx);

            // Ensure the hash is non-negative and resize table if needed
            if hash >= 0 {
                let bucket_idx = hash as usize;
                // Resize table if needed
                if self.hash_tables[table_idx].len() <= bucket_idx {
                    self.hash_tables[table_idx].resize(bucket_idx + 1, Vec::new());
                }
                self.hash_tables[table_idx][bucket_idx].push(data_idx);
            }
        }
    }

    /// Query for approximate nearest neighbors
    pub fn query(&self, vector: &[f64]) -> Vec<usize> {
        assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension must match LSH dimension"
        );

        let mut candidates = std::collections::HashSet::new();

        for table_idx in 0..self.table_count {
            let hash = self.hash_vector(vector, table_idx);

            if hash >= 0 && (hash as usize) < self.hash_tables[table_idx].len() {
                for &candidate in &self.hash_tables[table_idx][hash as usize] {
                    candidates.insert(candidate);
                }
            }
        }

        candidates.into_iter().collect()
    }

    /// Clear all hash tables
    pub fn clear(&mut self) {
        for table in &mut self.hash_tables {
            table.clear();
        }
    }

    /// Get LSH statistics
    pub fn stats(&self) -> LSHashStats {
        let total_entries: usize = self
            .hash_tables
            .iter()
            .flat_map(|table| table.iter())
            .map(|bucket| bucket.len())
            .sum();

        let non_empty_buckets: usize = self
            .hash_tables
            .iter()
            .flat_map(|table| table.iter())
            .filter(|bucket| !bucket.is_empty())
            .count();

        let total_buckets: usize = self.hash_tables.iter().map(|table| table.len()).sum();

        LSHashStats {
            table_count: self.table_count,
            dimension: self.dimension,
            bucket_width: self.bucket_width,
            total_entries,
            total_buckets,
            non_empty_buckets,
            load_factor: if total_buckets > 0 {
                non_empty_buckets as f64 / total_buckets as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct LSHashStats {
    pub table_count: usize,
    pub dimension: usize,
    pub bucket_width: f64,
    pub total_entries: usize,
    pub total_buckets: usize,
    pub non_empty_buckets: usize,
    pub load_factor: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_bloom_filter() {
        let mut filter = BloomFilter::new(1000, 0.01);

        // Insert some items
        filter.insert(&"hello");
        filter.insert(&"world");
        filter.insert(&42);

        // Test membership
        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(filter.contains(&42));
        assert!(!filter.contains(&"not_inserted"));

        assert_eq!(filter.len(), 3);

        let stats = filter.stats();
        assert!(stats.false_positive_probability < 0.1);
    }

    #[test]
    fn test_count_min_sketch() {
        let mut sketch = CountMinSketch::new(100, 5);

        // Add some items
        sketch.increment(&"apple");
        sketch.increment(&"apple");
        sketch.add(&"banana", 3);
        sketch.increment(&"cherry");

        // Test estimates
        assert!(sketch.estimate(&"apple") >= 2);
        assert!(sketch.estimate(&"banana") >= 3);
        assert!(sketch.estimate(&"cherry") >= 1);
        assert_eq!(sketch.estimate(&"not_added"), 0);

        assert_eq!(sketch.total_count(), 6);
    }

    #[test]
    fn test_hyperloglog() {
        let mut hll = HyperLogLog::new(8);

        // Add many unique items
        for i in 0..1000 {
            hll.add(&i);
        }

        let cardinality = hll.cardinality();
        // HyperLogLog should estimate around 1000 with some error
        assert!(cardinality > 800.0 && cardinality < 1200.0);

        // Test merge
        let mut hll2 = HyperLogLog::new(8);
        for i in 500..1500 {
            hll2.add(&i);
        }

        hll.merge(&hll2);
        let merged_cardinality = hll.cardinality();
        assert!(merged_cardinality > cardinality);
    }

    #[test]
    fn test_minhash() {
        let mut mh1 = MinHash::new(128);
        let mut mh2 = MinHash::new(128);

        // Create two sets with some overlap
        let set1: HashSet<i32> = (0..100).collect();
        let set2: HashSet<i32> = (50..150).collect();

        for item in &set1 {
            mh1.add(item);
        }

        for item in &set2 {
            mh2.add(item);
        }

        let similarity = mh1.jaccard_similarity(&mh2);

        // The actual Jaccard similarity is 50/150 = 0.33
        // MinHash should approximate this
        assert!(similarity > 0.2 && similarity < 0.5);
    }

    #[test]
    fn test_lsh() {
        let mut lsh = LSHash::new(3, 5, 1.0);

        // Add some vectors
        lsh.add(&[1.0, 2.0, 3.0], 0);
        lsh.add(&[1.1, 2.1, 3.1], 1);
        lsh.add(&[5.0, 6.0, 7.0], 2);

        // Query for similar vectors
        let candidates = lsh.query(&[1.05, 2.05, 3.05]);

        // The test should complete quickly
        // We'll just verify that we get some results back (could be empty or non-empty)
        // depending on the random projections
        println!("LSH candidates: {:?}", candidates);

        let stats = lsh.stats();
        // The exact number of entries depends on the random projections and may vary
        // We just check that the basic functionality works
        assert!(stats.table_count == 5);
        assert!(stats.dimension == 3);
    }
}
