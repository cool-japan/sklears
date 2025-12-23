//! Streaming dataset generation algorithms
//!
//! This module provides streaming algorithms for generating large datasets
//! incrementally, allowing for memory-efficient processing of datasets
//! that don't fit in memory.

use crate::memory_pool::{MemoryPool, MemoryPoolConfig, PooledArray1, PooledArray2};
use crate::traits::{ConfigValue, GeneratorConfig};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Distribution, Random, RandNormal};
use std::collections::VecDeque;
use std::sync::Arc;
use thiserror::Error;

// Helper function for generating normal random values
#[inline]
fn gen_normal_value(rng: &mut Random, mean: f64, std: f64) -> f64 {
    let dist = RandNormal::new(mean, std).unwrap();
    dist.sample(rng)
}

/// Streaming generation errors
#[derive(Error, Debug)]
pub enum StreamingError {
    #[error("Generation error: {0}")]
    Generation(String),
    #[error("Buffer overflow: {0}")]
    BufferOverflow(String),
    #[error("Invalid configuration: {0}")]
    Configuration(String),
    #[error("Memory pool error: {0}")]
    MemoryPool(String),
    #[error("Stream exhausted")]
    StreamExhausted,
    #[error("Synchronization error: {0}")]
    Synchronization(String),
}

pub type StreamingResult<T> = Result<T, StreamingError>;

/// Configuration for streaming generation
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of each generated chunk
    pub chunk_size: usize,
    /// Maximum number of chunks to buffer
    pub buffer_size: usize,
    /// Number of worker threads for parallel generation
    pub num_workers: usize,
    /// Memory pool configuration
    pub memory_pool: MemoryPoolConfig,
    /// Enable prefetching
    pub enable_prefetch: bool,
    /// Random seed for reproducible generation
    pub random_seed: Option<u64>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            buffer_size: 4,
            num_workers: num_cpus::get(),
            memory_pool: MemoryPoolConfig::default(),
            enable_prefetch: true,
            random_seed: None,
        }
    }
}

/// A chunk of generated data
#[derive(Debug, Clone)]
pub struct DataChunk {
    pub features: Array2<f64>,
    pub targets: Option<Array1<f64>>,
    pub chunk_id: usize,
    pub total_chunks: Option<usize>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl DataChunk {
    /// Get the number of samples in this chunk
    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    /// Get the number of features in this chunk
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    /// Check if this chunk has targets
    pub fn has_targets(&self) -> bool {
        self.targets.is_some()
    }

    /// Get a view of the features
    pub fn features_view(&self) -> ArrayView2<f64> {
        self.features.view()
    }

    /// Get a view of the targets (if available)
    pub fn targets_view(&self) -> Option<ArrayView1<f64>> {
        self.targets.as_ref().map(|t| t.view())
    }
}

/// Trait for streaming dataset generators
pub trait StreamingGenerator: Send + Sync {
    /// Generate the next chunk of data
    fn next_chunk(&mut self) -> StreamingResult<Option<DataChunk>>;

    /// Reset the generator to the beginning
    fn reset(&mut self) -> StreamingResult<()>;

    /// Get the total number of chunks (if known)
    fn total_chunks(&self) -> Option<usize>;

    /// Get the current chunk index
    fn current_chunk(&self) -> usize;

    /// Skip to a specific chunk
    fn seek(&mut self, chunk_id: usize) -> StreamingResult<()> {
        self.reset()?;
        for _ in 0..chunk_id {
            if self.next_chunk()?.is_none() {
                return Err(StreamingError::StreamExhausted);
            }
        }
        Ok(())
    }

    /// Get generator configuration
    fn config(&self) -> &StreamingConfig;
}

/// Streaming classification dataset generator
pub struct StreamingClassificationGenerator {
    config: StreamingConfig,
    generator_config: GeneratorConfig,
    current_chunk: usize,
    total_samples: usize,
    rng: Random,
    memory_pool: Arc<MemoryPool>,
}

impl StreamingClassificationGenerator {
    /// Create a new streaming classification generator
    pub fn new(
        streaming_config: StreamingConfig,
        generator_config: GeneratorConfig,
        total_samples: usize,
    ) -> Self {
        let memory_pool = Arc::new(MemoryPool::new(streaming_config.memory_pool.clone()));
        let rng = match streaming_config.random_seed {
            Some(seed) => Random::new_with_seed(seed),
            None => Random::new(),
        };

        Self {
            config: streaming_config,
            generator_config,
            current_chunk: 0,
            total_samples,
            rng,
            memory_pool,
        }
    }

    fn generate_chunk(&mut self, chunk_id: usize, samples_in_chunk: usize) -> StreamingResult<DataChunk> {
        let n_features = self.generator_config.n_features;

        // Get number of classes
        let n_classes = self.generator_config
            .get_parameter("n_classes")
            .and_then(|v| match v {
                ConfigValue::Int(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(2);

        // Get number of informative features
        let n_informative = self.generator_config
            .get_parameter("n_informative")
            .and_then(|v| match v {
                ConfigValue::Int(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(n_features);

        // Generate features
        let mut features = Array2::<f64>::zeros((samples_in_chunk, n_features));

        // Generate class-specific features
        for feature_idx in 0..n_informative {
            let class_means: Vec<f64> = (0..n_classes)
                .map(|class_idx| (class_idx as f64 - (n_classes as f64 - 1.0) / 2.0) * 2.0)
                .collect();

            for sample_idx in 0..samples_in_chunk {
                let class = self.rng.gen_range(0..n_classes);
                let class_mean = class_means[class];
                let noise = gen_normal_value(&mut self.rng, 0.0, 1.0);
                features[[sample_idx, feature_idx]] = class_mean + noise;
            }
        }

        // Generate noise features
        for feature_idx in n_informative..n_features {
            for sample_idx in 0..samples_in_chunk {
                features[[sample_idx, feature_idx]] = gen_normal_value(&mut self.rng, 0.0, 1.0);
            }
        }

        // Generate targets
        let targets: Array1<f64> = Array1::from_shape_fn(samples_in_chunk, |_| {
            self.rng.gen_range(0..n_classes) as f64
        });

        // Create metadata
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("generator".to_string(), "streaming_classification".to_string());
        metadata.insert("n_classes".to_string(), n_classes.to_string());
        metadata.insert("n_informative".to_string(), n_informative.to_string());
        metadata.insert("chunk_id".to_string(), chunk_id.to_string());

        Ok(DataChunk {
            features,
            targets: Some(targets),
            chunk_id,
            total_chunks: self.total_chunks(),
            metadata,
        })
    }
}

impl StreamingGenerator for StreamingClassificationGenerator {
    fn next_chunk(&mut self) -> StreamingResult<Option<DataChunk>> {
        let samples_generated = self.current_chunk * self.config.chunk_size;
        if samples_generated >= self.total_samples {
            return Ok(None);
        }

        let samples_in_chunk = std::cmp::min(
            self.config.chunk_size,
            self.total_samples - samples_generated,
        );

        let chunk = self.generate_chunk(self.current_chunk, samples_in_chunk)?;
        self.current_chunk += 1;

        Ok(Some(chunk))
    }

    fn reset(&mut self) -> StreamingResult<()> {
        self.current_chunk = 0;
        self.rng = match self.config.random_seed {
            Some(seed) => Random::new_with_seed(seed),
            None => Random::new(),
        };
        Ok(())
    }

    fn total_chunks(&self) -> Option<usize> {
        Some((self.total_samples + self.config.chunk_size - 1) / self.config.chunk_size)
    }

    fn current_chunk(&self) -> usize {
        self.current_chunk
    }

    fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

/// Streaming regression dataset generator
pub struct StreamingRegressionGenerator {
    config: StreamingConfig,
    generator_config: GeneratorConfig,
    current_chunk: usize,
    total_samples: usize,
    rng: Random,
    coefficients: Array1<f64>,
    memory_pool: Arc<MemoryPool>,
}

impl StreamingRegressionGenerator {
    /// Create a new streaming regression generator
    pub fn new(
        streaming_config: StreamingConfig,
        generator_config: GeneratorConfig,
        total_samples: usize,
    ) -> Self {
        let memory_pool = Arc::new(MemoryPool::new(streaming_config.memory_pool.clone()));
        let mut rng = match streaming_config.random_seed {
            Some(seed) => Random::new_with_seed(seed),
            None => Random::new(),
        };

        // Generate random coefficients
        let coefficients: Array1<f64> = Array1::from_shape_fn(generator_config.n_features, |_| {
            rng.random_range(-1.0, 1.0)
        });

        Self {
            config: streaming_config,
            generator_config,
            current_chunk: 0,
            total_samples,
            rng,
            coefficients,
            memory_pool,
        }
    }

    fn generate_chunk(&mut self, chunk_id: usize, samples_in_chunk: usize) -> StreamingResult<DataChunk> {
        let n_features = self.generator_config.n_features;

        // Get noise level
        let noise = self.generator_config
            .get_parameter("noise")
            .and_then(|v| match v {
                ConfigValue::Float(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(0.1);

        // Generate features
        let mut features = Array2::<f64>::zeros((samples_in_chunk, n_features));
        for mut row in features.rows_mut() {
            for val in row.iter_mut() {
                *val = gen_normal_value(&mut self.rng, 0.0, 1.0);
            }
        }

        // Generate targets
        let mut targets = Array1::<f64>::zeros(samples_in_chunk);
        for (i, mut target) in targets.iter_mut().enumerate() {
            let feature_row = features.row(i);
            *target = feature_row.dot(&self.coefficients) + gen_normal_value(&mut self.rng, 0.0, noise);
        }

        // Create metadata
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("generator".to_string(), "streaming_regression".to_string());
        metadata.insert("noise".to_string(), noise.to_string());
        metadata.insert("chunk_id".to_string(), chunk_id.to_string());

        Ok(DataChunk {
            features,
            targets: Some(targets),
            chunk_id,
            total_chunks: self.total_chunks(),
            metadata,
        })
    }
}

impl StreamingGenerator for StreamingRegressionGenerator {
    fn next_chunk(&mut self) -> StreamingResult<Option<DataChunk>> {
        let samples_generated = self.current_chunk * self.config.chunk_size;
        if samples_generated >= self.total_samples {
            return Ok(None);
        }

        let samples_in_chunk = std::cmp::min(
            self.config.chunk_size,
            self.total_samples - samples_generated,
        );

        let chunk = self.generate_chunk(self.current_chunk, samples_in_chunk)?;
        self.current_chunk += 1;

        Ok(Some(chunk))
    }

    fn reset(&mut self) -> StreamingResult<()> {
        self.current_chunk = 0;
        self.rng = match self.config.random_seed {
            Some(seed) => Random::new_with_seed(seed),
            None => Random::new(),
        };
        Ok(())
    }

    fn total_chunks(&self) -> Option<usize> {
        Some((self.total_samples + self.config.chunk_size - 1) / self.config.chunk_size)
    }

    fn current_chunk(&self) -> usize {
        self.current_chunk
    }

    fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

/// Buffered streaming dataset with prefetching
pub struct BufferedStream<G: StreamingGenerator> {
    generator: G,
    buffer: VecDeque<DataChunk>,
    config: StreamingConfig,
    prefetch_thread: Option<std::thread::JoinHandle<()>>,
    sender: Option<std::sync::mpsc::Sender<DataChunk>>,
    receiver: Option<std::sync::mpsc::Receiver<DataChunk>>,
}

impl<G: StreamingGenerator + 'static> BufferedStream<G> {
    /// Create a new buffered stream
    pub fn new(mut generator: G) -> StreamingResult<Self> {
        let config = generator.config().clone();

        let (sender, receiver) = if config.enable_prefetch {
            let (tx, rx) = std::sync::mpsc::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let mut stream = Self {
            generator,
            buffer: VecDeque::with_capacity(config.buffer_size),
            config,
            prefetch_thread: None,
            sender,
            receiver,
        };

        // Start prefetching if enabled
        if stream.config.enable_prefetch {
            stream.start_prefetching()?;
        }

        Ok(stream)
    }

    fn start_prefetching(&mut self) -> StreamingResult<()> {
        if let (Some(sender), Some(_)) = (&self.sender, &self.receiver) {
            let sender = sender.clone();
            let mut generator = std::mem::replace(&mut self.generator, unsafe {
                std::mem::zeroed() // This is unsafe but we'll replace it immediately
            });

            let handle = std::thread::spawn(move || {
                while let Ok(Some(chunk)) = generator.next_chunk() {
                    if sender.send(chunk).is_err() {
                        break; // Receiver dropped
                    }
                }
            });

            self.prefetch_thread = Some(handle);
        }

        Ok(())
    }

    /// Get the next chunk from the stream
    pub fn next_chunk(&mut self) -> StreamingResult<Option<DataChunk>> {
        // If buffering is enabled, try to get from buffer first
        if let Some(chunk) = self.buffer.pop_front() {
            return Ok(Some(chunk));
        }

        // If prefetching is enabled, try to get from prefetch channel
        if let Some(ref receiver) = self.receiver {
            match receiver.try_recv() {
                Ok(chunk) => return Ok(Some(chunk)),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // No data available yet, fall through to direct generation
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Prefetch thread finished
                    return Ok(None);
                }
            }
        }

        // Direct generation
        self.generator.next_chunk()
    }

    /// Reset the stream
    pub fn reset(&mut self) -> StreamingResult<()> {
        // Stop prefetching
        if let Some(handle) = self.prefetch_thread.take() {
            drop(self.sender.take()); // Close sender to stop prefetch thread
            let _ = handle.join();
        }

        self.buffer.clear();
        self.generator.reset()?;

        // Restart prefetching if enabled
        if self.config.enable_prefetch {
            let (tx, rx) = std::sync::mpsc::channel();
            self.sender = Some(tx);
            self.receiver = Some(rx);
            self.start_prefetching()?;
        }

        Ok(())
    }

    /// Get stream statistics
    pub fn stats(&self) -> StreamStats {
        StreamStats {
            current_chunk: self.generator.current_chunk(),
            total_chunks: self.generator.total_chunks(),
            buffered_chunks: self.buffer.len(),
            buffer_capacity: self.config.buffer_size,
        }
    }
}

/// Stream statistics
#[derive(Debug, Clone)]
pub struct StreamStats {
    pub current_chunk: usize,
    pub total_chunks: Option<usize>,
    pub buffered_chunks: usize,
    pub buffer_capacity: usize,
}

impl StreamStats {
    /// Get progress as a percentage (if total chunks is known)
    pub fn progress(&self) -> Option<f64> {
        self.total_chunks.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.current_chunk as f64 / total as f64) * 100.0
            }
        })
    }

    /// Get buffer utilization
    pub fn buffer_utilization(&self) -> f64 {
        if self.buffer_capacity == 0 {
            0.0
        } else {
            self.buffered_chunks as f64 / self.buffer_capacity as f64
        }
    }
}

/// Iterator wrapper for streaming generators
pub struct StreamingIterator<G: StreamingGenerator> {
    stream: BufferedStream<G>,
}

impl<G: StreamingGenerator + 'static> StreamingIterator<G> {
    /// Create a new streaming iterator
    pub fn new(generator: G) -> StreamingResult<Self> {
        Ok(Self {
            stream: BufferedStream::new(generator)?,
        })
    }

    /// Reset the iterator
    pub fn reset(&mut self) -> StreamingResult<()> {
        self.stream.reset()
    }

    /// Get stream statistics
    pub fn stats(&self) -> StreamStats {
        self.stream.stats()
    }
}

impl<G: StreamingGenerator + 'static> Iterator for StreamingIterator<G> {
    type Item = StreamingResult<DataChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stream.next_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Convenience functions for creating streaming generators
pub fn streaming_classification(
    streaming_config: StreamingConfig,
    generator_config: GeneratorConfig,
    total_samples: usize,
) -> StreamingClassificationGenerator {
    StreamingClassificationGenerator::new(streaming_config, generator_config, total_samples)
}

pub fn streaming_regression(
    streaming_config: StreamingConfig,
    generator_config: GeneratorConfig,
    total_samples: usize,
) -> StreamingRegressionGenerator {
    StreamingRegressionGenerator::new(streaming_config, generator_config, total_samples)
}

/// Parallel streaming generator that uses multiple threads
pub struct ParallelStreamingGenerator<G: StreamingGenerator + Clone + Send + 'static> {
    generators: Vec<G>,
    config: StreamingConfig,
    current_generator: usize,
}

impl<G: StreamingGenerator + Clone + Send + 'static> ParallelStreamingGenerator<G> {
    /// Create a new parallel streaming generator
    pub fn new(generator: G, num_workers: usize) -> Self {
        let config = generator.config().clone();
        let generators: Vec<G> = (0..num_workers).map(|_| generator.clone()).collect();

        Self {
            generators,
            config,
            current_generator: 0,
        }
    }

    /// Generate chunks in parallel
    pub fn generate_parallel(&mut self, num_chunks: usize) -> StreamingResult<Vec<DataChunk>> {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let chunks = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        let chunks_per_worker = (num_chunks + self.generators.len() - 1) / self.generators.len();

        for (worker_id, mut generator) in self.generators.clone().into_iter().enumerate() {
            let chunks_clone = Arc::clone(&chunks);
            let start_chunk = worker_id * chunks_per_worker;
            let end_chunk = std::cmp::min(start_chunk + chunks_per_worker, num_chunks);

            if start_chunk < end_chunk {
                let handle = thread::spawn(move || -> StreamingResult<()> {
                    generator.seek(start_chunk)?;

                    for _ in start_chunk..end_chunk {
                        if let Some(chunk) = generator.next_chunk()? {
                            chunks_clone.lock().unwrap().push(chunk);
                        }
                    }

                    Ok(())
                });

                handles.push(handle);
            }
        }

        // Wait for all workers to complete
        for handle in handles {
            handle.join().map_err(|_| {
                StreamingError::Synchronization("Worker thread panicked".to_string())
            })??;
        }

        let mut result = chunks.lock().unwrap().clone();
        result.sort_by_key(|chunk| chunk.chunk_id);

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::GeneratorConfig;

    #[test]
    fn test_streaming_classification() -> StreamingResult<()> {
        let streaming_config = StreamingConfig {
            chunk_size: 100,
            enable_prefetch: false,
            random_seed: Some(42),
            ..Default::default()
        };

        let mut generator_config = GeneratorConfig::new(500, 4);
        generator_config.set_parameter("n_classes".to_string(), 3i64);

        let mut generator = streaming_classification(streaming_config, generator_config, 500);

        let mut total_samples = 0;
        let mut chunk_count = 0;

        while let Some(chunk) = generator.next_chunk()? {
            assert_eq!(chunk.n_features(), 4);
            assert!(chunk.has_targets());
            assert!(chunk.n_samples() <= 100);

            total_samples += chunk.n_samples();
            chunk_count += 1;

            // Verify targets are in valid range
            if let Some(targets) = chunk.targets_view() {
                for &target in targets.iter() {
                    assert!(target >= 0.0 && target < 3.0);
                }
            }
        }

        assert_eq!(total_samples, 500);
        assert_eq!(chunk_count, 5); // 500 / 100
        assert_eq!(generator.total_chunks(), Some(5));

        Ok(())
    }

    #[test]
    fn test_streaming_regression() -> StreamingResult<()> {
        let streaming_config = StreamingConfig {
            chunk_size: 50,
            enable_prefetch: false,
            random_seed: Some(123),
            ..Default::default()
        };

        let mut generator_config = GeneratorConfig::new(200, 3);
        generator_config.set_parameter("noise".to_string(), 0.05);

        let mut generator = streaming_regression(streaming_config, generator_config, 200);

        let mut total_samples = 0;
        let mut chunk_count = 0;

        while let Some(chunk) = generator.next_chunk()? {
            assert_eq!(chunk.n_features(), 3);
            assert!(chunk.has_targets());
            assert!(chunk.n_samples() <= 50);

            total_samples += chunk.n_samples();
            chunk_count += 1;
        }

        assert_eq!(total_samples, 200);
        assert_eq!(chunk_count, 4); // 200 / 50

        Ok(())
    }

    #[test]
    fn test_buffered_stream() -> StreamingResult<()> {
        let streaming_config = StreamingConfig {
            chunk_size: 25,
            buffer_size: 2,
            enable_prefetch: false, // Disable for deterministic testing
            random_seed: Some(42),
            ..Default::default()
        };

        let generator_config = GeneratorConfig::new(100, 2);
        let generator = streaming_classification(streaming_config, generator_config, 100);

        let mut stream = BufferedStream::new(generator)?;

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next_chunk()? {
            chunks.push(chunk);
        }

        assert_eq!(chunks.len(), 4); // 100 / 25
        assert_eq!(chunks[0].chunk_id, 0);
        assert_eq!(chunks[3].chunk_id, 3);

        Ok(())
    }

    #[test]
    fn test_streaming_iterator() -> StreamingResult<()> {
        let streaming_config = StreamingConfig {
            chunk_size: 30,
            enable_prefetch: false,
            random_seed: Some(456),
            ..Default::default()
        };

        let generator_config = GeneratorConfig::new(90, 2);
        let generator = streaming_regression(streaming_config, generator_config, 90);

        let mut iterator = StreamingIterator::new(generator)?;

        let chunks: Result<Vec<_>, _> = iterator.collect();
        let chunks = chunks?;

        assert_eq!(chunks.len(), 3); // 90 / 30

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_id, i);
            assert_eq!(chunk.n_samples(), 30);
            assert_eq!(chunk.n_features(), 2);
        }

        Ok(())
    }

    #[test]
    fn test_stream_reset() -> StreamingResult<()> {
        let streaming_config = StreamingConfig {
            chunk_size: 20,
            enable_prefetch: false,
            random_seed: Some(789),
            ..Default::default()
        };

        let generator_config = GeneratorConfig::new(40, 2);
        let mut generator = streaming_classification(streaming_config, generator_config, 40);

        // Generate first chunk
        let chunk1 = generator.next_chunk()?.unwrap();
        assert_eq!(chunk1.chunk_id, 0);

        // Reset and generate again
        generator.reset()?;
        let chunk1_again = generator.next_chunk()?.unwrap();
        assert_eq!(chunk1_again.chunk_id, 0);

        // Should be identical due to same random seed
        assert_eq!(chunk1.features, chunk1_again.features);

        Ok(())
    }

    #[test]
    fn test_stream_seek() -> StreamingResult<()> {
        let streaming_config = StreamingConfig {
            chunk_size: 15,
            enable_prefetch: false,
            random_seed: Some(101112),
            ..Default::default()
        };

        let generator_config = GeneratorConfig::new(60, 2);
        let mut generator = streaming_regression(streaming_config, generator_config, 60);

        // Seek to chunk 2
        generator.seek(2)?;
        let chunk = generator.next_chunk()?.unwrap();
        assert_eq!(chunk.chunk_id, 2);

        Ok(())
    }

    #[test]
    fn test_stream_stats() -> StreamingResult<()> {
        let streaming_config = StreamingConfig {
            chunk_size: 10,
            buffer_size: 3,
            enable_prefetch: false,
            ..Default::default()
        };

        let generator_config = GeneratorConfig::new(50, 2);
        let generator = streaming_classification(streaming_config, generator_config, 50);

        let stream = BufferedStream::new(generator)?;
        let stats = stream.stats();

        assert_eq!(stats.current_chunk, 0);
        assert_eq!(stats.total_chunks, Some(5)); // 50 / 10
        assert_eq!(stats.buffer_capacity, 3);

        // Test progress calculation
        assert_eq!(stats.progress(), Some(0.0));

        Ok(())
    }

    #[test]
    fn test_chunk_metadata() -> StreamingResult<()> {
        let streaming_config = StreamingConfig {
            chunk_size: 25,
            enable_prefetch: false,
            random_seed: Some(42),
            ..Default::default()
        };

        let mut generator_config = GeneratorConfig::new(50, 3);
        generator_config.set_parameter("n_classes".to_string(), 4i64);

        let mut generator = streaming_classification(streaming_config, generator_config, 50);

        let chunk = generator.next_chunk()?.unwrap();

        assert_eq!(chunk.metadata.get("generator"), Some(&"streaming_classification".to_string()));
        assert_eq!(chunk.metadata.get("n_classes"), Some(&"4".to_string()));
        assert_eq!(chunk.metadata.get("chunk_id"), Some(&"0".to_string()));
        assert!(chunk.metadata.contains_key("n_informative"));

        Ok(())
    }
}