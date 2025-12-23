//! Memory-efficient kernel approximation methods
//!
//! This module provides memory-efficient implementations of kernel approximation methods
//! that can handle large datasets that don't fit entirely in memory.

use crate::nystroem::Kernel;
use crate::{Nystroem, RBFSampler, SamplingStrategy};
use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform},
};
use std::sync::{Arc, Mutex};

/// Configuration for memory-efficient operations
#[derive(Debug, Clone)]
/// MemoryConfig
pub struct MemoryConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Number of parallel workers
    pub n_workers: usize,
    /// Enable disk caching for intermediate results
    pub enable_disk_cache: bool,
    /// Temporary directory for disk cache
    pub temp_dir: String,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            chunk_size: 10000,
            n_workers: num_cpus::get(),
            enable_disk_cache: false,
            temp_dir: "/tmp".to_string(),
        }
    }
}

/// Memory-efficient RBF sampler with chunked processing
#[derive(Debug, Clone)]
/// MemoryEfficientRBFSampler
pub struct MemoryEfficientRBFSampler {
    n_components: usize,
    gamma: f64,
    config: MemoryConfig,
    random_seed: Option<u64>,
}

impl MemoryEfficientRBFSampler {
    /// Create a new memory-efficient RBF sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            gamma: 1.0,
            config: MemoryConfig::default(),
            random_seed: None,
        }
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set memory configuration
    pub fn config(mut self, config: MemoryConfig) -> Self {
        self.config = config;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Process data in chunks
    pub fn transform_chunked(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let _n_features = x.ncols();
        let chunk_size = self.config.chunk_size.min(n_samples);

        // Initialize output array
        let mut output = Array2::zeros((n_samples, self.n_components));

        // Create RBF sampler for consistent random features
        let rbf_sampler = RBFSampler::new(self.n_components).gamma(self.gamma);
        let fitted_sampler = rbf_sampler.fit(x, &())?;

        // Process in chunks
        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            let chunk = x.slice(s![chunk_start..chunk_end, ..]);

            // Transform chunk
            let chunk_transformed = fitted_sampler.transform(&chunk.to_owned())?;

            // Store result
            output
                .slice_mut(s![chunk_start..chunk_end, ..])
                .assign(&chunk_transformed);
        }

        Ok(output)
    }

    /// Parallel chunked processing
    pub fn transform_chunked_parallel(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let chunk_size = self.config.chunk_size.min(n_samples);

        // Create RBF sampler for consistent random features
        let rbf_sampler = RBFSampler::new(self.n_components).gamma(self.gamma);
        let fitted_sampler = Arc::new(rbf_sampler.fit(x, &())?);

        // Create chunks
        let chunks: Vec<_> = (0..n_samples)
            .step_by(chunk_size)
            .map(|start| {
                let end = (start + chunk_size).min(n_samples);
                (start, end)
            })
            .collect();

        // Process chunks in parallel
        let results: Result<Vec<_>> = chunks
            .par_iter()
            .map(|&(start, end)| {
                let chunk = x.slice(s![start..end, ..]).to_owned();
                fitted_sampler
                    .transform(&chunk)
                    .map(|result| (start, result))
            })
            .collect();

        let results = results?;

        // Combine results
        let mut output = Array2::zeros((n_samples, self.n_components));
        for (start, chunk_result) in results {
            let end = start + chunk_result.nrows();
            output.slice_mut(s![start..end, ..]).assign(&chunk_result);
        }

        Ok(output)
    }
}

/// Fitted memory-efficient RBF sampler
pub struct FittedMemoryEfficientRBFSampler {
    random_weights: Array2<f64>,
    random_offset: Array1<f64>,
    gamma: f64,
    config: MemoryConfig,
}

impl Fit<Array2<f64>, ()> for MemoryEfficientRBFSampler {
    type Fitted = FittedMemoryEfficientRBFSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let n_features = x.ncols();

        let mut rng = if let Some(seed) = self.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_seed(thread_rng().gen())
        };

        // Generate random weights and offsets
        let random_weights = Array2::from_shape_fn((self.n_components, n_features), |_| {
            rng.sample(RandNormal::new(0.0, (2.0 * self.gamma).sqrt()).unwrap())
        });

        let random_offset = Array1::from_shape_fn(self.n_components, |_| {
            rng.sample(RandUniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap())
        });

        Ok(FittedMemoryEfficientRBFSampler {
            random_weights,
            random_offset,
            gamma: self.gamma,
            config: self.config.clone(),
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedMemoryEfficientRBFSampler {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let chunk_size = self.config.chunk_size.min(n_samples);

        if n_samples <= chunk_size {
            // Small dataset, process normally
            self.transform_small(x)
        } else {
            // Large dataset, use chunked processing
            self.transform_chunked(x)
        }
    }
}

impl FittedMemoryEfficientRBFSampler {
    fn transform_small(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let projection = x.dot(&self.random_weights.t());
        let scaled_projection = projection + &self.random_offset;

        let normalization = (2.0 / self.random_weights.nrows() as f64).sqrt();
        Ok(scaled_projection.mapv(|v| v.cos() * normalization))
    }

    fn transform_chunked(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let chunk_size = self.config.chunk_size;
        let mut output = Array2::zeros((n_samples, self.random_weights.nrows()));

        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            let chunk = x.slice(s![chunk_start..chunk_end, ..]);

            let chunk_transformed = self.transform_small(&chunk.to_owned())?;
            output
                .slice_mut(s![chunk_start..chunk_end, ..])
                .assign(&chunk_transformed);
        }

        Ok(output)
    }
}

/// Memory-efficient Nyström approximation
#[derive(Debug, Clone)]
/// MemoryEfficientNystroem
pub struct MemoryEfficientNystroem {
    n_components: usize,
    kernel: String,
    gamma: Option<f64>,
    degree: Option<i32>,
    coef0: Option<f64>,
    sampling: SamplingStrategy,
    config: MemoryConfig,
    random_seed: Option<u64>,
}

impl MemoryEfficientNystroem {
    /// Create a new memory-efficient Nyström approximation
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            kernel: "rbf".to_string(),
            gamma: None,
            degree: None,
            coef0: None,
            sampling: SamplingStrategy::Random,
            config: MemoryConfig::default(),
            random_seed: None,
        }
    }

    /// Set kernel type
    pub fn kernel(mut self, kernel: &str) -> Self {
        self.kernel = kernel.to_string();
        self
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set sampling strategy
    pub fn sampling(mut self, sampling: SamplingStrategy) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set memory configuration
    pub fn config(mut self, config: MemoryConfig) -> Self {
        self.config = config;
        self
    }

    /// Out-of-core training for large datasets
    pub fn fit_incremental(
        &self,
        x_chunks: Vec<Array2<f64>>,
    ) -> Result<FittedMemoryEfficientNystroem> {
        // Collect representative samples from all chunks
        let mut representative_samples = Vec::new();
        let samples_per_chunk = self.n_components / x_chunks.len().max(1);

        for chunk in &x_chunks {
            let n_samples = chunk.nrows().min(samples_per_chunk);
            if n_samples > 0 {
                let indices: Vec<usize> = (0..chunk.nrows()).collect();
                let selected_indices = &indices[..n_samples];

                for &idx in selected_indices {
                    representative_samples.push(chunk.row(idx).to_owned());
                }
            }
        }

        if representative_samples.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No samples found in chunks".to_string(),
            ));
        }

        // Create combined dataset from representative samples
        let n_selected = representative_samples.len().min(self.n_components);
        let n_features = representative_samples[0].len();
        let mut combined_data = Array2::zeros((n_selected, n_features));

        for (i, sample) in representative_samples.iter().take(n_selected).enumerate() {
            combined_data.row_mut(i).assign(sample);
        }

        // Fit standard Nyström on representative samples
        let kernel = match self.kernel.as_str() {
            "rbf" => Kernel::Rbf {
                gamma: self.gamma.unwrap_or(1.0),
            },
            "linear" => Kernel::Linear,
            "polynomial" => Kernel::Polynomial {
                gamma: self.gamma.unwrap_or(1.0),
                degree: self.degree.unwrap_or(3) as u32,
                coef0: self.coef0.unwrap_or(1.0),
            },
            _ => Kernel::Rbf { gamma: 1.0 }, // default
        };
        let nystroem = Nystroem::new(kernel, n_selected).sampling_strategy(self.sampling.clone());

        let fitted_nystroem = nystroem.fit(&combined_data, &())?;

        Ok(FittedMemoryEfficientNystroem {
            fitted_nystroem,
            config: self.config.clone(),
        })
    }
}

/// Fitted memory-efficient Nyström approximation
pub struct FittedMemoryEfficientNystroem {
    fitted_nystroem: crate::nystroem::Nystroem<Trained>,
    config: MemoryConfig,
}

impl Fit<Array2<f64>, ()> for MemoryEfficientNystroem {
    type Fitted = FittedMemoryEfficientNystroem;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let kernel = match self.kernel.as_str() {
            "rbf" => Kernel::Rbf {
                gamma: self.gamma.unwrap_or(1.0),
            },
            "linear" => Kernel::Linear,
            "polynomial" => Kernel::Polynomial {
                gamma: self.gamma.unwrap_or(1.0),
                degree: self.degree.unwrap_or(3) as u32,
                coef0: self.coef0.unwrap_or(1.0),
            },
            _ => Kernel::Rbf { gamma: 1.0 }, // default
        };
        let nystroem =
            Nystroem::new(kernel, self.n_components).sampling_strategy(self.sampling.clone());

        let fitted_nystroem = nystroem.fit(x, &())?;

        Ok(FittedMemoryEfficientNystroem {
            fitted_nystroem,
            config: self.config.clone(),
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedMemoryEfficientNystroem {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let chunk_size = self.config.chunk_size;

        if n_samples <= chunk_size {
            // Small dataset, process normally
            self.fitted_nystroem.transform(x)
        } else {
            // Large dataset, use chunked processing
            self.transform_chunked(x)
        }
    }
}

impl FittedMemoryEfficientNystroem {
    fn transform_chunked(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let chunk_size = self.config.chunk_size;
        let n_components = self
            .fitted_nystroem
            .transform(&x.slice(s![0..1, ..]).to_owned())?
            .ncols();

        let mut output = Array2::zeros((n_samples, n_components));

        for chunk_start in (0..n_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_samples);
            let chunk = x.slice(s![chunk_start..chunk_end, ..]);

            let chunk_transformed = self.fitted_nystroem.transform(&chunk.to_owned())?;
            output
                .slice_mut(s![chunk_start..chunk_end, ..])
                .assign(&chunk_transformed);
        }

        Ok(output)
    }
}

/// Memory usage monitoring utilities
pub struct MemoryMonitor {
    max_memory_bytes: usize,
    current_usage: Arc<Mutex<usize>>,
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new(max_memory_bytes: usize) -> Self {
        Self {
            max_memory_bytes,
            current_usage: Arc::new(Mutex::new(0)),
        }
    }

    /// Check if we can allocate more memory
    pub fn can_allocate(&self, bytes: usize) -> bool {
        let current = *self.current_usage.lock().unwrap();
        current + bytes <= self.max_memory_bytes
    }

    /// Allocate memory (tracking purposes)
    pub fn allocate(&self, bytes: usize) -> Result<()> {
        let mut current = self.current_usage.lock().unwrap();
        if *current + bytes > self.max_memory_bytes {
            return Err(SklearsError::InvalidInput(format!(
                "Memory limit exceeded: {} + {} > {}",
                *current, bytes, self.max_memory_bytes
            )));
        }
        *current += bytes;
        Ok(())
    }

    /// Deallocate memory
    pub fn deallocate(&self, bytes: usize) {
        let mut current = self.current_usage.lock().unwrap();
        *current = current.saturating_sub(bytes);
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        *self.current_usage.lock().unwrap()
    }

    /// Get memory usage percentage
    pub fn usage_percentage(&self) -> f64 {
        let current = *self.current_usage.lock().unwrap();
        (current as f64 / self.max_memory_bytes as f64) * 100.0
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_memory_efficient_rbf_sampler() {
        let x = Array2::from_shape_vec((100, 10), (0..1000).map(|i| i as f64).collect()).unwrap();

        let sampler = MemoryEfficientRBFSampler::new(50)
            .gamma(0.1)
            .config(MemoryConfig {
                chunk_size: 30,
                ..Default::default()
            });

        let fitted = sampler.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[100, 50]);

        // Test chunked processing gives same results as small dataset
        let small_x = x.slice(s![0..10, ..]).to_owned();
        let small_transformed = fitted.transform(&small_x).unwrap();
        let chunked_transformed = transformed.slice(s![0..10, ..]);

        assert_abs_diff_eq!(small_transformed, chunked_transformed, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_efficient_rbf_chunked_parallel() {
        let x =
            Array2::from_shape_vec((200, 5), (0..1000).map(|i| i as f64 * 0.1).collect()).unwrap();

        let sampler = MemoryEfficientRBFSampler::new(30)
            .gamma(1.0)
            .config(MemoryConfig {
                chunk_size: 50,
                n_workers: 2,
                ..Default::default()
            });

        let result = sampler.transform_chunked_parallel(&x).unwrap();
        assert_eq!(result.shape(), &[200, 30]);

        // Verify output is reasonable (not all zeros, not all same values)
        let mean_val = result.mean().unwrap();
        let std_val = result.std(0.0);
        assert!(mean_val.abs() < 0.5); // Should be roughly centered
        assert!(std_val > 0.1); // Should have some variance
    }

    #[test]
    fn test_memory_efficient_nystroem() {
        let x =
            Array2::from_shape_vec((80, 6), (0..480).map(|i| i as f64 * 0.01).collect()).unwrap();

        let nystroem = MemoryEfficientNystroem::new(20)
            .kernel("rbf")
            .gamma(0.5)
            .config(MemoryConfig {
                chunk_size: 25,
                ..Default::default()
            });

        let fitted = nystroem.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[80, 20]);
    }

    #[test]
    fn test_memory_efficient_nystroem_incremental() {
        // Create multiple chunks
        let chunk1 =
            Array2::from_shape_vec((30, 4), (0..120).map(|i| i as f64 * 0.1).collect()).unwrap();
        let chunk2 =
            Array2::from_shape_vec((40, 4), (120..280).map(|i| i as f64 * 0.1).collect()).unwrap();
        let chunk3 =
            Array2::from_shape_vec((30, 4), (280..400).map(|i| i as f64 * 0.1).collect()).unwrap();

        let chunks = vec![chunk1, chunk2.clone(), chunk3];

        let nystroem = MemoryEfficientNystroem::new(15)
            .kernel("rbf")
            .config(MemoryConfig {
                chunk_size: 20,
                ..Default::default()
            });

        let fitted = nystroem.fit_incremental(chunks).unwrap();
        let transformed = fitted.transform(&chunk2).unwrap();

        assert_eq!(transformed.shape(), &[40, 15]);
    }

    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::new(1000);

        assert!(monitor.can_allocate(500));
        assert!(monitor.allocate(500).is_ok());
        assert_eq!(monitor.current_usage(), 500);
        assert_eq!(monitor.usage_percentage(), 50.0);

        assert!(!monitor.can_allocate(600)); // Would exceed limit
        assert!(monitor.allocate(400).is_ok()); // Total = 900, still OK

        assert!(monitor.allocate(200).is_err()); // Would exceed limit

        monitor.deallocate(300);
        assert_eq!(monitor.current_usage(), 600);
        assert!(monitor.can_allocate(300));
    }

    #[test]
    fn test_memory_config() {
        let config = MemoryConfig::default();
        assert_eq!(config.max_memory_bytes, 1024 * 1024 * 1024);
        assert_eq!(config.chunk_size, 10000);
        assert!(config.n_workers > 0);

        let custom_config = MemoryConfig {
            max_memory_bytes: 512 * 1024 * 1024,
            chunk_size: 5000,
            n_workers: 4,
            enable_disk_cache: true,
            temp_dir: "/custom/temp".to_string(),
        };

        let sampler = MemoryEfficientRBFSampler::new(50).config(custom_config.clone());
        assert_eq!(sampler.config.max_memory_bytes, 512 * 1024 * 1024);
        assert_eq!(sampler.config.chunk_size, 5000);
        assert_eq!(sampler.config.n_workers, 4);
        assert!(sampler.config.enable_disk_cache);
        assert_eq!(sampler.config.temp_dir, "/custom/temp");
    }

    #[test]
    fn test_reproducibility() {
        let x =
            Array2::from_shape_vec((50, 8), (0..400).map(|i| i as f64 * 0.05).collect()).unwrap();

        let sampler1 = MemoryEfficientRBFSampler::new(20)
            .gamma(0.2)
            .random_seed(42);

        let sampler2 = MemoryEfficientRBFSampler::new(20)
            .gamma(0.2)
            .random_seed(42);

        let fitted1 = sampler1.fit(&x, &()).unwrap();
        let fitted2 = sampler2.fit(&x, &()).unwrap();

        let result1 = fitted1.transform(&x).unwrap();
        let result2 = fitted2.transform(&x).unwrap();

        assert_abs_diff_eq!(result1, result2, epsilon = 1e-10);
    }
}
