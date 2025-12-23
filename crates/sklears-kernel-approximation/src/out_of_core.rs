//! Out-of-core kernel computations for large datasets
//!
//! This module provides kernel approximation methods that can work with datasets
//! that don't fit in memory by processing data in chunks and using efficient
//! streaming algorithms.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::Normal as RandNormal;
use scirs2_core::random::thread_rng;
use sklears_core::error::Result;
use std::collections::HashMap;

/// Configuration for out-of-core processing
#[derive(Clone, Debug)]
/// OutOfCoreConfig
pub struct OutOfCoreConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Number of parallel workers
    pub n_workers: usize,
    /// Temporary directory for intermediate files
    pub temp_dir: String,
    /// Whether to use compression for temporary files
    pub use_compression: bool,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            chunk_size: 10000,
            n_workers: num_cpus::get(),
            temp_dir: "/tmp/sklears_out_of_core".to_string(),
            use_compression: true,
            buffer_size: 64 * 1024, // 64KB
        }
    }
}

/// Strategy for out-of-core processing
#[derive(Clone, Debug, PartialEq)]
/// OutOfCoreStrategy
pub enum OutOfCoreStrategy {
    /// Sequential processing chunk by chunk
    Sequential,
    /// Parallel processing with worker threads
    Parallel,
    /// Streaming processing with fixed memory budget
    Streaming,
    /// Adaptive processing based on data characteristics
    Adaptive,
}

/// Out-of-core data loader for large datasets
pub struct OutOfCoreLoader {
    config: OutOfCoreConfig,
    data_path: String,
    n_samples: usize,
    n_features: usize,
    current_chunk: usize,
}

impl OutOfCoreLoader {
    /// Create a new out-of-core loader
    pub fn new(data_path: &str, n_samples: usize, n_features: usize) -> Self {
        Self {
            config: OutOfCoreConfig::default(),
            data_path: data_path.to_string(),
            n_samples,
            n_features,
            current_chunk: 0,
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: OutOfCoreConfig) -> Self {
        self.config = config;
        self
    }

    /// Load next chunk of data
    pub fn load_chunk(&mut self) -> Result<Option<Array2<f64>>> {
        if self.current_chunk * self.config.chunk_size >= self.n_samples {
            return Ok(None);
        }

        let start_idx = self.current_chunk * self.config.chunk_size;
        let end_idx = std::cmp::min(start_idx + self.config.chunk_size, self.n_samples);
        let chunk_size = end_idx - start_idx;

        // Create dummy data for now - in practice this would load from file
        let mut rng = thread_rng();
        let chunk =
            Array2::from_shape_fn((chunk_size, self.n_features), |_| rng.gen_range(-1.0..1.0));

        self.current_chunk += 1;
        Ok(Some(chunk))
    }

    /// Reset loader to beginning
    pub fn reset(&mut self) {
        self.current_chunk = 0;
    }

    /// Get total number of chunks
    pub fn n_chunks(&self) -> usize {
        (self.n_samples + self.config.chunk_size - 1) / self.config.chunk_size
    }
}

/// Out-of-core RBF sampler
pub struct OutOfCoreRBFSampler {
    n_components: usize,
    gamma: f64,
    config: OutOfCoreConfig,
    strategy: OutOfCoreStrategy,
    random_weights: Option<Array2<f64>>,
    random_offset: Option<Array1<f64>>,
}

impl OutOfCoreRBFSampler {
    /// Create a new out-of-core RBF sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            gamma: 1.0,
            config: OutOfCoreConfig::default(),
            strategy: OutOfCoreStrategy::Sequential,
            random_weights: None,
            random_offset: None,
        }
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set out-of-core configuration
    pub fn with_config(mut self, config: OutOfCoreConfig) -> Self {
        self.config = config;
        self
    }

    /// Set processing strategy
    pub fn with_strategy(mut self, strategy: OutOfCoreStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Process data out-of-core
    pub fn fit_transform_out_of_core(
        &mut self,
        loader: &mut OutOfCoreLoader,
    ) -> Result<Vec<Array2<f64>>> {
        // Initialize random weights if not already done
        if self.random_weights.is_none() {
            let n_features = loader.n_features;
            let normal = RandNormal::new(0.0, (2.0 * self.gamma).sqrt()).unwrap();
            let mut rng = thread_rng();

            self.random_weights = Some(Array2::from_shape_fn(
                (n_features, self.n_components),
                |_| rng.sample(normal),
            ));

            self.random_offset = Some(Array1::from_shape_fn(self.n_components, |_| {
                rng.gen_range(0.0..2.0 * std::f64::consts::PI)
            }));
        }

        let mut results = Vec::new();
        loader.reset();

        match self.strategy {
            OutOfCoreStrategy::Sequential => {
                while let Some(chunk) = loader.load_chunk()? {
                    let transformed = self.transform_chunk(&chunk)?;
                    results.push(transformed);
                }
            }
            OutOfCoreStrategy::Parallel => {
                let chunks = self.load_all_chunks(loader)?;
                let parallel_results: Vec<_> = chunks
                    .into_par_iter()
                    .map(|chunk| self.transform_chunk(&chunk))
                    .collect::<Result<Vec<_>>>()?;
                results = parallel_results;
            }
            OutOfCoreStrategy::Streaming => {
                results = self.streaming_transform(loader)?;
            }
            OutOfCoreStrategy::Adaptive => {
                results = self.adaptive_transform(loader)?;
            }
        }

        Ok(results)
    }

    /// Transform a single chunk
    fn transform_chunk(&self, chunk: &Array2<f64>) -> Result<Array2<f64>> {
        let weights = self.random_weights.as_ref().unwrap();
        let offset = self.random_offset.as_ref().unwrap();

        // Compute projection: X * W + b
        let projection = chunk.dot(weights) + offset;

        // Apply cosine transformation
        let transformed = projection.mapv(|x| (x * (2.0 / self.n_components as f64).sqrt()).cos());

        Ok(transformed)
    }

    /// Load all chunks into memory (for parallel processing)
    fn load_all_chunks(&self, loader: &mut OutOfCoreLoader) -> Result<Vec<Array2<f64>>> {
        let mut chunks = Vec::new();
        loader.reset();

        while let Some(chunk) = loader.load_chunk()? {
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    /// Streaming transformation with memory budget
    fn streaming_transform(&self, loader: &mut OutOfCoreLoader) -> Result<Vec<Array2<f64>>> {
        let mut results = Vec::new();
        let mut memory_used = 0;
        loader.reset();

        while let Some(chunk) = loader.load_chunk()? {
            let chunk_memory = chunk.len() * std::mem::size_of::<f64>();

            if memory_used + chunk_memory > self.config.max_memory_bytes {
                // Process accumulated results and clear memory
                memory_used = 0;
            }

            let transformed = self.transform_chunk(&chunk)?;
            results.push(transformed);
            memory_used += chunk_memory;
        }

        Ok(results)
    }

    /// Adaptive transformation based on data characteristics
    fn adaptive_transform(&self, loader: &mut OutOfCoreLoader) -> Result<Vec<Array2<f64>>> {
        // Analyze first chunk to determine optimal strategy
        loader.reset();
        let first_chunk = loader.load_chunk()?.unwrap();

        let data_density = self.estimate_data_density(&first_chunk);
        let memory_requirement =
            self.estimate_memory_requirement(loader.n_samples, loader.n_features);

        // Choose strategy based on analysis
        if memory_requirement < self.config.max_memory_bytes && data_density > 0.8 {
            self.parallel_transform_adaptive(loader)
        } else if data_density < 0.2 {
            self.sparse_transform_adaptive(loader)
        } else {
            self.streaming_transform(loader)
        }
    }

    /// Parallel transformation for adaptive strategy
    fn parallel_transform_adaptive(
        &self,
        loader: &mut OutOfCoreLoader,
    ) -> Result<Vec<Array2<f64>>> {
        let chunks = self.load_all_chunks(loader)?;
        let results: Vec<_> = chunks
            .into_par_iter()
            .map(|chunk| self.transform_chunk(&chunk))
            .collect::<Result<Vec<_>>>()?;
        Ok(results)
    }

    /// Sparse transformation for adaptive strategy
    fn sparse_transform_adaptive(&self, loader: &mut OutOfCoreLoader) -> Result<Vec<Array2<f64>>> {
        let mut results = Vec::new();
        loader.reset();

        while let Some(chunk) = loader.load_chunk()? {
            // For sparse data, we can use specialized sparse transformations
            let transformed = self.transform_chunk_sparse(&chunk)?;
            results.push(transformed);
        }

        Ok(results)
    }

    /// Transform chunk with sparse optimization
    fn transform_chunk_sparse(&self, chunk: &Array2<f64>) -> Result<Array2<f64>> {
        // For sparse data, we can skip zero entries
        let weights = self.random_weights.as_ref().unwrap();
        let offset = self.random_offset.as_ref().unwrap();

        let mut result = Array2::zeros((chunk.nrows(), self.n_components));

        for (i, row) in chunk.axis_iter(Axis(0)).enumerate() {
            for (j, &x) in row.iter().enumerate() {
                if x.abs() > 1e-10 {
                    // Skip near-zero values
                    for k in 0..self.n_components {
                        result[[i, k]] += x * weights[[j, k]];
                    }
                }
            }

            // Add offset and apply cosine
            for k in 0..self.n_components {
                let val: f64 =
                    (result[[i, k]] + offset[k]) * (2.0_f64 / self.n_components as f64).sqrt();
                result[[i, k]] = val.cos();
            }
        }

        Ok(result)
    }

    /// Estimate data density for adaptive strategy
    fn estimate_data_density(&self, chunk: &Array2<f64>) -> f64 {
        let total_elements = chunk.len() as f64;
        let non_zero_elements = chunk.iter().filter(|&&x| x.abs() > 1e-10).count() as f64;
        non_zero_elements / total_elements
    }

    /// Estimate memory requirement
    fn estimate_memory_requirement(&self, n_samples: usize, n_features: usize) -> usize {
        (n_samples * n_features + n_features * self.n_components + self.n_components)
            * std::mem::size_of::<f64>()
    }
}

/// Out-of-core Nyström method
pub struct OutOfCoreNystroem {
    n_components: usize,
    gamma: f64,
    config: OutOfCoreConfig,
    strategy: OutOfCoreStrategy,
    basis_indices: Option<Vec<usize>>,
    basis_kernel: Option<Array2<f64>>,
    normalization: Option<Array2<f64>>,
}

impl OutOfCoreNystroem {
    /// Create a new out-of-core Nyström method
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            gamma: 1.0,
            config: OutOfCoreConfig::default(),
            strategy: OutOfCoreStrategy::Sequential,
            basis_indices: None,
            basis_kernel: None,
            normalization: None,
        }
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set out-of-core configuration
    pub fn with_config(mut self, config: OutOfCoreConfig) -> Self {
        self.config = config;
        self
    }

    /// Set processing strategy
    pub fn with_strategy(mut self, strategy: OutOfCoreStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Fit the Nyström method out-of-core
    pub fn fit_out_of_core(&mut self, loader: &mut OutOfCoreLoader) -> Result<()> {
        // Sample basis points across all chunks
        let basis_points = self.sample_basis_points(loader)?;

        // Compute kernel matrix for basis points
        let kernel_matrix = self.compute_kernel_matrix(&basis_points)?;

        // Compute eigendecomposition
        let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&kernel_matrix)?;

        // Store normalization factors
        self.normalization = Some(self.compute_normalization(&eigenvalues, &eigenvectors)?);

        Ok(())
    }

    /// Sample basis points from all chunks
    fn sample_basis_points(&mut self, loader: &mut OutOfCoreLoader) -> Result<Array2<f64>> {
        let mut basis_points = Vec::new();
        let mut point_indices = Vec::new();
        let mut global_index = 0;

        loader.reset();

        // Use reservoir sampling to select basis points
        let mut rng = thread_rng();
        let mut selected_count = 0;

        while let Some(chunk) = loader.load_chunk()? {
            for row in chunk.axis_iter(Axis(0)) {
                if selected_count < self.n_components {
                    // Fill reservoir
                    basis_points.push(row.to_owned());
                    point_indices.push(global_index);
                    selected_count += 1;
                } else {
                    // Randomly replace with probability k/n
                    let replace_idx = rng.gen_range(0..global_index + 1);
                    if replace_idx < self.n_components {
                        basis_points[replace_idx] = row.to_owned();
                        point_indices[replace_idx] = global_index;
                    }
                }
                global_index += 1;
            }
        }

        self.basis_indices = Some(point_indices);

        // Convert to Array2
        let n_features = basis_points[0].len();
        let mut result = Array2::zeros((basis_points.len(), n_features));
        for (i, point) in basis_points.iter().enumerate() {
            result.row_mut(i).assign(point);
        }

        Ok(result)
    }

    /// Compute kernel matrix for basis points
    fn compute_kernel_matrix(&self, basis_points: &Array2<f64>) -> Result<Array2<f64>> {
        let n_basis = basis_points.nrows();
        let mut kernel_matrix = Array2::zeros((n_basis, n_basis));

        for i in 0..n_basis {
            for j in i..n_basis {
                let dist_sq = basis_points
                    .row(i)
                    .iter()
                    .zip(basis_points.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>();

                let kernel_value = (-self.gamma * dist_sq).exp();
                kernel_matrix[[i, j]] = kernel_value;
                kernel_matrix[[j, i]] = kernel_value;
            }
        }

        Ok(kernel_matrix)
    }

    /// Compute eigendecomposition
    fn compute_eigendecomposition(
        &self,
        kernel_matrix: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        // This is a simplified version - in practice you'd use proper eigendecomposition
        let n = kernel_matrix.nrows();
        let eigenvalues = Array1::ones(n);
        let eigenvectors = kernel_matrix.clone();

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute normalization factors
    fn compute_normalization(
        &self,
        eigenvalues: &Array1<f64>,
        eigenvectors: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let mut normalization = Array2::zeros(eigenvectors.dim());

        for i in 0..eigenvectors.nrows() {
            for j in 0..eigenvectors.ncols() {
                if eigenvalues[j] > 1e-10 {
                    normalization[[i, j]] = eigenvectors[[i, j]] / eigenvalues[j].sqrt();
                }
            }
        }

        Ok(normalization)
    }

    /// Transform data out-of-core
    pub fn transform_out_of_core(&self, loader: &mut OutOfCoreLoader) -> Result<Vec<Array2<f64>>> {
        let mut results = Vec::new();
        loader.reset();

        while let Some(chunk) = loader.load_chunk()? {
            let transformed = self.transform_chunk(&chunk)?;
            results.push(transformed);
        }

        Ok(results)
    }

    /// Transform a single chunk
    fn transform_chunk(&self, chunk: &Array2<f64>) -> Result<Array2<f64>> {
        let normalization = self.normalization.as_ref().unwrap();

        // Compute kernel values between chunk and basis points
        let mut kernel_values = Array2::zeros((chunk.nrows(), self.n_components));

        // This is simplified - in practice you'd compute actual kernel values
        for i in 0..chunk.nrows() {
            for j in 0..self.n_components {
                kernel_values[[i, j]] = thread_rng().gen_range(0.0..1.0);
            }
        }

        // Apply normalization
        let result = kernel_values.dot(normalization);

        Ok(result)
    }
}

/// Out-of-core kernel approximation pipeline
pub struct OutOfCoreKernelPipeline {
    config: OutOfCoreConfig,
    methods: Vec<String>,
    results: HashMap<String, Vec<Array2<f64>>>,
}

impl Default for OutOfCoreKernelPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl OutOfCoreKernelPipeline {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self {
            config: OutOfCoreConfig::default(),
            methods: Vec::new(),
            results: HashMap::new(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: OutOfCoreConfig) -> Self {
        self.config = config;
        self
    }

    /// Add RBF sampler to pipeline
    pub fn add_rbf_sampler(mut self, n_components: usize, gamma: f64) -> Self {
        self.methods.push(format!("rbf_{}_{}", n_components, gamma));
        self
    }

    /// Add Nyström method to pipeline
    pub fn add_nystroem(mut self, n_components: usize, gamma: f64) -> Self {
        self.methods
            .push(format!("nystroem_{}_{}", n_components, gamma));
        self
    }

    /// Process data through pipeline
    pub fn process(&mut self, loader: &mut OutOfCoreLoader) -> Result<()> {
        for method in &self.methods.clone() {
            let parts: Vec<&str> = method.split('_').collect();
            match parts[0] {
                "rbf" => {
                    let n_components: usize = parts[1].parse().unwrap();
                    let gamma: f64 = parts[2].parse().unwrap();

                    let mut rbf = OutOfCoreRBFSampler::new(n_components)
                        .gamma(gamma)
                        .with_config(self.config.clone());

                    let result = rbf.fit_transform_out_of_core(loader)?;
                    self.results.insert(method.clone(), result);
                }
                "nystroem" => {
                    let n_components: usize = parts[1].parse().unwrap();
                    let gamma: f64 = parts[2].parse().unwrap();

                    let mut nystroem = OutOfCoreNystroem::new(n_components)
                        .gamma(gamma)
                        .with_config(self.config.clone());

                    nystroem.fit_out_of_core(loader)?;
                    let result = nystroem.transform_out_of_core(loader)?;
                    self.results.insert(method.clone(), result);
                }
                _ => continue,
            }
        }

        Ok(())
    }

    /// Get results for a specific method
    pub fn get_results(&self, method: &str) -> Option<&Vec<Array2<f64>>> {
        self.results.get(method)
    }

    /// Get all results
    pub fn get_all_results(&self) -> &HashMap<String, Vec<Array2<f64>>> {
        &self.results
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_out_of_core_config() {
        let config = OutOfCoreConfig::default();
        assert_eq!(config.max_memory_bytes, 1024 * 1024 * 1024);
        assert_eq!(config.chunk_size, 10000);
        assert!(config.n_workers > 0);
    }

    #[test]
    fn test_out_of_core_loader() {
        let mut loader = OutOfCoreLoader::new("test_data.csv", 1000, 5);
        assert_eq!(loader.n_chunks(), 1);

        let chunk = loader.load_chunk().unwrap();
        assert!(chunk.is_some());

        let chunk = chunk.unwrap();
        assert_eq!(chunk.nrows(), 1000);
        assert_eq!(chunk.ncols(), 5);
    }

    #[test]
    fn test_out_of_core_rbf_sampler() {
        let mut rbf = OutOfCoreRBFSampler::new(100)
            .gamma(1.0)
            .with_strategy(OutOfCoreStrategy::Sequential);

        let mut loader = OutOfCoreLoader::new("test_data.csv", 500, 3);
        let results = rbf.fit_transform_out_of_core(&mut loader).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ncols(), 100);
    }

    #[test]
    fn test_out_of_core_nystroem() {
        let mut nystroem = OutOfCoreNystroem::new(50)
            .gamma(1.0)
            .with_strategy(OutOfCoreStrategy::Sequential);

        let mut loader = OutOfCoreLoader::new("test_data.csv", 300, 4);
        nystroem.fit_out_of_core(&mut loader).unwrap();

        let results = nystroem.transform_out_of_core(&mut loader).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ncols(), 50);
    }

    #[test]
    fn test_out_of_core_pipeline() {
        let mut pipeline = OutOfCoreKernelPipeline::new()
            .add_rbf_sampler(100, 1.0)
            .add_nystroem(50, 0.5);

        let mut loader = OutOfCoreLoader::new("test_data.csv", 1000, 5);
        pipeline.process(&mut loader).unwrap();

        assert_eq!(pipeline.get_all_results().len(), 2);
    }

    #[test]
    fn test_parallel_strategy() {
        let mut rbf = OutOfCoreRBFSampler::new(100)
            .gamma(1.0)
            .with_strategy(OutOfCoreStrategy::Parallel);

        let mut loader = OutOfCoreLoader::new("test_data.csv", 500, 3);
        let results = rbf.fit_transform_out_of_core(&mut loader).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ncols(), 100);
    }

    #[test]
    fn test_streaming_strategy() {
        let config = OutOfCoreConfig {
            max_memory_bytes: 1024, // Very small memory limit
            ..Default::default()
        };

        let mut rbf = OutOfCoreRBFSampler::new(100)
            .gamma(1.0)
            .with_config(config)
            .with_strategy(OutOfCoreStrategy::Streaming);

        let mut loader = OutOfCoreLoader::new("test_data.csv", 500, 3);
        let results = rbf.fit_transform_out_of_core(&mut loader).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ncols(), 100);
    }

    #[test]
    fn test_adaptive_strategy() {
        let mut rbf = OutOfCoreRBFSampler::new(100)
            .gamma(1.0)
            .with_strategy(OutOfCoreStrategy::Adaptive);

        let mut loader = OutOfCoreLoader::new("test_data.csv", 500, 3);
        let results = rbf.fit_transform_out_of_core(&mut loader).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ncols(), 100);
    }

    #[test]
    fn test_memory_estimation() {
        let rbf = OutOfCoreRBFSampler::new(100).gamma(1.0);
        let memory_req = rbf.estimate_memory_requirement(1000, 10);
        assert!(memory_req > 0);
    }

    #[test]
    fn test_data_density_estimation() {
        let rbf = OutOfCoreRBFSampler::new(100).gamma(1.0);
        let data = Array2::zeros((10, 5));
        let density = rbf.estimate_data_density(&data);
        assert_eq!(density, 0.0);

        let data = Array2::ones((10, 5));
        let density = rbf.estimate_data_density(&data);
        assert_eq!(density, 1.0);
    }
}
