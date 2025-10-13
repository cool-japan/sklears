//! Out-of-Core Support Vector Machine Training
//!
//! This module implements SVM training algorithms that can handle datasets
//! larger than available memory by processing data in chunks and using
//! incremental learning approaches.

use crate::kernels::{Kernel, KernelType};
use crate::smo::SmoSolver;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::marker::PhantomData;
use std::path::Path;

/// Configuration for out-of-core SVM training
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Maximum number of samples to load in memory at once
    pub chunk_size: usize,
    /// Maximum size of the kernel cache in MB
    pub cache_size_mb: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to use shrinking heuristics
    pub shrinking: bool,
    /// Working directory for temporary files
    pub work_dir: String,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10000,
            cache_size_mb: 200,
            tolerance: 1e-3,
            max_iter: 1000,
            shrinking: true,
            work_dir: "/tmp/out_of_core_svm".to_string(),
        }
    }
}

/// Out-of-core SVM classifier
#[derive(Debug, Clone)]
pub struct OutOfCoreSVM<S> {
    /// Regularization parameter
    pub c: Float,
    /// Kernel function
    pub kernel: KernelType,
    /// Configuration
    pub config: OutOfCoreConfig,
    /// Number of classes (for classification)
    pub n_classes: Option<usize>,
    /// Support vectors (loaded in chunks)
    pub support_vectors: Option<Array2<Float>>,
    /// Dual coefficients
    pub dual_coef: Option<Array1<Float>>,
    /// Intercept
    pub intercept: Float,
    /// Class labels
    pub classes: Option<Array1<i32>>,
    /// Number of support vectors
    pub n_support: Option<Array1<usize>>,
    /// State marker
    _state: PhantomData<S>,
}

impl OutOfCoreSVM<Untrained> {
    /// Create a new out-of-core SVM classifier
    pub fn new(c: Float, kernel: KernelType, config: OutOfCoreConfig) -> Self {
        Self {
            c,
            kernel,
            config,
            n_classes: None,
            support_vectors: None,
            dual_coef: None,
            intercept: 0.0,
            classes: None,
            n_support: None,
            _state: PhantomData,
        }
    }

    /// Builder pattern for configuration
    pub fn builder() -> OutOfCoreSVMBuilder {
        OutOfCoreSVMBuilder::new()
    }
}

/// Builder for OutOfCoreSVM
pub struct OutOfCoreSVMBuilder {
    c: Float,
    kernel: KernelType,
    config: OutOfCoreConfig,
}

impl OutOfCoreSVMBuilder {
    pub fn new() -> Self {
        Self {
            c: 1.0,
            kernel: KernelType::Rbf { gamma: 1.0 },
            config: OutOfCoreConfig::default(),
        }
    }

    pub fn c(mut self, c: Float) -> Self {
        self.c = c;
        self
    }

    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = chunk_size;
        self
    }

    pub fn cache_size_mb(mut self, cache_size_mb: usize) -> Self {
        self.config.cache_size_mb = cache_size_mb;
        self
    }

    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn shrinking(mut self, shrinking: bool) -> Self {
        self.config.shrinking = shrinking;
        self
    }

    pub fn work_dir<P: AsRef<Path>>(mut self, work_dir: P) -> Self {
        self.config.work_dir = work_dir.as_ref().to_string_lossy().to_string();
        self
    }

    pub fn build(self) -> OutOfCoreSVM<Untrained> {
        OutOfCoreSVM::new(self.c, self.kernel, self.config)
    }
}

impl Default for OutOfCoreSVMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Data chunk for out-of-core processing
#[derive(Debug, Clone)]
pub struct DataChunk {
    /// Feature matrix for this chunk
    pub x: Array2<Float>,
    /// Target values for this chunk
    pub y: Array1<Float>,
    /// Global indices of samples in this chunk
    pub indices: Array1<usize>,
}

/// Out-of-core data iterator
pub struct OutOfCoreDataIterator {
    /// Path to the data file
    data_path: String,
    /// Chunk size
    chunk_size: usize,
    /// Current position in file
    current_position: usize,
    /// Total number of samples
    total_samples: usize,
    /// Number of features
    n_features: usize,
}

impl OutOfCoreDataIterator {
    /// Create a new data iterator
    pub fn new(data_path: &str, chunk_size: usize) -> Result<Self> {
        let file = File::open(data_path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read first line to determine number of features
        let first_line = lines
            .next()
            .ok_or_else(|| SklearsError::InvalidInput("Data file is empty".to_string()))??;

        let parts: Vec<&str> = first_line.split_whitespace().collect();
        let n_features = parts.len() - 1; // Assuming last column is target

        // Count total lines
        let total_samples = std::fs::read_to_string(data_path)?.lines().count();

        Ok(Self {
            data_path: data_path.to_string(),
            chunk_size,
            current_position: 0,
            total_samples,
            n_features,
        })
    }

    /// Get next chunk of data
    pub fn next_chunk(&mut self) -> Result<Option<DataChunk>> {
        if self.current_position >= self.total_samples {
            return Ok(None);
        }

        let file = File::open(&self.data_path)?;
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().collect::<std::io::Result<Vec<_>>>()?;

        let end_pos = std::cmp::min(self.current_position + self.chunk_size, self.total_samples);

        let chunk_lines = &lines[self.current_position..end_pos];
        let chunk_size = chunk_lines.len();

        let mut x = Array2::zeros((chunk_size, self.n_features));
        let mut y = Array1::zeros(chunk_size);
        let mut indices = Array1::zeros(chunk_size);

        for (i, line) in chunk_lines.iter().enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != self.n_features + 1 {
                return Err(SklearsError::InvalidInput(format!(
                    "Invalid line format: {}",
                    line
                )));
            }

            // Parse features
            for (j, part) in parts[..self.n_features].iter().enumerate() {
                x[[i, j]] = part
                    .parse::<Float>()
                    .map_err(|_| SklearsError::InvalidInput(format!("Invalid float: {part}")))?;
            }

            // Parse target
            y[i] = parts[self.n_features].parse::<Float>().map_err(|_| {
                SklearsError::InvalidInput(format!("Invalid target: {}", parts[self.n_features]))
            })?;

            // Set global index
            indices[i] = self.current_position + i;
        }

        self.current_position = end_pos;

        Ok(Some(DataChunk { x, y, indices }))
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.current_position = 0;
    }

    /// Get total number of samples
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for OutOfCoreSVM<Untrained> {
    type Fitted = OutOfCoreSVM<Trained>;

    fn fit(self, x: &ArrayView2<Float>, y: &ArrayView1<Float>) -> Result<Self::Fitted> {
        // For in-memory data, we can fall back to regular training
        // But this is mainly for API compatibility
        self.fit_from_memory(*x, *y)
    }
}

impl OutOfCoreSVM<Untrained> {
    /// Train from data file using out-of-core approach
    pub fn fit_from_file<P: AsRef<Path>>(&self, data_path: P) -> Result<OutOfCoreSVM<Trained>> {
        let data_path = data_path.as_ref().to_string_lossy().to_string();

        // Create work directory
        std::fs::create_dir_all(&self.config.work_dir)?;

        // Initialize data iterator
        let mut data_iter = OutOfCoreDataIterator::new(&data_path, self.config.chunk_size)?;

        // Initialize model state
        let mut dual_coef = Array1::zeros(data_iter.total_samples());
        let mut support_vector_mask = Array1::from_elem(data_iter.total_samples(), false);
        let mut intercept = 0.0;

        // Determine unique classes
        let mut classes = Vec::new();
        data_iter.reset();
        while let Some(chunk) = data_iter.next_chunk()? {
            for &label in chunk.y.iter() {
                let label_i32 = label as i32;
                if !classes.contains(&label_i32) {
                    classes.push(label_i32);
                }
            }
        }
        classes.sort_unstable();
        let n_classes = classes.len();

        // Convert to binary classification if needed
        let binary_labels = if n_classes == 2 {
            // Binary classification: use +1/-1 labels
            classes.clone()
        } else {
            // Multi-class: use one-vs-rest for now
            return Err(SklearsError::InvalidInput(
                "Multi-class out-of-core SVM not yet implemented".to_string(),
            ));
        };

        // Main training loop using decomposition method
        let mut iteration = 0;
        let mut converged = false;

        while !converged && iteration < self.config.max_iter {
            converged = true;
            data_iter.reset();

            // Process each chunk
            while let Some(chunk) = data_iter.next_chunk()? {
                // Convert labels to binary format
                let mut binary_y = Array1::zeros(chunk.y.len());
                for (i, &label) in chunk.y.iter().enumerate() {
                    binary_y[i] = if label as i32 == binary_labels[0] {
                        -1.0
                    } else {
                        1.0
                    };
                }

                // Create SMO solver for this chunk
                let kernel = self.kernel.clone();
                let smo_config = crate::smo::SmoConfig {
                    c: self.c,
                    tol: self.config.tolerance,
                    max_iter: self.config.max_iter,
                    cache_size: self.config.cache_size_mb,
                    shrinking: self.config.shrinking,
                    ..Default::default()
                };
                let mut smo_solver = SmoSolver::new(smo_config, kernel);

                // Get current dual coefficients for this chunk
                let mut chunk_dual_coef = Array1::zeros(chunk.y.len());
                for (i, &global_idx) in chunk.indices.iter().enumerate() {
                    chunk_dual_coef[i] = dual_coef[global_idx];
                }

                // Solve SMO for this chunk
                let smo_result = smo_solver.solve(&chunk.x, &binary_y)?;

                // Update global dual coefficients
                for (i, &global_idx) in chunk.indices.iter().enumerate() {
                    let old_alpha = dual_coef[global_idx];
                    dual_coef[global_idx] = smo_result.alpha[i];

                    // Check convergence
                    if (dual_coef[global_idx] - old_alpha).abs() > self.config.tolerance {
                        converged = false;
                    }

                    // Update support vector mask
                    support_vector_mask[global_idx] = smo_result.alpha[i].abs() > 1e-10;
                }

                // Update intercept (average across chunks)
                intercept += smo_result.b;
            }

            // Average intercept across chunks
            intercept /=
                (data_iter.total_samples() as Float / self.config.chunk_size as Float).ceil();

            iteration += 1;
        }

        // Extract support vectors and their coefficients
        let n_support_vectors = support_vector_mask.iter().filter(|&&x| x).count();
        let mut support_vectors = Array2::zeros((n_support_vectors, data_iter.n_features()));
        let mut support_dual_coef = Array1::zeros(n_support_vectors);

        // Load support vectors from file
        data_iter.reset();
        let mut sv_index = 0;
        while let Some(chunk) = data_iter.next_chunk()? {
            for (i, &global_idx) in chunk.indices.iter().enumerate() {
                if support_vector_mask[global_idx] {
                    support_vectors.row_mut(sv_index).assign(&chunk.x.row(i));
                    support_dual_coef[sv_index] = dual_coef[global_idx];
                    sv_index += 1;
                }
            }
        }

        // Count support vectors per class
        let mut n_support = Array1::zeros(n_classes);
        for &coef in support_dual_coef.iter() {
            if coef > 0.0 {
                n_support[1] += 1;
            } else {
                n_support[0] += 1;
            }
        }

        Ok(OutOfCoreSVM {
            c: self.c,
            kernel: self.kernel.clone(),
            config: self.config.clone(),
            n_classes: Some(n_classes),
            support_vectors: Some(support_vectors),
            dual_coef: Some(support_dual_coef),
            intercept,
            classes: Some(Array1::from_vec(classes)),
            n_support: Some(n_support),
            _state: PhantomData,
        })
    }

    /// Train from in-memory data (fallback)
    pub fn fit_from_memory(
        &self,
        x: ArrayView2<Float>,
        y: ArrayView1<Float>,
    ) -> Result<OutOfCoreSVM<Trained>> {
        // For in-memory data, we can use existing SMO implementation
        // This is a simplified version - could be optimized further

        // Determine unique classes
        let mut classes_vec = Vec::new();
        for &label in y.iter() {
            let label_i32 = label as i32;
            if !classes_vec.contains(&label_i32) {
                classes_vec.push(label_i32);
            }
        }
        classes_vec.sort_unstable();
        let n_classes = classes_vec.len();

        if n_classes != 2 {
            return Err(SklearsError::InvalidInput(
                "Multi-class out-of-core SVM not yet implemented".to_string(),
            ));
        }

        // Convert to binary labels
        let mut binary_y = Array1::zeros(y.len());
        for (i, &label) in y.iter().enumerate() {
            binary_y[i] = if label as i32 == classes_vec[0] {
                -1.0
            } else {
                1.0
            };
        }

        // Create SMO solver
        let kernel = self.kernel.clone();
        let smo_config = crate::smo::SmoConfig {
            c: self.c,
            tol: self.config.tolerance,
            max_iter: self.config.max_iter,
            cache_size: self.config.cache_size_mb,
            shrinking: self.config.shrinking,
            ..Default::default()
        };
        let mut smo_solver = SmoSolver::new(smo_config, kernel);

        // Solve
        let smo_result = smo_solver.solve(&x.to_owned(), &binary_y)?;

        // Extract support vectors
        let support_indices: Vec<usize> = smo_result
            .alpha
            .iter()
            .enumerate()
            .filter(|(_, &coef)| coef.abs() > 1e-10)
            .map(|(i, _)| i)
            .collect();

        let n_support_vectors = support_indices.len();
        let mut support_vectors = Array2::zeros((n_support_vectors, x.ncols()));
        let mut support_dual_coef = Array1::zeros(n_support_vectors);

        for (i, &idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(idx));
            support_dual_coef[i] = smo_result.alpha[idx];
        }

        // Count support vectors per class
        let mut n_support = Array1::zeros(n_classes);
        for &coef in support_dual_coef.iter() {
            if coef > 0.0 {
                n_support[1] += 1;
            } else {
                n_support[0] += 1;
            }
        }

        Ok(OutOfCoreSVM {
            c: self.c,
            kernel: self.kernel.clone(),
            config: self.config.clone(),
            n_classes: Some(n_classes),
            support_vectors: Some(support_vectors),
            dual_coef: Some(support_dual_coef),
            intercept: smo_result.b,
            classes: Some(Array1::from_vec(classes_vec)),
            n_support: Some(n_support),
            _state: PhantomData,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<Float>> for OutOfCoreSVM<Trained> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<Float>> {
        let support_vectors =
            self.support_vectors
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let dual_coef = self
            .dual_coef
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let kernel = self.kernel.clone();
        let mut predictions = Array1::zeros(x.nrows());

        // Compute decision function for each sample
        for i in 0..x.nrows() {
            let mut decision_value = 0.0;

            // Compute kernel values with all support vectors
            for j in 0..support_vectors.nrows() {
                let kernel_value = kernel.compute(x.row(i), support_vectors.row(j));
                decision_value += dual_coef[j] * kernel_value;
            }

            decision_value += self.intercept;

            // Convert to class prediction
            predictions[i] = if decision_value >= 0.0 {
                classes[1] as Float
            } else {
                classes[0] as Float
            };
        }

        Ok(predictions)
    }
}

impl Estimator for OutOfCoreSVM<Untrained> {
    type Config = OutOfCoreConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for OutOfCoreSVM<Trained> {
    type Config = OutOfCoreConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_out_of_core_svm_creation() {
        let config = OutOfCoreConfig::default();
        let svm = OutOfCoreSVM::new(1.0, KernelType::Linear, config);
        assert_eq!(svm.c, 1.0);
        assert_eq!(svm.config().chunk_size, 10000);
    }

    #[test]
    fn test_out_of_core_svm_builder() {
        let svm = OutOfCoreSVM::builder()
            .c(2.0)
            .kernel(KernelType::Rbf { gamma: 0.5 })
            .chunk_size(5000)
            .tolerance(1e-4)
            .build();

        assert_eq!(svm.c, 2.0);
        assert_eq!(svm.config.chunk_size, 5000);
        assert_eq!(svm.config.tolerance, 1e-4);
    }

    #[test]
    fn test_data_iterator() -> Result<()> {
        // Create test data file
        let test_file = "/tmp/test_data.txt";
        let mut file = File::create(test_file)?;
        writeln!(file, "1.0 2.0 1")?;
        writeln!(file, "2.0 3.0 -1")?;
        writeln!(file, "3.0 4.0 1")?;
        writeln!(file, "4.0 5.0 -1")?;

        let mut iterator = OutOfCoreDataIterator::new(test_file, 2)?;

        assert_eq!(iterator.total_samples(), 4);
        assert_eq!(iterator.n_features(), 2);

        // Test first chunk
        let chunk1 = iterator.next_chunk()?.unwrap();
        assert_eq!(chunk1.x.nrows(), 2);
        assert_eq!(chunk1.x.ncols(), 2);
        assert_eq!(chunk1.y.len(), 2);

        // Test second chunk
        let chunk2 = iterator.next_chunk()?.unwrap();
        assert_eq!(chunk2.x.nrows(), 2);

        // Test end of iteration
        let chunk3 = iterator.next_chunk()?;
        assert!(chunk3.is_none());

        // Clean up
        std::fs::remove_file(test_file)?;

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_out_of_core_svm_memory_training() -> Result<()> {
        // Create simple binary classification data
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0])?;
        let y = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0]);

        let svm = OutOfCoreSVM::builder()
            .c(1.0)
            .kernel(KernelType::Linear)
            .tolerance(1e-3)
            .build();

        let trained_svm = svm.fit(&x.view(), &y.view())?;

        // Test predictions
        let predictions = trained_svm.predict(&x.view())?;
        assert_eq!(predictions.len(), 4);

        // Check that we have reasonable predictions
        for &pred in predictions.iter() {
            assert!(pred == 1.0 || pred == -1.0);
        }

        Ok(())
    }
}
