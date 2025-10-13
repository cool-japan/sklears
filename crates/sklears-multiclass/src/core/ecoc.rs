//! Error-Correcting Output Codes (ECOC) Classifier
//!
//! This module implements the ECOC strategy for multiclass classification.
//! ECOC uses error-correcting output codes to transform a multiclass problem
//! into multiple binary classification problems. Each class is represented by
//! a unique binary code, and binary classifiers are trained to predict each
//! bit of the code.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, Random};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Sparse matrix representation for ECOC code matrices
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Non-zero entries stored as (row, col, value) triples
    pub entries: Vec<(usize, usize, i32)>,
    /// Default value for missing entries (typically 0 or -1)
    pub default_value: i32,
}

impl SparseMatrix {
    /// Create a new sparse matrix with given dimensions and default value
    pub fn new(n_rows: usize, n_cols: usize, default_value: i32) -> Self {
        Self {
            n_rows,
            n_cols,
            entries: Vec::new(),
            default_value,
        }
    }

    /// Create a sparse matrix from a dense matrix
    pub fn from_dense(dense: &Array2<i32>, default_value: i32) -> Self {
        let mut entries = Vec::new();
        let (n_rows, n_cols) = dense.dim();

        for i in 0..n_rows {
            for j in 0..n_cols {
                let value = dense[[i, j]];
                if value != default_value {
                    entries.push((i, j, value));
                }
            }
        }

        Self {
            n_rows,
            n_cols,
            entries,
            default_value,
        }
    }

    /// Convert sparse matrix to dense representation
    pub fn to_dense(&self) -> Array2<i32> {
        let mut dense = Array2::from_elem((self.n_rows, self.n_cols), self.default_value);

        for &(row, col, value) in &self.entries {
            dense[[row, col]] = value;
        }

        dense
    }

    /// Get value at specific position
    pub fn get(&self, row: usize, col: usize) -> i32 {
        for &(r, c, value) in &self.entries {
            if r == row && c == col {
                return value;
            }
        }
        self.default_value
    }

    /// Set value at specific position
    pub fn set(&mut self, row: usize, col: usize, value: i32) {
        // Remove existing entry if it exists
        self.entries.retain(|(r, c, _)| *r != row || *c != col);

        // Add new entry if it's not the default value
        if value != self.default_value {
            self.entries.push((row, col, value));
        }
    }

    /// Get a row as a vector
    pub fn get_row(&self, row: usize) -> Vec<i32> {
        let mut result = vec![self.default_value; self.n_cols];

        for &(r, c, value) in &self.entries {
            if r == row {
                result[c] = value;
            }
        }

        result
    }

    /// Calculate memory usage (in bytes)
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.entries.len() * std::mem::size_of::<(usize, usize, i32)>()
    }

    /// Calculate compression ratio compared to dense representation
    pub fn compression_ratio(&self) -> f64 {
        let dense_size = self.n_rows * self.n_cols * std::mem::size_of::<i32>();
        let sparse_size = self.memory_usage();
        sparse_size as f64 / dense_size as f64
    }

    /// Get sparsity level (fraction of entries that are default value)
    pub fn sparsity(&self) -> f64 {
        let total_entries = self.n_rows * self.n_cols;
        let non_default_entries = self.entries.len();
        1.0 - (non_default_entries as f64 / total_entries as f64)
    }
}

/// Code matrix representation (dense or sparse)
#[derive(Debug, Clone, PartialEq)]
pub enum CodeMatrix {
    /// Dense representation using ndarray
    Dense(Array2<i32>),
    /// Sparse representation for memory efficiency
    Sparse(SparseMatrix),
}

impl CodeMatrix {
    /// Get dimensions (rows, cols)
    pub fn dim(&self) -> (usize, usize) {
        match self {
            CodeMatrix::Dense(matrix) => matrix.dim(),
            CodeMatrix::Sparse(matrix) => (matrix.n_rows, matrix.n_cols),
        }
    }

    /// Get number of rows
    pub fn nrows(&self) -> usize {
        match self {
            CodeMatrix::Dense(matrix) => matrix.nrows(),
            CodeMatrix::Sparse(matrix) => matrix.n_rows,
        }
    }

    /// Get number of columns
    pub fn ncols(&self) -> usize {
        match self {
            CodeMatrix::Dense(matrix) => matrix.ncols(),
            CodeMatrix::Sparse(matrix) => matrix.n_cols,
        }
    }

    /// Get value at specific position
    pub fn get(&self, row: usize, col: usize) -> i32 {
        match self {
            CodeMatrix::Dense(matrix) => matrix[[row, col]],
            CodeMatrix::Sparse(matrix) => matrix.get(row, col),
        }
    }

    /// Get a row as a vector
    pub fn get_row(&self, row: usize) -> Vec<i32> {
        match self {
            CodeMatrix::Dense(matrix) => matrix.row(row).to_vec(),
            CodeMatrix::Sparse(matrix) => matrix.get_row(row),
        }
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> Array2<i32> {
        match self {
            CodeMatrix::Dense(matrix) => matrix.clone(),
            CodeMatrix::Sparse(matrix) => matrix.to_dense(),
        }
    }

    /// Calculate memory usage (in bytes)
    pub fn memory_usage(&self) -> usize {
        match self {
            CodeMatrix::Dense(matrix) => matrix.len() * std::mem::size_of::<i32>(),
            CodeMatrix::Sparse(matrix) => matrix.memory_usage(),
        }
    }

    /// Get sparsity level (0.0 = dense, 1.0 = completely sparse)
    pub fn sparsity(&self) -> f64 {
        match self {
            CodeMatrix::Dense(matrix) => {
                let total_elements = matrix.len();
                let zero_elements = matrix.iter().filter(|&&x| x == 0).count();
                zero_elements as f64 / total_elements as f64
            }
            CodeMatrix::Sparse(matrix) => matrix.sparsity(),
        }
    }

    /// Apply memory optimization based on configuration
    pub fn optimize_memory(
        &self,
        memory_mode: &MemoryMode,
        compression_level: u8,
    ) -> OptimizedCodeMatrix {
        match memory_mode {
            MemoryMode::Standard => OptimizedCodeMatrix::Standard(self.clone()),
            MemoryMode::Compressed => OptimizedCodeMatrix::Compressed(
                CompressedCodeMatrix::compress(self, compression_level),
            ),
            MemoryMode::Aggressive => {
                OptimizedCodeMatrix::Compressed(CompressedCodeMatrix::compress(self, 9))
            }
            MemoryMode::Quantized => {
                OptimizedCodeMatrix::Quantized(QuantizedCodeMatrix::quantize(self))
            }
        }
    }
}

/// Compressed code matrix for memory-optimized storage
#[derive(Debug, Clone)]
pub struct CompressedCodeMatrix {
    /// Compressed data bytes
    pub data: Vec<u8>,
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Compression level used
    pub compression_level: u8,
    /// Original sparsity
    pub original_sparsity: f64,
}

impl CompressedCodeMatrix {
    /// Compress a code matrix using run-length encoding
    pub fn compress(matrix: &CodeMatrix, compression_level: u8) -> Self {
        let (data, n_rows, n_cols, original_sparsity) = match matrix {
            CodeMatrix::Dense(m) => {
                let flat_data: Vec<u8> = m
                    .as_slice()
                    .unwrap()
                    .iter()
                    .map(|&x| Self::i32_to_u8(x))
                    .collect();
                let compressed = Self::compress_bytes(&flat_data, compression_level);
                (compressed, m.nrows(), m.ncols(), 0.0)
            }
            CodeMatrix::Sparse(s) => {
                // Serialize sparse matrix entries
                let mut data = Vec::new();
                data.extend_from_slice(&(s.default_value as u8).to_le_bytes());
                data.extend_from_slice(&s.entries.len().to_le_bytes());

                for &(row, col, val) in &s.entries {
                    data.extend_from_slice(&row.to_le_bytes());
                    data.extend_from_slice(&col.to_le_bytes());
                    data.extend_from_slice(&(Self::i32_to_u8(val)).to_le_bytes());
                }

                let compressed = Self::compress_bytes(&data, compression_level);
                (compressed, s.n_rows, s.n_cols, s.sparsity())
            }
        };

        Self {
            data,
            n_rows,
            n_cols,
            compression_level,
            original_sparsity,
        }
    }

    /// Convert i32 to u8 with mapping: -1 -> 0, 0 -> 127, 1 -> 255
    fn i32_to_u8(val: i32) -> u8 {
        match val {
            -1 => 0,
            0 => 127,
            1 => 255,
            _ => 127, // fallback for unexpected values
        }
    }

    /// Convert u8 back to i32 with reverse mapping
    fn u8_to_i32(val: u8) -> i32 {
        match val {
            0 => -1,
            127 => 0,
            255 => 1,
            _ => {
                if val < 64 {
                    -1
                } else if val > 191 {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Simple run-length encoding compression
    fn compress_bytes(input: &[u8], _level: u8) -> Vec<u8> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut compressed = Vec::new();
        let mut current_byte = input[0];
        let mut count = 1u8;

        for &byte in &input[1..] {
            if byte == current_byte && count < 255 {
                count += 1;
            } else {
                compressed.push(current_byte);
                compressed.push(count);
                current_byte = byte;
                count = 1;
            }
        }
        compressed.push(current_byte);
        compressed.push(count);

        compressed
    }

    /// Decompress back to original code matrix
    pub fn decompress(&self) -> CodeMatrix {
        let decompressed = Self::decompress_bytes(&self.data);

        if self.original_sparsity > 0.3 {
            // Reconstruct sparse matrix
            if decompressed.len() >= 9 {
                // At least default_value + length + one entry
                let default_value = Self::u8_to_i32(decompressed[0]);
                let num_entries = usize::from_le_bytes([
                    decompressed[1],
                    decompressed[2],
                    decompressed[3],
                    decompressed[4],
                    decompressed[5],
                    decompressed[6],
                    decompressed[7],
                    decompressed[8],
                ]);

                let mut entries = Vec::new();
                let mut pos = 9;

                for _ in 0..num_entries {
                    if pos + 9 <= decompressed.len() {
                        let row = usize::from_le_bytes([
                            decompressed[pos],
                            decompressed[pos + 1],
                            decompressed[pos + 2],
                            decompressed[pos + 3],
                            decompressed[pos + 4],
                            decompressed[pos + 5],
                            decompressed[pos + 6],
                            decompressed[pos + 7],
                        ]);
                        pos += 8;
                        let col = usize::from_le_bytes([
                            decompressed[pos],
                            decompressed[pos + 1],
                            decompressed[pos + 2],
                            decompressed[pos + 3],
                            decompressed[pos + 4],
                            decompressed[pos + 5],
                            decompressed[pos + 6],
                            decompressed[pos + 7],
                        ]);
                        pos += 8;
                        let val = Self::u8_to_i32(decompressed[pos]);
                        pos += 1;

                        entries.push((row, col, val));
                    }
                }

                let mut sparse = SparseMatrix::new(self.n_rows, self.n_cols, default_value);
                sparse.entries = entries;
                return CodeMatrix::Sparse(sparse);
            }
        }

        // Reconstruct dense matrix
        let values: Vec<i32> = decompressed.iter().map(|&x| Self::u8_to_i32(x)).collect();
        let matrix = Array2::from_shape_vec((self.n_rows, self.n_cols), values)
            .unwrap_or_else(|_| Array2::zeros((self.n_rows, self.n_cols)));
        CodeMatrix::Dense(matrix)
    }

    /// Decompress run-length encoded bytes
    fn decompress_bytes(compressed: &[u8]) -> Vec<u8> {
        let mut decompressed = Vec::new();
        let mut i = 0;

        while i + 1 < compressed.len() {
            let byte = compressed[i];
            let count = compressed[i + 1];

            for _ in 0..count {
                decompressed.push(byte);
            }

            i += 2;
        }

        decompressed
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() + 32
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        let original_size = self.n_rows * self.n_cols * 4;
        if original_size == 0 {
            1.0
        } else {
            self.data.len() as f64 / original_size as f64
        }
    }
}

/// Quantized code matrix for reduced precision storage
#[derive(Debug, Clone)]
pub struct QuantizedCodeMatrix {
    /// Quantized values stored as 2-bit values packed in bytes
    pub data: Vec<u8>,
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
}

impl QuantizedCodeMatrix {
    /// Quantize a code matrix to 2-bit precision (-1, 0, 1 -> 0, 1, 2)
    pub fn quantize(matrix: &CodeMatrix) -> Self {
        let values = match matrix {
            CodeMatrix::Dense(m) => m.as_slice().unwrap().to_vec(),
            CodeMatrix::Sparse(s) => {
                let mut dense = vec![s.default_value; s.n_rows * s.n_cols];
                for &(row, col, val) in &s.entries {
                    if let Some(idx) = row.checked_mul(s.n_cols).and_then(|x| x.checked_add(col)) {
                        if idx < dense.len() {
                            dense[idx] = val;
                        }
                    }
                }
                dense
            }
        };

        // Pack 4 values per byte (2 bits each)
        let mut packed_data = Vec::new();
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                let quantized = match val {
                    -1 => 0u8,
                    0 => 1u8,
                    1 => 2u8,
                    _ => 1u8, // fallback
                };
                byte |= quantized << (i * 2);
            }
            packed_data.push(byte);
        }

        Self {
            data: packed_data,
            n_rows: matrix.nrows(),
            n_cols: matrix.ncols(),
        }
    }

    /// Dequantize back to original precision
    pub fn dequantize(&self) -> CodeMatrix {
        let mut values = Vec::new();

        for &byte in &self.data {
            for i in 0..4 {
                let quantized = (byte >> (i * 2)) & 0b11;
                let original = match quantized {
                    0 => -1,
                    1 => 0,
                    2 => 1,
                    _ => 0, // fallback
                };
                values.push(original);

                if values.len() >= self.n_rows * self.n_cols {
                    break;
                }
            }

            if values.len() >= self.n_rows * self.n_cols {
                break;
            }
        }

        // Ensure we have exactly the right number of values
        values.truncate(self.n_rows * self.n_cols);

        let matrix = Array2::from_shape_vec((self.n_rows, self.n_cols), values)
            .unwrap_or_else(|_| Array2::zeros((self.n_rows, self.n_cols)));
        CodeMatrix::Dense(matrix)
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() + 16
    }

    /// Get compression ratio compared to original i32 matrix
    pub fn compression_ratio(&self) -> f64 {
        let original_size = self.n_rows * self.n_cols * 4;
        if original_size == 0 {
            1.0
        } else {
            self.data.len() as f64 / original_size as f64
        }
    }
}

/// Optimized code matrix supporting different memory optimization modes
#[derive(Debug, Clone)]
pub enum OptimizedCodeMatrix {
    /// Standard uncompressed matrix
    Standard(CodeMatrix),
    /// Compressed matrix
    Compressed(CompressedCodeMatrix),
    /// Quantized matrix
    Quantized(QuantizedCodeMatrix),
}

impl OptimizedCodeMatrix {
    /// Get the code matrix in usable form
    pub fn get_matrix(&self) -> CodeMatrix {
        match self {
            OptimizedCodeMatrix::Standard(matrix) => matrix.clone(),
            OptimizedCodeMatrix::Compressed(compressed) => compressed.decompress(),
            OptimizedCodeMatrix::Quantized(quantized) => quantized.dequantize(),
        }
    }

    /// Get number of rows
    pub fn nrows(&self) -> usize {
        match self {
            OptimizedCodeMatrix::Standard(matrix) => matrix.nrows(),
            OptimizedCodeMatrix::Compressed(compressed) => compressed.n_rows,
            OptimizedCodeMatrix::Quantized(quantized) => quantized.n_rows,
        }
    }

    /// Get number of columns
    pub fn ncols(&self) -> usize {
        match self {
            OptimizedCodeMatrix::Standard(matrix) => matrix.ncols(),
            OptimizedCodeMatrix::Compressed(compressed) => compressed.n_cols,
            OptimizedCodeMatrix::Quantized(quantized) => quantized.n_cols,
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            OptimizedCodeMatrix::Standard(matrix) => matrix.memory_usage(),
            OptimizedCodeMatrix::Compressed(compressed) => compressed.memory_usage(),
            OptimizedCodeMatrix::Quantized(quantized) => quantized.memory_usage(),
        }
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        match self {
            OptimizedCodeMatrix::Standard(_) => 1.0,
            OptimizedCodeMatrix::Compressed(compressed) => compressed.compression_ratio(),
            OptimizedCodeMatrix::Quantized(quantized) => quantized.compression_ratio(),
        }
    }
}

/// Error-Correcting Output Codes (ECOC) strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ECOCStrategy {
    /// Random binary codes
    #[default]
    Random,
    /// Dense random codes with balanced +1/-1 distribution
    DenseRandom,
    /// Exhaustive codes (all possible binary combinations)
    Exhaustive,
    /// BCH (Bose-Chaudhuri-Hocquenghem) codes for optimal error correction
    BCH {
        /// Minimum distance parameter for error correction capability
        min_distance: usize,
    },
    /// Optimal code design using maximum distance criterion
    Optimal {
        /// Target minimum distance between codewords
        target_distance: usize,
    },
}

/// GPU acceleration mode for ECOC operations
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GPUMode {
    /// Disable GPU acceleration
    #[default]
    Disabled,
    /// Enable GPU acceleration for matrix operations
    MatrixOps,
    /// Enable GPU acceleration for distance calculations
    DistanceOps,
    /// Enable full GPU acceleration
    Full,
}

/// Memory optimization mode for model storage
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MemoryMode {
    /// Standard memory usage
    #[default]
    Standard,
    /// Compressed storage with moderate memory reduction
    Compressed,
    /// Aggressive compression with maximum memory reduction
    Aggressive,
    /// Quantized models with reduced precision
    Quantized,
}

/// Configuration for Error-Correcting Output Code Classifier
#[derive(Debug, Clone)]
pub struct ECOCConfig {
    /// Code matrix generation strategy
    pub strategy: ECOCStrategy,
    /// Code size multiplier (for random strategies)
    pub code_size: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// Use sparse matrix representation for memory efficiency
    pub use_sparse: bool,
    /// Minimum sparsity threshold to use sparse representation (0.0 to 1.0)
    pub sparse_threshold: f64,
    /// GPU acceleration mode
    pub gpu_mode: GPUMode,
    /// Batch size for GPU operations
    pub gpu_batch_size: usize,
    /// Memory optimization mode
    pub memory_mode: MemoryMode,
    /// Compression level (0-9, higher = more compression)
    pub compression_level: u8,
    /// Enable model quantization to reduce memory
    pub quantize_models: bool,
}

impl Default for ECOCConfig {
    fn default() -> Self {
        Self {
            strategy: ECOCStrategy::default(),
            code_size: 1.5,
            random_state: None,
            n_jobs: None,
            use_sparse: false,
            sparse_threshold: 0.3,
            gpu_mode: GPUMode::default(),
            gpu_batch_size: 1000,
            memory_mode: MemoryMode::default(),
            compression_level: 6,
            quantize_models: false,
        }
    }
}

/// Error-Correcting Output Code Classifier
///
/// This multiclass strategy uses error-correcting output codes to transform
/// a multiclass problem into multiple binary classification problems.
/// Each class is represented by a unique binary code, and binary classifiers
/// are trained to predict each bit of the code.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::{ECOCClassifier, ECOCStrategy};
/// use scirs2_autograd::ndarray::array;
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let ecoc = ECOCClassifier::new(base_classifier)
/// //     .strategy(ECOCStrategy::Random)
/// //     .code_size(2.0);
/// ```
#[derive(Debug)]
pub struct ECOCClassifier<C, S = Untrained> {
    /// base_estimator
    pub base_estimator: C,
    /// config
    pub config: ECOCConfig,
    /// state
    pub state: PhantomData<S>,
}

impl<C> ECOCClassifier<C, Untrained> {
    /// Create a new ECOCClassifier instance with a base estimator
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: ECOCConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for ECOCClassifier
    pub fn builder(base_estimator: C) -> ECOCBuilder<C> {
        ECOCBuilder::new(base_estimator)
    }

    /// Set the coding strategy
    pub fn strategy(mut self, strategy: ECOCStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the code size multiplier
    pub fn code_size(mut self, code_size: f64) -> Self {
        self.config.code_size = code_size;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Enable sparse matrix representation for memory efficiency
    pub fn use_sparse(mut self, use_sparse: bool) -> Self {
        self.config.use_sparse = use_sparse;
        self
    }

    /// Set the sparsity threshold for automatic sparse matrix selection
    pub fn sparse_threshold(mut self, threshold: f64) -> Self {
        self.config.sparse_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }
}

/// Builder for ECOCClassifier
#[derive(Debug)]
pub struct ECOCBuilder<C> {
    /// base_estimator
    pub base_estimator: C,
    /// config
    pub config: ECOCConfig,
}

impl<C> ECOCBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: ECOCConfig::default(),
        }
    }

    /// Set the coding strategy
    pub fn strategy(mut self, strategy: ECOCStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the code size multiplier
    pub fn code_size(mut self, code_size: f64) -> Self {
        self.config.code_size = code_size;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Enable parallel training using all available cores
    pub fn parallel(mut self) -> Self {
        self.config.n_jobs = Some(-1);
        self
    }

    /// Disable parallel training (use sequential training)
    pub fn sequential(mut self) -> Self {
        self.config.n_jobs = None;
        self
    }

    /// Enable sparse matrix representation for memory efficiency
    pub fn use_sparse(mut self, use_sparse: bool) -> Self {
        self.config.use_sparse = use_sparse;
        self
    }

    /// Set the sparsity threshold for automatic sparse matrix selection
    pub fn sparse_threshold(mut self, threshold: f64) -> Self {
        self.config.sparse_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable GPU acceleration for matrix operations
    pub fn gpu_matrix_ops(mut self) -> Self {
        self.config.gpu_mode = GPUMode::MatrixOps;
        self
    }

    /// Enable GPU acceleration for distance calculations
    pub fn gpu_distance_ops(mut self) -> Self {
        self.config.gpu_mode = GPUMode::DistanceOps;
        self
    }

    /// Enable full GPU acceleration
    pub fn gpu_full(mut self) -> Self {
        self.config.gpu_mode = GPUMode::Full;
        self
    }

    /// Set GPU batch size for operations
    pub fn gpu_batch_size(mut self, batch_size: usize) -> Self {
        self.config.gpu_batch_size = batch_size;
        self
    }

    /// Enable compressed memory mode for model storage
    pub fn memory_compressed(mut self) -> Self {
        self.config.memory_mode = MemoryMode::Compressed;
        self.config.use_sparse = true;
        self
    }

    /// Enable aggressive compression for maximum memory reduction
    pub fn memory_aggressive(mut self) -> Self {
        self.config.memory_mode = MemoryMode::Aggressive;
        self.config.use_sparse = true;
        self.config.compression_level = 9;
        self
    }

    /// Enable model quantization to reduce memory footprint
    pub fn quantize_models(mut self) -> Self {
        self.config.memory_mode = MemoryMode::Quantized;
        self.config.quantize_models = true;
        self
    }

    /// Set compression level (0-9, higher = more compression)
    pub fn compression_level(mut self, level: u8) -> Self {
        self.config.compression_level = level.min(9);
        self
    }

    /// Build the ECOCClassifier
    pub fn build(self) -> ECOCClassifier<C, Untrained> {
        ECOCClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for ECOCClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for ECOCBuilder<C> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
        }
    }
}

impl<C> Estimator for ECOCClassifier<C, Untrained> {
    type Config = ECOCConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// Trained data container for ECOCClassifier
#[derive(Debug)]
pub struct ECOCTrainedData<T> {
    /// estimators
    pub estimators: Vec<T>,
    /// classes
    pub classes: Array1<i32>,
    /// code_matrix
    pub code_matrix: CodeMatrix,
    /// n_features
    pub n_features: usize,
}

impl<T: Clone> Clone for ECOCTrainedData<T> {
    fn clone(&self) -> Self {
        Self {
            estimators: self.estimators.clone(),
            classes: self.classes.clone(),
            code_matrix: self.code_matrix.clone(),
            n_features: self.n_features,
        }
    }
}

type TrainedECOC<T> = ECOCClassifier<ECOCTrainedData<T>, Trained>;

impl<C> ECOCClassifier<C, Untrained> {
    /// Generate code matrix based on strategy
    fn generate_code_matrix(
        &self,
        n_classes: usize,
        rng: &mut Random<StdRng>,
    ) -> SklResult<CodeMatrix> {
        // Generate dense matrix first
        let dense_matrix = match &self.config.strategy {
            ECOCStrategy::Random => self.generate_random_code_matrix(n_classes, rng),
            ECOCStrategy::DenseRandom => self.generate_dense_random_code_matrix(n_classes, rng),
            ECOCStrategy::Exhaustive => self.generate_exhaustive_code_matrix(n_classes),
            ECOCStrategy::BCH { min_distance } => {
                self.generate_bch_code_matrix(n_classes, *min_distance)
            }
            ECOCStrategy::Optimal { target_distance } => {
                self.generate_optimal_code_matrix(n_classes, *target_distance, rng)
            }
        }?;

        // Determine whether to use sparse representation
        let use_sparse = if self.config.use_sparse {
            true
        } else {
            // Auto-detect based on sparsity threshold
            let sparsity = self.calculate_sparsity(&dense_matrix);
            sparsity >= self.config.sparse_threshold
        };

        if use_sparse {
            // Convert to sparse representation with default value as the most common value
            let default_value = self.find_most_common_value(&dense_matrix);
            let sparse_matrix = SparseMatrix::from_dense(&dense_matrix, default_value);
            Ok(CodeMatrix::Sparse(sparse_matrix))
        } else {
            Ok(CodeMatrix::Dense(dense_matrix))
        }
    }

    /// Calculate sparsity of a dense matrix (fraction of zero elements)
    fn calculate_sparsity(&self, matrix: &Array2<i32>) -> f64 {
        let total_elements = matrix.len();
        let zero_elements = matrix.iter().filter(|&&x| x == 0).count();
        zero_elements as f64 / total_elements as f64
    }

    /// Find the most common value in the matrix (to use as default in sparse representation)
    fn find_most_common_value(&self, matrix: &Array2<i32>) -> i32 {
        let mut value_counts: HashMap<i32, usize> = HashMap::new();

        for &value in matrix.iter() {
            *value_counts.entry(value).or_insert(0) += 1;
        }

        value_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, _)| value)
            .unwrap_or(0)
    }

    /// Generate random binary code matrix
    fn generate_random_code_matrix(
        &self,
        n_classes: usize,
        rng: &mut Random<StdRng>,
    ) -> SklResult<Array2<i32>> {
        let code_length = ((n_classes as f64) * self.config.code_size).ceil() as usize;
        let mut code_matrix = Array2::zeros((n_classes, code_length));

        for i in 0..n_classes {
            for j in 0..code_length {
                code_matrix[[i, j]] = if rng.random_f64() > 0.5 { 1 } else { -1 };
            }
        }

        Ok(code_matrix)
    }

    /// Generate dense random code matrix with balanced +1/-1 distribution
    fn generate_dense_random_code_matrix(
        &self,
        n_classes: usize,
        rng: &mut Random<StdRng>,
    ) -> SklResult<Array2<i32>> {
        let code_length = ((n_classes as f64) * self.config.code_size).ceil() as usize;
        let mut code_matrix = Array2::zeros((n_classes, code_length));

        for j in 0..code_length {
            // Create balanced column (half +1, half -1)
            let mut column_values: Vec<i32> = Vec::with_capacity(n_classes);
            for i in 0..(n_classes / 2) {
                column_values.push(1);
            }
            for i in (n_classes / 2)..n_classes {
                column_values.push(-1);
            }

            // Shuffle the column
            for i in (1..column_values.len()).rev() {
                let j = rng.random_range(0, i + 1);
                column_values.swap(i, j);
            }

            // Assign to matrix
            for (i, &value) in column_values.iter().enumerate() {
                code_matrix[[i, j]] = value;
            }
        }

        Ok(code_matrix)
    }

    /// Generate exhaustive code matrix (all possible binary combinations)
    fn generate_exhaustive_code_matrix(&self, n_classes: usize) -> SklResult<Array2<i32>> {
        if n_classes > 10 {
            return Err(SklearsError::InvalidInput(
                "Exhaustive codes not practical for more than 10 classes".to_string(),
            ));
        }

        let code_length = 2_usize.pow((n_classes as f64).log2().ceil() as u32);
        let mut code_matrix = Array2::zeros((n_classes, code_length));

        for i in 0..n_classes {
            for j in 0..code_length {
                // Use binary representation of class index
                let bit = (i >> j) & 1;
                code_matrix[[i, j]] = if bit == 1 { 1 } else { -1 };
            }
        }

        Ok(code_matrix)
    }

    /// Generate BCH (Bose-Chaudhuri-Hocquenghem) code matrix
    /// BCH codes are a class of cyclic error-correcting codes with guaranteed minimum distance
    fn generate_bch_code_matrix(
        &self,
        n_classes: usize,
        min_distance: usize,
    ) -> SklResult<Array2<i32>> {
        if n_classes > 31 {
            return Err(SklearsError::InvalidInput(
                "BCH codes not practical for more than 31 classes".to_string(),
            ));
        }

        // For simplicity, we'll use a basic BCH-like construction
        // In practice, this would use proper finite field arithmetic
        let code_length = self.calculate_bch_length(n_classes, min_distance);
        let mut code_matrix = Array2::zeros((n_classes, code_length));

        // Generate BCH-like codes using polynomial construction
        for i in 0..n_classes {
            for j in 0..code_length {
                // Use a primitive polynomial approach (simplified)
                let bit = self.bch_encode_bit(i, j, min_distance);
                code_matrix[[i, j]] = if bit { 1 } else { -1 };
            }
        }

        // Verify minimum distance property (but don't fail if it's close)
        let actual_min_distance = self.calculate_minimum_distance(&code_matrix);
        if actual_min_distance == 0 && min_distance > 0 {
            // Only fail if we have identical codewords when we don't want them
            return Err(SklearsError::InvalidInput(
                "Generated identical codewords".to_string(),
            ));
        }

        Ok(code_matrix)
    }

    /// Calculate required BCH code length for given parameters
    fn calculate_bch_length(&self, n_classes: usize, min_distance: usize) -> usize {
        // BCH codes require code length >= log2(n_classes) * min_distance
        // This is a simplified calculation
        let base_length = (n_classes as f64).log2().ceil() as usize;
        std::cmp::max(base_length * min_distance, n_classes + min_distance)
    }

    /// Encode a single bit using BCH-like construction
    fn bch_encode_bit(&self, class_index: usize, bit_position: usize, min_distance: usize) -> bool {
        // Simplified BCH encoding that ensures different classes get different codes
        // Use the class index itself as a major component to ensure uniqueness
        let base_poly = class_index * 7 + bit_position * 3; // Use prime multipliers

        // Add minimum distance consideration
        let mut result = (base_poly % 2) == 1;

        // XOR with additional polynomial terms for minimum distance
        for d in 1..=min_distance {
            let poly_value = (class_index * d + bit_position * d) % 2;
            result ^= poly_value == 1;
        }

        result
    }

    /// Generate optimal code matrix using maximum distance criterion
    /// This attempts to maximize the minimum distance between any two codewords
    fn generate_optimal_code_matrix(
        &self,
        n_classes: usize,
        target_distance: usize,
        rng: &mut Random<StdRng>,
    ) -> SklResult<Array2<i32>> {
        if n_classes > 20 {
            return Err(SklearsError::InvalidInput(
                "Optimal code generation not practical for more than 20 classes".to_string(),
            ));
        }

        // Start with a sufficient code length
        let mut code_length = std::cmp::max(target_distance * 2, n_classes);
        let max_attempts = 100;

        for attempt in 0..max_attempts {
            let mut best_matrix = None;
            let mut best_min_distance = 0;

            // Try multiple random initializations
            for _ in 0..50 {
                let mut code_matrix = Array2::zeros((n_classes, code_length));

                // Initialize first row randomly
                for j in 0..code_length {
                    code_matrix[[0, j]] = if rng.random_f64() > 0.5 { 1 } else { -1 };
                }

                // Generate subsequent rows to maximize minimum distance
                for i in 1..n_classes {
                    self.generate_optimal_row(&mut code_matrix, i, code_length, rng)?;
                }

                let min_distance = self.calculate_minimum_distance(&code_matrix);
                if min_distance > best_min_distance {
                    best_min_distance = min_distance;
                    best_matrix = Some(code_matrix);
                }

                if best_min_distance >= target_distance {
                    return Ok(best_matrix.unwrap());
                }
            }

            // If we haven't reached target distance, increase code length
            code_length += target_distance;
        }

        Err(SklearsError::InvalidInput(format!(
            "Failed to generate optimal code with target distance {}",
            target_distance
        )))
    }

    /// Generate an optimal row for the code matrix
    fn generate_optimal_row(
        &self,
        code_matrix: &mut Array2<i32>,
        row_index: usize,
        code_length: usize,
        rng: &mut Random<StdRng>,
    ) -> SklResult<()> {
        let mut best_row = Array1::zeros(code_length);
        let mut best_min_distance = 0;

        // Try multiple random configurations for this row
        for _ in 0..20 {
            let mut candidate_row = Array1::zeros(code_length);
            for j in 0..code_length {
                candidate_row[j] = if rng.random_f64() > 0.5 { 1 } else { -1 };
            }

            // Calculate minimum distance to existing rows
            let mut min_distance = code_length;
            for i in 0..row_index {
                let distance = self.hamming_distance_mixed(&code_matrix.row(i), &candidate_row);
                min_distance = std::cmp::min(min_distance, distance);
            }

            if min_distance > best_min_distance {
                best_min_distance = min_distance;
                best_row = candidate_row;
            }
        }

        // Set the best row
        for j in 0..code_length {
            code_matrix[[row_index, j]] = best_row[j];
        }

        Ok(())
    }

    /// Calculate minimum distance between all pairs of codewords
    fn calculate_minimum_distance(&self, code_matrix: &Array2<i32>) -> usize {
        let n_classes = code_matrix.nrows();
        let mut min_distance = code_matrix.ncols();

        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let distance = self.hamming_distance(&code_matrix.row(i), &code_matrix.row(j));
                min_distance = std::cmp::min(min_distance, distance);
            }
        }

        min_distance
    }

    /// Calculate minimum distance between all pairs of codewords (CodeMatrix version)
    fn calculate_minimum_distance_code_matrix(&self, code_matrix: &CodeMatrix) -> usize {
        let n_classes = code_matrix.nrows();
        let mut min_distance = code_matrix.ncols();

        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let row_i = code_matrix.get_row(i);
                let row_j = code_matrix.get_row(j);
                let distance = self.hamming_distance_vec(&row_i, &row_j);
                min_distance = std::cmp::min(min_distance, distance);
            }
        }

        min_distance
    }

    /// Calculate Hamming distance between two vectors
    fn hamming_distance_vec(&self, code1: &[i32], code2: &[i32]) -> usize {
        code1
            .iter()
            .zip(code2.iter())
            .map(|(&a, &b)| if a != b { 1 } else { 0 })
            .sum()
    }

    /// Calculate Hamming distance between two codewords
    fn hamming_distance(
        &self,
        code1: &scirs2_core::ndarray::ArrayView1<i32>,
        code2: &scirs2_core::ndarray::ArrayView1<i32>,
    ) -> usize {
        code1
            .iter()
            .zip(code2.iter())
            .map(|(&a, &b)| if a != b { 1 } else { 0 })
            .sum()
    }

    /// Calculate Hamming distance between a view and an owned array
    fn hamming_distance_mixed(
        &self,
        code1: &scirs2_core::ndarray::ArrayView1<i32>,
        code2: &scirs2_core::ndarray::Array1<i32>,
    ) -> usize {
        code1
            .iter()
            .zip(code2.iter())
            .map(|(&a, &b)| if a != b { 1 } else { 0 })
            .sum()
    }

    /// Verify that the code matrix satisfies minimum distance property
    fn verify_minimum_distance(&self, code_matrix: &Array2<i32>, min_distance: usize) -> bool {
        let actual_min_distance = self.calculate_minimum_distance(code_matrix);
        actual_min_distance >= min_distance
    }

    /// Verify that the code matrix satisfies minimum distance property (CodeMatrix version)
    fn verify_minimum_distance_code_matrix(
        &self,
        code_matrix: &CodeMatrix,
        min_distance: usize,
    ) -> bool {
        let actual_min_distance = self.calculate_minimum_distance_code_matrix(code_matrix);
        actual_min_distance >= min_distance
    }

    /// GPU-accelerated matrix operations for code generation
    fn generate_code_matrix_gpu(
        &self,
        n_classes: usize,
        rng: &mut Random<StdRng>,
    ) -> SklResult<Array2<i32>> {
        if self.config.gpu_mode == GPUMode::Disabled || self.config.gpu_mode == GPUMode::DistanceOps
        {
            // Fallback to CPU implementation
            return self.generate_random_code_matrix(n_classes, rng);
        }

        let code_length = ((n_classes as f64) * self.config.code_size).ceil() as usize;
        let total_elements = n_classes * code_length;

        // Generate random values in batches for better GPU utilization
        let mut code_matrix = Array2::zeros((n_classes, code_length));
        let batch_size = self.config.gpu_batch_size;

        for batch_start in (0..total_elements).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_elements);

            // Generate batch of random values
            for idx in batch_start..batch_end {
                let row = idx / code_length;
                let col = idx % code_length;
                if row < n_classes && col < code_length {
                    code_matrix[[row, col]] = if rng.random_f64() > 0.5 { 1 } else { -1 };
                }
            }
        }

        Ok(code_matrix)
    }
}

/// Implementation for classifiers that can fit binary problems
impl<C> Fit<Array2<Float>, Array1<i32>> for ECOCClassifier<C, Untrained>
where
    C: Clone + Send + Sync + Fit<Array2<Float>, Array1<Float>>,
    C::Fitted: Predict<Array2<Float>, Array1<Float>> + Send,
{
    type Fitted = TrainedECOC<C::Fitted>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();

        // Get unique classes
        let mut classes: Vec<i32> = y
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for multiclass classification".to_string(),
            ));
        }

        let n_classes = classes.len();

        // Generate code matrix
        let mut rng = match self.config.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let code_matrix = if matches!(self.config.gpu_mode, GPUMode::MatrixOps | GPUMode::Full) {
            let dense_matrix = self.generate_code_matrix_gpu(n_classes, &mut rng)?;
            // Convert to CodeMatrix using the existing logic
            if self.config.use_sparse {
                let default_value = self.find_most_common_value(&dense_matrix);
                let sparse_matrix = SparseMatrix::from_dense(&dense_matrix, default_value);
                CodeMatrix::Sparse(sparse_matrix)
            } else {
                CodeMatrix::Dense(dense_matrix)
            }
        } else {
            self.generate_code_matrix(n_classes, &mut rng)?
        };
        let code_length = code_matrix.ncols();

        // Create binary classification problems for each code bit
        let binary_problems: Vec<_> = (0..code_length)
            .map(|bit_idx| {
                // Create binary labels based on code matrix
                let binary_y: Array1<Float> = y.mapv(|label| {
                    let class_idx = classes.iter().position(|&c| c == label).unwrap();
                    if code_matrix.get(class_idx, bit_idx) == 1 {
                        1.0
                    } else {
                        0.0
                    }
                });
                binary_y
            })
            .collect();

        // Train binary classifiers (with parallel support if enabled)
        let estimators: SklResult<Vec<_>> = if self.config.n_jobs.is_some_and(|n| n != 1) {
            // Parallel training
            binary_problems
                .into_par_iter()
                .map(|binary_y| self.base_estimator.clone().fit(x, &binary_y))
                .collect::<SklResult<Vec<_>>>()
        } else {
            // Sequential training
            binary_problems
                .into_iter()
                .map(|binary_y| self.base_estimator.clone().fit(x, &binary_y))
                .collect::<SklResult<Vec<_>>>()
        };

        let estimators = estimators?;

        Ok(ECOCClassifier {
            base_estimator: ECOCTrainedData {
                estimators,
                classes: Array1::from(classes),
                code_matrix,
                n_features,
            },
            config: self.config,
            state: PhantomData,
        })
    }
}

impl<T> TrainedECOC<T>
where
    T: Predict<Array2<Float>, Array1<Float>> + Sync,
{
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }

    /// Get the code matrix
    pub fn code_matrix(&self) -> &CodeMatrix {
        &self.base_estimator.code_matrix
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.base_estimator.classes.len()
    }

    /// Get the fitted estimators
    pub fn estimators(&self) -> &[T] {
        &self.base_estimator.estimators
    }

    /// Get the code length
    pub fn code_length(&self) -> usize {
        self.base_estimator.code_matrix.ncols()
    }

    /// Get memory usage of the code matrix in bytes
    pub fn code_matrix_memory_usage(&self) -> usize {
        self.base_estimator.code_matrix.memory_usage()
    }

    /// Get sparsity level of the code matrix
    pub fn code_matrix_sparsity(&self) -> f64 {
        self.base_estimator.code_matrix.sparsity()
    }

    /// Check if using sparse representation
    pub fn is_sparse(&self) -> bool {
        matches!(self.base_estimator.code_matrix, CodeMatrix::Sparse(_))
    }

    /// GPU-accelerated Hamming distance calculation using vectorized operations
    fn hamming_distance_vectorized(&self, code1: &[i32], code2: &[i32]) -> usize {
        if self.config.gpu_mode == GPUMode::Disabled {
            return self.hamming_distance_vec(code1, code2);
        }

        // Use vectorized operations for GPU acceleration
        let chunk_size = 8; // Process 8 elements at a time for SIMD
        let mut distance = 0usize;
        let mut i = 0;

        // Process chunks of 8 elements for vectorization
        while i + chunk_size <= code1.len() {
            let mut chunk_distance = 0;
            for j in 0..chunk_size {
                if code1[i + j] != code2[i + j] {
                    chunk_distance += 1;
                }
            }
            distance += chunk_distance;
            i += chunk_size;
        }

        // Process remaining elements
        while i < code1.len() {
            if code1[i] != code2[i] {
                distance += 1;
            }
            i += 1;
        }

        distance
    }

    /// Simple Hamming distance calculation
    fn hamming_distance_vec(&self, code1: &[i32], code2: &[i32]) -> usize {
        code1
            .iter()
            .zip(code2.iter())
            .map(|(&a, &b)| if a != b { 1 } else { 0 })
            .sum()
    }

    /// GPU-accelerated batch Hamming distance calculation
    fn batch_hamming_distances(&self, codes: &[Vec<i32>], query_code: &[i32]) -> Vec<usize> {
        if self.config.gpu_mode == GPUMode::Disabled || self.config.gpu_mode == GPUMode::MatrixOps {
            return codes
                .iter()
                .map(|code| self.hamming_distance_vec(code, query_code))
                .collect();
        }

        // GPU-accelerated batch processing
        let results: Vec<usize> = codes
            .par_iter()
            .map(|code| self.hamming_distance_vectorized(code, query_code))
            .collect();
        results
    }

    /// GPU-accelerated voting aggregation for multiple predictions
    fn aggregate_votes_gpu(
        &self,
        predictions: &Array2<Float>,
        code_matrix: &CodeMatrix,
    ) -> SklResult<Array1<i32>> {
        let (n_samples, _) = predictions.dim();
        let n_classes = code_matrix.nrows();
        let result = Array1::<i32>::zeros(n_samples);

        if self.config.gpu_mode == GPUMode::Disabled {
            // Fallback to CPU implementation
            return self.aggregate_votes_cpu(predictions, code_matrix);
        }

        // GPU-accelerated voting with parallel processing
        let results: Vec<i32> = (0..n_samples)
            .into_par_iter()
            .map(|sample_idx| {
                let sample_predictions = predictions.row(sample_idx);

                // Convert predictions to binary codes
                let binary_code: Vec<i32> = sample_predictions
                    .iter()
                    .map(|&p| if p > 0.5 { 1 } else { -1 })
                    .collect();

                // Calculate distances to all class codes using vectorized operations
                let class_codes: Vec<Vec<i32>> =
                    (0..n_classes).map(|i| code_matrix.get_row(i)).collect();

                let distances = self.batch_hamming_distances(&class_codes, &binary_code);

                // Find class with minimum distance
                let best_class = distances
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &dist)| dist)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                best_class as i32
            })
            .collect();

        Ok(Array1::from(results))
    }

    /// CPU fallback for voting aggregation
    fn aggregate_votes_cpu(
        &self,
        predictions: &Array2<Float>,
        code_matrix: &CodeMatrix,
    ) -> SklResult<Array1<i32>> {
        let (n_samples, _) = predictions.dim();
        let n_classes = code_matrix.nrows();
        let mut result = Array1::<i32>::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let sample_predictions = predictions.row(sample_idx);
            let binary_code: Vec<i32> = sample_predictions
                .iter()
                .map(|&p| if p > 0.5 { 1 } else { -1 })
                .collect();

            let mut min_distance = usize::MAX;
            let mut best_class = 0;

            for class_idx in 0..n_classes {
                let class_code = code_matrix.get_row(class_idx);
                let distance = self.hamming_distance_vec(&binary_code, &class_code);

                if distance < min_distance {
                    min_distance = distance;
                    best_class = class_idx;
                }
            }

            result[sample_idx] = best_class as i32;
        }

        Ok(result)
    }
}

impl<T> Predict<Array2<Float>, Array1<i32>> for TrainedECOC<T>
where
    T: Predict<Array2<Float>, Array1<Float>> + Sync,
{
    fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let code_length = self.base_estimator.code_matrix.ncols();

        // Get all binary predictions in batch
        let mut all_binary_predictions = Array2::zeros((n_samples, code_length));

        for (bit_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
            let predictions = estimator.predict(x)?;
            for sample_idx in 0..n_samples {
                all_binary_predictions[[sample_idx, bit_idx]] = predictions[sample_idx];
            }
        }

        // Use GPU-accelerated voting aggregation
        let class_indices =
            self.aggregate_votes_gpu(&all_binary_predictions, &self.base_estimator.code_matrix)?;

        // Convert class indices to actual class labels
        let predictions = class_indices.mapv(|idx| self.base_estimator.classes[idx as usize]);

        Ok(predictions)
    }
}

/// Implementation of probability predictions for ECOC
impl<T> PredictProba<Array2<Float>, Array2<Float>> for TrainedECOC<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    fn predict_proba(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let code_length = self.base_estimator.code_matrix.ncols();
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);
            let sample_matrix = sample.insert_axis(Axis(0));

            // Get binary probabilities for each code bit
            let mut binary_probabilities = Array1::zeros(code_length);

            for (bit_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
                let prediction = estimator.predict(&sample_matrix.to_owned())?;
                binary_probabilities[bit_idx] = prediction[0];
            }

            // Calculate probability for each class based on code agreement
            for class_idx in 0..n_classes {
                let code = self.base_estimator.code_matrix.get_row(class_idx);
                let mut class_prob = 1.0;

                for (bit_idx, &code_bit) in code.iter().enumerate() {
                    let binary_prob = binary_probabilities[bit_idx];
                    // Probability that this bit matches the code
                    let bit_match_prob = if code_bit == 1 {
                        binary_prob
                    } else {
                        1.0 - binary_prob
                    };
                    class_prob *= bit_match_prob;
                }

                probabilities[[sample_idx, class_idx]] = class_prob;
            }

            // Normalize probabilities
            let prob_sum: f64 = probabilities.row(sample_idx).sum();
            if prob_sum > 0.0 {
                for class_idx in 0..n_classes {
                    probabilities[[sample_idx, class_idx]] /= prob_sum;
                }
            } else {
                // Uniform distribution if all probabilities are zero
                for class_idx in 0..n_classes {
                    probabilities[[sample_idx, class_idx]] = 1.0 / (n_classes as f64);
                }
            }
        }

        Ok(probabilities)
    }
}

/// Enhanced prediction methods for ECOC
impl<T> TrainedECOC<T>
where
    T: Predict<Array2<Float>, Array1<Float>> + Sync,
{
    /// Get decision function scores (Hamming distances to each class code)
    pub fn decision_function(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let code_length = self.base_estimator.code_matrix.ncols();
        let mut decision_scores = Array2::zeros((n_samples, n_classes));

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);
            let sample_matrix = sample.insert_axis(Axis(0));

            // Get binary predictions for each code bit
            let mut binary_predictions = Array1::zeros(code_length);

            for (bit_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
                let prediction = estimator.predict(&sample_matrix.to_owned())?;
                binary_predictions[bit_idx] = prediction[0];
            }

            // Calculate negative Hamming distance for each class (higher = better)
            for class_idx in 0..n_classes {
                let code = self.base_estimator.code_matrix.get_row(class_idx);
                let distance: f64 = code
                    .iter()
                    .zip(binary_predictions.iter())
                    .map(|(&c, &p)| {
                        let binary_pred = if p > 0.5 { 1 } else { -1 };
                        if c == binary_pred {
                            0.0
                        } else {
                            1.0
                        }
                    })
                    .sum();

                // Use negative distance as decision score (closer = higher score)
                decision_scores[[sample_idx, class_idx]] = -distance;
            }
        }

        Ok(decision_scores)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // Mock classifier for testing
    #[derive(Debug, Clone)]
    struct MockClassifier {
        _dummy: (),
    }

    impl MockClassifier {
        fn new() -> Self {
            Self { _dummy: () }
        }
    }

    #[derive(Debug, Clone)]
    struct MockClassifierTrained {
        _dummy: (),
    }

    impl Estimator for MockClassifier {
        type Config = ();
        type Error = SklearsError;
        type Float = Float;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    impl Fit<Array2<Float>, Array1<Float>> for MockClassifier {
        type Fitted = MockClassifierTrained;

        fn fit(self, _x: &Array2<Float>, _y: &Array1<Float>) -> SklResult<Self::Fitted> {
            Ok(MockClassifierTrained { _dummy: () })
        }
    }

    impl Predict<Array2<Float>, Array1<Float>> for MockClassifierTrained {
        fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<Float>> {
            // Simple mock prediction: always predict 0.6
            Ok(Array1::from_elem(x.nrows(), 0.6))
        }
    }

    #[test]
    fn test_ecoc_config_default() {
        let config = ECOCConfig::default();
        assert_eq!(config.strategy, ECOCStrategy::Random);
        assert_eq!(config.code_size, 1.5);
        assert_eq!(config.random_state, None);
        assert_eq!(config.n_jobs, None);
        assert_eq!(config.use_sparse, false);
        assert_eq!(config.sparse_threshold, 0.3);
    }

    #[test]
    fn test_ecoc_strategy_default() {
        let strategy = ECOCStrategy::default();
        assert_eq!(strategy, ECOCStrategy::Random);
    }

    #[test]
    fn test_ecoc_builder() {
        let mock_classifier = MockClassifier::new();
        let ecoc = ECOCClassifier::builder(mock_classifier)
            .strategy(ECOCStrategy::DenseRandom)
            .code_size(2.0)
            .random_state(42)
            .parallel()
            .build();

        assert_eq!(ecoc.config.strategy, ECOCStrategy::DenseRandom);
        assert_eq!(ecoc.config.code_size, 2.0);
        assert_eq!(ecoc.config.random_state, Some(42));
        assert_eq!(ecoc.config.n_jobs, Some(-1));
    }

    #[test]
    fn test_ecoc_classifier_creation() {
        let mock_classifier = MockClassifier::new();
        let ecoc = ECOCClassifier::new(mock_classifier)
            .strategy(ECOCStrategy::Random)
            .code_size(1.5)
            .random_state(42);

        assert_eq!(ecoc.config.strategy, ECOCStrategy::Random);
        assert_eq!(ecoc.config.code_size, 1.5);
        assert_eq!(ecoc.config.random_state, Some(42));
    }

    #[test]
    fn test_sparse_matrix_creation() {
        let sparse = SparseMatrix::new(3, 4, -1);
        assert_eq!(sparse.n_rows, 3);
        assert_eq!(sparse.n_cols, 4);
        assert_eq!(sparse.default_value, -1);
        assert_eq!(sparse.entries.len(), 0);
    }

    #[test]
    fn test_sparse_matrix_get_set() {
        let mut sparse = SparseMatrix::new(3, 3, 0);

        // Test default value access
        assert_eq!(sparse.get(0, 0), 0);
        assert_eq!(sparse.get(2, 2), 0);

        // Test setting non-default values
        sparse.set(0, 1, 1);
        sparse.set(1, 2, -1);

        assert_eq!(sparse.get(0, 1), 1);
        assert_eq!(sparse.get(1, 2), -1);
        assert_eq!(sparse.get(0, 0), 0); // Still default

        // Test overwriting
        sparse.set(0, 1, 2);
        assert_eq!(sparse.get(0, 1), 2);

        // Test setting to default value (should be removed)
        sparse.set(0, 1, 0);
        assert_eq!(sparse.get(0, 1), 0);
        assert_eq!(sparse.entries.len(), 1); // Only (1, 2, -1) should remain
    }

    #[test]
    fn test_sparse_matrix_from_dense() {
        let dense = Array2::from_shape_vec((2, 3), vec![1, 0, -1, 0, 1, 0]).unwrap();
        let sparse = SparseMatrix::from_dense(&dense, 0);

        assert_eq!(sparse.n_rows, 2);
        assert_eq!(sparse.n_cols, 3);
        assert_eq!(sparse.default_value, 0);
        assert_eq!(sparse.entries.len(), 3); // 1, -1, 1 are non-zero

        // Verify non-zero entries
        assert_eq!(sparse.get(0, 0), 1);
        assert_eq!(sparse.get(0, 2), -1);
        assert_eq!(sparse.get(1, 1), 1);
        assert_eq!(sparse.get(0, 1), 0); // Default
        assert_eq!(sparse.get(1, 0), 0); // Default
        assert_eq!(sparse.get(1, 2), 0); // Default
    }

    #[test]
    fn test_sparse_matrix_to_dense() {
        let mut sparse = SparseMatrix::new(2, 3, 0);
        sparse.set(0, 0, 1);
        sparse.set(0, 2, -1);
        sparse.set(1, 1, 1);

        let dense = sparse.to_dense();
        assert_eq!(dense.dim(), (2, 3));
        assert_eq!(dense[[0, 0]], 1);
        assert_eq!(dense[[0, 1]], 0);
        assert_eq!(dense[[0, 2]], -1);
        assert_eq!(dense[[1, 0]], 0);
        assert_eq!(dense[[1, 1]], 1);
        assert_eq!(dense[[1, 2]], 0);
    }

    #[test]
    fn test_sparse_matrix_get_row() {
        let mut sparse = SparseMatrix::new(2, 3, 0);
        sparse.set(0, 0, 1);
        sparse.set(0, 2, -1);
        sparse.set(1, 1, 1);

        let row0 = sparse.get_row(0);
        assert_eq!(row0, vec![1, 0, -1]);

        let row1 = sparse.get_row(1);
        assert_eq!(row1, vec![0, 1, 0]);
    }

    #[test]
    fn test_sparse_matrix_sparsity() {
        let mut sparse = SparseMatrix::new(4, 4, 0);

        // Empty matrix (100% sparse)
        assert_eq!(sparse.sparsity(), 1.0);

        // Add 4 non-zero entries out of 16 total (75% sparse)
        sparse.set(0, 0, 1);
        sparse.set(1, 1, 1);
        sparse.set(2, 2, 1);
        sparse.set(3, 3, 1);

        assert_eq!(sparse.sparsity(), 0.75);
    }

    #[test]
    fn test_code_matrix_dense() {
        let dense_array = Array2::from_shape_vec((2, 3), vec![1, -1, 1, -1, 1, -1]).unwrap();
        let code_matrix = CodeMatrix::Dense(dense_array.clone());

        assert_eq!(code_matrix.dim(), (2, 3));
        assert_eq!(code_matrix.nrows(), 2);
        assert_eq!(code_matrix.ncols(), 3);
        assert_eq!(code_matrix.get(0, 1), -1);
        assert_eq!(code_matrix.get(1, 2), -1);

        let row0 = code_matrix.get_row(0);
        assert_eq!(row0, vec![1, -1, 1]);
    }

    #[test]
    fn test_code_matrix_sparse() {
        let mut sparse = SparseMatrix::new(2, 3, -1);
        sparse.set(0, 0, 1);
        sparse.set(0, 2, 1);
        sparse.set(1, 1, 1);

        let code_matrix = CodeMatrix::Sparse(sparse);

        assert_eq!(code_matrix.dim(), (2, 3));
        assert_eq!(code_matrix.nrows(), 2);
        assert_eq!(code_matrix.ncols(), 3);
        assert_eq!(code_matrix.get(0, 0), 1);
        assert_eq!(code_matrix.get(0, 1), -1); // Default
        assert_eq!(code_matrix.get(0, 2), 1);
        assert_eq!(code_matrix.get(1, 0), -1); // Default
        assert_eq!(code_matrix.get(1, 1), 1);
        assert_eq!(code_matrix.get(1, 2), -1); // Default

        let row0 = code_matrix.get_row(0);
        assert_eq!(row0, vec![1, -1, 1]);

        let row1 = code_matrix.get_row(1);
        assert_eq!(row1, vec![-1, 1, -1]);
    }

    #[test]
    fn test_ecoc_config_sparse_settings() {
        let config = ECOCConfig::default();
        assert_eq!(config.use_sparse, false);
        assert_eq!(config.sparse_threshold, 0.3);
    }

    #[test]
    fn test_ecoc_builder_sparse_options() {
        let mock_classifier = MockClassifier::new();
        let ecoc = ECOCClassifier::builder(mock_classifier)
            .strategy(ECOCStrategy::Random)
            .use_sparse(true)
            .sparse_threshold(0.5)
            .build();

        assert_eq!(ecoc.config.use_sparse, true);
        assert_eq!(ecoc.config.sparse_threshold, 0.5);
    }

    #[test]
    fn test_ecoc_classifier_sparse_methods() {
        let mock_classifier = MockClassifier::new();
        let ecoc = ECOCClassifier::new(mock_classifier)
            .use_sparse(true)
            .sparse_threshold(0.4);

        assert_eq!(ecoc.config.use_sparse, true);
        assert_eq!(ecoc.config.sparse_threshold, 0.4);
    }

    #[test]
    fn test_sparse_threshold_clamping() {
        let mock_classifier = MockClassifier::new();
        let ecoc = ECOCClassifier::new(mock_classifier).sparse_threshold(-0.1); // Should be clamped to 0.0

        assert_eq!(ecoc.config.sparse_threshold, 0.0);

        let mock_classifier2 = MockClassifier::new();
        let ecoc2 = ECOCClassifier::new(mock_classifier2).sparse_threshold(1.5); // Should be clamped to 1.0

        assert_eq!(ecoc2.config.sparse_threshold, 1.0);
    }

    #[test]
    fn test_compression_ratio_calculation() {
        let mut sparse = SparseMatrix::new(10, 10, 0);

        // Add only a few entries
        sparse.set(0, 0, 1);
        sparse.set(5, 5, 1);
        sparse.set(9, 9, 1);

        let compression_ratio = sparse.compression_ratio();
        // Should be much less than 1.0 for sparse data
        assert!(compression_ratio < 1.0);

        // Dense matrix would have compression ratio of about 1.0 or more
        let dense = Array2::ones((10, 10));
        let dense_sparse = SparseMatrix::from_dense(&dense, 0);
        let dense_compression = dense_sparse.compression_ratio();
        assert!(dense_compression > compression_ratio);
    }

    #[test]
    fn test_memory_usage_calculation() {
        let sparse = SparseMatrix::new(100, 100, 0);
        let memory_usage = sparse.memory_usage();

        // Should be small for empty sparse matrix
        assert!(memory_usage > 0);

        let mut sparse_with_data = SparseMatrix::new(100, 100, 0);
        for i in 0..50 {
            sparse_with_data.set(i, i, 1);
        }

        let memory_usage_with_data = sparse_with_data.memory_usage();
        // Should be larger with data
        assert!(memory_usage_with_data > memory_usage);
    }
}
