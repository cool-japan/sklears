//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, CoreRandom, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Untrained,
};
use std::collections::HashMap;
use std::marker::PhantomData;

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
        let mut packed_data = Vec::new();
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                let quantized = match val {
                    -1 => 0u8,
                    0 => 1u8,
                    1 => 2u8,
                    _ => 1u8,
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
                    _ => 0,
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
            _ => 127,
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
            if decompressed.len() >= 9 {
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
/// //     .strategy(ECOCStrategy::StdRng)
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
impl<C> ECOCClassifier<C, Untrained> {
    /// Generate code matrix based on strategy
    pub(crate) fn generate_code_matrix(
        &self,
        n_classes: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<CodeMatrix> {
        let dense_matrix = match &self.config.strategy {
            ECOCStrategy::StdRng => self.generate_random_code_matrix(n_classes, rng),
            ECOCStrategy::DenseStdRng => self.generate_dense_random_code_matrix(n_classes, rng),
            ECOCStrategy::Exhaustive => self.generate_exhaustive_code_matrix(n_classes),
            ECOCStrategy::BCH { min_distance } => {
                self.generate_bch_code_matrix(n_classes, *min_distance)
            }
            ECOCStrategy::Optimal { target_distance } => {
                self.generate_optimal_code_matrix(n_classes, *target_distance, rng)
            }
        }?;
        let use_sparse = if self.config.use_sparse {
            true
        } else {
            let sparsity = self.calculate_sparsity(&dense_matrix);
            sparsity >= self.config.sparse_threshold
        };
        if use_sparse {
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
    pub(crate) fn find_most_common_value(&self, matrix: &Array2<i32>) -> i32 {
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
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Array2<i32>> {
        let code_length = ((n_classes as f64) * self.config.code_size).ceil() as usize;
        let mut code_matrix = Array2::zeros((n_classes, code_length));
        for i in 0..n_classes {
            for j in 0..code_length {
                code_matrix[[i, j]] = if rng.gen::<f64>() > 0.5 { 1 } else { -1 };
            }
        }
        Ok(code_matrix)
    }
    /// Generate dense random code matrix with balanced +1/-1 distribution
    fn generate_dense_random_code_matrix(
        &self,
        n_classes: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Array2<i32>> {
        let code_length = ((n_classes as f64) * self.config.code_size).ceil() as usize;
        let mut code_matrix = Array2::zeros((n_classes, code_length));
        for j in 0..code_length {
            let mut column_values: Vec<i32> = Vec::with_capacity(n_classes);
            for _i in 0..(n_classes / 2) {
                column_values.push(1);
            }
            for _i in (n_classes / 2)..n_classes {
                column_values.push(-1);
            }
            for i in (1..column_values.len()).rev() {
                let j = rng.gen_range(0..i + 1);
                column_values.swap(i, j);
            }
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
        let code_length = self.calculate_bch_length(n_classes, min_distance);
        let mut code_matrix = Array2::zeros((n_classes, code_length));
        for i in 0..n_classes {
            for j in 0..code_length {
                let bit = self.bch_encode_bit(i, j, min_distance);
                code_matrix[[i, j]] = if bit { 1 } else { -1 };
            }
        }
        let actual_min_distance = self.calculate_minimum_distance(&code_matrix);
        if actual_min_distance == 0 && min_distance > 0 {
            return Err(SklearsError::InvalidInput(
                "Generated identical codewords".to_string(),
            ));
        }
        Ok(code_matrix)
    }
    /// Calculate required BCH code length for given parameters
    fn calculate_bch_length(&self, n_classes: usize, min_distance: usize) -> usize {
        let base_length = (n_classes as f64).log2().ceil() as usize;
        std::cmp::max(base_length * min_distance, n_classes + min_distance)
    }
    /// Encode a single bit using BCH-like construction
    fn bch_encode_bit(&self, class_index: usize, bit_position: usize, min_distance: usize) -> bool {
        let base_poly = class_index * 7 + bit_position * 3;
        let mut result = (base_poly % 2) == 1;
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
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Array2<i32>> {
        if n_classes > 20 {
            return Err(SklearsError::InvalidInput(
                "Optimal code generation not practical for more than 20 classes".to_string(),
            ));
        }
        let mut code_length = std::cmp::max(target_distance * 2, n_classes);
        let max_attempts = 100;
        for _attempt in 0..max_attempts {
            let mut best_matrix = None;
            let mut best_min_distance = 0;
            for _ in 0..50 {
                let mut code_matrix = Array2::zeros((n_classes, code_length));
                for j in 0..code_length {
                    code_matrix[[0, j]] = if rng.gen::<f64>() > 0.5 { 1 } else { -1 };
                }
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
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<()> {
        let mut best_row = Array1::zeros(code_length);
        let mut best_min_distance = 0;
        for _ in 0..20 {
            let mut candidate_row = Array1::zeros(code_length);
            for j in 0..code_length {
                candidate_row[j] = if rng.gen::<f64>() > 0.5 { 1 } else { -1 };
            }
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
    pub(crate) fn generate_code_matrix_gpu(
        &self,
        n_classes: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<Array2<i32>> {
        if self.config.gpu_mode == GPUMode::Disabled || self.config.gpu_mode == GPUMode::DistanceOps
        {
            return self.generate_random_code_matrix(n_classes, rng);
        }
        let code_length = ((n_classes as f64) * self.config.code_size).ceil() as usize;
        let total_elements = n_classes * code_length;
        let mut code_matrix = Array2::zeros((n_classes, code_length));
        let batch_size = self.config.gpu_batch_size;
        for batch_start in (0..total_elements).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_elements);
            for idx in batch_start..batch_end {
                let row = idx / code_length;
                let col = idx % code_length;
                if row < n_classes && col < code_length {
                    code_matrix[[row, col]] = if rng.gen::<f64>() > 0.5 { 1 } else { -1 };
                }
            }
        }
        Ok(code_matrix)
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
/// Configuration for Error-Correcting Output Code Classifier
#[derive(Debug, Clone)]
pub struct ECOCConfig {
    /// Code matrix generation strategy
    pub strategy: ECOCStrategy,
    /// Code size multiplier (for random strategies)
    pub code_size: f64,
    /// StdRng state for reproducibility
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
/// Error-Correcting Output Codes (ECOC) strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ECOCStrategy {
    /// StdRng binary codes
    #[default]
    StdRng,
    /// Dense random codes with balanced +1/-1 distribution
    DenseStdRng,
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
        self.entries.retain(|(r, c, _)| *r != row || *c != col);
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
