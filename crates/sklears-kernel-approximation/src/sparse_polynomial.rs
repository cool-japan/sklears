//! Sparse polynomial features for memory-efficient computation

use scirs2_core::ndarray::Array2;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::{collections::HashMap, marker::PhantomData};

/// Sparsity threshold strategy
#[derive(Debug, Clone)]
/// SparsityStrategy
pub enum SparsityStrategy {
    /// Absolute threshold (values below this are considered zero)
    Absolute(Float),
    /// Relative threshold (fraction of max value)
    Relative(Float),
    /// Top-K sparsity (keep only K largest values)
    TopK(usize),
    /// Percentile-based (keep values above this percentile)
    Percentile(Float),
}

/// Sparse storage format
#[derive(Debug, Clone)]
/// SparseFormat
pub enum SparseFormat {
    /// Coordinate format (COO) - stores (row, col, value) triplets
    Coordinate,
    /// Compressed Sparse Row (CSR) - optimized for row operations
    CompressedSparseRow,
    /// Compressed Sparse Column (CSC) - optimized for column operations
    CompressedSparseColumn,
    /// Dictionary of Keys (DOK) - uses hash map for efficient updates
    DictionaryOfKeys,
}

/// Sparse matrix representation for polynomial features
#[derive(Debug, Clone)]
/// SparseMatrix
pub struct SparseMatrix {
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
    /// Storage format
    pub format: SparseFormat,
    /// Data storage - interpretation depends on format
    pub data: SparseData,
}

/// Sparse data storage variants
#[derive(Debug, Clone)]
/// SparseData
pub enum SparseData {
    /// Coordinate format: (row_indices, col_indices, values)
    Coordinate(Vec<usize>, Vec<usize>, Vec<Float>),
    /// CSR format: (row_ptr, col_indices, values)
    CSR(Vec<usize>, Vec<usize>, Vec<Float>),
    /// CSC format: (col_ptr, row_indices, values)
    CSC(Vec<usize>, Vec<usize>, Vec<Float>),
    /// DOK format: HashMap with (row, col) -> value mapping
    DOK(HashMap<(usize, usize), Float>),
}

impl SparseMatrix {
    /// Create a new sparse matrix
    pub fn new(nrows: usize, ncols: usize, format: SparseFormat) -> Self {
        let data = match format {
            SparseFormat::Coordinate => SparseData::Coordinate(Vec::new(), Vec::new(), Vec::new()),
            SparseFormat::CompressedSparseRow => {
                SparseData::CSR(vec![0; nrows + 1], Vec::new(), Vec::new())
            }
            SparseFormat::CompressedSparseColumn => {
                SparseData::CSC(vec![0; ncols + 1], Vec::new(), Vec::new())
            }
            SparseFormat::DictionaryOfKeys => SparseData::DOK(HashMap::new()),
        };

        Self {
            nrows,
            ncols,
            format,
            data,
        }
    }

    /// Insert a value at (row, col)
    pub fn insert(&mut self, row: usize, col: usize, value: Float) {
        if row >= self.nrows || col >= self.ncols {
            return;
        }

        match &mut self.data {
            SparseData::Coordinate(rows, cols, vals) => {
                rows.push(row);
                cols.push(col);
                vals.push(value);
            }
            SparseData::DOK(map) => {
                if value.abs() > 1e-15 {
                    map.insert((row, col), value);
                } else {
                    map.remove(&(row, col));
                }
            }
            _ => {
                // Convert to DOK for easy insertion, then convert back
                self.to_dok();
                if let SparseData::DOK(map) = &mut self.data {
                    if value.abs() > 1e-15 {
                        map.insert((row, col), value);
                    } else {
                        map.remove(&(row, col));
                    }
                }
            }
        }
    }

    /// Convert to DOK format
    pub fn to_dok(&mut self) {
        let mut map = HashMap::new();

        match &self.data {
            SparseData::Coordinate(rows, cols, vals) => {
                for ((row, col), val) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
                    if val.abs() > 1e-15 {
                        map.insert((*row, *col), *val);
                    }
                }
            }
            SparseData::CSR(row_ptr, col_indices, values) => {
                for row in 0..self.nrows {
                    for idx in row_ptr[row]..row_ptr[row + 1] {
                        if idx < col_indices.len() && idx < values.len() {
                            let col = col_indices[idx];
                            let val = values[idx];
                            if val.abs() > 1e-15 {
                                map.insert((row, col), val);
                            }
                        }
                    }
                }
            }
            SparseData::DOK(_) => return, // Already DOK
            _ => {}                       // Other formats not implemented for conversion
        }

        self.data = SparseData::DOK(map);
        self.format = SparseFormat::DictionaryOfKeys;
    }

    /// Convert to CSR format
    pub fn to_csr(&mut self) {
        self.to_dok(); // First convert to DOK for easy processing

        if let SparseData::DOK(map) = &self.data {
            let mut triplets: Vec<(usize, usize, Float)> = map
                .iter()
                .map(|((row, col), val)| (*row, *col, *val))
                .collect();

            // Sort by row, then by column
            triplets.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            let mut row_ptr = vec![0; self.nrows + 1];
            let mut col_indices = Vec::new();
            let mut values = Vec::new();

            for (row, col, val) in triplets {
                col_indices.push(col);
                values.push(val);
                row_ptr[row + 1] += 1;
            }

            // Convert counts to cumulative sums
            for i in 1..row_ptr.len() {
                row_ptr[i] += row_ptr[i - 1];
            }

            self.data = SparseData::CSR(row_ptr, col_indices, values);
            self.format = SparseFormat::CompressedSparseRow;
        }
    }

    /// Get value at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Float {
        if row >= self.nrows || col >= self.ncols {
            return 0.0;
        }

        match &self.data {
            SparseData::DOK(map) => map.get(&(row, col)).copied().unwrap_or(0.0),
            SparseData::CSR(row_ptr, col_indices, values) => {
                for idx in row_ptr[row]..row_ptr[row + 1] {
                    if idx < col_indices.len() && col_indices[idx] == col {
                        return values.get(idx).copied().unwrap_or(0.0);
                    }
                }
                0.0
            }
            SparseData::Coordinate(rows, cols, vals) => {
                for ((r, c), val) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
                    if *r == row && *c == col {
                        return *val;
                    }
                }
                0.0
            }
            _ => 0.0,
        }
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> Array2<Float> {
        let mut dense = Array2::zeros((self.nrows, self.ncols));

        match &self.data {
            SparseData::DOK(map) => {
                for ((row, col), val) in map {
                    dense[(*row, *col)] = *val;
                }
            }
            SparseData::CSR(row_ptr, col_indices, values) => {
                for row in 0..self.nrows {
                    for idx in row_ptr[row]..row_ptr[row + 1] {
                        if idx < col_indices.len() && idx < values.len() {
                            let col = col_indices[idx];
                            let val = values[idx];
                            if col < self.ncols {
                                dense[(row, col)] = val;
                            }
                        }
                    }
                }
            }
            SparseData::Coordinate(rows, cols, vals) => {
                for ((row, col), val) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
                    if *row < self.nrows && *col < self.ncols {
                        dense[(*row, *col)] = *val;
                    }
                }
            }
            _ => {}
        }

        dense
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        match &self.data {
            SparseData::DOK(map) => map.len(),
            SparseData::CSR(_, _, values) => values.len(),
            SparseData::CSC(_, _, values) => values.len(),
            SparseData::Coordinate(_, _, values) => values.len(),
        }
    }

    /// Apply sparsity threshold
    pub fn apply_sparsity(&mut self, strategy: &SparsityStrategy) {
        self.to_dok(); // Convert to DOK for easy manipulation

        if let SparseData::DOK(map) = &mut self.data {
            match strategy {
                SparsityStrategy::Absolute(threshold) => {
                    map.retain(|_, val| val.abs() >= *threshold);
                }
                SparsityStrategy::Relative(fraction) => {
                    if let Some(max_val) = map
                        .values()
                        .map(|v| v.abs())
                        .fold(None, |acc, v| Some(acc.map_or(v, |a: Float| a.max(v))))
                    {
                        let threshold = max_val * fraction;
                        map.retain(|_, val| val.abs() >= threshold);
                    }
                }
                SparsityStrategy::TopK(k) => {
                    if map.len() > *k {
                        let mut values: Vec<_> = map.iter().collect();
                        values.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

                        let new_map: HashMap<_, _> =
                            values.into_iter().take(*k).map(|(k, v)| (*k, *v)).collect();
                        *map = new_map;
                    }
                }
                SparsityStrategy::Percentile(percentile) => {
                    let mut abs_values: Vec<Float> = map.values().map(|v| v.abs()).collect();
                    abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    if !abs_values.is_empty() {
                        let idx = ((percentile / 100.0) * abs_values.len() as Float) as usize;
                        let threshold = abs_values.get(idx).copied().unwrap_or(0.0);
                        map.retain(|_, val| val.abs() >= threshold);
                    }
                }
            }
        }
    }
}

/// Sparse Polynomial Features
///
/// Generates polynomial features using sparse matrix representations for
/// memory-efficient computation, especially useful for high-dimensional
/// polynomial feature spaces with many zero values.
///
/// # Parameters
///
/// * `degree` - Maximum degree of polynomial features (default: 2)
/// * `interaction_only` - Include only interaction features (default: false)
/// * `include_bias` - Include bias column (default: true)
/// * `sparsity_strategy` - Strategy for enforcing sparsity
/// * `sparse_format` - Internal sparse matrix format
/// * `sparsity_threshold` - Minimum absolute value to keep (for automatic sparsity)
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::sparse_polynomial::{SparsePolynomialFeatures, SparsityStrategy};
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
///
/// let sparse_poly = SparsePolynomialFeatures::new(2)
///     .sparsity_strategy(SparsityStrategy::Absolute(0.1));
/// let fitted_sparse = sparse_poly.fit(&X, &()).unwrap();
/// let X_transformed = fitted_sparse.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
/// SparsePolynomialFeatures
pub struct SparsePolynomialFeatures<State = Untrained> {
    /// Maximum degree of polynomial features
    pub degree: u32,
    /// Include only interaction features
    pub interaction_only: bool,
    /// Include bias column
    pub include_bias: bool,
    /// Sparsity enforcement strategy
    pub sparsity_strategy: SparsityStrategy,
    /// Sparse matrix format
    pub sparse_format: SparseFormat,
    /// Automatic sparsity threshold
    pub sparsity_threshold: Float,

    // Fitted attributes
    n_input_features_: Option<usize>,
    n_output_features_: Option<usize>,
    powers_: Option<Vec<Vec<u32>>>,
    feature_indices_: Option<HashMap<Vec<u32>, usize>>,

    _state: PhantomData<State>,
}

impl SparsePolynomialFeatures<Untrained> {
    /// Create a new sparse polynomial features transformer
    pub fn new(degree: u32) -> Self {
        Self {
            degree,
            interaction_only: false,
            include_bias: true,
            sparsity_strategy: SparsityStrategy::Absolute(1e-10),
            sparse_format: SparseFormat::DictionaryOfKeys,
            sparsity_threshold: 1e-10,
            n_input_features_: None,
            n_output_features_: None,
            powers_: None,
            feature_indices_: None,
            _state: PhantomData,
        }
    }

    /// Set interaction_only parameter
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Set include_bias parameter
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Set sparsity strategy
    pub fn sparsity_strategy(mut self, strategy: SparsityStrategy) -> Self {
        self.sparsity_strategy = strategy;
        self
    }

    /// Set sparse matrix format
    pub fn sparse_format(mut self, format: SparseFormat) -> Self {
        self.sparse_format = format;
        self
    }

    /// Set automatic sparsity threshold
    pub fn sparsity_threshold(mut self, threshold: Float) -> Self {
        self.sparsity_threshold = threshold;
        self
    }
}

impl Estimator for SparsePolynomialFeatures<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for SparsePolynomialFeatures<Untrained> {
    type Fitted = SparsePolynomialFeatures<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (_, n_features) = x.dim();

        if self.degree == 0 {
            return Err(SklearsError::InvalidInput(
                "degree must be positive".to_string(),
            ));
        }

        // Generate all combinations of powers
        let powers = self.generate_powers(n_features)?;
        let n_output_features = powers.len();

        // Create feature index mapping
        let feature_indices: HashMap<Vec<u32>, usize> = powers
            .iter()
            .enumerate()
            .map(|(i, power)| (power.clone(), i))
            .collect();

        Ok(SparsePolynomialFeatures {
            degree: self.degree,
            interaction_only: self.interaction_only,
            include_bias: self.include_bias,
            sparsity_strategy: self.sparsity_strategy,
            sparse_format: self.sparse_format,
            sparsity_threshold: self.sparsity_threshold,
            n_input_features_: Some(n_features),
            n_output_features_: Some(n_output_features),
            powers_: Some(powers),
            feature_indices_: Some(feature_indices),
            _state: PhantomData,
        })
    }
}

impl SparsePolynomialFeatures<Untrained> {
    fn generate_powers(&self, n_features: usize) -> Result<Vec<Vec<u32>>> {
        let mut powers = Vec::new();

        // Add bias term if requested
        if self.include_bias {
            powers.push(vec![0; n_features]);
        }

        // Generate all combinations up to degree
        self.generate_all_combinations(n_features, self.degree, &mut powers);

        Ok(powers)
    }

    fn generate_all_combinations(
        &self,
        n_features: usize,
        max_degree: u32,
        powers: &mut Vec<Vec<u32>>,
    ) {
        // Generate all combinations with total degree from 1 to max_degree
        for total_degree in 1..=max_degree {
            self.generate_combinations_with_total_degree(
                n_features,
                total_degree,
                0,
                &mut vec![0; n_features],
                powers,
            );
        }
    }

    fn generate_combinations_with_total_degree(
        &self,
        n_features: usize,
        total_degree: u32,
        feature_idx: usize,
        current: &mut Vec<u32>,
        powers: &mut Vec<Vec<u32>>,
    ) {
        if feature_idx == n_features {
            let sum: u32 = current.iter().sum();
            if sum == total_degree {
                // Check if it's valid for interaction_only mode
                if !self.interaction_only || self.is_valid_for_interaction_only(current) {
                    powers.push(current.clone());
                }
            }
            return;
        }

        let current_sum: u32 = current.iter().sum();
        let remaining_degree = total_degree - current_sum;

        // Try different powers for current feature
        for power in 0..=remaining_degree {
            current[feature_idx] = power;
            self.generate_combinations_with_total_degree(
                n_features,
                total_degree,
                feature_idx + 1,
                current,
                powers,
            );
        }
        current[feature_idx] = 0;
    }

    fn is_valid_for_interaction_only(&self, powers: &[u32]) -> bool {
        let non_zero_count = powers.iter().filter(|&&p| p > 0).count();
        let max_power = powers.iter().max().unwrap_or(&0);

        // Valid if:
        // 1. It's a linear term (single variable with power 1)
        // 2. It's an interaction term (multiple variables, each with power 1)
        if non_zero_count == 1 {
            *max_power == 1
        } else {
            *max_power == 1
        }
    }
}

impl Transform<Array2<Float>, Array2<Float>> for SparsePolynomialFeatures<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let n_input_features = self.n_input_features_.unwrap();
        let n_output_features = self.n_output_features_.unwrap();
        let powers = self.powers_.as_ref().unwrap();

        if n_features != n_input_features {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but SparsePolynomialFeatures was fitted with {} features",
                n_features, n_input_features
            )));
        }

        // Create sparse matrix for intermediate computation
        let mut sparse_result =
            SparseMatrix::new(n_samples, n_output_features, self.sparse_format.clone());

        for i in 0..n_samples {
            for (j, power_combination) in powers.iter().enumerate() {
                let mut feature_value = 1.0;
                for (k, &power) in power_combination.iter().enumerate() {
                    if power > 0 {
                        feature_value *= x[[i, k]].powi(power as i32);
                    }
                }

                // Only store non-zero values
                if feature_value.abs() > self.sparsity_threshold {
                    sparse_result.insert(i, j, feature_value);
                }
            }
        }

        // Apply sparsity strategy
        sparse_result.apply_sparsity(&self.sparsity_strategy);

        // Convert to dense for return (could be optimized to return sparse in the future)
        Ok(sparse_result.to_dense())
    }
}

impl SparsePolynomialFeatures<Trained> {
    /// Get the number of input features
    pub fn n_input_features(&self) -> usize {
        self.n_input_features_.unwrap()
    }

    /// Get the number of output features
    pub fn n_output_features(&self) -> usize {
        self.n_output_features_.unwrap()
    }

    /// Get the powers for each feature
    pub fn powers(&self) -> &[Vec<u32>] {
        self.powers_.as_ref().unwrap()
    }

    /// Get the feature indices mapping
    pub fn feature_indices(&self) -> &HashMap<Vec<u32>, usize> {
        self.feature_indices_.as_ref().unwrap()
    }

    /// Transform and return as sparse matrix (more efficient for sparse data)
    pub fn transform_sparse(&self, x: &Array2<Float>) -> Result<SparseMatrix> {
        let (n_samples, n_features) = x.dim();
        let n_input_features = self.n_input_features_.unwrap();
        let n_output_features = self.n_output_features_.unwrap();
        let powers = self.powers_.as_ref().unwrap();

        if n_features != n_input_features {
            return Err(SklearsError::InvalidInput(format!(
                "X has {} features, but SparsePolynomialFeatures was fitted with {} features",
                n_features, n_input_features
            )));
        }

        let mut sparse_result =
            SparseMatrix::new(n_samples, n_output_features, self.sparse_format.clone());

        for i in 0..n_samples {
            for (j, power_combination) in powers.iter().enumerate() {
                let mut feature_value = 1.0;
                for (k, &power) in power_combination.iter().enumerate() {
                    if power > 0 {
                        feature_value *= x[[i, k]].powi(power as i32);
                    }
                }

                if feature_value.abs() > self.sparsity_threshold {
                    sparse_result.insert(i, j, feature_value);
                }
            }
        }

        sparse_result.apply_sparsity(&self.sparsity_strategy);
        Ok(sparse_result)
    }

    /// Estimate memory usage compared to dense representation
    pub fn memory_efficiency(&self, x: &Array2<Float>) -> Result<(usize, usize, Float)> {
        let sparse_result = self.transform_sparse(x)?;
        let nnz = sparse_result.nnz();
        let total_elements = sparse_result.nrows * sparse_result.ncols;
        let sparsity_ratio = 1.0 - (nnz as Float / total_elements as Float);

        Ok((nnz, total_elements, sparsity_ratio))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sparse_matrix_basic() {
        let mut sparse = SparseMatrix::new(3, 3, SparseFormat::DictionaryOfKeys);

        sparse.insert(0, 0, 1.0);
        sparse.insert(1, 1, 2.0);
        sparse.insert(2, 2, 3.0);

        assert_abs_diff_eq!(sparse.get(0, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(1, 1), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(2, 2), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(0, 1), 0.0, epsilon = 1e-10);

        assert_eq!(sparse.nnz(), 3);
    }

    #[test]
    fn test_sparse_matrix_to_dense() {
        let mut sparse = SparseMatrix::new(2, 2, SparseFormat::DictionaryOfKeys);
        sparse.insert(0, 0, 1.0);
        sparse.insert(0, 1, 2.0);
        sparse.insert(1, 0, 3.0);
        sparse.insert(1, 1, 4.0);

        let dense = sparse.to_dense();
        let expected = array![[1.0, 2.0], [3.0, 4.0]];

        for ((i, j), &expected_val) in expected.indexed_iter() {
            assert_abs_diff_eq!(dense[(i, j)], expected_val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sparse_matrix_csr_conversion() {
        let mut sparse = SparseMatrix::new(2, 3, SparseFormat::DictionaryOfKeys);
        sparse.insert(0, 0, 1.0);
        sparse.insert(0, 2, 2.0);
        sparse.insert(1, 1, 3.0);

        sparse.to_csr();

        assert_abs_diff_eq!(sparse.get(0, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(0, 2), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(1, 1), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(0, 1), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sparse_polynomial_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let sparse_poly = SparsePolynomialFeatures::new(2);
        let fitted = sparse_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 2);
        // Features: [1, a, b, a^2, ab, b^2] = 6 features
        assert_eq!(x_transformed.ncols(), 6);
    }

    #[test]
    fn test_sparse_polynomial_sparsity_strategies() {
        let x = array![[0.1, 0.0], [0.0, 0.2]]; // Sparse input

        let strategies = vec![
            SparsityStrategy::Absolute(0.01),
            SparsityStrategy::Relative(0.1),
            SparsityStrategy::TopK(3),
            SparsityStrategy::Percentile(50.0),
        ];

        for strategy in strategies {
            let sparse_poly = SparsePolynomialFeatures::new(2).sparsity_strategy(strategy);
            let fitted = sparse_poly.fit(&x, &()).unwrap();
            let x_transformed = fitted.transform(&x).unwrap();

            assert_eq!(x_transformed.nrows(), 2);
            assert!(x_transformed.ncols() > 0);
        }
    }

    #[test]
    fn test_sparse_polynomial_interaction_only() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let sparse_poly = SparsePolynomialFeatures::new(2).interaction_only(true);
        let fitted = sparse_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 2);
        // Should exclude pure powers like a^2, b^2
        assert!(x_transformed.ncols() >= 2); // At least bias + interaction terms
    }

    #[test]
    fn test_sparse_polynomial_no_bias() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let sparse_poly = SparsePolynomialFeatures::new(2).include_bias(false);
        let fitted = sparse_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 2);
        // Features: [a, b, a^2, ab, b^2] = 5 features (no bias)
        assert_eq!(x_transformed.ncols(), 5);
    }

    #[test]
    fn test_sparse_polynomial_transform_sparse() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let sparse_poly = SparsePolynomialFeatures::new(2);
        let fitted = sparse_poly.fit(&x, &()).unwrap();
        let sparse_result = fitted.transform_sparse(&x).unwrap();

        assert_eq!(sparse_result.nrows, 2);
        assert_eq!(sparse_result.ncols, 6);
        assert!(sparse_result.nnz() > 0);
    }

    #[test]
    fn test_sparse_polynomial_memory_efficiency() {
        let x = array![[0.1, 0.0], [0.0, 0.2], [0.0, 0.0]]; // Very sparse input

        let sparse_poly =
            SparsePolynomialFeatures::new(2).sparsity_strategy(SparsityStrategy::Absolute(0.01));
        let fitted = sparse_poly.fit(&x, &()).unwrap();

        let (nnz, total, sparsity_ratio) = fitted.memory_efficiency(&x).unwrap();

        assert!(nnz < total);
        assert!(sparsity_ratio > 0.0);
        assert!(sparsity_ratio <= 1.0);
    }

    #[test]
    fn test_sparse_polynomial_different_formats() {
        let x = array![[1.0, 2.0]];

        let formats = vec![
            SparseFormat::DictionaryOfKeys,
            SparseFormat::CompressedSparseRow,
            SparseFormat::Coordinate,
        ];

        for format in formats {
            let sparse_poly = SparsePolynomialFeatures::new(2).sparse_format(format);
            let fitted = sparse_poly.fit(&x, &()).unwrap();
            let x_transformed = fitted.transform(&x).unwrap();

            assert_eq!(x_transformed.nrows(), 1);
            assert_eq!(x_transformed.ncols(), 6);
        }
    }

    #[test]
    fn test_sparse_polynomial_feature_mismatch() {
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Different number of features

        let sparse_poly = SparsePolynomialFeatures::new(2);
        let fitted = sparse_poly.fit(&x_train, &()).unwrap();
        let result = fitted.transform(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_matrix_sparsity_thresholding() {
        let mut sparse = SparseMatrix::new(2, 2, SparseFormat::DictionaryOfKeys);
        sparse.insert(0, 0, 1.0);
        sparse.insert(0, 1, 0.001); // Small value
        sparse.insert(1, 0, 0.5);
        sparse.insert(1, 1, 0.0001); // Very small value

        sparse.apply_sparsity(&SparsityStrategy::Absolute(0.01));

        // Only values >= 0.01 should remain
        assert_abs_diff_eq!(sparse.get(0, 0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(0, 1), 0.0, epsilon = 1e-10); // Removed
        assert_abs_diff_eq!(sparse.get(1, 0), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(1, 1), 0.0, epsilon = 1e-10); // Removed

        assert_eq!(sparse.nnz(), 2);
    }

    #[test]
    fn test_sparse_polynomial_zero_degree() {
        let x = array![[1.0, 2.0]];
        let sparse_poly = SparsePolynomialFeatures::new(0);
        let result = sparse_poly.fit(&x, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_polynomial_single_feature() {
        let x = array![[2.0], [3.0]];

        let sparse_poly = SparsePolynomialFeatures::new(3);
        let fitted = sparse_poly.fit(&x, &()).unwrap();
        let x_transformed = fitted.transform(&x).unwrap();

        // Features: [1, a, a^2, a^3] = 4 features
        assert_eq!(x_transformed.shape(), &[2, 4]);
    }
}
