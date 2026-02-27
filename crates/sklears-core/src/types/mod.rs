pub mod advanced_numeric;
/// Type system for sklears machine learning library
///
/// This module provides a comprehensive type system for machine learning operations,
/// including core traits, array types, domain-specific types, and compile-time utilities.
///
/// The module is organized into focused sub-modules:
/// - `traits`: Core numeric trait definitions
/// - `arrays`: Array type aliases and utilities
/// - `domain`: Domain-specific ML types (Probability, LearningRate, etc.)
/// - `const_generic`: Compile-time fixed-size types
/// - `zero_copy`: Zero-copy array operations
pub mod arrays;
pub mod domain;
pub mod traits;
// pub mod memory_pool;  // TODO: Implement memory_pool module

// Re-export commonly used types for convenience
pub use arrays::{
    Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, CowFeatures, CowLabels,
    CowPredictions, CowProbabilities, CowSampleWeight, CowTarget, Distances, Features, Float, Int,
    Labels, Predictions, Probabilities, SampleWeight, Similarities, Target,
};

pub use domain::{
    common::{Prob, RegStrength, Tol, LR},
    FeatureCount, LearningRate, Probability, RegularizationStrength, SampleCount, Tolerance,
};

pub use traits::{Aggregatable, FloatBounds, IndexType, IntBounds, Numeric};

// Advanced numeric capabilities for high-performance ML
pub use advanced_numeric::{
    Complex, ComplexOps, GpuFloat, GpuOps, MemoryEfficientFloat, MemoryEfficientOps,
    NumericConversion, SimdF32, SimdF64, SimdOps,
};

// Memory management and zero-copy operations
// pub use memory_pool::{
//     CacheFriendlyArray, ChunkedProcessor, GlobalBufferPool, MemoryLayout, MemoryMappedArray,
//     MemoryPool, PooledBuffer, ZeroCopyArray, ZeroCopySubArray,
// };

// Const generic types for compile-time guarantees
pub mod const_generic {
    use super::traits::{FloatBounds, Numeric};
    use crate::error::{Result, SklearsError};
    // SciRS2 Policy: Using scirs2_core::ndarray for unified access (COMPLIANT)
    use scirs2_core::ndarray::Array2;

    /// Fixed-size feature vector with compile-time size checking
    #[derive(Debug, Clone, PartialEq)]
    pub struct FixedFeatures<T: FloatBounds, const N: usize> {
        data: [T; N],
    }

    impl<T: FloatBounds, const N: usize> FixedFeatures<T, N> {
        /// Create from array
        pub fn new(data: [T; N]) -> Self {
            Self { data }
        }

        /// Create from slice, checking length at runtime
        pub fn from_slice(slice: &[T]) -> Result<Self> {
            if slice.len() != N {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("{N}"),
                    actual: format!("{}", slice.len()),
                });
            }

            let mut data = [T::zero(); N];
            data.copy_from_slice(slice);
            Ok(Self { data })
        }

        /// Get as slice
        pub fn as_slice(&self) -> &[T] {
            &self.data
        }

        /// Get feature count at compile time
        pub const fn feature_count() -> usize {
            N
        }

        /// Dot product with another fixed features vector
        pub fn dot(&self, other: &Self) -> T {
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .fold(T::zero(), |acc, x| acc + x)
        }

        /// L2 norm
        pub fn norm(&self) -> T {
            self.dot(self).sqrt()
        }

        /// Normalize to unit length
        pub fn normalize(&self) -> Self {
            let norm = self.norm();
            if norm <= T::EPSILON {
                self.clone()
            } else {
                let mut normalized = [T::zero(); N];
                for (i, &val) in self.data.iter().enumerate() {
                    normalized[i] = val / norm;
                }
                Self::new(normalized)
            }
        }
    }

    impl<T: FloatBounds, const N: usize> std::ops::Index<usize> for FixedFeatures<T, N> {
        type Output = T;

        fn index(&self, index: usize) -> &Self::Output {
            &self.data[index]
        }
    }

    impl<T: FloatBounds, const N: usize> std::ops::IndexMut<usize> for FixedFeatures<T, N> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[index]
        }
    }

    /// Fixed-size dataset with compile-time shape checking
    #[derive(Debug, Clone, PartialEq)]
    pub struct FixedSamples<T: Numeric, const M: usize, const N: usize> {
        data: [[T; N]; M],
    }

    impl<T: Numeric, const M: usize, const N: usize> FixedSamples<T, M, N> {
        /// Create from nested array
        pub fn new(data: [[T; N]; M]) -> Self {
            Self { data }
        }

        /// Create from flat slice, checking length at runtime
        pub fn from_flat_slice(slice: &[T]) -> Result<Self> {
            if slice.len() != M * N {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("{}", M * N),
                    actual: format!("{}", slice.len()),
                });
            }

            let mut data = [[T::zero(); N]; M];
            for (i, chunk) in slice.chunks_exact(N).enumerate() {
                for (j, &val) in chunk.iter().enumerate() {
                    data[i][j] = val;
                }
            }
            Ok(Self { data })
        }

        /// Get sample count at compile time
        pub const fn sample_count() -> usize {
            M
        }

        /// Get feature count at compile time
        pub const fn feature_count() -> usize {
            N
        }

        /// Get shape at compile time
        pub const fn shape() -> (usize, usize) {
            (M, N)
        }

        /// Get a sample as FixedFeatures
        pub fn sample(&self, index: usize) -> FixedFeatures<T, N>
        where
            T: FloatBounds,
        {
            FixedFeatures::new(self.data[index])
        }

        /// Convert to dynamic Array2
        pub fn to_array2(&self) -> Array2<T> {
            let flat: Vec<T> = self.data.iter().flatten().copied().collect();
            Array2::from_shape_vec((M, N), flat).expect("Shape is guaranteed by const generics")
        }

        /// Create from Array2, checking shape at runtime
        pub fn from_array2(array: &Array2<T>) -> Result<Self> {
            let (rows, cols) = array.dim();
            if rows != M || cols != N {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("({M}, {N})"),
                    actual: format!("({rows}, {cols})"),
                });
            }

            let mut data = [[T::zero(); N]; M];
            for i in 0..M {
                for j in 0..N {
                    data[i][j] = array[[i, j]];
                }
            }
            Ok(Self { data })
        }
    }

    impl<T: Numeric, const M: usize, const N: usize> std::ops::Index<usize> for FixedSamples<T, M, N> {
        type Output = [T; N];

        fn index(&self, index: usize) -> &Self::Output {
            &self.data[index]
        }
    }

    impl<T: Numeric, const M: usize, const N: usize> std::ops::IndexMut<usize>
        for FixedSamples<T, M, N>
    {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[index]
        }
    }

    /// Matrix shape marker for compile-time verification
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MatrixShape<const ROWS: usize, const COLS: usize>;

    impl<const ROWS: usize, const COLS: usize> MatrixShape<ROWS, COLS> {
        /// Get the number of rows
        pub const fn rows() -> usize {
            ROWS
        }

        /// Get the number of columns
        pub const fn cols() -> usize {
            COLS
        }

        /// Get the total number of elements
        pub const fn total_elements() -> usize {
            ROWS * COLS
        }

        /// Check if the matrix is square
        pub const fn is_square() -> bool {
            ROWS == COLS
        }

        /// Check if shapes are compatible for matrix multiplication
        pub const fn matmul_compatible<const OTHER_COLS: usize>(
            _other: MatrixShape<COLS, OTHER_COLS>,
        ) -> bool {
            // The check is implicit in the type system:
            // COLS (our columns) must equal OTHER_ROWS (which is COLS in the parameter)
            true
        }
    }

    /// Type aliases for common fixed-size matrices
    pub type FixedFeaturesMatrix<T, const M: usize, const N: usize> = FixedSamples<T, M, N>;
    pub type FixedTarget<T, const N: usize> = FixedFeatures<T, N>;
    pub type FixedPredictions<T, const N: usize> = FixedFeatures<T, N>;

    /// Compile-time dimension constraints
    pub mod dimension_constraints {
        /// Ensures a type has a minimum number of features
        pub trait MinFeatures<const MIN: usize> {
            const FEATURE_COUNT: usize;
            const SATISFIES_MIN_FEATURES: bool = Self::FEATURE_COUNT >= MIN;
        }

        /// Ensures a type has a minimum number of samples
        pub trait MinSamples<const MIN: usize> {
            const SAMPLE_COUNT: usize;
            const SATISFIES_MIN_SAMPLES: bool = Self::SAMPLE_COUNT >= MIN;
        }

        /// Ensures matrix dimensions are valid for certain operations
        pub trait ValidDimensions {
            const IS_VALID: bool;
        }

        impl<T: super::Numeric, const M: usize, const N: usize> MinFeatures<1>
            for super::FixedSamples<T, M, N>
        {
            const FEATURE_COUNT: usize = N;
        }

        impl<T: super::Numeric, const M: usize, const N: usize> MinSamples<1>
            for super::FixedSamples<T, M, N>
        {
            const SAMPLE_COUNT: usize = M;
        }

        impl<T: super::Numeric, const M: usize, const N: usize> ValidDimensions
            for super::FixedSamples<T, M, N>
        {
            const IS_VALID: bool = M > 0 && N > 0;
        }
    }
}

// Zero-copy operations module
pub mod zero_copy {
    use super::arrays::{Array1, Array2, ArrayView2, Float};
    use crate::error::{Result, SklearsError};
    use scirs2_core::numeric::Zero;
    use std::borrow::Cow;

    /// Zero-copy array wrapper that can hold either owned or borrowed data
    #[derive(Debug)]
    pub enum ZeroCopyArray<'a, T> {
        /// Owned data
        Owned(Array2<T>),
        /// Borrowed data
        Borrowed(ArrayView2<'a, T>),
    }

    impl<'a, T> ZeroCopyArray<'a, T> {
        /// Create from owned array
        pub fn from_owned(array: Array2<T>) -> Self {
            ZeroCopyArray::Owned(array)
        }

        /// Create from borrowed array view
        pub fn from_borrowed(view: ArrayView2<'a, T>) -> Self {
            ZeroCopyArray::Borrowed(view)
        }

        /// Get shape
        pub fn shape(&self) -> (usize, usize) {
            match self {
                ZeroCopyArray::Owned(arr) => arr.dim(),
                ZeroCopyArray::Borrowed(view) => view.dim(),
            }
        }

        /// Get as array view
        pub fn view(&self) -> ArrayView2<'_, T> {
            match self {
                ZeroCopyArray::Owned(arr) => arr.view(),
                ZeroCopyArray::Borrowed(view) => view.view(),
            }
        }

        /// Convert to owned array (may clone if borrowed)
        pub fn into_owned(self) -> Array2<T>
        where
            T: Clone,
        {
            match self {
                ZeroCopyArray::Owned(arr) => arr,
                ZeroCopyArray::Borrowed(view) => view.to_owned(),
            }
        }

        /// Check if this is zero-copy (borrowed)
        pub fn is_zero_copy(&self) -> bool {
            matches!(self, ZeroCopyArray::Borrowed(_))
        }
    }

    /// Zero-copy features and target pair
    #[derive(Debug)]
    pub struct ZeroCopyDataset<'a, T = Float>
    where
        T: Clone,
    {
        pub features: ZeroCopyArray<'a, T>,
        pub target: Cow<'a, Array1<T>>,
    }

    impl<'a, T> ZeroCopyDataset<'a, T>
    where
        T: Clone,
    {
        /// Create from owned data
        pub fn from_owned(features: Array2<T>, target: Array1<T>) -> Self {
            Self {
                features: ZeroCopyArray::from_owned(features),
                target: Cow::Owned(target),
            }
        }

        /// Create from borrowed data
        pub fn from_borrowed(features: ArrayView2<'a, T>, target: &'a Array1<T>) -> Self {
            Self {
                features: ZeroCopyArray::from_borrowed(features),
                target: Cow::Borrowed(target),
            }
        }

        /// Check if both features and target are zero-copy
        pub fn is_fully_zero_copy(&self) -> bool {
            self.features.is_zero_copy() && matches!(self.target, Cow::Borrowed(_))
        }

        /// Get number of samples
        pub fn n_samples(&self) -> usize {
            self.features.shape().0
        }

        /// Get number of features
        pub fn n_features(&self) -> usize {
            self.features.shape().1
        }

        /// Validate that features and target have consistent sample counts
        pub fn validate(&self) -> Result<()> {
            let n_samples_features = self.n_samples();
            let n_samples_target = self.target.len();

            if n_samples_features != n_samples_target {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("features.shape[0] == target.len() ({n_samples_features})"),
                    actual: format!(
                        "features.shape[0]={n_samples_features}, target.len()={n_samples_target}"
                    ),
                });
            }
            Ok(())
        }
    }

    /// Trait for types that support zero-copy operations
    pub trait ZeroCopy<'a> {
        type Output;

        /// Convert to zero-copy representation
        fn to_zero_copy(&'a self) -> Self::Output;
    }

    /// Zero-copy array views utilities
    pub mod array_views {
        use super::*;

        /// Create array view with lifetime management
        pub fn create_view<'a, T>(array: &'a Array2<T>) -> ArrayView2<'a, T> {
            array.view()
        }

        /// Split array into multiple views along axis 0 (rows)
        pub fn split_rows<'a, T>(
            array: &'a Array2<T>,
            indices: &[usize],
        ) -> Result<Vec<ArrayView2<'a, T>>> {
            let (n_rows, _) = array.dim();
            let mut views = Vec::new();
            let mut start = 0;

            for &end in indices {
                if end > n_rows {
                    return Err(SklearsError::InvalidInput(format!(
                        "Split index {end} exceeds array rows {n_rows}"
                    )));
                }
                if end <= start {
                    return Err(SklearsError::InvalidInput(
                        "Split indices must be in ascending order".to_string(),
                    ));
                }

                let view = array.slice(scirs2_core::ndarray::s![start..end, ..]);
                views.push(view);
                start = end;
            }

            // Add remaining rows if any
            if start < n_rows {
                let view = array.slice(scirs2_core::ndarray::s![start.., ..]);
                views.push(view);
            }

            Ok(views)
        }

        /// Create sliding window views
        pub fn sliding_windows<'a, T>(
            array: &'a Array2<T>,
            window_size: usize,
            stride: usize,
        ) -> Result<Vec<ArrayView2<'a, T>>> {
            let (n_rows, _) = array.dim();

            if window_size == 0 || stride == 0 {
                return Err(SklearsError::InvalidInput(
                    "Window size and stride must be positive".to_string(),
                ));
            }

            if window_size > n_rows {
                return Err(SklearsError::InvalidInput(format!(
                    "Window size {window_size} exceeds array rows {n_rows}"
                )));
            }

            let mut views = Vec::new();
            let mut start = 0;

            while start + window_size <= n_rows {
                let end = start + window_size;
                let view = array.slice(scirs2_core::ndarray::s![start..end, ..]);
                views.push(view);
                start += stride;
            }

            Ok(views)
        }
    }

    /// Dataset operations with zero-copy semantics
    pub mod dataset_ops {
        use super::*;

        /// Split dataset into train/test (returns owned datasets for simplicity)
        pub fn train_test_split<T>(
            dataset: &ZeroCopyDataset<T>,
            train_ratio: f64,
        ) -> Result<(ZeroCopyDataset<'static, T>, ZeroCopyDataset<'static, T>)>
        where
            T: Clone,
        {
            if train_ratio <= 0.0 || train_ratio >= 1.0 {
                return Err(SklearsError::InvalidParameter {
                    name: "train_ratio".to_string(),
                    reason: "must be between 0 and 1".to_string(),
                });
            }

            let n_samples = dataset.n_samples();
            let train_size = (n_samples as f64 * train_ratio) as usize;

            if train_size == 0 || train_size == n_samples {
                return Err(SklearsError::InvalidInput(
                    "Train/test split results in empty set".to_string(),
                ));
            }

            // Convert to owned arrays to avoid lifetime issues
            let full_features = dataset.features.view().to_owned();
            let full_target = dataset.target.clone().into_owned();

            // Split the data
            let train_features = full_features
                .slice(scirs2_core::ndarray::s![..train_size, ..])
                .to_owned();
            let test_features = full_features
                .slice(scirs2_core::ndarray::s![train_size.., ..])
                .to_owned();

            let train_target = full_target
                .slice(scirs2_core::ndarray::s![..train_size])
                .to_owned();
            let test_target = full_target
                .slice(scirs2_core::ndarray::s![train_size..])
                .to_owned();

            let train_dataset = ZeroCopyDataset::from_owned(train_features, train_target);
            let test_dataset = ZeroCopyDataset::from_owned(test_features, test_target);

            Ok((train_dataset, test_dataset))
        }

        /// Create k-fold cross-validation splits (returns owned datasets for simplicity)
        pub fn k_fold_splits<T>(
            dataset: &ZeroCopyDataset<T>,
            k: usize,
        ) -> Result<Vec<(ZeroCopyDataset<'static, T>, ZeroCopyDataset<'static, T>)>>
        where
            T: Clone,
        {
            if k < 2 {
                return Err(SklearsError::InvalidParameter {
                    name: "k".to_string(),
                    reason: "must be at least 2".to_string(),
                });
            }

            let n_samples = dataset.n_samples();
            if k > n_samples {
                return Err(SklearsError::InvalidParameter {
                    name: "k".to_string(),
                    reason: format!("cannot exceed number of samples ({n_samples})"),
                });
            }

            // Convert to owned for easier manipulation
            let full_features = dataset.features.view().to_owned();
            let full_target = dataset.target.clone().into_owned();

            let fold_size = n_samples / k;
            let mut splits = Vec::new();

            for fold in 0..k {
                let test_start = fold * fold_size;
                let test_end = if fold == k - 1 {
                    n_samples // Last fold gets remaining samples
                } else {
                    test_start + fold_size
                };

                // Create test split
                let test_features = full_features
                    .slice(scirs2_core::ndarray::s![test_start..test_end, ..])
                    .to_owned();
                let test_target = full_target
                    .slice(scirs2_core::ndarray::s![test_start..test_end])
                    .to_owned();

                // Create train split (concatenate before and after test)
                let mut train_features_data = Vec::new();
                let mut train_target_data = Vec::new();

                // Add samples before test set
                for i in 0..test_start {
                    let row = full_features.row(i);
                    train_features_data.extend(row.iter().cloned());
                    train_target_data.push(full_target[i].clone());
                }

                // Add samples after test set
                for i in test_end..n_samples {
                    let row = full_features.row(i);
                    train_features_data.extend(row.iter().cloned());
                    train_target_data.push(full_target[i].clone());
                }

                let n_train = train_target_data.len();
                let n_features = dataset.n_features();

                let train_features =
                    Array2::from_shape_vec((n_train, n_features), train_features_data).map_err(
                        |e| SklearsError::Other(format!("Failed to create train features: {e}")),
                    )?;
                let train_target = Array1::from_vec(train_target_data);

                let train_dataset = ZeroCopyDataset::from_owned(train_features, train_target);
                let test_dataset = ZeroCopyDataset::from_owned(test_features, test_target);

                splits.push((train_dataset, test_dataset));
            }

            Ok(splits)
        }
    }

    /// Memory pool for efficient array allocation
    pub struct ArrayPool<T> {
        pool: std::collections::VecDeque<Array2<T>>,
        max_size: usize,
    }

    impl<T> ArrayPool<T> {
        /// Create a new array pool
        pub fn new(max_size: usize) -> Self {
            Self {
                pool: std::collections::VecDeque::new(),
                max_size,
            }
        }

        /// Get an array from the pool or create a new one
        pub fn get(&mut self, shape: (usize, usize)) -> Array2<T>
        where
            T: Clone + Zero,
        {
            // Try to reuse an array with compatible size
            if let Some(mut array) = self.pool.pop_front() {
                if array.dim() == shape {
                    array.fill(T::zero());
                    return array;
                }
                // Put it back if size doesn't match
                self.pool.push_back(array);
            }

            // Create new array
            Array2::zeros(shape)
        }

        /// Return an array to the pool
        pub fn return_array(&mut self, array: Array2<T>) {
            if self.pool.len() < self.max_size {
                self.pool.push_back(array);
            }
            // Otherwise drop the array
        }

        /// Clear the pool
        pub fn clear(&mut self) {
            self.pool.clear();
        }

        /// Get pool size
        pub fn size(&self) -> usize {
            self.pool.len()
        }
    }
}

// Type aliases for commonly used zero-copy types
pub use zero_copy::{
    array_views, dataset_ops, ArrayPool, ZeroCopyArray, ZeroCopyDataset as CowDataset,
};

// Re-export fixed-size types
pub use const_generic::{
    FixedFeatures, FixedFeaturesMatrix, FixedPredictions, FixedSamples, FixedTarget, MatrixShape,
};

// Type aliases for zero-copy operations
pub type ZeroCopyFeatures<'a, T = Float> = ZeroCopyArray<'a, T>;
pub type ZeroCopyTarget<'a, T = Float> = std::borrow::Cow<'a, Array1<T>>;

/// Trait for zero-copy conversion
pub trait ZeroCopy<'a, T> {
    type Output;

    /// Convert to zero-copy representation
    fn to_zero_copy(&'a self) -> Self::Output;
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_module_reexports() {
        // Test that we can use re-exported types
        let _array: Array2<Float> = Array2::zeros((3, 4));
        let _prob = Probability::new(0.5).expect("expected valid value");
        let _lr = LearningRate::new(0.01).expect("expected valid value");
        let _fc = FeatureCount::new(10).expect("expected valid value");
    }

    #[test]
    fn test_fixed_features() {
        let features = const_generic::FixedFeatures::<f64, 3>::new([1.0, 2.0, 3.0]);
        assert_eq!(FixedFeatures::<f64, 3>::feature_count(), 3);
        assert_eq!(features[0], 1.0);

        let norm = features.norm();
        assert_abs_diff_eq!(norm, 14.0_f64.sqrt());

        let normalized = features.normalize();
        assert_abs_diff_eq!(normalized.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fixed_samples() {
        let data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let samples = const_generic::FixedSamples::<f64, 3, 2>::new(data);

        assert_eq!(FixedSamples::<f64, 3, 2>::sample_count(), 3);
        assert_eq!(FixedSamples::<f64, 3, 2>::feature_count(), 2);
        assert_eq!(FixedSamples::<f64, 3, 2>::shape(), (3, 2));

        let array2 = samples.to_array2();
        assert_eq!(array2.dim(), (3, 2));
        assert_eq!(array2[[0, 0]], 1.0);
        assert_eq!(array2[[2, 1]], 6.0);
    }

    #[test]
    fn test_zero_copy_dataset() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid array shape");
        let target = Array1::from_vec(vec![0.0, 1.0, 2.0]);

        let dataset = zero_copy::ZeroCopyDataset::from_owned(features, target);
        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
        assert!(!dataset.is_fully_zero_copy());

        dataset.validate().expect("validate should succeed");
    }

    #[test]
    fn test_array_pool() {
        let mut pool = zero_copy::ArrayPool::<f64>::new(2);
        assert_eq!(pool.size(), 0);

        let array1 = pool.get((2, 3));
        assert_eq!(array1.dim(), (2, 3));

        pool.return_array(array1);
        assert_eq!(pool.size(), 1);

        let array2 = pool.get((2, 3)); // Should reuse
        assert_eq!(array2.dim(), (2, 3));
    }
}
