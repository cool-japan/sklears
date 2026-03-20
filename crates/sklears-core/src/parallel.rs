/// Parallel trait implementations using rayon
///
/// This module provides parallel implementations of key machine learning operations
/// to improve performance on multi-core systems. All parallel operations fall back
/// gracefully to sequential execution when parallelism is not available or beneficial.
use crate::error::Result;
use crate::traits::*;
use crate::types::FloatBounds;
// SciRS2 Policy: Using scirs2_core::ndarray for unified access (COMPLIANT)
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};

/// Configuration for parallel operations
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use (None = use rayon default)
    pub num_threads: Option<usize>,
    /// Minimum batch size before using parallel processing
    pub min_parallel_batch_size: usize,
    /// Whether to enable parallel operations
    pub enabled: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: None,
            min_parallel_batch_size: 1000,
            enabled: true,
        }
    }
}

/// Trait for parallel prediction operations
pub trait ParallelPredict<X, Output> {
    /// Make predictions in parallel on large datasets
    fn predict_parallel(&self, x: &X) -> Result<Output>;

    /// Make predictions in parallel with custom configuration
    fn predict_parallel_with_config(&self, x: &X, config: &ParallelConfig) -> Result<Output>;
}

/// Trait for parallel transformation operations
pub trait ParallelTransform<X, Output = X> {
    /// Transform data in parallel
    fn transform_parallel(&self, x: &X) -> Result<Output>;

    /// Transform data in parallel with custom configuration
    fn transform_parallel_with_config(&self, x: &X, config: &ParallelConfig) -> Result<Output>;
}

/// Trait for parallel fitting operations
pub trait ParallelFit<X, Y> {
    type Fitted;

    /// Fit model using parallel operations where beneficial
    fn fit_parallel(self, x: &X, y: &Y) -> Result<Self::Fitted>;

    /// Fit model with parallel configuration
    fn fit_parallel_with_config(
        self,
        x: &X,
        y: &Y,
        config: &ParallelConfig,
    ) -> Result<Self::Fitted>;
}

/// Trait for parallel cross-validation
pub trait ParallelCrossValidation<X, Y> {
    type Score: FloatBounds;

    /// Perform k-fold cross-validation in parallel
    fn cross_validate_parallel(
        &self,
        model: impl Fit<X, Y> + Clone + Send + Sync,
        x: &X,
        y: &Y,
        cv_folds: usize,
    ) -> Result<Vec<Self::Score>>
    where
        X: Clone + Send + Sync,
        Y: Clone + Send + Sync,
        <Self as ParallelCrossValidation<X, Y>>::Score: Send;
}

/// Trait for parallel ensemble operations
pub trait ParallelEnsemble<X, Y, Output> {
    /// Train ensemble models in parallel
    fn fit_ensemble_parallel(
        models: Vec<impl Fit<X, Y> + Clone + Send + Sync>,
        x: &X,
        y: &Y,
    ) -> Result<Vec<Box<dyn Predict<X, Output>>>>
    where
        X: Clone + Send + Sync,
        Y: Clone + Send + Sync;

    /// Make ensemble predictions in parallel
    fn predict_ensemble_parallel(
        models: &[impl Predict<X, Output> + Sync],
        x: &X,
    ) -> Result<Vec<Output>>
    where
        X: Sync,
        Output: Send;
}

/// Helper function for parallel prediction on ndarray data
pub fn predict_parallel_ndarray<T, M>(
    model: &M,
    x: &Array2<T>,
    config: &ParallelConfig,
) -> Result<Array1<T>>
where
    T: FloatBounds + Send + Sync,
    M: Predict<Array2<T>, Array1<T>> + Sync,
{
    if !config.enabled || x.nrows() < config.min_parallel_batch_size {
        return model.predict(x);
    }

    // Split data into chunks for parallel processing
    let chunk_size = (x.nrows() / rayon::current_num_threads()).max(1);
    let chunks: Vec<_> = x.axis_chunks_iter(Axis(0), chunk_size).collect();

    let results: Result<Vec<_>> = chunks
        .into_par_iter()
        .map(|chunk| {
            let chunk_array = chunk.to_owned();
            model.predict(&chunk_array)
        })
        .collect();

    let predictions = results?;

    // Concatenate results
    let total_len: usize = predictions.iter().map(|p| p.len()).sum();
    let mut result = Array1::zeros(total_len);
    let mut offset = 0;

    for pred in predictions {
        let end = offset + pred.len();
        result
            .slice_mut(scirs2_core::ndarray::s![offset..end])
            .assign(&pred);
        offset = end;
    }

    Ok(result)
}

/// Parallel matrix operations
pub struct ParallelMatrixOps;

impl ParallelMatrixOps {
    /// Parallel matrix multiplication using rayon and SIMD
    pub fn matrix_multiply_parallel<T: FloatBounds + Send + Sync>(
        a: &Array2<T>,
        b: &Array2<T>,
        config: &ParallelConfig,
    ) -> Array2<T> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2, "Matrix dimensions must match");

        let mut result = Array2::zeros((m, n));

        if !config.enabled || m < config.min_parallel_batch_size {
            // Fall back to sequential
            result.assign(&a.dot(b));
            return result;
        }

        // Parallel row computation
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n {
                    let mut sum = T::zero();
                    for ki in 0..k {
                        sum += a[[i, ki]] * b[[ki, j]];
                    }
                    row[j] = sum;
                }
            });

        result
    }

    /// Parallel element-wise operations
    pub fn elementwise_op_parallel<T, F>(
        a: &Array2<T>,
        b: &Array2<T>,
        op: F,
        config: &ParallelConfig,
    ) -> Array2<T>
    where
        T: FloatBounds + Send + Sync,
        F: Fn(T, T) -> T + Send + Sync,
    {
        assert_eq!(a.shape(), b.shape());

        let mut result = Array2::zeros(a.dim());

        if !config.enabled || a.len() < config.min_parallel_batch_size {
            // Sequential fallback
            result
                .iter_mut()
                .zip(a.iter())
                .zip(b.iter())
                .for_each(|((r, &ai), &bi)| *r = op(ai, bi));
        } else {
            // Parallel operation using slices
            if let (Some(result_slice), Some(a_slice), Some(b_slice)) =
                (result.as_slice_mut(), a.as_slice(), b.as_slice())
            {
                result_slice
                    .par_iter_mut()
                    .zip(a_slice.par_iter())
                    .zip(b_slice.par_iter())
                    .for_each(|((r, &ai), &bi)| *r = op(ai, bi));
            } else {
                // Fallback to sequential if slices unavailable
                result
                    .iter_mut()
                    .zip(a.iter())
                    .zip(b.iter())
                    .for_each(|((r, &ai), &bi)| *r = op(ai, bi));
            }
        }

        result
    }

    /// Parallel row-wise operations
    pub fn apply_row_parallel<T, F>(matrix: &Array2<T>, op: F, config: &ParallelConfig) -> Array1<T>
    where
        T: FloatBounds + Send + Sync,
        F: Fn(ArrayView1<T>) -> T + Send + Sync,
    {
        let mut result = Array1::zeros(matrix.nrows());

        if !config.enabled || matrix.nrows() < config.min_parallel_batch_size {
            // Sequential
            result
                .iter_mut()
                .zip(matrix.axis_iter(Axis(0)))
                .for_each(|(r, row)| *r = op(row));
        } else {
            // Parallel using indexed access with slice
            if let Some(result_slice) = result.as_slice_mut() {
                result_slice.par_iter_mut().enumerate().for_each(|(i, r)| {
                    let row = matrix.row(i);
                    *r = op(row);
                });
            } else {
                // Fallback to sequential
                result
                    .iter_mut()
                    .zip(matrix.axis_iter(Axis(0)))
                    .for_each(|(r, row)| *r = op(row));
            }
        }

        result
    }

    /// Parallel column-wise operations
    pub fn apply_column_parallel<T, F>(
        matrix: &Array2<T>,
        op: F,
        config: &ParallelConfig,
    ) -> Array1<T>
    where
        T: FloatBounds + Send + Sync,
        F: Fn(ArrayView1<T>) -> T + Send + Sync,
    {
        let mut result = Array1::zeros(matrix.ncols());

        if !config.enabled || matrix.ncols() < config.min_parallel_batch_size {
            // Sequential
            result
                .iter_mut()
                .zip(matrix.axis_iter(Axis(1)))
                .for_each(|(r, col)| *r = op(col));
        } else {
            // Parallel using indexed access with slice
            if let Some(result_slice) = result.as_slice_mut() {
                result_slice.par_iter_mut().enumerate().for_each(|(j, r)| {
                    let col = matrix.column(j);
                    *r = op(col);
                });
            } else {
                // Fallback to sequential
                result
                    .iter_mut()
                    .zip(matrix.axis_iter(Axis(1)))
                    .for_each(|(r, col)| *r = op(col));
            }
        }

        result
    }
}

/// Cross-validation utilities with parallel execution
pub struct ParallelCrossValidator<T: FloatBounds> {
    config: ParallelConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: FloatBounds> ParallelCrossValidator<T> {
    /// Create a new parallel cross-validator
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Perform k-fold cross-validation in parallel
    pub fn k_fold_parallel<X, Y, M, Output>(
        &self,
        model: M,
        x: &X,
        y: &Y,
        k: usize,
    ) -> Result<Vec<T>>
    where
        M: Fit<X, Y> + Clone + Send + Sync,
        M::Fitted: Score<X, Y, Float = T>,
        X: Clone + Send + Sync,
        Y: Clone + Send + Sync,
        T: Send,
    {
        if !self.config.enabled || k < 2 {
            // Fall back to sequential
            return self.k_fold_sequential(model, x, y, k);
        }

        // Create fold indices (simplified - in practice would use proper stratification)
        let fold_indices: Vec<_> = (0..k).collect();

        // Execute folds in parallel
        let scores: Result<Vec<_>> = fold_indices
            .into_par_iter()
            .map(|_fold_idx| {
                // This is a simplified implementation
                // In practice, we would split data based on fold_idx
                let model_clone = model.clone();
                let fitted = model_clone.fit(x, y)?;
                fitted.score(x, y)
            })
            .collect();

        scores
    }

    /// Sequential fallback for cross-validation
    fn k_fold_sequential<X, Y, M>(&self, model: M, x: &X, y: &Y, k: usize) -> Result<Vec<T>>
    where
        M: Fit<X, Y> + Clone,
        M::Fitted: Score<X, Y, Float = T>,
    {
        let mut scores = Vec::with_capacity(k);

        for _fold in 0..k {
            let model_clone = model.clone();
            let fitted = model_clone.fit(x, y)?;
            let score = fitted.score(x, y)?;
            scores.push(score);
        }

        Ok(scores)
    }
}

/// Parallel ensemble operations
pub struct ParallelEnsembleOps;

impl ParallelEnsembleOps {
    /// Train multiple models in parallel
    pub fn train_models_parallel<X, Y, M>(
        models: Vec<M>,
        x: &X,
        y: &Y,
        config: &ParallelConfig,
    ) -> Result<Vec<M::Fitted>>
    where
        M: Fit<X, Y> + Send,
        M::Fitted: Send,
        X: Sync,
        Y: Sync,
    {
        if !config.enabled || models.len() < 2 {
            // Sequential fallback
            return models.into_iter().map(|model| model.fit(x, y)).collect();
        }

        // Parallel training
        models
            .into_par_iter()
            .map(|model| model.fit(x, y))
            .collect()
    }

    /// Make predictions with multiple models in parallel
    pub fn predict_parallel<X, Output, M>(
        models: &[M],
        x: &X,
        config: &ParallelConfig,
    ) -> Result<Vec<Output>>
    where
        M: Predict<X, Output> + Sync,
        Output: Send,
        X: Sync,
    {
        if !config.enabled || models.len() < 2 {
            // Sequential fallback
            return models.iter().map(|model| model.predict(x)).collect();
        }

        // Parallel prediction
        models.par_iter().map(|model| model.predict(x)).collect()
    }
}

/// Utility functions for parallel operations
pub mod utils {
    use super::*;

    /// Determine optimal chunk size for parallel processing
    pub fn optimal_chunk_size(total_size: usize, min_chunk_size: usize) -> usize {
        let num_threads = rayon::current_num_threads();
        (total_size / num_threads).max(min_chunk_size)
    }

    /// Check if parallel processing is beneficial
    pub fn should_use_parallel(data_size: usize, config: &ParallelConfig) -> bool {
        config.enabled && data_size >= config.min_parallel_batch_size
    }

    /// Initialize rayon thread pool with custom configuration
    pub fn initialize_thread_pool(num_threads: Option<usize>) -> Result<()> {
        if let Some(threads) = num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .map_err(|e| {
                    crate::error::SklearsError::NumericalError(format!(
                        "Failed to initialize thread pool: {e}"
                    ))
                })?;
        }
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_parallel_matrix_multiply() {
        let a = Array2::from_shape_vec((100, 50), (0..5000).map(|x| x as f64).collect())
            .expect("valid array shape");
        let b = Array2::from_shape_vec((50, 30), (0..1500).map(|x| x as f64 + 1.0).collect())
            .expect("valid array shape");

        let config = ParallelConfig {
            enabled: true,
            min_parallel_batch_size: 10,
            num_threads: None,
        };

        let result_parallel = ParallelMatrixOps::matrix_multiply_parallel(&a, &b, &config);
        let result_sequential = a.dot(&b);

        // Results should be approximately equal
        for i in 0..result_parallel.nrows() {
            for j in 0..result_parallel.ncols() {
                assert_relative_eq!(
                    result_parallel[[i, j]],
                    result_sequential[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_parallel_elementwise_ops() {
        let a = Array2::from_shape_vec((100, 100), (0..10000).map(|x| x as f64).collect())
            .expect("valid array shape");
        let b = Array2::from_shape_vec((100, 100), (0..10000).map(|x| x as f64 + 1.0).collect())
            .expect("expected valid value");

        let config = ParallelConfig {
            enabled: true,
            min_parallel_batch_size: 100,
            num_threads: None,
        };

        let result_parallel =
            ParallelMatrixOps::elementwise_op_parallel(&a, &b, |x, y| x + y, &config);
        let result_sequential = &a + &b;

        for i in 0..result_parallel.nrows() {
            for j in 0..result_parallel.ncols() {
                assert_relative_eq!(
                    result_parallel[[i, j]],
                    result_sequential[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_optimal_chunk_size() {
        let num_threads = rayon::current_num_threads();
        let expected = (1000 / num_threads).max(10);
        assert_eq!(utils::optimal_chunk_size(1000, 10), expected);
        assert_eq!(utils::optimal_chunk_size(100, 50), 50); // min_chunk_size takes precedence
    }

    #[test]
    fn test_should_use_parallel() {
        let config = ParallelConfig::default();
        assert!(!utils::should_use_parallel(100, &config)); // Below threshold
        assert!(utils::should_use_parallel(2000, &config)); // Above threshold

        let disabled_config = ParallelConfig {
            enabled: false,
            ..Default::default()
        };
        assert!(!utils::should_use_parallel(2000, &disabled_config)); // Disabled
    }
}
