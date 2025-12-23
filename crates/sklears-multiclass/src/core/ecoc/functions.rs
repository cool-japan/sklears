//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Predict, PredictProba, Trained},
    types::Float,
};

pub type TrainedECOC<T> = ECOCClassifier<ECOCTrainedData<T>, Trained>;
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
        let chunk_size = 8;
        let mut distance = 0usize;
        let mut i = 0;
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
        let _result = Array1::<i32>::zeros(n_samples);
        if self.config.gpu_mode == GPUMode::Disabled {
            return self.aggregate_votes_cpu(predictions, code_matrix);
        }
        let results: Vec<i32> = (0..n_samples)
            .into_par_iter()
            .map(|sample_idx| {
                let sample_predictions = predictions.row(sample_idx);
                let binary_code: Vec<i32> = sample_predictions
                    .iter()
                    .map(|&p| if p > 0.5 { 1 } else { -1 })
                    .collect();
                let class_codes: Vec<Vec<i32>> =
                    (0..n_classes).map(|i| code_matrix.get_row(i)).collect();
                let distances = self.batch_hamming_distances(&class_codes, &binary_code);
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
        let mut all_binary_predictions = Array2::zeros((n_samples, code_length));
        for (bit_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
            let predictions = estimator.predict(x)?;
            for sample_idx in 0..n_samples {
                all_binary_predictions[[sample_idx, bit_idx]] = predictions[sample_idx];
            }
        }
        let class_indices =
            self.aggregate_votes_gpu(&all_binary_predictions, &self.base_estimator.code_matrix)?;
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
            let mut binary_probabilities = Array1::zeros(code_length);
            for (bit_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
                let prediction = estimator.predict(&sample_matrix.to_owned())?;
                binary_probabilities[bit_idx] = prediction[0];
            }
            for class_idx in 0..n_classes {
                let code = self.base_estimator.code_matrix.get_row(class_idx);
                let mut class_prob = 1.0;
                for (bit_idx, &code_bit) in code.iter().enumerate() {
                    let binary_prob = binary_probabilities[bit_idx];
                    let bit_match_prob = if code_bit == 1 {
                        binary_prob
                    } else {
                        1.0 - binary_prob
                    };
                    class_prob *= bit_match_prob;
                }
                probabilities[[sample_idx, class_idx]] = class_prob;
            }
            let prob_sum: f64 = probabilities.row(sample_idx).sum();
            if prob_sum > 0.0 {
                for class_idx in 0..n_classes {
                    probabilities[[sample_idx, class_idx]] /= prob_sum;
                }
            } else {
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
            let mut binary_predictions = Array1::zeros(code_length);
            for (bit_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
                let prediction = estimator.predict(&sample_matrix.to_owned())?;
                binary_predictions[bit_idx] = prediction[0];
            }
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
    use sklears_core::traits::{Estimator, Fit};
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
            Ok(Array1::from_elem(x.nrows(), 0.6))
        }
    }
    #[test]
    fn test_ecoc_config_default() {
        let config = ECOCConfig::default();
        assert_eq!(config.strategy, ECOCStrategy::StdRng);
        assert_eq!(config.code_size, 1.5);
        assert_eq!(config.random_state, None);
        assert_eq!(config.n_jobs, None);
        assert_eq!(config.use_sparse, false);
        assert_eq!(config.sparse_threshold, 0.3);
    }
    #[test]
    fn test_ecoc_strategy_default() {
        let strategy = ECOCStrategy::default();
        assert_eq!(strategy, ECOCStrategy::StdRng);
    }
    #[test]
    fn test_ecoc_builder() {
        let mock_classifier = MockClassifier::new();
        let ecoc = ECOCClassifier::builder(mock_classifier)
            .strategy(ECOCStrategy::DenseStdRng)
            .code_size(2.0)
            .random_state(42)
            .parallel()
            .build();
        assert_eq!(ecoc.config.strategy, ECOCStrategy::DenseStdRng);
        assert_eq!(ecoc.config.code_size, 2.0);
        assert_eq!(ecoc.config.random_state, Some(42));
        assert_eq!(ecoc.config.n_jobs, Some(-1));
    }
    #[test]
    fn test_ecoc_classifier_creation() {
        let mock_classifier = MockClassifier::new();
        let ecoc = ECOCClassifier::new(mock_classifier)
            .strategy(ECOCStrategy::StdRng)
            .code_size(1.5)
            .random_state(42);
        assert_eq!(ecoc.config.strategy, ECOCStrategy::StdRng);
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
        assert_eq!(sparse.get(0, 0), 0);
        assert_eq!(sparse.get(2, 2), 0);
        sparse.set(0, 1, 1);
        sparse.set(1, 2, -1);
        assert_eq!(sparse.get(0, 1), 1);
        assert_eq!(sparse.get(1, 2), -1);
        assert_eq!(sparse.get(0, 0), 0);
        sparse.set(0, 1, 2);
        assert_eq!(sparse.get(0, 1), 2);
        sparse.set(0, 1, 0);
        assert_eq!(sparse.get(0, 1), 0);
        assert_eq!(sparse.entries.len(), 1);
    }
    #[test]
    fn test_sparse_matrix_from_dense() {
        let dense = Array2::from_shape_vec((2, 3), vec![1, 0, -1, 0, 1, 0]).unwrap();
        let sparse = SparseMatrix::from_dense(&dense, 0);
        assert_eq!(sparse.n_rows, 2);
        assert_eq!(sparse.n_cols, 3);
        assert_eq!(sparse.default_value, 0);
        assert_eq!(sparse.entries.len(), 3);
        assert_eq!(sparse.get(0, 0), 1);
        assert_eq!(sparse.get(0, 2), -1);
        assert_eq!(sparse.get(1, 1), 1);
        assert_eq!(sparse.get(0, 1), 0);
        assert_eq!(sparse.get(1, 0), 0);
        assert_eq!(sparse.get(1, 2), 0);
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
        assert_eq!(sparse.sparsity(), 1.0);
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
        assert_eq!(code_matrix.get(0, 1), -1);
        assert_eq!(code_matrix.get(0, 2), 1);
        assert_eq!(code_matrix.get(1, 0), -1);
        assert_eq!(code_matrix.get(1, 1), 1);
        assert_eq!(code_matrix.get(1, 2), -1);
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
            .strategy(ECOCStrategy::StdRng)
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
        let ecoc = ECOCClassifier::new(mock_classifier).sparse_threshold(-0.1);
        assert_eq!(ecoc.config.sparse_threshold, 0.0);
        let mock_classifier2 = MockClassifier::new();
        let ecoc2 = ECOCClassifier::new(mock_classifier2).sparse_threshold(1.5);
        assert_eq!(ecoc2.config.sparse_threshold, 1.0);
    }
    #[test]
    fn test_compression_ratio_calculation() {
        let mut sparse = SparseMatrix::new(10, 10, 0);
        sparse.set(0, 0, 1);
        sparse.set(5, 5, 1);
        sparse.set(9, 9, 1);
        let compression_ratio = sparse.compression_ratio();
        assert!(compression_ratio < 1.0);
        let dense = Array2::ones((10, 10));
        let dense_sparse = SparseMatrix::from_dense(&dense, 0);
        let dense_compression = dense_sparse.compression_ratio();
        assert!(dense_compression > compression_ratio);
    }
    #[test]
    fn test_memory_usage_calculation() {
        let sparse = SparseMatrix::new(100, 100, 0);
        let memory_usage = sparse.memory_usage();
        assert!(memory_usage > 0);
        let mut sparse_with_data = SparseMatrix::new(100, 100, 0);
        for i in 0..50 {
            sparse_with_data.set(i, i, 1);
        }
        let memory_usage_with_data = sparse_with_data.memory_usage();
        assert!(memory_usage_with_data > memory_usage);
    }
}
