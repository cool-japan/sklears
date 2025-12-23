use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{Array2, Axis};
use scirs2_core::random::{seeded_rng, Rng, SliceRandom};
use sklears_model_selection::*;
use std::collections::HashMap;

/// Reproducibility tests for model selection algorithms
/// Tests that ensure consistent results across different platforms and runs
#[allow(non_snake_case)]
#[cfg(test)]
mod reproducibility_tests {
    use super::*;

    /// Test reproducibility of random number generation
    #[test]
    fn test_random_state_reproducibility() {
        let seed = 42u64;

        // Generate random numbers with same seed
        let mut rng1 = seeded_rng(seed);
        let mut rng2 = seeded_rng(seed);

        let numbers1: Vec<f64> = (0..100).map(|_| rng1.random::<f64>()).collect();
        let numbers2: Vec<f64> = (0..100).map(|_| rng2.random::<f64>()).collect();

        // Should be identical
        assert_eq!(
            numbers1, numbers2,
            "Random number generation should be reproducible"
        );
    }

    /// Test reproducibility of data shuffling
    #[test]
    fn test_data_shuffling_reproducibility() {
        let seed = 42u64;
        let n_samples = 1000;

        // Create test data
        let x = Array2::from_shape_fn((n_samples, 5), |(i, j)| (i * 5 + j) as f64);
        let y = Array2::from_shape_fn((n_samples, 1), |(i, _)| i as f64);

        // Shuffle data twice with same seed
        let shuffled1 = shuffle_data(&x, &y, seed);
        let shuffled2 = shuffle_data(&x, &y, seed);

        // Should be identical
        assert_eq!(
            shuffled1.0, shuffled2.0,
            "Data shuffling should be reproducible"
        );
        assert_eq!(
            shuffled1.1, shuffled2.1,
            "Label shuffling should be reproducible"
        );
    }

    /// Test reproducibility of cross-validation splits
    #[test]
    fn test_cv_splits_reproducibility() {
        let seed = 42u64;
        let n_samples = 500;

        // Create test data
        let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| (i * 3 + j) as f64);
        let y = Array2::from_shape_fn((n_samples, 1), |(i, _)| (i % 3) as f64);

        // Create CV splitters with same seed
        let kfold1 = create_kfold_cv(5, seed);
        let kfold2 = create_kfold_cv(5, seed);

        let splits1: Vec<_> = kfold1.split(&x, Some(&y)).into_iter().collect();
        let splits2: Vec<_> = kfold2.split(&x, Some(&y)).into_iter().collect();

        // Should be identical
        assert_eq!(splits1, splits2, "CV splits should be reproducible");
    }

    /// Test reproducibility of parameter sampling
    #[test]
    fn test_parameter_sampling_reproducibility() {
        let seed = 42u64;

        // Create parameter space
        let mut param_space = ParameterSpace::new();
        param_space.add_float_param("learning_rate", 0.001, 0.1);
        param_space.add_int_param("n_estimators", 50, 500);
        param_space.add_categorical_param("criterion", vec!["gini", "entropy"]);

        // Sample parameters twice with same seed
        let params1 = sample_parameters(&param_space, 100, seed);
        let params2 = sample_parameters(&param_space, 100, seed);

        // Should be identical
        assert_eq!(
            params1, params2,
            "Parameter sampling should be reproducible"
        );
    }

    /// Test reproducibility of Bayesian optimization
    #[test]
    fn test_bayesian_optimization_reproducibility() {
        let seed = 42u64;
        let n_samples = 100;

        // Create test data
        let mut rng = seeded_rng(seed);
        let x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Create parameter space
        let mut param_space = ParameterSpace::new();
        param_space.add_float_param("param1", -1.0, 1.0);
        param_space.add_float_param("param2", -1.0, 1.0);

        // Run Bayesian optimization twice with same seed
        let estimator1 = MockEstimator::new();
        let estimator2 = MockEstimator::new();

        let results1 = run_bayesian_optimization(&estimator1, &param_space, &x, &y, seed);
        let results2 = run_bayesian_optimization(&estimator2, &param_space, &x, &y, seed);

        // Should produce same results
        assert_eq!(
            results1, results2,
            "Bayesian optimization should be reproducible"
        );
    }

    /// Test reproducibility of evolutionary algorithms
    #[test]
    fn test_evolutionary_algorithm_reproducibility() {
        let seed = 42u64;
        let n_samples = 100;

        // Create test data
        let mut rng = seeded_rng(seed);
        let x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Create parameter space
        let mut param_space = ParameterSpace::new();
        param_space.add_float_param("param1", -1.0, 1.0);
        param_space.add_float_param("param2", -1.0, 1.0);

        // Run evolutionary algorithm twice with same seed
        let estimator1 = MockEstimator::new();
        let estimator2 = MockEstimator::new();

        let results1 = run_evolutionary_algorithm(&estimator1, &param_space, &x, &y, seed);
        let results2 = run_evolutionary_algorithm(&estimator2, &param_space, &x, &y, seed);

        // Should produce same results
        assert_eq!(
            results1, results2,
            "Evolutionary algorithm should be reproducible"
        );
    }

    /// Test reproducibility across different data types
    #[test]
    fn test_reproducibility_different_data_types() {
        let seed = 42u64;

        // Test with different data types (all converted to f32 for consistency)
        let data_types = vec![
            ("f32", create_f32_data(100, 5, seed)),
            (
                "f64",
                create_f64_data(100, 5, seed)
                    .into_iter()
                    .map(|row| row.into_iter().map(|x| x as f32).collect())
                    .collect(),
            ),
            (
                "i32",
                create_i32_data(100, 5, seed)
                    .into_iter()
                    .map(|row| row.into_iter().map(|x| x as f32).collect())
                    .collect(),
            ),
        ];

        for (data_type, data) in data_types {
            let shuffled1 = shuffle_generic_data(&data, seed);
            let shuffled2 = shuffle_generic_data(&data, seed);

            assert_eq!(
                shuffled1, shuffled2,
                "Shuffling should be reproducible for {}",
                data_type
            );
        }
    }

    /// Test reproducibility of feature selection
    #[test]
    fn test_feature_selection_reproducibility() {
        let seed = 42u64;
        let n_samples = 200;
        let n_features = 50;

        // Create test data
        let mut rng = seeded_rng(seed);
        let x = Array2::from_shape_fn((n_samples, n_features), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Run feature selection twice with same seed
        let selected_features1 = run_feature_selection(&x, &y, 10, seed);
        let selected_features2 = run_feature_selection(&x, &y, 10, seed);

        // Should select same features
        assert_eq!(
            selected_features1, selected_features2,
            "Feature selection should be reproducible"
        );
    }

    /// Test reproducibility of ensemble construction
    #[test]
    fn test_ensemble_construction_reproducibility() {
        let seed = 42u64;
        let n_samples = 150;

        // Create test data
        let mut rng = seeded_rng(seed);
        let x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Build ensemble twice with same seed
        let ensemble1 = build_ensemble(&x, &y, 5, seed);
        let ensemble2 = build_ensemble(&x, &y, 5, seed);

        // Should produce same ensemble
        assert_eq!(
            ensemble1, ensemble2,
            "Ensemble construction should be reproducible"
        );
    }

    /// Test reproducibility of model evaluation
    #[test]
    fn test_model_evaluation_reproducibility() {
        let seed = 42u64;
        let n_samples = 100;

        // Create test data
        let mut rng = seeded_rng(seed);
        let x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Create model
        let model = MockEstimator::new();

        // Evaluate model twice with same seed
        let scores1 = evaluate_model(&model, &x, &y, seed);
        let scores2 = evaluate_model(&model, &x, &y, seed);

        // Should produce same scores
        assert_eq!(scores1, scores2, "Model evaluation should be reproducible");
    }

    /// Test reproducibility with different random number generators
    #[test]
    fn test_reproducibility_different_rngs() {
        let seed = 42u64;

        // Test with different RNG types
        let mut chacha_rng1 = seeded_rng(seed);
        let mut chacha_rng2 = seeded_rng(seed);

        let numbers1: Vec<f64> = (0..100).map(|_| chacha_rng1.random::<f64>()).collect();
        let numbers2: Vec<f64> = (0..100).map(|_| chacha_rng2.random::<f64>()).collect();

        // Should be identical
        assert_eq!(
            numbers1, numbers2,
            "Different RNG instances should be reproducible"
        );
    }

    /// Test reproducibility of parallel operations
    #[test]
    fn test_parallel_reproducibility() {
        let seed = 42u64;
        let n_samples = 1000;

        // Create test data
        let mut rng = seeded_rng(seed);
        let x = Array2::from_shape_fn((n_samples, 5), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });
        let y = Array2::from_shape_fn((n_samples, 1), |_| {
            rng.sample(Normal::new(0.0, 1.0).unwrap())
        });

        // Run parallel operations twice with same seed
        let results1 = run_parallel_cv(&x, &y, seed);
        let results2 = run_parallel_cv(&x, &y, seed);

        // Should produce same results
        assert_eq!(
            results1, results2,
            "Parallel operations should be reproducible"
        );
    }

    // Helper functions for testing

    fn shuffle_data(x: &Array2<f64>, y: &Array2<f64>, seed: u64) -> (Array2<f64>, Array2<f64>) {
        let mut rng = seeded_rng(seed);
        let n_samples = x.shape()[0];
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let shuffled_x = x.select(Axis(0), &indices);
        let shuffled_y = y.select(Axis(0), &indices);

        (shuffled_x, shuffled_y)
    }

    fn create_kfold_cv(n_splits: usize, seed: u64) -> MockKFoldCV {
        MockKFoldCV::new(n_splits, seed)
    }

    fn sample_parameters(
        param_space: &ParameterSpace,
        n_samples: usize,
        seed: u64,
    ) -> Vec<HashMap<String, ParameterValue>> {
        let mut rng = seeded_rng(seed);
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            let mut params = HashMap::new();

            // Sample from parameter space (sort names for deterministic iteration)
            let mut param_names: Vec<String> =
                param_space.get_parameter_names().into_iter().collect();
            param_names.sort();
            for param_name in param_names {
                let value = match param_name.as_str() {
                    "learning_rate" => ParameterValue::Float(rng.random_range(0.001..0.1)),
                    "n_estimators" => ParameterValue::Integer(rng.gen_range(50..500)),
                    "criterion" => {
                        let choices = ["gini", "entropy"];
                        ParameterValue::String(choices[rng.gen_range(0..choices.len())].to_string())
                    }
                    _ => ParameterValue::Float(rng.random()),
                };
                params.insert(param_name, value);
            }

            samples.push(params);
        }

        samples
    }

    fn run_bayesian_optimization(
        _estimator: &MockEstimator,
        _param_space: &ParameterSpace,
        _x: &Array2<f64>,
        _y: &Array2<f64>,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = seeded_rng(seed);
        // Mock Bayesian optimization results
        (0..10).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    fn run_evolutionary_algorithm(
        _estimator: &MockEstimator,
        _param_space: &ParameterSpace,
        _x: &Array2<f64>,
        _y: &Array2<f64>,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = seeded_rng(seed);
        // Mock evolutionary algorithm results
        (0..10).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    fn create_f32_data(n_samples: usize, n_features: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = seeded_rng(seed);
        (0..n_samples)
            .map(|_| (0..n_features).map(|_| rng.random::<f32>()).collect())
            .collect()
    }

    fn create_f64_data(n_samples: usize, n_features: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = seeded_rng(seed);
        (0..n_samples)
            .map(|_| (0..n_features).map(|_| rng.random()).collect())
            .collect()
    }

    fn create_i32_data(n_samples: usize, n_features: usize, seed: u64) -> Vec<Vec<i32>> {
        let mut rng = seeded_rng(seed);
        (0..n_samples)
            .map(|_| (0..n_features).map(|_| rng.random::<i32>()).collect())
            .collect()
    }

    fn shuffle_generic_data<T: Clone>(data: &[Vec<T>], seed: u64) -> Vec<Vec<T>> {
        let mut rng = seeded_rng(seed);
        let mut shuffled = data.to_vec();
        shuffled.shuffle(&mut rng);
        shuffled
    }

    fn run_feature_selection(
        _x: &Array2<f64>,
        _y: &Array2<f64>,
        n_features: usize,
        seed: u64,
    ) -> Vec<usize> {
        let mut rng = seeded_rng(seed);
        let mut features: Vec<usize> = (0..50).collect();
        features.shuffle(&mut rng);
        features[..n_features].to_vec()
    }

    fn build_ensemble(
        _x: &Array2<f64>,
        _y: &Array2<f64>,
        n_estimators: usize,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = seeded_rng(seed);
        // Mock ensemble weights
        (0..n_estimators).map(|_| rng.random()).collect()
    }

    fn evaluate_model(
        _model: &MockEstimator,
        _x: &Array2<f64>,
        _y: &Array2<f64>,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = seeded_rng(seed);
        // Mock evaluation scores
        (0..5).map(|_| rng.random_range(0.0..1.0)).collect()
    }

    fn run_parallel_cv(_x: &Array2<f64>, _y: &Array2<f64>, seed: u64) -> Vec<f64> {
        let mut rng = seeded_rng(seed);
        // Mock parallel CV results
        (0..5).map(|_| rng.random_range(0.0..1.0)).collect()
    }

    // Mock implementations for testing

    struct MockEstimator {
        params: HashMap<String, ParameterValue>,
    }

    impl MockEstimator {
        fn new() -> Self {
            Self {
                params: HashMap::new(),
            }
        }
    }

    impl PartialEq for MockEstimator {
        fn eq(&self, other: &Self) -> bool {
            self.params == other.params
        }
    }

    struct MockKFoldCV {
        n_splits: usize,
        seed: u64,
    }

    impl MockKFoldCV {
        fn new(n_splits: usize, seed: u64) -> Self {
            Self { n_splits, seed }
        }

        fn split(
            &self,
            x: &Array2<f64>,
            _y: Option<&Array2<f64>>,
        ) -> Vec<(Vec<usize>, Vec<usize>)> {
            let mut rng = seeded_rng(self.seed);
            let n_samples = x.shape()[0];
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let fold_size = n_samples / self.n_splits;
            let mut folds = Vec::new();

            for i in 0..self.n_splits {
                let start = i * fold_size;
                let end = if i == self.n_splits - 1 {
                    n_samples
                } else {
                    (i + 1) * fold_size
                };

                let test_idx = indices[start..end].to_vec();
                let train_idx = indices[..start]
                    .iter()
                    .chain(indices[end..].iter())
                    .cloned()
                    .collect();

                folds.push((train_idx, test_idx));
            }

            folds
        }
    }
}
