#![cfg(feature = "incomplete-benchmarks")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_chacha::ChaCha8Rng;
use sklears_model_selection::{
    BanditOptimization, BanditStrategy, BayesSearchCV, GeneticAlgorithmCV, GeneticAlgorithmConfig,
    GridSearchCV, ParameterDistribution as BayesParamDistribution, ParameterSpace,
};

// Mock estimator for benchmarking
#[derive(Clone)]
struct MockEstimator {
    param1: f64,
    param2: i32,
}

impl MockEstimator {
    fn new() -> Self {
        Self {
            param1: 1.0,
            param2: 1,
        }
    }

    fn set_params(&mut self, params: &HashMap<String, ParameterValue>) {
        if let Some(ParameterValue::Float(val)) = params.get("param1") {
            self.param1 = *val;
        }
        if let Some(ParameterValue::Integer(val)) = params.get("param2") {
            self.param2 = *val;
        }
    }

    // Mock objective function - sum of squared differences from optimal values
    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        // Optimal values are param1=2.5, param2=5
        let param1_loss = (self.param1 - 2.5).powi(2);
        let param2_loss = ((self.param2 as f64) - 5.0).powi(2);
        let total_loss = param1_loss + param2_loss;

        // Convert to score (higher is better)
        let base_score = 0.9; // Base score
        let noise = (x.sum() + y.sum()) * 1e-6; // Small noise based on data

        base_score - total_loss * 0.1 + noise
    }
}

fn generate_test_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let x = Array2::random_using(
        (n_samples, n_features),
        rand_distr::StandardNormal,
        &mut rng,
    );
    let y = Array1::random_using(n_samples, rand_distr::StandardNormal, &mut rng);
    (x, y)
}

fn create_parameter_space() -> ParameterSpace {
    let mut space = ParameterSpace::new();
    space.add_parameter(
        "param1",
        Parameter::Float {
            low: 0.0,
            high: 5.0,
            log: false,
        },
    );
    space.add_parameter("param2", Parameter::Integer { low: 1, high: 10 });
    space
}

fn bench_grid_search_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Grid Search Optimization");

    for &n_samples in &[100, 500, 1000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("GridSearch-Small", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let param_grid = vec![
                        (
                            "param1".to_string(),
                            vec![
                                ParameterValue::Float(1.0),
                                ParameterValue::Float(2.0),
                                ParameterValue::Float(3.0),
                            ],
                        ),
                        (
                            "param2".to_string(),
                            vec![
                                ParameterValue::Integer(1),
                                ParameterValue::Integer(5),
                                ParameterValue::Integer(10),
                            ],
                        ),
                    ]
                    .into_iter()
                    .collect();

                    let mut estimator = MockEstimator::new();
                    let mut best_score = f64::NEG_INFINITY;
                    let mut best_params = HashMap::new();

                    // Generate all combinations
                    let param1_values = &param_grid["param1"];
                    let param2_values = &param_grid["param2"];

                    for p1 in param1_values {
                        for p2 in param2_values {
                            let mut params = HashMap::new();
                            params.insert("param1".to_string(), p1.clone());
                            params.insert("param2".to_string(), p2.clone());

                            estimator.set_params(&params);
                            let score = estimator.score(black_box(&x), black_box(&y));

                            if score > best_score {
                                best_score = score;
                                best_params = params.clone();
                            }
                        }
                    }

                    black_box((best_score, best_params))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("GridSearch-Large", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let param_grid = vec![
                        (
                            "param1".to_string(),
                            (0..10)
                                .map(|i| ParameterValue::Float(i as f64 * 0.5))
                                .collect(),
                        ),
                        (
                            "param2".to_string(),
                            (1..11).map(|i| ParameterValue::Integer(i)).collect(),
                        ),
                    ]
                    .into_iter()
                    .collect();

                    let mut estimator = MockEstimator::new();
                    let mut best_score = f64::NEG_INFINITY;
                    let mut best_params = HashMap::new();

                    // Generate all combinations
                    let param1_values = &param_grid["param1"];
                    let param2_values = &param_grid["param2"];

                    for p1 in param1_values {
                        for p2 in param2_values {
                            let mut params = HashMap::new();
                            params.insert("param1".to_string(), p1.clone());
                            params.insert("param2".to_string(), p2.clone());

                            estimator.set_params(&params);
                            let score = estimator.score(black_box(&x), black_box(&y));

                            if score > best_score {
                                best_score = score;
                                best_params = params.clone();
                            }
                        }
                    }

                    black_box((best_score, best_params))
                })
            },
        );
    }

    group.finish();
}

fn bench_random_search_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Random Search Optimization");

    for &n_samples in &[100, 500, 1000] {
        let (x, y) = generate_test_data(n_samples, 10);
        let parameter_space = create_parameter_space();

        for &n_iterations in &[10, 50, 100] {
            group.bench_with_input(
                BenchmarkId::new(format!("RandomSearch-{}", n_iterations), n_samples),
                &n_samples,
                |b, _| {
                    b.iter(|| {
                        let mut rng = ChaCha8Rng::seed_from_u64(42);
                        let mut estimator = MockEstimator::new();
                        let mut best_score = f64::NEG_INFINITY;
                        let mut best_params = HashMap::new();

                        for _ in 0..n_iterations {
                            let params = parameter_space.sample(&mut rng);
                            estimator.set_params(&params);
                            let score = estimator.score(black_box(&x), black_box(&y));

                            if score > best_score {
                                best_score = score;
                                best_params = params.clone();
                            }
                        }

                        black_box((best_score, best_params))
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_bandit_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bandit Optimization");

    for &n_samples in &[100, 500, 1000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("BanditUCB", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut optimizer = BanditOptimizer::new(BanditStrategy::UCB { c: 1.41 });
                    let parameter_space = create_parameter_space();
                    let mut estimator = MockEstimator::new();
                    let mut best_score = f64::NEG_INFINITY;
                    let mut best_params = HashMap::new();

                    for _ in 0..50 {
                        let params = optimizer.suggest_parameters(&parameter_space);
                        estimator.set_params(&params);
                        let score = estimator.score(black_box(&x), black_box(&y));

                        optimizer.update(params.clone(), score);

                        if score > best_score {
                            best_score = score;
                            best_params = params.clone();
                        }
                    }

                    black_box((best_score, best_params))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("BanditEpsilonGreedy", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut optimizer =
                        BanditOptimizer::new(BanditStrategy::EpsilonGreedy { epsilon: 0.1 });
                    let parameter_space = create_parameter_space();
                    let mut estimator = MockEstimator::new();
                    let mut best_score = f64::NEG_INFINITY;
                    let mut best_params = HashMap::new();

                    for _ in 0..50 {
                        let params = optimizer.suggest_parameters(&parameter_space);
                        estimator.set_params(&params);
                        let score = estimator.score(black_box(&x), black_box(&y));

                        optimizer.update(params.clone(), score);

                        if score > best_score {
                            best_score = score;
                            best_params = params.clone();
                        }
                    }

                    black_box((best_score, best_params))
                })
            },
        );
    }

    group.finish();
}

fn bench_evolutionary_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Evolutionary Optimization");

    for &n_samples in &[100, 500, 1000] {
        let (x, y) = generate_test_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("GeneticAlgorithm", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let config = GeneticAlgorithmConfig {
                        population_size: 20,
                        n_generations: 10,
                        crossover_rate: 0.8,
                        mutation_rate: 0.1,
                        tournament_size: 3,
                        elitism_count: 2,
                        random_state: Some(42),
                    };

                    let parameter_space = create_parameter_space();
                    let mut ga = GeneticAlgorithm::new(config, parameter_space);
                    let mut estimator = MockEstimator::new();

                    // Objective function closure
                    let objective = |params: &HashMap<String, ParameterValue>| -> f64 {
                        let mut est = estimator.clone();
                        est.set_params(params);
                        est.score(&x, &y)
                    };

                    let (best_params, best_score) = ga.optimize(black_box(objective));
                    black_box((best_score, best_params))
                })
            },
        );
    }

    group.finish();
}

fn bench_optimization_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimization Convergence");

    let (x, y) = generate_test_data(500, 10);

    // Compare convergence speed of different algorithms
    group.bench_function("GridSearch-Convergence", |b| {
        b.iter(|| {
            let param_grid = vec![
                (
                    "param1".to_string(),
                    (0..20)
                        .map(|i| ParameterValue::Float(i as f64 * 0.25))
                        .collect(),
                ),
                (
                    "param2".to_string(),
                    (1..21).map(|i| ParameterValue::Integer(i)).collect(),
                ),
            ]
            .into_iter()
            .collect();

            let mut estimator = MockEstimator::new();
            let mut scores = Vec::new();

            let param1_values = &param_grid["param1"];
            let param2_values = &param_grid["param2"];

            for p1 in param1_values {
                for p2 in param2_values {
                    let mut params = HashMap::new();
                    params.insert("param1".to_string(), p1.clone());
                    params.insert("param2".to_string(), p2.clone());

                    estimator.set_params(&params);
                    let score = estimator.score(black_box(&x), black_box(&y));
                    scores.push(score);
                }
            }

            black_box(scores)
        })
    });

    group.bench_function("RandomSearch-Convergence", |b| {
        b.iter(|| {
            let parameter_space = create_parameter_space();
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let mut estimator = MockEstimator::new();
            let mut scores = Vec::new();

            for _ in 0..100 {
                let params = parameter_space.sample(&mut rng);
                estimator.set_params(&params);
                let score = estimator.score(black_box(&x), black_box(&y));
                scores.push(score);
            }

            black_box(scores)
        })
    });

    group.finish();
}

fn bench_optimization_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimization Scalability");

    // Test performance with different parameter space sizes
    for &n_params in &[2, 5, 10] {
        group.bench_with_input(
            BenchmarkId::new("RandomSearch-ParamSpace", n_params),
            &n_params,
            |b, _| {
                b.iter(|| {
                    let mut parameter_space = ParameterSpace::new();
                    for i in 0..n_params {
                        parameter_space.add_parameter(
                            &format!("param{}", i),
                            Parameter::Float {
                                low: 0.0,
                                high: 10.0,
                                log: false,
                            },
                        );
                    }

                    let mut rng = ChaCha8Rng::seed_from_u64(42);
                    let mut scores = Vec::new();

                    for _ in 0..50 {
                        let params = parameter_space.sample(&mut rng);
                        // Mock scoring based on parameter count
                        let score: f64 = params
                            .values()
                            .map(|v| match v {
                                ParameterValue::Float(f) => *f,
                                ParameterValue::Integer(i) => *i as f64,
                                _ => 0.0,
                            })
                            .sum::<f64>()
                            / params.len() as f64;
                        scores.push(score);
                    }

                    black_box(scores)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_grid_search_optimization,
    bench_random_search_optimization,
    bench_bandit_optimization,
    bench_evolutionary_optimization,
    bench_optimization_convergence,
    bench_optimization_scalability
);

criterion_main!(benches);
