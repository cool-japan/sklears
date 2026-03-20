//! Performance benchmarks for sklears
//! 
//! This module provides performance benchmarks and comparisons
//! to track performance regressions and measure improvements.

use std::time::Instant;
use ndarray::{Array1, Array2};
use sklears_core::traits::{Transform};
use sklears_metrics::{accuracy_score, mean_squared_error};
use sklears_utils::data_generation::{make_classification, make_regression, make_blobs};
use sklears_utils::random::{train_test_split_indices, set_random_state};
use sklears_preprocessing::{StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer, PolynomialFeatures};
use sklears_model_selection::{KFold, cross_val_score};
use sklears_tree::{DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier};

/// Benchmark configuration
#[derive(Clone)]
pub struct BenchmarkConfig {
    pub name: String,
    pub n_samples: usize,
    pub n_features: usize,
    pub n_iterations: usize,
}

impl BenchmarkConfig {
    pub fn new(name: &str, n_samples: usize, n_features: usize, n_iterations: usize) -> Self {
        Self {
            name: name.to_string(),
            n_samples,
            n_features,
            n_iterations,
        }
    }
}

/// Benchmark result
#[derive(Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub avg_time_ms: f64,
    pub std_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub throughput_samples_per_sec: f64,
}

impl BenchmarkResult {
    pub fn from_times(name: String, times_ms: &[f64], n_samples: usize) -> Self {
        let avg_time_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let variance = times_ms.iter()
            .map(|&t| (t - avg_time_ms).powi(2))
            .sum::<f64>() / times_ms.len() as f64;
        let std_time_ms = variance.sqrt();
        let min_time_ms = times_ms.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time_ms = times_ms.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let throughput_samples_per_sec = (n_samples as f64) / (avg_time_ms / 1000.0);
        
        Self {
            name,
            avg_time_ms,
            std_time_ms,
            min_time_ms,
            max_time_ms,
            throughput_samples_per_sec,
        }
    }
}

/// Utility function to run a benchmark
pub fn run_benchmark<F>(config: &BenchmarkConfig, mut benchmark_fn: F) -> BenchmarkResult
where
    F: FnMut() -> (),
{
    let mut times = Vec::with_capacity(config.n_iterations);
    
    // Warmup
    for _ in 0..2 {
        benchmark_fn();
    }
    
    // Actual benchmark
    for _ in 0..config.n_iterations {
        let start = Instant::now();
        benchmark_fn();
        let duration = start.elapsed();
        times.push(duration.as_secs_f64() * 1000.0); // Convert to milliseconds
    }
    
    BenchmarkResult::from_times(config.name.clone(), &times, config.n_samples)
}

#[test]
fn benchmark_standard_scaler() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("StandardScaler_100x10", 100, 10, 100),
        BenchmarkConfig::new("StandardScaler_1000x10", 1000, 10, 50),
        BenchmarkConfig::new("StandardScaler_1000x50", 1000, 50, 50),
        BenchmarkConfig::new("StandardScaler_5000x20", 5000, 20, 20),
    ];
    
    println!("StandardScaler Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        let (x, _) = make_regression(config.n_samples, config.n_features, 1, 0.1, 42).unwrap();
        
        let result = run_benchmark(&config, || {
            let scaler = StandardScaler::new().fit(&x, &()).unwrap();
            let _x_scaled = scaler.transform(&x).unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

#[test]
fn benchmark_minmax_scaler() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("MinMaxScaler_100x10", 100, 10, 100),
        BenchmarkConfig::new("MinMaxScaler_1000x10", 1000, 10, 50),
        BenchmarkConfig::new("MinMaxScaler_1000x50", 1000, 50, 50),
        BenchmarkConfig::new("MinMaxScaler_5000x20", 5000, 20, 20),
    ];
    
    println!("\nMinMaxScaler Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        let (x, _) = make_regression(config.n_samples, config.n_features, 1, 0.1, 42).unwrap();
        
        let result = run_benchmark(&config, || {
            let scaler = MinMaxScaler::new().fit(&x, &()).unwrap();
            let _x_scaled = scaler.transform(&x).unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

#[test]
fn benchmark_decision_tree() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("DecisionTree_100x5", 100, 5, 50),
        BenchmarkConfig::new("DecisionTree_500x10", 500, 10, 20),
        BenchmarkConfig::new("DecisionTree_1000x10", 1000, 10, 10),
        BenchmarkConfig::new("DecisionTree_2000x20", 2000, 20, 5),
    ];
    
    println!("\nDecision Tree Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        let (x, y) = make_classification(config.n_samples, config.n_features, 3, 2, 0.1, 42).unwrap();
        
        let result = run_benchmark(&config, || {
            let dt = DecisionTreeClassifier::new()
                .max_depth(Some(10))
                .fit(&x, &y).unwrap();
            let _predictions = dt.predict(&x).unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

#[test]
fn benchmark_random_forest() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("RandomForest_100x5", 100, 5, 20),
        BenchmarkConfig::new("RandomForest_500x10", 500, 10, 10),
        BenchmarkConfig::new("RandomForest_1000x10", 1000, 10, 5),
    ];
    
    println!("\nRandom Forest Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        let (x, y) = make_classification(config.n_samples, config.n_features, 3, 2, 0.1, 42).unwrap();
        
        let result = run_benchmark(&config, || {
            let rf = RandomForestClassifier::new()
                .n_estimators(10)
                .max_depth(Some(5))
                .fit(&x, &y).unwrap();
            let _predictions = rf.predict(&x).unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

#[test]
fn benchmark_polynomial_features() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("PolyFeatures_100x5_deg2", 100, 5, 100),
        BenchmarkConfig::new("PolyFeatures_500x5_deg2", 500, 5, 50),
        BenchmarkConfig::new("PolyFeatures_100x10_deg2", 100, 10, 50),
        BenchmarkConfig::new("PolyFeatures_100x5_deg3", 100, 5, 50),
    ];
    
    println!("\nPolynomial Features Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        let (x, _) = make_regression(config.n_samples, config.n_features, 1, 0.1, 42).unwrap();
        let degree = if config.name.contains("deg3") { 3 } else { 2 };
        
        let result = run_benchmark(&config, || {
            let poly = PolynomialFeatures::new()
                .degree(degree)
                .fit(&x, &()).unwrap();
            let _x_poly = poly.transform(&x).unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

#[test]
fn benchmark_one_hot_encoder() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("OneHotEncoder_100x3", 100, 3, 100),
        BenchmarkConfig::new("OneHotEncoder_500x3", 500, 3, 50),
        BenchmarkConfig::new("OneHotEncoder_1000x5", 1000, 5, 20),
    ];
    
    println!("\nOne-Hot Encoder Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        // Create categorical data
        let mut categorical_data = Vec::new();
        for _ in 0..config.n_samples {
            let mut row = Vec::new();
            for j in 0..config.n_features {
                let category = format!("cat_{}_{}", j, (rand::random::<usize>() % 5));
                row.push(category);
            }
            categorical_data.push(row);
        }
        
        let x = Array2::from_shape_fn((config.n_samples, config.n_features), |(i, j)| {
            categorical_data[i][j].clone()
        });
        
        let result = run_benchmark(&config, || {
            let encoder = OneHotEncoder::new().fit(&x, &()).unwrap();
            let _x_encoded = encoder.transform(&x).unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

#[test]
fn benchmark_simple_imputer() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("SimpleImputer_100x10", 100, 10, 100),
        BenchmarkConfig::new("SimpleImputer_1000x10", 1000, 10, 50),
        BenchmarkConfig::new("SimpleImputer_5000x20", 5000, 20, 20),
    ];
    
    println!("\nSimple Imputer Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        let (mut x, _) = make_regression(config.n_samples, config.n_features, 1, 0.1, 42).unwrap();
        
        // Add missing values (10% missing)
        let n_missing = (config.n_samples * config.n_features) / 10;
        for _ in 0..n_missing {
            let i = rand::random::<usize>() % config.n_samples;
            let j = rand::random::<usize>() % config.n_features;
            x[[i, j]] = f64::NAN;
        }
        
        let result = run_benchmark(&config, || {
            let imputer = SimpleImputer::new()
                .strategy(sklears_preprocessing::ImputationStrategy::Mean)
                .fit(&x, &()).unwrap();
            let _x_imputed = imputer.transform(&x).unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

#[test]
fn benchmark_cross_validation() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("CrossVal_100x5_3fold", 100, 5, 10),
        BenchmarkConfig::new("CrossVal_500x10_5fold", 500, 10, 5),
        BenchmarkConfig::new("CrossVal_1000x10_3fold", 1000, 10, 3),
    ];
    
    println!("\nCross-Validation Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        let (x, y) = make_classification(config.n_samples, config.n_features, 3, 2, 0.1, 42).unwrap();
        let n_folds = if config.name.contains("5fold") { 5 } else { 3 };
        
        let result = run_benchmark(&config, || {
            let dt = DecisionTreeClassifier::new().max_depth(Some(5));
            let cv = KFold::new(n_folds).shuffle(true).random_state(42);
            let _scores = cross_val_score(dt, &x, &y, cv, "accuracy").unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

#[test]
fn benchmark_metrics() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("Metrics_1000", 1000, 0, 1000),
        BenchmarkConfig::new("Metrics_10000", 10000, 0, 100),
        BenchmarkConfig::new("Metrics_100000", 100000, 0, 10),
    ];
    
    println!("\nMetrics Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        let y_true = Array1::from_iter((0..config.n_samples).map(|i| (i % 3) as i32));
        let y_pred = Array1::from_iter((0..config.n_samples).map(|i| ((i + 1) % 3) as i32));
        
        let result = run_benchmark(&config, || {
            let _accuracy = accuracy_score(&y_true, &y_pred).unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

#[test]
fn benchmark_end_to_end_pipeline() {
    set_random_state(42);
    
    let configs = vec![
        BenchmarkConfig::new("E2E_Pipeline_500x10", 500, 10, 10),
        BenchmarkConfig::new("E2E_Pipeline_1000x20", 1000, 20, 5),
    ];
    
    println!("\nEnd-to-End Pipeline Performance Benchmarks:");
    println!("{:<25} {:>12} {:>12} {:>15}", "Name", "Avg (ms)", "Std (ms)", "Throughput (sps)");
    println!("{}", "-".repeat(70));
    
    for config in configs {
        let (mut x, y) = make_classification(config.n_samples, config.n_features, 3, 2, 0.1, 42).unwrap();
        
        // Add some missing values
        for i in 0..config.n_samples/20 {
            x[[i, 0]] = f64::NAN;
        }
        
        let result = run_benchmark(&config, || {
            // Complete pipeline: imputation -> scaling -> polynomial features -> model training
            let imputer = SimpleImputer::new()
                .strategy(sklears_preprocessing::ImputationStrategy::Mean)
                .fit(&x, &()).unwrap();
            let x_imputed = imputer.transform(&x).unwrap();
            
            let scaler = StandardScaler::new().fit(&x_imputed, &()).unwrap();
            let x_scaled = scaler.transform(&x_imputed).unwrap();
            
            let poly = PolynomialFeatures::new()
                .degree(2)
                .interaction_only(true)
                .fit(&x_scaled, &()).unwrap();
            let x_poly = poly.transform(&x_scaled).unwrap();
            
            let dt = DecisionTreeClassifier::new()
                .max_depth(Some(5))
                .fit(&x_poly, &y).unwrap();
            let _predictions = dt.predict(&x_poly).unwrap();
        });
        
        println!("{:<25} {:>12.2} {:>12.2} {:>15.0}", 
                result.name, result.avg_time_ms, result.std_time_ms, result.throughput_samples_per_sec);
    }
}

/// Memory usage benchmark helper
#[test]
fn memory_usage_profile() {
    set_random_state(42);
    
    println!("\nMemory Usage Profile:");
    println!("Note: These are relative comparisons, not absolute memory measurements");
    
    // Small dataset baseline
    let (x_small, y_small) = make_classification(100, 5, 3, 2, 0.1, 42).unwrap();
    
    // Large dataset for comparison
    let (x_large, y_large) = make_classification(5000, 20, 3, 2, 0.1, 42).unwrap();
    
    println!("Dataset sizes:");
    println!("  Small: {} samples × {} features", x_small.nrows(), x_small.ncols());
    println!("  Large: {} samples × {} features", x_large.nrows(), x_large.ncols());
    
    // Test memory scaling with different algorithms
    let algorithms = vec![
        ("Decision Tree", "scales well with data size"),
        ("Random Forest", "memory scales with n_estimators"),
        ("StandardScaler", "stores statistics per feature"),
        ("PolynomialFeatures", "can grow exponentially with degree"),
    ];
    
    for (name, note) in algorithms {
        println!("  {}: {}", name, note);
    }
}