// Simple SVM Performance Test
// This tests the performance improvements from the recent optimizations

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::Fit;
use sklears_svm::svc::SVC;
use std::time::Instant;

fn generate_simple_dataset(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();

    // Generate positive class samples
    for i in 0..(n_samples / 2) {
        for j in 0..n_features {
            x_vec.push((i as f64 + 1.0) + (j as f64 * 0.1));
        }
        y_vec.push(1.0);
    }

    // Generate negative class samples
    for i in 0..(n_samples / 2) {
        for j in 0..n_features {
            x_vec.push(-(i as f64 + 1.0) - (j as f64 * 0.1));
        }
        y_vec.push(0.0);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_vec).expect("Failed to create array");
    let y = Array1::from_vec(y_vec);

    (x, y)
}

fn test_dataset_size(n_samples: usize, n_features: usize, n_runs: usize) -> (f64, f64, f64) {
    let (x, y) = generate_simple_dataset(n_samples, n_features);

    let mut times = Vec::new();

    for _ in 0..n_runs {
        let start = Instant::now();
        let svc = SVC::new().linear().tol(0.01).max_iter(100);
        match svc.fit(&x, &y) {
            Ok(_) => {
                let elapsed = start.elapsed();
                times.push(elapsed.as_secs_f64());
            }
            Err(e) => {
                eprintln!("Error fitting SVM: {}", e);
                return (0.0, 0.0, 0.0);
            }
        }
    }

    // Calculate statistics
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let max = times[times.len() - 1];
    let median = times[times.len() / 2];

    (min, median, max)
}

fn main() {
    println!("=== SVM Performance Benchmark ===\n");
    println!("Testing SVM training performance with various dataset sizes");
    println!("Note: Performance optimization ongoing, see benchmarks for current status\n");

    let test_configs = vec![
        (6, 2, 10, "Tiny (6 samples)"),
        (20, 2, 10, "Small (20 samples)"),
        (50, 5, 5, "Medium (50 samples)"),
        (100, 5, 5, "Large (100 samples)"),
        (200, 10, 3, "Very Large (200 samples)"),
    ];

    println!(
        "{:<25} {:>12} {:>12} {:>12}",
        "Dataset Size", "Min (s)", "Median (s)", "Max (s)"
    );
    println!("{}", "-".repeat(65));

    for (n_samples, n_features, n_runs, label) in test_configs {
        let (min, median, max) = test_dataset_size(n_samples, n_features, n_runs);
        println!("{:<25} {:>12.4} {:>12.4} {:>12.4}", label, min, median, max);
    }

    println!("\n=== Performance Targets ===");
    println!("6-20 samples:   < 0.5s (target)");
    println!("50-100 samples: < 2s (target)");
    println!("200 samples:    < 5s (target)");
    println!("\nNote: Performance optimization is ongoing");
}
