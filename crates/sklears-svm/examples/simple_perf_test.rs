// Simple single-run SVM performance test

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::Fit;
use sklears_svm::svc::SVC;
use std::time::Instant;

fn main() {
    println!("=== Simple SVM Performance Test ===\n");

    // Test 1: Very small dataset (6 samples)
    println!("Test 1: 6 samples, 2 features");
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 0.1, // class 1
            2.0, 0.2, // class 1
            3.0, 0.3, // class 1
            -1.0, -0.1, // class 0
            -2.0, -0.2, // class 0
            -3.0, -0.3, // class 0
        ],
    )
    .expect("Failed to create array");
    let y = Array1::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

    let start = Instant::now();
    let svc = SVC::new().linear().tol(0.01).max_iter(100);
    match svc.fit(&x, &y) {
        Ok(_model) => {
            let elapsed = start.elapsed();
            println!("  Training time: {:.4}s", elapsed.as_secs_f64());
            println!("  Expected: < 0.5s");
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    // Test 2: Small dataset (20 samples)
    println!("\nTest 2: 20 samples, 2 features");
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    for i in 0..10 {
        x_vec.push(i as f64 + 1.0);
        x_vec.push(i as f64 * 0.1);
        y_vec.push(1.0);
    }
    for i in 0..10 {
        x_vec.push(-(i as f64 + 1.0));
        x_vec.push(-(i as f64 * 0.1));
        y_vec.push(0.0);
    }
    let x = Array2::from_shape_vec((20, 2), x_vec).expect("Failed to create array");
    let y = Array1::from_vec(y_vec);

    let start = Instant::now();
    let svc = SVC::new().linear().tol(0.01).max_iter(100);
    match svc.fit(&x, &y) {
        Ok(_model) => {
            let elapsed = start.elapsed();
            println!("  Training time: {:.4}s", elapsed.as_secs_f64());
            println!("  Expected: < 1s");
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    // Test 3: Medium dataset (100 samples)
    println!("\nTest 3: 100 samples, 5 features");
    let mut x_vec = Vec::new();
    let mut y_vec = Vec::new();
    for i in 0..50 {
        for j in 0..5 {
            x_vec.push((i as f64 + 1.0) + (j as f64 * 0.1));
        }
        y_vec.push(1.0);
    }
    for i in 0..50 {
        for j in 0..5 {
            x_vec.push(-(i as f64 + 1.0) - (j as f64 * 0.1));
        }
        y_vec.push(0.0);
    }
    let x = Array2::from_shape_vec((100, 5), x_vec).expect("Failed to create array");
    let y = Array1::from_vec(y_vec);

    let start = Instant::now();
    let svc = SVC::new().linear().tol(0.01).max_iter(100);
    match svc.fit(&x, &y) {
        Ok(_model) => {
            let elapsed = start.elapsed();
            println!("  Training time: {:.4}s", elapsed.as_secs_f64());
            println!("  Expected: < 5s");
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    println!("\n=== Summary ===");
    println!("These times reflect the current SVM implementation.");
    println!("Performance optimization is ongoing.");
}
