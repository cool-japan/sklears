//! Restricted Boltzmann Machine (RBM) Demo
//!
//! This example demonstrates how to use RBMs for:
//! - Feature learning / dimensionality reduction
//! - Data reconstruction
//! - Generative modeling

use scirs2_core::ndarray::{array, Array1, Array2};
use sklears_core::traits::{Fit, Transform};
use sklears_neural::RBM;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Restricted Boltzmann Machine Demo ===\n");

    // Create a simple dataset with patterns
    // Bars and stripes patterns (4x4 images flattened to 16 features)
    let patterns = array![
        // Vertical stripes
        [1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
        // Horizontal stripes
        [1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.],
        // Diagonal patterns
        [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
        // Mixed patterns
        [1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1.],
        [0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0.],
    ];

    println!("Dataset shape: {:?}", patterns.shape());
    println!("Number of patterns: {}", patterns.nrows());
    println!("Features per pattern: {}\n", patterns.ncols());

    // Test 1: Feature Learning
    println!("1. Feature Learning with RBM:");
    let rbm = RBM::new()
        .n_hidden(6) // Learn 6 hidden features
        .learning_rate(0.1)
        .n_epochs(100)
        .batch_size(4)
        .random_state(42);

    let fitted_rbm = rbm.fit(&patterns, &())?;

    // Transform data to hidden representation
    let hidden_features = fitted_rbm.transform(&patterns)?;
    println!("   Hidden features shape: {:?}", hidden_features.shape());
    println!(
        "   First pattern hidden activations: {:.3}",
        hidden_features.row(0)
    );
    println!(
        "   Second pattern hidden activations: {:.3}\n",
        hidden_features.row(1)
    );

    // Test 2: Data Reconstruction
    println!("2. Data Reconstruction:");
    let reconstructed = fitted_rbm.reconstruct(&patterns)?;

    // Calculate reconstruction error
    let reconstruction_error = (&patterns - &reconstructed).mapv(|x: f64| x.powi(2)).sum()
        / (patterns.nrows() * patterns.ncols()) as f64;

    println!(
        "   Mean squared reconstruction error: {:.6}",
        reconstruction_error
    );

    // Show original vs reconstructed for first pattern
    println!("\n   Original pattern 1:");
    print_pattern(&patterns.row(0).to_owned());
    println!("\n   Reconstructed pattern 1 (probabilities):");
    print_pattern(&reconstructed.row(0).to_owned());

    // Test 3: Generative Modeling
    println!("\n3. Generative Modeling (Sampling):");
    let generated_samples = fitted_rbm.sample(4, 100)?;

    println!("   Generated {} new samples", generated_samples.nrows());
    println!("\n   Generated sample 1:");
    print_pattern(&generated_samples.row(0).to_owned());
    println!("\n   Generated sample 2:");
    print_pattern(&generated_samples.row(1).to_owned());

    // Test 4: Different Configurations
    println!("\n4. RBM with Different Configurations:");

    // Deeper RBM
    let deep_rbm = RBM::new()
        .n_hidden(10)
        .learning_rate(0.05)
        .n_epochs(200)
        .l2_reg(0.001) // Add regularization
        .momentum(0.5) // Use momentum
        .random_state(42);

    let fitted_deep = deep_rbm.fit(&patterns, &())?;
    let deep_features = fitted_deep.transform(&patterns)?;

    println!("   Deep RBM with 10 hidden units trained");
    println!(
        "   Feature dimensionality: {} -> {}",
        patterns.ncols(),
        deep_features.ncols()
    );

    // Test 5: Real-world Application - MNIST-like data
    println!("\n5. Application: Learning digit features");

    // Create simplified digit-like patterns (8x8 = 64 features)
    let digits = create_digit_patterns();

    // Simple min-max scaling to [0, 1]
    let mut scaled_digits = digits.clone();
    for j in 0..scaled_digits.ncols() {
        let column = scaled_digits.column(j);
        let min_val = column.fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = column.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() > 1e-8 {
            let mut col_mut = scaled_digits.column_mut(j);
            col_mut.mapv_inplace(|x| (x - min_val) / (max_val - min_val));
        }
    }

    // Train RBM
    let digit_rbm = RBM::new()
        .n_hidden(20)
        .learning_rate(0.01)
        .n_epochs(50)
        .batch_size(10)
        .random_state(42);

    let fitted_digit_rbm = digit_rbm.fit(&scaled_digits, &())?;

    println!("   Trained RBM on digit patterns");
    println!(
        "   Learned weights shape: {:?}",
        fitted_digit_rbm.weights().shape()
    );

    // Visualize learned features (weights)
    println!("\n   Visualization of first 3 hidden unit weights:");
    for i in 0..3.min(fitted_digit_rbm.weights().ncols()) {
        println!("   Hidden unit {}:", i);
        let weights = fitted_digit_rbm.weights().column(i);
        // Show top 10 strongest connections
        let mut weight_indices: Vec<_> =
            (0..weights.len()).map(|j| (j, weights[j].abs())).collect();
        weight_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        print!("   Top connections: ");
        for &(feat_idx, _weight_abs) in weight_indices.iter().take(5) {
            print!(
                "feat{}: {:.2}, ",
                feat_idx,
                fitted_digit_rbm.weights()[[feat_idx, i]]
            );
        }
        println!();
    }

    println!("\n=== Summary ===");
    println!("RBMs are powerful for:");
    println!("- Unsupervised feature learning");
    println!("- Dimensionality reduction");
    println!("- Data reconstruction and denoising");
    println!("- Generative modeling");
    println!("- Pre-training deep neural networks");

    Ok(())
}

/// Helper function to print a pattern as a 4x4 grid
fn print_pattern(pattern: &Array1<f64>) {
    for i in 0..4 {
        print!("   ");
        for j in 0..4 {
            let val = pattern[i * 4 + j];
            if val > 0.5 {
                print!("██ ");
            } else {
                print!("░░ ");
            }
        }
        println!();
    }
}

/// Create simplified digit-like patterns
fn create_digit_patterns() -> Array2<f64> {
    // Simple 8x8 patterns representing digits 0-9
    array![
        // "0" pattern
        [
            0., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
            1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
            0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0.
        ],
        // "1" pattern
        [
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0.
        ],
        // More patterns...
        [
            1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.
        ],
    ]
    .mapv(|x| x * 255.0) // Scale to 0-255 range like real images
}
