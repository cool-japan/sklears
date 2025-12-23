//! Autoencoder Demo
//!
//! This example demonstrates various autoencoder applications:
//! - Standard autoencoder for dimensionality reduction
//! - Denoising autoencoder for data cleaning
//! - Sparse autoencoder for feature learning
//! - Deep autoencoder for complex representations

use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{array, s, Array2};
use scirs2_core::random::seeded_rng;
use sklears_core::traits::{Fit, Transform};
use sklears_neural::{Activation, Autoencoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Autoencoder Demo ===\n");

    // Generate synthetic data (simple blob-like patterns)
    let mut rng = seeded_rng(42);
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let mut data = Array2::zeros((200, 10));

    // Create three blob-like clusters
    for i in 0..200 {
        let cluster = i / 67; // Three clusters
        let cluster_center = match cluster {
            0 => array![-2.0, -2.0, 0.0, 1.0, 0.5, -1.0, 2.0, -0.5, 1.5, -1.5],
            1 => array![2.0, 1.5, -1.0, -0.5, 2.5, 0.0, -2.0, 1.0, -1.5, 2.0],
            _ => array![0.0, 2.0, 1.5, -2.0, -1.0, 2.0, 0.5, -2.5, 0.0, 1.0],
        };

        for j in 0..10 {
            data[[i, j]] = cluster_center[j] + rng.sample(normal_dist) * 0.5;
        }
    }

    // Simple standardization (z-score normalization)
    let mut scaled_data = data.clone();
    for j in 0..scaled_data.ncols() {
        let column = scaled_data.column(j);
        let mean = column.mean().unwrap();
        let std = column
            .mapv(|x: f64| (x - mean).powi(2))
            .mean()
            .unwrap()
            .sqrt();
        let mut col_mut = scaled_data.column_mut(j);
        col_mut.mapv_inplace(|x| (x - mean) / std);
    }

    println!("Dataset shape: {:?}", scaled_data.shape());
    println!("Number of samples: {}", scaled_data.nrows());
    println!("Original dimensionality: {}\n", scaled_data.ncols());

    // Test 1: Standard Autoencoder for Dimensionality Reduction
    println!("1. Standard Autoencoder for Dimensionality Reduction:");
    let ae = Autoencoder::new()
        .encoder_layers(vec![8, 6, 4, 2]) // Compress to 2D
        .activation(Activation::Relu)
        .learning_rate(0.001)
        .n_epochs(50)
        .batch_size(32)
        .random_state(42);

    let fitted_ae = ae.fit(&scaled_data, &())?;

    // Transform to low-dimensional representation
    let encoded = fitted_ae.transform(&scaled_data)?;
    println!(
        "   Encoded shape: {:?} (reduced from {} to {} dimensions)",
        encoded.shape(),
        scaled_data.ncols(),
        encoded.ncols()
    );

    // Reconstruct data
    let reconstructed = fitted_ae.reconstruct(&scaled_data)?;
    let reconstruction_error = (&scaled_data - &reconstructed)
        .mapv(|x: f64| x.powi(2))
        .mean()
        .unwrap();
    println!(
        "   Mean squared reconstruction error: {:.6}",
        reconstruction_error
    );

    // Show sample encodings
    println!("\n   Sample encodings (first 5 samples):");
    for i in 0..5.min(encoded.nrows()) {
        println!(
            "   Sample {}: [{:.3}, {:.3}]",
            i,
            encoded[[i, 0]],
            encoded[[i, 1]]
        );
    }

    // Test 2: Denoising Autoencoder
    println!("\n2. Denoising Autoencoder:");

    // Add noise to data
    let mut rng = seeded_rng(42);
    let normal_dist = Normal::new(0.0, 0.3).unwrap();
    let mut noise = Array2::zeros(scaled_data.dim());
    for i in 0..noise.nrows() {
        for j in 0..noise.ncols() {
            noise[[i, j]] = rng.sample(normal_dist);
        }
    }
    let noisy_data = &scaled_data + &noise;

    let denoising_ae = Autoencoder::new()
        .encoder_layers(vec![8, 6, 4, 6, 8]) // Symmetric architecture
        .activation(Activation::Tanh)
        .noise_factor(0.2) // Add noise during training
        .learning_rate(0.001)
        .n_epochs(100)
        .l2_reg(0.001)
        .random_state(42);

    let fitted_denoising = denoising_ae.fit(&scaled_data, &())?;

    // Denoise the noisy data
    let denoised = fitted_denoising.reconstruct(&noisy_data)?;

    // Compare errors
    let noise_level = (&noisy_data - &scaled_data)
        .mapv(|x: f64| x.powi(2))
        .mean()
        .unwrap();
    let denoised_error = (&denoised - &scaled_data)
        .mapv(|x: f64| x.powi(2))
        .mean()
        .unwrap();

    println!("   Original noise level: {:.6}", noise_level);
    println!("   After denoising error: {:.6}", denoised_error);
    println!(
        "   Noise reduction: {:.1}%",
        (1.0 - denoised_error / noise_level) * 100.0
    );

    // Test 3: Sparse Autoencoder
    println!("\n3. Sparse Autoencoder:");
    let sparse_ae = Autoencoder::new()
        .encoder_layers(vec![20, 15]) // Overcomplete representation
        .activation(Activation::Logistic)
        .sparsity(0.05, 0.1) // Target 5% activation with weight 0.1
        .learning_rate(0.001)
        .n_epochs(50)
        .random_state(42);

    let fitted_sparse = sparse_ae.fit(&scaled_data, &())?;
    let sparse_encoding = fitted_sparse.transform(&scaled_data)?;

    // Calculate sparsity (percentage of near-zero activations)
    let threshold = 0.1;
    let total_activations = sparse_encoding.len() as f64;
    let sparse_activations = sparse_encoding
        .iter()
        .filter(|&&x| x.abs() < threshold)
        .count() as f64;
    let sparsity = sparse_activations / total_activations;

    println!("   Sparse encoding shape: {:?}", sparse_encoding.shape());
    println!(
        "   Achieved sparsity: {:.1}% of activations < {}",
        sparsity * 100.0,
        threshold
    );

    // Test 4: Autoencoder with Different Activation Functions
    println!("\n4. Autoencoder with Different Activation Functions:");

    // Different activation functions
    let ae_relu = Autoencoder::new()
        .encoding_dim(3)
        .activation(Activation::Relu)
        .learning_rate(0.01)
        .n_epochs(30)
        .random_state(42);

    let fitted_relu = ae_relu.fit(&scaled_data, &())?;

    let ae_tanh = Autoencoder::new()
        .encoding_dim(3)
        .activation(Activation::Tanh)
        .learning_rate(0.01)
        .n_epochs(30)
        .random_state(42);

    let fitted_tanh = ae_tanh.fit(&scaled_data, &())?;

    // Compare reconstruction errors
    let recon_relu = fitted_relu.reconstruct(&scaled_data)?;
    let recon_tanh = fitted_tanh.reconstruct(&scaled_data)?;

    let error_relu = (&scaled_data - &recon_relu)
        .mapv(|x: f64| x.powi(2))
        .mean()
        .unwrap();
    let error_tanh = (&scaled_data - &recon_tanh)
        .mapv(|x: f64| x.powi(2))
        .mean()
        .unwrap();

    println!("   ReLU activation MSE: {:.6}", error_relu);
    println!("   Tanh activation MSE: {:.6}", error_tanh);

    // Test 5: Anomaly Detection with Autoencoder
    println!("\n5. Anomaly Detection with Autoencoder:");

    // Create some anomalous samples
    let mut anomalies = Array2::zeros((5, scaled_data.ncols()));
    for i in 0..5 {
        for j in 0..scaled_data.ncols() {
            anomalies[[i, j]] = if j % 2 == 0 { 3.0 } else { -3.0 };
        }
    }

    // Use reconstruction error for anomaly detection
    let normal_reconstruction = fitted_ae.reconstruct(&scaled_data)?;
    let anomaly_reconstruction = fitted_ae.reconstruct(&anomalies)?;

    let normal_errors: Vec<f64> = (0..scaled_data.nrows())
        .map(|i| {
            let orig = scaled_data.slice(s![i, ..]);
            let recon = normal_reconstruction.slice(s![i, ..]);
            (&orig - &recon).mapv(|x: f64| x.powi(2)).sum()
        })
        .collect();

    let anomaly_errors: Vec<f64> = (0..anomalies.nrows())
        .map(|i| {
            let orig = anomalies.slice(s![i, ..]);
            let recon = anomaly_reconstruction.slice(s![i, ..]);
            (&orig - &recon).mapv(|x: f64| x.powi(2)).sum()
        })
        .collect();

    let normal_mean_error = normal_errors.iter().sum::<f64>() / normal_errors.len() as f64;
    let anomaly_mean_error = anomaly_errors.iter().sum::<f64>() / anomaly_errors.len() as f64;

    println!(
        "   Normal samples mean reconstruction error: {:.6}",
        normal_mean_error
    );
    println!(
        "   Anomaly samples mean reconstruction error: {:.6}",
        anomaly_mean_error
    );
    println!(
        "   Anomaly detection ratio: {:.2}x higher error",
        anomaly_mean_error / normal_mean_error
    );

    println!("\n=== Summary ===");
    println!("Autoencoders are versatile for:");
    println!("- Dimensionality reduction (like nonlinear PCA)");
    println!("- Data denoising and cleaning");
    println!("- Feature learning with sparsity constraints");
    println!("- Anomaly detection via reconstruction error");
    println!("- Pre-training deep networks");
    println!("- Generative modeling (with variational autoencoders)");

    Ok(())
}
