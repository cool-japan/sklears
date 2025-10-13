//! EMD Decomposition Demo
//!
//! This example demonstrates the usage of the EMD (Empirical Mode Decomposition) module.

use scirs2_core::ndarray::Array1;
use sklears_decomposition::emd_decomposition::{
    BoundaryCondition, EmpiricalModeDecomposition, InterpolationMethod,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("EMD Decomposition Demo");
    println!("=====================");

    // Create a composite signal with multiple frequency components
    let n = 200;
    let signal = Array1::from_vec(
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.01;
                // Combine low-frequency trend, medium frequency, and high frequency components
                0.5 * t +                                    // Linear trend
                (2.0 * PI * 2.0 * t).sin() +               // Low frequency (2 Hz)
                0.5 * (2.0 * PI * 10.0 * t).sin() +        // Medium frequency (10 Hz)
                0.2 * (2.0 * PI * 50.0 * t).sin() // High frequency (50 Hz)
            })
            .collect(),
    );

    println!("Original signal length: {} samples", signal.len());
    println!("Signal energy: {:.4}", signal.mapv(|x| x * x).sum().sqrt());

    // Configure EMD with custom parameters
    let emd = EmpiricalModeDecomposition::new()
        .max_sift_iter(100)
        .tolerance(1e-6)
        .max_imfs(6)
        .boundary_condition(BoundaryCondition::Mirror)
        .interpolation(InterpolationMethod::CubicSpline);

    // Perform EMD decomposition
    println!("\nPerforming EMD decomposition...");
    let start_time = std::time::Instant::now();

    match emd.decompose(&signal) {
        Ok(result) => {
            let duration = start_time.elapsed();

            println!("✅ EMD decomposition completed in {:?}", duration);
            println!(
                "📊 Extracted {} Intrinsic Mode Functions (IMFs)",
                result.n_imfs
            );
            println!(
                "📈 Residual energy: {:.6}",
                result.residual.mapv(|x| x * x).sum().sqrt()
            );

            // Test perfect reconstruction
            let reconstructed = result.reconstruct();
            let reconstruction_error = (&signal - &reconstructed).mapv(|x| x * x).sum().sqrt();
            println!("🔍 Reconstruction error: {:.2e}", reconstruction_error);

            // Display IMF statistics
            println!("\nIMF Analysis:");
            println!("=============");
            for i in 0..result.n_imfs {
                if let Some(imf) = result.imf(i) {
                    let energy = imf.mapv(|x| x * x).sum().sqrt();
                    let mean_abs = imf.mapv(|x| x.abs()).mean().unwrap();
                    println!(
                        "IMF {}: Energy = {:.6}, Mean Abs = {:.6}",
                        i + 1,
                        energy,
                        mean_abs
                    );
                }
            }

            // Demonstrate instantaneous frequency analysis
            match result.instantaneous_frequency(100.0) {
                Ok(frequencies) => {
                    println!("\n📡 Instantaneous frequency analysis completed");
                    println!("   Frequency matrix shape: {:?}", frequencies.shape());
                }
                Err(e) => println!("⚠️  Instantaneous frequency analysis failed: {}", e),
            }

            println!("\n✨ EMD Demo completed successfully!");
        }
        Err(e) => {
            println!("❌ EMD decomposition failed: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
