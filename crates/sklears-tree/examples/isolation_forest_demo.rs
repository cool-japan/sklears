//! Isolation Forest Demo
//!
//! This example demonstrates how to use Isolation Forest for anomaly detection.
//! It creates a dataset with normal samples and outliers, then uses both standard
//! and extended Isolation Forest to identify the anomalies.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict};
use sklears_tree::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Isolation Forest for Anomaly Detection ===\n");

    // Generate synthetic data with normal samples and outliers
    let (x, true_labels) = generate_anomaly_data();

    println!("Dataset:");
    println!("- {} total samples", x.nrows());
    println!(
        "- {} normal samples (label=1)",
        true_labels.iter().filter(|&&x| x == 1).count()
    );
    println!(
        "- {} anomalies (label=-1)\n",
        true_labels.iter().filter(|&&x| x == -1).count()
    );

    // Demo 1: Standard Isolation Forest
    println!("1. Standard Isolation Forest");
    println!("{}", "=".repeat(50));

    let y_dummy = Array1::zeros(x.nrows()); // Unsupervised, so dummy labels

    let model = IsolationForest::new()
        .n_estimators(100)
        .contamination(0.1) // Expect 10% outliers
        .random_state(42);

    let fitted_model = model.fit(&x, &y_dummy)?;

    // Get anomaly scores
    let scores = fitted_model.decision_function(&x)?;

    // Make predictions
    let predictions = fitted_model.predict(&x)?;

    // Calculate accuracy
    let correct = predictions
        .iter()
        .zip(true_labels.iter())
        .filter(|(pred, true_label)| pred == true_label)
        .count();
    let accuracy = (correct as f64) / (true_labels.len() as f64) * 100.0;

    println!("Accuracy: {:.2}%", accuracy);
    println!("Threshold: {:.4}", fitted_model.threshold());

    // Show some examples
    println!("\nSample predictions (first 10):");
    for i in 0..10.min(predictions.len()) {
        println!(
            "  Sample {}: score={:.4}, prediction={:2}, actual={:2}",
            i, scores[i], predictions[i], true_labels[i]
        );
    }

    // Demo 2: Extended Isolation Forest
    println!("\n2. Extended Isolation Forest (Hyperplane Splits)");
    println!("{}", "=".repeat(50));

    let extended_model = IsolationForest::new()
        .n_estimators(100)
        .contamination(0.1)
        .extended(true) // Enable Extended IF
        .extension_level(2) // Use 2D hyperplanes
        .random_state(42);

    let fitted_extended = extended_model.fit(&x, &y_dummy)?;

    let extended_predictions = fitted_extended.predict(&x)?;
    let extended_scores = fitted_extended.decision_function(&x)?;

    let extended_correct = extended_predictions
        .iter()
        .zip(true_labels.iter())
        .filter(|(pred, true_label)| pred == true_label)
        .count();
    let extended_accuracy = (extended_correct as f64) / (true_labels.len() as f64) * 100.0;

    println!("Accuracy: {:.2}%", extended_accuracy);
    println!("Threshold: {:.4}", fitted_extended.threshold());

    println!("\nSample predictions (first 10):");
    for i in 0..10.min(extended_predictions.len()) {
        println!(
            "  Sample {}: score={:.4}, prediction={:2}, actual={:2}",
            i, extended_scores[i], extended_predictions[i], true_labels[i]
        );
    }

    // Demo 3: Streaming Isolation Forest
    println!("\n3. Streaming Isolation Forest");
    println!("{}", "=".repeat(50));

    let config = IsolationForestConfig {
        n_estimators: 50,
        contamination: 0.1,
        ..Default::default()
    };

    let mut streaming_if = StreamingIsolationForest::new(
        config, 100, // window size
        50,  // update frequency
    );

    println!("Processing samples online...");

    let mut streaming_predictions = Vec::new();
    for i in 0..x.nrows() {
        let sample = x.row(i).to_owned();
        let score = streaming_if.process_sample(sample)?;
        let prediction = if score >= 0.5 { -1 } else { 1 }; // Simple threshold
        streaming_predictions.push(prediction);
    }

    let streaming_correct = streaming_predictions
        .iter()
        .zip(true_labels.iter())
        .filter(|(pred, true_label)| pred == true_label)
        .count();
    let streaming_accuracy = (streaming_correct as f64) / (true_labels.len() as f64) * 100.0;

    println!("Streaming accuracy: {:.2}%", streaming_accuracy);

    // Summary
    println!("\n=== Summary ===");
    println!("Standard IF accuracy:  {:.2}%", accuracy);
    println!("Extended IF accuracy:  {:.2}%", extended_accuracy);
    println!("Streaming IF accuracy: {:.2}%", streaming_accuracy);

    Ok(())
}

/// Generate synthetic anomaly detection data
fn generate_anomaly_data() -> (Array2<f64>, Array1<i32>) {
    let n_normal = 900;
    let n_anomalies = 100;
    let n_samples = n_normal + n_anomalies;

    let mut x = Array2::zeros((n_samples, 2));
    let mut labels = Array1::zeros(n_samples);

    // Generate normal samples (clustered around origin)
    for i in 0..n_normal {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / (n_normal as f64);
        let radius = 1.0 + ((i % 10) as f64) * 0.1 + rand_normal() * 0.2;
        x[[i, 0]] = radius * angle.cos();
        x[[i, 1]] = radius * angle.sin();
        labels[i] = 1; // Normal
    }

    // Generate anomalies (scattered far from cluster)
    for i in 0..n_anomalies {
        let idx = n_normal + i;
        // Anomalies are far from the normal cluster
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / (n_anomalies as f64);
        let radius = 5.0 + ((i % 5) as f64) * 2.0;
        x[[idx, 0]] = radius * angle.cos();
        x[[idx, 1]] = radius * angle.sin();
        labels[idx] = -1; // Anomaly
    }

    (x, labels)
}

/// Simple pseudo-random normal distribution (for demo purposes)
fn rand_normal() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    // Box-Muller transform (simplified)
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let u1 = ((seed % 1000000) as f64) / 1000000.0;
    let u2 = (((seed / 1000000) % 1000000) as f64) / 1000000.0;

    let z = ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()).abs();

    // Clamp to reasonable range
    z.clamp(-3.0, 3.0)
}
