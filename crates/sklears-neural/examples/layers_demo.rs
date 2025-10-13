//! Demonstration of new neural network layers: Batch Normalization and Dropout
//!
//! This example shows how to use the new layer implementations for building
//! modern neural networks with improved training stability and regularization.

use scirs2_core::ndarray::{array, Array2, Axis};
use sklears_neural::{activation::Activation, BatchNorm1d, Dropout, Layer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Network Layers Demo");
    println!("===============================\n");

    // 1. Batch Normalization Demo
    println!("1. Batch Normalization Layer");
    println!("----------------------------");

    // Create a batch normalization layer for 4 features
    let mut batch_norm = BatchNorm1d::new(4);
    batch_norm.initialize();

    // Create some sample data (batch of 3 samples, 4 features each)
    let input = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ];

    println!("Input data:");
    println!("{:8.3}", input);

    // Forward pass in training mode
    let normalized = batch_norm.forward(&input, true)?;
    println!("\nAfter batch normalization (training):");
    println!("{:8.3}", normalized);

    // Check that normalized data has approximately zero mean and unit variance
    let mean = normalized.mean_axis(Axis(0)).unwrap();
    let var = normalized.mapv(|x| x * x).mean_axis(Axis(0)).unwrap();
    println!("\nNormalized data statistics:");
    println!("Mean: {:8.3}", mean);
    println!("Variance: {:8.3}", var);

    // Show running statistics
    if let Some(running_mean) = batch_norm.running_mean() {
        println!("Running mean: {:8.3}", running_mean);
    }
    if let Some(running_var) = batch_norm.running_var() {
        println!("Running variance: {:8.3}", running_var);
    }

    // 2. Dropout Demo
    println!("\n2. Dropout Layer");
    println!("----------------");

    // Create a dropout layer with 50% dropout rate
    let mut dropout = Dropout::new(0.5).seed(42); // Fixed seed for reproducibility

    let input = array![
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0]
    ];

    println!("Input data:");
    println!("{:6.1}", input);

    // Forward pass in training mode (dropout active)
    let dropout_output = dropout.forward(&input, true)?;
    println!("\nAfter dropout (training mode):");
    println!("{:6.1}", dropout_output);

    // Forward pass in inference mode (no dropout)
    dropout.reset(); // Clear cached mask
    let inference_output = dropout.forward(&input, false)?;
    println!("\nAfter dropout (inference mode):");
    println!("{:6.1}", inference_output);

    // 3. Combined Layer Demo
    println!("\n3. Combined Layers Demo");
    println!("-----------------------");

    // Simulate a simple neural network layer with batch norm + activation + dropout
    let mut batch_norm_combined = BatchNorm1d::new(3);
    batch_norm_combined.initialize();
    let mut dropout_combined = Dropout::new(0.3).seed(123);

    let layer_input = array![
        [-2.0, 0.0, 2.0],
        [-1.0, 1.0, 3.0],
        [0.0, 2.0, 4.0],
        [1.0, 3.0, 5.0]
    ];

    println!("Layer input:");
    println!("{:6.1}", layer_input);

    // Step 1: Batch normalization
    let step1 = batch_norm_combined.forward(&layer_input, true)?;
    println!("\nAfter batch normalization:");
    println!("{:8.3}", step1);

    // Step 2: ReLU activation
    let step2 = Activation::Relu.apply(&step1);
    println!("\nAfter ReLU activation:");
    println!("{:8.3}", step2);

    // Step 3: Dropout
    let step3 = dropout_combined.forward(&step2, true)?;
    println!("\nAfter dropout:");
    println!("{:8.3}", step3);

    // 4. Backward Pass Demo
    println!("\n4. Backward Pass Demo");
    println!("---------------------");

    // Simulate gradients flowing backward
    let grad_output = Array2::ones((4, 3));
    println!("Gradient output (from next layer):");
    println!("{:6.1}", grad_output);

    // Backward through dropout
    let grad_step3 = dropout_combined.backward(&grad_output)?;
    println!("\nGradient after dropout backward:");
    println!("{:6.1}", grad_step3);

    // Backward through batch norm
    let grad_step1 = batch_norm_combined.backward(&grad_step3)?;
    println!("\nGradient after batch norm backward:");
    println!("{:8.3}", grad_step1);

    // Show parameter gradients
    if let Some(weight_grad) = batch_norm_combined.weight_grad() {
        println!("\nBatch norm weight gradients:");
        println!("{:8.3}", weight_grad);
    }
    if let Some(bias_grad) = batch_norm_combined.bias_grad() {
        println!("Batch norm bias gradients:");
        println!("{:8.3}", bias_grad);
    }

    // 5. Layer Configuration Demo
    println!("\n5. Layer Configuration Demo");
    println!("---------------------------");

    // Custom batch normalization configuration
    let custom_bn = BatchNorm1d::new(2)
        .momentum(0.01) // Slower moving average
        .epsilon(1e-3) // Larger epsilon for stability
        .affine(false) // No learnable parameters
        .track_running_stats(true);

    println!("Custom batch norm configuration:");
    println!("- Momentum: 0.01 (slower adaptation)");
    println!("- Epsilon: 1e-3 (more stable)");
    println!("- Affine: false (no learnable params)");
    println!("- Track running stats: true");

    // Custom dropout configuration
    let custom_dropout = Dropout::new(0.2).seed(999);

    println!("\nCustom dropout configuration:");
    println!("- Rate: 0.2 (light regularization)");
    println!("- Seed: 999 (reproducible)");

    println!("\n✅ Layer demonstrations completed successfully!");
    println!("\nKey Benefits:");
    println!("• Batch Normalization: Stabilizes training, reduces internal covariate shift");
    println!("• Dropout: Prevents overfitting, improves generalization");
    println!("• Modular design: Easy to combine and customize layers");
    println!("• Type safety: Compile-time validation of layer configurations");
    println!("• Performance: Efficient implementations with minimal allocations");

    Ok(())
}
