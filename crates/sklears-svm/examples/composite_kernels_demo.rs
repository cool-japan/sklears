//! Demonstration of composite kernel functionality
//!
//! NOTE: This example demonstrates composite kernels, but SumKernel, ProductKernel,
//! and WeightedSumKernel are not yet implemented in this crate.
//! This example shows basic kernel usage instead.

use scirs2_core::ndarray::array;
use sklears_svm::kernels::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "=== Composite Kernels Demo ===
"
    );

    // Create sample data points
    let x1 = array![1.0, 2.0, 3.0];
    let x2 = array![4.0, 5.0, 6.0];
    let x3 = array![1.0, 2.0, 3.0]; // Same as x1 for testing self-similarity

    println!("Sample data:");
    println!("x1 = {:?}", x1);
    println!("x2 = {:?}", x2);
    println!("x3 = {:?} (same as x1)", x3);
    println!();

    // Create individual kernels
    let linear_kernel = LinearKernel;
    let rbf_kernel = RbfKernel::new(0.5);
    let cosine_kernel = CosineKernel;

    // Test individual kernels first
    println!("=== Individual Kernel Results ===");
    let linear_val = linear_kernel.compute(x1.view(), x2.view());
    let rbf_val = rbf_kernel.compute(x1.view(), x2.view());
    let cosine_val = cosine_kernel.compute(x1.view(), x2.view());

    println!("Linear K(x1, x2) = {:.6}", linear_val);
    println!("RBF K(x1, x2) = {:.6}", rbf_val);
    println!("Cosine K(x1, x2) = {:.6}", cosine_val);
    println!();

    // Test self-similarity (should be higher than cross-similarity)
    println!("=== Self-similarity Tests ===");
    let linear_self = linear_kernel.compute(x1.view(), x3.view());
    let rbf_self = rbf_kernel.compute(x1.view(), x3.view());
    let cosine_self = cosine_kernel.compute(x1.view(), x3.view());

    println!("Linear K(x1, x1) = {:.6}", linear_self);
    println!("RBF K(x1, x1) = {:.6}", rbf_self);
    println!("Cosine K(x1, x1) = {:.6}", cosine_self);
    println!();

    // Show kernel matrix computation for a small dataset
    println!("=== Kernel Matrix Example ===");
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let kernel_for_matrix = LinearKernel;
    let kernel_matrix = kernel_for_matrix.compute_matrix(&data, &data);
    println!("Kernel matrix for linear kernel:");
    for i in 0..3 {
        print!("[");
        for j in 0..3 {
            print!("{:8.4}", kernel_matrix[[i, j]]);
            if j < 2 {
                print!(", ");
            }
        }
        println!("]");
    }
    println!();

    println!("✅ Kernel demo completed successfully!");
    println!();
    println!("Key features demonstrated:");
    println!("• Linear kernel: Simple dot product similarity");
    println!("• RBF kernel: Gaussian similarity with gamma parameter");
    println!("• Cosine kernel: Angular similarity between vectors");
    println!("• Self-similarity: Kernels produce higher values for identical inputs");
    println!("• Matrix operations: Efficient kernel matrix computation");
    println!();
    println!(
        "NOTE: Composite kernels (Sum, Product, WeightedSum) are planned for future releases."
    );

    /* Composite kernels (not yet implemented):

    // Create composite kernels
    println!("=== Composite Kernel Results ===");

    // Sum kernel: K_sum(x1, x2) = K_linear(x1, x2) + K_rbf(x1, x2)
    let sum_kernel = SumKernel::new(Box::new(LinearKernel), Box::new(RbfKernel::new(0.5)));
    let sum_val = sum_kernel.compute(x1.view(), x2.view());
    println!(
        "Sum kernel K(x1, x2) = {:.6} (expected: {:.6})",
        sum_val,
        linear_val + rbf_val
    );

    // Product kernel: K_prod(x1, x2) = K_linear(x1, x2) * K_cosine(x1, x2)
    let product_kernel =
        ProductKernel::new(Box::new(LinearKernel), Box::new(CosineKernel));
    let product_val = product_kernel.compute(x1.view(), x2.view());
    println!(
        "Product kernel K(x1, x2) = {:.6} (expected: {:.6})",
        product_val,
        linear_val * cosine_val
    );

    // Weighted sum kernel: K_weighted(x1, x2) = 0.7 * K_linear(x1, x2) + 0.3 * K_rbf(x1, x2)
    let weighted_kernel = WeightedSumKernel::new(
        Box::new(LinearKernel),
        Box::new(RbfKernel::new(0.5)),
        0.7,
        0.3,
    );
    let weighted_val = weighted_kernel.compute(x1.view(), x2.view());
    println!(
        "Weighted sum kernel K(x1, x2) = {:.6} (expected: {:.6})",
        weighted_val,
        0.7 * linear_val + 0.3 * rbf_val
    );

    // Balanced weighted sum kernel (equal weights)
    let balanced_kernel =
        WeightedSumKernel::balanced(Box::new(LinearKernel), Box::new(RbfKernel::new(0.5)));
    let balanced_val = balanced_kernel.compute(x1.view(), x2.view());
    println!(
        "Balanced weighted sum K(x1, x2) = {:.6} (expected: {:.6})",
        balanced_val,
        0.5 * linear_val + 0.5 * rbf_val
    );
    println!();

    // Demonstrate nested composite kernels
    println!("=== Nested Composite Kernels ===");

    // Create a sum of (linear + rbf) and (cosine)
    let nested_sum = SumKernel::new(
        Box::new(SumKernel::new(
            Box::new(LinearKernel),
            Box::new(RbfKernel::new(0.5)),
        )),
        Box::new(CosineKernel),
    );
    let nested_val = nested_sum.compute(x1.view(), x2.view());
    println!(
        "Nested sum [(Linear + RBF) + Cosine] K(x1, x2) = {:.6}",
        nested_val
    );
    println!("Expected: {:.6}", linear_val + rbf_val + cosine_val);
    println!();
    */

    Ok(())
}
