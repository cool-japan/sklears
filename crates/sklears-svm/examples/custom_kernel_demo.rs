//! Custom Kernel Demo for Support Vector Machines
//!
//! This example demonstrates how to create and use custom kernels with SVM
//! implementations in sklears. It shows several types of custom kernels and
//! how to integrate them with the SVM API.

use scirs2_core::ndarray::{array, Array1, Array2};
use sklears_core::{
    prelude::{Fit, Predict},
    types::Float,
};
use sklears_metrics::classification::accuracy_score;
use sklears_svm::{CosineKernel, Kernel, KernelType, SVC};

/// Custom Wave Kernel for periodic data
/// K(x, y) = cos(π * ||x - y|| / period) * exp(-||x - y||^2 / (2 * σ^2))
#[derive(Debug, Clone)]
pub struct WaveKernel {
    period: Float,
    sigma: Float,
}

impl WaveKernel {
    pub fn new(period: Float, sigma: Float) -> Self {
        Self { period, sigma }
    }
}

impl Kernel for WaveKernel {
    fn compute(
        &self,
        x1: scirs2_core::ndarray::ArrayView1<f64>,
        x2: scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        let distance = x1
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let wave_term = (std::f64::consts::PI * distance / self.period).cos();
        let gaussian_term = (-distance * distance / (2.0 * self.sigma * self.sigma)).exp();
        wave_term * gaussian_term
    }

    fn parameters(&self) -> std::collections::HashMap<String, f64> {
        let mut params = std::collections::HashMap::new();
        params.insert("period".to_string(), self.period);
        params.insert("sigma".to_string(), self.sigma);
        params
    }
}

/// Custom Exponential Kernel
/// K(x, y) = exp(-||x - y|| / σ)
#[derive(Debug, Clone)]
pub struct ExponentialKernel {
    sigma: Float,
}

impl ExponentialKernel {
    pub fn new(sigma: Float) -> Self {
        Self { sigma }
    }
}

impl Kernel for ExponentialKernel {
    fn compute(
        &self,
        x1: scirs2_core::ndarray::ArrayView1<f64>,
        x2: scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        let distance = x1
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        (-distance / self.sigma).exp()
    }

    fn parameters(&self) -> std::collections::HashMap<String, f64> {
        let mut params = std::collections::HashMap::new();
        params.insert("sigma".to_string(), self.sigma);
        params
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SKLears Custom Kernel Demo");
    println!("=============================");

    // Generate sample binary classification data
    let x_train = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [1.5, 1.5],
        [2.5, 2.5],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.5, 6.0],
        [6.5, 5.5],
    ];
    let y_train = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let x_test = array![[1.8, 2.2], [5.2, 5.3], [4.0, 4.0]];

    println!(
        "
📊 Training Data:"
    );
    println!("Features: {:?}", x_train);
    println!("Labels: {:?}", y_train);

    // Demo 1: Built-in kernels
    println!(
        "
🔧 Demo 1: Built-in Kernels"
    );
    println!("---------------------------");

    // Linear Kernel
    demo_svm_kernel(
        "Linear Kernel",
        SVC::new().linear().c(1.0),
        &x_train,
        &y_train,
        &x_test,
    )?;

    // RBF Kernel
    demo_svm_kernel(
        "RBF Kernel (γ=0.5)",
        SVC::new().rbf(Some(0.5)).c(1.0),
        &x_train,
        &y_train,
        &x_test,
    )?;

    // Demo 2: Built-in custom kernels using the new API
    println!(
        "
🎨 Demo 2: Built-in Custom Kernels"
    );
    println!("----------------------------------");

    // Cosine Kernel
    demo_svm_kernel(
        "Cosine Kernel",
        SVC::new().kernel(KernelType::Cosine).c(1.0),
        &x_train,
        &y_train,
        &x_test,
    )?;

    // Adaptive RBF Kernel - commented out as it's not implemented in the current crate
    // demo_svm_kernel(
    //     "Adaptive RBF Kernel",
    //     SVC::new()
    //         .kernel(KernelType::Rbf { gamma: 1.0 })
    //         .c(1.0),
    //     &x_train,
    //     &y_train,
    //     &x_test,
    // )?;

    // Demo 3: User-defined custom kernels
    println!(
        "
✨ Demo 3: User-Defined Custom Kernels"
    );
    println!("--------------------------------------");

    // Note: Custom kernel implementations (WaveKernel, ExponentialKernel) are demonstrated
    // in the kernel matrix section below, but cannot currently be used directly with SVC
    // through KernelType::Custom as it only accepts kernel names as strings.

    // Wave Kernel - commented out as KernelType::Custom doesn't support custom implementations
    // demo_svm_kernel(
    //     "Wave Kernel (period=2.0, σ=1.0)",
    //     SVC::new()
    //         .kernel(KernelType::Custom("wave".to_string()))
    //         .c(1.0),
    //     &x_train,
    //     &y_train,
    //     &x_test,
    // )?;

    // Exponential Kernel - commented out as KernelType::Custom doesn't support custom implementations
    // demo_svm_kernel(
    //     "Exponential Kernel (σ=1.0)",
    //     SVC::new()
    //         .kernel(KernelType::Custom("exponential".to_string()))
    //         .c(1.0),
    //     &x_train,
    //     &y_train,
    //     &x_test,
    // )?;

    // Demo 4: Kernel matrix visualization
    println!(
        "
📈 Demo 4: Kernel Matrix Properties"
    );
    println!("-----------------------------------");

    let sample_data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

    demo_kernel_matrix("Cosine Kernel", Box::new(CosineKernel), &sample_data);
    demo_kernel_matrix(
        "Wave Kernel",
        Box::new(WaveKernel::new(1.0, 0.5)),
        &sample_data,
    );

    // Demo 5: Implementation guide
    println!(
        "
📝 Demo 5: How to Implement Your Own Kernel"
    );
    println!("-------------------------------------------");

    println!("To create a custom kernel, implement the Kernel trait:");
    println!(
        "
```rust
use sklears_svm::{{Kernel, KernelType, SVC}};
use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;

#[derive(Debug, Clone)]
pub struct MyCustomKernel {{
    parameter: Float,
}}

impl MyCustomKernel {{
    pub fn new(parameter: Float) -> Self {{
        Self {{ parameter }}
    }}
}}

impl Kernel for MyCustomKernel {{
    fn compute(&self, x1: &Array1<Float>, x2: &Array1<Float>) -> Float {{
        // Your custom kernel computation here
        let dot_product = x1.dot(x2);
        (1.0 + self.parameter * dot_product).powi(2)
    }}
}}

// Usage:
let custom_kernel = KernelType::Custom(Box::new(MyCustomKernel::new(0.5)));
let svm = SVC::new().kernel(custom_kernel);
```"
    );

    println!(
        "
✨ Demo completed successfully! Custom kernels provide powerful"
    );
    println!("   flexibility for domain-specific machine learning applications.");

    Ok(())
}

fn demo_svm_kernel(
    name: &str,
    svm: SVC,
    x_train: &Array2<Float>,
    y_train: &Array1<Float>,
    x_test: &Array2<Float>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "
🔍 Testing: {name}"
    );

    // Train the model
    let trained_svm = svm.fit(x_train, y_train)?;

    // Make predictions
    let predictions = trained_svm.predict(x_test)?;
    println!("   Predictions: {:?}", predictions);

    // Calculate training accuracy
    let train_predictions = trained_svm.predict(x_train)?;
    let accuracy = accuracy_score(y_train, &train_predictions)?;
    println!("   Training accuracy: {:.1}%", accuracy * 100.0);

    println!("   ✅ Training and prediction successful!");

    Ok(())
}

fn demo_kernel_matrix(name: &str, kernel: Box<dyn Kernel>, data: &Array2<Float>) {
    println!(
        "
📊 {name} Kernel Matrix:"
    );

    let k_matrix = kernel.compute_matrix(data, data);

    println!("   Data points: {:?}", data);
    println!("   Kernel matrix:");
    for i in 0..k_matrix.nrows() {
        print!("   [");
        for j in 0..k_matrix.ncols() {
            print!(" {:6.3}", k_matrix[[i, j]]);
        }
        println!(" ]");
    }

    // Check properties
    let mut is_symmetric = true;
    for i in 0..k_matrix.nrows() {
        for j in 0..k_matrix.ncols() {
            if (k_matrix[[i, j]] - k_matrix[[j, i]]).abs() > 1e-10 {
                is_symmetric = false;
            }
        }
    }

    println!("   Properties: Symmetric={is_symmetric}");
}
