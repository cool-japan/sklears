//! Demo script for SVM functionality

use scirs2_core::ndarray::array;
use sklears_core::traits::{Fit, Predict};
use sklears_svm::{Kernel, KernelType, SVC, SVR};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "=== sklears SVM Demo ===
"
    );

    // SVC Demo with Linear Kernel
    println!("1. SVC with Linear Kernel");
    let x_svc = array![
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [-1.0, -1.0],
        [-2.0, -2.0],
        [-3.0, -3.0],
    ];
    let y_svc = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

    let svc = SVC::new().linear().c(1.0).fit(&x_svc, &y_svc)?;

    println!("  - Fitted SVC successfully");
    println!(
        "  - Number of support vectors: {}",
        svc.support_vectors().nrows()
    );
    println!("  - Classes: {:?}", svc.classes().to_vec());
    println!("  - Intercept: {:.4}", svc.intercept());

    // Test prediction
    let x_test = array![
        [4.0, 4.0],   // Should be class 1
        [-4.0, -4.0], // Should be class 0
    ];
    let predictions = svc.predict(&x_test)?;
    println!("  - Predictions for test data: {:?}", predictions.to_vec());

    // SVC Demo with RBF Kernel
    println!(
        "
2. SVC with RBF Kernel"
    );
    let svc_rbf = SVC::new().rbf(Some(1.0)).c(1.0).fit(&x_svc, &y_svc)?;

    println!("  - Fitted RBF SVC successfully");
    println!(
        "  - Number of support vectors: {}",
        svc_rbf.support_vectors().nrows()
    );

    let predictions_rbf = svc_rbf.predict(&x_test)?;
    println!("  - RBF Predictions: {:?}", predictions_rbf.to_vec());

    // SVR Demo
    println!(
        "
3. SVR (Support Vector Regression)"
    );
    let x_svr = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
    let y_svr = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Linear relationship: y = 2*x

    let svr = SVR::new()
        .linear()
        .c(1.0)
        .epsilon(0.1)
        .fit(&x_svr, &y_svr)?;

    println!("  - Fitted SVR successfully");
    println!(
        "  - Number of support vectors: {}",
        svr.support_vectors().nrows()
    );
    println!("  - Intercept: {:.4}", svr.intercept());

    // Test regression prediction
    let x_test_svr = array![[3.5]];
    let predictions_svr = svr.predict(&x_test_svr)?;
    println!("  - SVR prediction for x=3.5: {:.4}", predictions_svr[0]);

    // Kernel Demo
    println!(
        "
4. Kernel Functions Demo"
    );
    let kernel_linear = KernelType::Linear;
    let kernel_rbf = KernelType::Rbf { gamma: 1.0 };
    let kernel_poly = KernelType::Polynomial {
        gamma: 1.0,
        degree: 2.0,
        coef0: 1.0,
    };

    let x1 = array![1.0, 2.0];
    let x2 = array![3.0, 4.0];

    println!(
        "  - Linear kernel K(x1, x2): {:.4}",
        kernel_linear.compute(x1.view(), x2.view())
    );
    println!(
        "  - RBF kernel K(x1, x2): {:.4}",
        kernel_rbf.compute(x1.view(), x2.view())
    );
    println!(
        "  - Polynomial kernel K(x1, x2): {:.4}",
        kernel_poly.compute(x1.view(), x2.view())
    );

    println!(
        "
=== Demo completed successfully! ==="
    );

    Ok(())
}
