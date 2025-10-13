//! Demonstration of sparse matrix support in sklears-linear
//!
//! This example shows how to use the sparse matrix functionality for
//! memory-efficient linear regression on high-dimensional sparse data.

#[cfg(feature = "sparse")]
use sklears_linear::{
    SparseElasticNet, SparseLasso, SparseLinearRegression, SparseMatrix, SparseMatrixCSR,
};

#[cfg(feature = "sparse")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Sparse Matrix Linear Regression Demo");
    println!("=========================================\n");

    // 1. Create a sparse dataset
    println!("1. Creating synthetic sparse dataset...");
    let (x_dense, y, true_coeffs) = create_sparse_dataset();
    println!(
        "   Dataset: {} samples × {} features",
        x_dense.nrows(),
        x_dense.ncols()
    );

    // Analyze sparsity
    // TODO: Re-enable when analyze_sparsity function is implemented
    // let analysis = sklears_linear::sparse::utils::analyze_sparsity(&x_dense, 1e-10);
    // println!(
    //     "   Sparsity: {:.1}% non-zero elements",
    //     analysis.sparsity_ratio * 100.0
    // );
    // println!(
    //     "   Memory savings with sparse: {:.1}%",
    //     analysis.memory_savings_ratio(x_dense.nrows(), x_dense.ncols()) * 100.0
    // );

    // 2. Convert to sparse format
    println!("\n2. Converting to sparse format...");
    let x_sparse = SparseMatrixCSR::from_dense(&x_dense, 1e-10);
    println!(
        "   Sparse matrix: {} non-zeros out of {} elements",
        x_sparse.nnz(),
        x_dense.nrows() * x_dense.ncols()
    );

    // 3. Sparse Linear Regression
    println!("\n3. Training Sparse Linear Regression...");
    let linear_model = SparseLinearRegression::new()
        .fit_intercept(true)
        .auto_sparse_conversion(true)
        .fit(&x_sparse, &y)?;

    println!(
        "   Model fitted using sparse algorithms: {}",
        linear_model.is_sparse_fitted()
    );
    print_coefficients(
        "Linear Regression",
        linear_model.coefficients(),
        &true_coeffs,
    );

    // 4. Sparse Lasso Regression
    println!("\n4. Training Sparse Lasso Regression...");
    let lasso_model = SparseLasso::new(0.1)
        .fit_intercept(true)
        .max_iter(1000)
        .fit(&x_sparse, &y)?;

    println!(
        "   Non-zero coefficients: {}/{}",
        lasso_model.n_nonzero_coefficients(),
        lasso_model.n_features()
    );
    println!(
        "   Coefficient sparsity: {:.1}%",
        lasso_model.coefficient_sparsity() * 100.0
    );
    print_coefficients("Lasso", lasso_model.coefficients(), &true_coeffs);

    // 5. Sparse ElasticNet Regression
    println!("\n5. Training Sparse ElasticNet Regression...");
    let elastic_model = SparseElasticNet::new(0.1, 0.5)
        .fit_intercept(true)
        .max_iter(1000)
        .fit(&x_sparse, &y)?;

    println!(
        "   Non-zero coefficients: {}/{}",
        elastic_model.n_nonzero_coefficients(),
        elastic_model.n_features()
    );
    println!(
        "   Coefficient sparsity: {:.1}%",
        elastic_model.coefficient_sparsity() * 100.0
    );
    print_coefficients("ElasticNet", elastic_model.coefficients(), &true_coeffs);

    // 6. Performance comparison
    println!("\n6. Performance Comparison");
    println!("=========================");

    // Create test data
    let x_test_sparse = SparseMatrixCSR::from_dense(&x_dense, 1e-10);

    // Test predictions
    let linear_pred = linear_model.predict(&x_test_sparse)?;
    let lasso_pred = lasso_model.predict(&x_test_sparse)?;
    let elastic_pred = elastic_model.predict(&x_test_sparse)?;

    // Calculate MSE
    let linear_mse = mean_squared_error(&y, &linear_pred);
    let lasso_mse = mean_squared_error(&y, &lasso_pred);
    let elastic_mse = mean_squared_error(&y, &elastic_pred);

    println!("   Linear Regression MSE: {:.6}", linear_mse);
    println!("   Lasso MSE:             {:.6}", lasso_mse);
    println!("   ElasticNet MSE:        {:.6}", elastic_mse);

    // 7. Auto-conversion demonstration
    println!("\n7. Auto-conversion Demonstration");
    println!("=================================");

    // Dense data that should use dense algorithms
    let x_dense_small = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let y_small = array![1.0, 2.0, 3.0];

    let auto_model = SparseLinearRegression::new()
        .auto_sparse_conversion(true)
        .min_sparsity_ratio(0.5) // Only use sparse if <50% elements are non-zero
        .fit(&x_dense_small, &y_small)?;

    println!(
        "   Dense data (100% non-zero) → Uses sparse algorithms: {}",
        auto_model.is_sparse_fitted()
    );

    // Very sparse data that should use sparse algorithms
    let mut x_very_sparse = Array2::zeros((20, 50));
    x_very_sparse[[0, 0]] = 1.0;
    x_very_sparse[[1, 10]] = 1.0;
    x_very_sparse[[2, 20]] = 1.0;
    let y_sparse = Array1::ones(20);

    let sparse_auto_model = SparseLinearRegression::new()
        .auto_sparse_conversion(true)
        .min_sparsity_ratio(0.1) // Use sparse if <10% elements are non-zero
        .fit(&x_very_sparse, &y_sparse)?;

    // TODO: Re-enable when analyze_sparsity function is implemented
    // let sparse_analysis = sklears_linear::sparse::utils::analyze_sparsity(&x_very_sparse, 1e-10);
    println!(
        "   Sparse data (very sparse) → Uses sparse algorithms: {}",
        sparse_auto_model.is_sparse_fitted()
    );

    println!("\n✅ Sparse regression demo completed successfully!");
    Ok(())
}

#[cfg(not(feature = "sparse"))]
fn main() {
    println!("❌ This example requires the 'sparse' feature to be enabled.");
    println!("Run with: cargo run --example sparse_regression_demo --features sparse");
}

/// Create a synthetic sparse dataset for demonstration
#[cfg(feature = "sparse")]
fn create_sparse_dataset() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_samples = 100;
    let n_features = 50;
    let n_informative = 5;

    // Create a mostly sparse feature matrix
    let mut x = Array2::zeros((n_samples, n_features));
    let mut true_coeffs = Array1::zeros(n_features);

    // Set up informative features with sparse patterns
    for i in 0..n_samples {
        // Only a few features are non-zero for each sample
        let feature_indices = [
            i % n_informative,
            (i * 2) % n_informative,
            (i * 3) % n_informative,
        ];
        for &j in &feature_indices {
            x[[i, j]] = (i as f64 * 0.1 + j as f64 * 0.05).sin();
            x[[i, j + n_informative]] = (i as f64 * 0.05).cos();
        }
    }

    // Set true coefficients (sparse)
    true_coeffs[0] = 2.0;
    true_coeffs[1] = -1.5;
    true_coeffs[2] = 1.0;
    true_coeffs[5] = 0.5;
    true_coeffs[6] = -0.8;

    // Generate target variable
    let y = x.dot(&true_coeffs) + Array1::from_elem(n_samples, 0.1); // Small noise

    (x, y, true_coeffs)
}

/// Print coefficient comparison
#[cfg(feature = "sparse")]
fn print_coefficients(model_name: &str, fitted_coeffs: &Array1<f64>, true_coeffs: &Array1<f64>) {
    println!("   {} coefficients (showing first 10):", model_name);
    println!("     True:   {:?}", &true_coeffs.slice(s![0..10]).to_vec());
    println!(
        "     Fitted: {:?}",
        &fitted_coeffs.slice(s![0..10]).to_vec()
    );
}

/// Calculate mean squared error
#[cfg(feature = "sparse")]
fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.dot(&diff) / y_true.len() as f64
}
