//! Simple manual test for kernel approximation implementations

use crate::nystroem::*;
use crate::rbf_sampler::*;
use scirs2_core::ndarray::array;
use sklears_core::traits::{Fit, Transform};

pub fn test_implementations() {
    println!("Testing kernel approximation implementations...");

    // Test data
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

    // Test RBF Sampler
    println!("Testing RBF Sampler...");
    match test_rbf_sampler(&x) {
        Ok(_) => println!("✓ RBF Sampler works"),
        Err(e) => println!("✗ RBF Sampler failed: {}", e),
    }

    // Test Laplacian Sampler
    println!("Testing Laplacian Sampler...");
    match test_laplacian_sampler(&x) {
        Ok(_) => println!("✓ Laplacian Sampler works"),
        Err(e) => println!("✗ Laplacian Sampler failed: {}", e),
    }

    // Test Polynomial Sampler
    println!("Testing Polynomial Sampler...");
    match test_polynomial_sampler(&x) {
        Ok(_) => println!("✓ Polynomial Sampler works"),
        Err(e) => println!("✗ Polynomial Sampler failed: {}", e),
    }

    // Test Arc-cosine Sampler
    println!("Testing Arc-cosine Sampler...");
    match test_arc_cosine_sampler(&x) {
        Ok(_) => println!("✓ Arc-cosine Sampler works"),
        Err(e) => println!("✗ Arc-cosine Sampler failed: {}", e),
    }

    // Test Nyström Method
    println!("Testing Nyström Method...");
    match test_nystroem(&x) {
        Ok(_) => println!("✓ Nyström Method works"),
        Err(e) => println!("✗ Nyström Method failed: {}", e),
    }

    println!("Testing complete!");
}

fn test_rbf_sampler(
    x: &scirs2_core::ndarray::Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let rbf = RBFSampler::new(10).gamma(0.1);
    let fitted = rbf.fit(x, &())?;
    let result = fitted.transform(x)?;
    assert_eq!(result.shape(), &[4, 10]);
    Ok(())
}

fn test_laplacian_sampler(
    x: &scirs2_core::ndarray::Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let laplacian = LaplacianSampler::new(10).gamma(0.1);
    let fitted = laplacian.fit(x, &())?;
    let result = fitted.transform(x)?;
    assert_eq!(result.shape(), &[4, 10]);
    Ok(())
}

fn test_polynomial_sampler(
    x: &scirs2_core::ndarray::Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let poly = PolynomialSampler::new(10).degree(2).gamma(1.0).coef0(1.0);
    let fitted = poly.fit(x, &())?;
    let result = fitted.transform(x)?;
    assert_eq!(result.shape(), &[4, 10]);
    Ok(())
}

fn test_arc_cosine_sampler(
    x: &scirs2_core::ndarray::Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let arc_cosine = ArcCosineSampler::new(10).degree(1);
    let fitted = arc_cosine.fit(x, &())?;
    let result = fitted.transform(x)?;
    assert_eq!(result.shape(), &[4, 10]);
    Ok(())
}

fn test_nystroem(x: &scirs2_core::ndarray::Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // Test with Linear kernel and Random sampling
    let nystroem = Nystroem::new(Kernel::Linear, 3).sampling_strategy(SamplingStrategy::Random);
    let fitted = nystroem.fit(x, &())?;
    let result = fitted.transform(x)?;
    assert_eq!(result.nrows(), 4);

    // Test with RBF kernel and Leverage Score sampling
    let nystroem_rbf = Nystroem::new(Kernel::Rbf { gamma: 0.1 }, 3)
        .sampling_strategy(SamplingStrategy::LeverageScore);
    let fitted_rbf = nystroem_rbf.fit(x, &())?;
    let result_rbf = fitted_rbf.transform(x)?;
    assert_eq!(result_rbf.nrows(), 4);

    Ok(())
}
