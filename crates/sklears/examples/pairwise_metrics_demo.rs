//! Example demonstrating pairwise distance and similarity metrics

use scirs2_core::ndarray::array;
use sklears_metrics::pairwise::{
    euclidean_distances, nan_euclidean_distances, pairwise_distances, pairwise_distances_argmin,
    pairwise_distances_argmin_min, pairwise_kernels, DistanceMetric, KernelFunction,
};

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pairwise Metrics Demo ===\n");

    // Sample data
    let X = array![[0., 0.], [1., 1.], [2., 2.], [3., 3.],];

    let Y = array![[0.5, 0.5], [2.5, 2.5],];

    // 1. Euclidean distances
    println!("1. Euclidean Distances");
    println!("X:\n{}", X);
    println!("\nY:\n{}", Y);

    let distances = euclidean_distances(&X.view(), Some(&Y.view()))?;
    println!("\nEuclidean distances from X to Y:\n{:.3}", distances);

    // Self distances
    let self_distances = euclidean_distances(&X.view(), None)?;
    println!("\nEuclidean distances within X:\n{:.3}", self_distances);

    // 2. NaN-aware Euclidean distances
    println!("\n2. NaN-aware Euclidean Distances");
    let X_with_nan = array![[1., 2., f64::NAN], [4., f64::NAN, 6.], [7., 8., 9.],];
    let Y_ref = array![[1., 2., 3.]];

    println!("X with NaN:\n{}", X_with_nan);
    println!("\nY:\n{}", Y_ref);

    let nan_distances = nan_euclidean_distances(&X_with_nan.view(), Some(&Y_ref.view()))?;
    println!("\nNaN-aware distances:\n{:.3}", nan_distances);

    // 3. Different distance metrics
    println!("\n3. Different Distance Metrics");
    let metrics = vec![
        ("Euclidean", DistanceMetric::Euclidean),
        ("Manhattan", DistanceMetric::Manhattan),
        ("Chebyshev", DistanceMetric::Chebyshev),
        ("Minkowski(p=3)", DistanceMetric::Minkowski(3.0)),
        ("Cosine", DistanceMetric::Cosine),
    ];

    for (name, metric) in metrics {
        let dists = pairwise_distances(&X.view(), Some(&Y.view()), metric)?;
        println!("\n{} distances:\n{:.3}", name, dists);
    }

    // 4. Finding nearest neighbors
    println!("\n4. Finding Nearest Neighbors");
    let argmin = pairwise_distances_argmin(&X.view(), Some(&Y.view()), DistanceMetric::Euclidean)?;
    println!("Nearest neighbor indices: {}", argmin);

    let (argmin, min_dists) =
        pairwise_distances_argmin_min(&X.view(), Some(&Y.view()), DistanceMetric::Euclidean)?;
    println!("Nearest neighbor indices: {}", argmin);
    println!("Distances to nearest neighbors: {:.3}", min_dists);

    // 5. Kernel functions
    println!("\n5. Kernel Functions");
    let X_kernel = array![[1., 0.], [0., 1.], [1., 1.], [-1., -1.],];

    println!("X for kernels:\n{}", X_kernel);

    // Linear kernel
    let linear_kernel = pairwise_kernels(&X_kernel.view(), None, KernelFunction::Linear)?;
    println!("\nLinear kernel:\n{:.3}", linear_kernel);

    // RBF kernel
    let rbf_kernel = pairwise_kernels(&X_kernel.view(), None, KernelFunction::RBF { gamma: 0.5 })?;
    println!("\nRBF kernel (gamma=0.5):\n{:.3}", rbf_kernel);

    // Polynomial kernel
    let poly_kernel = pairwise_kernels(
        &X_kernel.view(),
        None,
        KernelFunction::Polynomial {
            degree: 2.0,
            gamma: 1.0,
            coef0: 1.0,
        },
    )?;
    println!(
        "\nPolynomial kernel (degree=2, gamma=1, coef0=1):\n{:.3}",
        poly_kernel
    );

    // Sigmoid kernel
    let sigmoid_kernel = pairwise_kernels(
        &X_kernel.view(),
        None,
        KernelFunction::Sigmoid {
            gamma: 0.1,
            coef0: 0.0,
        },
    )?;
    println!(
        "\nSigmoid kernel (gamma=0.1, coef0=0):\n{:.3}",
        sigmoid_kernel
    );

    // Cosine kernel
    let cosine_kernel = pairwise_kernels(&X_kernel.view(), None, KernelFunction::Cosine)?;
    println!("\nCosine kernel:\n{:.3}", cosine_kernel);

    Ok(())
}
