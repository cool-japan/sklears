//! Example: Gaussian Mixture Models (GMM)
//!
//! This example demonstrates:
//! - GMM for probabilistic clustering
//! - Different covariance types
//! - Soft clustering with probability predictions
//! - Model selection using information criteria

use scirs2_core::ndarray::{array, Array1, Array2};
use sklears::clustering::{CovarianceType, GaussianMixture, PredictProba, WeightInit};
use sklears::prelude::*;

fn main() -> Result<()> {
    println!("=== Gaussian Mixture Model Example ===\n");

    // 1. Basic GMM clustering
    println!("1. Basic GMM Clustering");
    println!("----------------------");

    // Create data from actual Gaussian mixture
    let data = create_gaussian_mixture_data();

    let model: GaussianMixture = GaussianMixture::new()
        .n_components(3)
        .covariance_type(CovarianceType::Full)
        .random_state(42)
        .fit(&data.view(), &Array1::ones(data.nrows()).view())?;

    println!("Model converged: {}", model.converged());
    println!("Number of iterations: {}", model.n_iter());
    println!("Log-likelihood lower bound: {:.2}", model.lower_bound());

    // Display component parameters
    println!("\nComponent parameters:");
    let weights = model.weights()?;
    let means = model.means()?;

    for i in 0..3 {
        println!(
            "  Component {}: weight={:.3}, mean=[{:.2}, {:.2}]",
            i,
            weights[i],
            means[[i, 0]],
            means[[i, 1]]
        );
    }

    // 2. Soft clustering with probabilities
    println!("\n2. Soft Clustering (Probability Predictions)");
    println!("-------------------------------------------");

    // Predict probabilities for some points
    let test_points = array![
        [0.0, 0.0], // Near component 1
        [5.0, 0.0], // Near component 2
        [2.5, 4.0], // Near component 3
        [2.5, 2.0], // Between components
    ];

    let probabilities = model.predict_proba(&test_points.view())?;
    let hard_labels = model.predict(&test_points.view())?;

    println!("Point         Component Probabilities    Hard Label");
    println!("----------    -----------------------   ----------");
    for i in 0..test_points.nrows() {
        print!(
            "[{:3.1},{:3.1}]  ",
            test_points[[i, 0]],
            test_points[[i, 1]]
        );
        print!("[");
        for j in 0..3 {
            print!("{:.3}", probabilities[[i, j]]);
            if j < 2 {
                print!(", ");
            }
        }
        println!("]     {}", hard_labels[i]);
    }

    // 3. Different covariance types
    println!("\n3. Comparing Covariance Types");
    println!("-----------------------------");

    let cov_types = vec![
        (CovarianceType::Full, "Full"),
        (CovarianceType::Diagonal, "Diagonal"),
        (CovarianceType::Tied, "Tied"),
        (CovarianceType::Spherical, "Spherical"),
    ];

    println!("Type        Converged  Iterations  Log-likelihood");
    println!("----------  ---------  ----------  --------------");

    for (cov_type, name) in cov_types {
        let model: GaussianMixture = GaussianMixture::new()
            .n_components(3)
            .covariance_type(cov_type)
            .random_state(42)
            .fit(&data.view(), &Array1::ones(data.nrows()).view())?;

        println!(
            "{:10}  {:9}  {:10}  {:14.2}",
            name,
            model.converged(),
            model.n_iter(),
            model.lower_bound()
        );
    }

    // 4. Model selection with different numbers of components
    println!("\n4. Model Selection (Choosing Number of Components)");
    println!("-------------------------------------------------");

    let n_components_range = vec![1, 2, 3, 4, 5, 6];

    println!("Components  Log-likelihood  BIC (lower is better)");
    println!("----------  --------------  --------------------");

    for &n_comp in &n_components_range {
        let model: GaussianMixture = GaussianMixture::new()
            .n_components(n_comp)
            .covariance_type(CovarianceType::Full)
            .random_state(42)
            .fit(&data.view(), &Array1::ones(data.nrows()).view())?;

        // Calculate BIC (Bayesian Information Criterion)
        // BIC = -2 * log_likelihood + k * log(n)
        // where k is the number of parameters and n is the number of samples
        let n_features = 2;
        let n_params = n_comp * (1 + n_features + n_features * (n_features + 1) / 2) - 1;
        let bic = -2.0 * model.lower_bound() + (n_params as f64) * (data.nrows() as f64).ln();

        println!("{:10}  {:14.2}  {:20.2}", n_comp, model.lower_bound(), bic);
    }

    println!("\nNote: The optimal number of components minimizes BIC");

    // 5. Initialization methods
    println!("\n5. Comparing Initialization Methods");
    println!("-----------------------------------");

    let init_methods = vec![
        (WeightInit::KMeans, "K-Means"),
        (WeightInit::Random, "Random"),
    ];

    for (init_method, name) in init_methods {
        let model: GaussianMixture = GaussianMixture::new()
            .n_components(3)
            .init_params(init_method)
            .n_init(5) // Multiple initializations
            .random_state(42)
            .fit(&data.view(), &Array1::ones(data.nrows()).view())?;

        println!(
            "{} initialization: {} iterations, log-likelihood: {:.2}",
            name,
            model.n_iter(),
            model.lower_bound()
        );
    }

    // 6. Detecting overlapping clusters
    println!("\n6. Overlapping Clusters Detection");
    println!("---------------------------------");

    // Create data with overlapping clusters
    let overlap_data = create_overlapping_clusters();

    let model: GaussianMixture = GaussianMixture::new()
        .n_components(2)
        .covariance_type(CovarianceType::Full)
        .fit(
            &overlap_data.view(),
            &Array1::ones(overlap_data.nrows()).view(),
        )?;

    // Find points with high uncertainty (similar probabilities)
    let all_probs = model.predict_proba(&overlap_data.view())?;
    let mut uncertain_points = Vec::new();

    for i in 0..overlap_data.nrows() {
        let p1 = all_probs[[i, 0]];
        let p2 = all_probs[[i, 1]];
        let uncertainty = 1.0 - (p1 - p2).abs();

        if uncertainty > 0.8 {
            // High uncertainty
            uncertain_points.push((i, uncertainty, p1, p2));
        }
    }

    println!(
        "Found {} points in overlap region (uncertainty > 0.8)",
        uncertain_points.len()
    );
    println!("\nExample uncertain points:");
    for (idx, (i, unc, p1, p2)) in uncertain_points.iter().take(5).enumerate() {
        println!(
            "  Point {}: uncertainty={:.3}, P(C1)={:.3}, P(C2)={:.3}",
            idx, unc, p1, p2
        );
    }

    Ok(())
}

/// Create data from a true Gaussian mixture
fn create_gaussian_mixture_data() -> Array2<f64> {
    use scirs2_core::essentials::Normal;
    use scirs2_core::random::{rngs::StdRng, Distribution, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::new();

    // Component 1: mean=[0,0], cov=[[0.5,0],[0,0.5]]
    let normal1_x = Normal::new(0.0, 0.7).unwrap();
    let normal1_y = Normal::new(0.0, 0.7).unwrap();
    for _ in 0..100 {
        data.push([normal1_x.sample(&mut rng), normal1_y.sample(&mut rng)]);
    }

    // Component 2: mean=[5,0], cov=[[1,0.5],[0.5,1]]
    let normal2_x = Normal::new(5.0, 1.0).unwrap();
    let normal2_y = Normal::new(0.0, 1.0).unwrap();
    for _ in 0..150 {
        let x = normal2_x.sample(&mut rng);
        let y = normal2_y.sample(&mut rng) + 0.3 * (x - 5.0); // Add correlation
        data.push([x, y]);
    }

    // Component 3: mean=[2.5,4], cov=[[0.8,0],[0,0.3]]
    let normal3_x = Normal::new(2.5, 0.9).unwrap();
    let normal3_y = Normal::new(4.0, 0.55).unwrap();
    for _ in 0..75 {
        data.push([normal3_x.sample(&mut rng), normal3_y.sample(&mut rng)]);
    }

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}

/// Create data with overlapping clusters
fn create_overlapping_clusters() -> Array2<f64> {
    use scirs2_core::essentials::Normal;
    use scirs2_core::random::{rngs::StdRng, Distribution, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::new();

    // Cluster 1: centered at (0, 0)
    let normal1 = Normal::new(0.0, 1.5).unwrap();
    for _ in 0..200 {
        data.push([normal1.sample(&mut rng), normal1.sample(&mut rng)]);
    }

    // Cluster 2: centered at (2, 0) - overlapping with cluster 1
    let normal2_x = Normal::new(2.0, 1.5).unwrap();
    let normal2_y = Normal::new(0.0, 1.5).unwrap();
    for _ in 0..200 {
        data.push([normal2_x.sample(&mut rng), normal2_y.sample(&mut rng)]);
    }

    Array2::from_shape_vec((data.len(), 2), data.into_iter().flatten().collect()).unwrap()
}
