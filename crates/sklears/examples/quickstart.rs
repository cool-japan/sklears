//! Quick start example for sklears
//!
//! This example demonstrates basic usage of linear regression
//! and clustering algorithms.

use sklears::dataset::{make_blobs, make_regression};
use sklears::prelude::*;

fn main() -> Result<()> {
    println!("sklears Quick Start Example\n");

    // Example 1: Linear Regression
    println!("=== Linear Regression Example ===");
    linear_regression_example()?;

    // Example 2: Clustering
    println!("\n=== K-Means Clustering Example ===");
    clustering_example()?;

    Ok(())
}

fn linear_regression_example() -> Result<()> {
    // Generate synthetic regression data
    let dataset = make_regression(100, 5, 0.1)?;
    let (x_train, y_train) = (&dataset.data, &dataset.target);

    // This would work once LinearRegression is implemented:
    /*
    let model = LinearRegression::new()
        .fit_intercept(true)
        .fit(x_train, y_train)?;

    // Make predictions
    let predictions = model.predict(x_train)?;

    // Calculate R² score
    let score = model.score(x_train, y_train)?;
    println!("R² score: {:.4}", score);
    */

    println!(
        "Generated regression dataset with {} samples and {} features",
        x_train.nrows(),
        x_train.ncols()
    );

    Ok(())
}

fn clustering_example() -> Result<()> {
    // Generate synthetic clustered data
    let dataset = make_blobs(150, 2, 3, 0.5)?;
    let x = &dataset.data;

    // This would work once KMeans is implemented:
    /*
    let kmeans = KMeans::new(3)
        .max_iter(300)
        .random_state(42);

    let fitted = kmeans.fit(x)?;
    let labels = fitted.labels();
    let centers = fitted.cluster_centers();

    println!("Found {} clusters", centers.nrows());
    println!("Inertia: {:.4}", fitted.inertia());

    // Predict cluster for new points
    let new_points = array![[0.0, 0.0], [5.0, 5.0]];
    let predictions = fitted.predict(&new_points)?;
    */

    println!(
        "Generated blob dataset with {} samples and {} features",
        x.nrows(),
        x.ncols()
    );

    Ok(())
}
