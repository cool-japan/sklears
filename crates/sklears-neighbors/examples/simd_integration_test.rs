//! Test SIMD integration in KNN algorithms

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::traits::{Fit, Predict};
use sklears_neighbors::{Distance, KNeighborsClassifier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing KNN with SIMD-optimized distance functions");

    // Create simple test data
    let x_train = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 1.0, // Class 0
            1.5, 1.5, // Class 0
            2.0, 2.0, // Class 0
            3.0, 3.0, // Class 1
            3.5, 3.5, // Class 1
            4.0, 4.0, // Class 1
        ],
    )?;
    let y_train = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

    // Test data
    let x_test = Array2::from_shape_vec(
        (3, 2),
        vec![
            1.2, 1.2, // Should be class 0
            3.8, 3.8, // Should be class 1
            2.5, 2.5, // Should be close to both classes
        ],
    )?;

    // Test Euclidean distance with SIMD optimization
    println!("Testing Euclidean distance...");
    let classifier = KNeighborsClassifier::new(3).with_metric(Distance::Euclidean);
    let fitted = classifier.fit(&x_train, &y_train)?;
    let predictions = fitted.predict(&x_test)?;
    println!("Euclidean predictions: {:?}", predictions);

    // Test Manhattan distance with SIMD optimization
    println!("Testing Manhattan distance...");
    let classifier = KNeighborsClassifier::new(3).with_metric(Distance::Manhattan);
    let fitted = classifier.fit(&x_train, &y_train)?;
    let predictions = fitted.predict(&x_test)?;
    println!("Manhattan predictions: {:?}", predictions);

    // Test Cosine distance with SIMD optimization
    println!("Testing Cosine distance...");
    let classifier = KNeighborsClassifier::new(3).with_metric(Distance::Cosine);
    let fitted = classifier.fit(&x_train, &y_train)?;
    let predictions = fitted.predict(&x_test)?;
    println!("Cosine predictions: {:?}", predictions);

    // Test individual distance functions
    println!("\nTesting individual distance functions:");
    let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

    let euclidean = sklears_neighbors::distance::euclidean_distance(&a.view(), &b.view());
    let manhattan = sklears_neighbors::distance::manhattan_distance(&a.view(), &b.view());
    let cosine = sklears_neighbors::distance::cosine_distance(&a.view(), &b.view());

    println!("Euclidean distance: {:.6}", euclidean);
    println!("Manhattan distance: {:.6}", manhattan);
    println!("Cosine distance: {:.6}", cosine);

    println!("\nSIMD integration test completed successfully!");
    Ok(())
}
