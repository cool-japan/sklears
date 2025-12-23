//! Classification and regression dataset generators
//!
//! This module provides specialized generators for classification and regression tasks,
//! including multilabel classification, sparse uncorrelated data, and polynomial regression.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, rng};
use scirs2_core::random::distributions::{Normal, StandardNormal};
use sklears_core::error::{Result, SklearsError};

/// Generate multilabel classification dataset
///
/// Creates a dataset where each sample can belong to multiple classes simultaneously.
/// Features are generated from a standard normal distribution, and labels are assigned
/// randomly according to the specified constraints.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `n_features`: Number of features per sample
/// - `n_classes`: Total number of possible classes
/// - `n_labels`: Number of labels assigned to each sample
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (features, multilabel_targets) where targets is a binary matrix
/// with shape (n_samples, n_classes)
pub fn make_multilabel_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_labels: usize,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array2<i32>)> {
    if n_samples == 0 || n_features == 0 || n_classes == 0 || n_labels == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, n_features, n_classes, and n_labels must be positive".to_string(),
        ));
    }

    if n_labels > n_classes {
        return Err(SklearsError::InvalidInput(
            "n_labels cannot exceed n_classes".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));
    let normal = StandardNormal;

    // Generate feature matrix
    let mut x = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    // Generate multilabel target matrix
    let mut y = Array2::zeros((n_samples, n_classes));

    // For each sample, randomly assign n_labels classes
    for i in 0..n_samples {
        // Create a vector of all possible class indices
        let mut available_classes: Vec<usize> = (0..n_classes).collect();

        // Randomly shuffle the classes
        for j in (1..available_classes.len()).rev() {
            let k = rng.gen_range(0..j + 1);
            available_classes.swap(j, k);
        }

        // Assign the first n_labels classes to this sample
        for &class_idx in available_classes.iter().take(n_labels) {
            y[[i, class_idx]] = 1;
        }
    }

    Ok((x, y))
}

/// Generate sparse uncorrelated classification dataset
///
/// Creates a dataset where only the first few features are informative for the target,
/// while the remaining features are noise. This is useful for testing feature selection
/// and dimensionality reduction algorithms.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `n_features`: Number of features per sample (must be >= 4)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (features, binary_targets)
pub fn make_sparse_uncorrelated(
    n_samples: usize,
    n_features: usize,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_features must be positive".to_string(),
        ));
    }

    if n_features < 4 {
        return Err(SklearsError::InvalidInput(
            "n_features must be >= 4 for sparse uncorrelated generation".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));
    let normal = StandardNormal;

    // Generate features with specific properties
    let mut x = Array2::zeros((n_samples, n_features));

    // First 4 features are informative
    for i in 0..n_samples {
        for j in 0..4 {
            x[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    // Remaining features are noise
    for i in 0..n_samples {
        for j in 4..n_features {
            x[[i, j]] = rng.sample::<f64, _>(normal) * 0.01; // Small noise
        }
    }

    // Generate targets based on a sparse linear combination
    let mut y = Array1::zeros(n_samples);

    // Only first 4 features are informative for the target
    for i in 0..n_samples {
        // Simple linear combination with sparse coefficients
        let target = x[[i, 0]] + 2.0 * x[[i, 1]] - x[[i, 2]] + 0.5 * x[[i, 3]];

        // Convert to binary classification
        y[i] = if target > 0.0 { 1 } else { 0 };
    }

    Ok((x, y))
}

/// Generate polynomial regression dataset
///
/// Creates a regression dataset where the target is a polynomial function of the features
/// with specified degree. Features are generated uniformly between -1 and 1, and optional
/// Gaussian noise can be added to the targets.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `n_features`: Number of input features
/// - `degree`: Degree of the polynomial (1=linear, 2=quadratic, etc.)
/// - `noise`: Standard deviation of Gaussian noise added to targets
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (features, continuous_targets)
pub fn make_polynomial_regression(
    n_samples: usize,
    n_features: usize,
    degree: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 || n_features == 0 || degree == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, n_features, and degree must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // Generate features uniformly between -1 and 1
    let mut x = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.random_range(-1.0, 1.0);
        }
    }

    // Generate polynomial coefficients
    let n_terms = polynomial_terms_count(n_features, degree);
    let mut coeffs = Array1::zeros(n_terms);
    for i in 0..n_terms {
        coeffs[i] = rng.random_range(-1.0, 1.0);
    }

    // Compute polynomial target values
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let mut target = 0.0;
        let mut term_idx = 0;

        // Add terms for each degree up to the specified degree
        for d in 0..=degree {
            let terms = generate_polynomial_terms(&x.row(i).to_owned(), d);
            for term_value in terms {
                if term_idx < coeffs.len() {
                    target += coeffs[term_idx] * term_value;
                    term_idx += 1;
                }
            }
        }

        // Add noise
        if noise > 0.0 {
            let noise_dist = Normal::new(0.0, noise).unwrap();
            target += rng.sample(noise_dist);
        }

        y[i] = target;
    }

    Ok((x, y))
}

// Helper function to count the number of polynomial terms
fn polynomial_terms_count(n_features: usize, max_degree: usize) -> usize {
    let mut count = 0;
    for degree in 0..=max_degree {
        count += combinations_with_repetition(n_features, degree);
    }
    count
}

// Helper function to calculate combinations with repetition
fn combinations_with_repetition(n: usize, k: usize) -> usize {
    if k == 0 {
        return 1;
    }
    if n == 0 {
        return 0;
    }

    let mut result = 1;
    for i in 0..k {
        result = result * (n + i) / (i + 1);
    }
    result
}

// Helper function to generate polynomial terms for a given degree
fn generate_polynomial_terms(features: &Array1<f64>, degree: usize) -> Vec<f64> {
    if degree == 0 {
        return vec![1.0];
    }

    let n_features = features.len();
    let mut terms = Vec::new();

    if degree == 1 {
        for i in 0..n_features {
            terms.push(features[i]);
        }
    } else {
        // For higher degrees, generate all combinations
        generate_combinations_recursive(features, degree, &mut vec![], &mut terms, 0);
    }

    terms
}

// Recursive helper to generate combinations for polynomial terms
fn generate_combinations_recursive(
    features: &Array1<f64>,
    remaining_degree: usize,
    current_indices: &mut Vec<usize>,
    terms: &mut Vec<f64>,
    start_idx: usize,
) {
    if remaining_degree == 0 {
        let mut term_value = 1.0;
        for idx in current_indices.iter() {
            term_value *= features[*idx];
        }
        terms.push(term_value);
        return;
    }

    for i in start_idx..features.len() {
        current_indices.push(i);
        generate_combinations_recursive(features, remaining_degree - 1, current_indices, terms, i);
        current_indices.pop();
    }
}