/// Utility functions for machine learning operations
///
/// This module provides common utility functions that are frequently needed
/// across different machine learning algorithms and workflows.
use crate::types::Float;
// SciRS2 Policy: Using scirs2_core::ndarray (COMPLIANT)
use scirs2_core::ndarray::Array1;

/// Generate a random seed from system entropy
pub fn generate_random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Calculate the entropy of a discrete distribution
///
/// # Arguments
/// * `probabilities` - Array of probabilities that should sum to 1.0
///
/// # Returns
/// The Shannon entropy in bits
///
/// # Example
/// ```
/// use sklears_core::utils::entropy;
/// use scirs2_core::ndarray::array;
///
/// let probs = array![0.5, 0.5];
/// let ent = entropy(&probs);
/// assert!((ent - 1.0).abs() < 1e-10);
/// ```
pub fn entropy(probabilities: &Array1<Float>) -> Float {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Calculate Gini impurity for a discrete distribution
///
/// # Arguments
/// * `probabilities` - Array of probabilities that should sum to 1.0
///
/// # Returns
/// The Gini impurity score
///
/// # Example
/// ```
/// use sklears_core::utils::gini_impurity;
/// use scirs2_core::ndarray::array;
///
/// let probs = array![0.5, 0.5];
/// let gini = gini_impurity(&probs);
/// assert!((gini - 0.5).abs() < 1e-10);
/// ```
pub fn gini_impurity(probabilities: &Array1<Float>) -> Float {
    1.0 - probabilities.iter().map(|&p| p * p).sum::<Float>()
}

/// Normalize an array to have zero mean and unit variance
///
/// # Arguments
/// * `array` - Input array to normalize
///
/// # Returns
/// Normalized array with zero mean and unit variance
///
/// # Example
/// ```
/// use sklears_core::utils::standardize;
/// use scirs2_core::ndarray::array;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let normalized = standardize(&data);
/// let mean = normalized.mean().unwrap();
/// assert!(mean.abs() < 1e-10);
/// ```
pub fn standardize(array: &Array1<Float>) -> Array1<Float> {
    let mean = array.mean().unwrap();
    let std = array.std(0.0);

    if std > 1e-10 {
        (array - mean) / std
    } else {
        array.clone()
    }
}

/// Min-max normalize an array to the range [0, 1]
///
/// # Arguments
/// * `array` - Input array to normalize
///
/// # Returns
/// Normalized array with values in [0, 1]
///
/// # Example
/// ```
/// use sklears_core::utils::min_max_normalize;
/// use scirs2_core::ndarray::array;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let normalized = min_max_normalize(&data);
/// assert!((normalized[[0]] - 0.0).abs() < 1e-10);
/// assert!((normalized[[4]] - 1.0).abs() < 1e-10);
/// ```
pub fn min_max_normalize(array: &Array1<Float>) -> Array1<Float> {
    let min_val = array.iter().fold(Float::INFINITY, |a, &b| a.min(b));
    let max_val = array.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    if range > 1e-10 {
        (array - min_val) / range
    } else {
        Array1::zeros(array.len())
    }
}

/// Calculate the cosine similarity between two vectors
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity value between -1 and 1
///
/// # Example
/// ```
/// use sklears_core::utils::cosine_similarity;
/// use scirs2_core::ndarray::array;
///
/// let a = array![1.0, 0.0];
/// let b = array![0.0, 1.0];
/// let sim = cosine_similarity(&a, &b);
/// assert!((sim - 0.0).abs() < 1e-10);
/// ```
pub fn cosine_similarity(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    if a.len() != b.len() {
        panic!("Arrays must have the same length");
    }

    let dot_product = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<Float>();
    let norm_a = a.iter().map(|&x| x * x).sum::<Float>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<Float>().sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Calculate Euclidean distance between two points
///
/// # Arguments
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
/// Euclidean distance
///
/// # Example
/// ```
/// use sklears_core::utils::euclidean_distance;
/// use scirs2_core::ndarray::array;
///
/// let a = array![0.0, 0.0];
/// let b = array![3.0, 4.0];
/// let dist = euclidean_distance(&a, &b);
/// assert!((dist - 5.0).abs() < 1e-10);
/// ```
pub fn euclidean_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    if a.len() != b.len() {
        panic!("Arrays must have the same length");
    }

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<Float>()
        .sqrt()
}

/// Calculate Manhattan distance between two points
///
/// # Arguments
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
/// Manhattan distance
///
/// # Example
/// ```
/// use sklears_core::utils::manhattan_distance;
/// use scirs2_core::ndarray::array;
///
/// let a = array![0.0, 0.0];
/// let b = array![3.0, 4.0];
/// let dist = manhattan_distance(&a, &b);
/// assert!((dist - 7.0).abs() < 1e-10);
/// ```
pub fn manhattan_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    if a.len() != b.len() {
        panic!("Arrays must have the same length");
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
}

/// Check if a value is approximately zero within a tolerance
///
/// # Arguments
/// * `value` - Value to check
/// * `tolerance` - Tolerance level (default: 1e-10)
///
/// # Returns
/// True if the value is within tolerance of zero
pub fn is_zero(value: Float, tolerance: Option<Float>) -> bool {
    let tol = tolerance.unwrap_or(1e-10);
    value.abs() < tol
}

/// Clamp a value between minimum and maximum bounds
///
/// # Arguments
/// * `value` - Value to clamp
/// * `min_val` - Minimum bound
/// * `max_val` - Maximum bound
///
/// # Returns
/// Clamped value
pub fn clamp(value: Float, min_val: Float, max_val: Float) -> Float {
    value.clamp(min_val, max_val)
}

/// Calculate the number of combinations (n choose k)
///
/// # Arguments
/// * `n` - Total number of items
/// * `k` - Number of items to choose
///
/// # Returns
/// Number of combinations
pub fn combinations(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k); // Take advantage of symmetry
    let mut result = 1;

    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }

    result
}

/// Generate samples from a multivariate normal distribution
///
/// Generates samples from a multivariate normal distribution with specified mean
/// and identity covariance matrix (independent components with unit variance).
///
/// # Arguments
/// * `mean` - Mean vector of the distribution
/// * `n_samples` - Number of samples to generate
/// * `rng` - Random number generator
///
/// # Returns
/// Array of shape (n_samples, n_features) containing the generated samples
///
/// # Example
/// ```
/// use sklears_core::utils::multivariate_normal_samples;
/// use scirs2_core::ndarray::array;
/// use scirs2_core::random::rngs::StdRng;
/// use scirs2_core::random::SeedableRng;
///
/// let mean = array![0.0, 1.0];
/// let mut rng = StdRng::seed_from_u64(42);
/// let samples = multivariate_normal_samples(&mean, 100, &mut rng);
/// assert_eq!(samples.shape(), &[100, 2]);
/// ```
pub fn multivariate_normal_samples<R: scirs2_core::random::Rng>(
    mean: &Array1<Float>,
    n_samples: usize,
    rng: &mut R,
) -> scirs2_core::ndarray::Array2<Float> {
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::essentials::Normal;
    use scirs2_core::Distribution;

    let n_features = mean.len();
    let mut samples = Array2::zeros((n_samples, n_features));

    // Standard normal distribution (mean=0, std=1)
    let standard_normal =
        Normal::new(0.0, 1.0).expect("Failed to create standard normal distribution");

    // Generate samples: X = μ + Z where Z ~ N(0, I)
    for i in 0..n_samples {
        for j in 0..n_features {
            let z = standard_normal.sample(rng);
            samples[(i, j)] = mean[j] + z;
        }
    }

    samples
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // SciRS2 Policy: Using scirs2_core::ndarray and scirs2_core::random (COMPLIANT)
    use scirs2_core::ndarray::array;

    #[test]
    fn test_entropy_uniform() {
        let probs = array![0.25, 0.25, 0.25, 0.25];
        let ent = entropy(&probs);
        assert!((ent - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_certain() {
        let probs = array![1.0, 0.0, 0.0, 0.0];
        let ent = entropy(&probs);
        assert!(ent.abs() < 1e-10);
    }

    #[test]
    fn test_gini_impurity_uniform() {
        let probs = array![0.5, 0.5];
        let gini = gini_impurity(&probs);
        assert!((gini - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gini_impurity_pure() {
        let probs = array![1.0, 0.0];
        let gini = gini_impurity(&probs);
        assert!(gini.abs() < 1e-10);
    }

    #[test]
    fn test_standardize() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = standardize(&data);
        let mean = normalized.mean().unwrap();
        let std = normalized.std(0.0);
        assert!(mean.abs() < 1e-10);
        assert!((std - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_normalize() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = min_max_normalize(&data);
        assert!((normalized[[0]] - 0.0).abs() < 1e-10);
        assert!((normalized[[4]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = array![1.0, 0.0];
        let b = array![1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);

        let c = array![0.0, 1.0];
        let sim2 = cosine_similarity(&a, &c);
        assert!(sim2.abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let dist = manhattan_distance(&a, &b);
        assert!((dist - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_multivariate_normal_samples() {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;

        let mean = array![0.0, 1.0];
        let mut rng = StdRng::seed_from_u64(42);
        let samples = multivariate_normal_samples(&mean, 100, &mut rng);

        // Check shape
        assert_eq!(samples.shape(), &[100, 2]);

        // Check that sample means are approximately correct
        let sample_mean_0 = samples.column(0).mean().unwrap();
        let sample_mean_1 = samples.column(1).mean().unwrap();

        // With 100 samples, means should be within ~0.3 of true means (roughly 3 * std_err)
        assert!(
            (sample_mean_0 - 0.0).abs() < 0.3,
            "Mean of first component should be close to 0.0"
        );
        assert!(
            (sample_mean_1 - 1.0).abs() < 0.3,
            "Mean of second component should be close to 1.0"
        );

        // Check that samples have reasonable variance (should be close to 1.0)
        let sample_std_0 = samples.column(0).std(0.0);
        let sample_std_1 = samples.column(1).std(0.0);
        assert!(
            sample_std_0 > 0.7 && sample_std_0 < 1.3,
            "Std of first component should be close to 1.0"
        );
        assert!(
            sample_std_1 > 0.7 && sample_std_1 < 1.3,
            "Std of second component should be close to 1.0"
        );
    }

    #[test]
    fn test_is_zero() {
        assert!(is_zero(0.0, None));
        assert!(is_zero(1e-12, None));
        assert!(!is_zero(1e-8, None));
        assert!(is_zero(0.01, Some(0.1)));
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-1.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_combinations() {
        assert_eq!(combinations(5, 0), 1);
        assert_eq!(combinations(5, 1), 5);
        assert_eq!(combinations(5, 2), 10);
        assert_eq!(combinations(5, 3), 10);
        assert_eq!(combinations(5, 5), 1);
        assert_eq!(combinations(3, 5), 0);
    }

    #[test]
    fn test_generate_random_seed() {
        let seed1 = generate_random_seed();
        std::thread::sleep(std::time::Duration::from_nanos(1000));
        let seed2 = generate_random_seed();
        // Seeds should be different (extremely unlikely to be the same)
        assert_ne!(seed1, seed2);
    }
}
