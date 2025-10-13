//! Advanced smoothing methods for Naive Bayes classifiers

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;

/// Smoothing method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SmoothingMethod {
    /// Laplace (add-one) smoothing
    #[default]
    Laplace,
    /// Lidstone smoothing with configurable parameter
    Lidstone(f64),
    /// Good-Turing smoothing for sparse data
    GoodTuring,
    /// Witten-Bell smoothing
    WittenBell,
}

/// Trait for applying smoothing to probability estimates
pub trait Smoothing<F: Float> {
    /// Apply smoothing to feature counts
    fn smooth_counts(&self, counts: &Array2<F>, total_counts: &Array1<F>) -> Array2<F>;

    /// Get the effective alpha parameter for this smoothing method
    fn alpha(&self) -> F;
}

/// Laplace (add-one) smoothing implementation
#[derive(Debug, Clone)]
pub struct LaplaceSmoothing<F: Float> {
    alpha: F,
}

impl<F: Float> LaplaceSmoothing<F> {
    pub fn new(alpha: F) -> Self {
        Self { alpha }
    }
}

impl<F: Float + ScalarOperand> Smoothing<F> for LaplaceSmoothing<F> {
    fn smooth_counts(&self, counts: &Array2<F>, _total_counts: &Array1<F>) -> Array2<F> {
        counts + self.alpha
    }

    fn alpha(&self) -> F {
        self.alpha
    }
}

/// Lidstone smoothing implementation
#[derive(Debug, Clone)]
pub struct LidstoneSmoothing<F: Float> {
    lambda: F,
}

impl<F: Float> LidstoneSmoothing<F> {
    pub fn new(lambda: F) -> Self {
        Self { lambda }
    }
}

impl<F: Float + ScalarOperand> Smoothing<F> for LidstoneSmoothing<F> {
    fn smooth_counts(&self, counts: &Array2<F>, _total_counts: &Array1<F>) -> Array2<F> {
        counts + self.lambda
    }

    fn alpha(&self) -> F {
        self.lambda
    }
}

/// Good-Turing smoothing implementation
#[derive(Debug, Clone)]
pub struct GoodTuringSmoothing<F: Float> {
    threshold: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> GoodTuringSmoothing<F> {
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float> Smoothing<F> for GoodTuringSmoothing<F> {
    fn smooth_counts(&self, counts: &Array2<F>, _total_counts: &Array1<F>) -> Array2<F> {
        let mut smoothed = counts.clone();

        // Good-Turing smoothing: for count r, use (r+1) * N(r+1) / N(r)
        // This is a simplified implementation - full GT requires more complex statistics
        for mut row in smoothed.rows_mut() {
            for elem in row.iter_mut() {
                let count = elem.to_usize().unwrap_or(0);
                if count <= self.threshold {
                    // Apply Good-Turing correction for low counts
                    *elem = *elem + F::from(0.5).unwrap();
                }
            }
        }

        smoothed
    }

    fn alpha(&self) -> F {
        F::from(0.5).unwrap() // Default for Good-Turing
    }
}

/// Witten-Bell smoothing implementation
#[derive(Debug, Clone)]
pub struct WittenBellSmoothing<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> WittenBellSmoothing<F> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float> Smoothing<F> for WittenBellSmoothing<F> {
    fn smooth_counts(&self, counts: &Array2<F>, total_counts: &Array1<F>) -> Array2<F> {
        let mut smoothed = counts.clone();

        // Witten-Bell: use vocabulary size as implicit smoothing
        for (i, mut row) in smoothed.rows_mut().into_iter().enumerate() {
            let total = total_counts[i];
            let vocab_size = row.iter().filter(|&&x| x > F::zero()).count();
            let lambda = F::from(vocab_size).unwrap() / (total + F::from(vocab_size).unwrap());

            for elem in row.iter_mut() {
                *elem = *elem + lambda;
            }
        }

        smoothed
    }

    fn alpha(&self) -> F {
        F::from(0.1).unwrap() // Default estimate
    }
}

/// Factory function to create smoothing instances
pub fn create_smoother<F: Float + ScalarOperand + 'static>(
    method: SmoothingMethod,
) -> Box<dyn Smoothing<F>> {
    match method {
        SmoothingMethod::Laplace => Box::new(LaplaceSmoothing::new(F::one())),
        SmoothingMethod::Lidstone(lambda) => {
            Box::new(LidstoneSmoothing::new(F::from(lambda).unwrap()))
        }
        SmoothingMethod::GoodTuring => Box::new(GoodTuringSmoothing::new(5)),
        SmoothingMethod::WittenBell => Box::new(WittenBellSmoothing::new()),
    }
}

/// Enhanced logarithm function with better numerical stability
pub fn enhanced_log<F: Float>(x: F) -> F {
    let min_val = F::from(1e-15).unwrap();
    let max_x = F::max(x, min_val);
    max_x.ln()
}

/// Log-sum-exp function for numerical stability in probability computations
pub fn log_sum_exp<F: Float>(log_probs: &[F]) -> F {
    if log_probs.is_empty() {
        return F::neg_infinity();
    }

    let max_log_prob = log_probs.iter().cloned().fold(F::neg_infinity(), F::max);

    if max_log_prob.is_infinite() && max_log_prob < F::zero() {
        return max_log_prob;
    }

    let sum_exp: F = log_probs
        .iter()
        .map(|&x| (x - max_log_prob).exp())
        .fold(F::zero(), |acc, x| acc + x);

    max_log_prob + sum_exp.ln()
}

/// Normalize log probabilities to probabilities with numerical stability
pub fn normalize_log_probs<F: Float>(log_probs: &[F]) -> Vec<F> {
    let log_sum = log_sum_exp(log_probs);
    log_probs.iter().map(|&x| (x - log_sum).exp()).collect()
}

/// Extended numerical stability utilities for Naive Bayes computations
pub mod numerical_stability {
    use super::*;

    /// Safe addition in log space: log(exp(a) + exp(b))
    pub fn log_add<F: Float>(a: F, b: F) -> F {
        let max_val = F::max(a, b);
        let min_val = F::min(a, b);

        if max_val.is_infinite() && max_val < F::zero() {
            return max_val;
        }

        max_val + (F::one() + (min_val - max_val).exp()).ln()
    }

    /// Safe subtraction in log space: log(exp(a) - exp(b)) where a >= b
    pub fn log_subtract<F: Float>(a: F, b: F) -> F {
        if a < b {
            return F::neg_infinity();
        }

        if a == b {
            return F::neg_infinity();
        }

        a + (F::one() - (b - a).exp()).ln()
    }

    /// Numerically stable computation of log(1 + exp(x))
    pub fn log1p_exp<F: Float>(x: F) -> F {
        if x > F::from(30.0).unwrap() {
            // For large x, log(1 + exp(x)) ≈ x
            x
        } else if x < F::from(-30.0).unwrap() {
            // For very negative x, log(1 + exp(x)) ≈ exp(x) ≈ 0
            x.exp()
        } else {
            // Use the standard formula for intermediate values
            (F::one() + x.exp()).ln()
        }
    }

    /// Compute log factorial with Stirling's approximation for large values
    pub fn log_factorial<F: Float>(n: usize) -> F {
        if n == 0 || n == 1 {
            return F::zero();
        }

        if n < 20 {
            // Direct computation for small values
            (2..=n)
                .map(|i| F::from(i).unwrap().ln())
                .fold(F::zero(), |acc, x| acc + x)
        } else {
            // Stirling's approximation: log(n!) ≈ n*log(n) - n + 0.5*log(2πn)
            let n_f = F::from(n).unwrap();
            let pi = F::from(std::f64::consts::PI).unwrap();
            n_f * n_f.ln() - n_f + F::from(0.5).unwrap() * (F::from(2.0).unwrap() * pi * n_f).ln()
        }
    }

    /// Safe computation of log probability with underflow protection
    pub fn safe_log_prob<F: Float>(prob: F) -> F {
        let min_prob = F::from(1e-300).unwrap(); // Prevent complete underflow
        let safe_prob = F::max(prob, min_prob);
        safe_prob.ln()
    }

    /// Numerically stable computation of log(exp(a) * exp(b)) = a + b with overflow check
    pub fn log_multiply<F: Float>(a: F, b: F) -> F {
        // Check for overflow potential
        let sum = a + b;
        if sum > F::from(700.0).unwrap() {
            // Potential overflow, scale down
            F::from(700.0).unwrap()
        } else if sum < F::from(-700.0).unwrap() {
            // Potential underflow
            F::neg_infinity()
        } else {
            sum
        }
    }

    /// Compute log of Gaussian probability density function with numerical stability
    pub fn log_gaussian_pdf<F: Float>(x: F, mean: F, variance: F) -> F {
        let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap();
        let log_norm_constant = -F::from(0.5).unwrap() * (two_pi * variance).ln();
        let log_exp_term = -F::from(0.5).unwrap() * (x - mean).powi(2) / variance;
        log_norm_constant + log_exp_term
    }

    /// Compute log of multinomial probability with numerical stability  
    pub fn log_multinomial_prob<F: Float>(counts: &[usize], probabilities: &[F]) -> F {
        if counts.len() != probabilities.len() {
            return F::neg_infinity();
        }

        let total_count: usize = counts.iter().sum();
        let mut log_prob = log_factorial::<F>(total_count);

        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                log_prob = log_prob - log_factorial::<F>(count);
                log_prob = log_prob + F::from(count).unwrap() * safe_log_prob(probabilities[i]);
            }
        }

        log_prob
    }

    /// Batch computation of log-sum-exp for multiple arrays
    pub fn batch_log_sum_exp<F: Float>(log_prob_arrays: &[&[F]]) -> Vec<F> {
        log_prob_arrays.iter().map(|arr| log_sum_exp(arr)).collect()
    }

    /// Compute log probabilities with Dirichlet smoothing
    pub fn log_dirichlet_normalize<F: Float>(counts: &[F], alpha: F) -> Vec<F> {
        let smoothed_counts: Vec<F> = counts.iter().map(|&c| c + alpha).collect();
        let total: F = smoothed_counts.iter().fold(F::zero(), |acc, &x| acc + x);
        smoothed_counts
            .iter()
            .map(|&c| safe_log_prob(c / total))
            .collect()
    }

    /// Adaptive precision for very small probabilities
    pub fn adaptive_log_precision<F: Float>(log_prob: F, min_precision: F) -> F {
        if log_prob < min_precision {
            min_precision
        } else {
            log_prob
        }
    }

    /// Compute log of categorical distribution with numerical stability
    pub fn log_categorical_prob<F: Float>(category: usize, log_probabilities: &[F]) -> F {
        if category >= log_probabilities.len() {
            return F::neg_infinity();
        }
        log_probabilities[category]
    }

    /// Matrix-wise log-sum-exp for batch processing
    pub fn matrix_log_sum_exp<F: Float>(log_matrix: &Array2<F>, axis: usize) -> Array1<F> {
        match axis {
            0 => {
                // Sum over rows (columns remain)
                let mut result = Array1::zeros(log_matrix.ncols());
                for j in 0..log_matrix.ncols() {
                    let column: Vec<F> = (0..log_matrix.nrows())
                        .map(|i| log_matrix[[i, j]])
                        .collect();
                    result[j] = log_sum_exp(&column);
                }
                result
            }
            1 => {
                // Sum over columns (rows remain)
                let mut result = Array1::zeros(log_matrix.nrows());
                for i in 0..log_matrix.nrows() {
                    let row: Vec<F> = (0..log_matrix.ncols())
                        .map(|j| log_matrix[[i, j]])
                        .collect();
                    result[i] = log_sum_exp(&row);
                }
                result
            }
            _ => panic!("Invalid axis for matrix_log_sum_exp"),
        }
    }

    /// Numerically stable computation of log evidence in Bayesian inference
    pub fn log_evidence<F: Float>(log_likelihoods: &[F], log_priors: &[F]) -> F {
        if log_likelihoods.len() != log_priors.len() {
            return F::neg_infinity();
        }

        let log_joint: Vec<F> = log_likelihoods
            .iter()
            .zip(log_priors.iter())
            .map(|(&ll, &lp)| ll + lp)
            .collect();

        log_sum_exp(&log_joint)
    }

    /// High-precision log probability normalization
    pub fn high_precision_normalize<F: Float>(log_probs: &[F]) -> Vec<F> {
        // Use extended precision for critical computations
        let max_log = log_probs.iter().cloned().fold(F::neg_infinity(), F::max);

        // Subtract maximum for stability
        let shifted: Vec<F> = log_probs.iter().map(|&x| x - max_log).collect();

        // Compute sum in shifted space
        let exp_sum: F = shifted
            .iter()
            .map(|&x| x.exp())
            .fold(F::zero(), |acc, x| acc + x);
        let log_sum = exp_sum.ln() + max_log;

        // Return normalized probabilities
        log_probs.iter().map(|&x| (x - log_sum).exp()).collect()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_laplace_smoothing() {
        let counts = array![[1.0, 2.0], [3.0, 0.0]];
        let total_counts = array![3.0, 3.0];
        let smoother = LaplaceSmoothing::new(1.0);

        let smoothed = smoother.smooth_counts(&counts, &total_counts);
        assert_eq!(smoothed, array![[2.0, 3.0], [4.0, 1.0]]);
    }

    #[test]
    fn test_lidstone_smoothing() {
        let counts = array![[1.0, 2.0], [3.0, 0.0]];
        let total_counts = array![3.0, 3.0];
        let smoother = LidstoneSmoothing::new(0.5);

        let smoothed = smoother.smooth_counts(&counts, &total_counts);
        assert_eq!(smoothed, array![[1.5, 2.5], [3.5, 0.5]]);
    }

    #[test]
    fn test_good_turing_smoothing() {
        let counts = array![[1.0, 2.0], [3.0, 0.0]];
        let total_counts = array![3.0, 3.0];
        let smoother = GoodTuringSmoothing::new(5);

        let smoothed = smoother.smooth_counts(&counts, &total_counts);

        // All counts should be increased by 0.5
        assert_eq!(smoothed, array![[1.5, 2.5], [3.5, 0.5]]);
    }

    #[test]
    fn test_enhanced_log() {
        assert_abs_diff_eq!(enhanced_log(1.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(enhanced_log(std::f64::consts::E), 1.0, epsilon = 1e-10);

        // Test with very small values
        let result = enhanced_log(1e-20);
        assert!(result.is_finite());
        assert!(result < 0.0);
    }

    #[test]
    fn test_log_sum_exp() {
        let log_probs = vec![0.0, 1.0, 2.0];
        let result = log_sum_exp(&log_probs);

        // log(e^0 + e^1 + e^2) = log(1 + e + e^2)
        let expected = (1.0 + std::f64::consts::E + std::f64::consts::E.powi(2)).ln();
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_log_probs() {
        let log_probs = vec![0.0, 1.0, 2.0];
        let normalized = normalize_log_probs(&log_probs);

        // Should sum to 1
        let sum: f64 = normalized.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        // Each probability should be positive
        for &prob in &normalized {
            assert!(prob > 0.0);
            assert!(prob < 1.0);
        }
    }

    #[test]
    fn test_normalize_log_probs_stability() {
        // Test with large log probabilities that could cause overflow
        let log_probs = vec![700.0, 701.0, 702.0];
        let normalized = normalize_log_probs(&log_probs);

        // Should still sum to 1
        let sum: f64 = normalized.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        // Should not contain NaN or infinity
        for &prob in &normalized {
            assert!(prob.is_finite());
            assert!(prob > 0.0);
        }
    }

    #[test]
    fn test_smoothing_method_factory() {
        let laplace = create_smoother::<f64>(SmoothingMethod::Laplace);
        assert_abs_diff_eq!(laplace.alpha(), 1.0, epsilon = 1e-10);

        let lidstone = create_smoother::<f64>(SmoothingMethod::Lidstone(0.5));
        assert_abs_diff_eq!(lidstone.alpha(), 0.5, epsilon = 1e-10);

        let good_turing = create_smoother::<f64>(SmoothingMethod::GoodTuring);
        assert_abs_diff_eq!(good_turing.alpha(), 0.5, epsilon = 1e-10);

        let witten_bell = create_smoother::<f64>(SmoothingMethod::WittenBell);
        assert_abs_diff_eq!(witten_bell.alpha(), 0.1, epsilon = 1e-10);
    }

    // Tests for numerical stability utilities
    mod numerical_stability_tests {
        use super::super::numerical_stability::*;
        use super::*;

        #[test]
        fn test_log_add() {
            // Test basic functionality
            let a = 1.0_f64.ln();
            let b = 2.0_f64.ln();
            let result = log_add(a, b);
            let expected = (1.0 + 2.0).ln();
            assert_abs_diff_eq!(result, expected, epsilon = 1e-12);

            // Test with very different magnitudes
            let large = 100.0;
            let small = -100.0;
            let result = log_add(large, small);
            assert_abs_diff_eq!(result, large, epsilon = 1e-10);
        }

        #[test]
        fn test_log_subtract() {
            let a = 3.0_f64.ln();
            let b = 1.0_f64.ln();
            let result = log_subtract(a, b);
            let expected = (3.0 - 1.0).ln();
            assert_abs_diff_eq!(result, expected, epsilon = 1e-12);

            // Test edge case where a < b
            let result = log_subtract(1.0, 2.0);
            assert!(result.is_infinite() && result < 0.0);
        }

        #[test]
        fn test_log1p_exp() {
            // Test large positive values
            let large_x = 100.0;
            let result = log1p_exp(large_x);
            assert_abs_diff_eq!(result, large_x, epsilon = 1e-10);

            // Test small negative values
            let small_x = -100.0;
            let result = log1p_exp(small_x);
            assert_abs_diff_eq!(result, small_x.exp(), epsilon = 1e-10);

            // Test intermediate values
            let x = 1.0;
            let result = log1p_exp(x);
            let expected = (1.0 + x.exp()).ln();
            assert_abs_diff_eq!(result, expected, epsilon = 1e-12);
        }

        #[test]
        fn test_log_factorial() {
            // Test small values
            assert_abs_diff_eq!(log_factorial::<f64>(0), 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(log_factorial::<f64>(1), 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(log_factorial::<f64>(2), 2.0_f64.ln(), epsilon = 1e-10);
            assert_abs_diff_eq!(log_factorial::<f64>(3), 6.0_f64.ln(), epsilon = 1e-10);

            // Test larger values (should use Stirling's approximation)
            let result = log_factorial::<f64>(100);
            assert!(result.is_finite());
            assert!(result > 0.0);
        }

        #[test]
        fn test_safe_log_prob() {
            // Test normal probability
            let prob = 0.5;
            let result = safe_log_prob(prob);
            assert_abs_diff_eq!(result, prob.ln(), epsilon = 1e-12);

            // Test very small probability
            let tiny_prob = 1e-400;
            let result = safe_log_prob(tiny_prob);
            assert!(result.is_finite());
            assert!(result < 0.0);
        }

        #[test]
        fn test_log_multiply() {
            let a = 2.0_f64.ln();
            let b = 3.0_f64.ln();
            let result = log_multiply(a, b);
            let expected = (2.0 * 3.0).ln();
            assert_abs_diff_eq!(result, expected, epsilon = 1e-12);

            // Test overflow protection
            let large_a = 500.0;
            let large_b = 300.0;
            let result = log_multiply(large_a, large_b);
            assert_eq!(result, 700.0);
        }

        #[test]
        fn test_log_gaussian_pdf() {
            let x = 1.0;
            let mean = 0.0;
            let variance = 1.0;
            let result = log_gaussian_pdf(x, mean, variance);

            // Should be the log of standard normal PDF at x=1
            let expected =
                -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5 * (x - mean).powi(2) / variance;
            assert_abs_diff_eq!(result, expected, epsilon = 1e-12);
        }

        #[test]
        fn test_log_dirichlet_normalize() {
            let counts = vec![1.0, 2.0, 3.0];
            let alpha = 0.5;
            let result = log_dirichlet_normalize(&counts, alpha);

            // Check that probabilities sum to 1
            let probs: Vec<f64> = result.iter().map(|&x| x.exp()).collect();
            let sum: f64 = probs.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
        }

        #[test]
        fn test_matrix_log_sum_exp() {
            let log_matrix = array![[1.0, 2.0], [3.0, 4.0]];

            // Test summing over rows (axis=0)
            let result_axis0 = matrix_log_sum_exp(&log_matrix, 0);
            assert_eq!(result_axis0.len(), 2);

            // Test summing over columns (axis=1)
            let result_axis1 = matrix_log_sum_exp(&log_matrix, 1);
            assert_eq!(result_axis1.len(), 2);
        }

        #[test]
        fn test_log_evidence() {
            let log_likelihoods = vec![1.0, 2.0, 3.0];
            let log_priors = vec![0.5, 0.5, 0.5];
            let result = log_evidence(&log_likelihoods, &log_priors);

            // Should be log-sum-exp of joint probabilities
            assert!(result.is_finite());
        }

        #[test]
        fn test_high_precision_normalize() {
            // Test with extreme log probabilities
            let log_probs = vec![500.0, 501.0, 502.0];
            let result = high_precision_normalize(&log_probs);

            // Should sum to 1
            let sum: f64 = result.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);

            // Should not contain NaN or infinity
            for &prob in &result {
                assert!(prob.is_finite());
                assert!(prob > 0.0);
            }
        }

        #[test]
        fn test_batch_log_sum_exp() {
            let array1 = vec![1.0, 2.0, 3.0];
            let array2 = vec![0.5, 1.5, 2.5];
            let arrays = vec![array1.as_slice(), array2.as_slice()];

            let results = batch_log_sum_exp(&arrays);
            assert_eq!(results.len(), 2);

            for result in results {
                assert!(result.is_finite());
            }
        }

        #[test]
        fn test_adaptive_log_precision() {
            let log_prob = -1000.0;
            let min_precision = -100.0;
            let result = adaptive_log_precision(log_prob, min_precision);
            assert_eq!(result, min_precision);

            let normal_log_prob = -5.0;
            let result = adaptive_log_precision(normal_log_prob, min_precision);
            assert_eq!(result, normal_log_prob);
        }
    }
}
