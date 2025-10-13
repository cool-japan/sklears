//! Error-Correcting Output Codes (ECOC) for Multi-Class Discriminant Analysis
//!
//! This module implements Error-Correcting Output Codes, a technique for extending
//! binary classifiers to multi-class problems using error-correcting codes.

use crate::lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig};
// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained},
    types::Float,
};

/// Configuration for Error-Correcting Output Codes
#[derive(Debug, Clone)]
pub struct ErrorCorrectingOutputCodesConfig {
    /// Code matrix generation method
    pub code_method: String,
    /// Number of binary classifiers (code length)
    pub n_codes: Option<usize>,
    /// Base classifier configuration
    pub base_classifier_config: LinearDiscriminantAnalysisConfig,
    /// Error correction method during prediction
    pub correction_method: String,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for ErrorCorrectingOutputCodesConfig {
    fn default() -> Self {
        Self {
            code_method: "random".to_string(),
            n_codes: None,
            base_classifier_config: LinearDiscriminantAnalysisConfig::default(),
            correction_method: "hamming".to_string(),
            random_state: None,
        }
    }
}

/// Error-Correcting Output Codes discriminant analysis
#[derive(Debug, Clone)]
pub struct ErrorCorrectingOutputCodes {
    config: ErrorCorrectingOutputCodesConfig,
}

/// Trained Error-Correcting Output Codes model
#[derive(Debug, Clone)]
pub struct TrainedErrorCorrectingOutputCodes {
    /// Trained binary classifiers
    classifiers: Vec<LinearDiscriminantAnalysis<Trained>>,
    /// Code matrix (n_classes x n_codes)
    code_matrix: Array2<i32>,
    /// Class labels
    classes: Array1<i32>,
    /// Configuration used for training
    config: ErrorCorrectingOutputCodesConfig,
}

impl Default for ErrorCorrectingOutputCodes {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorCorrectingOutputCodes {
    /// Create a new Error-Correcting Output Codes model
    pub fn new() -> Self {
        Self {
            config: ErrorCorrectingOutputCodesConfig::default(),
        }
    }

    /// Set the code generation method
    pub fn code_method(mut self, method: &str) -> Self {
        self.config.code_method = method.to_string();
        self
    }

    /// Set the number of binary classifiers
    pub fn n_codes(mut self, n_codes: usize) -> Self {
        self.config.n_codes = Some(n_codes);
        self
    }

    /// Set the base classifier configuration
    pub fn base_classifier_config(mut self, config: LinearDiscriminantAnalysisConfig) -> Self {
        self.config.base_classifier_config = config;
        self
    }

    /// Set the error correction method
    pub fn correction_method(mut self, method: &str) -> Self {
        self.config.correction_method = method.to_string();
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Generate the code matrix
    fn generate_code_matrix(&self, n_classes: usize) -> Result<Array2<i32>> {
        let n_codes = self.config.n_codes.unwrap_or_else(|| {
            // Default: use approximately log2(n_classes) * 4 codes for good error correction
            ((n_classes as f64).log2().ceil() as usize * 4).max(n_classes)
        });

        match self.config.code_method.as_str() {
            "random" => self.generate_random_code_matrix(n_classes, n_codes),
            "dense_random" => self.generate_dense_random_code_matrix(n_classes, n_codes),
            "sparse_random" => self.generate_sparse_random_code_matrix(n_classes, n_codes),
            "exhaustive" => self.generate_exhaustive_code_matrix(n_classes),
            _ => Err(SklearsError::InvalidParameter {
                name: "code_method".to_string(),
                reason: format!("Unknown code method: {}. Available methods: random, dense_random, sparse_random, exhaustive", self.config.code_method)
            }),
        }
    }

    /// Generate random binary code matrix
    fn generate_random_code_matrix(&self, n_classes: usize, n_codes: usize) -> Result<Array2<i32>> {
        let mut rng_state = self.config.random_state.unwrap_or(42);
        let mut codes = Array2::zeros((n_classes, n_codes));

        for i in 0..n_classes {
            for j in 0..n_codes {
                // Simple LCG for reproducible random numbers
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                codes[[i, j]] = if (rng_state % 2) == 0 { -1 } else { 1 };
            }
        }

        // Ensure each code has at least one +1 and one -1
        for j in 0..n_codes {
            let col = codes.column(j);
            let pos_count = col.iter().filter(|&&x| x == 1).count();
            let neg_count = col.iter().filter(|&&x| x == -1).count();

            if pos_count == 0 || neg_count == 0 {
                // Fix degenerate codes
                codes[[0, j]] = 1;
                codes[[1 % n_classes, j]] = -1;
            }
        }

        Ok(codes)
    }

    /// Generate dense random code matrix (more balanced +1/-1 distribution)
    fn generate_dense_random_code_matrix(
        &self,
        n_classes: usize,
        n_codes: usize,
    ) -> Result<Array2<i32>> {
        let mut rng_state = self.config.random_state.unwrap_or(42);
        let mut codes = Array2::zeros((n_classes, n_codes));

        for j in 0..n_codes {
            // For each code, randomly assign half the classes to +1 and half to -1
            let mut class_indices: Vec<usize> = (0..n_classes).collect();

            // Shuffle the indices
            for i in 0..n_classes {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let swap_idx = (rng_state as usize) % (i + 1);
                class_indices.swap(i, swap_idx);
            }

            let half = n_classes / 2;
            for i in 0..n_classes {
                codes[[class_indices[i], j]] = if i < half { 1 } else { -1 };
            }
        }

        Ok(codes)
    }

    /// Generate sparse random code matrix (mostly 0s with some +1/-1)
    fn generate_sparse_random_code_matrix(
        &self,
        n_classes: usize,
        n_codes: usize,
    ) -> Result<Array2<i32>> {
        let mut rng_state = self.config.random_state.unwrap_or(42);
        let mut codes = Array2::zeros((n_classes, n_codes));

        // Sparsity: only 30% of entries are non-zero
        let non_zero_prob = 0.3;

        for i in 0..n_classes {
            for j in 0..n_codes {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let r1 = (rng_state as f64) / (u64::MAX as f64);

                if r1 < non_zero_prob {
                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    codes[[i, j]] = if (rng_state % 2) == 0 { -1 } else { 1 };
                }
            }
        }

        // Ensure each code and each class has at least one non-zero entry
        for j in 0..n_codes {
            let col = codes.column(j);
            if col.iter().all(|&x| x == 0) {
                codes[[j % n_classes, j]] = 1;
                codes[[(j + 1) % n_classes, j]] = -1;
            }
        }

        for i in 0..n_classes {
            let row = codes.row(i);
            if row.iter().all(|&x| x == 0) {
                codes[[i, i % n_codes]] = 1;
            }
        }

        Ok(codes)
    }

    /// Generate exhaustive code matrix (all possible binary combinations)
    fn generate_exhaustive_code_matrix(&self, n_classes: usize) -> Result<Array2<i32>> {
        if n_classes > 10 {
            return Err(SklearsError::InvalidParameter {
                name: "n_classes".to_string(),
                reason: "Exhaustive code matrix only supported for up to 10 classes due to exponential growth".to_string()
            });
        }

        let n_codes = 2_usize.pow(n_classes as u32 - 1) - 1;
        let mut codes = Array2::zeros((n_classes, n_codes));

        for j in 0..n_codes {
            for i in 0..n_classes {
                codes[[i, j]] = if ((j + 1) >> i) & 1 == 1 { 1 } else { -1 };
            }
        }

        Ok(codes)
    }
}

impl Estimator for ErrorCorrectingOutputCodes {
    type Config = ErrorCorrectingOutputCodesConfig;
    type Float = Float;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for ErrorCorrectingOutputCodes {
    type Fitted = TrainedErrorCorrectingOutputCodes;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<TrainedErrorCorrectingOutputCodes> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.len()".to_string(),
                actual: format!("X.shape[0]={}, y.len()={}", x.nrows(), y.len()),
            });
        }

        // Get unique classes
        let mut classes: Vec<i32> = y.iter().cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_classes".to_string(),
                reason: "Need at least 2 classes for classification".to_string(),
            });
        }

        // Generate the code matrix
        let code_matrix = self.generate_code_matrix(n_classes)?;
        let n_codes = code_matrix.ncols();

        // Train binary classifiers for each code
        let mut classifiers = Vec::with_capacity(n_codes);

        for j in 0..n_codes {
            let code_column = code_matrix.column(j);

            // Create binary labels based on the code
            let mut binary_y = Array1::zeros(y.len());
            for (i, &label) in y.iter().enumerate() {
                let class_idx = classes.iter().position(|&c| c == label).unwrap();
                binary_y[i] = code_column[class_idx];
            }

            // Skip codes that result in all samples having the same label
            let unique_labels: Vec<i32> = binary_y
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            if unique_labels.len() < 2 {
                return Err(SklearsError::InvalidParameter {
                    name: "code_matrix".to_string(),
                    reason: "Degenerate code matrix: some codes result in only one class"
                        .to_string(),
                });
            }

            // Train binary classifier
            let base_classifier = LinearDiscriminantAnalysis::new();

            let trained_classifier = base_classifier.fit(x, &binary_y)?;
            classifiers.push(trained_classifier);
        }

        Ok(TrainedErrorCorrectingOutputCodes {
            classifiers,
            code_matrix,
            classes,
            config: self.config.clone(),
        })
    }
}

impl TrainedErrorCorrectingOutputCodes {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get the code matrix
    pub fn code_matrix(&self) -> &Array2<i32> {
        &self.code_matrix
    }

    /// Get the number of binary classifiers
    pub fn n_classifiers(&self) -> usize {
        self.classifiers.len()
    }

    /// Calculate distance between two code vectors
    fn code_distance(&self, code1: &[i32], code2: &[i32]) -> Float {
        match self.config.correction_method.as_str() {
            "hamming" => code1
                .iter()
                .zip(code2.iter())
                .map(|(&a, &b)| if a != b { 1.0 } else { 0.0 })
                .sum(),
            "euclidean" => code1
                .iter()
                .zip(code2.iter())
                .map(|(&a, &b)| (a as Float - b as Float).powi(2))
                .sum::<Float>()
                .sqrt(),
            "manhattan" => code1
                .iter()
                .zip(code2.iter())
                .map(|(&a, &b)| (a as Float - b as Float).abs())
                .sum(),
            _ => {
                // Default to Hamming distance
                code1
                    .iter()
                    .zip(code2.iter())
                    .map(|(&a, &b)| if a != b { 1.0 } else { 0.0 })
                    .sum()
            }
        }
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedErrorCorrectingOutputCodes {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let n_samples = x.nrows();
        let n_codes = self.classifiers.len();
        let n_classes = self.classes.len();

        // Get predictions from all binary classifiers
        let mut predictions = Array2::zeros((n_samples, n_codes));

        for (j, classifier) in self.classifiers.iter().enumerate() {
            let binary_pred = classifier.predict(x)?;
            for i in 0..n_samples {
                predictions[[i, j]] = binary_pred[i];
            }
        }

        // Decode predictions using error correction
        let mut final_predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample_code: Vec<i32> = predictions.row(i).iter().cloned().collect();

            // Find the closest class code
            let mut min_distance = Float::INFINITY;
            let mut best_class_idx = 0;

            for class_idx in 0..n_classes {
                let class_code: Vec<i32> =
                    self.code_matrix.row(class_idx).iter().cloned().collect();
                let distance = self.code_distance(&sample_code, &class_code);

                if distance < min_distance {
                    min_distance = distance;
                    best_class_idx = class_idx;
                }
            }

            final_predictions[i] = self.classes[best_class_idx];
        }

        Ok(final_predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedErrorCorrectingOutputCodes {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_codes = self.classifiers.len();
        let n_classes = self.classes.len();

        // Get probability predictions from all binary classifiers
        let mut predictions = Array2::zeros((n_samples, n_codes));

        for (j, classifier) in self.classifiers.iter().enumerate() {
            let binary_proba = classifier.predict_proba(x)?;
            // Use probability of positive class (class 1)
            for i in 0..n_samples {
                predictions[[i, j]] = binary_proba[[i, 1]];
            }
        }

        // Convert probabilities to soft codes and decode
        let mut final_probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let sample_proba: Vec<Float> = predictions.row(i).iter().cloned().collect();

            // Calculate soft distance to each class code
            let mut distances = Vec::with_capacity(n_classes);

            for class_idx in 0..n_classes {
                let class_code: Vec<i32> =
                    self.code_matrix.row(class_idx).iter().cloned().collect();

                // Soft distance calculation
                let distance: Float = sample_proba
                    .iter()
                    .zip(class_code.iter())
                    .map(|(&p, &c)| {
                        if c == 1 {
                            (1.0 - p).powi(2) // Distance to +1
                        } else {
                            p.powi(2) // Distance to -1
                        }
                    })
                    .sum();

                distances.push(distance);
            }

            // Convert distances to probabilities (inverse distance weighting)
            let max_distance = distances.iter().cloned().fold(0.0, Float::max);
            let mut inv_distances: Vec<Float> = distances
                .iter()
                .map(|&d| (max_distance - d + 1e-8).max(1e-8))
                .collect();

            // Normalize to probabilities
            let sum: Float = inv_distances.iter().sum();
            for prob in &mut inv_distances {
                *prob /= sum;
            }

            for (j, &prob) in inv_distances.iter().enumerate() {
                final_probas[[i, j]] = prob;
            }
        }

        Ok(final_probas)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Axis};

    #[test]
    fn test_error_correcting_output_codes_basic() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let ecoc = ErrorCorrectingOutputCodes::new()
            .code_method("random")
            .n_codes(5);

        let fitted = ecoc.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 3);
        assert_eq!(fitted.n_classifiers(), 5);
    }

    #[test]
    fn test_error_correcting_output_codes_predict_proba() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let ecoc = ErrorCorrectingOutputCodes::new()
            .code_method("dense_random")
            .n_codes(4);

        let fitted = ecoc.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (6, 3));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_different_code_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let methods = ["random", "dense_random", "sparse_random"];

        for method in &methods {
            let ecoc = ErrorCorrectingOutputCodes::new()
                .code_method(method)
                .n_codes(3);

            let fitted = ecoc.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 4);
            assert_eq!(fitted.classes().len(), 2);
        }
    }

    #[test]
    fn test_exhaustive_code_method() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let ecoc = ErrorCorrectingOutputCodes::new().code_method("exhaustive");

        let fitted = ecoc.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_code_matrix_properties() {
        let ecoc = ErrorCorrectingOutputCodes::new()
            .code_method("random")
            .n_codes(4)
            .random_state(42);

        let code_matrix = ecoc.generate_code_matrix(3).unwrap();

        assert_eq!(code_matrix.dim(), (3, 4));

        // Check that each code has both +1 and -1 values
        for j in 0..4 {
            let col = code_matrix.column(j);
            let has_pos = col.iter().any(|&x| x == 1);
            let has_neg = col.iter().any(|&x| x == -1);
            assert!(
                has_pos && has_neg,
                "Code {} should have both +1 and -1 values",
                j
            );
        }
    }

    #[test]
    fn test_different_correction_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let methods = ["hamming", "euclidean", "manhattan"];

        for method in &methods {
            let ecoc = ErrorCorrectingOutputCodes::new()
                .correction_method(method)
                .n_codes(3);

            let fitted = ecoc.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 4);
            assert_eq!(fitted.classes().len(), 2);
        }
    }
}
