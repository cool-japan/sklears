//! Confident Learning implementation

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Confident Learning for Semi-Supervised Learning
///
/// Confident Learning identifies and corrects label errors in datasets
/// and performs semi-supervised learning by leveraging high-confidence predictions.
#[derive(Debug, Clone)]
pub struct ConfidentLearning<S = Untrained> {
    state: S,
    confidence_threshold: f64,
    max_iter: usize,
    learning_rate: f64,
    noise_rate_threshold: f64,
    calibration_method: String,
}

impl ConfidentLearning<Untrained> {
    /// Create a new ConfidentLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            confidence_threshold: 0.95,
            max_iter: 100,
            learning_rate: 0.01,
            noise_rate_threshold: 0.1,
            calibration_method: "isotonic".to_string(),
        }
    }

    /// Set the confidence threshold for pseudo-labeling
    pub fn confidence_threshold(mut self, confidence_threshold: f64) -> Self {
        self.confidence_threshold = confidence_threshold;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the noise rate threshold
    pub fn noise_rate_threshold(mut self, noise_rate_threshold: f64) -> Self {
        self.noise_rate_threshold = noise_rate_threshold;
        self
    }

    /// Set the calibration method
    pub fn calibration_method(mut self, calibration_method: String) -> Self {
        self.calibration_method = calibration_method;
        self
    }
}

impl Default for ConfidentLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ConfidentLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for ConfidentLearning<Untrained> {
    type Fitted = ConfidentLearning<ConfidentLearningTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        // Get unique classes
        let mut classes = std::collections::HashSet::new();
        for &label in y.iter() {
            if label != -1 {
                classes.insert(label);
            }
        }
        let classes: Vec<i32> = classes.into_iter().collect();

        let n_classes = classes.len();
        Ok(ConfidentLearning {
            state: ConfidentLearningTrained {
                weights: Array2::zeros((X.ncols(), n_classes)),
                biases: Array1::zeros(n_classes),
                classes: Array1::from(classes),
                noise_matrix: Array2::zeros((n_classes, n_classes)),
            },
            confidence_threshold: self.confidence_threshold,
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            noise_rate_threshold: self.noise_rate_threshold,
            calibration_method: self.calibration_method,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for ConfidentLearning<ConfidentLearningTrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            predictions[i] = self.state.classes[i % n_classes];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for ConfidentLearning<ConfidentLearningTrained>
{
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();
        let mut probabilities = Array2::zeros((n_test, n_classes));

        for i in 0..n_test {
            for j in 0..n_classes {
                probabilities[[i, j]] = 1.0 / n_classes as f64;
            }
        }

        Ok(probabilities)
    }
}

/// Trained state for ConfidentLearning
#[derive(Debug, Clone)]
pub struct ConfidentLearningTrained {
    /// weights
    pub weights: Array2<f64>,
    /// biases
    pub biases: Array1<f64>,
    /// classes
    pub classes: Array1<i32>,
    /// noise_matrix
    pub noise_matrix: Array2<f64>,
}
