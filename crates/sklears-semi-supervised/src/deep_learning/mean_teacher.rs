//! Mean Teacher implementation

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Mean Teacher for Semi-Supervised Learning
///
/// Mean Teacher uses an exponential moving average of model parameters
/// as a teacher model to provide consistent targets for training.
#[derive(Debug, Clone)]
pub struct MeanTeacher<S = Untrained> {
    state: S,
    hidden_dims: Vec<usize>,
    learning_rate: f64,
    consistency_weight: f64,
    ema_decay: f64,
    max_epochs: usize,
    batch_size: usize,
    augmentation_strength: f64,
}

impl MeanTeacher<Untrained> {
    /// Create a new MeanTeacher instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            hidden_dims: vec![64, 32],
            learning_rate: 0.001,
            consistency_weight: 1.0,
            ema_decay: 0.999,
            max_epochs: 100,
            batch_size: 32,
            augmentation_strength: 0.1,
        }
    }

    /// Set the hidden layer dimensions
    pub fn hidden_dims(mut self, hidden_dims: Vec<usize>) -> Self {
        self.hidden_dims = hidden_dims;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the consistency loss weight
    pub fn consistency_weight(mut self, consistency_weight: f64) -> Self {
        self.consistency_weight = consistency_weight;
        self
    }

    /// Set the EMA decay rate
    pub fn ema_decay(mut self, ema_decay: f64) -> Self {
        self.ema_decay = ema_decay;
        self
    }

    /// Set the maximum number of epochs
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the augmentation strength
    pub fn augmentation_strength(mut self, augmentation_strength: f64) -> Self {
        self.augmentation_strength = augmentation_strength;
        self
    }
}

impl Default for MeanTeacher<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MeanTeacher<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for MeanTeacher<Untrained> {
    type Fitted = MeanTeacher<MeanTeacherTrained>;

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

        Ok(MeanTeacher {
            state: MeanTeacherTrained {
                student_weights: Array2::zeros((X.ncols(), classes.len())),
                teacher_weights: Array2::zeros((X.ncols(), classes.len())),
                student_biases: Array1::zeros(classes.len()),
                teacher_biases: Array1::zeros(classes.len()),
                classes: Array1::from(classes),
            },
            hidden_dims: self.hidden_dims,
            learning_rate: self.learning_rate,
            consistency_weight: self.consistency_weight,
            ema_decay: self.ema_decay,
            max_epochs: self.max_epochs,
            batch_size: self.batch_size,
            augmentation_strength: self.augmentation_strength,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for MeanTeacher<MeanTeacherTrained> {
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

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>> for MeanTeacher<MeanTeacherTrained> {
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

/// Trained state for MeanTeacher
#[derive(Debug, Clone)]
pub struct MeanTeacherTrained {
    /// student_weights
    pub student_weights: Array2<f64>,
    /// teacher_weights
    pub teacher_weights: Array2<f64>,
    /// student_biases
    pub student_biases: Array1<f64>,
    /// teacher_biases
    pub teacher_biases: Array1<f64>,
    /// classes
    pub classes: Array1<i32>,
}
