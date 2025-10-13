//! Crammer-Singer multi-class SVM implementation
//!
//! This module implements the Crammer-Singer approach for multi-class SVMs,
//! which formulates the multi-class problem as a single optimization problem
//! rather than decomposing it into binary sub-problems.

use crate::kernels::Kernel;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use sklears_core::{
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Crammer-Singer SVM
#[derive(Debug, Clone)]
pub struct CrammerSingerConfig {
    /// Regularization parameter
    pub c: Float,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Learning rate for gradient descent
    pub learning_rate: Float,
    /// Decay factor for learning rate
    pub lr_decay: Float,
    /// Early stopping tolerance
    pub early_stopping_tol: Float,
}

impl Default for CrammerSingerConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            tol: 1e-3,
            max_iter: 1000,
            learning_rate: 0.01,
            lr_decay: 0.99,
            early_stopping_tol: 1e-6,
        }
    }
}

/// Crammer-Singer multi-class SVM
#[derive(Debug)]
pub struct CrammerSingerSVM<K: Kernel, State = Untrained> {
    config: CrammerSingerConfig,
    kernel: K,
    state: PhantomData<State>,
    // Fitted parameters
    w_: Option<Array2<Float>>, // Weight matrix (n_features × n_classes)
    classes_: Option<Array1<Float>>,
    n_features_in_: Option<usize>,
    training_x_: Option<Array2<Float>>, // Store training data for kernel SVMs
}

impl<K: Kernel> CrammerSingerSVM<K, Untrained> {
    /// Create a new Crammer-Singer SVM
    pub fn new(kernel: K) -> Self {
        Self {
            config: CrammerSingerConfig::default(),
            kernel,
            state: PhantomData,
            w_: None,
            classes_: None,
            n_features_in_: None,
            training_x_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Build the classifier
    pub fn build(self) -> Self {
        self
    }

    /// Find unique classes and create mapping
    fn prepare_classes(&self, y: &Array1<Float>) -> Array1<Float> {
        let mut classes: Vec<Float> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();

        Array1::from_vec(classes)
    }

    /// Find class index in the classes array
    fn find_class_index(&self, classes: &Array1<Float>, class_label: Float) -> Option<usize> {
        classes
            .iter()
            .position(|&x| (x - class_label).abs() < Float::EPSILON)
    }

    /// Convert class labels to one-hot encoding
    fn create_one_hot(&self, y: &Array1<Float>, classes: &Array1<Float>) -> Array2<Float> {
        let n_samples = y.len();
        let n_classes = classes.len();
        let mut one_hot = Array2::zeros((n_samples, n_classes));

        for (i, &label) in y.iter().enumerate() {
            if let Some(class_idx) = self.find_class_index(classes, label) {
                one_hot[[i, class_idx]] = 1.0;
            }
        }

        one_hot
    }

    /// Compute kernel matrix
    fn compute_kernel_matrix(&self, x: &Array2<Float>) -> Array2<Float> {
        self.kernel.compute_matrix(x, x)
    }

    /// Compute scores for Crammer-Singer formulation
    fn compute_scores(
        &self,
        k_matrix: &Array2<Float>,
        w: &Array2<Float>,
        sample_idx: usize,
    ) -> Array1<Float> {
        let n_classes = w.ncols();
        let mut scores = Array1::zeros(n_classes);

        for c in 0..n_classes {
            let mut score = 0.0;
            for j in 0..k_matrix.ncols() {
                score += w[[j, c]] * k_matrix[[sample_idx, j]];
            }
            scores[c] = score;
        }

        scores
    }

    /// Compute loss and gradients for Crammer-Singer formulation
    fn compute_loss_and_gradients(
        &self,
        k_matrix: &Array2<Float>,
        w: &Array2<Float>,
        y_one_hot: &Array2<Float>,
    ) -> (Float, Array2<Float>) {
        let (n_samples, n_classes) = y_one_hot.dim();
        let mut total_loss = 0.0;
        let mut gradients = Array2::zeros(w.dim());

        // Add regularization term to loss
        let mut reg_loss = 0.0;
        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                reg_loss += w[[i, j]] * w[[i, j]];
            }
        }
        total_loss += 0.5 * reg_loss / self.config.c;

        // Compute hinge loss and gradients
        for i in 0..n_samples {
            let scores = self.compute_scores(k_matrix, w, i);
            let true_class = y_one_hot
                .row(i)
                .iter()
                .enumerate()
                .find(|(_, &val)| val > 0.5)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Find the highest scoring incorrect class
            let mut max_score = Float::NEG_INFINITY;
            let mut max_class = 0;
            for c in 0..n_classes {
                if c != true_class {
                    let margin_score = scores[c] - scores[true_class] + 1.0;
                    if margin_score > max_score {
                        max_score = margin_score;
                        max_class = c;
                    }
                }
            }

            // Compute hinge loss
            let loss = (scores[max_class] - scores[true_class] + 1.0).max(0.0);
            total_loss += loss;

            // Compute gradients if loss > 0
            if loss > 0.0 {
                for j in 0..k_matrix.ncols() {
                    // Gradient w.r.t. true class weights (negative)
                    gradients[[j, true_class]] -= k_matrix[[i, j]];
                    // Gradient w.r.t. max incorrect class weights (positive)
                    gradients[[j, max_class]] += k_matrix[[i, j]];
                }
            }
        }

        // Add regularization gradients
        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                gradients[[i, j]] += w[[i, j]] / self.config.c;
            }
        }

        (total_loss / n_samples as Float, gradients)
    }
}

impl<K: Kernel> CrammerSingerSVM<K, Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<Float> {
        self.classes_
            .as_ref()
            .expect("CrammerSingerSVM should be fitted")
    }

    /// Get the weight matrix
    pub fn weights(&self) -> &Array2<Float> {
        self.w_.as_ref().expect("CrammerSingerSVM should be fitted")
    }

    /// Compute decision scores for samples
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_.unwrap() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_.unwrap(),
                actual: n_features,
            });
        }

        let training_x = self.training_x_.as_ref().unwrap();
        let w = self.weights();
        let k_cross = self.kernel.compute_matrix(x, training_x);

        let n_classes = w.ncols();
        let mut scores = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for c in 0..n_classes {
                let mut score = 0.0;
                for j in 0..training_x.nrows() {
                    score += w[[j, c]] * k_cross[[i, j]];
                }
                scores[[i, c]] = score;
            }
        }

        Ok(scores)
    }
}

impl<K: Kernel> Fit<Array2<Float>, Array1<Float>> for CrammerSingerSVM<K, Untrained> {
    type Fitted = CrammerSingerSVM<K, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit CrammerSingerSVM on empty dataset".to_string(),
            ));
        }

        // Prepare classes
        let classes = self.prepare_classes(y);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "CrammerSingerSVM requires at least 2 classes".to_string(),
            ));
        }

        // Create one-hot encoding of labels
        let y_one_hot = self.create_one_hot(y, &classes);

        // Compute kernel matrix
        let k_matrix = self.compute_kernel_matrix(x);

        // Initialize weights
        let mut w = Array2::zeros((n_samples, n_classes));

        // Training loop using gradient descent
        let mut learning_rate = self.config.learning_rate;
        let mut prev_loss = Float::INFINITY;

        for iter in 0..self.config.max_iter {
            // Compute loss and gradients
            let (loss, gradients) = self.compute_loss_and_gradients(&k_matrix, &w, &y_one_hot);

            // Check for convergence
            if iter > 0 && (prev_loss - loss).abs() < self.config.early_stopping_tol {
                break;
            }

            // Update weights using gradient descent
            for i in 0..w.nrows() {
                for j in 0..w.ncols() {
                    w[[i, j]] -= learning_rate * gradients[[i, j]];
                }
            }

            // Decay learning rate
            learning_rate *= self.config.lr_decay;
            prev_loss = loss;

            // Print progress occasionally
            if iter % 100 == 0 {
                println!("Iteration {}: Loss = {:.6}", iter, loss);
            }
        }

        Ok(CrammerSingerSVM {
            config: self.config,
            kernel: self.kernel,
            state: PhantomData,
            w_: Some(w),
            classes_: Some(classes),
            n_features_in_: Some(n_features),
            training_x_: Some(x.clone()),
        })
    }
}

impl<K: Kernel> Predict<Array2<Float>, Array1<Float>> for CrammerSingerSVM<K, Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let scores = self.decision_function(x)?;
        let (n_samples, _) = scores.dim();
        let classes = self.classes();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Find class with highest score
            let mut max_score = Float::NEG_INFINITY;
            let mut best_class_idx = 0;

            for j in 0..scores.ncols() {
                if scores[[i, j]] > max_score {
                    max_score = scores[[i, j]];
                    best_class_idx = j;
                }
            }

            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::LinearKernel;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_crammer_singer_svm_basic() {
        // Create a simple 3-class dataset
        let x = array![
            [1.0, 1.0], // Class 0
            [1.1, 1.1], // Class 0
            [5.0, 5.0], // Class 1
            [5.1, 5.1], // Class 1
            [9.0, 9.0], // Class 2
            [9.1, 9.1], // Class 2
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let svm = CrammerSingerSVM::new(LinearKernel::new())
            .c(1.0)
            .tol(1e-2)
            .max_iter(100)
            .learning_rate(0.1)
            .build()
            .fit(&x, &y)
            .unwrap();

        assert_eq!(svm.classes().len(), 3);

        // Test prediction
        let x_test = array![
            [1.0, 1.0], // Should be class 0
            [5.0, 5.0], // Should be class 1
            [9.0, 9.0], // Should be class 2
        ];
        let predictions = svm.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 3);

        // Test decision function
        let scores = svm.decision_function(&x_test).unwrap();
        assert_eq!(scores.dim(), (3, 3)); // 3 samples, 3 classes
    }

    #[test]
    fn test_crammer_singer_binary_case() {
        // Test binary classification
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]];
        let y = array![1.0, 1.0, 0.0, 0.0];

        let svm = CrammerSingerSVM::new(LinearKernel::new())
            .c(1.0)
            .max_iter(50)
            .build()
            .fit(&x, &y)
            .unwrap();

        assert_eq!(svm.classes().len(), 2);

        let x_test = array![[1.5, 1.5], [-1.5, -1.5]];
        let predictions = svm.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_crammer_singer_invalid_input() {
        let x = array![[1.0, 1.0], [2.0, 2.0]];
        let y = array![1.0]; // Wrong size

        let svm = CrammerSingerSVM::new(LinearKernel::new());
        let result = svm.fit(&x, &y);
        assert!(result.is_err());
    }
}
