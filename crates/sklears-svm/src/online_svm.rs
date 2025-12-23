//! Online SVM algorithms for streaming data and incremental learning
//!
//! This module provides online learning algorithms for SVMs that can process
//! data points one at a time or in small batches, making them suitable for
//! streaming applications and large-scale datasets that don't fit in memory.

use crate::kernels::Kernel;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{error::Result, types::Float};
use std::collections::VecDeque;

/// Configuration for online SVM algorithms
#[derive(Debug, Clone)]
pub struct OnlineSvmConfig {
    /// Regularization parameter
    pub c: Float,
    /// Learning rate for online updates
    pub learning_rate: Float,
    /// Learning rate decay factor
    pub learning_rate_decay: Float,
    /// Minimum learning rate
    pub min_learning_rate: Float,
    /// Maximum number of support vectors to maintain
    pub max_support_vectors: usize,
    /// Budget for removing old support vectors
    pub budget_strategy: BudgetStrategy,
    /// Tolerance for support vector removal
    pub removal_tolerance: Float,
    /// Number of samples to average for gradient estimation
    pub gradient_window_size: usize,
}

/// Strategy for managing support vector budget
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetStrategy {
    /// Remove oldest support vectors
    Fifo,
    /// Remove support vectors with smallest coefficients
    MinimumCoeff,
    /// Remove support vectors that contribute least to margin
    MinimumMargin,
    /// Merge similar support vectors
    Merging,
}

impl Default for OnlineSvmConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            learning_rate: 0.01,
            learning_rate_decay: 0.99,
            min_learning_rate: 1e-6,
            max_support_vectors: 1000,
            budget_strategy: BudgetStrategy::MinimumCoeff,
            removal_tolerance: 1e-6,
            gradient_window_size: 100,
        }
    }
}

/// Online SVM classifier for streaming data
pub struct OnlineSvm<K: Kernel> {
    config: OnlineSvmConfig,
    kernel: K,
    support_vectors: Array2<Float>,
    support_labels: Array1<Float>,
    alpha: Array1<Float>,
    bias: Float,
    n_samples_seen: usize,
    current_learning_rate: Float,
    gradient_buffer: VecDeque<Float>,
}

impl<K: Kernel> OnlineSvm<K> {
    /// Create a new online SVM
    pub fn new(kernel: K, config: OnlineSvmConfig) -> Self {
        Self {
            current_learning_rate: config.learning_rate,
            config,
            kernel,
            support_vectors: Array2::zeros((0, 0)),
            support_labels: Array1::zeros(0),
            alpha: Array1::zeros(0),
            bias: 0.0,
            n_samples_seen: 0,
            gradient_buffer: VecDeque::new(),
        }
    }

    /// Process a single sample
    pub fn partial_fit(&mut self, x: &Array1<Float>, y: Float) -> Result<()> {
        if self.support_vectors.nrows() == 0 {
            // Initialize with first sample
            self.initialize_with_sample(x, y)?;
            self.n_samples_seen += 1;
            return Ok(());
        }

        // Compute current prediction
        let decision_value = self.decision_function_single(x)?;
        let margin = y * decision_value;

        // Check if sample violates margin
        if margin < 1.0 {
            // Update model with new sample
            self.add_support_vector(x.clone(), y)?;

            // Update existing support vector weights
            let loss_gradient = if margin < 0.0 { 1.0 } else { 1.0 - margin };
            self.update_weights(x, y, loss_gradient)?;
        }

        // Update learning rate
        self.update_learning_rate();

        // Maintain budget
        if self.support_vectors.nrows() > self.config.max_support_vectors {
            self.maintain_budget()?;
        }

        self.n_samples_seen += 1;
        Ok(())
    }

    /// Process a batch of samples
    pub fn partial_fit_batch(&mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<()> {
        assert_eq!(
            x.nrows(),
            y.len(),
            "Number of samples in X and y must match"
        );

        for (sample, &label) in x.axis_iter(Axis(0)).zip(y.iter()) {
            self.partial_fit(&sample.to_owned(), label)?;
        }
        Ok(())
    }

    /// Initialize model with first sample
    fn initialize_with_sample(&mut self, x: &Array1<Float>, y: Float) -> Result<()> {
        let n_features = x.len();
        self.support_vectors = Array2::zeros((1, n_features));
        self.support_vectors.row_mut(0).assign(x);
        self.support_labels = Array1::from_vec(vec![y]);
        self.alpha = Array1::from_vec(vec![self.config.c]);
        self.bias = 0.0;
        Ok(())
    }

    /// Add a new support vector
    fn add_support_vector(&mut self, x: Array1<Float>, y: Float) -> Result<()> {
        let n_features = x.len();
        let current_n_sv = self.support_vectors.nrows();

        // Expand arrays
        let mut new_support_vectors = Array2::zeros((current_n_sv + 1, n_features));
        let mut new_support_labels = Array1::zeros(current_n_sv + 1);
        let mut new_alpha = Array1::zeros(current_n_sv + 1);

        // Copy existing data
        if current_n_sv > 0 {
            new_support_vectors
                .slice_mut(s![0..current_n_sv, ..])
                .assign(&self.support_vectors);
            new_support_labels
                .slice_mut(s![0..current_n_sv])
                .assign(&self.support_labels);
            new_alpha.slice_mut(s![0..current_n_sv]).assign(&self.alpha);
        }

        // Add new sample
        new_support_vectors.row_mut(current_n_sv).assign(&x);
        new_support_labels[current_n_sv] = y;
        new_alpha[current_n_sv] = 0.0; // Will be updated by weight update

        self.support_vectors = new_support_vectors;
        self.support_labels = new_support_labels;
        self.alpha = new_alpha;

        Ok(())
    }

    /// Update support vector weights using online learning
    fn update_weights(&mut self, x: &Array1<Float>, y: Float, loss_gradient: Float) -> Result<()> {
        let n_sv = self.support_vectors.nrows();

        // Compute kernel values between new sample and all support vectors
        let mut kernels = Array1::zeros(n_sv);
        for i in 0..n_sv {
            kernels[i] = self.kernel.compute(x.view(), self.support_vectors.row(i));
        }

        // Update alpha values
        for i in 0..n_sv {
            let gradient = -y * self.support_labels[i] * kernels[i] * loss_gradient;
            let update = self.current_learning_rate * gradient;
            self.alpha[i] = (self.alpha[i] - update).max(0.0).min(self.config.c);
        }

        // Update newest support vector (if it exists)
        if n_sv > 0 {
            let newest_idx = n_sv - 1;
            self.alpha[newest_idx] = self.current_learning_rate * self.config.c * loss_gradient;
            self.alpha[newest_idx] = self.alpha[newest_idx].min(self.config.c);
        }

        // Update bias using running average
        let bias_gradient = -y * loss_gradient;
        self.bias -= self.current_learning_rate * bias_gradient;

        Ok(())
    }

    /// Update learning rate with decay
    fn update_learning_rate(&mut self) {
        self.current_learning_rate *= self.config.learning_rate_decay;
        self.current_learning_rate = self
            .current_learning_rate
            .max(self.config.min_learning_rate);
    }

    /// Maintain support vector budget by removing less important vectors
    fn maintain_budget(&mut self) -> Result<()> {
        let target_size = (self.config.max_support_vectors as f64 * 0.8) as usize;
        let n_to_remove = self.support_vectors.nrows() - target_size;

        match self.config.budget_strategy {
            BudgetStrategy::Fifo => self.remove_oldest(n_to_remove)?,
            BudgetStrategy::MinimumCoeff => self.remove_minimum_coefficients(n_to_remove)?,
            BudgetStrategy::MinimumMargin => self.remove_minimum_margin(n_to_remove)?,
            BudgetStrategy::Merging => self.merge_similar_vectors(n_to_remove)?,
        }

        Ok(())
    }

    /// Remove oldest support vectors
    fn remove_oldest(&mut self, n_to_remove: usize) -> Result<()> {
        let n_sv = self.support_vectors.nrows();
        let n_keep = n_sv - n_to_remove;

        if n_keep > 0 {
            let new_support_vectors = self.support_vectors.slice(s![n_to_remove.., ..]).to_owned();
            let new_support_labels = self.support_labels.slice(s![n_to_remove..]).to_owned();
            let new_alpha = self.alpha.slice(s![n_to_remove..]).to_owned();

            self.support_vectors = new_support_vectors;
            self.support_labels = new_support_labels;
            self.alpha = new_alpha;
        }

        Ok(())
    }

    /// Remove support vectors with smallest coefficients
    fn remove_minimum_coefficients(&mut self, n_to_remove: usize) -> Result<()> {
        let n_sv = self.support_vectors.nrows();

        // Create indices sorted by alpha values (ascending)
        let mut indices: Vec<usize> = (0..n_sv).collect();
        indices.sort_by(|&a, &b| self.alpha[a].partial_cmp(&self.alpha[b]).unwrap());

        // Keep the largest coefficients
        let keep_indices: Vec<usize> = indices.into_iter().skip(n_to_remove).collect();

        self.reorder_support_vectors(&keep_indices)?;
        Ok(())
    }

    /// Remove support vectors that contribute least to margin
    fn remove_minimum_margin(&mut self, n_to_remove: usize) -> Result<()> {
        let n_sv = self.support_vectors.nrows();
        let mut margins = Vec::with_capacity(n_sv);

        // Compute margin contribution for each support vector
        for i in 0..n_sv {
            let x_i = self.support_vectors.row(i);
            let decision = self.decision_function_single(&x_i.to_owned())?;
            let margin = self.support_labels[i] * decision;
            margins.push((i, margin));
        }

        // Sort by margin (ascending) and keep the ones with larger margins
        margins.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let keep_indices: Vec<usize> = margins
            .into_iter()
            .skip(n_to_remove)
            .map(|(idx, _)| idx)
            .collect();

        self.reorder_support_vectors(&keep_indices)?;
        Ok(())
    }

    /// Merge similar support vectors to reduce model size
    fn merge_similar_vectors(&mut self, n_to_remove: usize) -> Result<()> {
        let n_sv = self.support_vectors.nrows();
        let mut merged_count = 0;
        let mut keep_mask = vec![true; n_sv];

        for i in 0..n_sv {
            if !keep_mask[i] || merged_count >= n_to_remove {
                continue;
            }

            #[allow(clippy::needless_range_loop)]
            for j in (i + 1)..n_sv {
                if !keep_mask[j] {
                    continue;
                }

                // Check if vectors are similar
                let kernel_val = self
                    .kernel
                    .compute(self.support_vectors.row(i), self.support_vectors.row(j));

                if kernel_val > 0.9 && self.support_labels[i] == self.support_labels[j] {
                    // Merge j into i
                    self.alpha[i] += self.alpha[j];
                    self.alpha[i] = self.alpha[i].min(self.config.c);
                    keep_mask[j] = false;
                    merged_count += 1;

                    if merged_count >= n_to_remove {
                        break;
                    }
                }
            }
        }

        // Create new arrays with kept vectors
        let keep_indices: Vec<usize> = keep_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();

        self.reorder_support_vectors(&keep_indices)?;
        Ok(())
    }

    /// Reorder support vectors based on given indices
    fn reorder_support_vectors(&mut self, keep_indices: &[usize]) -> Result<()> {
        let n_keep = keep_indices.len();
        let n_features = self.support_vectors.ncols();

        let mut new_support_vectors = Array2::zeros((n_keep, n_features));
        let mut new_support_labels = Array1::zeros(n_keep);
        let mut new_alpha = Array1::zeros(n_keep);

        for (new_idx, &old_idx) in keep_indices.iter().enumerate() {
            new_support_vectors
                .row_mut(new_idx)
                .assign(&self.support_vectors.row(old_idx));
            new_support_labels[new_idx] = self.support_labels[old_idx];
            new_alpha[new_idx] = self.alpha[old_idx];
        }

        self.support_vectors = new_support_vectors;
        self.support_labels = new_support_labels;
        self.alpha = new_alpha;

        Ok(())
    }

    /// Compute decision function for a single sample
    pub fn decision_function_single(&self, x: &Array1<Float>) -> Result<Float> {
        let mut decision = self.bias;

        for i in 0..self.support_vectors.nrows() {
            let kernel_val = self.kernel.compute(x.view(), self.support_vectors.row(i));
            decision += self.alpha[i] * self.support_labels[i] * kernel_val;
        }

        Ok(decision)
    }

    /// Compute decision function for multiple samples
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let mut decisions = Array1::zeros(n_samples);

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            decisions[i] = self.decision_function_single(&sample.to_owned())?;
        }

        Ok(decisions)
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let decisions = self.decision_function(x)?;
        Ok(decisions.map(|&d| if d >= 0.0 { 1.0 } else { -1.0 }))
    }

    /// Predict single sample
    pub fn predict_single(&self, x: &Array1<Float>) -> Result<Float> {
        let decision = self.decision_function_single(x)?;
        Ok(if decision >= 0.0 { 1.0 } else { -1.0 })
    }

    /// Get current model statistics
    pub fn get_stats(&self) -> OnlineSvmStats {
        OnlineSvmStats {
            n_support_vectors: self.support_vectors.nrows(),
            n_samples_seen: self.n_samples_seen,
            current_learning_rate: self.current_learning_rate,
            average_alpha: if !self.alpha.is_empty() {
                self.alpha.mean().unwrap_or(0.0)
            } else {
                0.0
            },
            bias: self.bias,
        }
    }

    /// Reset the model to initial state
    pub fn reset(&mut self) {
        self.support_vectors = Array2::zeros((0, 0));
        self.support_labels = Array1::zeros(0);
        self.alpha = Array1::zeros(0);
        self.bias = 0.0;
        self.n_samples_seen = 0;
        self.current_learning_rate = self.config.learning_rate;
        self.gradient_buffer.clear();
    }
}

/// Statistics about the online SVM model
#[derive(Debug, Clone)]
pub struct OnlineSvmStats {
    pub n_support_vectors: usize,
    pub n_samples_seen: usize,
    pub current_learning_rate: Float,
    pub average_alpha: Float,
    pub bias: Float,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RbfKernel;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_online_svm_basic() {
        let kernel = RbfKernel::new(1.0);
        let config = OnlineSvmConfig::default();
        let mut online_svm = OnlineSvm::new(kernel, config);

        // Train on some samples
        let samples = [
            (array![1.0, 2.0], 1.0),
            (array![2.0, 3.0], 1.0),
            (array![-1.0, -2.0], -1.0),
            (array![-2.0, -3.0], -1.0),
        ];

        for (x, y) in &samples {
            online_svm.partial_fit(x, *y).unwrap();
        }

        // Check predictions
        let test_x = array![[1.5, 2.5], [-1.5, -2.5]];
        let predictions = online_svm.predict(&test_x).unwrap();

        assert_eq!(predictions[0], 1.0);
        assert_eq!(predictions[1], -1.0);
    }

    #[test]
    fn test_online_svm_batch() {
        let kernel = RbfKernel::new(1.0);
        let config = OnlineSvmConfig::default();
        let mut online_svm = OnlineSvm::new(kernel, config);

        let x = array![[1.0, 2.0], [2.0, 3.0], [-1.0, -2.0], [-2.0, -3.0]];
        let y = array![1.0, 1.0, -1.0, -1.0];

        online_svm.partial_fit_batch(&x, &y).unwrap();

        let stats = online_svm.get_stats();
        assert!(stats.n_support_vectors > 0);
        assert_eq!(stats.n_samples_seen, 4);
    }

    #[test]
    fn test_budget_maintenance() {
        let kernel = RbfKernel::new(1.0);
        let mut config = OnlineSvmConfig::default();
        config.max_support_vectors = 3;
        let mut online_svm = OnlineSvm::new(kernel, config);

        // Add more samples than the budget allows
        for i in 0..10 {
            let x = array![i as Float, (i * 2) as Float];
            let y = if i % 2 == 0 { 1.0 } else { -1.0 };
            online_svm.partial_fit(&x, y).unwrap();
        }

        let stats = online_svm.get_stats();
        assert!(stats.n_support_vectors <= 3);
    }

    #[test]
    fn test_learning_rate_decay() {
        let kernel = RbfKernel::new(1.0);
        let mut config = OnlineSvmConfig::default();
        config.learning_rate = 0.1;
        config.learning_rate_decay = 0.9;
        let mut online_svm = OnlineSvm::new(kernel, config);

        let initial_lr = online_svm.current_learning_rate;

        // Process some samples
        for i in 0..5 {
            let x = array![i as Float, 0.0];
            let y = 1.0;
            online_svm.partial_fit(&x, y).unwrap();
        }

        assert!(online_svm.current_learning_rate < initial_lr);
    }

    #[test]
    fn test_reset() {
        let kernel = RbfKernel::new(1.0);
        let config = OnlineSvmConfig::default();
        let mut online_svm = OnlineSvm::new(kernel, config);

        // Train on some data
        online_svm.partial_fit(&array![1.0, 2.0], 1.0).unwrap();
        assert!(online_svm.get_stats().n_support_vectors > 0);

        // Reset
        online_svm.reset();
        let stats = online_svm.get_stats();
        assert_eq!(stats.n_support_vectors, 0);
        assert_eq!(stats.n_samples_seen, 0);
    }
}
