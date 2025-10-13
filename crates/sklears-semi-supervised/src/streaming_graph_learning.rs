//! Streaming Graph Learning for Dynamic Semi-Supervised Learning
//!
//! This module provides algorithms for learning and updating graph structures
//! incrementally as new data arrives in streaming scenarios.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};
use std::collections::{HashMap, VecDeque};

/// Streaming Graph Learning for Dynamic Semi-Supervised Learning
///
/// This method continuously updates graph structures as new data points arrive,
/// making it suitable for dynamic environments where the data distribution
/// may change over time. It maintains a sliding window of recent data points
/// and efficiently updates the graph structure and label propagation.
///
/// # Parameters
///
/// * `window_size` - Size of the sliding window for maintaining recent data
/// * `lambda_sparse` - Sparsity regularization parameter for graph learning
/// * `alpha_decay` - Decay factor for edge weights over time
/// * `update_frequency` - Frequency of full graph reconstruction
/// * `forgetting_factor` - Factor for exponential forgetting of old connections
/// * `adaptive_threshold` - Whether to use adaptive thresholds for edge addition
/// * `min_samples_update` - Minimum samples required before updating the graph
///
/// # Examples
///
/// ```
/// use scirs2_core::array;
/// use sklears_semi_supervised::StreamingGraphLearning;
/// use sklears_core::traits::{Predict, Fit};
///
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
/// let y = array![0, 1, -1, -1]; // -1 indicates unlabeled
///
/// let mut sgl = StreamingGraphLearning::new()
///     .window_size(100)
///     .lambda_sparse(0.1)
///     .alpha_decay(0.95);
///
/// let mut fitted = sgl.fit(&X.view(), &y.view()).unwrap();
/// let predictions = fitted.predict(&X.view()).unwrap();
///
/// // Update with new data
/// let X_new = array![[5.0, 6.0], [6.0, 7.0]];
/// let y_new = array![-1, 0];
/// let updated = fitted.update(&X_new.view(), &y_new.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StreamingGraphLearning<S = Untrained> {
    state: S,
    window_size: usize,
    lambda_sparse: f64,
    alpha_decay: f64,
    update_frequency: usize,
    forgetting_factor: f64,
    adaptive_threshold: bool,
    min_samples_update: usize,
    k_neighbors: usize,
    similarity_threshold: f64,
}

impl StreamingGraphLearning<Untrained> {
    /// Create a new StreamingGraphLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            window_size: 1000,
            lambda_sparse: 0.1,
            alpha_decay: 0.95,
            update_frequency: 50,
            forgetting_factor: 0.99,
            adaptive_threshold: true,
            min_samples_update: 10,
            k_neighbors: 5,
            similarity_threshold: 0.5,
        }
    }

    /// Set the sliding window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the sparsity regularization parameter
    pub fn lambda_sparse(mut self, lambda_sparse: f64) -> Self {
        self.lambda_sparse = lambda_sparse;
        self
    }

    /// Set the decay factor for edge weights
    pub fn alpha_decay(mut self, alpha_decay: f64) -> Self {
        self.alpha_decay = alpha_decay;
        self
    }

    /// Set the frequency of full graph reconstruction
    pub fn update_frequency(mut self, frequency: usize) -> Self {
        self.update_frequency = frequency;
        self
    }

    /// Set the forgetting factor for old connections
    pub fn forgetting_factor(mut self, factor: f64) -> Self {
        self.forgetting_factor = factor;
        self
    }

    /// Enable/disable adaptive threshold for edge addition
    pub fn adaptive_threshold(mut self, adaptive: bool) -> Self {
        self.adaptive_threshold = adaptive;
        self
    }

    /// Set minimum samples required before updating the graph
    pub fn min_samples_update(mut self, min_samples: usize) -> Self {
        self.min_samples_update = min_samples;
        self
    }

    /// Set the number of nearest neighbors to consider
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the similarity threshold for edge creation
    pub fn similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    fn compute_similarity(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let diff = x1 - x2;
        let dist = diff.mapv(|x: f64| x * x).sum().sqrt();
        (-dist / (2.0 * 1.0_f64.powi(2))).exp()
    }

    fn build_initial_graph(&self, X: &Array2<f64>) -> Array2<f64> {
        let n_samples = X.nrows();
        let mut W = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut similarities: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let sim = self.compute_similarity(&X.row(i), &X.row(j));
                    similarities.push((j, sim));
                }
            }

            // Sort by similarity (descending)
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Connect to k nearest neighbors
            for &(j, sim) in similarities.iter().take(self.k_neighbors) {
                if sim > self.similarity_threshold {
                    W[[i, j]] = sim;
                    W[[j, i]] = sim; // Ensure symmetry
                }
            }
        }

        // Apply sparsity threshold
        let threshold = self.lambda_sparse;
        W.mapv_inplace(|x| if x > threshold { x - threshold } else { 0.0 });
        W.mapv_inplace(|x| x.max(0.0));

        // Zero diagonal
        for i in 0..n_samples {
            W[[i, i]] = 0.0;
        }

        W
    }

    #[allow(non_snake_case)]
    fn propagate_labels(&self, W: &Array2<f64>, Y_init: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = W.nrows();
        let n_classes = Y_init.ncols();

        // Compute transition matrix
        let D = W.sum_axis(Axis(1));
        let mut P = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            if D[i] > 0.0 {
                for j in 0..n_samples {
                    P[[i, j]] = W[[i, j]] / D[i];
                }
            }
        }

        let mut Y = Y_init.clone();
        let Y_static = Y_init.clone();

        // Label propagation iterations
        for _iter in 0..30 {
            let prev_Y = Y.clone();
            Y = 0.8 * P.dot(&Y) + 0.2 * &Y_static;

            // Check convergence
            let diff = (&Y - &prev_Y).mapv(|x| x.abs()).sum();
            if diff < 1e-6 {
                break;
            }
        }

        Ok(Y)
    }
}

impl Default for StreamingGraphLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StreamingGraphLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for StreamingGraphLearning<Untrained> {
    type Fitted = StreamingGraphLearning<StreamingGraphLearningTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();
        let (n_samples, n_features) = X.dim();

        // Identify labeled samples and classes
        let mut labeled_indices = Vec::new();
        let mut classes = std::collections::HashSet::new();

        for (i, &label) in y.iter().enumerate() {
            if label != -1 {
                labeled_indices.push(i);
                classes.insert(label);
            }
        }

        if labeled_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples provided".to_string(),
            ));
        }

        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Build initial graph
        let W = self.build_initial_graph(&X);

        // Initialize label matrix
        let mut Y = Array2::zeros((n_samples, n_classes));
        for &idx in &labeled_indices {
            if let Some(class_idx) = classes.iter().position(|&c| c == y[idx]) {
                Y[[idx, class_idx]] = 1.0;
            }
        }

        // Perform initial label propagation
        let Y_final = self.propagate_labels(&W, &Y)?;

        // Initialize sliding window with current data
        let mut data_window = VecDeque::with_capacity(self.window_size);
        let mut label_window = VecDeque::with_capacity(self.window_size);

        for i in 0..n_samples {
            data_window.push_back(X.row(i).to_owned());
            label_window.push_back(y[i]);
        }

        Ok(StreamingGraphLearning {
            state: StreamingGraphLearningTrained {
                X_train: X,
                y_train: y,
                classes: Array1::from(classes),
                current_graph: W,
                label_distributions: Y_final,
                data_window,
                label_window,
                update_count: 0,
                edge_ages: HashMap::new(),
                adaptive_threshold_value: self.similarity_threshold,
            },
            window_size: self.window_size,
            lambda_sparse: self.lambda_sparse,
            alpha_decay: self.alpha_decay,
            update_frequency: self.update_frequency,
            forgetting_factor: self.forgetting_factor,
            adaptive_threshold: self.adaptive_threshold,
            min_samples_update: self.min_samples_update,
            k_neighbors: self.k_neighbors,
            similarity_threshold: self.similarity_threshold,
        })
    }
}

impl StreamingGraphLearning<StreamingGraphLearningTrained> {
    fn compute_similarity(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let diff = x1 - x2;
        let dist = diff.mapv(|x: f64| x * x).sum().sqrt();
        (-dist / (2.0 * 1.0_f64.powi(2))).exp()
    }

    fn build_initial_graph(&self, X: &Array2<f64>) -> Array2<f64> {
        let n_samples = X.nrows();
        let mut W = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut similarities: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_samples {
                if i != j {
                    let sim = self.compute_similarity(&X.row(i), &X.row(j));
                    similarities.push((j, sim));
                }
            }

            // Sort by similarity (descending)
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Connect to k nearest neighbors
            for &(j, sim) in similarities.iter().take(self.k_neighbors) {
                if sim > self.similarity_threshold {
                    W[[i, j]] = sim;
                    W[[j, i]] = sim; // Ensure symmetry
                }
            }
        }

        // Apply sparsity threshold
        let threshold = self.lambda_sparse;
        W.mapv_inplace(|x| if x > threshold { x - threshold } else { 0.0 });
        W.mapv_inplace(|x| x.max(0.0));

        // Zero diagonal
        for i in 0..n_samples {
            W[[i, i]] = 0.0;
        }

        W
    }

    #[allow(non_snake_case)]
    fn propagate_labels(&self, W: &Array2<f64>, Y_init: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = W.nrows();
        let n_classes = Y_init.ncols();

        // Compute transition matrix
        let D = W.sum_axis(Axis(1));
        let mut P = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            if D[i] > 0.0 {
                for j in 0..n_samples {
                    P[[i, j]] = W[[i, j]] / D[i];
                }
            }
        }

        let mut Y = Y_init.clone();
        let Y_static = Y_init.clone();

        // Label propagation iterations
        for _iter in 0..30 {
            let prev_Y = Y.clone();
            Y = 0.8 * P.dot(&Y) + 0.2 * &Y_static;

            // Check convergence
            let diff = (&Y - &prev_Y).mapv(|x| x.abs()).sum();
            if diff < 1e-6 {
                break;
            }
        }

        Ok(Y)
    }
    /// Update the model with new streaming data
    #[allow(non_snake_case)]
    pub fn update(
        &mut self,
        X_new: &ArrayView2<'_, Float>,
        y_new: &ArrayView1<'_, i32>,
    ) -> SklResult<()> {
        let X_new = X_new.to_owned();
        let y_new = y_new.to_owned();
        let (n_new, _) = X_new.dim();

        // Add new data to sliding window
        for i in 0..n_new {
            // Remove oldest data if window is full
            if self.state.data_window.len() >= self.window_size {
                self.state.data_window.pop_front();
                self.state.label_window.pop_front();
            }

            self.state.data_window.push_back(X_new.row(i).to_owned());
            self.state.label_window.push_back(y_new[i]);
        }

        self.state.update_count += n_new;

        // Decay existing edge weights
        self.state
            .current_graph
            .mapv_inplace(|x| x * self.alpha_decay);

        // Update adaptive threshold if enabled
        if self.adaptive_threshold {
            self.update_adaptive_threshold();
        }

        // Age all edges
        let mut aged_edges = HashMap::new();
        for ((i, j), age) in &self.state.edge_ages {
            aged_edges.insert((*i, *j), age + 1);
        }
        self.state.edge_ages = aged_edges;

        // Incremental graph update
        self.incremental_graph_update(&X_new, &y_new)?;

        // Full reconstruction if update frequency is reached
        if self.state.update_count % self.update_frequency == 0 {
            self.full_graph_reconstruction()?;
        }

        Ok(())
    }

    fn update_adaptive_threshold(&mut self) {
        let current_data: Vec<Array1<f64>> = self.state.data_window.iter().cloned().collect();
        if current_data.len() < 2 {
            return;
        }

        let mut similarities = Vec::new();
        for i in 0..current_data.len().min(100) {
            for j in (i + 1)..current_data.len().min(100) {
                let sim = self.compute_similarity(&current_data[i].view(), &current_data[j].view());
                similarities.push(sim);
            }
        }

        if !similarities.is_empty() {
            similarities.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_idx = similarities.len() / 2;
            self.state.adaptive_threshold_value = similarities[median_idx] * 0.8;
        }
    }

    fn incremental_graph_update(
        &mut self,
        X_new: &Array2<f64>,
        y_new: &Array1<i32>,
    ) -> SklResult<()> {
        let current_data: Vec<Array1<f64>> = self.state.data_window.iter().cloned().collect();
        let current_labels: Vec<i32> = self.state.label_window.iter().cloned().collect();
        let n_current = current_data.len();
        let n_new = X_new.nrows();

        // Extend current graph to accommodate new nodes
        let mut new_graph = Array2::zeros((n_current, n_current));

        // Copy existing graph (with aging applied)
        let old_size = self.state.current_graph.nrows().min(n_current);
        for i in 0..old_size {
            for j in 0..old_size {
                new_graph[[i, j]] = self.state.current_graph[[i, j]];
            }
        }

        // Add connections for new nodes
        let start_idx = n_current - n_new;
        for i in start_idx..n_current {
            let mut similarities: Vec<(usize, f64)> = Vec::new();

            for j in 0..n_current {
                if i != j {
                    let sim =
                        self.compute_similarity(&current_data[i].view(), &current_data[j].view());
                    similarities.push((j, sim));
                }
            }

            // Sort by similarity (descending)
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Connect to k nearest neighbors
            let threshold = if self.adaptive_threshold {
                self.state.adaptive_threshold_value
            } else {
                self.similarity_threshold
            };

            for &(j, sim) in similarities.iter().take(self.k_neighbors) {
                if sim > threshold {
                    new_graph[[i, j]] = sim;
                    new_graph[[j, i]] = sim; // Ensure symmetry

                    // Track edge age
                    self.state.edge_ages.insert((i, j), 0);
                    self.state.edge_ages.insert((j, i), 0);
                }
            }
        }

        // Apply forgetting to old edges
        for ((i, j), age) in &self.state.edge_ages {
            if *i < n_current && *j < n_current {
                let forgetting_weight = self.forgetting_factor.powi(*age as i32);
                new_graph[[*i, *j]] *= forgetting_weight;
            }
        }

        // Apply sparsity threshold
        let threshold = self.lambda_sparse;
        new_graph.mapv_inplace(|x| if x > threshold { x - threshold } else { 0.0 });
        new_graph.mapv_inplace(|x| x.max(0.0));

        // Zero diagonal
        for i in 0..n_current {
            new_graph[[i, i]] = 0.0;
        }

        self.state.current_graph = new_graph;

        // Update label propagation
        self.update_label_propagation(&current_data, &current_labels)?;

        Ok(())
    }

    fn full_graph_reconstruction(&mut self) -> SklResult<()> {
        let current_data: Vec<Array1<f64>> = self.state.data_window.iter().cloned().collect();
        let current_labels: Vec<i32> = self.state.label_window.iter().cloned().collect();

        if current_data.is_empty() {
            return Ok(());
        }

        let n_samples = current_data.len();

        // Convert data to Array2
        let mut X = Array2::zeros((n_samples, current_data[0].len()));
        for (i, data_point) in current_data.iter().enumerate() {
            X.row_mut(i).assign(data_point);
        }

        // Rebuild graph from scratch
        self.state.current_graph = self.build_initial_graph(&X);

        // Clear edge ages
        self.state.edge_ages.clear();

        // Update label propagation
        self.update_label_propagation(&current_data, &current_labels)?;

        Ok(())
    }

    #[allow(non_snake_case)]
    fn update_label_propagation(
        &mut self,
        current_data: &[Array1<f64>],
        current_labels: &[i32],
    ) -> SklResult<()> {
        let n_samples = current_data.len();
        let n_classes = self.state.classes.len();

        if n_samples == 0 {
            return Ok(());
        }

        // Initialize label matrix
        let mut Y = Array2::zeros((n_samples, n_classes));
        for (i, &label) in current_labels.iter().enumerate() {
            if label != -1 {
                if let Some(class_idx) = self.state.classes.iter().position(|&c| c == label) {
                    Y[[i, class_idx]] = 1.0;
                }
            }
        }

        // Perform label propagation
        let Y_final = self.propagate_labels(&self.state.current_graph, &Y)?;
        self.state.label_distributions = Y_final;

        Ok(())
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for StreamingGraphLearning<StreamingGraphLearningTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let mut predictions = Array1::zeros(n_test);

        let current_data: Vec<Array1<f64>> = self.state.data_window.iter().cloned().collect();

        for i in 0..n_test {
            let mut max_sim = -1.0;
            let mut best_idx = 0;

            // Find most similar sample in current window
            for (j, data_point) in current_data.iter().enumerate() {
                let sim = self.compute_similarity(&X.row(i), &data_point.view());
                if sim > max_sim {
                    max_sim = sim;
                    best_idx = j;
                }
            }

            // Use the label distribution of the most similar sample
            if best_idx < self.state.label_distributions.nrows() {
                let distributions = self.state.label_distributions.row(best_idx);
                let max_idx = distributions
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;

                predictions[i] = self.state.classes[max_idx];
            }
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for StreamingGraphLearning<StreamingGraphLearningTrained>
{
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();
        let mut probas = Array2::zeros((n_test, n_classes));

        let current_data: Vec<Array1<f64>> = self.state.data_window.iter().cloned().collect();

        for i in 0..n_test {
            let mut max_sim = -1.0;
            let mut best_idx = 0;

            // Find most similar sample in current window
            for (j, data_point) in current_data.iter().enumerate() {
                let sim = self.compute_similarity(&X.row(i), &data_point.view());
                if sim > max_sim {
                    max_sim = sim;
                    best_idx = j;
                }
            }

            // Copy the label distribution
            if best_idx < self.state.label_distributions.nrows() {
                for k in 0..n_classes {
                    probas[[i, k]] = self.state.label_distributions[[best_idx, k]];
                }
            }
        }

        Ok(probas)
    }
}

/// Trained state for StreamingGraphLearning
#[derive(Debug, Clone)]
pub struct StreamingGraphLearningTrained {
    /// X_train
    pub X_train: Array2<f64>,
    /// y_train
    pub y_train: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
    /// current_graph
    pub current_graph: Array2<f64>,
    /// label_distributions
    pub label_distributions: Array2<f64>,
    /// data_window
    pub data_window: VecDeque<Array1<f64>>,
    /// label_window
    pub label_window: VecDeque<i32>,
    /// update_count
    pub update_count: usize,
    /// edge_ages
    pub edge_ages: HashMap<(usize, usize), usize>,
    /// adaptive_threshold_value
    pub adaptive_threshold_value: f64,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_graph_learning_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let sgl = StreamingGraphLearning::new()
            .window_size(10)
            .lambda_sparse(0.1)
            .alpha_decay(0.9)
            .update_frequency(5);
        let fitted = sgl.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let probas = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probas.dim(), (4, 2));

        // Check that labeled samples maintain their labels
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_graph_learning_update() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let sgl = StreamingGraphLearning::new()
            .window_size(10)
            .update_frequency(3)
            .alpha_decay(0.95);
        let mut fitted = sgl.fit(&X.view(), &y.view()).unwrap();

        // Initial graph size
        let initial_graph_size = fitted.state.current_graph.dim();
        assert_eq!(initial_graph_size, (4, 4));

        // Add new streaming data
        let X_new = array![[5.0, 6.0], [6.0, 7.0]];
        let y_new = array![-1, 0];
        fitted.update(&X_new.view(), &y_new.view()).unwrap();

        // Check that data window is updated
        assert_eq!(fitted.state.data_window.len(), 6);
        assert_eq!(fitted.state.label_window.len(), 6);

        // Graph should be updated to accommodate new data
        let updated_graph_size = fitted.state.current_graph.dim();
        assert_eq!(updated_graph_size, (6, 6));

        // Test predictions with updated model
        let predictions = fitted.predict(&X_new.view()).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_graph_learning_window_overflow() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 1];

        let sgl = StreamingGraphLearning::new()
            .window_size(3) // Small window size
            .update_frequency(2);
        let mut fitted = sgl.fit(&X.view(), &y.view()).unwrap();

        // Add more data than window size
        let X_new1 = array![[3.0, 4.0]];
        let y_new1 = array![-1];
        fitted.update(&X_new1.view(), &y_new1.view()).unwrap();

        let X_new2 = array![[4.0, 5.0]];
        let y_new2 = array![0];
        fitted.update(&X_new2.view(), &y_new2.view()).unwrap();

        // Window should maintain size limit
        assert_eq!(fitted.state.data_window.len(), 3);
        assert_eq!(fitted.state.label_window.len(), 3);

        // Should still be able to make predictions
        let predictions = fitted.predict(&X_new2.view()).unwrap();
        assert_eq!(predictions.len(), 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_graph_learning_adaptive_threshold() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1];

        let sgl = StreamingGraphLearning::new()
            .window_size(10)
            .adaptive_threshold(true)
            .similarity_threshold(0.5);
        let mut fitted = sgl.fit(&X.view(), &y.view()).unwrap();

        let initial_threshold = fitted.state.adaptive_threshold_value;

        // Add new data with different characteristics
        let X_new = array![[10.0, 20.0], [20.0, 30.0]];
        let y_new = array![-1, 1];
        fitted.update(&X_new.view(), &y_new.view()).unwrap();

        // Adaptive threshold should potentially change
        // (depends on the similarity distribution)
        assert!(fitted.state.adaptive_threshold_value > 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_graph_learning_edge_aging() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 1];

        let sgl = StreamingGraphLearning::new()
            .window_size(10)
            .forgetting_factor(0.8)
            .alpha_decay(0.9);
        let mut fitted = sgl.fit(&X.view(), &y.view()).unwrap();

        // Check initial state
        assert_eq!(fitted.state.update_count, 0);

        // Add new data multiple times to age edges
        for i in 0..3 {
            let X_new = array![[3.0 + i as f64, 4.0 + i as f64]];
            let y_new = array![-1];
            fitted.update(&X_new.view(), &y_new.view()).unwrap();
        }

        // Update count should be incremented
        assert_eq!(fitted.state.update_count, 3);

        // Some edges should have aged
        assert!(!fitted.state.edge_ages.is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_streaming_graph_learning_full_reconstruction() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 1];

        let sgl = StreamingGraphLearning::new()
            .window_size(10)
            .update_frequency(2); // Trigger full reconstruction frequently
        let mut fitted = sgl.fit(&X.view(), &y.view()).unwrap();

        // Add data to trigger full reconstruction
        let X_new1 = array![[3.0, 4.0]];
        let y_new1 = array![-1];
        fitted.update(&X_new1.view(), &y_new1.view()).unwrap();

        let X_new2 = array![[4.0, 5.0]];
        let y_new2 = array![0];
        fitted.update(&X_new2.view(), &y_new2.view()).unwrap();

        // Full reconstruction should have been triggered
        // Edge ages should be cleared
        assert!(
            fitted.state.edge_ages.is_empty()
                || fitted.state.edge_ages.values().all(|&age| age == 0)
        );

        // Should still be able to make predictions
        let predictions = fitted.predict(&X_new2.view()).unwrap();
        assert_eq!(predictions.len(), 1);
    }
}
