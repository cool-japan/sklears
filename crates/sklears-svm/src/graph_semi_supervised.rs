//! Graph-based Semi-supervised Support Vector Machines
//!
//! This module implements graph-based semi-supervised SVMs that leverage
//! graph structure to propagate label information from labeled to unlabeled
//! examples, particularly useful when labeled data is scarce.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Predict},
    types::Float,
};

/// Graph-based Semi-supervised Support Vector Machine
///
/// This implementation combines SVM with graph-based label propagation,
/// using the graph structure to regularize the learning process and
/// propagate information from labeled to unlabeled examples.
///
/// # Parameters
/// * `C` - Regularization parameter for SVM (default: 1.0)
/// * `gamma_a` - Graph regularization parameter (default: 0.1)
/// * `gamma_i` - Intrinsic regularization parameter (default: 0.01)
/// * `kernel` - Kernel function for SVM ('rbf', 'linear', default: 'rbf')
/// * `graph_kernel` - Kernel for graph construction ('rbf', 'knn', default: 'rbf')
/// * `n_neighbors` - Number of neighbors for k-NN graph (default: 10)
/// * `sigma` - Bandwidth for RBF kernel (default: 1.0)
/// * `max_iter` - Maximum number of iterations (default: 1000)
/// * `tol` - Tolerance for convergence (default: 1e-4)
/// * `verbose` - Enable verbose output (default: false)
///
/// # Example
/// ```rust
/// use sklears_svm::GraphSemiSupervisedSVM;
/// use sklears_core::traits::{Predict};
/// use scirs2_core::ndarray::array;
///
/// let X_labeled = array![[1.0, 2.0], [2.0, 3.0]];
/// let y_labeled = array![0, 1];
/// let X_unlabeled = array![[1.5, 2.5], [2.5, 3.5]];
///
/// let model = GraphSemiSupervisedSVM::new()
///     .with_c(1.0)
///     .with_gamma_a(0.1);
///
/// let trained_model = model
///     .fit_semi_supervised(&X_labeled, &y_labeled, &X_unlabeled)
///     .unwrap();
/// let predictions = trained_model.predict(&X_labeled).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GraphSemiSupervisedSVM {
    /// SVM regularization parameter
    pub c: f64,
    /// Graph regularization parameter
    pub gamma_a: f64,
    /// Intrinsic regularization parameter
    pub gamma_i: f64,
    /// Kernel function for SVM
    pub kernel: String,
    /// Kernel function for graph construction
    pub graph_kernel: String,
    /// Number of neighbors for k-NN graph
    pub n_neighbors: usize,
    /// Bandwidth for RBF kernel
    pub sigma: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Verbose output
    pub verbose: bool,
}

/// Trained Graph-based Semi-supervised SVM model
#[derive(Debug, Clone)]
pub struct TrainedGraphSemiSupervisedSVM {
    /// Model weights (coefficients)
    pub coef_: Array2<f64>,
    /// Intercept terms
    pub intercept_: Array1<f64>,
    /// Unique class labels
    pub classes_: Array1<i32>,
    /// Number of features
    pub n_features_in_: usize,
    /// Training data (labeled + unlabeled)
    pub training_data_: Array2<f64>,
    /// Graph adjacency matrix
    pub graph_matrix_: Array2<f64>,
    /// Predicted labels for unlabeled data
    pub unlabeled_predictions_: Array1<f64>,
    /// Training parameters
    _params: GraphSemiSupervisedSVM,
}

/// Graph construction utilities
#[derive(Debug)]
pub struct GraphBuilder {
    kernel_type: String,
    n_neighbors: usize,
    sigma: f64,
}

impl GraphBuilder {
    pub fn new(kernel_type: &str, n_neighbors: usize, sigma: f64) -> Self {
        Self {
            kernel_type: kernel_type.to_string(),
            n_neighbors,
            sigma,
        }
    }

    /// Compute RBF similarity between two points
    fn rbf_similarity(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        let dist_sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        (-dist_sq / (2.0 * self.sigma.powi(2))).exp()
    }

    /// Build k-NN graph
    fn build_knn_graph(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let n_samples = x.nrows();
        let mut graph = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let xi = x.row(i);

            // Compute distances to all other points
            let mut distances: Vec<(usize, f64)> = Vec::new();
            for j in 0..n_samples {
                if i != j {
                    let xj = x.row(j);
                    let dist: f64 = xi
                        .iter()
                        .zip(xj.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    distances.push((j, dist));
                }
            }

            // Sort by distance and take k nearest neighbors
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let k = std::cmp::min(self.n_neighbors, distances.len());
            for &(j, dist) in distances.iter().take(k) {
                // Use RBF similarity as edge weight
                let weight = (-dist.powi(2) / (2.0 * self.sigma.powi(2))).exp();
                graph[[i, j]] = weight;
                graph[[j, i]] = weight; // Symmetric graph
            }
        }

        graph
    }

    /// Build RBF similarity graph
    fn build_rbf_graph(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let n_samples = x.nrows();
        let mut graph = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let similarity = self.rbf_similarity(x.row(i), x.row(j));
                graph[[i, j]] = similarity;
                graph[[j, i]] = similarity;
            }
        }

        graph
    }

    /// Build graph based on specified kernel type
    pub fn build_graph(&self, x: ArrayView2<f64>) -> Array2<f64> {
        match self.kernel_type.as_str() {
            "knn" => self.build_knn_graph(x),
            "rbf" => self.build_rbf_graph(x),
            _ => self.build_rbf_graph(x), // Default to RBF
        }
    }
}

impl Default for GraphSemiSupervisedSVM {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSemiSupervisedSVM {
    /// Create a new GraphSemiSupervisedSVM with default parameters
    pub fn new() -> Self {
        Self {
            c: 1.0,
            gamma_a: 0.1,
            gamma_i: 0.01,
            kernel: "rbf".to_string(),
            graph_kernel: "rbf".to_string(),
            n_neighbors: 10,
            sigma: 1.0,
            max_iter: 1000,
            tol: 1e-4,
            verbose: false,
        }
    }

    /// Set the SVM regularization parameter C
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the graph regularization parameter
    pub fn with_gamma_a(mut self, gamma_a: f64) -> Self {
        self.gamma_a = gamma_a;
        self
    }

    /// Set the intrinsic regularization parameter
    pub fn with_gamma_i(mut self, gamma_i: f64) -> Self {
        self.gamma_i = gamma_i;
        self
    }

    /// Set the SVM kernel function
    pub fn with_kernel(mut self, kernel: &str) -> Self {
        self.kernel = kernel.to_string();
        self
    }

    /// Set the graph construction kernel
    pub fn with_graph_kernel(mut self, graph_kernel: &str) -> Self {
        self.graph_kernel = graph_kernel.to_string();
        self
    }

    /// Set the number of neighbors for k-NN graph
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the RBF kernel bandwidth
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Compute graph Laplacian matrix
    fn compute_laplacian(&self, adjacency: &Array2<f64>) -> Array2<f64> {
        let n = adjacency.nrows();
        let mut laplacian = Array2::zeros((n, n));

        // Compute degree matrix
        let mut degrees = Array1::zeros(n);
        for i in 0..n {
            degrees[i] = adjacency.row(i).sum();
        }

        // Compute normalized Laplacian: I - D^(-1/2) * A * D^(-1/2)
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    laplacian[[i, j]] = 1.0;
                    if degrees[i] > 0.0 {
                        laplacian[[i, j]] -= adjacency[[i, j]] / degrees[i];
                    }
                } else if degrees[i] > 0.0 && degrees[j] > 0.0 {
                    laplacian[[i, j]] = -adjacency[[i, j]] / (degrees[i] * degrees[j]).sqrt();
                }
            }
        }

        laplacian
    }

    /// Label propagation using graph structure
    fn propagate_labels(
        &self,
        graph: &Array2<f64>,
        labeled_indices: &[usize],
        unlabeled_indices: &[usize],
        y_labeled: &Array1<f64>,
    ) -> Array1<f64> {
        let n_total = graph.nrows();
        let mut labels = Array1::zeros(n_total);

        // Initialize labeled points
        for (i, &idx) in labeled_indices.iter().enumerate() {
            labels[idx] = y_labeled[i];
        }

        let laplacian = self.compute_laplacian(graph);

        // Iterative label propagation
        for iteration in 0..self.max_iter {
            let old_labels = labels.clone();

            // Update unlabeled points
            for &u_idx in unlabeled_indices {
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for j in 0..n_total {
                    if j != u_idx {
                        let weight = graph[[u_idx, j]];
                        weighted_sum += weight * labels[j];
                        weight_sum += weight;
                    }
                }

                if weight_sum > 0.0 {
                    labels[u_idx] = weighted_sum / weight_sum;
                }
            }

            // Reset labeled points (clamping)
            for (i, &idx) in labeled_indices.iter().enumerate() {
                labels[idx] = y_labeled[i];
            }

            // Check convergence
            let change: f64 = labels
                .iter()
                .zip(old_labels.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum::<f64>()
                .sqrt();

            if change < self.tol {
                if self.verbose {
                    println!("Label propagation converged at iteration {iteration}");
                }
                break;
            }

            if self.verbose && iteration % 100 == 0 {
                println!(
                    "Label propagation iteration {}, change: {:.6}",
                    iteration, change
                );
            }
        }

        labels
    }

    /// Train graph-based semi-supervised SVM
    pub fn fit_semi_supervised(
        self,
        x_labeled: &Array2<f64>,
        y_labeled: &Array1<i32>,
        x_unlabeled: &Array2<f64>,
    ) -> Result<TrainedGraphSemiSupervisedSVM> {
        let (n_labeled, n_features) = x_labeled.dim();
        let (n_unlabeled, _) = x_unlabeled.dim();

        if n_labeled == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Labeled data cannot be empty".to_string(),
            ));
        }

        if x_labeled.ncols() != x_unlabeled.ncols() {
            return Err(SklearsError::InvalidInput(
                "Labeled and unlabeled data must have same number of features".to_string(),
            ));
        }

        // Combine labeled and unlabeled data
        let mut x_combined = Array2::zeros((n_labeled + n_unlabeled, n_features));
        for i in 0..n_labeled {
            x_combined.row_mut(i).assign(&x_labeled.row(i));
        }
        for i in 0..n_unlabeled {
            x_combined
                .row_mut(n_labeled + i)
                .assign(&x_unlabeled.row(i));
        }

        // Get unique classes
        let mut classes: Vec<i32> = y_labeled.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from(classes);

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Build graph
        let graph_builder = GraphBuilder::new(&self.graph_kernel, self.n_neighbors, self.sigma);
        let graph = graph_builder.build_graph(x_combined.view());

        // Prepare indices
        let labeled_indices: Vec<usize> = (0..n_labeled).collect();
        let unlabeled_indices: Vec<usize> = (n_labeled..n_labeled + n_unlabeled).collect();

        // Convert labels to f64 for propagation (-1, 1 for binary)
        let y_labeled_f64 = if classes.len() == 2 {
            y_labeled.map(|&label| if label == classes[1] { 1.0 } else { -1.0 })
        } else {
            // For multi-class, use one-vs-rest approach
            y_labeled.map(|&label| label as f64)
        };

        // Perform label propagation
        let propagated_labels =
            self.propagate_labels(&graph, &labeled_indices, &unlabeled_indices, &y_labeled_f64);

        // Extract predictions for unlabeled data
        let mut unlabeled_predictions_ = Array1::zeros(n_unlabeled);
        for (i, &idx) in unlabeled_indices.iter().enumerate() {
            unlabeled_predictions_[i] = propagated_labels[idx];
        }

        // Train SVM on combined data with propagated labels
        let y_combined = if classes.len() == 2 {
            propagated_labels.map(|&f_label| {
                if f_label >= 0.0 {
                    classes[1]
                } else {
                    classes[0]
                }
            })
        } else {
            propagated_labels.map(|&f_label| f_label.round() as i32)
        };

        // Simple linear SVM training (simplified for demonstration)
        let mut coef = Array1::zeros(n_features);
        let mut intercept = 0.0;

        // Use gradient descent for training
        let learning_rate = 0.01;

        for iteration in 0..self.max_iter {
            let mut w_gradient: Array1<f64> = Array1::zeros(n_features);
            let mut intercept_gradient = 0.0;

            for i in 0..x_combined.nrows() {
                let xi = x_combined.row(i);
                let yi = if y_combined[i] == classes[1] {
                    1.0
                } else {
                    -1.0
                };

                // Compute prediction
                let mut prediction = intercept;
                for j in 0..n_features {
                    prediction += coef[j] * xi[j];
                }

                let margin = yi * prediction;

                // Hinge loss gradient
                if margin < 1.0 {
                    for j in 0..n_features {
                        w_gradient[j] += -yi * xi[j];
                    }
                    intercept_gradient += -yi;
                }
            }

            // Add regularization
            for j in 0..n_features {
                w_gradient[j] += coef[j] / self.c;
            }

            // Update parameters
            for j in 0..n_features {
                coef[j] -= learning_rate * w_gradient[j];
            }
            intercept -= learning_rate * intercept_gradient;

            // Check convergence (simplified)
            if iteration % 100 == 0 {
                let gradient_norm = w_gradient.iter().map(|&g| g * g).sum::<f64>().sqrt();
                if gradient_norm < self.tol {
                    if self.verbose {
                        println!("Graph SVM converged at iteration {iteration}");
                    }
                    break;
                }
            }
        }

        let coef_matrix = coef.insert_axis(Axis(0));
        let intercept_arr = Array1::from(vec![intercept]);

        Ok(TrainedGraphSemiSupervisedSVM {
            coef_: coef_matrix,
            intercept_: intercept_arr,
            classes_: classes,
            n_features_in_: n_features,
            training_data_: x_combined,
            graph_matrix_: graph,
            unlabeled_predictions_,
            _params: self,
        })
    }
}

impl Estimator for GraphSemiSupervisedSVM {
    type Config = Self;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        self
    }
}

// Note: We use a custom fit method instead of the standard Fit trait
// because semi-supervised learning requires both labeled and unlabeled data

impl Predict<Array2<f64>, Array1<i32>> for TrainedGraphSemiSupervisedSVM {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let decision_values = self.decision_function(x)?;

        if self.classes_.len() == 2 {
            // Binary classification
            let predictions = decision_values.map(|&score| {
                if score >= 0.0 {
                    self.classes_[1]
                } else {
                    self.classes_[0]
                }
            });
            Ok(predictions.remove_axis(Axis(1)))
        } else {
            // Multi-class: predict class with highest score
            let mut predictions = Array1::zeros(x.len_of(Axis(0)));
            for (i, row) in decision_values.axis_iter(Axis(0)).enumerate() {
                let best_class_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                predictions[i] = self.classes_[best_class_idx];
            }
            Ok(predictions)
        }
    }
}

impl TrainedGraphSemiSupervisedSVM {
    /// Compute decision function values
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_ {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_,
                actual: n_features,
            });
        }

        if self.classes_.len() == 2 {
            // Binary classification: single decision function
            let mut scores = Array1::zeros(n_samples);
            let w = self.coef_.row(0);
            let intercept = self.intercept_[0];

            for (i, x_row) in x.axis_iter(Axis(0)).enumerate() {
                let mut score = intercept;
                for (j, &x_val) in x_row.iter().enumerate() {
                    score += w[j] * x_val;
                }
                scores[i] = score;
            }

            Ok(scores.insert_axis(Axis(1)))
        } else {
            // Multi-class: one score per class
            let mut scores = Array2::zeros((n_samples, self.classes_.len()));

            for (class_idx, coef_row) in self.coef_.axis_iter(Axis(0)).enumerate() {
                let intercept = self.intercept_[class_idx];

                for (i, x_row) in x.axis_iter(Axis(0)).enumerate() {
                    let mut score = intercept;
                    for (j, &x_val) in x_row.iter().enumerate() {
                        score += coef_row[j] * x_val;
                    }
                    scores[[i, class_idx]] = score;
                }
            }

            Ok(scores)
        }
    }

    /// Get the model coefficients
    pub fn coef(&self) -> &Array2<f64> {
        &self.coef_
    }

    /// Get the intercept terms
    pub fn intercept(&self) -> &Array1<f64> {
        &self.intercept_
    }

    /// Get the class labels
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes_
    }

    /// Get the graph adjacency matrix
    pub fn graph_matrix(&self) -> &Array2<f64> {
        &self.graph_matrix_
    }

    /// Get the predictions for unlabeled data from training
    pub fn unlabeled_predictions(&self) -> &Array1<f64> {
        &self.unlabeled_predictions_
    }

    /// Get the combined training data (labeled + unlabeled)
    pub fn training_data(&self) -> &Array2<f64> {
        &self.training_data_
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_graph_semi_supervised_svm() {
        let X_labeled_var = array![[1.0, 2.0], [2.0, 3.0]];
        let y_labeled = array![0, 1];
        let X_unlabeled_var = array![[1.5, 2.5], [2.5, 3.5]];

        let model = GraphSemiSupervisedSVM::new()
            .with_c(1.0)
            .with_gamma_a(0.1)
            .with_verbose(true);

        let trained_model = model
            .fit_semi_supervised(&X_labeled_var, &y_labeled, &X_unlabeled_var)
            .unwrap();

        // Test predictions on labeled data
        let predictions = trained_model.predict(&X_labeled_var).unwrap();
        assert_eq!(predictions.len(), 2);

        // Test decision function
        let scores = trained_model.decision_function(&X_labeled_var).unwrap();
        assert_eq!(scores.dim(), (2, 1));

        // Check that unlabeled predictions were generated
        assert_eq!(trained_model.unlabeled_predictions().len(), 2);
    }

    #[test]
    fn test_graph_builder_rbf() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let builder = GraphBuilder::new("rbf", 2, 1.0);
        let graph = builder.build_graph(X_var.view());

        assert_eq!(graph.dim(), (3, 3));

        // Graph should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(graph[[i, j]], graph[[j, i]], epsilon = 1e-10);
            }
        }

        // Diagonal should be zero
        for i in 0..3 {
            assert_abs_diff_eq!(graph[[i, i]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_graph_builder_knn() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [10.0, 10.0]];

        let builder = GraphBuilder::new("knn", 2, 1.0);
        let graph = builder.build_graph(X_var.view());

        assert_eq!(graph.dim(), (4, 4));

        // Check that each row has at most k non-zero entries (excluding diagonal)
        for i in 0..4 {
            let non_zero_count = graph
                .row(i)
                .iter()
                .enumerate()
                .filter(|(j, &val)| *j != i && val > 1e-10)
                .count();
            assert!(
                non_zero_count <= 2,
                "Row {} has {} non-zero entries",
                i,
                non_zero_count
            );
        }
    }

    #[test]
    fn test_graph_semi_supervised_parameters() {
        let model = GraphSemiSupervisedSVM::new()
            .with_c(0.5)
            .with_gamma_a(0.2)
            .with_gamma_i(0.05)
            .with_kernel("linear")
            .with_graph_kernel("knn")
            .with_n_neighbors(5)
            .with_sigma(2.0)
            .with_max_iter(500)
            .with_tol(1e-5);

        assert_eq!(model.c, 0.5);
        assert_abs_diff_eq!(model.gamma_a, 0.2);
        assert_abs_diff_eq!(model.gamma_i, 0.05);
        assert_eq!(model.kernel, "linear");
        assert_eq!(model.graph_kernel, "knn");
        assert_eq!(model.n_neighbors, 5);
        assert_abs_diff_eq!(model.sigma, 2.0);
        assert_eq!(model.max_iter, 500);
        assert_abs_diff_eq!(model.tol, 1e-5);
    }

    #[test]
    fn test_label_propagation_convergence() {
        let X_labeled_var = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y_labeled = array![0, 1, 1];
        let X_unlabeled_var = array![[0.5, 0.5], [1.5, 1.5]];

        let model = GraphSemiSupervisedSVM::new()
            .with_gamma_a(0.5)
            .with_max_iter(100)
            .with_tol(1e-6)
            .with_verbose(true);

        let result = model.fit_semi_supervised(&X_labeled_var, &y_labeled, &X_unlabeled_var);
        assert!(result.is_ok());

        let trained_model = result.unwrap();

        // Check that unlabeled predictions are reasonable
        let unlabeled_preds = trained_model.unlabeled_predictions();
        println!("Unlabeled predictions: {:?}", unlabeled_preds);

        // First unlabeled point (0.5, 0.5) should be closer to class 0
        // Second unlabeled point (1.5, 1.5) should be closer to class 1
        assert!(unlabeled_preds[1] > unlabeled_preds[0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let X_labeled_var = array![[1.0, 2.0], [2.0, 3.0]];
        let y_labeled = array![0, 1];
        let X_unlabeled_var = array![[1.5, 2.5, 3.5]]; // Wrong dimension

        let model = GraphSemiSupervisedSVM::new();
        let result = model.fit_semi_supervised(&X_labeled_var, &y_labeled, &X_unlabeled_var);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_labeled_data() {
        let X_labeled_var = Array2::zeros((0, 2));
        let y_labeled = Array1::zeros(0);
        let X_unlabeled_var = array![[1.5, 2.5]];

        let model = GraphSemiSupervisedSVM::new();
        let result = model.fit_semi_supervised(&X_labeled_var, &y_labeled, &X_unlabeled_var);
        assert!(result.is_err());
    }
}
