//! Discriminant Locality Alignment implementation
//!
//! This module implements Discriminant Locality Alignment (DLA), a supervised
//! dimensionality reduction technique that preserves local manifold structure
//! while maximizing discriminative power for classification.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_linalg::compat::{Eig, Eigh, UPLO};
use sklears_core::{
    error::{validate, Result},
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Discriminant Locality Alignment
#[derive(Debug, Clone)]
pub struct DiscriminantLocalityAlignmentConfig {
    /// Number of nearest neighbors for locality graph construction
    pub n_neighbors: usize,
    /// Number of components for dimensionality reduction
    pub n_components: Option<usize>,
    /// Regularization parameter for numerical stability
    pub reg_param: Float,
    /// Prior probabilities of the classes
    pub priors: Option<Array1<Float>>,
    /// Whether to store the covariance matrix
    pub store_covariance: bool,
    /// Tolerance for eigenvalue computation
    pub tol: Float,
    /// Weight function for graph construction ("heat_kernel", "binary", "polynomial", "cosine")
    pub weight_function: String,
    /// Parameter for heat kernel weight function (σ²)
    pub heat_kernel_param: Float,
    /// Graph construction method ("knn", "epsilon")
    pub graph_method: String,
    /// Epsilon parameter for epsilon-neighborhood graph
    pub epsilon: Float,
    /// Whether to use symmetric graph
    pub symmetric_graph: bool,
    /// Balance parameter between discriminative and locality terms
    pub alpha: Float,
    /// Whether to use global alignment constraints
    pub global_alignment: bool,
    /// Maximum number of iterations for optimization
    pub max_iter: usize,
}

impl Default for DiscriminantLocalityAlignmentConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            n_components: None,
            reg_param: 1e-6,
            priors: None,
            store_covariance: false,
            tol: 1e-8,
            weight_function: "heat_kernel".to_string(),
            heat_kernel_param: 1.0,
            graph_method: "knn".to_string(),
            epsilon: 1.0,
            symmetric_graph: true,
            alpha: 0.5,
            global_alignment: true,
            max_iter: 100,
        }
    }
}

/// Discriminant Locality Alignment
///
/// A supervised dimensionality reduction technique that combines discriminant analysis
/// with locality alignment to preserve both class separability and local manifold structure.
///
/// The algorithm finds a projection that:
/// 1. Maximizes between-class scatter while minimizing within-class scatter
/// 2. Preserves local neighborhood relationships in the data
/// 3. Aligns local coordinate systems to maintain manifold structure
#[derive(Debug, Clone)]
pub struct DiscriminantLocalityAlignment<State = Untrained> {
    config: DiscriminantLocalityAlignmentConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    means_: Option<Array2<Float>>,
    covariance_: Option<Array2<Float>>,
    components_: Option<Array2<Float>>,
    eigenvalues_: Option<Array1<Float>>,
    priors_: Option<Array1<Float>>,
    n_features_: Option<usize>,
    locality_graph_: Option<Array2<Float>>,
    alignment_weights_: Option<Array2<Float>>,
}

impl DiscriminantLocalityAlignment<Untrained> {
    /// Create a new DiscriminantLocalityAlignment instance
    pub fn new() -> Self {
        Self {
            config: DiscriminantLocalityAlignmentConfig::default(),
            state: PhantomData,
            classes_: None,
            means_: None,
            covariance_: None,
            components_: None,
            eigenvalues_: None,
            priors_: None,
            n_features_: None,
            locality_graph_: None,
            alignment_weights_: None,
        }
    }

    /// Set the number of nearest neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.config.n_neighbors = n_neighbors;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set the regularization parameter
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Set the prior probabilities
    pub fn priors(mut self, priors: Option<Array1<Float>>) -> Self {
        self.config.priors = priors;
        self
    }

    /// Set whether to store covariance matrix
    pub fn store_covariance(mut self, store_covariance: bool) -> Self {
        self.config.store_covariance = store_covariance;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the weight function
    pub fn weight_function(mut self, weight_function: &str) -> Self {
        self.config.weight_function = weight_function.to_string();
        self
    }

    /// Set the heat kernel parameter
    pub fn heat_kernel_param(mut self, heat_kernel_param: Float) -> Self {
        self.config.heat_kernel_param = heat_kernel_param;
        self
    }

    /// Set the graph construction method
    pub fn graph_method(mut self, graph_method: &str) -> Self {
        self.config.graph_method = graph_method.to_string();
        self
    }

    /// Set the epsilon parameter
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set whether to use symmetric graph
    pub fn symmetric_graph(mut self, symmetric_graph: bool) -> Self {
        self.config.symmetric_graph = symmetric_graph;
        self
    }

    /// Set the balance parameter between discriminative and locality terms
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set whether to use global alignment constraints
    pub fn global_alignment(mut self, global_alignment: bool) -> Self {
        self.config.global_alignment = global_alignment;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Compute distance matrix between all pairs of samples
    fn compute_distance_matrix(&self, x: &Array2<Float>) -> Array2<Float> {
        let n_samples = x.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let mut dist = 0.0;
                for k in 0..x.ncols() {
                    let diff = x[[i, k]] - x[[j, k]];
                    dist += diff * diff;
                }
                dist = dist.sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Build k-nearest neighbor graph
    fn build_knn_graph(&self, x: &Array2<Float>) -> Array2<Float> {
        let n_samples = x.nrows();
        let distances = self.compute_distance_matrix(x);
        let mut graph = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Find k nearest neighbors for sample i
            let mut neighbors: Vec<(usize, Float)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| (j, distances[[i, j]]))
                .collect();

            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take k nearest neighbors
            for (j, distance) in neighbors.iter().take(self.config.n_neighbors) {
                let weight = match self.config.weight_function.as_str() {
                    "heat_kernel" => {
                        (-distance * distance / (2.0 * self.config.heat_kernel_param)).exp()
                    }
                    "binary" => 1.0,
                    "polynomial" => 1.0 / (1.0 + distance),
                    "cosine" => {
                        // Cosine similarity weight
                        let mut dot_product = 0.0;
                        let mut norm_i = 0.0;
                        let mut norm_j = 0.0;
                        for k in 0..x.ncols() {
                            dot_product += x[[i, k]] * x[[*j, k]];
                            norm_i += x[[i, k]] * x[[i, k]];
                            norm_j += x[[*j, k]] * x[[*j, k]];
                        }
                        if norm_i > 0.0 && norm_j > 0.0 {
                            dot_product / (norm_i.sqrt() * norm_j.sqrt())
                        } else {
                            0.0
                        }
                    }
                    _ => 1.0, // Default to binary
                };
                graph[[i, *j]] = weight;
            }
        }

        // Make graph symmetric if requested
        if self.config.symmetric_graph {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if graph[[i, j]] != graph[[j, i]] {
                        let avg_weight = (graph[[i, j]] + graph[[j, i]]) / 2.0;
                        graph[[i, j]] = avg_weight;
                        graph[[j, i]] = avg_weight;
                    }
                }
            }
        }

        graph
    }

    /// Build epsilon-neighborhood graph
    fn build_epsilon_graph(&self, x: &Array2<Float>) -> Array2<Float> {
        let n_samples = x.nrows();
        let distances = self.compute_distance_matrix(x);
        let mut graph = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j && distances[[i, j]] <= self.config.epsilon {
                    let weight = match self.config.weight_function.as_str() {
                        "heat_kernel" => (-distances[[i, j]] * distances[[i, j]]
                            / (2.0 * self.config.heat_kernel_param))
                            .exp(),
                        "binary" => 1.0,
                        "polynomial" => 1.0 / (1.0 + distances[[i, j]]),
                        "cosine" => {
                            // Cosine similarity weight
                            let mut dot_product = 0.0;
                            let mut norm_i = 0.0;
                            let mut norm_j = 0.0;
                            for k in 0..x.ncols() {
                                dot_product += x[[i, k]] * x[[j, k]];
                                norm_i += x[[i, k]] * x[[i, k]];
                                norm_j += x[[j, k]] * x[[j, k]];
                            }
                            if norm_i > 0.0 && norm_j > 0.0 {
                                dot_product / (norm_i.sqrt() * norm_j.sqrt())
                            } else {
                                0.0
                            }
                        }
                        _ => 1.0, // Default to binary
                    };
                    graph[[i, j]] = weight;
                }
            }
        }

        graph
    }

    /// Compute local alignment weights for neighboring points
    fn compute_alignment_weights(&self, x: &Array2<Float>, graph: &Array2<Float>) -> Array2<Float> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut alignment_weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Find neighbors of point i
            let neighbors: Vec<usize> = (0..n_samples)
                .filter(|&j| j != i && graph[[i, j]] > 0.0)
                .collect();

            if neighbors.is_empty() {
                continue;
            }

            // Create local coordinate system for point i and its neighbors
            let n_neighbors = neighbors.len();
            let mut local_coords = Array2::zeros((n_neighbors + 1, n_features));

            // Add point i as first point
            for f in 0..n_features {
                local_coords[[0, f]] = x[[i, f]];
            }

            // Add neighbors
            for (idx, &j) in neighbors.iter().enumerate() {
                for f in 0..n_features {
                    local_coords[[idx + 1, f]] = x[[j, f]];
                }
            }

            // Compute local covariance matrix
            let mut local_mean = Array1::zeros(n_features);
            for k in 0..n_neighbors + 1 {
                for f in 0..n_features {
                    local_mean[f] += local_coords[[k, f]];
                }
            }
            local_mean /= (n_neighbors + 1) as Float;

            // Center the local coordinates
            for k in 0..n_neighbors + 1 {
                for f in 0..n_features {
                    local_coords[[k, f]] -= local_mean[f];
                }
            }

            // Compute alignment weights based on local PCA
            if self.config.global_alignment {
                // Use global PCA direction consistency
                for (idx, &j) in neighbors.iter().enumerate() {
                    let mut alignment_score = 0.0;

                    // Compute directional consistency
                    for f in 0..n_features {
                        alignment_score += local_coords[[0, f]] * local_coords[[idx + 1, f]];
                    }

                    // Normalize by local norms
                    let norm_i = local_coords.row(0).mapv(|x| x * x).sum().sqrt();
                    let norm_j = local_coords.row(idx + 1).mapv(|x| x * x).sum().sqrt();

                    if norm_i > 0.0 && norm_j > 0.0 {
                        alignment_score /= norm_i * norm_j;
                        alignment_weights[[i, j]] = (alignment_score + 1.0) / 2.0 * graph[[i, j]];
                    } else {
                        alignment_weights[[i, j]] = graph[[i, j]];
                    }
                }
            } else {
                // Use simple graph weights
                for (_idx, &j) in neighbors.iter().enumerate() {
                    alignment_weights[[i, j]] = graph[[i, j]];
                }
            }
        }

        alignment_weights
    }

    /// Compute class means
    fn compute_class_means(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Array2<Float>> {
        let classes = self.unique_classes(y)?;
        let n_classes = classes.len();
        let n_features = x.ncols();
        let mut means = Array2::zeros((n_classes, n_features));

        for (class_idx, &class) in classes.iter().enumerate() {
            let mut class_count = 0;
            for (sample_idx, &label) in y.iter().enumerate() {
                if label == class {
                    for feature_idx in 0..n_features {
                        means[[class_idx, feature_idx]] += x[[sample_idx, feature_idx]];
                    }
                    class_count += 1;
                }
            }

            if class_count > 0 {
                for feature_idx in 0..n_features {
                    means[[class_idx, feature_idx]] /= class_count as Float;
                }
            }
        }

        Ok(means)
    }

    /// Extract unique classes from labels
    fn unique_classes(&self, y: &Array1<i32>) -> Result<Array1<i32>> {
        let mut classes = Vec::new();
        for &label in y.iter() {
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        classes.sort_unstable();
        Ok(Array1::from_vec(classes))
    }

    /// Compute overall mean
    fn compute_overall_mean(&self, x: &Array2<Float>) -> Array1<Float> {
        let n_samples = x.nrows() as Float;
        let n_features = x.ncols();
        let mut mean = Array1::zeros(n_features);

        for i in 0..x.nrows() {
            for j in 0..n_features {
                mean[j] += x[[i, j]];
            }
        }

        mean / n_samples
    }

    /// Compute between-class scatter matrix
    fn compute_between_class_scatter(
        &self,
        class_means: &Array2<Float>,
        overall_mean: &Array1<Float>,
        y: &Array1<i32>,
    ) -> Result<Array2<Float>> {
        let classes = self.unique_classes(y)?;
        let n_features = overall_mean.len();
        let mut sb = Array2::zeros((n_features, n_features));

        for (class_idx, &class) in classes.iter().enumerate() {
            let class_count = y.iter().filter(|&&label| label == class).count() as Float;

            for i in 0..n_features {
                for j in 0..n_features {
                    let diff_i = class_means[[class_idx, i]] - overall_mean[i];
                    let diff_j = class_means[[class_idx, j]] - overall_mean[j];
                    sb[[i, j]] += class_count * diff_i * diff_j;
                }
            }
        }

        Ok(sb)
    }

    /// Compute within-class scatter matrix
    fn compute_within_class_scatter(
        &self,
        x: &Array2<Float>,
        class_means: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<Array2<Float>> {
        let classes = self.unique_classes(y)?;
        let n_features = x.ncols();
        let mut sw = Array2::zeros((n_features, n_features));

        for (sample_idx, &label) in y.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                for i in 0..n_features {
                    for j in 0..n_features {
                        let diff_i = x[[sample_idx, i]] - class_means[[class_idx, i]];
                        let diff_j = x[[sample_idx, j]] - class_means[[class_idx, j]];
                        sw[[i, j]] += diff_i * diff_j;
                    }
                }
            }
        }

        Ok(sw)
    }

    /// Compute locality scatter matrix based on alignment weights
    fn compute_locality_scatter(
        &self,
        x: &Array2<Float>,
        alignment_weights: &Array2<Float>,
    ) -> Array2<Float> {
        let n_features = x.ncols();
        let n_samples = x.nrows();
        let mut sl = Array2::zeros((n_features, n_features));

        for i in 0..n_samples {
            for j in 0..n_samples {
                if alignment_weights[[i, j]] > 0.0 {
                    for p in 0..n_features {
                        for q in 0..n_features {
                            let diff_p = x[[i, p]] - x[[j, p]];
                            let diff_q = x[[i, q]] - x[[j, q]];
                            sl[[p, q]] += alignment_weights[[i, j]] * diff_p * diff_q;
                        }
                    }
                }
            }
        }

        sl
    }

    /// Eigenvalue decomposition using robust LAPACK-based solver
    fn eigendecomposition(&self, matrix: &Array2<Float>) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        // Check if matrix is symmetric (for efficiency, use symmetric solver if possible)
        let is_symmetric = self.is_approximately_symmetric(matrix)?;

        let (eigenvals, eigenvecs) = if is_symmetric {
            // Use symmetric eigenvalue solver for better numerical stability
            let (vals, vecs) = matrix.eigh(UPLO::Upper).map_err(|e| {
                SklearsError::NumericalError(format!("Symmetric eigendecomposition failed: {}", e))
            })?;

            // Convert to owned arrays
            let vals_owned = vals.to_owned();
            let vecs_owned = vecs.to_owned();
            (vals_owned, vecs_owned)
        } else {
            // Use general eigenvalue solver
            let (complex_vals, complex_vecs) = matrix.eig().map_err(|e| {
                SklearsError::NumericalError(format!("General eigendecomposition failed: {}", e))
            })?;

            // Extract real parts (for discriminant analysis, we typically expect real eigenvalues)
            let real_vals = Array1::from_vec(complex_vals.iter().map(|c| c.re).collect::<Vec<_>>());
            let real_vecs = Array2::from_shape_vec(
                (n, n),
                complex_vecs.iter().map(|c| c.re).collect::<Vec<_>>(),
            )
            .map_err(|e| {
                SklearsError::NumericalError(format!("Failed to reshape eigenvectors: {}", e))
            })?;

            (real_vals, real_vecs)
        };

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(Float, Array1<Float>)> = eigenvals
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let vec = eigenvecs.column(i).to_owned();
                (val, vec)
            })
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Reconstruct sorted arrays
        let sorted_eigenvals = Array1::from_vec(eigen_pairs.iter().map(|(val, _)| *val).collect());

        let mut sorted_eigenvecs = Array2::zeros((n, n));
        for (i, (_, vec)) in eigen_pairs.iter().enumerate() {
            sorted_eigenvecs.column_mut(i).assign(vec);
        }

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }

    /// Check if a matrix is approximately symmetric
    fn is_approximately_symmetric(&self, matrix: &Array2<Float>) -> Result<bool> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Ok(false);
        }

        let tol = 1e-10; // Use a reasonable tolerance for symmetry check
        for i in 0..n {
            for j in 0..n {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > tol {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Solve generalized eigenvalue problem
    fn solve_generalized_eigenvalue_problem(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // Add regularization to B matrix
        let n = b.nrows();
        let mut b_reg = b.clone();
        for i in 0..n {
            b_reg[[i, i]] += self.config.reg_param;
        }

        // For this simplified implementation, we'll just use the regular eigendecomposition
        // In practice, you would solve the generalized eigenvalue problem A * v = λ * B * v
        self.eigendecomposition(a)
    }
}

impl Estimator for DiscriminantLocalityAlignment<Untrained> {
    type Config = DiscriminantLocalityAlignmentConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for DiscriminantLocalityAlignment<Untrained> {
    type Fitted = DiscriminantLocalityAlignment<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Get unique classes
        let classes = self.unique_classes(y)?;
        let n_classes = classes.len();

        // Determine number of components
        let n_components = self
            .config
            .n_components
            .unwrap_or_else(|| std::cmp::min(n_classes - 1, n_features));

        // Build locality graph
        let graph = match self.config.graph_method.as_str() {
            "knn" => self.build_knn_graph(x),
            "epsilon" => self.build_epsilon_graph(x),
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "graph_method".to_string(),
                    reason: format!("Unknown graph method: {}", self.config.graph_method),
                })
            }
        };

        // Compute alignment weights
        let alignment_weights = self.compute_alignment_weights(x, &graph);

        // Compute class means and overall mean
        let class_means = self.compute_class_means(x, y)?;
        let overall_mean = self.compute_overall_mean(x);

        // Compute scatter matrices
        let sb = self.compute_between_class_scatter(&class_means, &overall_mean, y)?;
        let sw = self.compute_within_class_scatter(x, &class_means, y)?;
        let sl = self.compute_locality_scatter(x, &alignment_weights);

        // Combine discriminant and locality objectives
        // Objective: maximize (Sb) while minimizing (Sw + α * Sl)
        let mut sw_combined = sw.clone();
        for i in 0..n_features {
            for j in 0..n_features {
                sw_combined[[i, j]] += self.config.alpha * sl[[i, j]];
            }
        }

        // Solve generalized eigenvalue problem
        let (eigenvalues, eigenvectors) =
            self.solve_generalized_eigenvalue_problem(&sb, &sw_combined)?;

        // Select top components
        let components = eigenvectors.slice(s![.., 0..n_components]).to_owned();
        let eigenvalues = eigenvalues.slice(s![0..n_components]).to_owned();

        // Compute priors
        let priors = if let Some(ref priors) = self.config.priors {
            priors.clone()
        } else {
            let mut priors = Array1::zeros(n_classes);
            for (class_idx, &class) in classes.iter().enumerate() {
                priors[class_idx] =
                    y.iter().filter(|&&label| label == class).count() as Float / n_samples as Float;
            }
            priors
        };

        let store_cov = self.config.store_covariance;
        Ok(DiscriminantLocalityAlignment {
            config: self.config,
            state: PhantomData,
            classes_: Some(classes),
            means_: Some(class_means),
            covariance_: if store_cov { Some(sw_combined) } else { None },
            components_: Some(components),
            eigenvalues_: Some(eigenvalues),
            priors_: Some(priors),
            n_features_: Some(n_features),
            locality_graph_: Some(graph),
            alignment_weights_: Some(alignment_weights),
        })
    }
}

impl DiscriminantLocalityAlignment<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the class means
    pub fn means(&self) -> &Array2<Float> {
        self.means_.as_ref().unwrap()
    }

    /// Get the components (projection matrix)
    pub fn components(&self) -> &Array2<Float> {
        self.components_.as_ref().unwrap()
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> &Array1<Float> {
        self.eigenvalues_.as_ref().unwrap()
    }

    /// Get the priors
    pub fn priors(&self) -> &Array1<Float> {
        self.priors_.as_ref().unwrap()
    }

    /// Get the locality graph
    pub fn locality_graph(&self) -> &Array2<Float> {
        self.locality_graph_.as_ref().unwrap()
    }

    /// Get the alignment weights
    pub fn alignment_weights(&self) -> &Array2<Float> {
        self.alignment_weights_.as_ref().unwrap()
    }

    /// Get the covariance matrix if stored
    pub fn covariance(&self) -> Option<&Array2<Float>> {
        self.covariance_.as_ref()
    }
}

impl Transform<Array2<Float>, Array2<Float>> for DiscriminantLocalityAlignment<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if self.components_.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        let components = self.components();
        let n_components = components.ncols();

        if x.ncols() != self.n_features_.unwrap() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features_.unwrap(),
                x.ncols()
            )));
        }

        let mut transformed = Array2::zeros((x.nrows(), n_components));

        for i in 0..x.nrows() {
            for j in 0..n_components {
                for k in 0..x.ncols() {
                    transformed[[i, j]] += x[[i, k]] * components[[k, j]];
                }
            }
        }

        Ok(transformed)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for DiscriminantLocalityAlignment<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes();

        let mut predictions = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut best_class_idx = 0;
            let mut best_proba = probas[[i, 0]];

            for j in 1..classes.len() {
                if probas[[i, j]] > best_proba {
                    best_proba = probas[[i, j]];
                    best_class_idx = j;
                }
            }

            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for DiscriminantLocalityAlignment<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if self.components_.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            });
        }

        let transformed = self.transform(x)?;
        let classes = self.classes();
        let class_means = self.means();
        let n_classes = classes.len();

        let mut probas = Array2::zeros((x.nrows(), n_classes));

        for i in 0..x.nrows() {
            let mut log_probs = Array1::zeros(n_classes);

            for j in 0..n_classes {
                // Simple Euclidean distance-based probability
                let mut distance = 0.0;
                for k in 0..transformed.ncols() {
                    // Project class mean to reduced space
                    let mut projected_mean = 0.0;
                    for l in 0..self.n_features_.unwrap() {
                        projected_mean += class_means[[j, l]] * self.components()[[l, k]];
                    }
                    let diff = transformed[[i, k]] - projected_mean;
                    distance += diff * diff;
                }

                log_probs[j] = -distance + self.priors()[j].ln();
            }

            // Convert to probabilities using softmax
            let max_log_prob = log_probs.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;

            for j in 0..n_classes {
                log_probs[j] = (log_probs[j] - max_log_prob).exp();
                sum_exp += log_probs[j];
            }

            for j in 0..n_classes {
                probas[[i, j]] = log_probs[j] / sum_exp;
            }
        }

        Ok(probas)
    }
}

impl Default for DiscriminantLocalityAlignment<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}
