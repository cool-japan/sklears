//! Locality Preserving Discriminant Analysis implementation

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Locality Preserving Discriminant Analysis
#[derive(Debug, Clone)]
pub struct LocalityPreservingDiscriminantAnalysisConfig {
    /// Number of nearest neighbors for graph construction
    pub n_neighbors: usize,
    /// Weight function for graph construction ("heat_kernel", "binary", "polynomial")
    pub weight_function: String,
    /// Parameter for heat kernel weight function (σ²)
    pub heat_kernel_param: Float,
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
    /// Graph construction method ("knn", "epsilon")
    pub graph_method: String,
    /// Epsilon parameter for epsilon-neighborhood graph
    pub epsilon: Float,
    /// Whether to use symmetric graph
    pub symmetric_graph: bool,
    /// Locality preservation weight
    pub locality_weight: Float,
}

impl Default for LocalityPreservingDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            weight_function: "heat_kernel".to_string(),
            heat_kernel_param: 1.0,
            n_components: None,
            reg_param: 1e-6,
            priors: None,
            store_covariance: false,
            tol: 1e-8,
            graph_method: "knn".to_string(),
            epsilon: 1.0,
            symmetric_graph: true,
            locality_weight: 1.0,
        }
    }
}

/// Locality Preserving Discriminant Analysis
///
/// A dimensionality reduction technique that combines Linear Discriminant Analysis
/// with locality preserving projections. It preserves the local manifold structure
/// while maximizing class separability.
#[derive(Debug, Clone)]
pub struct LocalityPreservingDiscriminantAnalysis<State = Untrained> {
    config: LocalityPreservingDiscriminantAnalysisConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    means_: Option<Array2<Float>>,
    covariance_: Option<Array2<Float>>,
    components_: Option<Array2<Float>>,
    eigenvalues_: Option<Array1<Float>>,
    priors_: Option<Array1<Float>>,
    n_features_: Option<usize>,
    graph_matrix_: Option<Array2<Float>>,
}

impl LocalityPreservingDiscriminantAnalysis<Untrained> {
    /// Create a new LocalityPreservingDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            config: LocalityPreservingDiscriminantAnalysisConfig::default(),
            state: PhantomData,
            classes_: None,
            means_: None,
            covariance_: None,
            components_: None,
            eigenvalues_: None,
            priors_: None,
            n_features_: None,
            graph_matrix_: None,
        }
    }

    /// Set the number of nearest neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.config.n_neighbors = n_neighbors;
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

    /// Set the locality preservation weight
    pub fn locality_weight(mut self, locality_weight: Float) -> Self {
        self.config.locality_weight = locality_weight;
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

            for &(j, dist) in neighbors.iter().take(self.config.n_neighbors) {
                let weight = self.compute_weight(dist);
                graph[[i, j]] = weight;

                if self.config.symmetric_graph {
                    graph[[j, i]] = weight;
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
                    let weight = self.compute_weight(distances[[i, j]]);
                    graph[[i, j]] = weight;

                    if self.config.symmetric_graph {
                        graph[[j, i]] = weight;
                    }
                }
            }
        }

        graph
    }

    /// Compute weight based on distance and weight function
    fn compute_weight(&self, distance: Float) -> Float {
        match self.config.weight_function.as_str() {
            "heat_kernel" => (-distance * distance / self.config.heat_kernel_param).exp(),
            "binary" => 1.0,
            "polynomial" => {
                if distance == 0.0 {
                    1.0
                } else {
                    1.0 / (1.0 + distance)
                }
            }
            _ => 1.0,
        }
    }

    /// Compute the graph Laplacian matrix
    fn compute_graph_laplacian(&self, weight_matrix: &Array2<Float>) -> Array2<Float> {
        let n_samples = weight_matrix.nrows();
        let mut laplacian = Array2::zeros((n_samples, n_samples));

        // Compute degree matrix
        let mut degrees = Array1::zeros(n_samples);
        for i in 0..n_samples {
            degrees[i] = weight_matrix.row(i).sum();
        }

        // L = D - W
        for i in 0..n_samples {
            laplacian[[i, i]] = degrees[i];
            for j in 0..n_samples {
                if i != j {
                    laplacian[[i, j]] = -weight_matrix[[i, j]];
                }
            }
        }

        laplacian
    }

    /// Compute between-class scatter matrix
    fn compute_between_class_scatter(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        overall_mean: &Array1<Float>,
        class_means: &Array2<Float>,
        classes: &[i32],
    ) -> Array2<Float> {
        let n_features = x.ncols();
        let mut sb = Array2::zeros((n_features, n_features));

        for (i, &class) in classes.iter().enumerate() {
            let class_count = y.iter().filter(|&&yi| yi == class).count() as Float;
            let class_mean = class_means.row(i);
            let diff = &class_mean - overall_mean;

            for j in 0..n_features {
                for k in 0..n_features {
                    sb[[j, k]] += class_count * diff[j] * diff[k];
                }
            }
        }

        sb
    }

    /// Compute within-class scatter matrix
    fn compute_within_class_scatter(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        class_means: &Array2<Float>,
        classes: &[i32],
    ) -> Array2<Float> {
        let (n_samples, n_features) = x.dim();
        let mut sw = Array2::zeros((n_features, n_features));

        for i in 0..n_samples {
            let class_idx = classes.iter().position(|&c| c == y[i]).unwrap();
            let class_mean = class_means.row(class_idx);
            let sample = x.row(i);
            let diff = &sample - &class_mean;

            for j in 0..n_features {
                for k in 0..n_features {
                    sw[[j, k]] += diff[j] * diff[k];
                }
            }
        }

        sw
    }

    /// Compute locality preserving scatter matrix
    fn compute_locality_scatter(
        &self,
        x: &Array2<Float>,
        laplacian: &Array2<Float>,
    ) -> Array2<Float> {
        // S_l = X^T L X
        let xt = x.t();
        let xl = xt.dot(laplacian);
        xl.dot(x)
    }

    /// Solve generalized eigenvalue problem with regularization
    fn solve_generalized_eigenvalue_problem(
        &self,
        numerator: &Array2<Float>,
        denominator: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = numerator.nrows();

        // Add regularization to denominator to ensure it's positive definite
        let mut regularized_denom = denominator.clone();
        for i in 0..n {
            regularized_denom[[i, i]] += self.config.reg_param;
        }

        // Simplified eigenvalue decomposition (using power iteration for largest eigenvalues)
        let n_components = self.config.n_components.unwrap_or((n - 1).min(n));
        let mut eigenvalues = Array1::zeros(n_components);
        let mut eigenvectors = Array2::zeros((n, n_components));

        // Initialize with random vectors
        for i in 0..n_components {
            eigenvectors.column_mut(i).fill(1.0 / (n as Float).sqrt());
        }

        // Power iteration for each component
        for comp in 0..n_components {
            let mut v = eigenvectors.column(comp).to_owned();

            for _iter in 0..100 {
                let v_old = v.clone();

                // v = A * v where A = inv(denominator) * numerator
                // Simplified: use diagonal approximation for inverse
                let mut av = Array1::zeros(n);
                for i in 0..n {
                    for j in 0..n {
                        av[i] += numerator[[i, j]] * v[j] / regularized_denom[[j, j]];
                    }
                }

                // Normalize
                let norm = av.iter().map(|&x| x * x).sum::<Float>().sqrt();
                if norm > self.config.tol {
                    v = av / norm;
                } else {
                    break;
                }

                // Check convergence
                let diff: Float = v.iter().zip(v_old.iter()).map(|(a, b)| (a - b).abs()).sum();
                if diff < self.config.tol {
                    break;
                }
            }

            // Compute eigenvalue
            let mut num_val = 0.0;
            let mut denom_val = 0.0;
            for i in 0..n {
                for j in 0..n {
                    num_val += v[i] * numerator[[i, j]] * v[j];
                    denom_val += v[i] * regularized_denom[[i, j]] * v[j];
                }
            }

            eigenvalues[comp] = if denom_val.abs() > self.config.tol {
                num_val / denom_val
            } else {
                0.0
            };

            eigenvectors.column_mut(comp).assign(&v);
        }

        Ok((eigenvalues, eigenvectors))
    }
}

impl Default for LocalityPreservingDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LocalityPreservingDiscriminantAnalysis<Untrained> {
    type Config = LocalityPreservingDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for LocalityPreservingDiscriminantAnalysis<Untrained> {
    type Fitted = LocalityPreservingDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        // Basic validation
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        // Get unique classes
        let mut classes: Vec<i32> = y
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        // Calculate class means and overall mean
        let mut means = Array2::zeros((n_classes, n_features));
        let mut class_counts = Array1::zeros(n_classes);
        let mut overall_mean = Array1::zeros(n_features);

        // Compute overall mean
        for i in 0..n_samples {
            for j in 0..n_features {
                overall_mean[j] += x[[i, j]];
            }
        }
        overall_mean /= n_samples as Float;

        // Compute class means
        for (i, &class) in classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&yi| yi == class).collect();
            let class_samples: Vec<ArrayView1<Float>> = x
                .axis_iter(Axis(0))
                .enumerate()
                .filter(|(j, _)| class_mask[*j])
                .map(|(_, sample)| sample)
                .collect();

            let count = class_samples.len();
            class_counts[i] = count as Float;

            if count == 0 {
                return Err(SklearsError::InvalidInput(format!(
                    "Class {} has no samples",
                    class
                )));
            }

            let mut class_mean = Array1::zeros(n_features);
            for sample in &class_samples {
                for j in 0..n_features {
                    class_mean[j] += sample[j];
                }
            }
            class_mean /= count as Float;
            means.row_mut(i).assign(&class_mean);
        }

        // Build locality preserving graph
        let weight_matrix = match self.config.graph_method.as_str() {
            "knn" => self.build_knn_graph(x),
            "epsilon" => self.build_epsilon_graph(x),
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown graph method: {}",
                    self.config.graph_method
                )))
            }
        };

        let laplacian = self.compute_graph_laplacian(&weight_matrix);

        // Compute scatter matrices
        let sb = self.compute_between_class_scatter(x, y, &overall_mean, &means, &classes);
        let sw = self.compute_within_class_scatter(x, y, &means, &classes);
        let sl = self.compute_locality_scatter(x, &laplacian);

        // Combined scatter matrices for LPDA
        // Maximize: S_b - λ * S_l
        // Subject to: S_w
        let numerator = &sb - &(&sl * self.config.locality_weight);
        let denominator = &sw;

        // Solve generalized eigenvalue problem
        let (eigenvalues, eigenvectors) =
            self.solve_generalized_eigenvalue_problem(&numerator, denominator)?;

        // Calculate priors
        let priors = if let Some(ref p) = self.config.priors {
            p.clone()
        } else {
            &class_counts / n_samples as Float
        };

        // Store covariance if requested
        let covariance = if self.config.store_covariance {
            Some(sw.clone())
        } else {
            None
        };

        Ok(LocalityPreservingDiscriminantAnalysis {
            config: self.config,
            state: PhantomData,
            classes_: Some(Array1::from(classes)),
            means_: Some(means),
            covariance_: covariance,
            components_: Some(eigenvectors),
            eigenvalues_: Some(eigenvalues),
            priors_: Some(priors),
            n_features_: Some(n_features),
            graph_matrix_: Some(weight_matrix),
        })
    }
}

impl LocalityPreservingDiscriminantAnalysis<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().expect("Model is trained")
    }

    /// Get the means
    pub fn means(&self) -> &Array2<Float> {
        self.means_.as_ref().expect("Model is trained")
    }

    /// Get the components (eigenvectors)
    pub fn components(&self) -> &Array2<Float> {
        self.components_.as_ref().expect("Model is trained")
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> &Array1<Float> {
        self.eigenvalues_.as_ref().expect("Model is trained")
    }

    /// Get the priors
    pub fn priors(&self) -> &Array1<Float> {
        self.priors_.as_ref().expect("Model is trained")
    }

    /// Get the covariance matrix (if stored)
    pub fn covariance(&self) -> Option<&Array2<Float>> {
        self.covariance_.as_ref()
    }

    /// Get the graph matrix
    pub fn graph_matrix(&self) -> &Array2<Float> {
        self.graph_matrix_.as_ref().expect("Model is trained")
    }
}

impl Transform<Array2<Float>, Array2<Float>> for LocalityPreservingDiscriminantAnalysis<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        validate::check_n_features(x, n_features)?;

        let components = self.components_.as_ref().expect("Model is trained");

        // Project data onto the discriminant components
        let transformed = x.dot(components);
        Ok(transformed)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for LocalityPreservingDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().expect("Model is trained");

        let predictions: Vec<i32> = probas
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0;
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from(predictions))
    }
}

impl PredictProba<Array2<Float>, Array2<Float>>
    for LocalityPreservingDiscriminantAnalysis<Trained>
{
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        validate::check_n_features(x, n_features)?;

        let transformed = self.transform(x)?;
        let means = self.means_.as_ref().expect("Model is trained");
        let priors = self.priors_.as_ref().expect("Model is trained");
        let components = self.components_.as_ref().expect("Model is trained");

        let n_classes = means.nrows();
        let n_samples = x.nrows();

        // Transform class means to the new space
        let transformed_means = means.dot(components);

        let mut log_likelihoods = Array2::zeros((n_samples, n_classes));

        for i in 0..n_classes {
            let class_mean = transformed_means.row(i);

            for j in 0..n_samples {
                let sample = transformed.row(j);
                let diff = &sample - &class_mean;

                // Simple squared distance in transformed space
                let dist_sq = diff.iter().map(|&d| d * d).sum::<Float>();

                log_likelihoods[[j, i]] = priors[i].ln() - 0.5 * dist_sq;
            }
        }

        // Convert log-likelihoods to probabilities using softmax
        let mut probabilities = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let row = log_likelihoods.row(i);
            let max_log_like = row.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

            let exp_likes: Vec<Float> = row.iter().map(|&x| (x - max_log_like).exp()).collect();
            let sum_exp: Float = exp_likes.iter().sum();

            for j in 0..n_classes {
                probabilities[[i, j]] = exp_likes[j] / sum_exp;
            }
        }

        Ok(probabilities)
    }
}
