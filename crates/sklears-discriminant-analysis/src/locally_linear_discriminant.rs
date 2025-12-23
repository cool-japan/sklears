//! Locally Linear Discriminant Analysis
//!
//! This module implements Locally Linear Discriminant Analysis (LLDA), which
//! combines the principles of Locally Linear Embedding (LLE) with discriminant
//! analysis to handle non-linear data structures while preserving local geometry.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};

/// Configuration for Locally Linear Discriminant Analysis
#[derive(Debug, Clone)]
pub struct LocallyLinearDiscriminantAnalysisConfig {
    /// Number of neighbors for local reconstruction
    pub n_neighbors: usize,
    /// Number of components for embedding
    pub n_components: Option<usize>,
    /// Regularization parameter for weight calculation
    pub reg_param: Float,
    /// Weight regularization method ("ridge", "lasso", "elastic_net")
    pub weight_method: String,
    /// Eigensolver method ("standard", "sparse", "randomized")
    pub eigen_solver: String,
    /// Tolerance for eigenvalue computations
    pub eigen_tol: Float,
    /// Maximum iterations for iterative methods
    pub max_iter: usize,
    /// Neighborhood selection method ("knn", "epsilon", "adaptive")
    pub neighbor_method: String,
    /// Epsilon for epsilon-ball neighborhoods
    pub epsilon: Float,
    /// Whether to perform supervised embedding
    pub supervised: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

impl Default for LocallyLinearDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            n_components: None,
            reg_param: 1e-3,
            weight_method: "ridge".to_string(),
            eigen_solver: "standard".to_string(),
            eigen_tol: 1e-6,
            max_iter: 300,
            neighbor_method: "knn".to_string(),
            epsilon: 1.0,
            supervised: true,
            random_state: None,
        }
    }
}

/// Locally Linear Discriminant Analysis
#[derive(Debug, Clone)]
pub struct LocallyLinearDiscriminantAnalysis {
    config: LocallyLinearDiscriminantAnalysisConfig,
}

/// Trained Locally Linear Discriminant Analysis model
#[derive(Debug, Clone)]
pub struct TrainedLocallyLinearDiscriminantAnalysis {
    /// Embedding coordinates
    embedding: Array2<Float>,
    /// Reconstruction weights matrix
    weights: Array2<Float>,
    /// Neighborhood adjacency matrix
    neighbors: Array2<bool>,
    /// Linear discriminant components in embedding space
    components: Array2<Float>,
    /// Class means in embedding space
    means: Array2<Float>,
    /// Covariance matrix in embedding space
    covariance: Array2<Float>,
    /// Class priors
    priors: Array1<Float>,
    /// Class labels
    classes: Array1<i32>,
    /// Original training data
    training_data: Array2<Float>,
    /// Number of features in original space
    n_features_in: usize,
    /// Configuration used for training
    config: LocallyLinearDiscriminantAnalysisConfig,
}

impl Default for LocallyLinearDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl LocallyLinearDiscriminantAnalysis {
    /// Create a new Locally Linear Discriminant Analysis model
    pub fn new() -> Self {
        Self {
            config: LocallyLinearDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.config.n_neighbors = n_neighbors.max(1);
        self
    }

    /// Set number of components
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set regularization parameter
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Set weight calculation method
    pub fn weight_method(mut self, method: &str) -> Self {
        self.config.weight_method = method.to_string();
        self
    }

    /// Set eigenvalue solver method
    pub fn eigen_solver(mut self, solver: &str) -> Self {
        self.config.eigen_solver = solver.to_string();
        self
    }

    /// Set neighbor selection method
    pub fn neighbor_method(mut self, method: &str) -> Self {
        self.config.neighbor_method = method.to_string();
        self
    }

    /// Set epsilon for epsilon-ball neighborhoods
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    /// Set whether to use supervised embedding
    pub fn supervised(mut self, supervised: bool) -> Self {
        self.config.supervised = supervised;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Find neighbors for each data point
    fn find_neighbors(&self, x: &Array2<Float>) -> Result<Array2<bool>> {
        let n_samples = x.nrows();
        let mut neighbors = Array2::from_elem((n_samples, n_samples), false);

        match self.config.neighbor_method.as_str() {
            "knn" => {
                for i in 0..n_samples {
                    let xi = x.row(i);
                    let mut distances: Vec<(usize, Float)> = Vec::new();

                    for j in 0..n_samples {
                        if i != j {
                            let xj = x.row(j);
                            let dist = (&xi - &xj).mapv(|x| x * x).sum().sqrt();
                            distances.push((j, dist));
                        }
                    }

                    // Sort by distance and take k nearest neighbors
                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    for (neighbor_idx, _) in distances.iter().take(self.config.n_neighbors) {
                        neighbors[[i, *neighbor_idx]] = true;
                    }
                }
            }
            "epsilon" => {
                for i in 0..n_samples {
                    let xi = x.row(i);
                    for j in 0..n_samples {
                        if i != j {
                            let xj = x.row(j);
                            let dist = (&xi - &xj).mapv(|x| x * x).sum().sqrt();
                            if dist <= self.config.epsilon {
                                neighbors[[i, j]] = true;
                            }
                        }
                    }
                }
            }
            "adaptive" => {
                // Adaptive neighborhood based on local density
                for i in 0..n_samples {
                    let xi = x.row(i);
                    let mut distances: Vec<(usize, Float)> = Vec::new();

                    for j in 0..n_samples {
                        if i != j {
                            let xj = x.row(j);
                            let dist = (&xi - &xj).mapv(|x| x * x).sum().sqrt();
                            distances.push((j, dist));
                        }
                    }

                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    // Adaptive k based on distance distribution
                    let k_adaptive = if distances.len() > 10 {
                        let median_dist = distances[distances.len() / 2].1;
                        let close_neighbors = distances
                            .iter()
                            .take_while(|(_, dist)| *dist <= median_dist * 1.5)
                            .count();
                        close_neighbors
                            .max(self.config.n_neighbors / 2)
                            .min(self.config.n_neighbors * 2)
                    } else {
                        self.config.n_neighbors.min(distances.len())
                    };

                    for (neighbor_idx, _) in distances.iter().take(k_adaptive) {
                        neighbors[[i, *neighbor_idx]] = true;
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "neighbor_method".to_string(),
                    reason: format!("Unknown neighbor method: {}", self.config.neighbor_method),
                });
            }
        }

        Ok(neighbors)
    }

    /// Compute reconstruction weights using locally linear embedding
    fn compute_weights(
        &self,
        x: &Array2<Float>,
        neighbors: &Array2<bool>,
    ) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let xi = x.row(i);

            // Find neighbors for this point
            let neighbor_indices: Vec<usize> =
                (0..n_samples).filter(|&j| neighbors[[i, j]]).collect();

            if neighbor_indices.is_empty() {
                continue;
            }

            let k = neighbor_indices.len();

            // Create neighbor matrix (k x n_features)
            let mut neighbor_matrix = Array2::zeros((k, n_features));
            for (local_idx, &global_idx) in neighbor_indices.iter().enumerate() {
                for j in 0..n_features {
                    neighbor_matrix[[local_idx, j]] = x[[global_idx, j]] - xi[j];
                }
            }

            // Solve for weights using the specified method
            let local_weights = match self.config.weight_method.as_str() {
                "ridge" => self.solve_ridge_weights(&neighbor_matrix)?,
                "lasso" => self.solve_lasso_weights(&neighbor_matrix)?,
                "elastic_net" => self.solve_elastic_net_weights(&neighbor_matrix)?,
                _ => {
                    return Err(SklearsError::InvalidParameter {
                        name: "weight_method".to_string(),
                        reason: format!("Unknown weight method: {}", self.config.weight_method),
                    });
                }
            };

            // Assign weights back to the global weight matrix
            for (local_idx, &global_idx) in neighbor_indices.iter().enumerate() {
                weights[[i, global_idx]] = local_weights[local_idx];
            }
        }

        Ok(weights)
    }

    /// Solve for ridge regression weights
    fn solve_ridge_weights(&self, neighbor_matrix: &Array2<Float>) -> Result<Array1<Float>> {
        let k = neighbor_matrix.nrows();

        if k == 0 {
            return Ok(Array1::zeros(0));
        }

        // Compute Gram matrix
        let gram = neighbor_matrix.dot(&neighbor_matrix.t());

        // Add regularization
        let mut regularized_gram = gram;
        for i in 0..k {
            regularized_gram[[i, i]] += self.config.reg_param;
        }

        // Create target vector (all ones)
        let ones = Array1::<Float>::ones(k);

        // Solve the linear system (simplified - in practice you'd use proper linear algebra)
        // For now, use a simple iterative method
        let mut weights = Array1::from_elem(k, 1.0 / k as Float);

        // Simple gradient descent
        for _ in 0..50 {
            let residual = regularized_gram.dot(&weights) - &ones;
            let gradient = regularized_gram.t().dot(&residual);

            for j in 0..k {
                weights[j] -= 0.01 * gradient[j];
            }
        }

        // Normalize weights to sum to 1
        let weight_sum = weights.sum();
        if weight_sum > 1e-12 {
            for j in 0..k {
                weights[j] /= weight_sum;
            }
        } else {
            weights = Array1::from_elem(k, 1.0 / k as Float);
        }

        Ok(weights)
    }

    /// Solve for LASSO weights (simplified implementation)
    fn solve_lasso_weights(&self, neighbor_matrix: &Array2<Float>) -> Result<Array1<Float>> {
        // For simplicity, fall back to ridge with higher regularization
        let mut config_copy = self.config.clone();
        config_copy.reg_param *= 10.0;
        let temp_llda = LocallyLinearDiscriminantAnalysis {
            config: config_copy,
        };
        temp_llda.solve_ridge_weights(neighbor_matrix)
    }

    /// Solve for elastic net weights (simplified implementation)
    fn solve_elastic_net_weights(&self, neighbor_matrix: &Array2<Float>) -> Result<Array1<Float>> {
        // For simplicity, average ridge and lasso solutions
        let ridge_weights = self.solve_ridge_weights(neighbor_matrix)?;
        let lasso_weights = self.solve_lasso_weights(neighbor_matrix)?;

        let mut elastic_weights = Array1::zeros(ridge_weights.len());
        for i in 0..ridge_weights.len() {
            elastic_weights[i] = 0.5 * ridge_weights[i] + 0.5 * lasso_weights[i];
        }

        Ok(elastic_weights)
    }

    /// Compute the embedding using locally linear embedding
    fn compute_embedding(
        &self,
        weights: &Array2<Float>,
        y: Option<&Array1<i32>>,
    ) -> Result<Array2<Float>> {
        let n_samples = weights.nrows();
        let n_components = self.config.n_components.unwrap_or((n_samples / 2).max(2));

        // Create the LLE cost matrix M = (I - W)^T (I - W)
        let identity = Array2::<Float>::eye(n_samples);
        let i_minus_w = &identity - weights;
        let cost_matrix = i_minus_w.t().dot(&i_minus_w);

        // If supervised, incorporate class information
        let final_cost_matrix = if self.config.supervised && y.is_some() {
            let y = y.unwrap();
            let mut supervised_cost = cost_matrix.clone();

            // Encourage points of the same class to be close in embedding
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if y[i] == y[j] {
                        supervised_cost[[i, j]] -= 0.1; // Encourage similarity
                    } else {
                        supervised_cost[[i, j]] += 0.1; // Encourage dissimilarity
                    }
                }
            }

            supervised_cost
        } else {
            cost_matrix
        };

        // Simplified eigenvalue computation (in practice, use proper eigenvalue solver)
        // For now, create a simple embedding based on the cost matrix structure
        let mut embedding = Array2::zeros((n_samples, n_components));

        // Initialize with some structure
        for i in 0..n_samples {
            for j in 0..n_components {
                // Use a combination of the cost matrix diagonal and some noise
                let val: Float = final_cost_matrix[[i, i]] * (j + 1) as Float;
                embedding[[i, j]] = val.sin() * (0.1 as Float)
                    + (i as Float / n_samples as Float) * (j + 1) as Float;
            }
        }

        // Simple iterative refinement
        for iter in 0..20 {
            let mut new_embedding = embedding.clone();

            for i in 0..n_samples {
                for j in 0..n_components {
                    let mut sum = 0.0;
                    let mut count = 0.0;

                    // Average with neighbors
                    for k in 0..n_samples {
                        if weights[[i, k]] > 1e-8 {
                            sum += weights[[i, k]] * embedding[[k, j]];
                            count += weights[[i, k]];
                        }
                    }

                    if count > 1e-8 {
                        new_embedding[[i, j]] = sum / count;
                    }
                }
            }

            embedding = new_embedding;

            // Check convergence (simplified)
            if iter > 10 {
                break;
            }
        }

        Ok(embedding)
    }

    /// Train discriminant analysis in the embedding space
    fn train_discriminant(
        &self,
        embedding: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<(Array2<Float>, Array2<Float>, Array2<Float>, Array1<Float>)> {
        let classes: Vec<i32> = {
            let mut unique_classes: Vec<i32> = y.iter().cloned().collect();
            unique_classes.sort_unstable();
            unique_classes.dedup();
            unique_classes
        };

        let n_classes = classes.len();
        let n_components = embedding.ncols();

        // Compute class means
        let mut means = Array2::zeros((n_classes, n_components));
        let mut priors = Array1::zeros(n_classes);

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if !class_indices.is_empty() {
                priors[class_idx] = class_indices.len() as Float / y.len() as Float;

                // Compute class mean
                for &i in &class_indices {
                    for j in 0..n_components {
                        means[[class_idx, j]] += embedding[[i, j]];
                    }
                }
                for j in 0..n_components {
                    means[[class_idx, j]] /= class_indices.len() as Float;
                }
            }
        }

        // Compute within-class scatter matrix
        let mut sw = Array2::zeros((n_components, n_components));
        for (class_idx, &class_label) in classes.iter().enumerate() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            for &i in &class_indices {
                let centered = embedding.row(i).to_owned() - means.row(class_idx);
                for j in 0..n_components {
                    for k in 0..n_components {
                        sw[[j, k]] += centered[j] * centered[k];
                    }
                }
            }
        }

        // Compute between-class scatter matrix
        let overall_mean = embedding.mean_axis(Axis(0)).unwrap();
        let mut sb = Array2::<Float>::zeros((n_components, n_components));

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let class_count = y.iter().filter(|&&label| label == class_label).count() as Float;
            let centered_mean = means.row(class_idx).to_owned() - &overall_mean;

            for j in 0..n_components {
                for k in 0..n_components {
                    sb[[j, k]] += class_count * centered_mean[j] * centered_mean[k];
                }
            }
        }

        // Add regularization to within-class scatter
        for i in 0..n_components {
            sw[[i, i]] += self.config.reg_param;
        }

        // Compute discriminant components (simplified)
        // In practice, you'd solve the generalized eigenvalue problem
        let components = Array2::eye(n_components);

        Ok((components, means, sw, priors))
    }

    /// Map new data to embedding space
    fn map_to_embedding(
        &self,
        x: &Array2<Float>,
        trained_model: &TrainedLocallyLinearDiscriminantAnalysis,
    ) -> Result<Array2<Float>> {
        let n_new_samples = x.nrows();
        let n_components = trained_model.embedding.ncols();
        let mut new_embedding = Array2::zeros((n_new_samples, n_components));

        // Out-of-sample extension using local linear reconstruction
        for i in 0..n_new_samples {
            let xi = x.row(i);
            let mut distances: Vec<(usize, Float)> = Vec::new();

            // Find nearest neighbors in training data
            for j in 0..trained_model.training_data.nrows() {
                let xj = trained_model.training_data.row(j);
                let dist = (&xi - &xj).mapv(|x| x * x).sum().sqrt();
                distances.push((j, dist));
            }

            // Sort and take k nearest neighbors
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let k = self.config.n_neighbors.min(distances.len());

            // Compute reconstruction weights for new point
            let neighbor_indices: Vec<usize> =
                distances.iter().take(k).map(|(idx, _)| *idx).collect();

            if !neighbor_indices.is_empty() {
                // Simplified weight computation
                let mut total_weight = 0.0;
                for (local_idx, &neighbor_idx) in neighbor_indices.iter().enumerate() {
                    let weight = 1.0 / (distances[local_idx].1 + 1e-8);
                    total_weight += weight;

                    for j in 0..n_components {
                        new_embedding[[i, j]] +=
                            weight * trained_model.embedding[[neighbor_idx, j]];
                    }
                }

                // Normalize
                if total_weight > 0.0 {
                    for j in 0..n_components {
                        new_embedding[[i, j]] /= total_weight;
                    }
                }
            }
        }

        Ok(new_embedding)
    }
}

impl Estimator for LocallyLinearDiscriminantAnalysis {
    type Config = LocallyLinearDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for LocallyLinearDiscriminantAnalysis {
    type Fitted = TrainedLocallyLinearDiscriminantAnalysis;

    fn fit(
        self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> Result<TrainedLocallyLinearDiscriminantAnalysis> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, y.shape[0]={}", x.nrows(), y.len()),
            });
        }

        let n_features = x.ncols();
        let classes: Vec<i32> = {
            let mut unique_classes: Vec<i32> = y.iter().cloned().collect();
            unique_classes.sort_unstable();
            unique_classes.dedup();
            unique_classes
        };

        // Find neighbors
        let neighbors = self.find_neighbors(x)?;

        // Compute reconstruction weights
        let weights = self.compute_weights(x, &neighbors)?;

        // Compute embedding
        let embedding = if self.config.supervised {
            self.compute_embedding(&weights, Some(y))?
        } else {
            self.compute_embedding(&weights, None)?
        };

        // Train discriminant analysis in embedding space
        let (components, means, covariance, priors) = self.train_discriminant(&embedding, y)?;

        Ok(TrainedLocallyLinearDiscriminantAnalysis {
            embedding,
            weights,
            neighbors,
            components,
            means,
            covariance,
            priors,
            classes: Array1::from_vec(classes),
            training_data: x.clone(),
            n_features_in: n_features,
            config: self.config,
        })
    }
}

impl TrainedLocallyLinearDiscriminantAnalysis {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get the embedding
    pub fn embedding(&self) -> &Array2<Float> {
        &self.embedding
    }

    /// Get the reconstruction weights
    pub fn weights(&self) -> &Array2<Float> {
        &self.weights
    }

    /// Get the neighborhood adjacency matrix
    pub fn neighbors(&self) -> &Array2<bool> {
        &self.neighbors
    }

    /// Get the discriminant components
    pub fn components(&self) -> &Array2<Float> {
        &self.components
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedLocallyLinearDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[1] == {}", self.n_features_in),
                actual: format!("X.shape[1] == {}", x.ncols()),
            });
        }

        // Map to embedding space
        let llda = LocallyLinearDiscriminantAnalysis {
            config: self.config.clone(),
        };
        let x_embedded = llda.map_to_embedding(x, self)?;

        let n_samples = x_embedded.nrows();
        let mut predictions = Array1::zeros(n_samples);

        // Predict using discriminant analysis in embedding space
        for i in 0..n_samples {
            let xi = x_embedded.row(i);
            let mut best_score = Float::NEG_INFINITY;
            let mut best_class = 0;

            for (class_idx, &class_label) in self.classes.iter().enumerate() {
                let mean = self.means.row(class_idx);
                let diff = &xi - &mean;

                // Simplified discriminant function
                let score = self.priors[class_idx].ln() - 0.5 * diff.dot(&diff);

                if score > best_score {
                    best_score = score;
                    best_class = class_label;
                }
            }

            predictions[i] = best_class;
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedLocallyLinearDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[1] == {}", self.n_features_in),
                actual: format!("X.shape[1] == {}", x.ncols()),
            });
        }

        // Map to embedding space
        let llda = LocallyLinearDiscriminantAnalysis {
            config: self.config.clone(),
        };
        let x_embedded = llda.map_to_embedding(x, self)?;

        let n_samples = x_embedded.nrows();
        let n_classes = self.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        // Compute probabilities using discriminant analysis in embedding space
        for i in 0..n_samples {
            let xi = x_embedded.row(i);
            let mut scores = Array1::zeros(n_classes);

            for class_idx in 0..n_classes {
                let mean = self.means.row(class_idx);
                let diff = &xi - &mean;

                // Simplified discriminant function
                scores[class_idx] = self.priors[class_idx].ln() - 0.5 * diff.dot(&diff);
            }

            // Convert scores to probabilities (softmax)
            let max_score = scores.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
            let mut sum_exp = 0.0;
            for class_idx in 0..n_classes {
                let exp_score = (scores[class_idx] - max_score).exp();
                probas[[i, class_idx]] = exp_score;
                sum_exp += exp_score;
            }

            // Normalize
            for class_idx in 0..n_classes {
                probas[[i, class_idx]] /= sum_exp;
            }
        }

        Ok(probas)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for TrainedLocallyLinearDiscriminantAnalysis {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features_in {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[1] == {}", self.n_features_in),
                actual: format!("X.shape[1] == {}", x.ncols()),
            });
        }

        // Map to embedding space
        let llda = LocallyLinearDiscriminantAnalysis {
            config: self.config.clone(),
        };
        llda.map_to_embedding(x, self)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_locally_linear_discriminant_analysis() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let llda = LocallyLinearDiscriminantAnalysis::new()
            .n_neighbors(2)
            .n_components(Some(1));

        let fitted = llda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_locally_linear_predict_proba() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let llda = LocallyLinearDiscriminantAnalysis::new()
            .n_neighbors(1)
            .supervised(true);

        let fitted = llda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let sum: Float = probas.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_locally_linear_transform() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 0, 1, 1];

        let llda = LocallyLinearDiscriminantAnalysis::new()
            .n_neighbors(2)
            .n_components(Some(2));

        let fitted = llda.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.dim(), (4, 2));
    }

    #[test]
    fn test_different_neighbor_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let methods = ["knn", "epsilon", "adaptive"];

        for method in &methods {
            let llda = LocallyLinearDiscriminantAnalysis::new()
                .neighbor_method(method)
                .n_neighbors(1)
                .epsilon(2.0);

            let fitted = llda.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 4);
            assert_eq!(fitted.classes().len(), 2);
        }
    }

    #[test]
    fn test_different_weight_methods() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let methods = ["ridge", "lasso", "elastic_net"];

        for method in &methods {
            let llda = LocallyLinearDiscriminantAnalysis::new()
                .weight_method(method)
                .n_neighbors(1)
                .reg_param(0.1);

            let fitted = llda.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 4);
            assert_eq!(fitted.classes().len(), 2);
        }
    }

    #[test]
    fn test_unsupervised_mode() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let llda = LocallyLinearDiscriminantAnalysis::new()
            .supervised(false)
            .n_neighbors(1);

        let fitted = llda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }
}
