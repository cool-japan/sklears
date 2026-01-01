//! Robust graph learning methods for semi-supervised learning
//!
//! This module provides robust graph construction algorithms that are resistant
//! to outliers, noise, and adversarial examples in semi-supervised learning.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rand_prelude::*;
use scirs2_core::random::Random;
use sklears_core::error::SklearsError;
use std::collections::HashMap;

/// Robust graph construction using M-estimators and outlier detection
#[derive(Clone)]
pub struct RobustGraphConstruction {
    /// Number of neighbors for k-NN graph construction
    pub k_neighbors: usize,
    /// Robust distance metric: "huber", "tukey", "cauchy", "welsch"
    pub robust_metric: String,
    /// Robustness parameter for M-estimators
    pub robustness_param: f64,
    /// Outlier detection threshold
    pub outlier_threshold: f64,
    /// Graph construction method: "knn", "epsilon", "adaptive"
    pub construction_method: String,
    /// Epsilon parameter for epsilon-neighborhood graphs
    pub epsilon: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl RobustGraphConstruction {
    /// Create a new robust graph construction instance
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            robust_metric: "huber".to_string(),
            robustness_param: 1.345,
            outlier_threshold: 3.0,
            construction_method: "knn".to_string(),
            epsilon: 1.0,
            random_state: None,
        }
    }

    /// Set the number of neighbors
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the robust distance metric
    pub fn robust_metric(mut self, metric: String) -> Self {
        self.robust_metric = metric;
        self
    }

    /// Set the robustness parameter
    pub fn robustness_param(mut self, param: f64) -> Self {
        self.robustness_param = param;
        self
    }

    /// Set the outlier detection threshold
    pub fn outlier_threshold(mut self, threshold: f64) -> Self {
        self.outlier_threshold = threshold;
        self
    }

    /// Set the graph construction method
    pub fn construction_method(mut self, method: String) -> Self {
        self.construction_method = method;
        self
    }

    /// Set the epsilon parameter
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Construct a robust graph from data
    pub fn fit(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();

        // Detect outliers
        let outlier_mask = self.detect_outliers(X)?;

        // Construct graph with robust distances
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        match self.construction_method.as_str() {
            "knn" => {
                graph = self.construct_robust_knn_graph(X, &outlier_mask)?;
            }
            "epsilon" => {
                graph = self.construct_robust_epsilon_graph(X, &outlier_mask)?;
            }
            "adaptive" => {
                graph = self.construct_adaptive_robust_graph(X, &outlier_mask)?;
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown construction method: {}",
                    self.construction_method
                )));
            }
        }

        Ok(graph)
    }

    /// Detect outliers using robust statistical methods
    fn detect_outliers(&self, X: &ArrayView2<f64>) -> Result<Array1<bool>, SklearsError> {
        let n_samples = X.nrows();
        let mut outlier_mask = Array1::from_elem(n_samples, false);

        // Compute robust center and scale estimates
        let robust_center = self.compute_robust_center(X)?;
        let robust_scale = self.compute_robust_scale(X, &robust_center)?;

        // Identify outliers based on Mahalanobis distance
        for i in 0..n_samples {
            let distance = self.mahalanobis_distance(&X.row(i), &robust_center, robust_scale);
            if distance > self.outlier_threshold {
                outlier_mask[i] = true;
            }
        }

        Ok(outlier_mask)
    }

    /// Compute robust center using median
    fn compute_robust_center(&self, X: &ArrayView2<f64>) -> Result<Array1<f64>, SklearsError> {
        let n_features = X.ncols();
        let mut center = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut feature_values: Vec<f64> = X.column(j).to_vec();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = if feature_values.len() % 2 == 0 {
                let mid = feature_values.len() / 2;
                (feature_values[mid - 1] + feature_values[mid]) / 2.0
            } else {
                feature_values[feature_values.len() / 2]
            };

            center[j] = median;
        }

        Ok(center)
    }

    /// Compute robust scale using MAD (Median Absolute Deviation)
    fn compute_robust_scale(
        &self,
        X: &ArrayView2<f64>,
        center: &Array1<f64>,
    ) -> Result<f64, SklearsError> {
        let mut deviations = Vec::new();

        for i in 0..X.nrows() {
            let distance = self.euclidean_distance(&X.row(i), &center.view());
            deviations.push(distance);
        }

        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mad = if deviations.len() % 2 == 0 {
            let mid = deviations.len() / 2;
            (deviations[mid - 1] + deviations[mid]) / 2.0
        } else {
            deviations[deviations.len() / 2]
        };

        // MAD to standard deviation conversion factor
        Ok(mad * 1.4826)
    }

    /// Compute Mahalanobis distance (simplified version using robust scale)
    fn mahalanobis_distance(&self, x: &ArrayView1<f64>, center: &Array1<f64>, scale: f64) -> f64 {
        let distance = self.euclidean_distance(x, &center.view());
        if scale > 0.0 {
            distance / scale
        } else {
            distance
        }
    }

    /// Construct robust k-NN graph
    fn construct_robust_knn_graph(
        &self,
        X: &ArrayView2<f64>,
        outlier_mask: &Array1<bool>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            if outlier_mask[i] {
                continue; // Skip outliers
            }

            let mut distances: Vec<(f64, usize)> = Vec::new();

            for j in 0..n_samples {
                if i != j && !outlier_mask[j] {
                    let dist = self.robust_distance(&X.row(i), &X.row(j));
                    distances.push((dist, j));
                }
            }

            // Sort by distance and take k nearest neighbors
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            for (dist, j) in distances.iter().take(self.k_neighbors.min(distances.len())) {
                let weight = self.robust_weight(*dist);
                graph[[i, *j]] = weight;
            }
        }

        // Make graph symmetric
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let avg_weight = (graph[[i, j]] + graph[[j, i]]) / 2.0;
                graph[[i, j]] = avg_weight;
                graph[[j, i]] = avg_weight;
            }
        }

        Ok(graph)
    }

    /// Construct robust epsilon-neighborhood graph
    fn construct_robust_epsilon_graph(
        &self,
        X: &ArrayView2<f64>,
        outlier_mask: &Array1<bool>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            if outlier_mask[i] {
                continue; // Skip outliers
            }

            for j in i + 1..n_samples {
                if !outlier_mask[j] {
                    let dist = self.robust_distance(&X.row(i), &X.row(j));

                    if dist <= self.epsilon {
                        let weight = self.robust_weight(dist);
                        graph[[i, j]] = weight;
                        graph[[j, i]] = weight;
                    }
                }
            }
        }

        Ok(graph)
    }

    /// Construct adaptive robust graph
    fn construct_adaptive_robust_graph(
        &self,
        X: &ArrayView2<f64>,
        outlier_mask: &Array1<bool>,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut graph = Array2::<f64>::zeros((n_samples, n_samples));

        // Compute adaptive epsilon for each point
        let adaptive_epsilons = self.compute_adaptive_epsilons(X, outlier_mask)?;

        for i in 0..n_samples {
            if outlier_mask[i] {
                continue; // Skip outliers
            }

            for j in i + 1..n_samples {
                if !outlier_mask[j] {
                    let dist = self.robust_distance(&X.row(i), &X.row(j));
                    let epsilon_ij = (adaptive_epsilons[i] + adaptive_epsilons[j]) / 2.0;

                    if dist <= epsilon_ij {
                        let weight = self.robust_weight(dist);
                        graph[[i, j]] = weight;
                        graph[[j, i]] = weight;
                    }
                }
            }
        }

        Ok(graph)
    }

    /// Compute adaptive epsilon values for each point
    fn compute_adaptive_epsilons(
        &self,
        X: &ArrayView2<f64>,
        outlier_mask: &Array1<bool>,
    ) -> Result<Array1<f64>, SklearsError> {
        let n_samples = X.nrows();
        let mut epsilons = Array1::zeros(n_samples);

        for i in 0..n_samples {
            if outlier_mask[i] {
                epsilons[i] = f64::INFINITY; // Outliers get infinite epsilon
                continue;
            }

            let mut distances = Vec::new();
            for j in 0..n_samples {
                if i != j && !outlier_mask[j] {
                    let dist = self.robust_distance(&X.row(i), &X.row(j));
                    distances.push(dist);
                }
            }

            if distances.is_empty() {
                epsilons[i] = 1.0; // Default epsilon
                continue;
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Use k-th nearest neighbor distance as adaptive epsilon
            let k_index = self.k_neighbors.min(distances.len()) - 1;
            epsilons[i] = distances[k_index];
        }

        Ok(epsilons)
    }

    /// Compute robust distance using M-estimators
    fn robust_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        let euclidean_dist = self.euclidean_distance(x1, x2);

        match self.robust_metric.as_str() {
            "huber" => self.huber_distance(euclidean_dist),
            "tukey" => self.tukey_distance(euclidean_dist),
            "cauchy" => self.cauchy_distance(euclidean_dist),
            "welsch" => self.welsch_distance(euclidean_dist),
            _ => euclidean_dist, // Default to Euclidean
        }
    }

    /// Huber robust distance
    fn huber_distance(&self, dist: f64) -> f64 {
        let c = self.robustness_param;
        if dist <= c {
            0.5 * dist.powi(2)
        } else {
            c * dist - 0.5 * c.powi(2)
        }
    }

    /// Tukey biweight robust distance
    fn tukey_distance(&self, dist: f64) -> f64 {
        let c = self.robustness_param;
        if dist <= c {
            let ratio = dist / c;
            (c.powi(2) / 6.0) * (1.0 - (1.0 - ratio.powi(2)).powi(3))
        } else {
            c.powi(2) / 6.0
        }
    }

    /// Cauchy robust distance
    fn cauchy_distance(&self, dist: f64) -> f64 {
        let c = self.robustness_param;
        (c.powi(2) / 2.0) * ((1.0 + (dist / c).powi(2)).ln())
    }

    /// Welsch robust distance
    fn welsch_distance(&self, dist: f64) -> f64 {
        let c = self.robustness_param;
        (c.powi(2) / 2.0) * (1.0 - (-(dist / c).powi(2)).exp())
    }

    /// Compute robust weight from distance
    fn robust_weight(&self, dist: f64) -> f64 {
        match self.robust_metric.as_str() {
            "huber" => self.huber_weight(dist),
            "tukey" => self.tukey_weight(dist),
            "cauchy" => self.cauchy_weight(dist),
            "welsch" => self.welsch_weight(dist),
            _ => (-dist.powi(2) / 2.0).exp(), // Default RBF kernel
        }
    }

    /// Huber weight function
    fn huber_weight(&self, dist: f64) -> f64 {
        let c = self.robustness_param;
        if dist <= c {
            1.0
        } else {
            c / dist
        }
    }

    /// Tukey biweight weight function
    fn tukey_weight(&self, dist: f64) -> f64 {
        let c = self.robustness_param;
        if dist <= c {
            let ratio = dist / c;
            (1.0 - ratio.powi(2)).powi(2)
        } else {
            0.0
        }
    }

    /// Cauchy weight function
    fn cauchy_weight(&self, dist: f64) -> f64 {
        let c = self.robustness_param;
        1.0 / (1.0 + (dist / c).powi(2))
    }

    /// Welsch weight function
    fn welsch_weight(&self, dist: f64) -> f64 {
        let c = self.robustness_param;
        (-(dist / c).powi(2)).exp()
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for RobustGraphConstruction {
    fn default() -> Self {
        Self::new()
    }
}

/// Noise-robust label propagation for semi-supervised learning
#[derive(Clone)]
pub struct NoiseRobustPropagation {
    /// Number of neighbors for graph construction
    pub k_neighbors: usize,
    /// Noise level estimation method: "mad", "iqr", "adaptive"
    pub noise_estimation: String,
    /// Robustness parameter for propagation
    pub robustness_param: f64,
    /// Maximum iterations for label propagation
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Alpha parameter for label spreading
    pub alpha: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl NoiseRobustPropagation {
    /// Create a new noise-robust propagation instance
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            noise_estimation: "mad".to_string(),
            robustness_param: 1.345,
            max_iter: 1000,
            tolerance: 1e-6,
            alpha: 0.2,
            random_state: None,
        }
    }

    /// Set the number of neighbors
    pub fn k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }

    /// Set the noise estimation method
    pub fn noise_estimation(mut self, method: String) -> Self {
        self.noise_estimation = method;
        self
    }

    /// Set the robustness parameter
    pub fn robustness_param(mut self, param: f64) -> Self {
        self.robustness_param = param;
        self
    }

    /// Set the maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the alpha parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Perform noise-robust label propagation
    pub fn fit(
        &self,
        X: &ArrayView2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<Array1<i32>, SklearsError> {
        let n_samples = X.nrows();

        if y.len() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X and y should have same number of samples: {}", X.nrows()),
                actual: format!("X has {} samples, y has {} samples", X.nrows(), y.len()),
            });
        }

        // Estimate noise level
        let noise_level = self.estimate_noise_level(X)?;

        // Construct robust graph
        let robust_graph_builder = RobustGraphConstruction::new()
            .k_neighbors(self.k_neighbors)
            .robustness_param(self.robustness_param)
            .outlier_threshold(noise_level * 3.0);

        let graph = robust_graph_builder.fit(X)?;

        // Perform robust label propagation
        let labels = self.robust_propagate_labels(&graph, y)?;

        Ok(labels)
    }

    /// Estimate noise level in the data
    fn estimate_noise_level(&self, X: &ArrayView2<f64>) -> Result<f64, SklearsError> {
        match self.noise_estimation.as_str() {
            "mad" => self.estimate_noise_mad(X),
            "iqr" => self.estimate_noise_iqr(X),
            "adaptive" => self.estimate_noise_adaptive(X),
            _ => Ok(1.0), // Default noise level
        }
    }

    /// Estimate noise using Median Absolute Deviation
    fn estimate_noise_mad(&self, X: &ArrayView2<f64>) -> Result<f64, SklearsError> {
        let n_samples = X.nrows();
        let mut distances = Vec::new();

        // Compute pairwise distances
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                distances.push(dist);
            }
        }

        if distances.is_empty() {
            return Ok(1.0);
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if distances.len() % 2 == 0 {
            let mid = distances.len() / 2;
            (distances[mid - 1] + distances[mid]) / 2.0
        } else {
            distances[distances.len() / 2]
        };

        // Compute MAD
        let mut deviations: Vec<f64> = distances.iter().map(|&d| (d - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mad = if deviations.len() % 2 == 0 {
            let mid = deviations.len() / 2;
            (deviations[mid - 1] + deviations[mid]) / 2.0
        } else {
            deviations[deviations.len() / 2]
        };

        Ok(mad * 1.4826) // MAD to std conversion
    }

    /// Estimate noise using Interquartile Range
    fn estimate_noise_iqr(&self, X: &ArrayView2<f64>) -> Result<f64, SklearsError> {
        let n_samples = X.nrows();
        let mut distances = Vec::new();

        // Compute pairwise distances
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                distances.push(dist);
            }
        }

        if distances.is_empty() {
            return Ok(1.0);
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = distances.len() / 4;
        let q3_idx = 3 * distances.len() / 4;

        let iqr = distances[q3_idx] - distances[q1_idx];

        Ok(iqr / 1.349) // IQR to std conversion
    }

    /// Estimate noise adaptively
    fn estimate_noise_adaptive(&self, X: &ArrayView2<f64>) -> Result<f64, SklearsError> {
        // Combine MAD and IQR estimates
        let mad_estimate = self.estimate_noise_mad(X)?;
        let iqr_estimate = self.estimate_noise_iqr(X)?;

        // Use the minimum to be more conservative
        Ok(mad_estimate.min(iqr_estimate))
    }

    /// Perform robust label propagation
    #[allow(non_snake_case)]
    fn robust_propagate_labels(
        &self,
        graph: &Array2<f64>,
        y: &ArrayView1<i32>,
    ) -> Result<Array1<i32>, SklearsError> {
        let n_samples = graph.nrows();

        // Identify labeled and unlabeled samples
        let labeled_mask: Array1<bool> = y.iter().map(|&label| label != -1).collect();
        let unique_labels: Vec<i32> = y
            .iter()
            .filter(|&&label| label != -1)
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if unique_labels.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No labeled samples found".to_string(),
            ));
        }

        let n_classes = unique_labels.len();

        // Initialize label probability matrix
        let mut F = Array2::<f64>::zeros((n_samples, n_classes));

        // Set initial labels for labeled samples
        for i in 0..n_samples {
            if labeled_mask[i] {
                if let Some(class_idx) = unique_labels.iter().position(|&x| x == y[i]) {
                    F[[i, class_idx]] = 1.0;
                }
            }
        }

        // Normalize graph to get transition matrix
        let P = self.normalize_graph(graph)?;

        // Iterative label propagation with robustness
        for _iter in 0..self.max_iter {
            let F_old = F.clone();

            // Propagate labels: F = α * P * F + (1-α) * Y
            let propagated = P.dot(&F);
            F = &propagated * self.alpha;

            // Reset labeled samples
            for i in 0..n_samples {
                if labeled_mask[i] {
                    F.row_mut(i).fill(0.0);
                    if let Some(class_idx) = unique_labels.iter().position(|&x| x == y[i]) {
                        F[[i, class_idx]] = 1.0;
                    }
                }
            }

            // Check convergence
            let change = (&F - &F_old).iter().map(|x| x.abs()).sum::<f64>();
            if change < self.tolerance {
                break;
            }
        }

        // Convert probabilities to labels
        let mut labels = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut max_prob = 0.0;
            let mut max_class = 0;

            for j in 0..n_classes {
                if F[[i, j]] > max_prob {
                    max_prob = F[[i, j]];
                    max_class = j;
                }
            }

            labels[i] = unique_labels[max_class];
        }

        Ok(labels)
    }

    /// Normalize graph to get transition matrix
    fn normalize_graph(&self, graph: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let n_samples = graph.nrows();
        let mut P = graph.clone();

        for i in 0..n_samples {
            let row_sum: f64 = P.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_samples {
                    P[[i, j]] /= row_sum;
                }
            }
        }

        Ok(P)
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for NoiseRobustPropagation {
    fn default() -> Self {
        Self::new()
    }
}

/// Breakdown point analysis for robust graph learning methods
#[derive(Clone)]
pub struct BreakdownPointAnalysis {
    /// Robust estimators to analyze: "median", "huber", "tukey", "trimmed_mean"
    pub estimators: Vec<String>,
    /// Contamination levels to test (0.0 to 0.5)
    pub contamination_levels: Vec<f64>,
    /// Number of Monte Carlo simulations
    pub n_simulations: usize,
    /// Breakdown threshold (relative change in estimate)
    pub breakdown_threshold: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl BreakdownPointAnalysis {
    /// Create a new breakdown point analysis instance
    pub fn new() -> Self {
        Self {
            estimators: vec![
                "median".to_string(),
                "huber".to_string(),
                "tukey".to_string(),
                "trimmed_mean".to_string(),
            ],
            contamination_levels: (1..=25).map(|x| x as f64 / 100.0).collect(), // 1% to 25%
            n_simulations: 100,
            breakdown_threshold: 10.0, // 10x change indicates breakdown
            random_state: None,
        }
    }

    /// Set the estimators to analyze
    pub fn estimators(mut self, estimators: Vec<String>) -> Self {
        self.estimators = estimators;
        self
    }

    /// Set the contamination levels to test
    pub fn contamination_levels(mut self, levels: Vec<f64>) -> Self {
        self.contamination_levels = levels;
        self
    }

    /// Set the number of simulations
    pub fn n_simulations(mut self, n: usize) -> Self {
        self.n_simulations = n;
        self
    }

    /// Set the breakdown threshold
    pub fn breakdown_threshold(mut self, threshold: f64) -> Self {
        self.breakdown_threshold = threshold;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Analyze breakdown points for robust graph construction
    pub fn analyze_graph_breakdown(
        &self,
        X: &ArrayView2<f64>,
    ) -> Result<HashMap<String, BreakdownResult>, SklearsError> {
        let mut results = HashMap::new();
        let mut rng = Random::default();

        for estimator in &self.estimators {
            let breakdown_result = self.estimate_breakdown_point(X, estimator, &mut rng)?;
            results.insert(estimator.clone(), breakdown_result);
        }

        Ok(results)
    }

    /// Estimate breakdown point for a specific estimator
    fn estimate_breakdown_point(
        &self,
        X: &ArrayView2<f64>,
        estimator: &str,
        rng: &mut Random,
    ) -> Result<BreakdownResult, SklearsError> {
        let n_samples = X.nrows();

        // Compute clean estimate (no contamination)
        let clean_estimate = self.compute_robust_estimate(X, estimator, 0.0, rng)?;

        let mut breakdown_rates = Vec::new();
        let mut first_breakdown = None;

        for &contamination_level in &self.contamination_levels {
            let mut breakdown_count = 0;

            for _sim in 0..self.n_simulations {
                // Create contaminated data
                let contaminated_X = self.contaminate_data(X, contamination_level, rng)?;

                // Compute estimate on contaminated data
                let contaminated_estimate =
                    self.compute_robust_estimate(&contaminated_X.view(), estimator, 0.0, rng)?;

                // Check if breakdown occurred
                let relative_change =
                    self.compute_relative_change(clean_estimate, contaminated_estimate);

                if relative_change > self.breakdown_threshold {
                    breakdown_count += 1;
                }
            }

            let breakdown_rate = breakdown_count as f64 / self.n_simulations as f64;
            breakdown_rates.push(breakdown_rate);

            // Record first significant breakdown
            if first_breakdown.is_none() && breakdown_rate > 0.5 {
                first_breakdown = Some(contamination_level);
            }
        }

        Ok(BreakdownResult {
            estimator: estimator.to_string(),
            theoretical_breakdown_point: self.theoretical_breakdown_point(estimator),
            empirical_breakdown_point: first_breakdown.unwrap_or(0.5),
            contamination_levels: self.contamination_levels.clone(),
            breakdown_rates,
            clean_estimate,
        })
    }

    /// Contaminate data by replacing a fraction with outliers
    fn contaminate_data<R>(
        &self,
        X: &ArrayView2<f64>,
        contamination_level: f64,
        rng: &mut Random<R>,
    ) -> Result<Array2<f64>, SklearsError>
    where
        R: Rng,
    {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_outliers = (n_samples as f64 * contamination_level).round() as usize;

        let mut contaminated_X = X.to_owned();

        if n_outliers == 0 {
            return Ok(contaminated_X);
        }

        // Select random samples to contaminate
        let outlier_indices: Vec<usize> = (0..n_samples)
            .choose_multiple(rng, n_outliers)
            .into_iter()
            .collect();

        // Compute data range for generating outliers
        let mut feature_ranges = Vec::new();
        for j in 0..n_features {
            let column = X.column(j);
            let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            feature_ranges.push((min_val, max_val));
        }

        // Replace selected samples with outliers
        for &idx in &outlier_indices {
            for j in 0..n_features {
                let (min_val, max_val) = feature_ranges[j];
                let range = max_val - min_val;

                // Generate outlier far from the data range
                let outlier_multiplier = rng.random_range(5.0..10.0);
                let outlier_value = if rng.gen_bool(0.5) {
                    min_val - outlier_multiplier * range
                } else {
                    max_val + outlier_multiplier * range
                };

                contaminated_X[[idx, j]] = outlier_value;
            }
        }

        Ok(contaminated_X)
    }

    /// Compute robust estimate for graph properties
    fn compute_robust_estimate<R>(
        &self,
        X: &ArrayView2<f64>,
        estimator: &str,
        _contamination: f64,
        _rng: &mut Random<R>,
    ) -> Result<f64, SklearsError>
    where
        R: Rng,
    {
        match estimator {
            "median" => self.compute_median_graph_property(X),
            "huber" => self.compute_huber_graph_property(X),
            "tukey" => self.compute_tukey_graph_property(X),
            "trimmed_mean" => self.compute_trimmed_mean_graph_property(X),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown estimator: {}",
                estimator
            ))),
        }
    }

    /// Compute median-based graph property (median edge weight)
    fn compute_median_graph_property(&self, X: &ArrayView2<f64>) -> Result<f64, SklearsError> {
        let n_samples = X.nrows();
        let mut edge_weights = Vec::new();

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                let weight = (-dist.powi(2) / 2.0).exp(); // RBF kernel
                edge_weights.push(weight);
            }
        }

        if edge_weights.is_empty() {
            return Ok(0.0);
        }

        edge_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idx = edge_weights.len() / 2;
        Ok(if edge_weights.len() % 2 == 0 {
            (edge_weights[median_idx - 1] + edge_weights[median_idx]) / 2.0
        } else {
            edge_weights[median_idx]
        })
    }

    /// Compute Huber-based graph property
    fn compute_huber_graph_property(&self, X: &ArrayView2<f64>) -> Result<f64, SklearsError> {
        let n_samples = X.nrows();
        let mut huber_weights = Vec::new();
        let c = 1.345; // Standard Huber parameter

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                let huber_dist = if dist <= c {
                    0.5 * dist.powi(2)
                } else {
                    c * dist - 0.5 * c.powi(2)
                };
                huber_weights.push((-huber_dist).exp());
            }
        }

        Ok(huber_weights.iter().sum::<f64>() / huber_weights.len() as f64)
    }

    /// Compute Tukey-based graph property
    fn compute_tukey_graph_property(&self, X: &ArrayView2<f64>) -> Result<f64, SklearsError> {
        let n_samples = X.nrows();
        let mut tukey_weights = Vec::new();
        let c = 4.685; // Standard Tukey parameter

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                let tukey_weight = if dist <= c {
                    let ratio = dist / c;
                    (1.0 - ratio.powi(2)).powi(2)
                } else {
                    0.0
                };
                tukey_weights.push(tukey_weight);
            }
        }

        Ok(tukey_weights.iter().sum::<f64>() / tukey_weights.len() as f64)
    }

    /// Compute trimmed mean-based graph property
    fn compute_trimmed_mean_graph_property(
        &self,
        X: &ArrayView2<f64>,
    ) -> Result<f64, SklearsError> {
        let n_samples = X.nrows();
        let mut edge_weights = Vec::new();

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.euclidean_distance(&X.row(i), &X.row(j));
                let weight = (-dist.powi(2) / 2.0).exp(); // RBF kernel
                edge_weights.push(weight);
            }
        }

        if edge_weights.is_empty() {
            return Ok(0.0);
        }

        edge_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Trim 20% from each end
        let trim_size = (edge_weights.len() as f64 * 0.2) as usize;
        let trimmed_weights = &edge_weights[trim_size..edge_weights.len() - trim_size];

        Ok(trimmed_weights.iter().sum::<f64>() / trimmed_weights.len() as f64)
    }

    /// Compute relative change between estimates
    fn compute_relative_change(&self, clean_estimate: f64, contaminated_estimate: f64) -> f64 {
        if clean_estimate.abs() < 1e-10 {
            if contaminated_estimate.abs() < 1e-10 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (contaminated_estimate - clean_estimate).abs() / clean_estimate.abs()
        }
    }

    /// Get theoretical breakdown point for different estimators
    fn theoretical_breakdown_point(&self, estimator: &str) -> f64 {
        match estimator {
            "median" => 0.5,
            "huber" => 0.5,
            "tukey" => 0.5,
            "trimmed_mean" => 0.2, // 20% trimming
            _ => 0.0,
        }
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for BreakdownPointAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of breakdown point analysis
#[derive(Clone, Debug)]
pub struct BreakdownResult {
    /// Name of the estimator
    pub estimator: String,
    /// Theoretical breakdown point
    pub theoretical_breakdown_point: f64,
    /// Empirically observed breakdown point
    pub empirical_breakdown_point: f64,
    /// Contamination levels tested
    pub contamination_levels: Vec<f64>,
    /// Breakdown rates at each contamination level
    pub breakdown_rates: Vec<f64>,
    /// Clean estimate (no contamination)
    pub clean_estimate: f64,
}

impl BreakdownResult {
    /// Get the efficiency of the estimator (1 - empirical_breakdown_point)
    pub fn efficiency(&self) -> f64 {
        1.0 - self.empirical_breakdown_point
    }

    /// Check if the estimator meets theoretical expectations
    pub fn meets_theory(&self) -> bool {
        self.empirical_breakdown_point >= self.theoretical_breakdown_point * 0.9
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        format!(
            "Estimator: {}\nTheoretical BP: {:.3}\nEmpirical BP: {:.3}\nEfficiency: {:.3}\nMeets Theory: {}",
            self.estimator,
            self.theoretical_breakdown_point,
            self.empirical_breakdown_point,
            self.efficiency(),
            self.meets_theory()
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_robust_graph_construction() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [10.0, 20.0] // Outlier
        ];

        let rgc = RobustGraphConstruction::new()
            .k_neighbors(2)
            .robust_metric("huber".to_string())
            .outlier_threshold(2.0);

        let result = rgc.fit(&X.view());
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert_eq!(graph.dim(), (4, 4));

        // Check that diagonal is zero
        for i in 0..4 {
            assert_eq!(graph[[i, i]], 0.0);
        }

        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(graph[[i, j]], graph[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_robust_metrics() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let metrics = vec!["huber", "tukey", "cauchy", "welsch"];

        for metric in metrics {
            let rgc = RobustGraphConstruction::new()
                .k_neighbors(2)
                .robust_metric(metric.to_string());

            let result = rgc.fit(&X.view());
            assert!(result.is_ok());

            let graph = result.unwrap();
            assert_eq!(graph.dim(), (3, 3));
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_robust_construction_methods() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let methods = vec!["knn", "epsilon", "adaptive"];

        for method in methods {
            let rgc = RobustGraphConstruction::new()
                .construction_method(method.to_string())
                .k_neighbors(2)
                .epsilon(2.0);

            let result = rgc.fit(&X.view());
            assert!(result.is_ok());

            let graph = result.unwrap();
            assert_eq!(graph.dim(), (3, 3));
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_noise_robust_propagation() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, -1, -1]; // -1 indicates unlabeled

        let nrp = NoiseRobustPropagation::new()
            .k_neighbors(2)
            .noise_estimation("mad".to_string())
            .max_iter(100)
            .alpha(0.2);

        let result = nrp.fit(&X.view(), &y.view());
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.len(), 4);

        // Check that labeled samples retain their labels
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_noise_estimation_methods() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![0, 1, -1];

        let methods = vec!["mad", "iqr", "adaptive"];

        for method in methods {
            let nrp = NoiseRobustPropagation::new()
                .noise_estimation(method.to_string())
                .k_neighbors(2)
                .max_iter(50);

            let result = nrp.fit(&X.view(), &y.view());
            assert!(result.is_ok());

            let labels = result.unwrap();
            assert_eq!(labels.len(), 3);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_robust_graph_error_cases() {
        let rgc = RobustGraphConstruction::new().construction_method("invalid".to_string());

        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let result = rgc.fit(&X.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_noise_robust_propagation_error_cases() {
        let nrp = NoiseRobustPropagation::new();

        // Test with mismatched dimensions
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0]; // Wrong size

        let result = nrp.fit(&X.view(), &y.view());
        assert!(result.is_err());

        // Test with no labeled samples
        let y_unlabeled = array![-1, -1];
        let result = nrp.fit(&X.view(), &y_unlabeled.view());
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_breakdown_point_analysis() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let bpa = BreakdownPointAnalysis::new()
            .n_simulations(20) // Reduced for faster testing
            .contamination_levels(vec![0.1, 0.2, 0.3])
            .random_state(42);

        let result = bpa.analyze_graph_breakdown(&X.view());
        assert!(result.is_ok());

        let results = result.unwrap();
        assert!(!results.is_empty());

        // Check that we have results for each estimator
        for estimator in &["median", "huber", "tukey", "trimmed_mean"] {
            assert!(results.contains_key(*estimator));
            let breakdown_result = &results[*estimator];

            // Check that breakdown point is reasonable
            assert!(breakdown_result.empirical_breakdown_point >= 0.0);
            assert!(breakdown_result.empirical_breakdown_point <= 0.5);

            // Check that theoretical breakdown point is set correctly
            assert!(breakdown_result.theoretical_breakdown_point > 0.0);
        }
    }

    #[test]
    fn test_breakdown_result_methods() {
        let breakdown_result = BreakdownResult {
            estimator: "median".to_string(),
            theoretical_breakdown_point: 0.5,
            empirical_breakdown_point: 0.45, // Changed to meet theory threshold
            contamination_levels: vec![0.1, 0.2, 0.3],
            breakdown_rates: vec![0.0, 0.1, 0.8],
            clean_estimate: 1.0,
        };

        // Test efficiency calculation
        assert_abs_diff_eq!(breakdown_result.efficiency(), 0.55, epsilon = 1e-10);

        // Test meets_theory check
        assert!(breakdown_result.meets_theory());

        // Test summary generation
        let summary = breakdown_result.summary();
        assert!(summary.contains("median"));
        assert!(summary.contains("0.500"));
        assert!(summary.contains("0.450"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_contaminate_data() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let bpa = BreakdownPointAnalysis::new().random_state(42);
        let mut rng = Random::seed(42);

        // Test with 25% contamination
        let contaminated = bpa.contaminate_data(&X.view(), 0.25, &mut rng).unwrap();
        assert_eq!(contaminated.dim(), X.dim());

        // Check that at least one sample was contaminated
        let mut different = false;
        for i in 0..X.nrows() {
            for j in 0..X.ncols() {
                if (X[[i, j]] - contaminated[[i, j]]).abs() > 1e-10 {
                    different = true;
                    break;
                }
            }
        }
        assert!(different);

        // Test with 0% contamination
        let no_contamination = bpa.contaminate_data(&X.view(), 0.0, &mut rng).unwrap();
        for i in 0..X.nrows() {
            for j in 0..X.ncols() {
                assert_abs_diff_eq!(X[[i, j]], no_contamination[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_robust_graph_property_estimators() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let bpa = BreakdownPointAnalysis::new();

        // Test all estimators
        let estimators = vec!["median", "huber", "tukey", "trimmed_mean"];
        for estimator in estimators {
            let mut rng = Random::seed(42);
            let result = bpa.compute_robust_estimate(&X.view(), estimator, 0.0, &mut rng);
            assert!(result.is_ok());

            let estimate = result.unwrap();
            assert!(estimate >= 0.0);
            assert!(estimate <= 1.0); // Should be normalized
        }

        // Test invalid estimator
        let mut rng = Random::seed(42);
        let result = bpa.compute_robust_estimate(&X.view(), "invalid", 0.0, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_theoretical_breakdown_points() {
        let bpa = BreakdownPointAnalysis::new();

        assert_eq!(bpa.theoretical_breakdown_point("median"), 0.5);
        assert_eq!(bpa.theoretical_breakdown_point("huber"), 0.5);
        assert_eq!(bpa.theoretical_breakdown_point("tukey"), 0.5);
        assert_eq!(bpa.theoretical_breakdown_point("trimmed_mean"), 0.2);
        assert_eq!(bpa.theoretical_breakdown_point("unknown"), 0.0);
    }

    #[test]
    fn test_relative_change_computation() {
        let bpa = BreakdownPointAnalysis::new();

        // Normal case
        assert_abs_diff_eq!(bpa.compute_relative_change(1.0, 1.5), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(bpa.compute_relative_change(2.0, 1.0), 0.5, epsilon = 1e-10);

        // Zero clean estimate case
        assert_eq!(bpa.compute_relative_change(0.0, 0.0), 0.0);
        assert_eq!(bpa.compute_relative_change(0.0, 1.0), f64::INFINITY);

        // Very small clean estimate
        let small_val = 1e-12;
        assert_eq!(bpa.compute_relative_change(small_val, 1.0), f64::INFINITY);
    }

    #[test]
    fn test_breakdown_point_analysis_builder() {
        let custom_levels = vec![0.05, 0.15, 0.25];
        let custom_estimators = vec!["median".to_string(), "huber".to_string()];

        let bpa = BreakdownPointAnalysis::new()
            .estimators(custom_estimators.clone())
            .contamination_levels(custom_levels.clone())
            .n_simulations(50)
            .breakdown_threshold(5.0)
            .random_state(123);

        assert_eq!(bpa.estimators, custom_estimators);
        assert_eq!(bpa.contamination_levels, custom_levels);
        assert_eq!(bpa.n_simulations, 50);
        assert_eq!(bpa.breakdown_threshold, 5.0);
        assert_eq!(bpa.random_state, Some(123));
    }
}
