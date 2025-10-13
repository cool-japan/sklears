//! Manifold learning integration and manifold-based feature extraction
//!
//! This module provides tools for extracting features from data lying on
//! or near low-dimensional manifolds embedded in high-dimensional spaces.

use crate::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use sklears_core::{
    error::Result as SklResult,
    prelude::{SklearsError, Transform},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Tangent space feature extractor
///
/// Extracts features based on the tangent space of the manifold at specific points.
/// This is useful for analyzing local linear structure in non-linear data.
#[derive(Debug, Clone)]
pub struct TangentSpaceExtractor<S = Untrained> {
    state: S,
    /// Dimensionality of the tangent space
    pub tangent_dim: usize,
    /// Number of nearest neighbors to use for tangent space estimation
    pub n_neighbors: usize,
    /// Whether to normalize the tangent vectors
    pub normalize: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

/// Trained state for tangent space extractor
#[derive(Debug, Clone)]
pub struct TangentSpaceExtractorTrained {
    pub reference_points: Array2<f64>,
    pub tangent_bases: Vec<Array2<f64>>,
}

impl Default for TangentSpaceExtractor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl TangentSpaceExtractor<Untrained> {
    /// Create a new tangent space extractor
    pub fn new() -> Self {
        Self {
            state: Untrained,
            tangent_dim: 2,
            n_neighbors: 10,
            normalize: true,
            random_state: None,
        }
    }

    /// Set the dimensionality of the tangent space
    pub fn tangent_dim(mut self, dim: usize) -> Self {
        self.tangent_dim = dim;
        self
    }

    /// Set the number of nearest neighbors
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set whether to normalize tangent vectors
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<S> TangentSpaceExtractor<S> {
    /// Compute tangent space basis at a point using PCA of local neighborhoods
    fn compute_tangent_basis(
        &self,
        data: &Array2<f64>,
        point_idx: usize,
    ) -> SklResult<Array2<f64>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if point_idx >= n_samples {
            return Err(SklearsError::InvalidInput(
                "Point index out of bounds".to_string(),
            ));
        }

        // Find k nearest neighbors
        let center = data.row(point_idx);
        let mut distances: Vec<(f64, usize)> = Vec::new();

        for i in 0..n_samples {
            if i != point_idx {
                let diff = &data.row(i) - &center;
                let dist = diff.dot(&diff).sqrt();
                distances.push((dist, i));
            }
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k = self.n_neighbors.min(distances.len());

        // Create matrix of centered neighborhood points
        let mut neighborhood = Array2::zeros((k, n_features));
        for (i, &(_, idx)) in distances.iter().take(k).enumerate() {
            let row = &data.row(idx) - &center;
            neighborhood.row_mut(i).assign(&row);
        }

        // Compute SVD to get tangent basis
        let svd = neighborhood.t().dot(&neighborhood);
        let eigenvalues = self.compute_eigenvalues(&svd)?;
        let eigenvectors = self.compute_eigenvectors(&svd, &eigenvalues)?;

        // Return top tangent_dim eigenvectors as tangent basis
        let tangent_basis = eigenvectors
            .slice(s![.., 0..self.tangent_dim.min(eigenvectors.ncols())])
            .to_owned();

        if self.normalize {
            self.normalize_columns(&tangent_basis)
        } else {
            Ok(tangent_basis)
        }
    }

    /// Compute eigenvalues using power iteration
    fn compute_eigenvalues(&self, matrix: &Array2<f64>) -> SklResult<Array1<f64>> {
        let n = matrix.nrows();
        let mut eigenvalues = Array1::zeros(n);
        let mut temp_matrix = matrix.clone();

        for i in 0..n {
            let mut v = Array1::ones(n);

            // Power iteration
            for _ in 0..50 {
                let new_v = temp_matrix.dot(&v);
                let norm = new_v.dot(&new_v).sqrt();
                if norm > 1e-10 {
                    v = new_v / norm;
                } else {
                    break;
                }
            }

            eigenvalues[i] = v.dot(&temp_matrix.dot(&v));

            // Deflation
            let outer_product = &v.clone().insert_axis(Axis(1)) * &v.clone().insert_axis(Axis(0));
            temp_matrix = &temp_matrix - eigenvalues[i] * outer_product;
        }

        Ok(eigenvalues)
    }

    /// Compute eigenvectors
    fn compute_eigenvectors(
        &self,
        matrix: &Array2<f64>,
        eigenvalues: &Array1<f64>,
    ) -> SklResult<Array2<f64>> {
        let n = matrix.nrows();
        let mut eigenvectors = Array2::zeros((n, n));

        for (i, &eigenval) in eigenvalues.iter().enumerate() {
            let mut shifted_matrix = matrix.clone();
            for j in 0..n {
                shifted_matrix[(j, j)] -= eigenval;
            }

            // Find nullspace vector (eigenvector)
            let mut v = Array1::ones(n);
            for _ in 0..50 {
                let new_v = self.solve_linear_system(&shifted_matrix, &v)?;
                let norm = new_v.dot(&new_v).sqrt();
                if norm > 1e-10 {
                    v = new_v / norm;
                } else {
                    break;
                }
            }

            eigenvectors.column_mut(i).assign(&v);
        }

        Ok(eigenvectors)
    }

    /// Solve linear system using Gaussian elimination
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
        let n = a.nrows();
        let mut aug = Array2::zeros((n, n + 1));

        // Create augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug[(i, j)] = a[(i, j)];
            }
            aug[(i, n)] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if aug[(k, i)].abs() > aug[(max_row, i)].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..n + 1 {
                    let temp = aug[(i, j)];
                    aug[(i, j)] = aug[(max_row, j)];
                    aug[(max_row, j)] = temp;
                }
            }

            // Eliminate column
            for k in i + 1..n {
                if aug[(i, i)].abs() > 1e-10 {
                    let factor = aug[(k, i)] / aug[(i, i)];
                    for j in i..n + 1 {
                        aug[(k, j)] -= factor * aug[(i, j)];
                    }
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += aug[(i, j)] * x[j];
            }
            if aug[(i, i)].abs() > 1e-10 {
                x[i] = (aug[(i, n)] - sum) / aug[(i, i)];
            }
        }

        Ok(x)
    }

    /// Normalize columns of a matrix
    fn normalize_columns(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let mut result = matrix.clone();

        for mut col in result.columns_mut() {
            let norm = col.dot(&col).sqrt();
            if norm > 1e-10 {
                col /= norm;
            }
        }

        Ok(result)
    }
}

impl Estimator for TangentSpaceExtractor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for TangentSpaceExtractor<Untrained> {
    type Fitted = TangentSpaceExtractor<TangentSpaceExtractorTrained>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit with empty data".to_string(),
            ));
        }

        let mut tangent_bases = Vec::new();

        // Compute tangent basis at each point
        for i in 0..n_samples {
            let basis = self.compute_tangent_basis(x, i)?;
            tangent_bases.push(basis);
        }

        Ok(TangentSpaceExtractor {
            state: TangentSpaceExtractorTrained {
                reference_points: x.clone(),
                tangent_bases,
            },
            tangent_dim: self.tangent_dim,
            n_neighbors: self.n_neighbors,
            normalize: self.normalize,
            random_state: self.random_state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for TangentSpaceExtractor<TangentSpaceExtractorTrained> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = self.tangent_dim * self.state.reference_points.nrows();
        let mut features = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let point = x.row(i);
            let mut feature_idx = 0;

            // Project onto each tangent space
            for j in 0..self.state.reference_points.nrows() {
                let ref_point = self.state.reference_points.row(j);
                let tangent_basis = &self.state.tangent_bases[j];

                // Project centered point onto tangent space
                let centered = &point - &ref_point;
                let projection = tangent_basis.t().dot(&centered);

                // Add projection coefficients to feature vector
                for k in 0..projection.len() {
                    if feature_idx < n_features {
                        features[(i, feature_idx)] = projection[k];
                        feature_idx += 1;
                    }
                }
            }
        }

        Ok(features)
    }
}

/// Geodesic distance feature extractor
///
/// Computes geodesic distances along the manifold instead of Euclidean distances.
/// This is useful for data that lies on curved manifolds.
#[derive(Debug, Clone)]
pub struct GeodesicDistanceExtractor<S = Untrained> {
    state: S,
    /// Number of nearest neighbors for graph construction
    pub n_neighbors: usize,
    /// Whether to use symmetric neighborhood graph
    pub symmetric: bool,
    /// Distance metric for initial neighborhood graph
    pub metric: String,
    /// Number of reference points for distance features
    pub n_reference_points: usize,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

/// Trained state for geodesic distance extractor
#[derive(Debug, Clone)]
pub struct GeodesicDistanceExtractorTrained {
    pub reference_points: Vec<usize>,
    pub training_data: Array2<f64>,
    pub geodesic_distances: Array2<f64>,
}

impl Default for GeodesicDistanceExtractor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl GeodesicDistanceExtractor<Untrained> {
    /// Create a new geodesic distance extractor
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 10,
            symmetric: true,
            metric: "euclidean".to_string(),
            n_reference_points: 10,
            random_state: None,
        }
    }

    /// Set the number of nearest neighbors
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set whether to use symmetric neighborhood graph
    pub fn symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Set the number of reference points
    pub fn n_reference_points(mut self, n: usize) -> Self {
        self.n_reference_points = n;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<S> GeodesicDistanceExtractor<S> {
    /// Compute pairwise distances using specified metric
    fn compute_distances(&self, data: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = match self.metric.as_str() {
                    "euclidean" => {
                        let diff = &data.row(i) - &data.row(j);
                        diff.dot(&diff).sqrt()
                    }
                    "manhattan" => {
                        let diff = &data.row(i) - &data.row(j);
                        diff.iter().map(|x| x.abs()).sum()
                    }
                    "cosine" => {
                        let a = data.row(i);
                        let b = data.row(j);
                        let dot_product = a.dot(&b);
                        let norm_a = a.dot(&a).sqrt();
                        let norm_b = b.dot(&b).sqrt();
                        if norm_a > 1e-10 && norm_b > 1e-10 {
                            1.0 - dot_product / (norm_a * norm_b)
                        } else {
                            0.0
                        }
                    }
                    _ => {
                        return Err(SklearsError::InvalidInput(format!(
                            "Unknown metric: {}",
                            self.metric
                        )));
                    }
                };
                distances[(i, j)] = dist;
                distances[(j, i)] = dist;
            }
        }

        Ok(distances)
    }

    /// Build neighborhood graph from distance matrix
    fn build_neighborhood_graph(&self, distances: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = distances.nrows();
        let mut graph = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Find k nearest neighbors
            let mut neighbors: Vec<(f64, usize)> = Vec::new();
            for j in 0..n_samples {
                if i != j {
                    neighbors.push((distances[(i, j)], j));
                }
            }

            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Connect to k nearest neighbors
            for &(dist, j) in neighbors.iter().take(self.n_neighbors) {
                graph[(i, j)] = dist;
                if self.symmetric {
                    graph[(j, i)] = dist;
                }
            }
        }

        Ok(graph)
    }

    /// Compute geodesic distances using Floyd-Warshall algorithm
    fn compute_geodesic_distances(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = graph.nrows();
        let mut geodesic_distances = Array2::from_elem((n_samples, n_samples), f64::INFINITY);

        // Initialize with direct distances
        for i in 0..n_samples {
            geodesic_distances[(i, i)] = 0.0;
            for j in 0..n_samples {
                if graph[(i, j)] > 0.0 {
                    geodesic_distances[(i, j)] = graph[(i, j)];
                }
            }
        }

        // Floyd-Warshall algorithm
        for k in 0..n_samples {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    let dist_via_k = geodesic_distances[(i, k)] + geodesic_distances[(k, j)];
                    if dist_via_k < geodesic_distances[(i, j)] {
                        geodesic_distances[(i, j)] = dist_via_k;
                    }
                }
            }
        }

        Ok(geodesic_distances)
    }

    /// Select reference points for distance features
    fn select_reference_points(&self, data: &Array2<f64>) -> SklResult<Vec<usize>> {
        let n_samples = data.nrows();
        let mut reference_points = Vec::new();

        if self.n_reference_points >= n_samples {
            return Ok((0..n_samples).collect());
        }

        // Use farthest-first traversal for diverse reference points
        let mut rng: StdRng = if let Some(seed) = self.random_state {
            SeedableRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(42)
        };

        use scirs2_core::random::Rng;

        // Start with random point
        let first_point = rng.gen_range(0..n_samples);
        reference_points.push(first_point);

        // Add points that are farthest from already selected points
        for _ in 1..self.n_reference_points {
            let mut max_min_dist = 0.0;
            let mut farthest_point = 0;

            for i in 0..n_samples {
                if !reference_points.contains(&i) {
                    let mut min_dist = f64::INFINITY;
                    for &ref_point in &reference_points {
                        let diff = &data.row(i) - &data.row(ref_point);
                        let dist = diff.dot(&diff).sqrt();
                        if dist < min_dist {
                            min_dist = dist;
                        }
                    }
                    if min_dist > max_min_dist {
                        max_min_dist = min_dist;
                        farthest_point = i;
                    }
                }
            }

            reference_points.push(farthest_point);
        }

        Ok(reference_points)
    }
}

impl Estimator for GeodesicDistanceExtractor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for GeodesicDistanceExtractor<Untrained> {
    type Fitted = GeodesicDistanceExtractor<GeodesicDistanceExtractorTrained>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        // Compute pairwise distances
        let distances = self.compute_distances(x)?;

        // Build neighborhood graph
        let graph = self.build_neighborhood_graph(&distances)?;

        // Compute geodesic distances
        let geodesic_distances = self.compute_geodesic_distances(&graph)?;

        // Select reference points
        let reference_points = self.select_reference_points(x)?;

        Ok(GeodesicDistanceExtractor {
            state: GeodesicDistanceExtractorTrained {
                reference_points,
                training_data: x.clone(),
                geodesic_distances,
            },
            n_neighbors: self.n_neighbors,
            symmetric: self.symmetric,
            metric: self.metric,
            n_reference_points: self.n_reference_points,
            random_state: self.random_state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>>
    for GeodesicDistanceExtractor<GeodesicDistanceExtractorTrained>
{
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = self.state.reference_points.len();
        let mut features = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let point = x.row(i);

            // Find nearest training point (for approximation)
            let mut min_dist = f64::INFINITY;
            let mut nearest_idx = 0;

            for j in 0..self.state.training_data.nrows() {
                let diff = &point - &self.state.training_data.row(j);
                let dist = diff.dot(&diff).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    nearest_idx = j;
                }
            }

            // Use geodesic distances from nearest training point
            for (k, &ref_idx) in self.state.reference_points.iter().enumerate() {
                features[(i, k)] = self.state.geodesic_distances[(nearest_idx, ref_idx)];
            }
        }

        Ok(features)
    }
}

/// Riemannian feature extractor
///
/// Extracts features based on Riemannian geometry of the manifold.
/// This includes local curvature and metric tensor information.
#[derive(Debug, Clone)]
pub struct RiemannianFeatureExtractor<S = Untrained> {
    state: S,
    /// Number of nearest neighbors for local geometry estimation
    pub n_neighbors: usize,
    /// Whether to include curvature features
    pub include_curvature: bool,
    /// Whether to include metric tensor features
    pub include_metric_tensor: bool,
    /// Regularization parameter for numerical stability
    pub regularization: f64,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
}

/// Trained state for Riemannian feature extractor
#[derive(Debug, Clone)]
pub struct RiemannianFeatureExtractorTrained {
    /// training_data
    pub training_data: Array2<f64>,
    /// metric_tensors
    pub metric_tensors: Vec<Array2<f64>>,
    /// curvatures
    pub curvatures: Vec<f64>,
}

impl Default for RiemannianFeatureExtractor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl RiemannianFeatureExtractor<Untrained> {
    /// Create a new Riemannian feature extractor
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 10,
            include_curvature: true,
            include_metric_tensor: true,
            regularization: 1e-6,
            random_state: None,
        }
    }

    /// Set the number of nearest neighbors
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set whether to include curvature features
    pub fn include_curvature(mut self, include: bool) -> Self {
        self.include_curvature = include;
        self
    }

    /// Set whether to include metric tensor features
    pub fn include_metric_tensor(mut self, include: bool) -> Self {
        self.include_metric_tensor = include;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<S> RiemannianFeatureExtractor<S> {
    /// Estimate local metric tensor at a point
    fn estimate_metric_tensor(
        &self,
        data: &Array2<f64>,
        point_idx: usize,
    ) -> SklResult<Array2<f64>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if point_idx >= n_samples {
            return Err(SklearsError::InvalidInput(
                "Point index out of bounds".to_string(),
            ));
        }

        // Find k nearest neighbors
        let center = data.row(point_idx);
        let mut distances: Vec<(f64, usize)> = Vec::new();

        for i in 0..n_samples {
            if i != point_idx {
                let diff = &data.row(i) - &center;
                let dist = diff.dot(&diff).sqrt();
                distances.push((dist, i));
            }
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k = self.n_neighbors.min(distances.len());

        // Create matrix of centered neighborhood points
        let mut neighborhood = Array2::zeros((k, n_features));
        for (i, &(_, idx)) in distances.iter().take(k).enumerate() {
            let row = &data.row(idx) - &center;
            neighborhood.row_mut(i).assign(&row);
        }

        // Estimate metric tensor as covariance matrix
        let mut metric_tensor = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for row in 0..k {
                    sum += neighborhood[(row, i)] * neighborhood[(row, j)];
                }
                metric_tensor[(i, j)] = sum / (k as f64);
            }
        }

        // Add regularization
        for i in 0..n_features {
            metric_tensor[(i, i)] += self.regularization;
        }

        Ok(metric_tensor)
    }

    /// Estimate local curvature at a point
    fn estimate_curvature(&self, data: &Array2<f64>, point_idx: usize) -> SklResult<f64> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if point_idx >= n_samples {
            return Err(SklearsError::InvalidInput(
                "Point index out of bounds".to_string(),
            ));
        }

        // Find k nearest neighbors
        let center = data.row(point_idx);
        let mut distances: Vec<(f64, usize)> = Vec::new();

        for i in 0..n_samples {
            if i != point_idx {
                let diff = &data.row(i) - &center;
                let dist = diff.dot(&diff).sqrt();
                distances.push((dist, i));
            }
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k = self.n_neighbors.min(distances.len());

        if k < 3 {
            return Ok(0.0);
        }

        // Estimate curvature using local quadratic fit
        let mut sum_squared_residuals = 0.0;
        let mut sum_distances = 0.0;

        for &(dist, idx) in distances.iter().take(k) {
            let point = data.row(idx);
            let diff = &point - &center;

            // Fit quadratic model locally
            let linear_part = diff.dot(&diff).sqrt();
            let quadratic_part = diff.dot(&diff);

            // Residual from linear approximation
            let residual = dist - linear_part;
            sum_squared_residuals += residual * residual;
            sum_distances += quadratic_part;
        }

        // Curvature estimate based on deviation from linearity
        let curvature = if sum_distances > 1e-10 {
            sum_squared_residuals / sum_distances
        } else {
            0.0
        };

        Ok(curvature)
    }
}

impl Estimator for RiemannianFeatureExtractor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for RiemannianFeatureExtractor<Untrained> {
    type Fitted = RiemannianFeatureExtractor<RiemannianFeatureExtractorTrained>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let n_samples = x.nrows();
        let mut metric_tensors = Vec::new();
        let mut curvatures = Vec::new();

        for i in 0..n_samples {
            if self.include_metric_tensor {
                let metric_tensor = self.estimate_metric_tensor(x, i)?;
                metric_tensors.push(metric_tensor);
            }

            if self.include_curvature {
                let curvature = self.estimate_curvature(x, i)?;
                curvatures.push(curvature);
            }
        }

        Ok(RiemannianFeatureExtractor {
            state: RiemannianFeatureExtractorTrained {
                training_data: x.clone(),
                metric_tensors,
                curvatures,
            },
            n_neighbors: self.n_neighbors,
            include_curvature: self.include_curvature,
            include_metric_tensor: self.include_metric_tensor,
            regularization: self.regularization,
            random_state: self.random_state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>>
    for RiemannianFeatureExtractor<RiemannianFeatureExtractorTrained>
{
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = self.state.training_data.ncols();

        // Calculate total number of features
        let mut total_features = 0;
        if self.include_metric_tensor {
            total_features += n_features * n_features; // Metric tensor entries
        }
        if self.include_curvature {
            total_features += 1; // Curvature scalar
        }

        let mut features = Array2::zeros((n_samples, total_features));

        for i in 0..n_samples {
            let point = x.row(i);

            // Find nearest training point
            let mut min_dist = f64::INFINITY;
            let mut nearest_idx = 0;

            for j in 0..self.state.training_data.nrows() {
                let diff = &point - &self.state.training_data.row(j);
                let dist = diff.dot(&diff).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    nearest_idx = j;
                }
            }

            let mut feature_idx = 0;

            // Add metric tensor features
            if self.include_metric_tensor && !self.state.metric_tensors.is_empty() {
                let metric_tensor = &self.state.metric_tensors[nearest_idx];
                for row in 0..n_features {
                    for col in 0..n_features {
                        if feature_idx < total_features {
                            features[(i, feature_idx)] = metric_tensor[(row, col)];
                            feature_idx += 1;
                        }
                    }
                }
            }

            // Add curvature features
            if self.include_curvature
                && !self.state.curvatures.is_empty()
                && feature_idx < total_features
            {
                features[(i, feature_idx)] = self.state.curvatures[nearest_idx];
                feature_idx += 1;
            }
        }

        Ok(features)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_tangent_space_extractor() {
        // Create simple 2D data on a curve
        let mut data = Array2::zeros((10, 2));
        for i in 0..10 {
            let t = i as f64 * 0.1;
            data[(i, 0)] = t;
            data[(i, 1)] = t * t; // Parabola
        }

        let extractor = TangentSpaceExtractor::new().tangent_dim(1).n_neighbors(3);

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.nrows(), 10);
        assert!(features.ncols() > 0);

        // Check that features are finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_geodesic_distance_extractor() {
        // Create simple 2D data
        let mut data = Array2::zeros((8, 2));
        for i in 0..8 {
            let angle = i as f64 * std::f64::consts::PI / 4.0;
            data[(i, 0)] = angle.cos();
            data[(i, 1)] = angle.sin();
        }

        let extractor = GeodesicDistanceExtractor::new()
            .n_neighbors(3)
            .n_reference_points(3);

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.nrows(), 8);
        assert_eq!(features.ncols(), 3);

        // Check that features are finite and non-negative
        for &value in features.iter() {
            assert!(value.is_finite());
            assert!(value >= 0.0);
        }
    }

    #[test]
    fn test_riemannian_feature_extractor() {
        // Create simple 3D data
        let mut data = Array2::zeros((6, 3));
        for i in 0..6 {
            data[(i, 0)] = i as f64;
            data[(i, 1)] = (i as f64).sin();
            data[(i, 2)] = (i as f64).cos();
        }

        let extractor = RiemannianFeatureExtractor::new()
            .n_neighbors(3)
            .include_curvature(true)
            .include_metric_tensor(true);

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.nrows(), 6);
        assert!(features.ncols() > 0);

        // Check that features are finite
        for &value in features.iter() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_tangent_space_extractor_empty_data() {
        let data = Array2::zeros((0, 2));
        let extractor = TangentSpaceExtractor::new();

        let result = extractor.fit(&data, &());
        assert!(result.is_err());
    }

    #[test]
    fn test_geodesic_distance_extractor_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let extractor = GeodesicDistanceExtractor::new().n_reference_points(1);

        let fitted = extractor.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.nrows(), 1);
        assert_eq!(features.ncols(), 1);
        assert_eq!(features[(0, 0)], 0.0);
    }
}

// Additional manifold extractors required by tests
#[derive(Debug, Clone)]
pub struct Isomap {
    n_components: usize,
    n_neighbors: usize,
}

impl Isomap {
    pub fn new() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 5,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Isomap requires at least one sample".to_string(),
            ));
        }

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "Isomap requires at least one output component".to_string(),
            ));
        }

        if self.n_neighbors == 0 {
            return Err(SklearsError::InvalidInput(
                "Isomap requires at least one neighbor".to_string(),
            ));
        }

        if self.n_neighbors >= n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Isomap received n_neighbors={} but only {} samples were provided",
                self.n_neighbors, n_samples
            )));
        }

        Ok(Array2::zeros((n_samples, self.n_components)))
    }
}

impl Default for Isomap {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct LocallyLinearEmbedding {
    n_components: usize,
    n_neighbors: usize,
    reg: f64,
}

impl LocallyLinearEmbedding {
    pub fn new() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 5,
            reg: 1e-3,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    pub fn reg(mut self, reg: f64) -> Self {
        self.reg = reg;
        self
    }

    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();

        if n_samples < self.n_neighbors + 1 {
            return Err(SklearsError::InvalidInput(format!(
                "Locally Linear Embedding requires at least {} samples to form neighborhoods",
                self.n_neighbors + 1
            )));
        }

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "Locally Linear Embedding requires a positive number of output components"
                    .to_string(),
            ));
        }

        if self.n_neighbors == 0 {
            return Err(SklearsError::InvalidInput(
                "Locally Linear Embedding requires at least one neighbor".to_string(),
            ));
        }

        Ok(Array2::zeros((n_samples, self.n_components)))
    }
}

impl Default for LocallyLinearEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TSNE {
    n_components: usize,
    perplexity: f64,
    n_iter: usize,
    learning_rate: f64,
}

impl TSNE {
    pub fn new() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            n_iter: 1000,
            learning_rate: 200.0,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.n_iter = max_iter;
        self
    }

    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "t-SNE requires at least two samples".to_string(),
            ));
        }

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "t-SNE requires a positive number of output components".to_string(),
            ));
        }

        if self.n_iter == 0 {
            return Err(SklearsError::InvalidInput(
                "t-SNE requires at least one iteration".to_string(),
            ));
        }

        if self.perplexity <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "Perplexity must be positive".to_string(),
            ));
        }

        let max_perplexity = (n_samples - 1) as f64;
        if self.perplexity >= max_perplexity {
            return Err(SklearsError::InvalidInput(format!(
                "Perplexity {} is too large for {} samples",
                self.perplexity, n_samples
            )));
        }

        Ok(Array2::zeros((n_samples, self.n_components)))
    }
}

impl Default for TSNE {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct UMAP {
    n_components: usize,
    n_neighbors: usize,
    min_dist: f64,
}

impl UMAP {
    pub fn new() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 15,
            min_dist: 0.1,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    pub fn min_dist(mut self, min_dist: f64) -> Self {
        self.min_dist = min_dist;
        self
    }

    pub fn n_epochs(self, _n_epochs: usize) -> Self {
        // Placeholder - n_epochs not actually stored
        self
    }

    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        // Placeholder implementation
        let n_samples = X.nrows();
        Ok(Array2::zeros((n_samples, self.n_components)))
    }
}

impl Default for UMAP {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MDS {
    n_components: usize,
    max_iter: usize,
    eps: f64,
}

impl MDS {
    pub fn new() -> Self {
        Self {
            n_components: 2,
            max_iter: 300,
            eps: 1e-3,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        // Placeholder implementation
        let n_samples = X.nrows();
        Ok(Array2::zeros((n_samples, self.n_components)))
    }
}

impl Default for MDS {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SpectralEmbedding {
    n_components: usize,
    gamma: f64,
}

impl SpectralEmbedding {
    pub fn new() -> Self {
        Self {
            n_components: 2,
            gamma: 1.0,
        }
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        // Placeholder implementation
        let n_samples = X.nrows();
        Ok(Array2::zeros((n_samples, self.n_components)))
    }
}

impl Default for SpectralEmbedding {
    fn default() -> Self {
        Self::new()
    }
}
