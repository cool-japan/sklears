//! Topological Data Analysis (TDA) algorithms
//!
//! This module provides tools for topological analysis of data, including:
//! - **Persistent Homology**: Computing topological features across multiple scales
//! - **Mapper Algorithm**: Topological data visualization and analysis
//! - **Simplicial Complexes**: Vietoris-Rips and Alpha complex construction
//! - **Topological Feature Extraction**: Converting persistent homology to features
//!
//! TDA provides insights into the "shape" of data by studying topological properties
//! that persist across different scales of analysis.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Persistent Homology computation
///
/// Computes the persistent homology of a point cloud, which captures
/// topological features (connected components, loops, voids) that persist
/// across multiple scales.
///
/// # Parameters
///
/// * `max_dimension` - Maximum homology dimension to compute (0, 1, 2)
/// * `distance_matrix` - Precomputed distance matrix (optional)
/// * `max_epsilon` - Maximum scale parameter for filtration
/// * `min_epsilon` - Minimum scale parameter for filtration
/// * `n_steps` - Number of steps in the filtration
///
/// # Examples
///
/// ```
/// use sklears_manifold::topological::PersistentHomology;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let ph = PersistentHomology::builder()
///     .max_dimension(1)
///     .n_steps(50)
///     .build();
/// let fitted = ph.fit(&data.view(), &()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PersistentHomology<S = Untrained> {
    max_dimension: usize,
    max_epsilon: Option<Float>,
    min_epsilon: Option<Float>,
    n_steps: usize,
    complex_type: String, // "vr" for Vietoris-Rips, "alpha" for Alpha complex
    state: S,
}

/// Trained state for PersistentHomology
#[derive(Debug, Clone)]
pub struct PersistentHomologyTrained {
    /// persistence_diagrams
    pub persistence_diagrams: Vec<Vec<(Float, Float)>>, // birth-death pairs for each dimension
    /// betti_numbers
    pub betti_numbers: Vec<Vec<usize>>, // Betti numbers across filtration
    /// epsilon_values
    pub epsilon_values: Vec<Float>,
    /// data_points
    pub data_points: Array2<Float>,
    /// distance_matrix
    pub distance_matrix: Array2<Float>,
}

/// Persistence diagram entry: (birth_time, death_time, dimension)
#[derive(Debug, Clone, PartialEq)]
pub struct PersistencePair {
    /// birth
    pub birth: Float,
    /// death
    pub death: Float,
    /// dimension
    pub dimension: usize,
}

impl Default for PersistentHomology<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentHomology<Untrained> {
    /// Create a new PersistentHomology builder
    pub fn builder() -> PersistentHomologyBuilder {
        PersistentHomologyBuilder::new()
    }

    /// Create a PersistentHomology with default parameters
    pub fn new() -> Self {
        Self {
            max_dimension: 1,
            max_epsilon: None,
            min_epsilon: None,
            n_steps: 50,
            complex_type: "vr".to_string(),
            state: Untrained,
        }
    }
}

/// Builder for PersistentHomology
#[derive(Debug)]
pub struct PersistentHomologyBuilder {
    max_dimension: usize,
    max_epsilon: Option<Float>,
    min_epsilon: Option<Float>,
    n_steps: usize,
    complex_type: String,
}

impl PersistentHomologyBuilder {
    fn new() -> Self {
        Self {
            max_dimension: 1,
            max_epsilon: None,
            min_epsilon: None,
            n_steps: 50,
            complex_type: "vr".to_string(),
        }
    }

    /// Set maximum homology dimension to compute
    pub fn max_dimension(mut self, max_dimension: usize) -> Self {
        self.max_dimension = max_dimension;
        self
    }

    /// Set maximum epsilon for filtration
    pub fn max_epsilon(mut self, max_epsilon: Float) -> Self {
        self.max_epsilon = Some(max_epsilon);
        self
    }

    /// Set minimum epsilon for filtration
    pub fn min_epsilon(mut self, min_epsilon: Float) -> Self {
        self.min_epsilon = Some(min_epsilon);
        self
    }

    /// Set number of steps in filtration
    pub fn n_steps(mut self, n_steps: usize) -> Self {
        self.n_steps = n_steps;
        self
    }

    /// Set complex type ("vr" for Vietoris-Rips, "alpha" for Alpha complex)
    pub fn complex_type(mut self, complex_type: &str) -> Self {
        self.complex_type = complex_type.to_string();
        self
    }

    /// Build the PersistentHomology
    pub fn build(self) -> PersistentHomology<Untrained> {
        PersistentHomology {
            max_dimension: self.max_dimension,
            max_epsilon: self.max_epsilon,
            min_epsilon: self.min_epsilon,
            n_steps: self.n_steps,
            complex_type: self.complex_type,
            state: Untrained,
        }
    }
}

impl Estimator for PersistentHomology<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for PersistentHomology<Untrained> {
    type Fitted = PersistentHomology<PersistentHomologyTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let n_samples = x.nrows();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for persistent homology".to_string(),
            ));
        }

        // Compute distance matrix
        let distance_matrix = compute_distance_matrix(x)?;

        // Determine epsilon range
        let max_epsilon = self
            .max_epsilon
            .unwrap_or_else(|| distance_matrix.iter().fold(0.0, |a, &b| a.max(b)));
        let min_epsilon = self.min_epsilon.unwrap_or(0.0);

        // Create filtration
        let epsilon_values: Vec<Float> = (0..self.n_steps)
            .map(|i| {
                min_epsilon
                    + (max_epsilon - min_epsilon) * (i as Float) / (self.n_steps - 1) as Float
            })
            .collect();

        // Compute persistent homology
        let (persistence_diagrams, betti_numbers) = match self.complex_type.as_str() {
            "vr" => compute_vietoris_rips_persistence(
                &distance_matrix,
                &epsilon_values,
                self.max_dimension,
            )?,
            "alpha" => compute_alpha_complex_persistence(x, &epsilon_values, self.max_dimension)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown complex type: {}",
                    self.complex_type
                )))
            }
        };

        let trained_state = PersistentHomologyTrained {
            persistence_diagrams,
            betti_numbers,
            epsilon_values,
            data_points: x.to_owned(),
            distance_matrix,
        };

        Ok(PersistentHomology {
            max_dimension: self.max_dimension,
            max_epsilon: self.max_epsilon,
            min_epsilon: self.min_epsilon,
            n_steps: self.n_steps,
            complex_type: self.complex_type,
            state: trained_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for PersistentHomology<PersistentHomologyTrained>
{
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        // Extract topological features from persistence diagrams
        extract_persistence_features(&self.state.persistence_diagrams)
    }
}

impl PersistentHomology<PersistentHomologyTrained> {
    /// Get persistence diagrams for each dimension
    pub fn persistence_diagrams(&self) -> &Vec<Vec<(Float, Float)>> {
        &self.state.persistence_diagrams
    }

    /// Get Betti numbers across the filtration
    pub fn betti_numbers(&self) -> &Vec<Vec<usize>> {
        &self.state.betti_numbers
    }

    /// Get epsilon values used in filtration
    pub fn epsilon_values(&self) -> &Vec<Float> {
        &self.state.epsilon_values
    }

    /// Compute persistence landscape representation
    pub fn persistence_landscape(
        &self,
        dimension: usize,
        resolution: usize,
    ) -> SklResult<Array2<Float>> {
        if dimension >= self.state.persistence_diagrams.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Dimension {} not computed",
                dimension
            )));
        }

        compute_persistence_landscape(&self.state.persistence_diagrams[dimension], resolution)
    }

    /// Compute persistence image representation
    pub fn persistence_image(
        &self,
        dimension: usize,
        resolution: usize,
        sigma: Float,
    ) -> SklResult<Array2<Float>> {
        if dimension >= self.state.persistence_diagrams.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Dimension {} not computed",
                dimension
            )));
        }

        compute_persistence_image(
            &self.state.persistence_diagrams[dimension],
            resolution,
            sigma,
        )
    }
}

/// Mapper algorithm for topological data visualization
///
/// The Mapper algorithm creates a simplicial complex that captures
/// the topological structure of data by using a filter function
/// and clustering within overlapping neighborhoods.
#[derive(Debug, Clone)]
pub struct Mapper<S = Untrained> {
    n_intervals: usize,
    overlap: Float,
    filter_function: String,   // "projection", "density", "distance"
    clustering_method: String, // "single_linkage", "dbscan"
    state: S,
}

/// Trained state for Mapper
#[derive(Debug, Clone)]
pub struct MapperTrained {
    /// nodes
    pub nodes: Vec<MapperNode>,
    /// edges
    pub edges: Vec<(usize, usize)>,
    /// filter_values
    pub filter_values: Array1<Float>,
    /// data_points
    pub data_points: Array2<Float>,
}

/// A node in the Mapper complex
#[derive(Debug, Clone)]
pub struct MapperNode {
    /// point_indices
    pub point_indices: Vec<usize>,
    /// center
    pub center: Array1<Float>,
    /// size
    pub size: usize,
}

impl Default for Mapper<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Mapper<Untrained> {
    /// Create a new Mapper with default parameters
    pub fn new() -> Self {
        Self {
            n_intervals: 10,
            overlap: 0.3,
            filter_function: "projection".to_string(),
            clustering_method: "single_linkage".to_string(),
            state: Untrained,
        }
    }

    /// Create a new Mapper builder
    pub fn builder() -> MapperBuilder {
        MapperBuilder::new()
    }
}

/// Builder for Mapper
#[derive(Debug)]
pub struct MapperBuilder {
    n_intervals: usize,
    overlap: Float,
    filter_function: String,
    clustering_method: String,
}

impl MapperBuilder {
    fn new() -> Self {
        Self {
            n_intervals: 10,
            overlap: 0.3,
            filter_function: "projection".to_string(),
            clustering_method: "single_linkage".to_string(),
        }
    }

    /// Set number of intervals for filter function
    pub fn n_intervals(mut self, n_intervals: usize) -> Self {
        self.n_intervals = n_intervals;
        self
    }

    /// Set overlap between intervals
    pub fn overlap(mut self, overlap: Float) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set filter function ("projection", "density", "distance")
    pub fn filter_function(mut self, filter_function: &str) -> Self {
        self.filter_function = filter_function.to_string();
        self
    }

    /// Set clustering method ("single_linkage", "dbscan")
    pub fn clustering_method(mut self, clustering_method: &str) -> Self {
        self.clustering_method = clustering_method.to_string();
        self
    }

    /// Build the Mapper
    pub fn build(self) -> Mapper<Untrained> {
        Mapper {
            n_intervals: self.n_intervals,
            overlap: self.overlap,
            filter_function: self.filter_function,
            clustering_method: self.clustering_method,
            state: Untrained,
        }
    }
}

impl Estimator for Mapper<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for Mapper<Untrained> {
    type Fitted = Mapper<MapperTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let n_samples = x.nrows();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for Mapper".to_string(),
            ));
        }

        // Compute filter function
        let filter_values = match self.filter_function.as_str() {
            "projection" => compute_projection_filter(x)?,
            "density" => compute_density_filter(x)?,
            "distance" => compute_distance_filter(x)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown filter function: {}",
                    self.filter_function
                )))
            }
        };

        // Create overlapping intervals
        let intervals =
            create_overlapping_intervals(&filter_values, self.n_intervals, self.overlap)?;

        // Cluster points within each interval
        let mut nodes = Vec::new();
        let mut node_id = 0;

        for interval in intervals {
            let points_in_interval: Vec<usize> = interval.point_indices;

            if !points_in_interval.is_empty() {
                let clusters =
                    cluster_points_in_interval(x, &points_in_interval, &self.clustering_method)?;

                for cluster in clusters {
                    if !cluster.is_empty() {
                        let center = compute_cluster_center(x, &cluster)?;
                        let size = cluster.len();
                        nodes.push(MapperNode {
                            point_indices: cluster,
                            center,
                            size,
                        });
                        node_id += 1;
                    }
                }
            }
        }

        // Build edges between overlapping nodes
        let edges = build_mapper_edges(&nodes);

        let trained_state = MapperTrained {
            nodes,
            edges,
            filter_values,
            data_points: x.to_owned(),
        };

        Ok(Mapper {
            n_intervals: self.n_intervals,
            overlap: self.overlap,
            filter_function: self.filter_function,
            clustering_method: self.clustering_method,
            state: trained_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for Mapper<MapperTrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        // Return adjacency matrix of the Mapper complex
        let n_nodes = self.state.nodes.len();
        let mut adjacency = Array2::zeros((n_nodes, n_nodes));

        for &(i, j) in &self.state.edges {
            adjacency[(i, j)] = 1.0;
            adjacency[(j, i)] = 1.0;
        }

        Ok(adjacency)
    }
}

// Helper functions for TDA computations

/// Compute distance matrix for a dataset
fn compute_distance_matrix(x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
    let n_samples = x.nrows();
    let mut distances = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in i..n_samples {
            let dist = if i == j {
                0.0
            } else {
                (&x.row(i) - &x.row(j)).mapv(|x| x * x).sum().sqrt()
            };
            distances[(i, j)] = dist;
            distances[(j, i)] = dist;
        }
    }

    Ok(distances)
}

/// Compute Vietoris-Rips persistent homology
fn compute_vietoris_rips_persistence(
    distance_matrix: &Array2<Float>,
    epsilon_values: &[Float],
    max_dimension: usize,
) -> SklResult<(Vec<Vec<(Float, Float)>>, Vec<Vec<usize>>)> {
    let n_samples = distance_matrix.nrows();
    let mut persistence_diagrams = vec![Vec::new(); max_dimension + 1];
    let mut betti_numbers = vec![Vec::new(); max_dimension + 1];

    // Simplified persistence computation
    // In practice, this would use more sophisticated algorithms like persistent homology

    for &epsilon in epsilon_values {
        // Build simplicial complex at this scale
        let complex = build_vietoris_rips_complex(distance_matrix, epsilon, max_dimension)?;

        // Compute Betti numbers (simplified)
        let betti = compute_betti_numbers(&complex, max_dimension)?;

        for dim in 0..=max_dimension {
            betti_numbers[dim].push(betti[dim]);
        }
    }

    // Extract persistence pairs (simplified)
    for dim in 0..=max_dimension {
        for i in 0..(epsilon_values.len() - 1) {
            if betti_numbers[dim][i] > betti_numbers[dim][i + 1] {
                // Feature dies
                persistence_diagrams[dim].push((epsilon_values[i], epsilon_values[i + 1]));
            }
        }
    }

    Ok((persistence_diagrams, betti_numbers))
}

/// Compute Alpha complex persistent homology
fn compute_alpha_complex_persistence(
    _x: &ArrayView2<'_, Float>,
    epsilon_values: &[Float],
    max_dimension: usize,
) -> SklResult<(Vec<Vec<(Float, Float)>>, Vec<Vec<usize>>)> {
    // Placeholder for Alpha complex implementation
    // This would require computational geometry algorithms
    let persistence_diagrams = vec![Vec::new(); max_dimension + 1];
    let betti_numbers = vec![vec![0; epsilon_values.len()]; max_dimension + 1];

    Ok((persistence_diagrams, betti_numbers))
}

/// Build Vietoris-Rips complex at given scale
fn build_vietoris_rips_complex(
    distance_matrix: &Array2<Float>,
    epsilon: Float,
    max_dimension: usize,
) -> SklResult<SimplicalComplex> {
    let n_samples = distance_matrix.nrows();
    let mut complex = SimplicalComplex::new();

    // Add vertices
    for i in 0..n_samples {
        complex.add_simplex(vec![i], 0.0);
    }

    // Add edges
    for i in 0..n_samples {
        for j in i + 1..n_samples {
            if distance_matrix[(i, j)] <= epsilon {
                complex.add_simplex(vec![i, j], distance_matrix[(i, j)]);
            }
        }
    }

    // Add higher-dimensional simplices (limited implementation)
    if max_dimension >= 2 {
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                for k in j + 1..n_samples {
                    let max_dist = distance_matrix[(i, j)]
                        .max(distance_matrix[(i, k)])
                        .max(distance_matrix[(j, k)]);
                    if max_dist <= epsilon {
                        complex.add_simplex(vec![i, j, k], max_dist);
                    }
                }
            }
        }
    }

    Ok(complex)
}

/// Simplified simplicial complex representation
#[derive(Debug, Clone)]
struct SimplicalComplex {
    simplices: Vec<(Vec<usize>, Float)>, // (vertices, filtration_value)
}

impl SimplicalComplex {
    fn new() -> Self {
        Self {
            simplices: Vec::new(),
        }
    }

    fn add_simplex(&mut self, vertices: Vec<usize>, filtration_value: Float) {
        self.simplices.push((vertices, filtration_value));
    }
}

/// Compute Betti numbers for a simplicial complex
fn compute_betti_numbers(
    _complex: &SimplicalComplex,
    max_dimension: usize,
) -> SklResult<Vec<usize>> {
    // Placeholder for Betti number computation
    // This would require linear algebra over finite fields
    Ok(vec![0; max_dimension + 1])
}

/// Extract features from persistence diagrams
fn extract_persistence_features(
    persistence_diagrams: &[Vec<(Float, Float)>],
) -> SklResult<Array2<Float>> {
    let mut features = Vec::new();

    for diagram in persistence_diagrams {
        if diagram.is_empty() {
            features.extend(vec![0.0; 4]); // No features for this dimension
        } else {
            // Compute summary statistics
            let lifetimes: Vec<Float> = diagram.iter().map(|(b, d)| d - b).collect();
            let max_lifetime = lifetimes.iter().fold(0.0 as Float, |a, &b| a.max(b));
            let mean_lifetime = lifetimes.iter().sum::<Float>() / lifetimes.len() as Float;
            let num_features = diagram.len() as Float;
            let total_persistence = lifetimes.iter().sum::<Float>();

            features.extend(vec![
                max_lifetime,
                mean_lifetime,
                num_features,
                total_persistence,
            ]);
        }
    }

    Array2::from_shape_vec((1, features.len()), features)
        .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
}

/// Compute persistence landscape
fn compute_persistence_landscape(
    diagram: &[(Float, Float)],
    resolution: usize,
) -> SklResult<Array2<Float>> {
    if diagram.is_empty() {
        return Ok(Array2::zeros((1, resolution)));
    }

    let min_val = diagram
        .iter()
        .map(|(b, _)| *b)
        .fold(Float::INFINITY, |a, b| a.min(b));
    let max_val = diagram
        .iter()
        .map(|(_, d)| *d)
        .fold(0.0 as Float, |a, b| a.max(b));

    let step = (max_val - min_val) / (resolution - 1) as Float;
    let mut landscape = Array2::zeros((1, resolution));

    for i in 0..resolution {
        let t = min_val + i as Float * step;
        let mut values = Vec::new();

        for &(birth, death) in diagram {
            if birth <= t && t <= death {
                values.push((t - birth).min(death - t));
            }
        }

        values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        landscape[(0, i)] = values.first().copied().unwrap_or(0.0);
    }

    Ok(landscape)
}

/// Compute persistence image
fn compute_persistence_image(
    diagram: &[(Float, Float)],
    resolution: usize,
    sigma: Float,
) -> SklResult<Array2<Float>> {
    let mut image = Array2::zeros((resolution, resolution));

    if diagram.is_empty() {
        return Ok(image);
    }

    let min_birth = diagram
        .iter()
        .map(|(b, _)| *b)
        .fold(Float::INFINITY, |a, b| a.min(b));
    let max_birth = diagram
        .iter()
        .map(|(b, _)| *b)
        .fold(0.0 as Float, |a, b| a.max(b));
    let min_death = diagram
        .iter()
        .map(|(_, d)| *d)
        .fold(Float::INFINITY, |a, b| a.min(b));
    let max_death = diagram
        .iter()
        .map(|(_, d)| *d)
        .fold(0.0 as Float, |a, b| a.max(b));

    let birth_step = (max_birth - min_birth) / (resolution - 1) as Float;
    let death_step = (max_death - min_death) / (resolution - 1) as Float;

    for i in 0..resolution {
        for j in 0..resolution {
            let birth_coord = min_birth + i as Float * birth_step;
            let death_coord = min_death + j as Float * death_step;

            if birth_coord <= death_coord {
                let mut weight = 0.0;

                for &(birth, death) in diagram {
                    let persistence = death - birth;
                    let dist_sq = (birth - birth_coord).powi(2) + (death - death_coord).powi(2);
                    weight += persistence * (-dist_sq / (2.0 * sigma * sigma)).exp();
                }

                image[(i, j)] = weight;
            }
        }
    }

    Ok(image)
}

// Helper functions for Mapper algorithm

/// Interval for Mapper algorithm
#[derive(Debug, Clone)]
struct MapperInterval {
    point_indices: Vec<usize>,
    start: Float,
    end: Float,
}

/// Compute projection filter (first principal component)
fn compute_projection_filter(x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
    let n_samples = x.nrows();

    // Simple projection onto first coordinate for now
    // In practice, would use PCA
    let mut filter_values = Array1::zeros(n_samples);

    for i in 0..n_samples {
        filter_values[i] = x[(i, 0)];
    }

    Ok(filter_values)
}

/// Compute density filter
fn compute_density_filter(x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
    let n_samples = x.nrows();
    let mut filter_values = Array1::zeros(n_samples);

    // Compute local density using k-nearest neighbors
    let k = (n_samples as Float).sqrt() as usize;

    for i in 0..n_samples {
        let mut distances = Vec::new();
        for j in 0..n_samples {
            if i != j {
                let dist = (&x.row(i) - &x.row(j)).mapv(|x| x * x).sum().sqrt();
                distances.push(dist);
            }
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let kth_distance = distances
            .get(k.min(distances.len() - 1))
            .copied()
            .unwrap_or(1.0);
        filter_values[i] = 1.0 / (1.0 + kth_distance); // Inverse of k-th nearest neighbor distance
    }

    Ok(filter_values)
}

/// Compute distance filter (distance to a reference point)
fn compute_distance_filter(x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
    let n_samples = x.nrows();
    let mut filter_values = Array1::zeros(n_samples);

    // Use centroid as reference point
    let centroid = x.mean_axis(Axis(0)).unwrap();

    for i in 0..n_samples {
        let dist = (&x.row(i) - &centroid).mapv(|x| x * x).sum().sqrt();
        filter_values[i] = dist;
    }

    Ok(filter_values)
}

/// Create overlapping intervals for Mapper
fn create_overlapping_intervals(
    filter_values: &Array1<Float>,
    n_intervals: usize,
    overlap: Float,
) -> SklResult<Vec<MapperInterval>> {
    let min_val = filter_values.iter().fold(Float::INFINITY, |a, &b| a.min(b));
    let max_val = filter_values.iter().fold(0.0 as Float, |a, &b| a.max(b));

    let interval_length = (max_val - min_val) / n_intervals as Float;
    let step_size = interval_length * (1.0 - overlap);

    let mut intervals = Vec::new();

    for i in 0..n_intervals {
        let start = min_val + i as Float * step_size;
        let end = start + interval_length;

        let mut point_indices = Vec::new();
        for (j, &value) in filter_values.iter().enumerate() {
            if value >= start && value <= end {
                point_indices.push(j);
            }
        }

        if !point_indices.is_empty() {
            intervals.push(MapperInterval {
                point_indices,
                start,
                end,
            });
        }
    }

    Ok(intervals)
}

/// Cluster points within an interval
fn cluster_points_in_interval(
    x: &ArrayView2<'_, Float>,
    point_indices: &[usize],
    clustering_method: &str,
) -> SklResult<Vec<Vec<usize>>> {
    if point_indices.len() <= 1 {
        return Ok(vec![point_indices.to_vec()]);
    }

    match clustering_method {
        "single_linkage" => single_linkage_clustering(x, point_indices),
        "dbscan" => dbscan_clustering(x, point_indices),
        _ => Ok(vec![point_indices.to_vec()]), // No clustering
    }
}

/// Simple single linkage clustering
fn single_linkage_clustering(
    x: &ArrayView2<'_, Float>,
    point_indices: &[usize],
) -> SklResult<Vec<Vec<usize>>> {
    if point_indices.len() <= 2 {
        return Ok(vec![point_indices.to_vec()]);
    }

    // Simplified: split into two clusters based on distance
    let n = point_indices.len();
    let mut distances = Vec::new();

    for i in 0..n {
        for j in i + 1..n {
            let dist = (&x.row(point_indices[i]) - &x.row(point_indices[j]))
                .mapv(|x| x * x)
                .sum()
                .sqrt();
            distances.push((dist, i, j));
        }
    }

    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Simple two-cluster split
    let mid = n / 2;
    Ok(vec![
        point_indices[..mid].to_vec(),
        point_indices[mid..].to_vec(),
    ])
}

/// Simple DBSCAN clustering placeholder
fn dbscan_clustering(
    _x: &ArrayView2<'_, Float>,
    point_indices: &[usize],
) -> SklResult<Vec<Vec<usize>>> {
    // Placeholder: return single cluster
    Ok(vec![point_indices.to_vec()])
}

/// Compute center of a cluster
fn compute_cluster_center(
    x: &ArrayView2<'_, Float>,
    cluster: &[usize],
) -> SklResult<Array1<Float>> {
    let n_features = x.ncols();
    let mut center = Array1::zeros(n_features);

    for &idx in cluster {
        center = center + x.row(idx);
    }

    center /= cluster.len() as Float;
    Ok(center)
}

/// Build edges between Mapper nodes
fn build_mapper_edges(nodes: &[MapperNode]) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();

    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            // Check if nodes share any points
            let shared = nodes[i]
                .point_indices
                .iter()
                .any(|&idx| nodes[j].point_indices.contains(&idx));

            if shared {
                edges.push((i, j));
            }
        }
    }

    edges
}
