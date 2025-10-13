//! Extensible distance metrics for manifold learning
//!
//! This module provides a registry system for distance metrics that can be
//! used across different manifold learning algorithms, with support for
//! custom user-defined metrics and automatic metric selection.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::sync::{Arc, RwLock};

/// Trait for distance metrics
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
pub trait DistanceMetric: Send + Sync {
    /// Compute distance between two points
    fn distance(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> SklResult<f64>;

    /// Compute pairwise distances between all points in X
    fn pairwise_distances(&self, x: ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let n = x.shape()[0];
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let dist = self.distance(x.row(i), x.row(j))?;
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        Ok(distances)
    }

    /// Compute distances between points in X and Y
    fn pairwise_distances_cross(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_x = x.shape()[0];
        let n_y = y.shape()[0];
        let mut distances = Array2::zeros((n_x, n_y));

        for i in 0..n_x {
            for j in 0..n_y {
                distances[[i, j]] = self.distance(x.row(i), y.row(j))?;
            }
        }

        Ok(distances)
    }

    /// Get the name of the metric
    fn name(&self) -> &str;

    /// Check if the metric satisfies the triangle inequality
    fn is_metric(&self) -> bool {
        true // Most distance metrics satisfy this
    }

    /// Check if the metric is symmetric
    fn is_symmetric(&self) -> bool {
        true // Most distance metrics are symmetric
    }

    /// Get metric-specific parameters
    fn parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Euclidean distance metric
#[derive(Debug, Clone)]
pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> SklResult<f64> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Points must have the same dimensionality".to_string(),
            ));
        }

        let diff = &x - &y;
        Ok(diff.dot(&diff).sqrt())
    }

    fn name(&self) -> &str {
        "euclidean"
    }
}

/// Manhattan (L1) distance metric
#[derive(Debug, Clone)]
pub struct ManhattanDistance;

impl DistanceMetric for ManhattanDistance {
    fn distance(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> SklResult<f64> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Points must have the same dimensionality".to_string(),
            ));
        }

        Ok((&x - &y).map(|a| a.abs()).sum())
    }

    fn name(&self) -> &str {
        "manhattan"
    }
}

/// Chebyshev (L∞) distance metric
#[derive(Debug, Clone)]
pub struct ChebyshevDistance;

impl DistanceMetric for ChebyshevDistance {
    fn distance(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> SklResult<f64> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Points must have the same dimensionality".to_string(),
            ));
        }

        Ok((&x - &y)
            .map(|a| a.abs())
            .fold(0.0, |acc, &val| acc.max(val)))
    }

    fn name(&self) -> &str {
        "chebyshev"
    }
}

/// Minkowski distance metric (generalization of Euclidean and Manhattan)
#[derive(Debug, Clone)]
pub struct MinkowskiDistance {
    p: f64,
}

impl MinkowskiDistance {
    /// Create a new Minkowski distance with parameter p
    pub fn new(p: f64) -> SklResult<Self> {
        if p < 1.0 {
            return Err(SklearsError::InvalidInput(
                "Minkowski p parameter must be >= 1.0".to_string(),
            ));
        }
        Ok(Self { p })
    }
}

impl DistanceMetric for MinkowskiDistance {
    fn distance(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> SklResult<f64> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Points must have the same dimensionality".to_string(),
            ));
        }

        if self.p == 1.0 {
            Ok((&x - &y).map(|a| a.abs()).sum())
        } else if self.p == 2.0 {
            let diff = &x - &y;
            Ok(diff.dot(&diff).sqrt())
        } else if self.p.is_infinite() {
            Ok((&x - &y)
                .map(|a| a.abs())
                .fold(0.0, |acc, &val| acc.max(val)))
        } else {
            Ok((&x - &y)
                .map(|a| a.abs().powf(self.p))
                .sum()
                .powf(1.0 / self.p))
        }
    }

    fn name(&self) -> &str {
        "minkowski"
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("p".to_string(), self.p);
        params
    }
}

/// Cosine distance metric (1 - cosine similarity)
#[derive(Debug, Clone)]
pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    fn distance(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> SklResult<f64> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Points must have the same dimensionality".to_string(),
            ));
        }

        let dot_product = x.dot(&y);
        let norm_x = x.dot(&x).sqrt();
        let norm_y = y.dot(&y).sqrt();

        if norm_x == 0.0 || norm_y == 0.0 {
            return Ok(1.0); // Maximum distance for zero vectors
        }

        let cosine_similarity = dot_product / (norm_x * norm_y);
        Ok(1.0 - cosine_similarity)
    }

    fn name(&self) -> &str {
        "cosine"
    }
}

/// Correlation distance metric (1 - Pearson correlation)
#[derive(Debug, Clone)]
pub struct CorrelationDistance;

impl DistanceMetric for CorrelationDistance {
    fn distance(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> SklResult<f64> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Points must have the same dimensionality".to_string(),
            ));
        }

        let n = x.len() as f64;
        if n < 2.0 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 dimensions for correlation".to_string(),
            ));
        }

        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let x_centered: Array1<f64> = x.map(|&v| v - mean_x);
        let y_centered: Array1<f64> = y.map(|&v| v - mean_y);

        let numerator = x_centered.dot(&y_centered);
        let denominator = (x_centered.dot(&x_centered) * y_centered.dot(&y_centered)).sqrt();

        if denominator == 0.0 {
            return Ok(1.0); // Maximum distance for constant vectors
        }

        let correlation = numerator / denominator;
        Ok(1.0 - correlation)
    }

    fn name(&self) -> &str {
        "correlation"
    }
}

/// Hamming distance metric (for binary data)
#[derive(Debug, Clone)]
pub struct HammingDistance;

impl DistanceMetric for HammingDistance {
    fn distance(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> SklResult<f64> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Points must have the same dimensionality".to_string(),
            ));
        }

        let n = x.len() as f64;
        let different = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| if (a - b).abs() > 1e-10 { 1.0 } else { 0.0 })
            .sum::<f64>();

        Ok(different / n)
    }

    fn name(&self) -> &str {
        "hamming"
    }
}

/// Mahalanobis distance metric (with precomputed inverse covariance)
#[derive(Debug, Clone)]
pub struct MahalanobisDistance {
    inv_cov: Array2<f64>,
}

impl MahalanobisDistance {
    /// Create a new Mahalanobis distance with inverse covariance matrix
    pub fn new(inv_cov: Array2<f64>) -> SklResult<Self> {
        let shape = inv_cov.shape();
        if shape[0] != shape[1] {
            return Err(SklearsError::InvalidInput(
                "Inverse covariance matrix must be square".to_string(),
            ));
        }
        Ok(Self { inv_cov })
    }

    /// Create from covariance matrix (will compute inverse)
    pub fn from_covariance(cov: Array2<f64>) -> SklResult<Self> {
        use scirs2_core::ndarray::ndarray_linalg::Inverse;
        let inv_cov = cov.inv().map_err(|_| {
            SklearsError::InvalidInput("Covariance matrix is not invertible".to_string())
        })?;
        Self::new(inv_cov)
    }
}

impl DistanceMetric for MahalanobisDistance {
    fn distance(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> SklResult<f64> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Points must have the same dimensionality".to_string(),
            ));
        }

        if x.len() != self.inv_cov.shape()[0] {
            return Err(SklearsError::InvalidInput(
                "Point dimensionality doesn't match covariance matrix".to_string(),
            ));
        }

        let diff = &x - &y;
        let temp = self.inv_cov.dot(&diff);
        Ok(diff.dot(&temp).sqrt())
    }

    fn name(&self) -> &str {
        "mahalanobis"
    }
}

/// Global registry for distance metrics
static METRIC_REGISTRY: RwLock<Option<MetricRegistry>> = RwLock::new(None);

/// Registry for distance metrics
pub struct MetricRegistry {
    metrics: HashMap<String, Arc<dyn DistanceMetric>>,
}

impl MetricRegistry {
    /// Create a new metric registry with default metrics
    pub fn new() -> Self {
        let mut registry = Self {
            metrics: HashMap::new(),
        };

        // Register default metrics
        registry.register("euclidean", Arc::new(EuclideanDistance));
        registry.register("manhattan", Arc::new(ManhattanDistance));
        registry.register("chebyshev", Arc::new(ChebyshevDistance));
        registry.register("cosine", Arc::new(CosineDistance));
        registry.register("correlation", Arc::new(CorrelationDistance));
        registry.register("hamming", Arc::new(HammingDistance));

        // Register some common Minkowski metrics
        if let Ok(l1) = MinkowskiDistance::new(1.0) {
            registry.register("l1", Arc::new(l1));
        }
        if let Ok(l2) = MinkowskiDistance::new(2.0) {
            registry.register("l2", Arc::new(l2));
        }

        registry
    }

    /// Register a new metric
    pub fn register(&mut self, name: impl Into<String>, metric: Arc<dyn DistanceMetric>) {
        self.metrics.insert(name.into(), metric);
    }

    /// Get a metric by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn DistanceMetric>> {
        self.metrics.get(name).cloned()
    }

    /// List all available metrics
    pub fn list_metrics(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }

    /// Check if a metric is registered
    pub fn contains(&self, name: &str) -> bool {
        self.metrics.contains_key(name)
    }

    /// Remove a metric
    pub fn remove(&mut self, name: &str) -> Option<Arc<dyn DistanceMetric>> {
        self.metrics.remove(name)
    }
}

impl Default for MetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the global metric registry
pub fn get_global_registry() -> Arc<RwLock<MetricRegistry>> {
    {
        let registry = METRIC_REGISTRY.read().unwrap();
        if registry.is_some() {
            return Arc::new(RwLock::new(registry.as_ref().unwrap().clone()));
        }
    }

    let mut registry = METRIC_REGISTRY.write().unwrap();
    if registry.is_none() {
        *registry = Some(MetricRegistry::new());
    }
    Arc::new(RwLock::new(registry.as_ref().unwrap().clone()))
}

/// Initialize the global metric registry
pub fn init_global_registry() {
    let mut registry = METRIC_REGISTRY.write().unwrap();
    *registry = Some(MetricRegistry::new());
}

/// Register a metric globally
pub fn register_metric(name: impl Into<String>, metric: Arc<dyn DistanceMetric>) -> SklResult<()> {
    let registry = get_global_registry();
    let mut reg = registry
        .write()
        .map_err(|_| SklearsError::InvalidInput("Failed to acquire registry lock".to_string()))?;
    reg.register(name, metric);
    Ok(())
}

/// Get a metric from the global registry
pub fn get_metric(name: &str) -> Option<Arc<dyn DistanceMetric>> {
    let registry = get_global_registry();
    let reg = registry.read().ok()?;
    reg.get(name)
}

/// List all available metrics in the global registry
pub fn list_available_metrics() -> Vec<String> {
    let registry = get_global_registry();
    registry
        .read()
        .map(|reg| reg.list_metrics())
        .unwrap_or_else(|_| Vec::new())
}

/// Create a metric from name and parameters
pub fn create_metric(
    name: &str,
    params: Option<HashMap<String, f64>>,
) -> SklResult<Arc<dyn DistanceMetric>> {
    match name {
        "euclidean" => Ok(Arc::new(EuclideanDistance)),
        "manhattan" | "l1" => Ok(Arc::new(ManhattanDistance)),
        "chebyshev" => Ok(Arc::new(ChebyshevDistance)),
        "cosine" => Ok(Arc::new(CosineDistance)),
        "correlation" => Ok(Arc::new(CorrelationDistance)),
        "hamming" => Ok(Arc::new(HammingDistance)),
        "minkowski" => {
            let p = params.and_then(|p| p.get("p").copied()).unwrap_or(2.0);
            Ok(Arc::new(MinkowskiDistance::new(p)?))
        }
        _ => {
            // Try to get from global registry
            get_metric(name)
                .ok_or_else(|| SklearsError::InvalidInput(format!("Unknown metric: {}", name)))
        }
    }
}

/// Metric selector that chooses appropriate metrics based on data characteristics
pub struct MetricSelector;

impl MetricSelector {
    /// Suggest appropriate metrics based on data characteristics
    pub fn suggest_metrics(
        data_type: DataType,
        dimensionality: usize,
        sparsity: f64,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        match data_type {
            DataType::Continuous => {
                suggestions.push("euclidean".to_string());
                if dimensionality < 50 {
                    suggestions.push("correlation".to_string());
                }
                if sparsity > 0.5 {
                    suggestions.push("cosine".to_string());
                }
                suggestions.push("manhattan".to_string());
            }
            DataType::Binary => {
                suggestions.push("hamming".to_string());
                suggestions.push("cosine".to_string());
            }
            DataType::Categorical => {
                suggestions.push("hamming".to_string());
            }
            DataType::Mixed => {
                suggestions.push("hamming".to_string());
                suggestions.push("manhattan".to_string());
            }
        }

        suggestions
    }

    /// Get the default metric for a data type
    pub fn default_metric(data_type: DataType) -> String {
        match data_type {
            DataType::Continuous => "euclidean".to_string(),
            DataType::Binary => "hamming".to_string(),
            DataType::Categorical => "hamming".to_string(),
            DataType::Mixed => "manhattan".to_string(),
        }
    }
}

/// Data type for metric selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Continuous numerical data
    Continuous,
    /// Binary data (0/1)
    Binary,
    /// Categorical data
    Categorical,
    /// Mixed data types
    Mixed,
}

/// Clone implementation for MetricRegistry
impl Clone for MetricRegistry {
    fn clone(&self) -> Self {
        Self {
            metrics: self.metrics.clone(),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_euclidean_distance() {
        let metric = EuclideanDistance;
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];

        let dist = metric.distance(x.view(), y.view()).unwrap();
        let expected = ((3.0_f64).powi(2) + (3.0_f64).powi(2) + (3.0_f64).powi(2)).sqrt();
        assert!((dist - expected).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let metric = ManhattanDistance;
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];

        let dist = metric.distance(x.view(), y.view()).unwrap();
        assert_eq!(dist, 9.0);
    }

    #[test]
    fn test_cosine_distance() {
        let metric = CosineDistance;
        let x = array![1.0, 0.0, 0.0];
        let y = array![0.0, 1.0, 0.0];

        let dist = metric.distance(x.view(), y.view()).unwrap();
        assert!((dist - 1.0).abs() < 1e-10); // Orthogonal vectors

        let x = array![1.0, 0.0, 0.0];
        let y = array![1.0, 0.0, 0.0];

        let dist = metric.distance(x.view(), y.view()).unwrap();
        assert!(dist.abs() < 1e-10); // Same vectors
    }

    #[test]
    fn test_minkowski_distance() {
        // Test p=1 (Manhattan)
        let metric = MinkowskiDistance::new(1.0).unwrap();
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];

        let dist = metric.distance(x.view(), y.view()).unwrap();
        assert_eq!(dist, 9.0);

        // Test p=2 (Euclidean)
        let metric = MinkowskiDistance::new(2.0).unwrap();
        let dist = metric.distance(x.view(), y.view()).unwrap();
        let expected = ((3.0_f64).powi(2) + (3.0_f64).powi(2) + (3.0_f64).powi(2)).sqrt();
        assert!((dist - expected).abs() < 1e-10);
    }

    #[test]
    fn test_metric_registry() {
        let mut registry = MetricRegistry::new();

        // Test default metrics
        assert!(registry.contains("euclidean"));
        assert!(registry.contains("manhattan"));
        assert!(registry.contains("cosine"));

        // Test custom metric registration
        let custom_metric = Arc::new(ChebyshevDistance);
        registry.register("custom_chebyshev", custom_metric);
        assert!(registry.contains("custom_chebyshev"));

        // Test metric retrieval
        let metric = registry.get("euclidean").unwrap();
        assert_eq!(metric.name(), "euclidean");

        // Test listing metrics
        let metrics = registry.list_metrics();
        assert!(metrics.contains(&"euclidean".to_string()));
        assert!(metrics.contains(&"custom_chebyshev".to_string()));
    }

    #[test]
    fn test_metric_creation() {
        let metric = create_metric("euclidean", None).unwrap();
        assert_eq!(metric.name(), "euclidean");

        let mut params = HashMap::new();
        params.insert("p".to_string(), 3.0);
        let metric = create_metric("minkowski", Some(params)).unwrap();
        assert_eq!(metric.name(), "minkowski");

        // Test unknown metric
        assert!(create_metric("unknown", None).is_err());
    }

    #[test]
    fn test_metric_selector() {
        let suggestions = MetricSelector::suggest_metrics(DataType::Continuous, 10, 0.1);
        assert!(suggestions.contains(&"euclidean".to_string()));

        let suggestions = MetricSelector::suggest_metrics(DataType::Binary, 100, 0.8);
        assert!(suggestions.contains(&"hamming".to_string()));

        let default = MetricSelector::default_metric(DataType::Continuous);
        assert_eq!(default, "euclidean");
    }

    #[test]
    fn test_pairwise_distances() {
        let metric = EuclideanDistance;
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let distances = metric.pairwise_distances(x.view()).unwrap();
        assert_eq!(distances.shape(), &[3, 3]);

        // Diagonal should be zero
        for i in 0..3 {
            assert!(distances[[i, i]].abs() < 1e-10);
        }

        // Should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((distances[[i, j]] - distances[[j, i]]).abs() < 1e-10);
            }
        }
    }
}
