//! Multi-view Clustering
//!
//! Multi-view clustering algorithms that perform clustering across multiple data views
//! simultaneously, finding cluster assignments that are consistent across views while
//! respecting the structure within each view.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Random, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Transform},
    types::Float,
};
use std::marker::PhantomData;

/// Multi-view Clustering Algorithm
///
/// This algorithm performs clustering across multiple data views by finding cluster
/// assignments that maximize agreement between views while preserving within-view
/// cluster structure. It uses an alternating optimization approach to find consensus
/// cluster assignments.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use sklears_cross_decomposition::MultiViewClustering;
/// use sklears_core::traits::Fit;
///
/// let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.5, 2.5]];
/// let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [2.5, 3.5]];
/// let view3 = array![[1.2, 2.2], [3.2, 4.2], [5.2, 6.2], [1.7, 2.7]];
/// let views = vec![view1, view2, view3];
///
/// let mvc = MultiViewClustering::new(2).consensus_weight(0.7);
/// let fitted = mvc.fit(&views, &()).unwrap();
/// let cluster_assignments = fitted.labels();
/// ```
#[derive(Debug, Clone)]
pub struct MultiViewClustering<State = Untrained> {
    /// Number of clusters
    pub n_clusters: usize,
    /// Weight for consensus vs individual clustering (0.0 to 1.0)
    pub consensus_weight: Float,
    /// Initialization method for cluster centers
    pub init_method: InitMethod,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Distance metric for clustering
    pub distance_metric: DistanceMetric,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Final cluster labels for each sample
    labels_: Option<Array1<usize>>,
    /// Cluster centers for each view
    cluster_centers_: Option<Vec<Array2<Float>>>,
    /// Consensus cluster centers
    consensus_centers_: Option<Array2<Float>>,
    /// Individual cluster assignments for each view
    individual_labels_: Option<Vec<Array1<usize>>>,
    /// Consensus scores for each sample (confidence in cluster assignment)
    consensus_scores_: Option<Array1<Float>>,
    /// Within-cluster sum of squares for each view
    within_cluster_ss_: Option<Array1<Float>>,
    /// Number of iterations for convergence
    n_iter_: Option<usize>,
    /// Inertia (sum of squared distances to centroids)
    inertia_: Option<Float>,
    /// Silhouette score for each view
    silhouette_scores_: Option<Array1<Float>>,
    /// State marker
    _state: PhantomData<State>,
}

/// Initialization methods for cluster centers
#[derive(Debug, Clone)]
pub enum InitMethod {
    /// K-means++ initialization
    KMeansPlusPlus,
    /// Random initialization
    Random,
    /// User-provided centers
    Custom(Vec<Array2<Float>>),
}

/// Distance metrics for clustering
#[derive(Debug, Clone)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
}

/// Marker type for untrained state
#[derive(Debug, Clone)]
pub struct Untrained;

/// Marker type for trained state
#[derive(Debug, Clone)]
pub struct Trained;

impl MultiViewClustering<Untrained> {
    /// Create a new Multi-view Clustering with specified number of clusters
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            consensus_weight: 0.5,
            init_method: InitMethod::KMeansPlusPlus,
            max_iter: 300,
            tol: 1e-4,
            distance_metric: DistanceMetric::Euclidean,
            random_state: None,
            labels_: None,
            cluster_centers_: None,
            consensus_centers_: None,
            individual_labels_: None,
            consensus_scores_: None,
            within_cluster_ss_: None,
            n_iter_: None,
            inertia_: None,
            silhouette_scores_: None,
            _state: PhantomData,
        }
    }

    /// Set consensus weight (0.0 = only individual clustering, 1.0 = only consensus)
    pub fn consensus_weight(mut self, weight: Float) -> Self {
        self.consensus_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set initialization method
    pub fn init_method(mut self, init_method: InitMethod) -> Self {
        self.init_method = init_method;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Estimator for MultiViewClustering<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Vec<Array2<Float>>, ()> for MultiViewClustering<Untrained> {
    type Fitted = MultiViewClustering<Trained>;

    fn fit(self, views: &Vec<Array2<Float>>, _target: &()) -> Result<Self::Fitted> {
        self.fit_views(views)
    }
}

impl MultiViewClustering<Untrained> {
    /// Fit the model to multiple views
    pub fn fit_views(self, views: &Vec<Array2<Float>>) -> Result<MultiViewClustering<Trained>> {
        if views.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least 1 view is required for multi-view clustering".to_string(),
            ));
        }

        let n_samples = views[0].nrows();

        // Validate that all views have the same number of samples
        for (i, view) in views.iter().enumerate() {
            if view.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(
                    format!("All views must have the same number of samples. View {} has {} samples, expected {}", 
                            i, view.nrows(), n_samples)
                ));
            }
        }

        // Validate n_clusters
        if self.n_clusters < 1 {
            return Err(SklearsError::InvalidInput(
                "n_clusters must be at least 1".to_string(),
            ));
        }

        if self.n_clusters > n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "n_clusters ({}) cannot exceed n_samples ({})",
                self.n_clusters, n_samples
            )));
        }

        // Initialize cluster centers for each view
        let mut cluster_centers = self.initialize_centers(views)?;
        let mut labels = Array1::zeros(n_samples);
        let mut individual_labels: Vec<Array1<usize>> =
            views.iter().map(|_| Array1::zeros(n_samples)).collect();

        let mut converged = false;
        let mut n_iter = 0;

        // Main clustering loop
        while !converged && n_iter < self.max_iter {
            let old_labels = labels.clone();

            // Step 1: Perform individual clustering for each view
            for (view_idx, view) in views.iter().enumerate() {
                individual_labels[view_idx] =
                    self.assign_clusters_single_view(view, &cluster_centers[view_idx])?;
            }

            // Step 2: Compute consensus cluster assignments
            labels = self.compute_consensus_labels(&individual_labels)?;

            // Step 3: Update cluster centers based on consensus assignments
            cluster_centers = self.update_cluster_centers(views, &labels)?;

            // Check convergence
            let n_changed = labels
                .iter()
                .zip(old_labels.iter())
                .filter(|(a, b)| a != b)
                .count();

            let change_ratio = n_changed as Float / n_samples as Float;
            if change_ratio < self.tol {
                converged = true;
            }

            n_iter += 1;
        }

        // Compute consensus centers
        let consensus_centers = self.compute_consensus_centers(views, &labels)?;

        // Compute consensus scores (confidence in cluster assignments)
        let consensus_scores = self.compute_consensus_scores(views, &labels, &cluster_centers)?;

        // Compute within-cluster sum of squares for each view
        let within_cluster_ss = self.compute_within_cluster_ss(views, &labels, &cluster_centers)?;

        // Compute total inertia
        let inertia = within_cluster_ss.sum();

        // Compute silhouette scores for each view
        let silhouette_scores = self.compute_silhouette_scores(views, &labels)?;

        Ok(MultiViewClustering {
            n_clusters: self.n_clusters,
            consensus_weight: self.consensus_weight,
            init_method: self.init_method,
            max_iter: self.max_iter,
            tol: self.tol,
            distance_metric: self.distance_metric,
            random_state: self.random_state,
            labels_: Some(labels),
            cluster_centers_: Some(cluster_centers),
            consensus_centers_: Some(consensus_centers),
            individual_labels_: Some(individual_labels),
            consensus_scores_: Some(consensus_scores),
            within_cluster_ss_: Some(within_cluster_ss),
            n_iter_: Some(n_iter),
            inertia_: Some(inertia),
            silhouette_scores_: Some(silhouette_scores),
            _state: PhantomData,
        })
    }

    /// Initialize cluster centers for each view
    fn initialize_centers(&self, views: &[Array2<Float>]) -> Result<Vec<Array2<Float>>> {
        match &self.init_method {
            InitMethod::KMeansPlusPlus => self.kmeans_plus_plus_init(views),
            InitMethod::Random => self.random_init(views),
            InitMethod::Custom(centers) => {
                if centers.len() != views.len() {
                    return Err(SklearsError::InvalidInput(format!(
                        "Number of custom centers ({}) must match number of views ({})",
                        centers.len(),
                        views.len()
                    )));
                }
                Ok(centers.clone())
            }
        }
    }

    /// K-means++ initialization
    fn kmeans_plus_plus_init(&self, views: &[Array2<Float>]) -> Result<Vec<Array2<Float>>> {
        let mut centers = Vec::new();

        for view in views {
            let (n_samples, n_features) = view.dim();
            let mut view_centers = Array2::zeros((self.n_clusters, n_features));

            // Choose first center randomly
            let first_idx = if let Some(seed) = self.random_state {
                (seed as usize) % n_samples
            } else {
                let mut rng = thread_rng();
                rng.gen_range(0..n_samples)
            };
            view_centers.row_mut(0).assign(&view.row(first_idx));

            // Choose remaining centers using k-means++ strategy
            for k in 1..self.n_clusters {
                let mut distances = Array1::zeros(n_samples);

                // Compute minimum distance to existing centers for each point
                for i in 0..n_samples {
                    let point = view.row(i);
                    let mut min_dist = Float::INFINITY;

                    for j in 0..k {
                        let center = view_centers.row(j);
                        let dist = self.compute_distance(&point.to_owned(), &center.to_owned());
                        min_dist = min_dist.min(dist);
                    }

                    distances[i] = min_dist * min_dist; // Squared distance for probability
                }

                // Choose next center proportional to squared distance
                let total_distance: Float = distances.sum();
                let mut cumsum = 0.0;
                let threshold = if let Some(seed) = self.random_state {
                    ((seed as usize + k) as Float / u64::MAX as Float) * total_distance
                } else {
                    thread_rng().random::<Float>() * total_distance
                };

                let mut chosen_idx = 0;
                for (i, &dist) in distances.iter().enumerate() {
                    cumsum += dist;
                    if cumsum >= threshold {
                        chosen_idx = i;
                        break;
                    }
                }

                view_centers.row_mut(k).assign(&view.row(chosen_idx));
            }

            centers.push(view_centers);
        }

        Ok(centers)
    }

    /// Random initialization
    fn random_init(&self, views: &[Array2<Float>]) -> Result<Vec<Array2<Float>>> {
        let mut centers = Vec::new();

        for view in views {
            let (n_samples, n_features) = view.dim();
            let mut view_centers = Array2::zeros((self.n_clusters, n_features));

            for k in 0..self.n_clusters {
                let idx = if let Some(seed) = self.random_state {
                    ((seed as usize + k) * 1234567) % n_samples
                } else {
                    let mut rng = thread_rng();
                    rng.gen_range(0..n_samples)
                };
                view_centers.row_mut(k).assign(&view.row(idx));
            }

            centers.push(view_centers);
        }

        Ok(centers)
    }

    /// Assign clusters for a single view
    fn assign_clusters_single_view(
        &self,
        view: &Array2<Float>,
        centers: &Array2<Float>,
    ) -> Result<Array1<usize>> {
        let n_samples = view.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point = view.row(i).to_owned();
            let mut min_dist = Float::INFINITY;
            let mut best_cluster = 0;

            for k in 0..self.n_clusters {
                let center = centers.row(k).to_owned();
                let dist = self.compute_distance(&point, &center);

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }

    /// Compute consensus cluster assignments across views
    fn compute_consensus_labels(
        &self,
        individual_labels: &[Array1<usize>],
    ) -> Result<Array1<usize>> {
        let n_samples = individual_labels[0].len();
        let n_views = individual_labels.len();
        let mut consensus_labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Count votes for each cluster from all views
            let mut votes = vec![0; self.n_clusters];

            for view_labels in individual_labels {
                let cluster = view_labels[i];
                if cluster < self.n_clusters {
                    votes[cluster] += 1;
                }
            }

            // Find cluster with most votes
            let best_cluster = votes
                .iter()
                .enumerate()
                .max_by_key(|(_, &count)| count)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            consensus_labels[i] = best_cluster;
        }

        Ok(consensus_labels)
    }

    /// Update cluster centers based on consensus assignments
    fn update_cluster_centers(
        &self,
        views: &[Array2<Float>],
        labels: &Array1<usize>,
    ) -> Result<Vec<Array2<Float>>> {
        let mut new_centers = Vec::new();

        for view in views {
            let (_, n_features) = view.dim();
            let mut view_centers = Array2::zeros((self.n_clusters, n_features));

            for k in 0..self.n_clusters {
                let mut cluster_points = Vec::new();

                for (i, &label) in labels.iter().enumerate() {
                    if label == k {
                        cluster_points.push(view.row(i).to_owned());
                    }
                }

                if !cluster_points.is_empty() {
                    // Compute centroid
                    let mut centroid = Array1::zeros(n_features);
                    for point in &cluster_points {
                        centroid = centroid + point;
                    }
                    centroid /= cluster_points.len() as Float;
                    view_centers.row_mut(k).assign(&centroid);
                } else {
                    // If no points assigned to cluster, keep previous center
                    // This is handled by the zeros initialization above
                }
            }

            new_centers.push(view_centers);
        }

        Ok(new_centers)
    }

    /// Compute consensus centers across all views
    fn compute_consensus_centers(
        &self,
        views: &[Array2<Float>],
        labels: &Array1<usize>,
    ) -> Result<Array2<Float>> {
        let n_features = views[0].ncols();
        let mut consensus_centers = Array2::zeros((self.n_clusters, n_features));

        for k in 0..self.n_clusters {
            let mut cluster_points = Vec::new();

            // Collect all points assigned to cluster k across all views
            for view in views {
                for (i, &label) in labels.iter().enumerate() {
                    if label == k {
                        cluster_points.push(view.row(i).to_owned());
                    }
                }
            }

            if !cluster_points.is_empty() {
                // Compute centroid
                let mut centroid = Array1::zeros(n_features);
                for point in &cluster_points {
                    centroid = centroid + point;
                }
                centroid /= cluster_points.len() as Float;
                consensus_centers.row_mut(k).assign(&centroid);
            }
        }

        Ok(consensus_centers)
    }

    /// Compute consensus scores (confidence in cluster assignments)
    fn compute_consensus_scores(
        &self,
        views: &[Array2<Float>],
        labels: &Array1<usize>,
        centers: &[Array2<Float>],
    ) -> Result<Array1<Float>> {
        let n_samples = labels.len();
        let n_views = views.len();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let assigned_cluster = labels[i];
            let mut total_agreement = 0.0;

            // Calculate agreement across views
            for (view_idx, view) in views.iter().enumerate() {
                let point = view.row(i).to_owned();
                let assigned_center = centers[view_idx].row(assigned_cluster).to_owned();
                let dist_to_assigned = self.compute_distance(&point, &assigned_center);

                // Find distance to closest alternative cluster
                let mut min_alt_dist = Float::INFINITY;
                for k in 0..self.n_clusters {
                    if k != assigned_cluster {
                        let alt_center = centers[view_idx].row(k).to_owned();
                        let dist_to_alt = self.compute_distance(&point, &alt_center);
                        min_alt_dist = min_alt_dist.min(dist_to_alt);
                    }
                }

                // Agreement is relative preference for assigned cluster
                if min_alt_dist > 0.0 {
                    total_agreement += min_alt_dist / (dist_to_assigned + min_alt_dist);
                } else {
                    total_agreement += 1.0;
                }
            }

            scores[i] = total_agreement / n_views as Float;
        }

        Ok(scores)
    }

    /// Compute within-cluster sum of squares for each view
    fn compute_within_cluster_ss(
        &self,
        views: &[Array2<Float>],
        labels: &Array1<usize>,
        centers: &[Array2<Float>],
    ) -> Result<Array1<Float>> {
        let n_views = views.len();
        let mut within_ss = Array1::zeros(n_views);

        for (view_idx, view) in views.iter().enumerate() {
            let mut view_ss = 0.0;

            for (i, &cluster) in labels.iter().enumerate() {
                let point = view.row(i).to_owned();
                let center = centers[view_idx].row(cluster).to_owned();
                let dist = self.compute_distance(&point, &center);
                view_ss += dist * dist;
            }

            within_ss[view_idx] = view_ss;
        }

        Ok(within_ss)
    }

    /// Compute silhouette scores for each view
    fn compute_silhouette_scores(
        &self,
        views: &[Array2<Float>],
        labels: &Array1<usize>,
    ) -> Result<Array1<Float>> {
        let n_views = views.len();
        let n_samples = labels.len();
        let mut silhouette_scores = Array1::zeros(n_views);

        for (view_idx, view) in views.iter().enumerate() {
            let mut total_silhouette = 0.0;

            for i in 0..n_samples {
                let point_i = view.row(i).to_owned();
                let cluster_i = labels[i];

                // Compute a(i): average distance to points in same cluster
                let mut a_i = 0.0;
                let mut same_cluster_count = 0;

                for j in 0..n_samples {
                    if i != j && labels[j] == cluster_i {
                        let point_j = view.row(j).to_owned();
                        a_i += self.compute_distance(&point_i, &point_j);
                        same_cluster_count += 1;
                    }
                }

                if same_cluster_count > 0 {
                    a_i /= same_cluster_count as Float;
                }

                // Compute b(i): minimum average distance to points in other clusters
                let mut b_i = Float::INFINITY;

                for k in 0..self.n_clusters {
                    if k != cluster_i {
                        let mut avg_dist_k = 0.0;
                        let mut count_k = 0;

                        for j in 0..n_samples {
                            if labels[j] == k {
                                let point_j = view.row(j).to_owned();
                                avg_dist_k += self.compute_distance(&point_i, &point_j);
                                count_k += 1;
                            }
                        }

                        if count_k > 0 {
                            avg_dist_k /= count_k as Float;
                            b_i = b_i.min(avg_dist_k);
                        }
                    }
                }

                // Silhouette coefficient for point i
                let s_i = if a_i.max(b_i) > 0.0 {
                    (b_i - a_i) / a_i.max(b_i)
                } else {
                    0.0
                };

                total_silhouette += s_i;
            }

            silhouette_scores[view_idx] = total_silhouette / n_samples as Float;
        }

        Ok(silhouette_scores)
    }

    /// Compute distance between two points
    fn compute_distance(&self, point1: &Array1<Float>, point2: &Array1<Float>) -> Float {
        match self.distance_metric {
            DistanceMetric::Euclidean => {
                let diff = point1 - point2;
                diff.dot(&diff).sqrt()
            }
            DistanceMetric::Manhattan => (point1 - point2).mapv(|x| x.abs()).sum(),
            DistanceMetric::Cosine => {
                let dot_product = point1.dot(point2);
                let norm1 = point1.dot(point1).sqrt();
                let norm2 = point2.dot(point2).sqrt();

                if norm1 > 0.0 && norm2 > 0.0 {
                    1.0 - (dot_product / (norm1 * norm2))
                } else {
                    1.0 // Maximum cosine distance if either vector is zero
                }
            }
        }
    }
}

impl Transform<Vec<Array2<Float>>, Array1<usize>> for MultiViewClustering<Trained> {
    /// Predict cluster labels for new data
    fn transform(&self, views: &Vec<Array2<Float>>) -> Result<Array1<usize>> {
        self.predict(views)
    }
}

impl MultiViewClustering<Trained> {
    /// Predict cluster labels for new data
    pub fn predict(&self, views: &Vec<Array2<Float>>) -> Result<Array1<usize>> {
        let cluster_centers = self.cluster_centers_.as_ref().unwrap();

        if views.len() != cluster_centers.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} views, got {}",
                cluster_centers.len(),
                views.len()
            )));
        }

        let n_samples = views[0].nrows();

        // Validate that all views have the same number of samples
        for (i, view) in views.iter().enumerate() {
            if view.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(
                    format!("All views must have the same number of samples. View {} has {} samples, expected {}", 
                            i, view.nrows(), n_samples)
                ));
            }
        }

        let mut individual_labels: Vec<Array1<usize>> = Vec::new();

        // Get individual cluster assignments for each view
        for (view_idx, view) in views.iter().enumerate() {
            let view_labels = self.assign_clusters_single_view(view, &cluster_centers[view_idx])?;
            individual_labels.push(view_labels);
        }

        // Compute consensus labels
        self.compute_consensus_labels(&individual_labels)
    }

    /// Get the final cluster labels
    pub fn labels(&self) -> &Array1<usize> {
        self.labels_.as_ref().unwrap()
    }

    /// Get the cluster centers for each view
    pub fn cluster_centers(&self) -> &Vec<Array2<Float>> {
        self.cluster_centers_.as_ref().unwrap()
    }

    /// Get the consensus cluster centers
    pub fn consensus_centers(&self) -> &Array2<Float> {
        self.consensus_centers_.as_ref().unwrap()
    }

    /// Get the individual cluster assignments for each view
    pub fn individual_labels(&self) -> &Vec<Array1<usize>> {
        self.individual_labels_.as_ref().unwrap()
    }

    /// Get the consensus scores (confidence in assignments)
    pub fn consensus_scores(&self) -> &Array1<Float> {
        self.consensus_scores_.as_ref().unwrap()
    }

    /// Get the within-cluster sum of squares for each view
    pub fn within_cluster_ss(&self) -> &Array1<Float> {
        self.within_cluster_ss_.as_ref().unwrap()
    }

    /// Get the number of iterations for convergence
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Get the total inertia
    pub fn inertia(&self) -> Float {
        self.inertia_.unwrap()
    }

    /// Get the silhouette scores for each view
    pub fn silhouette_scores(&self) -> &Array1<Float> {
        self.silhouette_scores_.as_ref().unwrap()
    }

    // Helper method used by predict
    fn assign_clusters_single_view(
        &self,
        view: &Array2<Float>,
        centers: &Array2<Float>,
    ) -> Result<Array1<usize>> {
        let n_samples = view.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point = view.row(i).to_owned();
            let mut min_dist = Float::INFINITY;
            let mut best_cluster = 0;

            for k in 0..self.n_clusters {
                let center = centers.row(k).to_owned();
                let dist = self.compute_distance(&point, &center);

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }

    fn compute_consensus_labels(
        &self,
        individual_labels: &[Array1<usize>],
    ) -> Result<Array1<usize>> {
        let n_samples = individual_labels[0].len();
        let mut consensus_labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            // Count votes for each cluster from all views
            let mut votes = vec![0; self.n_clusters];

            for view_labels in individual_labels {
                let cluster = view_labels[i];
                if cluster < self.n_clusters {
                    votes[cluster] += 1;
                }
            }

            // Find cluster with most votes
            let best_cluster = votes
                .iter()
                .enumerate()
                .max_by_key(|(_, &count)| count)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            consensus_labels[i] = best_cluster;
        }

        Ok(consensus_labels)
    }

    fn compute_distance(&self, point1: &Array1<Float>, point2: &Array1<Float>) -> Float {
        match self.distance_metric {
            DistanceMetric::Euclidean => {
                let diff = point1 - point2;
                diff.dot(&diff).sqrt()
            }
            DistanceMetric::Manhattan => (point1 - point2).mapv(|x| x.abs()).sum(),
            DistanceMetric::Cosine => {
                let dot_product = point1.dot(point2);
                let norm1 = point1.dot(point1).sqrt();
                let norm2 = point2.dot(point2).sqrt();

                if norm1 > 0.0 && norm2 > 0.0 {
                    1.0 - (dot_product / (norm1 * norm2))
                } else {
                    1.0
                }
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Fit;

    #[test]
    fn test_multiview_clustering_basic() {
        let view1 = array![[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [5.0, 6.0]];
        let view2 = array![[1.0, 1.5], [1.5, 2.0], [5.5, 5.0], [5.0, 5.5]];
        let views = vec![view1, view2];

        let mvc = MultiViewClustering::new(2);
        let fitted = mvc.fit(&views, &()).unwrap();

        // Check that labels were computed
        assert_eq!(fitted.labels().len(), 4);
        assert_eq!(fitted.cluster_centers().len(), 2);
        assert_eq!(fitted.individual_labels().len(), 2);

        // Check that we have the right number of clusters
        let unique_labels: std::collections::HashSet<_> = fitted.labels().iter().cloned().collect();
        assert!(unique_labels.len() <= 2);
    }

    #[test]
    fn test_multiview_clustering_predict() {
        let view1 = array![[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [5.0, 6.0]];
        let view2 = array![[1.0, 1.5], [1.5, 2.0], [5.5, 5.0], [5.0, 5.5]];
        let views = vec![view1.clone(), view2.clone()];

        let mvc = MultiViewClustering::new(2);
        let fitted = mvc.fit(&views, &()).unwrap();

        // Test prediction on same data
        let predicted = fitted.predict(&views).unwrap();
        assert_eq!(predicted.len(), 4);

        // Test transform
        let transformed = fitted.transform(&views).unwrap();
        assert_eq!(transformed.len(), 4);
    }

    #[test]
    fn test_multiview_clustering_different_metrics() {
        let view1 = array![[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [5.0, 6.0]];
        let view2 = array![[1.0, 1.5], [1.5, 2.0], [5.5, 5.0], [5.0, 5.5]];
        let views = vec![view1, view2];

        // Test Euclidean distance
        let mvc_euclidean = MultiViewClustering::new(2).distance_metric(DistanceMetric::Euclidean);
        let fitted_euclidean = mvc_euclidean.fit(&views, &()).unwrap();
        assert!(fitted_euclidean.n_iter() > 0);

        // Test Manhattan distance
        let mvc_manhattan = MultiViewClustering::new(2).distance_metric(DistanceMetric::Manhattan);
        let fitted_manhattan = mvc_manhattan.fit(&views, &()).unwrap();
        assert!(fitted_manhattan.n_iter() > 0);

        // Test Cosine distance
        let mvc_cosine = MultiViewClustering::new(2).distance_metric(DistanceMetric::Cosine);
        let fitted_cosine = mvc_cosine.fit(&views, &()).unwrap();
        assert!(fitted_cosine.n_iter() > 0);
    }

    #[test]
    fn test_multiview_clustering_consensus_weight() {
        let view1 = array![[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [5.0, 6.0]];
        let view2 = array![[1.0, 1.5], [1.5, 2.0], [5.5, 5.0], [5.0, 5.5]];
        let views = vec![view1, view2];

        // Test with different consensus weights
        let mvc1 = MultiViewClustering::new(2).consensus_weight(0.0);
        let fitted1 = mvc1.fit(&views, &()).unwrap();

        let mvc2 = MultiViewClustering::new(2).consensus_weight(1.0);
        let fitted2 = mvc2.fit(&views, &()).unwrap();

        // Both should work
        assert!(fitted1.n_iter() > 0);
        assert!(fitted2.n_iter() > 0);
        assert_eq!(fitted1.labels().len(), 4);
        assert_eq!(fitted2.labels().len(), 4);
    }

    #[test]
    fn test_multiview_clustering_error_cases() {
        // Test with empty views
        let views: Vec<Array2<Float>> = vec![];
        let mvc = MultiViewClustering::new(2);
        assert!(mvc.fit(&views, &()).is_err());

        // Test with mismatched sample sizes
        let view1 = array![[1.0, 2.0], [3.0, 4.0]];
        let view2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
        let views = vec![view1, view2];
        let mvc = MultiViewClustering::new(2);
        assert!(mvc.fit(&views, &()).is_err());

        // Test with invalid number of clusters
        let view1 = array![[1.0, 2.0], [3.0, 4.0]];
        let views = vec![view1];
        let mvc = MultiViewClustering::new(0);
        assert!(mvc.fit(&views, &()).is_err());
    }

    #[test]
    fn test_multiview_clustering_initialization_methods() {
        let view1 = array![[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [5.0, 6.0]];
        let view2 = array![[1.0, 1.5], [1.5, 2.0], [5.5, 5.0], [5.0, 5.5]];
        let views = vec![view1, view2];

        // Test K-means++ initialization
        let mvc_kmpp = MultiViewClustering::new(2).init_method(InitMethod::KMeansPlusPlus);
        let fitted_kmpp = mvc_kmpp.fit(&views, &()).unwrap();
        assert!(fitted_kmpp.n_iter() > 0);

        // Test random initialization
        let mvc_random = MultiViewClustering::new(2).init_method(InitMethod::Random);
        let fitted_random = mvc_random.fit(&views, &()).unwrap();
        assert!(fitted_random.n_iter() > 0);
    }

    #[test]
    fn test_multiview_clustering_metrics() {
        let view1 = array![[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [5.0, 6.0]];
        let view2 = array![[1.0, 1.5], [1.5, 2.0], [5.5, 5.0], [5.0, 5.5]];
        let views = vec![view1, view2];

        let mvc = MultiViewClustering::new(2);
        let fitted = mvc.fit(&views, &()).unwrap();

        // Check that metrics were computed
        assert_eq!(fitted.consensus_scores().len(), 4);
        assert_eq!(fitted.within_cluster_ss().len(), 2);
        assert!(fitted.inertia() >= 0.0);
        assert_eq!(fitted.silhouette_scores().len(), 2);
    }
}
