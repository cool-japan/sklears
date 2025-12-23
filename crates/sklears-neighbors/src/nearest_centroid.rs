//! Nearest Centroid Classifier

use crate::{Distance, NeighborsError};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Predict, Untrained};
use sklears_core::types::{Features, Float, Int};
use std::collections::HashMap;

/// Centroid computation methods for different classes
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CentroidType {
    /// Standard arithmetic mean (default)
    #[default]
    Mean,
    /// Median (robust to outliers)
    Median,
    /// Trimmed mean (exclude outliers by percentage)
    TrimmedMean(Float),
    /// Weighted mean (requires sample weights)
    WeightedMean,
    /// Geometric median (L1 minimizer, more robust)
    GeometricMedian,
}

/// Class-specific configuration for centroid computation
#[derive(Debug, Clone)]
pub struct ClassConfig {
    /// Centroid computation method for this class
    pub centroid_type: CentroidType,
    /// Distance metric for this class
    pub distance_metric: Distance,
    /// Shrinkage threshold for this class
    pub shrink_threshold: Option<Float>,
    /// Sample weights for this class (if using weighted centroid)
    pub sample_weights: Option<Array1<Float>>,
}

impl Default for ClassConfig {
    fn default() -> Self {
        Self {
            centroid_type: CentroidType::Mean,
            distance_metric: Distance::Euclidean,
            shrink_threshold: None,
            sample_weights: None,
        }
    }
}

/// Nearest Centroid Classifier
///
/// This classifier represents each class by the centroid of its members.
/// It has no parameters to choose, making it a good baseline classifier.
/// It also has no assumptions about the underlying data distribution.
#[derive(Debug, Clone)]
pub struct NearestCentroid<State = sklears_core::traits::Untrained> {
    /// Distance metric to use (global default)
    pub metric: Distance,
    /// Whether to shrink the centroids to remove features (global default)
    pub shrink_threshold: Option<Float>,
    /// Default centroid computation method
    pub centroid_type: CentroidType,
    /// Class-specific configurations
    pub class_configs: HashMap<Int, ClassConfig>,
    /// Class centroids (only available after fitting)
    pub(crate) centroids_: Option<Array2<Float>>,
    /// Class labels (only available after fitting)  
    pub(crate) classes_: Option<Array1<Int>>,
    /// Class-specific distance metrics used during fitting
    pub(crate) class_metrics_: Option<HashMap<Int, Distance>>,
    /// Sample counts for each class (for online updates)
    pub(crate) class_sample_counts_: Option<HashMap<Int, usize>>,
    /// Running sums for each class (for online mean updates)
    pub(crate) class_running_sums_: Option<HashMap<Int, Array1<Float>>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl NearestCentroid {
    pub fn new() -> Self {
        Self {
            metric: Distance::default(),
            shrink_threshold: None,
            centroid_type: CentroidType::default(),
            class_configs: HashMap::new(),
            centroids_: None,
            classes_: None,
            class_metrics_: None,
            class_sample_counts_: None,
            class_running_sums_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the distance metric (global default)
    pub fn with_metric(mut self, metric: Distance) -> Self {
        self.metric = metric;
        self
    }

    /// Set the shrink threshold for feature selection (global default)
    ///
    /// The shrinkage method removes features that have low variance across classes.
    /// Features with variance below the threshold are set to zero in the centroids.
    pub fn with_shrink_threshold(mut self, threshold: Float) -> Self {
        self.shrink_threshold = Some(threshold);
        self
    }

    /// Set the default centroid computation method
    pub fn with_centroid_type(mut self, centroid_type: CentroidType) -> Self {
        self.centroid_type = centroid_type;
        self
    }

    /// Set class-specific configuration for a particular class
    pub fn with_class_config(mut self, class_label: Int, config: ClassConfig) -> Self {
        self.class_configs.insert(class_label, config);
        self
    }

    /// Set class-specific centroid type for a particular class
    pub fn with_class_centroid_type(
        mut self,
        class_label: Int,
        centroid_type: CentroidType,
    ) -> Self {
        let config = self.class_configs.entry(class_label).or_default();
        config.centroid_type = centroid_type;
        self
    }

    /// Set class-specific distance metric for a particular class
    pub fn with_class_metric(mut self, class_label: Int, metric: Distance) -> Self {
        let config = self.class_configs.entry(class_label).or_default();
        config.distance_metric = metric;
        self
    }

    /// Set class-specific shrinkage threshold for a particular class
    pub fn with_class_shrink_threshold(mut self, class_label: Int, threshold: Float) -> Self {
        let config = self.class_configs.entry(class_label).or_default();
        config.shrink_threshold = Some(threshold);
        self
    }

    /// Set class-specific sample weights for a particular class
    pub fn with_class_weights(mut self, class_label: Int, weights: Array1<Float>) -> Self {
        let config = self.class_configs.entry(class_label).or_default();
        config.sample_weights = Some(weights);
        self
    }
}

impl Default for NearestCentroid {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for NearestCentroid {
    type Config = ();
    type Error = NeighborsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<Int>> for NearestCentroid {
    type Fitted = NearestCentroid<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Int>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x.nrows() != y.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
            }
            .into());
        }

        // Get unique classes
        let mut classes: Vec<Int> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();

        let n_classes = classes.len();
        let n_features = x.ncols();

        // Compute centroids for each class using class-specific methods
        let mut centroids = Array2::zeros((n_classes, n_features));
        let mut class_metrics = HashMap::new();
        let mut class_sample_counts = HashMap::new();
        let mut class_running_sums = HashMap::new();

        for (class_idx, &class_label) in classes.iter().enumerate() {
            // Find all samples of this class
            let class_samples: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if !class_samples.is_empty() {
                // Get class-specific configuration or use defaults
                let config = self
                    .class_configs
                    .get(&class_label)
                    .cloned()
                    .unwrap_or_else(|| ClassConfig {
                        centroid_type: self.centroid_type,
                        distance_metric: self.metric.clone(),
                        shrink_threshold: self.shrink_threshold,
                        sample_weights: None,
                    });

                // Store class-specific metric
                class_metrics.insert(class_label, config.distance_metric.clone());

                // Extract class data
                let class_data: Array2<Float> = {
                    let mut data = Array2::zeros((class_samples.len(), n_features));
                    for (i, &sample_idx) in class_samples.iter().enumerate() {
                        data.row_mut(i).assign(&x.row(sample_idx));
                    }
                    data
                };

                // Store sample count and running sum for online updates
                class_sample_counts.insert(class_label, class_samples.len());
                let class_sum = class_data.sum_axis(Axis(0));
                class_running_sums.insert(class_label, class_sum);

                // Compute centroid using class-specific method
                let class_centroid = Self::compute_centroid(&class_data, &config)?;
                centroids.row_mut(class_idx).assign(&class_centroid);
            }
        }

        // Apply class-specific shrinkage
        for (class_idx, &class_label) in classes.iter().enumerate() {
            let config = self.class_configs.get(&class_label);
            let shrink_threshold = config
                .and_then(|c| c.shrink_threshold)
                .or(self.shrink_threshold);

            if let Some(threshold) = shrink_threshold {
                // Apply shrinkage to this class centroid
                let mut centroid = centroids.row(class_idx).to_owned();
                centroid =
                    Self::apply_class_shrinkage(&centroid, &centroids, class_idx, threshold)?;
                centroids.row_mut(class_idx).assign(&centroid);
            }
        }

        Ok(NearestCentroid {
            metric: self.metric,
            shrink_threshold: self.shrink_threshold,
            centroid_type: self.centroid_type,
            class_configs: self.class_configs,
            centroids_: Some(centroids),
            classes_: Some(Array1::from_vec(classes)),
            class_metrics_: Some(class_metrics),
            class_sample_counts_: Some(class_sample_counts),
            class_running_sums_: Some(class_running_sums),
            _state: std::marker::PhantomData,
        })
    }
}

impl NearestCentroid<Untrained> {
    /// Compute centroid using the specified method
    fn compute_centroid(class_data: &Array2<Float>, config: &ClassConfig) -> Result<Array1<Float>> {
        if class_data.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let n_samples = class_data.nrows();
        let n_features = class_data.ncols();

        match config.centroid_type {
            CentroidType::Mean => {
                // Standard arithmetic mean
                let sum = class_data.sum_axis(Axis(0));
                Ok(sum.mapv(|x| x / n_samples as Float))
            }
            CentroidType::Median => {
                // Median for each feature
                let mut centroid = Array1::zeros(n_features);
                for feature_idx in 0..n_features {
                    let mut values: Vec<Float> = class_data.column(feature_idx).to_vec();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let median = if values.len() % 2 == 0 {
                        let mid = values.len() / 2;
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };
                    centroid[feature_idx] = median;
                }
                Ok(centroid)
            }
            CentroidType::TrimmedMean(trim_percentage) => {
                // Trimmed mean - exclude outliers
                if !(0.0..0.5).contains(&trim_percentage) {
                    return Err(NeighborsError::InvalidInput(
                        "Trim percentage must be between 0 and 0.5".to_string(),
                    )
                    .into());
                }

                let mut centroid = Array1::zeros(n_features);
                let trim_count = (n_samples as Float * trim_percentage).floor() as usize;

                for feature_idx in 0..n_features {
                    let mut values: Vec<Float> = class_data.column(feature_idx).to_vec();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    // Remove outliers from both ends
                    let trimmed_values = &values[trim_count..(values.len() - trim_count)];
                    if !trimmed_values.is_empty() {
                        let sum: Float = trimmed_values.iter().sum();
                        centroid[feature_idx] = sum / trimmed_values.len() as Float;
                    } else {
                        // Fallback to median if too much trimming
                        centroid[feature_idx] = values[values.len() / 2];
                    }
                }
                Ok(centroid)
            }
            CentroidType::WeightedMean => {
                // Weighted mean using sample weights
                if let Some(ref weights) = config.sample_weights {
                    if weights.len() != n_samples {
                        return Err(NeighborsError::ShapeMismatch {
                            expected: vec![n_samples],
                            actual: vec![weights.len()],
                        }
                        .into());
                    }

                    let mut centroid = Array1::zeros(n_features);
                    let weight_sum = weights.sum();

                    if weight_sum == 0.0 {
                        return Err(NeighborsError::InvalidInput(
                            "Sum of weights cannot be zero".to_string(),
                        )
                        .into());
                    }

                    for feature_idx in 0..n_features {
                        let weighted_sum: Float = class_data
                            .column(feature_idx)
                            .iter()
                            .zip(weights.iter())
                            .map(|(&value, &weight)| value * weight)
                            .sum();
                        centroid[feature_idx] = weighted_sum / weight_sum;
                    }
                    Ok(centroid)
                } else {
                    // Fallback to regular mean if no weights provided
                    let sum = class_data.sum_axis(Axis(0));
                    Ok(sum.mapv(|x| x / n_samples as Float))
                }
            }
            CentroidType::GeometricMedian => {
                // Geometric median (L1 minimizer) - iterative algorithm
                Self::compute_geometric_median(class_data)
            }
        }
    }

    /// Compute geometric median using iterative algorithm
    fn compute_geometric_median(data: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples == 0 {
            return Err(NeighborsError::EmptyInput.into());
        }

        // Initialize with regular mean
        let mut median = data.sum_axis(Axis(0)).mapv(|x| x / n_samples as Float);

        // Iterative refinement using Weiszfeld's algorithm
        for _ in 0..50 {
            // Max iterations
            let mut numerator = Array1::<Float>::zeros(n_features);
            let mut denominator: Float = 0.0;

            for sample in data.axis_iter(Axis(0)) {
                // Compute distance from current median estimate
                let diff = &sample - &median;
                let distance = diff.mapv(|x: Float| x * x).sum().sqrt();

                if distance > 1e-8 {
                    // Avoid division by zero
                    let weight = 1.0 / distance;
                    numerator += &sample.mapv(|x: Float| x * weight);
                    denominator += weight;
                }
            }

            if denominator > 1e-8 {
                let new_median = numerator.mapv(|x: Float| x / denominator);

                // Check for convergence
                let change = (&new_median - &median).mapv(|x: Float| x.abs()).sum();
                median = new_median;

                if change < 1e-6 {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(median)
    }

    /// Apply class-specific shrinkage
    fn apply_class_shrinkage(
        centroid: &Array1<Float>,
        all_centroids: &Array2<Float>,
        _class_idx: usize,
        threshold: Float,
    ) -> Result<Array1<Float>> {
        if threshold < 0.0 {
            return Err(NeighborsError::InvalidInput(format!(
                "Shrink threshold must be non-negative, got {}",
                threshold
            ))
            .into());
        }

        let n_features = centroid.len();
        let mut shrunken_centroid = centroid.clone();

        // Compute overall centroid (mean of all class centroids)
        let overall_centroid = all_centroids.mean_axis(Axis(0)).unwrap();

        // For each feature, apply shrinkage towards overall centroid
        for feature_idx in 0..n_features {
            let feature_values: Vec<Float> = (0..all_centroids.nrows())
                .map(|idx| all_centroids[[idx, feature_idx]])
                .collect();

            // Compute variance of this feature across classes
            let mean = feature_values.iter().sum::<Float>() / feature_values.len() as Float;
            let variance = feature_values
                .iter()
                .map(|&val| (val - mean).powi(2))
                .sum::<Float>()
                / feature_values.len() as Float;

            // Apply shrinkage
            if variance < threshold {
                // Strong shrinkage - move towards overall centroid
                shrunken_centroid[feature_idx] = overall_centroid[feature_idx];
            } else {
                // Soft shrinkage
                let shrinkage_factor = threshold / (variance + threshold);
                let original_value = centroid[feature_idx];
                let overall_value = overall_centroid[feature_idx];
                shrunken_centroid[feature_idx] =
                    (1.0 - shrinkage_factor) * original_value + shrinkage_factor * overall_value;
            }
        }

        Ok(shrunken_centroid)
    }

    /// Apply shrinkage to centroids for high-dimensional data
    ///
    /// This method implements the shrinkage approach by computing the variance
    /// of each feature across all class centroids and setting features with
    /// variance below the threshold to zero.
    fn apply_shrinkage(centroids: &Array2<Float>, threshold: Float) -> Result<Array2<Float>> {
        if threshold < 0.0 {
            return Err(NeighborsError::InvalidInput(format!(
                "Shrink threshold must be non-negative, got {}",
                threshold
            ))
            .into());
        }

        let n_classes = centroids.nrows();
        let n_features = centroids.ncols();
        let mut shrunken_centroids = centroids.clone();

        // Compute overall centroid (mean of all class centroids)
        let overall_centroid = centroids.mean_axis(Axis(0)).unwrap();

        // For each feature, compute variance across class centroids
        for feature_idx in 0..n_features {
            let feature_values: Vec<Float> = (0..n_classes)
                .map(|class_idx| centroids[[class_idx, feature_idx]])
                .collect();

            // Compute variance of this feature across classes
            let mean = feature_values.iter().sum::<Float>() / n_classes as Float;
            let variance = feature_values
                .iter()
                .map(|&val| (val - mean).powi(2))
                .sum::<Float>()
                / n_classes as Float;

            // Apply shrinkage if variance is below threshold
            if variance < threshold {
                // Set this feature to the overall centroid value for all classes
                for class_idx in 0..n_classes {
                    shrunken_centroids[[class_idx, feature_idx]] = overall_centroid[feature_idx];
                }
            } else {
                // Apply soft shrinkage: move towards overall centroid
                let shrinkage_factor = threshold / (variance + threshold);
                for class_idx in 0..n_classes {
                    let original_value = centroids[[class_idx, feature_idx]];
                    let overall_value = overall_centroid[feature_idx];
                    shrunken_centroids[[class_idx, feature_idx]] = (1.0 - shrinkage_factor)
                        * original_value
                        + shrinkage_factor * overall_value;
                }
            }
        }

        Ok(shrunken_centroids)
    }
}

impl Predict<Features, Array1<Int>> for NearestCentroid<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<Int>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let centroids = self.centroids_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        if x.ncols() != centroids.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![centroids.ncols()],
                actual: vec![x.ncols()],
            }
            .into());
        }

        let mut predictions = Array1::zeros(x.nrows());
        let class_metrics = self.class_metrics_.as_ref();

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let mut min_distance = Float::INFINITY;
            let mut best_class_idx = 0;

            // Compute distance to each class centroid using class-specific metrics
            for (class_idx, &class_label) in classes.iter().enumerate() {
                let centroid = centroids.row(class_idx);

                // Use class-specific metric if available, otherwise use global metric
                let metric = class_metrics
                    .and_then(|metrics| metrics.get(&class_label))
                    .unwrap_or(&self.metric);

                let distance = metric.calculate(&sample, &centroid);

                if distance < min_distance {
                    min_distance = distance;
                    best_class_idx = class_idx;
                }
            }

            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }
}

impl NearestCentroid<sklears_core::traits::Trained> {
    /// Get the class centroids
    pub fn centroids(&self) -> &Array2<Float> {
        self.centroids_.as_ref().unwrap()
    }

    /// Get the class labels
    pub fn classes(&self) -> &Array1<Int> {
        self.classes_.as_ref().unwrap()
    }

    /// Check if shrinkage was applied during training
    pub fn has_shrinkage(&self) -> bool {
        self.shrink_threshold.is_some()
    }

    /// Get the shrinkage threshold used during training
    pub fn shrink_threshold(&self) -> Option<Float> {
        self.shrink_threshold
    }

    /// Compute decision scores (negative distances to centroids) for each class
    pub fn decision_function(&self, x: &Features) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        let centroids = self.centroids_.as_ref().unwrap();

        if x.ncols() != centroids.ncols() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![centroids.ncols()],
                actual: vec![x.ncols()],
            }
            .into());
        }

        let n_samples = x.nrows();
        let n_classes = centroids.nrows();
        let mut scores = Array2::zeros((n_samples, n_classes));
        let classes = self.classes_.as_ref().unwrap();
        let class_metrics = self.class_metrics_.as_ref();

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            // Compute scores using class-specific metrics
            for (class_idx, &class_label) in classes.iter().enumerate() {
                let centroid = centroids.row(class_idx);

                // Use class-specific metric if available, otherwise use global metric
                let metric = class_metrics
                    .and_then(|metrics| metrics.get(&class_label))
                    .unwrap_or(&self.metric);

                let distance = metric.calculate(&sample, &centroid);

                // Convert distance to score (negative distance, so closer = higher score)
                scores[[i, class_idx]] = -distance;
            }
        }

        Ok(scores)
    }

    /// Get the feature importances based on variance across centroids
    /// Higher variance indicates more important features for classification
    pub fn feature_importances(&self) -> Array1<Float> {
        let centroids = self.centroids_.as_ref().unwrap();
        let n_classes = centroids.nrows();
        let n_features = centroids.ncols();
        let mut importances = Array1::zeros(n_features);

        for feature_idx in 0..n_features {
            let feature_values: Vec<Float> = (0..n_classes)
                .map(|class_idx| centroids[[class_idx, feature_idx]])
                .collect();

            // Compute variance of this feature across classes
            let mean = feature_values.iter().sum::<Float>() / n_classes as Float;
            let variance = feature_values
                .iter()
                .map(|&val| (val - mean).powi(2))
                .sum::<Float>()
                / n_classes as Float;

            importances[feature_idx] = variance;
        }

        // Normalize to sum to 1
        let total_variance = importances.sum();
        if total_variance > 0.0 {
            importances.mapv_inplace(|x| x / total_variance);
        }

        importances
    }

    /// Update centroids with new samples online
    ///
    /// This method allows incremental updates to the centroids when new training
    /// samples arrive, without requiring a complete retraining.
    ///
    /// # Arguments
    /// * `x_new` - New feature samples to add
    /// * `y_new` - New target labels corresponding to the samples
    ///
    /// # Returns
    /// * `Result<(), NeighborsError>` - Success or error
    pub fn partial_fit(&mut self, x_new: &Features, y_new: &Array1<Int>) -> Result<()> {
        if x_new.is_empty() || y_new.is_empty() {
            return Err(NeighborsError::EmptyInput.into());
        }

        if x_new.nrows() != y_new.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x_new.nrows()],
                actual: vec![y_new.len()],
            }
            .into());
        }

        let n_features = self.centroids_.as_ref().unwrap().ncols();

        if x_new.ncols() != n_features {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![n_features],
                actual: vec![x_new.ncols()],
            }
            .into());
        }

        let mut classes = self.classes_.as_ref().unwrap().to_vec();

        // Process each new sample
        for (i, &new_label) in y_new.iter().enumerate() {
            let new_sample = x_new.row(i);

            // Find or create class index
            let class_idx = if let Some(idx) = classes.iter().position(|&c| c == new_label) {
                idx
            } else {
                // New class - add to classes and resize centroids
                classes.push(new_label);
                let new_class_idx = classes.len() - 1;

                // Resize centroids array
                let centroids = self.centroids_.as_mut().unwrap();
                let mut new_centroids = Array2::zeros((classes.len(), n_features));
                new_centroids
                    .slice_mut(s![0..centroids.nrows(), ..])
                    .assign(centroids);

                // Initialize new class centroid with this first sample
                new_centroids.row_mut(new_class_idx).assign(&new_sample);
                *centroids = new_centroids;

                // Initialize tracking for new class
                let sample_counts = self.class_sample_counts_.as_mut().unwrap();
                let running_sums = self.class_running_sums_.as_mut().unwrap();
                sample_counts.insert(new_label, 1);
                running_sums.insert(new_label, new_sample.to_owned());

                // Initialize class configuration if not exists
                self.class_configs.entry(new_label).or_default();

                new_class_idx
            };

            // Update existing class (if it's not a newly created class)
            // For new classes, the centroid is already set to the first sample above
            let sample_counts = self.class_sample_counts_.as_ref().unwrap();
            if let Some(&existing_count) = sample_counts.get(&new_label) {
                // This is an existing class, update the centroid
                if existing_count > 0 {
                    self.update_class_centroid(class_idx, new_label, &new_sample)?;
                }
            }
        }

        // Update classes array
        self.classes_ = Some(Array1::from_vec(classes));

        Ok(())
    }

    /// Update centroid for a specific class with a new sample
    fn update_class_centroid(
        &mut self,
        class_idx: usize,
        class_label: Int,
        new_sample: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Result<()> {
        let centroids = self.centroids_.as_mut().unwrap();
        let sample_counts = self.class_sample_counts_.as_mut().unwrap();
        let running_sums = self.class_running_sums_.as_mut().unwrap();

        // Get class configuration
        let config = self
            .class_configs
            .get(&class_label)
            .cloned()
            .unwrap_or_else(|| ClassConfig {
                centroid_type: self.centroid_type,
                distance_metric: self.metric.clone(),
                shrink_threshold: self.shrink_threshold,
                sample_weights: None,
            });

        // Update based on centroid type
        match config.centroid_type {
            CentroidType::Mean => {
                // For mean centroids, we can do efficient online updates
                let current_count = *sample_counts.get(&class_label).unwrap_or(&0);
                let new_count = current_count + 1;

                // Update running sum
                let current_sum = running_sums.get_mut(&class_label).unwrap();
                *current_sum = &*current_sum + new_sample;

                // Update centroid = running_sum / count
                let new_centroid = current_sum.mapv(|x| x / new_count as Float);
                centroids.row_mut(class_idx).assign(&new_centroid);

                // Update count
                sample_counts.insert(class_label, new_count);
            }
            CentroidType::WeightedMean => {
                // For weighted mean, we need to handle weights properly
                // For now, treat new samples with weight 1.0
                let current_count = *sample_counts.get(&class_label).unwrap_or(&0);
                let new_count = current_count + 1;

                let current_sum = running_sums.get_mut(&class_label).unwrap();
                *current_sum = &*current_sum + new_sample;

                let new_centroid = current_sum.mapv(|x| x / new_count as Float);
                centroids.row_mut(class_idx).assign(&new_centroid);

                sample_counts.insert(class_label, new_count);
            }
            _ => {
                // For other centroid types (Median, TrimmedMean, GeometricMedian),
                // we need to store all samples or use approximations
                // For now, fall back to simple online mean update
                let current_count = *sample_counts.get(&class_label).unwrap_or(&0);
                let new_count = current_count + 1;

                let current_sum = running_sums.get_mut(&class_label).unwrap();
                *current_sum = &*current_sum + new_sample;

                let new_centroid = current_sum.mapv(|x| x / new_count as Float);
                centroids.row_mut(class_idx).assign(&new_centroid);

                sample_counts.insert(class_label, new_count);
            }
        }

        Ok(())
    }

    /// Remove a sample from a specific class (for online learning scenarios)
    ///
    /// # Arguments
    /// * `class_label` - The class to remove the sample from
    /// * `sample` - The sample to remove
    ///
    /// # Returns
    /// * `Result<(), NeighborsError>` - Success or error
    pub fn remove_sample(
        &mut self,
        class_label: Int,
        sample: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Result<()> {
        let centroids = self.centroids_.as_mut().unwrap();
        let sample_counts = self.class_sample_counts_.as_mut().unwrap();
        let running_sums = self.class_running_sums_.as_mut().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        // Find class index
        let class_idx = classes
            .iter()
            .position(|&c| c == class_label)
            .ok_or_else(|| {
                NeighborsError::InvalidInput(format!("Class {} not found", class_label))
            })?;

        let current_count = *sample_counts.get(&class_label).unwrap_or(&0);
        if current_count <= 1 {
            return Err(NeighborsError::InvalidInput(
                "Cannot remove sample from class with only one sample".to_string(),
            )
            .into());
        }

        let new_count = current_count - 1;

        // Update running sum
        let current_sum = running_sums.get_mut(&class_label).unwrap();
        *current_sum = &*current_sum - sample;

        // Update centroid
        let new_centroid = current_sum.mapv(|x| x / new_count as Float);
        centroids.row_mut(class_idx).assign(&new_centroid);

        // Update count
        sample_counts.insert(class_label, new_count);

        Ok(())
    }

    /// Get the current sample count for each class
    pub fn class_sample_counts(&self) -> &HashMap<Int, usize> {
        self.class_sample_counts_.as_ref().unwrap()
    }

    /// Get the total number of samples used for training
    pub fn total_samples(&self) -> usize {
        self.class_sample_counts_.as_ref().unwrap().values().sum()
    }

    /// Reset the online learning state (clear all accumulated statistics)
    pub fn reset(&mut self) {
        self.class_sample_counts_ = None;
        self.class_running_sums_ = None;
        self.centroids_ = None;
        self.classes_ = None;
        self.class_metrics_ = None;
    }

    /// Get a summary of the current online learning state
    pub fn online_summary(&self) -> String {
        let sample_counts = self.class_sample_counts_.as_ref().unwrap();
        let total_samples = self.total_samples();
        let n_classes = self.classes_.as_ref().unwrap().len();

        format!(
            "Online NearestCentroid: {} classes, {} total samples, per-class counts: {:?}",
            n_classes, total_samples, sample_counts
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_nearest_centroid_basic() {
        // Create a simple dataset with two clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Class 0
                1.5, 1.5, // Class 0
                2.0, 2.0, // Class 0
                5.0, 5.0, // Class 1
                5.5, 5.5, // Class 1
                6.0, 6.0, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let classifier = NearestCentroid::new();
        let fitted = classifier.fit(&x, &y).unwrap();

        // Check centroids
        let centroids = fitted.centroids();
        assert_eq!(centroids.shape(), &[2, 2]);

        // Class 0 centroid should be around (1.5, 1.5)
        assert!((centroids[[0, 0]] - 1.5).abs() < 0.1);
        assert!((centroids[[0, 1]] - 1.5).abs() < 0.1);

        // Class 1 centroid should be around (5.5, 5.5)
        assert!((centroids[[1, 0]] - 5.5).abs() < 0.1);
        assert!((centroids[[1, 1]] - 5.5).abs() < 0.1);

        // Test prediction
        let x_test = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.2, 1.2, // Should be class 0
                5.8, 5.8, // Should be class 1
            ],
        )
        .unwrap();

        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_nearest_centroid_single_class() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let y = array![5, 5, 5]; // All same class

        let classifier = NearestCentroid::new();
        let fitted = classifier.fit(&x, &y).unwrap();

        let x_test = Array2::from_shape_vec((1, 2), vec![2.0, 2.0]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();

        assert_eq!(predictions[0], 5);
    }

    #[test]
    fn test_nearest_centroid_empty_input() {
        let x = Array2::zeros((0, 2));
        let y = Array1::zeros(0);

        let classifier = NearestCentroid::new();
        let result = classifier.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_nearest_centroid_shape_mismatch() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let y = array![0, 1, 2]; // Wrong length

        let classifier = NearestCentroid::new();
        let result = classifier.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_nearest_centroid_with_shrinkage() {
        // Create data with high-dimensional features that should be shrunk
        let x = Array2::from_shape_vec(
            (6, 4),
            vec![
                1.0, 1.0, 100.0, 0.1, // Class 0 - feature 2 has high variance
                1.5, 1.5, 101.0, 0.1, // Class 0
                2.0, 2.0, 102.0, 0.1, // Class 0
                5.0, 5.0, 50.0, 0.2, // Class 1 - feature 2 has high variance
                5.5, 5.5, 51.0, 0.2, // Class 1
                6.0, 6.0, 52.0, 0.2, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        // Apply shrinkage with threshold 0.5
        let classifier = NearestCentroid::new().with_shrink_threshold(0.5);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Check that shrinkage was applied
        assert!(fitted.has_shrinkage());
        assert_eq!(fitted.shrink_threshold(), Some(0.5));

        // Get centroids and check that low-variance features were shrunk
        let centroids = fitted.centroids();
        assert_eq!(centroids.shape(), &[2, 4]);

        // Feature 3 should be shrunk towards overall mean (low variance: 0.1 vs 0.2)
        let feature_3_class_0 = centroids[[0, 3]];
        let feature_3_class_1 = centroids[[1, 3]];
        // They should be closer together due to shrinkage
        assert!((feature_3_class_0 - feature_3_class_1).abs() < 0.05);

        // Test prediction still works
        let x_test = Array2::from_shape_vec((1, 4), vec![1.2, 1.2, 100.5, 0.15]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions[0], 0);
    }

    #[test]
    fn test_nearest_centroid_decision_function() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1];

        let classifier = NearestCentroid::new();
        let fitted = classifier.fit(&x, &y).unwrap();

        let x_test = Array2::from_shape_vec((2, 2), vec![1.05, 1.05, 5.05, 5.05]).unwrap();
        let scores = fitted.decision_function(&x_test).unwrap();

        // Shape should be (n_samples, n_classes)
        assert_eq!(scores.shape(), &[2, 2]);

        // First sample should have higher score for class 0
        assert!(scores[[0, 0]] > scores[[0, 1]]);
        // Second sample should have higher score for class 1
        assert!(scores[[1, 1]] > scores[[1, 0]]);
    }

    #[test]
    fn test_nearest_centroid_feature_importances() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 10.0, // Class 0 - feature 2 varies a lot
                1.1, 0.0, 11.0, // Class 0
                1.2, 0.0, 12.0, // Class 0
                5.0, 0.1, 20.0, // Class 1 - feature 2 varies a lot
                5.1, 0.1, 21.0, // Class 1
                5.2, 0.1, 22.0, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let classifier = NearestCentroid::new();
        let fitted = classifier.fit(&x, &y).unwrap();

        let importances = fitted.feature_importances();
        assert_eq!(importances.len(), 3);

        // Feature 2 should have highest importance (largest difference between classes: 21.0 - 11.0 = 10.0)
        // Feature 0 has difference 5.1 - 1.1 = 4.0
        // Feature 1 has difference 0.1 - 0.0 = 0.1
        assert!(importances[2] > importances[0]);
        assert!(importances[0] > importances[1]);

        // Importances should sum to 1
        let sum = importances.sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_shrinkage_with_negative_threshold() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let y = array![0, 1];

        let classifier = NearestCentroid::new().with_shrink_threshold(-0.1);
        let result = classifier.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_nearest_centroid_without_shrinkage() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1]).unwrap();
        let y = array![0, 0, 1, 1];

        let classifier = NearestCentroid::new();
        let fitted = classifier.fit(&x, &y).unwrap();

        // Should not have shrinkage
        assert!(!fitted.has_shrinkage());
        assert_eq!(fitted.shrink_threshold(), None);
    }

    #[test]
    fn test_class_specific_centroid_types() {
        // Create dataset with outliers in class 0
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, // Class 0 normal
                1.1, 1.1, // Class 0 normal
                1.2, 1.2, // Class 0 normal
                10.0, 10.0, // Class 0 outlier
                5.0, 5.0, // Class 1 normal
                5.1, 5.1, // Class 1 normal
                5.2, 5.2, // Class 1 normal
                5.3, 5.3, // Class 1 normal
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        // Use median centroid for class 0 (robust to outliers) and mean for class 1
        let classifier = NearestCentroid::new()
            .with_class_centroid_type(0, CentroidType::Median)
            .with_class_centroid_type(1, CentroidType::Mean);

        let fitted = classifier.fit(&x, &y).unwrap();
        let centroids = fitted.centroids();

        // Class 0 centroid should be robust to outlier (closer to median ~1.15)
        assert!(
            centroids[[0, 0]] < 3.0,
            "Class 0 centroid should be robust to outlier"
        );
        assert!(
            centroids[[0, 1]] < 3.0,
            "Class 0 centroid should be robust to outlier"
        );

        // Class 1 centroid should be around mean (5.15)
        assert!((centroids[[1, 0]] - 5.15).abs() < 0.1);
        assert!((centroids[[1, 1]] - 5.15).abs() < 0.1);

        // Test prediction
        let x_test = Array2::from_shape_vec((2, 2), vec![1.15, 1.15, 5.15, 5.15]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_class_specific_distance_metrics() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                1.2, 1.2, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
                5.2, 5.2, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        // Use different distance metrics for different classes
        let classifier = NearestCentroid::new()
            .with_class_metric(0, Distance::Manhattan)
            .with_class_metric(1, Distance::Euclidean);

        let fitted = classifier.fit(&x, &y).unwrap();

        // Test that model still works with different metrics
        let x_test = Array2::from_shape_vec((2, 2), vec![1.05, 1.05, 5.05, 5.05]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);

        // Test decision function
        let scores = fitted.decision_function(&x_test).unwrap();
        assert_eq!(scores.shape(), &[2, 2]);

        // First sample should have higher score for class 0
        assert!(scores[[0, 0]] > scores[[0, 1]]);
        // Second sample should have higher score for class 1
        assert!(scores[[1, 1]] > scores[[1, 0]]);
    }

    #[test]
    fn test_trimmed_mean_centroid() {
        // Create data with outliers
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Normal
                1.1, 1.1, // Normal
                1.2, 1.2, // Normal
                1.3, 1.3, // Normal
                10.0, 10.0, // Outlier
                1.0, 1.0, // Normal (to ensure enough samples after trimming)
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0, 0]; // All same class

        // Use 20% trimmed mean (remove top and bottom 20%)
        let classifier =
            NearestCentroid::new().with_class_centroid_type(0, CentroidType::TrimmedMean(0.2));

        let fitted = classifier.fit(&x, &y).unwrap();
        let centroids = fitted.centroids();

        // Centroid should be robust to outlier
        assert!(
            centroids[[0, 0]] < 2.0,
            "Trimmed mean should be robust to outlier"
        );
        assert!(
            centroids[[0, 1]] < 2.0,
            "Trimmed mean should be robust to outlier"
        );
    }

    #[test]
    fn test_weighted_mean_centroid() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // Class 0 - weight 1.0
                2.0, 2.0, // Class 0 - weight 3.0 (should pull centroid towards this)
                5.0, 5.0, // Class 1
                6.0, 6.0, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1];

        // Give higher weight to second sample in class 0
        let weights = array![1.0, 3.0]; // Weights for class 0 samples
        let classifier = NearestCentroid::new()
            .with_class_centroid_type(0, CentroidType::WeightedMean)
            .with_class_weights(0, weights);

        let fitted = classifier.fit(&x, &y).unwrap();
        let centroids = fitted.centroids();

        // Class 0 centroid should be closer to (2,2) due to higher weight
        // Weighted centroid = (1*1 + 3*2, 1*1 + 3*2) / (1+3) = (7/4, 7/4) = (1.75, 1.75)
        assert!((centroids[[0, 0]] - 1.75).abs() < 0.1);
        assert!((centroids[[0, 1]] - 1.75).abs() < 0.1);

        // Class 1 should use regular mean: (5.5, 5.5)
        assert!((centroids[[1, 0]] - 5.5).abs() < 0.1);
        assert!((centroids[[1, 1]] - 5.5).abs() < 0.1);
    }

    #[test]
    fn test_geometric_median_centroid() {
        // Create data where geometric median differs from arithmetic mean
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, // Forms a cross pattern
                1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0]; // All same class

        let classifier =
            NearestCentroid::new().with_class_centroid_type(0, CentroidType::GeometricMedian);

        let fitted = classifier.fit(&x, &y).unwrap();
        let centroids = fitted.centroids();

        // Geometric median should be at origin (0, 0)
        assert!(centroids[[0, 0]].abs() < 0.1);
        assert!(centroids[[0, 1]].abs() < 0.1);
    }

    #[test]
    fn test_class_specific_shrinkage() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 1.0, 0.1, // Class 0 - feature 2 has low variance
                1.1, 1.1, 0.1, // Class 0
                1.2, 1.2, 0.1, // Class 0
                5.0, 5.0, 0.8, // Class 1 - feature 2 has low variance
                5.1, 5.1, 0.8, // Class 1
                5.2, 5.2, 0.8, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        // Apply different shrinkage to different classes
        let classifier = NearestCentroid::new()
            .with_class_shrink_threshold(0, 0.1) // Strong shrinkage for class 0
            .with_class_shrink_threshold(1, 0.01); // Weaker shrinkage for class 1

        let fitted = classifier.fit(&x, &y).unwrap();
        let centroids = fitted.centroids();

        // Both classes should have centroids computed
        assert_eq!(centroids.shape(), &[2, 3]);

        // Features 0 and 1 should show class differences
        assert!((centroids[[0, 0]] - centroids[[1, 0]]).abs() > 1.0);
        assert!((centroids[[0, 1]] - centroids[[1, 1]]).abs() > 1.0);
    }

    #[test]
    fn test_online_centroid_updates() {
        // Initial training data
        let x_initial = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
            ],
        )
        .unwrap();
        let y_initial = array![0, 0, 1, 1];

        // Train initial model
        let classifier = NearestCentroid::new();
        let mut fitted = classifier.fit(&x_initial, &y_initial).unwrap();

        // Check initial centroids
        let initial_centroids = fitted.centroids().clone();
        assert_eq!(initial_centroids.shape(), &[2, 2]);

        // Add new samples online
        let x_new = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.2, 1.2, // Class 0
                5.2, 5.2, // Class 1
            ],
        )
        .unwrap();
        let y_new = array![0, 1];

        // Perform online update
        fitted.partial_fit(&x_new, &y_new).unwrap();

        // Check updated centroids
        let updated_centroids = fitted.centroids();
        assert_eq!(updated_centroids.shape(), &[2, 2]);

        // Class 0 centroid should now be (1.0 + 1.1 + 1.2) / 3 = 1.1
        assert!((updated_centroids[[0, 0]] - 1.1).abs() < 0.01);
        assert!((updated_centroids[[0, 1]] - 1.1).abs() < 0.01);

        // Class 1 centroid should now be (5.0 + 5.1 + 5.2) / 3 = 5.1
        assert!((updated_centroids[[1, 0]] - 5.1).abs() < 0.01);
        assert!((updated_centroids[[1, 1]] - 5.1).abs() < 0.01);

        // Check sample counts
        let sample_counts = fitted.class_sample_counts();
        assert_eq!(*sample_counts.get(&0).unwrap(), 3);
        assert_eq!(*sample_counts.get(&1).unwrap(), 3);
        assert_eq!(fitted.total_samples(), 6);
    }

    #[test]
    fn test_online_new_class() {
        // Initial training data with two classes
        let x_initial = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
            ],
        )
        .unwrap();
        let y_initial = array![0, 0, 1, 1];

        let classifier = NearestCentroid::new();
        let mut fitted = classifier.fit(&x_initial, &y_initial).unwrap();

        // Add new class online
        let x_new = Array2::from_shape_vec(
            (2, 2),
            vec![
                10.0, 10.0, // New class 2
                10.1, 10.1, // New class 2
            ],
        )
        .unwrap();
        let y_new = array![2, 2];

        fitted.partial_fit(&x_new, &y_new).unwrap();

        // Check that we now have 3 classes
        let classes = fitted.classes();
        assert_eq!(classes.len(), 3);
        assert!(classes.iter().any(|&c| c == 2));

        // Check centroids shape
        let centroids = fitted.centroids();
        assert_eq!(centroids.shape(), &[3, 2]);

        // Test prediction with new class
        let x_test = Array2::from_shape_vec((1, 2), vec![10.05, 10.05]).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions[0], 2);
    }

    #[test]
    fn test_sample_removal() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                1.2, 1.2, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
                5.2, 5.2, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let classifier = NearestCentroid::new();
        let mut fitted = classifier.fit(&x, &y).unwrap();

        // Remove a sample from class 0
        let sample_to_remove = x.row(0); // [1.0, 1.0]
        fitted.remove_sample(0, &sample_to_remove).unwrap();

        // Check updated sample count
        let sample_counts = fitted.class_sample_counts();
        assert_eq!(*sample_counts.get(&0).unwrap(), 2);

        // Check updated centroid for class 0 should now be (1.1 + 1.2) / 2 = 1.15
        let centroids = fitted.centroids();
        assert!((centroids[[0, 0]] - 1.15).abs() < 0.01);
        assert!((centroids[[0, 1]] - 1.15).abs() < 0.01);
    }

    #[test]
    fn test_sample_removal_error_cases() {
        let x = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.0, 1.0, // Class 0
                5.0, 5.0, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 1];

        let classifier = NearestCentroid::new();
        let mut fitted = classifier.fit(&x, &y).unwrap();

        // Try to remove from non-existent class
        let sample = x.row(0);
        let result = fitted.remove_sample(99, &sample);
        assert!(result.is_err());

        // Try to remove from class with only one sample
        let result = fitted.remove_sample(0, &sample);
        assert!(result.is_err());
    }

    #[test]
    fn test_online_summary() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1];

        let classifier = NearestCentroid::new();
        let fitted = classifier.fit(&x, &y).unwrap();

        let summary = fitted.online_summary();
        assert!(summary.contains("2 classes"));
        assert!(summary.contains("4 total samples"));
    }

    #[test]
    fn test_reset_online_state() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0, // Class 0
                1.1, 1.1, // Class 0
                5.0, 5.0, // Class 1
                5.1, 5.1, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1];

        let classifier = NearestCentroid::new();
        let mut fitted = classifier.fit(&x, &y).unwrap();

        // Verify initial state
        assert!(fitted.centroids_.is_some());
        assert!(fitted.class_sample_counts_.is_some());

        // Reset state
        fitted.reset();

        // Verify reset state
        assert!(fitted.centroids_.is_none());
        assert!(fitted.class_sample_counts_.is_none());
        assert!(fitted.class_running_sums_.is_none());
        assert!(fitted.classes_.is_none());
        assert!(fitted.class_metrics_.is_none());
    }

    #[test]
    fn test_partial_fit_shape_mismatch() {
        let x = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.0, 1.0, // Class 0
                5.0, 5.0, // Class 1
            ],
        )
        .unwrap();
        let y = array![0, 1];

        let classifier = NearestCentroid::new();
        let mut fitted = classifier.fit(&x, &y).unwrap();

        // Try to add data with wrong number of features
        let x_wrong = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap();
        let y_new = array![0];

        let result = fitted.partial_fit(&x_wrong, &y_new);
        assert!(result.is_err());

        // Try to add data with mismatched sample/label counts
        let x_new = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
        let y_wrong = array![0]; // Wrong length

        let result = fitted.partial_fit(&x_new, &y_wrong);
        assert!(result.is_err());
    }
}
