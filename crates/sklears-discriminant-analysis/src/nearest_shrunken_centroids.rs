//! Nearest Shrunken Centroids (NSC)
//!
//! This module implements Nearest Shrunken Centroids classification, also known as
//! Prediction Analysis of Microarrays (PAM). This method is particularly useful for
//! high-dimensional data with many features and few samples, commonly found in
//! genomics and other biological applications.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct NearestShrunkenCentroidsConfig {
    /// Shrinkage threshold parameter
    pub threshold: Float,
    /// Whether to standardize features
    pub standardize: bool,
    /// Prior probability type: "uniform" or "class_prior"
    pub prior_type: String,
    /// Cross-validation folds for threshold selection
    pub cv_folds: Option<usize>,
    /// Range of thresholds to test (start, end, num_points)
    pub threshold_range: Option<(Float, Float, usize)>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Minimum threshold value
    pub min_threshold: Float,
    /// Maximum threshold value
    pub max_threshold: Float,
}

impl Default for NearestShrunkenCentroidsConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            standardize: true,
            prior_type: "uniform".to_string(),
            cv_folds: None,
            threshold_range: None,
            random_state: None,
            min_threshold: 0.0,
            max_threshold: 10.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NearestShrunkenCentroids<State = Untrained> {
    config: NearestShrunkenCentroidsConfig,
    data: Option<TrainedData>,
    _state: PhantomData<State>,
}

#[derive(Debug, Clone)]
struct TrainedData {
    /// Shrunken centroids [n_classes x n_features]
    shrunken_centroids: Array2<Float>,
    /// Original centroids [n_classes x n_features]
    original_centroids: Array2<Float>,
    /// Class priors [n_classes]
    class_priors: Array1<Float>,
    /// Unique class labels
    classes: Array1<i32>,
    /// Number of features
    n_features: usize,
    /// Feature means (for standardization)
    feature_means: Array1<Float>,
    /// Feature standard deviations (for standardization)
    feature_stds: Array1<Float>,
    /// Overall centroid (global mean)
    overall_centroid: Array1<Float>,
    /// Pooled within-class standard deviations
    pooled_stds: Array1<Float>,
    /// Selected features (non-zero after shrinkage)
    selected_features: Array1<bool>,
    /// Shrinkage factors applied
    shrinkage_factors: Array1<Float>,
    /// Cross-validation results (if performed)
    cv_results: Option<Vec<(Float, Float)>>, // (threshold, accuracy)
}

impl NearestShrunkenCentroids<Untrained> {
    pub fn new() -> Self {
        Self {
            config: NearestShrunkenCentroidsConfig::default(),
            data: None,
            _state: PhantomData,
        }
    }

    pub fn with_config(config: NearestShrunkenCentroidsConfig) -> Self {
        Self {
            config,
            data: None,
            _state: PhantomData,
        }
    }

    pub fn threshold(mut self, threshold: Float) -> Self {
        self.config.threshold = threshold;
        self
    }

    pub fn standardize(mut self, standardize: bool) -> Self {
        self.config.standardize = standardize;
        self
    }

    pub fn prior_type(mut self, prior_type: &str) -> Self {
        self.config.prior_type = prior_type.to_string();
        self
    }

    pub fn cv_folds(mut self, cv_folds: Option<usize>) -> Self {
        self.config.cv_folds = cv_folds;
        self
    }

    pub fn threshold_range(mut self, start: Float, end: Float, num_points: usize) -> Self {
        self.config.threshold_range = Some((start, end, num_points));
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    pub fn min_threshold(mut self, min_threshold: Float) -> Self {
        self.config.min_threshold = min_threshold;
        self
    }

    pub fn max_threshold(mut self, max_threshold: Float) -> Self {
        self.config.max_threshold = max_threshold;
        self
    }
}

impl Default for NearestShrunkenCentroids<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

pub type TrainedNearestShrunkenCentroids = NearestShrunkenCentroids<Trained>;

impl NearestShrunkenCentroids<Trained> {
    pub fn shrunken_centroids(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().shrunken_centroids
    }

    pub fn original_centroids(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().original_centroids
    }

    pub fn class_priors(&self) -> &Array1<Float> {
        &self.data.as_ref().unwrap().class_priors
    }

    pub fn classes(&self) -> &Array1<i32> {
        &self.data.as_ref().unwrap().classes
    }

    pub fn n_features(&self) -> usize {
        self.data.as_ref().unwrap().n_features
    }

    pub fn overall_centroid(&self) -> &Array1<Float> {
        &self.data.as_ref().unwrap().overall_centroid
    }

    pub fn pooled_stds(&self) -> &Array1<Float> {
        &self.data.as_ref().unwrap().pooled_stds
    }

    pub fn selected_features(&self) -> &Array1<bool> {
        &self.data.as_ref().unwrap().selected_features
    }

    pub fn shrinkage_factors(&self) -> &Array1<Float> {
        &self.data.as_ref().unwrap().shrinkage_factors
    }

    pub fn cv_results(&self) -> Option<&Vec<(Float, Float)>> {
        self.data.as_ref().unwrap().cv_results.as_ref()
    }

    /// Get the number of selected features
    pub fn n_selected_features(&self) -> usize {
        self.selected_features().iter().filter(|&&x| x).count()
    }

    /// Get indices of selected features
    pub fn selected_feature_indices(&self) -> Vec<usize> {
        self.selected_features()
            .iter()
            .enumerate()
            .filter(|(_, &selected)| selected)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Compute feature scores based on centroid differences
    pub fn feature_scores(&self) -> Array1<Float> {
        let data = self.data.as_ref().unwrap();
        let n_features = data.n_features;
        let n_classes = data.classes.len();
        let mut scores = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut max_diff: Float = 0.0;

            // Find maximum difference between any pair of class centroids
            for i in 0..n_classes {
                for k in (i + 1)..n_classes {
                    let diff =
                        (data.shrunken_centroids[[i, j]] - data.shrunken_centroids[[k, j]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            scores[j] = max_diff;
        }

        scores
    }

    /// Get the effective threshold used (after cross-validation if performed)
    pub fn effective_threshold(&self) -> Float {
        if let Some(cv_results) = self.cv_results() {
            // Return threshold with highest accuracy
            cv_results
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(threshold, _)| *threshold)
                .unwrap_or(0.0)
        } else {
            self.config.threshold
        }
    }
}

impl Estimator<Untrained> for NearestShrunkenCentroids<Untrained> {
    type Config = NearestShrunkenCentroidsConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for NearestShrunkenCentroids<Untrained> {
    type Fitted = NearestShrunkenCentroids<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Input data and targets have different lengths".to_string(),
            ));
        }

        // Extract unique classes
        let mut unique_classes = y.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Standardize features if requested
        let (x_standardized, feature_means, feature_stds) = if self.config.standardize {
            self.standardize_features(x)
        } else {
            (
                x.clone(),
                Array1::zeros(n_features),
                Array1::ones(n_features),
            )
        };

        // Compute class centroids and priors
        let (original_centroids, class_priors) =
            self.compute_centroids_and_priors(&x_standardized, y, &classes)?;

        // Compute overall centroid
        let overall_centroid = self.compute_overall_centroid(&x_standardized);

        // Compute pooled within-class standard deviations
        let pooled_stds =
            self.compute_pooled_stds(&x_standardized, y, &classes, &original_centroids)?;

        // Perform cross-validation if requested
        let (optimal_threshold, cv_results) = if let Some(cv_folds) = self.config.cv_folds {
            self.cross_validate_threshold(&x_standardized, y, &classes, cv_folds)?
        } else {
            (self.config.threshold, None)
        };

        // Apply shrinkage with optimal threshold
        let (shrunken_centroids, shrinkage_factors) = self.apply_shrinkage(
            &original_centroids,
            &overall_centroid,
            &pooled_stds,
            &class_priors,
            optimal_threshold,
        )?;

        // Determine selected features
        let selected_features = self.determine_selected_features(&shrunken_centroids);

        let trained_data = TrainedData {
            shrunken_centroids,
            original_centroids,
            class_priors,
            classes,
            n_features,
            feature_means,
            feature_stds,
            overall_centroid,
            pooled_stds,
            selected_features,
            shrinkage_factors,
            cv_results,
        };

        Ok(NearestShrunkenCentroids {
            config: self.config,
            data: Some(trained_data),
            _state: PhantomData,
        })
    }
}

impl NearestShrunkenCentroids<Untrained> {
    /// Standardize features to have zero mean and unit variance
    fn standardize_features(
        &self,
        x: &Array2<Float>,
    ) -> (Array2<Float>, Array1<Float>, Array1<Float>) {
        let n_features = x.ncols();
        let means = x.mean_axis(Axis(0)).unwrap();
        let mut stds = Array1::zeros(n_features);

        // Compute standard deviations
        for j in 0..n_features {
            let var = x
                .column(j)
                .mapv(|val| (val - means[j]).powi(2))
                .mean()
                .unwrap();
            stds[j] = (var + 1e-8).sqrt(); // Add small epsilon to avoid division by zero
        }

        // Standardize
        let mut x_standardized = x.clone();
        for (_i, mut row) in x_standardized.axis_iter_mut(Axis(0)).enumerate() {
            for j in 0..n_features {
                row[j] = (row[j] - means[j]) / stds[j];
            }
        }

        (x_standardized, means, stds)
    }

    /// Compute class centroids and priors
    fn compute_centroids_and_priors(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let n_features = x.ncols();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        let mut centroids = Array2::zeros((n_classes, n_features));
        let mut class_counts = Array1::<Float>::zeros(n_classes);

        // Compute centroids
        for (sample_idx, &label) in y.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                let sample = x.row(sample_idx);
                for j in 0..n_features {
                    centroids[[class_idx, j]] += sample[j];
                }
                class_counts[class_idx] += 1.0;
            }
        }

        // Normalize centroids
        for (class_idx, &count) in class_counts.iter().enumerate() {
            if count > 0.0 {
                for j in 0..n_features {
                    centroids[[class_idx, j]] /= count;
                }
            }
        }

        // Compute priors
        let class_priors = match self.config.prior_type.as_str() {
            "uniform" => Array1::from_elem(n_classes, 1.0 / n_classes as Float),
            "class_prior" => class_counts.mapv(|count| count / n_samples as Float),
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown prior type: {}",
                    self.config.prior_type
                )))
            }
        };

        Ok((centroids, class_priors))
    }

    /// Compute overall centroid (global mean)
    fn compute_overall_centroid(&self, x: &Array2<Float>) -> Array1<Float> {
        x.mean_axis(Axis(0)).unwrap()
    }

    /// Compute pooled within-class standard deviations
    fn compute_pooled_stds(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        centroids: &Array2<Float>,
    ) -> Result<Array1<Float>> {
        let n_features = x.ncols();
        let n_samples = x.nrows();
        let n_classes = classes.len();

        let mut pooled_vars = Array1::zeros(n_features);

        for (sample_idx, &label) in y.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| c == label) {
                let sample = x.row(sample_idx);
                let centroid = centroids.row(class_idx);

                for j in 0..n_features {
                    let diff = sample[j] - centroid[j];
                    pooled_vars[j] += diff * diff;
                }
            }
        }

        // Normalize by degrees of freedom
        let df = (n_samples - n_classes) as Float;
        if df <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "Too few samples relative to number of classes".to_string(),
            ));
        }

        for j in 0..n_features {
            let var_with_reg: Float = pooled_vars[j] / df + 1e-8 as Float;
            pooled_vars[j] = var_with_reg.sqrt(); // Add small epsilon
        }

        Ok(pooled_vars)
    }

    /// Cross-validate threshold selection
    fn cross_validate_threshold(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        cv_folds: usize,
    ) -> Result<(Float, Option<Vec<(Float, Float)>>)> {
        let (start, end, num_points) = self.config.threshold_range.unwrap_or((
            self.config.min_threshold,
            self.config.max_threshold,
            20,
        ));

        let thresholds: Vec<Float> = (0..num_points)
            .map(|i| start + (end - start) * i as Float / (num_points - 1) as Float)
            .collect();

        let mut cv_results = Vec::new();
        let n_samples = x.nrows();
        let fold_size = n_samples / cv_folds;

        for &threshold in &thresholds {
            let mut fold_accuracies = Vec::new();

            for fold in 0..cv_folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == cv_folds - 1 {
                    n_samples
                } else {
                    (fold + 1) * fold_size
                };

                // Create train/test splits
                let mut train_indices = Vec::new();
                let mut test_indices = Vec::new();

                for i in 0..n_samples {
                    if i >= start_idx && i < end_idx {
                        test_indices.push(i);
                    } else {
                        train_indices.push(i);
                    }
                }

                if train_indices.is_empty() || test_indices.is_empty() {
                    continue;
                }

                // Extract train/test data
                let x_train = x.select(Axis(0), &train_indices);
                let y_train = y.select(Axis(0), &train_indices);
                let x_test = x.select(Axis(0), &test_indices);
                let y_test = y.select(Axis(0), &test_indices);

                // Train with current threshold
                let accuracy = self
                    .evaluate_threshold(&x_train, &y_train, &x_test, &y_test, classes, threshold)?;
                fold_accuracies.push(accuracy);
            }

            let mean_accuracy =
                fold_accuracies.iter().sum::<Float>() / fold_accuracies.len() as Float;
            cv_results.push((threshold, mean_accuracy));
        }

        // Find best threshold
        let best_threshold = cv_results
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(threshold, _)| *threshold)
            .unwrap_or(0.0);

        Ok((best_threshold, Some(cv_results)))
    }

    /// Evaluate a specific threshold
    fn evaluate_threshold(
        &self,
        x_train: &Array2<Float>,
        y_train: &Array1<i32>,
        x_test: &Array2<Float>,
        y_test: &Array1<i32>,
        classes: &Array1<i32>,
        threshold: Float,
    ) -> Result<Float> {
        // Compute training statistics
        let (original_centroids, class_priors) =
            self.compute_centroids_and_priors(x_train, y_train, classes)?;
        let overall_centroid = self.compute_overall_centroid(x_train);
        let pooled_stds =
            self.compute_pooled_stds(x_train, y_train, classes, &original_centroids)?;

        // Apply shrinkage
        let (shrunken_centroids, _) = self.apply_shrinkage(
            &original_centroids,
            &overall_centroid,
            &pooled_stds,
            &class_priors,
            threshold,
        )?;

        // Make predictions on test set
        let mut correct = 0;
        for (i, &true_label) in y_test.iter().enumerate() {
            let sample = x_test.row(i);
            let predicted_label =
                NearestShrunkenCentroids::<Untrained>::predict_single_sample_static(
                    &sample,
                    &shrunken_centroids,
                    &class_priors,
                    classes,
                )?;
            if predicted_label == true_label {
                correct += 1;
            }
        }

        Ok(correct as Float / y_test.len() as Float)
    }

    /// Apply shrinkage to centroids
    fn apply_shrinkage(
        &self,
        original_centroids: &Array2<Float>,
        overall_centroid: &Array1<Float>,
        pooled_stds: &Array1<Float>,
        class_priors: &Array1<Float>,
        threshold: Float,
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let (n_classes, n_features) = original_centroids.dim();
        let mut shrunken_centroids = original_centroids.clone();
        let mut shrinkage_factors = Array1::zeros(n_features);

        for j in 0..n_features {
            for i in 0..n_classes {
                let dk = (original_centroids[[i, j]] - overall_centroid[j])
                    / (pooled_stds[j] * (1.0 / class_priors[i] + 1.0).sqrt());

                let dk_prime = if dk.abs() > threshold {
                    dk.signum() * (dk.abs() - threshold)
                } else {
                    0.0
                };

                shrunken_centroids[[i, j]] = overall_centroid[j]
                    + pooled_stds[j] * (1.0 / class_priors[i] + 1.0).sqrt() * dk_prime;

                // Record shrinkage factor for this feature
                if i == 0 {
                    shrinkage_factors[j] = if dk.abs() > 0.0 {
                        (dk.abs() - dk_prime.abs()) / dk.abs()
                    } else {
                        0.0
                    };
                }
            }
        }

        Ok((shrunken_centroids, shrinkage_factors))
    }

    /// Determine which features are selected (non-zero after shrinkage)
    fn determine_selected_features(&self, shrunken_centroids: &Array2<Float>) -> Array1<bool> {
        let (n_classes, n_features) = shrunken_centroids.dim();
        let mut selected = Array1::from_elem(n_features, false);

        for j in 0..n_features {
            for i in 0..n_classes {
                if shrunken_centroids[[i, j]].abs() > 1e-10 {
                    selected[j] = true;
                    break;
                }
            }
        }

        selected
    }

    /// Predict single sample
    fn predict_single_sample(
        &self,
        sample: &ArrayView1<Float>,
        shrunken_centroids: &Array2<Float>,
        class_priors: &Array1<Float>,
        classes: &Array1<i32>,
    ) -> Result<i32> {
        let _n_classes = classes.len();
        let mut min_distance = Float::INFINITY;
        let mut predicted_class = classes[0];

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let centroid = shrunken_centroids.row(class_idx);

            // Compute squared Euclidean distance
            let mut distance = 0.0;
            for j in 0..sample.len() {
                let diff = sample[j] - centroid[j];
                distance += diff * diff;
            }

            // Add prior probability penalty
            distance -= 2.0 * class_priors[class_idx].ln();

            if distance < min_distance {
                min_distance = distance;
                predicted_class = class_label;
            }
        }

        Ok(predicted_class)
    }

    /// Static version of predict_single_sample for use when we have mutable borrows
    fn predict_single_sample_static(
        sample: &ArrayView1<Float>,
        shrunken_centroids: &Array2<Float>,
        class_priors: &Array1<Float>,
        classes: &Array1<i32>,
    ) -> Result<i32> {
        let _n_classes = classes.len();
        let mut min_distance = Float::INFINITY;
        let mut predicted_class = classes[0];

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let centroid = shrunken_centroids.row(class_idx);

            // Compute squared Euclidean distance
            let mut distance = 0.0;
            for j in 0..sample.len() {
                let diff = sample[j] - centroid[j];
                distance += diff * diff;
            }

            // Add prior probability penalty
            distance -= 2.0 * class_priors[class_idx].ln();

            if distance < min_distance {
                min_distance = distance;
                predicted_class = class_label;
            }
        }

        Ok(predicted_class)
    }
}

impl NearestShrunkenCentroids<Trained> {
    /// Preprocess input data according to training standardization
    fn preprocess_input(&self, x: &Array2<Float>) -> Array2<Float> {
        if !self.config.standardize {
            return x.clone();
        }

        let data = self.data.as_ref().unwrap();
        let mut x_standardized = x.clone();

        for (_i, mut row) in x_standardized.axis_iter_mut(Axis(0)).enumerate() {
            for j in 0..data.n_features {
                row[j] = (row[j] - data.feature_means[j]) / data.feature_stds[j];
            }
        }

        x_standardized
    }
}

impl Predict<Array2<Float>, Array1<i32>> for NearestShrunkenCentroids<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let (n_samples, n_features) = x.dim();
        let data = self.data.as_ref().unwrap();

        if n_features != data.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                data.n_features, n_features
            )));
        }

        // Preprocess input data
        let x_preprocessed = self.preprocess_input(x);

        let mut predictions = Array1::zeros(n_samples);

        for (i, sample) in x_preprocessed.axis_iter(Axis(0)).enumerate() {
            predictions[i] = NearestShrunkenCentroids::<Untrained>::predict_single_sample_static(
                &sample,
                &data.shrunken_centroids,
                &data.class_priors,
                &data.classes,
            )?;
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for NearestShrunkenCentroids<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let data = self.data.as_ref().unwrap();

        if n_features != data.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                data.n_features, n_features
            )));
        }

        // Preprocess input data
        let x_preprocessed = self.preprocess_input(x);

        let n_classes = data.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x_preprocessed.axis_iter(Axis(0)).enumerate() {
            let mut distances = Array1::zeros(n_classes);

            // Compute distances to each centroid
            for (class_idx, _) in data.classes.iter().enumerate() {
                let centroid = data.shrunken_centroids.row(class_idx);

                let mut distance = 0.0;
                for j in 0..data.n_features {
                    let diff = sample[j] - centroid[j];
                    distance += diff * diff;
                }

                // Add prior probability
                distance -= 2.0 * data.class_priors[class_idx].ln();
                distances[class_idx] = distance;
            }

            // Convert distances to probabilities using softmax
            let min_distance = distances.iter().fold(Float::INFINITY, |a, &b| a.min(b));
            let exp_neg_distances: Array1<Float> = distances.mapv(|d| (-(d - min_distance)).exp());
            let sum_exp = exp_neg_distances.sum();

            if sum_exp > 1e-10 {
                probas.row_mut(i).assign(&(exp_neg_distances / sum_exp));
            } else {
                probas.row_mut(i).fill(1.0 / n_classes as Float);
            }
        }

        Ok(probas)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for NearestShrunkenCentroids<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (_n_samples, n_features) = x.dim();
        let data = self.data.as_ref().unwrap();

        if n_features != data.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                data.n_features, n_features
            )));
        }

        // Preprocess input data
        let x_preprocessed = self.preprocess_input(x);

        // Select only the features that survived shrinkage
        let selected_indices = self.selected_feature_indices();

        if selected_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected after shrinkage".to_string(),
            ));
        }

        Ok(x_preprocessed.select(Axis(1), &selected_indices))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nearest_shrunken_centroids_basic() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2], // Class 0
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2] // Class 1
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let nsc = NearestShrunkenCentroids::new().threshold(0.0);
        let fitted = nsc.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.n_features(), 2);
    }

    #[test]
    fn test_nearest_shrunken_centroids_predict_proba() {
        let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
        let y = array![0, 0, 1, 1];

        let nsc = NearestShrunkenCentroids::new().threshold(0.1);
        let fitted = nsc.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_nearest_shrunken_centroids_with_shrinkage() {
        let x = array![
            [1.0, 2.0, 10.0],
            [1.1, 2.1, 10.1],
            [3.0, 4.0, 10.0],
            [3.1, 4.1, 10.1]
        ];
        let y = array![0, 0, 1, 1];

        let nsc = NearestShrunkenCentroids::new().threshold(2.0);
        let fitted = nsc.fit(&x, &y).unwrap();

        // Third feature should be shrunk to zero due to low discrimination
        let selected_features = fitted.selected_features();
        assert!(selected_features[0] || selected_features[1]); // At least one of first two selected

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_nearest_shrunken_centroids_standardization() {
        let x = array![[10.0, 200.0], [11.0, 210.0], [30.0, 400.0], [31.0, 410.0]];
        let y = array![0, 0, 1, 1];

        let nsc_std = NearestShrunkenCentroids::new()
            .threshold(0.1)
            .standardize(true);
        let fitted_std = nsc_std.fit(&x, &y).unwrap();

        let nsc_no_std = NearestShrunkenCentroids::new()
            .threshold(0.1)
            .standardize(false);
        let fitted_no_std = nsc_no_std.fit(&x, &y).unwrap();

        // Both should work but may give different results
        let predictions_std = fitted_std.predict(&x).unwrap();
        let predictions_no_std = fitted_no_std.predict(&x).unwrap();

        assert_eq!(predictions_std.len(), 4);
        assert_eq!(predictions_no_std.len(), 4);
    }

    #[test]
    fn test_nearest_shrunken_centroids_transform() {
        let x = array![
            [1.0, 2.0, 10.0],
            [1.1, 2.1, 10.0],
            [3.0, 4.0, 10.0],
            [3.1, 4.1, 10.0]
        ];
        let y = array![0, 0, 1, 1];

        let nsc = NearestShrunkenCentroids::new().threshold(1.0);
        let fitted = nsc.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        // Should select features with good discrimination
        assert!(transformed.ncols() > 0);
        assert!(transformed.ncols() <= 3);
        assert_eq!(transformed.nrows(), 4);
    }

    #[test]
    fn test_nearest_shrunken_centroids_feature_scores() {
        let x = array![
            [1.0, 2.0, 5.0],
            [1.1, 2.1, 5.1],
            [3.0, 4.0, 5.0],
            [3.1, 4.1, 5.1]
        ];
        let y = array![0, 0, 1, 1];

        let nsc = NearestShrunkenCentroids::new().threshold(0.0);
        let fitted = nsc.fit(&x, &y).unwrap();
        let scores = fitted.feature_scores();

        assert_eq!(scores.len(), 3);

        // Features 0 and 1 should have higher scores than feature 2
        assert!(scores[0] > scores[2]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn test_nearest_shrunken_centroids_cross_validation() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.3, 2.3],
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2],
            [3.3, 4.3]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let nsc = NearestShrunkenCentroids::new()
            .cv_folds(Some(2))
            .threshold_range(0.0, 2.0, 5);
        let fitted = nsc.fit(&x, &y).unwrap();

        assert!(fitted.cv_results().is_some());
        let cv_results = fitted.cv_results().unwrap();
        assert!(!cv_results.is_empty());

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 8);
    }

    #[test]
    fn test_nearest_shrunken_centroids_multiclass() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [3.0, 4.0],
            [3.1, 4.1],
            [5.0, 6.0],
            [5.1, 6.1]
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let nsc = NearestShrunkenCentroids::new().threshold(0.1);
        let fitted = nsc.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 3);
        assert_eq!(probas.dim(), (6, 3));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_nearest_shrunken_centroids_selected_feature_indices() {
        let x = array![
            [1.0, 2.0, 10.0, 20.0],
            [1.1, 2.1, 10.0, 20.0],
            [3.0, 4.0, 10.0, 20.0],
            [3.1, 4.1, 10.0, 20.0]
        ];
        let y = array![0, 0, 1, 1];

        let nsc = NearestShrunkenCentroids::new().threshold(1.5);
        let fitted = nsc.fit(&x, &y).unwrap();
        let selected_indices = fitted.selected_feature_indices();

        // Should select features with good discrimination
        assert!(!selected_indices.is_empty());
        assert!(selected_indices.len() <= 4);
    }

    #[test]
    fn test_nearest_shrunken_centroids_prior_types() {
        let x = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2], // Class 0 (3 samples)
            [3.0, 4.0],
            [3.1, 4.1] // Class 1 (2 samples)
        ];
        let y = array![0, 0, 0, 1, 1];

        // Test uniform priors
        let nsc_uniform = NearestShrunkenCentroids::new()
            .threshold(0.0)
            .prior_type("uniform");
        let fitted_uniform = nsc_uniform.fit(&x, &y).unwrap();

        // Test class priors
        let nsc_class = NearestShrunkenCentroids::new()
            .threshold(0.0)
            .prior_type("class_prior");
        let fitted_class = nsc_class.fit(&x, &y).unwrap();

        // Both should work
        let predictions_uniform = fitted_uniform.predict(&x).unwrap();
        let predictions_class = fitted_class.predict(&x).unwrap();

        assert_eq!(predictions_uniform.len(), 5);
        assert_eq!(predictions_class.len(), 5);

        // Check class priors
        let uniform_priors = fitted_uniform.class_priors();
        let class_priors = fitted_class.class_priors();

        assert_abs_diff_eq!(uniform_priors[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(uniform_priors[1], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(class_priors[0], 0.6, epsilon = 1e-6);
        assert_abs_diff_eq!(class_priors[1], 0.4, epsilon = 1e-6);
    }
}
