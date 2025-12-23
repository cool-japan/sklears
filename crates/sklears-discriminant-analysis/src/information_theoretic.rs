//! Information-Theoretic Discriminant Analysis implementation
//!
//! This module implements discriminant analysis methods based on information theory,
//! including mutual information, maximum entropy discrimination, and information gain.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::{collections::HashMap, marker::PhantomData};

/// Configuration for Information-Theoretic Discriminant Analysis
#[derive(Debug, Clone)]
pub struct InformationTheoreticDiscriminantAnalysisConfig {
    /// Information criterion to use
    pub criterion: InformationCriterion,
    /// Number of bins for histogram-based estimation
    pub n_bins: usize,
    /// Regularization parameter for entropy estimation
    pub entropy_regularization: Float,
    /// Method for handling continuous variables
    pub discretization_method: DiscretizationMethod,
    /// Number of components for dimensionality reduction
    pub n_components: Option<usize>,
    /// Feature selection threshold based on information gain
    pub information_threshold: Float,
    /// Maximum iterations for optimization algorithms
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for InformationTheoreticDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            criterion: InformationCriterion::MutualInformation,
            n_bins: 10,
            entropy_regularization: 1e-8,
            discretization_method: DiscretizationMethod::EqualWidth,
            n_components: None,
            information_threshold: 0.01,
            max_iter: 100,
            tol: 1e-6,
            random_state: None,
        }
    }
}

/// Information criteria for discriminant analysis
#[derive(Debug, Clone)]
pub enum InformationCriterion {
    /// Mutual information between features and classes
    MutualInformation,
    /// Information gain (reduction in entropy)
    InformationGain,
    /// Conditional mutual information
    ConditionalMutualInformation,
    /// Normalized mutual information
    NormalizedMutualInformation,
    /// Joint mutual information for feature pairs
    JointMutualInformation,
    /// Maximum entropy discrimination
    MaximumEntropy { lambda: Float },
}

/// Methods for discretizing continuous variables
#[derive(Debug, Clone)]
pub enum DiscretizationMethod {
    /// Equal-width binning
    EqualWidth,
    /// Equal-frequency binning (quantiles)
    EqualFrequency,
    /// K-means based binning
    KMeans,
    /// Entropy-based binning (minimum description length)
    EntropyBased,
    /// Supervised binning using class information
    Supervised,
}

/// Information-Theoretic Discriminant Analysis
///
/// A discriminant analysis method based on information theory principles,
/// using mutual information, entropy, and information gain for feature selection
/// and classification.
#[derive(Debug, Clone)]
pub struct InformationTheoreticDiscriminantAnalysis<State = Untrained> {
    config: InformationTheoreticDiscriminantAnalysisConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    feature_scores_: Option<Array1<Float>>,
    selected_features_: Option<Array1<bool>>,
    discretization_thresholds_: Option<Vec<Array1<Float>>>,
    class_priors_: Option<Array1<Float>>,
    feature_distributions_: Option<Vec<HashMap<(i32, i32), Float>>>, // (feature_bin, class) -> probability
    mutual_information_matrix_: Option<Array2<Float>>,
    entropy_weights_: Option<Array1<Float>>,
    n_features_: Option<usize>,
}

impl InformationTheoreticDiscriminantAnalysis<Untrained> {
    /// Create a new InformationTheoreticDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            config: InformationTheoreticDiscriminantAnalysisConfig::default(),
            state: PhantomData,
            classes_: None,
            feature_scores_: None,
            selected_features_: None,
            discretization_thresholds_: None,
            class_priors_: None,
            feature_distributions_: None,
            mutual_information_matrix_: None,
            entropy_weights_: None,
            n_features_: None,
        }
    }

    /// Set the information criterion
    pub fn criterion(mut self, criterion: InformationCriterion) -> Self {
        self.config.criterion = criterion;
        self
    }

    /// Set number of bins for discretization
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.config.n_bins = n_bins;
        self
    }

    /// Set entropy regularization parameter
    pub fn entropy_regularization(mut self, reg: Float) -> Self {
        self.config.entropy_regularization = reg;
        self
    }

    /// Set discretization method
    pub fn discretization_method(mut self, method: DiscretizationMethod) -> Self {
        self.config.discretization_method = method;
        self
    }

    /// Set number of components
    pub fn n_components(mut self, n: Option<usize>) -> Self {
        self.config.n_components = n;
        self
    }

    /// Set information threshold for feature selection
    pub fn information_threshold(mut self, threshold: Float) -> Self {
        self.config.information_threshold = threshold;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}

impl Default for InformationTheoreticDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for InformationTheoreticDiscriminantAnalysis<Untrained> {
    type Config = InformationTheoreticDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for InformationTheoreticDiscriminantAnalysis<Untrained> {
    type Fitted = InformationTheoreticDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of labels must match number of samples".to_string(),
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
                "At least 2 classes are required".to_string(),
            ));
        }

        // Compute class priors
        let class_priors = self.compute_class_priors(y, &classes)?;

        // Discretize features
        let (discretized_x, discretization_thresholds) = self.discretize_features(x)?;

        // Compute feature distributions
        let feature_distributions =
            self.compute_feature_distributions(&discretized_x, y, &classes)?;

        // Compute information scores for each feature
        let feature_scores =
            self.compute_information_scores(&discretized_x, y, &classes, &class_priors)?;

        // Select features based on information threshold
        let selected_features =
            feature_scores.mapv(|score| score >= self.config.information_threshold);

        // Compute mutual information matrix for feature interactions
        let mutual_information_matrix =
            self.compute_mutual_information_matrix(&discretized_x, y, &classes)?;

        // Compute entropy-based weights for features
        let entropy_weights = self.compute_entropy_weights(&discretized_x, y, &classes)?;

        Ok(InformationTheoreticDiscriminantAnalysis {
            config: self.config,
            state: PhantomData,
            classes_: Some(classes),
            feature_scores_: Some(feature_scores),
            selected_features_: Some(selected_features),
            discretization_thresholds_: Some(discretization_thresholds),
            class_priors_: Some(class_priors),
            feature_distributions_: Some(feature_distributions),
            mutual_information_matrix_: Some(mutual_information_matrix),
            entropy_weights_: Some(entropy_weights),
            n_features_: Some(n_features),
        })
    }
}

impl InformationTheoreticDiscriminantAnalysis<Untrained> {
    fn compute_class_priors(
        &self,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<Array1<Float>> {
        let n_samples = y.len() as Float;
        let mut priors = Array1::zeros(classes.len());

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let count = y.iter().filter(|&&label| label == class_label).count() as Float;
            priors[class_idx] = count / n_samples;
        }

        Ok(priors)
    }

    fn discretize_features(&self, x: &Array2<Float>) -> Result<(Array2<i32>, Vec<Array1<Float>>)> {
        let n_features = x.ncols();
        let mut discretized_x = Array2::zeros((x.nrows(), n_features));
        let mut thresholds = Vec::new();

        for feature_idx in 0..n_features {
            let feature_column = x.column(feature_idx);
            let (discretized_feature, feature_thresholds) =
                self.discretize_single_feature(&feature_column)?;

            discretized_x
                .column_mut(feature_idx)
                .assign(&discretized_feature);
            thresholds.push(feature_thresholds);
        }

        Ok((discretized_x, thresholds))
    }

    fn discretize_single_feature(
        &self,
        feature: &ArrayView1<Float>,
    ) -> Result<(Array1<i32>, Array1<Float>)> {
        let n_bins = self.config.n_bins;
        let mut feature_values = feature.to_vec();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let thresholds = match self.config.discretization_method {
            DiscretizationMethod::EqualWidth => {
                let min_val = feature_values[0];
                let max_val = feature_values[feature_values.len() - 1];
                let width = (max_val - min_val) / n_bins as Float;

                let mut thresholds = Vec::new();
                for i in 1..n_bins {
                    thresholds.push(min_val + i as Float * width);
                }
                thresholds
            }
            DiscretizationMethod::EqualFrequency => {
                let mut thresholds = Vec::new();
                let n_samples = feature_values.len();
                for i in 1..n_bins {
                    let idx = (i * n_samples) / n_bins;
                    if idx < n_samples {
                        thresholds.push(feature_values[idx]);
                    }
                }
                thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
                thresholds
            }
            DiscretizationMethod::KMeans => self.kmeans_discretization(&feature_values, n_bins)?,
            DiscretizationMethod::EntropyBased => {
                self.entropy_based_discretization(&feature_values, n_bins)?
            }
            DiscretizationMethod::Supervised => {
                // For supervised discretization, we need class information
                // For now, fall back to entropy-based
                self.entropy_based_discretization(&feature_values, n_bins)?
            }
        };

        // Apply discretization
        let mut discretized = Array1::zeros(feature.len());
        for (i, &value) in feature.iter().enumerate() {
            let mut bin = 0;
            for &threshold in &thresholds {
                if value > threshold {
                    bin += 1;
                } else {
                    break;
                }
            }
            discretized[i] = bin;
        }

        Ok((discretized, Array1::from_vec(thresholds)))
    }

    fn compute_feature_distributions(
        &self,
        x: &Array2<i32>,
        y: &Array1<i32>,
        _classes: &Array1<i32>,
    ) -> Result<Vec<HashMap<(i32, i32), Float>>> {
        let n_features = x.ncols();
        let n_samples = x.nrows() as Float;
        let mut distributions = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let mut feature_dist = HashMap::new();
            let feature_column = x.column(feature_idx);

            // Count occurrences of each (feature_bin, class) pair
            for (sample_idx, &feature_bin) in feature_column.iter().enumerate() {
                let class_label = y[sample_idx];
                let key = (feature_bin, class_label);
                *feature_dist.entry(key).or_insert(0.0) += 1.0;
            }

            // Normalize to probabilities
            for value in feature_dist.values_mut() {
                *value /= n_samples;
            }

            distributions.push(feature_dist);
        }

        Ok(distributions)
    }

    fn compute_information_scores(
        &self,
        x: &Array2<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        class_priors: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n_features = x.ncols();
        let mut scores = Array1::zeros(n_features);

        for feature_idx in 0..n_features {
            let feature_column = x.column(feature_idx);
            let score = match &self.config.criterion {
                InformationCriterion::MutualInformation => {
                    self.compute_mutual_information(&feature_column, y, classes, class_priors)?
                }
                InformationCriterion::InformationGain => {
                    self.compute_information_gain(&feature_column, y, classes, class_priors)?
                }
                InformationCriterion::NormalizedMutualInformation => self
                    .compute_normalized_mutual_information(
                        &feature_column,
                        y,
                        classes,
                        class_priors,
                    )?,
                InformationCriterion::ConditionalMutualInformation => self
                    .compute_conditional_mutual_information(
                        &feature_column,
                        y,
                        classes,
                        class_priors,
                        x,
                        feature_idx,
                    )?,
                InformationCriterion::JointMutualInformation => self
                    .compute_joint_mutual_information(
                        &feature_column,
                        y,
                        classes,
                        class_priors,
                        x,
                        feature_idx,
                    )?,
                InformationCriterion::MaximumEntropy { lambda } => self
                    .compute_maximum_entropy_discrimination(
                        &feature_column,
                        y,
                        classes,
                        class_priors,
                        *lambda,
                    )?,
            };
            scores[feature_idx] = score;
        }

        Ok(scores)
    }

    fn compute_mutual_information(
        &self,
        feature: &ArrayView1<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        class_priors: &Array1<Float>,
    ) -> Result<Float> {
        let n_samples = feature.len() as Float;
        let mut mutual_info = 0.0;

        // Get unique feature values
        let mut feature_values: Vec<i32> = feature.to_vec();
        feature_values.sort_unstable();
        feature_values.dedup();

        // Compute feature value probabilities
        let mut feature_probs = HashMap::new();
        for &value in &feature_values {
            let count = feature.iter().filter(|&&v| v == value).count() as Float;
            feature_probs.insert(value, count / n_samples);
        }

        // Compute joint probabilities and mutual information
        for &feature_value in &feature_values {
            for (class_idx, &class_label) in classes.iter().enumerate() {
                let joint_count = feature
                    .iter()
                    .zip(y.iter())
                    .filter(|(&f, &c)| f == feature_value && c == class_label)
                    .count() as Float;

                if joint_count > 0.0 {
                    let joint_prob = joint_count / n_samples;
                    let feature_prob = feature_probs[&feature_value];
                    let class_prob = class_priors[class_idx];

                    if feature_prob > 0.0 && class_prob > 0.0 {
                        mutual_info += joint_prob * (joint_prob / (feature_prob * class_prob)).ln();
                    }
                }
            }
        }

        Ok(mutual_info)
    }

    fn compute_information_gain(
        &self,
        feature: &ArrayView1<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        class_priors: &Array1<Float>,
    ) -> Result<Float> {
        // Information gain = H(Y) - H(Y|X)
        let class_entropy = self.compute_entropy(class_priors)?;
        let conditional_entropy = self.compute_conditional_entropy(feature, y, classes)?;
        Ok(class_entropy - conditional_entropy)
    }

    fn compute_normalized_mutual_information(
        &self,
        feature: &ArrayView1<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        class_priors: &Array1<Float>,
    ) -> Result<Float> {
        let mutual_info = self.compute_mutual_information(feature, y, classes, class_priors)?;
        let class_entropy = self.compute_entropy(class_priors)?;

        // Get feature entropy
        let n_samples = feature.len() as Float;
        let mut feature_values: Vec<i32> = feature.to_vec();
        feature_values.sort_unstable();
        feature_values.dedup();

        let mut feature_probs = Vec::new();
        for &value in &feature_values {
            let count = feature.iter().filter(|&&v| v == value).count() as Float;
            feature_probs.push(count / n_samples);
        }

        let feature_entropy = self.compute_entropy(&Array1::from_vec(feature_probs))?;

        if class_entropy == 0.0 && feature_entropy == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * mutual_info / (class_entropy + feature_entropy))
        }
    }

    fn compute_entropy(&self, probabilities: &Array1<Float>) -> Result<Float> {
        let mut entropy = 0.0;
        for &prob in probabilities.iter() {
            if prob > self.config.entropy_regularization {
                entropy -= prob * prob.ln();
            }
        }
        Ok(entropy)
    }

    fn compute_conditional_entropy(
        &self,
        feature: &ArrayView1<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<Float> {
        let n_samples = feature.len() as Float;
        let mut feature_values: Vec<i32> = feature.to_vec();
        feature_values.sort_unstable();
        feature_values.dedup();

        let mut conditional_entropy = 0.0;

        for &feature_value in &feature_values {
            // Get indices where feature has this value
            let indices: Vec<usize> = feature
                .iter()
                .enumerate()
                .filter(|(_, &v)| v == feature_value)
                .map(|(i, _)| i)
                .collect();

            if indices.is_empty() {
                continue;
            }

            let feature_prob = indices.len() as Float / n_samples;

            // Compute class distribution for this feature value
            let mut class_counts = vec![0; classes.len()];
            for &idx in &indices {
                let class_label = y[idx];
                if let Some(class_idx) = classes.iter().position(|&c| c == class_label) {
                    class_counts[class_idx] += 1;
                }
            }

            // Convert counts to probabilities
            let mut class_probs = Array1::zeros(classes.len());
            let total_count = indices.len() as Float;
            for (i, &count) in class_counts.iter().enumerate() {
                class_probs[i] = count as Float / total_count;
            }

            // Compute entropy for this feature value
            let feature_entropy = self.compute_entropy(&class_probs)?;
            conditional_entropy += feature_prob * feature_entropy;
        }

        Ok(conditional_entropy)
    }

    fn compute_mutual_information_matrix(
        &self,
        x: &Array2<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<Array2<Float>> {
        let n_features = x.ncols();
        let mut mi_matrix = Array2::zeros((n_features, n_features));

        let class_priors = self.compute_class_priors(y, classes)?;

        for i in 0..n_features {
            for j in i..n_features {
                let feature_i = x.column(i);
                let feature_j = x.column(j);

                let mi = if i == j {
                    // Self mutual information (entropy)
                    self.compute_mutual_information(&feature_i, y, classes, &class_priors)?
                } else {
                    // Mutual information between features (simplified)
                    self.compute_feature_mutual_information(&feature_i, &feature_j)?
                };

                mi_matrix[[i, j]] = mi;
                mi_matrix[[j, i]] = mi;
            }
        }

        Ok(mi_matrix)
    }

    fn compute_feature_mutual_information(
        &self,
        feature1: &ArrayView1<i32>,
        feature2: &ArrayView1<i32>,
    ) -> Result<Float> {
        let n_samples = feature1.len() as Float;
        let mut mutual_info = 0.0;

        // Get unique values for both features
        let mut values1: Vec<i32> = feature1.to_vec();
        values1.sort_unstable();
        values1.dedup();

        let mut values2: Vec<i32> = feature2.to_vec();
        values2.sort_unstable();
        values2.dedup();

        // Compute marginal probabilities
        let mut probs1 = HashMap::new();
        let mut probs2 = HashMap::new();
        let mut joint_probs = HashMap::new();

        for &v1 in &values1 {
            let count = feature1.iter().filter(|&&v| v == v1).count() as Float;
            probs1.insert(v1, count / n_samples);
        }

        for &v2 in &values2 {
            let count = feature2.iter().filter(|&&v| v == v2).count() as Float;
            probs2.insert(v2, count / n_samples);
        }

        // Compute joint probabilities
        for &v1 in &values1 {
            for &v2 in &values2 {
                let count = feature1
                    .iter()
                    .zip(feature2.iter())
                    .filter(|(&f1, &f2)| f1 == v1 && f2 == v2)
                    .count() as Float;
                joint_probs.insert((v1, v2), count / n_samples);
            }
        }

        // Compute mutual information
        for &v1 in &values1 {
            for &v2 in &values2 {
                let joint_prob = joint_probs[&(v1, v2)];
                if joint_prob > 0.0 {
                    let prob1 = probs1[&v1];
                    let prob2 = probs2[&v2];
                    if prob1 > 0.0 && prob2 > 0.0 {
                        mutual_info += joint_prob * (joint_prob / (prob1 * prob2)).ln();
                    }
                }
            }
        }

        Ok(mutual_info)
    }

    fn compute_entropy_weights(
        &self,
        x: &Array2<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<Array1<Float>> {
        let n_features = x.ncols();
        let mut weights = Array1::zeros(n_features);
        let class_priors = self.compute_class_priors(y, classes)?;

        for feature_idx in 0..n_features {
            let feature_column = x.column(feature_idx);
            let information_gain =
                self.compute_information_gain(&feature_column, y, classes, &class_priors)?;
            weights[feature_idx] = information_gain.exp(); // Exponential weighting
        }

        // Normalize weights
        let sum_weights = weights.sum();
        if sum_weights > 0.0 {
            weights /= sum_weights;
        } else {
            weights.fill(1.0 / n_features as Float);
        }

        Ok(weights)
    }

    fn compute_conditional_mutual_information(
        &self,
        feature: &ArrayView1<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        class_priors: &Array1<Float>,
        x: &Array2<i32>,
        feature_idx: usize,
    ) -> Result<Float> {
        // Compute I(X_i; Y | X_rest) = I(X_i, X_rest; Y) - I(X_rest; Y)
        // Simplified implementation using a subset of other features as conditioning set

        let n_features = x.ncols();
        if n_features <= 1 {
            // Fall back to regular mutual information if only one feature
            return self.compute_mutual_information(feature, y, classes, class_priors);
        }

        // Use up to 3 other features as conditioning variables (for computational efficiency)
        let conditioning_features: Vec<usize> = (0..n_features)
            .filter(|&i| i != feature_idx)
            .take(3)
            .collect();

        if conditioning_features.is_empty() {
            return self.compute_mutual_information(feature, y, classes, class_priors);
        }

        let n_samples = feature.len() as Float;
        let mut conditional_mi = 0.0;

        // Create joint conditioning context
        let mut conditioning_states = Vec::new();
        for &sample_idx in (0..feature.len()).collect::<Vec<_>>().iter() {
            let mut state = Vec::new();
            for &cond_feat in &conditioning_features {
                state.push(x[[sample_idx, cond_feat]]);
            }
            conditioning_states.push(state);
        }

        // Get unique conditioning states
        let mut unique_cond_states = conditioning_states.clone();
        unique_cond_states.sort();
        unique_cond_states.dedup();

        // Compute conditional mutual information for each conditioning state
        for cond_state in &unique_cond_states {
            // Find samples with this conditioning state
            let matching_indices: Vec<usize> = conditioning_states
                .iter()
                .enumerate()
                .filter(|(_, state)| *state == cond_state)
                .map(|(i, _)| i)
                .collect();

            if matching_indices.len() < 2 {
                continue; // Need at least 2 samples for meaningful MI computation
            }

            let cond_prob = matching_indices.len() as Float / n_samples;

            // Extract feature and class values for matching samples
            let cond_feature_values: Vec<i32> =
                matching_indices.iter().map(|&i| feature[i]).collect();
            let cond_class_values: Vec<i32> = matching_indices.iter().map(|&i| y[i]).collect();

            // Compute MI for this conditioning state
            let cond_feature_array = Array1::from_vec(cond_feature_values);
            let cond_class_array = Array1::from_vec(cond_class_values);

            // Compute conditional class priors
            let mut cond_class_priors = Array1::zeros(classes.len());
            for (class_idx, &class_label) in classes.iter().enumerate() {
                let count = cond_class_array
                    .iter()
                    .filter(|&&c| c == class_label)
                    .count() as Float;
                cond_class_priors[class_idx] = count / cond_class_array.len() as Float;
            }

            let cond_mi = self.compute_mutual_information(
                &cond_feature_array.view(),
                &cond_class_array,
                classes,
                &cond_class_priors,
            )?;

            conditional_mi += cond_prob * cond_mi;
        }

        Ok(conditional_mi)
    }

    fn compute_joint_mutual_information(
        &self,
        feature: &ArrayView1<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        class_priors: &Array1<Float>,
        x: &Array2<i32>,
        feature_idx: usize,
    ) -> Result<Float> {
        // Compute joint MI with other features: I(X_i, X_j; Y) for pairs of features
        let n_features = x.ncols();
        if n_features <= 1 {
            return self.compute_mutual_information(feature, y, classes, class_priors);
        }

        let mut max_joint_mi: f64 = 0.0;

        // Try pairing with each other feature and take the maximum joint MI
        for other_feature_idx in 0..n_features {
            if other_feature_idx == feature_idx {
                continue;
            }

            let other_feature = x.column(other_feature_idx);
            let joint_mi = self.compute_joint_feature_mutual_information(
                feature,
                &other_feature,
                y,
                classes,
                class_priors,
            )?;

            max_joint_mi = max_joint_mi.max(joint_mi);
        }

        Ok(max_joint_mi)
    }

    fn compute_joint_feature_mutual_information(
        &self,
        feature1: &ArrayView1<i32>,
        feature2: &ArrayView1<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        class_priors: &Array1<Float>,
    ) -> Result<Float> {
        let n_samples = feature1.len() as Float;
        let mut joint_mi = 0.0;

        // Get unique value combinations for the two features
        let mut joint_feature_values = Vec::new();
        for i in 0..feature1.len() {
            joint_feature_values.push((feature1[i], feature2[i]));
        }

        let mut unique_joint_values = joint_feature_values.clone();
        unique_joint_values.sort();
        unique_joint_values.dedup();

        // Compute joint probabilities
        let mut joint_probs = std::collections::HashMap::new();
        for &(f1, f2) in &unique_joint_values {
            let count = joint_feature_values
                .iter()
                .filter(|&&(v1, v2)| v1 == f1 && v2 == f2)
                .count() as Float;
            joint_probs.insert((f1, f2), count / n_samples);
        }

        // Compute joint mutual information
        for &(f1, f2) in &unique_joint_values {
            for (class_idx, &class_label) in classes.iter().enumerate() {
                let joint_class_count = feature1
                    .iter()
                    .zip(feature2.iter())
                    .zip(y.iter())
                    .filter(|((&v1, &v2), &c)| v1 == f1 && v2 == f2 && c == class_label)
                    .count() as Float;

                if joint_class_count > 0.0 {
                    let joint_class_prob = joint_class_count / n_samples;
                    let joint_feature_prob = joint_probs[&(f1, f2)];
                    let class_prob = class_priors[class_idx];

                    if joint_feature_prob > 0.0 && class_prob > 0.0 {
                        joint_mi += joint_class_prob
                            * (joint_class_prob / (joint_feature_prob * class_prob)).ln();
                    }
                }
            }
        }

        Ok(joint_mi)
    }

    fn compute_maximum_entropy_discrimination(
        &self,
        feature: &ArrayView1<i32>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        class_priors: &Array1<Float>,
        lambda: Float,
    ) -> Result<Float> {
        // Maximum entropy discrimination: H(Y) + λ * I(X; Y)
        // This balances entropy maximization with mutual information maximization

        let class_entropy = self.compute_entropy(class_priors)?;
        let mutual_info = self.compute_mutual_information(feature, y, classes, class_priors)?;

        // Combine entropy and mutual information with lambda weighting
        let max_entropy_score = class_entropy + lambda * mutual_info;

        Ok(max_entropy_score)
    }

    fn kmeans_discretization(&self, feature_values: &[Float], n_bins: usize) -> Result<Vec<Float>> {
        if feature_values.len() < n_bins {
            return Ok(Vec::new());
        }

        let min_val = feature_values[0];
        let max_val = feature_values[feature_values.len() - 1];

        // Initialize centroids evenly distributed
        let mut centroids: Vec<Float> = (0..n_bins)
            .map(|i| min_val + (i as Float / (n_bins - 1) as Float) * (max_val - min_val))
            .collect();

        // Simple k-means iterations
        for _ in 0..20 {
            // Maximum 20 iterations
            let mut new_centroids = vec![0.0; n_bins];
            let mut counts = vec![0; n_bins];

            // Assign each point to nearest centroid
            for &value in feature_values {
                let mut best_cluster = 0;
                let mut best_distance = Float::INFINITY;

                for (i, &centroid) in centroids.iter().enumerate() {
                    let distance = (value - centroid).abs();
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = i;
                    }
                }

                new_centroids[best_cluster] += value;
                counts[best_cluster] += 1;
            }

            // Update centroids
            let mut converged = true;
            for i in 0..n_bins {
                if counts[i] > 0 {
                    new_centroids[i] /= counts[i] as Float;
                    if (new_centroids[i] - centroids[i]).abs() > 1e-6 {
                        converged = false;
                    }
                    centroids[i] = new_centroids[i];
                }
            }

            if converged {
                break;
            }
        }

        // Convert centroids to thresholds (midpoints between consecutive centroids)
        centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut thresholds = Vec::new();
        for i in 0..centroids.len() - 1 {
            thresholds.push((centroids[i] + centroids[i + 1]) / 2.0);
        }

        Ok(thresholds)
    }

    fn entropy_based_discretization(
        &self,
        feature_values: &[Float],
        n_bins: usize,
    ) -> Result<Vec<Float>> {
        // Simplified entropy-based discretization using MDLP (Minimum Description Length Principle)
        if feature_values.len() < n_bins {
            return Ok(Vec::new());
        }

        let min_val = feature_values[0];
        let max_val = feature_values[feature_values.len() - 1];

        // Start with equal-width binning and refine based on entropy
        let mut thresholds: Vec<Float> = (1..n_bins)
            .map(|i| min_val + (i as Float / n_bins as Float) * (max_val - min_val))
            .collect();

        // Refine thresholds to minimize entropy
        for _iteration in 0..10 {
            // Maximum 10 refinement iterations
            let mut improved = false;

            for i in 0..thresholds.len() {
                let original_threshold = thresholds[i];
                let mut best_threshold = original_threshold;
                let mut best_entropy = self.compute_binning_entropy(feature_values, &thresholds)?;

                // Try small adjustments to the threshold
                let adjustment_range = (max_val - min_val) / (n_bins as Float * 10.0);
                for delta in &[-adjustment_range, adjustment_range] {
                    let new_threshold = original_threshold + delta;
                    if new_threshold > min_val && new_threshold < max_val {
                        thresholds[i] = new_threshold;
                        let entropy = self.compute_binning_entropy(feature_values, &thresholds)?;
                        if entropy < best_entropy {
                            best_entropy = entropy;
                            best_threshold = new_threshold;
                            improved = true;
                        }
                    }
                }

                thresholds[i] = best_threshold;
            }

            if !improved {
                break;
            }
        }

        Ok(thresholds)
    }

    fn compute_binning_entropy(
        &self,
        feature_values: &[Float],
        thresholds: &[Float],
    ) -> Result<Float> {
        let n_samples = feature_values.len();
        let n_bins = thresholds.len() + 1;
        let mut bin_counts = vec![0; n_bins];

        // Count samples in each bin
        for &value in feature_values {
            let mut bin = 0;
            for &threshold in thresholds {
                if value > threshold {
                    bin += 1;
                } else {
                    break;
                }
            }
            if bin < n_bins {
                bin_counts[bin] += 1;
            }
        }

        // Compute entropy
        let mut entropy = 0.0;
        for &count in &bin_counts {
            if count > 0 {
                let prob = count as Float / n_samples as Float;
                entropy -= prob * prob.ln();
            }
        }

        Ok(entropy)
    }
}

impl Estimator for InformationTheoreticDiscriminantAnalysis<Trained> {
    type Config = InformationTheoreticDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Predict<Array2<Float>, Array1<i32>> for InformationTheoreticDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().unwrap();

        let mut predictions = Vec::new();
        for row in probas.axis_iter(Axis(0)) {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions.push(classes[max_idx]);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl PredictProba<Array2<Float>, Array2<Float>>
    for InformationTheoreticDiscriminantAnalysis<Trained>
{
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let classes = self.classes_.as_ref().unwrap();
        let n_classes = classes.len();
        let thresholds = self.discretization_thresholds_.as_ref().unwrap();
        let feature_distributions = self.feature_distributions_.as_ref().unwrap();
        let class_priors = self.class_priors_.as_ref().unwrap();
        let entropy_weights = self.entropy_weights_.as_ref().unwrap();

        if x.ncols() != thresholds.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                thresholds.len(),
                x.ncols()
            )));
        }

        // Discretize input features
        let mut discretized_x = Array2::zeros((n_samples, x.ncols()));
        for feature_idx in 0..x.ncols() {
            let feature_column = x.column(feature_idx);
            let feature_thresholds = &thresholds[feature_idx];

            for (sample_idx, &value) in feature_column.iter().enumerate() {
                let mut bin = 0;
                for &threshold in feature_thresholds.iter() {
                    if value > threshold {
                        bin += 1;
                    } else {
                        break;
                    }
                }
                discretized_x[[sample_idx, feature_idx]] = bin;
            }
        }

        // Compute probabilities for each sample
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        for sample_idx in 0..n_samples {
            let sample = discretized_x.row(sample_idx);
            let mut log_probs = Array1::zeros(n_classes);

            for (class_idx, &class_label) in classes.iter().enumerate() {
                // Start with prior probability
                log_probs[class_idx] = class_priors[class_idx].ln();

                // Add likelihood for each feature
                for (feature_idx, &feature_bin) in sample.iter().enumerate() {
                    let key = (feature_bin, class_label);
                    let likelihood = feature_distributions[feature_idx]
                        .get(&key)
                        .unwrap_or(&self.config.entropy_regularization);

                    // Weight by information content
                    let weighted_likelihood = likelihood * entropy_weights[feature_idx];
                    log_probs[class_idx] +=
                        (weighted_likelihood + self.config.entropy_regularization).ln();
                }
            }

            // Convert log probabilities to probabilities using softmax
            let max_log_prob = log_probs.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_probs: Array1<Float> = log_probs.mapv(|x| (x - max_log_prob).exp());
            let sum_exp = exp_probs.sum();

            for (class_idx, &exp_prob) in exp_probs.iter().enumerate() {
                probabilities[[sample_idx, class_idx]] = exp_prob / sum_exp;
            }
        }

        Ok(probabilities)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for InformationTheoreticDiscriminantAnalysis<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_components = self.config.n_components.unwrap_or(
            selected_features
                .iter()
                .filter(|&&selected| selected)
                .count(),
        );

        // Get indices of selected features
        let selected_indices: Vec<usize> = selected_features
            .iter()
            .enumerate()
            .filter(|(_, &selected)| selected)
            .map(|(i, _)| i)
            .take(n_components)
            .collect();

        if selected_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected".to_string(),
            ));
        }

        // Select features based on information criteria
        let mut transformed = Array2::zeros((x.nrows(), selected_indices.len()));
        for (new_idx, &original_idx) in selected_indices.iter().enumerate() {
            transformed
                .column_mut(new_idx)
                .assign(&x.column(original_idx));
        }

        Ok(transformed)
    }
}

impl InformationTheoreticDiscriminantAnalysis<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }

    /// Get feature scores based on information criteria
    pub fn feature_scores(&self) -> &Array1<Float> {
        self.feature_scores_.as_ref().unwrap()
    }

    /// Get selected features mask
    pub fn selected_features(&self) -> &Array1<bool> {
        self.selected_features_.as_ref().unwrap()
    }

    /// Get discretization thresholds for each feature
    pub fn discretization_thresholds(&self) -> &[Array1<Float>] {
        self.discretization_thresholds_.as_ref().unwrap()
    }

    /// Get class priors
    pub fn class_priors(&self) -> &Array1<Float> {
        self.class_priors_.as_ref().unwrap()
    }

    /// Get mutual information matrix
    pub fn mutual_information_matrix(&self) -> &Array2<Float> {
        self.mutual_information_matrix_.as_ref().unwrap()
    }

    /// Get entropy-based feature weights
    pub fn entropy_weights(&self) -> &Array1<Float> {
        self.entropy_weights_.as_ref().unwrap()
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.n_features_.unwrap()
    }

    /// Get number of selected features
    pub fn n_selected_features(&self) -> usize {
        self.selected_features_
            .as_ref()
            .unwrap()
            .iter()
            .filter(|&&selected| selected)
            .count()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_information_theoretic_discriminant_analysis() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 3.0, 4.0, 5.0, 3.1, 4.1, 5.1, 3.2,
                4.2, 5.2,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new()
            .criterion(InformationCriterion::MutualInformation)
            .n_bins(3);

        let fitted = itda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.feature_scores().len(), 3);
    }

    #[test]
    fn test_information_gain_criterion() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 5.0, 5.0, 5.5, 5.5, 6.0, 6.0, 6.5, 6.5,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new()
            .criterion(InformationCriterion::InformationGain)
            .n_bins(4)
            .information_threshold(0.1);

        let fitted = itda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (8, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_feature_selection() {
        let x = Array2::from_shape_vec(
            (6, 4),
            vec![
                1.0, 2.0, 100.0, 200.0, // First two features discriminative
                1.1, 2.1, 101.0, 199.0, 1.2, 2.2, 102.0, 201.0, 3.0, 4.0, 103.0,
                198.0, // Last two features less discriminative
                3.1, 4.1, 104.0, 202.0, 3.2, 4.2, 105.0, 197.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new()
            .criterion(InformationCriterion::InformationGain)
            .information_threshold(0.01)
            .n_components(Some(2));

        let fitted = itda.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert!(fitted.n_selected_features() >= 1);
        assert!(transformed.ncols() <= 4);
        assert_eq!(transformed.nrows(), 6);
    }

    #[test]
    fn test_different_discretization_methods() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0, 70.0, 8.0,
                80.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);

        let methods = vec![
            DiscretizationMethod::EqualWidth,
            DiscretizationMethod::EqualFrequency,
        ];

        for method in methods {
            let itda = InformationTheoreticDiscriminantAnalysis::new()
                .discretization_method(method)
                .n_bins(4);

            let fitted = itda.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 8);
            assert_eq!(fitted.classes().len(), 2);
        }
    }

    #[test]
    fn test_mutual_information_matrix() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 3.0, 4.0, 5.0, 3.1, 4.1, 5.1, 3.2,
                4.2, 5.2,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new();
        let fitted = itda.fit(&x, &y).unwrap();

        let mi_matrix = fitted.mutual_information_matrix();
        assert_eq!(mi_matrix.dim(), (3, 3));

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(mi_matrix[[i, j]], mi_matrix[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_entropy_weights() {
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 100.0, // Second feature should have higher entropy weight
                2.0, 100.1, 5.0, 200.0, 6.0, 200.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new().n_bins(2);

        let fitted = itda.fit(&x, &y).unwrap();
        let weights = fitted.entropy_weights();

        assert_eq!(weights.len(), 2);
        assert_abs_diff_eq!(weights.sum(), 1.0, epsilon = 1e-6);

        // All weights should be non-negative
        for &weight in weights.iter() {
            assert!(weight >= 0.0);
        }
    }

    #[test]
    fn test_conditional_mutual_information() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 2.0, 1.0, 1.1, 2.1, 1.1, 1.2, 2.2, 1.2, 1.3, 2.3, 1.3, 3.0, 4.0, 3.0, 3.1,
                4.1, 3.1, 3.2, 4.2, 3.2, 3.3, 4.3, 3.3,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new()
            .criterion(InformationCriterion::ConditionalMutualInformation)
            .n_bins(3);

        let fitted = itda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.feature_scores().len(), 3);

        // All scores should be finite
        for &score in fitted.feature_scores().iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_joint_mutual_information() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 3.0, 4.0, 5.0, 3.1, 4.1, 5.1, 3.2,
                4.2, 5.2,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new()
            .criterion(InformationCriterion::JointMutualInformation)
            .n_bins(3);

        let fitted = itda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.feature_scores().len(), 3);
    }

    #[test]
    fn test_maximum_entropy_discrimination() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.1, 2.1, 1.2, 2.2, 3.0, 4.0, 3.1, 4.1, 3.2, 4.2],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new()
            .criterion(InformationCriterion::MaximumEntropy { lambda: 0.5 })
            .n_bins(3);

        let fitted = itda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_kmeans_discretization() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 8.0, 80.0, 9.0, 90.0, 10.0, 100.0,
                11.0, 110.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new()
            .discretization_method(DiscretizationMethod::KMeans)
            .n_bins(3);

        let fitted = itda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_entropy_based_discretization() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 8.0, 80.0, 9.0, 90.0, 10.0, 100.0,
                11.0, 110.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]);

        let itda = InformationTheoreticDiscriminantAnalysis::new()
            .discretization_method(DiscretizationMethod::EntropyBased)
            .n_bins(4);

        let fitted = itda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);
        assert_eq!(fitted.classes().len(), 2);

        // Check that discretization thresholds were created
        let thresholds = fitted.discretization_thresholds();
        assert_eq!(thresholds.len(), 2); // One threshold array per feature
        for threshold_array in thresholds {
            assert!(threshold_array.len() <= 3); // n_bins - 1 thresholds max
        }
    }

    #[test]
    fn test_combined_discretization_and_criteria() {
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 5.0, 10.0, 1.5, 5.5, 10.5, 2.0, 6.0, 11.0, 2.5, 6.5, 11.5, 3.0, 7.0, 12.0,
                7.0, 1.0, 2.0, 7.5, 1.5, 2.5, 8.0, 2.0, 3.0, 8.5, 2.5, 3.5, 9.0, 3.0, 4.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

        let methods = vec![
            DiscretizationMethod::EqualWidth,
            DiscretizationMethod::EqualFrequency,
            DiscretizationMethod::KMeans,
            DiscretizationMethod::EntropyBased,
        ];

        for method in methods {
            let itda = InformationTheoreticDiscriminantAnalysis::new()
                .discretization_method(method)
                .criterion(InformationCriterion::InformationGain)
                .n_bins(3);

            let fitted = itda.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 10);
            assert_eq!(fitted.classes().len(), 2);
            assert_eq!(fitted.feature_scores().len(), 3);
        }
    }
}
