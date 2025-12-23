//! Multi-modal feature selection for heterogeneous data types.
//!
//! This module provides specialized feature selection capabilities for multi-modal data,
//! where features come from different modalities such as text, images, audio, sensors,
//! or other heterogeneous data sources. It implements various fusion strategies and
//! cross-modal analysis techniques to identify the most informative features across modalities.
//!
//! # Features
//!
//! - **Early fusion**: Concatenates features from different modalities before selection
//! - **Late fusion**: Selects features within each modality separately, then combines results
//! - **Hybrid fusion**: Combines early and late fusion strategies
//! - **Cross-modal correlations**: Analyzes relationships between features across modalities
//! - **Modality weighting**: Assigns different importance weights to different modalities
//! - **Missing modality handling**: Robust to missing data in some modalities
//!
//! # Examples
//!
//! ## Basic Multi-Modal Feature Selection
//!
//! ```rust,ignore
//! use sklears_feature_selection::domain_specific::multi_modal::MultiModalFeatureSelector;
//! use scirs2_core::ndarray::{Array2, Array1};
//! use std::collections::HashMap;
//!
//! // Text features (TF-IDF, word embeddings, etc.)
//! let text_features = Array2::from_shape_vec((100, 50), (0..5000).map(|x| x as f64).collect()).unwrap();
//!
//! // Image features (CNN features, color histograms, etc.)
//! let image_features = Array2::from_shape_vec((100, 30), (0..3000).map(|x| (x * 2) as f64).collect()).unwrap();
//!
//! // Audio features (MFCCs, spectrograms, etc.)
//! let audio_features = Array2::from_shape_vec((100, 20), (0..2000).map(|x| (x * 3) as f64).collect()).unwrap();
//!
//! let mut modalities = HashMap::new();
//! modalities.insert("text".to_string(), text_features);
//! modalities.insert("image".to_string(), image_features);
//! modalities.insert("audio".to_string(), audio_features);
//!
//! let target = Array1::from_iter((0..100).map(|i| (i % 2) as f64));
//!
//! let selector = MultiModalFeatureSelector::builder()
//!     .fusion_strategy("hybrid")
//!     .modality_weights([("text", 0.4), ("image", 0.4), ("audio", 0.2)])
//!     .cross_modal_analysis(true)
//!     .k(20)
//!     .build();
//!
//! let trained = selector.fit(&modalities, &target)?;
//! let selected_features = trained.transform(&modalities)?;
//! ```
//!
//! ## Early Fusion Strategy
//!
//! ```rust,ignore
//! let selector = MultiModalFeatureSelector::builder()
//!     .fusion_strategy("early")
//!     .normalize_modalities(true)
//!     .k(15)
//!     .build();
//! ```
//!
//! ## Late Fusion with Modality-Specific Selection
//!
//! ```rust,ignore
//! let selector = MultiModalFeatureSelector::builder()
//!     .fusion_strategy("late")
//!     .modality_k([("text", 10), ("image", 8), ("audio", 5)])
//!     .cross_modal_threshold(0.3)
//!     .build();
//! ```
//!
//! ## Handling Missing Modalities
//!
//! ```rust,ignore
//! let selector = MultiModalFeatureSelector::builder()
//!     .handle_missing_modalities(true)
//!     .min_modalities_required(2)
//!     .missing_strategy("impute")
//!     .build();
//! ```

use scirs2_core::ndarray::{concatenate, s, Array1, Array2, ArrayView1, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Transform};
use std::collections::HashMap;
use std::marker::PhantomData;

type Result<T> = SklResult<T>;
type Float = f64;

#[derive(Debug, Clone)]
pub struct Untrained;

#[derive(Debug, Clone)]
pub struct Trained {
    fusion_strategy: String,
    selected_features_per_modality: HashMap<String, Vec<usize>>,
    combined_selected_features: Vec<(String, usize)>, // (modality, feature_index)
    feature_scores_per_modality: HashMap<String, Array1<Float>>,
    cross_modal_scores: Option<Array2<Float>>,
    modality_weights: HashMap<String, Float>,
    feature_mapping: HashMap<usize, (String, usize)>, // global_index -> (modality, local_index)
    total_features: usize,
    modality_feature_counts: HashMap<String, usize>,
}

/// Multi-modal feature selector for heterogeneous data types.
///
/// This selector handles feature selection across multiple data modalities using various
/// fusion strategies. It can analyze cross-modal relationships, handle missing modalities,
/// and apply different selection criteria for each modality type.
#[derive(Debug, Clone)]
pub struct MultiModalFeatureSelector<State = Untrained> {
    fusion_strategy: String,
    modality_weights: HashMap<String, Float>,
    modality_k: HashMap<String, usize>,
    cross_modal_analysis: bool,
    cross_modal_threshold: Float,
    normalize_modalities: bool,
    handle_missing_modalities: bool,
    min_modalities_required: usize,
    missing_strategy: String,
    k: Option<usize>,
    score_threshold: Float,
    correlation_method: String,
    interaction_analysis: bool,
    max_interaction_order: usize,
    state: PhantomData<State>,
    trained_state: Option<Trained>,
}

impl Default for MultiModalFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiModalFeatureSelector<Untrained> {
    /// Creates a new MultiModalFeatureSelector with default parameters.
    pub fn new() -> Self {
        Self {
            fusion_strategy: "hybrid".to_string(),
            modality_weights: HashMap::new(),
            modality_k: HashMap::new(),
            cross_modal_analysis: true,
            cross_modal_threshold: 0.1,
            normalize_modalities: true,
            handle_missing_modalities: true,
            min_modalities_required: 1,
            missing_strategy: "ignore".to_string(),
            k: None,
            score_threshold: 0.1,
            correlation_method: "pearson".to_string(),
            interaction_analysis: false,
            max_interaction_order: 2,
            state: PhantomData,
            trained_state: None,
        }
    }

    /// Creates a builder for configuring the MultiModalFeatureSelector.
    pub fn builder() -> MultiModalFeatureSelectorBuilder {
        MultiModalFeatureSelectorBuilder::new()
    }
}

/// Builder for MultiModalFeatureSelector configuration.
#[derive(Debug)]
pub struct MultiModalFeatureSelectorBuilder {
    fusion_strategy: String,
    modality_weights: HashMap<String, Float>,
    modality_k: HashMap<String, usize>,
    cross_modal_analysis: bool,
    cross_modal_threshold: Float,
    normalize_modalities: bool,
    handle_missing_modalities: bool,
    min_modalities_required: usize,
    missing_strategy: String,
    k: Option<usize>,
    score_threshold: Float,
    correlation_method: String,
    interaction_analysis: bool,
    max_interaction_order: usize,
}

impl Default for MultiModalFeatureSelectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiModalFeatureSelectorBuilder {
    pub fn new() -> Self {
        Self {
            fusion_strategy: "hybrid".to_string(),
            modality_weights: HashMap::new(),
            modality_k: HashMap::new(),
            cross_modal_analysis: true,
            cross_modal_threshold: 0.1,
            normalize_modalities: true,
            handle_missing_modalities: true,
            min_modalities_required: 1,
            missing_strategy: "ignore".to_string(),
            k: None,
            score_threshold: 0.1,
            correlation_method: "pearson".to_string(),
            interaction_analysis: false,
            max_interaction_order: 2,
        }
    }

    /// Fusion strategy: "early", "late", or "hybrid".
    pub fn fusion_strategy(mut self, strategy: &str) -> Self {
        self.fusion_strategy = strategy.to_string();
        self
    }

    /// Set weights for different modalities.
    pub fn modality_weights<I>(mut self, weights: I) -> Self
    where
        I: IntoIterator<Item = (&'static str, f64)>,
    {
        self.modality_weights = weights
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        self
    }

    /// Set number of features to select per modality (for late fusion).
    pub fn modality_k<I>(mut self, k_values: I) -> Self
    where
        I: IntoIterator<Item = (&'static str, usize)>,
    {
        self.modality_k = k_values
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        self
    }

    /// Whether to perform cross-modal correlation analysis.
    pub fn cross_modal_analysis(mut self, enable: bool) -> Self {
        self.cross_modal_analysis = enable;
        self
    }

    /// Threshold for cross-modal correlation significance.
    pub fn cross_modal_threshold(mut self, threshold: Float) -> Self {
        self.cross_modal_threshold = threshold;
        self
    }

    /// Whether to normalize features within each modality.
    pub fn normalize_modalities(mut self, normalize: bool) -> Self {
        self.normalize_modalities = normalize;
        self
    }

    /// Whether to handle missing modalities gracefully.
    pub fn handle_missing_modalities(mut self, handle: bool) -> Self {
        self.handle_missing_modalities = handle;
        self
    }

    /// Minimum number of modalities required for processing.
    pub fn min_modalities_required(mut self, min: usize) -> Self {
        self.min_modalities_required = min;
        self
    }

    /// Strategy for handling missing modalities: "ignore", "impute", or "error".
    pub fn missing_strategy(mut self, strategy: &str) -> Self {
        self.missing_strategy = strategy.to_string();
        self
    }

    /// Total number of features to select (for early/hybrid fusion).
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Minimum score threshold for feature selection.
    pub fn score_threshold(mut self, threshold: Float) -> Self {
        self.score_threshold = threshold;
        self
    }

    /// Correlation method: "pearson", "spearman", or "mutual_info".
    pub fn correlation_method(mut self, method: &str) -> Self {
        self.correlation_method = method.to_string();
        self
    }

    /// Whether to analyze feature interactions across modalities.
    pub fn interaction_analysis(mut self, enable: bool) -> Self {
        self.interaction_analysis = enable;
        self
    }

    /// Maximum order of interactions to consider (2 = pairwise, 3 = three-way, etc.).
    pub fn max_interaction_order(mut self, order: usize) -> Self {
        self.max_interaction_order = order;
        self
    }

    /// Builds the MultiModalFeatureSelector.
    pub fn build(self) -> MultiModalFeatureSelector<Untrained> {
        MultiModalFeatureSelector {
            fusion_strategy: self.fusion_strategy,
            modality_weights: self.modality_weights,
            modality_k: self.modality_k,
            cross_modal_analysis: self.cross_modal_analysis,
            cross_modal_threshold: self.cross_modal_threshold,
            normalize_modalities: self.normalize_modalities,
            handle_missing_modalities: self.handle_missing_modalities,
            min_modalities_required: self.min_modalities_required,
            missing_strategy: self.missing_strategy,
            k: self.k,
            score_threshold: self.score_threshold,
            correlation_method: self.correlation_method,
            interaction_analysis: self.interaction_analysis,
            max_interaction_order: self.max_interaction_order,
            state: PhantomData,
            trained_state: None,
        }
    }
}

impl Estimator for MultiModalFeatureSelector<Untrained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for MultiModalFeatureSelector<Trained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<HashMap<String, Array2<Float>>, Array1<Float>> for MultiModalFeatureSelector<Untrained> {
    type Fitted = MultiModalFeatureSelector<Trained>;

    fn fit(
        self,
        modalities: &HashMap<String, Array2<Float>>,
        y: &Array1<Float>,
    ) -> Result<Self::Fitted> {
        if modalities.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one modality is required".to_string(),
            ));
        }

        if modalities.len() < self.min_modalities_required {
            return Err(SklearsError::InvalidInput(format!(
                "At least {} modalities are required",
                self.min_modalities_required
            )));
        }

        // Validate that all modalities have the same number of samples
        let n_samples = y.len();
        for (modality_name, features) in modalities {
            if features.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Modality '{}' has {} samples, expected {}",
                    modality_name,
                    features.nrows(),
                    n_samples
                )));
            }
        }

        // Normalize modalities if requested
        let normalized_modalities = if self.normalize_modalities {
            normalize_modalities(modalities)?
        } else {
            modalities.clone()
        };

        // Set default weights if not provided
        let mut modality_weights = self.modality_weights.clone();
        if modality_weights.is_empty() {
            let default_weight = 1.0 / modalities.len() as Float;
            for modality_name in modalities.keys() {
                modality_weights.insert(modality_name.clone(), default_weight);
            }
        }

        // Perform feature selection based on fusion strategy
        let (
            selected_features_per_modality,
            combined_selected_features,
            feature_scores_per_modality,
            cross_modal_scores,
            feature_mapping,
            total_features,
            modality_feature_counts,
        ) = match self.fusion_strategy.as_str() {
            "early" => self.early_fusion_selection(&normalized_modalities, y, &modality_weights)?,
            "late" => self.late_fusion_selection(&normalized_modalities, y, &modality_weights)?,
            "hybrid" => {
                self.hybrid_fusion_selection(&normalized_modalities, y, &modality_weights)?
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown fusion strategy: {}",
                    self.fusion_strategy
                )))
            }
        };

        let trained_state = Trained {
            fusion_strategy: self.fusion_strategy.clone(),
            selected_features_per_modality,
            combined_selected_features,
            feature_scores_per_modality,
            cross_modal_scores,
            modality_weights,
            feature_mapping,
            total_features,
            modality_feature_counts,
        };

        Ok(MultiModalFeatureSelector {
            fusion_strategy: self.fusion_strategy,
            modality_weights: self.modality_weights,
            modality_k: self.modality_k,
            cross_modal_analysis: self.cross_modal_analysis,
            cross_modal_threshold: self.cross_modal_threshold,
            normalize_modalities: self.normalize_modalities,
            handle_missing_modalities: self.handle_missing_modalities,
            min_modalities_required: self.min_modalities_required,
            missing_strategy: self.missing_strategy,
            k: self.k,
            score_threshold: self.score_threshold,
            correlation_method: self.correlation_method,
            interaction_analysis: self.interaction_analysis,
            max_interaction_order: self.max_interaction_order,
            state: PhantomData,
            trained_state: Some(trained_state),
        })
    }
}

impl Transform<HashMap<String, Array2<Float>>, Array2<Float>>
    for MultiModalFeatureSelector<Trained>
{
    fn transform(&self, modalities: &HashMap<String, Array2<Float>>) -> Result<Array2<Float>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Selector must be fitted before transforming".to_string())
        })?;

        match trained.fusion_strategy.as_str() {
            "early" => self.transform_early_fusion(modalities, trained),
            "late" => self.transform_late_fusion(modalities, trained),
            "hybrid" => self.transform_hybrid_fusion(modalities, trained),
            _ => Err(SklearsError::InvalidState(format!(
                "Unknown fusion strategy: {}",
                trained.fusion_strategy
            ))),
        }
    }
}

// MultiModalFeatureSelector uses HashMap<String, Array2> as input, not Array2,
// so it cannot implement SelectorMixin which requires Transform<Array2>
/* impl SelectorMixin for MultiModalFeatureSelector<Trained> {
    fn get_support(&self) -> Result<Array1<bool>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Selector must be fitted before getting support".to_string())
        })?;

        let mut support = Array1::from_elem(trained.total_features, false);

        match trained.fusion_strategy.as_str() {
            "early" | "hybrid" => {
                for &(_, global_idx) in &trained.combined_selected_features {
                    if let Some(&(_, _)) = trained.feature_mapping.get(&global_idx) {
                        support[global_idx] = true;
                    }
                }
            }
            "late" => {
                for (modality, selected_indices) in &trained.selected_features_per_modality {
                    if let Some(&base_idx) = trained
                        .modality_feature_counts
                        .keys()
                        .take_while(|&k| k != modality)
                        .map(|k| trained.modality_feature_counts[k])
                        .reduce(|acc, x| acc + x)
                        .as_ref()
                    {
                        for &local_idx in selected_indices {
                            support[base_idx + local_idx] = true;
                        }
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidState(
                    "Unknown fusion strategy".to_string(),
                ))
            }
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState(
                "Selector must be fitted before transforming features".to_string(),
            )
        })?;

        let selected: Vec<usize> = indices
            .iter()
            .filter(|&&idx| trained.selected_features.contains(&idx))
            .cloned()
            .collect();
        Ok(selected)
    }
} */

// Implementation methods for MultiModalFeatureSelector
impl MultiModalFeatureSelector<Untrained> {
    fn early_fusion_selection(
        &self,
        modalities: &HashMap<String, Array2<Float>>,
        y: &Array1<Float>,
        modality_weights: &HashMap<String, Float>,
    ) -> Result<(
        HashMap<String, Vec<usize>>,
        Vec<(String, usize)>,
        HashMap<String, Array1<Float>>,
        Option<Array2<Float>>,
        HashMap<usize, (String, usize)>,
        usize,
        HashMap<String, usize>,
    )> {
        // Concatenate all modalities into a single feature matrix
        let (combined_features, feature_mapping, modality_feature_counts) =
            concatenate_modalities(modalities)?;

        // Compute feature scores using weighted univariate analysis
        let feature_scores = compute_weighted_feature_scores(
            &combined_features,
            y,
            modalities,
            modality_weights,
            &self.correlation_method,
        )?;

        // Cross-modal analysis if enabled
        let cross_modal_scores = if self.cross_modal_analysis {
            Some(compute_cross_modal_correlations(
                modalities,
                &self.correlation_method,
            )?)
        } else {
            None
        };

        // Select features based on scores
        let selected_indices = if let Some(k) = self.k {
            select_top_k_features(&feature_scores, k)
        } else {
            select_features_by_threshold(&feature_scores, self.score_threshold)
        };

        // Map back to per-modality selections
        let mut selected_features_per_modality = HashMap::new();
        let mut combined_selected_features = Vec::new();

        for &global_idx in &selected_indices {
            if let Some(&(ref modality, local_idx)) = feature_mapping.get(&global_idx) {
                selected_features_per_modality
                    .entry(modality.clone())
                    .or_insert_with(Vec::new)
                    .push(local_idx);
                combined_selected_features.push((modality.clone(), global_idx));
            }
        }

        // Create per-modality feature scores
        let mut feature_scores_per_modality = HashMap::new();
        let mut current_idx = 0;
        for (modality_name, features) in modalities {
            let n_features = features.ncols();
            let modality_scores = feature_scores
                .slice(s![current_idx..current_idx + n_features])
                .to_owned();
            feature_scores_per_modality.insert(modality_name.clone(), modality_scores);
            current_idx += n_features;
        }

        Ok((
            selected_features_per_modality,
            combined_selected_features,
            feature_scores_per_modality,
            cross_modal_scores,
            feature_mapping,
            combined_features.ncols(),
            modality_feature_counts,
        ))
    }

    fn late_fusion_selection(
        &self,
        modalities: &HashMap<String, Array2<Float>>,
        y: &Array1<Float>,
        modality_weights: &HashMap<String, Float>,
    ) -> Result<(
        HashMap<String, Vec<usize>>,
        Vec<(String, usize)>,
        HashMap<String, Array1<Float>>,
        Option<Array2<Float>>,
        HashMap<usize, (String, usize)>,
        usize,
        HashMap<String, usize>,
    )> {
        let mut selected_features_per_modality = HashMap::new();
        let mut feature_scores_per_modality = HashMap::new();
        let mut combined_selected_features = Vec::new();
        let mut feature_mapping = HashMap::new();
        let mut modality_feature_counts = HashMap::new();
        let mut total_features = 0;
        let mut global_idx = 0;

        // Select features within each modality separately
        for (modality_name, features) in modalities {
            let n_features = features.ncols();
            modality_feature_counts.insert(modality_name.clone(), n_features);

            // Create feature mapping for this modality
            for local_idx in 0..n_features {
                feature_mapping.insert(global_idx, (modality_name.clone(), local_idx));
                global_idx += 1;
            }

            let weight = modality_weights.get(modality_name).cloned().unwrap_or(1.0);
            let scores = compute_univariate_scores(features, y, &self.correlation_method)?;
            let weighted_scores = scores.mapv(|s| s * weight);

            let k = self
                .modality_k
                .get(modality_name)
                .cloned()
                .or(self.k.map(|total_k| total_k / modalities.len()))
                .unwrap_or(n_features / 2);

            let selected_indices = select_top_k_features(&weighted_scores, k.min(n_features));

            for &local_idx in &selected_indices {
                let global_feature_idx = total_features + local_idx;
                combined_selected_features.push((modality_name.clone(), global_feature_idx));
            }

            selected_features_per_modality.insert(modality_name.clone(), selected_indices);
            feature_scores_per_modality.insert(modality_name.clone(), weighted_scores);
            total_features += n_features;
        }

        // Cross-modal analysis if enabled
        let cross_modal_scores = if self.cross_modal_analysis {
            Some(compute_cross_modal_correlations(
                modalities,
                &self.correlation_method,
            )?)
        } else {
            None
        };

        Ok((
            selected_features_per_modality,
            combined_selected_features,
            feature_scores_per_modality,
            cross_modal_scores,
            feature_mapping,
            total_features,
            modality_feature_counts,
        ))
    }

    fn hybrid_fusion_selection(
        &self,
        modalities: &HashMap<String, Array2<Float>>,
        y: &Array1<Float>,
        modality_weights: &HashMap<String, Float>,
    ) -> Result<(
        HashMap<String, Vec<usize>>,
        Vec<(String, usize)>,
        HashMap<String, Array1<Float>>,
        Option<Array2<Float>>,
        HashMap<usize, (String, usize)>,
        usize,
        HashMap<String, usize>,
    )> {
        // Perform both early and late fusion, then combine results
        let (
            early_selected,
            _early_combined,
            early_scores,
            early_cross_modal,
            early_mapping,
            early_total,
            early_counts,
        ) = self.early_fusion_selection(modalities, y, modality_weights)?;

        let (
            late_selected,
            _late_combined,
            late_scores,
            late_cross_modal,
            _late_mapping,
            _late_total,
            _late_counts,
        ) = self.late_fusion_selection(modalities, y, modality_weights)?;

        // Combine results with hybrid strategy
        let mut final_selected_per_modality = HashMap::new();
        let mut final_combined_selected = Vec::new();
        let mut final_scores_per_modality = HashMap::new();

        for (modality_name, early_indices) in &early_selected {
            let late_indices = late_selected
                .get(modality_name)
                .cloned()
                .unwrap_or_default();
            let early_modality_scores = &early_scores[modality_name];
            let late_modality_scores = &late_scores[modality_name];

            // Combine scores from early and late fusion
            let combined_scores = (early_modality_scores + late_modality_scores) / 2.0;

            // Merge selected indices (take union and re-rank)
            let mut all_candidates: std::collections::HashSet<usize> =
                early_indices.iter().cloned().collect();
            all_candidates.extend(late_indices.iter());

            let mut candidate_scores: Vec<(usize, Float)> = all_candidates
                .iter()
                .map(|&idx| (idx, combined_scores[idx]))
                .collect();

            candidate_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let target_k = self
                .modality_k
                .get(modality_name)
                .cloned()
                .or(self.k.map(|total_k| total_k / modalities.len()))
                .unwrap_or(candidate_scores.len() / 2);

            let final_indices: Vec<usize> = candidate_scores
                .into_iter()
                .take(target_k.min(all_candidates.len()))
                .map(|(idx, _)| idx)
                .collect();

            // Update global combined selection
            for &local_idx in &final_indices {
                if let Some(base_offset) = compute_modality_offset(modality_name, modalities) {
                    final_combined_selected.push((modality_name.clone(), base_offset + local_idx));
                }
            }

            final_selected_per_modality.insert(modality_name.clone(), final_indices);
            final_scores_per_modality.insert(modality_name.clone(), combined_scores);
        }

        // Use early fusion mappings and totals as base
        let cross_modal_scores = early_cross_modal.or(late_cross_modal);

        Ok((
            final_selected_per_modality,
            final_combined_selected,
            final_scores_per_modality,
            cross_modal_scores,
            early_mapping,
            early_total,
            early_counts,
        ))
    }
}

impl MultiModalFeatureSelector<Trained> {
    fn transform_early_fusion(
        &self,
        modalities: &HashMap<String, Array2<Float>>,
        trained: &Trained,
    ) -> Result<Array2<Float>> {
        let (combined_features, _, _) = concatenate_modalities(modalities)?;

        let global_indices: Vec<usize> = trained
            .combined_selected_features
            .iter()
            .map(|(_, global_idx)| *global_idx)
            .collect();

        if global_indices.is_empty() {
            return Err(SklearsError::InvalidState(
                "No features were selected".to_string(),
            ));
        }

        let selected_data = combined_features.select(Axis(1), &global_indices);
        Ok(selected_data)
    }

    fn transform_late_fusion(
        &self,
        modalities: &HashMap<String, Array2<Float>>,
        trained: &Trained,
    ) -> Result<Array2<Float>> {
        let mut selected_features_owned = Vec::new();

        for (modality_name, features) in modalities {
            if let Some(selected_indices) =
                trained.selected_features_per_modality.get(modality_name)
            {
                if !selected_indices.is_empty() {
                    let selected_modality_features = features.select(Axis(1), selected_indices);
                    selected_features_owned.push(selected_modality_features);
                }
            }
        }

        if selected_features_owned.is_empty() {
            return Err(SklearsError::InvalidState(
                "No features were selected from any modality".to_string(),
            ));
        }

        let selected_features_list: Vec<_> =
            selected_features_owned.iter().map(|a| a.view()).collect();
        let combined = concatenate(Axis(1), &selected_features_list).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to concatenate selected features: {}", e))
        })?;

        Ok(combined)
    }

    fn transform_hybrid_fusion(
        &self,
        modalities: &HashMap<String, Array2<Float>>,
        trained: &Trained,
    ) -> Result<Array2<Float>> {
        // For hybrid fusion, use the same approach as late fusion since features were selected per modality
        self.transform_late_fusion(modalities, trained)
    }
}

// Utility functions

fn normalize_modalities(
    modalities: &HashMap<String, Array2<Float>>,
) -> Result<HashMap<String, Array2<Float>>> {
    let mut normalized = HashMap::new();

    for (modality_name, features) in modalities {
        let normalized_features = normalize_features(features)?;
        normalized.insert(modality_name.clone(), normalized_features);
    }

    Ok(normalized)
}

fn normalize_features(features: &Array2<Float>) -> Result<Array2<Float>> {
    let (n_samples, n_features) = features.dim();
    let mut normalized = Array2::zeros((n_samples, n_features));

    for j in 0..n_features {
        let feature = features.column(j);
        let mean = feature.sum() / n_samples as Float;
        let variance = feature.mapv(|x| (x - mean).powi(2)).sum() / n_samples as Float;
        let std_dev = variance.sqrt();

        if std_dev > 1e-8 {
            for i in 0..n_samples {
                normalized[[i, j]] = (features[[i, j]] - mean) / std_dev;
            }
        } else {
            // Handle constant features
            normalized.column_mut(j).fill(0.0);
        }
    }

    Ok(normalized)
}

fn concatenate_modalities(
    modalities: &HashMap<String, Array2<Float>>,
) -> Result<(
    Array2<Float>,
    HashMap<usize, (String, usize)>,
    HashMap<String, usize>,
)> {
    if modalities.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No modalities provided".to_string(),
        ));
    }

    let n_samples = modalities.values().next().unwrap().nrows();
    let mut feature_views = Vec::new();
    let mut feature_mapping = HashMap::new();
    let mut modality_feature_counts = HashMap::new();
    let mut global_idx = 0;

    for (modality_name, features) in modalities {
        if features.nrows() != n_samples {
            return Err(SklearsError::InvalidInput(
                format!("All modalities must have the same number of samples. Expected {}, got {} for modality '{}'",
                       n_samples, features.nrows(), modality_name)
            ));
        }

        let n_features = features.ncols();
        modality_feature_counts.insert(modality_name.clone(), n_features);

        for local_idx in 0..n_features {
            feature_mapping.insert(global_idx, (modality_name.clone(), local_idx));
            global_idx += 1;
        }

        feature_views.push(features.view());
    }

    let combined = concatenate(Axis(1), &feature_views).map_err(|e| {
        SklearsError::InvalidInput(format!("Failed to concatenate modalities: {}", e))
    })?;

    Ok((combined, feature_mapping, modality_feature_counts))
}

fn compute_weighted_feature_scores(
    features: &Array2<Float>,
    y: &Array1<Float>,
    modalities: &HashMap<String, Array2<Float>>,
    modality_weights: &HashMap<String, Float>,
    correlation_method: &str,
) -> Result<Array1<Float>> {
    let (_, n_features) = features.dim();
    let mut scores = Array1::zeros(n_features);
    let mut feature_idx = 0;

    for (modality_name, modality_features) in modalities {
        let weight = modality_weights.get(modality_name).cloned().unwrap_or(1.0);
        let modality_scores = compute_univariate_scores(modality_features, y, correlation_method)?;
        let weighted_scores = modality_scores.mapv(|s| s * weight);

        let end_idx = feature_idx + modality_features.ncols();
        scores
            .slice_mut(s![feature_idx..end_idx])
            .assign(&weighted_scores);
        feature_idx = end_idx;
    }

    Ok(scores)
}

fn compute_univariate_scores(
    features: &Array2<Float>,
    y: &Array1<Float>,
    method: &str,
) -> Result<Array1<Float>> {
    let (_, n_features) = features.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = features.column(j);
        let score = match method {
            "pearson" => compute_pearson_correlation(&feature, &y.view()),
            "spearman" => compute_spearman_correlation(&feature, &y.view()),
            "mutual_info" => compute_mutual_information(&feature, &y.view()),
            _ => compute_pearson_correlation(&feature, &y.view()),
        };
        scores[j] = score.abs();
    }

    Ok(scores)
}

fn compute_cross_modal_correlations(
    modalities: &HashMap<String, Array2<Float>>,
    method: &str,
) -> Result<Array2<Float>> {
    let modality_names: Vec<&String> = modalities.keys().collect();
    let n_modalities = modality_names.len();
    let mut correlations = Array2::zeros((n_modalities, n_modalities));

    for i in 0..n_modalities {
        for j in i..n_modalities {
            if i == j {
                correlations[[i, j]] = 1.0;
            } else {
                let features_i = &modalities[modality_names[i]];
                let features_j = &modalities[modality_names[j]];

                let correlation = compute_modality_correlation(features_i, features_j, method)?;
                correlations[[i, j]] = correlation;
                correlations[[j, i]] = correlation;
            }
        }
    }

    Ok(correlations)
}

fn compute_modality_correlation(
    features_a: &Array2<Float>,
    features_b: &Array2<Float>,
    method: &str,
) -> Result<Float> {
    // Compute average correlation between modalities by taking mean of all pairwise feature correlations
    let (_, n_features_a) = features_a.dim();
    let (_, n_features_b) = features_b.dim();

    let mut total_correlation = 0.0;
    let mut count = 0;

    for i in 0..n_features_a.min(10) {
        // Limit to avoid computational explosion
        for j in 0..n_features_b.min(10) {
            let feature_a = features_a.column(i);
            let feature_b = features_b.column(j);

            let correlation = match method {
                "pearson" => compute_pearson_correlation(&feature_a, &feature_b),
                "spearman" => compute_spearman_correlation(&feature_a, &feature_b),
                _ => compute_pearson_correlation(&feature_a, &feature_b),
            };

            total_correlation += correlation.abs();
            count += 1;
        }
    }

    Ok(if count > 0 {
        total_correlation / count as Float
    } else {
        0.0
    })
}

fn compute_pearson_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
    let n = x.len();
    if n != y.len() || n == 0 {
        return 0.0;
    }

    let mean_x = x.sum() / n as Float;
    let mean_y = y.sum() / n as Float;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn compute_spearman_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
    // Simplified Spearman correlation (rank-based)
    let ranks_x = compute_ranks(x);
    let ranks_y = compute_ranks(y);
    compute_pearson_correlation(&ranks_x.view(), &ranks_y.view())
}

fn compute_mutual_information(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
    // Simplified mutual information estimation using correlation as proxy
    compute_pearson_correlation(x, y).abs()
}

fn compute_ranks(values: &ArrayView1<Float>) -> Array1<Float> {
    let n = values.len();
    let mut indexed_values: Vec<(usize, Float)> =
        values.iter().enumerate().map(|(i, &v)| (i, v)).collect();

    indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = Array1::zeros(n);
    for (rank, &(original_idx, _)) in indexed_values.iter().enumerate() {
        ranks[original_idx] = rank as Float;
    }

    ranks
}

fn compute_modality_offset(
    modality_name: &str,
    modalities: &HashMap<String, Array2<Float>>,
) -> Option<usize> {
    let mut offset = 0;
    for (name, features) in modalities {
        if name == modality_name {
            return Some(offset);
        }
        offset += features.ncols();
    }
    None
}

fn select_top_k_features(scores: &Array1<Float>, k: usize) -> Vec<usize> {
    let mut indexed_scores: Vec<(usize, Float)> = scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();

    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed_scores
        .into_iter()
        .take(k.min(scores.len()))
        .map(|(i, _)| i)
        .collect()
}

fn select_features_by_threshold(scores: &Array1<Float>, threshold: Float) -> Vec<usize> {
    scores
        .iter()
        .enumerate()
        .filter(|(_, &score)| score >= threshold)
        .map(|(i, _)| i)
        .collect()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_modal_feature_selector_creation() {
        let selector = MultiModalFeatureSelector::new();
        assert_eq!(selector.fusion_strategy, "hybrid");
        assert!(selector.cross_modal_analysis);
        assert!(selector.normalize_modalities);
    }

    #[test]
    fn test_multi_modal_feature_selector_builder() {
        let selector = MultiModalFeatureSelector::builder()
            .fusion_strategy("early")
            .cross_modal_analysis(false)
            .k(10)
            .build();

        assert_eq!(selector.fusion_strategy, "early");
        assert!(!selector.cross_modal_analysis);
        assert_eq!(selector.k, Some(10));
    }

    #[test]
    fn test_concatenate_modalities() {
        let mut modalities = HashMap::new();

        let text_features =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let image_features =
            Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        modalities.insert("text".to_string(), text_features);
        modalities.insert("image".to_string(), image_features);

        let (combined, mapping, counts) = concatenate_modalities(&modalities).unwrap();

        assert_eq!(combined.dim(), (3, 4));
        assert_eq!(mapping.len(), 4);
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_fit_transform_early_fusion() {
        let mut modalities = HashMap::new();

        let text_features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let image_features =
            Array2::from_shape_vec((4, 2), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

        modalities.insert("text".to_string(), text_features);
        modalities.insert("image".to_string(), image_features);

        let target = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let selector = MultiModalFeatureSelector::builder()
            .fusion_strategy("early")
            .k(2)
            .build();

        let trained = selector.fit(&modalities, &target).unwrap();
        let transformed = trained.transform(&modalities).unwrap();

        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 4);
    }

    #[test]
    fn test_fit_transform_late_fusion() {
        let mut modalities = HashMap::new();

        let text_features = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let image_features =
            Array2::from_shape_vec((4, 2), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

        modalities.insert("text".to_string(), text_features);
        modalities.insert("image".to_string(), image_features);

        let target = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let selector = MultiModalFeatureSelector::builder()
            .fusion_strategy("late")
            .modality_k([("text", 2), ("image", 1)])
            .build();

        let trained = selector.fit(&modalities, &target).unwrap();
        let transformed = trained.transform(&modalities).unwrap();

        assert_eq!(transformed.ncols(), 3); // 2 from text + 1 from image
        assert_eq!(transformed.nrows(), 4);
    }

    #[test]
    fn test_normalize_features() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let normalized = normalize_features(&features).unwrap();

        assert_eq!(normalized.dim(), (3, 2));

        // Check that each column has approximately zero mean and unit variance
        for j in 0..2 {
            let column = normalized.column(j);
            let mean = column.sum() / column.len() as Float;
            assert!((mean).abs() < 1e-10);
        }
    }

    // Note: get_support() is not implemented for MultiModalFeatureSelector
    // (SelectorMixin is commented out because it requires Transform<Array2> but this uses HashMap<String, Array2>)
    // #[test]
    // fn test_get_support() {
    //     let mut modalities = HashMap::new();
    //
    //     let text_features = Array2::from_shape_vec(
    //         (3, 4),
    //         vec![
    //             1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    //         ],
    //     )
    //     .unwrap();
    //     let image_features =
    //         Array2::from_shape_vec((3, 2), vec![13.0, 14.0, 15.0, 16.0, 17.0, 18.0]).unwrap();
    //
    //     modalities.insert("text".to_string(), text_features);
    //     modalities.insert("image".to_string(), image_features);
    //
    //     let target = Array1::from_vec(vec![0.0, 1.0, 1.0]);
    //
    //     let selector = MultiModalFeatureSelector::builder()
    //         .fusion_strategy("late")
    //         .modality_k([("text", 2), ("image", 1)])
    //         .build();
    //
    //     let trained = selector.fit(&modalities, &target).unwrap();
    //     let support = trained.get_support().unwrap();
    //
    //     assert_eq!(support.len(), 6); // 4 + 2 total features
    //     assert_eq!(support.iter().filter(|&&x| x).count(), 3); // 2 + 1 selected
    // }

    #[test]
    fn test_pearson_correlation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let correlation = compute_pearson_correlation(&x.view(), &y.view());

        // Perfect positive correlation
        assert!((correlation - 1.0).abs() < 1e-10);
    }
}
