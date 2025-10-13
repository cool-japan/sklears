//! Advanced NLP feature selection for sophisticated text analysis.
//!
//! This module provides advanced natural language processing feature selection capabilities
//! that go beyond basic text features. It includes syntactic analysis, semantic understanding,
//! discourse structure, pragmatic features, and contextual embeddings for sophisticated
//! text modeling tasks such as sentiment analysis, document classification, and text generation.
//!
//! # Features
//!
//! - **Syntactic features**: Dependency parsing, constituency parsing, POS tagging patterns
//! - **Semantic features**: Word embeddings, semantic roles, named entity recognition
//! - **Discourse features**: Coherence, cohesion, rhetorical structure, topic modeling
//! - **Pragmatic features**: Sentiment, emotion, subjectivity, stance detection
//! - **Contextual embeddings**: BERT-like features, attention patterns, transformer outputs
//! - **Cross-lingual features**: Multilingual embeddings and transfer learning
//!
//! # Examples
//!
//! ## Syntactic Feature Selection
//!
//! ```rust,ignore
//! use sklears_feature_selection::domain_specific::advanced_nlp::AdvancedNLPFeatureSelector;
//! use scirs2_core::ndarray::{Array2, Array1};
//!
//! // NLP features: [pos_tags, dependency_relations, syntactic_complexity, ...]
//! let nlp_features = Array2::from_shape_vec((500, 100),
//!     (0..50000).map(|x| (x as f64) * 0.001).collect()).unwrap();
//!
//! // Target labels (e.g., document categories, sentiment scores)
//! let labels = Array1::from_iter((0..500).map(|i|
//!     match i % 4 { 0 => 0.0, 1 => 1.0, 2 => 2.0, _ => 3.0 }
//! ));
//!
//! let selector = AdvancedNLPFeatureSelector::builder()
//!     .feature_type("syntactic")
//!     .include_pos_patterns(true)
//!     .include_dependency_features(true)
//!     .include_syntactic_complexity(true)
//!     .pos_ngram_range((1, 3))
//!     .dependency_window_size(5)
//!     .k(20)
//!     .build();
//!
//! let trained = selector.fit(&nlp_features, &labels)?;
//! let selected_features = trained.transform(&nlp_features)?;
//! ```
//!
//! ## Semantic Feature Selection
//!
//! ```rust,ignore
//! let selector = AdvancedNLPFeatureSelector::builder()
//!     .feature_type("semantic")
//!     .include_word_embeddings(true)
//!     .include_semantic_roles(true)
//!     .include_named_entities(true)
//!     .embedding_dimensions(300)
//!     .semantic_similarity_threshold(0.7)
//!     .entity_type_importance(0.8)
//!     .build();
//! ```
//!
//! ## Discourse and Pragmatic Analysis
//!
//! ```rust,ignore
//! let selector = AdvancedNLPFeatureSelector::builder()
//!     .feature_type("discourse")
//!     .include_coherence_features(true)
//!     .include_rhetorical_structure(true)
//!     .include_topic_modeling(true)
//!     .coherence_window_size(10)
//!     .topic_model_dimensions(50)
//!     .rhetorical_relation_types(vec!["contrast", "elaboration", "cause"])
//!     .build();
//! ```
//!
//! ## Contextual Embeddings and Transformers
//!
//! ```rust,ignore
//! let selector = AdvancedNLPFeatureSelector::builder()
//!     .feature_type("contextual")
//!     .include_transformer_features(true)
//!     .include_attention_patterns(true)
//!     .transformer_layers(vec![6, 9, 12])
//!     .attention_head_analysis(true)
//!     .contextual_similarity_weight(0.6)
//!     .build();
//! ```

use crate::base::SelectorMixin;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::{Estimator, Fit, Transform};
use std::collections::HashMap;
use std::marker::PhantomData;

type Result<T> = SklResult<T>;

type Float = f64;

/// Strategy for NLP feature selection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NLPStrategy {
    /// InformationTheoretic
    InformationTheoretic,
    /// SyntacticAnalysis
    SyntacticAnalysis,
    /// SemanticAnalysis
    SemanticAnalysis,
    /// DiscourseAnalysis
    DiscourseAnalysis,
    /// PragmaticAnalysis
    PragmaticAnalysis,
    /// TransformerBased
    TransformerBased,
}

#[derive(Debug, Clone)]
pub struct Untrained;

#[derive(Debug, Clone)]
pub struct Trained {
    selected_features: Vec<usize>,
    feature_scores: Array1<Float>,
    syntactic_scores: Option<HashMap<String, Array1<Float>>>,
    semantic_scores: Option<HashMap<String, Array1<Float>>>,
    discourse_scores: Option<HashMap<String, Array1<Float>>>,
    pragmatic_scores: Option<HashMap<String, Array1<Float>>>,
    contextual_scores: Option<HashMap<String, Array1<Float>>>,
    attention_weights: Option<Array2<Float>>,
    topic_distributions: Option<Array2<Float>>,
    semantic_similarity_matrix: Option<Array2<Float>>,
    n_features: usize,
    feature_type: String,
}

/// Advanced NLP feature selector for sophisticated text analysis.
///
/// This selector provides comprehensive natural language processing capabilities for
/// feature selection, including syntactic, semantic, discourse, and pragmatic analysis.
/// It supports modern NLP techniques including transformer-based features, attention
/// mechanisms, and contextual embeddings for state-of-the-art text modeling.
#[derive(Debug, Clone)]
pub struct AdvancedNLPFeatureSelector<State = Untrained> {
    feature_type: String,
    include_pos_patterns: bool,
    include_dependency_features: bool,
    include_syntactic_complexity: bool,
    include_word_embeddings: bool,
    include_semantic_roles: bool,
    include_named_entities: bool,
    include_coherence_features: bool,
    include_rhetorical_structure: bool,
    include_topic_modeling: bool,
    include_sentiment_features: bool,
    include_emotion_features: bool,
    include_transformer_features: bool,
    include_attention_patterns: bool,
    pos_ngram_range: (usize, usize),
    dependency_window_size: usize,
    embedding_dimensions: usize,
    semantic_similarity_threshold: Float,
    entity_type_importance: Float,
    coherence_window_size: usize,
    topic_model_dimensions: usize,
    rhetorical_relation_types: Vec<String>,
    transformer_layers: Vec<usize>,
    attention_head_analysis: bool,
    contextual_similarity_weight: Float,
    cross_lingual_features: bool,
    language_transfer_weight: Float,
    discourse_marker_weight: Float,
    pub k: usize,
    score_threshold: Float,
    strategy: NLPStrategy,
    state: PhantomData<State>,
    trained_state: Option<Trained>,
}

impl Default for AdvancedNLPFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedNLPFeatureSelector<Untrained> {
    /// Creates a new AdvancedNLPFeatureSelector with default parameters.
    pub fn new() -> Self {
        Self {
            feature_type: "syntactic".to_string(),
            include_pos_patterns: true,
            include_dependency_features: true,
            include_syntactic_complexity: true,
            include_word_embeddings: false,
            include_semantic_roles: false,
            include_named_entities: false,
            include_coherence_features: false,
            include_rhetorical_structure: false,
            include_topic_modeling: false,
            include_sentiment_features: false,
            include_emotion_features: false,
            include_transformer_features: false,
            include_attention_patterns: false,
            pos_ngram_range: (1, 3),
            dependency_window_size: 5,
            embedding_dimensions: 300,
            semantic_similarity_threshold: 0.7,
            entity_type_importance: 0.8,
            coherence_window_size: 10,
            topic_model_dimensions: 50,
            rhetorical_relation_types: vec!["elaboration".to_string(), "contrast".to_string()],
            transformer_layers: vec![6, 9, 12],
            attention_head_analysis: false,
            contextual_similarity_weight: 0.6,
            cross_lingual_features: false,
            language_transfer_weight: 0.3,
            discourse_marker_weight: 0.4,
            k: 10,
            score_threshold: 0.1,
            strategy: NLPStrategy::InformationTheoretic,
            state: PhantomData,
            trained_state: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the NLP strategy
    pub fn strategy(mut self, strategy: NLPStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Creates a builder for configuring the AdvancedNLPFeatureSelector.
    pub fn builder() -> AdvancedNLPFeatureSelectorBuilder {
        AdvancedNLPFeatureSelectorBuilder::new()
    }
}

/// Builder for AdvancedNLPFeatureSelector configuration.
#[derive(Debug)]
pub struct AdvancedNLPFeatureSelectorBuilder {
    feature_type: String,
    include_pos_patterns: bool,
    include_dependency_features: bool,
    include_syntactic_complexity: bool,
    include_word_embeddings: bool,
    include_semantic_roles: bool,
    include_named_entities: bool,
    include_coherence_features: bool,
    include_rhetorical_structure: bool,
    include_topic_modeling: bool,
    include_sentiment_features: bool,
    include_emotion_features: bool,
    include_transformer_features: bool,
    include_attention_patterns: bool,
    pos_ngram_range: (usize, usize),
    dependency_window_size: usize,
    embedding_dimensions: usize,
    semantic_similarity_threshold: Float,
    entity_type_importance: Float,
    coherence_window_size: usize,
    topic_model_dimensions: usize,
    rhetorical_relation_types: Vec<String>,
    transformer_layers: Vec<usize>,
    attention_head_analysis: bool,
    contextual_similarity_weight: Float,
    cross_lingual_features: bool,
    language_transfer_weight: Float,
    discourse_marker_weight: Float,
    k: Option<usize>,
    score_threshold: Float,
    strategy: NLPStrategy,
}

impl Default for AdvancedNLPFeatureSelectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedNLPFeatureSelectorBuilder {
    pub fn new() -> Self {
        Self {
            feature_type: "syntactic".to_string(),
            include_pos_patterns: true,
            include_dependency_features: true,
            include_syntactic_complexity: true,
            include_word_embeddings: false,
            include_semantic_roles: false,
            include_named_entities: false,
            include_coherence_features: false,
            include_rhetorical_structure: false,
            include_topic_modeling: false,
            include_sentiment_features: false,
            include_emotion_features: false,
            include_transformer_features: false,
            include_attention_patterns: false,
            pos_ngram_range: (1, 3),
            dependency_window_size: 5,
            embedding_dimensions: 300,
            semantic_similarity_threshold: 0.7,
            entity_type_importance: 0.8,
            coherence_window_size: 10,
            topic_model_dimensions: 50,
            rhetorical_relation_types: vec!["elaboration".to_string(), "contrast".to_string()],
            transformer_layers: vec![6, 9, 12],
            attention_head_analysis: false,
            contextual_similarity_weight: 0.6,
            cross_lingual_features: false,
            language_transfer_weight: 0.3,
            discourse_marker_weight: 0.4,
            k: None,
            score_threshold: 0.1,
            strategy: NLPStrategy::InformationTheoretic,
        }
    }

    /// Type of NLP features: "syntactic", "semantic", "discourse", "pragmatic", "contextual".
    pub fn feature_type(mut self, feature_type: &str) -> Self {
        self.feature_type = feature_type.to_string();
        self
    }

    /// Whether to include POS tag patterns.
    pub fn include_pos_patterns(mut self, include: bool) -> Self {
        self.include_pos_patterns = include;
        self
    }

    /// Whether to include dependency parsing features.
    pub fn include_dependency_features(mut self, include: bool) -> Self {
        self.include_dependency_features = include;
        self
    }

    /// Whether to include syntactic complexity metrics.
    pub fn include_syntactic_complexity(mut self, include: bool) -> Self {
        self.include_syntactic_complexity = include;
        self
    }

    /// Whether to include word embedding features.
    pub fn include_word_embeddings(mut self, include: bool) -> Self {
        self.include_word_embeddings = include;
        self
    }

    /// Whether to include semantic role labeling features.
    pub fn include_semantic_roles(mut self, include: bool) -> Self {
        self.include_semantic_roles = include;
        self
    }

    /// Whether to include named entity features.
    pub fn include_named_entities(mut self, include: bool) -> Self {
        self.include_named_entities = include;
        self
    }

    /// Whether to include coherence and cohesion features.
    pub fn include_coherence_features(mut self, include: bool) -> Self {
        self.include_coherence_features = include;
        self
    }

    /// Whether to include rhetorical structure features.
    pub fn include_rhetorical_structure(mut self, include: bool) -> Self {
        self.include_rhetorical_structure = include;
        self
    }

    /// Whether to include topic modeling features.
    pub fn include_topic_modeling(mut self, include: bool) -> Self {
        self.include_topic_modeling = include;
        self
    }

    /// Whether to include sentiment analysis features.
    pub fn include_sentiment_features(mut self, include: bool) -> Self {
        self.include_sentiment_features = include;
        self
    }

    /// Whether to include emotion detection features.
    pub fn include_emotion_features(mut self, include: bool) -> Self {
        self.include_emotion_features = include;
        self
    }

    /// Whether to include transformer-based features.
    pub fn include_transformer_features(mut self, include: bool) -> Self {
        self.include_transformer_features = include;
        self
    }

    /// Whether to include attention pattern analysis.
    pub fn include_attention_patterns(mut self, include: bool) -> Self {
        self.include_attention_patterns = include;
        self
    }

    /// N-gram range for POS tag patterns.
    pub fn pos_ngram_range(mut self, range: (usize, usize)) -> Self {
        self.pos_ngram_range = range;
        self
    }

    /// Window size for dependency relation analysis.
    pub fn dependency_window_size(mut self, size: usize) -> Self {
        self.dependency_window_size = size;
        self
    }

    /// Number of embedding dimensions.
    pub fn embedding_dimensions(mut self, dims: usize) -> Self {
        self.embedding_dimensions = dims;
        self
    }

    /// Threshold for semantic similarity.
    pub fn semantic_similarity_threshold(mut self, threshold: Float) -> Self {
        self.semantic_similarity_threshold = threshold;
        self
    }

    /// Importance weight for entity type features.
    pub fn entity_type_importance(mut self, importance: Float) -> Self {
        self.entity_type_importance = importance;
        self
    }

    /// Window size for coherence analysis.
    pub fn coherence_window_size(mut self, size: usize) -> Self {
        self.coherence_window_size = size;
        self
    }

    /// Number of dimensions for topic modeling.
    pub fn topic_model_dimensions(mut self, dims: usize) -> Self {
        self.topic_model_dimensions = dims;
        self
    }

    /// Types of rhetorical relations to analyze.
    pub fn rhetorical_relation_types(mut self, types: Vec<&str>) -> Self {
        self.rhetorical_relation_types = types.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Transformer layers to analyze.
    pub fn transformer_layers(mut self, layers: Vec<usize>) -> Self {
        self.transformer_layers = layers;
        self
    }

    /// Whether to analyze individual attention heads.
    pub fn attention_head_analysis(mut self, analyze: bool) -> Self {
        self.attention_head_analysis = analyze;
        self
    }

    /// Weight for contextual similarity in scoring.
    pub fn contextual_similarity_weight(mut self, weight: Float) -> Self {
        self.contextual_similarity_weight = weight;
        self
    }

    /// Whether to include cross-lingual features.
    pub fn cross_lingual_features(mut self, include: bool) -> Self {
        self.cross_lingual_features = include;
        self
    }

    /// Weight for language transfer features.
    pub fn language_transfer_weight(mut self, weight: Float) -> Self {
        self.language_transfer_weight = weight;
        self
    }

    /// Weight for discourse marker features.
    pub fn discourse_marker_weight(mut self, weight: Float) -> Self {
        self.discourse_marker_weight = weight;
        self
    }

    /// Number of top features to select.
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Minimum score threshold for feature selection.
    pub fn score_threshold(mut self, threshold: Float) -> Self {
        self.score_threshold = threshold;
        self
    }

    /// Builds the AdvancedNLPFeatureSelector.
    pub fn build(self) -> AdvancedNLPFeatureSelector<Untrained> {
        AdvancedNLPFeatureSelector {
            feature_type: self.feature_type,
            include_pos_patterns: self.include_pos_patterns,
            include_dependency_features: self.include_dependency_features,
            include_syntactic_complexity: self.include_syntactic_complexity,
            include_word_embeddings: self.include_word_embeddings,
            include_semantic_roles: self.include_semantic_roles,
            include_named_entities: self.include_named_entities,
            include_coherence_features: self.include_coherence_features,
            include_rhetorical_structure: self.include_rhetorical_structure,
            include_topic_modeling: self.include_topic_modeling,
            include_sentiment_features: self.include_sentiment_features,
            include_emotion_features: self.include_emotion_features,
            include_transformer_features: self.include_transformer_features,
            include_attention_patterns: self.include_attention_patterns,
            pos_ngram_range: self.pos_ngram_range,
            dependency_window_size: self.dependency_window_size,
            embedding_dimensions: self.embedding_dimensions,
            semantic_similarity_threshold: self.semantic_similarity_threshold,
            entity_type_importance: self.entity_type_importance,
            coherence_window_size: self.coherence_window_size,
            topic_model_dimensions: self.topic_model_dimensions,
            rhetorical_relation_types: self.rhetorical_relation_types,
            transformer_layers: self.transformer_layers,
            attention_head_analysis: self.attention_head_analysis,
            contextual_similarity_weight: self.contextual_similarity_weight,
            cross_lingual_features: self.cross_lingual_features,
            language_transfer_weight: self.language_transfer_weight,
            discourse_marker_weight: self.discourse_marker_weight,
            k: self.k.unwrap_or(100),
            score_threshold: self.score_threshold,
            strategy: self.strategy,
            state: PhantomData,
            trained_state: None,
        }
    }
}

impl Estimator for AdvancedNLPFeatureSelector<Untrained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for AdvancedNLPFeatureSelector<Trained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for AdvancedNLPFeatureSelector<Untrained> {
    type Fitted = AdvancedNLPFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Perform analysis based on feature type
        let (
            feature_scores,
            syntactic_scores,
            semantic_scores,
            discourse_scores,
            pragmatic_scores,
            contextual_scores,
            attention_weights,
            topic_distributions,
            semantic_similarity_matrix,
        ) = match self.feature_type.as_str() {
            "syntactic" => self.analyze_syntactic_features(x, y)?,
            "semantic" => self.analyze_semantic_features(x, y)?,
            "discourse" => self.analyze_discourse_features(x, y)?,
            "pragmatic" => self.analyze_pragmatic_features(x, y)?,
            "contextual" => self.analyze_contextual_features(x, y)?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown feature type: {}",
                    self.feature_type
                )))
            }
        };

        // Select features based on scores
        let selected_features = if self.k > 0 {
            select_top_k_features(&feature_scores, self.k)
        } else {
            select_features_by_threshold(&feature_scores, self.score_threshold)
        };

        let trained_state = Trained {
            selected_features,
            feature_scores,
            syntactic_scores,
            semantic_scores,
            discourse_scores,
            pragmatic_scores,
            contextual_scores,
            attention_weights,
            topic_distributions,
            semantic_similarity_matrix,
            n_features,
            feature_type: self.feature_type.clone(),
        };

        Ok(AdvancedNLPFeatureSelector {
            feature_type: self.feature_type,
            include_pos_patterns: self.include_pos_patterns,
            include_dependency_features: self.include_dependency_features,
            include_syntactic_complexity: self.include_syntactic_complexity,
            include_word_embeddings: self.include_word_embeddings,
            include_semantic_roles: self.include_semantic_roles,
            include_named_entities: self.include_named_entities,
            include_coherence_features: self.include_coherence_features,
            include_rhetorical_structure: self.include_rhetorical_structure,
            include_topic_modeling: self.include_topic_modeling,
            include_sentiment_features: self.include_sentiment_features,
            include_emotion_features: self.include_emotion_features,
            include_transformer_features: self.include_transformer_features,
            include_attention_patterns: self.include_attention_patterns,
            pos_ngram_range: self.pos_ngram_range,
            dependency_window_size: self.dependency_window_size,
            embedding_dimensions: self.embedding_dimensions,
            semantic_similarity_threshold: self.semantic_similarity_threshold,
            entity_type_importance: self.entity_type_importance,
            coherence_window_size: self.coherence_window_size,
            topic_model_dimensions: self.topic_model_dimensions,
            rhetorical_relation_types: self.rhetorical_relation_types,
            transformer_layers: self.transformer_layers,
            attention_head_analysis: self.attention_head_analysis,
            contextual_similarity_weight: self.contextual_similarity_weight,
            cross_lingual_features: self.cross_lingual_features,
            language_transfer_weight: self.language_transfer_weight,
            discourse_marker_weight: self.discourse_marker_weight,
            k: self.k,
            score_threshold: self.score_threshold,
            strategy: self.strategy,
            state: PhantomData,
            trained_state: Some(trained_state),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for AdvancedNLPFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Selector must be fitted before transforming".to_string())
        })?;

        let (n_samples, n_features) = x.dim();

        if n_features != trained.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                trained.n_features, n_features
            )));
        }

        if trained.selected_features.is_empty() {
            return Err(SklearsError::InvalidState(
                "No features were selected".to_string(),
            ));
        }

        let selected_data = x.select(Axis(1), &trained.selected_features);
        Ok(selected_data)
    }
}

impl SelectorMixin for AdvancedNLPFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState("Selector must be fitted before getting support".to_string())
        })?;

        let mut support = Array1::from_elem(trained.n_features, false);
        for &idx in &trained.selected_features {
            support[idx] = true;
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
}

impl AdvancedNLPFeatureSelector<Trained> {
    /// Get the selected feature indices
    pub fn selected_features(&self) -> Result<&Vec<usize>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState(
                "Selector must be fitted before getting selected features".to_string(),
            )
        })?;
        Ok(&trained.selected_features)
    }
}

// Implementation methods for AdvancedNLPFeatureSelector
impl AdvancedNLPFeatureSelector<Untrained> {
    fn analyze_syntactic_features(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut syntactic_scores = HashMap::new();

        // POS pattern analysis
        if self.include_pos_patterns {
            let pos_scores = compute_pos_pattern_scores(x, y, self.pos_ngram_range)?;
            feature_scores = &feature_scores + &pos_scores;
            syntactic_scores.insert("pos_patterns".to_string(), pos_scores);
        }

        // Dependency feature analysis
        if self.include_dependency_features {
            let dep_scores = compute_dependency_scores(x, y, self.dependency_window_size)?;
            feature_scores = &feature_scores + &dep_scores;
            syntactic_scores.insert("dependency_features".to_string(), dep_scores);
        }

        // Syntactic complexity analysis
        if self.include_syntactic_complexity {
            let complexity_scores = compute_syntactic_complexity_scores(x, y)?;
            feature_scores = &feature_scores + &complexity_scores;
            syntactic_scores.insert("syntactic_complexity".to_string(), complexity_scores);
        }

        Ok((
            feature_scores,
            Some(syntactic_scores),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ))
    }

    fn analyze_semantic_features(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut semantic_scores = HashMap::new();

        // Word embedding analysis
        if self.include_word_embeddings {
            let embedding_scores = compute_embedding_scores(x, y, self.embedding_dimensions)?;
            feature_scores = &feature_scores + &embedding_scores;
            semantic_scores.insert("word_embeddings".to_string(), embedding_scores);
        }

        // Semantic role analysis
        if self.include_semantic_roles {
            let role_scores = compute_semantic_role_scores(x, y)?;
            feature_scores = &feature_scores + &role_scores;
            semantic_scores.insert("semantic_roles".to_string(), role_scores);
        }

        // Named entity analysis
        if self.include_named_entities {
            let entity_scores = compute_named_entity_scores(x, y, self.entity_type_importance)?;
            feature_scores = &feature_scores + &entity_scores;
            semantic_scores.insert("named_entities".to_string(), entity_scores);
        }

        // Semantic similarity matrix
        let semantic_similarity_matrix = if self.include_word_embeddings {
            Some(compute_semantic_similarity_matrix(
                x,
                self.semantic_similarity_threshold,
            )?)
        } else {
            None
        };

        Ok((
            feature_scores,
            None,
            Some(semantic_scores),
            None,
            None,
            None,
            None,
            None,
            semantic_similarity_matrix,
        ))
    }

    fn analyze_discourse_features(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut discourse_scores = HashMap::new();

        // Coherence and cohesion analysis
        if self.include_coherence_features {
            let coherence_scores = compute_coherence_scores(x, y, self.coherence_window_size)?;
            feature_scores = &feature_scores + &coherence_scores;
            discourse_scores.insert("coherence_features".to_string(), coherence_scores);
        }

        // Rhetorical structure analysis
        if self.include_rhetorical_structure {
            let rhetorical_scores =
                compute_rhetorical_structure_scores(x, y, &self.rhetorical_relation_types)?;
            feature_scores = &feature_scores + &rhetorical_scores;
            discourse_scores.insert("rhetorical_structure".to_string(), rhetorical_scores);
        }

        // Topic modeling analysis
        let topic_distributions = if self.include_topic_modeling {
            let (topic_scores, topic_dists) =
                compute_topic_modeling_scores(x, y, self.topic_model_dimensions)?;
            feature_scores = &feature_scores + &topic_scores;
            discourse_scores.insert("topic_modeling".to_string(), topic_scores);
            Some(topic_dists)
        } else {
            None
        };

        Ok((
            feature_scores,
            None,
            None,
            Some(discourse_scores),
            None,
            None,
            None,
            topic_distributions,
            None,
        ))
    }

    fn analyze_pragmatic_features(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut pragmatic_scores = HashMap::new();

        // Sentiment analysis
        if self.include_sentiment_features {
            let sentiment_scores = compute_sentiment_scores(x, y)?;
            feature_scores = &feature_scores + &sentiment_scores;
            pragmatic_scores.insert("sentiment_features".to_string(), sentiment_scores);
        }

        // Emotion detection
        if self.include_emotion_features {
            let emotion_scores = compute_emotion_scores(x, y)?;
            feature_scores = &feature_scores + &emotion_scores;
            pragmatic_scores.insert("emotion_features".to_string(), emotion_scores);
        }

        // Cross-lingual features
        if self.cross_lingual_features {
            let cross_lingual_scores =
                compute_cross_lingual_scores(x, y, self.language_transfer_weight)?;
            feature_scores = &feature_scores + &cross_lingual_scores;
            pragmatic_scores.insert("cross_lingual_features".to_string(), cross_lingual_scores);
        }

        Ok((
            feature_scores,
            None,
            None,
            None,
            Some(pragmatic_scores),
            None,
            None,
            None,
            None,
        ))
    }

    fn analyze_contextual_features(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<HashMap<String, Array1<Float>>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
        Option<Array2<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut feature_scores = Array1::zeros(n_features);
        let mut contextual_scores = HashMap::new();

        // Transformer-based features
        if self.include_transformer_features {
            let transformer_scores = compute_transformer_scores(x, y, &self.transformer_layers)?;
            feature_scores = &feature_scores + &transformer_scores;
            contextual_scores.insert("transformer_features".to_string(), transformer_scores);
        }

        // Attention pattern analysis
        let attention_weights = if self.include_attention_patterns {
            let (attention_scores, attention_matrix) =
                compute_attention_scores(x, y, self.attention_head_analysis)?;
            feature_scores = &feature_scores + &attention_scores;
            contextual_scores.insert("attention_patterns".to_string(), attention_scores);
            Some(attention_matrix)
        } else {
            None
        };

        // Contextual similarity weighting
        if self.contextual_similarity_weight > 0.0 {
            let contextual_adjustment =
                compute_contextual_similarity_adjustment(x, self.contextual_similarity_weight)?;
            for j in 0..n_features {
                feature_scores[j] *= contextual_adjustment[j];
            }
        }

        Ok((
            feature_scores,
            None,
            None,
            None,
            None,
            Some(contextual_scores),
            attention_weights,
            None,
            None,
        ))
    }
}

// Syntactic feature computation functions

fn compute_pos_pattern_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    ngram_range: (usize, usize),
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Simplified POS pattern analysis
        let mut pattern_score = 0.0;

        for n in ngram_range.0..=ngram_range.1 {
            if feature.len() >= n {
                // Compute n-gram patterns (simplified)
                for i in 0..feature.len() - n + 1 {
                    let ngram_sum = feature.slice(s![i..i + n]).sum();
                    let target_correlation = if i < y.len() {
                        (ngram_sum * y[i]).abs()
                    } else {
                        0.0
                    };
                    pattern_score += target_correlation;
                }
            }
        }

        scores[j] = pattern_score / (ngram_range.1 - ngram_range.0 + 1) as Float;
    }

    Ok(scores)
}

fn compute_dependency_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    window_size: usize,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Simplified dependency analysis with windowed correlations
        let mut dependency_score = 0.0;
        let mut count = 0;

        for i in 0..feature.len() {
            let start = i.saturating_sub(window_size);
            let end = (i + window_size + 1).min(feature.len());

            if start < end && i < y.len() {
                let window = feature.slice(s![start..end]);
                let window_mean = window.sum() / window.len() as Float;
                let correlation = (window_mean - feature[i]).abs() * y[i].abs();
                dependency_score += correlation;
                count += 1;
            }
        }

        scores[j] = if count > 0 {
            dependency_score / count as Float
        } else {
            0.0
        };
    }

    Ok(scores)
}

fn compute_syntactic_complexity_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Simplified syntactic complexity metrics
        let mean_value = feature.sum() / feature.len() as Float;
        let variance =
            feature.mapv(|val| (val - mean_value).powi(2)).sum() / feature.len() as Float;
        let complexity = variance.sqrt();

        let correlation = compute_pearson_correlation(&feature, &y.view());
        scores[j] = complexity * correlation.abs();
    }

    Ok(scores)
}

// Semantic feature computation functions

fn compute_embedding_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    embedding_dims: usize,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Assume features are organized as embedding dimensions
    let features_per_embedding = embedding_dims.min(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Compute embedding-based similarity scores
        let mut embedding_score = 0.0;

        // Group features into embedding vectors
        let embedding_group = j / features_per_embedding;
        let start_idx = embedding_group * features_per_embedding;
        let end_idx = (start_idx + features_per_embedding).min(n_features);

        if start_idx < end_idx {
            // Compute centroid of embedding group
            let mut centroid = 0.0;
            for k in start_idx..end_idx {
                let other_feature = x.column(k);
                centroid += other_feature.sum() / other_feature.len() as Float;
            }
            centroid /= (end_idx - start_idx) as Float;

            let feature_mean = feature.sum() / feature.len() as Float;
            embedding_score = (feature_mean - centroid).abs();
        }

        let correlation = compute_pearson_correlation(&feature, &y.view());
        scores[j] = embedding_score * correlation.abs();
    }

    Ok(scores)
}

fn compute_semantic_role_scores(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Simplified semantic role analysis
    for j in 0..n_features {
        let feature = x.column(j);

        // Role-based scoring (simplified)
        let role_type = j % 5; // 5 different semantic roles
        let role_weight = match role_type {
            0 => 1.5, // Agent
            1 => 1.3, // Patient
            2 => 1.1, // Theme
            3 => 1.0, // Location
            _ => 0.8, // Other
        };

        let correlation = compute_pearson_correlation(&feature, &y.view());
        scores[j] = role_weight * correlation.abs();
    }

    Ok(scores)
}

fn compute_named_entity_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    entity_importance: Float,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Entity type scoring (simplified)
        let entity_type = j % 7; // 7 different entity types
        let entity_weight = match entity_type {
            0 => 1.4, // PERSON
            1 => 1.3, // ORGANIZATION
            2 => 1.2, // LOCATION
            3 => 1.1, // DATE
            4 => 1.0, // MONEY
            5 => 0.9, // PRODUCT
            _ => 0.8, // MISC
        };

        let correlation = compute_pearson_correlation(&feature, &y.view());
        scores[j] = entity_weight * entity_importance * correlation.abs();
    }

    Ok(scores)
}

fn compute_semantic_similarity_matrix(
    x: &Array2<Float>,
    threshold: Float,
) -> Result<Array2<Float>> {
    let (_, n_features) = x.dim();
    let mut similarity_matrix = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in i..n_features {
            let similarity = if i == j {
                1.0
            } else {
                let feature_i = x.column(i);
                let feature_j = x.column(j);
                let correlation = compute_pearson_correlation(&feature_i, &feature_j);
                if correlation.abs() >= threshold {
                    correlation
                } else {
                    0.0
                }
            };

            similarity_matrix[[i, j]] = similarity;
            similarity_matrix[[j, i]] = similarity;
        }
    }

    Ok(similarity_matrix)
}

// Discourse feature computation functions

fn compute_coherence_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    window_size: usize,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Coherence analysis using local consistency
        let mut coherence_score = 0.0;
        let mut count = 0;

        for i in window_size..feature.len() {
            if i < y.len() {
                let current_window = feature.slice(s![i - window_size..i]);
                let next_value = feature[i];

                let window_mean = current_window.sum() / window_size as Float;
                let consistency = 1.0 / (1.0 + (next_value - window_mean).abs());

                coherence_score += consistency * y[i].abs();
                count += 1;
            }
        }

        scores[j] = if count > 0 {
            coherence_score / count as Float
        } else {
            0.0
        };
    }

    Ok(scores)
}

fn compute_rhetorical_structure_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    relation_types: &[String],
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Rhetorical relation analysis (simplified)
        let mut rhetorical_score = 0.0;

        for (idx, relation_type) in relation_types.iter().enumerate() {
            let relation_weight = match relation_type.as_str() {
                "elaboration" => 1.3,
                "contrast" => 1.2,
                "cause" => 1.4,
                "condition" => 1.1,
                _ => 1.0,
            };

            // Features associated with this relation type
            if j % relation_types.len() == idx {
                let correlation = compute_pearson_correlation(&feature, &y.view());
                rhetorical_score += relation_weight * correlation.abs();
            }
        }

        scores[j] = rhetorical_score / relation_types.len() as Float;
    }

    Ok(scores)
}

fn compute_topic_modeling_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    topic_dims: usize,
) -> Result<(Array1<Float>, Array2<Float>)> {
    let (n_samples, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Simplified topic modeling (LDA-like)
    let n_topics = topic_dims.min(10);
    let mut topic_distributions = Array2::zeros((n_samples, n_topics));

    // Initialize topic distributions
    for i in 0..n_samples {
        let doc_features = x.row(i);
        let doc_sum = doc_features.sum();

        for t in 0..n_topics {
            // Simple topic assignment based on feature groups
            let topic_start = (t * n_features) / n_topics;
            let topic_end = ((t + 1) * n_features) / n_topics;

            let topic_sum = doc_features.slice(s![topic_start..topic_end]).sum();
            topic_distributions[[i, t]] = if doc_sum > 0.0 {
                topic_sum / doc_sum
            } else {
                1.0 / n_topics as Float
            };
        }
    }

    // Compute feature scores based on topic coherence
    for j in 0..n_features {
        let feature = x.column(j);
        let topic_idx = (j * n_topics) / n_features;

        if topic_idx < n_topics {
            let topic_dist = topic_distributions.column(topic_idx);
            let correlation = compute_pearson_correlation(&feature, &topic_dist);
            scores[j] = correlation.abs();
        }
    }

    Ok((scores, topic_distributions))
}

// Pragmatic feature computation functions

fn compute_sentiment_scores(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Sentiment polarity analysis (simplified)
        let mean_value = feature.sum() / feature.len() as Float;
        let polarity: Float = if mean_value > 0.5 {
            1.0
        } else if mean_value < -0.5 {
            -1.0
        } else {
            0.0
        };

        let correlation = compute_pearson_correlation(&feature, &y.view());
        scores[j] = polarity.abs() * correlation.abs();
    }

    Ok(scores)
}

fn compute_emotion_scores(x: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Emotion categories: joy, anger, fear, sadness, surprise, disgust
    let emotion_weights = [1.2, 1.4, 1.1, 1.0, 1.3, 0.9];

    for j in 0..n_features {
        let feature = x.column(j);

        // Emotion classification (simplified)
        let emotion_type = j % emotion_weights.len();
        let emotion_weight = emotion_weights[emotion_type];

        let correlation = compute_pearson_correlation(&feature, &y.view());
        scores[j] = emotion_weight * correlation.abs();
    }

    Ok(scores)
}

fn compute_cross_lingual_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    transfer_weight: Float,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Cross-lingual transfer analysis (simplified)
        let language_group = j % 3; // 3 language families
        let transfer_effectiveness = match language_group {
            0 => 1.3, // Germanic languages
            1 => 1.1, // Romance languages
            _ => 0.9, // Other language families
        };

        let correlation = compute_pearson_correlation(&feature, &y.view());
        scores[j] = transfer_weight * transfer_effectiveness * correlation.abs();
    }

    Ok(scores)
}

// Contextual feature computation functions

fn compute_transformer_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    layers: &[usize],
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    for j in 0..n_features {
        let feature = x.column(j);

        // Layer-specific analysis
        let layer_group = j % layers.len();
        let layer_depth = layers[layer_group];

        // Deeper layers typically capture more abstract features
        let layer_weight = 1.0 + (layer_depth as Float) * 0.1;

        let correlation = compute_pearson_correlation(&feature, &y.view());
        scores[j] = layer_weight * correlation.abs();
    }

    Ok(scores)
}

fn compute_attention_scores(
    x: &Array2<Float>,
    y: &Array1<Float>,
    head_analysis: bool,
) -> Result<(Array1<Float>, Array2<Float>)> {
    let (n_samples, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Simplified attention mechanism
    let n_heads = if head_analysis { 8 } else { 1 };
    let mut attention_matrix = Array2::zeros((n_features, n_features));

    // Compute attention weights
    for i in 0..n_features {
        for j in 0..n_features {
            let feature_i = x.column(i);
            let feature_j = x.column(j);

            let attention_weight = if i == j {
                1.0
            } else {
                let correlation = compute_pearson_correlation(&feature_i, &feature_j);
                (correlation.abs() * n_heads as Float).tanh()
            };

            attention_matrix[[i, j]] = attention_weight;
        }
    }

    // Compute feature scores based on attention patterns
    for j in 0..n_features {
        let feature = x.column(j);
        let attention_row = attention_matrix.row(j);

        let attention_sum = attention_row.sum();
        let attention_entropy = compute_attention_entropy(&attention_row);

        let correlation = compute_pearson_correlation(&feature, &y.view());
        scores[j] = attention_sum * attention_entropy * correlation.abs();
    }

    Ok((scores, attention_matrix))
}

fn compute_contextual_similarity_adjustment(
    x: &Array2<Float>,
    similarity_weight: Float,
) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut adjustments = Array1::ones(n_features);

    for j in 0..n_features {
        let feature_j = x.column(j);
        let mut context_score = 0.0;

        // Contextual similarity with neighboring features
        for k in 0..n_features {
            if j != k {
                let feature_k = x.column(k);
                let similarity = compute_pearson_correlation(&feature_j, &feature_k);

                // Distance-based weighting
                let distance = (j as i32 - k as i32).abs() as Float;
                let weight = (-distance / n_features as Float).exp();

                context_score += similarity.abs() * weight;
            }
        }

        adjustments[j] = 1.0 + similarity_weight * context_score / n_features as Float;
    }

    Ok(adjustments)
}

// Utility functions

fn compute_attention_entropy(attention_weights: &ArrayView1<Float>) -> Float {
    let sum = attention_weights.sum();
    if sum == 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &weight in attention_weights.iter() {
        if weight > 0.0 {
            let prob = weight / sum;
            entropy -= prob * prob.ln();
        }
    }

    entropy
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
    fn test_advanced_nlp_feature_selector_creation() {
        let selector = AdvancedNLPFeatureSelector::new();
        assert_eq!(selector.feature_type, "syntactic");
        assert!(selector.include_pos_patterns);
        assert!(selector.include_dependency_features);
        assert!(selector.include_syntactic_complexity);
    }

    #[test]
    fn test_advanced_nlp_feature_selector_builder() {
        let selector = AdvancedNLPFeatureSelector::builder()
            .feature_type("semantic")
            .include_word_embeddings(true)
            .embedding_dimensions(512)
            .k(15)
            .build();

        assert_eq!(selector.feature_type, "semantic");
        assert!(selector.include_word_embeddings);
        assert_eq!(selector.embedding_dimensions, 512);
        assert_eq!(selector.k, 15);
    }

    #[test]
    fn test_syntactic_analysis() {
        let nlp_features =
            Array2::from_shape_vec((20, 10), (0..200).map(|x| x as f64 * 0.01).collect()).unwrap();
        let labels = Array1::from_iter((0..20).map(|i| (i % 3) as f64));

        let selector = AdvancedNLPFeatureSelector::builder()
            .feature_type("syntactic")
            .k(5)
            .build();

        let trained = selector.fit(&nlp_features, &labels).unwrap();
        let transformed = trained.transform(&nlp_features).unwrap();

        assert_eq!(transformed.ncols(), 5);
        assert_eq!(transformed.nrows(), 20);
    }

    #[test]
    fn test_semantic_analysis() {
        let nlp_features =
            Array2::from_shape_vec((15, 12), (0..180).map(|x| (x as f64) * 0.02).collect())
                .unwrap();
        let labels = Array1::from_iter((0..15).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }));

        let selector = AdvancedNLPFeatureSelector::builder()
            .feature_type("semantic")
            .include_word_embeddings(true)
            .include_named_entities(true)
            .k(6)
            .build();

        let trained = selector.fit(&nlp_features, &labels).unwrap();
        let transformed = trained.transform(&nlp_features).unwrap();

        assert_eq!(transformed.ncols(), 6);
        assert_eq!(transformed.nrows(), 15);
    }

    #[test]
    fn test_pos_pattern_scores() {
        let features =
            Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let target = Array1::from_iter((0..10).map(|i| (i % 2) as f64));

        let scores = compute_pos_pattern_scores(&features, &target, (1, 2)).unwrap();

        assert_eq!(scores.len(), 3);
        for &score in scores.iter() {
            assert!(score >= 0.0);
        }
    }

    #[test]
    fn test_semantic_similarity_matrix() {
        let features = Array2::from_shape_vec(
            (8, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 1.5, 3.0, 4.5, 6.0, 3.0, 6.0, 9.0, 12.0,
                1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 12.0, 16.0,
            ],
        )
        .unwrap();

        let similarity_matrix = compute_semantic_similarity_matrix(&features, 0.5).unwrap();

        assert_eq!(similarity_matrix.dim(), (4, 4));
        // Check diagonal elements
        for i in 0..4 {
            assert!((similarity_matrix[[i, i]] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_attention_scores() {
        let features =
            Array2::from_shape_vec((12, 6), (0..72).map(|x| x as f64).collect()).unwrap();
        let target = Array1::from_iter((0..12).map(|i| (i % 3) as f64));

        let (scores, attention_matrix) =
            compute_attention_scores(&features, &target, true).unwrap();

        assert_eq!(scores.len(), 6);
        assert_eq!(attention_matrix.dim(), (6, 6));

        for &score in scores.iter() {
            assert!(score >= 0.0);
        }
    }

    #[test]
    fn test_discourse_features() {
        let nlp_features =
            Array2::from_shape_vec((25, 8), (0..200).map(|x| (x as f64) * 0.05).collect()).unwrap();
        let labels = Array1::from_iter((0..25).map(|i| (i % 4) as f64));

        let selector = AdvancedNLPFeatureSelector::builder()
            .feature_type("discourse")
            .include_coherence_features(true)
            .include_topic_modeling(true)
            .k(4)
            .build();

        let trained = selector.fit(&nlp_features, &labels).unwrap();
        let transformed = trained.transform(&nlp_features).unwrap();

        assert_eq!(transformed.ncols(), 4);
        assert_eq!(transformed.nrows(), 25);
    }

    #[test]
    fn test_get_support() {
        let nlp_features =
            Array2::from_shape_vec((10, 8), (0..80).map(|x| x as f64).collect()).unwrap();
        let labels = Array1::from_iter((0..10).map(|i| (i % 2) as f64));

        let selector = AdvancedNLPFeatureSelector::builder().k(5).build();

        let trained = selector.fit(&nlp_features, &labels).unwrap();
        let support = trained.get_support().unwrap();

        assert_eq!(support.len(), 8);
        assert_eq!(support.iter().filter(|&&x| x).count(), 5);
    }

    #[test]
    fn test_attention_entropy() {
        let attention_weights = Array1::from_vec(vec![0.4, 0.3, 0.2, 0.1]);
        let entropy = compute_attention_entropy(&attention_weights.view());

        assert!(entropy > 0.0);
        assert!(entropy < 10.0); // Reasonable bound for entropy
    }

    #[test]
    fn test_contextual_features() {
        let nlp_features =
            Array2::from_shape_vec((18, 6), (0..108).map(|x| (x as f64) * 0.03).collect()).unwrap();
        let labels = Array1::from_iter((0..18).map(|i| if i < 9 { 0.0 } else { 1.0 }));

        let selector = AdvancedNLPFeatureSelector::builder()
            .feature_type("contextual")
            .include_transformer_features(true)
            .include_attention_patterns(true)
            .k(3)
            .build();

        let trained = selector.fit(&nlp_features, &labels).unwrap();
        let transformed = trained.transform(&nlp_features).unwrap();

        assert_eq!(transformed.ncols(), 3);
        assert_eq!(transformed.nrows(), 18);
    }
}
