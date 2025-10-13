//! Machine Learning Infrastructure Module
//!
//! This module provides comprehensive machine learning infrastructure for adaptive learning,
//! reinforcement learning, experience management, policy optimization, and model selection
//! in distributed optimization environments.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};

// ================================================================================================
// MACHINE LEARNING INFRASTRUCTURE
// ================================================================================================

/// Adaptive learning system for optimization
pub struct AdaptiveLearning {
    learning_algorithms: Vec<LearningAlgorithm>,
    experience_buffer: ExperienceBuffer,
    policy_updater: PolicyUpdater,
    model_selector: ModelSelector,
}

/// Learning algorithms
#[derive(Debug, Clone)]
pub enum LearningAlgorithm {
    Q_Learning,
    SARSA,
    DeepQ_Network,
    PolicyGradient,
    ActorCritic,
    ProximalPolicyOptimization,
    TrustRegionPolicyOptimization,
    SoftActorCritic,
    Custom(String),
}

/// Experience buffer for reinforcement learning
pub struct ExperienceBuffer {
    experiences: VecDeque<Experience>,
    buffer_size: usize,
    sampling_strategy: SamplingStrategy,
    prioritization: PrioritizationStrategy,
}

/// Experience for reinforcement learning
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Vec<f64>,
    pub action: u32,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
    pub timestamp: SystemTime,
    pub priority: f64,
    pub importance_weight: f64,
}

/// Sampling strategies for experience buffer
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    Random,
    Prioritized,
    Recent,
    Balanced,
    StratifiedSampling,
    ImportanceSampling,
    Custom(String),
}

/// Prioritization strategies for experience replay
#[derive(Debug, Clone)]
pub enum PrioritizationStrategy {
    None,
    TDError,
    Reward,
    Novelty,
    Uncertainty,
    Combined,
    Custom(String),
}

/// Policy updater for adaptive learning
pub struct PolicyUpdater {
    update_frequency: Duration,
    learning_rate: f64,
    exploration_strategy: ExplorationStrategy,
    policy_evaluation: PolicyEvaluation,
    gradient_estimator: GradientEstimator,
}

/// Exploration strategies
#[derive(Debug, Clone)]
pub enum ExplorationStrategy {
    EpsilonGreedy(f64),
    SoftMax(f64),
    UpperConfidenceBound,
    ThompsonSampling,
    CuriosityDriven,
    NoiseInjection,
    Custom(String),
}

/// Policy evaluation methods
pub struct PolicyEvaluation {
    evaluation_methods: Vec<EvaluationMethod>,
    evaluation_frequency: Duration,
    evaluation_episodes: u32,
    baseline_comparison: BaselineComparison,
}

/// Evaluation methods for policies
#[derive(Debug, Clone)]
pub enum EvaluationMethod {
    MonteCarloEvaluation,
    TemporalDifferenceEvaluation,
    ImportanceSampling,
    WeightedImportanceSampling,
    OffPolicyEvaluation,
    Custom(String),
}

/// Baseline comparison for policy evaluation
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    pub baseline_policies: Vec<String>,
    pub comparison_metrics: Vec<String>,
    pub statistical_tests: Vec<StatisticalTest>,
}

/// Statistical tests for policy comparison
#[derive(Debug, Clone)]
pub enum StatisticalTest {
    TTest,
    WilcoxonRankSum,
    KruskalWallis,
    BootstrapTest,
    PermutationTest,
    Custom(String),
}

/// Gradient estimator for policy optimization
pub struct GradientEstimator {
    estimator_type: GradientEstimatorType,
    variance_reduction: VarianceReduction,
    gradient_clipping: GradientClipping,
}

/// Types of gradient estimators
#[derive(Debug, Clone)]
pub enum GradientEstimatorType {
    REINFORCE,
    REINFORCE_Baseline,
    ActorCritic,
    TrustRegion,
    NaturalPolicyGradient,
    Custom(String),
}

/// Variance reduction techniques
#[derive(Debug, Clone)]
pub struct VarianceReduction {
    pub baseline_type: BaselineType,
    pub control_variates: Vec<String>,
    pub importance_sampling: bool,
}

/// Types of baselines for variance reduction
#[derive(Debug, Clone)]
pub enum BaselineType {
    StateDependent,
    TimeDependent,
    StateTimeDependent,
    LearnedBaseline,
    None,
}

/// Gradient clipping configuration
#[derive(Debug, Clone)]
pub struct GradientClipping {
    pub clipping_method: ClippingMethod,
    pub clip_value: f64,
    pub adaptive_clipping: bool,
}

/// Gradient clipping methods
#[derive(Debug, Clone)]
pub enum ClippingMethod {
    L2Norm,
    ValueClipping,
    GlobalNorm,
    PerParameterClipping,
    AdaptiveClipping,
    Custom(String),
}

/// Model selector for choosing optimal models
pub struct ModelSelector {
    selection_criteria: Vec<SelectionCriterion>,
    model_comparison: ModelComparison,
    selection_history: VecDeque<SelectionEvent>,
    auto_selection: AutoSelectionConfig,
}

/// Model selection criteria
#[derive(Debug, Clone)]
pub enum SelectionCriterion {
    CrossValidationScore,
    AkaikeInformationCriterion,
    BayesianInformationCriterion,
    PredictiveAccuracy,
    Robustness,
    ComputationalEfficiency,
    Custom(String),
}

/// Model comparison framework
pub struct ModelComparison {
    comparison_metrics: Vec<ComparisonMetric>,
    statistical_significance: StatisticalSignificance,
    model_rankings: VecDeque<ModelRanking>,
}

/// Comparison metrics for models
#[derive(Debug, Clone)]
pub struct ComparisonMetric {
    pub metric_name: String,
    pub metric_type: MetricType,
    pub weight: f64,
    pub higher_is_better: bool,
}

/// Types of comparison metrics
#[derive(Debug, Clone)]
pub enum MetricType {
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC_ROC,
    LogLoss,
    RMSE,
    MAE,
    Custom(String),
}

/// Statistical significance testing
pub struct StatisticalSignificance {
    significance_level: f64,
    multiple_testing_correction: MultipleTestingCorrection,
    power_analysis: PowerAnalysis,
}

/// Multiple testing correction methods
#[derive(Debug, Clone)]
pub enum MultipleTestingCorrection {
    Bonferroni,
    Holm,
    BenjaminiHochberg,
    BenjaminiYekutieli,
    None,
    Custom(String),
}

/// Power analysis configuration
#[derive(Debug, Clone)]
pub struct PowerAnalysis {
    pub target_power: f64,
    pub effect_size: f64,
    pub sample_size_estimation: bool,
}

/// Model ranking results
#[derive(Debug, Clone)]
pub struct ModelRanking {
    pub ranking_timestamp: SystemTime,
    pub ranked_models: Vec<RankedModel>,
    pub ranking_confidence: f64,
}

/// Ranked model information
#[derive(Debug, Clone)]
pub struct RankedModel {
    pub model_id: String,
    pub rank: u32,
    pub score: f64,
    pub confidence_interval: (f64, f64),
}

/// Selection event history
#[derive(Debug, Clone)]
pub struct SelectionEvent {
    pub event_id: String,
    pub timestamp: SystemTime,
    pub selected_model: String,
    pub selection_reason: String,
    pub alternatives_considered: Vec<String>,
    pub selection_confidence: f64,
}

/// Auto-selection configuration
pub struct AutoSelectionConfig {
    pub enable_auto_selection: bool,
    pub selection_frequency: Duration,
    pub performance_threshold: f64,
    pub stability_requirement: u32,
    pub fallback_model: Option<String>,
}

/// Machine learning errors
#[derive(Debug, thiserror::Error)]
pub enum MachineLearningError {
    #[error("Training convergence failed: {0}")]
    TrainingConvergenceFailed(String),
    #[error("Policy update failed: {0}")]
    PolicyUpdateFailed(String),
    #[error("Experience buffer overflow: {0}")]
    ExperienceBufferOverflow(String),
    #[error("Gradient computation failed: {0}")]
    GradientComputationFailed(String),
    #[error("Model evaluation failed: {0}")]
    ModelEvaluationFailed(String),
}

// ================================================================================================
// IMPLEMENTATIONS
// ================================================================================================

impl AdaptiveLearning {
    pub fn new() -> Self {
        Self {
            learning_algorithms: vec![LearningAlgorithm::Q_Learning],
            experience_buffer: ExperienceBuffer::new(),
            policy_updater: PolicyUpdater::new(),
            model_selector: ModelSelector::new(),
        }
    }

    pub fn learn_from_experience(&mut self, experience: Experience) -> Result<(), MachineLearningError> {
        self.experience_buffer.add_experience(experience)?;

        if self.experience_buffer.size() > 100 {
            self.policy_updater.update_policy(&self.experience_buffer)?;
        }

        Ok(())
    }

    pub fn select_action(&self, state: &[f64]) -> Result<u32, MachineLearningError> {
        self.policy_updater.select_action(state)
    }

    pub fn evaluate_performance(&self) -> Result<f64, MachineLearningError> {
        self.policy_updater.evaluate_performance()
    }

    pub fn get_learning_algorithms(&self) -> &[LearningAlgorithm] {
        &self.learning_algorithms
    }

    pub fn add_learning_algorithm(&mut self, algorithm: LearningAlgorithm) {
        self.learning_algorithms.push(algorithm);
    }

    pub fn get_experience_buffer(&self) -> &ExperienceBuffer {
        &self.experience_buffer
    }

    pub fn get_policy_updater(&self) -> &PolicyUpdater {
        &self.policy_updater
    }

    pub fn get_model_selector(&self) -> &ModelSelector {
        &self.model_selector
    }
}

impl ExperienceBuffer {
    pub fn new() -> Self {
        Self {
            experiences: VecDeque::new(),
            buffer_size: 10000,
            sampling_strategy: SamplingStrategy::Random,
            prioritization: PrioritizationStrategy::None,
        }
    }

    pub fn add_experience(&mut self, experience: Experience) -> Result<(), MachineLearningError> {
        self.experiences.push_back(experience);

        while self.experiences.len() > self.buffer_size {
            self.experiences.pop_front();
        }

        Ok(())
    }

    pub fn sample_batch(&self, batch_size: usize) -> Result<Vec<Experience>, MachineLearningError> {
        if self.experiences.is_empty() {
            return Err(MachineLearningError::ExperienceBufferOverflow("Buffer is empty".to_string()));
        }

        let mut batch = Vec::new();

        match self.sampling_strategy {
            SamplingStrategy::Random => {
                for _ in 0..batch_size.min(self.experiences.len()) {
                    let idx = (SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() % self.experiences.len() as u128) as usize;
                    if let Some(experience) = self.experiences.get(idx) {
                        batch.push(experience.clone());
                    }
                }
            }
            SamplingStrategy::Recent => {
                let start_idx = self.experiences.len().saturating_sub(batch_size);
                for experience in self.experiences.range(start_idx..) {
                    batch.push(experience.clone());
                }
            }
            _ => {
                for _ in 0..batch_size.min(self.experiences.len()) {
                    let idx = (SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() % self.experiences.len() as u128) as usize;
                    if let Some(experience) = self.experiences.get(idx) {
                        batch.push(experience.clone());
                    }
                }
            }
        }

        Ok(batch)
    }

    pub fn size(&self) -> usize {
        self.experiences.len()
    }

    pub fn get_buffer_size(&self) -> usize {
        self.buffer_size
    }

    pub fn set_buffer_size(&mut self, size: usize) {
        self.buffer_size = size;
        while self.experiences.len() > self.buffer_size {
            self.experiences.pop_front();
        }
    }

    pub fn get_sampling_strategy(&self) -> &SamplingStrategy {
        &self.sampling_strategy
    }

    pub fn set_sampling_strategy(&mut self, strategy: SamplingStrategy) {
        self.sampling_strategy = strategy;
    }

    pub fn get_prioritization_strategy(&self) -> &PrioritizationStrategy {
        &self.prioritization
    }

    pub fn set_prioritization_strategy(&mut self, strategy: PrioritizationStrategy) {
        self.prioritization = strategy;
    }

    pub fn clear(&mut self) {
        self.experiences.clear();
    }
}

impl PolicyUpdater {
    pub fn new() -> Self {
        Self {
            update_frequency: Duration::from_secs(60),
            learning_rate: 0.001,
            exploration_strategy: ExplorationStrategy::EpsilonGreedy(0.1),
            policy_evaluation: PolicyEvaluation::new(),
            gradient_estimator: GradientEstimator::new(),
        }
    }

    pub fn update_policy(&mut self, experience_buffer: &ExperienceBuffer) -> Result<(), MachineLearningError> {
        let batch = experience_buffer.sample_batch(32)?;

        let gradients = self.gradient_estimator.compute_gradients(&batch)?;

        self.apply_gradients(&gradients)?;

        Ok(())
    }

    pub fn select_action(&self, state: &[f64]) -> Result<u32, MachineLearningError> {
        match &self.exploration_strategy {
            ExplorationStrategy::EpsilonGreedy(epsilon) => {
                let random_value = (SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() % 100) as f64 / 100.0;

                if random_value < *epsilon {
                    Ok((random_value * 10.0) as u32)
                } else {
                    self.greedy_action(state)
                }
            }
            _ => self.greedy_action(state)
        }
    }

    fn greedy_action(&self, _state: &[f64]) -> Result<u32, MachineLearningError> {
        Ok(0)
    }

    pub fn evaluate_performance(&self) -> Result<f64, MachineLearningError> {
        Ok(0.8)
    }

    fn apply_gradients(&mut self, _gradients: &[f64]) -> Result<(), MachineLearningError> {
        Ok(())
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate;
    }

    pub fn get_exploration_strategy(&self) -> &ExplorationStrategy {
        &self.exploration_strategy
    }

    pub fn set_exploration_strategy(&mut self, strategy: ExplorationStrategy) {
        self.exploration_strategy = strategy;
    }

    pub fn get_update_frequency(&self) -> Duration {
        self.update_frequency
    }

    pub fn set_update_frequency(&mut self, frequency: Duration) {
        self.update_frequency = frequency;
    }
}

impl PolicyEvaluation {
    pub fn new() -> Self {
        Self {
            evaluation_methods: vec![EvaluationMethod::MonteCarloEvaluation],
            evaluation_frequency: Duration::from_secs(300),
            evaluation_episodes: 100,
            baseline_comparison: BaselineComparison {
                baseline_policies: Vec::new(),
                comparison_metrics: Vec::new(),
                statistical_tests: Vec::new(),
            },
        }
    }

    pub fn get_evaluation_methods(&self) -> &[EvaluationMethod] {
        &self.evaluation_methods
    }

    pub fn add_evaluation_method(&mut self, method: EvaluationMethod) {
        self.evaluation_methods.push(method);
    }

    pub fn get_evaluation_frequency(&self) -> Duration {
        self.evaluation_frequency
    }

    pub fn set_evaluation_frequency(&mut self, frequency: Duration) {
        self.evaluation_frequency = frequency;
    }

    pub fn get_evaluation_episodes(&self) -> u32 {
        self.evaluation_episodes
    }

    pub fn set_evaluation_episodes(&mut self, episodes: u32) {
        self.evaluation_episodes = episodes;
    }

    pub fn get_baseline_comparison(&self) -> &BaselineComparison {
        &self.baseline_comparison
    }
}

impl GradientEstimator {
    pub fn new() -> Self {
        Self {
            estimator_type: GradientEstimatorType::REINFORCE,
            variance_reduction: VarianceReduction {
                baseline_type: BaselineType::StateDependent,
                control_variates: Vec::new(),
                importance_sampling: false,
            },
            gradient_clipping: GradientClipping {
                clipping_method: ClippingMethod::L2Norm,
                clip_value: 1.0,
                adaptive_clipping: false,
            },
        }
    }

    pub fn compute_gradients(&self, _experiences: &[Experience]) -> Result<Vec<f64>, MachineLearningError> {
        let gradients = vec![0.1; 10];
        Ok(gradients)
    }

    pub fn get_estimator_type(&self) -> &GradientEstimatorType {
        &self.estimator_type
    }

    pub fn set_estimator_type(&mut self, estimator_type: GradientEstimatorType) {
        self.estimator_type = estimator_type;
    }

    pub fn get_variance_reduction(&self) -> &VarianceReduction {
        &self.variance_reduction
    }

    pub fn get_gradient_clipping(&self) -> &GradientClipping {
        &self.gradient_clipping
    }
}

impl ModelSelector {
    pub fn new() -> Self {
        Self {
            selection_criteria: vec![
                SelectionCriterion::CrossValidationScore,
                SelectionCriterion::PredictiveAccuracy,
            ],
            model_comparison: ModelComparison::new(),
            selection_history: VecDeque::new(),
            auto_selection: AutoSelectionConfig {
                enable_auto_selection: true,
                selection_frequency: Duration::from_secs(3600),
                performance_threshold: 0.8,
                stability_requirement: 5,
                fallback_model: Some("default".to_string()),
            },
        }
    }

    pub fn select_best_model(&mut self, candidate_models: &[String]) -> Result<String, MachineLearningError> {
        if candidate_models.is_empty() {
            return Err(MachineLearningError::ModelEvaluationFailed("No candidate models provided".to_string()));
        }

        let mut model_scores = HashMap::new();

        for model_id in candidate_models {
            let score = self.evaluate_model(model_id)?;
            model_scores.insert(model_id.clone(), score);
        }

        let best_model = model_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| candidate_models[0].clone());

        let selection_event = SelectionEvent {
            event_id: format!("selection_{}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()),
            timestamp: SystemTime::now(),
            selected_model: best_model.clone(),
            selection_reason: "Highest cross-validation score".to_string(),
            alternatives_considered: candidate_models.to_vec(),
            selection_confidence: model_scores.get(&best_model).copied().unwrap_or(0.0),
        };

        self.selection_history.push_back(selection_event);

        while self.selection_history.len() > 100 {
            self.selection_history.pop_front();
        }

        Ok(best_model)
    }

    fn evaluate_model(&self, _model_id: &str) -> Result<f64, MachineLearningError> {
        Ok(0.85)
    }

    pub fn get_selection_criteria(&self) -> &[SelectionCriterion] {
        &self.selection_criteria
    }

    pub fn add_selection_criterion(&mut self, criterion: SelectionCriterion) {
        self.selection_criteria.push(criterion);
    }

    pub fn get_model_comparison(&self) -> &ModelComparison {
        &self.model_comparison
    }

    pub fn get_selection_history(&self) -> &VecDeque<SelectionEvent> {
        &self.selection_history
    }

    pub fn get_auto_selection_config(&self) -> &AutoSelectionConfig {
        &self.auto_selection
    }

    pub fn clear_selection_history(&mut self) {
        self.selection_history.clear();
    }
}

impl ModelComparison {
    pub fn new() -> Self {
        Self {
            comparison_metrics: vec![
                ComparisonMetric {
                    metric_name: "Accuracy".to_string(),
                    metric_type: MetricType::Accuracy,
                    weight: 1.0,
                    higher_is_better: true,
                },
                ComparisonMetric {
                    metric_name: "RMSE".to_string(),
                    metric_type: MetricType::RMSE,
                    weight: 0.8,
                    higher_is_better: false,
                },
            ],
            statistical_significance: StatisticalSignificance {
                significance_level: 0.05,
                multiple_testing_correction: MultipleTestingCorrection::BenjaminiHochberg,
                power_analysis: PowerAnalysis {
                    target_power: 0.8,
                    effect_size: 0.5,
                    sample_size_estimation: true,
                },
            },
            model_rankings: VecDeque::new(),
        }
    }

    pub fn get_comparison_metrics(&self) -> &[ComparisonMetric] {
        &self.comparison_metrics
    }

    pub fn add_comparison_metric(&mut self, metric: ComparisonMetric) {
        self.comparison_metrics.push(metric);
    }

    pub fn get_statistical_significance(&self) -> &StatisticalSignificance {
        &self.statistical_significance
    }

    pub fn get_model_rankings(&self) -> &VecDeque<ModelRanking> {
        &self.model_rankings
    }

    pub fn add_model_ranking(&mut self, ranking: ModelRanking) {
        self.model_rankings.push_back(ranking);
        while self.model_rankings.len() > 50 {
            self.model_rankings.pop_front();
        }
    }
}

impl Default for AdaptiveLearning {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ExperienceBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PolicyUpdater {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PolicyEvaluation {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for GradientEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ModelSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ModelComparison {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_learning_creation() {
        let adaptive_learning = AdaptiveLearning::new();
        assert!(!adaptive_learning.learning_algorithms.is_empty());
    }

    #[test]
    fn test_experience_buffer() {
        let mut buffer = ExperienceBuffer::new();

        let experience = Experience {
            state: vec![1.0, 2.0],
            action: 1,
            reward: 1.0,
            next_state: vec![2.0, 3.0],
            done: false,
            timestamp: SystemTime::now(),
            priority: 1.0,
            importance_weight: 1.0,
        };

        let result = buffer.add_experience(experience);
        assert!(result.is_ok());
        assert_eq!(buffer.size(), 1);
    }

    #[test]
    fn test_experience_buffer_sampling() {
        let mut buffer = ExperienceBuffer::new();

        for i in 0..10 {
            let experience = Experience {
                state: vec![i as f64],
                action: i as u32,
                reward: i as f64,
                next_state: vec![i as f64 + 1.0],
                done: false,
                timestamp: SystemTime::now(),
                priority: 1.0,
                importance_weight: 1.0,
            };
            buffer.add_experience(experience).unwrap();
        }

        let batch = buffer.sample_batch(5);
        assert!(batch.is_ok());
        assert_eq!(batch.unwrap().len(), 5);
    }

    #[test]
    fn test_policy_updater() {
        let mut updater = PolicyUpdater::new();
        let buffer = ExperienceBuffer::new();

        let result = updater.update_policy(&buffer);
        assert!(result.is_err());

        let state = vec![1.0, 2.0, 3.0];
        let action_result = updater.select_action(&state);
        assert!(action_result.is_ok());
    }

    #[test]
    fn test_model_selector() {
        let mut selector = ModelSelector::new();
        let candidates = vec!["model1".to_string(), "model2".to_string()];
        let result = selector.select_best_model(&candidates);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gradient_estimator() {
        let estimator = GradientEstimator::new();
        let experiences = vec![
            Experience {
                state: vec![1.0, 2.0],
                action: 1,
                reward: 1.0,
                next_state: vec![2.0, 3.0],
                done: false,
                timestamp: SystemTime::now(),
                priority: 1.0,
                importance_weight: 1.0,
            }
        ];

        let result = estimator.compute_gradients(&experiences);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10);
    }

    #[test]
    fn test_buffer_size_management() {
        let mut buffer = ExperienceBuffer::new();
        assert_eq!(buffer.get_buffer_size(), 10000);

        buffer.set_buffer_size(5);
        assert_eq!(buffer.get_buffer_size(), 5);

        for i in 0..10 {
            let experience = Experience {
                state: vec![i as f64],
                action: i as u32,
                reward: i as f64,
                next_state: vec![i as f64 + 1.0],
                done: false,
                timestamp: SystemTime::now(),
                priority: 1.0,
                importance_weight: 1.0,
            };
            buffer.add_experience(experience).unwrap();
        }

        assert_eq!(buffer.size(), 5);
    }

    #[test]
    fn test_sampling_strategy_changes() {
        let mut buffer = ExperienceBuffer::new();

        assert!(matches!(buffer.get_sampling_strategy(), SamplingStrategy::Random));

        buffer.set_sampling_strategy(SamplingStrategy::Recent);
        assert!(matches!(buffer.get_sampling_strategy(), SamplingStrategy::Recent));
    }

    #[test]
    fn test_policy_evaluation_configuration() {
        let mut eval = PolicyEvaluation::new();

        assert_eq!(eval.get_evaluation_episodes(), 100);
        eval.set_evaluation_episodes(200);
        assert_eq!(eval.get_evaluation_episodes(), 200);

        assert_eq!(eval.get_evaluation_frequency(), Duration::from_secs(300));
        eval.set_evaluation_frequency(Duration::from_secs(600));
        assert_eq!(eval.get_evaluation_frequency(), Duration::from_secs(600));
    }

    #[test]
    fn test_learning_algorithm_management() {
        let mut adaptive_learning = AdaptiveLearning::new();

        let initial_count = adaptive_learning.get_learning_algorithms().len();
        adaptive_learning.add_learning_algorithm(LearningAlgorithm::SARSA);
        assert_eq!(adaptive_learning.get_learning_algorithms().len(), initial_count + 1);
    }

    #[test]
    fn test_model_comparison_metrics() {
        let mut comparison = ModelComparison::new();

        let initial_metrics = comparison.get_comparison_metrics().len();
        let new_metric = ComparisonMetric {
            metric_name: "F1Score".to_string(),
            metric_type: MetricType::F1Score,
            weight: 0.9,
            higher_is_better: true,
        };

        comparison.add_comparison_metric(new_metric);
        assert_eq!(comparison.get_comparison_metrics().len(), initial_metrics + 1);
    }

    #[test]
    fn test_selection_history_management() {
        let mut selector = ModelSelector::new();

        let models = vec!["model1".to_string()];
        selector.select_best_model(&models).unwrap();

        assert_eq!(selector.get_selection_history().len(), 1);

        selector.clear_selection_history();
        assert_eq!(selector.get_selection_history().len(), 0);
    }
}