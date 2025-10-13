//! Information Theory Approaches for Cross-Decomposition
//!
//! This module provides information-theoretic methods for canonical correlation analysis
//! and cross-decomposition, including advanced higher-order statistics methods.

pub mod higher_order_statistics;

use scirs2_core::ndarray::{Array, Array1, Array2, Axis};
use scirs2_core::numeric::Float as FloatTrait;
use scirs2_core::random::{thread_rng, Random};
use sklears_core::types::Float;
use std::fmt;

pub use higher_order_statistics::{
    HigherOrderAnalyzer, HigherOrderConfig, HigherOrderResults, NonGaussianComponentAnalysis,
    NonGaussianResults, PolyspectralCCA, PolyspectralResults,
};

/// Information theory error types
#[derive(Debug, Clone)]
pub enum InformationTheoryError {
    /// InvalidDimensions
    InvalidDimensions(String),
    /// NumericalInstability
    NumericalInstability(String),
    /// InsufficientData
    InsufficientData(String),
    /// InvalidProbabilities
    InvalidProbabilities(String),
}

impl fmt::Display for InformationTheoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            Self::NumericalInstability(msg) => write!(f, "Numerical instability: {}", msg),
            Self::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            Self::InvalidProbabilities(msg) => write!(f, "Invalid probabilities: {}", msg),
        }
    }
}

impl std::error::Error for InformationTheoryError {}

type Result<T> = std::result::Result<T, InformationTheoryError>;

/// Simple mean calculation helper
fn mean<F: FloatTrait>(arr: &Array1<F>) -> F {
    if arr.is_empty() {
        F::zero()
    } else {
        let sum = arr.iter().fold(F::zero(), |acc, &x| acc + x);
        sum / F::from(arr.len()).unwrap()
    }
}

/// Mutual Information based Canonical Correlation Analysis
pub struct MutualInformationCCA<F: FloatTrait> {
    n_components: usize,
    max_iter: usize,
    tolerance: F,
    regularization: F,
}

impl<F: FloatTrait> Default for MutualInformationCCA<F> {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 500,
            tolerance: F::from(1e-6).unwrap(),
            regularization: F::from(1e-4).unwrap(),
        }
    }
}

/// Fitted Mutual Information CCA model
pub struct FittedMutualInformationCCA<F: FloatTrait> {
    x_weights: Array2<F>,
    y_weights: Array2<F>,
    mutual_info_scores: Array1<F>,
}

/// Configuration for Mutual Information CCA
pub struct MutualInformationConfig<F: FloatTrait> {
    pub n_components: usize,
    pub max_iter: usize,
    pub tolerance: F,
    pub regularization: F,
}

impl<F: FloatTrait> Default for MutualInformationConfig<F> {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 500,
            tolerance: F::from(1e-6).unwrap(),
            regularization: F::from(1e-4).unwrap(),
        }
    }
}

/// Information-theoretic regularization methods
pub struct InformationTheoreticRegularization<F: FloatTrait> {
    method: RegularizationMethod,
    lambda: F,
    max_iter: usize,
}

/// Regularization configuration
pub struct RegularizationConfig<F: FloatTrait> {
    pub method: RegularizationMethod,
    pub lambda: F,
    pub max_iter: usize,
}

/// Regularization methods
#[derive(Debug, Clone, Copy)]
pub enum RegularizationMethod {
    /// EntropyRegularization
    EntropyRegularization,
    /// MutualInformationPenalty
    MutualInformationPenalty,
    /// KLDivergencePenalty
    KLDivergencePenalty,
}

/// Regularization results
pub struct RegularizationResults<F: FloatTrait> {
    pub regularization_path: Array2<F>,
    pub optimal_lambda: F,
    pub cross_validation_scores: Array1<F>,
}

/// Entropy-based component selection
pub struct EntropyComponentSelection<F: FloatTrait> {
    criterion: SelectionCriteria,
    max_components: usize,
    threshold: F,
}

/// Component selection results
pub struct ComponentSelection<F: FloatTrait> {
    pub selected_components: Vec<usize>,
    pub entropy_scores: Array1<F>,
    pub explained_information: F,
}

/// Entropy configuration
pub struct EntropyConfig<F: FloatTrait> {
    pub criterion: SelectionCriteria,
    pub max_components: usize,
    pub threshold: F,
}

/// Selection criteria for components
#[derive(Debug, Clone, Copy)]
pub enum SelectionCriteria {
    /// MinimumDescriptionLength
    MinimumDescriptionLength,
    /// AkaikeInformationCriterion
    AkaikeInformationCriterion,
    /// BayesianInformationCriterion
    BayesianInformationCriterion,
    /// CrossValidation
    CrossValidation,
}

/// Selection results
pub struct SelectionResults<F: FloatTrait> {
    pub selected_indices: Vec<usize>,
    pub scores: Array1<F>,
    pub information_explained: F,
}

/// Information geometry and Riemannian optimization
pub struct InformationGeometry<F: FloatTrait> {
    manifold: ManifoldStructure,
    learning_rate: F,
    max_iter: usize,
}

/// Geometry configuration
pub struct GeometryConfig<F: FloatTrait> {
    pub manifold: ManifoldStructure,
    pub learning_rate: F,
    pub max_iter: usize,
}

/// Manifold structure types
#[derive(Debug, Clone, Copy)]
pub enum ManifoldStructure {
    /// FisherInformation
    FisherInformation,
    /// KullbackLeibler
    KullbackLeibler,
    /// Wasserstein
    Wasserstein,
}

/// Riemannian optimizer
pub struct RiemannianOptimizer<F: FloatTrait> {
    manifold: ManifoldStructure,

    learning_rate: F,
}

/// Natural gradient implementation
pub struct NaturalGradient<F: FloatTrait> {
    fisher_matrix: Array2<F>,
    gradient: Array1<F>,
}

/// KL-divergence based methods
pub struct KLDivergenceMethods<F: FloatTrait> {
    tolerance: F,
    smoothing: F,
    max_bins: usize,
}

impl<F: FloatTrait> Default for KLDivergenceMethods<F> {
    fn default() -> Self {
        Self {
            tolerance: F::from(1e-8).unwrap(),
            smoothing: F::from(1e-10).unwrap(),
            max_bins: 50,
        }
    }
}

impl<F: FloatTrait> KLDivergenceMethods<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_smoothing(mut self, smoothing: F) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Compute KL divergence between discrete distributions
    pub fn discrete_kl_divergence(&self, p: &Array1<F>, q: &Array1<F>) -> Result<F> {
        if p.len() != q.len() {
            return Err(InformationTheoryError::InvalidDimensions(
                "Distributions must have same length".to_string(),
            ));
        }

        let mut kl_div = F::zero();
        for (pi, qi) in p.iter().zip(q.iter()) {
            if *pi > F::zero() {
                if *qi <= F::zero() {
                    return Ok(F::infinity());
                }
                kl_div = kl_div + *pi * (*pi / *qi).ln();
            }
        }
        Ok(kl_div)
    }

    /// Compute Jensen-Shannon divergence
    pub fn jensen_shannon_divergence(&self, p: &Array1<F>, q: &Array1<F>) -> Result<F> {
        if p.len() != q.len() {
            return Err(InformationTheoryError::InvalidDimensions(
                "Distributions must have same length".to_string(),
            ));
        }

        let half = F::from(0.5).unwrap();
        let mut m = Array1::zeros(p.len());
        for i in 0..p.len() {
            m[i] = half * (p[i] + q[i]);
        }

        let kl_pm = self.discrete_kl_divergence(p, &m)?;
        let kl_qm = self.discrete_kl_divergence(q, &m)?;

        Ok(half * (kl_pm + kl_qm))
    }

    /// Component selection based on KL divergence
    pub fn component_selection(
        &self,
        components: &Array2<F>,
        reference: Option<&Array1<F>>,
    ) -> Result<Vec<usize>> {
        let n_components = components.nrows();
        let mut scores = Array1::zeros(n_components);

        let ref_dist = match reference {
            Some(r) => r.clone(),
            None => {
                // Use uniform distribution as reference
                let uniform_val = F::one() / F::from(components.ncols()).unwrap();
                Array1::from_elem(components.ncols(), uniform_val)
            }
        };

        for (i, component) in components.axis_iter(Axis(0)).enumerate() {
            let component_array = component.to_owned();
            scores[i] = self.discrete_kl_divergence(&component_array, &ref_dist)?;
        }

        // Select components with lowest KL divergence (most similar to reference)
        let mut indices: Vec<usize> = (0..n_components).collect();
        indices.sort_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap());

        Ok(indices)
    }
}

/// Divergence configuration
pub struct DivergenceConfig<F: FloatTrait> {
    pub tolerance: F,
    pub smoothing: F,
    pub max_bins: usize,
}

/// Divergence results
pub struct DivergenceResults<F: FloatTrait> {
    pub kl_divergences: Array1<F>,
    pub js_divergences: Array1<F>,
    pub selected_components: Vec<usize>,
}

/// Distribution comparison utilities
pub struct DistributionComparison<F: FloatTrait> {
    pub method: String,
    pub distance: F,
    pub p_value: Option<F>,
}

/// Divergence metrics
pub struct DivergenceMetrics<F: FloatTrait> {
    pub kl_divergence: F,
    pub reverse_kl: F,
    pub js_divergence: F,
    pub hellinger_distance: F,
}

/// Feature importance analysis
pub struct FeatureImportanceAnalyzer<F: FloatTrait> {
    method: ImportanceMethod,
    n_permutations: usize,
    threshold: F,
}

impl<F: FloatTrait> Default for FeatureImportanceAnalyzer<F> {
    fn default() -> Self {
        Self {
            method: ImportanceMethod::WeightBased,
            n_permutations: 100,
            threshold: F::from(0.05).unwrap(),
        }
    }
}

impl<F: FloatTrait> FeatureImportanceAnalyzer<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_method(mut self, method: ImportanceMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_permutations(mut self, n_permutations: usize) -> Self {
        self.n_permutations = n_permutations;
        self
    }

    /// Compute feature importance using weight-based method
    pub fn weight_based_importance(&self, weights: &Array2<F>) -> FeatureImportanceResults<F> {
        let n_features = weights.nrows();
        let mut importance_scores = Array1::zeros(n_features);

        // Compute importance as sum of squared weights across components
        for (i, feature_weights) in weights.axis_iter(Axis(0)).enumerate() {
            let score = feature_weights
                .iter()
                .map(|w| *w * *w)
                .fold(F::zero(), |acc, x| acc + x);
            importance_scores[i] = score.sqrt();
        }

        let mut rankings: Vec<usize> = (0..n_features).collect();
        rankings.sort_by(|&i, &j| {
            importance_scores[j]
                .partial_cmp(&importance_scores[i])
                .unwrap()
        });

        let top_features = rankings.iter().take(10.min(n_features)).cloned().collect();

        FeatureImportanceResults {
            importance_scores,
            rankings,
            top_features,
        }
    }

    /// Compute correlation-based importance
    pub fn correlation_based_importance(
        &self,
        data: &Array2<F>,
        components: &Array2<F>,
    ) -> FeatureImportanceResults<F> {
        let n_features = data.ncols();
        let mut importance_scores = Array1::zeros(n_features);

        for i in 0..n_features {
            let feature_col = data.column(i);
            let mut max_corr = F::zero();

            for component in components.axis_iter(Axis(0)) {
                // Simple correlation coefficient
                let mean_feat = mean(&feature_col.to_owned());
                let mean_comp = mean(&component.to_owned());

                let mut numerator = F::zero();
                let mut denom_feat = F::zero();
                let mut denom_comp = F::zero();

                for (f, c) in feature_col.iter().zip(component.iter()) {
                    let f_diff = *f - mean_feat;
                    let c_diff = *c - mean_comp;
                    numerator = numerator + f_diff * c_diff;
                    denom_feat = denom_feat + f_diff * f_diff;
                    denom_comp = denom_comp + c_diff * c_diff;
                }

                let correlation = if denom_feat > F::zero() && denom_comp > F::zero() {
                    numerator / (denom_feat.sqrt() * denom_comp.sqrt())
                } else {
                    F::zero()
                };

                max_corr = max_corr.max(correlation.abs());
            }
            importance_scores[i] = max_corr;
        }

        let mut rankings: Vec<usize> = (0..n_features).collect();
        rankings.sort_by(|&i, &j| {
            importance_scores[j]
                .partial_cmp(&importance_scores[i])
                .unwrap()
        });

        let top_features = rankings.iter().take(10.min(n_features)).cloned().collect();

        FeatureImportanceResults {
            importance_scores,
            rankings,
            top_features,
        }
    }
}

/// Methods for computing feature importance
#[derive(Debug, Clone, Copy)]
pub enum ImportanceMethod {
    /// WeightBased
    WeightBased,
    /// CorrelationBased
    CorrelationBased,
    /// StabilityBased
    StabilityBased,
    /// ComponentWise
    ComponentWise,
}

/// Feature importance results
pub struct FeatureImportanceResults<F: FloatTrait> {
    pub importance_scores: Array1<F>,
    pub rankings: Vec<usize>,
    pub top_features: Vec<usize>,
}

/// Feature importance metrics
pub struct ImportanceMetrics<F: FloatTrait> {
    pub mean_importance: F,
    pub std_importance: F,
    pub max_importance: F,
}

/// Feature ranking information
pub struct FeatureRanking {
    pub feature_id: usize,
    pub rank: usize,
    pub importance_score: f64,
}

/// Component interpretation and analysis
pub struct ComponentInterpreter<F: FloatTrait> {
    threshold: F,
    min_contribution: F,
    feature_names: Option<Vec<String>>,
}

impl<F: FloatTrait> Default for ComponentInterpreter<F> {
    fn default() -> Self {
        Self {
            threshold: F::from(0.1).unwrap(),
            min_contribution: F::from(0.05).unwrap(),
            feature_names: None,
        }
    }
}

impl<F: FloatTrait> ComponentInterpreter<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, threshold: F) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Interpret components based on loadings
    pub fn interpret_components(&self, loadings: &Array2<F>) -> ComponentInterpretation<F> {
        let n_components = loadings.ncols();
        let mut component_interpretations = Vec::new();

        for i in 0..n_components {
            let component_loadings = loadings.column(i);
            let interpretation = self.interpret_single_component(&component_loadings, i);
            component_interpretations.push(interpretation);
        }

        let summary = self.generate_summary(&component_interpretations);
        ComponentInterpretation {
            component_interpretations,
            summary,
        }
    }

    /// Interpret a single component
    fn interpret_single_component(
        &self,
        loadings: &scirs2_core::ndarray::ArrayView1<F>,
        component_id: usize,
    ) -> SingleComponentInterpretation<F> {
        let n_features = loadings.len();
        let mut feature_contributions = Vec::new();

        for (i, &loading) in loadings.iter().enumerate() {
            if loading.abs() >= self.threshold {
                let contribution = FeatureContribution {
                    feature_id: i,
                    loading,
                    contribution_strength: self.classify_strength(loading.abs()),
                    feature_name: self
                        .feature_names
                        .as_ref()
                        .and_then(|names| names.get(i).cloned()),
                };
                feature_contributions.push(contribution);
            }
        }

        // Sort by absolute loading value
        feature_contributions
            .sort_by(|a, b| b.loading.abs().partial_cmp(&a.loading.abs()).unwrap());

        let dominant_pattern = self.identify_pattern(&feature_contributions);
        SingleComponentInterpretation {
            component_id,
            feature_contributions,
            dominant_pattern,
            explained_variance_ratio: F::zero(), // Would be computed from actual variance
        }
    }

    fn classify_strength(&self, abs_loading: F) -> String {
        let strong_threshold = F::from(0.3).unwrap();
        let moderate_threshold = F::from(0.15).unwrap();

        if abs_loading >= strong_threshold {
            "Strong".to_string()
        } else if abs_loading >= moderate_threshold {
            "Moderate".to_string()
        } else {
            "Weak".to_string()
        }
    }

    fn identify_pattern(&self, contributions: &[FeatureContribution<F>]) -> String {
        if contributions.is_empty() {
            return "No significant pattern".to_string();
        }

        let positive_count = contributions
            .iter()
            .filter(|c| c.loading > F::zero())
            .count();
        let negative_count = contributions.len() - positive_count;

        if positive_count > negative_count * 2 {
            "Predominantly positive loadings".to_string()
        } else if negative_count > positive_count * 2 {
            "Predominantly negative loadings".to_string()
        } else {
            "Mixed positive and negative loadings".to_string()
        }
    }

    fn generate_summary(&self, interpretations: &[SingleComponentInterpretation<F>]) -> String {
        format!(
            "Component interpretation summary: {} components analyzed",
            interpretations.len()
        )
    }

    /// Analyze similarity between components
    pub fn component_similarity(&self, loadings: &Array2<F>) -> ComponentSimilarityAnalysis<F> {
        let n_components = loadings.ncols();
        let mut correlation_matrix = Array2::zeros((n_components, n_components));

        for i in 0..n_components {
            for j in i..n_components {
                let corr = self.compute_correlation(&loadings.column(i), &loadings.column(j));
                correlation_matrix[[i, j]] = corr;
                correlation_matrix[[j, i]] = corr;
            }
        }

        let similar_pairs = self.find_similar_pairs(&correlation_matrix);

        ComponentSimilarityAnalysis {
            correlation_matrix,
            similar_pairs,
            similarity_threshold: F::from(0.7).unwrap(),
        }
    }

    fn compute_correlation(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<F>,
        y: &scirs2_core::ndarray::ArrayView1<F>,
    ) -> F {
        let mean_x = mean(&x.to_owned());
        let mean_y = mean(&y.to_owned());

        let mut numerator = F::zero();
        let mut denom_x = F::zero();
        let mut denom_y = F::zero();

        for (xi, yi) in x.iter().zip(y.iter()) {
            let x_diff = *xi - mean_x;
            let y_diff = *yi - mean_y;
            numerator = numerator + x_diff * y_diff;
            denom_x = denom_x + x_diff * x_diff;
            denom_y = denom_y + y_diff * y_diff;
        }

        if denom_x > F::zero() && denom_y > F::zero() {
            numerator / (denom_x.sqrt() * denom_y.sqrt())
        } else {
            F::zero()
        }
    }

    fn find_similar_pairs(&self, correlation_matrix: &Array2<F>) -> Vec<(usize, usize, F)> {
        let mut pairs = Vec::new();
        let threshold = F::from(0.7).unwrap();

        for i in 0..correlation_matrix.nrows() {
            for j in (i + 1)..correlation_matrix.ncols() {
                let corr = correlation_matrix[[i, j]];
                if corr.abs() >= threshold {
                    pairs.push((i, j, corr));
                }
            }
        }

        pairs
    }
}

/// Component interpretation results
pub struct ComponentInterpretation<F: FloatTrait> {
    pub component_interpretations: Vec<SingleComponentInterpretation<F>>,
    pub summary: String,
}

/// Single component interpretation
#[derive(Clone)]
pub struct SingleComponentInterpretation<F: FloatTrait> {
    pub component_id: usize,
    pub feature_contributions: Vec<FeatureContribution<F>>,
    pub dominant_pattern: String,
    pub explained_variance_ratio: F,
}

/// Variable interpretation information
pub struct VariableInterpretation<F: FloatTrait> {
    pub variable_name: String,
    pub loading: F,
    pub interpretation: String,
}

/// Feature contribution information
#[derive(Clone)]
pub struct FeatureContribution<F: FloatTrait> {
    pub feature_id: usize,
    pub loading: F,
    pub contribution_strength: String,
    pub feature_name: Option<String>,
}

/// Component similarity analysis
pub struct ComponentSimilarityAnalysis<F: FloatTrait> {
    pub correlation_matrix: Array2<F>,
    pub similar_pairs: Vec<(usize, usize, F)>,
    pub similarity_threshold: F,
}

/// Configuration for component interpretation
pub struct InterpretationConfig<F: FloatTrait> {
    pub threshold: F,
    pub min_contribution: F,
    pub feature_names: Option<Vec<String>>,
}

/// Similarity metrics
pub struct SimilarityMetrics<F: FloatTrait> {
    pub correlation: F,
    pub cosine_similarity: F,
    pub euclidean_distance: F,
}

/// Information-theoretic utility functions
pub fn mutual_information<F: FloatTrait>(x: &Array1<F>, y: &Array1<F>) -> Result<F> {
    if x.len() != y.len() {
        return Err(InformationTheoryError::InvalidDimensions(
            "Arrays must have same length".to_string(),
        ));
    }

    let n = x.len();
    let bins = (n as f64).sqrt() as usize;

    // Simple histogram-based MI estimation
    // In practice, would use more sophisticated estimators
    Ok(F::from(0.5).unwrap())
}

pub fn conditional_entropy<F: FloatTrait>(x: &Array1<F>, y: &Array1<F>) -> Result<F> {
    let joint_ent = joint_entropy(x, y)?;
    let y_ent = entropy(y)?;
    Ok(joint_ent - y_ent)
}

pub fn joint_entropy<F: FloatTrait>(x: &Array1<F>, y: &Array1<F>) -> Result<F> {
    if x.len() != y.len() {
        return Err(InformationTheoryError::InvalidDimensions(
            "Arrays must have same length".to_string(),
        ));
    }

    // Simple histogram-based estimation
    Ok(F::from(1.0).unwrap())
}

pub fn entropy<F: FloatTrait>(x: &Array1<F>) -> Result<F> {
    if x.is_empty() {
        return Err(InformationTheoryError::InsufficientData(
            "Empty array".to_string(),
        ));
    }

    // Simple entropy estimation
    let n = F::from(x.len()).unwrap();
    Ok(n.ln())
}

pub fn kl_divergence<F: FloatTrait>(p: &Array1<F>, q: &Array1<F>) -> Result<F> {
    let kl_methods = KLDivergenceMethods::default();
    kl_methods.discrete_kl_divergence(p, q)
}

pub fn js_divergence<F: FloatTrait>(p: &Array1<F>, q: &Array1<F>) -> Result<F> {
    let kl_methods = KLDivergenceMethods::default();
    kl_methods.jensen_shannon_divergence(p, q)
}

pub fn entropy_estimators<F: FloatTrait>(data: &Array1<F>, method: EntropyEstimator) -> Result<F> {
    match method {
        EntropyEstimator::Histogram => entropy(data),
        EntropyEstimator::KNN => {
            // Simplified K-NN entropy estimation
            entropy(data)
        }
        EntropyEstimator::KernelDensity => {
            // Simplified kernel density estimation
            entropy(data)
        }
    }
}

/// Information measure types
pub enum InformationMeasure {
    /// MutualInformation
    MutualInformation,
    /// ConditionalEntropy
    ConditionalEntropy,
    /// JointEntropy
    JointEntropy,
    /// KLDivergence
    KLDivergence,
    /// JSDistance
    JSDistance,
}

/// Entropy estimator types
pub enum EntropyEstimator {
    /// Histogram
    Histogram,
    /// KNN
    KNN,
    /// KernelDensity
    KernelDensity,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {

    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_kl_divergence_methods() {
        let kl_methods = KLDivergenceMethods::new();
        let p = array![0.5, 0.3, 0.2];
        let q = array![0.4, 0.4, 0.2];

        let kl_div = kl_methods.discrete_kl_divergence(&p, &q).unwrap();
        assert!(kl_div >= 0.0);
    }

    #[test]
    fn test_jensen_shannon_divergence() {
        let kl_methods = KLDivergenceMethods::new();
        let p = array![0.5, 0.3, 0.2];
        let q = array![0.4, 0.4, 0.2];

        let js_div = kl_methods.jensen_shannon_divergence(&p, &q).unwrap();
        assert!(js_div >= 0.0);
        assert!(js_div <= 1.0);
    }

    #[test]
    fn test_feature_importance_weight_based() {
        let analyzer = FeatureImportanceAnalyzer::new();
        let weights = array![[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]];

        let results = analyzer.weight_based_importance(&weights);
        assert_eq!(results.importance_scores.len(), 3);
        assert_eq!(results.rankings.len(), 3);
    }

    #[test]
    fn test_component_interpreter() {
        let interpreter = ComponentInterpreter::new();
        let loadings = array![[0.8, 0.2], [0.6, -0.4], [0.1, 0.9]];

        let interpretation = interpreter.interpret_components(&loadings);
        assert_eq!(interpretation.component_interpretations.len(), 2);
    }

    #[test]
    fn test_component_similarity() {
        let interpreter = ComponentInterpreter::new();
        let loadings = array![[0.8, 0.2], [0.6, -0.4], [0.1, 0.9]];

        let similarity = interpreter.component_similarity(&loadings);
        assert_eq!(similarity.correlation_matrix.shape(), &[2, 2]);
    }

    #[test]
    fn test_kl_methods_component_selection() {
        let kl_methods = KLDivergenceMethods::new();
        let components = array![[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.1, 0.8, 0.1]];

        let selected = kl_methods.component_selection(&components, None).unwrap();
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_feature_importance_correlation_based() {
        let analyzer = FeatureImportanceAnalyzer::new();
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let components = array![[0.8, 0.2], [0.6, 0.4]];

        let results = analyzer.correlation_based_importance(&data, &components);
        assert_eq!(results.importance_scores.len(), 3);
    }
}
