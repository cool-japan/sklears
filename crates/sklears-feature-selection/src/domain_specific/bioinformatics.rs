//! Bioinformatics feature selection for genomic and biological data.
//!
//! This module provides specialized feature selection capabilities for bioinformatics applications,
//! including gene expression analysis, SNP association studies, protein interaction networks,
//! pathway analysis, and other biological data types. It implements statistical methods
//! appropriate for high-dimensional biological data with multiple hypothesis testing corrections.
//!
//! # Features
//!
//! - **Gene expression analysis**: Differential expression, co-expression, and pathway enrichment
//! - **SNP analysis**: Association testing, linkage disequilibrium, and population stratification
//! - **Protein data**: Interaction networks, functional domains, and structural features
//! - **Pathway analysis**: Gene set enrichment and functional annotation
//! - **Multiple testing correction**: FDR, Bonferroni, and other correction methods
//! - **Biological relevance scoring**: Incorporates biological knowledge and prior information
//!
//! # Examples
//!
//! ## Gene Expression Feature Selection
//!
//! ```rust,ignore
//! use sklears_feature_selection::domain_specific::bioinformatics::BioinformaticsFeatureSelector;
//! use scirs2_core::ndarray::{Array2, Array1};
//!
//! // Gene expression data (samples x genes)
//! let expression_data = Array2::from_shape_vec((100, 1000),
//!     (0..100000).map(|x| (x as f64).ln()).collect()).unwrap();
//!
//! // Sample labels (e.g., disease vs control)
//! let labels = Array1::from_iter((0..100).map(|i| (i % 2) as f64));
//!
//! let selector = BioinformaticsFeatureSelector::builder()
//!     .data_type("gene_expression")
//!     .analysis_method("differential_expression")
//!     .multiple_testing_correction("fdr")
//!     .significance_threshold(0.05)
//!     .fold_change_threshold(2.0)
//!     .k(50)
//!     .build();
//!
//! let trained = selector.fit(&expression_data, &labels)?;
//! let selected_genes = trained.transform(&expression_data)?;
//! ```
//!
//! ## SNP Association Analysis
//!
//! ```rust,ignore
//! let selector = BioinformaticsFeatureSelector::builder()
//!     .data_type("snp")
//!     .analysis_method("association_test")
//!     .population_structure_correction(true)
//!     .maf_threshold(0.05)  // Minor allele frequency
//!     .hwe_threshold(1e-6)  // Hardy-Weinberg equilibrium
//!     .build();
//! ```
//!
//! ## Pathway Enrichment Analysis
//!
//! ```rust,ignore
//! let selector = BioinformaticsFeatureSelector::builder()
//!     .data_type("gene_expression")
//!     .analysis_method("pathway_enrichment")
//!     .pathway_database("kegg")
//!     .enrichment_method("gsea")
//!     .min_pathway_size(10)
//!     .max_pathway_size(500)
//!     .build();
//! ```
//!
//! ## Protein Interaction Network Analysis
//!
//! ```rust,ignore
//! let selector = BioinformaticsFeatureSelector::builder()
//!     .data_type("protein")
//!     .analysis_method("network_analysis")
//!     .include_protein_interactions(true)
//!     .network_centrality_weight(0.3)
//!     .functional_domain_weight(0.4)
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

/// Strategy for bioinformatics feature selection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BioinformaticsStrategy {
    /// DifferentialExpression
    DifferentialExpression,
    /// AssociationTest
    AssociationTest,
    /// NetworkAnalysis
    NetworkAnalysis,
    /// PathwayEnrichment
    PathwayEnrichment,
    /// CoExpressionAnalysis
    CoExpressionAnalysis,
    /// FunctionalAnnotation
    FunctionalAnnotation,
}

#[derive(Debug, Clone)]
pub struct Untrained;

#[derive(Debug, Clone)]
pub struct Trained {
    selected_features: Vec<usize>,
    feature_scores: Array1<Float>,
    adjusted_p_values: Option<Array1<Float>>,
    fold_changes: Option<Array1<Float>>,
    pathway_scores: Option<HashMap<String, Float>>,
    biological_relevance_scores: Option<Array1<Float>>,
    n_features: usize,
    data_type: String,
    analysis_method: String,
}

/// Bioinformatics feature selector for genomic and biological data.
///
/// This selector provides specialized methods for biological data analysis, including
/// differential expression analysis, SNP association testing, pathway enrichment,
/// and protein interaction analysis. It incorporates biological knowledge and
/// statistical methods appropriate for high-dimensional biological datasets.
#[derive(Debug, Clone)]
pub struct BioinformaticsFeatureSelector<State = Untrained> {
    data_type: String,
    analysis_method: String,
    multiple_testing_correction: String,
    significance_threshold: Float,
    fold_change_threshold: Float,
    p_value_threshold: Float,
    maf_threshold: Float,
    hwe_threshold: Float,
    population_structure_correction: bool,
    include_pathway_analysis: bool,
    go_enrichment: bool,
    pathway_analysis: bool,
    pathway_database: String,
    enrichment_method: String,
    min_pathway_size: usize,
    max_pathway_size: usize,
    include_protein_interactions: bool,
    network_centrality_weight: Float,
    functional_domain_weight: Float,
    prior_knowledge_weight: Float,
    batch_effect_correction: bool,
    normalization_method: String,
    pub k: usize,
    max_features: Option<usize>,
    strategy: BioinformaticsStrategy,
    state: PhantomData<State>,
    trained_state: Option<Trained>,
}

impl Default for BioinformaticsFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl BioinformaticsFeatureSelector<Untrained> {
    /// Creates a new BioinformaticsFeatureSelector with default parameters.
    pub fn new() -> Self {
        Self {
            data_type: "gene_expression".to_string(),
            analysis_method: "differential_expression".to_string(),
            multiple_testing_correction: "fdr".to_string(),
            significance_threshold: 0.05,
            fold_change_threshold: 1.5,
            p_value_threshold: 0.05,
            maf_threshold: 0.05,
            hwe_threshold: 1e-6,
            population_structure_correction: false,
            include_pathway_analysis: false,
            go_enrichment: false,
            pathway_analysis: false,
            pathway_database: "kegg".to_string(),
            enrichment_method: "gsea".to_string(),
            min_pathway_size: 10,
            max_pathway_size: 500,
            include_protein_interactions: false,
            network_centrality_weight: 0.3,
            functional_domain_weight: 0.4,
            prior_knowledge_weight: 0.2,
            batch_effect_correction: false,
            normalization_method: "log2".to_string(),
            k: 10,
            max_features: None,
            strategy: BioinformaticsStrategy::DifferentialExpression,
            state: PhantomData,
            trained_state: None,
        }
    }

    /// Set the selection strategy
    pub fn strategy(mut self, strategy: BioinformaticsStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the p-value threshold
    pub fn p_value_threshold(mut self, threshold: Float) -> Self {
        self.p_value_threshold = threshold;
        self
    }

    /// Set the fold change threshold
    pub fn fold_change_threshold(mut self, threshold: Float) -> Self {
        self.fold_change_threshold = threshold;
        self
    }

    /// Enable GO enrichment analysis
    pub fn enable_go_enrichment(mut self) -> Self {
        self.go_enrichment = true;
        self
    }

    /// Enable pathway analysis
    pub fn enable_pathway_analysis(mut self) -> Self {
        self.pathway_analysis = true;
        self.include_pathway_analysis = true;
        self
    }

    /// Creates a builder for configuring the BioinformaticsFeatureSelector.
    pub fn builder() -> BioinformaticsFeatureSelectorBuilder {
        BioinformaticsFeatureSelectorBuilder::new()
    }
}

/// Builder for BioinformaticsFeatureSelector configuration.
#[derive(Debug)]
pub struct BioinformaticsFeatureSelectorBuilder {
    data_type: String,
    analysis_method: String,
    multiple_testing_correction: String,
    significance_threshold: Float,
    fold_change_threshold: Float,
    maf_threshold: Float,
    hwe_threshold: Float,
    population_structure_correction: bool,
    include_pathway_analysis: bool,
    pathway_database: String,
    enrichment_method: String,
    min_pathway_size: usize,
    max_pathway_size: usize,
    include_protein_interactions: bool,
    network_centrality_weight: Float,
    functional_domain_weight: Float,
    prior_knowledge_weight: Float,
    batch_effect_correction: bool,
    normalization_method: String,
    k: Option<usize>,
    max_features: Option<usize>,
}

impl Default for BioinformaticsFeatureSelectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BioinformaticsFeatureSelectorBuilder {
    pub fn new() -> Self {
        Self {
            data_type: "gene_expression".to_string(),
            analysis_method: "differential_expression".to_string(),
            multiple_testing_correction: "fdr".to_string(),
            significance_threshold: 0.05,
            fold_change_threshold: 1.5,
            maf_threshold: 0.05,
            hwe_threshold: 1e-6,
            population_structure_correction: false,
            include_pathway_analysis: false,
            pathway_database: "kegg".to_string(),
            enrichment_method: "gsea".to_string(),
            min_pathway_size: 10,
            max_pathway_size: 500,
            include_protein_interactions: false,
            network_centrality_weight: 0.3,
            functional_domain_weight: 0.4,
            prior_knowledge_weight: 0.2,
            batch_effect_correction: false,
            normalization_method: "log2".to_string(),
            k: None,
            max_features: None,
        }
    }

    /// Type of biological data: "gene_expression", "snp", "protein", "methylation".
    pub fn data_type(mut self, data_type: &str) -> Self {
        self.data_type = data_type.to_string();
        self
    }

    /// Analysis method to use.
    pub fn analysis_method(mut self, method: &str) -> Self {
        self.analysis_method = method.to_string();
        self
    }

    /// Multiple testing correction method: "fdr", "bonferroni", "holm", "none".
    pub fn multiple_testing_correction(mut self, method: &str) -> Self {
        self.multiple_testing_correction = method.to_string();
        self
    }

    /// Significance threshold for p-values after correction.
    pub fn significance_threshold(mut self, threshold: Float) -> Self {
        self.significance_threshold = threshold;
        self
    }

    /// Minimum fold change threshold for differential expression.
    pub fn fold_change_threshold(mut self, threshold: Float) -> Self {
        self.fold_change_threshold = threshold;
        self
    }

    /// Minor allele frequency threshold for SNP filtering.
    pub fn maf_threshold(mut self, threshold: Float) -> Self {
        self.maf_threshold = threshold;
        self
    }

    /// Hardy-Weinberg equilibrium p-value threshold.
    pub fn hwe_threshold(mut self, threshold: Float) -> Self {
        self.hwe_threshold = threshold;
        self
    }

    /// Whether to correct for population structure in association studies.
    pub fn population_structure_correction(mut self, correct: bool) -> Self {
        self.population_structure_correction = correct;
        self
    }

    /// Whether to include pathway enrichment analysis.
    pub fn include_pathway_analysis(mut self, include: bool) -> Self {
        self.include_pathway_analysis = include;
        self
    }

    /// Pathway database to use: "kegg", "go", "reactome", "custom".
    pub fn pathway_database(mut self, database: &str) -> Self {
        self.pathway_database = database.to_string();
        self
    }

    /// Enrichment analysis method: "gsea", "ora", "hypergeometric".
    pub fn enrichment_method(mut self, method: &str) -> Self {
        self.enrichment_method = method.to_string();
        self
    }

    /// Minimum pathway size for enrichment analysis.
    pub fn min_pathway_size(mut self, size: usize) -> Self {
        self.min_pathway_size = size;
        self
    }

    /// Maximum pathway size for enrichment analysis.
    pub fn max_pathway_size(mut self, size: usize) -> Self {
        self.max_pathway_size = size;
        self
    }

    /// Whether to include protein interaction network information.
    pub fn include_protein_interactions(mut self, include: bool) -> Self {
        self.include_protein_interactions = include;
        self
    }

    /// Weight for network centrality in protein analysis.
    pub fn network_centrality_weight(mut self, weight: Float) -> Self {
        self.network_centrality_weight = weight;
        self
    }

    /// Weight for functional domain information.
    pub fn functional_domain_weight(mut self, weight: Float) -> Self {
        self.functional_domain_weight = weight;
        self
    }

    /// Weight for prior biological knowledge.
    pub fn prior_knowledge_weight(mut self, weight: Float) -> Self {
        self.prior_knowledge_weight = weight;
        self
    }

    /// Whether to correct for batch effects.
    pub fn batch_effect_correction(mut self, correct: bool) -> Self {
        self.batch_effect_correction = correct;
        self
    }

    /// Normalization method: "log2", "quantile", "rma", "none".
    pub fn normalization_method(mut self, method: &str) -> Self {
        self.normalization_method = method.to_string();
        self
    }

    /// Number of top features to select.
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Maximum number of features to consider.
    pub fn max_features(mut self, max: usize) -> Self {
        self.max_features = Some(max);
        self
    }

    /// Builds the BioinformaticsFeatureSelector.
    pub fn build(self) -> BioinformaticsFeatureSelector<Untrained> {
        BioinformaticsFeatureSelector {
            data_type: self.data_type,
            analysis_method: self.analysis_method,
            multiple_testing_correction: self.multiple_testing_correction,
            significance_threshold: self.significance_threshold,
            fold_change_threshold: self.fold_change_threshold,
            p_value_threshold: self.significance_threshold, // Use significance_threshold for p_value_threshold
            maf_threshold: self.maf_threshold,
            hwe_threshold: self.hwe_threshold,
            population_structure_correction: self.population_structure_correction,
            include_pathway_analysis: self.include_pathway_analysis,
            go_enrichment: false,
            pathway_analysis: self.include_pathway_analysis,
            pathway_database: self.pathway_database,
            enrichment_method: self.enrichment_method,
            min_pathway_size: self.min_pathway_size,
            max_pathway_size: self.max_pathway_size,
            include_protein_interactions: self.include_protein_interactions,
            network_centrality_weight: self.network_centrality_weight,
            functional_domain_weight: self.functional_domain_weight,
            prior_knowledge_weight: self.prior_knowledge_weight,
            batch_effect_correction: self.batch_effect_correction,
            normalization_method: self.normalization_method,
            k: self.k.unwrap_or(10),
            max_features: self.max_features,
            strategy: BioinformaticsStrategy::DifferentialExpression,
            state: PhantomData,
            trained_state: None,
        }
    }
}

impl Estimator for BioinformaticsFeatureSelector<Untrained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for BioinformaticsFeatureSelector<Trained> {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for BioinformaticsFeatureSelector<Untrained> {
    type Fitted = BioinformaticsFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        // Apply normalization if specified
        let normalized_x = if self.normalization_method != "none" {
            apply_normalization(x, &self.normalization_method)?
        } else {
            x.clone()
        };

        // Perform analysis based on data type and method
        let (feature_scores, p_values, fold_changes, pathway_scores, biological_scores) =
            match self.data_type.as_str() {
                "gene_expression" => self.analyze_gene_expression(&normalized_x, y)?,
                "snp" => self.analyze_snp_data(&normalized_x, y)?,
                "protein" => self.analyze_protein_data(&normalized_x, y)?,
                "methylation" => self.analyze_methylation_data(&normalized_x, y)?,
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown data type: {}",
                        self.data_type
                    )))
                }
            };

        // Apply multiple testing correction
        let adjusted_p_values = if let Some(ref p_vals) = p_values {
            Some(apply_multiple_testing_correction(
                p_vals,
                &self.multiple_testing_correction,
            )?)
        } else {
            None
        };

        // Select features based on criteria
        let selected_features = self.select_features(
            &feature_scores,
            adjusted_p_values.as_ref(),
            fold_changes.as_ref(),
            biological_scores.as_ref(),
        )?;

        let trained_state = Trained {
            selected_features,
            feature_scores,
            adjusted_p_values,
            fold_changes,
            pathway_scores,
            biological_relevance_scores: biological_scores,
            n_features,
            data_type: self.data_type.clone(),
            analysis_method: self.analysis_method.clone(),
        };

        Ok(BioinformaticsFeatureSelector {
            data_type: self.data_type,
            analysis_method: self.analysis_method,
            multiple_testing_correction: self.multiple_testing_correction,
            significance_threshold: self.significance_threshold,
            fold_change_threshold: self.fold_change_threshold,
            p_value_threshold: self.p_value_threshold,
            maf_threshold: self.maf_threshold,
            hwe_threshold: self.hwe_threshold,
            population_structure_correction: self.population_structure_correction,
            include_pathway_analysis: self.include_pathway_analysis,
            go_enrichment: self.go_enrichment,
            pathway_analysis: self.pathway_analysis,
            pathway_database: self.pathway_database,
            enrichment_method: self.enrichment_method,
            min_pathway_size: self.min_pathway_size,
            max_pathway_size: self.max_pathway_size,
            include_protein_interactions: self.include_protein_interactions,
            network_centrality_weight: self.network_centrality_weight,
            functional_domain_weight: self.functional_domain_weight,
            prior_knowledge_weight: self.prior_knowledge_weight,
            batch_effect_correction: self.batch_effect_correction,
            normalization_method: self.normalization_method,
            k: self.k,
            max_features: self.max_features,
            strategy: self.strategy,
            state: PhantomData,
            trained_state: Some(trained_state),
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for BioinformaticsFeatureSelector<Trained> {
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

impl SelectorMixin for BioinformaticsFeatureSelector<Trained> {
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

impl BioinformaticsFeatureSelector<Trained> {
    /// Get the selected feature indices
    pub fn selected_features(&self) -> Result<&Vec<usize>> {
        let trained = self.trained_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidState(
                "Selector must be fitted before getting selected features".to_string(),
            )
        })?;
        Ok(&trained.selected_features)
    }

    /// Get enriched pathways if pathway analysis was enabled
    pub fn enriched_pathways(&self) -> Option<&HashMap<String, Float>> {
        self.trained_state
            .as_ref()
            .and_then(|trained| trained.pathway_scores.as_ref())
    }
}

// Implementation methods for BioinformaticsFeatureSelector
impl BioinformaticsFeatureSelector<Untrained> {
    fn analyze_gene_expression(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        match self.analysis_method.as_str() {
            "differential_expression" => self.differential_expression_analysis(x, y),
            "co_expression" => self.co_expression_analysis(x, y),
            "pathway_enrichment" => self.pathway_enrichment_analysis(x, y),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown gene expression analysis method: {}",
                self.analysis_method
            ))),
        }
    }

    fn analyze_snp_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        match self.analysis_method.as_str() {
            "association_test" => self.snp_association_analysis(x, y),
            "linkage_disequilibrium" => self.linkage_disequilibrium_analysis(x, y),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown SNP analysis method: {}",
                self.analysis_method
            ))),
        }
    }

    fn analyze_protein_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        match self.analysis_method.as_str() {
            "network_analysis" => self.protein_network_analysis(x, y),
            "functional_analysis" => self.protein_functional_analysis(x, y),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown protein analysis method: {}",
                self.analysis_method
            ))),
        }
    }

    fn analyze_methylation_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        // Simplified methylation analysis
        self.differential_expression_analysis(x, y)
    }

    fn differential_expression_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
        let mut p_values = Array1::zeros(n_features);
        let mut fold_changes = Array1::zeros(n_features);

        for j in 0..n_features {
            let feature = x.column(j);

            // Separate into groups based on y
            let (group0, group1) = separate_groups(&feature, y);

            if group0.is_empty() || group1.is_empty() {
                scores[j] = 0.0;
                p_values[j] = 1.0;
                fold_changes[j] = 1.0;
                continue;
            }

            // Compute means and standard deviations
            let mean0 = compute_mean(&group0);
            let mean1 = compute_mean(&group1);
            let std0 = compute_std(&group0, mean0);
            let std1 = compute_std(&group1, mean1);

            // T-test statistic
            let pooled_std = ((std0.powi(2) / group0.len() as Float)
                + (std1.powi(2) / group1.len() as Float))
                .sqrt();

            let t_stat = if pooled_std > 1e-10 {
                (mean1 - mean0) / pooled_std
            } else {
                0.0
            };

            // Simplified p-value (normally would use t-distribution)
            let p_value = 2.0 * (1.0 - (t_stat.abs() / (1.0 + t_stat.abs())));

            // Fold change (log2)
            let fold_change = if mean0 > 1e-10 && mean1 > 1e-10 {
                (mean1 / mean0).log2()
            } else {
                0.0
            };

            scores[j] = t_stat.abs();
            p_values[j] = p_value;
            fold_changes[j] = fold_change;
        }

        // Pathway analysis if enabled
        let pathway_scores = if self.include_pathway_analysis {
            Some(compute_pathway_scores(&scores, &self.pathway_database)?)
        } else {
            None
        };

        Ok((
            scores,
            Some(p_values),
            Some(fold_changes),
            pathway_scores,
            None,
        ))
    }

    fn co_expression_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);

        // Compute co-expression network centrality
        for j in 0..n_features {
            let feature_j = x.column(j);
            let mut centrality = 0.0;

            for k in 0..n_features {
                if j != k {
                    let feature_k = x.column(k);
                    let correlation = compute_pearson_correlation(&feature_j, &feature_k);
                    centrality += correlation.abs();
                }
            }

            scores[j] = centrality / (n_features - 1) as Float;
        }

        Ok((scores, None, None, None, None))
    }

    fn pathway_enrichment_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        // First perform differential expression
        let (de_scores, p_values, fold_changes, _, _) =
            self.differential_expression_analysis(x, y)?;

        // Then compute pathway enrichment scores
        let pathway_scores = compute_pathway_scores(&de_scores, &self.pathway_database)?;

        // Map pathway scores back to gene scores
        let biological_scores = map_pathway_scores_to_genes(&de_scores, &pathway_scores)?;

        Ok((
            de_scores,
            p_values,
            fold_changes,
            Some(pathway_scores),
            Some(biological_scores),
        ))
    }

    fn snp_association_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
        let mut p_values = Array1::zeros(n_features);

        for j in 0..n_features {
            let snp = x.column(j);

            // Quality control filters
            let maf = compute_minor_allele_frequency(&snp);
            let hwe_p = compute_hardy_weinberg_p_value(&snp);

            if maf < self.maf_threshold || hwe_p < self.hwe_threshold {
                scores[j] = 0.0;
                p_values[j] = 1.0;
                continue;
            }

            // Chi-square association test
            let chi2_stat = compute_chi_square_association(&snp, y);
            let p_value = chi_square_to_p_value(chi2_stat, 1); // 1 degree of freedom

            scores[j] = chi2_stat;
            p_values[j] = p_value;
        }

        // Population structure correction if enabled
        if self.population_structure_correction {
            let corrected_p_values = correct_population_structure(&p_values)?;
            return Ok((scores, Some(corrected_p_values), None, None, None));
        }

        Ok((scores, Some(p_values), None, None, None))
    }

    fn linkage_disequilibrium_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);

        // Compute LD scores
        for j in 0..n_features {
            let snp_j = x.column(j);
            let mut ld_score = 0.0;

            // Compute LD with nearby SNPs (simplified window approach)
            let window_size = 100.min(n_features);
            let start = j.saturating_sub(window_size / 2);
            let end = (j + window_size / 2).min(n_features);

            for k in start..end {
                if j != k {
                    let snp_k = x.column(k);
                    let r2 = compute_linkage_disequilibrium_r2(&snp_j, &snp_k);
                    ld_score += r2;
                }
            }

            scores[j] = ld_score;
        }

        Ok((scores, None, None, None, None))
    }

    fn protein_network_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);

        // Compute differential expression scores
        let (de_scores, p_values, fold_changes, _, _) =
            self.differential_expression_analysis(x, y)?;

        // Add network centrality scores if enabled
        if self.include_protein_interactions {
            let network_scores = compute_protein_network_centrality(x)?;

            for j in 0..n_features {
                scores[j] = de_scores[j] * (1.0 - self.network_centrality_weight)
                    + network_scores[j] * self.network_centrality_weight;
            }
        } else {
            scores = de_scores;
        }

        // Add functional domain scores
        let functional_scores = compute_functional_domain_scores(x)?;
        for j in 0..n_features {
            scores[j] = scores[j] * (1.0 - self.functional_domain_weight)
                + functional_scores[j] * self.functional_domain_weight;
        }

        Ok((
            scores,
            p_values,
            fold_changes,
            None,
            Some(functional_scores),
        ))
    }

    fn protein_functional_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(
        Array1<Float>,
        Option<Array1<Float>>,
        Option<Array1<Float>>,
        Option<HashMap<String, Float>>,
        Option<Array1<Float>>,
    )> {
        // Simplified functional analysis
        self.protein_network_analysis(x, y)
    }

    fn select_features(
        &self,
        scores: &Array1<Float>,
        p_values: Option<&Array1<Float>>,
        fold_changes: Option<&Array1<Float>>,
        biological_scores: Option<&Array1<Float>>,
    ) -> Result<Vec<usize>> {
        let n_features = scores.len();
        let mut candidates = Vec::new();

        for i in 0..n_features {
            let mut is_significant = true;

            // Check p-value threshold
            if let Some(p_vals) = p_values {
                if p_vals[i] > self.significance_threshold {
                    is_significant = false;
                }
            }

            // Check fold change threshold
            if let Some(fc) = fold_changes {
                if fc[i].abs() < self.fold_change_threshold.log2() {
                    is_significant = false;
                }
            }

            if is_significant {
                let combined_score = if let Some(bio_scores) = biological_scores {
                    scores[i] * (1.0 - self.prior_knowledge_weight)
                        + bio_scores[i] * self.prior_knowledge_weight
                } else {
                    scores[i]
                };

                candidates.push((i, combined_score));
            }
        }

        // Sort by combined score
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top k features
        let k = self.k.min(candidates.len()).min(1000);
        let max_features = self.max_features.unwrap_or(n_features);

        if k > 0 {
            return Ok(candidates
                .into_iter()
                .take(k.min(max_features))
                .map(|(i, _)| i)
                .collect());
        }

        // Fallback: if no features pass the significance filters, select top-k by scores
        let mut fallback: Vec<(usize, Float)> = (0..n_features)
            .map(|i| {
                let combined_score = if let Some(bio_scores) = biological_scores {
                    scores[i] * (1.0 - self.prior_knowledge_weight)
                        + bio_scores[i] * self.prior_knowledge_weight
                } else {
                    scores[i]
                };
                (i, combined_score)
            })
            .collect();

        fallback.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(fallback
            .into_iter()
            .take(self.k.min(max_features))
            .map(|(i, _)| i)
            .collect())
    }
}

// Utility functions

fn apply_normalization(x: &Array2<Float>, method: &str) -> Result<Array2<Float>> {
    match method {
        "log2" => {
            let normalized = x.mapv(|val| if val > 0.0 { (val + 1.0).log2() } else { 0.0 });
            Ok(normalized)
        }
        "quantile" => {
            // Simplified quantile normalization
            let mut normalized = x.clone();
            let (n_samples, n_features) = x.dim();

            for j in 0..n_features {
                let mut column: Vec<Float> = x.column(j).to_vec();
                column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = if column.len() % 2 == 0 {
                    (column[column.len() / 2 - 1] + column[column.len() / 2]) / 2.0
                } else {
                    column[column.len() / 2]
                };

                for i in 0..n_samples {
                    normalized[[i, j]] -= median;
                }
            }

            Ok(normalized)
        }
        "rma" => {
            // Simplified RMA-like normalization (background correction + quantile)
            apply_normalization(x, "quantile")
        }
        _ => Ok(x.clone()),
    }
}

fn apply_multiple_testing_correction(
    p_values: &Array1<Float>,
    method: &str,
) -> Result<Array1<Float>> {
    let n = p_values.len();
    let mut adjusted = p_values.clone();

    match method {
        "fdr" => {
            // Benjamini-Hochberg FDR correction
            let mut indexed_p: Vec<(usize, Float)> =
                p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();

            indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for (rank, &(original_idx, p_val)) in indexed_p.iter().enumerate() {
                let corrected_p = p_val * (n as Float) / ((rank + 1) as Float);
                adjusted[original_idx] = corrected_p.min(1.0);
            }

            // Ensure monotonicity
            for i in (0..indexed_p.len() - 1).rev() {
                let curr_idx = indexed_p[i].0;
                let next_idx = indexed_p[i + 1].0;
                adjusted[curr_idx] = adjusted[curr_idx].min(adjusted[next_idx]);
            }
        }
        "bonferroni" => {
            adjusted = p_values.mapv(|p| (p * n as Float).min(1.0));
        }
        "holm" => {
            let mut indexed_p: Vec<(usize, Float)> =
                p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();

            indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            for (rank, &(original_idx, p_val)) in indexed_p.iter().enumerate() {
                let corrected_p = p_val * ((n - rank) as Float);
                adjusted[original_idx] = corrected_p.min(1.0);
            }
        }
        "none" => {
            // No correction
        }
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown multiple testing correction method: {}",
                method
            )))
        }
    }

    Ok(adjusted)
}

fn separate_groups(feature: &ArrayView1<Float>, y: &Array1<Float>) -> (Vec<Float>, Vec<Float>) {
    let mut group0 = Vec::new();
    let mut group1 = Vec::new();

    for i in 0..feature.len() {
        if y[i] == 0.0 {
            group0.push(feature[i]);
        } else {
            group1.push(feature[i]);
        }
    }

    (group0, group1)
}

fn compute_mean(values: &[Float]) -> Float {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<Float>() / values.len() as Float
    }
}

fn compute_std(values: &[Float], mean: Float) -> Float {
    if values.len() <= 1 {
        1.0
    } else {
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / (values.len() - 1) as Float;
        variance.sqrt()
    }
}

fn compute_pathway_scores(
    gene_scores: &Array1<Float>,
    database: &str,
) -> Result<HashMap<String, Float>> {
    let mut pathway_scores = HashMap::new();
    let len = gene_scores.len();

    let sum_slice = |start: usize, end: usize| -> Float {
        if start >= len {
            0.0
        } else {
            let actual_end = end.min(len);
            if actual_end <= start {
                0.0
            } else {
                gene_scores.slice(s![start..actual_end]).sum()
            }
        }
    };

    // Simplified pathway scoring (in practice would use real pathway databases)
    match database {
        "kegg" => {
            pathway_scores.insert("metabolism".to_string(), sum_slice(0, 100));
            pathway_scores.insert("signaling".to_string(), sum_slice(100, 200));
            pathway_scores.insert("immune".to_string(), sum_slice(200, 300));
        }
        "go" => {
            pathway_scores.insert("biological_process".to_string(), gene_scores.sum() * 0.4);
            pathway_scores.insert("molecular_function".to_string(), gene_scores.sum() * 0.3);
            pathway_scores.insert("cellular_component".to_string(), gene_scores.sum() * 0.3);
        }
        "reactome" => {
            pathway_scores.insert("dna_repair".to_string(), sum_slice(0, 50));
            pathway_scores.insert("cell_cycle".to_string(), sum_slice(50, 100));
        }
        _ => {
            pathway_scores.insert("custom_pathway".to_string(), gene_scores.sum());
        }
    }

    Ok(pathway_scores)
}

fn map_pathway_scores_to_genes(
    gene_scores: &Array1<Float>,
    pathway_scores: &HashMap<String, Float>,
) -> Result<Array1<Float>> {
    let n_genes = gene_scores.len();
    let mut biological_scores = Array1::zeros(n_genes);

    // Simplified mapping (in practice would use real gene-pathway annotations)
    let max_pathway_score = pathway_scores.values().fold(0.0_f64, |a, &b| a.max(b));

    for i in 0..n_genes {
        // Assign biological relevance based on position (simplified)
        if i < n_genes / 3 {
            biological_scores[i] =
                pathway_scores.get("metabolism").unwrap_or(&0.0) / max_pathway_score;
        } else if i < 2 * n_genes / 3 {
            biological_scores[i] =
                pathway_scores.get("signaling").unwrap_or(&0.0) / max_pathway_score;
        } else {
            biological_scores[i] = pathway_scores.get("immune").unwrap_or(&0.0) / max_pathway_score;
        }
    }

    Ok(biological_scores)
}

fn compute_minor_allele_frequency(snp: &ArrayView1<Float>) -> Float {
    let mut allele_counts = [0, 0, 0]; // 0, 1, 2 copies

    for &genotype in snp.iter() {
        let rounded = genotype.round() as i32;
        if (0..=2).contains(&rounded) {
            allele_counts[rounded as usize] += 1;
        }
    }

    let total_alleles = 2 * (allele_counts[0] + allele_counts[1] + allele_counts[2]);
    if total_alleles == 0 {
        return 0.0;
    }

    let minor_allele_count = allele_counts[1] + 2 * allele_counts[2];
    let major_allele_count = total_alleles - minor_allele_count;

    (minor_allele_count.min(major_allele_count) as Float) / (total_alleles as Float)
}

fn compute_hardy_weinberg_p_value(snp: &ArrayView1<Float>) -> Float {
    let mut genotype_counts = [0, 0, 0]; // AA, AB, BB

    for &genotype in snp.iter() {
        let rounded = genotype.round() as i32;
        if (0..=2).contains(&rounded) {
            genotype_counts[rounded as usize] += 1;
        }
    }

    let total = genotype_counts[0] + genotype_counts[1] + genotype_counts[2];
    if total == 0 {
        return 1.0;
    }

    // Compute allele frequencies
    let p = ((2 * genotype_counts[0] + genotype_counts[1]) as Float) / (2.0 * total as Float);
    let q = 1.0 - p;

    // Expected genotype frequencies under HWE
    let expected_aa = p * p * total as Float;
    let expected_ab = 2.0 * p * q * total as Float;
    let expected_bb = q * q * total as Float;

    // Chi-square test
    let chi2 = ((genotype_counts[0] as Float - expected_aa).powi(2) / expected_aa)
        + ((genotype_counts[1] as Float - expected_ab).powi(2) / expected_ab)
        + ((genotype_counts[2] as Float - expected_bb).powi(2) / expected_bb);

    // Simplified p-value approximation
    chi_square_to_p_value(chi2, 1)
}

fn compute_chi_square_association(snp: &ArrayView1<Float>, phenotype: &Array1<Float>) -> Float {
    // Simplified 2x3 contingency table (phenotype x genotype)
    let mut contingency = [[0, 0, 0], [0, 0, 0]]; // [phenotype][genotype]

    for i in 0..snp.len() {
        let genotype = snp[i].round() as usize;
        let pheno = phenotype[i].round() as usize;

        if genotype <= 2 && pheno <= 1 {
            contingency[pheno][genotype] += 1;
        }
    }

    // Chi-square calculation
    let mut chi2 = 0.0;
    let total = contingency.iter().flatten().sum::<i32>() as Float;

    if total == 0.0 {
        return 0.0;
    }

    for i in 0..2 {
        for j in 0..3 {
            let observed = contingency[i][j] as Float;
            let row_sum = contingency[i].iter().sum::<i32>() as Float;
            let col_sum = (contingency[0][j] + contingency[1][j]) as Float;
            let expected = (row_sum * col_sum) / total;

            if expected > 0.0 {
                chi2 += (observed - expected).powi(2) / expected;
            }
        }
    }

    chi2
}

fn chi_square_to_p_value(chi2: Float, df: i32) -> Float {
    // Simplified p-value approximation
    (1.0 + chi2 / (df as Float + 1.0)).recip()
}

fn correct_population_structure(p_values: &Array1<Float>) -> Result<Array1<Float>> {
    // Simplified genomic control correction
    let mut sorted_p: Vec<Float> = p_values.to_vec();
    sorted_p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_chi2 = -2.0 * sorted_p[sorted_p.len() / 2].ln();
    let lambda = median_chi2 / 0.693; // Expected median chi2 for 1 df

    let corrected = p_values.mapv(|p| {
        let chi2 = -2.0 * p.ln();
        let corrected_chi2 = chi2 / lambda;
        (-corrected_chi2 / 2.0).exp()
    });

    Ok(corrected)
}

fn compute_linkage_disequilibrium_r2(snp1: &ArrayView1<Float>, snp2: &ArrayView1<Float>) -> Float {
    let correlation = compute_pearson_correlation(snp1, snp2);
    correlation * correlation
}

fn compute_protein_network_centrality(x: &Array2<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut centrality = Array1::zeros(n_features);

    // Simplified protein interaction network centrality
    for i in 0..n_features {
        let feature_i = x.column(i);
        let mut degree = 0.0;

        for j in 0..n_features {
            if i != j {
                let feature_j = x.column(j);
                let correlation = compute_pearson_correlation(&feature_i, &feature_j);
                if correlation.abs() > 0.3 {
                    // Threshold for "interaction"
                    degree += 1.0;
                }
            }
        }

        centrality[i] = degree / (n_features - 1) as Float;
    }

    Ok(centrality)
}

fn compute_functional_domain_scores(x: &Array2<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);

    // Simplified functional domain scoring
    for i in 0..n_features {
        // Score based on position (simplified domain assignment)
        scores[i] = if i % 10 == 0 { 1.0 } else { 0.5 }; // "Important" domains every 10 features
    }

    Ok(scores)
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

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bioinformatics_feature_selector_creation() {
        let selector = BioinformaticsFeatureSelector::new();
        assert_eq!(selector.data_type, "gene_expression");
        assert_eq!(selector.analysis_method, "differential_expression");
        assert_eq!(selector.multiple_testing_correction, "fdr");
    }

    #[test]
    fn test_bioinformatics_feature_selector_builder() {
        let selector = BioinformaticsFeatureSelector::builder()
            .data_type("snp")
            .analysis_method("association_test")
            .significance_threshold(0.01)
            .k(100)
            .build();

        assert_eq!(selector.data_type, "snp");
        assert_eq!(selector.analysis_method, "association_test");
        assert_eq!(selector.significance_threshold, 0.01);
        assert_eq!(selector.k, 100);
    }

    #[test]
    fn test_differential_expression_analysis() {
        let expression_data = Array2::from_shape_vec(
            (6, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0,
                11.0, 12.0, 13.0, 14.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let selector = BioinformaticsFeatureSelector::builder()
            .data_type("gene_expression")
            .analysis_method("differential_expression")
            .k(2)
            .build();

        let trained = selector.fit(&expression_data, &labels).unwrap();
        let transformed = trained.transform(&expression_data).unwrap();

        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 6);
    }

    #[test]
    fn test_apply_normalization() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 8.0, 4.0, 16.0]).unwrap();

        let log_normalized = apply_normalization(&data, "log2").unwrap();
        assert_eq!(log_normalized.dim(), (3, 2));

        // Check that log2 normalization was applied
        assert!((log_normalized[[0, 0]] - (2.0_f64).log2()).abs() < 1e-6);
        assert!((log_normalized[[0, 1]] - (5.0_f64).log2()).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_testing_correction() {
        let p_values = Array1::from_vec(vec![0.01, 0.05, 0.1, 0.2, 0.5]);

        let fdr_corrected = apply_multiple_testing_correction(&p_values, "fdr").unwrap();
        assert_eq!(fdr_corrected.len(), 5);

        let bonferroni_corrected =
            apply_multiple_testing_correction(&p_values, "bonferroni").unwrap();
        assert_eq!(bonferroni_corrected.len(), 5);
        // Check that Bonferroni correction makes p-values more conservative
        assert!(bonferroni_corrected[0] >= p_values[0]);
    }

    #[test]
    fn test_minor_allele_frequency() {
        let snp = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]); // 2 AA, 2 AB, 2 BB
        let maf = compute_minor_allele_frequency(&snp.view());

        // With equal genotype frequencies, MAF should be 0.5
        assert!((maf - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_hardy_weinberg_test() {
        let snp = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let hwe_p = compute_hardy_weinberg_p_value(&snp.view());

        assert!(hwe_p >= 0.0 && hwe_p <= 1.0);
    }

    #[test]
    fn test_get_support() {
        let expression_data = Array2::from_shape_vec(
            (4, 6),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let selector = BioinformaticsFeatureSelector::builder().k(3).build();

        let trained = selector.fit(&expression_data, &labels).unwrap();
        let support = trained.get_support().unwrap();

        assert_eq!(support.len(), 6);
        assert_eq!(support.iter().filter(|&&x| x).count(), 3);
    }

    #[test]
    fn test_snp_association_analysis() {
        let snp_data = Array2::from_shape_vec(
            (6, 3),
            vec![
                0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0,
                0.0, 1.0,
            ],
        )
        .unwrap();

        let phenotype = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let selector = BioinformaticsFeatureSelector::builder()
            .data_type("snp")
            .analysis_method("association_test")
            .k(2)
            .build();

        let trained = selector.fit(&snp_data, &phenotype).unwrap();
        let transformed = trained.transform(&snp_data).unwrap();

        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 6);
    }
}
