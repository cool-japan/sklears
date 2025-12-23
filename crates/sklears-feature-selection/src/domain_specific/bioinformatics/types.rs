//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
pub use crate::base::SelectorMixin;
pub use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
pub use sklears_core::error::{Result as SklResult, SklearsError};
pub use sklears_core::traits::{Estimator, Fit, Transform};
pub use std::collections::HashMap;
pub use std::marker::PhantomData;

pub type Result<T> = SklResult<T>;
pub type Float = f64;

/// Complex return type for bioinformatics analysis methods
/// (scores, p_values, fold_changes, metrics, additional_data)
pub type BioinformaticsAnalysisResult = (
    Array1<Float>,
    Option<Array1<Float>>,
    Option<Array1<Float>>,
    Option<HashMap<String, Float>>,
    Option<Array1<Float>>,
);

/// Bioinformatics feature selector for genomic and biological data.
///
/// This selector provides specialized methods for biological data analysis, including
/// differential expression analysis, SNP association testing, pathway enrichment,
/// and protein interaction analysis. It incorporates biological knowledge and
/// statistical methods appropriate for high-dimensional biological datasets.
#[derive(Debug, Clone)]
pub struct BioinformaticsFeatureSelector<State = Untrained> {
    pub(crate) data_type: String,
    pub(crate) analysis_method: String,
    pub(crate) multiple_testing_correction: String,
    pub(crate) significance_threshold: Float,
    pub(crate) fold_change_threshold: Float,
    pub(crate) p_value_threshold: Float,
    pub(crate) maf_threshold: Float,
    pub(crate) hwe_threshold: Float,
    pub(crate) population_structure_correction: bool,
    pub(crate) include_pathway_analysis: bool,
    pub(crate) go_enrichment: bool,
    pub(crate) pathway_analysis: bool,
    pub(crate) pathway_database: String,
    pub(crate) enrichment_method: String,
    pub(crate) min_pathway_size: usize,
    pub(crate) max_pathway_size: usize,
    pub(crate) include_protein_interactions: bool,
    pub(crate) network_centrality_weight: Float,
    pub(crate) functional_domain_weight: Float,
    pub(crate) prior_knowledge_weight: Float,
    pub(crate) batch_effect_correction: bool,
    pub(crate) normalization_method: String,
    pub k: usize,
    pub(crate) max_features: Option<usize>,
    pub(crate) strategy: BioinformaticsStrategy,
    pub(crate) state: PhantomData<State>,
    pub(crate) trained_state: Option<Trained>,
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
impl BioinformaticsFeatureSelector<Untrained> {
    pub(crate) fn analyze_gene_expression(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
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
    pub(crate) fn analyze_snp_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        match self.analysis_method.as_str() {
            "association_test" => self.snp_association_analysis(x, y),
            "linkage_disequilibrium" => self.linkage_disequilibrium_analysis(x, y),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown SNP analysis method: {}",
                self.analysis_method
            ))),
        }
    }
    pub(crate) fn analyze_protein_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        match self.analysis_method.as_str() {
            "network_analysis" => self.protein_network_analysis(x, y),
            "functional_analysis" => self.protein_functional_analysis(x, y),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown protein analysis method: {}",
                self.analysis_method
            ))),
        }
    }
    pub(crate) fn analyze_methylation_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        self.differential_expression_analysis(x, y)
    }
    fn differential_expression_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
        let mut p_values = Array1::zeros(n_features);
        let mut fold_changes = Array1::zeros(n_features);
        for j in 0..n_features {
            let feature = x.column(j);
            let (group0, group1) = separate_groups(&feature, y);
            if group0.is_empty() || group1.is_empty() {
                scores[j] = 0.0;
                p_values[j] = 1.0;
                fold_changes[j] = 1.0;
                continue;
            }
            let mean0 = compute_mean(&group0);
            let mean1 = compute_mean(&group1);
            let std0 = compute_std(&group0, mean0);
            let std1 = compute_std(&group1, mean1);
            let pooled_std = ((std0.powi(2) / group0.len() as Float)
                + (std1.powi(2) / group1.len() as Float))
                .sqrt();
            let t_stat = if pooled_std > 1e-10 {
                (mean1 - mean0) / pooled_std
            } else {
                0.0
            };
            let p_value = 2.0 * (1.0 - (t_stat.abs() / (1.0 + t_stat.abs())));
            let fold_change = if mean0 > 1e-10 && mean1 > 1e-10 {
                (mean1 / mean0).log2()
            } else {
                0.0
            };
            scores[j] = t_stat.abs();
            p_values[j] = p_value;
            fold_changes[j] = fold_change;
        }
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
        _y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
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
    ) -> Result<BioinformaticsAnalysisResult> {
        let (de_scores, p_values, fold_changes, _, _) =
            self.differential_expression_analysis(x, y)?;
        let pathway_scores = compute_pathway_scores(&de_scores, &self.pathway_database)?;
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
    ) -> Result<BioinformaticsAnalysisResult> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
        let mut p_values = Array1::zeros(n_features);
        for j in 0..n_features {
            let snp = x.column(j);
            let maf = compute_minor_allele_frequency(&snp);
            let hwe_p = compute_hardy_weinberg_p_value(&snp);
            if maf < self.maf_threshold || hwe_p < self.hwe_threshold {
                scores[j] = 0.0;
                p_values[j] = 1.0;
                continue;
            }
            let chi2_stat = compute_chi_square_association(&snp, y);
            let p_value = chi_square_to_p_value(chi2_stat, 1);
            scores[j] = chi2_stat;
            p_values[j] = p_value;
        }
        if self.population_structure_correction {
            let corrected_p_values = correct_population_structure(&p_values)?;
            return Ok((scores, Some(corrected_p_values), None, None, None));
        }
        Ok((scores, Some(p_values), None, None, None))
    }
    fn linkage_disequilibrium_analysis(
        &self,
        x: &Array2<Float>,
        _y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
        for j in 0..n_features {
            let snp_j = x.column(j);
            let mut ld_score = 0.0;
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
    ) -> Result<BioinformaticsAnalysisResult> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
        let (de_scores, p_values, fold_changes, _, _) =
            self.differential_expression_analysis(x, y)?;
        if self.include_protein_interactions {
            let network_scores = compute_protein_network_centrality(x)?;
            for j in 0..n_features {
                scores[j] = de_scores[j] * (1.0 - self.network_centrality_weight)
                    + network_scores[j] * self.network_centrality_weight;
            }
        } else {
            scores = de_scores;
        }
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
    ) -> Result<BioinformaticsAnalysisResult> {
        self.protein_network_analysis(x, y)
    }
    /// DESeq2-like differential expression analysis using negative binomial distribution
    ///
    /// This method implements a simplified version of DESeq2's approach for differential
    /// expression analysis, modeling count data with a negative binomial distribution and
    /// estimating dispersion parameters for robust statistical testing.
    fn deseq2_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
        let mut p_values = Array1::zeros(n_features);
        let mut fold_changes = Array1::zeros(n_features);
        let (group0_indices, group1_indices): (Vec<(usize, &Float)>, Vec<(usize, &Float)>) =
            y.iter().enumerate().partition(|(_, &val)| val == 0.0);
        for j in 0..n_features {
            let feature = x.column(j);
            let counts0: Vec<Float> = group0_indices.iter().map(|(i, _)| feature[*i]).collect();
            let counts1: Vec<Float> = group1_indices.iter().map(|(i, _)| feature[*i]).collect();
            if counts0.is_empty() || counts1.is_empty() {
                continue;
            }
            let size_factor0 = compute_size_factor(&counts0);
            let size_factor1 = compute_size_factor(&counts1);
            let norm_counts0: Vec<Float> = counts0.iter().map(|&c| c / size_factor0).collect();
            let norm_counts1: Vec<Float> = counts1.iter().map(|&c| c / size_factor1).collect();
            let mean0 = compute_mean(&norm_counts0);
            let mean1 = compute_mean(&norm_counts1);
            let dispersion = estimate_dispersion(&norm_counts0, &norm_counts1, mean0, mean1);
            let log_fold_change = if mean0 > 1e-10 && mean1 > 1e-10 {
                (mean1 / mean0).log2()
            } else {
                0.0
            };
            let se =
                compute_wald_standard_error(mean0, mean1, dispersion, counts0.len(), counts1.len());
            let wald_stat = if se > 1e-10 {
                log_fold_change / se
            } else {
                0.0
            };
            let p_value = 2.0 * (1.0 - normal_cdf(wald_stat.abs()));
            scores[j] = wald_stat.abs();
            p_values[j] = p_value;
            fold_changes[j] = log_fold_change;
        }
        Ok((scores, Some(p_values), Some(fold_changes), None, None))
    }
    /// EdgeR-like exact test for differential expression
    ///
    /// Implements a simplified version of edgeR's exact test for comparing two groups,
    /// using a negative binomial model with empirical Bayes dispersion estimation.
    fn edger_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let (_, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
        let mut p_values = Array1::zeros(n_features);
        let mut fold_changes = Array1::zeros(n_features);
        let (group0, group1) = separate_groups_indices(y);
        for j in 0..n_features {
            let feature = x.column(j);
            let counts0: Vec<Float> = group0.iter().map(|&i| feature[i]).collect();
            let counts1: Vec<Float> = group1.iter().map(|&i| feature[i]).collect();
            if counts0.is_empty() || counts1.is_empty() {
                continue;
            }
            let _lib_size0: Float = counts0.iter().sum();
            let _lib_size1: Float = counts1.iter().sum();
            let n0 = counts0.len() as Float;
            let n1 = counts1.len() as Float;
            let total_count: Float = counts0.iter().sum::<Float>() + counts1.iter().sum::<Float>();
            let p0 = n0 / (n0 + n1);
            let dispersion = estimate_common_dispersion(&counts0, &counts1);
            let mean0 = counts0.iter().sum::<Float>() / n0;
            let mean1 = counts1.iter().sum::<Float>() / n1;
            let log_fc = if mean0 > 1e-10 && mean1 > 1e-10 {
                (mean1 / mean0).log2()
            } else {
                0.0
            };
            let p_value = compute_exact_test_p_value(
                counts0.iter().sum(),
                counts1.iter().sum(),
                total_count,
                p0,
                dispersion,
            );
            scores[j] = log_fc.abs();
            p_values[j] = p_value;
            fold_changes[j] = log_fc;
        }
        Ok((scores, Some(p_values), Some(fold_changes), None, None))
    }
    /// Limma-like linear models for microarray data
    ///
    /// Implements empirical Bayes moderated t-statistics for differential expression,
    /// borrowing information across genes to improve statistical power.
    fn limma_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let (_n_samples, n_features) = x.dim();
        let mut scores = Array1::zeros(n_features);
        let mut p_values = Array1::zeros(n_features);
        let mut fold_changes = Array1::zeros(n_features);
        let mut gene_variances = Vec::new();
        let mut t_stats = Vec::new();
        for j in 0..n_features {
            let feature = x.column(j);
            let (group0, group1) = separate_groups(&feature, y);
            if group0.is_empty() || group1.is_empty() {
                gene_variances.push(1.0);
                t_stats.push(0.0);
                continue;
            }
            let mean0 = compute_mean(&group0);
            let mean1 = compute_mean(&group1);
            let var0 = compute_variance(&group0, mean0);
            let var1 = compute_variance(&group1, mean1);
            let n0 = group0.len() as Float;
            let n1 = group1.len() as Float;
            let pooled_var = ((n0 - 1.0) * var0 + (n1 - 1.0) * var1) / (n0 + n1 - 2.0);
            gene_variances.push(pooled_var);
            let se = (pooled_var * (1.0 / n0 + 1.0 / n1)).sqrt();
            let t_stat = if se > 1e-10 {
                (mean1 - mean0) / se
            } else {
                0.0
            };
            t_stats.push(t_stat);
        }
        let (s0_squared, d0) = estimate_prior_variance(&gene_variances);
        for j in 0..n_features {
            let feature = x.column(j);
            let (group0, group1) = separate_groups(&feature, y);
            if group0.is_empty() || group1.is_empty() {
                continue;
            }
            let mean0 = compute_mean(&group0);
            let mean1 = compute_mean(&group1);
            let n0 = group0.len() as Float;
            let n1 = group1.len() as Float;
            let gene_var = gene_variances[j];
            let df_gene = n0 + n1 - 2.0;
            let moderated_var = (d0 * s0_squared + df_gene * gene_var) / (d0 + df_gene);
            let moderated_se = (moderated_var * (1.0 / n0 + 1.0 / n1)).sqrt();
            let moderated_t = if moderated_se > 1e-10 {
                (mean1 - mean0) / moderated_se
            } else {
                0.0
            };
            let _moderated_df = d0 + df_gene;
            let p_value = 2.0 * (1.0 - normal_cdf(moderated_t.abs()));
            let log_fc = if mean0 > 1e-10 && mean1 > 1e-10 {
                (mean1 / mean0).log2()
            } else if mean0 <= 1e-10 && mean1 > 1e-10 {
                10.0
            } else if mean1 <= 1e-10 && mean0 > 1e-10 {
                -10.0
            } else {
                0.0
            };
            scores[j] = moderated_t.abs();
            p_values[j] = p_value;
            fold_changes[j] = log_fc;
        }
        Ok((scores, Some(p_values), Some(fold_changes), None, None))
    }
    /// Gene Set Enrichment Analysis (GSEA)
    ///
    /// Determines whether a priori defined sets of genes show statistically significant,
    /// concordant differences between two biological states.
    fn gsea_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let (de_scores, _, fold_changes, _, _) = self.differential_expression_analysis(x, y)?;
        let mut ranked_genes: Vec<(usize, Float)> = de_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        ranked_genes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let gene_sets = define_gene_sets(x.ncols());
        let mut pathway_scores = HashMap::new();
        let mut biological_scores = Array1::zeros(x.ncols());
        for (set_name, gene_set) in gene_sets.iter() {
            let enrichment_score = compute_enrichment_score(&ranked_genes, gene_set);
            pathway_scores.insert(set_name.clone(), enrichment_score);
            for &gene_idx in gene_set {
                if gene_idx < biological_scores.len() {
                    biological_scores[gene_idx] += enrichment_score.abs();
                }
            }
        }
        let max_bio_score = biological_scores.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_bio_score > 1e-10 {
            biological_scores.mapv_inplace(|x| x / max_bio_score);
        }
        Ok((
            de_scores,
            None,
            fold_changes,
            Some(pathway_scores),
            Some(biological_scores),
        ))
    }
    /// Over-Representation Analysis (ORA)
    ///
    /// Tests whether genes in a predefined set are over-represented in a list of
    /// differentially expressed genes using hypergeometric or Fisher's exact test.
    fn ora_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let (de_scores, p_vals_opt, fold_changes, _, _) =
            self.differential_expression_analysis(x, y)?;
        let p_vals = p_vals_opt.unwrap_or_else(|| Array1::ones(de_scores.len()));
        let significant_genes: Vec<usize> = p_vals
            .iter()
            .enumerate()
            .filter(|(_, &p)| p < self.significance_threshold)
            .map(|(i, _)| i)
            .collect();
        let total_genes = x.ncols();
        let gene_sets = define_gene_sets(total_genes);
        let mut pathway_scores = HashMap::new();
        let mut biological_scores = Array1::zeros(total_genes);
        for (set_name, gene_set) in gene_sets.iter() {
            let overlap: usize = significant_genes
                .iter()
                .filter(|&&g| gene_set.contains(&g))
                .count();
            let gene_set_size = gene_set.len();
            let sig_gene_count = significant_genes.len();
            let ora_p_value = compute_hypergeometric_p_value(
                overlap,
                gene_set_size,
                total_genes - gene_set_size,
                sig_gene_count,
            );
            let enrichment = if ora_p_value > 1e-300 {
                -ora_p_value.log10()
            } else {
                300.0
            };
            pathway_scores.insert(set_name.clone(), enrichment);
            for &gene_idx in gene_set {
                if gene_idx < biological_scores.len() {
                    biological_scores[gene_idx] += enrichment;
                }
            }
        }
        let max_bio_score = biological_scores.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_bio_score > 1e-10 {
            biological_scores.mapv_inplace(|x| x / max_bio_score);
        }
        Ok((
            de_scores,
            Some(p_vals),
            fold_changes,
            Some(pathway_scores),
            Some(biological_scores),
        ))
    }
    /// Protein-Protein Interaction (PPI) Network Analysis
    ///
    /// Integrates protein interaction network topology with expression data to identify
    /// functionally important proteins based on network centrality and expression changes.
    fn ppi_network_analysis(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let (de_scores, p_values, fold_changes, _, _) =
            self.differential_expression_analysis(x, y)?;
        let network_matrix = build_ppi_network(x, 0.7);
        let degree_centrality = compute_degree_centrality(&network_matrix);
        let betweenness_centrality = compute_betweenness_centrality(&network_matrix);
        let closeness_centrality = compute_closeness_centrality(&network_matrix);
        let mut integrated_scores = Array1::zeros(x.ncols());
        for i in 0..x.ncols() {
            let network_score = 0.4 * degree_centrality[i]
                + 0.3 * betweenness_centrality[i]
                + 0.3 * closeness_centrality[i];
            integrated_scores[i] = self.network_centrality_weight * network_score
                + (1.0 - self.network_centrality_weight) * de_scores[i];
        }
        let mut network_metrics = HashMap::new();
        network_metrics.insert("avg_degree".to_string(), degree_centrality.mean().unwrap());
        network_metrics.insert(
            "avg_betweenness".to_string(),
            betweenness_centrality.mean().unwrap(),
        );
        network_metrics.insert(
            "avg_closeness".to_string(),
            closeness_centrality.mean().unwrap(),
        );
        Ok((
            integrated_scores,
            p_values,
            fold_changes,
            Some(network_metrics),
            Some(degree_centrality),
        ))
    }
    /// Variant Effect Prediction
    ///
    /// Predicts the functional impact of genetic variants based on conservation,
    /// protein structure, and functional domain information.
    fn variant_effect_prediction(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let n_features = x.ncols();
        let mut variant_scores = Array1::zeros(n_features);
        let mut p_values = Array1::zeros(n_features);
        let mut effect_predictions = Array1::zeros(n_features);
        for j in 0..n_features {
            let variant = x.column(j);
            let _maf = compute_minor_allele_frequency(&variant);
            let conservation_score = 1.0 - (variant.var(0.0) / 2.0).min(1.0);
            let protein_impact = compute_protein_impact_score(&variant, j);
            let domain_score = compute_variant_domain_score(j, n_features);
            let association = compute_chi_square_association(&variant, y);
            variant_scores[j] = 0.3 * conservation_score
                + 0.3 * protein_impact
                + 0.2 * domain_score
                + 0.2 * association;
            p_values[j] = chi_square_to_p_value(association, 1);
            effect_predictions[j] = if variant_scores[j] > 0.8 {
                2.0
            } else if variant_scores[j] > 0.5 {
                1.0
            } else {
                0.0
            };
        }
        let mut metrics = HashMap::new();
        metrics.insert(
            "high_impact_variants".to_string(),
            effect_predictions.iter().filter(|&&x| x >= 2.0).count() as Float,
        );
        metrics.insert(
            "moderate_impact_variants".to_string(),
            effect_predictions.iter().filter(|&&x| x == 1.0).count() as Float,
        );
        Ok((
            variant_scores,
            Some(p_values),
            Some(effect_predictions),
            Some(metrics),
            None,
        ))
    }
    /// Multi-Omics Integration
    ///
    /// Integrates multiple omics data types (transcriptomics, proteomics, metabolomics)
    /// to identify features with concordant signals across different molecular layers.
    fn multi_omics_integration(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BioinformaticsAnalysisResult> {
        let n_features = x.ncols();
        let n_per_omics = n_features / 3;
        let transcriptomics_data = x.slice(s![.., 0..n_per_omics]).to_owned();
        let proteomics_data = x.slice(s![.., n_per_omics..2 * n_per_omics]).to_owned();
        let metabolomics_data = x.slice(s![.., 2 * n_per_omics..]).to_owned();
        let (transcriptomics_scores, _, _, _, _) =
            self.differential_expression_analysis(&transcriptomics_data, y)?;
        let (proteomics_scores, _, _, _, _) =
            self.differential_expression_analysis(&proteomics_data, y)?;
        let (metabolomics_scores, _, _, _, _) =
            self.differential_expression_analysis(&metabolomics_data, y)?;
        let mut integrated_scores = Array1::zeros(n_features);
        for i in 0..n_per_omics.min(transcriptomics_scores.len()) {
            integrated_scores[i] = transcriptomics_scores[i];
        }
        for i in 0..n_per_omics.min(proteomics_scores.len()) {
            integrated_scores[n_per_omics + i] = proteomics_scores[i];
        }
        for i in 0..(n_features - 2 * n_per_omics).min(metabolomics_scores.len()) {
            integrated_scores[2 * n_per_omics + i] = metabolomics_scores[i];
        }
        let trans_prot_cor = compute_layer_correlation(&transcriptomics_data, &proteomics_data);
        let trans_metab_cor = compute_layer_correlation(&transcriptomics_data, &metabolomics_data);
        let prot_metab_cor = compute_layer_correlation(&proteomics_data, &metabolomics_data);
        let mut biological_scores = Array1::zeros(n_features);
        for i in 0..n_per_omics.min(transcriptomics_scores.len()) {
            let prot_idx = i.min(proteomics_scores.len() - 1);
            let metab_idx = i.min(metabolomics_scores.len() - 1);
            let concordance =
                (trans_prot_cor[[i, prot_idx]].abs() + trans_metab_cor[[i, metab_idx]].abs()) / 2.0;
            biological_scores[i] = concordance;
            if n_per_omics + i < biological_scores.len() {
                biological_scores[n_per_omics + i] = concordance;
            }
            if 2 * n_per_omics + i < biological_scores.len() {
                biological_scores[2 * n_per_omics + i] = concordance;
            }
        }
        let mut omics_metrics = HashMap::new();
        omics_metrics.insert(
            "transcriptomics_mean_score".to_string(),
            transcriptomics_scores.mean().unwrap_or(0.0),
        );
        omics_metrics.insert(
            "proteomics_mean_score".to_string(),
            proteomics_scores.mean().unwrap_or(0.0),
        );
        omics_metrics.insert(
            "metabolomics_mean_score".to_string(),
            metabolomics_scores.mean().unwrap_or(0.0),
        );
        omics_metrics.insert(
            "trans_prot_correlation".to_string(),
            trans_prot_cor.mean().unwrap_or(0.0),
        );
        omics_metrics.insert(
            "trans_metab_correlation".to_string(),
            trans_metab_cor.mean().unwrap_or(0.0),
        );
        omics_metrics.insert(
            "prot_metab_correlation".to_string(),
            prot_metab_cor.mean().unwrap_or(0.0),
        );
        Ok((
            integrated_scores,
            None,
            None,
            Some(omics_metrics),
            Some(biological_scores),
        ))
    }
    pub(crate) fn select_features(
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
            if let Some(p_vals) = p_values {
                if p_vals[i] > self.significance_threshold {
                    is_significant = false;
                }
            }
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
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.k.min(candidates.len()).min(1000);
        let max_features = self.max_features.unwrap_or(n_features);
        if k > 0 {
            return Ok(candidates
                .into_iter()
                .take(k.min(max_features))
                .map(|(i, _)| i)
                .collect());
        }
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
#[derive(Debug, Clone)]
pub struct Untrained;
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
    /// DESeq2-like differential expression with negative binomial
    DESeq2Like,
    /// EdgeR-like exact test for differential expression
    EdgeRLike,
    /// Limma-like linear models for microarray data
    LimmaLike,
    /// GSEA pathway enrichment analysis
    GSEA,
    /// Over-representation analysis (ORA)
    ORA,
    /// Protein-protein interaction network analysis
    PPINetwork,
    /// Variant effect prediction
    VariantEffectPrediction,
    /// Multi-omics integration
    MultiOmics,
}
#[derive(Debug, Clone)]
pub struct Trained {
    pub(crate) selected_features: Vec<usize>,
    pub(crate) feature_scores: Array1<Float>,
    pub(crate) adjusted_p_values: Option<Array1<Float>>,
    pub(crate) fold_changes: Option<Array1<Float>>,
    pub(crate) pathway_scores: Option<HashMap<String, Float>>,
    pub(crate) biological_relevance_scores: Option<Array1<Float>>,
    pub(crate) n_features: usize,
    pub(crate) data_type: String,
    pub(crate) analysis_method: String,
}
/// Builder for BioinformaticsFeatureSelector configuration.
#[derive(Debug)]
pub struct BioinformaticsFeatureSelectorBuilder {
    pub(crate) data_type: String,
    pub(crate) analysis_method: String,
    pub(crate) multiple_testing_correction: String,
    pub(crate) significance_threshold: Float,
    pub(crate) fold_change_threshold: Float,
    pub(crate) maf_threshold: Float,
    pub(crate) hwe_threshold: Float,
    pub(crate) population_structure_correction: bool,
    pub(crate) include_pathway_analysis: bool,
    pub(crate) pathway_database: String,
    pub(crate) enrichment_method: String,
    pub(crate) min_pathway_size: usize,
    pub(crate) max_pathway_size: usize,
    pub(crate) include_protein_interactions: bool,
    pub(crate) network_centrality_weight: Float,
    pub(crate) functional_domain_weight: Float,
    pub(crate) prior_knowledge_weight: Float,
    pub(crate) batch_effect_correction: bool,
    pub(crate) normalization_method: String,
    pub(crate) k: Option<usize>,
    pub(crate) max_features: Option<usize>,
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
            p_value_threshold: self.significance_threshold,
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
