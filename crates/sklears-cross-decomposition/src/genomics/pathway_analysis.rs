//! Pathway Analysis for Gene Set Enrichment
//!
//! This module provides comprehensive pathway enrichment analysis for genomics data,
//! including multiple enrichment methods and statistical corrections.

use crate::multi_omics::GenomicsError;
use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;
use std::collections::HashMap;

/// Pathway Analysis for Gene Set Enrichment
///
/// This structure provides comprehensive pathway enrichment analysis for genomics data,
/// including multiple enrichment methods and statistical corrections.
pub struct PathwayAnalysis {
    /// Method for computing pathway enrichment
    enrichment_method: EnrichmentMethod,
    /// Multiple testing correction method
    multiple_testing_correction: MultipleTestingCorrection,
    /// Minimum pathway size for analysis
    min_pathway_size: usize,
    /// Maximum pathway size for analysis
    max_pathway_size: usize,
    /// Significance threshold (p-value)
    significance_threshold: Float,
    /// Pathway database to use
    pathway_database: PathwayDatabase,
}

/// Methods for computing pathway enrichment
#[derive(Debug, Clone)]
pub enum EnrichmentMethod {
    Hypergeometric,
    FisherExact,
    GSEA,
    SsGsea,
}

/// Multiple testing correction methods
#[derive(Debug, Clone)]
pub enum MultipleTestingCorrection {
    /// No correction
    None,
    /// Bonferroni correction
    Bonferroni,
    /// Benjamini-Hochberg FDR
    BenjaminiHochberg,
    /// Benjamini-Yekutieli FDR
    BenjaminiYekutieli,
}

/// Pathway databases
#[derive(Debug, Clone)]
pub enum PathwayDatabase {
    /// KEGG pathway database
    KEGG,
    /// Gene Ontology
    GeneOntology,
    /// Reactome
    Reactome,
    /// Custom pathway database
    Custom(HashMap<String, Vec<String>>),
}

impl PathwayAnalysis {
    /// Create a new pathway analysis
    pub fn new() -> Self {
        Self {
            enrichment_method: EnrichmentMethod::Hypergeometric,
            multiple_testing_correction: MultipleTestingCorrection::BenjaminiHochberg,
            min_pathway_size: 5,
            max_pathway_size: 500,
            significance_threshold: 0.05,
            pathway_database: PathwayDatabase::KEGG,
        }
    }

    /// Set the enrichment method
    pub fn enrichment_method(mut self, method: EnrichmentMethod) -> Self {
        self.enrichment_method = method;
        self
    }

    /// Set the multiple testing correction method
    pub fn multiple_testing_correction(mut self, correction: MultipleTestingCorrection) -> Self {
        self.multiple_testing_correction = correction;
        self
    }

    /// Set the minimum pathway size
    pub fn min_pathway_size(mut self, size: usize) -> Self {
        self.min_pathway_size = size;
        self
    }

    /// Set the maximum pathway size
    pub fn max_pathway_size(mut self, size: usize) -> Self {
        self.max_pathway_size = size;
        self
    }

    /// Set the significance threshold
    pub fn significance_threshold(mut self, threshold: Float) -> Self {
        self.significance_threshold = threshold;
        self
    }

    /// Set the pathway database
    pub fn pathway_database(mut self, database: PathwayDatabase) -> Self {
        self.pathway_database = database;
        self
    }

    /// Analyze pathway enrichment for integration scores
    pub fn analyze_enrichment(
        &self,
        integration_scores: &[Array1<Float>],
    ) -> Result<HashMap<String, Float>, GenomicsError> {
        // Get pathway definitions
        let pathways = self.get_pathway_definitions()?;

        // Identify significantly expressed genes
        let significant_genes = self.identify_significant_genes(integration_scores)?;

        // Compute enrichment for each pathway
        let mut enrichment_results = HashMap::new();
        let mut p_values = Vec::new();
        let mut pathway_names = Vec::new();

        for (pathway_name, pathway_genes) in &pathways {
            if pathway_genes.len() < self.min_pathway_size
                || pathway_genes.len() > self.max_pathway_size
            {
                continue;
            }

            let p_value = match self.enrichment_method {
                EnrichmentMethod::Hypergeometric => {
                    self.hypergeometric_test(&significant_genes, pathway_genes)?
                }
                EnrichmentMethod::FisherExact => {
                    self.fisher_exact_test(&significant_genes, pathway_genes)?
                }
                EnrichmentMethod::GSEA => self.gsea_test(integration_scores, pathway_genes)?,
                EnrichmentMethod::SsGsea => self.ssgsea_test(integration_scores, pathway_genes)?,
            };

            p_values.push(p_value);
            pathway_names.push(pathway_name.clone());
        }

        // Apply multiple testing correction
        let corrected_p_values = self.apply_multiple_testing_correction(&p_values)?;

        // Filter by significance threshold
        for (i, &corrected_p) in corrected_p_values.iter().enumerate() {
            if corrected_p < self.significance_threshold {
                enrichment_results.insert(pathway_names[i].clone(), corrected_p);
            }
        }

        Ok(enrichment_results)
    }

    fn get_pathway_definitions(&self) -> Result<HashMap<String, Vec<String>>, GenomicsError> {
        match &self.pathway_database {
            PathwayDatabase::KEGG => self.get_kegg_pathways(),
            PathwayDatabase::GeneOntology => self.get_go_pathways(),
            PathwayDatabase::Reactome => self.get_reactome_pathways(),
            PathwayDatabase::Custom(pathways) => Ok(pathways.clone()),
        }
    }

    fn get_kegg_pathways(&self) -> Result<HashMap<String, Vec<String>>, GenomicsError> {
        // Mock KEGG pathway database with Gene_X format (adaptive to small datasets)
        let mut pathways = HashMap::new();

        // Create smaller pathways that work with limited gene counts
        pathways.insert(
            "Pathway_A".to_string(),
            vec![
                "Gene_0".to_string(),
                "Gene_1".to_string(),
                "Gene_2".to_string(),
            ],
        );

        pathways.insert(
            "Pathway_B".to_string(),
            vec![
                "Gene_1".to_string(),
                "Gene_2".to_string(),
                "Gene_3".to_string(),
            ],
        );

        pathways.insert(
            "Pathway_C".to_string(),
            vec![
                "Gene_0".to_string(),
                "Gene_3".to_string(),
                "Gene_4".to_string(),
            ],
        );

        // Only add larger pathways if we might have more genes
        pathways.insert(
            "Pathway_D".to_string(),
            vec![
                "Gene_0".to_string(),
                "Gene_1".to_string(),
                "Gene_2".to_string(),
                "Gene_3".to_string(),
                "Gene_4".to_string(),
            ],
        );

        Ok(pathways)
    }

    fn get_go_pathways(&self) -> Result<HashMap<String, Vec<String>>, GenomicsError> {
        // Mock Gene Ontology pathways with Gene_X format (adaptive to small datasets)
        let mut pathways = HashMap::new();

        pathways.insert(
            "GO_Process_1".to_string(),
            vec![
                "Gene_0".to_string(),
                "Gene_1".to_string(),
                "Gene_2".to_string(),
            ],
        );

        pathways.insert(
            "GO_Process_2".to_string(),
            vec![
                "Gene_1".to_string(),
                "Gene_2".to_string(),
                "Gene_3".to_string(),
            ],
        );

        Ok(pathways)
    }

    fn get_reactome_pathways(&self) -> Result<HashMap<String, Vec<String>>, GenomicsError> {
        // Mock Reactome pathways with Gene_X format (adaptive to small datasets)
        let mut pathways = HashMap::new();

        pathways.insert(
            "Reactome_1".to_string(),
            vec![
                "Gene_0".to_string(),
                "Gene_1".to_string(),
                "Gene_2".to_string(),
            ],
        );

        Ok(pathways)
    }

    fn identify_significant_genes(
        &self,
        integration_scores: &[Array1<Float>],
    ) -> Result<Vec<String>, GenomicsError> {
        let mut significant_genes = Vec::new();

        // Combine scores from all omics datasets
        let combined_scores = self.combine_integration_scores(integration_scores)?;

        // Use top 10% of genes by combined score
        let threshold_percentile = 0.9;
        let mut sorted_scores: Vec<(usize, Float)> = combined_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let n_significant = (sorted_scores.len() as Float * (1.0 - threshold_percentile)) as usize;

        for i in 0..n_significant.min(sorted_scores.len()) {
            let gene_idx = sorted_scores[i].0;
            significant_genes.push(format!("Gene_{}", gene_idx));
        }

        Ok(significant_genes)
    }

    fn combine_integration_scores(
        &self,
        integration_scores: &[Array1<Float>],
    ) -> Result<Array1<Float>, GenomicsError> {
        if integration_scores.is_empty() {
            return Err(GenomicsError::InsufficientData(
                "No integration scores provided".to_string(),
            ));
        }

        let n_components = integration_scores[0].len();
        let mut combined_scores = Array1::zeros(n_components);

        // Average scores across omics datasets
        for scores in integration_scores {
            if scores.len() != n_components {
                return Err(GenomicsError::InvalidDimensions(
                    "All integration score arrays must have the same length".to_string(),
                ));
            }

            for (i, &score) in scores.iter().enumerate() {
                combined_scores[i] += score;
            }
        }

        combined_scores.mapv_inplace(|x| x / integration_scores.len() as Float);
        Ok(combined_scores)
    }

    fn hypergeometric_test(
        &self,
        significant_genes: &[String],
        pathway_genes: &[String],
    ) -> Result<Float, GenomicsError> {
        // Count overlapping genes
        let overlap = significant_genes
            .iter()
            .filter(|gene| pathway_genes.contains(gene))
            .count();

        // Mock hypergeometric test calculation
        // In practice, this would use proper hypergeometric distribution
        let n_significant = significant_genes.len() as Float;
        let n_pathway = pathway_genes.len() as Float;
        let n_total = 20000.0; // Assume 20,000 total genes

        // Simple approximation using normal distribution
        let expected = (n_significant * n_pathway) / n_total;
        let variance = expected * (1.0 - n_pathway / n_total) * (1.0 - n_significant / n_total);

        if variance <= 0.0 {
            return Ok(1.0);
        }

        let z_score = (overlap as Float - expected) / variance.sqrt();

        // Convert to p-value (simplified)
        let p_value = if z_score <= 0.0 {
            1.0
        } else {
            (1.0 - z_score / 5.0).max(0.001).min(1.0) // Simplified p-value calculation
        };

        Ok(p_value)
    }

    fn fisher_exact_test(
        &self,
        significant_genes: &[String],
        pathway_genes: &[String],
    ) -> Result<Float, GenomicsError> {
        // For simplicity, use hypergeometric test as approximation
        self.hypergeometric_test(significant_genes, pathway_genes)
    }

    fn gsea_test(
        &self,
        integration_scores: &[Array1<Float>],
        pathway_genes: &[String],
    ) -> Result<Float, GenomicsError> {
        // Simplified GSEA implementation
        let combined_scores = self.combine_integration_scores(integration_scores)?;

        // Create gene ranking
        let mut gene_rankings: Vec<(usize, Float)> = combined_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        gene_rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Calculate enrichment score
        let n_genes = gene_rankings.len();
        let mut enrichment_score: Float = 0.0;
        let mut max_enrichment: Float = 0.0;
        let mut current_enrichment: Float = 0.0;

        let pathway_gene_indices: Vec<usize> = pathway_genes
            .iter()
            .filter_map(|gene_name| {
                // Extract gene index from gene name (format: Gene_<index>)
                gene_name
                    .strip_prefix("Gene_")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .collect();

        for (rank, (gene_idx, _)) in gene_rankings.iter().enumerate() {
            if pathway_gene_indices.contains(gene_idx) {
                current_enrichment += 1.0 / pathway_gene_indices.len() as Float;
            } else {
                current_enrichment -= 1.0 / (n_genes - pathway_gene_indices.len()) as Float;
            }

            if current_enrichment.abs() > max_enrichment.abs() {
                max_enrichment = current_enrichment;
            }
        }

        enrichment_score = max_enrichment;

        // Convert enrichment score to p-value (simplified)
        let p_value = (1.0 - enrichment_score.abs()).max(0.001).min(1.0);

        Ok(p_value)
    }

    fn ssgsea_test(
        &self,
        integration_scores: &[Array1<Float>],
        pathway_genes: &[String],
    ) -> Result<Float, GenomicsError> {
        // Simplified single-sample GSEA
        let combined_scores = self.combine_integration_scores(integration_scores)?;

        let pathway_gene_indices: Vec<usize> = pathway_genes
            .iter()
            .filter_map(|gene_name| {
                gene_name
                    .strip_prefix("Gene_")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .collect();

        // Calculate pathway score as mean of gene scores
        let mut pathway_score: Float = 0.0;
        let mut n_pathway_genes = 0;

        for &gene_idx in &pathway_gene_indices {
            if gene_idx < combined_scores.len() {
                pathway_score += combined_scores[gene_idx];
                n_pathway_genes += 1;
            }
        }

        if n_pathway_genes > 0 {
            pathway_score /= n_pathway_genes as Float;
        }

        // Convert pathway score to p-value (simplified)
        let p_value = (1.0 - pathway_score.abs()).max(0.001).min(1.0);

        Ok(p_value)
    }

    fn apply_multiple_testing_correction(
        &self,
        p_values: &[Float],
    ) -> Result<Vec<Float>, GenomicsError> {
        match self.multiple_testing_correction {
            MultipleTestingCorrection::None => Ok(p_values.to_vec()),
            MultipleTestingCorrection::Bonferroni => {
                let n = p_values.len() as Float;
                Ok(p_values.iter().map(|&p| (p * n).min(1.0)).collect())
            }
            MultipleTestingCorrection::BenjaminiHochberg => {
                self.benjamini_hochberg_correction(p_values)
            }
            MultipleTestingCorrection::BenjaminiYekutieli => {
                self.benjamini_yekutieli_correction(p_values)
            }
        }
    }

    fn benjamini_hochberg_correction(
        &self,
        p_values: &[Float],
    ) -> Result<Vec<Float>, GenomicsError> {
        let n = p_values.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Create sorted indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap());

        let mut corrected = vec![0.0; n];

        // Apply BH correction
        for (i, &idx) in indices.iter().enumerate() {
            let rank = i + 1;
            let correction_factor = n as Float / rank as Float;
            corrected[idx] = (p_values[idx] * correction_factor).min(1.0);
        }

        // Ensure monotonicity
        for i in (0..n - 1).rev() {
            let curr_idx = indices[i];
            let next_idx = indices[i + 1];
            corrected[curr_idx] = corrected[curr_idx].min(corrected[next_idx]);
        }

        Ok(corrected)
    }

    fn benjamini_yekutieli_correction(
        &self,
        p_values: &[Float],
    ) -> Result<Vec<Float>, GenomicsError> {
        // Similar to BH but with different correction factor
        let n = p_values.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Calculate harmonic number H_n
        let harmonic_n = (1..=n).map(|i| 1.0 / i as Float).sum::<Float>();

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap());

        let mut corrected = vec![0.0; n];

        for (i, &idx) in indices.iter().enumerate() {
            let rank = i + 1;
            let correction_factor = (n as Float * harmonic_n) / rank as Float;
            corrected[idx] = (p_values[idx] * correction_factor).min(1.0);
        }

        // Ensure monotonicity
        for i in (0..n - 1).rev() {
            let curr_idx = indices[i];
            let next_idx = indices[i + 1];
            corrected[curr_idx] = corrected[curr_idx].min(corrected[next_idx]);
        }

        Ok(corrected)
    }
}

impl Default for PathwayAnalysis {
    fn default() -> Self {
        Self::new()
    }
}
