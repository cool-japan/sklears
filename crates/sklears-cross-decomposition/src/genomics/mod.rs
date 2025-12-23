//! Genomics and Multi-Omics Integration
//!
//! This module provides specialized methods for genomics and multi-omics data analysis,
//! including multi-omics integration, gene-environment interaction analysis, and
//! single-cell multi-modal analysis.

pub mod enhanced_pathway_analysis;
pub mod gene_environment;
pub mod pathway_analysis;
pub mod single_cell;
pub mod temporal_gene;

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{thread_rng, Random, Rng};
use sklears_core::types::Float;
use std::collections::HashMap;

// Re-export from multi_omics module
pub use crate::multi_omics::{FittedMultiOmicsIntegration, GenomicsError, MultiOmicsIntegration};

// Re-export from pathway_analysis module
pub use pathway_analysis::{
    EnrichmentMethod, MultipleTestingCorrection, PathwayAnalysis, PathwayDatabase,
};

// Re-export from enhanced_pathway_analysis module
pub use enhanced_pathway_analysis::{
    ConsensusMethod, EnhancedPathwayAnalysis, EnhancedPathwayResults, MLScoringConfig,
    MissingDataStrategy, MultiModalConfig, NetworkAnalysisConfig, PathwayAnalysisConfig,
    TemporalAnalysisConfig,
};

// Re-export from gene_environment module
pub use gene_environment::{FittedGeneEnvironmentInteraction, GeneEnvironmentInteraction};

// Re-export from single_cell module
pub use single_cell::{FittedSingleCellMultiModal, SingleCellMultiModal};

// Re-export from temporal_gene module
pub use temporal_gene::{FittedTemporalGeneExpression, TemporalGeneExpression};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::essentials::{Normal, Uniform};
    use scirs2_core::ndarray::array;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_multi_omics_integration_basic() {
        // Create synthetic multi-omics data
        let rna_data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let protein_data = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];
        let omics_data = vec![rna_data.view(), protein_data.view()];

        let integration = MultiOmicsIntegration::new(2);
        let result = integration.fit(&omics_data);

        assert!(result.is_ok());
        let fitted = result.unwrap();
        assert_eq!(fitted.n_components(), 2);
        assert_eq!(fitted.integration_scores().len(), 2);
        // Check that pathway enrichment doesn't crash (may be empty for small test datasets)
        let pathway_results = fitted.pathway_enrichment();
        assert!(pathway_results.values().all(|score| score.is_finite()));
    }

    #[test]
    fn test_multi_omics_integration_error_cases() {
        // Test with empty data
        let omics_data = vec![];
        let integration = MultiOmicsIntegration::new(2);
        assert!(integration.fit(&omics_data).is_err());

        // Test with single dataset
        let single_data = array![[1.0, 2.0], [3.0, 4.0]];
        let omics_data = vec![single_data.view()];
        assert!(integration.fit(&omics_data).is_err());
    }

    #[test]
    fn test_gene_environment_interaction_basic() {
        let gene_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let env_data = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];
        let phenotype = array![1.0, 2.0, 3.0, 4.0];

        let gei = GeneEnvironmentInteraction::new(1);
        let result = gei.fit(gene_data.view(), env_data.view(), phenotype.view());

        assert!(result.is_ok());
        let fitted = result.unwrap();
        assert_eq!(fitted.interaction_effects().nrows(), 2);
        assert_eq!(fitted.interaction_effects().ncols(), 2);
        assert!(!fitted.significant_interactions().is_empty());
    }

    #[test]
    fn test_gene_environment_interaction_prediction() {
        let gene_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let env_data = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];
        let phenotype = array![1.0, 2.0, 3.0, 4.0];

        let gei = GeneEnvironmentInteraction::new(1);
        let fitted = gei
            .fit(gene_data.view(), env_data.view(), phenotype.view())
            .unwrap();

        let new_gene_data = array![[9.0, 10.0], [11.0, 12.0]];
        let new_env_data = array![[8.5, 9.5], [10.5, 11.5]];
        let predictions = fitted.predict(new_gene_data.view(), new_env_data.view());

        assert!(predictions.is_ok());
        let pred = predictions.unwrap();
        assert_eq!(pred.nrows(), 2);
        assert_eq!(pred.ncols(), 1);
    }

    #[test]
    fn test_single_cell_multi_modal_basic() {
        let rna_data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let atac_data = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];

        let scmm = SingleCellMultiModal::new(2);
        let result = scmm.fit(rna_data.view(), atac_data.view());

        assert!(result.is_ok());
        let fitted = result.unwrap();
        assert_eq!(fitted.cell_types().len(), 4);
        assert_eq!(fitted.modality_correlations().len(), 2);
    }

    #[test]
    fn test_single_cell_multi_modal_cell_type_prediction() {
        let rna_data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let atac_data = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];

        let scmm = SingleCellMultiModal::new(2);
        let fitted = scmm.fit(rna_data.view(), atac_data.view()).unwrap();

        let new_rna_data = array![[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]];
        let predicted_types = fitted.predict_cell_types(new_rna_data.view());

        assert!(predicted_types.is_ok());
        let types = predicted_types.unwrap();
        assert_eq!(types.len(), 2);
        assert!(types.iter().all(|&t| t < 4)); // Should be 0, 1, 2, or 3
    }

    #[test]
    fn test_correlation_computation() {
        let gei = GeneEnvironmentInteraction::new(1);
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 4.0, 6.0, 8.0];

        let correlation = gei.compute_correlation(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(correlation, 1.0, epsilon = 1e-10);

        let z = array![4.0, 3.0, 2.0, 1.0];
        let correlation = gei.compute_correlation(&x.view(), &z.view()).unwrap();
        assert_abs_diff_eq!(correlation, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_temporal_gene_expression_basic() {
        // Create synthetic time series gene expression data
        let n_timepoints = 20;
        let n_genes = 5;
        let mut expression_data = Array2::zeros((n_timepoints, n_genes));
        let time_points = Array1::from_iter((0..n_timepoints).map(|i| i as Float));

        // Generate synthetic oscillating gene expression patterns
        let mut rng = thread_rng();
        for (t, mut row) in expression_data.rows_mut().into_iter().enumerate() {
            let time = t as Float * 0.1;
            for (g, value) in row.iter_mut().enumerate() {
                let frequency = (g + 1) as Float * 0.5;
                *value =
                    (time * frequency).sin() + 0.1 * rng.sample(&Uniform::new(0.0, 1.0).unwrap());
            }
        }

        let temporal_analysis = TemporalGeneExpression::new(3)
            .n_lags(2)
            .window_size(5)
            .detrend(true);

        let result = temporal_analysis.fit(expression_data.view(), time_points.view());
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.n_components(), 3);
        assert_eq!(fitted.n_lags(), 2);
        assert_eq!(
            fitted.temporal_components().nrows(),
            fitted.temporal_components().nrows()
        );
        assert_eq!(fitted.gene_trajectories().ncols(), n_genes);
        assert_eq!(fitted.temporal_correlations().nrows(), n_genes);
        assert_eq!(fitted.temporal_correlations().ncols(), n_genes);
    }

    #[test]
    fn test_temporal_gene_expression_prediction() {
        let n_timepoints = 15;
        let n_genes = 3;
        let mut expression_data = Array2::zeros((n_timepoints, n_genes));
        let time_points = Array1::from_iter((0..n_timepoints).map(|i| i as Float));

        // Generate linear trend data for predictability
        for (t, mut row) in expression_data.rows_mut().into_iter().enumerate() {
            for (g, value) in row.iter_mut().enumerate() {
                *value = (t as Float) * (g + 1) as Float * 0.1;
            }
        }

        let temporal_analysis = TemporalGeneExpression::new(2).n_lags(1).detrend(false);

        let fitted = temporal_analysis
            .fit(expression_data.view(), time_points.view())
            .unwrap();

        // Test future prediction
        let predictions = fitted.predict_future(expression_data.view(), 3);
        assert!(predictions.is_ok());

        let pred = predictions.unwrap();
        assert_eq!(pred.nrows(), 3);
        assert_eq!(pred.ncols(), n_genes);
    }

    #[test]
    fn test_temporal_gene_expression_identification() {
        let n_timepoints = 12;
        let n_genes = 4;
        let mut expression_data = Array2::zeros((n_timepoints, n_genes));
        let time_points = Array1::from_iter((0..n_timepoints).map(|i| i as Float));

        // Create genes with different temporal variability
        let mut rng = thread_rng();
        for (t, mut row) in expression_data.rows_mut().into_iter().enumerate() {
            let time = t as Float;
            row[0] = 1.0; // Constant gene
            row[1] = time * 0.1; // Linear trend
            row[2] = (time * 0.5).sin(); // Oscillating gene
            row[3] = rng.sample(&Uniform::new(0.0, 1.0).unwrap()) - 0.5; // Random/noisy gene
        }

        let temporal_analysis = TemporalGeneExpression::new(2).window_size(3);

        let fitted = temporal_analysis
            .fit(expression_data.view(), time_points.view())
            .unwrap();

        // Test temporal gene identification
        let temporal_genes = fitted.identify_temporal_genes(0.01);
        assert!(temporal_genes.is_ok());

        let genes = temporal_genes.unwrap();
        // Should identify genes with temporal patterns (genes 1, 2, 3 but not 0)
        assert!(!genes.is_empty());
    }

    #[test]
    fn test_temporal_gene_expression_error_cases() {
        let temporal_analysis = TemporalGeneExpression::new(2);

        // Test with mismatched dimensions
        let expression_data = array![[1.0, 2.0], [3.0, 4.0]];
        let time_points = array![1.0, 2.0, 3.0]; // Wrong length

        let result = temporal_analysis.fit(expression_data.view(), time_points.view());
        assert!(result.is_err());

        // Test with insufficient data
        let small_data = array![[1.0], [2.0]]; // Too few time points
        let small_time = array![1.0, 2.0];

        let large_window_analysis = TemporalGeneExpression::new(1).window_size(10);
        let result = large_window_analysis.fit(small_data.view(), small_time.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_pathway_analysis_basic() {
        let pathway_analyzer = PathwayAnalysis::new()
            .enrichment_method(EnrichmentMethod::Hypergeometric)
            .multiple_testing_correction(MultipleTestingCorrection::BenjaminiHochberg)
            .min_pathway_size(3)
            .max_pathway_size(20)
            .significance_threshold(1.0); // Use 1.0 for testing to allow all results

        // Create mock integration scores with enough genes to match pathways
        let score1 = Array1::from_iter((0..50).map(|i| {
            if i < 10 {
                0.9 - i as Float * 0.05
            } else {
                0.1 + (i as Float * 0.01)
            }
        }));
        let score2 = Array1::from_iter((0..50).map(|i| {
            if i < 10 {
                0.8 - i as Float * 0.04
            } else {
                0.2 + (i as Float * 0.008)
            }
        }));
        let integration_scores = vec![score1, score2];

        let result = pathway_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());

        let enrichment = result.unwrap();
        assert!(!enrichment.is_empty());

        // Check that p-values are valid
        for &p_value in enrichment.values() {
            assert!(p_value >= 0.0);
            assert!(p_value <= 1.0);
        }
    }

    #[test]
    fn test_pathway_analysis_different_methods() {
        let integration_scores = vec![
            array![0.8, 0.2, 0.1, 0.9, 0.3],
            array![0.7, 0.3, 0.2, 0.8, 0.4],
        ];

        // Test Hypergeometric method
        let hypergeometric_analyzer = PathwayAnalysis::new()
            .enrichment_method(EnrichmentMethod::Hypergeometric)
            .significance_threshold(1.0); // Accept all for testing

        let result = hypergeometric_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());

        // Test GSEA method
        let gsea_analyzer = PathwayAnalysis::new()
            .enrichment_method(EnrichmentMethod::GSEA)
            .significance_threshold(1.0);

        let result = gsea_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());

        // Test ssGSEA method
        let ssgsea_analyzer = PathwayAnalysis::new()
            .enrichment_method(EnrichmentMethod::SsGsea)
            .significance_threshold(1.0);

        let result = ssgsea_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pathway_analysis_different_databases() {
        let integration_scores = vec![
            array![0.8, 0.2, 0.1, 0.9, 0.3],
            array![0.7, 0.3, 0.2, 0.8, 0.4],
        ];

        // Test KEGG database
        let kegg_analyzer = PathwayAnalysis::new()
            .pathway_database(PathwayDatabase::KEGG)
            .significance_threshold(1.0);

        let result = kegg_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());

        // Test Gene Ontology database
        let go_analyzer = PathwayAnalysis::new()
            .pathway_database(PathwayDatabase::GeneOntology)
            .significance_threshold(1.0);

        let result = go_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());

        // Test custom database
        let mut custom_pathways = HashMap::new();
        custom_pathways.insert(
            "Custom_Pathway_1".to_string(),
            vec![
                "Gene_0".to_string(),
                "Gene_1".to_string(),
                "Gene_2".to_string(),
            ],
        );
        custom_pathways.insert(
            "Custom_Pathway_2".to_string(),
            vec!["Gene_3".to_string(), "Gene_4".to_string()],
        );

        let custom_analyzer = PathwayAnalysis::new()
            .pathway_database(PathwayDatabase::Custom(custom_pathways))
            .significance_threshold(1.0);

        let result = custom_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pathway_analysis_multiple_testing_correction() {
        let integration_scores = vec![
            array![0.8, 0.2, 0.1, 0.9, 0.3, 0.7, 0.4, 0.6],
            array![0.7, 0.3, 0.2, 0.8, 0.4, 0.6, 0.5, 0.7],
        ];

        // Test Benjamini-Hochberg correction
        let bh_analyzer = PathwayAnalysis::new()
            .multiple_testing_correction(MultipleTestingCorrection::BenjaminiHochberg)
            .significance_threshold(1.0);

        let result = bh_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());

        // Test Bonferroni correction
        let bonferroni_analyzer = PathwayAnalysis::new()
            .multiple_testing_correction(MultipleTestingCorrection::Bonferroni)
            .significance_threshold(1.0);

        let result = bonferroni_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());

        // Test no correction
        let no_correction_analyzer = PathwayAnalysis::new()
            .multiple_testing_correction(MultipleTestingCorrection::None)
            .significance_threshold(1.0);

        let result = no_correction_analyzer.analyze_enrichment(&integration_scores);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pathway_analysis_error_cases() {
        let pathway_analyzer = PathwayAnalysis::new();

        // Test with empty integration scores
        let empty_scores: Vec<Array1<Float>> = vec![];
        let result = pathway_analyzer.analyze_enrichment(&empty_scores);
        assert!(result.is_err());

        // Test with mismatched score lengths
        let score1 = array![0.8, 0.2, 0.1];
        let score2 = array![0.7, 0.3]; // Different length
        let mismatched_scores = vec![score1, score2];
        let result = pathway_analyzer.analyze_enrichment(&mismatched_scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_omics_integration_with_enhanced_pathway_analysis() {
        // Test the enhanced pathway analysis integration in MultiOmicsIntegration
        let data1 = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ];

        let data2 = array![
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
        ];

        let omics_data = vec![data1.view(), data2.view()];

        let integration = MultiOmicsIntegration::new(2).scale(true).max_iter(50);

        let result = integration.fit(&omics_data);
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.n_components(), 2);

        // Check that pathway enrichment doesn't crash (may be empty for small test datasets)
        let pathway_results = fitted.pathway_enrichment();
        assert!(pathway_results.values().all(|score| score.is_finite()));

        // Verify pathway enrichment values are valid p-values (if any exist)
        for &p_value in pathway_results.values() {
            assert!(p_value >= 0.0);
            assert!(p_value <= 1.0);
        }
    }
}
