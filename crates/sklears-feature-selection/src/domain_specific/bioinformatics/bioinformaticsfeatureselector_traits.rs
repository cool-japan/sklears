//! # BioinformaticsFeatureSelector - Trait Implementations
//!
//! This module contains trait implementations for `BioinformaticsFeatureSelector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Estimator`
//! - `Estimator`
//! - `Fit`
//! - `Transform`
//! - `SelectorMixin`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BioinformaticsFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
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
        let normalized_x = if self.normalization_method != "none" {
            apply_normalization(x, &self.normalization_method)?
        } else {
            x.clone()
        };
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
                    )));
                }
            };
        let adjusted_p_values = if let Some(ref p_vals) = p_values {
            Some(apply_multiple_testing_correction(
                p_vals,
                &self.multiple_testing_correction,
            )?)
        } else {
            None
        };
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
        let (_n_samples, n_features) = x.dim();
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
