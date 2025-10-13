//! Method Selection Module for AutoML Feature Selection
//!
//! Intelligently selects appropriate feature selection methods based on data characteristics.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

use super::automl_core::{AutoMLMethod, DataCharacteristics, TargetType};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;

/// Method selector for choosing appropriate feature selection methods
#[derive(Debug, Clone)]
pub struct MethodSelector;

impl MethodSelector {
    pub fn new() -> Self {
        Self
    }

    pub fn select_methods(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Result<Vec<AutoMLMethod>> {
        let mut selected_methods = Vec::new();

        // Rule-based method selection
        match characteristics.target_type {
            TargetType::BinaryClassification | TargetType::MultiClassification => {
                // Always include univariate filtering for classification
                selected_methods.push(AutoMLMethod::UnivariateFiltering);

                if characteristics.n_features > 100 {
                    selected_methods.push(AutoMLMethod::LassoBased);
                }

                if characteristics.computational_budget.allow_complex_methods {
                    selected_methods.push(AutoMLMethod::TreeBased);
                }
            }
            TargetType::Regression => {
                selected_methods.push(AutoMLMethod::CorrelationBased);

                if characteristics.n_features > 50 {
                    selected_methods.push(AutoMLMethod::LassoBased);
                }

                if characteristics.computational_budget.allow_complex_methods {
                    selected_methods.push(AutoMLMethod::WrapperBased);
                }
            }
            TargetType::MultiLabel => {
                selected_methods.push(AutoMLMethod::UnivariateFiltering);
                selected_methods.push(AutoMLMethod::EnsembleBased);
            }
            TargetType::Survival => {
                selected_methods.push(AutoMLMethod::UnivariateFiltering);
                selected_methods.push(AutoMLMethod::CorrelationBased);
            }
        }

        // Add correlation-based if high correlation detected
        if characteristics.correlation_structure.high_correlation_pairs
            > characteristics.n_features / 4
            && !selected_methods.contains(&AutoMLMethod::CorrelationBased)
        {
            selected_methods.push(AutoMLMethod::CorrelationBased);
        }

        // Add ensemble method for complex scenarios
        if characteristics.feature_to_sample_ratio > 0.5
            && characteristics.computational_budget.allow_complex_methods
        {
            selected_methods.push(AutoMLMethod::EnsembleBased);
        }

        // Hybrid approach for very complex data
        if characteristics.n_features > 1000 && characteristics.n_samples > 1000 {
            selected_methods.push(AutoMLMethod::Hybrid);
        }

        // Neural Architecture Search for very high-dimensional data
        if characteristics.feature_to_sample_ratio > 2.0
            && characteristics.computational_budget.allow_complex_methods
            && !characteristics.computational_budget.prefer_speed
        {
            selected_methods.push(AutoMLMethod::NeuralArchitectureSearch);
        }

        // Transfer Learning when we have moderate complexity but good compute budget
        if characteristics.n_features > 100
            && characteristics.n_samples >= 500
            && characteristics.computational_budget.allow_complex_methods
        {
            selected_methods.push(AutoMLMethod::TransferLearning);
        }

        // Meta-Learning Ensemble for complex scenarios with sufficient data
        if characteristics.n_features > 50
            && characteristics.n_samples > 200
            && characteristics.computational_budget.allow_complex_methods
            && selected_methods.len() >= 2
        {
            selected_methods.push(AutoMLMethod::MetaLearningEnsemble);
        }

        if selected_methods.is_empty() {
            // Fallback to univariate filtering
            selected_methods.push(AutoMLMethod::UnivariateFiltering);
        }

        Ok(selected_methods)
    }
}

impl Default for MethodSelector {
    fn default() -> Self {
        Self::new()
    }
}
