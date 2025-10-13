//! Scoring utilities for model evaluation
//!
//! This module provides utilities for creating custom scoring functions
//! and accessing built-in scorers for use in model selection and evaluation.

use crate::regression::{
    explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,
};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;

/// Scoring metric configuration
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Whether higher scores are better
    pub greater_is_better: bool,
    /// Whether the scorer requires probability predictions
    pub needs_proba: bool,
    /// Whether the scorer requires positive class probabilities only
    pub needs_threshold: bool,
}

/// Available scoring metrics
#[derive(Debug, Clone)]
pub enum ScoringMetric {
    // Classification metrics
    /// Accuracy
    Accuracy,
    /// Precision
    Precision,
    /// Recall
    Recall,
    /// F1
    F1,
    // Regression metrics
    /// R2
    R2,
    /// NegMeanSquaredError
    NegMeanSquaredError,
    /// NegMeanAbsoluteError
    NegMeanAbsoluteError,
    /// NegRootMeanSquaredError
    NegRootMeanSquaredError,
    ExplainedVariance,
    // Custom metric
    Custom(String),
}

/// A scorer configuration
#[derive(Debug, Clone)]
pub struct Scorer {
    /// The name of the scorer
    pub name: String,
    /// The scoring metric
    pub metric: ScoringMetric,
    /// Configuration for the scorer
    pub config: ScorerConfig,
}

impl Scorer {
    /// Score predictions using this scorer
    pub fn score<T>(&self, y_true: &[T], y_pred: &[T]) -> Result<f64>
    where
        T: PartialEq + Copy + Ord + Into<f64>,
    {
        use crate::classification::{accuracy_score, f1_score, precision_score, recall_score};
        use scirs2_core::ndarray::Array1;

        match &self.metric {
            ScoringMetric::Accuracy => {
                let y_true_arr = Array1::from_vec(y_true.to_vec());
                let y_pred_arr = Array1::from_vec(y_pred.to_vec());
                accuracy_score(&y_true_arr, &y_pred_arr)
                    .map_err(|e| SklearsError::InvalidInput(e.to_string()))
            }
            ScoringMetric::Precision => {
                let y_true_arr = Array1::from_vec(y_true.to_vec());
                let y_pred_arr = Array1::from_vec(y_pred.to_vec());
                precision_score(&y_true_arr, &y_pred_arr, None)
                    .map_err(|e| SklearsError::InvalidInput(e.to_string()))
            }
            ScoringMetric::Recall => {
                let y_true_arr = Array1::from_vec(y_true.to_vec());
                let y_pred_arr = Array1::from_vec(y_pred.to_vec());
                recall_score(&y_true_arr, &y_pred_arr, None)
                    .map_err(|e| SklearsError::InvalidInput(e.to_string()))
            }
            ScoringMetric::F1 => {
                let y_true_arr = Array1::from_vec(y_true.to_vec());
                let y_pred_arr = Array1::from_vec(y_pred.to_vec());
                f1_score(&y_true_arr, &y_pred_arr, None)
                    .map_err(|e| SklearsError::InvalidInput(e.to_string()))
            }
            // For regression metrics that don't need Ord, defer to score_float
            _ => self.score_float(
                &y_true.iter().map(|&x| x.into()).collect::<Vec<f64>>(),
                &y_pred.iter().map(|&x| x.into()).collect::<Vec<f64>>(),
            ),
        }
    }

    /// Score predictions using this scorer for float values (for regression)
    pub fn score_float(&self, y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
        use scirs2_core::ndarray::Array1;

        match &self.metric {
            ScoringMetric::R2 => {
                let y_true_arr = Array1::from_vec(y_true.to_vec());
                let y_pred_arr = Array1::from_vec(y_pred.to_vec());
                r2_score(&y_true_arr, &y_pred_arr)
                    .map_err(|e| SklearsError::InvalidInput(e.to_string()))
            }
            ScoringMetric::NegMeanSquaredError => {
                let y_true_arr = Array1::from_vec(y_true.to_vec());
                let y_pred_arr = Array1::from_vec(y_pred.to_vec());
                let mse = mean_squared_error(&y_true_arr, &y_pred_arr)
                    .map_err(|e| SklearsError::InvalidInput(e.to_string()))?;
                Ok(-mse) // Negative because we want higher scores to be better
            }
            ScoringMetric::NegMeanAbsoluteError => {
                let y_true_arr = Array1::from_vec(y_true.to_vec());
                let y_pred_arr = Array1::from_vec(y_pred.to_vec());
                let mae = mean_absolute_error(&y_true_arr, &y_pred_arr)
                    .map_err(|e| SklearsError::InvalidInput(e.to_string()))?;
                Ok(-mae) // Negative because we want higher scores to be better
            }
            ScoringMetric::NegRootMeanSquaredError => {
                let y_true_arr = Array1::from_vec(y_true.to_vec());
                let y_pred_arr = Array1::from_vec(y_pred.to_vec());
                let mse = mean_squared_error(&y_true_arr, &y_pred_arr)
                    .map_err(|e| SklearsError::InvalidInput(e.to_string()))?;
                Ok(-mse.sqrt()) // Negative because we want higher scores to be better
            }
            ScoringMetric::ExplainedVariance => {
                let y_true_arr = Array1::from_vec(y_true.to_vec());
                let y_pred_arr = Array1::from_vec(y_pred.to_vec());
                explained_variance_score(&y_true_arr, &y_pred_arr)
                    .map_err(|e| SklearsError::InvalidInput(e.to_string()))
            }
            // Classification metrics should not be called with float method
            _ => Err(SklearsError::InvalidInput(
                "Classification metrics require discrete labels, not float values".to_string(),
            )),
        }
    }
}

/// Make a scorer from a performance metric
///
/// # Arguments
/// * `name` - Name for the scorer
/// * `metric` - The scoring metric to use
/// * `greater_is_better` - Whether higher scores are better
/// * `needs_proba` - Whether the scorer requires probability predictions
/// * `needs_threshold` - Whether the scorer requires positive class probabilities
///
/// # Returns
/// A scorer object that can be used in cross-validation
pub fn make_scorer(
    name: String,
    metric: ScoringMetric,
    greater_is_better: bool,
    needs_proba: bool,
    needs_threshold: bool,
) -> Scorer {
    Scorer {
        name,
        metric,
        config: ScorerConfig {
            greater_is_better,
            needs_proba,
            needs_threshold,
        },
    }
}

/// Get a scorer by name
///
/// # Arguments
/// * `scoring` - The name of the scorer
///
/// # Returns
/// The requested scorer
///
/// # Available scorers
/// ## Classification
/// - "accuracy": Classification accuracy
/// - "precision": Precision score
/// - "recall": Recall score
/// - "f1": F1 score
///
/// ## Regression
/// - "r2": R² score
/// - "neg_mean_squared_error": Negative mean squared error
/// - "neg_mean_absolute_error": Negative mean absolute error
/// - "neg_root_mean_squared_error": Negative root mean squared error
/// - "explained_variance": Explained variance score
pub fn get_scorer(scoring: &str) -> Result<Scorer> {
    let scorer = match scoring {
        // Classification scorers
        "accuracy" => Scorer {
            name: "accuracy".to_string(),
            metric: ScoringMetric::Accuracy,
            config: ScorerConfig {
                greater_is_better: true,
                needs_proba: false,
                needs_threshold: false,
            },
        },
        "precision" => Scorer {
            name: "precision".to_string(),
            metric: ScoringMetric::Precision,
            config: ScorerConfig {
                greater_is_better: true,
                needs_proba: false,
                needs_threshold: false,
            },
        },
        "recall" => Scorer {
            name: "recall".to_string(),
            metric: ScoringMetric::Recall,
            config: ScorerConfig {
                greater_is_better: true,
                needs_proba: false,
                needs_threshold: false,
            },
        },
        "f1" => Scorer {
            name: "f1".to_string(),
            metric: ScoringMetric::F1,
            config: ScorerConfig {
                greater_is_better: true,
                needs_proba: false,
                needs_threshold: false,
            },
        },

        // Regression scorers
        "r2" => Scorer {
            name: "r2".to_string(),
            metric: ScoringMetric::R2,
            config: ScorerConfig {
                greater_is_better: true,
                needs_proba: false,
                needs_threshold: false,
            },
        },
        "neg_mean_squared_error" => Scorer {
            name: "neg_mean_squared_error".to_string(),
            metric: ScoringMetric::NegMeanSquaredError,
            config: ScorerConfig {
                greater_is_better: true,
                needs_proba: false,
                needs_threshold: false,
            },
        },
        "neg_mean_absolute_error" => Scorer {
            name: "neg_mean_absolute_error".to_string(),
            metric: ScoringMetric::NegMeanAbsoluteError,
            config: ScorerConfig {
                greater_is_better: true,
                needs_proba: false,
                needs_threshold: false,
            },
        },
        "neg_root_mean_squared_error" => Scorer {
            name: "neg_root_mean_squared_error".to_string(),
            metric: ScoringMetric::NegRootMeanSquaredError,
            config: ScorerConfig {
                greater_is_better: true,
                needs_proba: false,
                needs_threshold: false,
            },
        },
        "explained_variance" => Scorer {
            name: "explained_variance".to_string(),
            metric: ScoringMetric::ExplainedVariance,
            config: ScorerConfig {
                greater_is_better: true,
                needs_proba: false,
                needs_threshold: false,
            },
        },
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown scorer: {scoring}. Use get_scorer_names() to see available scorers."
            )));
        }
    };

    Ok(scorer)
}

/// Get the names of all available scorers
pub fn get_scorer_names() -> Vec<&'static str> {
    vec![
        // Classification
        "accuracy",
        "precision",
        "recall",
        "f1",
        // Regression
        "r2",
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
        "explained_variance",
    ]
}

/// Check that the scoring parameter is valid
///
/// # Arguments
/// * `scoring` - The scoring parameter to check
/// * `is_classifier` - Whether the estimator is a classifier
///
/// # Returns
/// The validated scorer
pub fn check_scoring(scoring: Option<&str>, is_classifier: bool) -> Result<Scorer> {
    match scoring {
        Some(name) => get_scorer(name),
        None => {
            // Default scorer based on estimator type
            if is_classifier {
                get_scorer("accuracy")
            } else {
                get_scorer("r2")
            }
        }
    }
}

/// Create scorers for multiple metrics
///
/// # Arguments
/// * `scoring` - A list of metric names
///
/// # Returns
/// A dictionary mapping scorer names to scorer objects
pub fn check_multimetric_scoring(scoring: &[&str]) -> Result<HashMap<String, Scorer>> {
    let mut scorers = HashMap::new();

    for &name in scoring {
        let scorer = get_scorer(name)?;
        scorers.insert(name.to_string(), scorer);
    }

    Ok(scorers)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_scorer() {
        let scorer = make_scorer(
            "custom".to_string(),
            ScoringMetric::Custom("my_metric".to_string()),
            true,
            false,
            false,
        );

        assert_eq!(scorer.name, "custom");
        assert!(scorer.config.greater_is_better);
        assert!(!scorer.config.needs_proba);
    }

    #[test]
    fn test_get_scorer() {
        // Test classification scorers
        let accuracy_scorer = get_scorer("accuracy").unwrap();
        assert_eq!(accuracy_scorer.name, "accuracy");
        assert!(accuracy_scorer.config.greater_is_better);
        assert!(!accuracy_scorer.config.needs_proba);

        // Test regression scorers
        let r2_scorer = get_scorer("r2").unwrap();
        assert_eq!(r2_scorer.name, "r2");
        assert!(r2_scorer.config.greater_is_better);

        let mse_scorer = get_scorer("neg_mean_squared_error").unwrap();
        assert_eq!(mse_scorer.name, "neg_mean_squared_error");
        assert!(mse_scorer.config.greater_is_better);

        // Test unknown scorer
        let result = get_scorer("unknown_scorer");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_scorer_names() {
        let names = get_scorer_names();
        assert!(names.contains(&"accuracy"));
        assert!(names.contains(&"r2"));
        assert!(names.contains(&"neg_mean_squared_error"));
        assert_eq!(names.len(), 9);
    }

    #[test]
    fn test_check_scoring() {
        // Test with explicit scorer
        let scorer = check_scoring(Some("f1"), true).unwrap();
        assert_eq!(scorer.name, "f1");

        // Test default for classifier
        let scorer = check_scoring(None, true).unwrap();
        assert_eq!(scorer.name, "accuracy");

        // Test default for regressor
        let scorer = check_scoring(None, false).unwrap();
        assert_eq!(scorer.name, "r2");
    }

    #[test]
    fn test_check_multimetric_scoring() {
        let metrics = vec!["accuracy", "precision", "recall"];
        let scorers = check_multimetric_scoring(&metrics).unwrap();

        assert_eq!(scorers.len(), 3);
        assert!(scorers.contains_key("accuracy"));
        assert!(scorers.contains_key("precision"));
        assert!(scorers.contains_key("recall"));
    }
}
