//! Hyperparameter Optimization Module for AutoML Feature Selection
//!
//! Optimizes hyperparameters for different feature selection methods based on data characteristics.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};

use super::automl_core::{AutoMLMethod, DataCharacteristics, TargetType};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;

#[derive(Debug, Clone)]
struct DatasetMetrics {
    avg_feature_magnitude: f64,
    target_variance: f64,
    class_balance: f64,
    sample_count: usize,
    feature_count: usize,
}

impl DatasetMetrics {
    fn from_data(X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Self {
        let (sample_count, feature_count) = X.dim();
        let total_entries = sample_count * feature_count;
        let avg_feature_magnitude = if total_entries > 0 {
            X.iter().map(|value| value.abs()).sum::<f64>() / total_entries as f64
        } else {
            0.0
        };

        let target_len = y.len();
        let (target_variance, class_balance) = if target_len > 0 {
            let target_mean = y.iter().copied().sum::<f64>() / target_len as f64;
            let variance = if target_len > 1 {
                y.iter()
                    .map(|value| (value - target_mean).powi(2))
                    .sum::<f64>()
                    / (target_len - 1) as f64
            } else {
                0.0
            };

            let positives = y.iter().filter(|value| **value >= target_mean).count();
            let balance = (positives as f64 / target_len as f64).clamp(0.0, 1.0);
            (variance, balance)
        } else {
            (0.0, 0.5)
        };

        Self {
            avg_feature_magnitude,
            target_variance,
            class_balance,
            sample_count,
            feature_count,
        }
    }
}

/// Hyperparameter optimizer for feature selection methods
#[derive(Debug, Clone)]
pub struct HyperparameterOptimizer {
    pub max_iterations: usize,
}

impl HyperparameterOptimizer {
    pub fn new() -> Self {
        Self { max_iterations: 20 }
    }

    pub fn optimize_method(
        &self,
        method: &AutoMLMethod,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        characteristics: &DataCharacteristics,
    ) -> Result<OptimizedMethod> {
        let metrics = DatasetMetrics::from_data(&X, &y);

        let mut config = match method {
            AutoMLMethod::UnivariateFiltering => self.optimize_univariate(characteristics)?,
            AutoMLMethod::CorrelationBased => self.optimize_correlation(characteristics)?,
            AutoMLMethod::TreeBased => self.optimize_tree(characteristics)?,
            AutoMLMethod::LassoBased => self.optimize_lasso(characteristics)?,
            AutoMLMethod::WrapperBased => self.optimize_wrapper(characteristics)?,
            AutoMLMethod::EnsembleBased => self.optimize_ensemble(characteristics)?,
            AutoMLMethod::Hybrid => self.optimize_hybrid(characteristics)?,
            AutoMLMethod::NeuralArchitectureSearch => self.optimize_nas(characteristics)?,
            AutoMLMethod::TransferLearning => self.optimize_transfer_learning(characteristics)?,
            AutoMLMethod::MetaLearningEnsemble => self.optimize_meta_learning(characteristics)?,
        };

        self.adjust_config_for_data(&mut config, characteristics, &metrics);

        let estimated_cost = self.estimate_computational_cost(method, characteristics, &metrics);

        Ok(OptimizedMethod {
            method_type: method.clone(),
            config,
            estimated_cost,
        })
    }

    fn optimize_univariate(&self, characteristics: &DataCharacteristics) -> Result<MethodConfig> {
        let k = if characteristics.n_features > 1000 {
            (characteristics.n_features / 10).min(100)
        } else {
            (characteristics.n_features / 2).min(50)
        };

        Ok(MethodConfig::Univariate { k })
    }

    fn optimize_correlation(&self, characteristics: &DataCharacteristics) -> Result<MethodConfig> {
        let threshold = if characteristics.correlation_structure.average_correlation > 0.5 {
            0.8
        } else {
            0.7
        };

        Ok(MethodConfig::Correlation { threshold })
    }

    fn optimize_tree(&self, characteristics: &DataCharacteristics) -> Result<MethodConfig> {
        let n_estimators = if characteristics.n_samples > 10000 {
            100
        } else {
            50
        };
        let max_depth = if characteristics.n_features > 100 {
            10
        } else {
            6
        };

        Ok(MethodConfig::Tree {
            n_estimators,
            max_depth,
        })
    }

    fn optimize_lasso(&self, characteristics: &DataCharacteristics) -> Result<MethodConfig> {
        let alpha = if characteristics.feature_to_sample_ratio > 1.0 {
            0.1
        } else {
            0.01
        };

        Ok(MethodConfig::Lasso { alpha })
    }

    fn optimize_wrapper(&self, _characteristics: &DataCharacteristics) -> Result<MethodConfig> {
        Ok(MethodConfig::Wrapper {
            cv_folds: 5,
            scoring: "accuracy".to_string(),
        })
    }

    fn optimize_ensemble(&self, _characteristics: &DataCharacteristics) -> Result<MethodConfig> {
        Ok(MethodConfig::Ensemble {
            n_methods: 3,
            aggregation: "voting".to_string(),
        })
    }

    fn optimize_hybrid(&self, characteristics: &DataCharacteristics) -> Result<MethodConfig> {
        let stage1_method = if characteristics.n_features > 1000 {
            "univariate"
        } else {
            "correlation"
        };

        Ok(MethodConfig::Hybrid {
            stage1_method: stage1_method.to_string(),
            stage2_method: "lasso".to_string(),
            stage1_features: characteristics.n_features / 3,
        })
    }

    fn optimize_nas(&self, characteristics: &DataCharacteristics) -> Result<MethodConfig> {
        let max_epochs = if characteristics.n_features > 1000 {
            100
        } else {
            50
        };

        let population_size = if characteristics.computational_budget.allow_complex_methods {
            20
        } else {
            10
        };

        Ok(MethodConfig::NeuralArchitectureSearch {
            max_epochs,
            population_size,
            mutation_rate: 0.1,
            early_stopping_patience: 10,
        })
    }

    fn optimize_transfer_learning(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Result<MethodConfig> {
        let source_domain = match characteristics.target_type {
            TargetType::BinaryClassification => "binary_classification",
            TargetType::MultiClassification => "multi_classification",
            TargetType::Regression => "regression",
            _ => "general",
        }
        .to_string();

        let fine_tuning_epochs = if characteristics.n_samples > 1000 {
            30
        } else {
            10
        };

        Ok(MethodConfig::TransferLearning {
            source_domain,
            adaptation_method: "fine_tuning".to_string(),
            fine_tuning_epochs,
            transfer_ratio: 0.7,
        })
    }

    fn optimize_meta_learning(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Result<MethodConfig> {
        let base_methods = vec![
            "univariate".to_string(),
            "correlation".to_string(),
            "lasso".to_string(),
        ];

        let ensemble_size = if characteristics.computational_budget.allow_complex_methods {
            5
        } else {
            3
        };

        Ok(MethodConfig::MetaLearningEnsemble {
            base_methods,
            meta_learner: "gradient_boosting".to_string(),
            adaptation_strategy: "online_learning".to_string(),
            ensemble_size,
        })
    }

    fn adjust_config_for_data(
        &self,
        config: &mut MethodConfig,
        characteristics: &DataCharacteristics,
        metrics: &DatasetMetrics,
    ) {
        match config {
            MethodConfig::Univariate { k } => {
                let feature_cap = std::cmp::max(metrics.feature_count, 1);
                if metrics.target_variance < 1e-3 {
                    let conservative_cap = std::cmp::max(feature_cap / 5, 1);
                    *k = (*k).min(conservative_cap);
                } else if metrics.target_variance > 1.0 {
                    let bonus =
                        ((metrics.target_variance.min(4.0)) * feature_cap as f64 * 0.05) as usize;
                    *k = (*k + bonus).min(feature_cap);
                } else {
                    *k = (*k).min(feature_cap);
                }
                *k = (*k).max(1);
            }
            MethodConfig::Correlation { threshold } => {
                let fluctuation = (metrics.avg_feature_magnitude - 1.0).abs().min(0.15);
                if metrics.target_variance < 0.3 {
                    *threshold = (*threshold - fluctuation).clamp(0.3, 0.95);
                } else {
                    *threshold = (*threshold + fluctuation).clamp(0.3, 0.95);
                }
            }
            MethodConfig::Tree {
                n_estimators,
                max_depth,
            } => {
                if metrics.sample_count > 5_000 {
                    *n_estimators = (*n_estimators).max(100);
                }
                if metrics.target_variance > 1.2 {
                    *max_depth = (*max_depth + 2).min(20);
                } else if metrics.target_variance < 0.2 {
                    *max_depth = (*max_depth).max(4);
                }
            }
            MethodConfig::Lasso { alpha } => {
                let scale_adjustment = metrics.avg_feature_magnitude.clamp(0.5, 2.0);
                *alpha = (*alpha * scale_adjustment).max(1e-4);
                if matches!(characteristics.target_type, TargetType::Regression)
                    && metrics.target_variance > 2.0
                {
                    *alpha *= 0.9;
                }
            }
            MethodConfig::Wrapper { scoring, cv_folds } => {
                let imbalance = (metrics.class_balance - 0.5).abs();
                if matches!(characteristics.target_type, TargetType::Regression) {
                    *scoring = "r2".to_string();
                } else if imbalance > 0.2 {
                    *scoring = "roc_auc".to_string();
                } else {
                    *scoring = "accuracy".to_string();
                }

                *cv_folds = if metrics.sample_count < 200 {
                    3
                } else if metrics.sample_count > 5_000 {
                    7
                } else {
                    5
                };
            }
            MethodConfig::Ensemble { n_methods, .. } => {
                if metrics.feature_count > 500 {
                    *n_methods = (*n_methods).max(4);
                }
                if metrics.class_balance < 0.35 || metrics.class_balance > 0.65 {
                    *n_methods = (*n_methods).max(5);
                }
            }
            MethodConfig::Hybrid {
                stage1_features, ..
            } => {
                let feature_cap = std::cmp::max(metrics.feature_count, 1);
                let mut desired = feature_cap / 3;
                if metrics.target_variance < 0.2 {
                    desired = std::cmp::max(feature_cap / 5, 1);
                } else if metrics.target_variance > 1.0 {
                    desired = std::cmp::max(feature_cap / 2, 1);
                }
                *stage1_features = desired.min(feature_cap);
            }
            MethodConfig::NeuralArchitectureSearch {
                max_epochs,
                population_size,
                early_stopping_patience,
                ..
            } => {
                if metrics.sample_count > 2_000 {
                    *population_size = (*population_size).max(25);
                }
                if metrics.target_variance < 0.4 {
                    *max_epochs = (*max_epochs).max(80);
                    *early_stopping_patience = (*early_stopping_patience).max(15);
                } else {
                    *max_epochs = (*max_epochs).min(150);
                }
            }
            MethodConfig::TransferLearning {
                transfer_ratio,
                fine_tuning_epochs,
                ..
            } => {
                if matches!(characteristics.target_type, TargetType::Regression) {
                    *transfer_ratio = 0.6;
                } else if metrics.target_variance > 1.0 {
                    *transfer_ratio = 0.8;
                } else {
                    *transfer_ratio = 0.7;
                }

                if metrics.sample_count > 2_500 {
                    *fine_tuning_epochs = (*fine_tuning_epochs).max(25);
                }
            }
            MethodConfig::MetaLearningEnsemble { ensemble_size, .. } => {
                if metrics.feature_count > 1_000 {
                    *ensemble_size = (*ensemble_size).max(6);
                }
                if metrics.sample_count < 500 {
                    *ensemble_size = (*ensemble_size).min(4);
                }
            }
        }
    }

    fn estimate_computational_cost(
        &self,
        method: &AutoMLMethod,
        characteristics: &DataCharacteristics,
        metrics: &DatasetMetrics,
    ) -> f64 {
        let base_cost =
            characteristics.n_samples as f64 * characteristics.n_features as f64 / 1_000_000.0;

        let scale_penalty = 1.0 + (metrics.avg_feature_magnitude - 1.0).abs().min(3.0) * 0.05;
        let variance_discount = if metrics.target_variance < 1e-6 {
            0.85
        } else {
            1.0
        };
        let imbalance_penalty = 1.0 + (metrics.class_balance - 0.5).abs() * 0.5;

        let method_multiplier = match method {
            AutoMLMethod::UnivariateFiltering => 0.1,
            AutoMLMethod::CorrelationBased => 0.5,
            AutoMLMethod::TreeBased => 2.0,
            AutoMLMethod::LassoBased => 1.5,
            AutoMLMethod::WrapperBased => 10.0,
            AutoMLMethod::EnsembleBased => 5.0,
            AutoMLMethod::Hybrid => 3.0,
            AutoMLMethod::NeuralArchitectureSearch => 15.0,
            AutoMLMethod::TransferLearning => 8.0,
            AutoMLMethod::MetaLearningEnsemble => 12.0,
        };

        base_cost * method_multiplier * scale_penalty * variance_discount * imbalance_penalty
    }
}

impl Default for HyperparameterOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimized method with hyperparameters
#[derive(Debug, Clone)]
pub struct OptimizedMethod {
    pub method_type: AutoMLMethod,
    pub config: MethodConfig,
    pub estimated_cost: f64,
}

/// Method configuration with optimized hyperparameters
#[derive(Debug, Clone)]
pub enum MethodConfig {
    /// Univariate
    Univariate {
        k: usize,
    },
    /// Correlation
    Correlation {
        threshold: f64,
    },
    /// Tree
    Tree {
        n_estimators: usize,

        max_depth: usize,
    },
    Lasso {
        alpha: f64,
    },
    Wrapper {
        cv_folds: usize,
        scoring: String,
    },
    Ensemble {
        n_methods: usize,
        aggregation: String,
    },
    Hybrid {
        stage1_method: String,
        stage2_method: String,
        stage1_features: usize,
    },
    NeuralArchitectureSearch {
        max_epochs: usize,
        population_size: usize,
        mutation_rate: f64,
        early_stopping_patience: usize,
    },
    TransferLearning {
        source_domain: String,
        adaptation_method: String,
        fine_tuning_epochs: usize,
        transfer_ratio: f64,
    },
    MetaLearningEnsemble {
        base_methods: Vec<String>,
        meta_learner: String,
        adaptation_strategy: String,
        ensemble_size: usize,
    },
}

impl OptimizedMethod {
    /// Fit the method to training data (stub implementation)
    pub fn fit(self, X: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<TrainedMethod> {
        // Simplified feature selection based on method type
        let mut selected_features: Vec<usize> = match &self.method_type {
            AutoMLMethod::UnivariateFiltering => {
                if let MethodConfig::Univariate { k } = &self.config {
                    (0..*k.min(&X.ncols())).collect()
                } else {
                    (0..X.ncols().min(10)).collect()
                }
            }
            AutoMLMethod::CorrelationBased => {
                // Select features with correlation above threshold
                (0..X.ncols().min(20)).collect()
            }
            AutoMLMethod::TreeBased => {
                // Select features based on tree importance (simplified)
                (0..X.ncols().min(30)).collect()
            }
            AutoMLMethod::LassoBased => {
                // Select features with non-zero Lasso coefficients (simplified)
                (0..X.ncols().min(15)).collect()
            }
            AutoMLMethod::WrapperBased => {
                // Select features using wrapper method (simplified)
                (0..X.ncols().min(25)).collect()
            }
            AutoMLMethod::EnsembleBased => {
                // Select features from ensemble (simplified)
                (0..X.ncols().min(35)).collect()
            }
            AutoMLMethod::Hybrid => {
                // Multi-stage feature selection (simplified)
                (0..X.ncols().min(20)).collect()
            }
            AutoMLMethod::NeuralArchitectureSearch => {
                // Features selected by NAS (simplified)
                (0..X.ncols().min(40)).collect()
            }
            AutoMLMethod::TransferLearning => {
                // Features from transfer learning (simplified)
                (0..X.ncols().min(30)).collect()
            }
            AutoMLMethod::MetaLearningEnsemble => {
                // Features from meta-learning ensemble (simplified)
                (0..X.ncols().min(50)).collect()
            }
        };

        let metrics = DatasetMetrics::from_data(&X, &y);

        if metrics.target_variance < 1e-6 && selected_features.len() > 10 {
            selected_features.truncate(10);
        }

        let denom = std::cmp::max(selected_features.len(), 1) as f64;
        let importance_scale = 1.0 + metrics.target_variance.sqrt().min(2.0);
        let balance_adjustment = 1.0 + (0.5 - metrics.class_balance).abs() * 0.5;
        let magnitude_adjustment = metrics.avg_feature_magnitude.max(0.1);

        let feature_importances: Vec<f64> = selected_features
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let rank = (denom - index as f64) / denom;
                (rank * importance_scale * balance_adjustment * magnitude_adjustment).max(0.05)
            })
            .collect();

        Ok(TrainedMethod {
            method_type: self.method_type,
            config: self.config,
            selected_features: selected_features.clone(),
            feature_importances,
        })
    }
}

/// Trained method with selected features
#[derive(Debug, Clone)]
pub struct TrainedMethod {
    pub method_type: AutoMLMethod,
    pub config: MethodConfig,
    pub selected_features: Vec<usize>,
    pub feature_importances: Vec<f64>,
}

impl TrainedMethod {
    pub fn transform_indices(&self) -> Result<Vec<usize>> {
        Ok(self.selected_features.clone())
    }
}
