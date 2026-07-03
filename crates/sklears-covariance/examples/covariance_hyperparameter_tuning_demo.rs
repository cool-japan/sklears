//! Hyperparameter Tuning Demo for Covariance Estimators
//!
//! This example demonstrates the automatic hyperparameter tuning capabilities
//! for covariance estimators (`CovarianceHyperparameterTuner`), showing how to
//! optimize regularization parameters and other hyperparameters using several
//! search strategies (grid search, random search, and Bayesian-style search),
//! how to compare multiple estimator families under a common tuning harness,
//! and how to plug in custom scoring metrics.
//!
//! **API note**: `LedoitWolf`'s shrinkage intensity is *computed automatically*
//! from the data and exposed as a read-only property of the fitted model via
//! `get_shrinkage()` -- it is not a settable builder parameter, so it cannot be
//! searched over by a hyperparameter tuner. `ShrunkCovariance`, in contrast,
//! exposes shrinkage as a settable builder parameter
//! (`ShrunkCovariance::new().shrinkage(value)`), which makes it the right
//! estimator to plug into the tuning framework below. Demo 1 tunes
//! `ShrunkCovariance`'s shrinkage and compares the result against
//! `LedoitWolf`'s automatically chosen shrinkage.
//!
//! Run with: `cargo run --example covariance_hyperparameter_tuning_demo`

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::{Distribution, SeedableRng, StdRng};
use sklears_core::error::Result as SklResult;
use sklears_core::traits::Fit;
use sklears_covariance::{
    tuning_presets, AcquisitionFunction, CovarianceEstimatorFitted, CovarianceEstimatorTunable,
    CovarianceHyperparameterTuner, CrossValidationConfig, EmpiricalCovariance,
    EmpiricalCovarianceTrained, GraphicalLasso, LedoitWolf, ParameterSpec, ParameterType,
    ParameterValue, ScoringMetric, SearchStrategy, ShrunkCovariance, ShrunkCovarianceTrained,
    TuningConfig, TuningResult,
};
use std::collections::HashMap;

/// Factory closures handed to [`CovarianceHyperparameterTuner::tune`].
type TunableFactory =
    Box<dyn Fn(&HashMap<String, ParameterValue>) -> SklResult<Box<dyn CovarianceEstimatorTunable<f64>>>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hyperparameter Tuning Demo for Covariance Estimators");
    println!("========================================================\n");

    demo_shrinkage_tuning()?;
    demo_graphical_lasso_tuning()?;
    demo_estimator_comparison()?;
    demo_search_strategies()?;
    demo_custom_scoring()?;

    println!("All hyperparameter tuning demos completed successfully!");
    Ok(())
}

// ============================================================================
// Synthetic data generators
// ============================================================================

/// Data driven by a single common factor plus feature-specific noise, so that
/// shrinkage-based regularization has a clear, measurable effect.
fn generate_factor_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("standard normal parameters are always valid");

    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        let common_factor = normal.sample(&mut rng);
        for j in 0..n_features {
            let loading = 0.3 + 0.4 * (j as f64 / n_features as f64);
            let noise = normal.sample(&mut rng);
            data[[i, j]] = common_factor * loading + 0.8 * noise;
        }
    }
    data
}

/// Data with an explicit block-sparse correlation structure: the first three
/// features share one latent factor, the next three share a second factor,
/// and the rest are independent noise. This gives `GraphicalLasso` a genuine
/// sparsity pattern to recover.
fn generate_block_sparse_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("standard normal parameters are always valid");

    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        let factor_a = normal.sample(&mut rng);
        let factor_b = normal.sample(&mut rng);
        for j in 0..n_features {
            let noise = normal.sample(&mut rng);
            data[[i, j]] = if j < 3 {
                0.9 * factor_a + 0.3 * noise
            } else if j < 6 {
                0.9 * factor_b + 0.3 * noise
            } else {
                noise
            };
        }
    }
    data
}

// ============================================================================
// Tunable / Fitted wrappers
//
// `CovarianceEstimatorTunable::fit` takes `&self`, so each wrapper owns just
// the candidate hyperparameter values and constructs a fresh, owned estimator
// on every call (matching the consuming `Fit::fit(self, ...)` signature of
// the underlying estimators).
// ============================================================================

/// Wraps [`EmpiricalCovariance`]. It has no hyperparameters of its own, which
/// makes it a useful parameter-free baseline in a multi-estimator comparison.
struct TunableEmpirical;

impl CovarianceEstimatorTunable<f64> for TunableEmpirical {
    fn fit(
        &self,
        x: &ArrayView2<f64>,
        _y: Option<ArrayView2<f64>>,
    ) -> SklResult<Box<dyn CovarianceEstimatorFitted<f64>>> {
        let fitted = EmpiricalCovariance::new().fit(x, &())?;
        Ok(Box::new(FittedEmpirical { inner: fitted }))
    }
}

struct FittedEmpirical {
    inner: EmpiricalCovariance<EmpiricalCovarianceTrained>,
}

impl CovarianceEstimatorFitted<f64> for FittedEmpirical {
    fn get_covariance(&self) -> &Array2<f64> {
        self.inner.get_covariance()
    }

    fn get_precision(&self) -> Option<&Array2<f64>> {
        self.inner.get_precision()
    }
}

/// Wraps [`ShrunkCovariance`] with a fixed shrinkage intensity so the tuner
/// can search over it directly.
struct TunableShrunk {
    shrinkage: f64,
}

impl CovarianceEstimatorTunable<f64> for TunableShrunk {
    fn fit(
        &self,
        x: &ArrayView2<f64>,
        _y: Option<ArrayView2<f64>>,
    ) -> SklResult<Box<dyn CovarianceEstimatorFitted<f64>>> {
        let fitted = ShrunkCovariance::new()
            .shrinkage(self.shrinkage)
            .fit(x, &())?;
        Ok(Box::new(FittedShrunk { inner: fitted }))
    }
}

struct FittedShrunk {
    inner: ShrunkCovariance<ShrunkCovarianceTrained>,
}

impl CovarianceEstimatorFitted<f64> for FittedShrunk {
    fn get_covariance(&self) -> &Array2<f64> {
        self.inner.get_covariance()
    }

    fn get_precision(&self) -> Option<&Array2<f64>> {
        self.inner.get_precision()
    }
}

/// Wraps [`GraphicalLasso`] so that both `alpha` (the L1 penalty) and
/// `max_iter` can be tuned together.
struct TunableGraphicalLasso {
    alpha: f64,
    max_iter: usize,
}

impl CovarianceEstimatorTunable<f64> for TunableGraphicalLasso {
    fn fit(
        &self,
        x: &ArrayView2<f64>,
        _y: Option<ArrayView2<f64>>,
    ) -> SklResult<Box<dyn CovarianceEstimatorFitted<f64>>> {
        let fitted = GraphicalLasso::new()
            .alpha(self.alpha)
            .max_iter(self.max_iter)
            .fit(x, &())?;

        // `get_covariance()` is the covariance implied by the fitted,
        // regularized precision matrix, so it already reflects `alpha` --
        // no manual ridge-then-invert workaround needed here.
        Ok(Box::new(FittedGraphicalLasso {
            covariance: fitted.get_covariance().clone(),
            precision: fitted.get_precision().clone(),
        }))
    }
}

struct FittedGraphicalLasso {
    covariance: Array2<f64>,
    precision: Array2<f64>,
}

impl CovarianceEstimatorFitted<f64> for FittedGraphicalLasso {
    fn get_covariance(&self) -> &Array2<f64> {
        &self.covariance
    }

    fn get_precision(&self) -> Option<&Array2<f64>> {
        // GraphicalLasso always computes a precision matrix (that is its
        // whole point), so this is never `None`.
        Some(&self.precision)
    }
}

// ============================================================================
// Demo 1: ShrunkCovariance shrinkage tuning, compared against LedoitWolf
// ============================================================================

fn demo_shrinkage_tuning() -> SklResult<()> {
    println!("Demo 1: ShrunkCovariance Shrinkage Parameter Tuning");
    println!("----------------------------------------------------");

    let data = generate_factor_data(100, 20, 42);
    println!(
        "Created synthetic data: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );

    let param_specs = vec![ParameterSpec {
        name: "shrinkage".to_string(),
        param_type: ParameterType::Continuous { min: 0.0, max: 1.0 },
        log_scale: false,
    }];

    let config = tuning_presets::grid_search_regularization(0.0, 1.0, 11);
    let tuner = CovarianceHyperparameterTuner::new(param_specs, config);

    let estimator_factory = |params: &HashMap<String, ParameterValue>| -> SklResult<
        Box<dyn CovarianceEstimatorTunable<f64>>,
    > {
        let shrinkage = match params.get("shrinkage") {
            Some(ParameterValue::Float(s)) => *s,
            _ => 0.1,
        };
        Ok(Box::new(TunableShrunk { shrinkage }))
    };

    println!("Starting grid search over the shrinkage intensity...");
    let result = tuner.tune(estimator_factory, &data.view(), None)?;

    println!("\nTuning results:");
    println!("  Best score: {:.6}", result.best_score);
    for (param, value) in &result.best_params {
        println!("  Best {param}: {value:?}");
    }
    println!("  Score std across folds: {:.6}", result.best_score_std);
    println!("  Evaluations: {}", result.n_evaluations);
    println!("  Total time: {:.3}s", result.total_time_seconds);

    // For reference, compare against LedoitWolf, whose shrinkage is
    // *computed* automatically rather than searched for -- a read-only
    // property of the trained estimator, not a hyperparameter.
    let lw_fitted = LedoitWolf::new().fit(&data.view(), &())?;
    println!(
        "\nFor reference, LedoitWolf's automatically chosen shrinkage = {:.6}",
        lw_fitted.get_shrinkage()
    );

    println!("\nCross-validation score at each grid point:");
    for (i, &score) in result.optimization_history.scores.iter().enumerate() {
        println!("  Evaluation {}: score = {:.6}", i + 1, score);
    }

    println!();
    Ok(())
}

// ============================================================================
// Demo 2: GraphicalLasso regularization parameter tuning
// ============================================================================

fn demo_graphical_lasso_tuning() -> SklResult<()> {
    println!("Demo 2: GraphicalLasso Regularization Parameter Tuning");
    println!("---------------------------------------------------------");

    let data = generate_block_sparse_data(80, 15, 7);
    println!(
        "Created block-sparse data: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );

    let param_specs = vec![
        ParameterSpec {
            name: "alpha".to_string(),
            param_type: ParameterType::Continuous {
                min: -3.0,
                max: 0.0,
            }, // log10 scale: alpha in [0.001, 1.0]
            log_scale: true,
        },
        ParameterSpec {
            name: "max_iter".to_string(),
            param_type: ParameterType::Integer { min: 50, max: 200 },
            log_scale: false,
        },
    ];

    // Random search explores the 2D parameter space faster than a full grid.
    let config = tuning_presets::random_search_default(20);
    let tuner = CovarianceHyperparameterTuner::new(param_specs, config);

    let estimator_factory = |params: &HashMap<String, ParameterValue>| -> SklResult<
        Box<dyn CovarianceEstimatorTunable<f64>>,
    > {
        let alpha = match params.get("alpha") {
            Some(ParameterValue::Float(a)) => *a,
            _ => 0.01,
        };
        let max_iter = match params.get("max_iter") {
            Some(ParameterValue::Int(iter)) => *iter as usize,
            _ => 100,
        };
        Ok(Box::new(TunableGraphicalLasso { alpha, max_iter }))
    };

    println!("Starting random search over GraphicalLasso parameters...");
    let result = tuner.tune(estimator_factory, &data.view(), None)?;

    println!("\nBest GraphicalLasso configuration:");
    println!("  Best score: {:.6}", result.best_score);
    for (param, value) in &result.best_params {
        match (param.as_str(), value) {
            ("alpha", ParameterValue::Float(alpha)) => {
                println!("  Regularization (alpha): {alpha:.6}");
            }
            ("max_iter", ParameterValue::Int(iter)) => {
                println!("  Max iterations: {iter}");
            }
            _ => println!("  {param}: {value:?}"),
        }
    }
    println!("  Evaluations: {}", result.n_evaluations);

    // Refit with the winning configuration and report the *actual* sparsity
    // of the resulting precision matrix (rather than a hand-wavy estimate).
    let best_alpha = match result.best_params.get("alpha") {
        Some(ParameterValue::Float(a)) => *a,
        _ => 0.01,
    };
    let best_max_iter = match result.best_params.get("max_iter") {
        Some(ParameterValue::Int(iter)) => *iter as usize,
        _ => 100,
    };
    let best_fit = GraphicalLasso::new()
        .alpha(best_alpha)
        .max_iter(best_max_iter)
        .fit(&data.view(), &())?;
    let precision = best_fit.get_precision();
    let n = precision.nrows();
    let near_zero = precision.iter().filter(|&&value| value.abs() < 1e-3).count();
    let sparsity = near_zero as f64 / (n * n) as f64;

    println!("\nSparsity of the tuned precision matrix:");
    println!("  Entries near zero (|.| < 1e-3): {:.1}%", sparsity * 100.0);

    println!();
    Ok(())
}

// ============================================================================
// Demo 3: Multiple estimator comparison under a common tuning harness
// ============================================================================

fn empirical_tuner() -> (CovarianceHyperparameterTuner<f64>, TunableFactory) {
    // No hyperparameters to search over; a single evaluation is enough.
    let config = tuning_presets::random_search_default(1);
    let tuner = CovarianceHyperparameterTuner::new(Vec::new(), config);
    let factory: TunableFactory = Box::new(|_params: &HashMap<String, ParameterValue>| {
        Ok(Box::new(TunableEmpirical) as Box<dyn CovarianceEstimatorTunable<f64>>)
    });
    (tuner, factory)
}

fn shrunk_tuner() -> (CovarianceHyperparameterTuner<f64>, TunableFactory) {
    let param_specs = vec![ParameterSpec {
        name: "shrinkage".to_string(),
        param_type: ParameterType::Continuous { min: 0.0, max: 1.0 },
        log_scale: false,
    }];
    let config = tuning_presets::random_search_default(10);
    let tuner = CovarianceHyperparameterTuner::new(param_specs, config);
    let factory: TunableFactory = Box::new(|params: &HashMap<String, ParameterValue>| {
        let shrinkage = match params.get("shrinkage") {
            Some(ParameterValue::Float(s)) => *s,
            _ => 0.1,
        };
        Ok(Box::new(TunableShrunk { shrinkage }) as Box<dyn CovarianceEstimatorTunable<f64>>)
    });
    (tuner, factory)
}

fn graphical_lasso_tuner() -> (CovarianceHyperparameterTuner<f64>, TunableFactory) {
    let param_specs = tuning_presets::graphical_lasso_params();
    let config = tuning_presets::random_search_default(15);
    let tuner = CovarianceHyperparameterTuner::new(param_specs, config);
    let factory: TunableFactory = Box::new(|params: &HashMap<String, ParameterValue>| {
        let alpha = match params.get("alpha") {
            Some(ParameterValue::Float(a)) => *a,
            _ => 0.01,
        };
        let max_iter = match params.get("max_iter") {
            Some(ParameterValue::Int(iter)) => *iter as usize,
            _ => 100,
        };
        Ok(
            Box::new(TunableGraphicalLasso { alpha, max_iter })
                as Box<dyn CovarianceEstimatorTunable<f64>>,
        )
    });
    (tuner, factory)
}

fn demo_estimator_comparison() -> SklResult<()> {
    println!("Demo 3: Multiple Estimator Comparison with Tuning");
    println!("----------------------------------------------------");

    let data = generate_factor_data(150, 10, 11);
    println!(
        "Created financial-style data: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );

    let (emp_tuner, emp_factory) = empirical_tuner();
    let (shrunk_tuner_inst, shrunk_factory) = shrunk_tuner();
    let (gl_tuner, gl_factory) = graphical_lasso_tuner();

    let estimators: Vec<(&str, CovarianceHyperparameterTuner<f64>, TunableFactory)> = vec![
        ("Empirical", emp_tuner, emp_factory),
        ("ShrunkCovariance", shrunk_tuner_inst, shrunk_factory),
        ("GraphicalLasso", gl_tuner, gl_factory),
    ];

    println!("\nComparison results:");
    println!("======================");

    let mut all_results: Vec<(&str, TuningResult)> = Vec::new();
    for (name, tuner, factory) in estimators {
        println!("\nTuning {name}...");
        let start_time = std::time::Instant::now();
        let result = tuner.tune(factory, &data.view(), None)?;
        let elapsed = start_time.elapsed().as_secs_f64();

        println!("  Tuning time: {elapsed:.3}s");
        println!("  Best score: {:.6}", result.best_score);
        println!("  Evaluations: {}", result.n_evaluations);

        if !result.best_params.is_empty() {
            println!("  Best parameters:");
            for (param, value) in &result.best_params {
                println!("    {param}: {value:?}");
            }
        }

        all_results.push((name, result));
    }

    let best_estimator = all_results
        .iter()
        .max_by(|(_, a), (_, b)| {
            a.best_score
                .partial_cmp(&b.best_score)
                .expect("cross-validation scores are always finite for these metrics")
        })
        .expect("at least one estimator was evaluated");

    println!(
        "\nWinner: {} with score {:.6}",
        best_estimator.0, best_estimator.1.best_score
    );

    println!("\nPerformance summary:");
    for (name, result) in &all_results {
        println!(
            "  {}: {:.6} +/- {:.6}",
            name, result.best_score, result.best_score_std
        );
    }

    println!();
    Ok(())
}

// ============================================================================
// Demo 4: Advanced search strategies
// ============================================================================

fn demo_search_strategies() -> SklResult<()> {
    println!("Demo 4: Advanced Search Strategies");
    println!("--------------------------------------");

    let data = generate_block_sparse_data(120, 8, 21);
    println!(
        "Test data: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );

    let param_specs = vec![ParameterSpec {
        name: "alpha".to_string(),
        param_type: ParameterType::Continuous {
            min: -3.0,
            max: 0.0,
        },
        log_scale: true,
    }];

    let strategies: Vec<(&str, SearchStrategy)> = vec![
        ("Grid Search", SearchStrategy::GridSearch),
        ("Random Search", SearchStrategy::RandomSearch { n_iter: 15 }),
        (
            "Bayesian Optimization",
            SearchStrategy::BayesianOptimization {
                acquisition_function: AcquisitionFunction::ExpectedImprovement,
                n_initial_points: 5,
            },
        ),
    ];

    println!("\nSearch strategy comparison:");
    for (strategy_name, strategy) in strategies {
        println!("\nTesting {strategy_name}...");

        let config = TuningConfig {
            cv_config: CrossValidationConfig {
                n_folds: 3,
                shuffle: true,
                random_seed: Some(42),
                stratify: false,
            },
            search_strategy: strategy,
            scoring: ScoringMetric::LogLikelihood,
            max_evaluations: 15,
            random_seed: Some(42),
            n_jobs: None,
            early_stopping: None,
        };

        let tuner = CovarianceHyperparameterTuner::new(param_specs.clone(), config);

        let estimator_factory = |params: &HashMap<String, ParameterValue>| -> SklResult<
            Box<dyn CovarianceEstimatorTunable<f64>>,
        > {
            let alpha = match params.get("alpha") {
                Some(ParameterValue::Float(a)) => *a,
                _ => 0.01,
            };
            Ok(Box::new(TunableGraphicalLasso {
                alpha,
                max_iter: 100,
            }))
        };

        let result = tuner.tune(estimator_factory, &data.view(), None)?;

        println!("  Best score: {:.6}", result.best_score);
        println!("  Evaluations: {}", result.n_evaluations);
        println!("  Time: {:.3}s", result.total_time_seconds);

        let history = &result.optimization_history.best_scores;
        if history.len() >= 2 {
            let improvement = history.last().expect("history is non-empty")
                - history.first().expect("history is non-empty");
            println!("  Improvement from first to last evaluation: {improvement:.6}");
        }
    }

    println!();
    Ok(())
}

// ============================================================================
// Demo 5: Custom scoring metrics
// ============================================================================

fn demo_custom_scoring() -> SklResult<()> {
    println!("Demo 5: Custom Scoring Metrics");
    println!("-----------------------------------");

    let data = generate_block_sparse_data(100, 6, 33);
    println!(
        "Test data: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );

    let param_specs = vec![ParameterSpec {
        name: "alpha".to_string(),
        param_type: ParameterType::Continuous {
            min: -3.0,
            max: -1.0,
        },
        log_scale: true,
    }];

    let scoring_metrics: Vec<(&str, ScoringMetric)> = vec![
        ("Log-Likelihood", ScoringMetric::LogLikelihood),
        ("Condition Number", ScoringMetric::ConditionNumber),
        (
            "Sparsity-seeking (custom)",
            ScoringMetric::Custom(Box::new(|cov_matrix, _true_cov| {
                // Reward matrices whose off-diagonal mass is small relative
                // to the trace, i.e. more diagonal-dominant / sparse-looking
                // covariance structure.
                let n = cov_matrix.nrows();
                let mut off_diagonal_penalty = 0.0;
                let mut trace = 0.0;
                for i in 0..n {
                    trace += cov_matrix[[i, i]];
                    for j in 0..n {
                        if i != j {
                            off_diagonal_penalty += cov_matrix[[i, j]].abs();
                        }
                    }
                }
                -off_diagonal_penalty / (trace + 1e-8)
            })),
        ),
    ];

    println!("\nScoring metric comparison:");
    for (metric_name, metric) in scoring_metrics {
        println!("\nUsing {metric_name} metric...");

        let config = TuningConfig {
            cv_config: CrossValidationConfig {
                n_folds: 3,
                shuffle: true,
                random_seed: Some(42),
                stratify: false,
            },
            search_strategy: SearchStrategy::GridSearch,
            scoring: metric,
            max_evaluations: 10,
            random_seed: Some(42),
            n_jobs: None,
            early_stopping: None,
        };

        let tuner = CovarianceHyperparameterTuner::new(param_specs.clone(), config);

        let estimator_factory = |params: &HashMap<String, ParameterValue>| -> SklResult<
            Box<dyn CovarianceEstimatorTunable<f64>>,
        > {
            let alpha = match params.get("alpha") {
                Some(ParameterValue::Float(a)) => *a,
                _ => 0.01,
            };
            Ok(Box::new(TunableGraphicalLasso {
                alpha,
                max_iter: 100,
            }))
        };

        let result = tuner.tune(estimator_factory, &data.view(), None)?;

        println!("  Best score: {:.6}", result.best_score);
        if let Some(ParameterValue::Float(alpha)) = result.best_params.get("alpha") {
            println!("  Best alpha: {alpha:.6}");
        }
    }

    println!();
    Ok(())
}
