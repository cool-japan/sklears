//! Comprehensive Covariance Estimation Cookbook
//!
//! This cookbook demonstrates how to use the automatic model selection
//! (`AutoCovarianceSelector`) and hyperparameter tuning
//! (`CovarianceHyperparameterTuner`) machinery in `sklears-covariance`
//! together, end to end: from raw data, to a tuned and selected covariance
//! estimator, to production-style diagnostics.
//!
//! Six recipes:
//! 1. Quick start: raw data -> an automatically selected covariance matrix.
//! 2. Professional workflow: a custom candidate estimator, a ranked
//!    comparison, and real diagnostics (`CovarianceDiagnostics`) instead of
//!    hand-rolled risk heuristics.
//! 3. Advanced model selection: the *candidate set* registered with a
//!    selector -- not its strategy label -- is what determines what it can
//!    pick; shown across several synthetic data profiles.
//! 4. Hyperparameter optimization: tune `ShrunkCovariance`'s shrinkage with
//!    `CovarianceHyperparameterTuner`, then register the tuned estimator as a
//!    model-selection candidate alongside untuned baselines.
//! 5. Production deployment: robust selection, a `CovarianceDiagnostics`
//!    quality gate, and a `CovarianceBenchmark` performance check.
//! 6. Troubleshooting: diagnosing a genuinely ill-conditioned dataset with
//!    real tooling rather than invented heuristics.
//!
//! Run with: `cargo run --example comprehensive_cookbook`

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::{SeedableRng, StdRng};
use sklears_core::error::Result as SklResult;
use sklears_core::traits::Fit;
use sklears_covariance::{
    model_selection_presets, tuning_presets, ArrayCovarianceResult, ArrayEstimator,
    AutoCovarianceSelector, ComputationalComplexity, CovarianceBenchmark, CovarianceDiagnostics,
    CovarianceEstimatorFitted, CovarianceEstimatorTunable, CovarianceHyperparameterTuner,
    DataCharacteristics, EmpiricalCovariance, GraphicalLasso, LedoitWolf, ModelSelectionResult,
    ModelSelectionScoring, ParameterSpec, ParameterType, ParameterValue, QualityAssessment,
    SelectionRule, SelectionStrategy, ShrunkCovariance, ShrunkCovarianceTrained,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comprehensive Covariance Estimation Cookbook");
    println!("===============================================\n");

    recipe_quick_start()?;
    recipe_professional_workflow()?;
    recipe_advanced_model_selection()?;
    recipe_hyperparameter_optimization()?;
    recipe_production_deployment()?;
    recipe_troubleshooting()?;

    println!("Cookbook complete -- every recipe ran against the real, current API.");
    Ok(())
}

// ============================================================================
// Recipe 1: Quick start
// ============================================================================

fn recipe_quick_start() -> SklResult<()> {
    println!("Recipe 1: Quick Start - From Raw Data to Results");
    println!("===================================================");
    println!("Goal: get from raw data to a covariance matrix in a few lines.\n");

    println!("Step 1: prepare your data");
    let data = generate_portfolio(12, 3, 1, 0.02, (0.6, 1.2), 0.01);
    let column_names = ["Asset_A", "Asset_B", "Asset_C"];
    println!(
        "  Created a {} x {} data matrix",
        data.nrows(),
        data.ncols()
    );

    println!("\nStep 2: automatic model selection (easiest approach)");
    let selector = model_selection_presets::basic_selector();
    let result = selector.select_best(&data.view())?;
    println!("  Selected estimator: {}", result.best_estimator.name);

    println!("\nStep 3: your covariance matrix");
    let cov_matrix = &result.best_estimator.result.covariance;
    print!("{:>12}", "");
    for name in &column_names {
        print!("{name:>12}");
    }
    println!();
    for (i, name) in column_names.iter().enumerate() {
        print!("{name:>12}");
        for j in 0..cov_matrix.ncols() {
            print!("{:12.4}", cov_matrix[[i, j]]);
        }
        println!();
    }

    println!("\nStep 4: quick insights");
    println!(
        "  Selection confidence: {:.1}%",
        result.best_estimator.confidence * 100.0
    );
    println!(
        "  Sample/feature ratio: {:.1}",
        result.data_characteristics.sample_feature_ratio
    );
    println!(
        "  Estimated sparsity: {:.1}%",
        result.data_characteristics.sparsity_level * 100.0
    );

    println!("\nNext: Recipe 2 walks through a full professional workflow.\n");
    Ok(())
}

// ============================================================================
// Recipe 2: Professional workflow
// ============================================================================

/// Wraps [`ShrunkCovariance`] with a fixed shrinkage intensity as a
/// model-selection candidate.
#[derive(Debug)]
struct ShrunkArrayEstimator {
    shrinkage: f64,
}

impl ArrayEstimator<f64> for ShrunkArrayEstimator {
    fn fit_array(&self, data: &ArrayView2<f64>) -> SklResult<ArrayCovarianceResult<f64>> {
        let fitted = ShrunkCovariance::new()
            .shrinkage(self.shrinkage)
            .fit(data, &())?;
        Ok(ArrayCovarianceResult {
            covariance: fitted.get_covariance().clone(),
            precision: fitted.get_precision().cloned(),
        })
    }

    fn name(&self) -> &str {
        "ShrunkCovariance"
    }
}

/// Wraps [`GraphicalLasso`] with a fixed regularization strength as a
/// model-selection candidate.
#[derive(Debug)]
struct GraphicalLassoArrayEstimator {
    alpha: f64,
}

impl ArrayEstimator<f64> for GraphicalLassoArrayEstimator {
    fn fit_array(&self, data: &ArrayView2<f64>) -> SklResult<ArrayCovarianceResult<f64>> {
        let fitted = GraphicalLasso::new().alpha(self.alpha).fit(data, &())?;

        // `get_covariance()` is the covariance implied by the fitted,
        // regularized precision matrix, so it already reflects `alpha` --
        // no manual ridge-then-invert workaround needed here.
        Ok(ArrayCovarianceResult {
            covariance: fitted.get_covariance().clone(),
            precision: Some(fitted.get_precision().clone()),
        })
    }

    fn name(&self) -> &str {
        "GraphicalLasso"
    }
}

/// A selector that starts from the basic preset (Empirical + LedoitWolf) and
/// registers a fixed-shrinkage `ShrunkCovariance` candidate alongside it.
fn professional_selector() -> AutoCovarianceSelector<f64> {
    let selector = model_selection_presets::basic_selector();
    let selector = selector.add_candidate(
        "ShrunkCovariance(0.3)".to_string(),
        Box::new(|_chars: &DataCharacteristics| {
            Ok(Box::new(ShrunkArrayEstimator { shrinkage: 0.3 }) as Box<dyn ArrayEstimator<f64>>)
        }),
        DataCharacteristics::default(),
        ComputationalComplexity::Quadratic,
    );
    // `.selection_strategy(...)` records a label in the selection metadata
    // for reporting/auditing; the actual winner is still driven by each
    // candidate's cross-validated score.
    selector.selection_strategy(SelectionStrategy::MultiObjective {
        performance_weight: 0.5,
        complexity_weight: 0.3,
        stability_weight: 0.2,
    })
}

struct ColumnSummary {
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
}

fn column_stats(data: &Array2<f64>) -> Vec<ColumnSummary> {
    let n_samples = data.nrows() as f64;
    (0..data.ncols())
        .map(|j| {
            let column = data.column(j);
            let mean = column.sum() / n_samples;
            let variance = column.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_samples;
            let min = column.iter().copied().fold(f64::INFINITY, f64::min);
            let max = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            ColumnSummary {
                mean,
                std_dev: variance.sqrt(),
                min,
                max,
            }
        })
        .collect()
}

fn sector_asset_names(n_assets: usize) -> Vec<String> {
    (0..n_assets)
        .map(|i| {
            let sector = match i % 3 {
                0 => "TECH",
                1 => "FIN",
                _ => "HLTH",
            };
            format!("{sector}_Asset_{i:02}")
        })
        .collect()
}

fn recipe_professional_workflow() -> SklResult<()> {
    println!("Recipe 2: Professional Data Analysis Workflow");
    println!("=================================================");
    println!("Goal: a complete workflow with summary statistics, ranked");
    println!("candidates, and real risk diagnostics.\n");

    println!("Step 1: realistic financial dataset");
    let financial_data = create_professional_dataset(200, 10, 100);
    let column_names = sector_asset_names(10);
    println!(
        "  Created {} samples x {} assets",
        financial_data.nrows(),
        financial_data.ncols()
    );

    println!("\nStep 2: summary statistics per asset");
    for (name, stats) in column_names.iter().zip(column_stats(&financial_data)) {
        println!(
            "  {name}: mean={:.5}, std={:.5}, range=[{:.4}, {:.4}]",
            stats.mean, stats.std_dev, stats.min, stats.max
        );
    }

    println!("\nStep 3: model selection with a custom candidate");
    let selector = professional_selector();
    let selection_result = selector.select_best(&financial_data.view())?;

    println!("  Best estimator: {}", selection_result.best_estimator.name);
    println!(
        "  Recorded strategy label: {}",
        selection_result.selection_metadata.strategy
    );
    println!(
        "  Cross-validation score (higher is better): {:.4}",
        selection_result.best_estimator.score
    );
    println!(
        "  Selection confidence: {:.1}%",
        selection_result.best_estimator.confidence * 100.0
    );

    println!("\nStep 4: ranked comparison across candidates");
    let mut candidates = selection_result.candidate_results.clone();
    candidates.sort_by(|a, b| {
        b.mean_score
            .partial_cmp(&a.mean_score)
            .expect("cross-validation scores are always finite")
    });
    for (rank, candidate) in candidates.iter().enumerate() {
        println!(
            "  {}. {} (score: {:.4}, std: {:.4})",
            rank + 1,
            candidate.name,
            candidate.mean_score,
            candidate.score_std
        );
    }

    println!("\nStep 5: risk diagnostics on the selected covariance matrix");
    let diagnostics =
        CovarianceDiagnostics::analyze(&selection_result.best_estimator.result.covariance)?;
    println!("  Quality assessment: {:?}", diagnostics.quality_assessment);
    println!(
        "  Condition number (max/min variance ratio): {:.2e}",
        diagnostics.properties.condition_number
    );
    println!(
        "  Mean |correlation|: {:.4}, max |correlation|: {:.4} ({} pairs > 0.8)",
        diagnostics.correlation_diagnostics.mean_abs_correlation,
        diagnostics.correlation_diagnostics.max_correlation,
        diagnostics.correlation_diagnostics.n_high_correlations
    );

    println!("\nStep 6: professional summary");
    println!("COVARIANCE ESTIMATION REPORT");
    println!("============================");
    println!(
        "Dataset:    {} observations, {} assets",
        financial_data.nrows(),
        financial_data.ncols()
    );
    println!("Method:     {}", selection_result.best_estimator.name);
    println!(
        "Validation: {}-fold cross-validation",
        selection_result.selection_metadata.cv_config.n_folds
    );
    println!(
        "Confidence: {:.1}%",
        selection_result.best_estimator.confidence * 100.0
    );

    println!("\nProfessional analysis complete.\n");
    Ok(())
}

// ============================================================================
// Recipe 3: Advanced model selection pipeline
// ============================================================================

/// A selector combining the basic preset with fixed-parameter
/// `ShrunkCovariance` and `GraphicalLasso` candidates.
fn comprehensive_selector() -> AutoCovarianceSelector<f64> {
    let selector = model_selection_presets::basic_selector();
    let selector = selector.add_candidate(
        "ShrunkCovariance(0.3)".to_string(),
        Box::new(|_chars: &DataCharacteristics| {
            Ok(Box::new(ShrunkArrayEstimator { shrinkage: 0.3 }) as Box<dyn ArrayEstimator<f64>>)
        }),
        DataCharacteristics::default(),
        ComputationalComplexity::Quadratic,
    );
    selector.add_candidate(
        "GraphicalLasso(0.05)".to_string(),
        Box::new(|_chars: &DataCharacteristics| {
            Ok(
                Box::new(GraphicalLassoArrayEstimator { alpha: 0.05 })
                    as Box<dyn ArrayEstimator<f64>>,
            )
        }),
        DataCharacteristics::default(),
        ComputationalComplexity::Cubic,
    )
}

fn recipe_advanced_model_selection() -> SklResult<()> {
    println!("Recipe 3: Advanced Model Selection Pipeline");
    println!("==============================================");
    println!("Goal: show that the *candidate set* registered with a selector");
    println!("-- not just its strategy label -- is what determines the winner,");
    println!("across several synthetic data profiles.\n");

    let datasets: Vec<(&str, Array2<f64>)> = vec![
        (
            "Conservative Portfolio",
            generate_portfolio(100, 8, 201, 0.005, (0.2, 0.6), 0.003),
        ),
        (
            "High-Growth Portfolio",
            generate_portfolio(120, 12, 202, 0.025, (1.2, 2.0), 0.015),
        ),
        (
            "International Portfolio",
            generate_international_portfolio(150, 15, 203),
        ),
    ];

    for (scenario, dataset) in &datasets {
        println!("\nScenario: {scenario}");
        println!("  Data shape: {} x {}", dataset.nrows(), dataset.ncols());

        let selector_results: Vec<(&str, SklResult<ModelSelectionResult<f64>>)> = vec![
            (
                "basic (Empirical + LedoitWolf)",
                model_selection_presets::basic_selector().select_best(&dataset.view()),
            ),
            (
                "high-dimensional (LedoitWolf only)",
                model_selection_presets::high_dimensional_selector().select_best(&dataset.view()),
            ),
            (
                "comprehensive (+ Shrunk + GraphicalLasso)",
                comprehensive_selector().select_best(&dataset.view()),
            ),
        ];

        for (selector_name, result) in &selector_results {
            match result {
                Ok(r) => println!(
                    "  [{selector_name}] winner: {} (score: {:.4})",
                    r.best_estimator.name, r.best_estimator.score
                ),
                Err(e) => println!("  [{selector_name}] selection failed: {e}"),
            }
        }

        if let Some((_, Ok(baseline))) = selector_results.first() {
            let chars = &baseline.data_characteristics;
            println!("  Characteristics (from the basic selector's analysis):");
            println!(
                "    {} samples / {} features = {:.1} ratio",
                chars.n_samples, chars.n_features, chars.sample_feature_ratio
            );
            println!(
                "    Estimated sparsity: {:.1}%, condition number: {:.2e}",
                chars.sparsity_level * 100.0,
                chars.condition_number
            );
            println!(
                "    Roughly normal: {}, skewness: {:.2}",
                chars.distribution.normality.is_normal, chars.distribution.skewness
            );
        }
    }

    println!("\nTakeaways:");
    println!("  - A selector can never choose an estimator it was never given: the");
    println!("    registered candidate set is what actually drives the outcome.");
    println!("  - Data characteristics (sparsity, conditioning, sample/feature ratio)");
    println!("    are computed independently of the candidate set and are a useful");
    println!("    first diagnostic before deciding which candidates to register.");
    println!();
    Ok(())
}

// ============================================================================
// Recipe 4: Hyperparameter optimization pipeline
// ============================================================================

/// Wraps [`ShrunkCovariance`] for [`CovarianceHyperparameterTuner`]. Unlike
/// `LedoitWolf` (whose shrinkage is computed automatically and read via
/// `get_shrinkage()`), `ShrunkCovariance` exposes shrinkage as a settable
/// builder parameter, which is what makes it tunable here.
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

fn recipe_hyperparameter_optimization() -> SklResult<()> {
    println!("Recipe 4: Hyperparameter Optimization Pipeline");
    println!("=================================================");
    println!("Goal: tune a regularization parameter, then hand the tuned");
    println!("estimator to automatic model selection alongside untuned");
    println!("baselines.\n");

    let data = generate_portfolio(100, 10, 301, 0.02, (0.5, 1.5), 0.01);
    println!(
        "Optimization dataset: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );

    println!("\nStep 1: tune ShrunkCovariance's shrinkage intensity");
    let param_specs = vec![ParameterSpec {
        name: "shrinkage".to_string(),
        param_type: ParameterType::Continuous { min: 0.0, max: 1.0 },
        log_scale: false,
    }];
    let config = tuning_presets::bayesian_optimization_default();
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

    let tuning_result = tuner.tune(estimator_factory, &data.view(), None)?;
    println!("  Best score: {:.6}", tuning_result.best_score);
    let best_shrinkage = match tuning_result.best_params.get("shrinkage") {
        Some(ParameterValue::Float(s)) => *s,
        _ => 0.1,
    };
    println!("  Best shrinkage: {best_shrinkage:.4}");
    println!("  Evaluations: {}", tuning_result.n_evaluations);
    println!("  Total time: {:.3}s", tuning_result.total_time_seconds);

    println!("\nStep 2: convergence analysis");
    let history = &tuning_result.optimization_history.best_scores;
    if history.len() >= 2 {
        let initial_score = history[0];
        let final_score = *history.last().expect("history is non-empty");
        println!("  Initial best score: {initial_score:.6}");
        println!("  Final best score:   {final_score:.6}");
        println!("  Improvement:        {:.6}", final_score - initial_score);
    }

    println!("\nStep 3: register the tuned estimator as a selection candidate");
    let selector = model_selection_presets::basic_selector().add_candidate(
        format!("ShrunkCovariance(tuned={best_shrinkage:.4})"),
        Box::new(move |_chars: &DataCharacteristics| {
            Ok(Box::new(ShrunkArrayEstimator {
                shrinkage: best_shrinkage,
            }) as Box<dyn ArrayEstimator<f64>>)
        }),
        DataCharacteristics::default(),
        ComputationalComplexity::Quadratic,
    );
    let selection_result = selector.select_best(&data.view())?;
    println!(
        "  Selector chose: {} (score: {:.4})",
        selection_result.best_estimator.name, selection_result.best_estimator.score
    );
    println!("  All registered candidates, ranked:");
    let mut ranked = selection_result.candidate_results.clone();
    ranked.sort_by(|a, b| {
        b.mean_score
            .partial_cmp(&a.mean_score)
            .expect("cross-validation scores are always finite")
    });
    for candidate in &ranked {
        println!("    {}: {:.4}", candidate.name, candidate.mean_score);
    }

    println!();
    Ok(())
}

// ============================================================================
// Recipe 5: Production deployment
// ============================================================================

fn recipe_production_deployment() -> SklResult<()> {
    println!("Recipe 5: Production Deployment Recipe");
    println!("=========================================");
    println!("Goal: select and validate a covariance estimator with");
    println!("production readiness checks before deployment.\n");

    let production_data = create_professional_dataset(500, 20, 401);
    println!(
        "Production dataset: {} samples, {} features",
        production_data.nrows(),
        production_data.ncols()
    );

    println!("\nStep 1: robust model selection");
    let selector = comprehensive_selector().selection_strategy(SelectionStrategy::CrossValidation {
        scoring: ModelSelectionScoring::CrossValidationStability,
        selection_rule: SelectionRule::OneStandardError,
    });
    let prod_result = selector.select_best(&production_data.view())?;

    println!("  Selected estimator: {}", prod_result.best_estimator.name);
    println!(
        "  Selection confidence: {:.1}%",
        prod_result.best_estimator.confidence * 100.0
    );

    println!("\nStep 2: quality gate via CovarianceDiagnostics");
    let diagnostics =
        CovarianceDiagnostics::analyze(&prod_result.best_estimator.result.covariance)?;
    println!("  Quality assessment: {:?}", diagnostics.quality_assessment);
    println!(
        "  Condition number (fitted covariance diagnostic): {:.2e}",
        diagnostics.properties.condition_number
    );
    println!(
        "  Positive definite: {}",
        diagnostics.properties.is_positive_definite
    );

    match diagnostics.quality_assessment {
        QualityAssessment::Excellent | QualityAssessment::Good => {
            println!("  -> Passed the quality gate: safe to deploy.");
        }
        QualityAssessment::Acceptable => {
            println!("  -> Marginal quality: deploy with monitoring.");
        }
        QualityAssessment::Poor | QualityAssessment::Failed => {
            println!("  -> Failed the quality gate: do not deploy without remediation.");
        }
    }

    println!("\nStep 3: performance benchmark");
    let benchmark = CovarianceBenchmark::new(5);
    let bench_result = benchmark.time_execution(|| {
        let _ = EmpiricalCovariance::new().fit(&production_data.view(), &());
    });
    println!(
        "  EmpiricalCovariance: {:.2}ms mean, {:.1} fits/sec",
        bench_result.mean_time_ms(),
        bench_result.throughput_ops_per_sec()
    );

    println!("\nStep 4: deployment recommendations from data characteristics");
    let chars = &prod_result.data_characteristics;
    if chars.sample_feature_ratio < 5.0 {
        println!("  Low sample/feature ratio -> prefer regularized estimators.");
    } else {
        println!(
            "  Sample/feature ratio ({:.1}) is healthy.",
            chars.sample_feature_ratio
        );
    }
    if chars.condition_number > 1e10 {
        println!("  High condition number -> prefer robust/regularized methods.");
    } else {
        println!(
            "  Condition number (raw-data characteristic, {:.2e}) is within a reasonable range.",
            chars.condition_number
        );
    }
    if chars.computational_constraints.can_parallelize {
        println!("  Parallelization is available for scaling to larger datasets.");
    }

    println!("\nProduction deployment checklist complete.\n");
    Ok(())
}

// ============================================================================
// Recipe 6: Troubleshooting and diagnostics
// ============================================================================

fn recipe_troubleshooting() -> SklResult<()> {
    println!("Recipe 6: Troubleshooting and Diagnostics");
    println!("============================================");
    println!("Goal: diagnose common covariance estimation problems with real");
    println!("diagnostic tooling rather than guesswork.\n");

    println!("Problem 1: a near-constant feature drives up the condition number");
    let problematic_data = create_problematic_dataset(50, 10, 501);
    let empirical_fit = EmpiricalCovariance::new().fit(&problematic_data.view(), &())?;
    let diagnostics = CovarianceDiagnostics::analyze(empirical_fit.get_covariance())?;

    println!(
        "  Condition number: {:.2e}",
        diagnostics.properties.condition_number
    );
    println!("  Quality assessment label: {:?}", diagnostics.quality_assessment);
    println!(
        "  (the label is a coarse, multi-factor summary -- always sanity-check the"
    );
    println!(
        "  condition number and other raw diagnostics directly rather than relying"
    );
    println!("  on the single label alone, as this example's numbers do.)");
    if diagnostics.properties.condition_number > 1e6 {
        println!("  Diagnosis: one feature's variance is orders of magnitude");
        println!("  smaller than the others (a near-constant column).");
        println!("  Fix: use a shrinkage-regularized estimator.");

        let lw_fit = LedoitWolf::new().fit(&problematic_data.view(), &())?;
        let lw_diagnostics = CovarianceDiagnostics::analyze(lw_fit.get_covariance())?;
        println!(
            "  After LedoitWolf shrinkage (shrinkage={:.4}): condition number {:.2e} (was {:.2e})",
            lw_fit.get_shrinkage(),
            lw_diagnostics.properties.condition_number,
            diagnostics.properties.condition_number
        );
    }

    println!("\nProblem 2: is there enough data?");
    println!(
        "  Sample size: {} (rule of thumb: prefer > 5x the feature count)",
        problematic_data.nrows()
    );
    println!(
        "  Dimensionality: {} features, sample/feature ratio = {:.1}",
        problematic_data.ncols(),
        problematic_data.nrows() as f64 / problematic_data.ncols() as f64
    );

    println!("\nProblem 3: how fast is estimation?");
    let benchmark = CovarianceBenchmark::new(10);
    let empirical_bench = benchmark.time_execution(|| {
        let _ = EmpiricalCovariance::new().fit(&problematic_data.view(), &());
    });
    let gl_bench = benchmark.time_execution(|| {
        let _ = GraphicalLasso::new()
            .alpha(0.1)
            .fit(&problematic_data.view(), &());
    });
    println!(
        "  EmpiricalCovariance: {:.3}ms mean",
        empirical_bench.mean_time_ms()
    );
    println!("  GraphicalLasso:      {:.3}ms mean", gl_bench.mean_time_ms());
    if gl_bench.mean_time_ms() > empirical_bench.mean_time_ms() {
        println!("  GraphicalLasso is slower here; prefer it when you specifically");
        println!("  need a sparse precision matrix, not as a default choice.");
    }

    println!("\nDiagnostic summary:");
    println!("  - Extreme per-feature variance ratios -> LedoitWolf or ShrunkCovariance.");
    println!("  - Low sample/feature ratio -> collect more data or regularize more.");
    println!("  - Slow computation -> benchmark first (CovarianceBenchmark), then decide");
    println!("    whether a cheaper estimator meets your accuracy bar.");
    println!("  - Always fix a random seed during development so results are reproducible.");

    println!("\nTroubleshooting guide complete.\n");
    Ok(())
}

// ============================================================================
// Synthetic data generators
// ============================================================================

/// Single-factor synthetic returns with a configurable market volatility and
/// beta range, used to build portfolios with different risk profiles.
fn generate_portfolio(
    n_samples: usize,
    n_assets: usize,
    seed: u64,
    market_vol: f64,
    beta_range: (f64, f64),
    idiosyncratic_vol: f64,
) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Array2::zeros((n_samples, n_assets));
    let (beta_min, beta_max) = beta_range;

    for i in 0..n_samples {
        let market_return = rng.gen_range(-market_vol..market_vol);
        for j in 0..n_assets {
            let beta = beta_min + (j as f64 / n_assets as f64) * (beta_max - beta_min);
            let idiosyncratic = rng.gen_range(-idiosyncratic_vol..idiosyncratic_vol);
            data[[i, j]] = market_return * beta + idiosyncratic;
        }
    }
    data
}

/// Returns driven by three independent regional market factors plus a shared
/// currency effect, giving each region a distinct correlation structure.
fn generate_international_portfolio(n_samples: usize, n_assets: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Array2::zeros((n_samples, n_assets));

    for i in 0..n_samples {
        let us_market = rng.gen_range(-0.015..0.015);
        let europe_market = rng.gen_range(-0.012..0.012);
        let asia_market = rng.gen_range(-0.018..0.018);
        let currency_effect = rng.gen_range(-0.008..0.008);

        for j in 0..n_assets {
            let (market_return, currency_exposure) = match j % 3 {
                0 => (us_market, 0.0),
                1 => (europe_market, currency_effect * 0.7),
                _ => (asia_market, currency_effect * 0.9),
            };
            let beta = 0.8 + (j as f64 / n_assets as f64) * 0.6;
            let idiosyncratic = rng.gen_range(-0.01..0.01);
            data[[i, j]] = market_return * beta + currency_exposure + idiosyncratic;
        }
    }
    data
}

/// A financial-style dataset with a market factor, three sector factors,
/// volatility clustering, and occasional jumps (fat tails).
fn create_professional_dataset(n_samples: usize, n_assets: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Array2::zeros((n_samples, n_assets));

    for i in 0..n_samples {
        let phase = i as f64 / n_samples as f64 * 2.0 * std::f64::consts::PI;
        let market_vol = 0.15 + 0.05 * phase.sin().abs();
        let market_return = rng.gen_range(-market_vol..market_vol) / 16.0;

        let tech_factor = rng.gen_range(-0.01..0.01);
        let finance_factor = rng.gen_range(-0.008..0.008);
        let healthcare_factor = rng.gen_range(-0.006..0.006);

        for j in 0..n_assets {
            let (sector_loading, sector_return) = match j % 3 {
                0 => (0.7, tech_factor),
                1 => (0.5, finance_factor),
                _ => (0.4, healthcare_factor),
            };

            let beta = 0.6 + (j as f64 / n_assets as f64) * 0.8;
            let idiosyncratic_vol = 0.01 + 0.005 * rng.gen_range(0.0..1.0);
            let idiosyncratic = rng.gen_range(-idiosyncratic_vol..idiosyncratic_vol);

            // Occasional jumps (fat tails), ~2% of observations.
            let jump = if rng.gen_range(0.0..1.0) < 0.02 {
                rng.gen_range(-0.05..0.05)
            } else {
                0.0
            };

            data[[i, j]] =
                market_return * beta + sector_return * sector_loading + idiosyncratic + jump;
        }
    }

    data
}

/// A dataset with one near-constant feature (tiny variance) among otherwise
/// normal-scale features. `CovarianceDiagnostics`'s condition-number
/// approximation is the ratio of the largest to smallest per-feature
/// variance, so this reliably produces a large, genuinely diagnosable value.
fn create_problematic_dataset(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = if j == 0 {
                1.0 + rng.gen_range(-1e-6..1e-6)
            } else {
                rng.gen_range(-1.0..1.0)
            };
        }
    }

    data
}
