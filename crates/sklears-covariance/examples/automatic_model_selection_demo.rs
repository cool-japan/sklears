//! Automatic Model Selection Demo for Covariance Estimation
//!
//! This example demonstrates the advanced model selection capabilities that automatically
//! choose the best covariance estimator for given data characteristics. The system
//! analyzes data properties and intelligently selects from multiple candidate estimators.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::Rng;
use sklears_core::error::Result as SklResult;
use sklears_covariance::{
    model_selection_presets, AutoCovarianceSelector, CovarianceDataFrame, DataFrameEstimator,
    EmpiricalCovariance, ModelSelectionScoring, SelectionRule, SelectionStrategy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Automatic Model Selection Demo for Covariance Estimation");
    println!("===========================================================\n");

    // Demo 1: Basic automatic model selection
    demo_basic_model_selection()?;

    // Demo 2: Selection for different data types
    demo_data_type_specific_selection()?;

    // Demo 3: Advanced selection strategies
    demo_advanced_selection_strategies()?;

    // Demo 4: Data characterization analysis
    demo_data_characterization()?;

    // Demo 5: Performance comparison and insights
    demo_performance_insights()?;

    println!("✅ All automatic model selection demos completed successfully!");
    Ok(())
}

fn demo_basic_model_selection() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Demo 1: Basic Automatic Model Selection");
    println!("------------------------------------------");

    // Create sample financial return data
    let financial_data = create_financial_return_data(100, 8)?;
    println!(
        "📊 Created financial returns data: {} samples, {} features",
        financial_data.shape().0,
        financial_data.shape().1
    );

    // Create a basic model selector with common estimators
    let mut selector = model_selection_presets::basic_selector();

    // Add additional candidate estimators
    selector = add_custom_candidates(selector);

    println!("\n🔍 Analyzing data and selecting best estimator...");
    let start_time = std::time::Instant::now();

    // Perform automatic model selection
    let selection_result = selector.select_best(&financial_data)?;
    let selection_time = start_time.elapsed().as_secs_f64();

    println!("\n🏆 Model Selection Results:");
    println!("==========================");
    println!(
        "⭐ Best Estimator: {}",
        selection_result.best_estimator.name
    );
    println!(
        "📊 Selection Score: {:.6}",
        selection_result.best_estimator.score
    );
    println!(
        "🎯 Confidence: {:.1}%",
        selection_result.best_estimator.confidence * 100.0
    );
    println!("⏱️  Selection Time: {:.2}s", selection_time);

    println!("\n💡 Selection Reasons:");
    for (i, reason) in selection_result
        .best_estimator
        .selection_reasons
        .iter()
        .enumerate()
    {
        println!("  {}. {}", i + 1, reason);
    }

    // Display data characteristics
    println!("\n📋 Data Characteristics Analysis:");
    let chars = &selection_result.data_characteristics;
    println!("  📏 Dimensions: {}×{}", chars.n_samples, chars.n_features);
    println!(
        "  📊 Sample/Feature Ratio: {:.2}",
        chars.sample_feature_ratio
    );
    println!("  🔍 Sparsity Level: {:.1}%", chars.sparsity_level * 100.0);
    println!("  🔢 Condition Number: {:.2e}", chars.condition_number);
    println!(
        "  📈 Normal Distribution: {}",
        chars.distribution.normality.is_normal
    );

    // Show all candidate performance
    println!("\n📊 All Candidates Performance:");
    for candidate in &selection_result.candidate_results {
        println!(
            "  {:20} Score: {:8.4} ± {:6.4} Time: {:6.2}s",
            candidate.name, candidate.mean_score, candidate.score_std, candidate.computation_time
        );
    }

    println!();
    Ok(())
}

fn demo_data_type_specific_selection() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗂️  Demo 2: Selection for Different Data Types");
    println!("----------------------------------------------");

    // Test different data types
    let test_cases = vec![
        ("High-Dimensional", create_high_dimensional_data(50, 100)?),
        ("Well-Conditioned", create_well_conditioned_data(200, 10)?),
        (
            "Sparse Correlation",
            create_sparse_correlation_data(150, 15)?,
        ),
        ("Noisy Financial", create_noisy_financial_data(100, 12)?),
    ];

    for (data_type, data) in test_cases {
        println!("\n📊 Testing: {} Data", data_type);
        println!("  Data shape: {:?}", data.shape());

        // Use appropriate selector for data type
        let selector = match data_type {
            "High-Dimensional" => model_selection_presets::high_dimensional_selector(),
            "Sparse Correlation" => model_selection_presets::sparse_selector(),
            _ => model_selection_presets::basic_selector(),
        };

        let selector = add_custom_candidates(selector);

        match selector.select_best(&data) {
            Ok(result) => {
                println!("  🏆 Selected: {}", result.best_estimator.name);
                println!("  📊 Score: {:.4}", result.best_estimator.score);
                println!(
                    "  🎯 Confidence: {:.1}%",
                    result.best_estimator.confidence * 100.0
                );

                // Show why this estimator was chosen
                let chars = &result.data_characteristics;
                println!("  📋 Key characteristics:");
                println!(
                    "    Sample/Feature Ratio: {:.2}",
                    chars.sample_feature_ratio
                );
                println!("    Sparsity: {:.1}%", chars.sparsity_level * 100.0);
                println!("    Condition Number: {:.2e}", chars.condition_number);
            }
            Err(e) => {
                println!("  ❌ Selection failed: {}", e);
            }
        }
    }

    println!();
    Ok(())
}

fn demo_advanced_selection_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Demo 3: Advanced Selection Strategies");
    println!("----------------------------------------");

    let test_data = create_financial_return_data(120, 10)?;
    println!(
        "📊 Test data: {} samples, {} features",
        test_data.shape().0,
        test_data.shape().1
    );

    // Test different selection strategies
    let strategies = vec![
        (
            "Best Score",
            SelectionStrategy::CrossValidation {
                scoring: ModelSelectionScoring::LogLikelihood,
                selection_rule: SelectionRule::BestScore,
            },
        ),
        (
            "One Standard Error",
            SelectionStrategy::CrossValidation {
                scoring: ModelSelectionScoring::LogLikelihood,
                selection_rule: SelectionRule::OneStandardError,
            },
        ),
        (
            "Statistical Test",
            SelectionStrategy::CrossValidation {
                scoring: ModelSelectionScoring::LogLikelihood,
                selection_rule: SelectionRule::StatisticalTest { alpha: 0.05 },
            },
        ),
        (
            "Multi-Objective",
            SelectionStrategy::MultiObjective {
                performance_weight: 0.7,
                complexity_weight: 0.2,
                stability_weight: 0.1,
            },
        ),
    ];

    println!("\n🔬 Strategy Comparison Results:");
    println!("==============================");

    for (strategy_name, strategy) in strategies {
        let selector = model_selection_presets::basic_selector().selection_strategy(strategy);
        let selector = add_custom_candidates(selector);

        let start_time = std::time::Instant::now();
        match selector.select_best(&test_data) {
            Ok(result) => {
                let elapsed = start_time.elapsed().as_secs_f64();
                println!("\n📊 {} Strategy:", strategy_name);
                println!("  🏆 Selected: {}", result.best_estimator.name);
                println!("  📊 Score: {:.6}", result.best_estimator.score);
                println!(
                    "  🎯 Confidence: {:.1}%",
                    result.best_estimator.confidence * 100.0
                );
                println!("  ⏱️  Time: {:.2}s", elapsed);

                // Show performance comparison
                let perf = &result.performance_comparison;
                println!(
                    "  📈 Score Range: {:.4} to {:.4}",
                    perf.score_range.0, perf.score_range.1
                );
                println!(
                    "  🎲 Ranking Stability: {:.1}%",
                    perf.ranking_stability * 100.0
                );
            }
            Err(e) => {
                println!("\n❌ {} Strategy failed: {}", strategy_name, e);
            }
        }
    }

    println!();
    Ok(())
}

fn demo_data_characterization() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Demo 4: Detailed Data Characterization Analysis");
    println!("--------------------------------------------------");

    let complex_data = create_complex_mixed_data(80, 15)?;
    println!(
        "📊 Complex mixed data: {} samples, {} features",
        complex_data.shape().0,
        complex_data.shape().1
    );

    let selector = model_selection_presets::basic_selector();
    let selector = add_custom_candidates(selector);

    let result = selector.select_best(&complex_data)?;
    let chars = &result.data_characteristics;

    println!("\n📋 Comprehensive Data Analysis:");
    println!("==============================");

    // Basic characteristics
    println!("📏 Dimensions & Scale:");
    println!("  Samples: {}", chars.n_samples);
    println!("  Features: {}", chars.n_features);
    println!("  Sample/Feature Ratio: {:.2}", chars.sample_feature_ratio);

    // Distribution characteristics
    println!("\n📊 Distribution Properties:");
    println!(
        "  Normal Distribution: {}",
        chars.distribution.normality.is_normal
    );
    println!("  Heavy Tails: {}", chars.distribution.heavy_tails);
    println!("  Average Skewness: {:.3}", chars.distribution.skewness);
    println!("  Average Kurtosis: {:.3}", chars.distribution.kurtosis);
    println!(
        "  Outlier Fraction: {:.1}%",
        chars.distribution.outliers.outlier_fraction * 100.0
    );

    // Correlation structure
    println!("\n🔗 Correlation Structure:");
    println!(
        "  Block Structure: {}",
        chars.correlation_structure.has_block_structure
    );
    if let Some(n_blocks) = chars.correlation_structure.n_blocks {
        println!("  Number of Blocks: {}", n_blocks);
    }
    println!(
        "  Sparse Correlations: {}",
        chars.correlation_structure.is_sparse
    );
    println!(
        "  Factor Structure: {}",
        chars.correlation_structure.has_factor_structure
    );
    if let Some(n_factors) = chars.correlation_structure.n_factors {
        println!("  Estimated Factors: {}", n_factors);
    }

    // Missing data analysis
    println!("\n❓ Missing Data Analysis:");
    println!(
        "  Missing Fraction: {:.1}%",
        chars.missing_data.missing_fraction * 100.0
    );
    println!("  Pattern: {:?}", chars.missing_data.pattern);
    println!("  Mechanism: {:?}", chars.missing_data.mechanism);

    // Numerical properties
    println!("\n🔢 Numerical Properties:");
    println!("  Sparsity Level: {:.1}%", chars.sparsity_level * 100.0);
    println!("  Condition Number: {:.2e}", chars.condition_number);

    // Computational constraints
    println!("\n💻 Computational Profile:");
    println!(
        "  Can Parallelize: {}",
        chars.computational_constraints.can_parallelize
    );
    println!(
        "  Real-time Required: {}",
        chars.computational_constraints.real_time
    );
    if let Some(max_time) = chars.computational_constraints.max_time {
        println!("  Max Time Limit: {:.1}s", max_time);
    }

    println!();
    Ok(())
}

fn demo_performance_insights() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Demo 5: Performance Comparison and Insights");
    println!("----------------------------------------------");

    let benchmark_data = create_benchmark_data(100, 12)?;
    println!(
        "📊 Benchmark data: {} samples, {} features",
        benchmark_data.shape().0,
        benchmark_data.shape().1
    );

    // Use a comprehensive selector
    let selector = create_comprehensive_selector();
    let result = selector.select_best(&benchmark_data)?;

    println!("\n🏆 Best Estimator Analysis:");
    println!("===========================");
    println!("Selected: {}", result.best_estimator.name);
    println!("Score: {:.6}", result.best_estimator.score);
    println!(
        "Confidence: {:.1}%",
        result.best_estimator.confidence * 100.0
    );

    println!("\n📊 Detailed Performance Comparison:");
    println!("===================================");

    // Sort candidates by performance
    let mut sorted_candidates = result.candidate_results.clone();
    sorted_candidates.sort_by(|a, b| b.mean_score.partial_cmp(&a.mean_score).unwrap());

    println!("Rank | Estimator          | Score      | Std Err    | Time    | Stability");
    println!("-----|-------------------|------------|------------|---------|----------");
    for (rank, candidate) in sorted_candidates.iter().enumerate() {
        let stability = 1.0 / (1.0 + candidate.score_std);
        println!(
            "{:^4} | {:17} | {:10.6} | {:10.6} | {:7.2}s | {:8.1}%",
            rank + 1,
            candidate.name,
            candidate.mean_score,
            candidate.score_std,
            candidate.computation_time,
            stability * 100.0
        );
    }

    // Performance insights
    let perf = &result.performance_comparison;
    println!("\n💡 Performance Insights:");
    println!("========================");
    println!(
        "📊 Score Range: {:.6} to {:.6}",
        perf.score_range.0, perf.score_range.1
    );
    println!(
        "📈 Performance Spread: {:.6}",
        perf.score_range.1 - perf.score_range.0
    );
    println!(
        "🎲 Ranking Stability: {:.1}%",
        perf.ranking_stability * 100.0
    );

    // Selection metadata
    let meta = &result.selection_metadata;
    println!("\n⚙️ Selection Metadata:");
    println!("======================");
    println!("Strategy Used: {}", meta.strategy);
    println!("Candidates Evaluated: {}", meta.n_candidates);
    println!("Total Selection Time: {:.2}s", meta.total_time);
    println!(
        "CV Configuration: {} folds × {} repeats",
        meta.cv_config.n_folds, meta.cv_config.n_repeats
    );

    // Recommendations
    println!("\n💡 Recommendations:");
    println!("==================");
    if result.best_estimator.confidence > 0.8 {
        println!("✅ High confidence in selection - recommended for production use");
    } else if result.best_estimator.confidence > 0.6 {
        println!("⚠️  Moderate confidence - consider additional validation");
    } else {
        println!("❌ Low confidence - multiple estimators perform similarly");
    }

    if perf.ranking_stability > 0.8 {
        println!("✅ Stable ranking across CV folds - robust selection");
    } else {
        println!("⚠️  Unstable ranking - consider more data or different strategy");
    }

    let best_time = sorted_candidates
        .iter()
        .map(|c| c.computation_time)
        .fold(f64::INFINITY, f64::min);
    let selected_time = result
        .best_estimator
        .result
        .estimator_info
        .metrics
        .as_ref()
        .map(|m| m.computation_time_ms / 1000.0)
        .unwrap_or(0.0);

    if selected_time > best_time * 2.0 {
        println!("⚠️  Selected estimator is computationally expensive");
        println!("   Consider simpler alternatives for real-time applications");
    }

    println!();
    Ok(())
}

// Helper functions for creating different types of data

fn create_financial_return_data(
    n_samples: usize,
    n_assets: usize,
) -> SklResult<CovarianceDataFrame> {
    use scirs2_core::random::thread_rng;

    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_assets));

    for i in 0..n_samples {
        // Market factor with volatility clustering
        let market_vol = 0.01 + 0.005 * (rng.random::<f64>() * 2.0 - 1.0).abs();
        let market_return = rng.random::<f64>() * 2.0 - 1.0 * market_vol;

        for j in 0..n_assets {
            // Asset-specific beta and idiosyncratic risk
            let beta = 0.5 + (j as f64 / n_assets as f64) * 1.0;
            let idiosyncratic_vol = 0.008 + 0.002 * (rng.random::<f64>() * 2.0 - 1.0).abs();
            let idiosyncratic = rng.random::<f64>() * 2.0 - 1.0 * idiosyncratic_vol;

            data[[i, j]] = market_return * beta + idiosyncratic;
        }
    }

    let column_names = (0..n_assets).map(|i| format!("Asset_{}", i)).collect();
    CovarianceDataFrame::new(data, column_names, None)
}

fn create_high_dimensional_data(
    n_samples: usize,
    n_features: usize,
) -> SklResult<CovarianceDataFrame> {
    use scirs2_core::random::thread_rng;

    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_features));

    // Create data with low effective dimensionality
    let n_factors = (n_features as f64 / 5.0).ceil() as usize;
    let mut factors = Array2::zeros((n_samples, n_factors));

    // Generate factors
    for i in 0..n_samples {
        for f in 0..n_factors {
            factors[[i, f]] = rng.random::<f64>() * 2.0 - 1.0;
        }
    }

    // Generate data from factors
    for i in 0..n_samples {
        for j in 0..n_features {
            let factor_loading = (j % n_factors) as f64 / n_factors as f64 + 0.1;
            let factor_idx = j % n_factors;
            data[[i, j]] =
                factors[[i, factor_idx]] * factor_loading + rng.random::<f64>() * 2.0 - 1.0 * 0.1;
        }
    }

    let column_names = (0..n_features).map(|i| format!("Var_{}", i)).collect();
    CovarianceDataFrame::new(data, column_names, None)
}

fn create_well_conditioned_data(
    n_samples: usize,
    n_features: usize,
) -> SklResult<CovarianceDataFrame> {
    use scirs2_core::random::thread_rng;

    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_features));

    // Create well-conditioned data (low correlations)
    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
        }
    }

    let column_names = (0..n_features)
        .map(|i| format!("Independent_{}", i))
        .collect();
    CovarianceDataFrame::new(data, column_names, None)
}

fn create_sparse_correlation_data(
    n_samples: usize,
    n_features: usize,
) -> SklResult<CovarianceDataFrame> {
    use scirs2_core::random::thread_rng;

    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_features));

    // Create block structure with sparse correlations
    let block_size = 3;
    let n_blocks = n_features / block_size;

    for i in 0..n_samples {
        for block in 0..n_blocks {
            // Block factor
            let block_factor = rng.random::<f64>() * 2.0 - 1.0 * 0.7;

            for j in 0..block_size {
                let feature_idx = block * block_size + j;
                if feature_idx < n_features {
                    data[[i, feature_idx]] = block_factor + rng.random::<f64>() * 2.0 - 1.0 * 0.5;
                }
            }
        }

        // Independent features
        for j in (n_blocks * block_size)..n_features {
            data[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
        }
    }

    let column_names = (0..n_features).map(|i| format!("Feature_{}", i)).collect();
    CovarianceDataFrame::new(data, column_names, None)
}

fn create_noisy_financial_data(
    n_samples: usize,
    n_features: usize,
) -> SklResult<CovarianceDataFrame> {
    use scirs2_core::random::thread_rng;

    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        // Add occasional jumps (outliers)
        let is_jump_day = rng.random::<f64>() < 0.05; // 5% chance of jump
        let jump_magnitude = if is_jump_day { 3.0 } else { 1.0 };

        let market_return = rng.random::<f64>() * 2.0 - 1.0 * 0.015 * jump_magnitude;

        for j in 0..n_features {
            let beta = 0.3 + (j as f64 / n_features as f64) * 1.4;
            let noise = rng.random::<f64>() * 2.0 - 1.0 * 0.02;

            data[[i, j]] = market_return * beta + noise;
        }
    }

    let column_names = (0..n_features).map(|i| format!("Stock_{}", i)).collect();
    CovarianceDataFrame::new(data, column_names, None)
}

fn create_complex_mixed_data(
    n_samples: usize,
    n_features: usize,
) -> SklResult<CovarianceDataFrame> {
    use scirs2_core::random::thread_rng;

    let mut rng = thread_rng();
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        // Mixed characteristics: some normal, some skewed, some heavy-tailed
        for j in 0..n_features {
            let value = match j % 3 {
                0 => rng.random::<f64>() * 2.0 - 1.0, // Normal
                1 => {
                    // Skewed distribution
                    let u = rng.random_range(0.0..1.0);
                    if u < 0.8 {
                        rng.random::<f64>() * 2.0 - 1.0 * 0.5
                    } else {
                        rng.random::<f64>() * 2.0 - 1.0 * 2.0 + 1.0
                    }
                }
                _ => {
                    // Heavy-tailed (t-distribution approximation)
                    let u = rng.random_range(0.0..1.0);
                    if u < 0.9 {
                        rng.random::<f64>() * 2.0 - 1.0
                    } else {
                        rng.random::<f64>() * 2.0 - 1.0 * 5.0
                    }
                }
            };
            data[[i, j]] = value;
        }
    }

    let column_names = (0..n_features).map(|i| format!("Mixed_{}", i)).collect();
    CovarianceDataFrame::new(data, column_names, None)
}

fn create_benchmark_data(n_samples: usize, n_features: usize) -> SklResult<CovarianceDataFrame> {
    // Create realistic benchmark data combining multiple characteristics
    let financial_data = create_financial_return_data(n_samples, n_features / 2)?;
    let _sparse_data = create_sparse_correlation_data(n_samples, n_features / 2)?;

    // Combine datasets (simplified - just use financial data for demo)
    Ok(financial_data)
}

// Helper function to add custom candidates to a selector
fn add_custom_candidates(mut selector: AutoCovarianceSelector<f64>) -> AutoCovarianceSelector<f64> {
    use sklears_covariance::{ComputationalComplexity, DataCharacteristics};

    // Add empirical covariance if not already present
    if !selector
        .candidates
        .iter()
        .any(|c| c.name == "EmpiricalCovariance")
    {
        selector = selector.add_candidate(
            "EmpiricalCovariance".to_string(),
            Box::new(|_chars| {
                let estimator = EmpiricalCovariance::new();
                Ok(Box::new(estimator) as Box<dyn DataFrameEstimator<f64>>)
            }),
            DataCharacteristics::default(),
            ComputationalComplexity::Quadratic,
        );
    }

    // Add Ledoit-Wolf with different shrinkage levels
    // TODO: Re-enable when LedoitWolf implements DataFrameEstimator trait
    // selector = selector.add_candidate(
    //     "LedoitWolf-Auto".to_string(),
    //     Box::new(|_chars| {
    //         let estimator = LedoitWolf::new(); // Auto shrinkage
    //         Ok(Box::new(estimator) as Box<dyn DataFrameEstimator<f64>>)
    //     }),
    //     DataCharacteristics::default(),
    //     ComputationalComplexity::Quadratic,
    // );

    // TODO: Re-enable when .shrinkage() method is implemented
    // selector = selector.add_candidate(
    //     "LedoitWolf-Conservative".to_string(),
    //     Box::new(|_chars| {
    //         let estimator = LedoitWolf::new().shrinkage(0.1); // Conservative shrinkage
    //         Ok(Box::new(estimator) as Box<dyn DataFrameEstimator<f64>>)
    //     }),
    //     DataCharacteristics::default(),
    //     ComputationalComplexity::Quadratic,
    // );

    selector
}

// Create a comprehensive selector with many candidates
fn create_comprehensive_selector() -> AutoCovarianceSelector<f64> {
    let selector = model_selection_presets::basic_selector();
    let selector = add_custom_candidates(selector);

    // Add more sophisticated selection strategy
    selector.selection_strategy(SelectionStrategy::MultiObjective {
        performance_weight: 0.6,
        complexity_weight: 0.25,
        stability_weight: 0.15,
    })
}
