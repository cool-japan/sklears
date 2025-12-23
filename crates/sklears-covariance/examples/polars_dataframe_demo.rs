//! Polars DataFrame Integration Demo
//!
//! This example demonstrates the seamless integration between Polars DataFrames
//! and sklears-covariance estimators, showing how to work with real-world data
//! formats and get rich, contextual results.

use scirs2_core::ndarray::array;
use scirs2_core::random::{thread_rng, Rng};
use sklears_covariance::{
    polars_utils, CovarianceDataFrame, DataFrameEstimator, EmpiricalCovariance,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Polars DataFrame Integration Demo for Covariance Estimation");
    println!("==============================================================\n");

    // Demo 1: Basic DataFrame creation and validation
    demo_basic_dataframe_creation()?;

    // Demo 2: Working with realistic financial data
    demo_financial_data_analysis()?;

    // Demo 3: Handling missing data
    demo_missing_data_handling()?;

    // Demo 4: Multiple estimator comparison with DataFrames
    demo_estimator_comparison()?;

    // Demo 5: Advanced DataFrame utilities
    demo_advanced_utilities()?;

    println!("✅ All demos completed successfully!");
    Ok(())
}

fn demo_basic_dataframe_creation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 1: Basic DataFrame Creation and Validation");
    println!("--------------------------------------------------");

    // Create sample financial return data
    let returns_data = vec![
        vec![0.02f64, 0.015f64, -0.01f64],   // Day 1: Stock A, B, C returns
        vec![0.01f64, 0.025f64, 0.005f64],   // Day 2
        vec![-0.005f64, 0.02f64, 0.01f64],   // Day 3
        vec![0.03f64, -0.01f64, 0.015f64],   // Day 4
        vec![0.008f64, 0.018f64, -0.002f64], // Day 5
    ];

    let column_names = vec![
        "AAPL_returns".to_string(),
        "GOOGL_returns".to_string(),
        "MSFT_returns".to_string(),
    ];

    // Convert to expected format for from_slices
    let returns_refs: Vec<&[f64]> = returns_data.iter().map(|v| v.as_slice()).collect();
    let df = polars_utils::from_slices(&returns_refs, column_names)?;

    println!("📈 Created DataFrame with shape: {:?}", df.shape());
    println!("📋 Columns: {:?}", df.column_names());

    // Validate the data
    df.validate()?;
    println!("✅ Data validation passed");

    // Get summary statistics
    let description = df.describe();
    println!("\n📊 DataFrame Summary:");
    println!("  Shape: {:?}", description.shape);
    for (col_name, stats) in &description.metadata.column_stats {
        println!(
            "  {}: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            col_name, stats.mean, stats.std_dev, stats.min, stats.max
        );
    }

    // Fit empirical covariance using DataFrame interface
    let estimator = EmpiricalCovariance::new();
    let result = estimator.fit_dataframe(&df)?;

    println!("\n🔍 Covariance Matrix:");
    for (i, row_name) in result.feature_names.iter().enumerate() {
        print!("  {:15}", row_name);
        for j in 0..result.covariance.ncols() {
            print!(" {:8.4}", result.covariance[[i, j]]);
        }
        println!();
    }

    println!(
        "⏱️  Computation time: {:.2}ms",
        result
            .estimator_info
            .metrics
            .as_ref()
            .unwrap()
            .computation_time_ms
    );

    println!();
    Ok(())
}

fn demo_financial_data_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("💹 Demo 2: Financial Data Analysis");
    println!("-----------------------------------");

    // Create more realistic financial data with different volatilities
    let mut financial_data = Vec::new();
    let mut rng = thread_rng();

    for _ in 0..100 {
        // Generate correlated returns for a portfolio using direct random generation
        let market_shock: f64 = (rng.random::<f64>() * 2.0 - 1.0) * 0.01;

        // Tech stock - high volatility, correlated with market
        let tech_return = market_shock * 1.5 + (rng.random::<f64>() * 2.0 - 1.0) * 0.02;

        // Utility stock - low volatility, less correlated
        let utility_return = market_shock * 0.5 + (rng.random::<f64>() * 2.0 - 1.0) * 0.008;

        // Bond - low volatility, slightly negatively correlated
        let bond_return = -market_shock * 0.2 + (rng.random::<f64>() * 2.0 - 1.0) * 0.005;

        // Commodity - moderate volatility, different correlation pattern
        let commodity_return = market_shock * 0.8 + (rng.random::<f64>() * 2.0 - 1.0) * 0.015;

        financial_data.push(vec![
            tech_return,
            utility_return,
            bond_return,
            commodity_return,
        ]);
    }

    let portfolio_columns = vec![
        "Technology".to_string(),
        "Utilities".to_string(),
        "Bonds".to_string(),
        "Commodities".to_string(),
    ];

    let portfolio_data: Vec<&[f64]> = financial_data.iter().map(|row| row.as_slice()).collect();
    let portfolio_df = polars_utils::from_slices(&portfolio_data, portfolio_columns)?;

    println!(
        "📊 Portfolio DataFrame: {} assets, {} observations",
        portfolio_df.shape().1,
        portfolio_df.shape().0
    );

    // Analyze with empirical estimator
    let empirical = EmpiricalCovariance::new();

    let emp_result = empirical.fit_dataframe(&portfolio_df)?;

    println!("\n📈 Empirical Covariance Matrix:");
    print_covariance_matrix(&emp_result);

    // Display condition number
    if let Some(emp_cond) = emp_result
        .estimator_info
        .metrics
        .as_ref()
        .and_then(|m| m.condition_number)
    {
        println!("\n🔢 Condition Number:");
        println!("  Empirical: {:.2e}", emp_cond);
    }

    // Calculate and display correlation matrix
    let emp_corr = emp_result.correlation()?;

    println!("\n📊 Empirical Correlation Matrix:");
    print_correlation_matrix(&emp_corr, &emp_result.feature_names);

    println!();
    Ok(())
}

fn demo_missing_data_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("❓ Demo 3: Missing Data Handling");
    println!("--------------------------------");

    // Create data with some missing values
    let complete_data = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ];

    let missing_mask = Some(array![
        [false, false, false],
        [true, false, false], // Missing value in first column
        [false, false, true], // Missing value in third column
        [false, false, false],
        [false, true, false] // Missing value in second column
    ]);

    let columns = vec!["X1".to_string(), "X2".to_string(), "X3".to_string()];
    let df_with_missing = CovarianceDataFrame::new(complete_data, columns, missing_mask)?;

    println!("📊 Data with missing values:");
    println!("  Shape: {:?}", df_with_missing.shape());
    println!("  Has missing: {}", df_with_missing.has_missing_values());

    let missing_ratios = df_with_missing.missing_ratios();
    for (col, ratio) in &missing_ratios {
        println!("  {}: {:.1}% missing", col, ratio * 100.0);
    }

    // Handle missing data by dropping rows
    println!("\n🧹 Dropping rows with missing values...");
    let clean_df = df_with_missing.drop_missing()?;
    println!("  New shape after dropping: {:?}", clean_df.shape());

    // Fit covariance on clean data
    let estimator = EmpiricalCovariance::new();
    let result = estimator.fit_dataframe(&clean_df)?;

    println!("\n📊 Covariance Matrix (after handling missing data):");
    print_covariance_matrix(&result);

    println!();
    Ok(())
}

fn demo_estimator_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Demo 4: Multiple Estimator Comparison");
    println!("----------------------------------------");

    // Create sample data for comparison
    let test_data = polars_utils::create_sample_data(50, 4)?;

    println!(
        "📊 Test data: {} samples, {} features",
        test_data.shape().0,
        test_data.shape().1
    );

    // Define estimators to compare
    let estimators: Vec<(&str, Box<dyn DataFrameEstimator<f64>>)> =
        vec![("Empirical", Box::new(EmpiricalCovariance::new()))];

    println!("\n🏆 Estimator Comparison Results:");
    println!("================================");

    let mut results = Vec::new();
    for (name, estimator) in estimators {
        let start_time = std::time::Instant::now();
        let result = estimator.fit_dataframe(&test_data)?;
        let elapsed = start_time.elapsed().as_millis();

        println!("\n📈 {}", name);
        println!("  ⏱️  Time: {}ms", elapsed);
        if let Some(metrics) = &result.estimator_info.metrics {
            if let Some(cond_num) = metrics.condition_number {
                println!("  🔢 Condition number: {:.2e}", cond_num);
            }
        }

        // Calculate Frobenius norm as a measure of matrix magnitude
        let frobenius_norm = result.covariance.iter().map(|&x| x * x).sum::<f64>().sqrt();
        println!("  📏 Frobenius norm: {:.4}", frobenius_norm);

        results.push((name, result));
    }

    // Compare estimators
    if results.len() >= 2 {
        println!("\n🔍 Comparison Summary:");
        let emp_result = &results[0].1;
        let lw_result = &results[1].1;

        // Calculate Frobenius norm of difference
        let diff_norm = emp_result
            .covariance
            .iter()
            .zip(lw_result.covariance.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        println!("  📏 Difference (Frobenius norm): {:.6}", diff_norm);
        println!(
            "  📊 Relative difference: {:.2}%",
            diff_norm
                / emp_result
                    .covariance
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f64>()
                    .sqrt()
                * 100.0
        );
    }

    println!();
    Ok(())
}

fn demo_advanced_utilities() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛠️  Demo 5: Advanced DataFrame Utilities");
    println!("----------------------------------------");

    // Create sample data
    let raw_data = vec![
        vec![100.0f64, 200.0f64, 50.0f64], // Different scales
        vec![110.0f64, 180.0f64, 52.0f64],
        vec![95.0f64, 220.0f64, 48.0f64],
        vec![105.0f64, 190.0f64, 51.0f64],
        vec![98.0f64, 210.0f64, 49.0f64],
    ];

    let columns = vec![
        "Price_A".to_string(),
        "Price_B".to_string(),
        "Price_C".to_string(),
    ];
    let raw_data_refs: Vec<&[f64]> = raw_data.iter().map(|v| v.as_slice()).collect();
    let original_df = polars_utils::from_slices(&raw_data_refs, columns)?;

    println!("📊 Original data (different scales):");
    let desc = original_df.describe();
    for (col_name, stats) in &desc.metadata.column_stats {
        println!(
            "  {}: mean={:.1}, std={:.1}",
            col_name, stats.mean, stats.std_dev
        );
    }

    // Standardize the data
    println!("\n📏 Standardizing data (zero mean, unit variance)...");
    let standardized_df = polars_utils::standardize(&original_df)?;

    println!("📊 Standardized data:");
    let std_desc = standardized_df.describe();
    for (col_name, stats) in &std_desc.metadata.column_stats {
        println!(
            "  {}: mean={:.3}, std={:.3}",
            col_name, stats.mean, stats.std_dev
        );
    }

    // Center the original data
    println!("\n🎯 Centering data (zero mean, original variance)...");
    let centered_df = polars_utils::center(&original_df)?;

    println!("📊 Centered data:");
    let cent_desc = centered_df.describe();
    for (col_name, stats) in &cent_desc.metadata.column_stats {
        println!(
            "  {}: mean={:.3}, std={:.1}",
            col_name, stats.mean, stats.std_dev
        );
    }

    // Fit covariance on different preprocessed versions
    let estimator = EmpiricalCovariance::new();

    let original_result = estimator.fit_dataframe(&original_df)?;
    let standardized_result = estimator.fit_dataframe(&standardized_df)?;
    let centered_result = estimator.fit_dataframe(&centered_df)?;

    println!("\n📈 Covariance Matrix Comparisons:");

    println!("\n🔸 Original data covariance:");
    print_matrix_summary(&original_result.covariance);

    println!("\n🔸 Standardized data covariance (= correlation matrix):");
    print_matrix_summary(&standardized_result.covariance);

    println!("\n🔸 Centered data covariance:");
    print_matrix_summary(&centered_result.covariance);

    // The standardized covariance should be the correlation matrix
    let correlation = original_result.correlation()?;
    println!("\n✅ Verification: Standardized covariance equals correlation matrix");
    let diff_norm = standardized_result
        .covariance
        .iter()
        .zip(correlation.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("  Max absolute difference: {:.2e}", diff_norm);

    println!();
    Ok(())
}

// Helper functions for pretty printing

fn print_covariance_matrix(result: &sklears_covariance::CovarianceResult<f64>) {
    for (i, row_name) in result.feature_names.iter().enumerate() {
        print!("  {:12}", row_name);
        for j in 0..result.covariance.ncols() {
            print!(" {:8.4}", result.covariance[[i, j]]);
        }
        println!();
    }
}

fn print_correlation_matrix(
    correlation: &scirs2_core::ndarray::Array2<f64>,
    feature_names: &[String],
) {
    for (i, row_name) in feature_names.iter().enumerate() {
        print!("  {:12}", row_name);
        for j in 0..correlation.ncols() {
            print!(" {:7.3}", correlation[[i, j]]);
        }
        println!();
    }
}

fn print_matrix_summary(matrix: &scirs2_core::ndarray::Array2<f64>) {
    let trace = (0..matrix.nrows()).map(|i| matrix[[i, i]]).sum::<f64>();
    let frobenius_norm = matrix.iter().map(|&x| x * x).sum::<f64>().sqrt();
    println!(
        "    Trace: {:.4}, Frobenius norm: {:.4}",
        trace, frobenius_norm
    );
}
