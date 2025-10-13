//! Simple Financial Data Imputation Example
//!
//! This example demonstrates basic financial data imputation using the
//! FinancialTimeSeriesImputer from the sklears-impute crate.
//!
//! ```bash
//! # Run this example
//! cargo run --example simple_financial_imputation
//! ```

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{thread_rng, Rng};
use sklears_impute::domain_specific::finance::FinancialTimeSeriesImputer;
use std::time::SystemTime;

/// Simple financial dataset structure
#[derive(Debug, Clone)]
pub struct SimpleFinancialData {
    /// Price data with missing values
    pub prices: Array2<f64>,
    /// Asset names
    pub asset_names: Vec<String>,
    /// Timestamps
    pub timestamps: Vec<SystemTime>,
}

/// Create synthetic financial data with missing values
fn create_sample_financial_data() -> SimpleFinancialData {
    let n_assets = 5;
    let n_periods = 100;
    let mut rng = thread_rng();

    // Generate synthetic price data
    let mut prices = Array2::zeros((n_periods, n_assets));

    for asset in 0..n_assets {
        let mut price = 100.0; // Starting price
        for time in 0..n_periods {
            // Random walk
            let return_rate = rng.gen_range(-0.02..0.02);
            price *= 1.0 + return_rate;
            prices[[time, asset]] = price;
        }
    }

    // Introduce missing values (5% randomly)
    for time in 0..n_periods {
        for asset in 0..n_assets {
            if rng.gen_range(0.0..1.0) < 0.05 {
                prices[[time, asset]] = f64::NAN;
            }
        }
    }

    // Create asset names
    let asset_names: Vec<String> = (0..n_assets).map(|i| format!("ASSET_{}", i)).collect();

    // Create timestamps (daily data)
    let base_time = SystemTime::now();
    let timestamps: Vec<SystemTime> = (0..n_periods)
        .map(|i| base_time - std::time::Duration::from_secs(i as u64 * 86400))
        .collect();

    SimpleFinancialData {
        prices,
        asset_names,
        timestamps,
    }
}

/// Analyze missing data patterns
fn analyze_missing_data(data: &SimpleFinancialData) {
    let total_values = data.prices.len();
    let missing_count = data.prices.iter().filter(|&&x| x.is_nan()).count();
    let missing_percentage = missing_count as f64 / total_values as f64 * 100.0;

    println!("📊 Missing Data Analysis:");
    println!("  - Total values: {}", total_values);
    println!("  - Missing values: {}", missing_count);
    println!("  - Missing percentage: {:.2}%", missing_percentage);

    // Analyze by asset
    for (i, asset_name) in data.asset_names.iter().enumerate() {
        let column = data.prices.column(i);
        let asset_missing = column.iter().filter(|&&x| x.is_nan()).count();
        let asset_percentage = asset_missing as f64 / column.len() as f64 * 100.0;

        if asset_percentage > 0.0 {
            println!("  - {}: {:.1}% missing", asset_name, asset_percentage);
        }
    }
}

/// Demonstrate financial imputation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏦 Simple Financial Data Imputation Example");
    println!("============================================");

    // Create sample data
    println!("\n📊 Creating synthetic financial dataset...");
    let financial_data = create_sample_financial_data();

    println!("  - Assets: {}", financial_data.asset_names.len());
    println!("  - Time periods: {}", financial_data.timestamps.len());
    println!("  - Data shape: {:?}", financial_data.prices.dim());

    // Analyze missing patterns
    println!("\n🔍 Analyzing missing data patterns:");
    analyze_missing_data(&financial_data);

    // Create and configure imputer
    println!("\n⚙️  Configuring financial time series imputer:");
    let imputer = FinancialTimeSeriesImputer::new()
        .with_model_type("garch")
        .with_volatility_window(20);

    println!("  - Model type: GARCH");
    println!("  - Volatility window: 20 periods");

    // Perform imputation
    println!("\n🔧 Performing imputation...");
    let start_time = std::time::Instant::now();

    match imputer.fit_transform(&financial_data.prices.view()) {
        Ok(imputed_prices) => {
            let duration = start_time.elapsed();
            println!("  ✅ Imputation completed in {:.2}ms", duration.as_millis());

            // Check results
            let remaining_missing = imputed_prices.iter().filter(|&&x| x.is_nan()).count();
            let completeness =
                (1.0 - remaining_missing as f64 / imputed_prices.len() as f64) * 100.0;

            println!("\n📈 Imputation Results:");
            println!("  - Completeness: {:.2}%", completeness);
            println!("  - Remaining missing values: {}", remaining_missing);

            // Show some sample results
            println!("\n📋 Sample Imputed Values (first 5 periods, first 3 assets):");
            println!("    {:>8} {:>8} {:>8}", "ASSET_0", "ASSET_1", "ASSET_2");
            for t in 0..5.min(imputed_prices.nrows()) {
                print!("    ");
                for a in 0..3.min(imputed_prices.ncols()) {
                    let original = financial_data.prices[[t, a]];
                    let imputed = imputed_prices[[t, a]];

                    if original.is_nan() {
                        print!("{:>8.2}*", imputed); // * indicates imputed value
                    } else {
                        print!("{:>8.2} ", imputed);
                    }
                }
                println!();
            }
            println!("  (* indicates imputed values)");
        }
        Err(e) => {
            println!("  ❌ Imputation failed: {:?}", e);
            return Err(Box::new(e));
        }
    }

    println!("\n✅ Financial imputation example completed successfully!");
    Ok(())
}
