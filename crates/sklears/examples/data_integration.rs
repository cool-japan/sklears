//! Example showing integration with Polars DataFrames
//!
//! This demonstrates the seamless data flow between Polars,
//! numrs2 arrays, and sklears models.

use polars::prelude::{col, lit, DataFrame as PolarsDataFrame, IntoLazy, NamedFrom, Series};
use sklears::prelude::*;

fn main() -> Result<()> {
    println!("sklears Data Integration Example\n");

    // Example of the intended data flow pattern
    data_flow_example()?;

    Ok(())
}

fn data_flow_example() -> Result<()> {
    // Create a sample DataFrame
    let df = PolarsDataFrame::new(vec![
        Series::new("feature1".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0]).into(),
        Series::new("feature2".into(), vec![2.0, 4.0, 6.0, 8.0, 10.0]).into(),
        Series::new("feature3".into(), vec![1.5, 3.0, 4.5, 6.0, 7.5]).into(),
        Series::new("target".into(), vec![10.0, 20.0, 30.0, 40.0, 50.0]).into(),
    ])
    .expect("Failed to create DataFrame");

    println!("Created DataFrame with shape: {:?}", df.shape());
    println!("{}", df);

    // This would work with proper integration:
    /*
    // Convert DataFrame columns to numrs2 arrays
    let features = df.to_numrs_array(&["feature1", "feature2", "feature3"])?;
    let targets = df.to_numrs_array(&["target"])?;

    // Use scirs2 for preprocessing
    let scaled_features = scirs2::preprocessing::standardize(&features)?;

    // Train sklears model
    let model = LogisticRegression::new()
        .penalty(Penalty::L2(0.1))
        .fit(&scaled_features, &targets)?;

    // Make predictions and add back to DataFrame
    let predictions = model.predict(&scaled_features)?;
    let mut result_df = df.clone();
    result_df.with_column(Series::new("predictions", predictions.to_vec()))?;

    println!("\nDataFrame with predictions:");
    println!("{}", result_df);
    */

    // Example of filtering and selecting data
    let filtered_df = df
        .clone()
        .lazy()
        .filter(col("target").gt(lit(20)))
        .select(&[col("feature1"), col("feature2"), col("target")])
        .collect()
        .expect("Failed to filter DataFrame");

    println!("\nFiltered DataFrame:");
    println!("{}", filtered_df);

    Ok(())
}

// Example helper function for DataFrame to array conversion
// This would be implemented in the integration layer
fn dataframe_to_array_example(
    df: &PolarsDataFrame,
    columns: &[&str],
) -> std::result::Result<Array2<f64>, Box<dyn std::error::Error>> {
    let n_rows = df.height();
    let n_cols = columns.len();
    let mut data = Vec::with_capacity(n_rows * n_cols);

    for col_name in columns {
        let series = df.column(col_name)?;
        let values: Vec<f64> = series
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();
        data.extend(values);
    }

    // Note: This is transposed, would need to be fixed in real implementation
    Ok(Array2::from_shape_vec((n_cols, n_rows), data)?
        .t()
        .to_owned())
}
