// This example is currently disabled as Pipeline and ScalarTransformer are not
// yet available in the main sklears crate API. They exist in sklears-compose but
// are not re-exported. This example will be re-enabled once the pipeline API is stable.

#[cfg(any())] // Never compile - disabled until Pipeline API is available
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::ndarray::array;
    use sklears::prelude::*;
    println!("🚀 Sklears Pipeline Demo");

    // Create sample data
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

    println!("Original data:");
    println!("{}", data);

    // Create a pipeline with multiple transformers
    let mut pipeline: Pipeline<ScalarTransformer> = Pipeline::new()
        .add_step(ScalarTransformer::new(2.0)) // Scale by 2
        .add_step(ScalarTransformer::new(0.5)); // Scale by 0.5 (net effect: no change)

    // Fit and transform the data
    let transformed_data = pipeline.fit_transform(data.clone(), None)?;

    println!("\nTransformed data (should be same as original):");
    println!("{}", transformed_data);

    // Test transform-only functionality
    let test_data = array![[10.0, 11.0, 12.0],];

    let test_transformed = pipeline.transform(&test_data)?;
    println!("\nTest data transformed:");
    println!("{}", test_transformed);

    // Note: Functional pipeline composition is not yet available in the current API
    // This feature is planned for future releases

    // Demonstrate manual composition as an alternative
    let mut pipeline2 = Pipeline::new()
        .add_step(ScalarTransformer::new(3.0))
        .add_step(ScalarTransformer::new(1.0 / 3.0));

    let manual_result = pipeline2.fit_transform(data.clone(), None)?;

    println!("\nManual composition pipeline result:");
    println!("{}", manual_result);

    println!("\n✅ Pipeline demo completed successfully!");

    Ok(())
}

// Placeholder main function
#[cfg(not(any()))] // Always compile
fn main() {
    println!("⚠️  Pipeline demo is currently disabled");
    println!(
        "Pipeline and ScalarTransformer APIs are not yet available in the main sklears crate."
    );
    println!("These features exist in sklears-compose but are not yet stabilized for public use.");
    println!("\nThis example will be re-enabled in a future release.");
}
