//! Demonstration of the shuffle-based cross-validation implementations
//! extracted into the shuffle_cv module.

use scirs2_core::ndarray::Array1;
use sklears_model_selection::{
    BootstrapCV, CrossValidator, MonteCarloCV, ShuffleSplit, StratifiedShuffleSplit,
};

fn main() {
    println!("=== Shuffle CV Module Demo ===\n");

    // Test ShuffleSplit
    println!("1. ShuffleSplit Example:");
    let shuffle_split = ShuffleSplit::new(3)
        .test_size(0.2)
        .train_size(0.6)
        .random_state(42);

    let splits = shuffle_split.split(100, None);
    println!("   Generated {} splits", splits.len());
    for (i, (train, test)) in splits.iter().enumerate() {
        println!(
            "   Split {}: train_size={}, test_size={}",
            i + 1,
            train.len(),
            test.len()
        );
    }

    // Test StratifiedShuffleSplit
    println!("\n2. StratifiedShuffleSplit Example:");
    let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2]);
    let stratified_shuffle = StratifiedShuffleSplit::new(2)
        .test_size(0.3)
        .random_state(42);

    let splits = stratified_shuffle.split(10, Some(&y));
    println!("   Generated {} splits", splits.len());
    for (i, (train, test)) in splits.iter().enumerate() {
        println!(
            "   Split {}: train_size={}, test_size={}",
            i + 1,
            train.len(),
            test.len()
        );
    }

    // Test BootstrapCV
    println!("\n3. BootstrapCV Example:");
    let bootstrap = BootstrapCV::new(3).random_state(42);

    let splits = bootstrap.split(50, None);
    println!("   Generated {} splits", splits.len());
    for (i, (train, test)) in splits.iter().enumerate() {
        println!(
            "   Split {}: train_size={}, test_size={}",
            i + 1,
            train.len(),
            test.len()
        );
    }

    // Test MonteCarloCV
    println!("\n4. MonteCarloCV Example:");
    let monte_carlo = MonteCarloCV::new(4, 0.25).random_state(42);

    let splits = monte_carlo.split(80, None);
    println!("   Generated {} splits", splits.len());
    for (i, (train, test)) in splits.iter().enumerate() {
        println!(
            "   Split {}: train_size={}, test_size={}",
            i + 1,
            train.len(),
            test.len()
        );
    }

    println!("\n=== Demo completed successfully! ===");
}
