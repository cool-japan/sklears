//! Demo of the cross-validation iterators

use scirs2_core::ndarray::{array, Array1};
use sklears_model_selection::{
    CrossValidator, GroupShuffleSplit, LeaveOneGroupOut, LeavePGroupsOut, PredefinedSplit,
    RepeatedKFold, RepeatedStratifiedKFold,
};

fn main() {
    println!("Cross-Validation Iterators Demo");
    println!("==============================\n");

    // Test GroupShuffleSplit
    println!("1. GroupShuffleSplit:");
    println!("   - Generates random train/test splits respecting group constraints");
    let groups = array![0, 0, 1, 1, 2, 2, 3, 3];
    let cv = GroupShuffleSplit::new(2).test_size(0.25).random_state(42);
    let splits = cv.split_with_groups(8, &groups);
    println!("   Number of splits: {}", splits.len());
    for (i, (train, test)) in splits.iter().enumerate() {
        println!(
            "   Split {}: train size = {}, test size = {}",
            i,
            train.len(),
            test.len()
        );
    }
    println!();

    // Test LeaveOneGroupOut
    println!("2. LeaveOneGroupOut:");
    println!("   - Each split leaves out one unique group");
    let groups = array![0, 0, 1, 1, 2, 2];
    let cv = LeaveOneGroupOut::new();
    let splits = cv.split_with_groups(6, &groups);
    println!("   Number of splits: {}", splits.len());
    for (i, (train, test)) in splits.iter().enumerate() {
        println!(
            "   Split {}: train size = {}, test size = {}",
            i,
            train.len(),
            test.len()
        );
    }
    println!();

    // Test LeavePGroupsOut
    println!("3. LeavePGroupsOut (p=2):");
    println!("   - Each split leaves out P groups");
    let groups = array![0, 0, 1, 1, 2, 2, 3, 3];
    let cv = LeavePGroupsOut::new(2);
    let splits = cv.split_with_groups(8, &groups);
    println!("   Number of splits: {} (C(4,2) = 6)", splits.len());
    println!();

    // Test RepeatedKFold
    println!("4. RepeatedKFold (n_splits=3, n_repeats=2):");
    println!("   - Repeats K-Fold with different randomization");
    let cv = RepeatedKFold::new(3, 2).random_state(42);
    let splits = cv.split(9, None);
    println!(
        "   Number of splits: {} (3 folds × 2 repeats)",
        splits.len()
    );
    println!();

    // Test RepeatedStratifiedKFold
    println!("5. RepeatedStratifiedKFold (n_splits=3, n_repeats=2):");
    println!("   - Repeats Stratified K-Fold with different randomization");
    let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let cv = RepeatedStratifiedKFold::new(3, 2).random_state(42);
    let splits = cv.split(9, Some(&y));
    println!(
        "   Number of splits: {} (3 folds × 2 repeats)",
        splits.len()
    );

    // Check stratification
    println!("   Checking stratification in first repeat:");
    for (i, (train, test)) in splits.iter().take(3).enumerate() {
        let test_classes: Vec<i32> = test.iter().map(|&idx| y[idx]).collect();
        println!("   Split {}: test classes = {:?}", i, test_classes);
    }
    println!();

    // Test PredefinedSplit
    println!("6. PredefinedSplit:");
    println!("   - Uses user-defined split indices");
    let test_fold = array![-1, 0, 0, 1, 1, 2, 2, -1];
    println!(
        "   test_fold: {:?} (-1 means always in training set)",
        test_fold
    );
    let cv = PredefinedSplit::new(test_fold);
    println!("   Number of splits: {}", cv.n_splits());
    let splits = cv.split(8, None::<&Array1<i32>>);
    for (i, (train, test)) in splits.iter().enumerate() {
        println!(
            "   Split {}: train indices = {:?}, test indices = {:?}",
            i, train, test
        );
    }
}
