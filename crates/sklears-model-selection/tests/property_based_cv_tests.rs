use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_model_selection::{CrossValidator, KFold};

/// Property-based tests for cross-validation splitters
///
/// These tests verify fundamental properties that should hold for all CV splitters:
/// 1. Train and test sets are disjoint for each fold
/// 2. All indices are used exactly once when combining all folds
/// 3. Each fold produces valid train/test splits
/// 4. Fold sizes are reasonably balanced (for standard CV methods)

#[allow(non_snake_case)]
#[cfg(test)]
mod property_tests {
    use super::*;

    /// Generate test data with variable size
    fn arb_dataset() -> impl Strategy<Value = (Array2<f64>, Array1<i32>)> {
        (10..100usize).prop_flat_map(|n_samples| {
            let n_features = 4;
            (
                proptest::collection::vec(proptest::num::f64::ANY, n_samples * n_features)
                    .prop_map(move |values| {
                        Array2::from_shape_vec((n_samples, n_features), values).unwrap()
                    }),
                proptest::collection::vec(0i32..3, n_samples).prop_map(Array1::from),
            )
        })
    }

    proptest! {
        #[test]
        fn test_kfold_disjoint_train_test_sets(
            data in arb_dataset(),
            n_folds in 2..=10usize
        ) {
            let (X, y) = data;
            let n_samples = X.nrows();

            // Skip if not enough samples for the number of folds
            if n_samples < n_folds {
                return Ok(());
            }

            let splitter = KFold::new(n_folds);
            let splits = splitter.split(n_samples, Some(&y));

            // Property 1: Train and test sets should be disjoint
            for (train_indices, test_indices) in &splits {
                let train_set: std::collections::HashSet<usize> =
                    train_indices.iter().copied().collect();
                let test_set: std::collections::HashSet<usize> =
                    test_indices.iter().copied().collect();

                prop_assert!(
                    train_set.is_disjoint(&test_set),
                    "Train and test sets should be disjoint"
                );
            }
        }

        #[test]
        fn test_kfold_complete_coverage(
            data in arb_dataset(),
            n_folds in 2..=10usize
        ) {
            let (X, y) = data;
            let n_samples = X.nrows();

            if n_samples < n_folds {
                return Ok(());
            }

            let splitter = KFold::new(n_folds);
            let splits = splitter.split(n_samples, Some(&y));

            // Property 2: All indices should be covered exactly once as test data
            let mut all_test_indices = std::collections::HashSet::new();
            for (_, test_indices) in &splits {
                for &idx in test_indices {
                    prop_assert!(
                        all_test_indices.insert(idx),
                        "Index {} appears in multiple test sets", idx
                    );
                }
            }

            prop_assert_eq!(
                all_test_indices.len(), n_samples,
                "Not all samples are covered in test sets"
            );

            // Verify all indices are in range
            for &idx in &all_test_indices {
                prop_assert!(
                    idx < n_samples,
                    "Index {} is out of range for {} samples", idx, n_samples
                );
            }
        }

        #[test]
        fn test_kfold_train_test_union_equals_all(
            data in arb_dataset(),
            n_folds in 2..=10usize
        ) {
            let (X, y) = data;
            let n_samples = X.nrows();

            if n_samples < n_folds {
                return Ok(());
            }

            let splitter = KFold::new(n_folds);
            let splits = splitter.split(n_samples, Some(&y));

            // Property 3: For each fold, train + test should equal all indices
            for (train_indices, test_indices) in &splits {
                let mut all_indices = std::collections::HashSet::new();

                for &idx in train_indices {
                    all_indices.insert(idx);
                }
                for &idx in test_indices {
                    all_indices.insert(idx);
                }

                prop_assert_eq!(
                    all_indices.len(), n_samples,
                    "Train + test indices don't cover all samples"
                );
            }
        }

        #[test]
        fn test_kfold_balanced_fold_sizes(
            data in arb_dataset(),
            n_folds in 2..=10usize
        ) {
            let (X, y) = data;
            let n_samples = X.nrows();

            if n_samples < n_folds {
                return Ok(());
            }

            let splitter = KFold::new(n_folds);
            let splits = splitter.split(n_samples, Some(&y));

            prop_assert_eq!(splits.len(), n_folds, "Number of splits should equal n_folds");

            // Property 4: Test set sizes should be reasonably balanced
            let expected_test_size = n_samples / n_folds;
            let max_allowed_diff = if n_samples % n_folds == 0 { 0 } else { 1 };

            for (i, (train_indices, test_indices)) in splits.iter().enumerate() {
                let test_size = test_indices.len();
                let diff = test_size.abs_diff(expected_test_size);

                prop_assert!(
                    diff <= max_allowed_diff,
                    "Fold {} test size {} differs too much from expected {}",
                    i, test_size, expected_test_size
                );

                // Train size should be n_samples - test_size
                prop_assert_eq!(
                    train_indices.len(),
                    n_samples - test_size,
                    "Train size should be n_samples - test_size"
                );
            }
        }

        #[test]
        fn test_kfold_valid_indices(
            data in arb_dataset(),
            n_folds in 2..=10usize
        ) {
            let (X, y) = data;
            let n_samples = X.nrows();

            if n_samples < n_folds {
                return Ok(());
            }

            let splitter = KFold::new(n_folds);
            let splits = splitter.split(n_samples, Some(&y));

            // Property 5: All indices should be valid (< n_samples)
            for (train_indices, test_indices) in &splits {
                for &idx in train_indices {
                    prop_assert!(
                        idx < n_samples,
                        "Train index {} is out of range for {} samples", idx, n_samples
                    );
                }
                for &idx in test_indices {
                    prop_assert!(
                        idx < n_samples,
                        "Test index {} is out of range for {} samples", idx, n_samples
                    );
                }
            }
        }
    }

    #[test]
    fn test_property_based_framework_works() {
        // Simple test to ensure proptest integration works
        proptest!(|(x in 0..100i32)| {
            prop_assert!(x >= 0);
            prop_assert!(x < 100);
        });
    }
}
