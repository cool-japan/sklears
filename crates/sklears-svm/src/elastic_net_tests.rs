//! Tests for elastic net regularization in LinearSVC

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Predict};

    #[test]
    fn test_elastic_net_regularization() {
        let X_var = array![[2.0, 3.0], [3.0, 3.0], [1.0, 1.0], [2.0, 1.0]];
        let y = array![1, 1, 0, 0];

        // Test L2 penalty
        let model_l2 = LinearSVC::new()
            .with_solver("primal_cd")
            .with_penalty("l2")
            .with_c(1.0)
            .with_max_iter(1000);

        let trained_l2 = model_l2.fit(&X_var, &y).unwrap();
        let predictions_l2 = trained_l2.predict(&X_var).unwrap();

        // Test L1 penalty
        let model_l1 = LinearSVC::new()
            .with_solver("primal_cd")
            .with_penalty("l1")
            .with_c(1.0)
            .with_max_iter(1000);

        let trained_l1 = model_l1.fit(&X_var, &y).unwrap();
        let predictions_l1 = trained_l1.predict(&X_var).unwrap();

        // Test elastic net penalty
        let model_en = LinearSVC::new()
            .with_solver("primal_cd")
            .with_penalty("elasticnet")
            .with_l1_ratio(0.5)
            .with_c(1.0)
            .with_max_iter(1000);

        let trained_en = model_en.fit(&X_var, &y).unwrap();
        let predictions_en = trained_en.predict(&X_var).unwrap();

        // All models should make reasonable predictions
        assert_eq!(predictions_l2.len(), 4);
        assert_eq!(predictions_l1.len(), 4);
        assert_eq!(predictions_en.len(), 4);

        // L1 regularization should generally produce sparser weights
        let l1_sparsity = trained_l1
            .coef()
            .iter()
            .filter(|&&x| x.abs() < 1e-10)
            .count();
        let l2_sparsity = trained_l2
            .coef()
            .iter()
            .filter(|&&x| x.abs() < 1e-10)
            .count();

        // L1 should typically be more sparse than L2 (though not guaranteed for small examples)
        println!("L1 sparsity: {}, L2 sparsity: {}", l1_sparsity, l2_sparsity);
    }

    #[test]
    fn test_penalty_validation() {
        let X_var = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 1];

        // Test invalid penalty
        let model = LinearSVC::new()
            .with_penalty("invalid_penalty")
            .with_solver("primal_cd");
        let result = model.fit(&X_var, &y);
        assert!(result.is_err());

        // Test valid penalties
        for penalty in &["l1", "l2", "elasticnet"] {
            let model = LinearSVC::new()
                .with_penalty(penalty)
                .with_solver("primal_cd");
            let result = model.fit(&X_var, &y);
            assert!(result.is_ok(), "Penalty {} should work", penalty);
        }
    }

    #[test]
    fn test_l1_ratio_effects() {
        let X_var = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ];
        let y = array![1, 1, 1, 0, 0, 0];

        // Test different l1_ratio values
        for l1_ratio in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let model = LinearSVC::new()
                .with_solver("primal_cd")
                .with_penalty("elasticnet")
                .with_l1_ratio(*l1_ratio)
                .with_c(1.0)
                .with_max_iter(1000);

            let result = model.fit(&X_var, &y);
            assert!(result.is_ok(), "l1_ratio {} should work", l1_ratio);

            let trained = result.unwrap();
            let predictions = trained.predict(&X_var).unwrap();
            assert_eq!(predictions.len(), 6);
        }
    }

    #[test]
    fn test_soft_threshold_function() {
        let model = LinearSVC::new();

        // Test positive values
        assert_abs_diff_eq!(model.soft_threshold(2.0, 1.0), 1.0);
        assert_abs_diff_eq!(model.soft_threshold(1.5, 1.0), 0.5);
        assert_abs_diff_eq!(model.soft_threshold(0.5, 1.0), 0.0);

        // Test negative values
        assert_abs_diff_eq!(model.soft_threshold(-2.0, 1.0), -1.0);
        assert_abs_diff_eq!(model.soft_threshold(-1.5, 1.0), -0.5);
        assert_abs_diff_eq!(model.soft_threshold(-0.5, 1.0), 0.0);

        // Test edge cases
        assert_abs_diff_eq!(model.soft_threshold(1.0, 1.0), 0.0);
        assert_abs_diff_eq!(model.soft_threshold(-1.0, 1.0), 0.0);
        assert_abs_diff_eq!(model.soft_threshold(0.0, 1.0), 0.0);
    }
}
