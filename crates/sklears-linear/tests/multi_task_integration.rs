//! Integration tests for multi-task linear models

#[cfg(any(feature = "multi-task-lasso", feature = "multi-task-elastic-net"))]
use scirs2_autograd::ndarray::{array, s, Array2};

#[cfg(any(feature = "multi-task-lasso", feature = "multi-task-elastic-net"))]
use sklears_core::traits::{Fit, Predict};

#[cfg(feature = "multi-task-elastic-net")]
use sklears_linear::MultiTaskElasticNet;
#[cfg(feature = "multi-task-lasso")]
use sklears_linear::MultiTaskLasso;

#[test]
#[cfg(feature = "multi-task-lasso")]
fn test_multi_task_lasso_integration() {
    // Create synthetic multi-output data
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],];

    let y = array![
        [3.0, 4.0],
        [5.0, 7.0],
        [7.0, 10.0],
        [9.0, 13.0],
        [11.0, 16.0],
    ];

    // Fit the model
    let model = MultiTaskLasso::new().alpha(0.1).fit_intercept(true);

    let fitted = model.fit(&x, &y).unwrap();

    // Make predictions
    let x_test = array![[6.0, 7.0], [7.0, 8.0]];
    let predictions = fitted.predict(&x_test).unwrap();

    // Check predictions shape
    assert_eq!(predictions.shape(), &[2, 2]);

    // Check coefficients shape
    assert_eq!(fitted.coef().unwrap().shape(), &[2, 2]);
    assert_eq!(fitted.intercept().unwrap().len(), 2);
}

#[test]
#[cfg(feature = "multi-task-elastic-net")]
fn test_multi_task_elastic_net_integration() {
    // Create data with more features than samples to test regularization
    let x = array![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ];

    let y = array![
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0],
    ];

    // Fit with different l1_ratios
    let models = vec![
        MultiTaskElasticNet::new().alpha(0.5).l1_ratio(0.0), // Ridge
        MultiTaskElasticNet::new().alpha(0.5).l1_ratio(0.5), // Elastic Net
        MultiTaskElasticNet::new().alpha(0.5).l1_ratio(1.0), // Lasso
    ];

    for (i, model) in models.into_iter().enumerate() {
        let fitted = model.fit(&x, &y).unwrap();
        let coef = fitted.coef().unwrap();

        // Count non-zero coefficients
        let non_zero_count = coef.iter().filter(|&&v| v.abs() > 1e-6).count();

        println!("Model {} non-zero coefficients: {}", i, non_zero_count);

        // Make predictions
        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[4, 3]);
    }
}

#[test]
#[cfg(feature = "multi-task-lasso")]
fn test_multi_task_models_error_handling() {
    let x = array![[1.0], [2.0]];
    let y = array![[1.0, 2.0], [3.0, 4.0]];

    // Test with mismatched dimensions
    let x_wrong = array![[1.0], [2.0], [3.0]];
    let model = MultiTaskLasso::new();

    let result = model.fit(&x_wrong, &y);
    assert!(result.is_err());

    // Test prediction with wrong number of features
    let fitted = MultiTaskLasso::new().fit(&x, &y).unwrap();
    let x_test_wrong = array![[1.0, 2.0]];

    let result = fitted.predict(&x_test_wrong);
    assert!(result.is_err());
}

#[test]
#[cfg(feature = "multi-task-lasso")]
fn test_multi_task_cross_validation_compatibility() {
    // This test ensures multi-task models work with cross-validation
    let n_samples = 20;
    let n_features = 5;
    let n_tasks = 3;

    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array2::zeros((n_samples, n_tasks));

    // Generate simple linear data
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = ((i + j) as f64) / 10.0;
        }
        for t in 0..n_tasks {
            y[[i, t]] = x.row(i).sum() * (t + 1) as f64;
        }
    }

    // Test different alpha values
    let alphas = vec![0.01, 0.1, 1.0];
    let mut best_alpha = 0.0;
    let mut best_score = f64::INFINITY;

    for &alpha in &alphas {
        let model = MultiTaskLasso::new().alpha(alpha);

        // Simple train/test split
        let split_idx = 15;
        let x_train = x.slice(s![..split_idx, ..]).to_owned();
        let y_train = y.slice(s![..split_idx, ..]).to_owned();
        let x_test = x.slice(s![split_idx.., ..]).to_owned();
        let y_test = y.slice(s![split_idx.., ..]).to_owned();

        let fitted = model.fit(&x_train, &y_train).unwrap();
        let predictions = fitted.predict(&x_test).unwrap();

        // Compute MSE
        let mse = (&predictions - &y_test).mapv(|v| v.powi(2)).mean().unwrap();

        if mse < best_score {
            best_score = mse;
            best_alpha = alpha;
        }
    }

    println!("Best alpha: {}, MSE: {}", best_alpha, best_score);
    assert!(best_score < 10.0); // Reasonable error for this simple problem
}
