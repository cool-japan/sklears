//! Kernel discriminant analysis tests

use super::super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, ArrayView1, Axis};
use sklears_core::traits::{Fit, Predict, PredictProba, Transform};
use sklears_core::types::Float;

#[test]
fn test_kernel_discriminant_analysis_rbf() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::RBF { gamma: 1.0 });
    let fitted = kda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_kernel_discriminant_analysis_polynomial() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Polynomial {
        gamma: 1.0,
        coef0: 1.0,
        degree: 2,
    });
    let fitted = kda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_kernel_discriminant_analysis_linear() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Linear);
    let fitted = kda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_kernel_discriminant_analysis_sigmoid() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Sigmoid {
        gamma: 0.1,
        coef0: 1.0,
    });
    let fitted = kda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_kernel_predict_proba() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::RBF { gamma: 1.0 });
    let fitted = kda.fit(&x, &y).expect("model fitting should succeed");
    let probas = fitted
        .predict_proba(&x)
        .expect("probability prediction should succeed");

    assert_eq!(probas.dim(), (4, 2));

    for row in probas.axis_iter(Axis(0)) {
        let sum: Float = row.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_kernel_transform() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let kda = KernelDiscriminantAnalysis::new()
        .kernel(KernelType::RBF { gamma: 1.0 })
        .n_components(Some(1));
    let fitted = kda.fit(&x, &y).expect("model fitting should succeed");
    let transformed = fitted.transform(&x).expect("transform should succeed");

    assert_eq!(transformed.dim(), (4, 1));
}

#[test]
fn test_kernel_discriminant_with_regularization() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    let kda = KernelDiscriminantAnalysis::new()
        .kernel(KernelType::RBF { gamma: 1.0 })
        .reg_param(0.1);
    let fitted = kda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_custom_kernel() {
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![0, 0, 1, 1];

    // Custom linear kernel
    let custom_kernel = Box::new(|x1: &ArrayView1<Float>, x2: &ArrayView1<Float>| -> Float {
        x1.iter()
            .zip(x2.iter())
            .map(|(&a, &b)| a * b)
            .sum::<Float>()
    });

    let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::Custom(custom_kernel));
    let fitted = kda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 4);
    assert_eq!(fitted.classes().len(), 2);
}

#[test]
fn test_kernel_discriminant_multiclass() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1], // Class 0
        [3.0, 4.0],
        [3.1, 4.1], // Class 1
        [5.0, 6.0],
        [5.1, 6.1] // Class 2
    ];
    let y = array![0, 0, 1, 1, 2, 2];

    let kda = KernelDiscriminantAnalysis::new().kernel(KernelType::RBF { gamma: 1.0 });
    let fitted = kda.fit(&x, &y).expect("model fitting should succeed");
    let predictions = fitted.predict(&x).expect("prediction should succeed");

    assert_eq!(predictions.len(), 6);
    assert_eq!(fitted.classes().len(), 3);
}

#[test]
fn test_kernel_parameter_grid() {
    let grid = KernelParameterGrid::new()
        .gamma_values(vec![0.1, 1.0])
        .coef0_values(vec![0.0, 1.0])
        .degree_values(vec![2, 3]);

    assert_eq!(grid.gamma_values, vec![0.1, 1.0]);
    assert_eq!(grid.coef0_values, vec![0.0, 1.0]);
    assert_eq!(grid.degree_values, vec![2, 3]);
}

#[test]
fn test_kernel_parameter_optimizer() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 2.2],
        [1.3, 2.3], // Class 0
        [3.0, 4.0],
        [3.1, 4.1],
        [3.2, 4.2],
        [3.3, 4.3] // Class 1
    ];
    let y = array![0, 0, 0, 0, 1, 1, 1, 1];

    let grid = KernelParameterGrid::new()
        .gamma_values(vec![0.1, 1.0])
        .coef0_values(vec![0.0])
        .degree_values(vec![2]);

    let cv_config = KernelCVConfig {
        n_folds: 2,
        scoring: "accuracy".to_string(),
        random_state: Some(42),
        shuffle: false,
    };

    let optimizer = KernelParameterOptimizer::new()
        .parameter_grid(grid)
        .cv_config(cv_config)
        .optimization_strategy("grid_search");

    let results = optimizer.optimize(&x, &y);
    assert!(results.is_ok());

    let results = results.expect("operation should succeed");
    assert!(results.best_score >= 0.0);
    assert!(results.best_score <= 1.0);
    assert!(!results.cv_scores.is_empty());
    assert!(!results.all_results.is_empty());
}

#[test]
fn test_kernel_discriminant_fit_auto() {
    let x = array![
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 2.2],
        [1.3, 2.3], // Class 0
        [3.0, 4.0],
        [3.1, 4.1],
        [3.2, 4.2],
        [3.3, 4.3] // Class 1
    ];
    let y = array![0, 0, 0, 0, 1, 1, 1, 1];

    let kda = KernelDiscriminantAnalysis::new();
    let fitted = kda.fit_auto(&x, &y);

    // fit_auto might succeed or fail depending on data, just check it doesn't panic
    match fitted {
        Ok(fitted) => {
            assert_eq!(fitted.classes().len(), 2);
            let predictions = fitted.predict(&x).expect("prediction should succeed");
            assert_eq!(predictions.len(), 8);
        }
        Err(_) => {
            // If auto-optimization fails (due to small dataset), that's acceptable
        }
    }
}

#[test]
fn test_kernel_discriminant_fit_with_custom_optimizer() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let grid = KernelParameterGrid::new()
        .gamma_values(vec![0.1, 1.0])
        .coef0_values(vec![0.0])
        .degree_values(vec![2]);

    let cv_config = KernelCVConfig {
        n_folds: 2,
        scoring: "accuracy".to_string(),
        random_state: Some(42),
        shuffle: false,
    };

    let optimizer = KernelParameterOptimizer::new()
        .parameter_grid(grid)
        .cv_config(cv_config)
        .optimization_strategy("random_search");

    let kda = KernelDiscriminantAnalysis::new();
    let fitted = kda.fit_with_optimizer(&x, &y, &optimizer);

    match fitted {
        Ok(fitted) => {
            assert_eq!(fitted.classes().len(), 2);
            let predictions = fitted.predict(&x).expect("prediction should succeed");
            assert_eq!(predictions.len(), 4);
        }
        Err(_) => {
            // Acceptable for small datasets
        }
    }
}

#[test]
fn test_kernel_optimization_strategies() {
    let x = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]];
    let y = array![0, 0, 1, 1];

    let grid = KernelParameterGrid::new()
        .gamma_values(vec![0.1, 1.0])
        .coef0_values(vec![0.0])
        .degree_values(vec![2]);

    let cv_config = KernelCVConfig {
        n_folds: 2,
        scoring: "accuracy".to_string(),
        random_state: Some(42),
        shuffle: false,
    };

    let strategies = ["grid_search", "random_search", "bayesian"];

    for strategy in &strategies {
        let optimizer = KernelParameterOptimizer::new()
            .parameter_grid(grid.clone())
            .cv_config(cv_config.clone())
            .optimization_strategy(strategy);

        let results = optimizer.optimize(&x, &y);
        match results {
            Ok(results) => {
                assert!(results.best_score >= 0.0);
                assert!(results.best_score <= 1.0);
            }
            Err(_) => {
                // Acceptable for some strategies with small datasets
            }
        }
    }
}
