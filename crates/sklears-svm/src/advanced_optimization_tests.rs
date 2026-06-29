use super::*;

fn create_test_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0])
        .expect("shape");
    let y = Array1::from_vec(vec![1.0, 1.0, -1.0, -1.0]);
    (x, y)
}

#[test]
fn test_admm_svm_basic() {
    let (x, y) = create_test_data();
    let mut admm = ADMMSVM::default();
    let result = admm.fit(&x, &y);
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert!(result.dual_coef.len() == x.nrows());
    assert!(!result.support_indices.is_empty());
}

#[test]
fn test_newton_svm_linear() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        kernel: KernelType::Linear,
        ..Default::default()
    };
    let mut newton = NewtonSVM::new(config);
    let result = newton.fit(&x, &y);
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert!(result.dual_coef.len() == x.nrows());
}

#[test]
fn test_newton_svm_rbf() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        kernel: KernelType::Rbf { gamma: 1.0 },
        ..Default::default()
    };
    let mut newton = NewtonSVM::new(config);
    let result = newton.fit(&x, &y);
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert!(result.dual_coef.len() == x.nrows());
}

#[test]
fn test_admm_convergence() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        max_iter: 100,
        tol: 1e-3,
        ..Default::default()
    };
    let mut admm = ADMMSVM::new(config);
    let result = admm.fit(&x, &y).expect("model fitting should succeed");

    // The ADMM alpha-subproblem maximizes the SVM dual objective
    // W(alpha) = e^T alpha - 0.5 alpha^T K alpha, so the reported objective
    // is expected to increase (or stay stable), consistent with the
    // trust-region and accelerated-gradient maximization solvers below.
    assert!(result.history.len() > 1);
    let first_obj = result.history[0];
    let last_obj = result.history.last().expect("operation should succeed");
    assert!(last_obj >= &first_obj || (last_obj - first_obj).abs() < 1e-6);
}

#[test]
fn test_advanced_optimization_config() {
    let config = AdvancedOptimizationConfig {
        c: 0.5,
        kernel: KernelType::Polynomial {
            gamma: 1.0,
            degree: 2.0,
            coef0: 1.0,
        },
        tol: 1e-5,
        max_iter: 500,
        rho: 2.0,
        trust_radius: 0.5,
        verbose: true,
        ..Default::default()
    };

    let admm = ADMMSVM::new(config.clone());
    assert_eq!(admm.config.c, 0.5);
    assert_eq!(admm.config.rho, 2.0);
    assert_eq!(admm.config.trust_radius, 0.5);
    assert!(admm.config.verbose);
}

#[test]
fn test_trust_region_svm_linear() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        kernel: KernelType::Linear,
        trust_radius: 1.0,
        max_iter: 50,
        tol: 1e-4,
        ..Default::default()
    };
    let mut trust_region = TrustRegionSVM::new(config);
    let result = trust_region.fit(&x, &y);
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert!(result.dual_coef.len() == x.nrows());
    assert!(result.n_iterations > 0);
}

#[test]
fn test_trust_region_svm_rbf() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        kernel: KernelType::Rbf { gamma: 1.0 },
        trust_radius: 0.5,
        max_iter: 50,
        tol: 1e-4,
        ..Default::default()
    };
    let mut trust_region = TrustRegionSVM::new(config);
    let result = trust_region.fit(&x, &y);
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert!(result.dual_coef.len() == x.nrows());
    assert!(result.n_iterations > 0); // Should have performed iterations
}

#[test]
fn test_trust_region_convergence() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        kernel: KernelType::Linear,
        trust_radius: 1.0,
        max_iter: 100,
        tol: 1e-5,
        verbose: false,
        ..Default::default()
    };
    let mut trust_region = TrustRegionSVM::new(config);
    let result = trust_region
        .fit(&x, &y)
        .expect("model fitting should succeed");

    // Check that objective improves
    assert!(result.history.len() > 1);
    let first_obj = result.history[0];
    let last_obj = result.history.last().expect("operation should succeed");

    // For maximization problem, objective should increase or stay same
    assert!(last_obj >= &first_obj || (last_obj - first_obj).abs() < 1e-6);
}

#[test]
fn test_trust_region_prediction() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        kernel: KernelType::Linear,
        trust_radius: 1.0,
        max_iter: 30,
        ..Default::default()
    };
    let mut trust_region = TrustRegionSVM::new(config);
    let result = trust_region
        .fit(&x, &y)
        .expect("model fitting should succeed");

    // Test prediction
    let predictions = trust_region
        .predict(&x, &result)
        .expect("prediction should succeed");
    assert_eq!(predictions.len(), x.nrows());

    // All predictions should be either 1.0 or -1.0
    for &pred in predictions.iter() {
        assert!(pred == 1.0 || pred == -1.0);
    }

    // Test decision function
    let decision_values = trust_region
        .decision_function(&x, &result)
        .expect("decision function should succeed");
    assert_eq!(decision_values.len(), x.nrows());

    // Decision values should be finite
    for &val in decision_values.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_accelerated_gradient_svm_nesterov() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        kernel: KernelType::Linear,
        max_iter: 100,
        tol: 1e-4,
        ..Default::default()
    };
    let mut accel_svm = AcceleratedGradientSVM::new(config)
        .with_momentum(0.9)
        .with_learning_rate(0.01)
        .with_method(AcceleratedMethod::Nesterov);

    let result = accel_svm.fit(&x, &y);
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert!(result.dual_coef.len() == x.nrows());
    assert!(result.n_iterations > 0);
}

#[test]
fn test_accelerated_gradient_svm_fista() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        kernel: KernelType::Linear,
        max_iter: 100,
        tol: 1e-4,
        ..Default::default()
    };
    let mut accel_svm = AcceleratedGradientSVM::new(config)
        .with_momentum(0.9)
        .with_learning_rate(0.01)
        .with_method(AcceleratedMethod::FISTA);

    let result = accel_svm.fit(&x, &y);
    assert!(result.is_ok());

    let result = result.expect("operation should succeed");
    assert!(result.dual_coef.len() == x.nrows());
    assert!(result.n_iterations > 0);
}

#[test]
fn test_accelerated_gradient_convergence() {
    let (x, y) = create_test_data();
    let config = AdvancedOptimizationConfig {
        kernel: KernelType::Linear,
        max_iter: 200,
        tol: 1e-5,
        ..Default::default()
    };
    let mut accel_svm =
        AcceleratedGradientSVM::new(config).with_method(AcceleratedMethod::Nesterov);

    let result = accel_svm.fit(&x, &y).expect("model fitting should succeed");

    // Check that objective improves
    assert!(result.history.len() > 1);
    let first_obj = result.history[0];
    let last_obj = result.history.last().expect("operation should succeed");

    // For dual maximization, objective should increase or stay stable
    assert!(last_obj >= &first_obj || (last_obj - first_obj).abs() < 1e-4);
}
