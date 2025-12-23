//! Tests for custom kernel functionality in SVM
//! NOTE: Custom kernel support is not yet implemented in SVC/SVR.
//! These tests are commented out until the feature is added.

#![allow(unexpected_cfgs)]

#[cfg(feature = "custom_kernels")] // Feature doesn't exist yet
mod custom_kernel_tests {

    use scirs2_autograd::ndarray::{array, Array1, Array2};
    use sklears_core::traits::Predict;
    use sklears_core::types::Float;
    use sklears_metrics::{classification::accuracy_score, regression::mean_squared_error};
    use sklears_svm::{Kernel, KernelType, SVC, SVR};
    use std::fmt::Debug;

    /// Simple custom linear kernel for testing
    #[derive(Debug)]
    struct TestLinearKernel {
        scale: Float,
    }

    impl Kernel for TestLinearKernel {
        fn compute(&self, x1: &Array1<Float>, x2: &Array1<Float>) -> Float {
            self.scale * x1.dot(x2)
        }
    }

    /// Custom polynomial-like kernel for testing
    #[derive(Debug)]
    struct TestPolyKernel {
        power: i32,
    }

    impl Kernel for TestPolyKernel {
        fn compute(&self, x1: &Array1<Float>, x2: &Array1<Float>) -> Float {
            (1.0 + x1.dot(x2)).powi(self.power)
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_custom_kernel_svc() {
        // Simple linearly separable data
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
            [-3.0, 0.0],
        ];
        let y = array![1, 1, 1, 0, 0, 0];

        // Test with custom linear kernel
        let custom_kernel = Box::new(TestLinearKernel { scale: 2.0 });
        let svc = SVC::new(KernelType::Custom(custom_kernel), 1.0);
        let trained = svc
            .fit(&x, &y)
            .expect("Failed to fit SVC with custom kernel");

        let predictions = trained.predict(&x).unwrap();
        let accuracy = accuracy_score(&y, &predictions);

        // Should achieve perfect accuracy on linearly separable data
        assert!(
            accuracy >= 0.95,
            "Custom kernel SVC should achieve high accuracy"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_custom_kernel_svr() {
        // Simple regression data
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        // Test with custom polynomial kernel
        let custom_kernel = Box::new(TestPolyKernel { power: 1 });
        let svr = SVR::new(KernelType::Custom(custom_kernel), 1.0, 0.1);
        let trained = svr
            .fit(&x, &y)
            .expect("Failed to fit SVR with custom kernel");

        let predictions = trained.predict(&x).unwrap();
        let mse = mean_squared_error(&y, &predictions);

        // Should achieve low MSE on linear data
        assert!(mse < 1.0, "Custom kernel SVR should achieve low MSE");
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_custom_kernel_multiclass() {
        // Multi-class classification
        let x = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [5.0, 1.0],
            [5.5, 1.5],
            [3.0, 5.0],
            [3.5, 5.5],
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        // Custom kernel that emphasizes certain dimensions
        #[derive(Debug)]
        struct WeightedKernel {
            weights: Array1<Float>,
        }

        impl Kernel for WeightedKernel {
            fn compute(&self, x1: &Array1<Float>, x2: &Array1<Float>) -> Float {
                let weighted_x1 = x1 * &self.weights;
                let weighted_x2 = x2 * &self.weights;
                weighted_x1.dot(&weighted_x2)
            }
        }

        let weights = array![2.0, 1.0]; // Emphasize first dimension
        let custom_kernel = Box::new(WeightedKernel { weights });
        let svc = SVC::new(KernelType::Custom(custom_kernel), 1.0);
        let trained = svc.fit(&x, &y).expect("Failed to fit multi-class SVC");

        let predictions = trained.predict(&x).unwrap();
        let accuracy = accuracy_score(&y, &predictions);

        // Should achieve good accuracy
        assert!(accuracy >= 0.8, "Custom kernel should work for multi-class");
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_custom_kernel_matrix_computation() {
        // Test that custom kernel matrix computation works correctly
        #[derive(Debug)]
        struct IdentityKernel;

        impl Kernel for IdentityKernel {
            fn compute(&self, x1: &Array1<Float>, x2: &Array1<Float>) -> Float {
                if x1 == x2 {
                    1.0
                } else {
                    0.0
                }
            }
        }

        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let kernel = IdentityKernel;
        let kernel_matrix = kernel.compute_matrix(&x);

        // Check diagonal is all 1s
        for i in 0..3 {
            assert_eq!(kernel_matrix[[i, i]], 1.0);
        }

        // Check off-diagonal based on actual computation
        // Since rows are different, off-diagonal should be 0
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert_eq!(kernel_matrix[[i, j]], 0.0);
                }
            }
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_custom_kernel_cross_computation() {
        // Test cross kernel computation (between train and test sets)
        #[derive(Debug)]
        struct DotProductKernel;

        impl Kernel for DotProductKernel {
            fn compute(&self, x1: &Array1<Float>, x2: &Array1<Float>) -> Float {
                x1.dot(x2)
            }
        }

        let x_train = array![[1.0, 0.0], [0.0, 1.0],];

        let x_test = array![[1.0, 1.0], [2.0, 0.0],];

        let kernel = DotProductKernel;
        let kernel_cross = kernel.compute_cross(&x_train, &x_test);

        // Verify dimensions
        assert_eq!(kernel_cross.shape(), &[2, 2]);

        // Verify values
        // x_test[0] . x_train[0] = [1,1] . [1,0] = 1
        assert_eq!(kernel_cross[[0, 0]], 1.0);
        // x_test[0] . x_train[1] = [1,1] . [0,1] = 1
        assert_eq!(kernel_cross[[0, 1]], 1.0);
        // x_test[1] . x_train[0] = [2,0] . [1,0] = 2
        assert_eq!(kernel_cross[[1, 0]], 2.0);
        // x_test[1] . x_train[1] = [2,0] . [0,1] = 0
        assert_eq!(kernel_cross[[1, 1]], 0.0);
    }
} // mod custom_kernel_tests

// Basic kernel tests that work with current implementation
#[allow(non_snake_case)]
#[cfg(test)]
mod basic_kernel_tests {
    use scirs2_core::ndarray::array;

    use sklears_svm::kernels::*;

    #[test]
    #[allow(non_snake_case)]
    fn test_linear_kernel() {
        let kernel = LinearKernel;
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![4.0, 5.0, 6.0];

        let result = kernel.compute(x1.view(), x2.view());
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_rbf_kernel() {
        let kernel = RbfKernel::new(1.0);
        let x1 = array![1.0, 0.0];
        let x2 = array![1.0, 0.0];

        let result = kernel.compute(x1.view(), x2.view());
        assert_eq!(result, 1.0); // Same vectors should give 1.0
    }
}
