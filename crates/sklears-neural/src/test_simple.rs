//! Simple test to verify neural network functionality

use crate::{activation::Activation, MLPClassifier};
use scirs2_core::ndarray::array;
use sklears_core::traits::{Fit, Predict};

#[test]
fn test_neural_network_basic() {
    // Simple XOR-like problem
    let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],];
    let y = vec![0, 1, 1, 0];

    let mlp = MLPClassifier::new()
        .hidden_layer_sizes(&[5])
        .activation(Activation::Tanh)
        .max_iter(50)
        .random_state(42);

    let trained_mlp = mlp.fit(&x, &y).unwrap();
    let predictions = trained_mlp.predict(&x).unwrap();

    // Just verify we get predictions of the right shape
    assert_eq!(predictions.len(), 4);

    // Verify classes
    assert_eq!(trained_mlp.classes(), &[0, 1]);

    // Verify probabilities
    let probabilities = trained_mlp.predict_proba(&x).unwrap();
    assert_eq!(probabilities.dim(), (4, 2));

    // Verify probabilities sum to 1
    for i in 0..4 {
        let sum: f64 = probabilities.row(i).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
