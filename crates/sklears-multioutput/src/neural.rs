//! Neural Network Multi-Output Learning
//!
//! This module provides comprehensive neural network implementations for multi-output learning,
//! including multi-layer perceptrons (MLP), recurrent neural networks (RNN/LSTM/GRU),
//! multi-task learning networks, and adversarial training approaches.
//!
//! The module is organized into specialized submodules for different neural network components:
//!
//! - `activation`: Activation functions (ReLU, Sigmoid, Tanh, Linear, Softmax)
//! - `loss`: Loss functions for training (MSE, Cross-Entropy, Binary Cross-Entropy)
//! - `mlp`: Multi-Layer Perceptron for basic feedforward neural networks
//! - `recurrent`: Recurrent neural networks (RNN, LSTM, GRU) for sequence modeling
//! - `multitask`: Multi-task learning with shared representations
//! - `adversarial`: Adversarial multi-task learning with feature disentanglement
//!
//! # Architecture Overview
//!
//! This implementation follows a modular design where each neural network type is
//! contained in its own module while sharing common components like activation and
//! loss functions. All implementations use SciRS2-Core for numerical operations
//! and follow the sklears trait system for consistent API.
//!
//! # Examples
//!
//! ## Basic Multi-Layer Perceptron
//!
//! ```rust
//! use sklears_multioutput::neural::{MultiOutputMLP, ActivationFunction, LossFunction};
//! use sklears_core::traits::{Fit, Predict};
//! use scirs2_core::ndarray::array;
//!
//! let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
//! let y = array![[0.5, 1.2], [1.0, 2.1], [1.5, 0.8], [2.0, 2.5]];
//!
//! let mlp = MultiOutputMLP::new()
//!     .hidden_layer_sizes(vec![10, 5])
//!     .activation(ActivationFunction::ReLU)
//!     .output_activation(ActivationFunction::Linear)
//!     .loss_function(LossFunction::MeanSquaredError)
//!     .learning_rate(0.01)
//!     .max_iter(1000)
//!     .random_state(Some(42));
//!
//! let trained_mlp = mlp.fit(&X.view(), &y)?;
//! let predictions = trained_mlp.predict(&X.view())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Recurrent Neural Network for Sequences
//!
//! ```rust
//! use sklears_multioutput::neural::{RecurrentNeuralNetwork, CellType, SequenceMode};
//! use sklears_core::traits::{Fit, Predict};
//! use scirs2_core::ndarray::Array3;
//!
//! // Create sequence data: (samples, timesteps, features)
//! let X = Array3::<f64>::zeros((2, 5, 3));
//! let y = Array3::<f64>::zeros((2, 5, 1));
//!
//! let rnn = RecurrentNeuralNetwork::new()
//!     .cell_type(CellType::LSTM)
//!     .hidden_size(50)
//!     .sequence_mode(SequenceMode::ManyToMany)
//!     .learning_rate(0.001)
//!     .max_iter(100);
//!
//! let trained_rnn = rnn.fit(&X.view(), &y)?;
//! let predictions = trained_rnn.predict(&X.view())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Multi-Task Learning
//!
//! ```rust
//! use sklears_multioutput::neural::{MultiTaskNeuralNetwork, TaskBalancing};
//! use sklears_core::traits::{Fit, Predict};
//! use std::collections::HashMap;
//! use scirs2_core::ndarray::array;
//!
//! let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
//! let mut tasks = HashMap::new();
//! tasks.insert("regression".to_string(), array![[0.5], [1.0], [1.5], [2.0]]);
//! tasks.insert("classification".to_string(), array![[1.0], [0.0], [1.0], [0.0]]);
//!
//! let mt_net = MultiTaskNeuralNetwork::new()
//!     .shared_layers(vec![20, 10])
//!     .task_specific_layers(vec![5])
//!     .task_outputs(&[("regression", 1), ("classification", 1)])
//!     .task_balancing(TaskBalancing::Adaptive)
//!     .learning_rate(0.01)
//!     .random_state(Some(42));
//!
//! let trained = mt_net.fit(&X.view(), &tasks)?;
//! let predictions = trained.predict(&X.view())?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Re-export commonly used types for convenience from sibling modules
pub use crate::activation::ActivationFunction;
pub use crate::loss::LossFunction;

// MLP types
pub use crate::mlp::{
    MultiOutputMLP, MultiOutputMLPClassifier, MultiOutputMLPRegressor, MultiOutputMLPTrained,
};

// Recurrent network types
pub use crate::recurrent::{
    CellType, RecurrentNeuralNetwork, RecurrentNeuralNetworkTrained, SequenceMode,
};

// Multi-task learning types
pub use crate::multitask::{MultiTaskNeuralNetwork, MultiTaskNeuralNetworkTrained, TaskBalancing};

// Adversarial learning types
pub use crate::adversarial::{
    AdversarialMultiTaskNetwork, AdversarialMultiTaskNetworkTrained, AdversarialStrategy,
    GradientReversalConfig, LambdaSchedule, TaskDiscriminator,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array3};
    use sklears_core::traits::{Fit, Predict};
    use std::collections::HashMap;

    #[test]
    fn test_activation_functions_integration() {
        let x = array![1.0, -1.0, 0.0, 2.0];

        // Test ReLU
        let relu_result = ActivationFunction::ReLU.apply(&x);
        assert_eq!(relu_result, array![1.0, 0.0, 0.0, 2.0]);

        // Test Linear
        let linear_result = ActivationFunction::Linear.apply(&x);
        assert_eq!(linear_result, x);
    }

    #[test]
    fn test_mlp_integration() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[0.5, 1.2], [1.0, 2.1]];

        let mlp = MultiOutputMLP::new()
            .hidden_layer_sizes(vec![5])
            .max_iter(5)
            .random_state(Some(42));

        let trained_mlp = mlp.fit(&X.view(), &y).unwrap();
        let predictions = trained_mlp.predict(&X.view()).unwrap();

        assert_eq!(predictions.dim(), (2, 2));
        assert!(!predictions.iter().any(|&x| x.is_nan()));
    }

    #[test]
    fn test_multitask_integration() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let mut tasks = HashMap::new();
        tasks.insert("task1".to_string(), array![[0.5], [1.0]]);
        tasks.insert("task2".to_string(), array![[1.0], [0.0]]);

        let mt_net = MultiTaskNeuralNetwork::new()
            .task_outputs(&[("task1", 1), ("task2", 1)])
            .max_iter(5)
            .random_state(Some(42));

        let trained = mt_net.fit(&X.view(), &tasks).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions.contains_key("task1"));
        assert!(predictions.contains_key("task2"));
    }

    #[test]
    fn test_adversarial_integration() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let mut tasks = HashMap::new();
        tasks.insert("task1".to_string(), array![[0.5], [1.0]]);

        let adv_net = AdversarialMultiTaskNetwork::new()
            .task_outputs(&[("task1", 1)])
            .max_iter(5)
            .random_state(Some(42));

        let trained = adv_net.fit(&X.view(), &tasks).unwrap();
        let predictions = trained.predict(&X.view()).unwrap();

        assert_eq!(predictions.len(), 1);
        assert!(predictions.contains_key("task1"));
    }

    #[test]
    fn test_module_exports() {
        // Test that all major types are properly exported
        use super::{
            ActivationFunction, AdversarialMultiTaskNetwork, AdversarialStrategy, CellType,
            LossFunction, MultiOutputMLP, MultiTaskNeuralNetwork, RecurrentNeuralNetwork,
            SequenceMode, TaskBalancing,
        };

        // Create instances to verify they're accessible
        let _mlp = MultiOutputMLP::new();
        let _rnn = RecurrentNeuralNetwork::new();
        let _mt = MultiTaskNeuralNetwork::new();
        let _adv = AdversarialMultiTaskNetwork::new();

        // Test enums are accessible
        let _act = ActivationFunction::ReLU;
        let _loss = LossFunction::MeanSquaredError;
        let _cell = CellType::LSTM;
        let _seq = SequenceMode::ManyToMany;
        let _bal = TaskBalancing::Equal;
        let _strat = AdversarialStrategy::GradientReversal;
    }
}
