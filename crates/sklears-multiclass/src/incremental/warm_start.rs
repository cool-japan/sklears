//! Warm Start Capabilities for Incremental Learning
//!
//! Provides functionality to resume training from saved model state,
//! enabling efficient incremental updates without full retraining.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;

/// Warm start configuration
#[derive(Debug, Clone)]
pub struct WarmStartConfig {
    /// Whether warm start is enabled
    pub enabled: bool,
    /// Initial learning rate for warm start
    pub initial_lr: f64,
    /// Learning rate decay factor
    pub lr_decay: f64,
    /// Minimum learning rate
    pub min_lr: f64,
    /// Number of iterations before learning rate decay
    pub decay_steps: usize,
}

impl Default for WarmStartConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_lr: 0.01,
            lr_decay: 0.95,
            min_lr: 1e-6,
            decay_steps: 100,
        }
    }
}

/// Model state for warm start
#[derive(Debug, Clone)]
pub struct ModelState {
    /// Model weights
    pub weights: Array2<f64>,
    /// Model biases
    pub biases: Array1<f64>,
    /// Classes seen during training
    pub classes: Vec<i32>,
    /// Number of training iterations
    pub n_iterations: usize,
    /// Current learning rate
    pub learning_rate: f64,
    /// Training history (loss values)
    pub training_history: Vec<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ModelState {
    /// Create a new model state
    pub fn new(weights: Array2<f64>, biases: Array1<f64>, classes: Vec<i32>) -> Self {
        Self {
            weights,
            biases,
            classes,
            n_iterations: 0,
            learning_rate: 0.01,
            training_history: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Update the model state with new training iteration
    pub fn update(&mut self, loss: f64, new_lr: f64) {
        self.n_iterations += 1;
        self.learning_rate = new_lr;
        self.training_history.push(loss);
    }

    /// Check if model has converged
    pub fn has_converged(&self, window_size: usize, tolerance: f64) -> bool {
        if self.training_history.len() < window_size {
            return false;
        }

        let recent = &self.training_history[self.training_history.len() - window_size..];
        let mean: f64 = recent.iter().sum::<f64>() / window_size as f64;
        let variance: f64 =
            recent.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window_size as f64;

        variance < tolerance
    }

    /// Get the average loss over last N iterations
    pub fn recent_loss(&self, n: usize) -> Option<f64> {
        if self.training_history.is_empty() {
            return None;
        }

        let n = n.min(self.training_history.len());
        let recent = &self.training_history[self.training_history.len() - n..];
        Some(recent.iter().sum::<f64>() / n as f64)
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

/// Warm start manager for incremental learning
pub struct WarmStartManager {
    config: WarmStartConfig,
    current_state: Option<ModelState>,
    checkpoints: Vec<ModelState>,
}

impl WarmStartManager {
    /// Create a new warm start manager
    pub fn new(config: WarmStartConfig) -> Self {
        Self {
            config,
            current_state: None,
            checkpoints: Vec::new(),
        }
    }

    /// Initialize with a model state
    pub fn initialize(&mut self, state: ModelState) -> SklResult<()> {
        if !self.config.enabled {
            return Err(SklearsError::InvalidInput(
                "Warm start is not enabled".to_string(),
            ));
        }

        self.current_state = Some(state);
        Ok(())
    }

    /// Get current model state
    pub fn current_state(&self) -> Option<&ModelState> {
        self.current_state.as_ref()
    }

    /// Get mutable current state
    pub fn current_state_mut(&mut self) -> Option<&mut ModelState> {
        self.current_state.as_mut()
    }

    /// Update current state
    pub fn update_state(&mut self, loss: f64) -> SklResult<()> {
        let state = self
            .current_state
            .as_mut()
            .ok_or_else(|| SklearsError::InvalidInput("No current state to update".to_string()))?;

        // Calculate new learning rate with decay (check before incrementing)
        let new_lr = if (state.n_iterations + 1) % self.config.decay_steps == 0 {
            (state.learning_rate * self.config.lr_decay).max(self.config.min_lr)
        } else {
            state.learning_rate
        };

        state.update(loss, new_lr);
        Ok(())
    }

    /// Create a checkpoint of current state
    pub fn checkpoint(&mut self) -> SklResult<()> {
        let state = self.current_state.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("No current state to checkpoint".to_string())
        })?;

        self.checkpoints.push(state.clone());
        Ok(())
    }

    /// Restore from latest checkpoint
    pub fn restore_checkpoint(&mut self) -> SklResult<()> {
        let state = self
            .checkpoints
            .pop()
            .ok_or_else(|| SklearsError::InvalidInput("No checkpoints available".to_string()))?;

        self.current_state = Some(state);
        Ok(())
    }

    /// Get number of checkpoints
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    /// Clear all checkpoints
    pub fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
    }

    /// Check if model should continue training
    pub fn should_continue_training(
        &self,
        max_iterations: usize,
        convergence_window: usize,
        tolerance: f64,
    ) -> bool {
        if let Some(state) = &self.current_state {
            if state.n_iterations >= max_iterations {
                return false;
            }

            if state.has_converged(convergence_window, tolerance) {
                return false;
            }

            true
        } else {
            false
        }
    }

    /// Get current learning rate
    pub fn current_learning_rate(&self) -> Option<f64> {
        self.current_state.as_ref().map(|s| s.learning_rate)
    }

    /// Get training progress
    pub fn training_progress(&self) -> Option<TrainingProgress> {
        self.current_state.as_ref().map(|state| TrainingProgress {
            iterations: state.n_iterations,
            current_lr: state.learning_rate,
            recent_loss: state.recent_loss(10),
            num_checkpoints: self.checkpoints.len(),
        })
    }
}

/// Training progress information
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    pub iterations: usize,
    pub current_lr: f64,
    pub recent_loss: Option<f64>,
    pub num_checkpoints: usize,
}

impl TrainingProgress {
    /// Print progress
    pub fn print(&self) {
        println!("Training Progress:");
        println!("  Iterations: {}", self.iterations);
        println!("  Learning Rate: {:.6}", self.current_lr);
        if let Some(loss) = self.recent_loss {
            println!("  Recent Loss: {:.6}", loss);
        }
        println!("  Checkpoints: {}", self.num_checkpoints);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_warm_start_config_default() {
        let config = WarmStartConfig::default();
        assert!(config.enabled);
        assert_eq!(config.initial_lr, 0.01);
        assert_eq!(config.decay_steps, 100);
    }

    #[test]
    fn test_model_state_creation() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let state = ModelState::new(weights.clone(), biases.clone(), classes.clone());
        assert_eq!(state.n_iterations, 0);
        assert_eq!(state.classes, classes);
    }

    #[test]
    fn test_model_state_update() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let mut state = ModelState::new(weights, biases, classes);
        state.update(0.5, 0.01);

        assert_eq!(state.n_iterations, 1);
        assert_eq!(state.learning_rate, 0.01);
        assert_eq!(state.training_history.len(), 1);
        assert_eq!(state.training_history[0], 0.5);
    }

    #[test]
    fn test_model_state_convergence() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let mut state = ModelState::new(weights, biases, classes);

        // Add converged history
        for _ in 0..10 {
            state.update(0.001, 0.01);
        }

        assert!(state.has_converged(5, 1e-6));
    }

    #[test]
    fn test_model_state_recent_loss() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let mut state = ModelState::new(weights, biases, classes);

        for i in 0..10 {
            state.update(i as f64, 0.01);
        }

        let recent = state.recent_loss(5).unwrap();
        // Average of [5, 6, 7, 8, 9] = 7.0
        assert!((recent - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_warm_start_manager_initialization() {
        let mut manager = WarmStartManager::new(WarmStartConfig::default());
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let state = ModelState::new(weights, biases, classes);
        assert!(manager.initialize(state).is_ok());
        assert!(manager.current_state().is_some());
    }

    #[test]
    fn test_warm_start_manager_update() {
        let mut manager = WarmStartManager::new(WarmStartConfig::default());
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let state = ModelState::new(weights, biases, classes);
        manager.initialize(state).unwrap();

        assert!(manager.update_state(0.5).is_ok());
        assert_eq!(manager.current_state().unwrap().n_iterations, 1);
    }

    #[test]
    fn test_warm_start_manager_checkpoint() {
        let mut manager = WarmStartManager::new(WarmStartConfig::default());
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let state = ModelState::new(weights, biases, classes);
        manager.initialize(state).unwrap();

        assert!(manager.checkpoint().is_ok());
        assert_eq!(manager.num_checkpoints(), 1);
    }

    #[test]
    fn test_warm_start_manager_restore() {
        let mut manager = WarmStartManager::new(WarmStartConfig::default());
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let state = ModelState::new(weights, biases, classes);
        manager.initialize(state).unwrap();

        manager.checkpoint().unwrap();
        manager.update_state(0.5).unwrap();

        // Restore checkpoint
        assert!(manager.restore_checkpoint().is_ok());
        assert_eq!(manager.current_state().unwrap().n_iterations, 0);
    }

    #[test]
    fn test_warm_start_learning_rate_decay() {
        let mut config = WarmStartConfig::default();
        config.initial_lr = 1.0;
        config.lr_decay = 0.5;
        config.decay_steps = 2;

        let mut manager = WarmStartManager::new(config);
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let mut state = ModelState::new(weights, biases, classes);
        state.learning_rate = 1.0;
        manager.initialize(state).unwrap();

        // First update (no decay)
        manager.update_state(0.5).unwrap();
        assert_eq!(manager.current_learning_rate().unwrap(), 1.0);

        // Second update (triggers decay)
        manager.update_state(0.4).unwrap();
        assert_eq!(manager.current_learning_rate().unwrap(), 0.5);
    }

    #[test]
    fn test_training_progress() {
        let mut manager = WarmStartManager::new(WarmStartConfig::default());
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let state = ModelState::new(weights, biases, classes);
        manager.initialize(state).unwrap();
        manager.update_state(0.5).unwrap();

        let progress = manager.training_progress().unwrap();
        assert_eq!(progress.iterations, 1);
        assert!(progress.recent_loss.is_some());
    }

    #[test]
    fn test_model_state_metadata() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let biases = array![0.1, 0.2];
        let classes = vec![0, 1];

        let mut state = ModelState::new(weights, biases, classes);
        state.add_metadata("model_type".to_string(), "linear".to_string());

        assert_eq!(
            state.get_metadata("model_type"),
            Some(&"linear".to_string())
        );
    }
}
