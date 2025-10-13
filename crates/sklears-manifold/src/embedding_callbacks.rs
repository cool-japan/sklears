//! Embedding callbacks for monitoring and customizing manifold learning
//!
//! This module provides a callback system that allows users to hook into the
//! training process of manifold learning algorithms to monitor progress,
//! save intermediate results, implement early stopping, or customize behavior.

use scirs2_core::ndarray::Array2;
use sklears_core::{error::Result as SklResult, types::Float};
use std::collections::HashMap;

/// Callback event that occurs during manifold learning
#[derive(Debug, Clone, PartialEq)]
pub enum CallbackEvent {
    /// Training started
    TrainingStarted,
    /// Iteration completed
    IterationCompleted { iteration: usize, loss: Float },
    /// Epoch completed (for algorithms with epochs)
    EpochCompleted { epoch: usize, loss: Float },
    /// Early stopping triggered
    EarlyStopping { reason: String },
    /// Training completed
    TrainingCompleted { final_loss: Float },
    /// Custom event with arbitrary data
    Custom {
        name: String,
        data: HashMap<String, Float>,
    },
}

/// Context information available to callbacks during training
#[derive(Debug, Clone)]
pub struct CallbackContext {
    /// Current iteration number
    pub iteration: usize,
    /// Current loss value
    pub loss: Float,
    /// Current embedding state
    pub embedding: Array2<Float>,
    /// Gradient norms (if available)
    pub gradient_norm: Option<Float>,
    /// Learning rate (if applicable)
    pub learning_rate: Option<Float>,
    /// Additional metrics
    pub metrics: HashMap<String, Float>,
}

/// Trait for implementing custom embedding callbacks
pub trait EmbeddingCallback: Send + Sync + std::fmt::Debug {
    /// Called when an event occurs during training
    fn on_event(&mut self, event: &CallbackEvent, context: &CallbackContext) -> SklResult<()>;

    /// Called to check if training should stop early
    fn should_stop(&self) -> bool {
        false
    }

    /// Get callback name for identification
    fn name(&self) -> &str;
}

/// Progress monitoring callback that prints training progress
#[derive(Debug, Clone)]
pub struct ProgressCallback {
    print_every: usize,
    last_printed: usize,
}

impl ProgressCallback {
    /// Create a new progress callback
    pub fn new(print_every: usize) -> Self {
        Self {
            print_every,
            last_printed: 0,
        }
    }
}

impl EmbeddingCallback for ProgressCallback {
    fn on_event(&mut self, event: &CallbackEvent, context: &CallbackContext) -> SklResult<()> {
        match event {
            CallbackEvent::TrainingStarted => {
                println!("Training started...");
            }
            CallbackEvent::IterationCompleted { iteration, loss } => {
                if iteration - self.last_printed >= self.print_every {
                    println!("Iteration {}: loss = {:.6}", iteration, loss);
                    self.last_printed = *iteration;
                }
            }
            CallbackEvent::TrainingCompleted { final_loss } => {
                println!("Training completed with final loss: {:.6}", final_loss);
            }
            _ => {}
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "ProgressCallback"
    }
}

/// Early stopping callback based on loss improvement
#[derive(Debug, Clone)]
pub struct EarlyStoppingCallback {
    patience: usize,
    min_delta: Float,
    best_loss: Float,
    wait: usize,
    stopped: bool,
}

impl EarlyStoppingCallback {
    /// Create a new early stopping callback
    pub fn new(patience: usize, min_delta: Float) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: Float::INFINITY,
            wait: 0,
            stopped: false,
        }
    }
}

impl EmbeddingCallback for EarlyStoppingCallback {
    fn on_event(&mut self, event: &CallbackEvent, _context: &CallbackContext) -> SklResult<()> {
        if let CallbackEvent::IterationCompleted { loss, .. } = event {
            if *loss < self.best_loss - self.min_delta {
                self.best_loss = *loss;
                self.wait = 0;
            } else {
                self.wait += 1;
                if self.wait >= self.patience {
                    self.stopped = true;
                    println!(
                        "Early stopping triggered after {} iterations without improvement",
                        self.wait
                    );
                }
            }
        }
        Ok(())
    }

    fn should_stop(&self) -> bool {
        self.stopped
    }

    fn name(&self) -> &str {
        "EarlyStoppingCallback"
    }
}

/// Callback for saving intermediate embeddings
#[derive(Debug, Clone)]
pub struct SaveEmbeddingCallback {
    save_every: usize,
    save_path: String,
    last_saved: usize,
}

impl SaveEmbeddingCallback {
    /// Create a new save embedding callback
    pub fn new(save_every: usize, save_path: String) -> Self {
        Self {
            save_every,
            save_path,
            last_saved: 0,
        }
    }
}

impl EmbeddingCallback for SaveEmbeddingCallback {
    fn on_event(&mut self, event: &CallbackEvent, context: &CallbackContext) -> SklResult<()> {
        if let CallbackEvent::IterationCompleted { iteration, .. } = event {
            if iteration - self.last_saved >= self.save_every {
                // Here you would implement actual saving logic
                // For now, just print a message
                println!(
                    "Would save embedding at iteration {} to {}",
                    iteration, self.save_path
                );
                self.last_saved = *iteration;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "SaveEmbeddingCallback"
    }
}

/// Callback for collecting training metrics
#[derive(Debug, Clone)]
pub struct MetricsCallback {
    history: Vec<(usize, Float)>,
    custom_metrics: HashMap<String, Vec<Float>>,
}

impl MetricsCallback {
    /// Create a new metrics callback
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Get the loss history
    pub fn loss_history(&self) -> &[(usize, Float)] {
        &self.history
    }

    /// Get custom metrics history
    pub fn custom_metrics(&self) -> &HashMap<String, Vec<Float>> {
        &self.custom_metrics
    }
}

impl Default for MetricsCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingCallback for MetricsCallback {
    fn on_event(&mut self, event: &CallbackEvent, context: &CallbackContext) -> SklResult<()> {
        if let CallbackEvent::IterationCompleted { iteration, loss } = event {
            self.history.push((*iteration, *loss));

            // Collect custom metrics
            for (name, value) in &context.metrics {
                self.custom_metrics
                    .entry(name.clone())
                    .or_default()
                    .push(*value);
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "MetricsCallback"
    }
}

/// Manager for handling multiple callbacks
#[derive(Debug, Default)]
pub struct CallbackManager {
    callbacks: Vec<Box<dyn EmbeddingCallback>>,
}

impl CallbackManager {
    /// Create a new callback manager
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback to the manager
    pub fn add_callback(&mut self, callback: Box<dyn EmbeddingCallback>) {
        self.callbacks.push(callback);
    }

    /// Trigger an event for all callbacks
    pub fn trigger_event(
        &mut self,
        event: &CallbackEvent,
        context: &CallbackContext,
    ) -> SklResult<()> {
        for callback in &mut self.callbacks {
            callback.on_event(event, context)?;
        }
        Ok(())
    }

    /// Check if any callback requests early stopping
    pub fn should_stop(&self) -> bool {
        self.callbacks.iter().any(|callback| callback.should_stop())
    }

    /// Get names of all registered callbacks
    pub fn callback_names(&self) -> Vec<&str> {
        self.callbacks.iter().map(|c| c.name()).collect()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_callback() {
        let mut callback = ProgressCallback::new(10);
        let context = CallbackContext {
            iteration: 5,
            loss: 1.0,
            embedding: Array2::zeros((10, 2)),
            gradient_norm: Some(0.1),
            learning_rate: Some(0.01),
            metrics: HashMap::new(),
        };

        let event = CallbackEvent::IterationCompleted {
            iteration: 5,
            loss: 1.0,
        };
        assert!(callback.on_event(&event, &context).is_ok());
        assert!(!callback.should_stop());
        assert_eq!(callback.name(), "ProgressCallback");
    }

    #[test]
    fn test_early_stopping_callback() {
        let mut callback = EarlyStoppingCallback::new(3, 0.01);
        let context = CallbackContext {
            iteration: 1,
            loss: 1.0,
            embedding: Array2::zeros((10, 2)),
            gradient_norm: None,
            learning_rate: None,
            metrics: HashMap::new(),
        };

        // First iteration - should not stop
        let event = CallbackEvent::IterationCompleted {
            iteration: 1,
            loss: 1.0,
        };
        assert!(callback.on_event(&event, &context).is_ok());
        assert!(!callback.should_stop());

        // Multiple iterations without improvement
        for i in 2..=5 {
            let event = CallbackEvent::IterationCompleted {
                iteration: i,
                loss: 1.0,
            };
            assert!(callback.on_event(&event, &context).is_ok());
        }

        // Should trigger early stopping after patience iterations
        assert!(callback.should_stop());
    }

    #[test]
    fn test_metrics_callback() {
        let mut callback = MetricsCallback::new();
        let mut metrics = HashMap::new();
        metrics.insert("custom_metric".to_string(), 0.5);

        let context = CallbackContext {
            iteration: 1,
            loss: 1.0,
            embedding: Array2::zeros((10, 2)),
            gradient_norm: None,
            learning_rate: None,
            metrics,
        };

        let event = CallbackEvent::IterationCompleted {
            iteration: 1,
            loss: 1.0,
        };
        assert!(callback.on_event(&event, &context).is_ok());

        let history = callback.loss_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0], (1, 1.0));

        let custom_metrics = callback.custom_metrics();
        assert!(custom_metrics.contains_key("custom_metric"));
        assert_eq!(custom_metrics["custom_metric"][0], 0.5);
    }

    #[test]
    fn test_callback_manager() {
        let mut manager = CallbackManager::new();

        manager.add_callback(Box::new(ProgressCallback::new(10)));
        manager.add_callback(Box::new(EarlyStoppingCallback::new(5, 0.01)));

        let context = CallbackContext {
            iteration: 1,
            loss: 1.0,
            embedding: Array2::zeros((10, 2)),
            gradient_norm: None,
            learning_rate: None,
            metrics: HashMap::new(),
        };

        let event = CallbackEvent::IterationCompleted {
            iteration: 1,
            loss: 1.0,
        };
        assert!(manager.trigger_event(&event, &context).is_ok());
        assert!(!manager.should_stop());

        let names = manager.callback_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"ProgressCallback"));
        assert!(names.contains(&"EarlyStoppingCallback"));
    }
}
