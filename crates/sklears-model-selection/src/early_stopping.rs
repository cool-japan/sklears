//! Early stopping strategies for hyperparameter optimization
//!
//! This module implements various early stopping criteria that can be used to terminate
//! optimization algorithms early when certain conditions are met, saving computational
//! resources while maintaining optimization quality.

use sklears_core::error::Result;
use std::collections::VecDeque;

/// Early stopping criterion configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Minimum number of iterations before early stopping can trigger
    pub min_iterations: usize,
    /// Patience: number of iterations to wait without improvement
    pub patience: usize,
    /// Minimum improvement threshold to reset patience
    pub min_delta: f64,
    /// Whether to restore best weights when stopping
    pub restore_best_weights: bool,
    /// Whether to monitor for maximum or minimum (true for max, false for min)
    pub maximize: bool,
    /// Baseline score to compare against
    pub baseline: Option<f64>,
    /// Smoothing factor for exponential moving average
    pub smoothing_factor: f64,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            min_iterations: 10,
            patience: 10,
            min_delta: 1e-4,
            restore_best_weights: true,
            maximize: true,
            baseline: None,
            smoothing_factor: 0.9,
        }
    }
}

/// Different early stopping strategies
#[derive(Debug, Clone)]
pub enum EarlyStoppingStrategy {
    /// Stop when no improvement for patience iterations
    Patience,
    /// Stop when improvement rate falls below threshold
    ImprovementRate(f64),
    /// Stop when exponential moving average converges
    ExponentialMovingAverage,
    /// Stop when validation loss starts increasing (overfitting detection)
    ValidationLoss,
    /// Stop when relative improvement becomes small
    RelativeImprovement(f64),
    /// Stop when absolute improvement becomes small
    AbsoluteImprovement(f64),
    /// Combination of multiple strategies (OR logic)
    Combined(Vec<EarlyStoppingStrategy>),
}

/// Early stopping state tracker
#[derive(Debug, Clone)]
pub struct EarlyStoppingState {
    /// Best score seen so far
    pub best_score: f64,
    /// Iteration when best score was achieved
    pub best_iteration: usize,
    /// Number of iterations without improvement
    pub patience_counter: usize,
    /// All scores seen so far
    pub score_history: Vec<f64>,
    /// Exponential moving average of scores
    pub ema_score: f64,
    /// Whether EMA has been initialized
    pub ema_initialized: bool,
    /// Recent scores for trend analysis
    pub recent_scores: VecDeque<f64>,
    /// Maximum window size for recent scores
    pub window_size: usize,
}

impl EarlyStoppingState {
    fn new(window_size: usize) -> Self {
        Self {
            best_score: f64::NEG_INFINITY,
            best_iteration: 0,
            patience_counter: 0,
            score_history: Vec::new(),
            ema_score: 0.0,
            ema_initialized: false,
            recent_scores: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    fn update(&mut self, score: f64, iteration: usize, config: &EarlyStoppingConfig) {
        self.score_history.push(score);

        // Update recent scores window
        if self.recent_scores.len() >= self.window_size {
            self.recent_scores.pop_front();
        }
        self.recent_scores.push_back(score);

        // Update exponential moving average
        if !self.ema_initialized {
            self.ema_score = score;
            self.ema_initialized = true;
        } else {
            self.ema_score =
                config.smoothing_factor * self.ema_score + (1.0 - config.smoothing_factor) * score;
        }

        // Check if this is the best score
        let is_improvement = if config.maximize {
            score > self.best_score + config.min_delta
        } else {
            score < self.best_score - config.min_delta
        };

        if is_improvement {
            self.best_score = score;
            self.best_iteration = iteration;
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
        }
    }

    fn improvement_rate(&self) -> f64 {
        if self.score_history.len() < 2 {
            return 0.0;
        }

        let recent_window = self.score_history.len().min(5);
        let recent_scores = &self.score_history[self.score_history.len() - recent_window..];

        if recent_scores.len() < 2 {
            return 0.0;
        }

        let start_score = recent_scores[0];
        let end_score = recent_scores[recent_scores.len() - 1];

        if start_score.abs() < 1e-8 {
            return 0.0;
        }

        (end_score - start_score) / start_score.abs()
    }

    fn relative_improvement(&self) -> f64 {
        if self.score_history.len() < 2 {
            return f64::INFINITY;
        }

        let current = self.score_history[self.score_history.len() - 1];
        let previous = self.score_history[self.score_history.len() - 2];

        if previous.abs() < 1e-8 {
            return f64::INFINITY;
        }

        (current - previous).abs() / previous.abs()
    }

    fn absolute_improvement(&self) -> f64 {
        if self.score_history.len() < 2 {
            return f64::INFINITY;
        }

        let current = self.score_history[self.score_history.len() - 1];
        let previous = self.score_history[self.score_history.len() - 2];

        (current - previous).abs()
    }

    fn ema_convergence(&self, threshold: f64) -> bool {
        if self.recent_scores.len() < self.window_size {
            return false;
        }

        let recent_avg: f64 =
            self.recent_scores.iter().sum::<f64>() / self.recent_scores.len() as f64;
        (self.ema_score - recent_avg).abs() < threshold
    }

    fn is_overfitting(&self, lookback: usize) -> bool {
        if self.score_history.len() < lookback + 2 {
            return false;
        }

        let len = self.score_history.len();
        let recent_scores = &self.score_history[len - lookback..];
        let previous_scores = &self.score_history[len - lookback - lookback..len - lookback];

        let recent_avg: f64 = recent_scores.iter().sum::<f64>() / recent_scores.len() as f64;
        let previous_avg: f64 = previous_scores.iter().sum::<f64>() / previous_scores.len() as f64;

        // For maximization: overfitting if recent scores are decreasing
        // For minimization: overfitting if recent scores are increasing
        recent_avg < previous_avg
    }
}

/// Early stopping monitor
pub struct EarlyStoppingMonitor {
    strategy: EarlyStoppingStrategy,
    config: EarlyStoppingConfig,
    state: EarlyStoppingState,
    current_iteration: usize,
}

impl EarlyStoppingMonitor {
    /// Create a new early stopping monitor
    pub fn new(strategy: EarlyStoppingStrategy, config: EarlyStoppingConfig) -> Self {
        Self {
            strategy,
            config,
            state: EarlyStoppingState::new(10), // Default window size
            current_iteration: 0,
        }
    }

    /// Update the monitor with a new score
    pub fn update(&mut self, score: f64) -> Result<()> {
        self.state
            .update(score, self.current_iteration, &self.config);
        self.current_iteration += 1;
        Ok(())
    }

    /// Check if early stopping criteria are met
    pub fn should_stop(&self) -> bool {
        if self.current_iteration < self.config.min_iterations {
            return false;
        }

        self.check_strategy(&self.strategy)
    }

    fn check_strategy(&self, strategy: &EarlyStoppingStrategy) -> bool {
        match strategy {
            EarlyStoppingStrategy::Patience => self.state.patience_counter > self.config.patience,
            EarlyStoppingStrategy::ImprovementRate(threshold) => {
                self.state.improvement_rate().abs() < *threshold
            }
            EarlyStoppingStrategy::ExponentialMovingAverage => {
                self.state.ema_convergence(self.config.min_delta)
            }
            EarlyStoppingStrategy::ValidationLoss => {
                self.state.is_overfitting(5) // Lookback of 5 iterations
            }
            EarlyStoppingStrategy::RelativeImprovement(threshold) => {
                self.state.relative_improvement() < *threshold
            }
            EarlyStoppingStrategy::AbsoluteImprovement(threshold) => {
                self.state.absolute_improvement() < *threshold
            }
            EarlyStoppingStrategy::Combined(strategies) => {
                strategies.iter().any(|s| self.check_strategy(s))
            }
        }
    }

    /// Get the current state
    pub fn state(&self) -> &EarlyStoppingState {
        &self.state
    }

    /// Get the best score and iteration
    pub fn best_result(&self) -> (f64, usize) {
        (self.state.best_score, self.state.best_iteration)
    }

    /// Reset the monitor
    pub fn reset(&mut self) {
        self.state = EarlyStoppingState::new(self.state.window_size);
        self.current_iteration = 0;
    }

    /// Check if minimum iterations have been reached
    pub fn min_iterations_reached(&self) -> bool {
        self.current_iteration >= self.config.min_iterations
    }

    /// Get convergence metrics
    pub fn convergence_metrics(&self) -> ConvergenceMetrics {
        ConvergenceMetrics {
            improvement_rate: self.state.improvement_rate(),
            relative_improvement: self.state.relative_improvement(),
            absolute_improvement: self.state.absolute_improvement(),
            patience_remaining: self
                .config
                .patience
                .saturating_sub(self.state.patience_counter),
            iterations_since_best: self
                .current_iteration
                .saturating_sub(self.state.best_iteration),
            ema_score: self.state.ema_score,
            current_score: self.state.score_history.last().copied().unwrap_or(0.0),
        }
    }
}

/// Convergence metrics for monitoring optimization progress
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Rate of improvement in recent iterations
    pub improvement_rate: f64,
    /// Relative improvement in the last iteration
    pub relative_improvement: f64,
    /// Absolute improvement in the last iteration
    pub absolute_improvement: f64,
    /// Number of patience iterations remaining
    pub patience_remaining: usize,
    /// Number of iterations since best score
    pub iterations_since_best: usize,
    /// Current exponential moving average score
    pub ema_score: f64,
    /// Most recent score
    pub current_score: f64,
}

/// Early stopping callback trait for use with optimizers
pub trait EarlyStoppingCallback {
    fn on_iteration(&mut self, score: f64) -> Result<bool>;

    fn on_early_stop(&mut self, reason: &str) -> Result<()>;

    fn best_score(&self) -> f64;

    fn convergence_info(&self) -> String;
}

impl EarlyStoppingCallback for EarlyStoppingMonitor {
    fn on_iteration(&mut self, score: f64) -> Result<bool> {
        self.update(score)?;
        Ok(self.should_stop())
    }

    fn on_early_stop(&mut self, _reason: &str) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    fn best_score(&self) -> f64 {
        self.state.best_score
    }

    fn convergence_info(&self) -> String {
        let metrics = self.convergence_metrics();
        format!(
            "Best: {:.6}, Current: {:.6}, Improvement Rate: {:.6}, Patience: {}/{}",
            self.state.best_score,
            metrics.current_score,
            metrics.improvement_rate,
            self.state.patience_counter,
            self.config.patience
        )
    }
}

/// Adaptive early stopping that adjusts parameters based on optimization progress
pub struct AdaptiveEarlyStopping {
    base_monitor: EarlyStoppingMonitor,
    adaptation_config: AdaptationConfig,
    adaptation_state: AdaptationState,
}

#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// How often to adapt parameters (in iterations)
    pub adaptation_frequency: usize,
    /// Factor to increase patience when making good progress
    pub patience_increase_factor: f64,
    /// Factor to decrease patience when making poor progress
    pub patience_decrease_factor: f64,
    /// Maximum patience allowed
    pub max_patience: usize,
    /// Minimum patience allowed
    pub min_patience: usize,
    /// Threshold for "good progress"
    pub good_progress_threshold: f64,
    /// Threshold for "poor progress"
    pub poor_progress_threshold: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            adaptation_frequency: 20,
            patience_increase_factor: 1.5,
            patience_decrease_factor: 0.8,
            max_patience: 50,
            min_patience: 5,
            good_progress_threshold: 0.01,
            poor_progress_threshold: 0.001,
        }
    }
}

#[derive(Debug, Clone)]
struct AdaptationState {
    last_adaptation_iteration: usize,
    adaptation_history: Vec<(usize, usize)>, // (iteration, patience)
}

impl AdaptiveEarlyStopping {
    /// Create a new adaptive early stopping monitor
    pub fn new(
        strategy: EarlyStoppingStrategy,
        config: EarlyStoppingConfig,
        adaptation_config: AdaptationConfig,
    ) -> Self {
        Self {
            base_monitor: EarlyStoppingMonitor::new(strategy, config),
            adaptation_config,
            adaptation_state: AdaptationState {
                last_adaptation_iteration: 0,
                adaptation_history: Vec::new(),
            },
        }
    }

    /// Update with adaptive behavior
    pub fn update_adaptive(&mut self, score: f64) -> Result<()> {
        self.base_monitor.update(score)?;

        // Check if it's time to adapt
        if self.base_monitor.current_iteration
            >= self.adaptation_state.last_adaptation_iteration
                + self.adaptation_config.adaptation_frequency
        {
            self.adapt_parameters();
        }

        Ok(())
    }

    fn adapt_parameters(&mut self) {
        let metrics = self.base_monitor.convergence_metrics();
        let current_patience = self.base_monitor.config.patience;

        let new_patience =
            if metrics.improvement_rate > self.adaptation_config.good_progress_threshold {
                // Good progress: increase patience
                let increased = (current_patience as f64
                    * self.adaptation_config.patience_increase_factor)
                    as usize;
                increased.min(self.adaptation_config.max_patience)
            } else if metrics.improvement_rate < self.adaptation_config.poor_progress_threshold {
                // Poor progress: decrease patience
                let decreased = (current_patience as f64
                    * self.adaptation_config.patience_decrease_factor)
                    as usize;
                decreased.max(self.adaptation_config.min_patience)
            } else {
                current_patience // No change
            };

        if new_patience != current_patience {
            self.base_monitor.config.patience = new_patience;
            self.adaptation_state
                .adaptation_history
                .push((self.base_monitor.current_iteration, new_patience));
        }

        self.adaptation_state.last_adaptation_iteration = self.base_monitor.current_iteration;
    }

    /// Get the underlying monitor
    pub fn monitor(&self) -> &EarlyStoppingMonitor {
        &self.base_monitor
    }

    /// Get the underlying monitor mutably
    pub fn monitor_mut(&mut self) -> &mut EarlyStoppingMonitor {
        &mut self.base_monitor
    }

    /// Get adaptation history
    pub fn adaptation_history(&self) -> &[(usize, usize)] {
        &self.adaptation_state.adaptation_history
    }
}

impl EarlyStoppingCallback for AdaptiveEarlyStopping {
    fn on_iteration(&mut self, score: f64) -> Result<bool> {
        self.update_adaptive(score)?;
        Ok(self.base_monitor.should_stop())
    }

    fn on_early_stop(&mut self, reason: &str) -> Result<()> {
        self.base_monitor.on_early_stop(reason)
    }

    fn best_score(&self) -> f64 {
        self.base_monitor.best_score()
    }

    fn convergence_info(&self) -> String {
        format!(
            "{} | Adaptations: {}",
            self.base_monitor.convergence_info(),
            self.adaptation_state.adaptation_history.len()
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_patience() {
        let config = EarlyStoppingConfig {
            min_iterations: 5,
            patience: 3,
            min_delta: 0.01,
            maximize: true,
            ..Default::default()
        };

        let mut monitor = EarlyStoppingMonitor::new(EarlyStoppingStrategy::Patience, config);

        // Should not stop before min_iterations
        for i in 0..5 {
            // Provide improving scores initially to avoid early patience trigger
            monitor.update(1.0 + i as f64 * 0.02).unwrap();
            assert!(!monitor.should_stop(), "Should not stop at iteration {}", i);
        }

        // Add scores without improvement
        monitor.update(1.0).unwrap(); // No improvement
        assert!(!monitor.should_stop());

        monitor.update(0.99).unwrap(); // Worse
        assert!(!monitor.should_stop());

        monitor.update(0.98).unwrap(); // Worse
        assert!(!monitor.should_stop());

        monitor.update(0.97).unwrap(); // Worse - should trigger early stopping
        assert!(monitor.should_stop());
    }

    #[test]
    fn test_early_stopping_improvement_rate() {
        let config = EarlyStoppingConfig {
            min_iterations: 3,
            maximize: true,
            ..Default::default()
        };

        let mut monitor = EarlyStoppingMonitor::new(
            EarlyStoppingStrategy::ImprovementRate(0.01), // 1% threshold
            config,
        );

        // Rapid improvement
        monitor.update(1.0).unwrap();
        monitor.update(1.1).unwrap();
        monitor.update(1.2).unwrap();
        assert!(!monitor.should_stop()); // Good improvement rate

        // Fill the window with consistently slow improvement
        // Starting from 1.2, add small increments that result in overall rate < 0.01
        monitor.update(1.2001).unwrap(); // Very small improvement
        monitor.update(1.2002).unwrap(); // Very small improvement
        monitor.update(1.2003).unwrap(); // Very small improvement
                                         // Now the window is [1.1, 1.2, 1.2001, 1.2002, 1.2003]
                                         // Improvement rate = (1.2003 - 1.1) / 1.1 = 0.0913... > 0.01, still too high

        // Need to replace the good improvement scores in the window
        monitor.update(1.2003).unwrap(); // No improvement
        monitor.update(1.2003).unwrap(); // No improvement
                                         // Now window is [1.2, 1.2001, 1.2002, 1.2003, 1.2003]
                                         // Improvement rate = (1.2003 - 1.2) / 1.2 = 0.00025 < 0.01
        assert!(monitor.should_stop()); // Should stop with poor improvement rate
    }

    #[test]
    fn test_early_stopping_combined_strategy() {
        let config = EarlyStoppingConfig {
            min_iterations: 2,
            patience: 5,
            maximize: true,
            ..Default::default()
        };

        let strategy = EarlyStoppingStrategy::Combined(vec![
            EarlyStoppingStrategy::Patience,
            EarlyStoppingStrategy::ImprovementRate(0.001),
        ]);

        let mut monitor = EarlyStoppingMonitor::new(strategy, config);

        monitor.update(1.0).unwrap();
        monitor.update(1.0001).unwrap(); // Tiny improvement
        monitor.update(1.0002).unwrap(); // Tiny improvement

        // Should stop due to low improvement rate, even though patience hasn't run out
        assert!(monitor.should_stop());
    }

    #[test]
    fn test_convergence_metrics() {
        let config = EarlyStoppingConfig::default();
        let mut monitor = EarlyStoppingMonitor::new(EarlyStoppingStrategy::Patience, config);

        monitor.update(1.0).unwrap();
        monitor.update(1.1).unwrap();
        monitor.update(1.05).unwrap();

        let metrics = monitor.convergence_metrics();
        assert!(metrics.improvement_rate.is_finite());
        assert!(metrics.relative_improvement >= 0.0);
        assert_eq!(
            metrics.patience_remaining,
            monitor.config.patience - monitor.state.patience_counter
        );
    }

    #[test]
    fn test_adaptive_early_stopping() {
        let config = EarlyStoppingConfig {
            min_iterations: 5,
            patience: 10,
            maximize: true,
            ..Default::default()
        };

        let adaptation_config = AdaptationConfig {
            adaptation_frequency: 5,
            good_progress_threshold: 0.1,
            poor_progress_threshold: 0.01,
            ..Default::default()
        };

        let mut adaptive =
            AdaptiveEarlyStopping::new(EarlyStoppingStrategy::Patience, config, adaptation_config);

        // Good progress should increase patience
        for i in 0..10 {
            adaptive.update_adaptive(1.0 + i as f64 * 0.2).unwrap();
        }

        assert!(adaptive.monitor().config.patience > 10); // Should have increased
        assert!(!adaptive.adaptation_history().is_empty());
    }

    #[test]
    fn test_early_stopping_callback() {
        let config = EarlyStoppingConfig {
            min_iterations: 2,
            patience: 2,
            maximize: true,
            min_delta: 0.0, // No minimum delta required for improvement
            ..Default::default()
        };

        let mut monitor = EarlyStoppingMonitor::new(EarlyStoppingStrategy::Patience, config);

        assert!(!monitor.on_iteration(1.0).unwrap()); // iteration 1: best = 1.0, patience = 0
        assert!(!monitor.on_iteration(1.0).unwrap()); // iteration 2: no improvement, patience = 1
        assert!(!monitor.on_iteration(0.9).unwrap()); // iteration 3: no improvement, patience = 2
        assert!(monitor.on_iteration(0.8).unwrap()); // iteration 4: no improvement, patience = 3, should stop (patience_counter > 2)

        assert_eq!(monitor.best_score(), 1.0);
        assert!(monitor.convergence_info().contains("Best: 1.000000"));
    }
}
