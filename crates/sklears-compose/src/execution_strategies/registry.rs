//! Strategy builder, factory, and registry for managing strategy instances.

use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::time::SystemTime;

use super::batch::BatchExecutionStrategy;
use super::sequential::SequentialExecutionStrategy;
use super::core::{ExecutionStrategy, StrategyConfig};


/// Strategy builder for creating strategies with configuration
#[allow(dead_code)]
pub struct StrategyBuilder {
    pub(super) strategy_type: StrategyType,
    pub(super) config: StrategyConfig,
}
/// Strategy factory for creating strategy instances
pub struct StrategyFactory;
impl StrategyFactory {
    /// Create a new strategy instance
    pub fn create_strategy(
        strategy_type: StrategyType,
        config: StrategyConfig,
    ) -> SklResult<Box<dyn ExecutionStrategy>> {
        match strategy_type {
            StrategyType::Sequential => {
                let mut strategy = SequentialExecutionStrategy::new();
                strategy.config = config;
                Ok(Box::new(strategy))
            }
            StrategyType::Batch => {
                let mut strategy = BatchExecutionStrategy::new();
                strategy.config = config;
                Ok(Box::new(strategy))
            }
            _ => {
                Err(
                    SklearsError::NotImplemented(
                        "Strategy type not implemented".to_string(),
                    ),
                )
            }
        }
    }
    /// Get available strategy types
    #[must_use]
    pub fn available_strategies() -> Vec<StrategyType> {
        vec![
            StrategyType::Sequential, StrategyType::Batch, StrategyType::Streaming,
            StrategyType::Gpu, StrategyType::Distributed, StrategyType::EventDriven,
        ]
    }
}
/// Strategy metadata
#[derive(Debug, Clone)]
pub struct StrategyMetadata {
    /// Strategy name
    pub name: String,
    /// Strategy description
    pub description: String,
    /// Strategy version
    pub version: String,
    /// Author
    pub author: String,
    /// Creation date
    pub created_at: SystemTime,
    /// Tags
    pub tags: Vec<String>,
}
/// Strategy registry for managing multiple strategies
#[derive(Debug)]
pub struct StrategyRegistry {
    /// Registered strategies
    pub(super) strategies: HashMap<String, Box<dyn ExecutionStrategy>>,
    /// Default strategy
    pub(super) default_strategy: Option<String>,
    /// Strategy metadata
    pub(super) metadata: HashMap<String, StrategyMetadata>,
}
impl StrategyRegistry {
    /// Create a new strategy registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            default_strategy: None,
            metadata: HashMap::new(),
        }
    }
    /// Register a strategy
    pub fn register(
        &mut self,
        name: String,
        strategy: Box<dyn ExecutionStrategy>,
    ) -> SklResult<()> {
        self.strategies.insert(name.clone(), strategy);
        self.metadata
            .insert(
                name.clone(),
                StrategyMetadata {
                    name: name.clone(),
                    description: format!("Strategy: {name}"),
                    version: "1.0.0".to_string(),
                    author: "SkleaRS".to_string(),
                    created_at: SystemTime::now(),
                    tags: Vec::new(),
                },
            );
        Ok(())
    }
    /// Get a strategy by name
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn ExecutionStrategy> {
        self.strategies.get(name).map(|b| b.as_ref())
    }
    /// List all registered strategies
    #[must_use]
    pub fn list(&self) -> Vec<String> {
        self.strategies.keys().cloned().collect()
    }
    /// Set default strategy
    pub fn set_default(&mut self, name: String) -> SklResult<()> {
        if self.strategies.contains_key(&name) {
            self.default_strategy = Some(name);
            Ok(())
        } else {
            Err(SklearsError::InvalidInput(format!("Strategy {name} not found")))
        }
    }
    /// Get default strategy name
    #[must_use]
    pub fn get_default(&self) -> Option<&String> {
        self.default_strategy.as_ref()
    }
}
impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}
/// Strategy types
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyType {
    /// Sequential
    Sequential,
    /// Batch
    Batch,
    /// Streaming
    Streaming,
    /// Gpu
    Gpu,
    /// Distributed
    Distributed,
    /// EventDriven
    EventDriven,
}
