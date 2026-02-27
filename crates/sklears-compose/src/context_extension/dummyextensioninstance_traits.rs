//! # DummyExtensionInstance - Trait Implementations
//!
//! This module contains trait implementations for `DummyExtensionInstance`.
//!
//! ## Implemented Traits
//!
//! - `ExtensionInstance`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::context_core::{
    ExecutionContextTrait, ContextType, ContextState, ContextError, ContextResult,
    ContextMetadata, ContextEvent,
};

use super::functions::ExtensionInstance;
use super::types::{DummyExtensionInstance, ExtensionConfiguration, ExtensionContext, ExtensionInfo, ExtensionInput, ExtensionOutput};

impl ExtensionInstance for DummyExtensionInstance {
    fn initialize(&mut self, _context: &ExtensionContext) -> ContextResult<()> {
        Ok(())
    }
    fn start(&mut self) -> ContextResult<()> {
        Ok(())
    }
    fn stop(&mut self) -> ContextResult<()> {
        Ok(())
    }
    fn execute(&mut self, _input: &ExtensionInput) -> ContextResult<ExtensionOutput> {
        Ok(ExtensionOutput {
            success: true,
            data: None,
            metadata: HashMap::new(),
            error: None,
            execution_time: Duration::from_millis(10),
        })
    }
    fn info(&self) -> &ExtensionInfo {
        &self.info
    }
    fn configure(&mut self, _config: &ExtensionConfiguration) -> ContextResult<()> {
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

