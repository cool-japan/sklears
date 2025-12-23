//! # ResourceContext - Trait Implementations
//!
//! This module contains trait implementations for `ResourceContext`.
//!
//! ## Implemented Traits
//!
//! - `ExecutionContextTrait`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl ExecutionContextTrait for ResourceContext {
    fn id(&self) -> &str {
        &self.context_id
    }
    fn context_type(&self) -> ContextType {
        ContextType::Extension("resource".to_string())
    }
    fn state(&self) -> ContextState {
        *self.state.read().unwrap()
    }
    fn is_active(&self) -> bool {
        matches!(self.state(), ContextState::Active)
    }
    fn metadata(&self) -> &ContextMetadata {
        unsafe { &*(self.metadata.read().unwrap().as_ref() as *const ContextMetadata) }
    }
    fn validate(&self) -> Result<(), ContextError> {
        self.check_resource_limits()
    }
    fn clone_with_id(
        &self,
        new_id: String,
    ) -> Result<Box<dyn ExecutionContextTrait>, ContextError> {
        let config = self.get_config()?;
        let new_context = ResourceContext::with_config(new_id, config)?;
        Ok(Box::new(new_context))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

