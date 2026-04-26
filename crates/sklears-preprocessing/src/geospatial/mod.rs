//! Auto-generated module structure

pub mod functions;
pub mod trait_impls;
pub mod types;

// Re-export all types
pub use functions::*;
// trait_impls contains only trait implementations (no named exports), suppress warning
#[allow(unused_imports)]
pub use trait_impls::*;
pub use types::*;
