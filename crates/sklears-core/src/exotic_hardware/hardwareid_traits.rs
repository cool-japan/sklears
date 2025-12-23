//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::fmt;

impl fmt::Display for HardwareId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}-{}-{}",
            self.device_type, self.vendor, self.model, self.device_index
        )
    }
}
