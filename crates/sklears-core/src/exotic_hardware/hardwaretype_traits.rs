//! Auto-generated trait implementations
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::fmt;

impl fmt::Display for HardwareType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HardwareType::TPU => write!(f, "TPU"),
            HardwareType::FPGA => write!(f, "FPGA"),
            HardwareType::Quantum => write!(f, "Quantum"),
            HardwareType::CustomASIC => write!(f, "Custom ASIC"),
            HardwareType::Neuromorphic => write!(f, "Neuromorphic"),
            HardwareType::Optical => write!(f, "Optical"),
        }
    }
}
