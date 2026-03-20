//! # Dual - Trait Implementations
//!
//! This module contains trait implementations for `Dual`.
//!
//! ## Implemented Traits
//!
//! - `Add`
//! - `Sub`
//! - `Mul`
//! - `Div`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Dual;

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, other: Dual) -> Self::Output {
        Dual::new(self.value + other.value, self.derivative + other.derivative)
    }
}

impl std::ops::Sub for Dual {
    type Output = Dual;
    fn sub(self, other: Dual) -> Self::Output {
        Dual::new(self.value - other.value, self.derivative - other.derivative)
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, other: Dual) -> Self::Output {
        Dual::new(
            self.value * other.value,
            self.derivative * other.value + self.value * other.derivative,
        )
    }
}

impl std::ops::Div for Dual {
    type Output = Dual;
    fn div(self, other: Dual) -> Self::Output {
        Dual::new(
            self.value / other.value,
            (self.derivative * other.value - self.value * other.derivative)
                / (other.value * other.value),
        )
    }
}
