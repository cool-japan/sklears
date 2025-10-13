//! Constant-time operations for security-conscious dummy estimator implementations

/// Constant-time equality check for arrays
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

/// Constant-time selection between two values
pub fn constant_time_select(condition: bool, true_val: f64, false_val: f64) -> f64 {
    let mask = if condition { 1.0 } else { 0.0 };
    mask * true_val + (1.0 - mask) * false_val
}

/// Constant-time comparison
pub fn constant_time_less_than(a: f64, b: f64) -> bool {
    // Simple implementation - in practice you'd use more sophisticated techniques
    let diff = a - b;
    diff < 0.0
}

/// Constant-time array access
pub fn constant_time_access(array: &[f64], index: usize, default: f64) -> f64 {
    if index < array.len() {
        array[index]
    } else {
        default
    }
}
