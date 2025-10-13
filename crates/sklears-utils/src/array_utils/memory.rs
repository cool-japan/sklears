//! Memory layout and optimization utilities

use scirs2_core::ndarray::Array2;

/// Check if array is contiguous in memory
pub fn is_contiguous<T>(array: &Array2<T>) -> bool {
    array.is_standard_layout()
}

/// Make array contiguous by copying if necessary
pub fn make_contiguous<T: Clone>(array: &Array2<T>) -> Array2<T> {
    if array.is_standard_layout() {
        array.clone()
    } else {
        array.to_owned()
    }
}

/// Efficient copy operation for arrays
pub fn efficient_copy<T: Clone>(array: &Array2<T>) -> Array2<T> {
    if array.is_standard_layout() {
        // Use faster clone for standard layout
        array.clone()
    } else {
        // Convert to standard layout first
        array.to_owned()
    }
}

/// Get memory strides for array
pub fn get_strides<T>(array: &Array2<T>) -> Vec<isize> {
    array.strides().to_vec()
}

/// Check if two arrays have compatible memory layout for operations
pub fn compatible_layout<T>(a: &Array2<T>, b: &Array2<T>) -> bool {
    a.raw_dim() == b.raw_dim() && a.strides() == b.strides()
}
