//! Performance-critical unsafe optimizations
//!
//! This module contains unsafe code for performance-critical paths in kernel
//! approximation. All unsafe code is carefully reviewed and documented.
//!
//! # Safety
//!
//! All functions in this module that use unsafe code have detailed safety
//! documentation explaining the invariants that must be upheld.

use scirs2_core::ndarray::{Array1, Array2};

/// Unsafe dot product with manual loop unrolling
///
/// # Safety
///
/// - `a` and `b` must have the same length
/// - All elements must be valid f64 values (not NaN that could cause UB in comparisons)
///
/// # Performance
///
/// This function uses manual loop unrolling to improve performance by:
/// - Reducing loop overhead
/// - Enabling better instruction-level parallelism
/// - Improving CPU pipeline utilization
#[inline]
pub unsafe fn dot_product_unrolled(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        sum0 += *a_ptr.add(idx) * *b_ptr.add(idx);
        sum1 += *a_ptr.add(idx + 1) * *b_ptr.add(idx + 1);
        sum2 += *a_ptr.add(idx + 2) * *b_ptr.add(idx + 2);
        sum3 += *a_ptr.add(idx + 3) * *b_ptr.add(idx + 3);
    }

    // Handle remainder
    let mut sum_remainder = 0.0;
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        sum_remainder += *a_ptr.add(idx) * *b_ptr.add(idx);
    }

    sum0 + sum1 + sum2 + sum3 + sum_remainder
}

/// Fast matrix-vector multiplication using unsafe optimizations
///
/// Computes `result = matrix * vector`
///
/// # Safety
///
/// - `matrix` must have the same number of columns as `vector` has elements
/// - `result` must have the same length as `matrix` has rows
/// - All slices must be properly aligned and valid
#[inline]
pub unsafe fn matvec_multiply_fast(matrix: &Array2<f64>, vector: &[f64], result: &mut [f64]) {
    let (n_rows, n_cols) = matrix.dim();
    debug_assert_eq!(n_cols, vector.len(), "Dimension mismatch");
    debug_assert_eq!(n_rows, result.len(), "Result size mismatch");

    let matrix_ptr = matrix.as_ptr();
    let vector_ptr = vector.as_ptr();
    let result_ptr = result.as_mut_ptr();

    for i in 0..n_rows {
        let row_offset = i * n_cols;
        let mut sum = 0.0;

        // Manual loop unrolling for better performance
        let chunks = n_cols / 4;
        let remainder = n_cols % 4;

        for j in 0..chunks {
            let idx = j * 4;
            sum += *matrix_ptr.add(row_offset + idx) * *vector_ptr.add(idx);
            sum += *matrix_ptr.add(row_offset + idx + 1) * *vector_ptr.add(idx + 1);
            sum += *matrix_ptr.add(row_offset + idx + 2) * *vector_ptr.add(idx + 2);
            sum += *matrix_ptr.add(row_offset + idx + 3) * *vector_ptr.add(idx + 3);
        }

        for j in 0..remainder {
            let idx = chunks * 4 + j;
            sum += *matrix_ptr.add(row_offset + idx) * *vector_ptr.add(idx);
        }

        *result_ptr.add(i) = sum;
    }
}

/// Fast element-wise operations with unrolled loops
///
/// # Safety
///
/// - All slices must have the same length
/// - Output slice must be valid for writing
#[inline]
pub unsafe fn elementwise_op_fast<F>(a: &[f64], b: &[f64], out: &mut [f64], mut op: F)
where
    F: FnMut(f64, f64) -> f64,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        *out_ptr.add(idx) = op(*a_ptr.add(idx), *b_ptr.add(idx));
        *out_ptr.add(idx + 1) = op(*a_ptr.add(idx + 1), *b_ptr.add(idx + 1));
        *out_ptr.add(idx + 2) = op(*a_ptr.add(idx + 2), *b_ptr.add(idx + 2));
        *out_ptr.add(idx + 3) = op(*a_ptr.add(idx + 3), *b_ptr.add(idx + 3));
    }

    // Handle remainder
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        *out_ptr.add(idx) = op(*a_ptr.add(idx), *b_ptr.add(idx));
    }
}

/// Fast exponential computation for RBF kernels
///
/// Computes exp(-gamma * ||x - y||^2) for kernel matrices
///
/// # Safety
///
/// - Input and output slices must be properly sized
/// - gamma must be a valid f64 value
#[inline]
pub unsafe fn rbf_kernel_fast(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    debug_assert_eq!(x.len(), y.len());

    let len = x.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    // Compute squared distance with loop unrolling
    for i in 0..chunks {
        let idx = i * 4;
        let diff0 = *x_ptr.add(idx) - *y_ptr.add(idx);
        let diff1 = *x_ptr.add(idx + 1) - *y_ptr.add(idx + 1);
        let diff2 = *x_ptr.add(idx + 2) - *y_ptr.add(idx + 2);
        let diff3 = *x_ptr.add(idx + 3) - *y_ptr.add(idx + 3);

        sum0 += diff0 * diff0;
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum3 += diff3 * diff3;
    }

    let mut sum_remainder = 0.0;
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        let diff = *x_ptr.add(idx) - *y_ptr.add(idx);
        sum_remainder += diff * diff;
    }

    let squared_dist = sum0 + sum1 + sum2 + sum3 + sum_remainder;
    (-gamma * squared_dist).exp()
}

/// Safe wrapper for dot product with bounds checking
#[inline]
pub fn safe_dot_product(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }

    // Check for NaN values
    if a.iter().any(|x| x.is_nan()) || b.iter().any(|x| x.is_nan()) {
        return None;
    }

    Some(unsafe { dot_product_unrolled(a, b) })
}

/// Safe wrapper for matrix-vector multiplication
#[inline]
pub fn safe_matvec_multiply(matrix: &Array2<f64>, vector: &Array1<f64>) -> Option<Array1<f64>> {
    let (n_rows, n_cols) = matrix.dim();
    if n_cols != vector.len() {
        return None;
    }

    let mut result = Array1::zeros(n_rows);
    unsafe {
        matvec_multiply_fast(
            matrix,
            vector.as_slice().unwrap(),
            result.as_slice_mut().unwrap(),
        );
    }
    Some(result)
}

/// Batch RBF kernel computation with unsafe optimizations
///
/// # Safety
///
/// - All matrices must have compatible dimensions
/// - gamma must be a valid positive f64
pub unsafe fn batch_rbf_kernel_fast(
    x_matrix: &Array2<f64>,
    y_matrix: &Array2<f64>,
    gamma: f64,
    output: &mut Array2<f64>,
) {
    let (n_x, d_x) = x_matrix.dim();
    let (n_y, d_y) = y_matrix.dim();
    let (out_rows, out_cols) = output.dim();

    debug_assert_eq!(d_x, d_y, "Feature dimensions must match");
    debug_assert_eq!(out_rows, n_x, "Output rows mismatch");
    debug_assert_eq!(out_cols, n_y, "Output cols mismatch");

    let x_ptr = x_matrix.as_ptr();
    let y_ptr = y_matrix.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..n_x {
        for j in 0..n_y {
            let mut squared_dist = 0.0;

            let x_offset = i * d_x;
            let y_offset = j * d_y;

            // Compute squared Euclidean distance
            for k in 0..d_x {
                let diff = *x_ptr.add(x_offset + k) - *y_ptr.add(y_offset + k);
                squared_dist += diff * diff;
            }

            *out_ptr.add(i * n_y + j) = (-gamma * squared_dist).exp();
        }
    }
}

/// Fast cosine features computation for Random Fourier Features
///
/// # Safety
///
/// - All arrays must be properly sized
/// - No aliasing between input and output
#[inline]
pub unsafe fn fast_cosine_features(
    projection: &[f64],
    offset: &[f64],
    scale: f64,
    output: &mut [f64],
) {
    debug_assert_eq!(projection.len(), offset.len());
    debug_assert_eq!(projection.len(), output.len());

    let len = projection.len();
    let proj_ptr = projection.as_ptr();
    let offset_ptr = offset.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..len {
        let val = *proj_ptr.add(i) + *offset_ptr.add(i);
        *out_ptr.add(i) = scale * val.cos();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_safe_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = safe_dot_product(&a, &b).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_safe_dot_product_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];

        assert!(safe_dot_product(&a, &b).is_none());
    }

    #[test]
    fn test_safe_dot_product_nan() {
        let a = vec![1.0, f64::NAN, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert!(safe_dot_product(&a, &b).is_none());
    }

    #[test]
    fn test_safe_matvec_multiply() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let vector = array![5.0, 6.0];

        let result = safe_matvec_multiply(&matrix, &vector).unwrap();
        assert_eq!(result[0], 17.0); // 1*5 + 2*6
        assert_eq!(result[1], 39.0); // 3*5 + 4*6
    }

    #[test]
    fn test_unsafe_rbf_kernel() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let gamma = 0.5;

        let result = unsafe { rbf_kernel_fast(&x, &y, gamma) };
        assert!((result - 1.0).abs() < 1e-10); // Same vectors should give 1.0
    }

    #[test]
    fn test_unsafe_rbf_kernel_different() {
        let x = vec![0.0, 0.0];
        let y = vec![1.0, 0.0];
        let gamma = 0.5;

        let result = unsafe { rbf_kernel_fast(&x, &y, gamma) };
        let expected = (-gamma * 1.0).exp(); // squared distance is 1.0
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fast_cosine_features() {
        let projection = vec![0.0, std::f64::consts::PI / 2.0];
        let offset = vec![0.0, 0.0];
        let scale = 1.0;
        let mut output = vec![0.0; 2];

        unsafe {
            fast_cosine_features(&projection, &offset, scale, &mut output);
        }

        assert!((output[0] - 1.0).abs() < 1e-10);
        assert!(output[1].abs() < 1e-10);
    }

    #[test]
    fn test_elementwise_op() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; 5];

        unsafe {
            elementwise_op_fast(&a, &b, &mut out, |x, y| x + y);
        }

        assert_eq!(out, vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    }
}
