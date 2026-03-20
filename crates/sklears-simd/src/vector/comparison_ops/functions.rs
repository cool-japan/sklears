//! Auto-generated module
//!
//! Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// Import ARM64 feature detection macro
#[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
use std::arch::is_aarch64_feature_detected;

/// SIMD-optimized element-wise equality comparison
///
/// Computes result\[i\] = a\[i\] == b\[i\] for all elements using SIMD instructions.
/// Returns a boolean mask where true indicates elements are equal.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must have same length as `a`)
/// * `result` - Output boolean vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::eq_vec;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![1.0, 3.0, 3.0, 5.0];
/// let mut result = vec![false; 4];
///
/// eq_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![true, false, true, false]);
/// ```
pub fn eq_vec(a: &[f32], b: &[f32], result: &mut [bool]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );
    if a.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { eq_vec_avx512(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { eq_vec_avx2(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { eq_vec_sse2(a, b, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { eq_vec_neon(a, b, result) };
            return;
        }
    }
    eq_vec_scalar(a, b, result);
}
/// SIMD-optimized element-wise not-equal comparison
///
/// Computes result\[i\] = a\[i\] != b\[i\] for all elements using SIMD instructions.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must have same length as `a`)
/// * `result` - Output boolean vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::ne_vec;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![1.0, 3.0, 3.0, 5.0];
/// let mut result = vec![false; 4];
///
/// ne_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![false, true, false, true]);
/// ```
pub fn ne_vec(a: &[f32], b: &[f32], result: &mut [bool]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );
    if a.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { ne_vec_avx512(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { ne_vec_avx2(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { ne_vec_sse2(a, b, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { ne_vec_neon(a, b, result) };
            return;
        }
    }
    ne_vec_scalar(a, b, result);
}
/// SIMD-optimized element-wise less-than comparison
///
/// Computes result\[i\] = a\[i\] < b\[i\] for all elements using SIMD instructions.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must have same length as `a`)
/// * `result` - Output boolean vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::lt_vec;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![2.0, 2.0, 2.0, 3.0];
/// let mut result = vec![false; 4];
///
/// lt_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![true, false, false, false]);
/// ```
pub fn lt_vec(a: &[f32], b: &[f32], result: &mut [bool]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );
    if a.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { lt_vec_avx512(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { lt_vec_avx2(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { lt_vec_sse2(a, b, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { lt_vec_neon(a, b, result) };
            return;
        }
    }
    lt_vec_scalar(a, b, result);
}
/// SIMD-optimized element-wise less-than-or-equal comparison
///
/// Computes result\[i\] = a\[i\] <= b\[i\] for all elements using SIMD instructions.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must have same length as `a`)
/// * `result` - Output boolean vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::le_vec;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![2.0, 2.0, 2.0, 3.0];
/// let mut result = vec![false; 4];
///
/// le_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![true, true, false, false]);
/// ```
pub fn le_vec(a: &[f32], b: &[f32], result: &mut [bool]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );
    if a.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { le_vec_avx512(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { le_vec_avx2(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { le_vec_sse2(a, b, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { le_vec_neon(a, b, result) };
            return;
        }
    }
    le_vec_scalar(a, b, result);
}
/// SIMD-optimized element-wise greater-than comparison
///
/// Computes result\[i\] = a\[i\] > b\[i\] for all elements using SIMD instructions.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must have same length as `a`)
/// * `result` - Output boolean vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::gt_vec;
///
/// let a = vec![3.0, 2.0, 1.0, 4.0];
/// let b = vec![2.0, 2.0, 2.0, 3.0];
/// let mut result = vec![false; 4];
///
/// gt_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![true, false, false, true]);
/// ```
pub fn gt_vec(a: &[f32], b: &[f32], result: &mut [bool]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );
    if a.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { gt_vec_avx512(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { gt_vec_avx2(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { gt_vec_sse2(a, b, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { gt_vec_neon(a, b, result) };
            return;
        }
    }
    gt_vec_scalar(a, b, result);
}
/// SIMD-optimized element-wise greater-than-or-equal comparison
///
/// Computes result\[i\] = a\[i\] >= b\[i\] for all elements using SIMD instructions.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must have same length as `a`)
/// * `result` - Output boolean vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::ge_vec;
///
/// let a = vec![3.0, 2.0, 1.0, 4.0];
/// let b = vec![2.0, 2.0, 2.0, 3.0];
/// let mut result = vec![false; 4];
///
/// ge_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![true, true, false, true]);
/// ```
pub fn ge_vec(a: &[f32], b: &[f32], result: &mut [bool]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );
    if a.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { ge_vec_avx512(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { ge_vec_avx2(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { ge_vec_sse2(a, b, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { ge_vec_neon(a, b, result) };
            return;
        }
    }
    ge_vec_scalar(a, b, result);
}
/// SIMD-optimized element-wise logical AND operation
///
/// Computes result\[i\] = a\[i\] && b\[i\] for all boolean elements.
/// This is typically used to combine boolean masks from comparison operations.
///
/// # Arguments
/// * `a` - First input boolean vector
/// * `b` - Second input boolean vector (must have same length as `a`)
/// * `result` - Output boolean vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::and_vec;
///
/// let a = vec![true, false, true, false];
/// let b = vec![true, true, false, false];
/// let mut result = vec![false; 4];
///
/// and_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![true, false, false, false]);
/// ```
pub fn and_vec(a: &[bool], b: &[bool], result: &mut [bool]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );
    if a.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { and_vec_avx512(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { and_vec_avx2(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { and_vec_sse2(a, b, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { and_vec_neon(a, b, result) };
            return;
        }
    }
    and_vec_scalar(a, b, result);
}
/// SIMD-optimized element-wise logical OR operation
///
/// Computes result\[i\] = a\[i\] || b\[i\] for all boolean elements.
///
/// # Arguments
/// * `a` - First input boolean vector
/// * `b` - Second input boolean vector (must have same length as `a`)
/// * `result` - Output boolean vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::or_vec;
///
/// let a = vec![true, false, true, false];
/// let b = vec![true, true, false, false];
/// let mut result = vec![false; 4];
///
/// or_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![true, true, true, false]);
/// ```
pub fn or_vec(a: &[bool], b: &[bool], result: &mut [bool]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );
    if a.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { or_vec_avx512(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { or_vec_avx2(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { or_vec_sse2(a, b, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { or_vec_neon(a, b, result) };
            return;
        }
    }
    or_vec_scalar(a, b, result);
}
/// SIMD-optimized element-wise logical XOR operation
///
/// Computes result\[i\] = a\[i\] ^ b\[i\] for all boolean elements.
///
/// # Arguments
/// * `a` - First input boolean vector
/// * `b` - Second input boolean vector (must have same length as `a`)
/// * `result` - Output boolean vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::xor_vec;
///
/// let a = vec![true, false, true, false];
/// let b = vec![true, true, false, false];
/// let mut result = vec![false; 4];
///
/// xor_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![false, true, true, false]);
/// ```
pub fn xor_vec(a: &[bool], b: &[bool], result: &mut [bool]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );
    if a.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { xor_vec_avx512(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { xor_vec_avx2(a, b, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { xor_vec_sse2(a, b, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { xor_vec_neon(a, b, result) };
            return;
        }
    }
    xor_vec_scalar(a, b, result);
}
/// SIMD-optimized element-wise logical NOT operation
///
/// Computes result\[i\] = !input\[i\] for all boolean elements.
///
/// # Arguments
/// * `input` - Input boolean vector
/// * `result` - Output boolean vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::comparison_ops::not_vec;
///
/// let input = vec![true, false, true, false];
/// let mut result = vec![false; 4];
///
/// not_vec(&input, &mut result);
/// assert_eq!(result, vec![false, true, false, true]);
/// ```
pub fn not_vec(input: &[bool], result: &mut [bool]) {
    assert_eq!(
        input.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );
    if input.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { not_vec_avx512(input, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { not_vec_avx2(input, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { not_vec_sse2(input, result) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { not_vec_neon(input, result) };
            return;
        }
    }
    not_vec_scalar(input, result);
}
fn eq_vec_scalar(a: &[f32], b: &[f32], result: &mut [bool]) {
    for i in 0..a.len() {
        result[i] = a[i] == b[i];
    }
}
fn ne_vec_scalar(a: &[f32], b: &[f32], result: &mut [bool]) {
    for i in 0..a.len() {
        result[i] = a[i] != b[i];
    }
}
fn lt_vec_scalar(a: &[f32], b: &[f32], result: &mut [bool]) {
    for i in 0..a.len() {
        result[i] = a[i] < b[i];
    }
}
fn le_vec_scalar(a: &[f32], b: &[f32], result: &mut [bool]) {
    for i in 0..a.len() {
        result[i] = a[i] <= b[i];
    }
}
fn gt_vec_scalar(a: &[f32], b: &[f32], result: &mut [bool]) {
    for i in 0..a.len() {
        result[i] = a[i] > b[i];
    }
}
fn ge_vec_scalar(a: &[f32], b: &[f32], result: &mut [bool]) {
    for i in 0..a.len() {
        result[i] = a[i] >= b[i];
    }
}
fn and_vec_scalar(a: &[bool], b: &[bool], result: &mut [bool]) {
    for i in 0..a.len() {
        result[i] = a[i] && b[i];
    }
}
fn or_vec_scalar(a: &[bool], b: &[bool], result: &mut [bool]) {
    for i in 0..a.len() {
        result[i] = a[i] || b[i];
    }
}
fn xor_vec_scalar(a: &[bool], b: &[bool], result: &mut [bool]) {
    for i in 0..a.len() {
        result[i] = a[i] ^ b[i];
    }
}
fn not_vec_scalar(input: &[bool], result: &mut [bool]) {
    for i in 0..input.len() {
        result[i] = !input[i];
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn eq_vec_sse2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm_cmpeq_ps(a_vec, b_vec);
        let mask = _mm_movemask_ps(cmp_result);
        result[i] = (mask & 0x1) != 0;
        result[i + 1] = (mask & 0x2) != 0;
        result[i + 2] = (mask & 0x4) != 0;
        result[i + 3] = (mask & 0x8) != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] == b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn ne_vec_sse2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm_cmpneq_ps(a_vec, b_vec);
        let mask = _mm_movemask_ps(cmp_result);
        result[i] = (mask & 0x1) != 0;
        result[i + 1] = (mask & 0x2) != 0;
        result[i + 2] = (mask & 0x4) != 0;
        result[i + 3] = (mask & 0x8) != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] != b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn lt_vec_sse2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm_cmplt_ps(a_vec, b_vec);
        let mask = _mm_movemask_ps(cmp_result);
        result[i] = (mask & 0x1) != 0;
        result[i + 1] = (mask & 0x2) != 0;
        result[i + 2] = (mask & 0x4) != 0;
        result[i + 3] = (mask & 0x8) != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] < b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn le_vec_sse2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm_cmple_ps(a_vec, b_vec);
        let mask = _mm_movemask_ps(cmp_result);
        result[i] = (mask & 0x1) != 0;
        result[i + 1] = (mask & 0x2) != 0;
        result[i + 2] = (mask & 0x4) != 0;
        result[i + 3] = (mask & 0x8) != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] <= b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn gt_vec_sse2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm_cmpgt_ps(a_vec, b_vec);
        let mask = _mm_movemask_ps(cmp_result);
        result[i] = (mask & 0x1) != 0;
        result[i + 1] = (mask & 0x2) != 0;
        result[i + 2] = (mask & 0x4) != 0;
        result[i + 3] = (mask & 0x8) != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] > b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn ge_vec_sse2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm_cmpge_ps(a_vec, b_vec);
        let mask = _mm_movemask_ps(cmp_result);
        result[i] = (mask & 0x1) != 0;
        result[i + 1] = (mask & 0x2) != 0;
        result[i + 2] = (mask & 0x4) != 0;
        result[i + 3] = (mask & 0x8) != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] >= b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn and_vec_sse2(a: &[bool], b: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 16 <= a.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let a_bytes = _mm_loadu_si128(a.as_ptr().add(i) as *const _);
        let b_bytes = _mm_loadu_si128(b.as_ptr().add(i) as *const _);
        let result_bytes = _mm_and_si128(a_bytes, b_bytes);
        _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] && b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn or_vec_sse2(a: &[bool], b: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 16 <= a.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let a_bytes = _mm_loadu_si128(a.as_ptr().add(i) as *const _);
        let b_bytes = _mm_loadu_si128(b.as_ptr().add(i) as *const _);
        let result_bytes = _mm_or_si128(a_bytes, b_bytes);
        _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] || b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn xor_vec_sse2(a: &[bool], b: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 16 <= a.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let a_bytes = _mm_loadu_si128(a.as_ptr().add(i) as *const _);
        let b_bytes = _mm_loadu_si128(b.as_ptr().add(i) as *const _);
        let result_bytes = _mm_xor_si128(a_bytes, b_bytes);
        _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] ^ b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn not_vec_sse2(input: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 16 <= input.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let input_bytes = _mm_loadu_si128(input.as_ptr().add(i) as *const _);
        let ones = _mm_set1_epi8(1);
        let result_bytes = _mm_xor_si128(input_bytes, ones);
        _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 16;
    }
    while i < input.len() {
        result[i] = !input[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn eq_vec_avx2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm256_cmp_ps(a_vec, b_vec, _CMP_EQ_OQ);
        let mask = _mm256_movemask_ps(cmp_result);
        result[i] = (mask & 0x01) != 0;
        result[i + 1] = (mask & 0x02) != 0;
        result[i + 2] = (mask & 0x04) != 0;
        result[i + 3] = (mask & 0x08) != 0;
        result[i + 4] = (mask & 0x10) != 0;
        result[i + 5] = (mask & 0x20) != 0;
        result[i + 6] = (mask & 0x40) != 0;
        result[i + 7] = (mask & 0x80) != 0;
        i += 8;
    }
    while i < a.len() {
        result[i] = a[i] == b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ne_vec_avx2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm256_cmp_ps(a_vec, b_vec, _CMP_NEQ_OQ);
        let mask = _mm256_movemask_ps(cmp_result);
        result[i] = (mask & 0x01) != 0;
        result[i + 1] = (mask & 0x02) != 0;
        result[i + 2] = (mask & 0x04) != 0;
        result[i + 3] = (mask & 0x08) != 0;
        result[i + 4] = (mask & 0x10) != 0;
        result[i + 5] = (mask & 0x20) != 0;
        result[i + 6] = (mask & 0x40) != 0;
        result[i + 7] = (mask & 0x80) != 0;
        i += 8;
    }
    while i < a.len() {
        result[i] = a[i] != b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn lt_vec_avx2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm256_cmp_ps(a_vec, b_vec, _CMP_LT_OQ);
        let mask = _mm256_movemask_ps(cmp_result);
        result[i] = (mask & 0x01) != 0;
        result[i + 1] = (mask & 0x02) != 0;
        result[i + 2] = (mask & 0x04) != 0;
        result[i + 3] = (mask & 0x08) != 0;
        result[i + 4] = (mask & 0x10) != 0;
        result[i + 5] = (mask & 0x20) != 0;
        result[i + 6] = (mask & 0x40) != 0;
        result[i + 7] = (mask & 0x80) != 0;
        i += 8;
    }
    while i < a.len() {
        result[i] = a[i] < b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn le_vec_avx2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm256_cmp_ps(a_vec, b_vec, _CMP_LE_OQ);
        let mask = _mm256_movemask_ps(cmp_result);
        result[i] = (mask & 0x01) != 0;
        result[i + 1] = (mask & 0x02) != 0;
        result[i + 2] = (mask & 0x04) != 0;
        result[i + 3] = (mask & 0x08) != 0;
        result[i + 4] = (mask & 0x10) != 0;
        result[i + 5] = (mask & 0x20) != 0;
        result[i + 6] = (mask & 0x40) != 0;
        result[i + 7] = (mask & 0x80) != 0;
        i += 8;
    }
    while i < a.len() {
        result[i] = a[i] <= b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn gt_vec_avx2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm256_cmp_ps(a_vec, b_vec, _CMP_GT_OQ);
        let mask = _mm256_movemask_ps(cmp_result);
        result[i] = (mask & 0x01) != 0;
        result[i + 1] = (mask & 0x02) != 0;
        result[i + 2] = (mask & 0x04) != 0;
        result[i + 3] = (mask & 0x08) != 0;
        result[i + 4] = (mask & 0x10) != 0;
        result[i + 5] = (mask & 0x20) != 0;
        result[i + 6] = (mask & 0x40) != 0;
        result[i + 7] = (mask & 0x80) != 0;
        i += 8;
    }
    while i < a.len() {
        result[i] = a[i] > b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ge_vec_avx2(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let cmp_result = _mm256_cmp_ps(a_vec, b_vec, _CMP_GE_OQ);
        let mask = _mm256_movemask_ps(cmp_result);
        result[i] = (mask & 0x01) != 0;
        result[i + 1] = (mask & 0x02) != 0;
        result[i + 2] = (mask & 0x04) != 0;
        result[i + 3] = (mask & 0x08) != 0;
        result[i + 4] = (mask & 0x10) != 0;
        result[i + 5] = (mask & 0x20) != 0;
        result[i + 6] = (mask & 0x40) != 0;
        result[i + 7] = (mask & 0x80) != 0;
        i += 8;
    }
    while i < a.len() {
        result[i] = a[i] >= b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn and_vec_avx2(a: &[bool], b: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 32 <= a.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let a_bytes = _mm256_loadu_si256(a.as_ptr().add(i) as *const _);
        let b_bytes = _mm256_loadu_si256(b.as_ptr().add(i) as *const _);
        let result_bytes = _mm256_and_si256(a_bytes, b_bytes);
        _mm256_storeu_si256(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 32;
    }
    while i < a.len() {
        result[i] = a[i] && b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn or_vec_avx2(a: &[bool], b: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 32 <= a.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let a_bytes = _mm256_loadu_si256(a.as_ptr().add(i) as *const _);
        let b_bytes = _mm256_loadu_si256(b.as_ptr().add(i) as *const _);
        let result_bytes = _mm256_or_si256(a_bytes, b_bytes);
        _mm256_storeu_si256(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 32;
    }
    while i < a.len() {
        result[i] = a[i] || b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn xor_vec_avx2(a: &[bool], b: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 32 <= a.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let a_bytes = _mm256_loadu_si256(a.as_ptr().add(i) as *const _);
        let b_bytes = _mm256_loadu_si256(b.as_ptr().add(i) as *const _);
        let result_bytes = _mm256_xor_si256(a_bytes, b_bytes);
        _mm256_storeu_si256(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 32;
    }
    while i < a.len() {
        result[i] = a[i] ^ b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn not_vec_avx2(input: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 32 <= input.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let input_bytes = _mm256_loadu_si256(input.as_ptr().add(i) as *const _);
        let ones = _mm256_set1_epi8(1);
        let result_bytes = _mm256_xor_si256(input_bytes, ones);
        _mm256_storeu_si256(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 32;
    }
    while i < input.len() {
        result[i] = !input[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn eq_vec_avx512(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let mask = _mm512_cmp_ps_mask(a_vec, b_vec, _CMP_EQ_OQ);
        for j in 0..16 {
            result[i + j] = (mask & (1 << j)) != 0;
        }
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] == b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn ne_vec_avx512(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let mask = _mm512_cmp_ps_mask(a_vec, b_vec, _CMP_NEQ_OQ);
        for j in 0..16 {
            result[i + j] = (mask & (1 << j)) != 0;
        }
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] != b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn lt_vec_avx512(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let mask = _mm512_cmp_ps_mask(a_vec, b_vec, _CMP_LT_OQ);
        for j in 0..16 {
            result[i + j] = (mask & (1 << j)) != 0;
        }
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] < b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn le_vec_avx512(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let mask = _mm512_cmp_ps_mask(a_vec, b_vec, _CMP_LE_OQ);
        for j in 0..16 {
            result[i + j] = (mask & (1 << j)) != 0;
        }
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] <= b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn gt_vec_avx512(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let mask = _mm512_cmp_ps_mask(a_vec, b_vec, _CMP_GT_OQ);
        for j in 0..16 {
            result[i + j] = (mask & (1 << j)) != 0;
        }
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] > b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn ge_vec_avx512(a: &[f32], b: &[f32], result: &mut [bool]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let mask = _mm512_cmp_ps_mask(a_vec, b_vec, _CMP_GE_OQ);
        for j in 0..16 {
            result[i + j] = (mask & (1 << j)) != 0;
        }
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] >= b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn and_vec_avx512(a: &[bool], b: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 64 <= a.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let a_bytes = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let b_bytes = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
        let result_bytes = _mm512_and_si512(a_bytes, b_bytes);
        _mm512_storeu_si512(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 64;
    }
    while i < a.len() {
        result[i] = a[i] && b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn or_vec_avx512(a: &[bool], b: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 64 <= a.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let a_bytes = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let b_bytes = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
        let result_bytes = _mm512_or_si512(a_bytes, b_bytes);
        _mm512_storeu_si512(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 64;
    }
    while i < a.len() {
        result[i] = a[i] || b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn xor_vec_avx512(a: &[bool], b: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 64 <= a.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let a_bytes = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let b_bytes = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
        let result_bytes = _mm512_xor_si512(a_bytes, b_bytes);
        _mm512_storeu_si512(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 64;
    }
    while i < a.len() {
        result[i] = a[i] ^ b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn not_vec_avx512(input: &[bool], result: &mut [bool]) {
    let mut i = 0;
    while i + 64 <= input.len() {
        #[cfg(feature = "no-std")]
        use core::arch::x86_64::*;
        #[cfg(not(feature = "no-std"))]
        use core::arch::x86_64::*;
        let input_bytes = _mm512_loadu_si512(input.as_ptr().add(i) as *const _);
        let ones = _mm512_set1_epi8(1);
        let result_bytes = _mm512_xor_si512(input_bytes, ones);
        _mm512_storeu_si512(result.as_mut_ptr().add(i) as *mut _, result_bytes);
        i += 64;
    }
    while i < input.len() {
        result[i] = !input[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn eq_vec_neon(a: &[f32], b: &[f32], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let cmp_result = vceqq_f32(a_vec, b_vec);
        let mask = vgetq_lane_u32(cmp_result, 0);
        result[i] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 1);
        result[i + 1] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 2);
        result[i + 2] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 3);
        result[i + 3] = mask != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] == b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ne_vec_neon(a: &[f32], b: &[f32], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let eq_result = vceqq_f32(a_vec, b_vec);
        let cmp_result = vmvnq_u32(eq_result);
        let mask = vgetq_lane_u32(cmp_result, 0);
        result[i] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 1);
        result[i + 1] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 2);
        result[i + 2] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 3);
        result[i + 3] = mask != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] != b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lt_vec_neon(a: &[f32], b: &[f32], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let cmp_result = vcltq_f32(a_vec, b_vec);
        let mask = vgetq_lane_u32(cmp_result, 0);
        result[i] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 1);
        result[i + 1] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 2);
        result[i + 2] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 3);
        result[i + 3] = mask != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] < b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn le_vec_neon(a: &[f32], b: &[f32], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let cmp_result = vcleq_f32(a_vec, b_vec);
        let mask = vgetq_lane_u32(cmp_result, 0);
        result[i] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 1);
        result[i + 1] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 2);
        result[i + 2] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 3);
        result[i + 3] = mask != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] <= b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gt_vec_neon(a: &[f32], b: &[f32], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let cmp_result = vcgtq_f32(a_vec, b_vec);
        let mask = vgetq_lane_u32(cmp_result, 0);
        result[i] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 1);
        result[i + 1] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 2);
        result[i + 2] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 3);
        result[i + 3] = mask != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] > b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ge_vec_neon(a: &[f32], b: &[f32], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let cmp_result = vcgeq_f32(a_vec, b_vec);
        let mask = vgetq_lane_u32(cmp_result, 0);
        result[i] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 1);
        result[i + 1] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 2);
        result[i + 2] = mask != 0;
        let mask = vgetq_lane_u32(cmp_result, 3);
        result[i + 3] = mask != 0;
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] >= b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_vec_neon(a: &[bool], b: &[bool], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_bytes = vld1q_u8(a.as_ptr().add(i) as *const u8);
        let b_bytes = vld1q_u8(b.as_ptr().add(i) as *const u8);
        let result_bytes = vandq_u8(a_bytes, b_bytes);
        vst1q_u8(result.as_mut_ptr().add(i) as *mut u8, result_bytes);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] && b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn or_vec_neon(a: &[bool], b: &[bool], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_bytes = vld1q_u8(a.as_ptr().add(i) as *const u8);
        let b_bytes = vld1q_u8(b.as_ptr().add(i) as *const u8);
        let result_bytes = vorrq_u8(a_bytes, b_bytes);
        vst1q_u8(result.as_mut_ptr().add(i) as *mut u8, result_bytes);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] || b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xor_vec_neon(a: &[bool], b: &[bool], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_bytes = vld1q_u8(a.as_ptr().add(i) as *const u8);
        let b_bytes = vld1q_u8(b.as_ptr().add(i) as *const u8);
        let result_bytes = veorq_u8(a_bytes, b_bytes);
        vst1q_u8(result.as_mut_ptr().add(i) as *mut u8, result_bytes);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] ^ b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn not_vec_neon(input: &[bool], result: &mut [bool]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 16 <= input.len() {
        let input_bytes = vld1q_u8(input.as_ptr().add(i) as *const u8);
        let ones = vdupq_n_u8(1);
        let result_bytes = veorq_u8(input_bytes, ones);
        vst1q_u8(result.as_mut_ptr().add(i) as *mut u8, result_bytes);
        i += 16;
    }
    while i < input.len() {
        result[i] = !input[i];
        i += 1;
    }
}
#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};
    #[test]
    fn test_eq_vec() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 3.0, 3.0, 5.0];
        let mut result = vec![false; 4];
        eq_vec(&a, &b, &mut result);
        assert_eq!(result, vec![true, false, true, false]);
        let empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        let mut empty_result: Vec<bool> = vec![];
        eq_vec(&empty_a, &empty_b, &mut empty_result);
        assert_eq!(empty_result, Vec::<bool>::new());
    }
    #[test]
    fn test_ne_vec() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 3.0, 3.0, 5.0];
        let mut result = vec![false; 4];
        ne_vec(&a, &b, &mut result);
        assert_eq!(result, vec![false, true, false, true]);
    }
    #[test]
    fn test_lt_vec() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 3.0];
        let mut result = vec![false; 4];
        lt_vec(&a, &b, &mut result);
        assert_eq!(result, vec![true, false, false, false]);
    }
    #[test]
    fn test_le_vec() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 3.0];
        let mut result = vec![false; 4];
        le_vec(&a, &b, &mut result);
        assert_eq!(result, vec![true, true, false, false]);
    }
    #[test]
    fn test_gt_vec() {
        let a = vec![3.0, 2.0, 1.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 3.0];
        let mut result = vec![false; 4];
        gt_vec(&a, &b, &mut result);
        assert_eq!(result, vec![true, false, false, true]);
    }
    #[test]
    fn test_ge_vec() {
        let a = vec![3.0, 2.0, 1.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 3.0];
        let mut result = vec![false; 4];
        ge_vec(&a, &b, &mut result);
        assert_eq!(result, vec![true, true, false, true]);
    }
    #[test]
    fn test_and_vec() {
        let a = vec![true, false, true, false];
        let b = vec![true, true, false, false];
        let mut result = vec![false; 4];
        and_vec(&a, &b, &mut result);
        assert_eq!(result, vec![true, false, false, false]);
        let empty_a: Vec<bool> = vec![];
        let empty_b: Vec<bool> = vec![];
        let mut empty_result: Vec<bool> = vec![];
        and_vec(&empty_a, &empty_b, &mut empty_result);
        assert_eq!(empty_result, Vec::<bool>::new());
    }
    #[test]
    fn test_or_vec() {
        let a = vec![true, false, true, false];
        let b = vec![true, true, false, false];
        let mut result = vec![false; 4];
        or_vec(&a, &b, &mut result);
        assert_eq!(result, vec![true, true, true, false]);
    }
    #[test]
    fn test_xor_vec() {
        let a = vec![true, false, true, false];
        let b = vec![true, true, false, false];
        let mut result = vec![false; 4];
        xor_vec(&a, &b, &mut result);
        assert_eq!(result, vec![false, true, true, false]);
    }
    #[test]
    fn test_not_vec() {
        let input = vec![true, false, true, false];
        let mut result = vec![false; 4];
        not_vec(&input, &mut result);
        assert_eq!(result, vec![false, true, false, true]);
        let temp_result = result.clone();
        not_vec(&temp_result, &mut result);
        assert_eq!(result, input);
    }
    #[test]
    fn test_special_values() {
        let a = vec![f32::NAN, 1.0, 2.0];
        let b = vec![f32::NAN, 1.0, 3.0];
        let mut result = vec![false; 3];
        eq_vec(&a, &b, &mut result);
        assert_eq!(result, vec![false, true, false]);
        let inf_a = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0];
        let inf_b = vec![f32::INFINITY, f32::NEG_INFINITY, 2.0];
        let mut inf_result = vec![false; 3];
        eq_vec(&inf_a, &inf_b, &mut inf_result);
        assert_eq!(inf_result, vec![true, true, false]);
    }
    #[test]
    fn test_comparison_properties() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 3.0];
        let mut eq_result = vec![false; 4];
        let mut ne_result = vec![false; 4];
        let mut lt_result = vec![false; 4];
        let mut ge_result = vec![false; 4];
        eq_vec(&a, &b, &mut eq_result);
        ne_vec(&a, &b, &mut ne_result);
        for i in 0..4 {
            assert_eq!(eq_result[i], !ne_result[i]);
        }
        lt_vec(&a, &b, &mut lt_result);
        ge_vec(&a, &b, &mut ge_result);
        for i in 0..4 {
            assert_eq!(lt_result[i], !ge_result[i]);
        }
    }
    #[test]
    fn test_logical_operations_properties() {
        let a = vec![true, false, true, false];
        let b = vec![true, true, false, false];
        let mut and_result = vec![false; 4];
        let mut or_result = vec![false; 4];
        let mut not_a = vec![false; 4];
        let mut not_b = vec![false; 4];
        and_vec(&a, &b, &mut and_result);
        or_vec(&a, &b, &mut or_result);
        not_vec(&a, &mut not_a);
        not_vec(&b, &mut not_b);
        let mut not_and = vec![false; 4];
        not_vec(&and_result, &mut not_and);
        let mut not_a_or_not_b = vec![false; 4];
        or_vec(&not_a, &not_b, &mut not_a_or_not_b);
        assert_eq!(not_and, not_a_or_not_b);
    }
    #[test]
    fn test_large_vectors() {
        let size = 100;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let mut result = vec![false; size];
        lt_vec(&a, &b, &mut result);
        assert!(result.iter().all(|&x| x));
    }
    #[test]
    #[should_panic(expected = "Input vectors must have the same length")]
    fn test_eq_vec_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let mut result = vec![false; 2];
        eq_vec(&a, &b, &mut result);
    }
    #[test]
    #[should_panic(expected = "Output vector must have the same length as input vectors")]
    fn test_eq_vec_output_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut result = vec![false; 2];
        eq_vec(&a, &b, &mut result);
    }
    #[test]
    #[should_panic(expected = "Input vectors must have the same length")]
    fn test_and_vec_dimension_mismatch() {
        let a = vec![true, false, true];
        let b = vec![false, true];
        let mut result = vec![false; 2];
        and_vec(&a, &b, &mut result);
    }
}
