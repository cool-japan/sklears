//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn scale_vec_avx2(vector: &[f32], scalar: f32, result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let scalar_vec = _mm256_set1_ps(scalar);
    let mut i = 0;
    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_mul_ps(vector_vec, scalar_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }
    while i < vector.len() {
        result[i] = vector[i] * scalar;
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn abs_vec_avx2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
    let mut i = 0;
    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_and_ps(vector_vec, abs_mask);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }
    while i < vector.len() {
        result[i] = vector[i].abs();
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn neg_vec_avx2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x80000000));
    let mut i = 0;
    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_xor_ps(vector_vec, sign_mask);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }
    while i < vector.len() {
        result[i] = -vector[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn reciprocal_vec_avx2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let one_vec = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_div_ps(one_vec, vector_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }
    while i < vector.len() {
        result[i] = 1.0 / vector[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn square_vec_avx2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_mul_ps(vector_vec, vector_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }
    while i < vector.len() {
        result[i] = vector[i] * vector[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn add_vec_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm512_add_ps(a_vec, b_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn subtract_vec_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm512_sub_ps(a_vec, b_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] - b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn multiply_vec_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm512_mul_ps(a_vec, b_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] * b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn divide_vec_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm512_div_ps(a_vec, b_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }
    while i < a.len() {
        result[i] = a[i] / b[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn scale_vec_avx512(vector: &[f32], scalar: f32, result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let scalar_vec = _mm512_set1_ps(scalar);
    let mut i = 0;
    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_mul_ps(vector_vec, scalar_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }
    while i < vector.len() {
        result[i] = vector[i] * scalar;
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn abs_vec_avx512(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_abs_ps(vector_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }
    while i < vector.len() {
        result[i] = vector[i].abs();
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn neg_vec_avx512(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let zero = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_sub_ps(zero, vector_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }
    while i < vector.len() {
        result[i] = -vector[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn reciprocal_vec_avx512(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let one_vec = _mm512_set1_ps(1.0);
    let mut i = 0;
    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_div_ps(one_vec, vector_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }
    while i < vector.len() {
        result[i] = 1.0 / vector[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn square_vec_avx512(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_mul_ps(vector_vec, vector_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }
    while i < vector.len() {
        result[i] = vector[i] * vector[i];
        i += 1;
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
pub(crate) unsafe fn fma_fma_intrinsic(a: &mut [f32], b: &[f32], c: &[f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let c_vec = _mm256_loadu_ps(c.as_ptr().add(i));
        let result_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
        _mm256_storeu_ps(a.as_mut_ptr().add(i), result_vec);
        i += 8;
    }
    while i < a.len() {
        a[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_vec_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let result_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn subtract_vec_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let result_vec = vsubq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] - b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn multiply_vec_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let result_vec = vmulq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] * b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn divide_vec_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let result_vec = vdivq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < a.len() {
        result[i] = a[i] / b[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn fma_neon(a: &mut [f32], b: &[f32], c: &[f32]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let c_vec = vld1q_f32(c.as_ptr().add(i));
        let result_vec = vfmaq_f32(c_vec, a_vec, b_vec);
        vst1q_f32(a.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < a.len() {
        a[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn scale_vec_neon(vector: &[f32], scalar: f32, result: &mut [f32]) {
    use core::arch::aarch64::*;
    let scalar_vec = vdupq_n_f32(scalar);
    let mut i = 0;
    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let result_vec = vmulq_f32(vector_vec, scalar_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < vector.len() {
        result[i] = vector[i] * scalar;
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn abs_vec_neon(vector: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let result_vec = vabsq_f32(vector_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < vector.len() {
        result[i] = vector[i].abs();
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neg_vec_neon(vector: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let result_vec = vnegq_f32(vector_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < vector.len() {
        result[i] = -vector[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn reciprocal_vec_neon(vector: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let one_vec = vdupq_n_f32(1.0);
        let result_vec = vdivq_f32(one_vec, vector_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < vector.len() {
        result[i] = 1.0 / vector[i];
        i += 1;
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn square_vec_neon(vector: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let result_vec = vmulq_f32(vector_vec, vector_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }
    while i < vector.len() {
        result[i] = vector[i] * vector[i];
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
    fn test_add_vec() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];
        add_vec(&a, &b, &mut result);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        let empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        let mut empty_result: Vec<f32> = vec![];
        add_vec(&empty_a, &empty_b, &mut empty_result);
        assert_eq!(empty_result, Vec::<f32>::new());
    }
    #[test]
    fn test_subtract_vec() {
        let a = vec![10.0, 8.0, 6.0, 4.0];
        let b = vec![3.0, 2.0, 1.0, 1.0];
        let mut result = vec![0.0; 4];
        subtract_vec(&a, &b, &mut result);
        assert_eq!(result, vec![7.0, 6.0, 5.0, 3.0]);
    }
    #[test]
    fn test_multiply_vec() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; 4];
        multiply_vec(&a, &b, &mut result);
        assert_eq!(result, vec![6.0, 12.0, 20.0, 30.0]);
    }
    #[test]
    fn test_divide_vec() {
        let a = vec![12.0, 15.0, 20.0, 25.0];
        let b = vec![3.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];
        divide_vec(&a, &b, &mut result);
        assert_eq!(result, vec![4.0, 5.0, 5.0, 5.0]);
        let div_zero_a = vec![1.0, 2.0];
        let div_zero_b = vec![0.0, 0.0];
        let mut div_zero_result = vec![0.0; 2];
        divide_vec(&div_zero_a, &div_zero_b, &mut div_zero_result);
        assert_eq!(div_zero_result[0], f32::INFINITY);
        assert_eq!(div_zero_result[1], f32::INFINITY);
    }
    #[test]
    fn test_fma() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        fma(&mut a, &b, &c);
        assert_eq!(a, vec![3.0, 7.0, 13.0, 21.0]);
        let mut empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        let empty_c: Vec<f32> = vec![];
        fma(&mut empty_a, &empty_b, &empty_c);
        assert_eq!(empty_a, Vec::<f32>::new());
    }
    #[test]
    fn test_scale_vec() {
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let mut result = vec![0.0; 4];
        scale_vec(&vector, 2.5, &mut result);
        assert_eq!(result, vec![2.5, 5.0, 7.5, 10.0]);
        scale_vec(&vector, 0.0, &mut result);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
        scale_vec(&vector, -2.0, &mut result);
        assert_eq!(result, vec![-2.0, -4.0, -6.0, -8.0]);
    }
    #[test]
    fn test_scale_vec_inplace() {
        let mut vector = vec![1.0, 2.0, 3.0, 4.0];
        scale_vec_inplace(&mut vector, 3.0);
        assert_eq!(vector, vec![3.0, 6.0, 9.0, 12.0]);
    }
    #[test]
    fn test_abs_vec() {
        let vector = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut result = vec![0.0; 5];
        abs_vec(&vector, &mut result);
        assert_eq!(result, vec![2.0, 1.0, 0.0, 1.0, 2.0]);
        let positive = vec![1.0, 2.0, 3.0];
        let mut positive_result = vec![0.0; 3];
        abs_vec(&positive, &mut positive_result);
        assert_eq!(positive_result, positive);
    }
    #[test]
    fn test_neg_vec() {
        let vector = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut result = vec![0.0; 5];
        neg_vec(&vector, &mut result);
        assert_eq!(result, vec![2.0, 1.0, 0.0, -1.0, -2.0]);
        let temp_result = result.clone();
        neg_vec(&temp_result, &mut result);
        assert_eq!(result, vector);
    }
    #[test]
    #[should_panic(expected = "Input vectors must have the same length")]
    fn test_add_vec_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let mut result = vec![0.0; 2];
        add_vec(&a, &b, &mut result);
    }
    #[test]
    #[should_panic(expected = "Output vector must have the same length as input vectors")]
    fn test_add_vec_output_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut result = vec![0.0; 2];
        add_vec(&a, &b, &mut result);
    }
    #[test]
    #[should_panic(expected = "Input vectors must have the same length")]
    fn test_fma_dimension_mismatch() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let c = vec![7.0, 8.0, 9.0];
        fma(&mut a, &b, &c);
    }
    #[test]
    fn test_large_vectors() {
        let size = 100;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let mut result = vec![0.0; size];
        add_vec(&a, &b, &mut result);
        for i in 0..size {
            assert_eq!(result[i], (2 * i + 1) as f32);
        }
    }
    #[test]
    fn test_arithmetic_properties() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result1 = vec![0.0; 4];
        let mut result2 = vec![0.0; 4];
        add_vec(&a, &b, &mut result1);
        add_vec(&b, &a, &mut result2);
        assert_eq!(result1, result2);
        multiply_vec(&a, &b, &mut result1);
        multiply_vec(&b, &a, &mut result2);
        assert_eq!(result1, result2);
        subtract_vec(&a, &b, &mut result1);
        subtract_vec(&b, &a, &mut result2);
        assert_ne!(result1, result2);
    }
    #[test]
    fn test_reciprocal_vec() {
        let vector = vec![1.0, 2.0, 4.0, 0.5];
        let mut result = vec![0.0; 4];
        reciprocal_vec(&vector, &mut result);
        assert_eq!(result, vec![1.0, 0.5, 0.25, 2.0]);
        let empty_vector: Vec<f32> = vec![];
        let mut empty_result: Vec<f32> = vec![];
        reciprocal_vec(&empty_vector, &mut empty_result);
        assert_eq!(empty_result, Vec::<f32>::new());
    }
    #[test]
    fn test_square_vec() {
        let vector = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let mut result = vec![0.0; 7];
        square_vec(&vector, &mut result);
        assert_eq!(result, vec![9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);
        let empty_vector: Vec<f32> = vec![];
        let mut empty_result: Vec<f32> = vec![];
        square_vec(&empty_vector, &mut empty_result);
        assert_eq!(empty_result, Vec::<f32>::new());
    }
    #[test]
    #[should_panic(expected = "Input and output vectors must have the same length")]
    fn test_reciprocal_vec_dimension_mismatch() {
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];
        reciprocal_vec(&vector, &mut result);
    }
    #[test]
    #[should_panic(expected = "Input and output vectors must have the same length")]
    fn test_square_vec_dimension_mismatch() {
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];
        square_vec(&vector, &mut result);
    }
}
