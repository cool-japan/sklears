//! SIMD-optimized distance calculations using platform intrinsics

#[cfg(feature = "no-std")]
use alloc::{format, vec, vec::Vec};

/// SIMD-optimized Euclidean distance calculation
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { euclidean_distance_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { euclidean_distance_sse2(a, b) };
        }
    }

    euclidean_distance_scalar(a, b)
}

/// Scalar fallback for Euclidean distance
fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn euclidean_distance_sse2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(a_vec, b_vec);
        let squared = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, squared);
        i += 4;
    }

    // Sum the 4 elements in the vector
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    // Handle remaining elements
    while i < a.len() {
        let diff = a[i] - b[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum.sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        let squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
        i += 8;
    }

    // Sum the 8 elements in the vector
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    // Handle remaining elements
    while i < a.len() {
        let diff = a[i] - b[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum.sqrt()
}

/// SIMD-optimized Manhattan distance
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { manhattan_distance_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { manhattan_distance_sse2(a, b) };
        }
    }

    manhattan_distance_scalar(a, b)
}

/// Scalar fallback for Manhattan distance
fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn manhattan_distance_sse2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let sign_mask = _mm_set1_ps(-0.0); // Sign bit mask for absolute value
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(a_vec, b_vec);
        // Absolute value by clearing sign bit
        let abs_diff = _mm_andnot_ps(sign_mask, diff);
        sum = _mm_add_ps(sum, abs_diff);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < a.len() {
        scalar_sum += (a[i] - b[i]).abs();
        i += 1;
    }

    scalar_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);
        sum = _mm256_add_ps(sum, abs_diff);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < a.len() {
        scalar_sum += (a[i] - b[i]).abs();
        i += 1;
    }

    scalar_sum
}

/// SIMD-optimized cosine distance
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { cosine_distance_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { cosine_distance_sse2(a, b) };
        }
    }

    cosine_distance_scalar(a, b)
}

/// Scalar fallback for cosine distance
fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    1.0 - (dot_product / (norm_a * norm_b))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn cosine_distance_sse2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut dot_sum = _mm_setzero_ps();
    let mut norm_a_sum = _mm_setzero_ps();
    let mut norm_b_sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));

        dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(a_vec, b_vec));
        norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(a_vec, a_vec));
        norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(b_vec, b_vec));

        i += 4;
    }

    let mut dot_result = [0.0f32; 4];
    let mut norm_a_result = [0.0f32; 4];
    let mut norm_b_result = [0.0f32; 4];

    _mm_storeu_ps(dot_result.as_mut_ptr(), dot_sum);
    _mm_storeu_ps(norm_a_result.as_mut_ptr(), norm_a_sum);
    _mm_storeu_ps(norm_b_result.as_mut_ptr(), norm_b_sum);

    let mut dot = dot_result.iter().sum::<f32>();
    let mut norm_a = norm_a_result.iter().sum::<f32>();
    let mut norm_b = norm_b_result.iter().sum::<f32>();

    while i < a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));

        dot_sum = _mm256_add_ps(dot_sum, _mm256_mul_ps(a_vec, b_vec));
        norm_a_sum = _mm256_add_ps(norm_a_sum, _mm256_mul_ps(a_vec, a_vec));
        norm_b_sum = _mm256_add_ps(norm_b_sum, _mm256_mul_ps(b_vec, b_vec));

        i += 8;
    }

    let mut dot_result = [0.0f32; 8];
    let mut norm_a_result = [0.0f32; 8];
    let mut norm_b_result = [0.0f32; 8];

    _mm256_storeu_ps(dot_result.as_mut_ptr(), dot_sum);
    _mm256_storeu_ps(norm_a_result.as_mut_ptr(), norm_a_sum);
    _mm256_storeu_ps(norm_b_result.as_mut_ptr(), norm_b_sum);

    let mut dot = dot_result.iter().sum::<f32>();
    let mut norm_a = norm_a_result.iter().sum::<f32>();
    let mut norm_b = norm_b_result.iter().sum::<f32>();

    while i < a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()))
}

/// Batch euclidean distance calculation
pub fn euclidean_distance_batch(points: &[Vec<f32>], query: &[f32]) -> Vec<f32> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        points
            .par_iter()
            .map(|point| euclidean_distance(point, query))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        points
            .iter()
            .map(|point| euclidean_distance(point, query))
            .collect()
    }
}

/// SIMD-optimized Chebyshev distance (L∞ norm)
pub fn chebyshev_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { chebyshev_distance_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { chebyshev_distance_sse2(a, b) };
        }
    }

    chebyshev_distance_scalar(a, b)
}

fn chebyshev_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn chebyshev_distance_sse2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut max_vec = _mm_setzero_ps();
    let sign_mask = _mm_set1_ps(-0.0);
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(a_vec, b_vec);
        let abs_diff = _mm_andnot_ps(sign_mask, diff);
        max_vec = _mm_max_ps(max_vec, abs_diff);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), max_vec);
    let mut max_val = result[0].max(result[1]).max(result[2]).max(result[3]);

    while i < a.len() {
        max_val = max_val.max((a[i] - b[i]).abs());
        i += 1;
    }

    max_val
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn chebyshev_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut max_vec = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);
        max_vec = _mm256_max_ps(max_vec, abs_diff);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), max_vec);
    let mut max_val = result.iter().fold(0.0f32, |a, &b| a.max(b));

    while i < a.len() {
        max_val = max_val.max((a[i] - b[i]).abs());
        i += 1;
    }

    max_val
}

/// SIMD-optimized Minkowski distance
pub fn minkowski_distance(a: &[f32], b: &[f32], p: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    if p == 1.0 {
        return manhattan_distance(a, b);
    } else if p == 2.0 {
        return euclidean_distance(a, b);
    } else if p == f32::INFINITY {
        return chebyshev_distance(a, b);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { minkowski_distance_avx2(a, b, p) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { minkowski_distance_sse2(a, b, p) };
        }
    }

    minkowski_distance_scalar(a, b, p)
}

fn minkowski_distance_scalar(a: &[f32], b: &[f32], p: f32) -> f32 {
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum();
    sum.powf(1.0 / p)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn minkowski_distance_sse2(a: &[f32], b: &[f32], p: f32) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let sign_mask = _mm_set1_ps(-0.0);
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(a_vec, b_vec);
        let abs_diff = _mm_andnot_ps(sign_mask, diff);

        // For SIMD efficiency, we use scalar pow for now
        let mut pow_result = [0.0f32; 4];
        _mm_storeu_ps(pow_result.as_mut_ptr(), abs_diff);

        for j in 0..4 {
            pow_result[j] = pow_result[j].powf(p);
        }

        let pow_vec = _mm_loadu_ps(pow_result.as_ptr());
        sum = _mm_add_ps(sum, pow_vec);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < a.len() {
        scalar_sum += (a[i] - b[i]).abs().powf(p);
        i += 1;
    }

    scalar_sum.powf(1.0 / p)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn minkowski_distance_avx2(a: &[f32], b: &[f32], p: f32) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);

        let mut pow_result = [0.0f32; 8];
        _mm256_storeu_ps(pow_result.as_mut_ptr(), abs_diff);

        for j in 0..8 {
            pow_result[j] = pow_result[j].powf(p);
        }

        let pow_vec = _mm256_loadu_ps(pow_result.as_ptr());
        sum = _mm256_add_ps(sum, pow_vec);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < a.len() {
        scalar_sum += (a[i] - b[i]).abs().powf(p);
        i += 1;
    }

    scalar_sum.powf(1.0 / p)
}

/// SIMD-optimized Hamming distance for binary data
pub fn hamming_distance(a: &[u32], b: &[u32]) -> u32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { hamming_distance_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { hamming_distance_sse2(a, b) };
        }
    }

    hamming_distance_scalar(a, b)
}

fn hamming_distance_scalar(a: &[u32], b: &[u32]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn hamming_distance_sse2(a: &[u32], b: &[u32]) -> u32 {
    use core::arch::x86_64::*;

    let mut sum = 0u32;
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_si128(a.as_ptr().add(i) as *const _);
        let b_vec = _mm_loadu_si128(b.as_ptr().add(i) as *const _);
        let xor_vec = _mm_xor_si128(a_vec, b_vec);

        let mut result = [0u32; 4];
        _mm_storeu_si128(result.as_mut_ptr() as *mut _, xor_vec);

        for j in 0..4 {
            sum += result[j].count_ones();
        }
        i += 4;
    }

    while i < a.len() {
        sum += (a[i] ^ b[i]).count_ones();
        i += 1;
    }

    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hamming_distance_avx2(a: &[u32], b: &[u32]) -> u32 {
    use core::arch::x86_64::*;

    let mut sum = 0u32;
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_si256(a.as_ptr().add(i) as *const _);
        let b_vec = _mm256_loadu_si256(b.as_ptr().add(i) as *const _);
        let xor_vec = _mm256_xor_si256(a_vec, b_vec);

        let mut result = [0u32; 8];
        _mm256_storeu_si256(result.as_mut_ptr() as *mut _, xor_vec);

        for j in 0..8 {
            sum += result[j].count_ones();
        }
        i += 8;
    }

    while i < a.len() {
        sum += (a[i] ^ b[i]).count_ones();
        i += 1;
    }

    sum
}

/// SIMD-optimized Jaccard similarity
pub fn jaccard_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { jaccard_similarity_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { jaccard_similarity_sse2(a, b) };
        }
    }

    jaccard_similarity_scalar(a, b)
}

fn jaccard_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut intersection = 0.0;
    let mut union = 0.0;

    for i in 0..a.len() {
        intersection += a[i].min(b[i]);
        union += a[i].max(b[i]);
    }

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn jaccard_similarity_sse2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut intersection_sum = _mm_setzero_ps();
    let mut union_sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));

        let intersection_vec = _mm_min_ps(a_vec, b_vec);
        let union_vec = _mm_max_ps(a_vec, b_vec);

        intersection_sum = _mm_add_ps(intersection_sum, intersection_vec);
        union_sum = _mm_add_ps(union_sum, union_vec);
        i += 4;
    }

    let mut intersection_result = [0.0f32; 4];
    let mut union_result = [0.0f32; 4];
    _mm_storeu_ps(intersection_result.as_mut_ptr(), intersection_sum);
    _mm_storeu_ps(union_result.as_mut_ptr(), union_sum);

    let mut intersection = intersection_result.iter().sum::<f32>();
    let mut union = union_result.iter().sum::<f32>();

    while i < a.len() {
        intersection += a[i].min(b[i]);
        union += a[i].max(b[i]);
        i += 1;
    }

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn jaccard_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut intersection_sum = _mm256_setzero_ps();
    let mut union_sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));

        let intersection_vec = _mm256_min_ps(a_vec, b_vec);
        let union_vec = _mm256_max_ps(a_vec, b_vec);

        intersection_sum = _mm256_add_ps(intersection_sum, intersection_vec);
        union_sum = _mm256_add_ps(union_sum, union_vec);
        i += 8;
    }

    let mut intersection_result = [0.0f32; 8];
    let mut union_result = [0.0f32; 8];
    _mm256_storeu_ps(intersection_result.as_mut_ptr(), intersection_sum);
    _mm256_storeu_ps(union_result.as_mut_ptr(), union_sum);

    let mut intersection = intersection_result.iter().sum::<f32>();
    let mut union = union_result.iter().sum::<f32>();

    while i < a.len() {
        intersection += a[i].min(b[i]);
        union += a[i].max(b[i]);
        i += 1;
    }

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Jaccard distance (1 - Jaccard similarity)
pub fn jaccard_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - jaccard_similarity(a, b)
}

/// SIMD-optimized Mahalanobis distance
///
/// Computes the Mahalanobis distance between two vectors given an inverse covariance matrix.
/// The distance is computed as sqrt((a-b)^T * inv_cov * (a-b))
pub fn mahalanobis_distance(a: &[f32], b: &[f32], inv_cov: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    let n = a.len();
    assert_eq!(
        inv_cov.len(),
        n * n,
        "Inverse covariance matrix must be n x n"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { mahalanobis_distance_avx2(a, b, inv_cov) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { mahalanobis_distance_sse2(a, b, inv_cov) };
        }
    }

    mahalanobis_distance_scalar(a, b, inv_cov)
}

fn mahalanobis_distance_scalar(a: &[f32], b: &[f32], inv_cov: &[f32]) -> f32 {
    let n = a.len();

    // Compute diff = a - b
    let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

    // Compute temp = inv_cov * diff
    let mut temp = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            temp[i] += inv_cov[i * n + j] * diff[j];
        }
    }

    // Compute diff^T * temp
    let result: f32 = diff.iter().zip(temp.iter()).map(|(x, y)| x * y).sum();

    result.sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mahalanobis_distance_sse2(a: &[f32], b: &[f32], inv_cov: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let n = a.len();

    // Compute diff = a - b with SIMD
    let mut diff = vec![0.0f32; n];
    let mut i = 0;

    while i + 4 <= n {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let diff_vec = _mm_sub_ps(a_vec, b_vec);
        _mm_storeu_ps(diff.as_mut_ptr().add(i), diff_vec);
        i += 4;
    }

    while i < n {
        diff[i] = a[i] - b[i];
        i += 1;
    }

    // Matrix-vector multiplication with SIMD
    let mut temp = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = _mm_setzero_ps();
        let mut j = 0;

        while j + 4 <= n {
            let inv_cov_vec = _mm_loadu_ps(inv_cov.as_ptr().add(i * n + j));
            let diff_vec = _mm_loadu_ps(diff.as_ptr().add(j));
            sum = _mm_add_ps(sum, _mm_mul_ps(inv_cov_vec, diff_vec));
            j += 4;
        }

        let mut result_array = [0.0f32; 4];
        _mm_storeu_ps(result_array.as_mut_ptr(), sum);
        let mut scalar_sum = result_array.iter().sum::<f32>();

        while j < n {
            scalar_sum += inv_cov[i * n + j] * diff[j];
            j += 1;
        }

        temp[i] = scalar_sum;
    }

    // Final dot product with SIMD
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= n {
        let diff_vec = _mm_loadu_ps(diff.as_ptr().add(i));
        let temp_vec = _mm_loadu_ps(temp.as_ptr().add(i));
        sum = _mm_add_ps(sum, _mm_mul_ps(diff_vec, temp_vec));
        i += 4;
    }

    let mut result_array = [0.0f32; 4];
    _mm_storeu_ps(result_array.as_mut_ptr(), sum);
    let mut scalar_sum = result_array.iter().sum::<f32>();

    while i < n {
        scalar_sum += diff[i] * temp[i];
        i += 1;
    }

    scalar_sum.sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mahalanobis_distance_avx2(a: &[f32], b: &[f32], inv_cov: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let n = a.len();

    // Compute diff = a - b with SIMD
    let mut diff = vec![0.0f32; n];
    let mut i = 0;

    while i + 8 <= n {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(diff.as_mut_ptr().add(i), diff_vec);
        i += 8;
    }

    while i < n {
        diff[i] = a[i] - b[i];
        i += 1;
    }

    // Matrix-vector multiplication with SIMD
    let mut temp = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = _mm256_setzero_ps();
        let mut j = 0;

        while j + 8 <= n {
            let inv_cov_vec = _mm256_loadu_ps(inv_cov.as_ptr().add(i * n + j));
            let diff_vec = _mm256_loadu_ps(diff.as_ptr().add(j));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(inv_cov_vec, diff_vec));
            j += 8;
        }

        let mut result_array = [0.0f32; 8];
        _mm256_storeu_ps(result_array.as_mut_ptr(), sum);
        let mut scalar_sum = result_array.iter().sum::<f32>();

        while j < n {
            scalar_sum += inv_cov[i * n + j] * diff[j];
            j += 1;
        }

        temp[i] = scalar_sum;
    }

    // Final dot product with SIMD
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        let diff_vec = _mm256_loadu_ps(diff.as_ptr().add(i));
        let temp_vec = _mm256_loadu_ps(temp.as_ptr().add(i));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff_vec, temp_vec));
        i += 8;
    }

    let mut result_array = [0.0f32; 8];
    _mm256_storeu_ps(result_array.as_mut_ptr(), sum);
    let mut scalar_sum = result_array.iter().sum::<f32>();

    while i < n {
        scalar_sum += diff[i] * temp[i];
        i += 1;
    }

    scalar_sum.sqrt()
}

/// SIMD-optimized Canberra distance
///
/// Computes the Canberra distance between two vectors.
/// Distance = sum(|a_i - b_i| / (|a_i| + |b_i|))
pub fn canberra_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { canberra_distance_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { canberra_distance_sse2(a, b) };
        }
    }

    canberra_distance_scalar(a, b)
}

fn canberra_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = (x - y).abs();
            let sum = x.abs() + y.abs();
            if sum == 0.0 {
                0.0
            } else {
                diff / sum
            }
        })
        .sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn canberra_distance_sse2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let sign_mask = _mm_set1_ps(-0.0);
    let zero = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));

        let diff = _mm_sub_ps(a_vec, b_vec);
        let abs_diff = _mm_andnot_ps(sign_mask, diff);

        let abs_a = _mm_andnot_ps(sign_mask, a_vec);
        let abs_b = _mm_andnot_ps(sign_mask, b_vec);
        let sum_abs = _mm_add_ps(abs_a, abs_b);

        // Check for division by zero
        let mask = _mm_cmpneq_ps(sum_abs, zero);
        let ratio = _mm_div_ps(abs_diff, sum_abs);
        let masked_ratio = _mm_and_ps(ratio, mask);

        sum = _mm_add_ps(sum, masked_ratio);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < a.len() {
        let diff = (a[i] - b[i]).abs();
        let sum_abs = a[i].abs() + b[i].abs();
        if sum_abs != 0.0 {
            scalar_sum += diff / sum_abs;
        }
        i += 1;
    }

    scalar_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn canberra_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let zero = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));

        let diff = _mm256_sub_ps(a_vec, b_vec);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);

        let abs_a = _mm256_andnot_ps(sign_mask, a_vec);
        let abs_b = _mm256_andnot_ps(sign_mask, b_vec);
        let sum_abs = _mm256_add_ps(abs_a, abs_b);

        // Check for division by zero
        let mask = _mm256_cmp_ps(sum_abs, zero, _CMP_NEQ_OQ);
        let ratio = _mm256_div_ps(abs_diff, sum_abs);
        let masked_ratio = _mm256_and_ps(ratio, mask);

        sum = _mm256_add_ps(sum, masked_ratio);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < a.len() {
        let diff = (a[i] - b[i]).abs();
        let sum_abs = a[i].abs() + b[i].abs();
        if sum_abs != 0.0 {
            scalar_sum += diff / sum_abs;
        }
        i += 1;
    }

    scalar_sum
}

/// SIMD-optimized correlation distance
///
/// Computes the correlation distance as 1 - correlation coefficient
pub fn correlation_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let correlation = correlation_coefficient(a, b);
    1.0 - correlation
}

/// SIMD-optimized Pearson correlation coefficient
pub fn correlation_coefficient(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { correlation_coefficient_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { correlation_coefficient_sse2(a, b) };
        }
    }

    correlation_coefficient_scalar(a, b)
}

fn correlation_coefficient_scalar(a: &[f32], b: &[f32]) -> f32 {
    let _n = a.len() as f32;

    let mean_a = crate::vector::mean(a);
    let mean_b = crate::vector::mean(b);

    let mut sum_ab = 0.0;
    let mut sum_a2 = 0.0;
    let mut sum_b2 = 0.0;

    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        sum_ab += da * db;
        sum_a2 += da * da;
        sum_b2 += db * db;
    }

    let denom = (sum_a2 * sum_b2).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        sum_ab / denom
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn correlation_coefficient_sse2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mean_a = crate::vector::mean(a);
    let mean_b = crate::vector::mean(b);

    let mean_a_vec = _mm_set1_ps(mean_a);
    let mean_b_vec = _mm_set1_ps(mean_b);

    let mut sum_ab = _mm_setzero_ps();
    let mut sum_a2 = _mm_setzero_ps();
    let mut sum_b2 = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));

        let da = _mm_sub_ps(a_vec, mean_a_vec);
        let db = _mm_sub_ps(b_vec, mean_b_vec);

        sum_ab = _mm_add_ps(sum_ab, _mm_mul_ps(da, db));
        sum_a2 = _mm_add_ps(sum_a2, _mm_mul_ps(da, da));
        sum_b2 = _mm_add_ps(sum_b2, _mm_mul_ps(db, db));
        i += 4;
    }

    let mut ab_result = [0.0f32; 4];
    let mut a2_result = [0.0f32; 4];
    let mut b2_result = [0.0f32; 4];

    _mm_storeu_ps(ab_result.as_mut_ptr(), sum_ab);
    _mm_storeu_ps(a2_result.as_mut_ptr(), sum_a2);
    _mm_storeu_ps(b2_result.as_mut_ptr(), sum_b2);

    let mut scalar_sum_ab = ab_result.iter().sum::<f32>();
    let mut scalar_sum_a2 = a2_result.iter().sum::<f32>();
    let mut scalar_sum_b2 = b2_result.iter().sum::<f32>();

    while i < a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        scalar_sum_ab += da * db;
        scalar_sum_a2 += da * da;
        scalar_sum_b2 += db * db;
        i += 1;
    }

    let denom = (scalar_sum_a2 * scalar_sum_b2).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        scalar_sum_ab / denom
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn correlation_coefficient_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mean_a = crate::vector::mean(a);
    let mean_b = crate::vector::mean(b);

    let mean_a_vec = _mm256_set1_ps(mean_a);
    let mean_b_vec = _mm256_set1_ps(mean_b);

    let mut sum_ab = _mm256_setzero_ps();
    let mut sum_a2 = _mm256_setzero_ps();
    let mut sum_b2 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));

        let da = _mm256_sub_ps(a_vec, mean_a_vec);
        let db = _mm256_sub_ps(b_vec, mean_b_vec);

        sum_ab = _mm256_add_ps(sum_ab, _mm256_mul_ps(da, db));
        sum_a2 = _mm256_add_ps(sum_a2, _mm256_mul_ps(da, da));
        sum_b2 = _mm256_add_ps(sum_b2, _mm256_mul_ps(db, db));
        i += 8;
    }

    let mut ab_result = [0.0f32; 8];
    let mut a2_result = [0.0f32; 8];
    let mut b2_result = [0.0f32; 8];

    _mm256_storeu_ps(ab_result.as_mut_ptr(), sum_ab);
    _mm256_storeu_ps(a2_result.as_mut_ptr(), sum_a2);
    _mm256_storeu_ps(b2_result.as_mut_ptr(), sum_b2);

    let mut scalar_sum_ab = ab_result.iter().sum::<f32>();
    let mut scalar_sum_a2 = a2_result.iter().sum::<f32>();
    let mut scalar_sum_b2 = b2_result.iter().sum::<f32>();

    while i < a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        scalar_sum_ab += da * db;
        scalar_sum_a2 += da * da;
        scalar_sum_b2 += db * db;
        i += 1;
    }

    let denom = (scalar_sum_a2 * scalar_sum_b2).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        scalar_sum_ab / denom
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = euclidean_distance(&a, &b);
        let expected = 8.0_f32; // sqrt(16 + 16 + 16 + 16)

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = manhattan_distance(&a, &b);
        let expected = 16.0_f32; // 4 + 4 + 4 + 4

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let result = cosine_distance(&a, &b);
        let expected = 1.0_f32; // Orthogonal vectors

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_batch_distance() {
        let points = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let query = vec![0.0, 0.0];

        let distances = euclidean_distance_batch(&points, &query);

        assert_eq!(distances.len(), 3);
        assert_relative_eq!(distances[0], (5.0_f32).sqrt(), epsilon = 1e-6);
        assert_relative_eq!(distances[1], 5.0, epsilon = 1e-6);
        assert_relative_eq!(distances[2], (61.0_f32).sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_chebyshev_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 1.0, 7.0];

        let result = chebyshev_distance(&a, &b);
        let expected = 4.0; // max(|1-4|, |2-1|, |3-7|) = max(3, 1, 4) = 4

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_minkowski_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];

        // Test p=1 (Manhattan)
        let result1 = minkowski_distance(&a, &b, 1.0);
        let expected1 = 12.0; // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        assert_relative_eq!(result1, expected1, epsilon = 1e-6);

        // Test p=2 (Euclidean)
        let result2 = minkowski_distance(&a, &b, 2.0);
        let expected2 = (9.0 + 16.0 + 25.0_f32).sqrt(); // sqrt(3² + 4² + 5²)
        assert_relative_eq!(result2, expected2, epsilon = 1e-6);
    }

    #[test]
    fn test_hamming_distance() {
        let a = vec![0b1010, 0b1100, 0b0001];
        let b = vec![0b1001, 0b1110, 0b0001];

        let result = hamming_distance(&a, &b);
        // XOR: 0b0011 (2 bits), 0b0010 (1 bit), 0b0000 (0 bits) = 2 + 1 + 0 = 3
        let expected = 3;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_jaccard_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];

        let result = jaccard_similarity(&a, &b);
        // Intersection: min(1,2) + min(2,3) + min(3,4) = 1 + 2 + 3 = 6
        // Union: max(1,2) + max(2,3) + max(3,4) = 2 + 3 + 4 = 9
        let expected = 6.0 / 9.0;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_jaccard_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];

        let similarity = jaccard_similarity(&a, &b);
        let distance = jaccard_distance(&a, &b);

        assert_relative_eq!(distance, 1.0 - similarity, epsilon = 1e-6);
    }

    #[test]
    fn test_mahalanobis_distance() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];

        // Identity matrix as inverse covariance (reduces to Euclidean distance)
        let inv_cov = vec![1.0, 0.0, 0.0, 1.0];

        let result = mahalanobis_distance(&a, &b, &inv_cov);
        let expected = euclidean_distance(&a, &b);

        assert_relative_eq!(result, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_canberra_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];

        let result = canberra_distance(&a, &b);
        // |1-2|/(|1|+|2|) + |2-4|/(|2|+|4|) + |3-6|/(|3|+|6|) = 1/3 + 2/6 + 3/9 = 1/3 + 1/3 + 1/3 = 1.0
        let expected = 1.0;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_correlation_coefficient() {
        // Perfect positive correlation
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = correlation_coefficient(&a, &b);
        assert_relative_eq!(result, 1.0, epsilon = 1e-5);

        // Perfect negative correlation
        let c = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result2 = correlation_coefficient(&a, &c);
        assert_relative_eq!(result2, -1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_correlation_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = correlation_coefficient(&a, &b);
        let distance = correlation_distance(&a, &b);

        assert_relative_eq!(distance, 1.0 - corr, epsilon = 1e-6);
    }
}
