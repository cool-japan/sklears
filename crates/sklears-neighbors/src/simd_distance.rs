use crate::{NeighborsError, NeighborsResult};
use scirs2_core::ndarray::ArrayView1;
use sklears_core::types::Float;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdCapability {
    None,
    #[cfg(target_arch = "x86_64")]
    SSE,
    #[cfg(target_arch = "x86_64")]
    AVX,
    #[cfg(target_arch = "x86_64")]
    AVX2,
    #[cfg(target_arch = "aarch64")]
    NEON,
}

pub struct SimdDistanceCalculator {
    capability: SimdCapability,
    use_simd: bool,
}

impl Default for SimdDistanceCalculator {
    fn default() -> Self {
        let capability = detect_simd_capability();
        Self {
            capability,
            use_simd: capability != SimdCapability::None,
        }
    }
}

impl SimdDistanceCalculator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_simd_disabled() -> Self {
        Self {
            capability: SimdCapability::None,
            use_simd: false,
        }
    }

    pub fn capability(&self) -> SimdCapability {
        self.capability
    }

    pub fn euclidean_distance(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if x1.len() != x2.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x1.len()],
                actual: vec![x2.len()],
            });
        }

        if !self.use_simd || x1.len() < 4 {
            return Ok(self.euclidean_distance_scalar(x1, x2));
        }

        match self.capability {
            #[cfg(target_arch = "x86_64")]
            SimdCapability::AVX2 => Ok(self.euclidean_distance_avx2(x1, x2)?),
            #[cfg(target_arch = "x86_64")]
            SimdCapability::AVX => Ok(self.euclidean_distance_avx(x1, x2)?),
            #[cfg(target_arch = "x86_64")]
            SimdCapability::SSE => Ok(self.euclidean_distance_sse(x1, x2)?),
            #[cfg(target_arch = "aarch64")]
            SimdCapability::NEON => Ok(self.euclidean_distance_neon(x1, x2)?),
            _ => Ok(self.euclidean_distance_scalar(x1, x2)),
        }
    }

    pub fn manhattan_distance(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if x1.len() != x2.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x1.len()],
                actual: vec![x2.len()],
            });
        }

        if !self.use_simd || x1.len() < 4 {
            return Ok(self.manhattan_distance_scalar(x1, x2));
        }

        match self.capability {
            #[cfg(target_arch = "x86_64")]
            SimdCapability::AVX2 => Ok(self.manhattan_distance_avx2(x1, x2)?),
            #[cfg(target_arch = "x86_64")]
            SimdCapability::AVX => Ok(self.manhattan_distance_avx(x1, x2)?),
            #[cfg(target_arch = "x86_64")]
            SimdCapability::SSE => Ok(self.manhattan_distance_sse(x1, x2)?),
            #[cfg(target_arch = "aarch64")]
            SimdCapability::NEON => Ok(self.manhattan_distance_neon(x1, x2)?),
            _ => Ok(self.manhattan_distance_scalar(x1, x2)),
        }
    }

    pub fn cosine_distance(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if x1.len() != x2.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![x1.len()],
                actual: vec![x2.len()],
            });
        }

        if !self.use_simd || x1.len() < 4 {
            return self.cosine_distance_scalar(x1, x2);
        }

        match self.capability {
            #[cfg(target_arch = "x86_64")]
            SimdCapability::AVX2 => Ok(self.cosine_distance_avx2(x1, x2)?),
            #[cfg(target_arch = "x86_64")]
            SimdCapability::AVX => Ok(self.cosine_distance_avx(x1, x2)?),
            #[cfg(target_arch = "x86_64")]
            SimdCapability::SSE => Ok(self.cosine_distance_sse(x1, x2)?),
            #[cfg(target_arch = "aarch64")]
            SimdCapability::NEON => Ok(self.cosine_distance_neon(x1, x2)?),
            _ => Ok(self.cosine_distance_scalar(x1, x2)?),
        }
    }

    // Scalar implementations (fallback)
    fn euclidean_distance_scalar(&self, x1: ArrayView1<Float>, x2: ArrayView1<Float>) -> Float {
        x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<Float>()
            .sqrt()
    }

    fn manhattan_distance_scalar(&self, x1: ArrayView1<Float>, x2: ArrayView1<Float>) -> Float {
        x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).sum()
    }

    fn cosine_distance_scalar(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        let dot_product: Float = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
        let norm1: Float = x1.iter().map(|x| x * x).sum::<Float>().sqrt();
        let norm2: Float = x2.iter().map(|x| x * x).sum::<Float>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Err(NeighborsError::InvalidInput(
                "Zero norm vector in cosine distance".to_string(),
            ));
        }

        Ok(1.0 - (dot_product / (norm1 * norm2)))
    }

    // AVX2 implementations for x86_64
    #[cfg(target_arch = "x86_64")]
    fn euclidean_distance_avx2(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if !is_x86_feature_detected!("avx2") {
            return Ok(self.euclidean_distance_scalar(x1, x2));
        }

        unsafe {
            let len = x1.len();
            let chunk_size = 8; // AVX2 can process 8 f32 or 4 f64 at once
            let mut sum = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    // f32 case
                    let a = _mm256_loadu_ps(x1.as_ptr().add(i) as *const f32);
                    let b = _mm256_loadu_ps(x2.as_ptr().add(i) as *const f32);
                    let diff = _mm256_sub_ps(a, b);
                    let squared = _mm256_mul_ps(diff, diff);

                    let mut result = [0.0f32; 8];
                    _mm256_storeu_ps(result.as_mut_ptr(), squared);
                    sum += result.iter().sum::<f32>() as Float;
                } else {
                    // f64 case
                    let chunk_size_f64 = 4;
                    if i + chunk_size_f64 <= len {
                        let a = _mm256_loadu_pd(x1.as_ptr().add(i) as *const f64);
                        let b = _mm256_loadu_pd(x2.as_ptr().add(i) as *const f64);
                        let diff = _mm256_sub_pd(a, b);
                        let squared = _mm256_mul_pd(diff, diff);

                        let mut result = [0.0f64; 4];
                        _mm256_storeu_pd(result.as_mut_ptr(), squared);
                        sum += result.iter().sum::<f64>() as Float;
                        i += chunk_size_f64;
                        continue;
                    }
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                let diff = x1[j] - x2[j];
                sum += diff * diff;
            }

            Ok(sum.sqrt())
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn manhattan_distance_avx2(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if !is_x86_feature_detected!("avx2") {
            return Ok(self.manhattan_distance_scalar(x1, x2));
        }

        unsafe {
            let len = x1.len();
            let chunk_size = 8;
            let mut sum = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    let a = _mm256_loadu_ps(x1.as_ptr().add(i) as *const f32);
                    let b = _mm256_loadu_ps(x2.as_ptr().add(i) as *const f32);
                    let diff = _mm256_sub_ps(a, b);

                    // Create mask for absolute value
                    let sign_mask = _mm256_set1_ps(-0.0);
                    let abs_diff = _mm256_andnot_ps(sign_mask, diff);

                    let mut result = [0.0f32; 8];
                    _mm256_storeu_ps(result.as_mut_ptr(), abs_diff);
                    sum += result.iter().sum::<f32>() as Float;
                } else {
                    let chunk_size_f64 = 4;
                    if i + chunk_size_f64 <= len {
                        let a = _mm256_loadu_pd(x1.as_ptr().add(i) as *const f64);
                        let b = _mm256_loadu_pd(x2.as_ptr().add(i) as *const f64);
                        let diff = _mm256_sub_pd(a, b);

                        let sign_mask = _mm256_set1_pd(-0.0);
                        let abs_diff = _mm256_andnot_pd(sign_mask, diff);

                        let mut result = [0.0f64; 4];
                        _mm256_storeu_pd(result.as_mut_ptr(), abs_diff);
                        sum += result.iter().sum::<f64>() as Float;
                        i += chunk_size_f64;
                        continue;
                    }
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                sum += (x1[j] - x2[j]).abs();
            }

            Ok(sum)
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn cosine_distance_avx2(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if !is_x86_feature_detected!("avx2") {
            return self.cosine_distance_scalar(x1, x2);
        }

        unsafe {
            let len = x1.len();
            let chunk_size = 8;
            let mut dot_product = 0.0;
            let mut norm1_sq = 0.0;
            let mut norm2_sq = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    let a = _mm256_loadu_ps(x1.as_ptr().add(i) as *const f32);
                    let b = _mm256_loadu_ps(x2.as_ptr().add(i) as *const f32);

                    let dot = _mm256_mul_ps(a, b);
                    let a_sq = _mm256_mul_ps(a, a);
                    let b_sq = _mm256_mul_ps(b, b);

                    let mut dot_result = [0.0f32; 8];
                    let mut a_sq_result = [0.0f32; 8];
                    let mut b_sq_result = [0.0f32; 8];

                    _mm256_storeu_ps(dot_result.as_mut_ptr(), dot);
                    _mm256_storeu_ps(a_sq_result.as_mut_ptr(), a_sq);
                    _mm256_storeu_ps(b_sq_result.as_mut_ptr(), b_sq);

                    dot_product += dot_result.iter().sum::<f32>() as Float;
                    norm1_sq += a_sq_result.iter().sum::<f32>() as Float;
                    norm2_sq += b_sq_result.iter().sum::<f32>() as Float;
                } else {
                    let chunk_size_f64 = 4;
                    if i + chunk_size_f64 <= len {
                        let a = _mm256_loadu_pd(x1.as_ptr().add(i) as *const f64);
                        let b = _mm256_loadu_pd(x2.as_ptr().add(i) as *const f64);

                        let dot = _mm256_mul_pd(a, b);
                        let a_sq = _mm256_mul_pd(a, a);
                        let b_sq = _mm256_mul_pd(b, b);

                        let mut dot_result = [0.0f64; 4];
                        let mut a_sq_result = [0.0f64; 4];
                        let mut b_sq_result = [0.0f64; 4];

                        _mm256_storeu_pd(dot_result.as_mut_ptr(), dot);
                        _mm256_storeu_pd(a_sq_result.as_mut_ptr(), a_sq);
                        _mm256_storeu_pd(b_sq_result.as_mut_ptr(), b_sq);

                        dot_product += dot_result.iter().sum::<f64>() as Float;
                        norm1_sq += a_sq_result.iter().sum::<f64>() as Float;
                        norm2_sq += b_sq_result.iter().sum::<f64>() as Float;
                        i += chunk_size_f64;
                        continue;
                    }
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                dot_product += x1[j] * x2[j];
                norm1_sq += x1[j] * x1[j];
                norm2_sq += x2[j] * x2[j];
            }

            let norm1 = norm1_sq.sqrt();
            let norm2 = norm2_sq.sqrt();

            if norm1 == 0.0 || norm2 == 0.0 {
                return Err(NeighborsError::InvalidInput(
                    "Zero norm vector in cosine distance".to_string(),
                ));
            }

            Ok(1.0 - (dot_product / (norm1 * norm2)))
        }
    }

    // AVX implementations for x86_64
    #[cfg(target_arch = "x86_64")]
    fn euclidean_distance_avx(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if !is_x86_feature_detected!("avx") {
            return Ok(self.euclidean_distance_scalar(x1, x2));
        }

        unsafe {
            let len = x1.len();
            let chunk_size = if std::mem::size_of::<Float>() == 4 {
                8
            } else {
                4
            };
            let mut sum = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    // f32 case
                    let a = _mm256_loadu_ps(x1.as_ptr().add(i) as *const f32);
                    let b = _mm256_loadu_ps(x2.as_ptr().add(i) as *const f32);
                    let diff = _mm256_sub_ps(a, b);
                    let squared = _mm256_mul_ps(diff, diff);

                    let mut result = [0.0f32; 8];
                    _mm256_storeu_ps(result.as_mut_ptr(), squared);
                    sum += result.iter().sum::<f32>() as Float;
                } else {
                    // f64 case
                    let a = _mm256_loadu_pd(x1.as_ptr().add(i) as *const f64);
                    let b = _mm256_loadu_pd(x2.as_ptr().add(i) as *const f64);
                    let diff = _mm256_sub_pd(a, b);
                    let squared = _mm256_mul_pd(diff, diff);

                    let mut result = [0.0f64; 4];
                    _mm256_storeu_pd(result.as_mut_ptr(), squared);
                    sum += result.iter().sum::<f64>() as Float;
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                let diff = x1[j] - x2[j];
                sum += diff * diff;
            }

            Ok(sum.sqrt())
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn euclidean_distance_sse(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if !is_x86_feature_detected!("sse") {
            return Ok(self.euclidean_distance_scalar(x1, x2));
        }

        unsafe {
            let len = x1.len();
            let chunk_size = if std::mem::size_of::<Float>() == 4 {
                4
            } else {
                2
            };
            let mut sum = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    // f32 case
                    let a = _mm_loadu_ps(x1.as_ptr().add(i) as *const f32);
                    let b = _mm_loadu_ps(x2.as_ptr().add(i) as *const f32);
                    let diff = _mm_sub_ps(a, b);
                    let squared = _mm_mul_ps(diff, diff);

                    let mut result = [0.0f32; 4];
                    _mm_storeu_ps(result.as_mut_ptr(), squared);
                    sum += result.iter().sum::<f32>() as Float;
                } else {
                    // f64 case
                    let a = _mm_loadu_pd(x1.as_ptr().add(i) as *const f64);
                    let b = _mm_loadu_pd(x2.as_ptr().add(i) as *const f64);
                    let diff = _mm_sub_pd(a, b);
                    let squared = _mm_mul_pd(diff, diff);

                    let mut result = [0.0f64; 2];
                    _mm_storeu_pd(result.as_mut_ptr(), squared);
                    sum += result.iter().sum::<f64>() as Float;
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                let diff = x1[j] - x2[j];
                sum += diff * diff;
            }

            Ok(sum.sqrt())
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn manhattan_distance_avx(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if !is_x86_feature_detected!("avx") {
            return Ok(self.manhattan_distance_scalar(x1, x2));
        }

        unsafe {
            let len = x1.len();
            let chunk_size = if std::mem::size_of::<Float>() == 4 {
                8
            } else {
                4
            };
            let mut sum = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    let a = _mm256_loadu_ps(x1.as_ptr().add(i) as *const f32);
                    let b = _mm256_loadu_ps(x2.as_ptr().add(i) as *const f32);
                    let diff = _mm256_sub_ps(a, b);

                    // Create mask for absolute value
                    let sign_mask = _mm256_set1_ps(-0.0);
                    let abs_diff = _mm256_andnot_ps(sign_mask, diff);

                    let mut result = [0.0f32; 8];
                    _mm256_storeu_ps(result.as_mut_ptr(), abs_diff);
                    sum += result.iter().sum::<f32>() as Float;
                } else {
                    let a = _mm256_loadu_pd(x1.as_ptr().add(i) as *const f64);
                    let b = _mm256_loadu_pd(x2.as_ptr().add(i) as *const f64);
                    let diff = _mm256_sub_pd(a, b);

                    let sign_mask = _mm256_set1_pd(-0.0);
                    let abs_diff = _mm256_andnot_pd(sign_mask, diff);

                    let mut result = [0.0f64; 4];
                    _mm256_storeu_pd(result.as_mut_ptr(), abs_diff);
                    sum += result.iter().sum::<f64>() as Float;
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                sum += (x1[j] - x2[j]).abs();
            }

            Ok(sum)
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn manhattan_distance_sse(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if !is_x86_feature_detected!("sse") {
            return Ok(self.manhattan_distance_scalar(x1, x2));
        }

        unsafe {
            let len = x1.len();
            let chunk_size = if std::mem::size_of::<Float>() == 4 {
                4
            } else {
                2
            };
            let mut sum = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    let a = _mm_loadu_ps(x1.as_ptr().add(i) as *const f32);
                    let b = _mm_loadu_ps(x2.as_ptr().add(i) as *const f32);
                    let diff = _mm_sub_ps(a, b);

                    // Create mask for absolute value
                    let sign_mask = _mm_set1_ps(-0.0);
                    let abs_diff = _mm_andnot_ps(sign_mask, diff);

                    let mut result = [0.0f32; 4];
                    _mm_storeu_ps(result.as_mut_ptr(), abs_diff);
                    sum += result.iter().sum::<f32>() as Float;
                } else {
                    let a = _mm_loadu_pd(x1.as_ptr().add(i) as *const f64);
                    let b = _mm_loadu_pd(x2.as_ptr().add(i) as *const f64);
                    let diff = _mm_sub_pd(a, b);

                    let sign_mask = _mm_set1_pd(-0.0);
                    let abs_diff = _mm_andnot_pd(sign_mask, diff);

                    let mut result = [0.0f64; 2];
                    _mm_storeu_pd(result.as_mut_ptr(), abs_diff);
                    sum += result.iter().sum::<f64>() as Float;
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                sum += (x1[j] - x2[j]).abs();
            }

            Ok(sum)
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn cosine_distance_avx(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if !is_x86_feature_detected!("avx") {
            return self.cosine_distance_scalar(x1, x2);
        }

        unsafe {
            let len = x1.len();
            let chunk_size = if std::mem::size_of::<Float>() == 4 {
                8
            } else {
                4
            };
            let mut dot_product = 0.0;
            let mut norm1_sq = 0.0;
            let mut norm2_sq = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    let a = _mm256_loadu_ps(x1.as_ptr().add(i) as *const f32);
                    let b = _mm256_loadu_ps(x2.as_ptr().add(i) as *const f32);

                    let dot = _mm256_mul_ps(a, b);
                    let a_sq = _mm256_mul_ps(a, a);
                    let b_sq = _mm256_mul_ps(b, b);

                    let mut dot_result = [0.0f32; 8];
                    let mut a_sq_result = [0.0f32; 8];
                    let mut b_sq_result = [0.0f32; 8];

                    _mm256_storeu_ps(dot_result.as_mut_ptr(), dot);
                    _mm256_storeu_ps(a_sq_result.as_mut_ptr(), a_sq);
                    _mm256_storeu_ps(b_sq_result.as_mut_ptr(), b_sq);

                    dot_product += dot_result.iter().sum::<f32>() as Float;
                    norm1_sq += a_sq_result.iter().sum::<f32>() as Float;
                    norm2_sq += b_sq_result.iter().sum::<f32>() as Float;
                } else {
                    let a = _mm256_loadu_pd(x1.as_ptr().add(i) as *const f64);
                    let b = _mm256_loadu_pd(x2.as_ptr().add(i) as *const f64);

                    let dot = _mm256_mul_pd(a, b);
                    let a_sq = _mm256_mul_pd(a, a);
                    let b_sq = _mm256_mul_pd(b, b);

                    let mut dot_result = [0.0f64; 4];
                    let mut a_sq_result = [0.0f64; 4];
                    let mut b_sq_result = [0.0f64; 4];

                    _mm256_storeu_pd(dot_result.as_mut_ptr(), dot);
                    _mm256_storeu_pd(a_sq_result.as_mut_ptr(), a_sq);
                    _mm256_storeu_pd(b_sq_result.as_mut_ptr(), b_sq);

                    dot_product += dot_result.iter().sum::<f64>() as Float;
                    norm1_sq += a_sq_result.iter().sum::<f64>() as Float;
                    norm2_sq += b_sq_result.iter().sum::<f64>() as Float;
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                dot_product += x1[j] * x2[j];
                norm1_sq += x1[j] * x1[j];
                norm2_sq += x2[j] * x2[j];
            }

            let norm1 = norm1_sq.sqrt();
            let norm2 = norm2_sq.sqrt();

            if norm1 == 0.0 || norm2 == 0.0 {
                return Err(NeighborsError::InvalidInput(
                    "Zero norm vector in cosine distance".to_string(),
                ));
            }

            Ok(1.0 - (dot_product / (norm1 * norm2)))
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn cosine_distance_sse(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        if !is_x86_feature_detected!("sse") {
            return self.cosine_distance_scalar(x1, x2);
        }

        unsafe {
            let len = x1.len();
            let chunk_size = if std::mem::size_of::<Float>() == 4 {
                4
            } else {
                2
            };
            let mut dot_product = 0.0;
            let mut norm1_sq = 0.0;
            let mut norm2_sq = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    let a = _mm_loadu_ps(x1.as_ptr().add(i) as *const f32);
                    let b = _mm_loadu_ps(x2.as_ptr().add(i) as *const f32);

                    let dot = _mm_mul_ps(a, b);
                    let a_sq = _mm_mul_ps(a, a);
                    let b_sq = _mm_mul_ps(b, b);

                    let mut dot_result = [0.0f32; 4];
                    let mut a_sq_result = [0.0f32; 4];
                    let mut b_sq_result = [0.0f32; 4];

                    _mm_storeu_ps(dot_result.as_mut_ptr(), dot);
                    _mm_storeu_ps(a_sq_result.as_mut_ptr(), a_sq);
                    _mm_storeu_ps(b_sq_result.as_mut_ptr(), b_sq);

                    dot_product += dot_result.iter().sum::<f32>() as Float;
                    norm1_sq += a_sq_result.iter().sum::<f32>() as Float;
                    norm2_sq += b_sq_result.iter().sum::<f32>() as Float;
                } else {
                    let a = _mm_loadu_pd(x1.as_ptr().add(i) as *const f64);
                    let b = _mm_loadu_pd(x2.as_ptr().add(i) as *const f64);

                    let dot = _mm_mul_pd(a, b);
                    let a_sq = _mm_mul_pd(a, a);
                    let b_sq = _mm_mul_pd(b, b);

                    let mut dot_result = [0.0f64; 2];
                    let mut a_sq_result = [0.0f64; 2];
                    let mut b_sq_result = [0.0f64; 2];

                    _mm_storeu_pd(dot_result.as_mut_ptr(), dot);
                    _mm_storeu_pd(a_sq_result.as_mut_ptr(), a_sq);
                    _mm_storeu_pd(b_sq_result.as_mut_ptr(), b_sq);

                    dot_product += dot_result.iter().sum::<f64>() as Float;
                    norm1_sq += a_sq_result.iter().sum::<f64>() as Float;
                    norm2_sq += b_sq_result.iter().sum::<f64>() as Float;
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                dot_product += x1[j] * x2[j];
                norm1_sq += x1[j] * x1[j];
                norm2_sq += x2[j] * x2[j];
            }

            let norm1 = norm1_sq.sqrt();
            let norm2 = norm2_sq.sqrt();

            if norm1 == 0.0 || norm2 == 0.0 {
                return Err(NeighborsError::InvalidInput(
                    "Zero norm vector in cosine distance".to_string(),
                ));
            }

            Ok(1.0 - (dot_product / (norm1 * norm2)))
        }
    }

    // NEON implementations for aarch64
    #[cfg(target_arch = "aarch64")]
    fn euclidean_distance_neon(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        unsafe {
            let len = x1.len();
            let chunk_size = if std::mem::size_of::<Float>() == 4 {
                4
            } else {
                2
            };
            let mut sum = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    // f32 case
                    let a = vld1q_f32(x1.as_ptr().add(i) as *const f32);
                    let b = vld1q_f32(x2.as_ptr().add(i) as *const f32);
                    let diff = vsubq_f32(a, b);
                    let squared = vmulq_f32(diff, diff);

                    let mut result = [0.0f32; 4];
                    vst1q_f32(result.as_mut_ptr(), squared);
                    sum += result.iter().sum::<f32>() as Float;
                } else {
                    // f64 case
                    let a = vld1q_f64(x1.as_ptr().add(i));
                    let b = vld1q_f64(x2.as_ptr().add(i));
                    let diff = vsubq_f64(a, b);
                    let squared = vmulq_f64(diff, diff);

                    let mut result = [0.0f64; 2];
                    vst1q_f64(result.as_mut_ptr(), squared);
                    sum += result.iter().sum::<f64>() as Float;
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                let diff = x1[j] - x2[j];
                sum += diff * diff;
            }

            Ok(sum.sqrt())
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn manhattan_distance_neon(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        unsafe {
            let len = x1.len();
            let chunk_size = if std::mem::size_of::<Float>() == 4 {
                4
            } else {
                2
            };
            let mut sum = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    let a = vld1q_f32(x1.as_ptr().add(i) as *const f32);
                    let b = vld1q_f32(x2.as_ptr().add(i) as *const f32);
                    let diff = vsubq_f32(a, b);
                    let abs_diff = vabsq_f32(diff);

                    let mut result = [0.0f32; 4];
                    vst1q_f32(result.as_mut_ptr(), abs_diff);
                    sum += result.iter().sum::<f32>() as Float;
                } else {
                    let a = vld1q_f64(x1.as_ptr().add(i));
                    let b = vld1q_f64(x2.as_ptr().add(i));
                    let diff = vsubq_f64(a, b);
                    let abs_diff = vabsq_f64(diff);

                    let mut result = [0.0f64; 2];
                    vst1q_f64(result.as_mut_ptr(), abs_diff);
                    sum += result.iter().sum::<f64>() as Float;
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                sum += (x1[j] - x2[j]).abs();
            }

            Ok(sum)
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn cosine_distance_neon(
        &self,
        x1: ArrayView1<Float>,
        x2: ArrayView1<Float>,
    ) -> NeighborsResult<Float> {
        unsafe {
            let len = x1.len();
            let chunk_size = if std::mem::size_of::<Float>() == 4 {
                4
            } else {
                2
            };
            let mut dot_product = 0.0;
            let mut norm1_sq = 0.0;
            let mut norm2_sq = 0.0;

            // Process chunks
            let mut i = 0;
            while i + chunk_size <= len {
                if std::mem::size_of::<Float>() == 4 {
                    let a = vld1q_f32(x1.as_ptr().add(i) as *const f32);
                    let b = vld1q_f32(x2.as_ptr().add(i) as *const f32);

                    let dot = vmulq_f32(a, b);
                    let a_sq = vmulq_f32(a, a);
                    let b_sq = vmulq_f32(b, b);

                    let mut dot_result = [0.0f32; 4];
                    let mut a_sq_result = [0.0f32; 4];
                    let mut b_sq_result = [0.0f32; 4];

                    vst1q_f32(dot_result.as_mut_ptr(), dot);
                    vst1q_f32(a_sq_result.as_mut_ptr(), a_sq);
                    vst1q_f32(b_sq_result.as_mut_ptr(), b_sq);

                    dot_product += dot_result.iter().sum::<f32>() as Float;
                    norm1_sq += a_sq_result.iter().sum::<f32>() as Float;
                    norm2_sq += b_sq_result.iter().sum::<f32>() as Float;
                } else {
                    let a = vld1q_f64(x1.as_ptr().add(i));
                    let b = vld1q_f64(x2.as_ptr().add(i));

                    let dot = vmulq_f64(a, b);
                    let a_sq = vmulq_f64(a, a);
                    let b_sq = vmulq_f64(b, b);

                    let mut dot_result = [0.0f64; 2];
                    let mut a_sq_result = [0.0f64; 2];
                    let mut b_sq_result = [0.0f64; 2];

                    vst1q_f64(dot_result.as_mut_ptr(), dot);
                    vst1q_f64(a_sq_result.as_mut_ptr(), a_sq);
                    vst1q_f64(b_sq_result.as_mut_ptr(), b_sq);

                    dot_product += dot_result.iter().sum::<f64>() as Float;
                    norm1_sq += a_sq_result.iter().sum::<f64>() as Float;
                    norm2_sq += b_sq_result.iter().sum::<f64>() as Float;
                }
                i += chunk_size;
            }

            // Process remaining elements
            for j in i..len {
                dot_product += x1[j] * x2[j];
                norm1_sq += x1[j] * x1[j];
                norm2_sq += x2[j] * x2[j];
            }

            let norm1 = norm1_sq.sqrt();
            let norm2 = norm2_sq.sqrt();

            if norm1 == 0.0 || norm2 == 0.0 {
                return Err(NeighborsError::InvalidInput(
                    "Zero norm vector in cosine distance".to_string(),
                ));
            }

            Ok(1.0 - (dot_product / (norm1 * norm2)))
        }
    }
}

fn detect_simd_capability() -> SimdCapability {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return SimdCapability::AVX2;
        }
        if is_x86_feature_detected!("avx") {
            return SimdCapability::AVX;
        }
        if is_x86_feature_detected!("sse") {
            return SimdCapability::SSE;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        SimdCapability::NEON
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        SimdCapability::None
    }
}

// Batch distance computation optimized with SIMD
pub fn batch_euclidean_distances(
    query: ArrayView1<Float>,
    targets: &[ArrayView1<Float>],
    calculator: &SimdDistanceCalculator,
) -> NeighborsResult<Vec<Float>> {
    let mut distances = Vec::with_capacity(targets.len());

    for target in targets {
        distances.push(calculator.euclidean_distance(query, *target)?);
    }

    Ok(distances)
}

pub fn pairwise_distances_simd(
    x: &[ArrayView1<Float>],
    y: &[ArrayView1<Float>],
    calculator: &SimdDistanceCalculator,
) -> NeighborsResult<Vec<Vec<Float>>> {
    let mut distances = Vec::with_capacity(x.len());

    for x_item in x {
        let mut row_distances = Vec::with_capacity(y.len());
        for y_item in y {
            row_distances.push(calculator.euclidean_distance(*x_item, *y_item)?);
        }
        distances.push(row_distances);
    }

    Ok(distances)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_capability_detection() {
        let calculator = SimdDistanceCalculator::new();
        // Just verify it doesn't panic
        let capability = calculator.capability();

        // Test based on target architecture
        #[cfg(target_arch = "x86_64")]
        {
            assert!(matches!(
                capability,
                SimdCapability::None
                    | SimdCapability::SSE
                    | SimdCapability::AVX
                    | SimdCapability::AVX2
            ));
        }

        #[cfg(target_arch = "aarch64")]
        {
            assert!(matches!(
                capability,
                SimdCapability::None | SimdCapability::NEON
            ));
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            assert!(matches!(capability, SimdCapability::None));
        }
    }

    #[test]
    fn test_euclidean_distance_simd_vs_scalar() {
        let calculator_simd = SimdDistanceCalculator::new();
        let calculator_scalar = SimdDistanceCalculator::with_simd_disabled();

        let x1 = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = array![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let dist_simd = calculator_simd
            .euclidean_distance(x1.view(), x2.view())
            .unwrap();
        let dist_scalar = calculator_scalar
            .euclidean_distance(x1.view(), x2.view())
            .unwrap();

        assert_relative_eq!(dist_simd, dist_scalar, epsilon = 1e-6);
    }

    #[test]
    fn test_manhattan_distance_simd_vs_scalar() {
        let calculator_simd = SimdDistanceCalculator::new();
        let calculator_scalar = SimdDistanceCalculator::with_simd_disabled();

        let x1 = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = array![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let dist_simd = calculator_simd
            .manhattan_distance(x1.view(), x2.view())
            .unwrap();
        let dist_scalar = calculator_scalar
            .manhattan_distance(x1.view(), x2.view())
            .unwrap();

        assert_relative_eq!(dist_simd, dist_scalar, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_simd_vs_scalar() {
        let calculator_simd = SimdDistanceCalculator::new();
        let calculator_scalar = SimdDistanceCalculator::with_simd_disabled();

        let x1 = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = array![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let dist_simd = calculator_simd
            .cosine_distance(x1.view(), x2.view())
            .unwrap();
        let dist_scalar = calculator_scalar
            .cosine_distance(x1.view(), x2.view())
            .unwrap();

        assert_relative_eq!(dist_simd, dist_scalar, epsilon = 1e-6);
    }

    #[test]
    fn test_batch_euclidean_distances() {
        let calculator = SimdDistanceCalculator::new();
        let query = array![1.0, 2.0, 3.0, 4.0];
        let target1 = array![2.0, 3.0, 4.0, 5.0];
        let target2 = array![3.0, 4.0, 5.0, 6.0];
        let targets = vec![target1.view(), target2.view()];

        let distances = batch_euclidean_distances(query.view(), &targets, &calculator).unwrap();

        assert_eq!(distances.len(), 2);
        assert_relative_eq!(distances[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(distances[1], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let calculator = SimdDistanceCalculator::new();
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![1.0, 2.0];

        let result = calculator.euclidean_distance(x1.view(), x2.view());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            NeighborsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_cosine_distance_zero_norm_error() {
        let calculator = SimdDistanceCalculator::new();
        let x1 = array![0.0, 0.0, 0.0];
        let x2 = array![1.0, 2.0, 3.0];

        let result = calculator.cosine_distance(x1.view(), x2.view());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            NeighborsError::InvalidInput(_)
        ));
    }

    #[test]
    fn test_small_vector_fallback() {
        let calculator = SimdDistanceCalculator::new();
        let x1 = array![1.0, 2.0]; // Small vector should fallback to scalar
        let x2 = array![3.0, 4.0];

        let dist = calculator.euclidean_distance(x1.view(), x2.view()).unwrap();
        assert_relative_eq!(
            dist,
            ((2.0_f64).powi(2) + (2.0_f64).powi(2)).sqrt(),
            epsilon = 1e-6
        );
    }
}
