//! SIMD-optimized bit-level operations for efficient data manipulation
//!
//! This module provides high-performance bit manipulation operations commonly used
//! in machine learning for sparse data processing, feature hashing, and boolean indexing.

/// Population count (popcount) operations for counting set bits
pub mod popcount {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use core::arch::x86_64::*;

    /// Count the number of set bits in a slice of u64 values using SIMD
    pub fn popcount_u64_slice(data: &[u64]) -> usize {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") && crate::simd_feature_detected!("popcnt") {
                return unsafe { popcount_u64_slice_avx2(data) };
            } else if crate::simd_feature_detected!("sse4.2")
                && crate::simd_feature_detected!("popcnt")
            {
                return unsafe { popcount_u64_slice_sse42(data) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            popcount_u64_slice_neon(data)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            popcount_u64_slice_scalar(data)
        }
    }

    /// Count set bits in u32 slice
    pub fn popcount_u32_slice(data: &[u32]) -> usize {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") && crate::simd_feature_detected!("popcnt") {
                return unsafe { popcount_u32_slice_avx2(data) };
            } else if crate::simd_feature_detected!("sse4.2")
                && crate::simd_feature_detected!("popcnt")
            {
                return unsafe { popcount_u32_slice_sse42(data) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            popcount_u32_slice_neon(data)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            popcount_u32_slice_scalar(data)
        }
    }

    /// Scalar implementation for u64 popcount
    fn popcount_u64_slice_scalar(data: &[u64]) -> usize {
        data.iter().map(|&x| x.count_ones() as usize).sum()
    }

    /// Scalar implementation for u32 popcount
    fn popcount_u32_slice_scalar(data: &[u32]) -> usize {
        data.iter().map(|&x| x.count_ones() as usize).sum()
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,popcnt")]
    unsafe fn popcount_u64_slice_avx2(data: &[u64]) -> usize {
        let mut total = 0usize;
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            // Extract lanes and use hardware popcount
            total += _popcnt64(_mm256_extract_epi64(vec, 0)) as usize;
            total += _popcnt64(_mm256_extract_epi64(vec, 1)) as usize;
            total += _popcnt64(_mm256_extract_epi64(vec, 2)) as usize;
            total += _popcnt64(_mm256_extract_epi64(vec, 3)) as usize;
        }

        // Handle remainder
        for &val in remainder {
            total += _popcnt64(val as i64) as usize;
        }

        total
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.2,popcnt")]
    unsafe fn popcount_u64_slice_sse42(data: &[u64]) -> usize {
        let mut total = 0usize;
        for &val in data {
            total += _popcnt64(val as i64) as usize;
        }
        total
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,popcnt")]
    unsafe fn popcount_u32_slice_avx2(data: &[u32]) -> usize {
        let mut total = 0usize;
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            // Extract lanes and use hardware popcount
            total += _popcnt32(_mm256_extract_epi32(vec, 0)) as usize;
            total += _popcnt32(_mm256_extract_epi32(vec, 1)) as usize;
            total += _popcnt32(_mm256_extract_epi32(vec, 2)) as usize;
            total += _popcnt32(_mm256_extract_epi32(vec, 3)) as usize;
            total += _popcnt32(_mm256_extract_epi32(vec, 4)) as usize;
            total += _popcnt32(_mm256_extract_epi32(vec, 5)) as usize;
            total += _popcnt32(_mm256_extract_epi32(vec, 6)) as usize;
            total += _popcnt32(_mm256_extract_epi32(vec, 7)) as usize;
        }

        // Handle remainder
        for &val in remainder {
            total += _popcnt32(val as i32) as usize;
        }

        total
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.2,popcnt")]
    unsafe fn popcount_u32_slice_sse42(data: &[u32]) -> usize {
        let mut total = 0usize;
        for &val in data {
            total += _popcnt32(val as i32) as usize;
        }
        total
    }

    #[cfg(target_arch = "aarch64")]
    fn popcount_u64_slice_neon(data: &[u64]) -> usize {
        let mut total = 0usize;
        for &val in data {
            total += val.count_ones() as usize;
        }
        total
    }

    #[cfg(target_arch = "aarch64")]
    fn popcount_u32_slice_neon(data: &[u32]) -> usize {
        let mut total = 0usize;
        for &val in data {
            total += val.count_ones() as usize;
        }
        total
    }
}

/// Bit manipulation operations for feature engineering and boolean indexing
pub mod bit_manipulation {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use core::arch::x86_64::*;

    #[cfg(feature = "no-std")]
    use alloc::vec::Vec;

    /// Reverse the bits in each u32 element of a slice
    pub fn reverse_bits_u32_slice(data: &mut [u32]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                return unsafe { reverse_bits_u32_slice_avx2(data) };
            } else if crate::simd_feature_detected!("sse2") {
                return unsafe { reverse_bits_u32_slice_sse2(data) };
            }
        }

        reverse_bits_u32_slice_scalar(data);
    }

    /// Parallel bit extraction using masks
    pub fn parallel_bit_extract(data: &[u64], mask: u64) -> Vec<u64> {
        let mut result = Vec::with_capacity(data.len());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("bmi2") {
                return unsafe { parallel_bit_extract_bmi2(data, mask) };
            }
        }

        // Scalar fallback
        for &val in data {
            result.push(parallel_bit_extract_scalar(val, mask));
        }
        result
    }

    /// Count leading zeros in parallel
    pub fn count_leading_zeros_slice(data: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(data.len());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") && crate::simd_feature_detected!("lzcnt") {
                return unsafe { count_leading_zeros_slice_avx2(data) };
            }
        }

        // Scalar fallback
        for &val in data {
            result.push(val.leading_zeros());
        }
        result
    }

    fn reverse_bits_u32_slice_scalar(data: &mut [u32]) {
        for val in data {
            *val = val.reverse_bits();
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn reverse_bits_u32_slice_avx2(data: &mut [u32]) {
        let mut chunks = data.chunks_exact_mut(8);
        let remainder_slice = chunks.by_ref();

        for chunk in remainder_slice {
            // Load 8 u32 values
            let vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            // Reverse bits using shuffles and bit operations
            // This is a simplified approach - full bit reversal requires more complex operations
            let reversed = reverse_bits_avx2(vec);

            _mm256_storeu_si256(chunk.as_mut_ptr() as *mut __m256i, reversed);
        }

        // Handle remainder with scalar
        let remainder = chunks.into_remainder();
        reverse_bits_u32_slice_scalar(remainder);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn reverse_bits_avx2(vec: __m256i) -> __m256i {
        // Simplified bit reversal using shifts and masks
        // Full implementation would need lookup tables or more complex bit manipulation
        let mask_55 = _mm256_set1_epi32(0x55555555u32 as i32);
        let mask_33 = _mm256_set1_epi32(0x33333333u32 as i32);
        let mask_0f = _mm256_set1_epi32(0x0f0f0f0fu32 as i32);
        let mask_ff = _mm256_set1_epi32(0x00ff00ffu32 as i32);

        let mut x = vec;

        // Swap pairs of bits
        x = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi32(x, 1), mask_55),
            _mm256_slli_epi32(_mm256_and_si256(x, mask_55), 1),
        );

        // Swap pairs of 2-bit groups
        x = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi32(x, 2), mask_33),
            _mm256_slli_epi32(_mm256_and_si256(x, mask_33), 2),
        );

        // Swap pairs of 4-bit groups
        x = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi32(x, 4), mask_0f),
            _mm256_slli_epi32(_mm256_and_si256(x, mask_0f), 4),
        );

        // Swap pairs of bytes
        x = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi32(x, 8), mask_ff),
            _mm256_slli_epi32(_mm256_and_si256(x, mask_ff), 8),
        );

        // Swap 16-bit halves
        _mm256_or_si256(_mm256_srli_epi32(x, 16), _mm256_slli_epi32(x, 16))
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn reverse_bits_u32_slice_sse2(data: &mut [u32]) {
        // SSE2 implementation similar to AVX2 but with 128-bit registers
        let mut chunks = data.chunks_exact_mut(4);
        let remainder_slice = chunks.by_ref();

        for chunk in remainder_slice {
            let vec = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
            let reversed = reverse_bits_sse2(vec);
            _mm_storeu_si128(chunk.as_mut_ptr() as *mut __m128i, reversed);
        }

        let remainder = chunks.into_remainder();
        reverse_bits_u32_slice_scalar(remainder);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn reverse_bits_sse2(vec: __m128i) -> __m128i {
        // Similar to AVX2 version but with SSE2 instructions
        let mask_55 = _mm_set1_epi32(0x55555555u32 as i32);
        let mask_33 = _mm_set1_epi32(0x33333333u32 as i32);
        let mask_0f = _mm_set1_epi32(0x0f0f0f0fu32 as i32);
        let mask_ff = _mm_set1_epi32(0x00ff00ffu32 as i32);

        let mut x = vec;

        x = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32(x, 1), mask_55),
            _mm_slli_epi32(_mm_and_si128(x, mask_55), 1),
        );

        x = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32(x, 2), mask_33),
            _mm_slli_epi32(_mm_and_si128(x, mask_33), 2),
        );

        x = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32(x, 4), mask_0f),
            _mm_slli_epi32(_mm_and_si128(x, mask_0f), 4),
        );

        x = _mm_or_si128(
            _mm_and_si128(_mm_srli_epi32(x, 8), mask_ff),
            _mm_slli_epi32(_mm_and_si128(x, mask_ff), 8),
        );

        _mm_or_si128(_mm_srli_epi32(x, 16), _mm_slli_epi32(x, 16))
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "bmi2")]
    unsafe fn parallel_bit_extract_bmi2(data: &[u64], mask: u64) -> Vec<u64> {
        let mut result = Vec::with_capacity(data.len());
        for &val in data {
            result.push(_pext_u64(val, mask));
        }
        result
    }

    fn parallel_bit_extract_scalar(val: u64, mask: u64) -> u64 {
        let mut result = 0u64;
        let mut mask_bit = 1u64;
        let mut result_bit = 1u64;

        for _ in 0..64 {
            if mask & mask_bit != 0 {
                if val & mask_bit != 0 {
                    result |= result_bit;
                }
                result_bit <<= 1;
            }
            mask_bit <<= 1;
        }

        result
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,lzcnt")]
    unsafe fn count_leading_zeros_slice_avx2(data: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(data.len());

        for &val in data {
            result.push(_lzcnt_u32(val));
        }

        result
    }
}

/// Hash functions for feature hashing and approximate algorithms
pub mod hash_functions {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use core::arch::x86_64::*;

    #[cfg(feature = "no-std")]
    use alloc::vec::Vec;

    /// Fast hash function using CRC32 instruction if available
    pub fn crc32_hash(data: &[u8]) -> u32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("sse4.2") {
                return unsafe { crc32_hash_sse42(data) };
            }
        }

        crc32_hash_scalar(data)
    }

    /// MurmurHash3 implementation for feature hashing
    pub fn murmur3_hash(data: &[u8], seed: u32) -> u32 {
        murmur3_hash_scalar(data, seed)
    }

    /// Fast hash for u64 values using multiplication and shifts
    pub fn fast_hash_u64(val: u64) -> u64 {
        // Multiply by a large prime and mix bits
        let mut x = val.wrapping_mul(0x9e3779b97f4a7c15);
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        x
    }

    /// Hash multiple u64 values in parallel
    pub fn fast_hash_u64_slice(data: &[u64]) -> Vec<u64> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                return unsafe { fast_hash_u64_slice_avx2(data) };
            }
        }

        data.iter().map(|&x| fast_hash_u64(x)).collect()
    }

    fn crc32_hash_scalar(data: &[u8]) -> u32 {
        // Simple polynomial-based hash as fallback
        let mut hash = 0xffffffffu32;
        for &byte in data {
            hash ^= byte as u32;
            for _ in 0..8 {
                if hash & 1 != 0 {
                    hash = (hash >> 1) ^ 0xedb88320;
                } else {
                    hash >>= 1;
                }
            }
        }
        !hash
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.2")]
    unsafe fn crc32_hash_sse42(data: &[u8]) -> u32 {
        let mut crc = 0xffffffffu32;

        // Process 8 bytes at a time
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let val = core::ptr::read_unaligned(chunk.as_ptr() as *const u64);
            crc = _mm_crc32_u64(crc as u64, val) as u32;
        }

        // Process remaining bytes
        for &byte in remainder {
            crc = _mm_crc32_u8(crc, byte);
        }

        !crc
    }

    fn murmur3_hash_scalar(data: &[u8], seed: u32) -> u32 {
        const C1: u32 = 0xcc9e2d51;
        const C2: u32 = 0x1b873593;
        const R1: u32 = 15;
        const R2: u32 = 13;
        const M: u32 = 5;
        const N: u32 = 0xe6546b64;

        let mut hash = seed;
        let len = data.len();

        // Process 4-byte chunks
        for chunk in data.chunks_exact(4) {
            let mut k = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            k = k.wrapping_mul(C1);
            k = k.rotate_left(R1);
            k = k.wrapping_mul(C2);

            hash ^= k;
            hash = hash.rotate_left(R2);
            hash = hash.wrapping_mul(M).wrapping_add(N);
        }

        // Handle remaining bytes
        let remaining = &data[data.len() & !3..];
        let mut k1 = 0u32;

        if remaining.len() >= 3 {
            k1 ^= (remaining[2] as u32) << 16;
        }
        if remaining.len() >= 2 {
            k1 ^= (remaining[1] as u32) << 8;
        }
        if !remaining.is_empty() {
            k1 ^= remaining[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = k1.rotate_left(R1);
            k1 = k1.wrapping_mul(C2);
            hash ^= k1;
        }

        // Finalization
        hash ^= len as u32;
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85ebca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2ae35);
        hash ^= hash >> 16;

        hash
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn fast_hash_u64_slice_avx2(data: &[u64]) -> Vec<u64> {
        let mut result = Vec::with_capacity(data.len());

        let prime1 = _mm256_set1_epi64x(0x9e3779b97f4a7c15u64 as i64);
        let prime2 = _mm256_set1_epi64x(0xbf58476d1ce4e5b9u64 as i64);
        let prime3 = _mm256_set1_epi64x(0x94d049bb133111ebu64 as i64);

        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            // Perform hash operations in parallel
            let mut x = _mm256_mul_epu32(vec, prime1);
            x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 30));
            x = _mm256_mul_epu32(x, prime2);
            x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 27));
            x = _mm256_mul_epu32(x, prime3);
            x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 31));

            // Store results
            let mut temp = [0u64; 4];
            _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, x);
            result.extend_from_slice(&temp);
        }

        // Handle remainder
        for &val in remainder {
            result.push(fast_hash_u64(val));
        }

        result
    }
}

/// Boolean indexing operations for filtering and selection
pub mod boolean_indexing {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use core::arch::x86_64::*;

    #[cfg(feature = "no-std")]
    use alloc::vec::Vec;

    /// Compress data based on boolean mask
    pub fn compress_by_mask_f32(data: &[f32], mask: &[bool]) -> Vec<f32> {
        assert_eq!(data.len(), mask.len());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                return unsafe { compress_by_mask_f32_avx2(data, mask) };
            }
        }

        compress_by_mask_f32_scalar(data, mask)
    }

    /// Create boolean mask from comparison operation
    pub fn create_mask_greater_than_f32(data: &[f32], threshold: f32) -> Vec<bool> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                return unsafe { create_mask_greater_than_f32_avx2(data, threshold) };
            }
        }

        create_mask_greater_than_f32_scalar(data, threshold)
    }

    /// Count true values in boolean mask
    pub fn count_true_mask(mask: &[bool]) -> usize {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("avx2") {
                return unsafe { count_true_mask_avx2(mask) };
            }
        }

        mask.iter().map(|&b| b as usize).sum()
    }

    fn compress_by_mask_f32_scalar(data: &[f32], mask: &[bool]) -> Vec<f32> {
        data.iter()
            .zip(mask.iter())
            .filter_map(|(&val, &include)| if include { Some(val) } else { None })
            .collect()
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn compress_by_mask_f32_avx2(data: &[f32], mask: &[bool]) -> Vec<f32> {
        let mut result = Vec::new();

        let chunks_data = data.chunks_exact(8);
        let chunks_mask = mask.chunks_exact(8);

        // Get remainders before consuming iterators
        let remaining_data = chunks_data.remainder();
        let remaining_mask = chunks_mask.remainder();

        for (data_chunk, mask_chunk) in chunks_data.zip(chunks_mask) {
            // Create mask from boolean array
            let mut mask_bits = 0u8;
            for (i, &b) in mask_chunk.iter().enumerate() {
                if b {
                    mask_bits |= 1 << i;
                }
            }

            // Extract elements based on mask
            for i in 0..8 {
                if mask_bits & (1 << i) != 0 {
                    result.push(data_chunk[i]);
                }
            }
        }

        // Handle remainder
        result.extend(compress_by_mask_f32_scalar(remaining_data, remaining_mask));

        result
    }

    fn create_mask_greater_than_f32_scalar(data: &[f32], threshold: f32) -> Vec<bool> {
        data.iter().map(|&x| x > threshold).collect()
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn create_mask_greater_than_f32_avx2(data: &[f32], threshold: f32) -> Vec<bool> {
        let mut result = Vec::with_capacity(data.len());
        let threshold_vec = _mm256_set1_ps(threshold);

        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let data_vec = _mm256_loadu_ps(chunk.as_ptr());
            let cmp_result = _mm256_cmp_ps(data_vec, threshold_vec, _CMP_GT_OQ);
            let mask = _mm256_movemask_ps(cmp_result);

            // Convert mask to boolean array
            for i in 0..8 {
                result.push((mask & (1 << i)) != 0);
            }
        }

        // Handle remainder
        result.extend(create_mask_greater_than_f32_scalar(remainder, threshold));

        result
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn count_true_mask_avx2(mask: &[bool]) -> usize {
        let mut count = 0usize;

        let chunks = mask.chunks_exact(32); // Process 32 bools at a time
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Convert booleans to packed bytes
            let mut packed = [0u8; 32];
            for (i, &b) in chunk.iter().enumerate() {
                packed[i] = b as u8;
            }

            let vec = _mm256_loadu_si256(packed.as_ptr() as *const __m256i);

            // Sum all bytes
            let zero = _mm256_setzero_si256();
            let sum = _mm256_sad_epu8(vec, zero);

            // Extract and sum the results
            count += _mm256_extract_epi64(sum, 0) as usize;
            count += _mm256_extract_epi64(sum, 1) as usize;
            count += _mm256_extract_epi64(sum, 2) as usize;
            count += _mm256_extract_epi64(sum, 3) as usize;
        }

        // Handle remainder
        count += remainder.iter().map(|&b| b as usize).sum::<usize>();

        count
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_popcount_u64() {
        let data = vec![0xFF, 0x00, 0xF0F0F0F0F0F0F0F0, 0x5555555555555555];
        let expected = 8 + 0 + 32 + 32; // 72 total bits

        let result = popcount::popcount_u64_slice(&data);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_popcount_u32() {
        let data = vec![0xFF, 0x00, 0xF0F0F0F0, 0x55555555];
        let expected = 8 + 0 + 16 + 16; // 40 total bits

        let result = popcount::popcount_u32_slice(&data);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_reverse_bits() {
        let mut data = vec![0x12345678u32, 0xABCDEF00u32];
        let expected = vec![0x1E6A2C48u32, 0x00F7B3D5u32];

        bit_manipulation::reverse_bits_u32_slice(&mut data);
        assert_eq!(data, expected);
    }

    #[test]
    fn test_parallel_bit_extract() {
        let data = vec![0b11110000u64, 0b10101010u64];
        let mask = 0b11001100u64;

        let result = bit_manipulation::parallel_bit_extract(&data, mask);

        // Verify that only bits at mask positions are extracted
        assert_eq!(result.len(), data.len());
        for &val in &result {
            assert!(val <= 0b1111); // At most 4 bits can be extracted
        }
    }

    #[test]
    fn test_count_leading_zeros() {
        let data = vec![0x00000001u32, 0x00000100u32, 0x80000000u32];
        let expected = vec![31, 23, 0];

        let result = bit_manipulation::count_leading_zeros_slice(&data);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_crc32_hash() {
        let data = b"hello world";
        let hash1 = hash_functions::crc32_hash(data);
        let hash2 = hash_functions::crc32_hash(data);

        // Same input should produce same hash
        assert_eq!(hash1, hash2);

        // Different input should (likely) produce different hash
        let different_hash = hash_functions::crc32_hash(b"hello world!");
        assert_ne!(hash1, different_hash);
    }

    #[test]
    fn test_murmur3_hash() {
        let data = b"test data";
        let seed = 42;

        let hash1 = hash_functions::murmur3_hash(data, seed);
        let hash2 = hash_functions::murmur3_hash(data, seed);

        assert_eq!(hash1, hash2);

        // Different seed should produce different hash
        let different_hash = hash_functions::murmur3_hash(data, seed + 1);
        assert_ne!(hash1, different_hash);
    }

    #[test]
    fn test_fast_hash_u64() {
        let data = vec![0, 1, 2, 0x123456789ABCDEF0];

        let result = hash_functions::fast_hash_u64_slice(&data);
        assert_eq!(result.len(), data.len());

        // Check that hashes are different for different inputs
        for i in 0..data.len() {
            for j in i + 1..data.len() {
                if data[i] != data[j] {
                    assert_ne!(result[i], result[j]);
                }
            }
        }
    }

    #[test]
    fn test_compress_by_mask() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true, false, true, false, true];
        let expected = vec![1.0, 3.0, 5.0];

        let result = boolean_indexing::compress_by_mask_f32(&data, &mask);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_create_mask_greater_than() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let threshold = 3.0;
        let expected = vec![false, false, false, true, true];

        let result = boolean_indexing::create_mask_greater_than_f32(&data, threshold);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_count_true_mask() {
        let mask = vec![true, false, true, true, false, true];
        let expected = 4;

        let result = boolean_indexing::count_true_mask(&mask);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_empty_inputs() {
        // Test empty slices don't crash
        assert_eq!(popcount::popcount_u64_slice(&[]), 0);
        assert_eq!(popcount::popcount_u32_slice(&[]), 0);

        let empty_data: Vec<f32> = vec![];
        let empty_mask: Vec<bool> = vec![];
        assert_eq!(
            boolean_indexing::compress_by_mask_f32(&empty_data, &empty_mask),
            vec![] as Vec<f32>
        );
        assert_eq!(boolean_indexing::count_true_mask(&empty_mask), 0);
    }

    #[test]
    fn test_large_inputs() {
        // Test with larger inputs to exercise SIMD paths
        let large_data: Vec<u64> = (0..1000).map(|i| i as u64).collect();
        let count = popcount::popcount_u64_slice(&large_data);

        // Should be non-zero for this range
        assert!(count > 0);

        // Test boolean operations on large data
        let large_float_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let mask = boolean_indexing::create_mask_greater_than_f32(&large_float_data, 500.0);
        let count_true = boolean_indexing::count_true_mask(&mask);

        assert_eq!(count_true, 499); // Numbers 501-999 are greater than 500
    }
}
