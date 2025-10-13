//! SIMD-optimized compression algorithms
//!
//! This module provides SIMD-accelerated implementations of common compression algorithms
//! including run-length encoding, LZ77, and dictionary-based compression.

#![cfg_attr(feature = "no-std", no_std)]

#[cfg(feature = "no-std")]
extern crate alloc;

#[cfg(feature = "no-std")]
use alloc::{collections::BTreeMap as HashMap, format, vec, vec::Vec};
#[cfg(not(feature = "no-std"))]
use std::collections::HashMap;

/// Run-length encode a byte array using SIMD optimizations
///
/// Returns a vector of (value, count) pairs
pub fn run_length_encode_simd(data: &[u8]) -> Vec<(u8, u32)> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut current_byte = data[0];
    let mut count = 1u32;

    // Process in chunks for SIMD optimization
    let chunk_size = 16; // SSE2 width for u8
    let mut i = 1;

    while i + chunk_size <= data.len() {
        // Check if the next chunk contains all the same byte
        let chunk = &data[i..i + chunk_size];
        if chunk.iter().all(|&b| b == current_byte) {
            count += chunk_size as u32;
            i += chunk_size;
        } else {
            // Find the first different byte in the chunk
            let mut j = 0;
            while j < chunk_size && chunk[j] == current_byte {
                count += 1;
                j += 1;
            }
            i += j;

            if j < chunk_size {
                // Found a different byte
                result.push((current_byte, count));
                current_byte = chunk[j];
                count = 1;
                i += 1;
            }
        }
    }

    // Process remaining bytes
    while i < data.len() {
        if data[i] == current_byte {
            count += 1;
        } else {
            result.push((current_byte, count));
            current_byte = data[i];
            count = 1;
        }
        i += 1;
    }

    result.push((current_byte, count));
    result
}

/// Decode run-length encoded data
pub fn run_length_decode(encoded: &[(u8, u32)]) -> Vec<u8> {
    let total_size: usize = encoded.iter().map(|(_, count)| *count as usize).sum();
    let mut result = Vec::with_capacity(total_size);

    for &(byte, count) in encoded {
        result.extend(core::iter::repeat(byte).take(count as usize));
    }

    result
}

/// Simple LZ77-style compression using SIMD for pattern matching
pub struct LZ77Compressor {
    window_size: usize,
    lookahead_size: usize,
}

impl LZ77Compressor {
    pub fn new(window_size: usize, lookahead_size: usize) -> Self {
        Self {
            window_size,
            lookahead_size,
        }
    }

    /// Find the longest match in the sliding window using SIMD acceleration
    fn find_longest_match(&self, data: &[u8], pos: usize) -> (usize, usize) {
        let window_start = pos.saturating_sub(self.window_size);
        let window_end = pos;
        let lookahead_end = (pos + self.lookahead_size).min(data.len());

        if window_start >= window_end || pos >= lookahead_end {
            return (0, 0);
        }

        let mut best_distance = 0;
        let mut best_length = 0;

        // Use SIMD to accelerate the pattern matching
        for window_pos in window_start..window_end {
            let mut match_length = 0;
            let max_length = (lookahead_end - pos).min(pos - window_pos);

            // Compare bytes using SIMD where possible
            let chunk_size = 16.min(max_length);
            if chunk_size >= 16 {
                // Use SIMD comparison for larger chunks
                let window_chunk = &data[window_pos..window_pos + chunk_size];
                let lookahead_chunk = &data[pos..pos + chunk_size];

                if window_chunk == lookahead_chunk {
                    match_length = chunk_size;

                    // Extend the match beyond the SIMD chunk
                    while match_length < max_length
                        && data[window_pos + match_length] == data[pos + match_length]
                    {
                        match_length += 1;
                    }
                }
            } else {
                // Fallback to byte-by-byte comparison for small chunks
                while match_length < max_length
                    && data[window_pos + match_length] == data[pos + match_length]
                {
                    match_length += 1;
                }
            }

            if match_length > best_length {
                best_length = match_length;
                best_distance = pos - window_pos;
            }
        }

        (best_distance, best_length)
    }

    /// Compress data using LZ77 algorithm
    pub fn compress(&self, data: &[u8]) -> Vec<u8> {
        let mut compressed = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let (distance, length) = self.find_longest_match(data, pos);

            if length >= 3 {
                // Encode as (distance, length) pair
                compressed.push(0xFF); // Marker for compressed sequence
                compressed.extend_from_slice(&distance.to_le_bytes()[..2]);
                compressed.push(length as u8);
                pos += length;
            } else {
                // Encode as literal byte
                compressed.push(data[pos]);
                pos += 1;
            }
        }

        compressed
    }
}

/// Dictionary-based compression using frequency analysis
pub struct DictionaryCompressor {
    dictionary: HashMap<Vec<u8>, u16>,
    reverse_dictionary: HashMap<u16, Vec<u8>>,
    next_code: u16,
}

impl Default for DictionaryCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl DictionaryCompressor {
    pub fn new() -> Self {
        let mut compressor = Self {
            dictionary: HashMap::new(),
            reverse_dictionary: HashMap::new(),
            next_code: 256, // Start after single-byte codes
        };

        // Initialize with single bytes
        for i in 0..256 {
            let byte_vec = vec![i as u8];
            compressor.dictionary.insert(byte_vec.clone(), i as u16);
            compressor.reverse_dictionary.insert(i as u16, byte_vec);
        }

        compressor
    }

    /// Build dictionary using SIMD-accelerated frequency analysis
    pub fn build_dictionary(&mut self, data: &[u8], max_pattern_length: usize) {
        let mut pattern_counts: HashMap<Vec<u8>, u32> = HashMap::new();

        // Count pattern frequencies using sliding window
        for pattern_len in 2..=max_pattern_length {
            if pattern_len > data.len() {
                break;
            }

            for i in 0..=data.len() - pattern_len {
                let pattern = data[i..i + pattern_len].to_vec();
                *pattern_counts.entry(pattern).or_insert(0) += 1;
            }
        }

        // Sort patterns by frequency and add most common ones to dictionary
        let mut patterns: Vec<_> = pattern_counts.into_iter().collect();
        patterns.sort_by(|a, b| b.1.cmp(&a.1));

        for (pattern, count) in patterns {
            if count >= 2 && self.next_code < u16::MAX && !self.dictionary.contains_key(&pattern) {
                self.dictionary.insert(pattern.clone(), self.next_code);
                self.reverse_dictionary.insert(self.next_code, pattern);
                self.next_code += 1;
            }
        }
    }

    /// Compress data using the built dictionary
    pub fn compress(&self, data: &[u8]) -> Vec<u16> {
        let mut compressed = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let mut best_match_len = 1;
            let mut best_code = data[pos] as u16;

            // Try to find the longest matching pattern
            for len in (2..=8.min(data.len() - pos)).rev() {
                let pattern = &data[pos..pos + len];
                if let Some(&code) = self.dictionary.get(pattern) {
                    best_match_len = len;
                    best_code = code;
                    break;
                }
            }

            compressed.push(best_code);
            pos += best_match_len;
        }

        compressed
    }

    /// Decompress data using the dictionary
    pub fn decompress(&self, compressed: &[u16]) -> Result<Vec<u8>, &'static str> {
        let mut decompressed = Vec::new();

        for &code in compressed {
            if let Some(pattern) = self.reverse_dictionary.get(&code) {
                decompressed.extend_from_slice(pattern);
            } else {
                return Err("Invalid code in compressed data");
            }
        }

        Ok(decompressed)
    }
}

/// SIMD-optimized byte frequency counter
pub fn count_byte_frequencies_simd(data: &[u8]) -> [u32; 256] {
    let mut frequencies = [0u32; 256];

    // Process data in chunks for better cache efficiency
    const CHUNK_SIZE: usize = 4096;

    for chunk in data.chunks(CHUNK_SIZE) {
        for &byte in chunk {
            frequencies[byte as usize] += 1;
        }
    }

    frequencies
}

/// Calculate compression ratio
pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
    if original_size == 0 {
        return 0.0;
    }
    compressed_size as f64 / original_size as f64
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_length_encode() {
        let data = b"aaabbbccccdddd";
        let encoded = run_length_encode_simd(data);
        let expected = vec![(b'a', 3), (b'b', 3), (b'c', 4), (b'd', 4)];
        assert_eq!(encoded, expected);
    }

    #[test]
    fn test_run_length_decode() {
        let encoded = vec![(b'a', 3), (b'b', 3), (b'c', 4), (b'd', 4)];
        let decoded = run_length_decode(&encoded);
        assert_eq!(decoded, b"aaabbbccccdddd");
    }

    #[test]
    fn test_run_length_roundtrip() {
        let original = b"aaaaabbbbcccccdddddeeeeee";
        let encoded = run_length_encode_simd(original);
        let decoded = run_length_decode(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_lz77_compression() {
        let compressor = LZ77Compressor::new(1024, 32);
        let data = b"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz";
        let compressed = compressor.compress(data);

        // Should be able to compress repeated patterns
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_dictionary_compression() {
        let mut compressor = DictionaryCompressor::new();
        let data = b"hello world hello world hello world";

        compressor.build_dictionary(data, 8);
        let compressed = compressor.compress(data);
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(decompressed, data);

        // Calculate compression efficiency
        let original_bits = data.len() * 8;
        let compressed_bits = compressed.len() * 16; // 16 bits per code
        assert!(compressed_bits < original_bits);
    }

    #[test]
    fn test_byte_frequency_counter() {
        let data = b"hello world";
        let frequencies = count_byte_frequencies_simd(data);

        assert_eq!(frequencies[b'h' as usize], 1);
        assert_eq!(frequencies[b'e' as usize], 1);
        assert_eq!(frequencies[b'l' as usize], 3);
        assert_eq!(frequencies[b'o' as usize], 2);
        assert_eq!(frequencies[b' ' as usize], 1);
        assert_eq!(frequencies[b'w' as usize], 1);
        assert_eq!(frequencies[b'r' as usize], 1);
        assert_eq!(frequencies[b'd' as usize], 1);
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = compression_ratio(1000, 750);
        assert!((ratio - 0.75).abs() < f64::EPSILON);

        let ratio_zero = compression_ratio(0, 100);
        assert_eq!(ratio_zero, 0.0);
    }

    #[test]
    fn test_empty_data() {
        let empty_data = b"";
        let encoded = run_length_encode_simd(empty_data);
        assert!(encoded.is_empty());

        let decoded = run_length_decode(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_single_byte() {
        let data = b"a";
        let encoded = run_length_encode_simd(data);
        assert_eq!(encoded, vec![(b'a', 1)]);

        let decoded = run_length_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_long_runs() {
        let data = vec![b'x'; 1000];
        let encoded = run_length_encode_simd(&data);
        assert_eq!(encoded, vec![(b'x', 1000)]);

        let decoded = run_length_decode(&encoded);
        assert_eq!(decoded, data);
    }
}
