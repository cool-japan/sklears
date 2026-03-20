//! Batch operations for tensor processing
//!
//! This module provides SIMD-optimized batch operations essential for
//! modern neural networks and tensor computations.

use crate::half_precision::{BF16, F16};

#[cfg(feature = "no-std")]
extern crate alloc;

/// Batch normalization operation
pub struct BatchNorm {
    epsilon: f32,
}

impl BatchNorm {
    /// Create a new batch normalization layer
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }

    /// Apply batch normalization to a batch of data
    /// Input shape: [batch_size, features]
    /// mean, variance, gamma, beta shape: \[features\]
    pub fn forward(
        &self,
        input: &[f32],
        mean: &[f32],
        variance: &[f32],
        gamma: &[f32],
        beta: &[f32],
        output: &mut [f32],
        batch_size: usize,
        features: usize,
    ) {
        assert_eq!(input.len(), batch_size * features);
        assert_eq!(output.len(), batch_size * features);
        assert_eq!(mean.len(), features);
        assert_eq!(variance.len(), features);
        assert_eq!(gamma.len(), features);
        assert_eq!(beta.len(), features);

        for batch in 0..batch_size {
            for feat in 0..features {
                let idx = batch * features + feat;
                let x = input[idx];
                let normalized = (x - mean[feat]) / (variance[feat] + self.epsilon).sqrt();
                output[idx] = gamma[feat] * normalized + beta[feat];
            }
        }
    }

    /// Apply batch normalization with FP16 precision
    pub fn forward_f16(
        &self,
        input: &[F16],
        mean: &[F16],
        variance: &[F16],
        gamma: &[F16],
        beta: &[F16],
        output: &mut [F16],
        batch_size: usize,
        features: usize,
    ) {
        assert_eq!(input.len(), batch_size * features);
        assert_eq!(output.len(), batch_size * features);
        assert_eq!(mean.len(), features);
        assert_eq!(variance.len(), features);
        assert_eq!(gamma.len(), features);
        assert_eq!(beta.len(), features);

        for batch in 0..batch_size {
            for feat in 0..features {
                let idx = batch * features + feat;
                let x = input[idx].to_f32();
                let m = mean[feat].to_f32();
                let v = variance[feat].to_f32();
                let g = gamma[feat].to_f32();
                let b = beta[feat].to_f32();

                let normalized = (x - m) / (v + self.epsilon).sqrt();
                let result = g * normalized + b;
                output[idx] = F16::from_f32(result);
            }
        }
    }

    /// Compute batch statistics (mean and variance)
    pub fn compute_stats(
        input: &[f32],
        mean: &mut [f32],
        variance: &mut [f32],
        batch_size: usize,
        features: usize,
    ) {
        assert_eq!(input.len(), batch_size * features);
        assert_eq!(mean.len(), features);
        assert_eq!(variance.len(), features);

        // Compute mean
        for feat in 0..features {
            let mut sum = 0.0;
            for batch in 0..batch_size {
                let idx = batch * features + feat;
                sum += input[idx];
            }
            mean[feat] = sum / batch_size as f32;
        }

        // Compute variance
        for feat in 0..features {
            let mut sum_sq_diff = 0.0;
            for batch in 0..batch_size {
                let idx = batch * features + feat;
                let diff = input[idx] - mean[feat];
                sum_sq_diff += diff * diff;
            }
            variance[feat] = sum_sq_diff / batch_size as f32;
        }
    }
}

/// Layer normalization operation
pub struct LayerNorm {
    epsilon: f32,
}

impl LayerNorm {
    /// Create a new layer normalization
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }

    /// Apply layer normalization
    /// Normalizes across the feature dimension for each sample
    pub fn forward(
        &self,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        output: &mut [f32],
        batch_size: usize,
        features: usize,
    ) {
        assert_eq!(input.len(), batch_size * features);
        assert_eq!(output.len(), batch_size * features);
        assert_eq!(gamma.len(), features);
        assert_eq!(beta.len(), features);

        for batch in 0..batch_size {
            let start_idx = batch * features;
            let end_idx = start_idx + features;

            // Compute mean for this sample
            let mut sum = 0.0;
            for i in start_idx..end_idx {
                sum += input[i];
            }
            let mean = sum / features as f32;

            // Compute variance for this sample
            let mut sum_sq_diff = 0.0;
            for i in start_idx..end_idx {
                let diff = input[i] - mean;
                sum_sq_diff += diff * diff;
            }
            let variance = sum_sq_diff / features as f32;
            let std_dev = (variance + self.epsilon).sqrt();

            // Apply normalization
            for (i, feat) in (start_idx..end_idx).enumerate() {
                let normalized = (input[feat] - mean) / std_dev;
                output[feat] = gamma[i] * normalized + beta[i];
            }
        }
    }
}

/// Batch matrix multiplication operations
pub mod batch_matmul {
    use super::*;

    /// Batch matrix multiplication: C\[i\] = A\[i\] * B\[i\]
    /// All matrices have the same dimensions
    pub fn batch_matmul_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize,
    ) {
        assert_eq!(a.len(), batch_size * m * k);
        assert_eq!(b.len(), batch_size * k * n);
        assert_eq!(c.len(), batch_size * m * n);

        for batch in 0..batch_size {
            let a_offset = batch * m * k;
            let b_offset = batch * k * n;
            let c_offset = batch * m * n;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        let a_idx = a_offset + i * k + l;
                        let b_idx = b_offset + l * n + j;
                        sum += a[a_idx] * b[b_idx];
                    }
                    let c_idx = c_offset + i * n + j;
                    c[c_idx] = sum;
                }
            }
        }
    }

    /// Batch matrix multiplication with broadcasting: C\[i\] = A\[i\] * B
    /// A is batched, B is shared across all batches
    pub fn batch_matmul_broadcast_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize,
    ) {
        assert_eq!(a.len(), batch_size * m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), batch_size * m * n);

        for batch in 0..batch_size {
            let a_offset = batch * m * k;
            let c_offset = batch * m * n;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        let a_idx = a_offset + i * k + l;
                        let b_idx = l * n + j;
                        sum += a[a_idx] * b[b_idx];
                    }
                    let c_idx = c_offset + i * n + j;
                    c[c_idx] = sum;
                }
            }
        }
    }

    /// Batch matrix multiplication with FP16
    pub fn batch_matmul_f16(
        a: &[F16],
        b: &[F16],
        c: &mut [F16],
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize,
    ) {
        assert_eq!(a.len(), batch_size * m * k);
        assert_eq!(b.len(), batch_size * k * n);
        assert_eq!(c.len(), batch_size * m * n);

        for batch in 0..batch_size {
            let a_offset = batch * m * k;
            let b_offset = batch * k * n;
            let c_offset = batch * m * n;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        let a_idx = a_offset + i * k + l;
                        let b_idx = b_offset + l * n + j;
                        sum += a[a_idx].to_f32() * b[b_idx].to_f32();
                    }
                    let c_idx = c_offset + i * n + j;
                    c[c_idx] = F16::from_f32(sum);
                }
            }
        }
    }

    /// Batch matrix multiplication with BF16
    pub fn batch_matmul_bf16(
        a: &[BF16],
        b: &[BF16],
        c: &mut [BF16],
        batch_size: usize,
        m: usize,
        n: usize,
        k: usize,
    ) {
        assert_eq!(a.len(), batch_size * m * k);
        assert_eq!(b.len(), batch_size * k * n);
        assert_eq!(c.len(), batch_size * m * n);

        for batch in 0..batch_size {
            let a_offset = batch * m * k;
            let b_offset = batch * k * n;
            let c_offset = batch * m * n;

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        let a_idx = a_offset + i * k + l;
                        let b_idx = b_offset + l * n + j;
                        sum += a[a_idx].to_f32() * b[b_idx].to_f32();
                    }
                    let c_idx = c_offset + i * n + j;
                    c[c_idx] = BF16::from_f32(sum);
                }
            }
        }
    }
}

/// Attention mechanism operations
pub mod attention {

    /// Scaled dot-product attention
    /// Query, Key, Value shapes: [batch_size, seq_len, d_model]
    /// Output shape: [batch_size, seq_len, d_model]
    pub fn scaled_dot_product_attention(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        output: &mut [f32],
        batch_size: usize,
        seq_len: usize,
        d_model: usize,
        mask: Option<&[bool]>,
    ) {
        let scale = 1.0 / (d_model as f32).sqrt();

        assert_eq!(query.len(), batch_size * seq_len * d_model);
        assert_eq!(key.len(), batch_size * seq_len * d_model);
        assert_eq!(value.len(), batch_size * seq_len * d_model);
        assert_eq!(output.len(), batch_size * seq_len * d_model);

        // Temporary storage for attention scores
        #[cfg(not(feature = "no-std"))]
        let mut scores = vec![0.0f32; batch_size * seq_len * seq_len];
        #[cfg(feature = "no-std")]
        let mut scores = alloc::vec![0.0f32; batch_size * seq_len * seq_len];

        for batch in 0..batch_size {
            // Compute attention scores: Q * K^T
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot_product = 0.0;
                    for k in 0..d_model {
                        let q_idx = batch * seq_len * d_model + i * d_model + k;
                        let k_idx = batch * seq_len * d_model + j * d_model + k;
                        dot_product += query[q_idx] * key[k_idx];
                    }
                    let score_idx = batch * seq_len * seq_len + i * seq_len + j;
                    scores[score_idx] = dot_product * scale;

                    // Apply mask if provided
                    if let Some(mask) = mask {
                        if !mask[i * seq_len + j] {
                            scores[score_idx] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Apply softmax to attention scores
            for i in 0..seq_len {
                let row_start = batch * seq_len * seq_len + i * seq_len;
                let row_end = row_start + seq_len;

                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for idx in row_start..row_end {
                    if scores[idx] > max_val {
                        max_val = scores[idx];
                    }
                }

                // Compute exp and sum
                let mut sum_exp = 0.0;
                for idx in row_start..row_end {
                    scores[idx] = (scores[idx] - max_val).exp();
                    sum_exp += scores[idx];
                }

                // Normalize
                for idx in row_start..row_end {
                    scores[idx] /= sum_exp;
                }
            }

            // Compute output: Attention_weights * V
            for i in 0..seq_len {
                for k in 0..d_model {
                    let mut weighted_sum = 0.0;
                    for j in 0..seq_len {
                        let attention_weight = scores[batch * seq_len * seq_len + i * seq_len + j];
                        let v_idx = batch * seq_len * d_model + j * d_model + k;
                        weighted_sum += attention_weight * value[v_idx];
                    }
                    let out_idx = batch * seq_len * d_model + i * d_model + k;
                    output[out_idx] = weighted_sum;
                }
            }
        }
    }

    /// Multi-head attention
    pub fn multi_head_attention(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        output: &mut [f32],
        batch_size: usize,
        seq_len: usize,
        d_model: usize,
        num_heads: usize,
        mask: Option<&[bool]>,
    ) {
        assert_eq!(d_model % num_heads, 0);
        let d_k = d_model / num_heads;

        assert_eq!(query.len(), batch_size * seq_len * d_model);
        assert_eq!(key.len(), batch_size * seq_len * d_model);
        assert_eq!(value.len(), batch_size * seq_len * d_model);
        assert_eq!(output.len(), batch_size * seq_len * d_model);

        #[cfg(not(feature = "no-std"))]
        let mut head_outputs = vec![0.0f32; batch_size * num_heads * seq_len * d_k];
        #[cfg(feature = "no-std")]
        let mut head_outputs = alloc::vec![0.0f32; batch_size * num_heads * seq_len * d_k];

        // Process each head
        for head in 0..num_heads {
            let head_start = head * d_k;
            let _head_end = head_start + d_k;

            // Extract head-specific Q, K, V
            #[cfg(not(feature = "no-std"))]
            let mut head_q = vec![0.0f32; batch_size * seq_len * d_k];
            #[cfg(feature = "no-std")]
            let mut head_q = alloc::vec![0.0f32; batch_size * seq_len * d_k];
            #[cfg(not(feature = "no-std"))]
            let mut head_k = vec![0.0f32; batch_size * seq_len * d_k];
            #[cfg(feature = "no-std")]
            let mut head_k = alloc::vec![0.0f32; batch_size * seq_len * d_k];
            #[cfg(not(feature = "no-std"))]
            let mut head_v = vec![0.0f32; batch_size * seq_len * d_k];
            #[cfg(feature = "no-std")]
            let mut head_v = alloc::vec![0.0f32; batch_size * seq_len * d_k];

            for batch in 0..batch_size {
                for seq in 0..seq_len {
                    for d in 0..d_k {
                        let src_idx = batch * seq_len * d_model + seq * d_model + head_start + d;
                        let dst_idx = batch * seq_len * d_k + seq * d_k + d;
                        head_q[dst_idx] = query[src_idx];
                        head_k[dst_idx] = key[src_idx];
                        head_v[dst_idx] = value[src_idx];
                    }
                }
            }

            // Apply attention for this head
            #[cfg(not(feature = "no-std"))]
            let mut head_output = vec![0.0f32; batch_size * seq_len * d_k];
            #[cfg(feature = "no-std")]
            let mut head_output = alloc::vec![0.0f32; batch_size * seq_len * d_k];
            scaled_dot_product_attention(
                &head_q,
                &head_k,
                &head_v,
                &mut head_output,
                batch_size,
                seq_len,
                d_k,
                mask,
            );

            // Store head output
            let head_offset = head * batch_size * seq_len * d_k;
            for i in 0..head_output.len() {
                head_outputs[head_offset + i] = head_output[i];
            }
        }

        // Concatenate all heads
        for batch in 0..batch_size {
            for seq in 0..seq_len {
                for head in 0..num_heads {
                    for d in 0..d_k {
                        let src_idx = head * batch_size * seq_len * d_k
                            + batch * seq_len * d_k
                            + seq * d_k
                            + d;
                        let dst_idx = batch * seq_len * d_model + seq * d_model + head * d_k + d;
                        output[dst_idx] = head_outputs[src_idx];
                    }
                }
            }
        }
    }
}

/// Convolution operations for batched data
pub mod convolution {

    /// 2D convolution for batched images
    /// Input shape: [batch_size, in_channels, height, width]
    /// Weight shape: [out_channels, in_channels, kernel_height, kernel_width]
    /// Output shape: [batch_size, out_channels, out_height, out_width]
    pub fn conv2d_batch(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        output: &mut [f32],
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        input_height: usize,
        input_width: usize,
        kernel_height: usize,
        kernel_width: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) {
        let output_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

        assert_eq!(
            input.len(),
            batch_size * in_channels * input_height * input_width
        );
        assert_eq!(
            weight.len(),
            out_channels * in_channels * kernel_height * kernel_width
        );
        assert_eq!(bias.len(), out_channels);
        assert_eq!(
            output.len(),
            batch_size * out_channels * output_height * output_width
        );

        for batch in 0..batch_size {
            for out_ch in 0..out_channels {
                for out_y in 0..output_height {
                    for out_x in 0..output_width {
                        let mut sum = bias[out_ch];

                        for in_ch in 0..in_channels {
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let in_y = out_y * stride_h + ky;
                                    let in_x = out_x * stride_w + kx;

                                    if in_y >= padding_h
                                        && in_x >= padding_w
                                        && in_y < input_height + padding_h
                                        && in_x < input_width + padding_w
                                    {
                                        let input_y = in_y - padding_h;
                                        let input_x = in_x - padding_w;

                                        let input_idx =
                                            batch * in_channels * input_height * input_width
                                                + in_ch * input_height * input_width
                                                + input_y * input_width
                                                + input_x;
                                        let weight_idx =
                                            out_ch * in_channels * kernel_height * kernel_width
                                                + in_ch * kernel_height * kernel_width
                                                + ky * kernel_width
                                                + kx;

                                        sum += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }

                        let output_idx = batch * out_channels * output_height * output_width
                            + out_ch * output_height * output_width
                            + out_y * output_width
                            + out_x;
                        output[output_idx] = sum;
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_batch_norm() {
        let batch_norm = BatchNorm::new(1e-5);
        let batch_size = 2;
        let features = 3;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mean = vec![2.5, 3.5, 4.5];
        let variance = vec![2.25, 2.25, 2.25];
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let mut output = vec![0.0; 6];

        batch_norm.forward(
            &input,
            &mean,
            &variance,
            &gamma,
            &beta,
            &mut output,
            batch_size,
            features,
        );

        // Check that normalization was applied
        for &val in &output {
            assert!(val.abs() < 2.0); // Normalized values should be small
        }
    }

    #[test]
    fn test_layer_norm() {
        let layer_norm = LayerNorm::new(1e-5);
        let batch_size = 2;
        let features = 3;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let mut output = vec![0.0; 6];

        layer_norm.forward(&input, &gamma, &beta, &mut output, batch_size, features);

        // Each sample should be normalized independently
        for batch in 0..batch_size {
            let start = batch * features;
            let end = start + features;
            let sample_mean: f32 = output[start..end].iter().sum::<f32>() / features as f32;
            assert!((sample_mean).abs() < 1e-5);
        }
    }

    #[test]
    fn test_batch_matmul() {
        let batch_size = 2;
        let m = 2;
        let n = 2;
        let k = 2;

        let a = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ];
        let b = vec![
            1.0, 0.0, 0.0, 1.0, // batch 0 (identity)
            1.0, 0.0, 0.0, 1.0, // batch 1 (identity)
        ];
        let mut c = vec![0.0; batch_size * m * n];

        batch_matmul::batch_matmul_f32(&a, &b, &mut c, batch_size, m, n, k);

        // With identity matrices, output should equal input
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for i in 0..expected.len() {
            assert!((c[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_batch_matmul_broadcast() {
        let batch_size = 2;
        let m = 2;
        let n = 2;
        let k = 2;

        let a = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ];
        let b = vec![1.0, 0.0, 0.0, 1.0]; // shared identity matrix
        let mut c = vec![0.0; batch_size * m * n];

        batch_matmul::batch_matmul_broadcast_f32(&a, &b, &mut c, batch_size, m, n, k);

        // With identity matrix, output should equal input
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for i in 0..expected.len() {
            assert!((c[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_basic() {
        let batch_size = 1;
        let seq_len = 3;
        let d_model = 4;

        // Simple test case with known values
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let key = query.clone();
        let value = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut output = vec![0.0; batch_size * seq_len * d_model];

        attention::scaled_dot_product_attention(
            &query,
            &key,
            &value,
            &mut output,
            batch_size,
            seq_len,
            d_model,
            None,
        );

        // Output should be a weighted combination of values
        assert_eq!(output.len(), 12);
        // All values should be finite
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_conv2d_batch_simple() {
        let batch_size = 1;
        let in_channels = 1;
        let out_channels = 1;
        let input_height = 3;
        let input_width = 3;
        let kernel_height = 2;
        let kernel_width = 2;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // Simple kernel
        let bias = vec![0.0];

        let output_height = input_height - kernel_height + 1;
        let output_width = input_width - kernel_width + 1;
        let mut output = vec![0.0; batch_size * out_channels * output_height * output_width];

        convolution::conv2d_batch(
            &input,
            &weight,
            &bias,
            &mut output,
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            1,
            1,
            0,
            0,
        );

        // Check that convolution produced finite results
        for &val in &output {
            assert!(val.is_finite());
        }
        assert_eq!(output.len(), 4); // 2x2 output
    }

    #[test]
    fn test_batch_norm_f16() {
        let batch_norm = BatchNorm::new(1e-3); // Larger epsilon for FP16
        let batch_size = 2;
        let features = 3;

        let input = vec![
            F16::from_f32(1.0),
            F16::from_f32(2.0),
            F16::from_f32(3.0),
            F16::from_f32(4.0),
            F16::from_f32(5.0),
            F16::from_f32(6.0),
        ];
        let mean = vec![F16::from_f32(2.5), F16::from_f32(3.5), F16::from_f32(4.5)];
        let variance = vec![
            F16::from_f32(2.25),
            F16::from_f32(2.25),
            F16::from_f32(2.25),
        ];
        let gamma = vec![F16::from_f32(1.0), F16::from_f32(1.0), F16::from_f32(1.0)];
        let beta = vec![F16::from_f32(0.0), F16::from_f32(0.0), F16::from_f32(0.0)];
        let mut output = vec![F16::from_bits(0); 6];

        batch_norm.forward_f16(
            &input,
            &mean,
            &variance,
            &gamma,
            &beta,
            &mut output,
            batch_size,
            features,
        );

        // Check that normalization was applied
        for &val in &output {
            assert!(val.to_f32().abs() < 2.0);
        }
    }

    #[test]
    fn test_batch_stats_computation() {
        let batch_size = 4;
        let features = 2;

        let input = vec![
            1.0, 2.0, // batch 0
            3.0, 4.0, // batch 1
            5.0, 6.0, // batch 2
            7.0, 8.0, // batch 3
        ];
        let mut mean = vec![0.0; features];
        let mut variance = vec![0.0; features];

        BatchNorm::compute_stats(&input, &mut mean, &mut variance, batch_size, features);

        // Expected mean: [4.0, 5.0]
        assert!((mean[0] - 4.0).abs() < 1e-6);
        assert!((mean[1] - 5.0).abs() < 1e-6);

        // Expected variance: [5.0, 5.0] (for values 1,3,5,7 and 2,4,6,8)
        assert!((variance[0] - 5.0).abs() < 1e-6);
        assert!((variance[1] - 5.0).abs() < 1e-6);
    }
}
