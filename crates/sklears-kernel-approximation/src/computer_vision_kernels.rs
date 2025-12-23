//! Computer vision kernel approximations
//!
//! This module provides kernel approximation methods specifically designed for computer vision tasks,
//! including spatial pyramid features, texture kernels, convolutional features, and scale-invariant methods.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::random::essentials::Normal as RandNormal;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use serde::{Deserialize, Serialize};

use sklears_core::error::Result;
use sklears_core::traits::{Fit, Transform};

/// Spatial pyramid kernel approximation for hierarchical spatial feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
/// SpatialPyramidFeatures
pub struct SpatialPyramidFeatures {
    /// levels
    pub levels: usize,
    /// feature_dim
    pub feature_dim: usize,
    /// pool_method
    pub pool_method: PoolingMethod,
    /// pyramid_weighting
    pub pyramid_weighting: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// PoolingMethod
pub enum PoolingMethod {
    /// Max
    Max,
    /// Average
    Average,
    /// Sum
    Sum,
    /// L2Norm
    L2Norm,
}

impl SpatialPyramidFeatures {
    pub fn new(levels: usize, feature_dim: usize) -> Self {
        Self {
            levels,
            feature_dim,
            pool_method: PoolingMethod::Max,
            pyramid_weighting: true,
        }
    }

    pub fn pool_method(mut self, method: PoolingMethod) -> Self {
        self.pool_method = method;
        self
    }

    pub fn pyramid_weighting(mut self, enable: bool) -> Self {
        self.pyramid_weighting = enable;
        self
    }

    fn spatial_pool(&self, features: &ArrayView2<f64>, grid_size: usize) -> Result<Array1<f64>> {
        let (height, width) = features.dim();
        let cell_h = height / grid_size;
        let cell_w = width / grid_size;

        let mut pooled = Vec::new();

        for i in 0..grid_size {
            for j in 0..grid_size {
                let start_h = i * cell_h;
                let end_h = if i == grid_size - 1 {
                    height
                } else {
                    (i + 1) * cell_h
                };
                let start_w = j * cell_w;
                let end_w = if j == grid_size - 1 {
                    width
                } else {
                    (j + 1) * cell_w
                };

                let cell = features.slice(s![start_h..end_h, start_w..end_w]);

                let pooled_value = match self.pool_method {
                    PoolingMethod::Max => cell.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)),
                    PoolingMethod::Average => cell.mean().unwrap_or(0.0),
                    PoolingMethod::Sum => cell.sum(),
                    PoolingMethod::L2Norm => cell.mapv(|x| x * x).sum().sqrt(),
                };

                pooled.push(pooled_value);
            }
        }

        Ok(Array1::from_vec(pooled))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// FittedSpatialPyramidFeatures
pub struct FittedSpatialPyramidFeatures {
    /// levels
    pub levels: usize,
    /// feature_dim
    pub feature_dim: usize,
    /// pool_method
    pub pool_method: PoolingMethod,
    /// pyramid_weighting
    pub pyramid_weighting: bool,
    /// level_weights
    pub level_weights: Array1<f64>,
}

impl Fit<Array2<f64>, ()> for SpatialPyramidFeatures {
    type Fitted = FittedSpatialPyramidFeatures;

    fn fit(self, _x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let level_weights = if self.pyramid_weighting {
            Array1::from_vec(
                (0..self.levels)
                    .map(|l| 2.0_f64.powi(-(l as i32 + 1)))
                    .collect(),
            )
        } else {
            Array1::ones(self.levels)
        };

        Ok(FittedSpatialPyramidFeatures {
            levels: self.levels,
            feature_dim: self.feature_dim,
            pool_method: self.pool_method.clone(),
            pyramid_weighting: self.pyramid_weighting,
            level_weights,
        })
    }
}

impl FittedSpatialPyramidFeatures {
    fn spatial_pool(&self, features: &ArrayView2<f64>, grid_size: usize) -> Result<Array1<f64>> {
        let (height, width) = features.dim();
        let cell_h = height / grid_size;
        let cell_w = width / grid_size;

        let mut pooled = Vec::new();

        for i in 0..grid_size {
            for j in 0..grid_size {
                let start_h = i * cell_h;
                let end_h = if i == grid_size - 1 {
                    height
                } else {
                    (i + 1) * cell_h
                };
                let start_w = j * cell_w;
                let end_w = if j == grid_size - 1 {
                    width
                } else {
                    (j + 1) * cell_w
                };

                let cell = features.slice(s![start_h..end_h, start_w..end_w]);

                let pooled_value = match self.pool_method {
                    PoolingMethod::Max => cell.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)),
                    PoolingMethod::Average => cell.mean().unwrap_or(0.0),
                    PoolingMethod::Sum => cell.sum(),
                    PoolingMethod::L2Norm => cell.mapv(|x| x * x).sum().sqrt(),
                };

                pooled.push(pooled_value);
            }
        }

        Ok(Array1::from_vec(pooled))
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedSpatialPyramidFeatures {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.shape()[0];
        let feature_size = x.shape()[1];

        // Assume square images for spatial pyramid
        let img_size = (feature_size as f64).sqrt() as usize;

        let mut all_features = Vec::new();

        for i in 0..n_samples {
            let sample = x.row(i).to_owned().into_shape((img_size, img_size))?;
            let mut pyramid_features = Vec::new();

            for level in 0..self.levels {
                let grid_size = 2_usize.pow(level as u32);
                let pooled = self.spatial_pool(&sample.view(), grid_size)?;

                // Apply pyramid weighting
                let weighted = &pooled * self.level_weights[level];
                pyramid_features.extend(weighted.iter().cloned());
            }

            all_features.push(pyramid_features);
        }

        // Convert to ndarray
        let n_features = all_features[0].len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (i, features) in all_features.iter().enumerate() {
            for (j, &feature) in features.iter().enumerate() {
                result[[i, j]] = feature;
            }
        }

        Ok(result)
    }
}

/// Texture kernel approximation using Local Binary Patterns and Gabor filters
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TextureKernelApproximation
pub struct TextureKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// use_lbp
    pub use_lbp: bool,
    /// use_gabor
    pub use_gabor: bool,
    /// gabor_frequencies
    pub gabor_frequencies: Vec<f64>,
    /// gabor_angles
    pub gabor_angles: Vec<f64>,
    /// lbp_radius
    pub lbp_radius: f64,
    /// lbp_n_points
    pub lbp_n_points: usize,
}

impl TextureKernelApproximation {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            use_lbp: true,
            use_gabor: true,
            gabor_frequencies: vec![0.1, 0.2, 0.3, 0.4],
            gabor_angles: vec![
                0.0,
                std::f64::consts::PI / 4.0,
                std::f64::consts::PI / 2.0,
                3.0 * std::f64::consts::PI / 4.0,
            ],
            lbp_radius: 1.0,
            lbp_n_points: 8,
        }
    }

    pub fn use_lbp(mut self, enable: bool) -> Self {
        self.use_lbp = enable;
        self
    }

    pub fn use_gabor(mut self, enable: bool) -> Self {
        self.use_gabor = enable;
        self
    }

    pub fn gabor_frequencies(mut self, frequencies: Vec<f64>) -> Self {
        self.gabor_frequencies = frequencies;
        self
    }

    pub fn gabor_angles(mut self, angles: Vec<f64>) -> Self {
        self.gabor_angles = angles;
        self
    }

    fn compute_lbp(&self, image: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let (height, width) = image.dim();
        let mut lbp_histogram = Array1::zeros(256);

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let center = image[[i, j]];
                let mut lbp_value = 0u8;

                // 8-neighborhood LBP
                let neighbors = [
                    image[[i - 1, j - 1]],
                    image[[i - 1, j]],
                    image[[i - 1, j + 1]],
                    image[[i, j + 1]],
                    image[[i + 1, j + 1]],
                    image[[i + 1, j]],
                    image[[i + 1, j - 1]],
                    image[[i, j - 1]],
                ];

                for (k, &neighbor) in neighbors.iter().enumerate() {
                    if neighbor >= center {
                        lbp_value |= 1 << k;
                    }
                }

                lbp_histogram[lbp_value as usize] += 1.0;
            }
        }

        // Normalize histogram
        let sum = lbp_histogram.sum();
        if sum > 0.0 {
            lbp_histogram /= sum;
        }

        Ok(lbp_histogram)
    }

    fn compute_gabor_features(&self, image: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let mut gabor_features = Vec::new();

        for &frequency in &self.gabor_frequencies {
            for &angle in &self.gabor_angles {
                let filtered = self.apply_gabor_filter(image, frequency, angle)?;

                // Extract statistical features from filtered image
                let mean = filtered.mean().unwrap_or(0.0);
                let variance = filtered.var(0.0);
                let energy = filtered.mapv(|x| x * x).sum();

                gabor_features.extend_from_slice(&[mean, variance, energy]);
            }
        }

        Ok(Array1::from_vec(gabor_features))
    }

    fn apply_gabor_filter(
        &self,
        image: &ArrayView2<f64>,
        frequency: f64,
        angle: f64,
    ) -> Result<Array2<f64>> {
        let (height, width) = image.dim();
        let mut filtered = Array2::zeros((height, width));

        let sigma_x = 1.0 / (2.0 * std::f64::consts::PI * frequency);
        let sigma_y = 1.0 / (2.0 * std::f64::consts::PI * frequency);

        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        for i in 0..height {
            for j in 0..width {
                let x = j as f64 - width as f64 / 2.0;
                let y = i as f64 - height as f64 / 2.0;

                let x_rot = x * cos_angle + y * sin_angle;
                let y_rot = -x * sin_angle + y * cos_angle;

                let gaussian = (-0.5
                    * (x_rot * x_rot / (sigma_x * sigma_x) + y_rot * y_rot / (sigma_y * sigma_y)))
                    .exp();
                let sinusoid = (2.0 * std::f64::consts::PI * frequency * x_rot).cos();

                let kernel_value = gaussian * sinusoid;

                if i < height && j < width {
                    filtered[[i, j]] = kernel_value;
                }
            }
        }

        // Convolve with image (simplified version)
        let mut result = Array2::zeros((height, width));
        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let mut sum = 0.0;
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;
                        let fi = (1 + di) as usize;
                        let fj = (1 + dj) as usize;
                        sum += image[[ni, nj]] * filtered[[fi, fj]];
                    }
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// FittedTextureKernelApproximation
pub struct FittedTextureKernelApproximation {
    /// n_components
    pub n_components: usize,
    /// use_lbp
    pub use_lbp: bool,
    /// use_gabor
    pub use_gabor: bool,
    /// gabor_frequencies
    pub gabor_frequencies: Vec<f64>,
    /// gabor_angles
    pub gabor_angles: Vec<f64>,
    /// lbp_radius
    pub lbp_radius: f64,
    /// lbp_n_points
    pub lbp_n_points: usize,
    /// feature_dim
    pub feature_dim: usize,
}

impl Fit<Array2<f64>, ()> for TextureKernelApproximation {
    type Fitted = FittedTextureKernelApproximation;

    fn fit(self, _x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let mut feature_dim = 0;

        if self.use_lbp {
            feature_dim += 256; // LBP histogram
        }

        if self.use_gabor {
            feature_dim += self.gabor_frequencies.len() * self.gabor_angles.len() * 3;
            // 3 stats per filter
        }

        Ok(FittedTextureKernelApproximation {
            n_components: self.n_components,
            use_lbp: self.use_lbp,
            use_gabor: self.use_gabor,
            gabor_frequencies: self.gabor_frequencies.clone(),
            gabor_angles: self.gabor_angles.clone(),
            lbp_radius: self.lbp_radius,
            lbp_n_points: self.lbp_n_points,
            feature_dim,
        })
    }
}

impl FittedTextureKernelApproximation {
    fn compute_lbp(&self, image: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let (height, width) = image.dim();
        let mut lbp_histogram = Array1::zeros(256);

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let center = image[[i, j]];
                let mut lbp_value = 0u8;

                // 8-neighborhood LBP
                let neighbors = [
                    image[[i - 1, j - 1]],
                    image[[i - 1, j]],
                    image[[i - 1, j + 1]],
                    image[[i, j + 1]],
                    image[[i + 1, j + 1]],
                    image[[i + 1, j]],
                    image[[i + 1, j - 1]],
                    image[[i, j - 1]],
                ];

                for (k, &neighbor) in neighbors.iter().enumerate() {
                    if neighbor >= center {
                        lbp_value |= 1 << k;
                    }
                }

                lbp_histogram[lbp_value as usize] += 1.0;
            }
        }

        // Normalize histogram
        let sum = lbp_histogram.sum();
        if sum > 0.0 {
            lbp_histogram /= sum;
        }

        Ok(lbp_histogram)
    }

    fn compute_gabor_features(&self, image: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let mut gabor_features = Vec::new();

        for &frequency in &self.gabor_frequencies {
            for &angle in &self.gabor_angles {
                let filtered = self.apply_gabor_filter(image, frequency, angle)?;

                // Extract statistical features from filtered image
                let mean = filtered.mean().unwrap_or(0.0);
                let variance = filtered.var(0.0);
                let energy = filtered.mapv(|x| x * x).sum();

                gabor_features.extend_from_slice(&[mean, variance, energy]);
            }
        }

        Ok(Array1::from_vec(gabor_features))
    }

    fn apply_gabor_filter(
        &self,
        image: &ArrayView2<f64>,
        frequency: f64,
        angle: f64,
    ) -> Result<Array2<f64>> {
        let (height, width) = image.dim();
        let mut filtered = Array2::zeros((height, width));

        let sigma_x = 1.0 / (2.0 * std::f64::consts::PI * frequency);
        let sigma_y = 1.0 / (2.0 * std::f64::consts::PI * frequency);

        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        for i in 0..height {
            for j in 0..width {
                let x = j as f64 - width as f64 / 2.0;
                let y = i as f64 - height as f64 / 2.0;

                let x_rot = x * cos_angle + y * sin_angle;
                let y_rot = -x * sin_angle + y * cos_angle;

                let gaussian = (-0.5
                    * (x_rot * x_rot / (sigma_x * sigma_x) + y_rot * y_rot / (sigma_y * sigma_y)))
                    .exp();
                let sinusoid = (2.0 * std::f64::consts::PI * frequency * x_rot).cos();

                let kernel_value = gaussian * sinusoid;

                if i < height && j < width {
                    filtered[[i, j]] = kernel_value;
                }
            }
        }

        // Convolve with image (simplified version)
        let mut result = Array2::zeros((height, width));
        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let mut sum = 0.0;
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;
                        let fi = (1 + di) as usize;
                        let fj = (1 + dj) as usize;
                        sum += image[[ni, nj]] * filtered[[fi, fj]];
                    }
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedTextureKernelApproximation {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.shape()[0];
        let feature_size = x.shape()[1];

        // Assume square images
        let img_size = (feature_size as f64).sqrt() as usize;

        let mut all_features = Vec::new();

        for i in 0..n_samples {
            let sample = x.row(i).to_owned().into_shape((img_size, img_size))?;
            let mut texture_features = Vec::new();

            if self.use_lbp {
                let lbp_features = self.compute_lbp(&sample.view())?;
                texture_features.extend(lbp_features.iter().cloned());
            }

            if self.use_gabor {
                let gabor_features = self.compute_gabor_features(&sample.view())?;
                texture_features.extend(gabor_features.iter().cloned());
            }

            all_features.push(texture_features);
        }

        // Convert to ndarray
        let n_features = all_features[0].len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (i, features) in all_features.iter().enumerate() {
            for (j, &feature) in features.iter().enumerate() {
                result[[i, j]] = feature;
            }
        }

        Ok(result)
    }
}

/// Scale-invariant feature transform (SIFT-like) kernel approximation
#[derive(Debug, Clone, Serialize, Deserialize)]
/// ScaleInvariantFeatures
pub struct ScaleInvariantFeatures {
    /// n_components
    pub n_components: usize,
    /// n_scales
    pub n_scales: usize,
    /// sigma
    pub sigma: f64,
    /// contrast_threshold
    pub contrast_threshold: f64,
    /// edge_threshold
    pub edge_threshold: f64,
}

impl ScaleInvariantFeatures {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_scales: 3,
            sigma: 1.6,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
        }
    }

    pub fn n_scales(mut self, n_scales: usize) -> Self {
        self.n_scales = n_scales;
        self
    }

    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    fn detect_keypoints(&self, image: &ArrayView2<f64>) -> Result<Vec<(usize, usize, f64)>> {
        let (height, width) = image.dim();
        let mut keypoints = Vec::new();

        // Simplified keypoint detection using Harris corner detector
        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let _ix = (image[[i, j + 1]] - image[[i, j - 1]]) / 2.0;
                let _iy = (image[[i + 1, j]] - image[[i - 1, j]]) / 2.0;
                let ixx = image[[i, j - 1]] - 2.0 * image[[i, j]] + image[[i, j + 1]];
                let iyy = image[[i - 1, j]] - 2.0 * image[[i, j]] + image[[i + 1, j]];
                let ixy = (image[[i - 1, j - 1]] + image[[i + 1, j + 1]]
                    - image[[i - 1, j + 1]]
                    - image[[i + 1, j - 1]])
                    / 4.0;

                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;

                if trace != 0.0 {
                    let harris_response = det - 0.04 * trace * trace;

                    if harris_response > self.contrast_threshold {
                        keypoints.push((i, j, harris_response));
                    }
                }
            }
        }

        Ok(keypoints)
    }

    fn compute_descriptor(
        &self,
        image: &ArrayView2<f64>,
        keypoint: (usize, usize, f64),
    ) -> Result<Array1<f64>> {
        let (y, x, _) = keypoint;
        let (height, width) = image.dim();

        // Simplified descriptor computation
        let window_size = 16;
        let half_window = window_size / 2;

        let mut descriptor = Vec::new();

        for i in 0..4 {
            for j in 0..4 {
                let start_y = y.saturating_sub(half_window) + i * 4;
                let start_x = x.saturating_sub(half_window) + j * 4;
                let end_y = (start_y + 4).min(height);
                let end_x = (start_x + 4).min(width);

                if start_y < end_y && start_x < end_x {
                    let patch = image.slice(s![start_y..end_y, start_x..end_x]);
                    let patch_mean = patch.mean().unwrap_or(0.0);
                    let patch_var = patch.var(0.0);

                    descriptor.push(patch_mean);
                    descriptor.push(patch_var);
                } else {
                    descriptor.push(0.0);
                    descriptor.push(0.0);
                }
            }
        }

        Ok(Array1::from_vec(descriptor))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// FittedScaleInvariantFeatures
pub struct FittedScaleInvariantFeatures {
    /// n_components
    pub n_components: usize,
    /// n_scales
    pub n_scales: usize,
    /// sigma
    pub sigma: f64,
    /// contrast_threshold
    pub contrast_threshold: f64,
    /// edge_threshold
    pub edge_threshold: f64,
}

impl Fit<Array2<f64>, ()> for ScaleInvariantFeatures {
    type Fitted = FittedScaleInvariantFeatures;

    fn fit(self, _x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        Ok(FittedScaleInvariantFeatures {
            n_components: self.n_components,
            n_scales: self.n_scales,
            sigma: self.sigma,
            contrast_threshold: self.contrast_threshold,
            edge_threshold: self.edge_threshold,
        })
    }
}

impl FittedScaleInvariantFeatures {
    fn detect_keypoints(&self, image: &ArrayView2<f64>) -> Result<Vec<(usize, usize, f64)>> {
        let (height, width) = image.dim();
        let mut keypoints = Vec::new();

        // Simplified keypoint detection using Harris corner detector
        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let _ix = (image[[i, j + 1]] - image[[i, j - 1]]) / 2.0;
                let _iy = (image[[i + 1, j]] - image[[i - 1, j]]) / 2.0;
                let ixx = image[[i, j - 1]] - 2.0 * image[[i, j]] + image[[i, j + 1]];
                let iyy = image[[i - 1, j]] - 2.0 * image[[i, j]] + image[[i + 1, j]];
                let ixy = (image[[i - 1, j - 1]] + image[[i + 1, j + 1]]
                    - image[[i - 1, j + 1]]
                    - image[[i + 1, j - 1]])
                    / 4.0;

                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;

                if trace != 0.0 {
                    let harris_response = det - 0.04 * trace * trace;

                    if harris_response > self.contrast_threshold {
                        keypoints.push((i, j, harris_response));
                    }
                }
            }
        }

        Ok(keypoints)
    }

    fn compute_descriptor(
        &self,
        image: &ArrayView2<f64>,
        keypoint: (usize, usize, f64),
    ) -> Result<Array1<f64>> {
        let (y, x, _) = keypoint;
        let (height, width) = image.dim();

        // Simplified descriptor computation
        let window_size = 16;
        let half_window = window_size / 2;

        let mut descriptor = Vec::new();

        for i in 0..4 {
            for j in 0..4 {
                let start_y = y.saturating_sub(half_window) + i * 4;
                let start_x = x.saturating_sub(half_window) + j * 4;
                let end_y = (start_y + 4).min(height);
                let end_x = (start_x + 4).min(width);

                if start_y < end_y && start_x < end_x {
                    let patch = image.slice(s![start_y..end_y, start_x..end_x]);
                    let patch_mean = patch.mean().unwrap_or(0.0);
                    let patch_var = patch.var(0.0);

                    descriptor.push(patch_mean);
                    descriptor.push(patch_var);
                } else {
                    descriptor.push(0.0);
                    descriptor.push(0.0);
                }
            }
        }

        Ok(Array1::from_vec(descriptor))
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedScaleInvariantFeatures {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.shape()[0];
        let feature_size = x.shape()[1];

        // Assume square images
        let img_size = (feature_size as f64).sqrt() as usize;

        let mut all_features = Vec::new();

        for i in 0..n_samples {
            let sample = x.row(i).to_owned().into_shape((img_size, img_size))?;

            let keypoints = self.detect_keypoints(&sample.view())?;
            let mut descriptors = Vec::new();

            for keypoint in keypoints.iter().take(self.n_components) {
                let descriptor = self.compute_descriptor(&sample.view(), *keypoint)?;
                descriptors.extend(descriptor.iter().cloned());
            }

            // Pad with zeros if not enough keypoints
            let expected_size = self.n_components * 32; // 32 features per keypoint
            while descriptors.len() < expected_size {
                descriptors.push(0.0);
            }
            descriptors.truncate(expected_size);

            all_features.push(descriptors);
        }

        // Convert to ndarray
        let n_features = all_features[0].len();
        let mut result = Array2::zeros((n_samples, n_features));

        for (i, features) in all_features.iter().enumerate() {
            for (j, &feature) in features.iter().enumerate() {
                result[[i, j]] = feature;
            }
        }

        Ok(result)
    }
}

/// Convolutional kernel features using random convolutions
#[derive(Debug, Clone, Serialize, Deserialize)]
/// ConvolutionalKernelFeatures
pub struct ConvolutionalKernelFeatures {
    /// n_components
    pub n_components: usize,
    /// kernel_size
    pub kernel_size: usize,
    /// stride
    pub stride: usize,
    /// padding
    pub padding: usize,
    /// activation
    pub activation: ActivationFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// ActivationFunction
pub enum ActivationFunction {
    /// ReLU
    ReLU,
    /// Tanh
    Tanh,
    /// Sigmoid
    Sigmoid,
    /// Linear
    Linear,
}

impl ConvolutionalKernelFeatures {
    pub fn new(n_components: usize, kernel_size: usize) -> Self {
        Self {
            n_components,
            kernel_size,
            stride: 1,
            padding: 0,
            activation: ActivationFunction::ReLU,
        }
    }

    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        self.activation = activation;
        self
    }

    fn apply_activation(&self, x: f64) -> f64 {
        match self.activation {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Linear => x,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// FittedConvolutionalKernelFeatures
pub struct FittedConvolutionalKernelFeatures {
    /// n_components
    pub n_components: usize,
    /// kernel_size
    pub kernel_size: usize,
    /// stride
    pub stride: usize,
    /// padding
    pub padding: usize,
    /// activation
    pub activation: ActivationFunction,
    /// conv_kernels
    pub conv_kernels: Array2<f64>,
    /// biases
    pub biases: Array1<f64>,
}

impl Fit<Array2<f64>, ()> for ConvolutionalKernelFeatures {
    type Fitted = FittedConvolutionalKernelFeatures;

    fn fit(self, _x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let mut rng = RealStdRng::from_seed(thread_rng().gen());
        let normal = RandNormal::new(0.0, 1.0).unwrap();

        // Generate random convolution kernels
        let kernel_elements = self.kernel_size * self.kernel_size;
        let mut conv_kernels = Array2::zeros((self.n_components, kernel_elements));

        for i in 0..self.n_components {
            for j in 0..kernel_elements {
                conv_kernels[[i, j]] = rng.sample(normal);
            }
        }

        // Initialize biases
        let biases = Array1::from_vec((0..self.n_components).map(|_| rng.sample(normal)).collect());

        Ok(FittedConvolutionalKernelFeatures {
            n_components: self.n_components,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            activation: self.activation,
            conv_kernels,
            biases,
        })
    }
}

impl FittedConvolutionalKernelFeatures {
    fn apply_activation(&self, x: f64) -> f64 {
        match self.activation {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Linear => x,
        }
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedConvolutionalKernelFeatures {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.shape()[0];
        let feature_size = x.shape()[1];

        // Assume square images
        let img_size = (feature_size as f64).sqrt() as usize;

        let output_size = (img_size + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let n_features = self.n_components * output_size * output_size;

        let mut result = Array2::zeros((n_samples, n_features));

        for sample_idx in 0..n_samples {
            let image = x
                .row(sample_idx)
                .to_owned()
                .into_shape((img_size, img_size))?;
            let mut feature_idx = 0;

            for kernel_idx in 0..self.n_components {
                let kernel = self
                    .conv_kernels
                    .row(kernel_idx)
                    .to_owned()
                    .into_shape((self.kernel_size, self.kernel_size))?;

                for i in 0..output_size {
                    for j in 0..output_size {
                        let start_i = i * self.stride;
                        let start_j = j * self.stride;

                        let mut conv_sum = 0.0;
                        for ki in 0..self.kernel_size {
                            for kj in 0..self.kernel_size {
                                let img_i = start_i + ki;
                                let img_j = start_j + kj;

                                if img_i < img_size && img_j < img_size {
                                    conv_sum += image[[img_i, img_j]] * kernel[[ki, kj]];
                                }
                            }
                        }

                        conv_sum += self.biases[kernel_idx];
                        let activated = self.apply_activation(conv_sum);

                        result[[sample_idx, feature_idx]] = activated;
                        feature_idx += 1;
                    }
                }
            }
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::essentials::Normal;

    use scirs2_core::ndarray::{Array, Array2};
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_spatial_pyramid_features() {
        let x: Array2<f64> = Array::from_shape_fn((10, 64), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let pyramid = SpatialPyramidFeatures::new(3, 64);

        let fitted = pyramid.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape()[0], 10);
        assert!(transformed.shape()[1] > 0);
    }

    #[test]
    fn test_texture_kernel_approximation() {
        let x: Array2<f64> = Array::from_shape_fn((5, 64), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let texture = TextureKernelApproximation::new(50);

        let fitted = texture.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape()[0], 5);
        assert!(transformed.shape()[1] > 0);
    }

    #[test]
    fn test_scale_invariant_features() {
        let x: Array2<f64> = Array::from_shape_fn((8, 64), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let sift = ScaleInvariantFeatures::new(10);

        let fitted = sift.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape()[0], 8);
        assert_eq!(transformed.shape()[1], 10 * 32);
    }

    #[test]
    fn test_convolutional_kernel_features() {
        let x: Array2<f64> = Array::from_shape_fn((6, 64), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let conv = ConvolutionalKernelFeatures::new(16, 3);

        let fitted = conv.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape()[0], 6);
        assert!(transformed.shape()[1] > 0);
    }
}
