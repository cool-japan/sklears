use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::Distribution;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

/// Image Patch Embedding for texture analysis and segmentation
#[derive(Debug, Clone)]
pub struct ImagePatchEmbedding<S = Untrained> {
    patch_size: (usize, usize),
    stride: (usize, usize),
    n_components: usize,
    embedding_method: PatchEmbeddingMethod,
    state: S,
}

#[derive(Debug, Clone)]
pub enum PatchEmbeddingMethod {
    /// PCA
    PCA,
    /// TSNE
    TSNE,
    /// UMAP
    UMAP,
    /// Isomap
    Isomap,
}

#[derive(Debug, Clone)]
pub struct TrainedPatchEmbedding {
    patch_size: (usize, usize),
    stride: (usize, usize),
    n_components: usize,
    embedding_method: PatchEmbeddingMethod,
    embedding_weights: Array2<f64>, // Learned embedding transformation
    patch_means: Array1<f64>,       // Mean patch for normalization
}

impl ImagePatchEmbedding<Untrained> {
    pub fn new(patch_size: (usize, usize)) -> Self {
        Self {
            patch_size,
            stride: (1, 1),
            n_components: 50,
            embedding_method: PatchEmbeddingMethod::PCA,
            state: Untrained,
        }
    }

    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_embedding_method(mut self, method: PatchEmbeddingMethod) -> Self {
        self.embedding_method = method;
        self
    }

    fn extract_patches(&self, image: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (height, width) = image.dim();
        let (patch_h, patch_w) = self.patch_size;
        let (stride_h, stride_w) = self.stride;

        if patch_h > height || patch_w > width {
            return Err(SklearsError::InvalidInput(
                "Patch size cannot be larger than image".to_string(),
            ));
        }

        let n_patches_h = (height - patch_h) / stride_h + 1;
        let n_patches_w = (width - patch_w) / stride_w + 1;
        let patch_dim = patch_h * patch_w;

        let mut patches = Array2::zeros((n_patches_h * n_patches_w, patch_dim));

        for i in 0..n_patches_h {
            for j in 0..n_patches_w {
                let start_h = i * stride_h;
                let start_w = j * stride_w;

                let patch = image.slice(scirs2_core::ndarray::s![
                    start_h..start_h + patch_h,
                    start_w..start_w + patch_w
                ]);

                let patch_idx = i * n_patches_w + j;
                for (flat_idx, &pixel) in patch.iter().enumerate() {
                    patches[[patch_idx, flat_idx]] = pixel;
                }
            }
        }

        Ok(patches)
    }

    fn compute_patch_embedding(
        &self,
        patches: &ArrayView2<f64>,
    ) -> SklResult<TrainedPatchEmbedding> {
        let (n_patches, patch_dim) = patches.dim();

        // Compute patch means for normalization
        let patch_means = patches.mean_axis(Axis(0)).unwrap();

        // Center patches
        let centered_patches = patches - &patch_means;

        // Compute embedding weights based on method
        let embedding_weights = match self.embedding_method {
            PatchEmbeddingMethod::PCA => self.compute_pca_embedding(&centered_patches.view())?,
            PatchEmbeddingMethod::TSNE => {
                // For t-SNE, we use a random projection as placeholder
                // In practice, would use full t-SNE implementation
                self.compute_random_projection(&centered_patches.view())?
            }
            PatchEmbeddingMethod::UMAP => {
                // For UMAP, we use a random projection as placeholder
                // In practice, would use full UMAP implementation
                self.compute_random_projection(&centered_patches.view())?
            }
            PatchEmbeddingMethod::Isomap => {
                // For Isomap, we use a random projection as placeholder
                // In practice, would use full Isomap implementation
                self.compute_random_projection(&centered_patches.view())?
            }
        };

        Ok(TrainedPatchEmbedding {
            patch_size: self.patch_size,
            stride: self.stride,
            n_components: self.n_components,
            embedding_method: self.embedding_method.clone(),
            embedding_weights,
            patch_means,
        })
    }

    fn compute_pca_embedding(&self, patches: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = patches.dim();

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Not enough patches for requested components".to_string(),
            ));
        }

        // Compute covariance matrix
        let cov = patches.t().dot(patches) / (n_samples - 1) as f64;

        // Compute eigendecomposition
        let (eigenvals, eigenvecs) = cov
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::from(format!("Eigendecomposition failed: {}", e)))?;

        // Sort eigenvalues in descending order and take top components
        let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvals
            .iter()
            .zip(eigenvecs.columns())
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take the top n_components eigenvectors
        let mut projection_matrix = Array2::zeros((n_features, self.n_components));
        for (i, (_, eigenvec)) in eigen_pairs.iter().take(self.n_components).enumerate() {
            projection_matrix.column_mut(i).assign(eigenvec);
        }

        Ok(projection_matrix)
    }

    fn compute_random_projection(&self, patches: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (_n_samples, n_features) = patches.dim();
        let mut rng = thread_rng();

        // Generate random projection matrix
        let normal = Normal::new(0.0, 1.0 / (n_features as f64).sqrt()).unwrap();
        let projection_matrix = Array2::from_shape_fn((n_features, self.n_components), |(_, _)| {
            normal.sample(&mut rng)
        });

        Ok(projection_matrix)
    }
}

impl ImagePatchEmbedding<TrainedPatchEmbedding> {
    pub fn transform_image(&self, image: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        // Extract patches from image
        let patches = ImagePatchEmbedding::<Untrained>::new(self.state.patch_size)
            .with_stride(self.state.stride)
            .extract_patches(image)?;

        // Center patches using learned means
        let centered_patches = &patches - &self.state.patch_means;

        // Apply embedding transformation
        let embedded_patches = centered_patches.dot(&self.state.embedding_weights);

        Ok(embedded_patches)
    }

    pub fn reconstruct_image(
        &self,
        embedded_patches: &ArrayView2<f64>,
        image_shape: (usize, usize),
    ) -> SklResult<Array2<f64>> {
        // Reconstruct patches from embedding
        let reconstructed_patches =
            embedded_patches.dot(&self.state.embedding_weights.t()) + &self.state.patch_means;

        // Reconstruct image from patches
        let (height, width) = image_shape;
        let (patch_h, patch_w) = self.state.patch_size;
        let (stride_h, stride_w) = self.state.stride;

        let n_patches_h = (height - patch_h) / stride_h + 1;
        let n_patches_w = (width - patch_w) / stride_w + 1;

        let mut reconstructed_image = Array2::zeros(image_shape);
        let mut count_matrix = Array2::zeros(image_shape);

        for i in 0..n_patches_h {
            for j in 0..n_patches_w {
                let start_h = i * stride_h;
                let start_w = j * stride_w;
                let patch_idx = i * n_patches_w + j;

                // Reshape patch back to 2D
                let patch_flat = reconstructed_patches.row(patch_idx);
                for (flat_idx, &pixel) in patch_flat.iter().enumerate() {
                    let patch_row = flat_idx / patch_w;
                    let patch_col = flat_idx % patch_w;
                    let img_row = start_h + patch_row;
                    let img_col = start_w + patch_col;

                    reconstructed_image[[img_row, img_col]] += pixel;
                    count_matrix[[img_row, img_col]] += 1.0;
                }
            }
        }

        // Average overlapping regions
        for ((i, j), count) in count_matrix.indexed_iter() {
            if *count > 0.0 {
                reconstructed_image[[i, j]] /= count;
            }
        }

        Ok(reconstructed_image)
    }
}

impl Estimator for ImagePatchEmbedding<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for ImagePatchEmbedding<Untrained> {
    type Fitted = ImagePatchEmbedding<TrainedPatchEmbedding>;

    fn fit(self, image: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (height, width) = image.dim();

        if height == 0 || width == 0 {
            return Err(SklearsError::InvalidInput("Empty image".to_string()));
        }

        // Extract patches from the image
        let patches = self.extract_patches(&image.view())?;

        // Compute embedding
        let trained_state = self.compute_patch_embedding(&patches.view())?;

        Ok(ImagePatchEmbedding {
            patch_size: self.patch_size,
            stride: self.stride,
            n_components: self.n_components,
            embedding_method: self.embedding_method,
            state: trained_state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for ImagePatchEmbedding<TrainedPatchEmbedding> {
    fn transform(&self, image: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.transform_image(&image.view())
    }
}

/// Face Manifold Learning for face recognition and expression analysis
#[derive(Debug, Clone)]
pub struct FaceManifoldLearning<S = Untrained> {
    image_size: (usize, usize),
    n_components: usize,
    preprocessing: FacePreprocessing,
    state: S,
}

#[derive(Debug, Clone)]
pub enum FacePreprocessing {
    /// Raw
    Raw,
    /// Histogram
    Histogram,
    /// GaussianBlur
    GaussianBlur { sigma: f64 },
    /// LocalBinaryPattern
    LocalBinaryPattern,
}

#[derive(Debug, Clone)]
pub struct TrainedFaceManifold {
    image_size: (usize, usize),
    n_components: usize,
    preprocessing: FacePreprocessing,
    face_embedding: Array2<f64>,
    mean_face: Array1<f64>,
}

impl FaceManifoldLearning<Untrained> {
    pub fn new(image_size: (usize, usize)) -> Self {
        Self {
            image_size,
            n_components: 50,
            preprocessing: FacePreprocessing::Raw,
            state: Untrained,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_preprocessing(mut self, preprocessing: FacePreprocessing) -> Self {
        self.preprocessing = preprocessing;
        self
    }

    fn preprocess_face(&self, face: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        match &self.preprocessing {
            FacePreprocessing::Raw => Ok(face.to_owned()),
            FacePreprocessing::Histogram => self.apply_histogram_equalization(face),
            FacePreprocessing::GaussianBlur { sigma } => self.apply_gaussian_blur(face, *sigma),
            FacePreprocessing::LocalBinaryPattern => self.compute_lbp_features(face),
        }
    }

    fn apply_histogram_equalization(&self, face: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        // Simple histogram equalization approximation
        let min_val = face.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = face.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if max_val == min_val {
            return Ok(face.to_owned());
        }

        let normalized = face.mapv(|x| (x - min_val) / (max_val - min_val));
        Ok(normalized)
    }

    fn apply_gaussian_blur(&self, face: &ArrayView2<f64>, sigma: f64) -> SklResult<Array2<f64>> {
        // Simplified Gaussian blur (just return smoothed version)
        let (height, width) = face.dim();
        let mut blurred = face.to_owned();

        // Simple box filter approximation
        let kernel_size = (6.0 * sigma) as usize;
        if kernel_size > 0 && kernel_size < height && kernel_size < width {
            for i in kernel_size..height - kernel_size {
                for j in kernel_size..width - kernel_size {
                    let mut sum = 0.0;
                    let mut count = 0;

                    for di in -(kernel_size as isize)..(kernel_size as isize) {
                        for dj in -(kernel_size as isize)..(kernel_size as isize) {
                            let ni = i as isize + di;
                            let nj = j as isize + dj;

                            if ni >= 0 && ni < height as isize && nj >= 0 && nj < width as isize {
                                sum += face[[ni as usize, nj as usize]];
                                count += 1;
                            }
                        }
                    }

                    blurred[[i, j]] = sum / count as f64;
                }
            }
        }

        Ok(blurred)
    }

    fn compute_lbp_features(&self, face: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (height, width) = face.dim();

        if height < 3 || width < 3 {
            return Err(SklearsError::InvalidInput(
                "Image too small for LBP features".to_string(),
            ));
        }

        let mut lbp_features = Array2::zeros((height - 2, width - 2));

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let center = face[[i, j]];
                let mut lbp_value = 0;

                // 8-neighborhood LBP
                let neighbors = [
                    face[[i - 1, j - 1]],
                    face[[i - 1, j]],
                    face[[i - 1, j + 1]],
                    face[[i, j - 1]],
                    face[[i, j + 1]],
                    face[[i + 1, j - 1]],
                    face[[i + 1, j]],
                    face[[i + 1, j + 1]],
                ];

                for (k, &neighbor) in neighbors.iter().enumerate() {
                    if neighbor >= center {
                        lbp_value |= 1 << k;
                    }
                }

                lbp_features[[i - 1, j - 1]] = lbp_value as f64;
            }
        }

        Ok(lbp_features)
    }

    fn compute_face_embedding(&self, faces: &ArrayView3<f64>) -> SklResult<TrainedFaceManifold> {
        let (n_faces, height, width) = faces.dim();
        let image_dim = height * width;

        // Flatten faces and compute mean
        let mut face_matrix = Array2::zeros((n_faces, image_dim));

        for i in 0..n_faces {
            let face = faces.index_axis(Axis(0), i);
            let processed_face = self.preprocess_face(&face)?;

            for (flat_idx, &pixel) in processed_face.iter().enumerate() {
                face_matrix[[i, flat_idx]] = pixel;
            }
        }

        // Compute mean face
        let mean_face = face_matrix.mean_axis(Axis(0)).unwrap();

        // Center faces
        let centered_faces = &face_matrix - &mean_face;

        // Compute PCA embedding
        let face_embedding = self.compute_pca_embedding(&centered_faces.view())?;
        let effective_components = face_embedding.ncols();

        Ok(TrainedFaceManifold {
            image_size: self.image_size,
            n_components: effective_components,
            preprocessing: self.preprocessing.clone(),
            face_embedding,
            mean_face,
        })
    }

    fn compute_pca_embedding(&self, faces: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = faces.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "At least two faces are required to build a manifold".to_string(),
            ));
        }

        let target_components = self.n_components.min(n_samples).min(n_features).max(1);

        // Compute covariance matrix
        let cov = faces.t().dot(faces) / (n_samples - 1) as f64;

        // Compute eigendecomposition
        let (eigenvals, eigenvecs) = cov
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::from(format!("Eigendecomposition failed: {}", e)))?;

        // Sort eigenvalues in descending order and take top components
        let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvals
            .iter()
            .zip(eigenvecs.columns())
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take the top n_components eigenvectors
        let mut projection_matrix = Array2::zeros((n_features, target_components));
        for (i, (_, eigenvec)) in eigen_pairs.iter().take(target_components).enumerate() {
            projection_matrix.column_mut(i).assign(eigenvec);
        }

        Ok(projection_matrix)
    }
}

impl FaceManifoldLearning<TrainedFaceManifold> {
    pub fn encode_face(&self, face: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let (height, width) = face.dim();

        if (height, width) != self.state.image_size {
            return Err(SklearsError::InvalidInput(
                "Face image size doesn't match trained model".to_string(),
            ));
        }

        // Preprocess face
        let processed_face = FaceManifoldLearning::<Untrained>::new(self.state.image_size)
            .with_preprocessing(self.state.preprocessing.clone())
            .preprocess_face(face)?;

        // Flatten and center
        let face_flat = processed_face.into_shape(height * width).unwrap();
        let centered_face = face_flat - &self.state.mean_face;

        // Project to face space
        let encoded = centered_face.dot(&self.state.face_embedding);

        Ok(encoded)
    }

    pub fn reconstruct_face(&self, encoding: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        // Reconstruct from encoding
        let reconstructed_flat =
            encoding.dot(&self.state.face_embedding.t()) + &self.state.mean_face;

        // Reshape back to image
        let (height, width) = self.state.image_size;
        let reconstructed_face = reconstructed_flat.into_shape((height, width)).unwrap();

        Ok(reconstructed_face)
    }

    pub fn face_similarity(
        &self,
        face1: &ArrayView2<f64>,
        face2: &ArrayView2<f64>,
    ) -> SklResult<f64> {
        let encoding1 = self.encode_face(face1)?;
        let encoding2 = self.encode_face(face2)?;

        // Compute cosine similarity
        let dot_product = encoding1.dot(&encoding2);
        let norm1 = encoding1.dot(&encoding1).sqrt();
        let norm2 = encoding2.dot(&encoding2).sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm1 * norm2))
    }
}

impl Estimator for FaceManifoldLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array3<f64>, ()> for FaceManifoldLearning<Untrained> {
    type Fitted = FaceManifoldLearning<TrainedFaceManifold>;

    fn fit(self, faces: &Array3<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_faces, height, width) = faces.dim();

        if n_faces == 0 || height == 0 || width == 0 {
            return Err(SklearsError::InvalidInput("Empty face dataset".to_string()));
        }

        if (height, width) != self.image_size {
            return Err(SklearsError::InvalidInput(
                "Face image size doesn't match expected size".to_string(),
            ));
        }

        if n_faces < 2 {
            return Err(SklearsError::InvalidInput(
                "At least two faces are required to fit the manifold".to_string(),
            ));
        }

        let trained_state = self.compute_face_embedding(&faces.view())?;

        Ok(FaceManifoldLearning {
            image_size: self.image_size,
            n_components: trained_state.n_components,
            preprocessing: self.preprocessing,
            state: trained_state,
        })
    }
}

impl Transform<Array2<f64>, Array1<f64>> for FaceManifoldLearning<TrainedFaceManifold> {
    fn transform(&self, face: &Array2<f64>) -> SklResult<Array1<f64>> {
        self.encode_face(&face.view())
    }
}

/// Manifold-based Image Denoising
#[derive(Debug, Clone)]
pub struct ManifoldImageDenoising<S = Untrained> {
    patch_size: (usize, usize),
    n_components: usize,
    overlap_threshold: f64,
    state: S,
}

#[derive(Debug, Clone)]
pub struct TrainedImageDenoising {
    patch_size: (usize, usize),
    n_components: usize,
    overlap_threshold: f64,
    patch_embedding: Array2<f64>,
    clean_patches: Array2<f64>,
}

impl ManifoldImageDenoising<Untrained> {
    pub fn new(patch_size: (usize, usize)) -> Self {
        Self {
            patch_size,
            n_components: 20,
            overlap_threshold: 0.8,
            state: Untrained,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_overlap_threshold(mut self, threshold: f64) -> Self {
        self.overlap_threshold = threshold;
        self
    }
}

impl ManifoldImageDenoising<TrainedImageDenoising> {
    pub fn denoise_image(&self, noisy_image: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        // Extract patches from noisy image
        let image_patch_embedding = ImagePatchEmbedding::<Untrained>::new(self.state.patch_size);
        let noisy_patches = image_patch_embedding.extract_patches(noisy_image)?;

        // Find similar patches in the clean patch dictionary
        let mut denoised_patches = Array2::zeros(noisy_patches.dim());

        for (i, noisy_patch) in noisy_patches.rows().into_iter().enumerate() {
            // Find the closest clean patch
            let mut best_similarity = -1.0;
            let mut best_patch_idx = 0;

            for (j, clean_patch) in self.state.clean_patches.rows().into_iter().enumerate() {
                // Compute similarity (cosine similarity)
                let dot_product = noisy_patch.dot(&clean_patch);
                let norm1 = noisy_patch.dot(&noisy_patch).sqrt();
                let norm2 = clean_patch.dot(&clean_patch).sqrt();

                let similarity = if norm1 > 0.0 && norm2 > 0.0 {
                    dot_product / (norm1 * norm2)
                } else {
                    0.0
                };

                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_patch_idx = j;
                }
            }

            // Use the best matching clean patch
            if best_similarity > self.state.overlap_threshold {
                denoised_patches
                    .row_mut(i)
                    .assign(&self.state.clean_patches.row(best_patch_idx));
            } else {
                // If no good match, use original patch
                denoised_patches.row_mut(i).assign(&noisy_patch);
            }
        }

        // Reconstruct image from denoised patches
        let image_shape = noisy_image.dim();
        let patch_embedding = ImagePatchEmbedding::<TrainedPatchEmbedding> {
            patch_size: self.state.patch_size,
            stride: (1, 1),
            n_components: self.state.n_components,
            embedding_method: PatchEmbeddingMethod::PCA,
            state: TrainedPatchEmbedding {
                patch_size: self.state.patch_size,
                stride: (1, 1),
                n_components: self.state.n_components,
                embedding_method: PatchEmbeddingMethod::PCA,
                embedding_weights: self.state.patch_embedding.clone(),
                patch_means: Array1::zeros(self.state.patch_size.0 * self.state.patch_size.1),
            },
        };

        let denoised_image =
            patch_embedding.reconstruct_image(&denoised_patches.view(), image_shape)?;

        Ok(denoised_image)
    }
}

impl Estimator for ManifoldImageDenoising<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, ()> for ManifoldImageDenoising<Untrained> {
    type Fitted = ManifoldImageDenoising<TrainedImageDenoising>;

    fn fit(self, clean_image: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        // Extract patches from clean image to build dictionary
        let image_patch_embedding = ImagePatchEmbedding::<Untrained>::new(self.patch_size);
        let clean_patches = image_patch_embedding.extract_patches(&clean_image.view())?;

        // Compute patch embedding for efficient similarity computation
        let patch_embedding = Array2::eye(clean_patches.ncols()); // Identity for now

        let trained_state = TrainedImageDenoising {
            patch_size: self.patch_size,
            n_components: self.n_components,
            overlap_threshold: self.overlap_threshold,
            patch_embedding,
            clean_patches,
        };

        Ok(ManifoldImageDenoising {
            patch_size: self.patch_size,
            n_components: self.n_components,
            overlap_threshold: self.overlap_threshold,
            state: trained_state,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for ManifoldImageDenoising<TrainedImageDenoising> {
    fn transform(&self, noisy_image: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.denoise_image(&noisy_image.view())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_image_patch_embedding_basic() {
        let image = Array2::from_shape_fn((10, 10), |(i, j)| i as f64 + j as f64);
        let patch_embedding = ImagePatchEmbedding::new((3, 3))
            .with_n_components(5)
            .with_stride((2, 2));

        let fitted = patch_embedding.fit(&image, &()).unwrap();
        let embedded = fitted.transform(&image).unwrap();

        assert_eq!(embedded.ncols(), 5);
        // Should have (10-3)/2+1 = 4 patches in each dimension = 16 total
        assert_eq!(embedded.nrows(), 16);
    }

    #[test]
    fn test_image_patch_embedding_reconstruction() {
        let image = Array2::from_shape_fn((8, 8), |(i, j)| (i + j) as f64);
        let patch_embedding = ImagePatchEmbedding::new((4, 4))
            .with_n_components(3)
            .with_stride((2, 2));

        let fitted = patch_embedding.fit(&image, &()).unwrap();
        let embedded = fitted.transform(&image).unwrap();
        let reconstructed = fitted
            .reconstruct_image(&embedded.view(), image.dim())
            .unwrap();

        assert_eq!(reconstructed.dim(), image.dim());
        // Check that reconstruction preserves general structure
        assert!(reconstructed[[0, 0]] < reconstructed[[7, 7]]);
    }

    #[test]
    fn test_face_manifold_learning_basic() {
        // Create synthetic face dataset (3 faces, 8x8 each)
        let faces = Array3::from_shape_fn((3, 8, 8), |(face_idx, i, j)| {
            face_idx as f64 * 10.0 + i as f64 + j as f64
        });

        let face_learning = FaceManifoldLearning::new((8, 8))
            .with_n_components(2)
            .with_preprocessing(FacePreprocessing::Raw);

        let fitted = face_learning.fit(&faces, &()).unwrap();

        // Test encoding a single face
        let face = faces.index_axis(Axis(0), 0);
        let encoded = fitted.encode_face(&face).unwrap();
        assert_eq!(encoded.len(), 2);

        // Test reconstruction
        let reconstructed = fitted.reconstruct_face(&encoded.view()).unwrap();
        assert_eq!(reconstructed.dim(), (8, 8));
    }

    #[test]
    fn test_face_similarity() {
        let faces = Array3::from_shape_fn((2, 6, 6), |(face_idx, i, j)| {
            face_idx as f64 * 5.0 + i as f64 + j as f64
        });

        let face_learning = FaceManifoldLearning::new((6, 6)).with_n_components(3);

        let fitted = face_learning.fit(&faces, &()).unwrap();

        let face1 = faces.index_axis(Axis(0), 0);
        let face2 = faces.index_axis(Axis(0), 1);

        let similarity = fitted.face_similarity(&face1, &face2).unwrap();
        assert!(similarity >= -1.0 && similarity <= 1.0);
    }

    #[test]
    fn test_manifold_image_denoising() {
        let clean_image = Array2::from_shape_fn((8, 8), |(i, j)| i as f64 + j as f64);
        let denoising = ManifoldImageDenoising::new((3, 3))
            .with_n_components(5)
            .with_overlap_threshold(0.5);

        let fitted = denoising.fit(&clean_image, &()).unwrap();

        // Create a noisy version
        let mut noisy_image = clean_image.clone();
        noisy_image[[2, 2]] += 10.0; // Add noise

        let denoised = fitted.denoise_image(&noisy_image.view()).unwrap();
        assert_eq!(denoised.dim(), clean_image.dim());

        // Check that denoising worked (noise should be reduced)
        let noise_reduction = (noisy_image[[2, 2]] - clean_image[[2, 2]]).abs()
            > (denoised[[2, 2]] - clean_image[[2, 2]]).abs();
        assert!(noise_reduction);
    }

    #[test]
    fn test_patch_embedding_invalid_params() {
        let small_image = Array2::from_shape_fn((2, 2), |(i, j)| i as f64 + j as f64);
        let patch_embedding = ImagePatchEmbedding::new((5, 5)); // Patch larger than image

        assert!(patch_embedding.fit(&small_image, &()).is_err());
    }

    #[test]
    fn test_face_preprocessing_methods() {
        let face = Array2::from_shape_fn((6, 6), |(i, j)| i as f64 + j as f64);

        // Test histogram equalization
        let face_learning_hist =
            FaceManifoldLearning::new((6, 6)).with_preprocessing(FacePreprocessing::Histogram);
        let processed_hist = face_learning_hist.preprocess_face(&face.view()).unwrap();
        assert_eq!(processed_hist.dim(), face.dim());

        // Test Gaussian blur
        let face_learning_blur = FaceManifoldLearning::new((6, 6))
            .with_preprocessing(FacePreprocessing::GaussianBlur { sigma: 1.0 });
        let processed_blur = face_learning_blur.preprocess_face(&face.view()).unwrap();
        assert_eq!(processed_blur.dim(), face.dim());

        // Test LBP features
        let face_learning_lbp = FaceManifoldLearning::new((6, 6))
            .with_preprocessing(FacePreprocessing::LocalBinaryPattern);
        let processed_lbp = face_learning_lbp.preprocess_face(&face.view()).unwrap();
        assert_eq!(processed_lbp.dim(), (4, 4)); // LBP reduces size by 2 in each dimension
    }
}
