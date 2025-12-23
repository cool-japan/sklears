use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::ndarray_linalg::{Eigh, SVD, UPLO};
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

/// Pose Estimation on Manifolds
///
/// Estimates human/object pose by learning a manifold of valid pose configurations.
/// Uses keypoint coordinates and learns the underlying pose manifold structure.
#[derive(Debug, Clone)]
pub struct PoseEstimationManifold<S = Untrained> {
    n_keypoints: usize,
    n_components: usize,
    embedding_method: PoseEmbeddingMethod,
    bone_constraints: Option<Vec<(usize, usize, f64)>>, // (joint1, joint2, expected_length)
    state: S,
}

#[derive(Debug, Clone)]
pub enum PoseEmbeddingMethod {
    /// PCA-based pose manifold
    PCA,
    /// Isomap for nonlinear pose manifolds
    Isomap { n_neighbors: usize },
    /// LLE for local pose structure
    LLE { n_neighbors: usize },
}

#[derive(Debug, Clone)]
pub struct TrainedPoseEstimation {
    n_keypoints: usize,
    n_components: usize,
    embedding_method: PoseEmbeddingMethod,
    bone_constraints: Option<Vec<(usize, usize, f64)>>,
    pose_manifold: Array2<f64>,     // Learned manifold embedding
    projection_matrix: Array2<f64>, // Projection from pose space to manifold
    mean_pose: Array1<f64>,         // Mean pose for normalization
    reference_poses: Array2<f64>,   // Reference training poses
}

impl PoseEstimationManifold<Untrained> {
    /// Create a new pose estimation model
    ///
    /// # Arguments
    /// * `n_keypoints` - Number of keypoints (joints) to track
    pub fn new(n_keypoints: usize) -> Self {
        Self {
            n_keypoints,
            n_components: 10,
            embedding_method: PoseEmbeddingMethod::Isomap { n_neighbors: 10 },
            bone_constraints: None,
            state: Untrained,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_embedding_method(mut self, method: PoseEmbeddingMethod) -> Self {
        self.embedding_method = method;
        self
    }

    /// Add bone length constraints for anatomically plausible poses
    ///
    /// # Arguments
    /// * `constraints` - Vec of (joint1_idx, joint2_idx, expected_length)
    pub fn with_bone_constraints(mut self, constraints: Vec<(usize, usize, f64)>) -> Self {
        self.bone_constraints = Some(constraints);
        self
    }

    fn validate_pose(&self, pose: &ArrayView1<f64>) -> SklResult<()> {
        let expected_dims = self.n_keypoints * 2; // x,y coordinates for each keypoint
        if pose.len() != expected_dims {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} dimensions ({}*2), got {}",
                expected_dims,
                self.n_keypoints,
                pose.len()
            )));
        }
        Ok(())
    }

    fn compute_pose_embedding(&self, poses: &Array2<f64>) -> SklResult<Array2<f64>> {
        match &self.embedding_method {
            PoseEmbeddingMethod::PCA => {
                // Perform PCA using SVD
                let n_samples = poses.nrows();
                let mean = poses.mean_axis(Axis(0)).unwrap();
                let centered = poses - &mean.insert_axis(Axis(0));

                let (_, s, vt) = centered
                    .svd(false, true)
                    .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

                let v = vt.unwrap().t().to_owned();
                let n_comp = self.n_components.min(v.ncols());
                let projection = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();

                Ok(centered.dot(&projection))
            }
            PoseEmbeddingMethod::Isomap { n_neighbors } => {
                // Simplified Isomap: Use geodesic distances via k-NN graph
                let n_samples = poses.nrows();
                let mut dist_matrix = Array2::zeros((n_samples, n_samples));

                // Compute pairwise Euclidean distances
                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        let pose_i = poses.row(i);
                        let pose_j = poses.row(j);
                        let dist = (&pose_i - &pose_j).mapv(|x| x * x).sum().sqrt();
                        dist_matrix[[i, j]] = dist;
                        dist_matrix[[j, i]] = dist;
                    }
                }

                // Build k-NN graph and compute geodesic distances
                let mut geodesic_dist = dist_matrix.clone();
                for i in 0..n_samples {
                    // Find k nearest neighbors
                    let mut neighbors: Vec<_> = (0..n_samples)
                        .filter(|&j| j != i)
                        .map(|j| (j, dist_matrix[[i, j]]))
                        .collect();
                    neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    // Set non-neighbor distances to infinity
                    let k_neighbors = std::cmp::min(*n_neighbors, neighbors.len());
                    for j in 0..n_samples {
                        if j != i && !neighbors[..k_neighbors].iter().any(|(idx, _)| *idx == j) {
                            geodesic_dist[[i, j]] = f64::INFINITY;
                        }
                    }
                }

                // Floyd-Warshall for shortest paths
                for k in 0..n_samples {
                    for i in 0..n_samples {
                        for j in 0..n_samples {
                            let dist_through_k = geodesic_dist[[i, k]] + geodesic_dist[[k, j]];
                            if dist_through_k < geodesic_dist[[i, j]] {
                                geodesic_dist[[i, j]] = dist_through_k;
                            }
                        }
                    }
                }

                // Classical MDS on geodesic distances
                let n = geodesic_dist.nrows();
                let mut gram = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        gram[[i, j]] = -0.5 * geodesic_dist[[i, j]].powi(2);
                    }
                }

                // Double centering
                let row_mean = gram.mean_axis(Axis(1)).unwrap();
                let col_mean = gram.mean_axis(Axis(0)).unwrap();
                let total_mean = gram.mean().unwrap();

                for i in 0..n {
                    for j in 0..n {
                        gram[[i, j]] = gram[[i, j]] - row_mean[i] - col_mean[j] + total_mean;
                    }
                }

                // Eigendecomposition
                let (eigenvalues, eigenvectors) = gram.eigh(UPLO::Lower).map_err(|e| {
                    SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
                })?;

                // Sort eigenvalues in descending order
                let mut eigen_pairs: Vec<_> = eigenvalues
                    .iter()
                    .enumerate()
                    .filter(|(_, &val)| val > 1e-10)
                    .collect();
                eigen_pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

                // Build embedding
                let n_comp = self.n_components.min(eigen_pairs.len());
                let mut embedding = Array2::zeros((n_samples, n_comp));
                for (comp_idx, (eig_idx, eigenval)) in eigen_pairs[..n_comp].iter().enumerate() {
                    let scale = eigenval.sqrt();
                    for i in 0..n_samples {
                        embedding[[i, comp_idx]] = eigenvectors[[i, *eig_idx]] * scale;
                    }
                }

                Ok(embedding)
            }
            PoseEmbeddingMethod::LLE { n_neighbors } => {
                // Simplified LLE for pose manifold
                let n_samples = poses.nrows();
                let n_features = poses.ncols();

                // Compute pairwise distances
                let mut dist_matrix = Array2::zeros((n_samples, n_samples));
                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        let pose_i = poses.row(i);
                        let pose_j = poses.row(j);
                        let dist = (&pose_i - &pose_j).mapv(|x| x * x).sum().sqrt();
                        dist_matrix[[i, j]] = dist;
                        dist_matrix[[j, i]] = dist;
                    }
                }

                // Reconstruct weights
                let mut weights = Array2::zeros((n_samples, n_samples));
                for i in 0..n_samples {
                    // Find k nearest neighbors
                    let mut neighbors: Vec<_> = (0..n_samples)
                        .filter(|&j| j != i)
                        .map(|j| (j, dist_matrix[[i, j]]))
                        .collect();
                    neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let k_count = std::cmp::min(*n_neighbors, neighbors.len());
                    let k_neighbors: Vec<_> =
                        neighbors[..k_count].iter().map(|(idx, _)| *idx).collect();

                    if k_neighbors.is_empty() {
                        continue;
                    }

                    // Build local Gram matrix
                    let k = k_neighbors.len();
                    let mut gram = Array2::<f64>::zeros((k, k));
                    for (idx_a, &a) in k_neighbors.iter().enumerate() {
                        for (idx_b, &b) in k_neighbors.iter().enumerate() {
                            let diff_a = poses.row(a).to_owned() - &poses.row(i).to_owned();
                            let diff_b = poses.row(b).to_owned() - &poses.row(i).to_owned();
                            gram[[idx_a, idx_b]] =
                                diff_a.iter().zip(diff_b.iter()).map(|(x, y)| x * y).sum();
                        }
                    }

                    // Add regularization
                    let trace = gram.diag().sum();
                    let reg = 1e-3 * trace / k as f64;
                    for idx in 0..k {
                        gram[[idx, idx]] += reg;
                    }

                    // Solve for weights (simplified: equal weights)
                    let weight_sum: f64 = k as f64;
                    for &neighbor_idx in &k_neighbors {
                        weights[[i, neighbor_idx]] = 1.0 / weight_sum;
                    }
                }

                // Compute embedding M = (I - W)^T (I - W)
                let mut m = Array2::zeros((n_samples, n_samples));
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        let mut sum = 0.0;
                        for k in 0..n_samples {
                            let w_ki = if k == i { 1.0 } else { -weights[[k, i]] };
                            let w_kj = if k == j { 1.0 } else { -weights[[k, j]] };
                            sum += w_ki * w_kj;
                        }
                        m[[i, j]] = sum;
                    }
                }

                // Eigendecomposition
                let (eigenvalues, eigenvectors) = m.eigh(UPLO::Lower).map_err(|e| {
                    SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
                })?;

                // Sort and select smallest non-zero eigenvalues (skip first which should be ~0)
                let mut eigen_pairs: Vec<_> = eigenvalues.iter().enumerate().collect();
                eigen_pairs.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

                let start_idx = eigen_pairs
                    .iter()
                    .position(|(_, &val)| val > 1e-6)
                    .unwrap_or(1);
                let n_comp = self.n_components.min(eigen_pairs.len() - start_idx);

                let mut embedding = Array2::zeros((n_samples, n_comp));
                for (comp_idx, (eig_idx, _)) in eigen_pairs[start_idx..start_idx + n_comp]
                    .iter()
                    .enumerate()
                {
                    for i in 0..n_samples {
                        embedding[[i, comp_idx]] = eigenvectors[[i, *eig_idx]];
                    }
                }

                Ok(embedding)
            }
        }
    }
}

impl PoseEstimationManifold<TrainedPoseEstimation> {
    /// Estimate pose by projecting to learned manifold
    pub fn estimate_pose(&self, initial_pose: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let normalized = initial_pose - &self.state.mean_pose;
        let embedded = normalized.dot(&self.state.projection_matrix);

        // Find nearest pose on manifold
        let mut min_dist = f64::INFINITY;
        let mut best_pose_idx = 0;

        for i in 0..self.state.pose_manifold.nrows() {
            let manifold_point = self.state.pose_manifold.row(i);
            let dist = (&embedded - &manifold_point).mapv(|x| x * x).sum();
            if dist < min_dist {
                min_dist = dist;
                best_pose_idx = i;
            }
        }

        Ok(self.state.reference_poses.row(best_pose_idx).to_owned())
    }

    /// Refine pose to satisfy bone constraints
    pub fn refine_with_constraints(&self, pose: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let mut refined = pose.to_owned();

        if let Some(ref constraints) = self.state.bone_constraints {
            // Iterative refinement to satisfy constraints
            for _iter in 0..10 {
                let mut adjustments = Array1::<f64>::zeros(pose.len());

                for &(j1, j2, expected_len) in constraints {
                    if j1 * 2 + 1 >= refined.len() || j2 * 2 + 1 >= refined.len() {
                        continue;
                    }

                    let x1 = refined[j1 * 2];
                    let y1 = refined[j1 * 2 + 1];
                    let x2 = refined[j2 * 2];
                    let y2 = refined[j2 * 2 + 1];

                    let dx = x2 - x1;
                    let dy = y2 - y1;
                    let current_len = (dx * dx + dy * dy).sqrt();

                    if current_len > 1e-10 {
                        let scale = expected_len / current_len;
                        let adjust_x = dx * (scale - 1.0) / 2.0;
                        let adjust_y = dy * (scale - 1.0) / 2.0;

                        adjustments[j1 * 2] -= adjust_x;
                        adjustments[j1 * 2 + 1] -= adjust_y;
                        adjustments[j2 * 2] += adjust_x;
                        adjustments[j2 * 2 + 1] += adjust_y;
                    }
                }

                refined = refined + &adjustments * 0.1; // Small step size
            }
        }

        Ok(refined)
    }

    /// Compute pose confidence based on distance to manifold
    pub fn pose_confidence(&self, pose: &ArrayView1<f64>) -> SklResult<f64> {
        let normalized = pose - &self.state.mean_pose;
        let embedded = normalized.dot(&self.state.projection_matrix);

        let mut min_dist = f64::INFINITY;
        for i in 0..self.state.pose_manifold.nrows() {
            let manifold_point = self.state.pose_manifold.row(i);
            let dist = (&embedded - &manifold_point).mapv(|x| x * x).sum().sqrt();
            min_dist = min_dist.min(dist);
        }

        // Convert distance to confidence (0 to 1)
        Ok((-min_dist / 10.0).exp())
    }
}

impl Fit<Array2<f64>, ()> for PoseEstimationManifold<Untrained> {
    type Fitted = PoseEstimationManifold<TrainedPoseEstimation>;

    fn fit(self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        if x.ncols() != self.n_keypoints * 2 {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} columns ({}*2), got {}",
                self.n_keypoints * 2,
                self.n_keypoints,
                x.ncols()
            )));
        }

        let mean_pose = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean_pose.clone().insert_axis(Axis(0));

        let pose_manifold = self.compute_pose_embedding(x)?;

        // Compute projection matrix (simplified PCA)
        let (_, _, vt) = centered
            .svd(false, true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
        let v = vt.unwrap().t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection_matrix = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();

        Ok(PoseEstimationManifold {
            n_keypoints: self.n_keypoints,
            n_components: self.n_components,
            embedding_method: self.embedding_method.clone(),
            bone_constraints: self.bone_constraints.clone(),
            state: TrainedPoseEstimation {
                n_keypoints: self.n_keypoints,
                n_components: self.n_components,
                embedding_method: self.embedding_method.clone(),
                bone_constraints: self.bone_constraints.clone(),
                pose_manifold,
                projection_matrix,
                mean_pose,
                reference_poses: x.to_owned(),
            },
        })
    }
}

impl Transform<Array1<f64>, Array1<f64>> for PoseEstimationManifold<TrainedPoseEstimation> {
    fn transform(&self, x: &Array1<f64>) -> SklResult<Array1<f64>> {
        self.estimate_pose(&x.view())
    }
}

/// Object Recognition using Manifold Embeddings
///
/// Learns discriminative manifold embeddings for object recognition.
#[derive(Debug, Clone)]
pub struct ObjectRecognitionEmbedding<S = Untrained> {
    n_components: usize,
    embedding_method: ObjectEmbeddingMethod,
    n_classes: usize,
    state: S,
}

#[derive(Debug, Clone)]
pub enum ObjectEmbeddingMethod {
    /// Siamese network-style contrastive learning
    Contrastive { margin: f64 },
    /// Triplet loss-based learning
    Triplet { margin: f64 },
    /// Supervised manifold learning
    Supervised,
}

#[derive(Debug, Clone)]
pub struct TrainedObjectRecognition {
    n_components: usize,
    embedding_method: ObjectEmbeddingMethod,
    n_classes: usize,
    embedding_matrix: Array2<f64>, // Learned embedding transformation
    class_prototypes: Array2<f64>, // Prototype for each class in embedding space
    feature_mean: Array1<f64>,     // Mean for normalization
}

impl ObjectRecognitionEmbedding<Untrained> {
    /// Create a new object recognition model
    ///
    /// # Arguments
    /// * `n_classes` - Number of object classes
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_components: 128,
            embedding_method: ObjectEmbeddingMethod::Contrastive { margin: 1.0 },
            n_classes,
            state: Untrained,
        }
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_embedding_method(mut self, method: ObjectEmbeddingMethod) -> Self {
        self.embedding_method = method;
        self
    }

    fn learn_contrastive_embedding(
        &self,
        X: &Array2<f64>,
        labels: &Array1<usize>,
        margin: f64,
    ) -> SklResult<Array2<f64>> {
        // Simplified contrastive learning: maximize inter-class distance, minimize intra-class distance
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Start with PCA initialization
        let mean = X.mean_axis(Axis(0)).unwrap();
        let centered = X - &mean.insert_axis(Axis(0));

        let (_, _, vt) = centered
            .svd(false, true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

        let v = vt.unwrap().t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection_matrix = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();

        // For now, return the PCA-based projection matrix directly
        // TODO: Implement proper contrastive learning that refines the transformation matrix
        // rather than per-sample embeddings
        Ok(projection_matrix)
    }
}

impl ObjectRecognitionEmbedding<TrainedObjectRecognition> {
    /// Recognize object by finding nearest class prototype
    pub fn recognize(&self, features: &ArrayView1<f64>) -> SklResult<usize> {
        let normalized = features - &self.state.feature_mean;
        let embedded = self.state.embedding_matrix.t().dot(&normalized);

        let mut min_dist = f64::INFINITY;
        let mut best_class = 0;

        for class_idx in 0..self.state.n_classes {
            let prototype = self.state.class_prototypes.row(class_idx);
            let dist = (&embedded - &prototype).mapv(|x| x * x).sum();
            if dist < min_dist {
                min_dist = dist;
                best_class = class_idx;
            }
        }

        Ok(best_class)
    }

    /// Compute recognition confidence
    pub fn recognition_confidence(&self, features: &ArrayView1<f64>) -> SklResult<f64> {
        let normalized = features - &self.state.feature_mean;
        let embedded = self.state.embedding_matrix.t().dot(&normalized);

        let mut distances: Vec<f64> = Vec::new();
        for class_idx in 0..self.state.n_classes {
            let prototype = self.state.class_prototypes.row(class_idx);
            let dist = (&embedded - &prototype).mapv(|x| x * x).sum().sqrt();
            distances.push(dist);
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if distances.len() < 2 {
            return Ok(1.0);
        }

        // Confidence based on separation between closest and second-closest
        let separation = (distances[1] - distances[0]) / distances[1];
        Ok(separation.max(0.0).min(1.0))
    }

    /// Get embedding for visualization
    pub fn embed_features(&self, features: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let normalized = features - &self.state.feature_mean;
        Ok(self.state.embedding_matrix.t().dot(&normalized))
    }
}

impl Fit<Array2<f64>, Array1<usize>> for ObjectRecognitionEmbedding<Untrained> {
    type Fitted = ObjectRecognitionEmbedding<TrainedObjectRecognition>;

    fn fit(self, x: &Array2<f64>, y: &Array1<usize>) -> SklResult<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        let feature_mean = x.mean_axis(Axis(0)).unwrap();

        let embedding_matrix = match &self.embedding_method {
            ObjectEmbeddingMethod::Contrastive { margin } => {
                self.learn_contrastive_embedding(x, y, *margin)?
            }
            ObjectEmbeddingMethod::Triplet { margin } => {
                self.learn_contrastive_embedding(x, y, *margin)?
            }
            ObjectEmbeddingMethod::Supervised => {
                // Supervised PCA-like embedding
                let centered = x - &feature_mean.clone().insert_axis(Axis(0));
                let (_, _, vt) = centered
                    .svd(false, true)
                    .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
                let v = vt.unwrap().t().to_owned();
                let n_comp = self.n_components.min(v.ncols());
                v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned()
            }
        };

        // Compute class prototypes
        let mut class_prototypes = Array2::zeros((self.n_classes, self.n_components));
        let mut class_counts = vec![0; self.n_classes];

        for (sample_idx, &label) in y.iter().enumerate() {
            if label < self.n_classes {
                let features = x.row(sample_idx).to_owned() - &feature_mean;
                let embedded = embedding_matrix.t().dot(&features);
                for comp_idx in 0..self.n_components.min(embedded.len()) {
                    class_prototypes[[label, comp_idx]] += embedded[comp_idx];
                }
                class_counts[label] += 1;
            }
        }

        // Normalize prototypes
        for class_idx in 0..self.n_classes {
            if class_counts[class_idx] > 0 {
                for comp_idx in 0..self.n_components {
                    class_prototypes[[class_idx, comp_idx]] /= class_counts[class_idx] as f64;
                }
            }
        }

        Ok(ObjectRecognitionEmbedding {
            n_components: self.n_components,
            embedding_method: self.embedding_method.clone(),
            n_classes: self.n_classes,
            state: TrainedObjectRecognition {
                n_components: self.n_components,
                embedding_method: self.embedding_method.clone(),
                n_classes: self.n_classes,
                embedding_matrix,
                class_prototypes,
                feature_mean,
            },
        })
    }
}

impl Transform<Array1<f64>, usize> for ObjectRecognitionEmbedding<TrainedObjectRecognition> {
    fn transform(&self, x: &Array1<f64>) -> SklResult<usize> {
        self.recognize(&x.view())
    }
}

/// Video Manifold Analysis
///
/// Analyzes temporal video sequences by learning manifolds of video patches/frames.
#[derive(Debug, Clone)]
pub struct VideoManifoldAnalysis<S = Untrained> {
    frame_size: (usize, usize),
    temporal_window: usize,
    n_components: usize,
    analysis_method: VideoAnalysisMethod,
    state: S,
}

#[derive(Debug, Clone)]
pub enum VideoAnalysisMethod {
    /// Temporal patch-based analysis
    TemporalPatch,
    /// Optical flow-based manifold
    OpticalFlow,
    /// Action recognition via temporal manifolds
    ActionRecognition,
}

#[derive(Debug, Clone)]
pub struct TrainedVideoAnalysis {
    frame_size: (usize, usize),
    temporal_window: usize,
    n_components: usize,
    analysis_method: VideoAnalysisMethod,
    temporal_embedding: Array2<f64>, // Learned temporal manifold
    projection_matrix: Array2<f64>,  // Projection to manifold space
    mean_frame: Array1<f64>,         // Mean frame for normalization
}

impl VideoManifoldAnalysis<Untrained> {
    /// Create a new video analysis model
    ///
    /// # Arguments
    /// * `frame_size` - (height, width) of video frames
    pub fn new(frame_size: (usize, usize)) -> Self {
        Self {
            frame_size,
            temporal_window: 8,
            n_components: 50,
            analysis_method: VideoAnalysisMethod::TemporalPatch,
            state: Untrained,
        }
    }

    pub fn with_temporal_window(mut self, window: usize) -> Self {
        self.temporal_window = window;
        self
    }

    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn with_analysis_method(mut self, method: VideoAnalysisMethod) -> Self {
        self.analysis_method = method;
        self
    }

    fn extract_temporal_features(&self, video: &Array3<f64>) -> SklResult<Array2<f64>> {
        let (n_frames, height, width) = video.dim();
        let frame_size = height * width;

        if n_frames < self.temporal_window {
            return Err(SklearsError::InvalidInput(format!(
                "Video must have at least {} frames, got {}",
                self.temporal_window, n_frames
            )));
        }

        let n_windows = n_frames - self.temporal_window + 1;
        let feature_dim = frame_size * self.temporal_window;

        let mut features = Array2::zeros((n_windows, feature_dim));

        for window_idx in 0..n_windows {
            for t in 0..self.temporal_window {
                let frame = video.slice(scirs2_core::ndarray::s![window_idx + t, .., ..]);
                let offset = t * frame_size;

                for (pixel_idx, &pixel) in frame.iter().enumerate() {
                    features[[window_idx, offset + pixel_idx]] = pixel;
                }
            }
        }

        Ok(features)
    }

    fn compute_optical_flow(
        &self,
        frame1: &ArrayView2<f64>,
        frame2: &ArrayView2<f64>,
    ) -> SklResult<Array2<f64>> {
        // Simplified optical flow using finite differences
        let (height, width) = frame1.dim();
        let mut flow = Array2::zeros((height - 1, width - 1));

        for i in 0..height - 1 {
            for j in 0..width - 1 {
                let dx = frame1[[i, j + 1]] - frame1[[i, j]];
                let dy = frame1[[i + 1, j]] - frame1[[i, j]];
                let dt = frame2[[i, j]] - frame1[[i, j]];

                // Simple gradient-based flow estimation
                let magnitude = (dx * dx + dy * dy).sqrt();
                if magnitude > 1e-6 {
                    flow[[i, j]] = dt / magnitude;
                }
            }
        }

        Ok(flow)
    }
}

impl VideoManifoldAnalysis<TrainedVideoAnalysis> {
    /// Analyze a video sequence and return temporal embedding
    pub fn analyze_video(&self, video: &Array3<f64>) -> SklResult<Array2<f64>> {
        let (n_frames, height, width) = video.dim();

        if height != self.state.frame_size.0 || width != self.state.frame_size.1 {
            return Err(SklearsError::InvalidInput(format!(
                "Frame size mismatch: expected {:?}, got ({}, {})",
                self.state.frame_size, height, width
            )));
        }

        if n_frames < self.state.temporal_window {
            return Err(SklearsError::InvalidInput(format!(
                "Video too short: need {} frames, got {}",
                self.state.temporal_window, n_frames
            )));
        }

        let n_windows = n_frames - self.state.temporal_window + 1;
        let frame_size = height * width;
        let feature_dim = frame_size * self.state.temporal_window;

        let mut features = Array2::zeros((n_windows, feature_dim));

        for window_idx in 0..n_windows {
            for t in 0..self.state.temporal_window {
                let frame = video.slice(scirs2_core::ndarray::s![window_idx + t, .., ..]);
                let offset = t * frame_size;

                for (pixel_idx, &pixel) in frame.iter().enumerate() {
                    features[[window_idx, offset + pixel_idx]] = pixel;
                }
            }
        }

        // Normalize and project
        let centered = &features - &self.state.mean_frame.clone().insert_axis(Axis(0));
        let embedded = centered.dot(&self.state.projection_matrix);

        Ok(embedded)
    }

    /// Detect action/event in video based on temporal manifold distance
    pub fn detect_action(&self, video: &Array3<f64>, threshold: f64) -> SklResult<Vec<usize>> {
        let embedded = self.analyze_video(video)?;
        let mut action_frames = Vec::new();

        for (frame_idx, embedding) in embedded.axis_iter(Axis(0)).enumerate() {
            let mut min_dist = f64::INFINITY;

            for reference_embedding in self.state.temporal_embedding.axis_iter(Axis(0)) {
                let dist = (&embedding - &reference_embedding)
                    .mapv(|x| x * x)
                    .sum()
                    .sqrt();
                min_dist = min_dist.min(dist);
            }

            if min_dist > threshold {
                action_frames.push(frame_idx);
            }
        }

        Ok(action_frames)
    }

    /// Compute temporal consistency score for video
    pub fn temporal_consistency(&self, video: &Array3<f64>) -> SklResult<f64> {
        let embedded = self.analyze_video(video)?;

        if embedded.nrows() < 2 {
            return Ok(1.0);
        }

        let mut total_diff = 0.0;
        for i in 0..embedded.nrows() - 1 {
            let curr = embedded.row(i);
            let next = embedded.row(i + 1);
            let diff = (&curr - &next).mapv(|x| x * x).sum().sqrt();
            total_diff += diff;
        }

        let avg_diff = total_diff / (embedded.nrows() - 1) as f64;
        Ok((-avg_diff).exp()) // Convert to consistency score (0 to 1)
    }
}

impl Fit<Array3<f64>, ()> for VideoManifoldAnalysis<Untrained> {
    type Fitted = VideoManifoldAnalysis<TrainedVideoAnalysis>;

    fn fit(self, x: &Array3<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_frames, height, width) = x.dim();

        if height != self.frame_size.0 || width != self.frame_size.1 {
            return Err(SklearsError::InvalidInput(format!(
                "Frame size mismatch: expected {:?}, got ({}, {})",
                self.frame_size, height, width
            )));
        }

        let features = self.extract_temporal_features(x)?;
        let mean_frame = features.mean_axis(Axis(0)).unwrap();
        let centered = &features - &mean_frame.clone().insert_axis(Axis(0));

        // Compute temporal embedding using PCA
        let (_, _, vt) = centered
            .svd(false, true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;

        let v = vt.unwrap().t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection_matrix = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();

        let temporal_embedding = centered.dot(&projection_matrix);

        Ok(VideoManifoldAnalysis {
            frame_size: self.frame_size,
            temporal_window: self.temporal_window,
            n_components: self.n_components,
            analysis_method: self.analysis_method.clone(),
            state: TrainedVideoAnalysis {
                frame_size: self.frame_size,
                temporal_window: self.temporal_window,
                n_components: self.n_components,
                analysis_method: self.analysis_method.clone(),
                temporal_embedding,
                projection_matrix,
                mean_frame,
            },
        })
    }
}

impl Transform<Array3<f64>, Array2<f64>> for VideoManifoldAnalysis<TrainedVideoAnalysis> {
    fn transform(&self, x: &Array3<f64>) -> SklResult<Array2<f64>> {
        self.analyze_video(x)
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

    #[test]
    fn test_pose_estimation_basic() {
        // Create synthetic pose dataset (10 poses, 5 keypoints each, 2D coordinates)
        let n_poses = 10;
        let n_keypoints = 5;
        let poses = Array2::from_shape_fn((n_poses, n_keypoints * 2), |(pose_idx, coord_idx)| {
            pose_idx as f64 + coord_idx as f64 * 0.1
        });

        let pose_model = PoseEstimationManifold::new(n_keypoints)
            .with_n_components(3)
            .with_embedding_method(PoseEmbeddingMethod::PCA);

        let fitted = pose_model.fit(&poses, &()).unwrap();

        // Test pose estimation
        let test_pose = poses.row(0);
        let estimated = fitted.estimate_pose(&test_pose).unwrap();
        assert_eq!(estimated.len(), n_keypoints * 2);
    }

    #[test]
    fn test_pose_estimation_with_constraints() {
        let n_keypoints = 4;
        let poses = Array2::from_shape_fn((8, n_keypoints * 2), |(pose_idx, coord_idx)| {
            pose_idx as f64 + coord_idx as f64
        });

        // Add bone constraints: joint 0-1 should have length 5.0, joint 1-2 should have length 3.0
        let constraints = vec![(0, 1, 5.0), (1, 2, 3.0)];

        let pose_model = PoseEstimationManifold::new(n_keypoints)
            .with_n_components(2)
            .with_bone_constraints(constraints);

        let fitted = pose_model.fit(&poses, &()).unwrap();

        let test_pose = poses.row(0);
        let refined = fitted.refine_with_constraints(&test_pose).unwrap();
        assert_eq!(refined.len(), n_keypoints * 2);
    }

    #[test]
    fn test_pose_confidence() {
        let n_keypoints = 3;
        let poses = Array2::from_shape_fn((6, n_keypoints * 2), |(pose_idx, coord_idx)| {
            pose_idx as f64 * 2.0 + coord_idx as f64
        });

        let pose_model = PoseEstimationManifold::new(n_keypoints).with_n_components(2);
        let fitted = pose_model.fit(&poses, &()).unwrap();

        let test_pose = poses.row(0);
        let confidence = fitted.pose_confidence(&test_pose).unwrap();
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_object_recognition_basic() {
        // Create synthetic object features (15 samples, 3 classes, 20 features each)
        let n_samples = 15;
        let n_features = 20;
        let n_classes = 3;

        let features = Array2::from_shape_fn((n_samples, n_features), |(sample_idx, feat_idx)| {
            (sample_idx / 5) as f64 * 10.0 + feat_idx as f64
        });

        let labels = Array1::from_shape_fn(n_samples, |i| i / 5); // 0,0,0,0,0,1,1,1,1,1,2,2,2,2,2

        let recognition_model = ObjectRecognitionEmbedding::new(n_classes).with_n_components(10);

        let fitted = recognition_model.fit(&features, &labels).unwrap();

        // Test recognition
        let test_sample = features.row(0);
        let predicted_class = fitted.recognize(&test_sample).unwrap();
        assert!(predicted_class < n_classes);
    }

    #[test]
    fn test_object_recognition_confidence() {
        let features = Array2::from_shape_fn((12, 15), |(sample_idx, feat_idx)| {
            (sample_idx / 4) as f64 * 5.0 + feat_idx as f64
        });
        let labels = Array1::from_shape_fn(12, |i| i / 4);

        let recognition_model = ObjectRecognitionEmbedding::new(3)
            .with_n_components(8)
            .with_embedding_method(ObjectEmbeddingMethod::Supervised);

        let fitted = recognition_model.fit(&features, &labels).unwrap();

        let test_sample = features.row(0);
        let confidence = fitted.recognition_confidence(&test_sample).unwrap();
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_object_embedding() {
        let features = Array2::from_shape_fn((9, 12), |(sample_idx, feat_idx)| {
            sample_idx as f64 + feat_idx as f64 * 0.5
        });
        let labels = Array1::from_shape_fn(9, |i| i / 3);

        let recognition_model = ObjectRecognitionEmbedding::new(3).with_n_components(5);
        let fitted = recognition_model.fit(&features, &labels).unwrap();

        let test_sample = features.row(0);
        let embedded = fitted.embed_features(&test_sample).unwrap();
        assert_eq!(embedded.len(), 5);
    }

    #[test]
    fn test_video_manifold_basic() {
        // Create synthetic video (12 frames, 8x8 resolution)
        let n_frames = 12;
        let height = 8;
        let width = 8;

        let video = Array3::from_shape_fn((n_frames, height, width), |(frame, i, j)| {
            frame as f64 + i as f64 * 0.5 + j as f64 * 0.3
        });

        let video_model = VideoManifoldAnalysis::new((height, width))
            .with_temporal_window(4)
            .with_n_components(10);

        let fitted = video_model.fit(&video, &()).unwrap();

        // Test video analysis
        let embedded = fitted.analyze_video(&video).unwrap();
        assert_eq!(embedded.ncols(), 10);
        // With 12 frames and window of 4, we get 12-4+1=9 temporal windows
        assert_eq!(embedded.nrows(), 9);
    }

    #[test]
    fn test_video_action_detection() {
        let video = Array3::from_shape_fn((10, 6, 6), |(frame, i, j)| {
            frame as f64 + i as f64 + j as f64
        });

        let video_model = VideoManifoldAnalysis::new((6, 6))
            .with_temporal_window(3)
            .with_n_components(5);

        let fitted = video_model.fit(&video, &()).unwrap();

        let action_frames = fitted.detect_action(&video, 10.0).unwrap();
        // Should return frame indices
        assert!(action_frames.len() <= 8); // 10-3+1=8 windows
    }

    #[test]
    fn test_video_temporal_consistency() {
        let video = Array3::from_shape_fn((8, 5, 5), |(frame, i, j)| {
            frame as f64 * 2.0 + i as f64 + j as f64
        });

        let video_model = VideoManifoldAnalysis::new((5, 5))
            .with_temporal_window(2)
            .with_n_components(4);

        let fitted = video_model.fit(&video, &()).unwrap();

        let consistency = fitted.temporal_consistency(&video).unwrap();
        assert!(consistency >= 0.0 && consistency <= 1.0);
    }

    #[test]
    fn test_pose_estimation_invalid_input() {
        let poses = Array2::from_shape_fn((5, 8), |(i, j)| i as f64 + j as f64); // 4 keypoints
        let pose_model = PoseEstimationManifold::new(5); // Expects 5 keypoints (10 coords)

        // Should fail because data has 8 columns (4 keypoints) but model expects 10 (5 keypoints)
        assert!(pose_model.fit(&poses, &()).is_err());
    }

    #[test]
    fn test_video_size_mismatch() {
        let video = Array3::from_shape_fn((10, 8, 8), |(f, i, j)| f as f64 + i as f64 + j as f64);
        let video_model = VideoManifoldAnalysis::new((6, 6)); // Expects 6x6 frames

        // Should fail because video has 8x8 frames but model expects 6x6
        assert!(video_model.fit(&video, &()).is_err());
    }
}
