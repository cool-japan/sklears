//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::Distribution;
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
};

#[derive(Debug, Clone)]
pub struct TrainedPatchEmbedding {
    patch_size: (usize, usize),
    stride: (usize, usize),
    n_components: usize,
    embedding_method: PatchEmbeddingMethod,
    embedding_weights: Array2<f64>,
    patch_means: Array1<f64>,
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
/// Pose Estimation on Manifolds
///
/// Estimates human/object pose by learning a manifold of valid pose configurations.
/// Uses keypoint coordinates and learns the underlying pose manifold structure.
#[derive(Debug, Clone)]
pub struct PoseEstimationManifold<S = Untrained> {
    pub(super) n_keypoints: usize,
    pub(super) n_components: usize,
    pub(super) embedding_method: PoseEmbeddingMethod,
    pub(super) bone_constraints: Option<Vec<(usize, usize, f64)>>,
    pub(super) state: S,
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
        let expected_dims = self.n_keypoints * 2;
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
    pub(super) fn compute_pose_embedding(&self, poses: &Array2<f64>) -> SklResult<Array2<f64>> {
        match &self.embedding_method {
            PoseEmbeddingMethod::PCA => {
                let n_samples = poses.nrows();
                let mean = poses.mean_axis(Axis(0)).expect("operation should succeed");
                let centered = poses - &mean.insert_axis(Axis(0));
                let (_, s, vt) = centered
                    .svd(false)
                    .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
                let v = vt.t().to_owned();
                let n_comp = self.n_components.min(v.ncols());
                let projection = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();
                Ok(centered.dot(&projection))
            }
            PoseEmbeddingMethod::Isomap { n_neighbors } => {
                let n_samples = poses.nrows();
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
                let mut geodesic_dist = dist_matrix.clone();
                for i in 0..n_samples {
                    let mut neighbors: Vec<_> = (0..n_samples)
                        .filter(|&j| j != i)
                        .map(|j| (j, dist_matrix[[i, j]]))
                        .collect();
                    neighbors
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).expect("operation should succeed"));
                    let k_neighbors = std::cmp::min(*n_neighbors, neighbors.len());
                    for j in 0..n_samples {
                        if j != i && !neighbors[..k_neighbors].iter().any(|(idx, _)| *idx == j) {
                            geodesic_dist[[i, j]] = f64::INFINITY;
                        }
                    }
                }
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
                let n = geodesic_dist.nrows();
                let mut gram = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        gram[[i, j]] = -0.5 * geodesic_dist[[i, j]].powi(2);
                    }
                }
                let row_mean = gram.mean_axis(Axis(1)).expect("operation should succeed");
                let col_mean = gram.mean_axis(Axis(0)).expect("operation should succeed");
                let total_mean = gram.mean().expect("operation should succeed");
                for i in 0..n {
                    for j in 0..n {
                        gram[[i, j]] = gram[[i, j]] - row_mean[i] - col_mean[j] + total_mean;
                    }
                }
                let symmetric_gram = (&gram + &gram.t()) / 2.0;
                let (eigenvalues, eigenvectors) =
                    symmetric_gram.eigh(UPLO::Lower).map_err(|e| {
                        SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
                    })?;
                let mut eigen_pairs: Vec<_> = eigenvalues
                    .iter()
                    .enumerate()
                    .filter(|(_, &val)| val > 1e-10)
                    .collect();
                eigen_pairs.sort_by(|a, b| b.1.partial_cmp(a.1).expect("operation should succeed"));
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
                let n_samples = poses.nrows();
                let n_features = poses.ncols();
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
                let mut weights = Array2::zeros((n_samples, n_samples));
                for i in 0..n_samples {
                    let mut neighbors: Vec<_> = (0..n_samples)
                        .filter(|&j| j != i)
                        .map(|j| (j, dist_matrix[[i, j]]))
                        .collect();
                    neighbors
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).expect("operation should succeed"));
                    let k_count = std::cmp::min(*n_neighbors, neighbors.len());
                    let k_neighbors: Vec<_> =
                        neighbors[..k_count].iter().map(|(idx, _)| *idx).collect();
                    if k_neighbors.is_empty() {
                        continue;
                    }
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
                    let trace = gram.diag().sum();
                    let reg = 1e-3 * trace / k as f64;
                    for idx in 0..k {
                        gram[[idx, idx]] += reg;
                    }
                    let weight_sum: f64 = k as f64;
                    for &neighbor_idx in &k_neighbors {
                        weights[[i, neighbor_idx]] = 1.0 / weight_sum;
                    }
                }
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
                let symmetric_m = (&m + &m.t()) / 2.0;
                let (eigenvalues, eigenvectors) = symmetric_m.eigh(UPLO::Lower).map_err(|e| {
                    SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
                })?;
                let mut eigen_pairs: Vec<_> = eigenvalues.iter().enumerate().collect();
                eigen_pairs.sort_by(|a, b| a.1.partial_cmp(b.1).expect("operation should succeed"));
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
                refined = refined + &adjustments * 0.1;
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
        Ok((-min_dist / 10.0).exp())
    }
}
/// Image Patch Embedding for texture analysis and segmentation
#[derive(Debug, Clone)]
pub struct ImagePatchEmbedding<S = Untrained> {
    pub(super) patch_size: (usize, usize),
    pub(super) stride: (usize, usize),
    pub(super) n_components: usize,
    pub(super) embedding_method: PatchEmbeddingMethod,
    pub(super) state: S,
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
    pub(super) fn extract_patches(&self, image: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
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
    pub(super) fn compute_patch_embedding(
        &self,
        patches: &ArrayView2<f64>,
    ) -> SklResult<TrainedPatchEmbedding> {
        let (n_patches, patch_dim) = patches.dim();
        let patch_means = patches
            .mean_axis(Axis(0))
            .expect("operation should succeed");
        let centered_patches = patches - &patch_means;
        let embedding_weights = match self.embedding_method {
            PatchEmbeddingMethod::PCA => self.compute_pca_embedding(&centered_patches.view())?,
            PatchEmbeddingMethod::TSNE => {
                self.compute_random_projection(&centered_patches.view())?
            }
            PatchEmbeddingMethod::UMAP => {
                self.compute_random_projection(&centered_patches.view())?
            }
            PatchEmbeddingMethod::Isomap => {
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
        let cov = patches.t().dot(patches) / (n_samples - 1) as f64;
        let symmetric_cov = (&cov + &cov.t()) / 2.0;
        let (eigenvals, eigenvecs) = symmetric_cov
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::from(format!("Eigendecomposition failed: {}", e)))?;
        let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvals
            .iter()
            .zip(eigenvecs.columns())
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("operation should succeed"));
        let mut projection_matrix = Array2::zeros((n_features, self.n_components));
        for (i, (_, eigenvec)) in eigen_pairs.iter().take(self.n_components).enumerate() {
            projection_matrix.column_mut(i).assign(eigenvec);
        }
        Ok(projection_matrix)
    }
    fn compute_random_projection(&self, patches: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let (_n_samples, n_features) = patches.dim();
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0 / (n_features as f64).sqrt())
            .expect("valid distribution parameters");
        let projection_matrix = Array2::from_shape_fn((n_features, self.n_components), |(_, _)| {
            normal.sample(&mut rng)
        });
        Ok(projection_matrix)
    }
}
impl ImagePatchEmbedding<TrainedPatchEmbedding> {
    pub fn transform_image(&self, image: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let patches = ImagePatchEmbedding::<Untrained>::new(self.state.patch_size)
            .with_stride(self.state.stride)
            .extract_patches(image)?;
        let centered_patches = &patches - &self.state.patch_means;
        let embedded_patches = centered_patches.dot(&self.state.embedding_weights);
        Ok(embedded_patches)
    }
    pub fn reconstruct_image(
        &self,
        embedded_patches: &ArrayView2<f64>,
        image_shape: (usize, usize),
    ) -> SklResult<Array2<f64>> {
        let reconstructed_patches =
            embedded_patches.dot(&self.state.embedding_weights.t()) + &self.state.patch_means;
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
        for ((i, j), count) in count_matrix.indexed_iter() {
            if *count > 0.0 {
                reconstructed_image[[i, j]] /= count;
            }
        }
        Ok(reconstructed_image)
    }
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
pub struct TrainedVideoAnalysis {
    pub(super) frame_size: (usize, usize),
    pub(super) temporal_window: usize,
    pub(super) n_components: usize,
    pub(super) analysis_method: VideoAnalysisMethod,
    pub(super) temporal_embedding: Array2<f64>,
    pub(super) projection_matrix: Array2<f64>,
    pub(super) mean_frame: Array1<f64>,
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
/// Face Manifold Learning for face recognition and expression analysis
#[derive(Debug, Clone)]
pub struct FaceManifoldLearning<S = Untrained> {
    pub(super) image_size: (usize, usize),
    pub(super) n_components: usize,
    pub(super) preprocessing: FacePreprocessing,
    pub(super) state: S,
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
    pub(crate) fn preprocess_face(&self, face: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        match &self.preprocessing {
            FacePreprocessing::Raw => Ok(face.to_owned()),
            FacePreprocessing::Histogram => self.apply_histogram_equalization(face),
            FacePreprocessing::GaussianBlur { sigma } => self.apply_gaussian_blur(face, *sigma),
            FacePreprocessing::LocalBinaryPattern => self.compute_lbp_features(face),
        }
    }
    fn apply_histogram_equalization(&self, face: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let min_val = face.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = face.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val == min_val {
            return Ok(face.to_owned());
        }
        let normalized = face.mapv(|x| (x - min_val) / (max_val - min_val));
        Ok(normalized)
    }
    fn apply_gaussian_blur(&self, face: &ArrayView2<f64>, sigma: f64) -> SklResult<Array2<f64>> {
        let (height, width) = face.dim();
        let mut blurred = face.to_owned();
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
    pub(super) fn compute_face_embedding(
        &self,
        faces: &ArrayView3<f64>,
    ) -> SklResult<TrainedFaceManifold> {
        let (n_faces, height, width) = faces.dim();
        let image_dim = height * width;
        let mut face_matrix = Array2::zeros((n_faces, image_dim));
        for i in 0..n_faces {
            let face = faces.index_axis(Axis(0), i);
            let processed_face = self.preprocess_face(&face)?;
            for (flat_idx, &pixel) in processed_face.iter().enumerate() {
                face_matrix[[i, flat_idx]] = pixel;
            }
        }
        let mean_face = face_matrix
            .mean_axis(Axis(0))
            .expect("operation should succeed");
        let centered_faces = &face_matrix - &mean_face;
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
        let cov = faces.t().dot(faces) / (n_samples - 1) as f64;
        let symmetric_cov = (&cov + &cov.t()) / 2.0;
        let (eigenvals, eigenvecs) = symmetric_cov
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::from(format!("Eigendecomposition failed: {}", e)))?;
        let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvals
            .iter()
            .zip(eigenvecs.columns())
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("operation should succeed"));
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
        let processed_face = FaceManifoldLearning::<Untrained>::new(self.state.image_size)
            .with_preprocessing(self.state.preprocessing.clone())
            .preprocess_face(face)?;
        let face_flat = processed_face
            .into_shape(height * width)
            .expect("valid reshape dimensions");
        let centered_face = face_flat - &self.state.mean_face;
        let encoded = centered_face.dot(&self.state.face_embedding);
        Ok(encoded)
    }
    pub fn reconstruct_face(&self, encoding: &ArrayView1<f64>) -> SklResult<Array2<f64>> {
        let reconstructed_flat =
            encoding.dot(&self.state.face_embedding.t()) + &self.state.mean_face;
        let (height, width) = self.state.image_size;
        let reconstructed_face = reconstructed_flat
            .into_shape((height, width))
            .expect("valid reshape dimensions");
        Ok(reconstructed_face)
    }
    pub fn face_similarity(
        &self,
        face1: &ArrayView2<f64>,
        face2: &ArrayView2<f64>,
    ) -> SklResult<f64> {
        let encoding1 = self.encode_face(face1)?;
        let encoding2 = self.encode_face(face2)?;
        let dot_product = encoding1.dot(&encoding2);
        let norm1 = encoding1.dot(&encoding1).sqrt();
        let norm2 = encoding2.dot(&encoding2).sqrt();
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        Ok(dot_product / (norm1 * norm2))
    }
}
/// Object Recognition using Manifold Embeddings
///
/// Learns discriminative manifold embeddings for object recognition.
#[derive(Debug, Clone)]
pub struct ObjectRecognitionEmbedding<S = Untrained> {
    pub(super) n_components: usize,
    pub(super) embedding_method: ObjectEmbeddingMethod,
    pub(super) n_classes: usize,
    pub(super) state: S,
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
    pub(super) fn learn_contrastive_embedding(
        &self,
        X: &Array2<f64>,
        labels: &Array1<usize>,
        margin: f64,
    ) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let mean = X.mean_axis(Axis(0)).expect("operation should succeed");
        let centered = X - &mean.insert_axis(Axis(0));
        let (_, _, vt) = centered
            .svd(false)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?;
        let v = vt.t().to_owned();
        let n_comp = self.n_components.min(v.ncols());
        let projection_matrix = v.slice(scirs2_core::ndarray::s![.., ..n_comp]).to_owned();
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
        distances.sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));
        if distances.len() < 2 {
            return Ok(1.0);
        }
        let separation = (distances[1] - distances[0]) / distances[1];
        Ok(separation.max(0.0).min(1.0))
    }
    /// Get embedding for visualization
    pub fn embed_features(&self, features: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
        let normalized = features - &self.state.feature_mean;
        Ok(self.state.embedding_matrix.t().dot(&normalized))
    }
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
pub struct TrainedImageDenoising {
    pub(super) patch_size: (usize, usize),
    pub(super) n_components: usize,
    pub(super) overlap_threshold: f64,
    pub(super) patch_embedding: Array2<f64>,
    pub(super) clean_patches: Array2<f64>,
}
/// Video Manifold Analysis
///
/// Analyzes temporal video sequences by learning manifolds of video patches/frames.
#[derive(Debug, Clone)]
pub struct VideoManifoldAnalysis<S = Untrained> {
    pub(super) frame_size: (usize, usize),
    pub(super) temporal_window: usize,
    pub(super) n_components: usize,
    pub(super) analysis_method: VideoAnalysisMethod,
    pub(super) state: S,
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
    pub(super) fn extract_temporal_features(&self, video: &Array3<f64>) -> SklResult<Array2<f64>> {
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
        let (height, width) = frame1.dim();
        let mut flow = Array2::zeros((height - 1, width - 1));
        for i in 0..height - 1 {
            for j in 0..width - 1 {
                let dx = frame1[[i, j + 1]] - frame1[[i, j]];
                let dy = frame1[[i + 1, j]] - frame1[[i, j]];
                let dt = frame2[[i, j]] - frame1[[i, j]];
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
        Ok((-avg_diff).exp())
    }
}
#[derive(Debug, Clone)]
pub struct TrainedPoseEstimation {
    pub(super) n_keypoints: usize,
    pub(super) n_components: usize,
    pub(super) embedding_method: PoseEmbeddingMethod,
    pub(super) bone_constraints: Option<Vec<(usize, usize, f64)>>,
    pub(super) pose_manifold: Array2<f64>,
    pub(super) projection_matrix: Array2<f64>,
    pub(super) mean_pose: Array1<f64>,
    pub(super) reference_poses: Array2<f64>,
}
#[derive(Debug, Clone)]
pub struct TrainedObjectRecognition {
    pub(super) n_components: usize,
    pub(super) embedding_method: ObjectEmbeddingMethod,
    pub(super) n_classes: usize,
    pub(super) embedding_matrix: Array2<f64>,
    pub(super) class_prototypes: Array2<f64>,
    pub(super) feature_mean: Array1<f64>,
}
/// Manifold-based Image Denoising
#[derive(Debug, Clone)]
pub struct ManifoldImageDenoising<S = Untrained> {
    pub(super) patch_size: (usize, usize),
    pub(super) n_components: usize,
    pub(super) overlap_threshold: f64,
    pub(super) state: S,
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
        let image_patch_embedding = ImagePatchEmbedding::<Untrained>::new(self.state.patch_size);
        let noisy_patches = image_patch_embedding.extract_patches(noisy_image)?;
        let mut denoised_patches = Array2::zeros(noisy_patches.dim());
        for (i, noisy_patch) in noisy_patches.rows().into_iter().enumerate() {
            let mut best_similarity = -1.0;
            let mut best_patch_idx = 0;
            for (j, clean_patch) in self.state.clean_patches.rows().into_iter().enumerate() {
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
            if best_similarity > self.state.overlap_threshold {
                denoised_patches
                    .row_mut(i)
                    .assign(&self.state.clean_patches.row(best_patch_idx));
            } else {
                denoised_patches.row_mut(i).assign(&noisy_patch);
            }
        }
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
#[derive(Debug, Clone)]
pub struct TrainedFaceManifold {
    image_size: (usize, usize),
    pub(super) n_components: usize,
    preprocessing: FacePreprocessing,
    face_embedding: Array2<f64>,
    mean_face: Array1<f64>,
}
