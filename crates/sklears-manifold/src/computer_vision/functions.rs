//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

use super::types::{
    FaceManifoldLearning, FacePreprocessing, ImagePatchEmbedding, ManifoldImageDenoising,
    ObjectEmbeddingMethod, ObjectRecognitionEmbedding, PoseEmbeddingMethod, PoseEstimationManifold,
    VideoManifoldAnalysis,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use sklears_core::traits::{Fit, Transform};
    #[test]
    fn test_image_patch_embedding_basic() {
        let image = Array2::from_shape_fn((10, 10), |(i, j)| i as f64 + j as f64);
        let patch_embedding = ImagePatchEmbedding::new((3, 3))
            .with_n_components(5)
            .with_stride((2, 2));
        let fitted = patch_embedding
            .fit(&image, &())
            .expect("model fitting should succeed");
        let embedded = fitted.transform(&image).expect("transform should succeed");
        assert_eq!(embedded.ncols(), 5);
        assert_eq!(embedded.nrows(), 16);
    }
    #[test]
    fn test_image_patch_embedding_reconstruction() {
        let image = Array2::from_shape_fn((8, 8), |(i, j)| (i + j) as f64);
        let patch_embedding = ImagePatchEmbedding::new((4, 4))
            .with_n_components(3)
            .with_stride((2, 2));
        let fitted = patch_embedding
            .fit(&image, &())
            .expect("model fitting should succeed");
        let embedded = fitted.transform(&image).expect("transform should succeed");
        let reconstructed = fitted
            .reconstruct_image(&embedded.view(), image.dim())
            .expect("operation should succeed");
        assert_eq!(reconstructed.dim(), image.dim());
        assert!(reconstructed[[0, 0]] < reconstructed[[7, 7]]);
    }
    #[test]
    fn test_face_manifold_learning_basic() {
        let faces = Array3::from_shape_fn((3, 8, 8), |(face_idx, i, j)| {
            face_idx as f64 * 10.0 + i as f64 + j as f64
        });
        let face_learning = FaceManifoldLearning::new((8, 8))
            .with_n_components(2)
            .with_preprocessing(FacePreprocessing::Raw);
        let fitted = face_learning
            .fit(&faces, &())
            .expect("model fitting should succeed");
        let face = faces.index_axis(Axis(0), 0);
        let encoded = fitted.encode_face(&face).expect("operation should succeed");
        assert_eq!(encoded.len(), 2);
        let reconstructed = fitted
            .reconstruct_face(&encoded.view())
            .expect("operation should succeed");
        assert_eq!(reconstructed.dim(), (8, 8));
    }
    #[test]
    fn test_face_similarity() {
        let faces = Array3::from_shape_fn((2, 6, 6), |(face_idx, i, j)| {
            face_idx as f64 * 5.0 + i as f64 + j as f64
        });
        let face_learning = FaceManifoldLearning::new((6, 6)).with_n_components(3);
        let fitted = face_learning
            .fit(&faces, &())
            .expect("model fitting should succeed");
        let face1 = faces.index_axis(Axis(0), 0);
        let face2 = faces.index_axis(Axis(0), 1);
        let similarity = fitted
            .face_similarity(&face1, &face2)
            .expect("operation should succeed");
        assert!(similarity >= -1.0 && similarity <= 1.0);
    }
    #[test]
    fn test_manifold_image_denoising() {
        let clean_image = Array2::from_shape_fn((8, 8), |(i, j)| i as f64 + j as f64);
        let denoising = ManifoldImageDenoising::new((3, 3))
            .with_n_components(5)
            .with_overlap_threshold(0.5);
        let fitted = denoising
            .fit(&clean_image, &())
            .expect("model fitting should succeed");
        let mut noisy_image = clean_image.clone();
        noisy_image[[2, 2]] += 10.0;
        let denoised = fitted
            .denoise_image(&noisy_image.view())
            .expect("operation should succeed");
        assert_eq!(denoised.dim(), clean_image.dim());
        let noise_reduction = (noisy_image[[2, 2]] - clean_image[[2, 2]]).abs()
            > (denoised[[2, 2]] - clean_image[[2, 2]]).abs();
        assert!(noise_reduction);
    }
    #[test]
    fn test_patch_embedding_invalid_params() {
        let small_image = Array2::from_shape_fn((2, 2), |(i, j)| i as f64 + j as f64);
        let patch_embedding = ImagePatchEmbedding::new((5, 5));
        assert!(patch_embedding.fit(&small_image, &()).is_err());
    }
    #[test]
    fn test_face_preprocessing_methods() {
        let face = Array2::from_shape_fn((6, 6), |(i, j)| i as f64 + j as f64);
        let face_learning_hist =
            FaceManifoldLearning::new((6, 6)).with_preprocessing(FacePreprocessing::Histogram);
        let processed_hist = face_learning_hist
            .preprocess_face(&face.view())
            .expect("operation should succeed");
        assert_eq!(processed_hist.dim(), face.dim());
        let face_learning_blur = FaceManifoldLearning::new((6, 6))
            .with_preprocessing(FacePreprocessing::GaussianBlur { sigma: 1.0 });
        let processed_blur = face_learning_blur
            .preprocess_face(&face.view())
            .expect("operation should succeed");
        assert_eq!(processed_blur.dim(), face.dim());
        let face_learning_lbp = FaceManifoldLearning::new((6, 6))
            .with_preprocessing(FacePreprocessing::LocalBinaryPattern);
        let processed_lbp = face_learning_lbp
            .preprocess_face(&face.view())
            .expect("operation should succeed");
        assert_eq!(processed_lbp.dim(), (4, 4));
    }
    #[test]
    fn test_pose_estimation_basic() {
        let n_poses = 10;
        let n_keypoints = 5;
        let poses = Array2::from_shape_fn((n_poses, n_keypoints * 2), |(pose_idx, coord_idx)| {
            pose_idx as f64 + coord_idx as f64 * 0.1
        });
        let pose_model = PoseEstimationManifold::new(n_keypoints)
            .with_n_components(3)
            .with_embedding_method(PoseEmbeddingMethod::PCA);
        let fitted = pose_model
            .fit(&poses, &())
            .expect("model fitting should succeed");
        let test_pose = poses.row(0);
        let estimated = fitted
            .estimate_pose(&test_pose)
            .expect("operation should succeed");
        assert_eq!(estimated.len(), n_keypoints * 2);
    }
    #[test]
    fn test_pose_estimation_with_constraints() {
        let n_keypoints = 4;
        let poses = Array2::from_shape_fn((8, n_keypoints * 2), |(pose_idx, coord_idx)| {
            pose_idx as f64 + coord_idx as f64
        });
        let constraints = vec![(0, 1, 5.0), (1, 2, 3.0)];
        let pose_model = PoseEstimationManifold::new(n_keypoints)
            .with_n_components(2)
            .with_bone_constraints(constraints);
        let fitted = pose_model
            .fit(&poses, &())
            .expect("model fitting should succeed");
        let test_pose = poses.row(0);
        let refined = fitted
            .refine_with_constraints(&test_pose)
            .expect("operation should succeed");
        assert_eq!(refined.len(), n_keypoints * 2);
    }
    #[test]
    fn test_pose_confidence() {
        let n_keypoints = 3;
        let poses = Array2::from_shape_fn((6, n_keypoints * 2), |(pose_idx, coord_idx)| {
            pose_idx as f64 * 2.0 + coord_idx as f64
        });
        let pose_model = PoseEstimationManifold::new(n_keypoints).with_n_components(2);
        let fitted = pose_model
            .fit(&poses, &())
            .expect("model fitting should succeed");
        let test_pose = poses.row(0);
        let confidence = fitted
            .pose_confidence(&test_pose)
            .expect("operation should succeed");
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
    #[test]
    fn test_object_recognition_basic() {
        let n_samples = 15;
        let n_features = 20;
        let n_classes = 3;
        let features = Array2::from_shape_fn((n_samples, n_features), |(sample_idx, feat_idx)| {
            (sample_idx / 5) as f64 * 10.0 + feat_idx as f64
        });
        let labels = Array1::from_shape_fn(n_samples, |i| i / 5);
        let recognition_model = ObjectRecognitionEmbedding::new(n_classes).with_n_components(10);
        let fitted = recognition_model
            .fit(&features, &labels)
            .expect("model fitting should succeed");
        let test_sample = features.row(0);
        let predicted_class = fitted
            .recognize(&test_sample)
            .expect("operation should succeed");
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
        let fitted = recognition_model
            .fit(&features, &labels)
            .expect("model fitting should succeed");
        let test_sample = features.row(0);
        let confidence = fitted
            .recognition_confidence(&test_sample)
            .expect("operation should succeed");
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
    #[test]
    fn test_object_embedding() {
        let features = Array2::from_shape_fn((9, 12), |(sample_idx, feat_idx)| {
            sample_idx as f64 + feat_idx as f64 * 0.5
        });
        let labels = Array1::from_shape_fn(9, |i| i / 3);
        let recognition_model = ObjectRecognitionEmbedding::new(3).with_n_components(5);
        let fitted = recognition_model
            .fit(&features, &labels)
            .expect("model fitting should succeed");
        let test_sample = features.row(0);
        let embedded = fitted
            .embed_features(&test_sample)
            .expect("operation should succeed");
        assert_eq!(embedded.len(), 5);
    }
    #[test]
    fn test_video_manifold_basic() {
        let n_frames = 12;
        let height = 8;
        let width = 8;
        let video = Array3::from_shape_fn((n_frames, height, width), |(frame, i, j)| {
            frame as f64 + i as f64 * 0.5 + j as f64 * 0.3
        });
        let video_model = VideoManifoldAnalysis::new((height, width))
            .with_temporal_window(4)
            .with_n_components(10);
        let fitted = video_model
            .fit(&video, &())
            .expect("model fitting should succeed");
        let embedded = fitted
            .analyze_video(&video)
            .expect("operation should succeed");
        assert_eq!(embedded.nrows(), 9);
        assert_eq!(embedded.ncols(), 9);
    }
    #[test]
    fn test_video_action_detection() {
        let video = Array3::from_shape_fn((10, 6, 6), |(frame, i, j)| {
            frame as f64 + i as f64 + j as f64
        });
        let video_model = VideoManifoldAnalysis::new((6, 6))
            .with_temporal_window(3)
            .with_n_components(5);
        let fitted = video_model
            .fit(&video, &())
            .expect("model fitting should succeed");
        let action_frames = fitted
            .detect_action(&video, 10.0)
            .expect("operation should succeed");
        assert!(action_frames.len() <= 8);
    }
    #[test]
    fn test_video_temporal_consistency() {
        let video = Array3::from_shape_fn((8, 5, 5), |(frame, i, j)| {
            frame as f64 * 2.0 + i as f64 + j as f64
        });
        let video_model = VideoManifoldAnalysis::new((5, 5))
            .with_temporal_window(2)
            .with_n_components(4);
        let fitted = video_model
            .fit(&video, &())
            .expect("model fitting should succeed");
        let consistency = fitted
            .temporal_consistency(&video)
            .expect("operation should succeed");
        assert!(consistency >= 0.0 && consistency <= 1.0);
    }
    #[test]
    fn test_pose_estimation_invalid_input() {
        let poses = Array2::from_shape_fn((5, 8), |(i, j)| i as f64 + j as f64);
        let pose_model = PoseEstimationManifold::new(5);
        assert!(pose_model.fit(&poses, &()).is_err());
    }
    #[test]
    fn test_video_size_mismatch() {
        let video = Array3::from_shape_fn((10, 8, 8), |(f, i, j)| f as f64 + i as f64 + j as f64);
        let video_model = VideoManifoldAnalysis::new((6, 6));
        assert!(video_model.fit(&video, &()).is_err());
    }
}
