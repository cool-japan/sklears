//! Computer Vision Metrics
//!
//! This module provides comprehensive metrics for computer vision tasks including
//! image quality assessment, object detection evaluation, and segmentation metrics.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{s, Array2};
use scirs2_core::numeric::Float;

/// Peak Signal-to-Noise Ratio (PSNR)
///
/// Measures the ratio between the maximum possible power of a signal and the power of
/// corrupting noise that affects the fidelity of its representation.
///
/// PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
/// where MAX_I is the maximum possible pixel value and MSE is the mean squared error.
///
/// # Arguments
/// * `original` - Original (reference) image as a 2D array
/// * `compressed` - Compressed/modified image as a 2D array
/// * `max_value` - Maximum possible pixel value (e.g., 255 for 8-bit images)
///
/// # Returns
/// PSNR value in decibels (dB). Higher values indicate better quality.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use sklears_metrics::computer_vision::psnr;
///
/// let original = array![[100.0, 150.0], [200.0, 250.0]];
/// let compressed = array![[95.0, 155.0], [205.0, 245.0]];
/// let psnr_value = psnr(&original, &compressed, 255.0).unwrap();
/// assert!(psnr_value > 0.0);
/// ```
pub fn psnr<T>(original: &Array2<T>, compressed: &Array2<T>, max_value: T) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    if original.shape() != compressed.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: original.shape().to_vec(),
            actual: compressed.shape().to_vec(),
        });
    }

    if original.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Calculate MSE
    let mse = original
        .iter()
        .zip(compressed.iter())
        .map(|(o, c)| (*o - *c).powi(2))
        .fold(T::zero(), |acc, x| acc + x)
        / T::from(original.len()).unwrap();

    if mse == T::zero() {
        // Images are identical, return infinity (very high PSNR)
        return Ok(T::infinity());
    }

    // PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    let psnr_value =
        T::from(20.0).unwrap() * max_value.log10() - T::from(10.0).unwrap() * mse.log10();

    Ok(psnr_value)
}

/// Structural Similarity Index (SSIM)
///
/// Measures the structural similarity between two images by comparing luminance,
/// contrast, and structure.
///
/// SSIM(x,y) = (2μxμy + c1)(2σxy + c2) / ((μx² + μy² + c1)(σx² + σy² + c2))
///
/// # Arguments
/// * `img1` - First image as a 2D array
/// * `img2` - Second image as a 2D array
/// * `window_size` - Size of the sliding window (must be odd)
/// * `k1` - Parameter for c1 calculation (default: 0.01)
/// * `k2` - Parameter for c2 calculation (default: 0.03)
/// * `max_value` - Maximum possible pixel value
///
/// # Returns
/// SSIM value between -1 and 1, where 1 indicates perfect similarity.
pub fn ssim<T>(
    img1: &Array2<T>,
    img2: &Array2<T>,
    window_size: usize,
    k1: T,
    k2: T,
    max_value: T,
) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    if img1.shape() != img2.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: img1.shape().to_vec(),
            actual: img2.shape().to_vec(),
        });
    }

    if window_size % 2 == 0 {
        return Err(MetricsError::InvalidParameter(
            "Window size must be odd".to_string(),
        ));
    }

    let (height, width) = img1.dim();
    let half_window = window_size / 2;

    if height < window_size || width < window_size {
        return Err(MetricsError::InvalidParameter(
            "Image dimensions must be larger than window size".to_string(),
        ));
    }

    // Constants for SSIM calculation
    let c1 = (k1 * max_value).powi(2);
    let c2 = (k2 * max_value).powi(2);

    let mut ssim_sum = T::zero();
    let mut count = 0;

    // Slide window across the image
    for i in half_window..(height - half_window) {
        for j in half_window..(width - half_window) {
            // Extract windows
            let window1 = img1.slice(s![
                i - half_window..=i + half_window,
                j - half_window..=j + half_window
            ]);
            let window2 = img2.slice(s![
                i - half_window..=i + half_window,
                j - half_window..=j + half_window
            ]);

            // Calculate means manually
            let mu1 =
                window1.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(window1.len()).unwrap();
            let mu2 =
                window2.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(window2.len()).unwrap();

            // Calculate variances and covariance
            let var1 = window1
                .iter()
                .map(|&x| (x - mu1).powi(2))
                .fold(T::zero(), |acc, x| acc + x)
                / T::from(window1.len() - 1).unwrap();

            let var2 = window2
                .iter()
                .map(|&x| (x - mu2).powi(2))
                .fold(T::zero(), |acc, x| acc + x)
                / T::from(window2.len() - 1).unwrap();

            let covar = window1
                .iter()
                .zip(window2.iter())
                .map(|(&x1, &x2)| (x1 - mu1) * (x2 - mu2))
                .fold(T::zero(), |acc, x| acc + x)
                / T::from(window1.len() - 1).unwrap();

            // Calculate SSIM for this window
            let numerator =
                (T::from(2.0).unwrap() * mu1 * mu2 + c1) * (T::from(2.0).unwrap() * covar + c2);
            let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (var1 + var2 + c2);

            if denominator != T::zero() {
                ssim_sum = ssim_sum + numerator / denominator;
                count += 1;
            }
        }
    }

    if count == 0 {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(ssim_sum / T::from(count).unwrap())
}

/// Intersection over Union (IoU) for bounding boxes
///
/// Calculates the intersection over union of two bounding boxes.
///
/// # Arguments
/// * `box1` - First bounding box [x1, y1, x2, y2]
/// * `box2` - Second bounding box [x1, y1, x2, y2]
///
/// # Returns
/// IoU value between 0 and 1, where 1 indicates perfect overlap.
pub fn iou_boxes<T>(box1: &[T; 4], box2: &[T; 4]) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug + PartialOrd,
{
    let [x1_1, y1_1, x2_1, y2_1] = *box1;
    let [x1_2, y1_2, x2_2, y2_2] = *box2;

    // Calculate intersection area
    let x_left = x1_1.max(x1_2);
    let y_top = y1_1.max(y1_2);
    let x_right = x2_1.min(x2_2);
    let y_bottom = y2_1.min(y2_2);

    if x_right <= x_left || y_bottom <= y_top {
        return Ok(T::zero()); // No intersection
    }

    let intersection_area = (x_right - x_left) * (y_bottom - y_top);

    // Calculate union area
    let area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    let area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    let union_area = area1 + area2 - intersection_area;

    if union_area == T::zero() {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(intersection_area / union_area)
}

/// Intersection over Union (IoU) for segmentation masks
///
/// Calculates the intersection over union for binary segmentation masks.
///
/// # Arguments
/// * `mask1` - First binary mask
/// * `mask2` - Second binary mask
///
/// # Returns
/// IoU value between 0 and 1.
pub fn iou_masks<T>(mask1: &Array2<T>, mask2: &Array2<T>) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug + PartialOrd,
{
    if mask1.shape() != mask2.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: mask1.shape().to_vec(),
            actual: mask2.shape().to_vec(),
        });
    }

    let intersection = mask1
        .iter()
        .zip(mask2.iter())
        .map(|(&a, &b)| {
            if a > T::zero() && b > T::zero() {
                T::one()
            } else {
                T::zero()
            }
        })
        .fold(T::zero(), |acc, x| acc + x);

    let union = mask1
        .iter()
        .zip(mask2.iter())
        .map(|(&a, &b)| {
            if a > T::zero() || b > T::zero() {
                T::one()
            } else {
                T::zero()
            }
        })
        .fold(T::zero(), |acc, x| acc + x);

    if union == T::zero() {
        return Ok(T::one()); // Both masks are empty, perfect match
    }

    Ok(intersection / union)
}

/// Detection result for object detection evaluation
#[derive(Debug, Clone)]
pub struct Detection<T> {
    pub bbox: [T; 4], // [x1, y1, x2, y2]
    pub confidence: T,
    pub class_id: usize,
}

/// Ground truth annotation for object detection evaluation
#[derive(Debug, Clone)]
pub struct GroundTruth<T> {
    pub bbox: [T; 4], // [x1, y1, x2, y2]
    pub class_id: usize,
}

/// Mean Average Precision (mAP) calculation for object detection
///
/// Calculates mAP across all classes using the COCO evaluation protocol.
///
/// # Arguments
/// * `detections` - Vector of detected objects
/// * `ground_truths` - Vector of ground truth annotations
/// * `iou_threshold` - IoU threshold for considering a detection as correct
/// * `num_classes` - Total number of classes
///
/// # Returns
/// mAP value between 0 and 1.
pub fn mean_average_precision<T>(
    detections: &[Detection<T>],
    ground_truths: &[GroundTruth<T>],
    iou_threshold: T,
    num_classes: usize,
) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug + PartialOrd,
{
    if num_classes == 0 {
        return Err(MetricsError::InvalidParameter(
            "Number of classes must be greater than 0".to_string(),
        ));
    }

    let mut class_aps = Vec::new();

    for class_id in 0..num_classes {
        // Filter detections and ground truths for this class
        let class_detections: Vec<_> = detections
            .iter()
            .filter(|det| det.class_id == class_id)
            .collect();

        let class_gts: Vec<_> = ground_truths
            .iter()
            .filter(|gt| gt.class_id == class_id)
            .collect();

        if class_gts.is_empty() {
            continue; // Skip classes with no ground truth
        }

        // Sort detections by confidence (descending)
        let mut sorted_detections = class_detections;
        sorted_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Calculate precision and recall
        let mut tp = vec![T::zero(); sorted_detections.len()];
        let mut fp = vec![T::zero(); sorted_detections.len()];
        let mut matched_gts = vec![false; class_gts.len()];

        for (det_idx, detection) in sorted_detections.iter().enumerate() {
            let mut max_iou = T::zero();
            let mut max_gt_idx = None;

            // Find the best matching ground truth
            for (gt_idx, gt) in class_gts.iter().enumerate() {
                if matched_gts[gt_idx] {
                    continue; // Already matched
                }

                let iou = iou_boxes(&detection.bbox, &gt.bbox)?;
                if iou > max_iou {
                    max_iou = iou;
                    max_gt_idx = Some(gt_idx);
                }
            }

            // Check if detection is correct
            if let Some(gt_idx) = max_gt_idx {
                if max_iou >= iou_threshold {
                    tp[det_idx] = T::one();
                    matched_gts[gt_idx] = true;
                } else {
                    fp[det_idx] = T::one();
                }
            } else {
                fp[det_idx] = T::one();
            }
        }

        // Calculate cumulative TP and FP
        for i in 1..tp.len() {
            tp[i] = tp[i] + tp[i - 1];
            fp[i] = fp[i] + fp[i - 1];
        }

        // Calculate precision and recall
        let num_gts = T::from(class_gts.len()).unwrap();
        let mut precisions = Vec::new();
        let mut recalls = Vec::new();

        for i in 0..tp.len() {
            let recall = tp[i] / num_gts;
            let precision = if (tp[i] + fp[i]) > T::zero() {
                tp[i] / (tp[i] + fp[i])
            } else {
                T::zero()
            };

            precisions.push(precision);
            recalls.push(recall);
        }

        // Calculate Average Precision using interpolation
        let ap = average_precision_interpolated(&precisions, &recalls)?;
        class_aps.push(ap);
    }

    if class_aps.is_empty() {
        return Ok(T::zero());
    }

    // Calculate mean AP
    let sum_ap = class_aps.iter().fold(T::zero(), |acc, &ap| acc + ap);
    Ok(sum_ap / T::from(class_aps.len()).unwrap())
}

/// Calculate Average Precision using 11-point interpolation
fn average_precision_interpolated<T>(precisions: &[T], recalls: &[T]) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug + PartialOrd,
{
    if precisions.len() != recalls.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![precisions.len()],
            actual: vec![recalls.len()],
        });
    }

    if precisions.is_empty() {
        return Ok(T::zero());
    }

    // 11-point interpolation
    let recall_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let mut interpolated_precisions = Vec::new();

    for &recall_level in &recall_levels {
        let recall_threshold = T::from(recall_level).unwrap();
        let mut max_precision = T::zero();

        // Find maximum precision for recalls >= recall_threshold
        for (i, &recall) in recalls.iter().enumerate() {
            if recall >= recall_threshold {
                max_precision = max_precision.max(precisions[i]);
            }
        }

        interpolated_precisions.push(max_precision);
    }

    // Calculate average
    let sum_precision = interpolated_precisions
        .iter()
        .fold(T::zero(), |acc, &p| acc + p);
    Ok(sum_precision / T::from(interpolated_precisions.len()).unwrap())
}

/// Mean Intersection over Union (mIoU) for semantic segmentation
///
/// Calculates the mean IoU across all classes for semantic segmentation.
///
/// # Arguments
/// * `predicted` - Predicted segmentation mask with class IDs
/// * `ground_truth` - Ground truth segmentation mask with class IDs
/// * `num_classes` - Total number of classes (including background)
///
/// # Returns
/// mIoU value between 0 and 1.
pub fn mean_iou<T>(
    predicted: &Array2<usize>,
    ground_truth: &Array2<usize>,
    num_classes: usize,
) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    if predicted.shape() != ground_truth.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: predicted.shape().to_vec(),
            actual: ground_truth.shape().to_vec(),
        });
    }

    if num_classes == 0 {
        return Err(MetricsError::InvalidParameter(
            "Number of classes must be greater than 0".to_string(),
        ));
    }

    let mut class_ious = Vec::new();

    for class_id in 0..num_classes {
        let mut intersection = 0;
        let mut union = 0;

        for (&pred, &gt) in predicted.iter().zip(ground_truth.iter()) {
            let pred_match = pred == class_id;
            let gt_match = gt == class_id;

            if pred_match && gt_match {
                intersection += 1;
            }
            if pred_match || gt_match {
                union += 1;
            }
        }

        if union > 0 {
            let iou = T::from(intersection).unwrap() / T::from(union).unwrap();
            class_ious.push(iou);
        }
    }

    if class_ious.is_empty() {
        return Ok(T::zero());
    }

    // Calculate mean IoU
    let sum_iou = class_ious.iter().fold(T::zero(), |acc, &iou| acc + iou);
    Ok(sum_iou / T::from(class_ious.len()).unwrap())
}

/// Pixel Accuracy for semantic segmentation
///
/// Calculates the percentage of correctly classified pixels.
///
/// # Arguments
/// * `predicted` - Predicted segmentation mask with class IDs
/// * `ground_truth` - Ground truth segmentation mask with class IDs
///
/// # Returns
/// Pixel accuracy value between 0 and 1.
pub fn pixel_accuracy<T>(
    predicted: &Array2<usize>,
    ground_truth: &Array2<usize>,
) -> MetricsResult<T>
where
    T: Float + std::fmt::Debug,
{
    if predicted.shape() != ground_truth.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: predicted.shape().to_vec(),
            actual: ground_truth.shape().to_vec(),
        });
    }

    if predicted.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let correct_pixels = predicted
        .iter()
        .zip(ground_truth.iter())
        .filter(|(&pred, &gt)| pred == gt)
        .count();

    let total_pixels = predicted.len();

    Ok(T::from(correct_pixels).unwrap() / T::from(total_pixels).unwrap())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_psnr_identical_images() {
        let img1 = array![[100.0, 150.0], [200.0, 250.0]];
        let img2 = array![[100.0, 150.0], [200.0, 250.0]];

        let result = psnr(&img1, &img2, 255.0).unwrap();
        assert!(result.is_infinite());
    }

    #[test]
    fn test_psnr_different_images() {
        let img1 = array![[100.0, 150.0], [200.0, 250.0]];
        let img2 = array![[95.0, 155.0], [205.0, 245.0]];

        let result = psnr(&img1, &img2, 255.0).unwrap();
        assert!(result > 0.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_iou_boxes_perfect_overlap() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [0.0, 0.0, 10.0, 10.0];

        let result = iou_boxes(&box1, &box2).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_iou_boxes_no_overlap() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [20.0, 20.0, 30.0, 30.0];

        let result = iou_boxes(&box1, &box2).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_iou_boxes_partial_overlap() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [5.0, 5.0, 15.0, 15.0];

        let result = iou_boxes(&box1, &box2).unwrap();
        // Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        let expected = 25.0 / 175.0;
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_pixel_accuracy() {
        let predicted = array![[0, 1, 2], [1, 2, 0], [2, 0, 1]];
        let ground_truth = array![[0, 1, 1], [1, 2, 0], [2, 1, 1]];

        let result: f64 = pixel_accuracy(&predicted, &ground_truth).unwrap();
        // 7 out of 9 pixels are correct: (0,0), (0,1), (1,0), (1,1), (1,2), (2,0), (2,2)
        assert_relative_eq!(result, 7.0 / 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mean_iou() {
        let predicted = array![[0, 1, 2], [1, 2, 0], [2, 0, 1]];
        let ground_truth = array![[0, 1, 1], [1, 2, 0], [2, 1, 1]];

        let result: f64 = mean_iou(&predicted, &ground_truth, 3).unwrap();
        assert!(result > 0.0 && result <= 1.0);
    }
}
