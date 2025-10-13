//! Zero-copy dataset views and efficient data access
//!
//! This module provides zero-copy dataset views that enable efficient access to
//! large datasets without unnecessary memory allocation or data copying.

use scirs2_core::ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, Slice, s};
use std::marker::PhantomData;
use std::ops::Range;
use thiserror::Error;

/// Errors for zero-copy operations
#[derive(Error, Debug)]
pub enum ZeroCopyError {
    #[error("Index out of bounds: {index} >= {len}")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("Range out of bounds: {start}..{end} exceeds {len}")]
    RangeOutOfBounds {

        start: usize,

        end: usize,

        len: usize,
    },
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
    #[error("Invalid slice parameters: {0}")]
    InvalidSlice(String),
    #[error("Alignment error: {0}")]
    Alignment(String),
}

pub type ZeroCopyResult<T> = Result<T, ZeroCopyError>;

/// Zero-copy view of a dataset
pub struct DatasetView<'a> {
    features: ArrayView2<'a, f64>,
    targets: Option<ArrayView1<'a, f64>>,
    feature_names: Option<&'a [String]>,
    sample_indices: Option<&'a [usize]>,
}

impl<'a> DatasetView<'a> {
    /// Create a new dataset view
    pub fn new(features: ArrayView2<'a, f64>, targets: Option<ArrayView1<'a, f64>>) -> Self {
        Self {
            features,
            targets,
            feature_names: None,
            sample_indices: None,
        }
    }

    /// Create a dataset view with feature names
    pub fn with_feature_names(
        features: ArrayView2<'a, f64>,
        targets: Option<ArrayView1<'a, f64>>,
        feature_names: &'a [String],
    ) -> ZeroCopyResult<Self> {
        if feature_names.len() != features.ncols() {
            return Err(ZeroCopyError::DimensionMismatch {
                expected: format!("{} feature names", features.ncols()),
                actual: format!("{} feature names", feature_names.len()),
            });
        }

        Ok(Self {
            features,
            targets,
            feature_names: Some(feature_names),
            sample_indices: None,
        })
    }

    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    /// Get the shape as (n_samples, n_features)
    pub fn shape(&self) -> (usize, usize) {
        (self.n_samples(), self.n_features())
    }

    /// Get a view of all features
    pub fn features(&self) -> ArrayView2<'a, f64> {
        self.features
    }

    /// Get a view of all targets (if available)
    pub fn targets(&self) -> Option<ArrayView1<'a, f64>> {
        self.targets
    }

    /// Get feature names
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_names
    }

    /// Get a specific sample (row) by index
    pub fn sample(&self, index: usize) -> ZeroCopyResult<ArrayView1<'a, f64>> {
        if index >= self.n_samples() {
            return Err(ZeroCopyError::IndexOutOfBounds {
                index,
                len: self.n_samples(),
            });
        }
        Ok(self.features.row(index))
    }

    /// Get a range of samples
    pub fn samples(&self, range: Range<usize>) -> ZeroCopyResult<ArrayView2<'a, f64>> {
        if range.end > self.n_samples() {
            return Err(ZeroCopyError::RangeOutOfBounds {
                start: range.start,
                end: range.end,
                len: self.n_samples(),
            });
        }
        Ok(self.features.slice(s![range, ..]))
    }

    /// Get a specific feature (column) by index
    pub fn feature(&self, index: usize) -> ZeroCopyResult<ArrayView1<'a, f64>> {
        if index >= self.n_features() {
            return Err(ZeroCopyError::IndexOutOfBounds {
                index,
                len: self.n_features(),
            });
        }
        Ok(self.features.column(index))
    }

    /// Get a range of features
    pub fn features_range(&self, range: Range<usize>) -> ZeroCopyResult<ArrayView2<'a, f64>> {
        if range.end > self.n_features() {
            return Err(ZeroCopyError::RangeOutOfBounds {
                start: range.start,
                end: range.end,
                len: self.n_features(),
            });
        }
        Ok(self.features.slice(s![.., range]))
    }

    /// Get a subview with specific sample and feature ranges
    pub fn subview(
        &self,
        sample_range: Range<usize>,
        feature_range: Range<usize>,
    ) -> ZeroCopyResult<ArrayView2<'a, f64>> {
        if sample_range.end > self.n_samples() {
            return Err(ZeroCopyError::RangeOutOfBounds {
                start: sample_range.start,
                end: sample_range.end,
                len: self.n_samples(),
            });
        }
        if feature_range.end > self.n_features() {
            return Err(ZeroCopyError::RangeOutOfBounds {
                start: feature_range.start,
                end: feature_range.end,
                len: self.n_features(),
            });
        }
        Ok(self.features.slice(s![sample_range, feature_range]))
    }

    /// Get targets for a specific range of samples
    pub fn targets_range(
        &self,
        range: Range<usize>,
    ) -> ZeroCopyResult<Option<ArrayView1<'a, f64>>> {
        if let Some(targets) = self.targets {
            if range.end > targets.len() {
                return Err(ZeroCopyError::RangeOutOfBounds {
                    start: range.start,
                    end: range.end,
                    len: targets.len(),
                });
            }
            Ok(Some(targets.slice(s![range])))
        } else {
            Ok(None)
        }
    }

    /// Create a filtered view based on a predicate
    pub fn filter<F>(&self, predicate: F) -> ZeroCopyResult<FilteredDatasetView<'a>>
    where
        F: Fn(ArrayView1<f64>) -> bool,
    {
        let mut selected_indices = Vec::new();

        for (i, sample) in self.features.axis_iter(Axis(0)).enumerate() {
            if predicate(sample) {
                selected_indices.push(i);
            }
        }

        Ok(FilteredDatasetView {
            original: self,
            indices: selected_indices,
        })
    }

    /// Create a view with only specific feature indices
    pub fn select_features(&self, indices: &[usize]) -> ZeroCopyResult<SelectedFeaturesView<'a>> {
        // Validate indices
        for &idx in indices {
            if idx >= self.n_features() {
                return Err(ZeroCopyError::IndexOutOfBounds {
                    index: idx,
                    len: self.n_features(),
                });
            }
        }

        Ok(SelectedFeaturesView {
            original: self,
            feature_indices: indices,
        })
    }

    /// Create a strided view (every nth sample)
    pub fn strided(&self, stride: usize, offset: usize) -> ZeroCopyResult<StridedDatasetView<'a>> {
        if stride == 0 {
            return Err(ZeroCopyError::InvalidSlice(
                "Stride cannot be zero".to_string(),
            ));
        }
        if offset >= self.n_samples() {
            return Err(ZeroCopyError::IndexOutOfBounds {
                index: offset,
                len: self.n_samples(),
            });
        }

        Ok(StridedDatasetView {
            original: self,
            stride,
            offset,
        })
    }

    /// Check if the view has targets
    pub fn has_targets(&self) -> bool {
        self.targets.is_some()
    }
}

/// Mutable zero-copy view of a dataset
pub struct DatasetViewMut<'a> {
    features: ArrayViewMut2<'a, f64>,
    targets: Option<ArrayViewMut1<'a, f64>>,
    feature_names: Option<&'a [String]>,
}

impl<'a> DatasetViewMut<'a> {
    /// Create a new mutable dataset view
    pub fn new(features: ArrayViewMut2<'a, f64>, targets: Option<ArrayViewMut1<'a, f64>>) -> Self {
        Self {
            features,
            targets,
            feature_names: None,
        }
    }

    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    /// Get a mutable view of a specific sample
    pub fn sample_mut(&mut self, index: usize) -> ZeroCopyResult<ArrayViewMut1<f64>> {
        if index >= self.n_samples() {
            return Err(ZeroCopyError::IndexOutOfBounds {
                index,
                len: self.n_samples(),
            });
        }
        Ok(self.features.row_mut(index))
    }

    /// Get a mutable view of a specific feature
    pub fn feature_mut(&mut self, index: usize) -> ZeroCopyResult<ArrayViewMut1<f64>> {
        if index >= self.n_features() {
            return Err(ZeroCopyError::IndexOutOfBounds {
                index,
                len: self.n_features(),
            });
        }
        Ok(self.features.column_mut(index))
    }

    /// Get mutable targets
    pub fn targets_mut(&mut self) -> Option<ArrayViewMut1<f64>> {
        self.targets.as_mut().map(|t| t.view_mut())
    }

    /// Convert to immutable view
    pub fn as_view(&self) -> DatasetView {
        DatasetView::new(
            self.features.view(),
            self.targets.as_ref().map(|t| t.view()),
        )
    }
}

/// Filtered dataset view that shows only samples matching a predicate
pub struct FilteredDatasetView<'a> {
    original: &'a DatasetView<'a>,
    indices: Vec<usize>,
}

impl<'a> FilteredDatasetView<'a> {
    /// Get the number of samples in the filtered view
    pub fn n_samples(&self) -> usize {
        self.indices.len()
    }

    /// Get the number of features (same as original)
    pub fn n_features(&self) -> usize {
        self.original.n_features()
    }

    /// Get a sample by index in the filtered view
    pub fn sample(&self, index: usize) -> ZeroCopyResult<ArrayView1<f64>> {
        if index >= self.indices.len() {
            return Err(ZeroCopyError::IndexOutOfBounds {
                index,
                len: self.indices.len(),
            });
        }
        let original_index = self.indices[index];
        self.original.sample(original_index)
    }

    /// Get the original indices of the filtered samples
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get target for a sample in the filtered view
    pub fn target(&self, index: usize) -> ZeroCopyResult<Option<f64>> {
        if index >= self.indices.len() {
            return Err(ZeroCopyError::IndexOutOfBounds {
                index,
                len: self.indices.len(),
            });
        }
        let original_index = self.indices[index];
        if let Some(targets) = self.original.targets() {
            Ok(Some(targets[original_index]))
        } else {
            Ok(None)
        }
    }
}

/// View with selected features only
pub struct SelectedFeaturesView<'a> {
    original: &'a DatasetView<'a>,
    feature_indices: &'a [usize],
}

impl<'a> SelectedFeaturesView<'a> {
    /// Get the number of samples (same as original)
    pub fn n_samples(&self) -> usize {
        self.original.n_samples()
    }

    /// Get the number of selected features
    pub fn n_features(&self) -> usize {
        self.feature_indices.len()
    }

    /// Get a sample with only selected features
    pub fn sample(&self, index: usize) -> ZeroCopyResult<Vec<f64>> {
        let original_sample = self.original.sample(index)?;
        let selected: Vec<f64> = self
            .feature_indices
            .iter()
            .map(|&feat_idx| original_sample[feat_idx])
            .collect();
        Ok(selected)
    }

    /// Get selected feature indices
    pub fn feature_indices(&self) -> &[usize] {
        self.feature_indices
    }

    /// Get target (same as original)
    pub fn target(&self, index: usize) -> ZeroCopyResult<Option<f64>> {
        if let Some(targets) = self.original.targets() {
            if index >= targets.len() {
                return Err(ZeroCopyError::IndexOutOfBounds {
                    index,
                    len: targets.len(),
                });
            }
            Ok(Some(targets[index]))
        } else {
            Ok(None)
        }
    }
}

/// Strided dataset view that shows every nth sample
pub struct StridedDatasetView<'a> {
    original: &'a DatasetView<'a>,
    stride: usize,
    offset: usize,
}

impl<'a> StridedDatasetView<'a> {
    /// Get the number of samples in the strided view
    pub fn n_samples(&self) -> usize {
        if self.offset >= self.original.n_samples() {
            0
        } else {
            (self.original.n_samples() - self.offset + self.stride - 1) / self.stride
        }
    }

    /// Get the number of features (same as original)
    pub fn n_features(&self) -> usize {
        self.original.n_features()
    }

    /// Get a sample by index in the strided view
    pub fn sample(&self, index: usize) -> ZeroCopyResult<ArrayView1<f64>> {
        if index >= self.n_samples() {
            return Err(ZeroCopyError::IndexOutOfBounds {
                index,
                len: self.n_samples(),
            });
        }
        let original_index = self.offset + index * self.stride;
        self.original.sample(original_index)
    }

    /// Get the original index for a strided index
    pub fn original_index(&self, strided_index: usize) -> Option<usize> {
        if strided_index < self.n_samples() {
            Some(self.offset + strided_index * self.stride)
        } else {
            None
        }
    }

    /// Get stride parameters
    pub fn stride_info(&self) -> (usize, usize) {
        (self.stride, self.offset)
    }
}

/// Iterator over dataset samples with zero-copy access
pub struct DatasetSampleIterator<'a> {
    view: &'a DatasetView<'a>,
    current: usize,
}

impl<'a> DatasetSampleIterator<'a> {
    fn new(view: &'a DatasetView<'a>) -> Self {
        Self { view, current: 0 }
    }
}

impl<'a> Iterator for DatasetSampleIterator<'a> {
    type Item = ZeroCopyResult<ArrayView1<'a, f64>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.view.n_samples() {
            let result = self.view.sample(self.current);
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.n_samples().saturating_sub(self.current);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for DatasetSampleIterator<'a> {}

impl<'a> DatasetView<'a> {
    /// Create an iterator over samples
    pub fn samples_iter(&self) -> DatasetSampleIterator {
        DatasetSampleIterator::new(self)
    }
}

/// Batch iterator for efficient processing of large datasets
pub struct BatchIterator<'a> {
    view: &'a DatasetView<'a>,
    batch_size: usize,
    current: usize,
}

impl<'a> BatchIterator<'a> {
    fn new(view: &'a DatasetView<'a>, batch_size: usize) -> Self {
        Self {
            view,
            batch_size,
            current: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = ZeroCopyResult<ArrayView2<'a, f64>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.view.n_samples() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.view.n_samples());
        let range = self.current..end;
        self.current = end;

        Some(self.view.samples(range))
    }
}

impl<'a> DatasetView<'a> {
    /// Create a batch iterator
    pub fn batches(&self, batch_size: usize) -> BatchIterator {
        BatchIterator::new(self, batch_size)
    }
}

/// Window view for sliding window operations
pub struct WindowView<'a> {
    view: &'a DatasetView<'a>,
    window_size: usize,
    step: usize,
}

impl<'a> WindowView<'a> {
    /// Create a new window view
    pub fn new(view: &'a DatasetView<'a>, window_size: usize, step: usize) -> ZeroCopyResult<Self> {
        if window_size == 0 {
            return Err(ZeroCopyError::InvalidSlice(
                "Window size cannot be zero".to_string(),
            ));
        }
        if step == 0 {
            return Err(ZeroCopyError::InvalidSlice(
                "Step cannot be zero".to_string(),
            ));
        }

        Ok(Self {
            view,
            window_size,
            step,
        })
    }

    /// Get the number of windows
    pub fn n_windows(&self) -> usize {
        if self.view.n_samples() < self.window_size {
            0
        } else {
            (self.view.n_samples() - self.window_size) / self.step + 1
        }
    }

    /// Get a specific window
    pub fn window(&self, index: usize) -> ZeroCopyResult<ArrayView2<'a, f64>> {
        if index >= self.n_windows() {
            return Err(ZeroCopyError::IndexOutOfBounds {
                index,
                len: self.n_windows(),
            });
        }

        let start = index * self.step;
        let end = start + self.window_size;
        self.view.samples(start..end)
    }
}

impl<'a> DatasetView<'a> {
    /// Create a window view
    pub fn windows(&self, window_size: usize, step: usize) -> ZeroCopyResult<WindowView<'a>> {
        WindowView::new(self, window_size, step)
    }
}

/// Convenient macro for creating dataset slices
macro_rules! s {
    ($($x:expr),*) => {
        ($(Slice::from($x),)*)
    };
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_dataset_view_basic() {
        let features = Array::from_shape_vec((6, 3), (0..18).map(|x| x as f64).collect()).unwrap();
        let targets = Array::from_vec((0..6).map(|x| x as f64).collect());

        let view = DatasetView::new(features.view(), Some(targets.view()));

        assert_eq!(view.n_samples(), 6);
        assert_eq!(view.n_features(), 3);
        assert_eq!(view.shape(), (6, 3));
        assert!(view.has_targets());

        // Test sample access
        let sample = view.sample(2).unwrap();
        assert_eq!(sample[0], 6.0); // 2 * 3 + 0
        assert_eq!(sample[1], 7.0); // 2 * 3 + 1
        assert_eq!(sample[2], 8.0); // 2 * 3 + 2

        // Test feature access
        let feature = view.feature(1).unwrap();
        assert_eq!(feature[0], 1.0); // 0 * 3 + 1
        assert_eq!(feature[2], 7.0); // 2 * 3 + 1

        // Test targets
        let targets_view = view.targets().unwrap();
        assert_eq!(targets_view[2], 2.0);
    }

    #[test]
    fn test_dataset_view_ranges() {
        let features = Array::from_shape_vec((10, 4), (0..40).map(|x| x as f64).collect()).unwrap();
        let view = DatasetView::new(features.view(), None);

        // Test sample range
        let samples = view.samples(2..5).unwrap();
        assert_eq!(samples.dim(), (3, 4));
        assert_eq!(samples[[0, 0]], 8.0); // Sample 2, feature 0

        // Test feature range
        let features_range = view.features_range(1..3).unwrap();
        assert_eq!(features_range.dim(), (10, 2));
        assert_eq!(features_range[[0, 0]], 1.0); // Sample 0, feature 1

        // Test subview
        let subview = view.subview(1..4, 1..3).unwrap();
        assert_eq!(subview.dim(), (3, 2));
        assert_eq!(subview[[0, 0]], 5.0); // Sample 1, feature 1
    }

    #[test]
    fn test_filtered_view() {
        let features = Array::from_shape_vec(
            (5, 2),
            vec![
                1.0, 2.0, // Sample 0: sum = 3
                3.0, 4.0, // Sample 1: sum = 7
                5.0, 6.0, // Sample 2: sum = 11
                7.0, 8.0, // Sample 3: sum = 15
                9.0, 10.0, // Sample 4: sum = 19
            ],
        )
        .unwrap();

        let view = DatasetView::new(features.view(), None);

        // Filter samples where sum > 10
        let filtered = view.filter(|sample| sample.sum() > 10.0).unwrap();

        assert_eq!(filtered.n_samples(), 3); // Samples 2, 3, 4
        assert_eq!(filtered.n_features(), 2);

        let sample0 = filtered.sample(0).unwrap();
        assert_eq!(sample0[0], 5.0); // Original sample 2
        assert_eq!(sample0[1], 6.0);

        // Check indices
        let indices = filtered.indices();
        assert_eq!(indices, &[2, 3, 4]);
    }

    #[test]
    fn test_selected_features_view() {
        let features = Array::from_shape_vec((3, 4), (0..12).map(|x| x as f64).collect()).unwrap();
        let view = DatasetView::new(features.view(), None);

        // Select features 0 and 2
        let selected = view.select_features(&[0, 2]).unwrap();

        assert_eq!(selected.n_samples(), 3);
        assert_eq!(selected.n_features(), 2);

        let sample0 = selected.sample(0).unwrap();
        assert_eq!(sample0, vec![0.0, 2.0]); // Features 0 and 2 of sample 0

        let sample1 = selected.sample(1).unwrap();
        assert_eq!(sample1, vec![4.0, 6.0]); // Features 0 and 2 of sample 1
    }

    #[test]
    fn test_strided_view() {
        let features = Array::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let view = DatasetView::new(features.view(), None);

        // Every 3rd sample starting from index 1
        let strided = view.strided(3, 1).unwrap();

        assert_eq!(strided.n_samples(), 3); // Samples 1, 4, 7
        assert_eq!(strided.n_features(), 2);

        let sample0 = strided.sample(0).unwrap();
        assert_eq!(sample0[0], 2.0); // Sample 1, feature 0: 1 * 2 + 0

        let sample1 = strided.sample(1).unwrap();
        assert_eq!(sample1[0], 8.0); // Sample 4, feature 0: 4 * 2 + 0

        // Test original index mapping
        assert_eq!(strided.original_index(0), Some(1));
        assert_eq!(strided.original_index(1), Some(4));
        assert_eq!(strided.original_index(2), Some(7));
    }

    #[test]
    fn test_sample_iterator() {
        let features = Array::from_shape_vec((3, 2), (0..6).map(|x| x as f64).collect()).unwrap();
        let view = DatasetView::new(features.view(), None);

        let mut iter = view.samples_iter();

        let sample0 = iter.next().unwrap().unwrap();
        assert_eq!(sample0[0], 0.0);
        assert_eq!(sample0[1], 1.0);

        let sample1 = iter.next().unwrap().unwrap();
        assert_eq!(sample1[0], 2.0);
        assert_eq!(sample1[1], 3.0);

        let sample2 = iter.next().unwrap().unwrap();
        assert_eq!(sample2[0], 4.0);
        assert_eq!(sample2[1], 5.0);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_batch_iterator() {
        let features = Array::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let view = DatasetView::new(features.view(), None);

        let mut iter = view.batches(3);

        let batch0 = iter.next().unwrap().unwrap();
        assert_eq!(batch0.dim(), (3, 2));

        let batch1 = iter.next().unwrap().unwrap();
        assert_eq!(batch1.dim(), (3, 2));

        let batch2 = iter.next().unwrap().unwrap();
        assert_eq!(batch2.dim(), (3, 2));

        let batch3 = iter.next().unwrap().unwrap();
        assert_eq!(batch3.dim(), (1, 2)); // Last batch is smaller

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_window_view() {
        let features = Array::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let view = DatasetView::new(features.view(), None);

        let windows = view.windows(3, 2).unwrap(); // Window size 3, step 2

        assert_eq!(windows.n_windows(), 4); // Windows at positions 0, 2, 4, 6

        let window0 = windows.window(0).unwrap();
        assert_eq!(window0.dim(), (3, 2)); // Samples 0, 1, 2

        let window1 = windows.window(1).unwrap();
        assert_eq!(window1.dim(), (3, 2)); // Samples 2, 3, 4

        // Test window content
        assert_eq!(window0[[0, 0]], 0.0); // Sample 0, feature 0
        assert_eq!(window1[[0, 0]], 4.0); // Sample 2, feature 0
    }

    #[test]
    fn test_mutable_dataset_view() {
        let mut features =
            Array::from_shape_vec((3, 2), (0..6).map(|x| x as f64).collect()).unwrap();
        let mut targets = Array::from_vec(vec![10.0, 20.0, 30.0]);

        let mut view = DatasetViewMut::new(features.view_mut(), Some(targets.view_mut()));

        assert_eq!(view.n_samples(), 3);
        assert_eq!(view.n_features(), 2);

        // Modify a sample
        {
            let mut sample = view.sample_mut(1).unwrap();
            sample[0] = 99.0;
            sample[1] = 88.0;
        }

        // Verify the change
        let immutable_view = view.as_view();
        let sample = immutable_view.sample(1).unwrap();
        assert_eq!(sample[0], 99.0);
        assert_eq!(sample[1], 88.0);

        // Modify targets
        if let Some(mut targets_mut) = view.targets_mut() {
            targets_mut[1] = 999.0;
        }

        let targets_view = immutable_view.targets().unwrap();
        assert_eq!(targets_view[1], 999.0);
    }

    #[test]
    fn test_error_handling() {
        let features = Array::from_shape_vec((3, 2), (0..6).map(|x| x as f64).collect()).unwrap();
        let view = DatasetView::new(features.view(), None);

        // Index out of bounds
        assert!(view.sample(10).is_err());
        assert!(view.feature(10).is_err());

        // Range out of bounds
        assert!(view.samples(0..10).is_err());
        assert!(view.features_range(0..10).is_err());

        // Invalid strided parameters
        assert!(view.strided(0, 0).is_err()); // Zero stride
        assert!(view.strided(1, 10).is_err()); // Offset out of bounds
    }
}
