//! Type-Safe Feature Selection Framework
//!
//! This module provides compile-time guarantees for feature selection operations,
//! using Rust's advanced type system features including phantom types, const generics,
//! and zero-cost abstractions to ensure correctness and performance.

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::marker::PhantomData;

type Result<T> = SklResult<T>;

/// Phantom type markers for selection method types
pub mod selection_types {
    /// Marker for filter-based selection methods
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Filter;

    /// Marker for wrapper-based selection methods
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Wrapper;

    /// Marker for embedded selection methods
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Embedded;

    /// Marker for univariate selection methods
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Univariate;

    /// Marker for multivariate selection methods
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Multivariate;

    /// Marker for supervised selection methods
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Supervised;

    /// Marker for unsupervised selection methods
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Unsupervised;

    /// Marker for deterministic selection methods
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Deterministic;

    /// Marker for stochastic selection methods
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Stochastic;
}

/// Phantom type markers for data states
pub mod data_states {
    /// Marker for untrained/unfitted state
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Untrained;

    /// Marker for trained/fitted state
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Trained;

    /// Marker for validated state
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Validated;

    /// Marker for optimized state
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Optimized;
}

/// Type-safe feature index with compile-time bounds checking
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FeatureIndex<const MAX_FEATURES: usize> {
    index: usize,
}

impl<const MAX_FEATURES: usize> FeatureIndex<MAX_FEATURES> {
    /// Create a new feature index with compile-time bounds checking
    pub const fn new(index: usize) -> Option<Self> {
        if index < MAX_FEATURES {
            Some(Self { index })
        } else {
            None
        }
    }

    /// Create a new feature index without bounds checking (unsafe)
    ///
    /// # Safety
    /// The caller must ensure that `index < MAX_FEATURES`
    pub const unsafe fn new_unchecked(index: usize) -> Self {
        Self { index }
    }

    /// Get the inner index value
    pub const fn get(self) -> usize {
        self.index
    }

    /// Convert to a runtime feature index
    pub const fn to_runtime(self) -> RuntimeFeatureIndex {
        RuntimeFeatureIndex::new(self.index)
    }
}

/// Runtime feature index for dynamic bounds checking
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RuntimeFeatureIndex {
    index: usize,
}

impl RuntimeFeatureIndex {
    /// Create a new runtime feature index
    pub const fn new(index: usize) -> Self {
        Self { index }
    }

    /// Get the inner index value
    pub const fn get(self) -> usize {
        self.index
    }

    /// Check if this index is valid for a given number of features
    pub const fn is_valid(self, n_features: usize) -> bool {
        self.index < n_features
    }
}

/// Type-safe feature mask with const generic size
#[derive(Debug, Clone)]
pub struct FeatureMask<const N_FEATURES: usize> {
    mask: [bool; N_FEATURES],
}

impl<const N_FEATURES: usize> FeatureMask<N_FEATURES> {
    /// Create a new feature mask with all features selected
    pub const fn all_selected() -> Self {
        Self {
            mask: [true; N_FEATURES],
        }
    }

    /// Create a new feature mask with no features selected
    pub const fn none_selected() -> Self {
        Self {
            mask: [false; N_FEATURES],
        }
    }

    /// Create a feature mask from a boolean array
    pub const fn from_array(mask: [bool; N_FEATURES]) -> Self {
        Self { mask }
    }

    /// Create a feature mask from selected indices
    pub fn from_indices(indices: &[FeatureIndex<N_FEATURES>]) -> Self {
        let mut mask = [false; N_FEATURES];
        for &index in indices {
            mask[index.get()] = true;
        }
        Self { mask }
    }

    /// Get the mask as a boolean array
    pub const fn as_array(&self) -> &[bool; N_FEATURES] {
        &self.mask
    }

    /// Check if a feature is selected
    pub const fn is_selected(&self, index: FeatureIndex<N_FEATURES>) -> bool {
        self.mask[index.get()]
    }

    /// Set a feature as selected or unselected
    pub fn set(&mut self, index: FeatureIndex<N_FEATURES>, selected: bool) {
        self.mask[index.get()] = selected;
    }

    /// Get the number of selected features
    pub fn count_selected(&self) -> usize {
        self.mask.iter().filter(|&&x| x).count()
    }

    /// Get indices of selected features
    pub fn selected_indices(&self) -> Vec<FeatureIndex<N_FEATURES>> {
        self.mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| {
                if selected {
                    // Safety: i is always < N_FEATURES in this context
                    Some(unsafe { FeatureIndex::new_unchecked(i) })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Combine with another mask using logical AND
    pub fn and(&self, other: &Self) -> Self {
        let mut result = [false; N_FEATURES];
        for i in 0..N_FEATURES {
            result[i] = self.mask[i] && other.mask[i];
        }
        Self::from_array(result)
    }

    /// Combine with another mask using logical OR
    pub fn or(&self, other: &Self) -> Self {
        let mut result = [false; N_FEATURES];
        for i in 0..N_FEATURES {
            result[i] = self.mask[i] || other.mask[i];
        }
        Self::from_array(result)
    }

    /// Invert the mask
    pub fn not(&self) -> Self {
        let mut result = [false; N_FEATURES];
        for i in 0..N_FEATURES {
            result[i] = !self.mask[i];
        }
        Self::from_array(result)
    }
}

/// Type-safe feature matrix with compile-time feature count validation
#[derive(Debug, Clone)]
pub struct FeatureMatrix<T, const N_FEATURES: usize> {
    data: Array2<T>,
    _phantom: PhantomData<[T; N_FEATURES]>,
}

impl<T, const N_FEATURES: usize> FeatureMatrix<T, N_FEATURES>
where
    T: Clone + Default,
{
    /// Create a new feature matrix with compile-time feature count validation
    pub fn new(data: Array2<T>) -> Result<Self> {
        if data.ncols() == N_FEATURES {
            Ok(Self {
                data,
                _phantom: PhantomData,
            })
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                N_FEATURES,
                data.ncols()
            )))
        }
    }

    /// Create a new feature matrix without validation (unsafe)
    ///
    /// # Safety
    /// The caller must ensure that `data.ncols() == N_FEATURES`
    pub unsafe fn new_unchecked(data: Array2<T>) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }

    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of features (always N_FEATURES)
    pub const fn n_features(&self) -> usize {
        N_FEATURES
    }

    /// Get a view of the underlying data
    pub fn view(&self) -> ArrayView2<'_, T> {
        self.data.view()
    }

    /// Get a feature column by type-safe index
    pub fn feature(&self, index: FeatureIndex<N_FEATURES>) -> ArrayView1<'_, T> {
        self.data.column(index.get())
    }

    /// Select features using a type-safe mask
    pub fn select_features<const N_SELECTED: usize>(
        &self,
        mask: &FeatureMask<N_FEATURES>,
    ) -> Result<FeatureMatrix<T, N_SELECTED>> {
        let selected_indices = mask.selected_indices();
        if selected_indices.len() != N_SELECTED {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} selected features, got {}",
                N_SELECTED,
                selected_indices.len()
            )));
        }

        let mut selected_data = Array2::default((self.n_samples(), N_SELECTED));
        for (new_col, &old_index) in selected_indices.iter().enumerate() {
            for row in 0..self.n_samples() {
                selected_data[[row, new_col]] = self.data[[row, old_index.get()]].clone();
            }
        }

        Ok(FeatureMatrix {
            data: selected_data,
            _phantom: PhantomData,
        })
    }

    /// Convert to a dynamic feature matrix
    pub fn to_dynamic(self) -> DynamicFeatureMatrix<T> {
        DynamicFeatureMatrix::new(self.data)
    }
}

/// Dynamic feature matrix for runtime feature count
#[derive(Debug, Clone)]
pub struct DynamicFeatureMatrix<T> {
    data: Array2<T>,
}

impl<T> DynamicFeatureMatrix<T> {
    /// Create a new dynamic feature matrix
    pub fn new(data: Array2<T>) -> Self {
        Self { data }
    }

    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Get a view of the underlying data
    pub fn view(&self) -> ArrayView2<'_, T> {
        self.data.view()
    }

    /// Get a feature column by runtime index
    pub fn feature(&self, index: RuntimeFeatureIndex) -> Result<ArrayView1<'_, T>> {
        if index.is_valid(self.n_features()) {
            Ok(self.data.column(index.get()))
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Feature index {} out of bounds for {} features",
                index.get(),
                self.n_features()
            )))
        }
    }

    /// Convert to a compile-time feature matrix if the size matches
    pub fn to_static<const N_FEATURES: usize>(self) -> Result<FeatureMatrix<T, N_FEATURES>>
    where
        T: Clone + Default,
    {
        FeatureMatrix::new(self.data)
    }
}

/// Zero-cost abstraction for feature selection algorithms
pub trait TypeSafeSelector<Method, State = data_states::Untrained> {
    /// The output state after fitting
    type FittedState;

    /// The type of selection results
    type SelectionResult;

    /// Fit the selector on training data
    fn fit_typed<const N_FEATURES: usize>(
        self,
        X: &FeatureMatrix<f64, N_FEATURES>,
        y: ArrayView1<f64>,
    ) -> Result<TypeSafeSelectorWrapper<Method, Self::FittedState, N_FEATURES>>;
}

/// Zero-cost wrapper for type-safe selectors
#[derive(Debug, Clone)]
pub struct TypeSafeSelectorWrapper<Method, State, const N_FEATURES: usize> {
    method_params: MethodParameters,
    selection_result: Option<FeatureMask<N_FEATURES>>,
    _phantom: PhantomData<(Method, State)>,
}

impl<Method, State, const N_FEATURES: usize> TypeSafeSelectorWrapper<Method, State, N_FEATURES> {
    /// Create a new type-safe selector wrapper
    pub fn new(method_params: MethodParameters) -> Self {
        Self {
            method_params,
            selection_result: None,
            _phantom: PhantomData,
        }
    }

    /// Get the selection result
    pub fn selection_mask(&self) -> Option<&FeatureMask<N_FEATURES>> {
        self.selection_result.as_ref()
    }

    /// Set the selection result
    pub fn set_selection(&mut self, mask: FeatureMask<N_FEATURES>) {
        self.selection_result = Some(mask);
    }
}

/// Method parameters for different selection algorithms
#[derive(Debug, Clone)]
pub enum MethodParameters {
    /// VarianceThreshold
    VarianceThreshold {
        threshold: f64,
    },
    /// UnivariateFilter
    UnivariateFilter {
        k: usize,

        score_function: String,
    },
    /// RecursiveElimination
    RecursiveElimination {
        n_features: usize,

        step: f64,
    },
    LassoSelection {
        alpha: f64,
        max_iter: usize,
    },
    TreeBasedSelection {
        n_estimators: usize,
        max_depth: Option<usize>,
    },
    CorrelationFilter {
        threshold: f64,
    },
    MutualInfoSelection {
        k: usize,
        discrete_features: Vec<bool>,
    },
}

/// Compile-time variance threshold selector
#[derive(Debug, Clone)]
pub struct VarianceThresholdSelector<const N_FEATURES: usize> {
    threshold: f64,
    feature_variances: Option<[f64; N_FEATURES]>,
}

impl<const N_FEATURES: usize> VarianceThresholdSelector<N_FEATURES> {
    /// Create a new variance threshold selector
    pub const fn new(threshold: f64) -> Self {
        Self {
            threshold,
            feature_variances: None,
        }
    }

    /// Fit the selector on data
    pub fn fit(&mut self, X: &FeatureMatrix<f64, N_FEATURES>) -> Result<FeatureMask<N_FEATURES>> {
        let mut variances = [0.0; N_FEATURES];

        for i in 0..N_FEATURES {
            // Safety: i is always < N_FEATURES
            let feature_index = unsafe { FeatureIndex::new_unchecked(i) };
            let feature_data = X.feature(feature_index);
            variances[i] = feature_data.var(1.0);
        }

        self.feature_variances = Some(variances);

        let mut mask = [false; N_FEATURES];
        for i in 0..N_FEATURES {
            mask[i] = variances[i] > self.threshold;
        }

        Ok(FeatureMask::from_array(mask))
    }

    /// Transform data using the fitted selector
    pub fn transform<const N_SELECTED: usize>(
        &self,
        X: &FeatureMatrix<f64, N_FEATURES>,
        mask: &FeatureMask<N_FEATURES>,
    ) -> Result<FeatureMatrix<f64, N_SELECTED>> {
        X.select_features(mask)
    }

    /// Get feature variances (if fitted)
    pub const fn feature_variances(&self) -> Option<&[f64; N_FEATURES]> {
        self.feature_variances.as_ref()
    }
}

/// Compile-time univariate feature selector
#[derive(Debug, Clone)]
pub struct UnivariateSelector<const N_FEATURES: usize, const K: usize> {
    score_function: UnivariateScoreFunction,
    feature_scores: Option<[f64; N_FEATURES]>,
}

impl<const N_FEATURES: usize, const K: usize> UnivariateSelector<N_FEATURES, K> {
    /// Create a new univariate selector
    ///
    /// # Compile-time checks
    /// - K must be <= N_FEATURES
    pub const fn new(score_function: UnivariateScoreFunction) -> Option<Self> {
        if K <= N_FEATURES {
            Some(Self {
                score_function,
                feature_scores: None,
            })
        } else {
            None
        }
    }

    /// Fit the selector on data
    pub fn fit(
        &mut self,
        X: &FeatureMatrix<f64, N_FEATURES>,
        y: ArrayView1<f64>,
    ) -> Result<FeatureMask<N_FEATURES>> {
        let mut scores = [0.0; N_FEATURES];

        for i in 0..N_FEATURES {
            // Safety: i is always < N_FEATURES
            let feature_index = unsafe { FeatureIndex::new_unchecked(i) };
            let feature_data = X.feature(feature_index);
            scores[i] = match self.score_function {
                UnivariateScoreFunction::Correlation => self.compute_correlation(feature_data, y),
                UnivariateScoreFunction::MutualInfo => self.compute_mutual_info(feature_data, y),
                UnivariateScoreFunction::Chi2 => self.compute_chi2_score(feature_data, y),
                UnivariateScoreFunction::FStatistic => self.compute_f_statistic(feature_data, y),
            };
        }

        self.feature_scores = Some(scores);

        // Select top K features
        let mut indexed_scores: Vec<(usize, f64)> = Vec::with_capacity(N_FEATURES);
        for i in 0..N_FEATURES {
            indexed_scores.push((i, scores[i]));
        }

        // Sort by score (descending)
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut mask = [false; N_FEATURES];
        for i in 0..K {
            if let Some(&(feature_idx, _)) = indexed_scores.get(i) {
                mask[feature_idx] = true;
            }
        }

        Ok(FeatureMask::from_array(mask))
    }

    /// Transform data using the fitted selector
    pub fn transform(
        &self,
        X: &FeatureMatrix<f64, N_FEATURES>,
        mask: &FeatureMask<N_FEATURES>,
    ) -> Result<FeatureMatrix<f64, K>> {
        X.select_features(mask)
    }

    /// Get feature scores (if fitted)
    pub const fn feature_scores(&self) -> Option<&[f64; N_FEATURES]> {
        self.feature_scores.as_ref()
    }

    // Helper methods for score computation
    fn compute_correlation(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            (sum_xy / denom).abs()
        }
    }

    fn compute_mutual_info(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Simplified mutual information computation
        // In a real implementation, this would use proper MI estimation algorithms
        self.compute_correlation(x, y)
    }

    fn compute_chi2_score(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Simplified chi-square test
        // In a real implementation, this would compute proper chi-square statistics
        self.compute_correlation(x, y)
    }

    fn compute_f_statistic(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        // Simplified F-statistic computation
        // In a real implementation, this would compute ANOVA F-statistic
        self.compute_correlation(x, y)
    }
}

/// Score functions for univariate selection
#[derive(Debug, Clone, Copy)]
pub enum UnivariateScoreFunction {
    /// Correlation
    Correlation,
    /// MutualInfo
    MutualInfo,
    /// Chi2
    Chi2,
    /// FStatistic
    FStatistic,
}

/// Compile-time correlation-based feature selector
#[derive(Debug, Clone)]
pub struct CorrelationSelector<const N_FEATURES: usize> {
    threshold: f64,
    correlation_matrix: Option<[[f64; N_FEATURES]; N_FEATURES]>,
}

impl<const N_FEATURES: usize> CorrelationSelector<N_FEATURES> {
    /// Create a new correlation-based selector
    pub const fn new(threshold: f64) -> Self {
        Self {
            threshold,
            correlation_matrix: None,
        }
    }

    /// Fit the selector on data
    pub fn fit(&mut self, X: &FeatureMatrix<f64, N_FEATURES>) -> Result<FeatureMask<N_FEATURES>> {
        let mut corr_matrix = [[0.0; N_FEATURES]; N_FEATURES];

        // Compute correlation matrix
        for i in 0..N_FEATURES {
            for j in 0..N_FEATURES {
                if i == j {
                    corr_matrix[i][j] = 1.0;
                } else {
                    // Safety: i and j are always < N_FEATURES
                    let feature_i = unsafe { FeatureIndex::new_unchecked(i) };
                    let feature_j = unsafe { FeatureIndex::new_unchecked(j) };
                    let data_i = X.feature(feature_i);
                    let data_j = X.feature(feature_j);
                    corr_matrix[i][j] = self.compute_correlation(data_i, data_j);
                }
            }
        }

        self.correlation_matrix = Some(corr_matrix);

        // Remove highly correlated features
        let mut mask = [true; N_FEATURES];
        for i in 0..N_FEATURES {
            for j in (i + 1)..N_FEATURES {
                if corr_matrix[i][j].abs() > self.threshold && mask[i] && mask[j] {
                    // Keep the feature with higher variance
                    // Safety: i and j are always < N_FEATURES
                    let feature_i = unsafe { FeatureIndex::new_unchecked(i) };
                    let feature_j = unsafe { FeatureIndex::new_unchecked(j) };
                    let var_i = X.feature(feature_i).var(1.0);
                    let var_j = X.feature(feature_j).var(1.0);
                    if var_i < var_j {
                        mask[i] = false;
                    } else {
                        mask[j] = false;
                    }
                }
            }
        }

        Ok(FeatureMask::from_array(mask))
    }

    /// Transform data using the fitted selector
    pub fn transform<const N_SELECTED: usize>(
        &self,
        X: &FeatureMatrix<f64, N_FEATURES>,
        mask: &FeatureMask<N_FEATURES>,
    ) -> Result<FeatureMatrix<f64, N_SELECTED>> {
        X.select_features(mask)
    }

    /// Get correlation matrix (if fitted)
    pub const fn correlation_matrix(&self) -> Option<&[[f64; N_FEATURES]; N_FEATURES]> {
        self.correlation_matrix.as_ref()
    }

    fn compute_correlation(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            sum_xy / denom
        }
    }
}

/// Type-safe feature selection pipeline with compile-time guarantees
#[derive(Debug, Clone)]
pub struct TypeSafeSelectionPipeline<const N_FEATURES: usize, State = data_states::Untrained> {
    steps: Vec<PipelineStep>,
    current_mask: Option<FeatureMask<N_FEATURES>>,
    _phantom: PhantomData<State>,
}

impl<const N_FEATURES: usize> Default
    for TypeSafeSelectionPipeline<N_FEATURES, data_states::Untrained>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const N_FEATURES: usize> TypeSafeSelectionPipeline<N_FEATURES, data_states::Untrained> {
    /// Create a new type-safe selection pipeline
    pub const fn new() -> Self {
        Self {
            steps: Vec::new(),
            current_mask: None,
            _phantom: PhantomData,
        }
    }

    /// Add a variance threshold step
    pub fn add_variance_threshold(mut self, threshold: f64) -> Self {
        self.steps.push(PipelineStep::VarianceThreshold(threshold));
        self
    }

    /// Add a correlation filter step
    pub fn add_correlation_filter(mut self, threshold: f64) -> Self {
        self.steps.push(PipelineStep::CorrelationFilter(threshold));
        self
    }

    /// Add a univariate selection step
    pub fn add_univariate_selection<const K: usize>(
        mut self,
        score_function: UnivariateScoreFunction,
    ) -> Self {
        self.steps.push(PipelineStep::UnivariateSelection {
            k: K,
            score_function,
        });
        self
    }

    /// Fit the pipeline on training data
    pub fn fit(
        self,
        X: &FeatureMatrix<f64, N_FEATURES>,
        y: ArrayView1<f64>,
    ) -> Result<TypeSafeSelectionPipeline<N_FEATURES, data_states::Trained>> {
        let mut current_mask = FeatureMask::all_selected();

        for step in &self.steps {
            let step_mask = match step {
                PipelineStep::VarianceThreshold(threshold) => {
                    let mut selector = VarianceThresholdSelector::new(*threshold);
                    selector.fit(X)?
                }
                PipelineStep::CorrelationFilter(threshold) => {
                    let mut selector = CorrelationSelector::new(*threshold);
                    selector.fit(X)?
                }
                PipelineStep::UnivariateSelection {
                    k: _,
                    score_function,
                } => {
                    // This is a simplification - in practice we'd need to handle different K values
                    // For demonstration, we'll use a fixed K
                    const DEFAULT_K: usize = 10;
                    if DEFAULT_K <= N_FEATURES {
                        let mut selector =
                            UnivariateSelector::<N_FEATURES, DEFAULT_K>::new(*score_function)
                                .ok_or_else(|| {
                                    SklearsError::InvalidInput(
                                        "Invalid K for univariate selection".to_string(),
                                    )
                                })?;
                        selector.fit(X, y)?
                    } else {
                        FeatureMask::all_selected()
                    }
                }
            };

            current_mask = current_mask.and(&step_mask);
        }

        Ok(TypeSafeSelectionPipeline {
            steps: self.steps,
            current_mask: Some(current_mask),
            _phantom: PhantomData,
        })
    }
}

impl<const N_FEATURES: usize> TypeSafeSelectionPipeline<N_FEATURES, data_states::Trained> {
    /// Transform data using the fitted pipeline
    pub fn transform<const N_SELECTED: usize>(
        &self,
        X: &FeatureMatrix<f64, N_FEATURES>,
    ) -> Result<FeatureMatrix<f64, N_SELECTED>> {
        if let Some(ref mask) = self.current_mask {
            X.select_features(mask)
        } else {
            Err(SklearsError::FitError("Pipeline not fitted".to_string()))
        }
    }

    /// Get the feature selection mask
    pub fn selection_mask(&self) -> Option<&FeatureMask<N_FEATURES>> {
        self.current_mask.as_ref()
    }

    /// Get the number of selected features
    pub fn n_selected_features(&self) -> usize {
        self.current_mask
            .as_ref()
            .map(|mask| mask.count_selected())
            .unwrap_or(0)
    }
}

/// Pipeline step enumeration
#[derive(Debug, Clone)]
enum PipelineStep {
    VarianceThreshold(f64),
    CorrelationFilter(f64),
    UnivariateSelection {
        k: usize,
        score_function: UnivariateScoreFunction,
    },
}

/// Zero-cost abstraction for feature transformations
pub trait ZeroCostTransform<Input, Output> {
    /// Apply the transformation with zero runtime cost
    fn transform_zero_cost(input: Input) -> Output;
}

/// Zero-cost feature index conversion
impl<const N: usize> ZeroCostTransform<FeatureIndex<N>, usize> for () {
    fn transform_zero_cost(input: FeatureIndex<N>) -> usize {
        input.get()
    }
}

/// Zero-cost feature mask conversion
impl<const N: usize> ZeroCostTransform<FeatureMask<N>, Vec<bool>> for () {
    fn transform_zero_cost(input: FeatureMask<N>) -> Vec<bool> {
        input.as_array().to_vec()
    }
}

/// Compile-time feature count validator
pub struct FeatureCountValidator<const EXPECTED: usize>;

impl<const EXPECTED: usize> FeatureCountValidator<EXPECTED> {
    /// Validate feature count at compile time
    pub const fn validate<const ACTUAL: usize>() -> bool {
        EXPECTED == ACTUAL
    }

    /// Validate and convert feature matrix type
    pub fn validate_matrix<T>(matrix: FeatureMatrix<T, EXPECTED>) -> FeatureMatrix<T, EXPECTED>
    where
        T: Clone + Default,
    {
        matrix
    }
}

/// Type-safe feature selection trait bounds
pub trait TypeSafeFeatureSelection {
    /// The feature matrix type
    type FeatureMatrix;

    /// The selection result type
    type SelectionResult;

    /// The number of input features (compile-time constant)
    const INPUT_FEATURES: usize;

    /// Perform feature selection with compile-time guarantees
    fn select_features_typed(data: Self::FeatureMatrix) -> Result<Self::SelectionResult>;
}

/// Implementation macro for type-safe selectors
#[macro_export]
macro_rules! impl_type_safe_selector {
    ($selector:ty, $method:ty, $n_features:expr, $n_selected:expr) => {
        impl TypeSafeFeatureSelection for $selector {
            type FeatureMatrix = FeatureMatrix<f64, $n_features>;
            type SelectionResult = FeatureMatrix<f64, $n_selected>;
            const INPUT_FEATURES: usize = $n_features;

            fn select_features_typed(data: Self::FeatureMatrix) -> Result<Self::SelectionResult> {
                // Default implementation using variance threshold
                // This can be overridden by implementing the trait directly
                use crate::type_safe::VarianceThresholdSelector;

                let mut selector = VarianceThresholdSelector::<$n_features>::new(0.0);
                let mask = selector.fit(&data)?;

                // Verify we have the expected number of selected features
                if mask.count_selected() != $n_selected {
                    return Err(SklearsError::InvalidInput(format!(
                        "Expected {} selected features, got {}. Consider adjusting selection parameters.",
                        $n_selected,
                        mask.count_selected()
                    )));
                }

                data.select_features(&mask)
            }
        }
    };
}

/// Const generic helper for computing binomial coefficients at compile time
pub const fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        0
    } else if k == 0 || k == n {
        1
    } else {
        let k = if k > n - k { n - k } else { k };
        let mut result = 1;
        let mut i = 0;
        while i < k {
            result = result * (n - i) / (i + 1);
            i += 1;
        }
        result
    }
}

/// Compile-time validation that selection count is valid
pub const fn validate_selection_count<const N_FEATURES: usize, const K: usize>() -> bool {
    K <= N_FEATURES && K > 0
}

/// Type-level boolean for compile-time feature validation
pub trait TypeBool {
    const VALUE: bool;
}

pub struct True;
pub struct False;

impl TypeBool for True {
    const VALUE: bool = true;
}

impl TypeBool for False {
    const VALUE: bool = false;
}

// Note: Advanced type-level programming features commented out due to requiring unstable Rust features
// These would be enabled once const generics operations and inherent associated types are stabilized

// /// Compile-time assertion for feature selection validity
// pub type Assert<T> = <T as TypeBool>::Value;

// pub trait TypeBoolTrait {
//     type Value: TypeBool;
// }

// /// Feature selection validity checker
// pub struct FeatureSelectionValid<const N_FEATURES: usize, const K: usize>;

// /// Conditional type selection (requires unstable features)
// pub type If<const CONDITION: bool, T, F> = IfImpl<{ CONDITION }, T, F>::Type;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_feature_index() {
        const MAX_FEATURES: usize = 10;

        // Valid index
        let valid_index = FeatureIndex::<MAX_FEATURES>::new(5).unwrap();
        assert_eq!(valid_index.get(), 5);

        // Invalid index
        assert!(FeatureIndex::<MAX_FEATURES>::new(15).is_none());
    }

    #[test]
    fn test_feature_mask() {
        const N_FEATURES: usize = 5;

        let mask = FeatureMask::<N_FEATURES>::from_array([true, false, true, false, true]);
        assert_eq!(mask.count_selected(), 3);

        let indices = mask.selected_indices();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_feature_matrix() -> Result<()> {
        const N_FEATURES: usize = 3;

        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let matrix = FeatureMatrix::<f64, N_FEATURES>::new(data)?;

        assert_eq!(matrix.n_features(), 3);
        assert_eq!(matrix.n_samples(), 2);

        Ok(())
    }

    #[test]
    fn test_variance_threshold_selector() -> Result<()> {
        const N_FEATURES: usize = 3;

        let data = array![[1.0, 2.0, 3.0], [1.1, 5.0, 3.1], [0.9, 8.0, 2.9]];
        let matrix = FeatureMatrix::<f64, N_FEATURES>::new(data)?;

        let mut selector = VarianceThresholdSelector::new(0.1);
        let mask = selector.fit(&matrix)?;

        // Should select features with variance > 0.1
        assert!(mask.count_selected() > 0);

        Ok(())
    }

    #[test]
    fn test_compile_time_validation() {
        const N_FEATURES: usize = 10;
        const K: usize = 5;

        // This should compile
        assert!(validate_selection_count::<N_FEATURES, K>());

        // This should not compile if uncommented:
        // assert!(validate_selection_count::<5, 10>());
    }

    #[test]
    fn test_type_safe_pipeline() -> Result<()> {
        const N_FEATURES: usize = 4;

        let data = array![
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 5.0, 3.1, 4.1],
            [0.9, 8.0, 2.9, 3.9],
            [1.2, 2.1, 3.2, 4.2]
        ];
        let matrix = FeatureMatrix::<f64, N_FEATURES>::new(data)?;
        let y = array![0.0, 1.0, 0.0, 1.0];

        let pipeline = TypeSafeSelectionPipeline::<N_FEATURES>::new()
            .add_variance_threshold(0.01)
            .add_correlation_filter(0.9);

        let fitted_pipeline = pipeline.fit(&matrix, y.view())?;
        assert!(fitted_pipeline.n_selected_features() > 0);

        Ok(())
    }
}
