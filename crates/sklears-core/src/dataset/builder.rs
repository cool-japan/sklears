/// Dataset builder pattern with compile-time validation
///
/// This module provides a type-safe builder pattern for constructing datasets.
/// The builder uses phantom types to ensure that both data and targets are
/// provided before the dataset can be built, catching errors at compile time.
use crate::dataset::core::Dataset;

/// Marker type indicating data has not been set in the builder
#[derive(Debug)]
pub struct NoData;

/// Marker type indicating data has been set in the builder
#[derive(Debug)]
pub struct HasData;

/// Marker type indicating target has not been set in the builder
#[derive(Debug)]
pub struct NoTarget;

/// Marker type indicating target has been set in the builder
#[derive(Debug)]
pub struct HasTarget;

/// Type-safe builder for Dataset construction with compile-time validation
///
/// The DatasetBuilder uses phantom types to track whether data and targets
/// have been set, preventing incomplete datasets from being constructed.
/// This provides compile-time safety without runtime overhead.
///
/// # Type Parameters
///
/// - `X`: Type of the feature data
/// - `Y`: Type of the target data
/// - `DataState`: Phantom type tracking data state (NoData/HasData)
/// - `TargetState`: Phantom type tracking target state (NoTarget/HasTarget)
///
/// # Examples
///
/// ```rust
/// use sklears_core::dataset::Dataset;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let features = Array2::`<f64>`::zeros((100, 4));
/// let targets = Array1::`<f64>`::zeros(100);
///
/// let dataset = Dataset::builder()
///     .data(features)
///     .target(targets)
///     .description("My dataset".to_string())
///     .feature_names(vec!["f1".to_string(), "f2".to_string()])
///     .build();
/// ```
#[derive(Debug)]
pub struct DatasetBuilder<X, Y, DataState, TargetState> {
    data: Option<X>,
    target: Option<Y>,
    feature_names: Vec<String>,
    target_names: Option<Vec<String>>,
    description: String,
    _phantom_data: std::marker::PhantomData<DataState>,
    _phantom_target: std::marker::PhantomData<TargetState>,
}

impl<X, Y> DatasetBuilder<X, Y, NoData, NoTarget> {
    /// Create a new dataset builder
    ///
    /// The builder starts in the initial state where neither data nor
    /// targets have been set. Both must be provided before build() can be called.
    ///
    /// # Returns
    ///
    /// A new DatasetBuilder in the initial (NoData, NoTarget) state
    pub fn new() -> Self {
        Self {
            data: None,
            target: None,
            feature_names: Vec::new(),
            target_names: None,
            description: String::new(),
            _phantom_data: std::marker::PhantomData,
            _phantom_target: std::marker::PhantomData,
        }
    }
}

impl<X, Y, TargetState> DatasetBuilder<X, Y, NoData, TargetState> {
    /// Set the feature data (required)
    ///
    /// This method transitions the builder from NoData to HasData state,
    /// bringing us closer to being able to build the dataset.
    ///
    /// # Arguments
    ///
    /// * `data` - The feature matrix or data structure
    ///
    /// # Returns
    ///
    /// DatasetBuilder with data set (HasData state)
    pub fn data(self, data: X) -> DatasetBuilder<X, Y, HasData, TargetState> {
        DatasetBuilder {
            data: Some(data),
            target: self.target,
            feature_names: self.feature_names,
            target_names: self.target_names,
            description: self.description,
            _phantom_data: std::marker::PhantomData,
            _phantom_target: std::marker::PhantomData,
        }
    }
}

impl<X, Y, DataState> DatasetBuilder<X, Y, DataState, NoTarget> {
    /// Set the target data (required)
    ///
    /// This method transitions the builder from NoTarget to HasTarget state,
    /// bringing us closer to being able to build the dataset.
    ///
    /// # Arguments
    ///
    /// * `target` - The target values corresponding to the features
    ///
    /// # Returns
    ///
    /// DatasetBuilder with target set (HasTarget state)
    pub fn target(self, target: Y) -> DatasetBuilder<X, Y, DataState, HasTarget> {
        DatasetBuilder {
            data: self.data,
            target: Some(target),
            feature_names: self.feature_names,
            target_names: self.target_names,
            description: self.description,
            _phantom_data: std::marker::PhantomData,
            _phantom_target: std::marker::PhantomData,
        }
    }
}

impl<X, Y, DataState, TargetState> DatasetBuilder<X, Y, DataState, TargetState> {
    /// Set feature names (optional)
    ///
    /// Feature names improve interpretability and are used in various
    /// visualization and analysis tools. This is an optional step.
    ///
    /// # Arguments
    ///
    /// * `names` - Vector of feature names
    ///
    /// # Returns
    ///
    /// Self with updated feature names
    pub fn feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Set target names (optional)
    ///
    /// Target names are particularly useful for multi-class classification
    /// where class labels need to be interpretable. This is an optional step.
    ///
    /// # Arguments
    ///
    /// * `names` - Vector of class/target names
    ///
    /// # Returns
    ///
    /// Self with updated target names
    pub fn target_names(mut self, names: Vec<String>) -> Self {
        self.target_names = Some(names);
        self
    }

    /// Set dataset description (optional)
    ///
    /// Descriptions are useful for documenting the source, preprocessing
    /// steps, or other relevant information. This is an optional step.
    ///
    /// # Arguments
    ///
    /// * `description` - String description of the dataset
    ///
    /// # Returns
    ///
    /// Self with updated description
    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = description.into();
        self
    }
}

impl<X, Y> DatasetBuilder<X, Y, HasData, HasTarget> {
    /// Build the final dataset
    ///
    /// This method is only available when both data and target have been set,
    /// ensuring compile-time safety. The unwrap() calls are safe because
    /// the type system guarantees the values are present.
    ///
    /// # Returns
    ///
    /// A completed Dataset instance
    pub fn build(self) -> Dataset<X, Y> {
        Dataset {
            data: self.data.unwrap(),     // Safe: HasData state guarantees this exists
            target: self.target.unwrap(), // Safe: HasTarget state guarantees this exists
            feature_names: self.feature_names,
            target_names: self.target_names,
            description: self.description,
        }
    }
}

/// Default implementation for the initial builder state
impl<X, Y> Default for DatasetBuilder<X, Y, NoData, NoTarget> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Array1, Array2};

    #[test]
    fn test_builder_pattern() {
        let data = Array2::<f64>::zeros((10, 3));
        let target = Array1::<f64>::zeros(10);

        let dataset = DatasetBuilder::new()
            .data(data)
            .target(target)
            .description("Test dataset")
            .feature_names(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()])
            .build();

        assert_eq!(dataset.description, "Test dataset");
        assert_eq!(dataset.feature_names.len(), 3);
    }

    #[test]
    fn test_builder_order_independence() {
        let data = Array2::<f64>::zeros((5, 2));
        let target = Array1::<f64>::zeros(5);

        // Test that data and target can be set in any order
        let dataset1 = DatasetBuilder::new()
            .data(data.clone())
            .target(target.clone())
            .build();

        let dataset2 = DatasetBuilder::new().target(target).data(data).build();

        // Both should have the same structure
        assert_eq!(dataset1.data.dim(), dataset2.data.dim());
        assert_eq!(dataset1.target.len(), dataset2.target.len());
    }

    #[test]
    fn test_builder_with_all_metadata() {
        let data = Array2::<f64>::ones((3, 2));
        let target = Array1::<f64>::ones(3);

        let dataset = DatasetBuilder::new()
            .data(data)
            .target(target)
            .feature_names(vec!["feature1".to_string(), "feature2".to_string()])
            .target_names(vec!["class_a".to_string(), "class_b".to_string()])
            .description("Complete dataset example")
            .build();

        assert_eq!(dataset.feature_names.len(), 2);
        assert!(dataset.target_names.is_some());
        assert_eq!(dataset.target_names.as_ref().unwrap().len(), 2);
        assert_eq!(dataset.description, "Complete dataset example");
    }
}
