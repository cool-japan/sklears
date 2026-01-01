/// Core Dataset structure and fundamental operations
///
/// This module contains the primary Dataset structure and its basic methods.
use crate::types::{Array1, Array2, Float};

/// A simple dataset structure for machine learning operations
///
/// The Dataset struct is the primary data container for sklears, holding
/// feature matrices and target values along with metadata.
///
/// # Type Parameters
///
/// - `X`: Type of the feature matrix (defaults to `Array2`<Float>``)
/// - `Y`: Type of the target values (defaults to `Array1`<Float>``)
///
/// # Examples
///
/// ```rust
/// use sklears_core::dataset::Dataset;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let features = Array2::`<f64>`::zeros((100, 4));
/// let targets = Array1::`<f64>`::zeros(100);
/// let dataset = Dataset::new(features, targets)
///     .with_description("Sample dataset".to_string());
/// ```
#[derive(Debug, Clone)]
pub struct Dataset<X = Array2<Float>, Y = Array1<Float>> {
    /// Feature matrix (n_samples x n_features)
    pub data: X,
    /// Target values (n_samples,)
    pub target: Y,
    /// Feature names for interpretability
    pub feature_names: Vec<String>,
    /// Target names (for classification tasks)
    pub target_names: Option<Vec<String>>,
    /// Dataset description for documentation
    pub description: String,
}

impl<X, Y> Dataset<X, Y> {
    /// Create a new dataset with the given data and target
    ///
    /// This is the primary constructor for creating datasets. Additional
    /// metadata can be added using builder methods.
    ///
    /// # Arguments
    ///
    /// * `data` - Feature matrix or data structure
    /// * `target` - Target values corresponding to the features
    ///
    /// # Returns
    ///
    /// A new Dataset instance with empty metadata
    pub fn new(data: X, target: Y) -> Self {
        Self {
            data,
            target,
            feature_names: Vec::new(),
            target_names: None,
            description: String::new(),
        }
    }

    /// Create a builder for constructing a dataset with compile-time validation
    ///
    /// The builder pattern provides compile-time guarantees that both data
    /// and targets are provided before the dataset can be constructed.
    ///
    /// # Returns
    ///
    /// A DatasetBuilder in its initial state
    pub fn builder() -> crate::dataset::builder::DatasetBuilder<
        X,
        Y,
        crate::dataset::builder::NoData,
        crate::dataset::builder::NoTarget,
    > {
        crate::dataset::builder::DatasetBuilder::new()
    }

    /// Set feature names for the dataset
    ///
    /// Feature names improve interpretability and are used in various
    /// visualization and analysis tools.
    ///
    /// # Arguments
    ///
    /// * `names` - Vector of feature names
    ///
    /// # Returns
    ///
    /// Self with updated feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Set target names for classification tasks
    ///
    /// Target names are particularly useful for multi-class classification
    /// where class labels need to be interpretable.
    ///
    /// # Arguments
    ///
    /// * `names` - Vector of class/target names
    ///
    /// # Returns
    ///
    /// Self with updated target names
    pub fn with_target_names(mut self, names: Vec<String>) -> Self {
        self.target_names = Some(names);
        self
    }

    /// Set a description for the dataset
    ///
    /// Descriptions are useful for documenting the source, preprocessing
    /// steps, or other relevant information about the dataset.
    ///
    /// # Arguments
    ///
    /// * `description` - String description of the dataset
    ///
    /// # Returns
    ///
    /// Self with updated description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Get the number of samples in the dataset
    ///
    /// This is a convenience method that should be implemented by
    /// types that can determine their sample count.
    pub fn n_samples(&self) -> Option<usize>
    where
        X: HasShape,
    {
        self.data.shape().map(|(n_samples, _)| n_samples)
    }

    /// Get the number of features in the dataset
    ///
    /// This is a convenience method that should be implemented by
    /// types that can determine their feature count.
    pub fn n_features(&self) -> Option<usize>
    where
        X: HasShape,
    {
        self.data.shape().map(|(_, n_features)| n_features)
    }
}

/// Trait for types that can provide shape information
///
/// This trait allows the Dataset to work with different backend types
/// while still providing shape information when available.
pub trait HasShape {
    /// Get the shape as (n_samples, n_features) if available
    fn shape(&self) -> Option<(usize, usize)>;
}

/// Implementation for ndarray Array2
impl HasShape for Array2<Float> {
    fn shape(&self) -> Option<(usize, usize)> {
        let dim = self.dim();
        Some((dim.0, dim.1))
    }
}

// /// Implementation for generic ndarray types (commented out due to conflicting implementations)
// impl<T, S> HasShape for ndarray::ArrayBase<S, ndarray::Ix2>
// where
//     S: ndarray::Data<Elem = T>,
// {
//     fn shape(&self) -> Option<(usize, usize)> {
//         let dim = self.dim();
//         Some((dim.0, dim.1))
//     }
// }

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_dataset_creation() {
        let data = Array2::<f64>::zeros((10, 3));
        let target = Array1::<f64>::zeros(10);

        let dataset = Dataset::new(data, target)
            .with_description("Test dataset".to_string())
            .with_feature_names(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()]);

        assert_eq!(dataset.description, "Test dataset");
        assert_eq!(dataset.feature_names.len(), 3);
        assert_eq!(dataset.n_samples(), Some(10));
        assert_eq!(dataset.n_features(), Some(3));
    }

    #[test]
    fn test_dataset_with_target_names() {
        let data = Array2::<f64>::zeros((5, 2));
        let target = Array1::<f64>::zeros(5);

        let dataset = Dataset::new(data, target)
            .with_target_names(vec!["class_a".to_string(), "class_b".to_string()]);

        assert!(dataset.target_names.is_some());
        assert_eq!(dataset.target_names.as_ref().unwrap().len(), 2);
    }
}
