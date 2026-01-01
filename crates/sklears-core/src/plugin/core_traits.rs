//! Core Plugin Traits
//!
//! This module defines the fundamental traits that all plugins must implement,
//! providing the interface for different types of machine learning algorithms
//! and transformations in the sklears plugin system.

use crate::error::Result;
use crate::traits::{Predict, Transform};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;

// Re-export types that will be used by traits
use super::types_config::{PluginConfig, PluginMetadata, PluginParameter};

/// Core trait that all plugins must implement
///
/// This trait defines the fundamental interface that every plugin in the sklears
/// ecosystem must provide. It ensures consistency across all plugin types and
/// enables the plugin system to manage plugins in a type-safe manner.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::{Plugin, PluginMetadata, PluginConfig};
/// use sklears_core::error::Result;
/// use std::any::{Any, TypeId};
///
/// #[derive(Debug)]
/// struct MyPlugin {
///     name: String,
/// }
///
/// impl Plugin for MyPlugin {
///     fn id(&self) -> &str {
///         &self.name
///     }
///
///     fn metadata(&self) -> PluginMetadata {
///         PluginMetadata::default()
///     }
///
///     fn initialize(&mut self, _config: &PluginConfig) -> Result<()> {
///         Ok(())
///     }
///
///     fn is_compatible(&self, _input_type: TypeId) -> bool {
///         true
///     }
///
///     fn as_any(&self) -> &dyn Any {
///         self
///     }
///
///     fn as_any_mut(&mut self) -> &mut dyn Any {
///         self
///     }
///
///     fn validate_config(&self, _config: &PluginConfig) -> Result<()> {
///         Ok(())
///     }
///
///     fn cleanup(&mut self) -> Result<()> {
///         Ok(())
///     }
/// }
/// ```
pub trait Plugin: Send + Sync + Debug {
    /// Unique identifier for the plugin
    ///
    /// This should be a unique string that identifies the plugin within
    /// the system. It's used for plugin discovery and registration.
    fn id(&self) -> &str;

    /// Plugin metadata
    ///
    /// Returns comprehensive metadata about the plugin including name,
    /// version, description, capabilities, and dependencies.
    fn metadata(&self) -> PluginMetadata;

    /// Initialize the plugin with configuration
    ///
    /// This method is called when the plugin is loaded and provides
    /// the plugin with its configuration. The plugin should perform
    /// any necessary initialization here.
    fn initialize(&mut self, config: &PluginConfig) -> Result<()>;

    /// Check if the plugin is compatible with the given input type
    ///
    /// This method allows the plugin system to determine if a plugin
    /// can handle a particular data type before attempting to use it.
    fn is_compatible(&self, input_type: TypeId) -> bool;

    /// Get the plugin as Any for downcasting
    ///
    /// This enables type-safe downcasting to the concrete plugin type
    /// when needed for specialized operations.
    fn as_any(&self) -> &dyn Any;

    /// Get the plugin as mutable Any for downcasting
    ///
    /// This enables mutable access to the concrete plugin type
    /// when needed for specialized operations.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Validate plugin configuration
    ///
    /// This method should check if the provided configuration is valid
    /// for this plugin. It's called before initialization to catch
    /// configuration errors early.
    fn validate_config(&self, config: &PluginConfig) -> Result<()>;

    /// Cleanup resources when plugin is unloaded
    ///
    /// This method is called when the plugin is being unloaded and
    /// should clean up any resources the plugin has allocated.
    fn cleanup(&mut self) -> Result<()>;
}

/// Trait for algorithm plugins that can fit and predict
///
/// This trait defines the interface for machine learning algorithms that
/// follow the fit/predict pattern. It extends the base Plugin trait with
/// algorithm-specific functionality.
///
/// # Type Parameters
///
/// * `X` - The input feature type (e.g., `Array2<f64>`)
/// * `Y` - The target/label type (e.g., `Array1<f64>`)
/// * `Output` - The prediction output type
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::{Plugin, AlgorithmPlugin, PluginParameter};
/// use sklears_core::traits::Predict;
/// use sklears_core::error::Result;
/// use std::collections::HashMap;
///
/// #[derive(Debug)]
/// struct LinearRegression {
///     // algorithm parameters
/// }
///
/// struct FittedLinearRegression {
///     // fitted model state
/// }
///
/// impl Predict<Vec<f64>, Vec<f64>> for FittedLinearRegression {
///     fn predict(&self, x: &Vec<f64>) -> Result<Vec<f64>> {
///         // prediction implementation
///         Ok(x.clone())
///     }
/// }
///
/// impl AlgorithmPlugin<Vec<f64>, Vec<f64>, Vec<f64>> for LinearRegression {
///     type Fitted = FittedLinearRegression;
///
///     fn fit(&self, x: &Vec<f64>, y: &Vec<f64>) -> Result<Self::Fitted> {
///         Ok(FittedLinearRegression {})
///     }
///
///     fn predict(&self, fitted: &Self::Fitted, x: &Vec<f64>) -> Result<Vec<f64>> {
///         fitted.predict(x)
///     }
///
///     fn get_parameters(&self) -> HashMap<String, PluginParameter> {
///         HashMap::new()
///     }
///
///     fn set_parameters(&mut self, _params: HashMap<String, PluginParameter>) -> Result<()> {
///         Ok(())
///     }
/// }
/// ```
pub trait AlgorithmPlugin<X, Y, Output>: Plugin + Send + Sync {
    /// The fitted model type
    ///
    /// This type represents the state of the algorithm after training.
    /// It must implement the Predict trait to enable predictions.
    type Fitted: Predict<X, Output> + Send + Sync;

    /// Fit the algorithm to training data
    ///
    /// This method trains the algorithm on the provided data and returns
    /// a fitted model that can be used for predictions.
    ///
    /// # Arguments
    ///
    /// * `x` - Training features
    /// * `y` - Training targets/labels
    ///
    /// # Returns
    ///
    /// A fitted model instance or an error if training fails.
    fn fit(&self, x: &X, y: &Y) -> Result<Self::Fitted>;

    /// Make predictions using the fitted model
    ///
    /// This method uses the fitted model to make predictions on new data.
    ///
    /// # Arguments
    ///
    /// * `fitted` - The fitted model from the fit method
    /// * `x` - Input features for prediction
    ///
    /// # Returns
    ///
    /// Predictions or an error if prediction fails.
    fn predict(&self, fitted: &Self::Fitted, x: &X) -> Result<Output>;

    /// Get algorithm-specific parameters
    ///
    /// Returns a map of all configurable parameters for this algorithm.
    /// This enables introspection and parameter tuning.
    fn get_parameters(&self) -> HashMap<String, PluginParameter>;

    /// Set algorithm-specific parameters
    ///
    /// Allows updating the algorithm's parameters. The algorithm should
    /// validate that the provided parameters are valid.
    ///
    /// # Arguments
    ///
    /// * `params` - Map of parameter names to values
    ///
    /// # Returns
    ///
    /// Ok(()) if parameters were set successfully, or an error if
    /// any parameter is invalid.
    fn set_parameters(&mut self, params: HashMap<String, PluginParameter>) -> Result<()>;
}

/// Trait for transformer plugins
///
/// This trait defines the interface for data transformation algorithms
/// that can fit to data and then transform new data using the learned
/// transformation.
///
/// # Type Parameters
///
/// * `X` - The input data type
/// * `Output` - The transformed output type (defaults to X)
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::{Plugin, TransformerPlugin};
/// use sklears_core::traits::Transform;
/// use sklears_core::error::Result;
///
/// #[derive(Debug)]
/// struct StandardScaler {
///     // scaler parameters
/// }
///
/// struct FittedStandardScaler {
///     // fitted scaler state
/// }
///
/// impl Transform<Vec<f64>, Vec<f64>> for FittedStandardScaler {
///     fn transform(&self, x: &Vec<f64>) -> Result<Vec<f64>> {
///         // transformation implementation
///         Ok(x.clone())
///     }
/// }
///
/// impl TransformerPlugin<Vec<f64>, Vec<f64>> for StandardScaler {
///     type Fitted = FittedStandardScaler;
///
///     fn fit_transform(&self, x: &Vec<f64>) -> Result<(Self::Fitted, Vec<f64>)> {
///         let fitted = FittedStandardScaler {};
///         let transformed = fitted.transform(x)?;
///         Ok((fitted, transformed))
///     }
///
///     fn transform(&self, fitted: &Self::Fitted, x: &Vec<f64>) -> Result<Vec<f64>> {
///         fitted.transform(x)
///     }
/// }
/// ```
pub trait TransformerPlugin<X, Output = X>: Plugin + Send + Sync {
    /// The fitted transformer type
    ///
    /// This type represents the state of the transformer after fitting.
    /// It must implement the Transform trait to enable transformations.
    type Fitted: Transform<X, Output> + Send + Sync;

    /// Fit the transformer to data and return the fitted transformer along with transformed data
    ///
    /// This method fits the transformer to the input data and immediately
    /// applies the transformation, returning both the fitted transformer
    /// and the transformed data.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data to fit and transform
    ///
    /// # Returns
    ///
    /// A tuple of (fitted transformer, transformed data) or an error.
    fn fit_transform(&self, x: &X) -> Result<(Self::Fitted, Output)>;

    /// Transform data using the fitted transformer
    ///
    /// This method applies the previously fitted transformation to new data.
    ///
    /// # Arguments
    ///
    /// * `fitted` - The fitted transformer from fit_transform
    /// * `x` - Input data to transform
    ///
    /// # Returns
    ///
    /// Transformed data or an error if transformation fails.
    fn transform(&self, fitted: &Self::Fitted, x: &X) -> Result<Output>;
}

/// Trait for clustering plugins
///
/// This trait defines the interface for clustering algorithms that
/// can group data points into clusters and provide cluster information.
///
/// # Type Parameters
///
/// * `X` - The input data type
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_core::plugin::{Plugin, ClusteringPlugin};
/// use sklears_core::error::Result;
/// use std::collections::HashMap;
///
/// #[derive(Debug)]
/// struct KMeans {
///     n_clusters: usize,
/// }
///
/// impl ClusteringPlugin<Vec<Vec<f64>>> for KMeans {
///     type Labels = Vec<usize>;
///
///     fn fit_predict(&self, x: &Vec<Vec<f64>>) -> Result<Self::Labels> {
///         // clustering implementation
///         Ok(vec![0; x.len()])
///     }
///
///     fn cluster_centers(&self) -> Option<Vec<Vec<f64>>> {
///         // return cluster centers if available
///         None
///     }
///
///     fn cluster_stats(&self) -> HashMap<String, f64> {
///         // return clustering statistics
///         HashMap::new()
///     }
/// }
/// ```
pub trait ClusteringPlugin<X>: Plugin + Send + Sync {
    /// The cluster labels type
    ///
    /// This type represents the cluster assignments for each data point.
    /// Typically this would be `Vec<usize>` for integer cluster labels.
    type Labels;

    /// Fit the clustering algorithm and return cluster assignments
    ///
    /// This method performs clustering on the input data and returns
    /// the cluster assignments for each data point.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data to cluster
    ///
    /// # Returns
    ///
    /// Cluster labels for each data point or an error.
    fn fit_predict(&self, x: &X) -> Result<Self::Labels>;

    /// Get cluster centers (if applicable)
    ///
    /// For algorithms that compute explicit cluster centers (like K-means),
    /// this method returns the center points. Returns None if the algorithm
    /// doesn't compute explicit centers.
    fn cluster_centers(&self) -> Option<X>;

    /// Get cluster statistics
    ///
    /// Returns various statistics about the clustering result, such as
    /// inertia, silhouette score, number of clusters, etc.
    fn cluster_stats(&self) -> HashMap<String, f64>;
}
