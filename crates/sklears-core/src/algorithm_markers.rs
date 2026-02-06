/// Marker traits for algorithm categorization and type safety
///
/// This module provides marker traits that categorize different types of
/// machine learning algorithms. These traits enable type-safe algorithm
/// composition, better API design, and compile-time verification of
/// algorithm compatibility.
use std::marker::PhantomData;

/// Marker trait for supervised learning algorithms
///
/// Algorithms implementing this trait learn from input-output pairs
/// and can make predictions on new data.
pub trait SupervisedLearning {}

/// Marker trait for unsupervised learning algorithms  
///
/// Algorithms implementing this trait learn patterns from input data
/// without explicit output targets.
pub trait UnsupervisedLearning {}

/// Marker trait for semi-supervised learning algorithms
///
/// Algorithms implementing this trait can learn from both labeled
/// and unlabeled data.
pub trait SemiSupervisedLearning {}

/// Marker trait for reinforcement learning algorithms
///
/// Algorithms implementing this trait learn through interaction
/// with an environment via rewards and penalties.
pub trait ReinforcementLearning {}

/// Marker trait for online learning algorithms
///
/// Algorithms implementing this trait can incrementally update
/// their model as new data arrives.
pub trait OnlineLearning {}

/// Marker trait for batch learning algorithms
///
/// Algorithms implementing this trait require the entire dataset
/// to be available for training.
pub trait BatchLearning {}

/// Marker trait for streaming algorithms
///
/// Algorithms implementing this trait can process data streams
/// in real-time with bounded memory usage.
pub trait StreamingLearning {}

// =============================================================================
// Algorithm Category Markers
// =============================================================================

/// Marker trait for linear model algorithms
///
/// Linear models make predictions using linear combinations of input features.
/// Examples: Linear Regression, Logistic Regression, SVM with linear kernel
#[cfg(feature = "linear_models")]
pub trait LinearModel: SupervisedLearning {}

/// Marker trait for tree-based algorithms
///
/// Tree-based algorithms use decision trees for prediction.
/// Examples: Decision Trees, Random Forest, Gradient Boosting
#[cfg(feature = "tree_models")]
pub trait TreeBased {}

/// Marker trait for neural network algorithms
///
/// Neural network algorithms use artificial neural networks for learning.
/// Examples: Multi-layer Perceptron, Convolutional Neural Networks
#[cfg(feature = "neural_networks")]
pub trait NeuralNetwork: SupervisedLearning {}

/// Marker trait for clustering algorithms
///
/// Clustering algorithms group similar data points together.
/// Examples: K-Means, DBSCAN, Hierarchical Clustering
#[cfg(feature = "clustering")]
pub trait Clustering: UnsupervisedLearning {}

/// Marker trait for dimensionality reduction algorithms
///
/// These algorithms reduce the number of features while preserving
/// important information.
/// Examples: PCA, t-SNE, UMAP
#[cfg(feature = "dimensionality_reduction")]
pub trait DimensionalityReduction: UnsupervisedLearning {}

/// Marker trait for ensemble methods
///
/// Ensemble methods combine multiple models to improve prediction accuracy.
/// Examples: Random Forest, AdaBoost, Voting Classifier
#[cfg(feature = "ensemble_methods")]
pub trait EnsembleMethod {}

// =============================================================================
// Problem Type Markers
// =============================================================================

/// Marker trait for classification algorithms
///
/// Classification algorithms predict discrete class labels.
pub trait Classification: SupervisedLearning {}

/// Marker trait for regression algorithms
///
/// Regression algorithms predict continuous numerical values.
pub trait Regression: SupervisedLearning {}

/// Marker trait for ranking algorithms
///
/// Ranking algorithms order items based on relevance or preference.
pub trait Ranking: SupervisedLearning {}

/// Marker trait for anomaly detection algorithms
///
/// Anomaly detection algorithms identify outliers or unusual patterns.
pub trait AnomalyDetection {}

/// Marker trait for density estimation algorithms
///
/// Density estimation algorithms estimate probability distributions.
pub trait DensityEstimation: UnsupervisedLearning {}

/// Marker trait for feature selection algorithms
///
/// Feature selection algorithms choose the most relevant features.
pub trait FeatureSelection {}

// =============================================================================
// Model Characteristics Markers
// =============================================================================

/// Marker trait for parametric models
///
/// Parametric models have a fixed number of parameters independent
/// of the training set size.
pub trait Parametric {}

/// Marker trait for non-parametric models
///
/// Non-parametric models have a number of parameters that grows
/// with the training set size.
pub trait NonParametric {}

/// Marker trait for probabilistic models
///
/// Probabilistic models output probability distributions rather
/// than point predictions.
pub trait Probabilistic {}

/// Marker trait for deterministic models
///
/// Deterministic models produce the same output given the same input.
pub trait Deterministic {}

/// Marker trait for interpretable models
///
/// Interpretable models provide insights into their decision-making process.
pub trait Interpretable {}

/// Marker trait for black-box models
///
/// Black-box models provide predictions without explainable reasoning.
pub trait BlackBox {}

/// Marker trait for models that support incremental learning
///
/// These models can update their parameters incrementally as new data arrives.
pub trait Incremental {}

/// Marker trait for models that require feature scaling
///
/// These models are sensitive to the scale of input features.
pub trait ScaleSensitive {}

/// Marker trait for models that are robust to outliers
///
/// These models perform well even in the presence of outliers.
pub trait OutlierRobust {}

/// Marker trait for models that handle missing values natively
///
/// These models can work with incomplete data without preprocessing.
pub trait MissingValueTolerant {}

// =============================================================================
// Computational Characteristics Markers
// =============================================================================

/// Marker trait for algorithms that support parallel training
///
/// These algorithms can utilize multiple CPU cores during training.
pub trait ParallelTraining {}

/// Marker trait for algorithms that support parallel prediction
///
/// These algorithms can make predictions in parallel.
pub trait ParallelPrediction {}

/// Marker trait for algorithms optimized for GPU computation
///
/// These algorithms can leverage GPU acceleration.
#[cfg(feature = "gpu_support")]
pub trait GpuAccelerated {}

/// Marker trait for algorithms that support distributed computing
///
/// These algorithms can run across multiple machines.
#[cfg(feature = "distributed")]
pub trait Distributed {}

/// Marker trait for memory-efficient algorithms
///
/// These algorithms have bounded memory usage regardless of dataset size.
pub trait MemoryEfficient {}

/// Marker trait for algorithms with fast training
///
/// These algorithms have sub-quadratic training time complexity.
pub trait FastTraining {}

/// Marker trait for algorithms with fast prediction
///
/// These algorithms have constant or logarithmic prediction time.
pub trait FastPrediction {}

// =============================================================================
// Data Type Markers
// =============================================================================

/// Marker trait for algorithms that work with numerical data
pub trait NumericalData {}

/// Marker trait for algorithms that work with categorical data
pub trait CategoricalData {}

/// Marker trait for algorithms that work with text data
pub trait TextData {}

/// Marker trait for algorithms that work with image data
pub trait ImageData {}

/// Marker trait for algorithms that work with time series data
pub trait TimeSeriesData {}

/// Marker trait for algorithms that work with graph data
pub trait GraphData {}

/// Marker trait for algorithms that work with sparse data
pub trait SparseData {}

/// Marker trait for algorithms that work with high-dimensional data
pub trait HighDimensionalData {}

// =============================================================================
// Combination and Utility Traits
// =============================================================================

/// Composite trait for algorithms that are both fast and memory-efficient
pub trait FastAndEfficient: FastTraining + FastPrediction + MemoryEfficient {}

/// Composite trait for algorithms suitable for real-time applications
pub trait RealTime: FastPrediction + StreamingLearning + MemoryEfficient {}

/// Composite trait for algorithms suitable for big data
pub trait BigData: ParallelTraining + MemoryEfficient + StreamingLearning {}

/// Composite trait for robust algorithms
pub trait Robust: OutlierRobust + MissingValueTolerant {}

/// Composite trait for explainable AI algorithms
pub trait ExplainableAI: Interpretable + Probabilistic {}

/// Algorithm complexity categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComplexityClass {
    /// Constant time complexity O(1)
    Constant,
    /// Logarithmic time complexity O(log n)
    Logarithmic,
    /// Linear time complexity O(n)
    Linear,
    /// Linearithmic time complexity O(n log n)
    Linearithmic,
    /// Quadratic time complexity O(n²)
    Quadratic,
    /// Cubic time complexity O(n³)
    Cubic,
    /// Exponential time complexity O(2^n)
    Exponential,
}

/// Trait for algorithms with known complexity characteristics
pub trait ComplexityBounds {
    /// Training time complexity
    fn training_complexity() -> ComplexityClass;
    /// Prediction time complexity  
    fn prediction_complexity() -> ComplexityClass;
    /// Space complexity
    fn space_complexity() -> ComplexityClass;
}

/// Algorithm stability categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityClass {
    /// Algorithm is stable to small changes in training data
    Stable,
    /// Algorithm is somewhat sensitive to training data changes
    ModeratelySensitive,
    /// Algorithm is highly sensitive to training data changes
    Unstable,
}

/// Trait for algorithms with known stability characteristics
pub trait StabilityBounds {
    /// Stability with respect to training data perturbations
    fn data_stability() -> StabilityClass;
    /// Stability with respect to hyperparameter changes
    fn hyperparameter_stability() -> StabilityClass;
}

/// Type-level algorithm category system
pub mod category_system {
    use super::*;

    /// Type-level representation of algorithm categories
    pub struct AlgorithmCategory<Learning, Problem, Model, Compute, Data> {
        _phantom: PhantomData<(Learning, Problem, Model, Compute, Data)>,
    }

    // Learning paradigm types
    pub struct Supervised;
    pub struct Unsupervised;
    pub struct SemiSupervised;
    pub struct Reinforcement;

    // Problem type types
    pub struct ClassificationProblem;
    pub struct RegressionProblem;
    pub struct ClusteringProblem;
    pub struct DimensionalityReductionProblem;

    // Model characteristic types
    pub struct ParametricModel;
    pub struct NonParametricModel;
    pub struct ProbabilisticModel;
    pub struct DeterministicModel;

    // Computational characteristic types
    pub struct ParallelCompute;
    pub struct SequentialCompute;
    pub struct GpuCompute;
    pub struct CpuCompute;

    // Data type types
    pub struct NumericalDataType;
    pub struct CategoricalDataType;
    pub struct MixedDataType;

    impl<L, P, M, C, D> AlgorithmCategory<L, P, M, C, D> {
        pub fn new() -> Self {
            Self {
                _phantom: PhantomData,
            }
        }
    }

    impl<L, P, M, C, D> Default for AlgorithmCategory<L, P, M, C, D> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Type alias for common algorithm categories
    pub type SupervisedClassifier = AlgorithmCategory<
        Supervised,
        ClassificationProblem,
        ParametricModel,
        ParallelCompute,
        NumericalDataType,
    >;

    pub type SupervisedRegressor = AlgorithmCategory<
        Supervised,
        RegressionProblem,
        ParametricModel,
        ParallelCompute,
        NumericalDataType,
    >;

    pub type UnsupervisedClusterer = AlgorithmCategory<
        Unsupervised,
        ClusteringProblem,
        NonParametricModel,
        ParallelCompute,
        NumericalDataType,
    >;
}

/// Macro for implementing multiple marker traits at once
#[macro_export]
macro_rules! impl_algorithm_markers {
    ($algorithm:ty: $($trait:path),+ $(,)?) => {
        $(
            impl $trait for $algorithm {}
        )+
    };
}

/// Macro for defining algorithm categories with compile-time checking
#[macro_export]
macro_rules! define_algorithm_category {
    (
        $name:ident:
        learning = $learning:ty,
        problem = $problem:ty,
        model = $model:ty,
        compute = $compute:ty,
        data = $data:ty,
    ) => {
        pub type $name = $crate::algorithm_markers::category_system::AlgorithmCategory<
            $learning,
            $problem,
            $model,
            $compute,
            $data,
        >;
    };
    (
        $name:ident:
        learning = $learning:ty,
        problem = $problem:ty,
        model = $model:ty,
    ) => {
        pub type $name = $crate::algorithm_markers::category_system::AlgorithmCategory<
            $learning,
            $problem,
            $model,
            $crate::algorithm_markers::category_system::CpuCompute,
            $crate::algorithm_markers::category_system::NumericalDataType,
        >;
    };
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    // Example algorithm for testing
    struct MockLinearRegression;

    // Implement marker traits for the mock algorithm
    impl_algorithm_markers!(
        MockLinearRegression:
        SupervisedLearning,
        Regression,
        Parametric,
        Deterministic,
        FastTraining,
        FastPrediction,
        ScaleSensitive,
        NumericalData,
        ParallelTraining,
        ParallelPrediction
    );

    #[cfg(feature = "linear_models")]
    impl LinearModel for MockLinearRegression {}

    impl ComplexityBounds for MockLinearRegression {
        fn training_complexity() -> ComplexityClass {
            ComplexityClass::Cubic // For normal equation: O(n³)
        }

        fn prediction_complexity() -> ComplexityClass {
            ComplexityClass::Linear // O(n) for matrix-vector multiplication
        }

        fn space_complexity() -> ComplexityClass {
            ComplexityClass::Quadratic // O(n²) for storing the design matrix
        }
    }

    impl StabilityBounds for MockLinearRegression {
        fn data_stability() -> StabilityClass {
            StabilityClass::Stable
        }

        fn hyperparameter_stability() -> StabilityClass {
            StabilityClass::Stable
        }
    }

    #[test]
    fn test_marker_traits() {
        fn is_supervised<T: SupervisedLearning>() -> bool {
            let _ = std::marker::PhantomData::<T>;
            true
        }
        fn is_regression<T: Regression>() -> bool {
            let _ = std::marker::PhantomData::<T>;
            true
        }
        fn is_parametric<T: Parametric>() -> bool {
            let _ = std::marker::PhantomData::<T>;
            true
        }

        assert!(is_supervised::<MockLinearRegression>());
        assert!(is_regression::<MockLinearRegression>());
        assert!(is_parametric::<MockLinearRegression>());
    }

    #[test]
    fn test_complexity_bounds() {
        assert_eq!(
            MockLinearRegression::training_complexity(),
            ComplexityClass::Cubic
        );
        assert_eq!(
            MockLinearRegression::prediction_complexity(),
            ComplexityClass::Linear
        );
        assert_eq!(
            MockLinearRegression::space_complexity(),
            ComplexityClass::Quadratic
        );
    }

    #[test]
    fn test_stability_bounds() {
        assert_eq!(
            MockLinearRegression::data_stability(),
            StabilityClass::Stable
        );
        assert_eq!(
            MockLinearRegression::hyperparameter_stability(),
            StabilityClass::Stable
        );
    }

    #[test]
    fn test_complexity_ordering() {
        assert!(ComplexityClass::Constant < ComplexityClass::Linear);
        assert!(ComplexityClass::Linear < ComplexityClass::Quadratic);
        assert!(ComplexityClass::Quadratic < ComplexityClass::Exponential);
    }

    #[test]
    fn test_category_system() {
        let _classifier = category_system::SupervisedClassifier::new();
        let _regressor = category_system::SupervisedRegressor::new();
        let _clusterer = category_system::UnsupervisedClusterer::new();
    }

    #[test]
    fn test_define_algorithm_category_macro() {
        // Test the macro for defining custom algorithm categories
        define_algorithm_category!(
            MyCustomAlgorithm:
            learning = category_system::Supervised,
            problem = category_system::ClassificationProblem,
            model = category_system::ParametricModel,
            compute = category_system::ParallelCompute,
            data = category_system::NumericalDataType,
        );

        let _my_algorithm = MyCustomAlgorithm::new();
    }

    // Test composite traits
    struct MockRandomForest;

    impl_algorithm_markers!(
        MockRandomForest:
        SupervisedLearning,
        Classification,
        NonParametric,
        OutlierRobust,
        MissingValueTolerant,
        ParallelTraining,
        ParallelPrediction,
        MemoryEfficient,
        StreamingLearning
    );

    #[cfg(feature = "tree_models")]
    impl TreeBased for MockRandomForest {}

    #[cfg(feature = "ensemble_methods")]
    impl EnsembleMethod for MockRandomForest {}

    impl Robust for MockRandomForest {}
    impl BigData for MockRandomForest {}

    #[test]
    fn test_composite_traits() {
        fn is_robust<T: Robust>() -> bool {
            let _ = std::marker::PhantomData::<T>;
            true
        }
        fn is_big_data<T: BigData>() -> bool {
            let _ = std::marker::PhantomData::<T>;
            true
        }

        assert!(is_robust::<MockRandomForest>());
        assert!(is_big_data::<MockRandomForest>());
    }

    // Test that algorithms can be categorized correctly
    #[test]
    fn test_algorithm_categorization() {
        // Function that only accepts supervised learning algorithms
        fn train_supervised<T: SupervisedLearning>(_algorithm: T) {
            // Training logic would go here
        }

        // Function that only accepts ensemble methods
        #[cfg(feature = "ensemble_methods")]
        fn ensemble_predict<T: EnsembleMethod>(_algorithm: T) {
            // Ensemble prediction logic would go here
        }

        // These should compile without issues
        train_supervised(MockLinearRegression);
        train_supervised(MockRandomForest);

        #[cfg(feature = "ensemble_methods")]
        ensemble_predict(MockRandomForest);

        // This would not compile (MockLinearRegression is not an EnsembleMethod):
        // ensemble_predict(MockLinearRegression);
    }
}
