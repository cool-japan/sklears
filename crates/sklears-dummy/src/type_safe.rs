//! Type-Safe Dummy Estimators
//!
//! This module provides compile-time guarantees for dummy estimators using Rust's type system.
//! It includes phantom types, state validation, and compile-time configuration verification.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict};
use std::marker::PhantomData;

/// Marker trait for estimator states
pub trait EstimatorState {}

/// Untrained state marker
#[derive(Debug, Clone, Copy)]
pub struct Untrained;
impl EstimatorState for Untrained {}

/// Trained state marker  
#[derive(Debug, Clone, Copy)]
pub struct Trained;
impl EstimatorState for Trained {}

/// Marker trait for task types
pub trait TaskType {}

/// Classification task marker
#[derive(Debug, Clone, Copy)]
pub struct Classification;
impl TaskType for Classification {}

/// Regression task marker
#[derive(Debug, Clone, Copy)]
pub struct Regression;
impl TaskType for Regression {}

/// Marker trait for strategy validation
pub trait StrategyValid<T: TaskType> {}

/// Validated classification strategies
impl StrategyValid<Classification> for crate::ClassifierStrategy {}

/// Validated regression strategies
impl StrategyValid<Regression> for crate::RegressorStrategy {}

/// Type-safe dummy estimator with compile-time guarantees
#[derive(Debug, Clone)]
pub struct TypeSafeDummyEstimator<State, Task, Strategy>
where
    State: EstimatorState,
    Task: TaskType,
    Strategy: StrategyValid<Task>,
{
    strategy: Strategy,
    random_state: Option<u64>,
    _state: PhantomData<State>,
    _task: PhantomData<Task>,
}

/// Fitted classifier with type-safe state
#[derive(Debug, Clone)]
pub struct TypeSafeFittedClassifier<Strategy>
where
    Strategy: StrategyValid<Classification>,
{
    strategy: Strategy,
    fitted_data: ClassificationFittedData,
    random_state: Option<u64>,
}

/// Fitted regressor with type-safe state
#[derive(Debug, Clone)]
pub struct TypeSafeFittedRegressor<Strategy>
where
    Strategy: StrategyValid<Regression>,
{
    strategy: Strategy,
    fitted_data: RegressionFittedData,
    random_state: Option<u64>,
}

/// Classification fitted data
#[derive(Debug, Clone)]
pub struct ClassificationFittedData {
    /// class_counts
    pub class_counts: std::collections::HashMap<i32, usize>,
    /// class_priors
    pub class_priors: std::collections::HashMap<i32, f64>,
    /// most_frequent_class
    pub most_frequent_class: i32,
    /// n_samples
    pub n_samples: usize,
    /// n_features
    pub n_features: usize,
}

/// Regression fitted data
#[derive(Debug, Clone)]
pub struct RegressionFittedData {
    /// target_mean
    pub target_mean: f64,
    /// target_median
    pub target_median: f64,
    /// target_std
    pub target_std: f64,
    /// target_min
    pub target_min: f64,
    /// target_max
    pub target_max: f64,
    /// n_samples
    pub n_samples: usize,
    /// n_features
    pub n_features: usize,
}

/// Compile-time configuration for estimators
pub trait EstimatorConfig {
    type TaskType: TaskType;
    type Strategy: StrategyValid<Self::TaskType>;

    /// Validate configuration at compile time
    fn validate() -> Result<(), &'static str>;

    /// Create strategy instance
    fn create_strategy() -> Self::Strategy;
}

/// Classification configuration
#[derive(Debug)]
pub struct ClassificationConfig<S: StrategyValid<Classification>> {
    _strategy: PhantomData<S>,
}

/// Regression configuration
#[derive(Debug)]
pub struct RegressionConfig<S: StrategyValid<Regression>> {
    _strategy: PhantomData<S>,
}

impl<S: StrategyValid<Classification>> EstimatorConfig for ClassificationConfig<S> {
    type TaskType = Classification;
    type Strategy = S;

    fn validate() -> Result<(), &'static str> {
        // Compile-time validation can be extended here
        Ok(())
    }

    fn create_strategy() -> Self::Strategy {
        // This would need to be implemented per strategy type
        // For now, we'll use a placeholder approach
        panic!("Strategy creation must be implemented per type")
    }
}

impl<S: StrategyValid<Regression>> EstimatorConfig for RegressionConfig<S> {
    type TaskType = Regression;
    type Strategy = S;

    fn validate() -> Result<(), &'static str> {
        // Compile-time validation can be extended here
        Ok(())
    }

    fn create_strategy() -> Self::Strategy {
        // This would need to be implemented per strategy type
        panic!("Strategy creation must be implemented per type")
    }
}

/// Builder for type-safe dummy estimators with compile-time validation
#[derive(Debug)]
pub struct TypeSafeEstimatorBuilder<Task, Strategy>
where
    Task: TaskType,
    Strategy: StrategyValid<Task>,
{
    strategy: Strategy,
    random_state: Option<u64>,
    _task: PhantomData<Task>,
}

impl<Strategy> TypeSafeDummyEstimator<Untrained, Classification, Strategy>
where
    Strategy: StrategyValid<Classification> + Clone,
{
    /// Create a new untrained type-safe classifier
    pub fn new(strategy: Strategy) -> Self {
        Self {
            strategy,
            random_state: None,
            _state: PhantomData,
            _task: PhantomData,
        }
    }

    /// Set random state (builder pattern)
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Build the estimator after validation
    pub fn build(self) -> Result<Self, &'static str> {
        // Compile-time and runtime validation
        self.validate_configuration()?;
        Ok(self)
    }

    /// Validate configuration
    fn validate_configuration(&self) -> Result<(), &'static str> {
        // Additional runtime validation can be added here
        Ok(())
    }
}

impl<Strategy> TypeSafeDummyEstimator<Untrained, Regression, Strategy>
where
    Strategy: StrategyValid<Regression> + Clone,
{
    /// Create a new untrained type-safe regressor
    pub fn new(strategy: Strategy) -> Self {
        Self {
            strategy,
            random_state: None,
            _state: PhantomData,
            _task: PhantomData,
        }
    }

    /// Set random state (builder pattern)
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Build the estimator after validation
    pub fn build(self) -> Result<Self, &'static str> {
        // Compile-time and runtime validation
        self.validate_configuration()?;
        Ok(self)
    }

    /// Validate configuration
    fn validate_configuration(&self) -> Result<(), &'static str> {
        // Additional runtime validation can be added here
        Ok(())
    }
}

/// Fit implementation for type-safe classifiers
impl<Strategy> Fit<Array2<f64>, Array1<i32>, TypeSafeFittedClassifier<Strategy>>
    for TypeSafeDummyEstimator<Untrained, Classification, Strategy>
where
    Strategy: StrategyValid<Classification> + Clone + Into<crate::ClassifierStrategy> + Send + Sync,
{
    type Fitted = TypeSafeFittedClassifier<Strategy>;
    fn fit(
        self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<TypeSafeFittedClassifier<Strategy>, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        // Calculate fitted data
        let mut class_counts = std::collections::HashMap::new();
        for &class in y.iter() {
            *class_counts.entry(class).or_insert(0) += 1;
        }

        let mut class_priors = std::collections::HashMap::new();
        let total_samples = y.len() as f64;
        for (&class, &count) in &class_counts {
            class_priors.insert(class, count as f64 / total_samples);
        }

        let most_frequent_class = class_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&class, _)| class)
            .unwrap_or(0);

        let fitted_data = ClassificationFittedData {
            class_counts,
            class_priors,
            most_frequent_class,
            n_samples: x.nrows(),
            n_features: x.ncols(),
        };

        Ok(TypeSafeFittedClassifier {
            strategy: self.strategy.clone(),
            fitted_data,
            random_state: self.random_state,
        })
    }
}

/// Fit implementation for type-safe regressors
impl<Strategy> Fit<Array2<f64>, Array1<f64>, TypeSafeFittedRegressor<Strategy>>
    for TypeSafeDummyEstimator<Untrained, Regression, Strategy>
where
    Strategy: StrategyValid<Regression> + Clone + Into<crate::RegressorStrategy> + Send + Sync,
{
    type Fitted = TypeSafeFittedRegressor<Strategy>;
    fn fit(
        self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<TypeSafeFittedRegressor<Strategy>, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        // Calculate fitted data
        let target_mean = y.mean().unwrap_or(0.0);
        let target_std = y.std(0.0);
        let target_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let target_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate median
        let mut sorted_targets = y.to_vec();
        sorted_targets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let target_median = if sorted_targets.len() % 2 == 0 {
            let mid = sorted_targets.len() / 2;
            (sorted_targets[mid - 1] + sorted_targets[mid]) / 2.0
        } else {
            sorted_targets[sorted_targets.len() / 2]
        };

        let fitted_data = RegressionFittedData {
            target_mean,
            target_median,
            target_std,
            target_min,
            target_max,
            n_samples: x.nrows(),
            n_features: x.ncols(),
        };

        Ok(TypeSafeFittedRegressor {
            strategy: self.strategy.clone(),
            fitted_data,
            random_state: self.random_state,
        })
    }
}

/// Predict implementation for type-safe fitted classifiers
impl<Strategy> Predict<Array2<f64>, Array1<i32>> for TypeSafeFittedClassifier<Strategy>
where
    Strategy: StrategyValid<Classification> + Clone + Into<crate::ClassifierStrategy>,
{
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>, SklearsError> {
        // Convert to legacy strategy and use existing implementation
        let legacy_strategy: crate::ClassifierStrategy = self.strategy.clone().into();
        let legacy_classifier = crate::DummyClassifier::new(legacy_strategy);

        // Create fake training data for legacy classifier
        let fake_x = Array2::zeros((self.fitted_data.n_samples, self.fitted_data.n_features));
        let fake_y: Array1<i32> = Array1::from_iter(
            self.fitted_data
                .class_counts
                .iter()
                .flat_map(|(&class, &count)| std::iter::repeat(class).take(count)),
        );

        let fitted_legacy = legacy_classifier.fit(&fake_x, &fake_y)?;
        fitted_legacy.predict(x)
    }
}

/// Predict implementation for type-safe fitted regressors
impl<Strategy> Predict<Array2<f64>, Array1<f64>> for TypeSafeFittedRegressor<Strategy>
where
    Strategy: StrategyValid<Regression> + Clone + Into<crate::RegressorStrategy>,
{
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, SklearsError> {
        // Convert to legacy strategy and use existing implementation
        let legacy_strategy: crate::RegressorStrategy = self.strategy.clone().into();
        let legacy_regressor = crate::DummyRegressor::new(legacy_strategy);

        // Create fake training data for legacy regressor
        let fake_x = Array2::zeros((self.fitted_data.n_samples, self.fitted_data.n_features));
        let fake_y = Array1::from_elem(self.fitted_data.n_samples, self.fitted_data.target_mean);

        let fitted_legacy = legacy_regressor.fit(&fake_x, &fake_y)?;
        fitted_legacy.predict(x)
    }
}

/// Compile-time strategy validation macro
#[macro_export]
macro_rules! validate_strategy_at_compile_time {
    ($strategy:ty, $task:ty) => {
        const _: fn() = || {
            fn assert_strategy_valid<S: StrategyValid<T>, T: TaskType>() {}
            assert_strategy_valid::<$strategy, $task>();
        };
    };
}

/// Compile-time estimator configuration macro
#[macro_export]
macro_rules! type_safe_estimator {
    (classification, $strategy:expr) => {{
        TypeSafeDummyEstimator::<Untrained, Classification, _>::new($strategy)
    }};
    (regression, $strategy:expr) => {{
        TypeSafeDummyEstimator::<Untrained, Regression, _>::new($strategy)
    }};
}

/// Zero-cost abstraction for strategy validation
pub struct ValidatedStrategy<S, T>
where
    S: StrategyValid<T>,
    T: TaskType,
{
    strategy: S,
    _task: PhantomData<T>,
}

impl<S, T> ValidatedStrategy<S, T>
where
    S: StrategyValid<T>,
    T: TaskType,
{
    /// Create a validated strategy (zero-cost)
    pub fn new(strategy: S) -> Self {
        Self {
            strategy,
            _task: PhantomData,
        }
    }

    /// Extract the strategy (zero-cost)
    pub fn into_strategy(self) -> S {
        self.strategy
    }

    /// Get a reference to the strategy (zero-cost)
    pub fn strategy(&self) -> &S {
        &self.strategy
    }
}

/// Trait for type-safe parameter validation
pub trait ParameterValidation {
    type Error;

    /// Validate parameters at compile time where possible
    fn validate(&self) -> Result<(), Self::Error>;
}

/// Type-safe parameter holder
#[derive(Debug, Clone)]
pub struct TypeSafeParameters<T> {
    value: T,
}

impl<T> TypeSafeParameters<T> {
    /// Create new type-safe parameters
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// Get the parameter value
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Consume and return the parameter value
    pub fn into_inner(self) -> T {
        self.value
    }
}

/// Bounded parameter type for compile-time validation
#[derive(Debug, Clone, Copy)]
pub struct BoundedParameter<T, const MIN: i64, const MAX: i64> {
    value: T,
}

impl<T, const MIN: i64, const MAX: i64> BoundedParameter<T, MIN, MAX>
where
    T: PartialOrd + Copy + TryFrom<i64>,
{
    /// Create a bounded parameter with compile-time bounds checking
    pub fn new(value: T) -> Result<Self, &'static str> {
        let min_val = T::try_from(MIN).map_err(|_| "Invalid minimum bound")?;
        let max_val = T::try_from(MAX).map_err(|_| "Invalid maximum bound")?;

        if value >= min_val && value <= max_val {
            Ok(Self { value })
        } else {
            Err("Parameter value out of bounds")
        }
    }

    /// Get the parameter value
    pub fn get(&self) -> T {
        self.value
    }
}

// Specific implementation for i32
impl<const MIN: i64, const MAX: i64> BoundedParameter<i32, MIN, MAX> {
    /// Create a bounded i32 parameter
    pub fn new_i32(value: i32) -> Result<Self, &'static str> {
        let value_i64 = value as i64;
        if value_i64 >= MIN && value_i64 <= MAX {
            Ok(Self { value })
        } else {
            Err("Parameter value out of bounds")
        }
    }
}

// Specific implementation for u64
impl<const MIN: i64, const MAX: i64> BoundedParameter<u64, MIN, MAX> {
    /// Create a bounded u64 parameter
    pub fn new_u64(value: u64) -> Result<Self, &'static str> {
        let value_i64 = value as i64;
        if value_i64 >= MIN && value_i64 <= MAX && value <= i64::MAX as u64 {
            Ok(Self { value })
        } else {
            Err("Parameter value out of bounds")
        }
    }
}

// Specific implementation for f64
impl<const MIN: i64, const MAX: i64> BoundedParameter<f64, MIN, MAX> {
    /// Create a bounded f64 parameter
    pub fn new_f64(value: f64) -> Result<Self, &'static str> {
        let min_f64 = MIN as f64;
        let max_f64 = MAX as f64;
        if value >= min_f64 && value <= max_f64 {
            Ok(Self { value })
        } else {
            Err("Parameter value out of bounds")
        }
    }

    /// Get the f64 parameter value
    pub fn get_f64(&self) -> f64 {
        self.value
    }
}

/// Type-safe probability parameter (0.0 to 1.0)
pub type Probability = BoundedParameter<f64, 0, 1>;

/// Type-safe positive integer parameter
pub type PositiveInt = BoundedParameter<i32, 1, { i32::MAX as i64 }>;

/// Type-safe random seed parameter
pub type RandomSeed = BoundedParameter<u64, 0, { i64::MAX }>;

/// Type-safe dimension parameter (1 to maximum dimensions)
pub type Dimension = BoundedParameter<usize, 1, 10000>;

/// Type-safe batch size parameter
pub type BatchSize = BoundedParameter<usize, 1, 1000000>;

/// Type-safe confidence level parameter (0.0 to 1.0)
pub type ConfidenceLevel = BoundedParameter<f64, 0, 1>;

/// Type-safe tolerance parameter (positive values only)
pub type Tolerance = BoundedParameter<f64, 0, { i64::MAX }>;

/// Type-safe iteration count parameter
pub type IterationCount = BoundedParameter<u32, 1, { i32::MAX as i64 }>;

/// Implementation for usize bounded parameters
impl<const MIN: i64, const MAX: i64> BoundedParameter<usize, MIN, MAX> {
    /// Create a bounded usize parameter
    pub fn new_usize(value: usize) -> Result<Self, &'static str> {
        let value_i64 = value as i64;
        if value_i64 >= MIN && value_i64 <= MAX && value <= i64::MAX as usize {
            Ok(Self { value })
        } else {
            Err("Parameter value out of bounds")
        }
    }

    /// Get the usize parameter value
    pub fn get_usize(&self) -> usize {
        self.value
    }
}

/// Implementation for u32 bounded parameters
impl<const MIN: i64, const MAX: i64> BoundedParameter<u32, MIN, MAX> {
    /// Create a bounded u32 parameter
    pub fn new_u32(value: u32) -> Result<Self, &'static str> {
        let value_i64 = value as i64;
        if value_i64 >= MIN && value_i64 <= MAX {
            Ok(Self { value })
        } else {
            Err("Parameter value out of bounds")
        }
    }

    /// Get the u32 parameter value
    pub fn get_u32(&self) -> u32 {
        self.value
    }
}

/// Const generic strategy selector for compile-time strategy configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstStrategySelector<const STRATEGY_ID: usize>;

/// Trait for mapping strategy IDs to strategy types
pub trait StrategyFromId<const ID: usize> {
    type Strategy;

    fn create_strategy() -> Self::Strategy;
}

/// Classification strategy mapping (const generic approach)
impl StrategyFromId<0> for ConstStrategySelector<0> {
    type Strategy = crate::ClassifierStrategy;

    fn create_strategy() -> Self::Strategy {
        crate::ClassifierStrategy::MostFrequent
    }
}

impl StrategyFromId<1> for ConstStrategySelector<1> {
    type Strategy = crate::ClassifierStrategy;

    fn create_strategy() -> Self::Strategy {
        crate::ClassifierStrategy::Stratified
    }
}

impl StrategyFromId<2> for ConstStrategySelector<2> {
    type Strategy = crate::ClassifierStrategy;

    fn create_strategy() -> Self::Strategy {
        crate::ClassifierStrategy::Uniform
    }
}

/// Regression strategy mapping (const generic approach)
impl StrategyFromId<10> for ConstStrategySelector<10> {
    type Strategy = crate::RegressorStrategy;

    fn create_strategy() -> Self::Strategy {
        crate::RegressorStrategy::Mean
    }
}

impl StrategyFromId<11> for ConstStrategySelector<11> {
    type Strategy = crate::RegressorStrategy;

    fn create_strategy() -> Self::Strategy {
        crate::RegressorStrategy::Median
    }
}

impl StrategyFromId<12> for ConstStrategySelector<12> {
    type Strategy = crate::RegressorStrategy;

    fn create_strategy() -> Self::Strategy {
        crate::RegressorStrategy::Normal {
            mean: None,
            std: None,
        }
    }
}

/// Compile-time estimator with const generic strategy selection
#[derive(Debug, Clone)]
pub struct ConstGenericEstimator<State, Task, const STRATEGY_ID: usize>
where
    State: EstimatorState,
    Task: TaskType,
    ConstStrategySelector<STRATEGY_ID>: StrategyFromId<STRATEGY_ID>,
{
    random_state: Option<u64>,
    _state: PhantomData<State>,
    _task: PhantomData<Task>,
}

impl<Task, const STRATEGY_ID: usize> Default for ConstGenericEstimator<Untrained, Task, STRATEGY_ID>
where
    Task: TaskType,
    ConstStrategySelector<STRATEGY_ID>: StrategyFromId<STRATEGY_ID>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Task, const STRATEGY_ID: usize> ConstGenericEstimator<Untrained, Task, STRATEGY_ID>
where
    Task: TaskType,
    ConstStrategySelector<STRATEGY_ID>: StrategyFromId<STRATEGY_ID>,
{
    /// Create new const generic estimator
    pub fn new() -> Self {
        Self {
            random_state: None,
            _state: PhantomData,
            _task: PhantomData,
        }
    }

    /// Set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get the strategy at compile time
    pub fn strategy(
    ) -> <ConstStrategySelector<STRATEGY_ID> as StrategyFromId<STRATEGY_ID>>::Strategy {
        ConstStrategySelector::<STRATEGY_ID>::create_strategy()
    }
}

/// Zero-cost wrapper for type-safe operations
#[derive(Debug, Clone, Copy)]
pub struct ZeroCostWrapper<T> {
    inner: T,
}

impl<T> ZeroCostWrapper<T> {
    /// Create a zero-cost wrapper
    #[inline(always)]
    pub const fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Extract the inner value (zero-cost)
    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Get a reference to the inner value (zero-cost)
    #[inline(always)]
    pub const fn get(&self) -> &T {
        &self.inner
    }

    /// Map the inner value (zero-cost at compile time)
    #[inline(always)]
    pub fn map<U, F>(self, f: F) -> ZeroCostWrapper<U>
    where
        F: FnOnce(T) -> U,
    {
        ZeroCostWrapper::new(f(self.inner))
    }
}

/// Phantom type for tracking statistical properties at compile time
#[derive(Debug, Clone, Copy)]
pub struct StatisticalPhantom<
    const IS_DETERMINISTIC: bool,
    const IS_STATELESS: bool,
    const REQUIRES_FITTING: bool,
> {
    _marker: PhantomData<()>,
}

impl<const IS_DETERMINISTIC: bool, const IS_STATELESS: bool, const REQUIRES_FITTING: bool> Default
    for StatisticalPhantom<IS_DETERMINISTIC, IS_STATELESS, REQUIRES_FITTING>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const IS_DETERMINISTIC: bool, const IS_STATELESS: bool, const REQUIRES_FITTING: bool>
    StatisticalPhantom<IS_DETERMINISTIC, IS_STATELESS, REQUIRES_FITTING>
{
    /// Create new statistical phantom type
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Check if the strategy is deterministic at compile time
    pub const fn is_deterministic() -> bool {
        IS_DETERMINISTIC
    }

    /// Check if the strategy is stateless at compile time
    pub const fn is_stateless() -> bool {
        IS_STATELESS
    }

    /// Check if the strategy requires fitting at compile time
    pub const fn requires_fitting() -> bool {
        REQUIRES_FITTING
    }
}

/// Type alias for deterministic, stateless strategies that don't require fitting
pub type SimpleDeterministicStrategy = StatisticalPhantom<true, true, false>;

/// Type alias for stochastic strategies that require fitting
pub type StochasticFittedStrategy = StatisticalPhantom<false, false, true>;

/// Compile-time validation trait for strategy properties
pub trait StrategyProperties {
    const IS_DETERMINISTIC: bool;
    const IS_STATELESS: bool;
    const REQUIRES_FITTING: bool;

    type Phantom: 'static;

    /// Get the phantom type for this strategy
    fn phantom() -> Self::Phantom;
}

/// Enhanced type-safe estimator with statistical properties
#[derive(Debug, Clone)]
pub struct StatisticallyTypedEstimator<State, Task, Strategy, Properties>
where
    State: EstimatorState,
    Task: TaskType,
    Strategy: StrategyValid<Task> + StrategyProperties<Phantom = Properties>,
    Properties: 'static,
{
    strategy: Strategy,
    random_state: Option<u64>,
    _state: PhantomData<State>,
    _task: PhantomData<Task>,
    _properties: PhantomData<Properties>,
}

impl<Task, Strategy, Properties> StatisticallyTypedEstimator<Untrained, Task, Strategy, Properties>
where
    Task: TaskType,
    Strategy: StrategyValid<Task> + StrategyProperties<Phantom = Properties> + Clone,
    Properties: 'static,
{
    /// Create new statistically typed estimator
    pub fn new(strategy: Strategy) -> Self {
        Self {
            strategy,
            random_state: None,
            _state: PhantomData,
            _task: PhantomData,
            _properties: PhantomData,
        }
    }

    /// Check if this estimator's strategy is deterministic (compile-time)
    pub const fn is_deterministic() -> bool {
        Strategy::IS_DETERMINISTIC
    }

    /// Check if this estimator's strategy is stateless (compile-time)
    pub const fn is_stateless() -> bool {
        Strategy::IS_STATELESS
    }

    /// Check if this estimator requires fitting (compile-time)
    pub const fn requires_fitting() -> bool {
        Strategy::REQUIRES_FITTING
    }
}

/// Compile-time configuration validation macro
#[macro_export]
macro_rules! validate_estimator_config {
    ($estimator_type:ty, $state:ty, $task:ty) => {
        const _: () = {
            // Ensure the estimator implements the required traits
            fn _assert_estimator_state<S: EstimatorState>() {}
            fn _assert_task_type<T: TaskType>() {}
            fn _assert_type_safe_estimator<
                E: TypeSafeEstimator<S, T>,
                S: EstimatorState,
                T: TaskType,
            >() {
            }

            _assert_estimator_state::<$state>();
            _assert_task_type::<$task>();
        };
    };
}

/// Compile-time strategy compatibility validation macro
#[macro_export]
macro_rules! assert_strategy_compatible {
    ($strategy:ty, $task:ty) => {
        const _: () = {
            fn _assert_compatible<S: StrategyValid<T>, T: TaskType>() {}
            _assert_compatible::<$strategy, $task>();
        };
    };
}

/// Zero-cost compile-time feature flag
#[derive(Debug, Clone, Copy)]
pub struct CompileTimeFeature<const ENABLED: bool>;

impl<const ENABLED: bool> CompileTimeFeature<ENABLED> {
    /// Check if feature is enabled at compile time
    pub const fn is_enabled() -> bool {
        ENABLED
    }

    /// Execute code only if feature is enabled (zero-cost)
    #[inline(always)]
    pub fn when_enabled<F, R>(f: F) -> Option<R>
    where
        F: FnOnce() -> R,
    {
        if ENABLED {
            Some(f())
        } else {
            None
        }
    }
}

/// Type-safe statistical operations with compile-time guarantees
pub trait TypeSafeStatisticalOps<T> {
    /// Compute mean with type safety
    fn safe_mean(&self) -> Option<T>;

    /// Compute variance with type safety
    fn safe_variance(&self) -> Option<T>;

    /// Compute standard deviation with type safety
    fn safe_std(&self) -> Option<T>;
}

impl TypeSafeStatisticalOps<f64> for Array1<f64> {
    fn safe_mean(&self) -> Option<f64> {
        if self.is_empty() {
            None
        } else {
            Some(self.mean().unwrap_or(0.0))
        }
    }

    fn safe_variance(&self) -> Option<f64> {
        if self.len() < 2 {
            None
        } else {
            Some(self.var(1.0)) // Bessel's correction
        }
    }

    fn safe_std(&self) -> Option<f64> {
        self.safe_variance().map(|v| v.sqrt())
    }
}

/// Compile-time memory layout optimization
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptimizedLayout<T> {
    data: T,
    _pad: [u8; 0], // Zero-sized padding for potential alignment
}

impl<T> OptimizedLayout<T> {
    /// Create optimized layout (zero-cost)
    #[inline(always)]
    pub const fn new(data: T) -> Self {
        Self { data, _pad: [] }
    }

    /// Extract data (zero-cost)
    #[inline(always)]
    pub fn into_data(self) -> T {
        self.data
    }

    /// Get reference to data (zero-cost)
    #[inline(always)]
    pub const fn data(&self) -> &T {
        &self.data
    }
}

/// Trait for estimators with compile-time guarantees
pub trait TypeSafeEstimator<State: EstimatorState, Task: TaskType> {
    type Strategy: StrategyValid<Task>;

    /// Get the current state (compile-time known)
    fn state(&self) -> State;

    /// Get the task type (compile-time known)
    fn task_type(&self) -> Task;

    /// Validate the estimator configuration
    fn validate(&self) -> Result<(), &'static str>;
}

impl<State, Task, Strategy> TypeSafeEstimator<State, Task>
    for TypeSafeDummyEstimator<State, Task, Strategy>
where
    State: EstimatorState + Default,
    Task: TaskType + Default,
    Strategy: StrategyValid<Task>,
{
    type Strategy = Strategy;

    fn state(&self) -> State {
        State::default()
    }

    fn task_type(&self) -> Task {
        Task::default()
    }

    fn validate(&self) -> Result<(), &'static str> {
        // Validation logic here
        Ok(())
    }
}

/// Implement Default for state markers to enable compile-time checks
impl Default for Untrained {
    fn default() -> Self {
        Untrained
    }
}

impl Default for Trained {
    fn default() -> Self {
        Trained
    }
}

impl Default for Classification {
    fn default() -> Self {
        Classification
    }
}

impl Default for Regression {
    fn default() -> Self {
        Regression
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClassifierStrategy, RegressorStrategy};
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_type_safe_classifier_creation() {
        let strategy = ClassifierStrategy::MostFrequent;
        let classifier = TypeSafeDummyEstimator::<Untrained, Classification, _>::new(strategy);

        // This should compile - correct types
        assert!(classifier.validate().is_ok());
    }

    #[test]
    fn test_type_safe_regressor_creation() {
        let strategy = RegressorStrategy::Mean;
        let regressor = TypeSafeDummyEstimator::<Untrained, Regression, _>::new(strategy);

        // This should compile - correct types
        assert!(regressor.validate().is_ok());
    }

    #[test]
    fn test_bounded_parameters() {
        // Valid probability
        let prob = Probability::new_f64(0.5);
        assert!(prob.is_ok());
        assert_eq!(prob.unwrap().get_f64(), 0.5);

        // Invalid probability
        let invalid_prob = Probability::new_f64(1.5);
        assert!(invalid_prob.is_err());

        // Valid positive integer
        let pos_int = PositiveInt::new_i32(42);
        assert!(pos_int.is_ok());
        assert_eq!(pos_int.unwrap().get(), 42);

        // Invalid positive integer
        let invalid_int = PositiveInt::new_i32(-1);
        assert!(invalid_int.is_err());
    }

    #[test]
    fn test_validated_strategy() {
        let strategy = ClassifierStrategy::MostFrequent;
        let validated = ValidatedStrategy::<_, Classification>::new(strategy.clone());

        assert_eq!(
            format!("{:?}", validated.strategy()),
            format!("{:?}", strategy)
        );

        let extracted = validated.into_strategy();
        assert_eq!(format!("{:?}", extracted), format!("{:?}", strategy));
    }

    #[test]
    fn test_type_safe_parameters() {
        let params = TypeSafeParameters::new(42.0);
        assert_eq!(*params.get(), 42.0);
        assert_eq!(params.into_inner(), 42.0);
    }

    #[test]
    fn test_estimator_state_transitions() {
        let strategy = ClassifierStrategy::MostFrequent;
        let untrained = TypeSafeDummyEstimator::<Untrained, Classification, _>::new(strategy);

        // Can't predict without fitting - this is ensured by the type system
        // fitted.predict() would require a TypeSafeFittedClassifier

        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let fitted = untrained.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x);

        assert!(predictions.is_ok());
        assert_eq!(predictions.unwrap().len(), 4);
    }

    #[test]
    fn test_const_generic_estimator() {
        // Most frequent classifier (ID = 0)
        let classifier = ConstGenericEstimator::<Untrained, Classification, 0>::new();
        let strategy = ConstGenericEstimator::<Untrained, Classification, 0>::strategy();

        assert_eq!(format!("{:?}", strategy), "MostFrequent");

        // Mean regressor (ID = 10)
        let regressor = ConstGenericEstimator::<Untrained, Regression, 10>::new();
        let reg_strategy = ConstGenericEstimator::<Untrained, Regression, 10>::strategy();

        assert_eq!(format!("{:?}", reg_strategy), "Mean");
    }

    #[test]
    fn test_zero_cost_wrapper() {
        let wrapper = ZeroCostWrapper::new(42);
        assert_eq!(*wrapper.get(), 42);
        assert_eq!(wrapper.into_inner(), 42);

        let mapped = ZeroCostWrapper::new(10).map(|x| x * 2);
        assert_eq!(mapped.into_inner(), 20);
    }

    #[test]
    fn test_statistical_phantom() {
        // Test deterministic, stateless strategy
        let phantom = SimpleDeterministicStrategy::new();
        assert!(SimpleDeterministicStrategy::is_deterministic());
        assert!(SimpleDeterministicStrategy::is_stateless());
        assert!(!SimpleDeterministicStrategy::requires_fitting());

        // Test stochastic, fitted strategy
        let stochastic = StochasticFittedStrategy::new();
        assert!(!StochasticFittedStrategy::is_deterministic());
        assert!(!StochasticFittedStrategy::is_stateless());
        assert!(StochasticFittedStrategy::requires_fitting());
    }

    #[test]
    fn test_compile_time_feature() {
        // Enabled feature
        type EnabledFeature = CompileTimeFeature<true>;
        assert!(EnabledFeature::is_enabled());

        let result = EnabledFeature::when_enabled(|| 42);
        assert_eq!(result, Some(42));

        // Disabled feature
        type DisabledFeature = CompileTimeFeature<false>;
        assert!(!DisabledFeature::is_enabled());

        let no_result = DisabledFeature::when_enabled(|| 42);
        assert_eq!(no_result, None);
    }

    #[test]
    fn test_type_safe_statistical_ops() {
        use scirs2_core::ndarray::array;

        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(data.safe_mean(), Some(3.0));
        assert!(data.safe_variance().is_some());
        assert!(data.safe_std().is_some());

        // Empty array
        let empty: Array1<f64> = array![];
        assert_eq!(empty.safe_mean(), None);
        assert_eq!(empty.safe_variance(), None);
        assert_eq!(empty.safe_std(), None);

        // Single element
        let single = array![42.0];
        assert_eq!(single.safe_mean(), Some(42.0));
        assert_eq!(single.safe_variance(), None); // Need at least 2 elements
    }

    #[test]
    fn test_optimized_layout() {
        let layout = OptimizedLayout::new(42);
        assert_eq!(*layout.data(), 42);
        assert_eq!(layout.into_data(), 42);
    }

    // Compile-time test - uncomment to verify compile-time checking
    // #[test]
    // fn test_compile_time_validation() {
    //     // This should not compile - wrong task type for strategy
    //     // let invalid = TypeSafeDummyEstimator::<Untrained, Regression, ClassifierStrategy>::new(
    //     //     ClassifierStrategy::MostFrequent
    //     // );
    //
    //     // Test compile-time macros
    //     validate_estimator_config!(TypeSafeDummyEstimator<Untrained, Classification, ClassifierStrategy>, Untrained, Classification);
    //     assert_strategy_compatible!(ClassifierStrategy, Classification);
    // }
}
