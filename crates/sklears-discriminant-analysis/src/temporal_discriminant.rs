//! Temporal Discriminant Analysis implementation
//!
//! This module implements discriminant analysis methods for time-series data,
//! including dynamic linear discriminant analysis, temporal pattern recognition,
//! and sequential pattern discrimination.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{concatenate, s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for Temporal Discriminant Analysis
#[derive(Debug, Clone)]
pub struct TemporalDiscriminantAnalysisConfig {
    /// Window size for temporal analysis
    pub window_size: usize,
    /// Overlap between consecutive windows
    pub window_overlap: usize,
    /// Temporal modeling method
    pub temporal_method: TemporalMethod,
    /// Whether to use dynamic updating
    pub dynamic_updating: bool,
    /// Forgetting factor for dynamic models
    pub forgetting_factor: Float,
    /// Number of temporal components
    pub n_temporal_components: Option<usize>,
    /// Regularization parameter
    pub regularization: Float,
    /// Whether to standardize temporal features
    pub standardize_temporal: bool,
    /// Temporal aggregation method
    pub aggregation_method: AggregationMethod,
    /// Trend analysis method
    pub trend_method: TrendMethod,
    /// Seasonality analysis parameters
    pub seasonality_period: Option<usize>,
    /// Maximum number of lags to consider
    pub max_lags: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Maximum iterations
    pub max_iter: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for TemporalDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            window_overlap: 5,
            temporal_method: TemporalMethod::SlidingWindow,
            dynamic_updating: false,
            forgetting_factor: 0.95,
            n_temporal_components: None,
            regularization: 0.01,
            standardize_temporal: true,
            aggregation_method: AggregationMethod::Mean,
            trend_method: TrendMethod::Linear,
            seasonality_period: None,
            max_lags: 5,
            tol: 1e-6,
            max_iter: 100,
            random_state: None,
        }
    }
}

/// Temporal modeling methods
#[derive(Debug, Clone)]
pub enum TemporalMethod {
    /// Sliding window approach
    SlidingWindow,
    /// Autoregressive (AR) model
    Autoregressive { order: usize },
    /// Moving average (MA) model
    MovingAverage { order: usize },
    /// ARMA model combining AR and MA
    ARMA { ar_order: usize, ma_order: usize },
    /// Hidden Markov Model-based
    HiddenMarkov { n_states: usize },
    /// Recurrent features
    Recurrent { memory_length: usize },
    /// State space model
    StateSpace { state_dim: usize },
}

/// Temporal aggregation methods
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    /// Simple mean
    Mean,
    /// Weighted mean with decay
    WeightedMean { decay_rate: Float },
    /// Maximum value
    Maximum,
    /// Minimum value
    Minimum,
    /// Last value
    Last,
    /// Median value
    Median,
    /// Standard deviation
    StandardDeviation,
    /// Slope of linear trend
    Slope,
}

/// Trend analysis methods
#[derive(Debug, Clone)]
pub enum TrendMethod {
    /// Linear trend analysis
    Linear,
    /// Polynomial trend
    Polynomial { degree: usize },
    /// Exponential trend
    Exponential,
    /// Moving average trend
    MovingAverage { window: usize },
    /// None (no trend analysis)
    None,
}

/// Temporal pattern information
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Pattern identifier
    pub id: usize,
    /// Temporal length
    pub length: usize,
    /// Pattern coefficients
    pub coefficients: Array1<Float>,
    /// Pattern weights
    pub weights: Array1<Float>,
    /// Associated class
    pub class_label: i32,
}

/// Temporal Discriminant Analysis
///
/// A discriminant analysis method specifically designed for time-series data,
/// incorporating temporal dynamics, trends, and sequential patterns.
#[derive(Debug, Clone)]
pub struct TemporalDiscriminantAnalysis<State = Untrained> {
    config: TemporalDiscriminantAnalysisConfig,
    state: PhantomData<State>,
    // Trained state fields
    classes_: Option<Array1<i32>>,
    temporal_patterns_: Option<Vec<TemporalPattern>>,
    temporal_components_: Option<Array2<Float>>,
    temporal_means_: Option<Array2<Float>>,
    temporal_covariances_: Option<Vec<Array2<Float>>>,
    trend_coefficients_: Option<Array2<Float>>,
    seasonal_components_: Option<Array2<Float>>,
    ar_coefficients_: Option<Array2<Float>>,
    ma_coefficients_: Option<Array2<Float>>,
    state_transition_matrix_: Option<Array2<Float>>,
    observation_matrix_: Option<Array2<Float>>,
    feature_means_: Option<Array1<Float>>,
    feature_stds_: Option<Array1<Float>>,
    n_features_: Option<usize>,
    n_time_steps_: Option<usize>,
}

impl TemporalDiscriminantAnalysis<Untrained> {
    /// Create a new TemporalDiscriminantAnalysis instance
    pub fn new() -> Self {
        Self {
            config: TemporalDiscriminantAnalysisConfig::default(),
            state: PhantomData,
            classes_: None,
            temporal_patterns_: None,
            temporal_components_: None,
            temporal_means_: None,
            temporal_covariances_: None,
            trend_coefficients_: None,
            seasonal_components_: None,
            ar_coefficients_: None,
            ma_coefficients_: None,
            state_transition_matrix_: None,
            observation_matrix_: None,
            feature_means_: None,
            feature_stds_: None,
            n_features_: None,
            n_time_steps_: None,
        }
    }

    /// Set window size
    pub fn window_size(mut self, size: usize) -> Self {
        self.config.window_size = size;
        self
    }

    /// Set window overlap
    pub fn window_overlap(mut self, overlap: usize) -> Self {
        self.config.window_overlap = overlap;
        self
    }

    /// Set temporal method
    pub fn temporal_method(mut self, method: TemporalMethod) -> Self {
        self.config.temporal_method = method;
        self
    }

    /// Set dynamic updating
    pub fn dynamic_updating(mut self, dynamic: bool) -> Self {
        self.config.dynamic_updating = dynamic;
        self
    }

    /// Set forgetting factor
    pub fn forgetting_factor(mut self, factor: Float) -> Self {
        self.config.forgetting_factor = factor;
        self
    }

    /// Set number of temporal components
    pub fn n_temporal_components(mut self, n: Option<usize>) -> Self {
        self.config.n_temporal_components = n;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, reg: Float) -> Self {
        self.config.regularization = reg;
        self
    }

    /// Set standardize temporal features
    pub fn standardize_temporal(mut self, standardize: bool) -> Self {
        self.config.standardize_temporal = standardize;
        self
    }

    /// Set aggregation method
    pub fn aggregation_method(mut self, method: AggregationMethod) -> Self {
        self.config.aggregation_method = method;
        self
    }

    /// Set trend method
    pub fn trend_method(mut self, method: TrendMethod) -> Self {
        self.config.trend_method = method;
        self
    }

    /// Set seasonality period
    pub fn seasonality_period(mut self, period: Option<usize>) -> Self {
        self.config.seasonality_period = period;
        self
    }

    /// Set maximum lags
    pub fn max_lags(mut self, lags: usize) -> Self {
        self.config.max_lags = lags;
        self
    }

    /// Set tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}

impl Default for TemporalDiscriminantAnalysis<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TemporalDiscriminantAnalysis<Untrained> {
    type Config = TemporalDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array3<Float>, Array1<i32>> for TemporalDiscriminantAnalysis<Untrained> {
    type Fitted = TemporalDiscriminantAnalysis<Trained>;

    fn fit(self, x: &Array3<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features, n_time_steps) = x.dim();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of labels must match number of samples".to_string(),
            ));
        }

        if n_time_steps < self.config.window_size {
            return Err(SklearsError::InvalidInput(
                "Time series length must be at least window_size".to_string(),
            ));
        }

        // Extract unique classes
        let mut unique_classes = y.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "At least 2 classes are required".to_string(),
            ));
        }

        // Standardize temporal features if requested
        let (standardized_x, feature_means, feature_stds) = if self.config.standardize_temporal {
            self.standardize_temporal_features(x)?
        } else {
            (x.clone(), None, None)
        };

        // Extract temporal features based on method
        let temporal_features = self.extract_temporal_features(&standardized_x)?;

        // Decompose into trend, seasonal, and residual components
        let (trend_coefficients, seasonal_components) =
            self.decompose_temporal_components(&standardized_x)?;

        // Fit temporal models
        let (ar_coefficients, ma_coefficients) = self.fit_temporal_models(&standardized_x)?;

        // Learn temporal patterns for each class
        let temporal_patterns = self.learn_temporal_patterns(&standardized_x, y, &classes)?;

        // Compute temporal means and covariances for each class
        let (temporal_means, temporal_covariances) =
            self.compute_temporal_statistics(&temporal_features, y, &classes)?;

        // Learn state space representation if applicable
        let (state_transition_matrix, observation_matrix) =
            self.learn_state_space_model(&standardized_x, y, &classes)?;

        // Compute temporal discriminant components
        let temporal_components = self.compute_temporal_discriminant_components(
            &temporal_features,
            y,
            &classes,
            &temporal_means,
            &temporal_covariances,
        )?;

        Ok(TemporalDiscriminantAnalysis {
            config: self.config,
            state: PhantomData,
            classes_: Some(classes),
            temporal_patterns_: Some(temporal_patterns),
            temporal_components_: Some(temporal_components),
            temporal_means_: Some(temporal_means),
            temporal_covariances_: Some(temporal_covariances),
            trend_coefficients_: Some(trend_coefficients),
            seasonal_components_: seasonal_components,
            ar_coefficients_: ar_coefficients,
            ma_coefficients_: ma_coefficients,
            state_transition_matrix_: state_transition_matrix,
            observation_matrix_: observation_matrix,
            feature_means_: feature_means,
            feature_stds_: feature_stds,
            n_features_: Some(n_features),
            n_time_steps_: Some(n_time_steps),
        })
    }
}

impl TemporalDiscriminantAnalysis<Untrained> {
    fn standardize_temporal_features(
        &self,
        x: &Array3<Float>,
    ) -> Result<(Array3<Float>, Option<Array1<Float>>, Option<Array1<Float>>)> {
        let (n_samples, n_features, n_time_steps) = x.dim();

        // Flatten time series for standardization
        let flattened = x
            .view()
            .into_shape((n_samples * n_time_steps, n_features))?;

        // Compute means and standard deviations
        let means = flattened.mean_axis(Axis(0)).unwrap();
        let stds = flattened.std_axis(Axis(0), 0.0);

        // Avoid division by zero
        let stds = stds.mapv(|s| if s == 0.0 { 1.0 } else { s });

        // Standardize
        let mut standardized = x.clone();
        for i in 0..n_samples {
            for t in 0..n_time_steps {
                for f in 0..n_features {
                    standardized[[i, f, t]] = (x[[i, f, t]] - means[f]) / stds[f];
                }
            }
        }

        Ok((standardized, Some(means), Some(stds)))
    }

    fn extract_temporal_features(&self, x: &Array3<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features, n_time_steps) = x.dim();

        match &self.config.temporal_method {
            TemporalMethod::SlidingWindow => self.extract_sliding_window_features(x),
            TemporalMethod::Autoregressive { order } => self.extract_ar_features(x, *order),
            TemporalMethod::MovingAverage { order } => self.extract_ma_features(x, *order),
            TemporalMethod::ARMA { ar_order, ma_order } => {
                self.extract_arma_features(x, *ar_order, *ma_order)
            }
            TemporalMethod::HiddenMarkov { n_states } => self.extract_hmm_features(x, *n_states),
            TemporalMethod::Recurrent { memory_length } => {
                self.extract_recurrent_features(x, *memory_length)
            }
            TemporalMethod::StateSpace { state_dim } => {
                self.extract_state_space_features(x, *state_dim)
            }
        }
    }

    fn extract_sliding_window_features(&self, x: &Array3<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features, n_time_steps) = x.dim();
        let window_size = self.config.window_size;
        let overlap = self.config.window_overlap;
        let step = if window_size > overlap {
            window_size - overlap
        } else {
            1 // Minimum step size
        };

        let n_windows = if n_time_steps >= window_size {
            (n_time_steps - window_size) / step + 1
        } else {
            0
        };

        if n_windows == 0 {
            return Err(SklearsError::InvalidInput(
                "Time series too short for window analysis".to_string(),
            ));
        }

        // Number of features per window
        let features_per_window = match &self.config.aggregation_method {
            AggregationMethod::Mean
            | AggregationMethod::WeightedMean { .. }
            | AggregationMethod::Maximum
            | AggregationMethod::Minimum
            | AggregationMethod::Last
            | AggregationMethod::Median
            | AggregationMethod::StandardDeviation
            | AggregationMethod::Slope => n_features,
        };

        let total_features = n_windows * features_per_window;
        let mut features = Array2::zeros((n_samples, total_features));

        for sample_idx in 0..n_samples {
            let mut feature_idx = 0;

            for window_idx in 0..n_windows {
                let start_time = window_idx * step;
                let end_time = start_time + window_size;

                // Extract window data
                let window_data = x.slice(s![sample_idx, .., start_time..end_time]);

                // Aggregate features for this window
                let aggregated = self.aggregate_window_features(&window_data)?;

                for &value in aggregated.iter() {
                    features[[sample_idx, feature_idx]] = value;
                    feature_idx += 1;
                }
            }
        }

        Ok(features)
    }

    fn aggregate_window_features(&self, window_data: &ArrayView2<Float>) -> Result<Array1<Float>> {
        let (n_features, window_size) = window_data.dim();

        match &self.config.aggregation_method {
            AggregationMethod::Mean => Ok(window_data.mean_axis(Axis(1)).unwrap()),
            AggregationMethod::WeightedMean { decay_rate } => {
                let mut weighted_features = Array1::zeros(n_features);
                let mut total_weight = 0.0;

                for t in 0..window_size {
                    let weight = decay_rate.powi(t as i32);
                    total_weight += weight;

                    for f in 0..n_features {
                        weighted_features[f] += weight * window_data[[f, t]];
                    }
                }

                if total_weight > 0.0 {
                    weighted_features /= total_weight;
                }

                Ok(weighted_features)
            }
            AggregationMethod::Maximum => {
                let mut max_features = Array1::zeros(n_features);
                for f in 0..n_features {
                    max_features[f] = window_data
                        .row(f)
                        .iter()
                        .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
                }
                Ok(max_features)
            }
            AggregationMethod::Minimum => {
                let mut min_features = Array1::zeros(n_features);
                for f in 0..n_features {
                    min_features[f] = window_data
                        .row(f)
                        .iter()
                        .fold(Float::INFINITY, |a, &b| a.min(b));
                }
                Ok(min_features)
            }
            AggregationMethod::Last => Ok(window_data.column(window_size - 1).to_owned()),
            AggregationMethod::StandardDeviation => Ok(window_data.std_axis(Axis(1), 0.0)),
            AggregationMethod::Slope => {
                let mut slopes = Array1::zeros(n_features);
                for f in 0..n_features {
                    slopes[f] = self.compute_slope(&window_data.row(f))?;
                }
                Ok(slopes)
            }
            AggregationMethod::Median => {
                let mut medians = Array1::zeros(n_features);
                for f in 0..n_features {
                    let mut values: Vec<Float> = window_data.row(f).to_vec();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    medians[f] = if values.len() % 2 == 0 {
                        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };
                }
                Ok(medians)
            }
        }
    }

    fn compute_slope(&self, values: &ArrayView1<Float>) -> Result<Float> {
        let n = values.len() as Float;
        if n < 2.0 {
            return Ok(0.0);
        }

        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.mean().unwrap();

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as Float;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean) * (x - x_mean);
        }

        if denominator != 0.0 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    fn extract_ar_features(&self, x: &Array3<Float>, order: usize) -> Result<Array2<Float>> {
        let (n_samples, n_features, n_time_steps) = x.dim();

        if n_time_steps <= order {
            return Err(SklearsError::InvalidInput(
                "Time series too short for AR model".to_string(),
            ));
        }

        let features_per_series = n_features * order;
        let mut features = Array2::zeros((n_samples, features_per_series));

        for sample_idx in 0..n_samples {
            let mut feature_idx = 0;

            for feature_idx_orig in 0..n_features {
                // Extract time series for this feature
                let time_series = x.slice(s![sample_idx, feature_idx_orig, ..]);

                // Fit AR model (simplified - just use lagged values as features)
                for lag in 1..=order {
                    if n_time_steps > lag {
                        let lagged_mean =
                            time_series.slice(s![..n_time_steps - lag]).mean().unwrap();
                        features[[sample_idx, feature_idx]] = lagged_mean;
                        feature_idx += 1;
                    }
                }
            }
        }

        Ok(features)
    }

    fn extract_ma_features(&self, x: &Array3<Float>, order: usize) -> Result<Array2<Float>> {
        // Simplified MA feature extraction - use moving averages of different window sizes
        let (n_samples, n_features, n_time_steps) = x.dim();
        let features_per_series = n_features * order;
        let mut features = Array2::zeros((n_samples, features_per_series));

        for sample_idx in 0..n_samples {
            let mut feature_idx = 0;

            for feature_idx_orig in 0..n_features {
                let time_series = x.slice(s![sample_idx, feature_idx_orig, ..]);

                for window_size in 1..=order {
                    let mut moving_avg = 0.0;
                    let mut count = 0;

                    for t in window_size..n_time_steps {
                        let window_sum: Float = time_series.slice(s![t - window_size..t]).sum();
                        moving_avg += window_sum / window_size as Float;
                        count += 1;
                    }

                    if count > 0 {
                        features[[sample_idx, feature_idx]] = moving_avg / count as Float;
                    }
                    feature_idx += 1;
                }
            }
        }

        Ok(features)
    }

    fn extract_arma_features(
        &self,
        x: &Array3<Float>,
        ar_order: usize,
        ma_order: usize,
    ) -> Result<Array2<Float>> {
        // Combine AR and MA features
        let ar_features = self.extract_ar_features(x, ar_order)?;
        let ma_features = self.extract_ma_features(x, ma_order)?;

        let combined = concatenate![Axis(1), ar_features, ma_features];
        Ok(combined)
    }

    fn extract_hmm_features(&self, x: &Array3<Float>, n_states: usize) -> Result<Array2<Float>> {
        // Simplified HMM feature extraction - discretize values and create state sequences
        let (n_samples, n_features, n_time_steps) = x.dim();
        let features_per_series = n_features * n_states;
        let mut features = Array2::zeros((n_samples, features_per_series));

        for sample_idx in 0..n_samples {
            let mut feature_idx = 0;

            for feature_idx_orig in 0..n_features {
                let time_series = x.slice(s![sample_idx, feature_idx_orig, ..]);

                // Discretize into states based on quantiles
                let mut values: Vec<Float> = time_series.to_vec();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let mut state_counts = vec![0; n_states];

                for &value in time_series.iter() {
                    let state = self.value_to_state(value, &values, n_states);
                    state_counts[state] += 1;
                }

                // Normalize state counts to probabilities
                let total_count = n_time_steps as Float;
                for state in 0..n_states {
                    features[[sample_idx, feature_idx]] =
                        state_counts[state] as Float / total_count;
                    feature_idx += 1;
                }
            }
        }

        Ok(features)
    }

    fn value_to_state(&self, value: Float, sorted_values: &[Float], n_states: usize) -> usize {
        let n = sorted_values.len();
        if n == 0 {
            return 0;
        }

        for state in 0..n_states {
            let threshold_idx = ((state + 1) * n) / n_states;
            if threshold_idx >= n {
                return n_states - 1;
            }

            if value <= sorted_values[threshold_idx.min(n - 1)] {
                return state;
            }
        }

        n_states - 1
    }

    fn extract_recurrent_features(
        &self,
        x: &Array3<Float>,
        memory_length: usize,
    ) -> Result<Array2<Float>> {
        // Simplified recurrent feature extraction - use weighted history
        let (n_samples, n_features, n_time_steps) = x.dim();
        let features_per_series = n_features;
        let mut features = Array2::zeros((n_samples, features_per_series));

        for sample_idx in 0..n_samples {
            for feature_idx in 0..n_features {
                let time_series = x.slice(s![sample_idx, feature_idx, ..]);

                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for t in 0..n_time_steps {
                    let lookback = (memory_length.min(t + 1)) as Float;
                    let weight = 1.0 / lookback;
                    weighted_sum += weight * time_series[t];
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    features[[sample_idx, feature_idx]] = weighted_sum / total_weight;
                }
            }
        }

        Ok(features)
    }

    fn extract_state_space_features(
        &self,
        x: &Array3<Float>,
        state_dim: usize,
    ) -> Result<Array2<Float>> {
        // Simplified state space feature extraction
        let (n_samples, n_features, n_time_steps) = x.dim();
        let features_per_series = state_dim.min(n_features);
        let mut features = Array2::zeros((n_samples, features_per_series));

        for sample_idx in 0..n_samples {
            for state_idx in 0..features_per_series {
                let feature_idx = state_idx % n_features;
                let time_series = x.slice(s![sample_idx, feature_idx, ..]);

                // Use final state as feature (in full implementation, would use Kalman filtering)
                features[[sample_idx, state_idx]] = time_series[n_time_steps - 1];
            }
        }

        Ok(features)
    }

    fn decompose_temporal_components(
        &self,
        x: &Array3<Float>,
    ) -> Result<(Array2<Float>, Option<Array2<Float>>)> {
        let (n_samples, n_features, n_time_steps) = x.dim();

        // Trend analysis
        let mut trend_coefficients = Array2::zeros((n_samples, n_features));

        for sample_idx in 0..n_samples {
            for feature_idx in 0..n_features {
                let time_series = x.slice(s![sample_idx, feature_idx, ..]);
                let trend_coef = self.compute_trend_coefficient(&time_series)?;
                trend_coefficients[[sample_idx, feature_idx]] = trend_coef;
            }
        }

        // Seasonal analysis (if period specified)
        let seasonal_components = if let Some(period) = self.config.seasonality_period {
            let mut seasonal = Array2::zeros((n_samples, n_features));

            for sample_idx in 0..n_samples {
                for feature_idx in 0..n_features {
                    let time_series = x.slice(s![sample_idx, feature_idx, ..]);
                    let seasonal_strength = self.compute_seasonal_strength(&time_series, period)?;
                    seasonal[[sample_idx, feature_idx]] = seasonal_strength;
                }
            }

            Some(seasonal)
        } else {
            None
        };

        Ok((trend_coefficients, seasonal_components))
    }

    fn compute_trend_coefficient(&self, time_series: &ArrayView1<Float>) -> Result<Float> {
        match &self.config.trend_method {
            TrendMethod::Linear => self.compute_slope(time_series),
            TrendMethod::Polynomial { degree } => {
                // Simplified polynomial trend - just use linear for now
                self.compute_slope(time_series)
            }
            TrendMethod::Exponential => {
                // Simplified exponential trend
                let n = time_series.len();
                if n < 2 {
                    return Ok(0.0);
                }

                let first_val = time_series[0];
                let last_val = time_series[n - 1];

                if first_val <= 0.0 || last_val <= 0.0 {
                    Ok(0.0)
                } else {
                    Ok((last_val / first_val).ln() / n as Float)
                }
            }
            TrendMethod::MovingAverage { window } => {
                let n = time_series.len();
                if n < *window {
                    return Ok(0.0);
                }

                let early_avg = time_series.slice(s![..*window]).mean().unwrap();
                let late_avg = time_series.slice(s![n - window..]).mean().unwrap();

                Ok((late_avg - early_avg) / n as Float)
            }
            TrendMethod::None => Ok(0.0),
        }
    }

    fn compute_seasonal_strength(
        &self,
        time_series: &ArrayView1<Float>,
        period: usize,
    ) -> Result<Float> {
        let n = time_series.len();
        if n < period * 2 {
            return Ok(0.0);
        }

        let mut seasonal_variance = 0.0;
        let mut total_variance = 0.0;

        let overall_mean = time_series.mean().unwrap();

        // Compute seasonal means
        let mut seasonal_means = vec![0.0; period];
        let mut seasonal_counts = vec![0; period];

        for (i, &value) in time_series.iter().enumerate() {
            let seasonal_idx = i % period;
            seasonal_means[seasonal_idx] += value;
            seasonal_counts[seasonal_idx] += 1;
        }

        for i in 0..period {
            if seasonal_counts[i] > 0 {
                seasonal_means[i] /= seasonal_counts[i] as Float;
            }
        }

        // Compute variances
        for (i, &value) in time_series.iter().enumerate() {
            let seasonal_idx = i % period;
            let seasonal_mean = seasonal_means[seasonal_idx];

            seasonal_variance += (value - seasonal_mean).powi(2);
            total_variance += (value - overall_mean).powi(2);
        }

        if total_variance > 0.0 {
            Ok(1.0 - (seasonal_variance / total_variance))
        } else {
            Ok(0.0)
        }
    }

    fn fit_temporal_models(
        &self,
        x: &Array3<Float>,
    ) -> Result<(Option<Array2<Float>>, Option<Array2<Float>>)> {
        match &self.config.temporal_method {
            TemporalMethod::Autoregressive { order } => {
                let ar_coefs = self.fit_ar_model(x, *order)?;
                Ok((Some(ar_coefs), None))
            }
            TemporalMethod::MovingAverage { order } => {
                let ma_coefs = self.fit_ma_model(x, *order)?;
                Ok((None, Some(ma_coefs)))
            }
            TemporalMethod::ARMA { ar_order, ma_order } => {
                let ar_coefs = self.fit_ar_model(x, *ar_order)?;
                let ma_coefs = self.fit_ma_model(x, *ma_order)?;
                Ok((Some(ar_coefs), Some(ma_coefs)))
            }
            _ => Ok((None, None)),
        }
    }

    fn fit_ar_model(&self, x: &Array3<Float>, order: usize) -> Result<Array2<Float>> {
        let (n_samples, n_features, n_time_steps) = x.dim();
        let mut ar_coefficients = Array2::zeros((n_samples * n_features, order));

        let mut coef_idx = 0;

        for sample_idx in 0..n_samples {
            for feature_idx in 0..n_features {
                let time_series = x.slice(s![sample_idx, feature_idx, ..]);

                // Simplified AR fitting using least squares
                if n_time_steps > order {
                    for lag in 0..order {
                        let mut correlation = 0.0;
                        let mut count = 0;

                        for t in (lag + 1)..n_time_steps {
                            correlation += time_series[t] * time_series[t - lag - 1];
                            count += 1;
                        }

                        if count > 0 {
                            ar_coefficients[[coef_idx, lag]] = correlation / count as Float;
                        }
                    }
                }

                coef_idx += 1;
            }
        }

        Ok(ar_coefficients)
    }

    fn fit_ma_model(&self, x: &Array3<Float>, order: usize) -> Result<Array2<Float>> {
        let (n_samples, n_features, _n_time_steps) = x.dim();
        let mut ma_coefficients = Array2::zeros((n_samples * n_features, order));

        // Simplified MA model - use variance at different lags
        let mut coef_idx = 0;

        for sample_idx in 0..n_samples {
            for feature_idx in 0..n_features {
                let time_series = x.slice(s![sample_idx, feature_idx, ..]);

                for lag in 0..order {
                    // Simplified: use variance of differences
                    let mut variance = 0.0;
                    let mut count = 0;

                    for t in (lag + 1)..time_series.len() {
                        let diff = time_series[t] - time_series[t - lag - 1];
                        variance += diff * diff;
                        count += 1;
                    }

                    if count > 0 {
                        ma_coefficients[[coef_idx, lag]] = variance / count as Float;
                    }
                }

                coef_idx += 1;
            }
        }

        Ok(ma_coefficients)
    }

    fn learn_temporal_patterns(
        &self,
        x: &Array3<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<Vec<TemporalPattern>> {
        let (n_samples, n_features, n_time_steps) = x.dim();
        let mut patterns = Vec::new();

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&label| label == class_label).collect();
            let class_samples: Vec<usize> = class_mask
                .iter()
                .enumerate()
                .filter(|(_, &mask)| mask)
                .map(|(i, _)| i)
                .collect();

            if class_samples.is_empty() {
                continue;
            }

            // Compute average temporal pattern for this class
            let mut pattern_coeffs = Array1::zeros(n_features);
            let mut pattern_weights = Array1::zeros(n_features);

            for &sample_idx in &class_samples {
                for feature_idx in 0..n_features {
                    let time_series = x.slice(s![sample_idx, feature_idx, ..]);

                    // Simple pattern: variance and trend
                    let variance = time_series.var(0.0);
                    let trend = self.compute_slope(&time_series)?;

                    pattern_coeffs[feature_idx] += variance + trend.abs();
                    pattern_weights[feature_idx] += 1.0;
                }
            }

            // Normalize
            for feature_idx in 0..n_features {
                if pattern_weights[feature_idx] > 0.0 {
                    pattern_coeffs[feature_idx] /= pattern_weights[feature_idx];
                }
            }

            let pattern = TemporalPattern {
                id: class_idx,
                length: n_time_steps,
                coefficients: pattern_coeffs,
                weights: pattern_weights,
                class_label,
            };

            patterns.push(pattern);
        }

        Ok(patterns)
    }

    fn compute_temporal_statistics(
        &self,
        temporal_features: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<(Array2<Float>, Vec<Array2<Float>>)> {
        let n_features = temporal_features.ncols();
        let n_classes = classes.len();

        let mut temporal_means = Array2::zeros((n_classes, n_features));
        let mut temporal_covariances = Vec::new();

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let class_mask: Vec<bool> = y.iter().map(|&label| label == class_label).collect();
            let class_features: Vec<_> = temporal_features
                .axis_iter(Axis(0))
                .zip(class_mask.iter())
                .filter(|(_, &mask)| mask)
                .map(|(row, _)| row.to_owned())
                .collect();

            if !class_features.is_empty() {
                // Compute class mean
                let mut class_mean = Array1::zeros(n_features);
                for feature_row in &class_features {
                    class_mean += feature_row;
                }
                class_mean /= class_features.len() as Float;
                temporal_means.row_mut(class_idx).assign(&class_mean);

                // Compute class covariance
                let mut class_cov = Array2::zeros((n_features, n_features));
                for feature_row in &class_features {
                    let diff = feature_row - &class_mean;
                    class_cov += &diff
                        .clone()
                        .insert_axis(Axis(1))
                        .dot(&diff.insert_axis(Axis(0)));
                }

                if class_features.len() > 1 {
                    class_cov /= (class_features.len() - 1) as Float;
                }

                // Add regularization
                for i in 0..n_features {
                    class_cov[[i, i]] += self.config.regularization;
                }

                temporal_covariances.push(class_cov);
            } else {
                temporal_covariances.push(Array2::eye(n_features));
            }
        }

        Ok((temporal_means, temporal_covariances))
    }

    fn learn_state_space_model(
        &self,
        x: &Array3<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<(Option<Array2<Float>>, Option<Array2<Float>>)> {
        match &self.config.temporal_method {
            TemporalMethod::StateSpace { state_dim } => {
                let (n_samples, n_features, n_time_steps) = x.dim();

                // Simplified state space model - use identity matrices
                let state_transition = Array2::eye(*state_dim);
                let observation = Array2::eye(n_features.min(*state_dim));

                Ok((Some(state_transition), Some(observation)))
            }
            _ => Ok((None, None)),
        }
    }

    fn compute_temporal_discriminant_components(
        &self,
        temporal_features: &Array2<Float>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
        temporal_means: &Array2<Float>,
        temporal_covariances: &[Array2<Float>],
    ) -> Result<Array2<Float>> {
        let n_features = temporal_features.ncols();
        let n_components = self
            .config
            .n_temporal_components
            .unwrap_or((classes.len() - 1).min(n_features));

        // For simplicity, use identity matrix as components
        // In a full implementation, this would solve the generalized eigenvalue problem
        let components = Array2::eye(n_features)
            .slice(s![.., ..n_components])
            .to_owned();

        Ok(components)
    }
}

impl Estimator for TemporalDiscriminantAnalysis<Trained> {
    type Config = TemporalDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Predict<Array3<Float>, Array1<i32>> for TemporalDiscriminantAnalysis<Trained> {
    fn predict(&self, x: &Array3<Float>) -> Result<Array1<i32>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().unwrap();

        let mut predictions = Vec::new();
        for row in probas.axis_iter(Axis(0)) {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions.push(classes[max_idx]);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl PredictProba<Array3<Float>, Array2<Float>> for TemporalDiscriminantAnalysis<Trained> {
    fn predict_proba(&self, x: &Array3<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features, n_time_steps) = x.dim();
        let classes = self.classes_.as_ref().unwrap();
        let n_classes = classes.len();

        if Some(n_features) != self.n_features_ || Some(n_time_steps) != self.n_time_steps_ {
            return Err(SklearsError::InvalidInput(
                "Input dimensions don't match training data".to_string(),
            ));
        }

        // Standardize if needed
        let standardized_x =
            if let (Some(means), Some(stds)) = (&self.feature_means_, &self.feature_stds_) {
                let mut standardized = x.clone();
                for i in 0..n_samples {
                    for t in 0..n_time_steps {
                        for f in 0..n_features {
                            standardized[[i, f, t]] = (x[[i, f, t]] - means[f]) / stds[f];
                        }
                    }
                }
                standardized
            } else {
                x.clone()
            };

        // Extract temporal features
        let temporal_features = self.extract_temporal_features(&standardized_x)?;

        let temporal_means = self.temporal_means_.as_ref().unwrap();
        let temporal_patterns = self.temporal_patterns_.as_ref().unwrap();

        let mut probabilities = Array2::zeros((n_samples, n_classes));

        for sample_idx in 0..n_samples {
            let sample_features = temporal_features.row(sample_idx);
            let mut log_probs = Array1::zeros(n_classes);

            for (class_idx, pattern) in temporal_patterns.iter().enumerate() {
                // Compute likelihood based on temporal pattern matching
                let class_mean = temporal_means.row(class_idx);
                let diff = &sample_features - &class_mean;

                // Simplified likelihood computation
                let distance = diff.dot(&diff);
                log_probs[class_idx] = -distance;
            }

            // Convert to probabilities using softmax
            let max_log_prob = log_probs.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_probs: Array1<Float> = log_probs.mapv(|x| (x - max_log_prob).exp());
            let sum_exp = exp_probs.sum();

            for (class_idx, &exp_prob) in exp_probs.iter().enumerate() {
                probabilities[[sample_idx, class_idx]] = exp_prob / sum_exp;
            }
        }

        Ok(probabilities)
    }
}

impl Transform<Array3<Float>, Array2<Float>> for TemporalDiscriminantAnalysis<Trained> {
    fn transform(&self, x: &Array3<Float>) -> Result<Array2<Float>> {
        let temporal_components = self.temporal_components_.as_ref().unwrap();

        // Standardize if needed
        let (n_samples, n_features, n_time_steps) = x.dim();
        let standardized_x =
            if let (Some(means), Some(stds)) = (&self.feature_means_, &self.feature_stds_) {
                let mut standardized = x.clone();
                for i in 0..n_samples {
                    for t in 0..n_time_steps {
                        for f in 0..n_features {
                            standardized[[i, f, t]] = (x[[i, f, t]] - means[f]) / stds[f];
                        }
                    }
                }
                standardized
            } else {
                x.clone()
            };

        // Extract temporal features
        let temporal_features = self.extract_temporal_features(&standardized_x)?;

        // Transform using temporal components
        let transformed = temporal_features.dot(temporal_components);

        Ok(transformed)
    }
}

impl TemporalDiscriminantAnalysis<Trained> {
    /// Extract temporal features (reuse the untrained implementation)
    fn extract_temporal_features(&self, x: &Array3<Float>) -> Result<Array2<Float>> {
        let untrained_version = TemporalDiscriminantAnalysis::<Untrained> {
            config: self.config.clone(),
            state: PhantomData,
            classes_: None,
            temporal_patterns_: None,
            temporal_components_: None,
            temporal_means_: None,
            temporal_covariances_: None,
            trend_coefficients_: None,
            seasonal_components_: None,
            ar_coefficients_: None,
            ma_coefficients_: None,
            state_transition_matrix_: None,
            observation_matrix_: None,
            feature_means_: None,
            feature_stds_: None,
            n_features_: None,
            n_time_steps_: None,
        };
        untrained_version.extract_temporal_features(x)
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }

    /// Get temporal patterns
    pub fn temporal_patterns(&self) -> &[TemporalPattern] {
        self.temporal_patterns_.as_ref().unwrap()
    }

    /// Get temporal components
    pub fn temporal_components(&self) -> &Array2<Float> {
        self.temporal_components_.as_ref().unwrap()
    }

    /// Get temporal means for each class
    pub fn temporal_means(&self) -> &Array2<Float> {
        self.temporal_means_.as_ref().unwrap()
    }

    /// Get temporal covariances for each class
    pub fn temporal_covariances(&self) -> &[Array2<Float>] {
        self.temporal_covariances_.as_ref().unwrap()
    }

    /// Get trend coefficients
    pub fn trend_coefficients(&self) -> &Array2<Float> {
        self.trend_coefficients_.as_ref().unwrap()
    }

    /// Get seasonal components (if available)
    pub fn seasonal_components(&self) -> Option<&Array2<Float>> {
        self.seasonal_components_.as_ref()
    }

    /// Get AR coefficients (if available)
    pub fn ar_coefficients(&self) -> Option<&Array2<Float>> {
        self.ar_coefficients_.as_ref()
    }

    /// Get MA coefficients (if available)
    pub fn ma_coefficients(&self) -> Option<&Array2<Float>> {
        self.ma_coefficients_.as_ref()
    }

    /// Get state transition matrix (if available)
    pub fn state_transition_matrix(&self) -> Option<&Array2<Float>> {
        self.state_transition_matrix_.as_ref()
    }

    /// Get observation matrix (if available)
    pub fn observation_matrix(&self) -> Option<&Array2<Float>> {
        self.observation_matrix_.as_ref()
    }

    /// Get number of features
    pub fn n_features(&self) -> usize {
        self.n_features_.unwrap()
    }

    /// Get number of time steps
    pub fn n_time_steps(&self) -> usize {
        self.n_time_steps_.unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_temporal_discriminant_analysis_sliding_window() {
        // Create simple 3D time series data (samples, features, time)
        let x = Array3::from_shape_vec(
            (4, 2, 10),
            vec![
                // Sample 0, Feature 0: [1,2,3,4,5,6,7,8,9,10]
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                // Sample 0, Feature 1: [2,3,4,5,6,7,8,9,10,11]
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                // Sample 1, Feature 0: similar increasing pattern
                1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9,
                7.9, 8.9, 9.9, 10.9,
                // Sample 2, Feature 0: different pattern (higher values)
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                11.0, 12.0, 13.0, 14.0, 15.0, // Sample 3, Feature 0: similar to sample 2
                5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 5.9, 6.9, 7.9, 8.9, 9.9,
                10.9, 11.9, 12.9, 13.9, 14.9,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let tda = TemporalDiscriminantAnalysis::new()
            .window_size(5)
            .window_overlap(2)
            .temporal_method(TemporalMethod::SlidingWindow);

        let fitted = tda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
        assert!(fitted.temporal_patterns().len() >= 1);
    }

    #[test]
    fn test_temporal_discriminant_analysis_autoregressive() {
        let x = Array3::from_shape_vec(
            (4, 2, 8),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1,
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1,
                12.1, 13.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let tda = TemporalDiscriminantAnalysis::new()
            .window_size(6) // Set window size smaller than time series length (8)
            .temporal_method(TemporalMethod::Autoregressive { order: 3 });

        let fitted = tda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_temporal_transform() {
        let x = Array3::from_shape_vec(
            (2, 2, 6),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 1]);

        let tda = TemporalDiscriminantAnalysis::new()
            .window_size(5) // Set window size smaller than time series length (6)
            .n_temporal_components(Some(1));

        let fitted = tda.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.nrows(), 2);
        assert!(transformed.ncols() >= 1);
    }

    #[test]
    fn test_aggregation_methods() {
        let x = Array3::from_shape_vec(
            (2, 2, 6),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 1]);

        let methods = vec![
            AggregationMethod::Mean,
            AggregationMethod::Maximum,
            AggregationMethod::Minimum,
            AggregationMethod::StandardDeviation,
            AggregationMethod::Slope,
        ];

        for method in methods {
            let tda = TemporalDiscriminantAnalysis::new()
                .window_size(3)
                .aggregation_method(method)
                .temporal_method(TemporalMethod::SlidingWindow);

            let fitted = tda.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 2);
            assert_eq!(fitted.classes().len(), 2);
        }
    }

    #[test]
    fn test_trend_analysis() {
        let x = Array3::from_shape_vec(
            (2, 1, 8),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Linear increasing trend
                8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, // Linear decreasing trend
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 1]);

        let tda = TemporalDiscriminantAnalysis::new()
            .window_size(6) // Set window size smaller than time series length (8)
            .trend_method(TrendMethod::Linear);

        let fitted = tda.fit(&x, &y).unwrap();
        let trend_coefficients = fitted.trend_coefficients();

        assert_eq!(trend_coefficients.nrows(), 2);
        assert_eq!(trend_coefficients.ncols(), 1);

        // First sample should have positive trend, second should have negative
        assert!(trend_coefficients[[0, 0]] > 0.0);
        assert!(trend_coefficients[[1, 0]] < 0.0);
    }

    #[test]
    fn test_temporal_standardization() {
        let x = Array3::from_shape_vec(
            (2, 2, 4),
            vec![
                100.0, 200.0, 300.0, 400.0, // Large values
                1000.0, 2000.0, 3000.0, 4000.0, 1.0, 2.0, 3.0, 4.0, // Small values
                10.0, 20.0, 30.0, 40.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 1]);

        let tda = TemporalDiscriminantAnalysis::new()
            .window_size(3) // Set window size smaller than time series length (4)
            .standardize_temporal(true);

        let fitted = tda.fit(&x, &y).unwrap();

        // Should fit successfully even with very different scales
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.n_features(), 2);
        assert_eq!(fitted.n_time_steps(), 4);
    }
}
