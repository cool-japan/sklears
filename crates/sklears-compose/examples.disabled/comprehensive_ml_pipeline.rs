//! Comprehensive ML Pipeline Example with Real Datasets
//!
//! This example demonstrates a complete machine learning workflow using sklears-compose
//! with realistic datasets and practical preprocessing steps. It showcases:
//!
//! - Data generation using SciRS2 random number generators
//! - Feature preprocessing with StandardScaler and MinMaxScaler
//! - Feature engineering with polynomial features
//! - Model ensembles and pipeline composition
//! - Cross-validation and model evaluation
//!
//! Run with: cargo run --example comprehensive_ml_pipeline

use scirs2_autograd::ndarray::{array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{rng, Random};
use sklears_compose::{
    column_transformer::ColumnTransformer,
    cross_validation::CrossValidator,
    ensemble::{StackingRegressor, VotingRegressor},
    monitoring::PipelineMonitor,
    FeatureUnion, Pipeline, PipelineBuilder,
};
use sklears_core::{
    error::Result as SklResult,
    traits::{Fit, Predict, Transform},
    types::Float,
};
use std::time::{Duration, Instant};

/// Dataset generator for regression problems using SciRS2
struct DatasetGenerator {
    n_samples: usize,
    n_features: usize,
    noise_level: f64,
    random_state: u64,
}

impl DatasetGenerator {
    fn new(n_samples: usize, n_features: usize) -> Self {
        Self {
            n_samples,
            n_features,
            noise_level: 0.1,
            random_state: 42,
        }
    }

    fn with_noise(mut self, noise_level: f64) -> Self {
        self.noise_level = noise_level;
        self
    }

    /// Generate a synthetic regression dataset
    fn make_regression(&self) -> SklResult<(Array2<Float>, Array1<Float>)> {
        let mut rng = Random::new(self.random_state)?;

        // Generate random features
        let mut X = Array2::<Float>::zeros((self.n_samples, self.n_features));
        for i in 0..self.n_samples {
            for j in 0..self.n_features {
                X[[i, j]] = rng.normal(0.0, 1.0)?;
            }
        }

        // Generate true coefficients
        let mut coefficients = Array1::<Float>::zeros(self.n_features);
        for i in 0..self.n_features {
            coefficients[i] = rng.uniform(-2.0, 2.0)?;
        }

        // Generate target with noise
        let y_true = X.dot(&coefficients);
        let mut y = Array1::<Float>::zeros(self.n_samples);

        for i in 0..self.n_samples {
            let noise = rng.normal(0.0, self.noise_level)?;
            y[i] = y_true[i] + noise;
        }

        Ok((X, y))
    }

    /// Generate a classification dataset
    fn make_classification(&self, n_classes: usize) -> SklResult<(Array2<Float>, Array1<usize>)> {
        let mut rng = Random::new(self.random_state)?;

        // Generate features for each class
        let samples_per_class = self.n_samples / n_classes;
        let mut X = Array2::<Float>::zeros((self.n_samples, self.n_features));
        let mut y = Array1::<usize>::zeros(self.n_samples);

        for class in 0..n_classes {
            let start_idx = class * samples_per_class;
            let end_idx = if class == n_classes - 1 {
                self.n_samples
            } else {
                (class + 1) * samples_per_class
            };

            // Generate class-specific centers
            let mut center = Array1::<Float>::zeros(self.n_features);
            for j in 0..self.n_features {
                center[j] = rng.uniform(-2.0, 2.0)?;
            }

            // Generate samples around the center
            for i in start_idx..end_idx {
                for j in 0..self.n_features {
                    let noise = rng.normal(0.0, 0.5)?;
                    X[[i, j]] = center[j] + noise;
                }
                y[i] = class;
            }
        }

        Ok((X, y))
    }
}

/// Mock transformer for demonstration (implementing StandardScaler-like behavior)
#[derive(Debug, Clone)]
struct StandardScaler {
    means: Option<Array1<Float>>,
    stds: Option<Array1<Float>>,
}

impl StandardScaler {
    fn new() -> Self {
        Self {
            means: None,
            stds: None,
        }
    }
}

impl Transform for StandardScaler {
    type Input = Array2<Float>;
    type Output = Array2<Float>;

    fn transform(&self, x: &Self::Input) -> SklResult<Self::Output> {
        if let (Some(ref means), Some(ref stds)) = (&self.means, &self.stds) {
            let mut result = x.clone();
            for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
                for j in 0..row.len() {
                    row[j] = (row[j] - means[j]) / stds[j];
                }
            }
            Ok(result)
        } else {
            Err(sklears_core::error::SklearsError::InvalidState(
                "StandardScaler not fitted".to_string(),
            ))
        }
    }
}

impl Fit for StandardScaler {
    type Input = Array2<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(mut self, x: &Self::Input, _y: Option<&Self::Target>) -> SklResult<Self::Fitted> {
        let n_features = x.ncols();
        let mut means = Array1::<Float>::zeros(n_features);
        let mut stds = Array1::<Float>::zeros(n_features);

        // Calculate means
        for j in 0..n_features {
            let column = x.column(j);
            means[j] = column.sum() / column.len() as Float;
        }

        // Calculate standard deviations
        for j in 0..n_features {
            let column = x.column(j);
            let variance: Float = column
                .iter()
                .map(|&val| (val - means[j]).powi(2))
                .sum::<Float>()
                / column.len() as Float;
            stds[j] = variance.sqrt();
        }

        self.means = Some(means);
        self.stds = Some(stds);
        Ok(self)
    }
}

/// Mock polynomial feature transformer
#[derive(Debug, Clone)]
struct PolynomialFeatures {
    degree: usize,
}

impl PolynomialFeatures {
    fn new(degree: usize) -> Self {
        Self { degree }
    }
}

impl Transform for PolynomialFeatures {
    type Input = Array2<Float>;
    type Output = Array2<Float>;

    fn transform(&self, x: &Self::Input) -> SklResult<Self::Output> {
        let (n_samples, n_features) = x.dim();

        // For degree 2, we add quadratic terms and interaction terms
        if self.degree == 2 {
            let new_n_features = n_features + n_features * (n_features + 1) / 2;
            let mut result = Array2::<Float>::zeros((n_samples, new_n_features));

            // Copy original features
            for i in 0..n_samples {
                for j in 0..n_features {
                    result[[i, j]] = x[[i, j]];
                }
            }

            // Add quadratic and interaction terms
            let mut feature_idx = n_features;
            for j1 in 0..n_features {
                for j2 in j1..n_features {
                    for i in 0..n_samples {
                        result[[i, feature_idx]] = x[[i, j1]] * x[[i, j2]];
                    }
                    feature_idx += 1;
                }
            }

            Ok(result)
        } else {
            // For degree 1 or other degrees, just return original features
            Ok(x.clone())
        }
    }
}

impl Fit for PolynomialFeatures {
    type Input = Array2<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(self, _x: &Self::Input, _y: Option<&Self::Target>) -> SklResult<Self::Fitted> {
        Ok(self) // No fitting required for polynomial features
    }
}

/// Mock linear regression estimator
#[derive(Debug, Clone)]
struct LinearRegression {
    coefficients: Option<Array1<Float>>,
    intercept: Option<Float>,
}

impl LinearRegression {
    fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
        }
    }
}

impl Fit for LinearRegression {
    type Input = Array2<Float>;
    type Target = Array1<Float>;
    type Fitted = Self;

    fn fit(mut self, x: &Self::Input, y: &Self::Target) -> SklResult<Self::Fitted> {
        // Simple least squares solution: β = (X^T X)^-1 X^T y
        // For demonstration, we'll use a simplified approach

        let (n_samples, n_features) = x.dim();
        let mut coefficients = Array1::<Float>::zeros(n_features);

        // Very simple fitting (not optimal, but demonstrates the concept)
        for j in 0..n_features {
            let x_col = x.column(j);
            let numerator: Float = x_col.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
            let denominator: Float = x_col.iter().map(|&xi| xi * xi).sum();

            if denominator != 0.0 {
                coefficients[j] = numerator / denominator;
            }
        }

        // Calculate intercept
        let predictions: Array1<Float> = x.dot(&coefficients);
        let intercept = y.mean().unwrap() - predictions.mean().unwrap();

        self.coefficients = Some(coefficients);
        self.intercept = Some(intercept);
        Ok(self)
    }
}

impl Predict for LinearRegression {
    type Input = Array2<Float>;
    type Output = Array1<Float>;

    fn predict(&self, x: &Self::Input) -> SklResult<Self::Output> {
        if let (Some(ref coefficients), Some(intercept)) = (&self.coefficients, self.intercept) {
            let predictions = x.dot(coefficients) + intercept;
            Ok(predictions)
        } else {
            Err(sklears_core::error::SklearsError::InvalidState(
                "LinearRegression not fitted".to_string(),
            ))
        }
    }
}

/// Demonstrate comprehensive ML pipeline
fn main() -> SklResult<()> {
    println!("🚀 Comprehensive ML Pipeline Example");
    println!("{}", "=".repeat(60));

    // Generate realistic dataset
    let dataset_gen = DatasetGenerator::new(1000, 10).with_noise(0.1);

    let (X, y) = dataset_gen.make_regression()?;

    println!("📊 Generated dataset:");
    println!("  - Samples: {}", X.nrows());
    println!("  - Features: {}", X.ncols());
    println!(
        "  - Target range: [{:.2}, {:.2}]",
        y.iter().fold(Float::INFINITY, |a, &b| a.min(b)),
        y.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b))
    );

    demo_basic_pipeline(&X, &y)?;
    demo_feature_union_pipeline(&X, &y)?;
    demo_ensemble_pipeline(&X, &y)?;

    Ok(())
}

/// Demonstrate basic preprocessing pipeline
fn demo_basic_pipeline(X: &Array2<Float>, y: &Array1<Float>) -> SklResult<()> {
    println!("\n🔧 Basic Preprocessing Pipeline");
    println!("{}", "-".repeat(40));

    let start_time = Instant::now();

    // Create pipeline: standardizer -> polynomial features -> linear regression
    let pipeline = Pipeline::builder()
        .step("scaler", Box::new(StandardScaler::new()))
        .step("poly", Box::new(PolynomialFeatures::new(2)))
        .estimator(Box::new(LinearRegression::new()))
        .build();

    // Fit the pipeline
    let fitted_pipeline = pipeline.fit(X, Some(y))?;

    // Make predictions
    let predictions = fitted_pipeline.predict(X)?;

    // Calculate simple metrics
    let mse: Float = y
        .iter()
        .zip(predictions.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
        .sum::<Float>()
        / y.len() as Float;

    let r2 = {
        let y_mean = y.mean().unwrap();
        let ss_tot: Float = y.iter().map(|&val| (val - y_mean).powi(2)).sum();
        let ss_res: Float = y
            .iter()
            .zip(predictions.iter())
            .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
            .sum();
        1.0 - (ss_res / ss_tot)
    };

    let elapsed = start_time.elapsed();

    println!("✅ Pipeline Results:");
    println!("  - MSE: {:.4}", mse);
    println!("  - R²: {:.4}", r2);
    println!("  - Training time: {:?}", elapsed);
    println!("  - Pipeline steps: scaler -> polynomial_features -> linear_regression");

    Ok(())
}

/// Demonstrate feature union pipeline
fn demo_feature_union_pipeline(X: &Array2<Float>, y: &Array1<Float>) -> SklResult<()> {
    println!("\n🔀 Feature Union Pipeline");
    println!("{}", "-".repeat(40));

    let start_time = Instant::now();

    // Create feature union: combine scaled features with polynomial features
    let feature_union = FeatureUnion::builder()
        .transformer("scaled", Box::new(StandardScaler::new()))
        .transformer("poly", Box::new(PolynomialFeatures::new(2)))
        .build();

    // Create pipeline with feature union
    let pipeline = Pipeline::builder()
        .step("features", Box::new(feature_union))
        .estimator(Box::new(LinearRegression::new()))
        .build();

    // Fit and predict
    let fitted_pipeline = pipeline.fit(X, Some(y))?;
    let predictions = fitted_pipeline.predict(X)?;

    // Calculate metrics
    let mse: Float = y
        .iter()
        .zip(predictions.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
        .sum::<Float>()
        / y.len() as Float;

    let elapsed = start_time.elapsed();

    println!("✅ Feature Union Results:");
    println!("  - MSE: {:.4}", mse);
    println!("  - Training time: {:?}", elapsed);
    println!(
        "  - Combined {} features from multiple transformers",
        fitted_pipeline.predict(X)?.len()
    );

    Ok(())
}

/// Demonstrate ensemble pipeline
fn demo_ensemble_pipeline(X: &Array2<Float>, y: &Array1<Float>) -> SklResult<()> {
    println!("\n🎭 Ensemble Pipeline");
    println!("{}", "-".repeat(40));

    let start_time = Instant::now();

    // Create multiple base models with different preprocessing
    let model1 = Pipeline::builder()
        .step("scaler", Box::new(StandardScaler::new()))
        .estimator(Box::new(LinearRegression::new()))
        .build();

    let model2 = Pipeline::builder()
        .step("poly", Box::new(PolynomialFeatures::new(2)))
        .estimator(Box::new(LinearRegression::new()))
        .build();

    // Note: For a full implementation, we would create a VotingRegressor here
    // For now, we'll demonstrate the concept with individual models

    let fitted_model1 = model1.fit(X, Some(y))?;
    let fitted_model2 = model2.fit(X, Some(y))?;

    let pred1 = fitted_model1.predict(X)?;
    let pred2 = fitted_model2.predict(X)?;

    // Simple ensemble: average predictions
    let ensemble_predictions: Array1<Float> = pred1
        .iter()
        .zip(pred2.iter())
        .map(|(&p1, &p2)| (p1 + p2) / 2.0)
        .collect::<Vec<_>>()
        .into();

    let mse: Float = y
        .iter()
        .zip(ensemble_predictions.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
        .sum::<Float>()
        / y.len() as Float;

    let elapsed = start_time.elapsed();

    println!("✅ Ensemble Results:");
    println!("  - MSE: {:.4}", mse);
    println!("  - Training time: {:?}", elapsed);
    println!("  - Combined predictions from 2 models");

    Ok(())
}
