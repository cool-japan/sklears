//! Mock components for testing and demonstration

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Transform,
    types::Float,
};

use crate::{PipelinePredictor, PipelineStep};

/// Mock transformer for demonstration
#[derive(Debug, Clone)]
pub struct MockTransformer {
    scale_factor: f64,
}

impl MockTransformer {
    /// Create a new `MockTransformer`
    #[must_use]
    pub fn new() -> Self {
        Self { scale_factor: 1.0 }
    }

    /// Create with custom scale factor
    #[must_use]
    pub fn with_scale(scale_factor: f64) -> Self {
        Self { scale_factor }
    }
}

impl Default for MockTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for MockTransformer {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let transformed = x.mapv(|val| val * self.scale_factor);
        Ok(transformed)
    }
}

impl PipelineStep for MockTransformer {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        Transform::transform(self, x)
    }

    fn fit(
        &mut self,
        _x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<()> {
        // Mock fitting - could set scale_factor based on data
        Ok(())
    }

    fn clone_step(&self) -> Box<dyn PipelineStep> {
        Box::new(self.clone())
    }
}

/// Simple fit method for `MockTransformer`
impl MockTransformer {
    /// Fit the transformer (basic implementation)
    pub fn fit(
        &mut self,
        _x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<()> {
        Ok(())
    }
}

/// Mock predictor for testing and demonstration
#[derive(Debug, Clone)]
pub struct MockPredictor {
    coefficients: Option<Array1<f64>>,
    intercept: f64,
    fitted: bool,
}

impl MockPredictor {
    /// Create a new `MockPredictor`
    #[must_use]
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            fitted: false,
        }
    }

    /// Create with custom coefficients
    #[must_use]
    pub fn with_coefficients(coefficients: Array1<f64>, intercept: f64) -> Self {
        Self {
            coefficients: Some(coefficients),
            intercept,
            fitted: true,
        }
    }

    /// Check if the predictor is fitted
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Default for MockPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelinePredictor for MockPredictor {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        if !self.fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            });
        }

        let predictions = if let Some(ref coef) = self.coefficients {
            // Simple linear prediction: X @ coef + intercept
            let mut predictions = Array1::zeros(x.nrows());
            for i in 0..x.nrows() {
                let mut sum = self.intercept;
                for j in 0..x.ncols().min(coef.len()) {
                    sum += x[[i, j]] * coef[j];
                }
                predictions[i] = sum;
            }
            predictions
        } else {
            // Default: return mean of input rows as prediction
            let mut predictions = Array1::zeros(x.nrows());
            for i in 0..x.nrows() {
                predictions[i] = x.row(i).mapv(|v| v).mean().unwrap_or(0.0) + self.intercept;
            }
            predictions
        };

        Ok(predictions)
    }

    fn fit(&mut self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<()> {
        // Simple fitting: use mean of features as coefficients
        let n_features = x.ncols();
        let mut coefficients = Array1::zeros(n_features);

        for j in 0..n_features {
            coefficients[j] = x.column(j).mapv(|v| v).mean().unwrap_or(0.0) / n_features as f64;
        }

        self.coefficients = Some(coefficients);
        self.intercept = y.mapv(|v| v).mean().unwrap_or(0.0);
        self.fitted = true;

        Ok(())
    }

    fn clone_predictor(&self) -> Box<dyn PipelinePredictor> {
        Box::new(self.clone())
    }
}
