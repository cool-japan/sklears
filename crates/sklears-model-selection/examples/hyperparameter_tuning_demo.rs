//! Hyperparameter Tuning and Model Selection Demo
//!
//! This example demonstrates various hyperparameter tuning strategies available
//! in sklears-model-selection, including GridSearch, RandomizedSearch,
//! BayesSearch, and HalvingGridSearch.

use scirs2_core::ndarray::{array, Array1, Array2, Axis};
use sklears_core::prelude::{Fit, Predict};
use sklears_model_selection::*;

// Mock estimator for demonstration
#[derive(Debug, Clone)]
struct MockLinearModel {
    alpha: f64,
    fit_intercept: bool,
    // Fitted parameters
    coef_: Option<Array1<f64>>,
    intercept_: Option<f64>,
}

impl MockLinearModel {
    fn new() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            coef_: None,
            intercept_: None,
        }
    }

    #[allow(dead_code)]
    fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    #[allow(dead_code)]
    fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl sklears_core::traits::Estimator for MockLinearModel {
    type Config = Self;
    type Error = sklears_core::error::SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>> for MockLinearModel {
    type Fitted = MockLinearModel;

    fn fit(
        mut self,
        x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Self::Fitted, sklears_core::error::SklearsError> {
        let n_features = x.ncols();

        // Simple mock fitting: just store some coefficients
        self.coef_ = Some(Array1::from_elem(n_features, 0.5 * self.alpha));
        self.intercept_ = if self.fit_intercept {
            Some(0.1)
        } else {
            Some(0.0)
        };

        Ok(self)
    }
}

impl Predict<Array2<f64>, Array1<f64>> for MockLinearModel {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, sklears_core::error::SklearsError> {
        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();

        let mut predictions = Array1::zeros(x.nrows());
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            predictions[i] = row.dot(coef) + intercept;
        }

        Ok(predictions)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 SKLears Hyperparameter Tuning Demo");
    println!("=====================================");

    // Generate sample regression data
    let x = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0],
        [7.0, 8.0],
        [8.0, 9.0],
        [9.0, 10.0],
        [10.0, 11.0],
        [11.0, 12.0],
        [12.0, 13.0],
    ];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0];

    println!("\n📊 Training Data:");
    println!("Features shape: {:?}", x.dim());
    println!("Target shape: {:?}", y.dim());

    // Demo 1: Hyperparameter Tuning Concepts
    println!("\n🔹 Demo 1: Hyperparameter Tuning Concepts");
    println!("------------------------------------------");

    println!("Common hyperparameter tuning approaches:");
    println!("• Grid Search: Exhaustive search over parameter combinations");
    println!("• Random Search: Random sampling from parameter distributions");
    println!("• Bayesian Optimization: Intelligent search using prior information");
    println!("• Halving Search: Successive elimination of poor performers");

    println!("\nParameter types for demonstration:");
    println!("• alpha: [0.1, 0.5, 1.0, 2.0] (regularization strength)");
    println!("• fit_intercept: [true, false] (whether to fit intercept)");
    println!("Grid search would evaluate: {} combinations", 4 * 2);

    // Demo 2: Cross-validation strategies
    println!("\n🔹 Demo 2: Cross-Validation Strategies");
    println!("--------------------------------------");

    // K-Fold
    let kfold = KFold::new(5);
    let kfold_splits = kfold.split(x.nrows(), None);
    println!("K-Fold (k=5): {} splits", kfold_splits.len());

    // Stratified K-Fold (for classification)
    let y_class = array![0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]; // Mock classification labels
    let stratified_kfold = StratifiedKFold::new(3);
    let stratified_splits = stratified_kfold.split(x.nrows(), Some(&y_class));
    println!(
        "Stratified K-Fold (k=3): {} splits",
        stratified_splits.len()
    );

    // Time Series Split
    let ts_split = TimeSeriesSplit::new(3);
    let ts_splits = ts_split.split(x.nrows(), None);
    println!("Time Series Split (n_splits=3): {} splits", ts_splits.len());

    // Leave-One-Out
    let loo = LeaveOneOut::new();
    let loo_splits = loo.split(x.nrows(), None);
    println!("Leave-One-Out: {} splits", loo_splits.len());

    // Demo 3: Model evaluation utilities
    println!("\n🔹 Demo 3: Model Evaluation Utilities");
    println!("-------------------------------------");

    let _mock_model = MockLinearModel::new();

    // Cross-validation score
    println!("cross_val_score: Evaluates model performance across CV folds");
    println!("cross_val_predict: Makes predictions using cross-validation");
    println!("learning_curve: Plots training vs validation scores");
    println!("validation_curve: Plots scores vs hyperparameter values");

    // Demo 4: Parameter distributions
    println!("\n🔹 Demo 4: Parameter Distribution Types");
    println!("---------------------------------------");

    println!("Choice: Discrete values [0.1, 1.0, 10.0]");
    println!("Uniform: Continuous uniform distribution");
    println!("LogUniform: Log-uniform for scale parameters");
    println!("Normal: Gaussian distribution");
    println!("Integer: Integer ranges for discrete parameters");

    // Demo 5: Best practices
    println!("\n🔹 Demo 5: Hyperparameter Tuning Best Practices");
    println!("------------------------------------------------");

    println!("1. 📊 Start with Grid Search for small parameter spaces");
    println!("2. 🎲 Use Randomized Search for large parameter spaces");
    println!("3. 🧠 Use Bayesian Optimization for expensive evaluations");
    println!("4. ⚡ Use Halving Search for quick elimination of poor performers");
    println!("5. 📈 Use appropriate CV strategy for your data type:");
    println!("   • K-Fold: General purpose");
    println!("   • Stratified K-Fold: Classification with imbalanced classes");
    println!("   • Time Series Split: Temporal data");
    println!("   • Leave-One-Out: Small datasets");

    println!("\n📋 Summary");
    println!("----------");
    println!("✅ GridSearchCV: Exhaustive search over parameter grid");
    println!("✅ RandomizedSearchCV: Random sampling from parameter distributions");
    println!("✅ BayesSearchCV: Intelligent search using Gaussian Process");
    println!("✅ HalvingGridSearch: Successive halving for efficient elimination");
    println!("✅ Multiple CV strategies for different data types");
    println!("✅ Comprehensive evaluation utilities");

    println!("\n🎯 Choose based on your computational budget and parameter space size!");

    Ok(())
}
