/// Example implementations showing how to use the validation framework
use crate::error::Result;
use crate::types::Float;
#[cfg(test)]
use crate::validation::ValidationContext;
use crate::validation::{ConfigValidation, Validate};

/// Example configuration with manual validation implementation
#[derive(Debug, Clone)]
pub struct LinearRegressionConfig {
    /// Learning rate for gradient descent
    pub learning_rate: Float,

    /// L2 regularization parameter
    pub alpha: Float,

    /// Maximum number of iterations
    pub max_iter: usize,

    /// Convergence tolerance
    pub tol: Float,

    /// Whether to fit intercept
    pub fit_intercept: bool,

    /// Solver method
    pub solver: String,
}

impl Validate for LinearRegressionConfig {
    fn validate(&self) -> Result<()> {
        // Validate learning rate
        crate::validation::ml::validate_learning_rate(self.learning_rate)?;

        // Validate regularization
        crate::validation::ml::validate_regularization(self.alpha)?;

        // Validate max_iter
        crate::validation::ml::validate_max_iter(self.max_iter)?;

        // Validate tolerance
        crate::validation::ValidationRules::new("tol")
            .add_rule(crate::validation::ValidationRule::Positive)
            .add_rule(crate::validation::ValidationRule::Finite)
            .validate_numeric(&self.tol)?;

        // Validate solver
        crate::validation::ValidationRules::new("solver")
            .add_rule(crate::validation::ValidationRule::OneOf(vec![
                "auto".to_string(),
                "svd".to_string(),
                "cholesky".to_string(),
                "lsqr".to_string(),
                "sparse_cg".to_string(),
                "sag".to_string(),
                "saga".to_string(),
            ]))
            .validate_string(&self.solver)?;

        Ok(())
    }
}

impl Default for LinearRegressionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            alpha: 1.0,
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            solver: "auto".to_string(),
        }
    }
}

impl ConfigValidation for LinearRegressionConfig {
    fn validate_config(&self) -> Result<()> {
        // First run basic validation
        self.validate()?;

        // Add algorithm-specific validation
        if self.solver == "cholesky" && !self.fit_intercept {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "solver".to_string(),
                reason: "cholesky solver requires fit_intercept=true".to_string(),
            });
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.learning_rate > 0.1 {
            warnings
                .push("Learning rate is quite high, consider using a smaller value".to_string());
        }

        if self.max_iter < 100 {
            warnings.push("Maximum iterations is quite low, model may not converge".to_string());
        }

        warnings
    }
}

/// Example clustering configuration
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,

    /// Maximum number of iterations  
    pub max_iter: usize,

    /// Convergence tolerance
    pub tol: Float,

    /// Initialization method
    pub init: String,

    /// Number of random initializations
    pub n_init: usize,

    /// Random seed
    pub random_state: Option<u64>,
}

impl Validate for KMeansConfig {
    fn validate(&self) -> Result<()> {
        // Validate n_clusters
        crate::validation::ml::validate_n_clusters(self.n_clusters, 100)?;

        // Validate max_iter
        crate::validation::ml::validate_max_iter(self.max_iter)?;

        // Validate tolerance
        crate::validation::ValidationRules::new("tol")
            .add_rule(crate::validation::ValidationRule::Positive)
            .add_rule(crate::validation::ValidationRule::Finite)
            .validate_numeric(&self.tol)?;

        // Validate initialization method
        crate::validation::ValidationRules::new("init")
            .add_rule(crate::validation::ValidationRule::OneOf(vec![
                "k-means++".to_string(),
                "random".to_string(),
                "custom".to_string(),
            ]))
            .validate_string(&self.init)?;

        // Validate n_init
        if self.n_init == 0 {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "n_init".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        Ok(())
    }
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            max_iter: 300,
            tol: 1e-4,
            init: "k-means++".to_string(),
            n_init: 10,
            random_state: None,
        }
    }
}

impl ConfigValidation for KMeansConfig {
    fn validate_config(&self) -> Result<()> {
        self.validate()?;

        // Additional validation for clustering
        if self.n_clusters == 1 {
            log::warn!("Using only 1 cluster - consider if clustering is necessary");
        }

        Ok(())
    }
}

/// Example neural network configuration with complex validation
#[derive(Debug, Clone)]
pub struct MLPConfig {
    /// Hidden layer sizes
    pub hidden_layer_sizes: Vec<usize>,

    /// Learning rate
    pub learning_rate: Float,

    /// Maximum number of iterations
    pub max_iter: usize,

    /// Dropout probability
    pub dropout: Float,

    /// Batch size
    pub batch_size: usize,

    /// L2 regularization
    pub alpha: Float,

    /// Activation function
    pub activation: String,

    /// Solver
    pub solver: String,
}

impl Validate for MLPConfig {
    fn validate(&self) -> Result<()> {
        // Validate hidden layer sizes
        crate::validation::ValidationRules::new("hidden_layer_sizes")
            .add_rule(crate::validation::ValidationRule::MinLength(1))
            .validate_array(&self.hidden_layer_sizes)?;

        // Validate learning rate
        crate::validation::ml::validate_learning_rate(self.learning_rate)?;

        // Validate max_iter
        crate::validation::ml::validate_max_iter(self.max_iter)?;

        // Validate dropout probability
        crate::validation::ml::validate_probability(self.dropout)?;

        // Validate batch size
        if self.batch_size == 0 {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "batch_size".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        // Validate regularization
        crate::validation::ml::validate_regularization(self.alpha)?;

        // Validate activation function
        crate::validation::ValidationRules::new("activation")
            .add_rule(crate::validation::ValidationRule::OneOf(vec![
                "relu".to_string(),
                "tanh".to_string(),
                "sigmoid".to_string(),
                "identity".to_string(),
            ]))
            .validate_string(&self.activation)?;

        // Validate solver
        crate::validation::ValidationRules::new("solver")
            .add_rule(crate::validation::ValidationRule::OneOf(vec![
                "adam".to_string(),
                "sgd".to_string(),
                "lbfgs".to_string(),
            ]))
            .validate_string(&self.solver)?;

        Ok(())
    }
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            hidden_layer_sizes: vec![100],
            learning_rate: 0.001,
            max_iter: 200,
            dropout: 0.0,
            batch_size: 32,
            alpha: 1e-4,
            activation: "relu".to_string(),
            solver: "adam".to_string(),
        }
    }
}

impl ConfigValidation for MLPConfig {
    fn validate_config(&self) -> Result<()> {
        self.validate()?;

        // Complex validation logic
        if self.solver == "lbfgs" && self.hidden_layer_sizes.len() > 3 {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "solver".to_string(),
                reason: "lbfgs solver may be inefficient for deep networks".to_string(),
            });
        }

        if self.batch_size > 1000 {
            log::warn!("Large batch size may lead to poor generalization");
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.hidden_layer_sizes.iter().any(|&size| size > 1000) {
            warnings.push("Very large hidden layers may cause overfitting".to_string());
        }

        if self.dropout > 0.5 {
            warnings.push("High dropout rate may prevent learning".to_string());
        }

        warnings
    }
}

/// Example of manual validation implementation for complex cases
pub struct CustomValidationExample {
    pub param1: Float,
    pub param2: usize,
    pub dependent_param: Float,
}

impl Validate for CustomValidationExample {
    fn validate(&self) -> Result<()> {
        // Basic validations
        if self.param1 <= 0.0 {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "param1".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        // Complex cross-parameter validation
        if self.param2 > 0 && self.dependent_param > self.param1 * 2.0 {
            return Err(crate::error::SklearsError::InvalidParameter {
                name: "dependent_param".to_string(),
                reason: "cannot be more than twice param1 when param2 > 0".to_string(),
            });
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_config_validation() {
        let mut config = LinearRegressionConfig::default();

        // Valid configuration
        assert!(config.validate().is_ok());

        // Invalid learning rate
        config.learning_rate = -0.1;
        assert!(config.validate().is_err());

        // Reset and test invalid solver
        config = LinearRegressionConfig::default();
        config.solver = "invalid_solver".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_kmeans_config_validation() {
        let mut config = KMeansConfig::default();

        // Valid configuration
        assert!(config.validate().is_ok());

        // Invalid n_clusters
        config.n_clusters = 0;
        assert!(config.validate().is_err());

        // Reset and test invalid tolerance
        config = KMeansConfig::default();
        config.tol = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mlp_config_validation() {
        let mut config = MLPConfig::default();

        // Valid configuration
        assert!(config.validate().is_ok());

        // Invalid hidden layer sizes (empty)
        config.hidden_layer_sizes = vec![];
        assert!(config.validate().is_err());

        // Reset and test invalid dropout
        config = MLPConfig::default();
        config.dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_trait() {
        let config = LinearRegressionConfig::default();

        // Test config validation
        assert!(config.validate_config().is_ok());

        // Test warnings
        let warnings = config.get_warnings();
        // Should be empty for default config
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validation_context() {
        let context =
            ValidationContext::new("LinearRegression", "fit").with_data_info(100, 5, "float64");

        let error = crate::error::SklearsError::InvalidParameter {
            name: "learning_rate".to_string(),
            reason: "must be positive".to_string(),
        };

        let formatted = context.format_error(&error);
        assert!(formatted.contains("LinearRegression"));
        assert!(formatted.contains("fit"));
        assert!(formatted.contains("100 samples"));
        assert!(formatted.contains("5 features"));
    }

    #[test]
    fn test_custom_validation() {
        let example = CustomValidationExample {
            param1: 1.0,
            param2: 0,
            dependent_param: 1.5,
        };

        // Should be valid
        assert!(example.validate().is_ok());

        let example2 = CustomValidationExample {
            param1: 1.0,
            param2: 1,
            dependent_param: 3.0, // > 2 * param1
        };

        // Should be invalid due to cross-parameter constraint
        assert!(example2.validate().is_err());
    }
}
