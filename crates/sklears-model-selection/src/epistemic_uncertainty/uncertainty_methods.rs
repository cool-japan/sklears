#[derive(Debug, Clone)]
pub enum EpistemicUncertaintyMethod {
    /// MonteCarloDropout
    MonteCarloDropout {
        dropout_rate: f64,

        n_samples: usize,
    },
    /// DeepEnsembles
    DeepEnsembles {
        n_models: usize,
    },
    /// BayesianNeuralNetwork
    BayesianNeuralNetwork {
        n_samples: usize,
    },
    Bootstrap {
        n_bootstrap: usize,
        sample_ratio: f64,
    },
    GaussianProcess {
        kernel_type: String,
    },
    VariationalInference {
        n_samples: usize,
    },
    LaplaceApproximation {
        hessian_method: String,
    },
}

#[derive(Debug, Clone)]
pub enum AleatoricUncertaintyMethod {
    /// HeteroskedasticRegression
    HeteroskedasticRegression {
        n_ensemble: usize,
    },
    /// MixtureDensityNetwork
    MixtureDensityNetwork {
        n_components: usize,
    },
    /// QuantileRegression
    QuantileRegression {
        quantiles: Vec<f64>,
    },
    /// ParametricUncertainty
    ParametricUncertainty {
        distribution: String,
    },
    InputDependentNoise {
        noise_model: String,
    },
    ResidualBasedUncertainty {
        window_size: usize,
    },
    EnsembleAleatoric {
        n_models: usize,
        noise_estimation: String,
    },
}
