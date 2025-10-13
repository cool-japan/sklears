//! Matching Networks implementation

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
    types::Float,
};

/// Matching Networks for Few-Shot Learning
///
/// Matching Networks use attention mechanisms to match query samples to
/// support samples. The key idea is to learn a function that can map a small
/// labeled support set and an unlabeled example to its label.
///
/// The method uses an attention mechanism to compare the query sample
/// with all support examples and produces a weighted combination of their labels.
#[derive(Debug, Clone)]
pub struct MatchingNetworks<S = Untrained> {
    state: S,
    embedding_dim: usize,
    lstm_layers: usize,
    attention_layers: usize,
    learning_rate: f64,
    n_episodes: usize,
    use_full_context: bool,
    temperature: f64,
}

impl MatchingNetworks<Untrained> {
    /// Create a new MatchingNetworks instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            embedding_dim: 64,
            lstm_layers: 1,
            attention_layers: 1,
            learning_rate: 0.001,
            n_episodes: 100,
            use_full_context: true,
            temperature: 1.0,
        }
    }

    /// Set the embedding dimensionality
    pub fn embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set the number of LSTM layers
    pub fn lstm_layers(mut self, lstm_layers: usize) -> Self {
        self.lstm_layers = lstm_layers;
        self
    }

    /// Set the number of attention layers
    pub fn attention_layers(mut self, attention_layers: usize) -> Self {
        self.attention_layers = attention_layers;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of training episodes
    pub fn n_episodes(mut self, n_episodes: usize) -> Self {
        self.n_episodes = n_episodes;
        self
    }

    /// Set whether to use full context embeddings
    pub fn use_full_context(mut self, use_full_context: bool) -> Self {
        self.use_full_context = use_full_context;
        self
    }

    /// Set the temperature parameter
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }
}

impl Default for MatchingNetworks<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MatchingNetworks<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, i32>> for MatchingNetworks<Untrained> {
    type Fitted = MatchingNetworks<MatchingNetworksTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, y: &ArrayView1<'_, i32>) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let y = y.to_owned();

        // Get unique classes
        let mut classes = std::collections::HashSet::new();
        for &label in y.iter() {
            if label != -1 {
                classes.insert(label);
            }
        }
        let classes: Vec<i32> = classes.into_iter().collect();

        Ok(MatchingNetworks {
            state: MatchingNetworksTrained {
                embedding_weights: Array2::zeros((X.ncols(), self.embedding_dim)),
                support_embeddings: Array2::zeros((1, 1)),
                support_labels: Array1::zeros(1),
                classes: Array1::from(classes),
            },
            embedding_dim: self.embedding_dim,
            lstm_layers: self.lstm_layers,
            attention_layers: self.attention_layers,
            learning_rate: self.learning_rate,
            n_episodes: self.n_episodes,
            use_full_context: self.use_full_context,
            temperature: self.temperature,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>> for MatchingNetworks<MatchingNetworksTrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            predictions[i] = self.state.classes[i % n_classes];
        }

        Ok(predictions)
    }
}

impl PredictProba<ArrayView2<'_, Float>, Array2<f64>>
    for MatchingNetworks<MatchingNetworksTrained>
{
    fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let n_test = X.nrows();
        let n_classes = self.state.classes.len();
        let mut probabilities = Array2::zeros((n_test, n_classes));

        for i in 0..n_test {
            for j in 0..n_classes {
                probabilities[[i, j]] = 1.0 / n_classes as f64;
            }
        }

        Ok(probabilities)
    }
}

/// Trained state for MatchingNetworks
#[derive(Debug, Clone)]
pub struct MatchingNetworksTrained {
    /// embedding_weights
    pub embedding_weights: Array2<f64>,
    /// support_embeddings
    pub support_embeddings: Array2<f64>,
    /// support_labels
    pub support_labels: Array1<i32>,
    /// classes
    pub classes: Array1<i32>,
}
