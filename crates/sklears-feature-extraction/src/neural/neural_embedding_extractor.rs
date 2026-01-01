use super::neural_types::*;
use super::neural_activations::*;
use super::neural_utilities::*;

pub struct NeuralEmbeddingExtractor {
    pub embedding_dim: usize,
    pub vocab_size: usize,
    pub learning_rate: f64,
    pub n_epochs: usize,
    pub batch_size: usize,
    pub regularization: f64,
    pub random_state: Option<u64>,
    pub dropout_rate: f64,
    pub use_pretrained: bool,
}

impl NeuralEmbeddingExtractor {
    pub fn new() -> Self {
        Self {
            embedding_dim: 128,
            vocab_size: 10000,
            learning_rate: 0.001,
            n_epochs: 50,
            batch_size: 64,
            regularization: 0.01,
            random_state: None,
            dropout_rate: 0.1,
            use_pretrained: false,
        }
    }

    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn n_epochs(mut self, epochs: usize) -> Self {
        self.n_epochs = epochs;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    pub fn use_pretrained(mut self, use_pretrained: bool) -> Self {
        self.use_pretrained = use_pretrained;
        self
    }

    fn initialize_embeddings(&self) -> Array2<f64> {
        let mut rng = match self.random_state {
            Some(seed) => Random::seed_from_u64(seed),
            None => Random::seed_from_u64(42),
        };

        let scale = (2.0 / self.embedding_dim as f64).sqrt();
        let mut embeddings = Array2::zeros((self.vocab_size, self.embedding_dim));

        for i in 0..self.vocab_size {
            for j in 0..self.embedding_dim {
                embeddings[(i, j)] = rng.gen_range(-scale..scale);
            }
        }

        embeddings
    }

    fn lookup_embeddings(&self, indices: &Array1<usize>, embeddings: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((indices.len(), self.embedding_dim));

        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.vocab_size {
                for j in 0..self.embedding_dim {
                    result[(i, j)] = embeddings[(idx, j)];
                }
            }
        }

        result
    }

    fn apply_dropout(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut rng = Random::seed_from_u64(42);
        let mut result = x.clone();

        for elem in result.iter_mut() {
            if rng.random_range(0.0..1.0) < self.dropout_rate {
                *elem = 0.0;
            } else {
                *elem /= 1.0 - self.dropout_rate;
            }
        }

        result
    }
}

pub struct FittedNeuralEmbeddingExtractor {
    extractor: NeuralEmbeddingExtractor,
    embeddings: Array2<f64>,
    vocab_size: usize,
}

impl Estimator<Untrained> for NeuralEmbeddingExtractor {
    fn estimator_type(&self) -> &'static str {
        "NeuralEmbeddingExtractor"
    }

    fn complexity(&self) -> f64 {
        (self.vocab_size * self.embedding_dim) as f64
    }
}

impl Fit<Array1<usize>, ()> for NeuralEmbeddingExtractor {
    type Output = FittedNeuralEmbeddingExtractor;

    fn fit(&self, x: &Array1<usize>, _y: &()) -> SklResult<Self::Output> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let max_idx = x.iter().max().copied().unwrap_or(0);
        if max_idx >= self.vocab_size {
            return Err(SklearsError::InvalidInput(format!(
                "Index {} exceeds vocabulary size {}",
                max_idx, self.vocab_size
            )));
        }

        let embeddings = if self.use_pretrained {
            self.initialize_embeddings()
        } else {
            let mut embeddings = self.initialize_embeddings();

            for _epoch in 0..self.n_epochs {
                let embedded = self.lookup_embeddings(x, &embeddings);
                let dropped = self.apply_dropout(&embedded);

                let lr_decay = 0.99_f64.powi(_epoch as i32);
                let effective_lr = self.learning_rate * lr_decay;

                embeddings *= 1.0 - effective_lr * self.regularization;
            }

            embeddings
        };

        Ok(FittedNeuralEmbeddingExtractor {
            extractor: self.clone(),
            embeddings,
            vocab_size: self.vocab_size,
        })
    }
}

impl Transform<Array1<usize>, Array2<f64>> for FittedNeuralEmbeddingExtractor {
    fn transform(&self, x: &Array1<usize>) -> SklResult<Array2<f64>> {
        if x.is_empty() {
            return Ok(Array2::zeros((0, self.extractor.embedding_dim)));
        }

        let max_idx = x.iter().max().copied().unwrap_or(0);
        if max_idx >= self.vocab_size {
            return Err(SklearsError::InvalidInput(format!(
                "Index {} exceeds vocabulary size {}",
                max_idx, self.vocab_size
            )));
        }

        Ok(self.extractor.lookup_embeddings(x, &self.embeddings))
    }
}

impl FittedNeuralEmbeddingExtractor {
    pub fn get_embeddings(&self) -> &Array2<f64> {
        &self.embeddings
    }

    pub fn similarity(&self, idx1: usize, idx2: usize) -> SklResult<f64> {
        if idx1 >= self.vocab_size || idx2 >= self.vocab_size {
            return Err(SklearsError::InvalidInput("Index out of bounds".to_string()));
        }

        let emb1 = self.embeddings.row(idx1);
        let emb2 = self.embeddings.row(idx2);

        let dot_product = emb1.iter().zip(emb2.iter()).map(|(&a, &b)| a * b).sum::<f64>();
        let norm1 = emb1.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm2 = emb2.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm1 * norm2))
        }
    }

    pub fn most_similar(&self, idx: usize, top_k: usize) -> SklResult<Vec<(usize, f64)>> {
        if idx >= self.vocab_size {
            return Err(SklearsError::InvalidInput("Index out of bounds".to_string()));
        }

        let mut similarities = Vec::new();

        for i in 0..self.vocab_size {
            if i != idx {
                let sim = self.similarity(idx, i)?;
                similarities.push((i, sim));
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(top_k);

        Ok(similarities)
    }
}