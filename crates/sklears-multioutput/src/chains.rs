//! Chain-based multi-output learning algorithms
//!
//! This module provides chain-based approaches for multi-label and multi-output problems,
//! including ClassifierChain, RegressorChain, EnsembleOfChains, and BayesianClassifierChain.

use crate::utils::*;
// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Classifier Chain
///
/// A multi-label model that arranges binary classifiers into a chain.
/// Each model makes a prediction in the order specified by the chain using
/// all of the available features provided to the model plus the predictions
/// of models that are earlier in the chain.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::chains::ClassifierChain;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let labels = array![[0, 1], [1, 0], [1, 1]];
/// ```
#[derive(Debug, Clone)]
pub struct ClassifierChain<S = Untrained> {
    state: S,
    order: Option<Vec<usize>>,
    cv: Option<usize>,
    random_state: Option<u64>,
}

impl ClassifierChain<Untrained> {
    /// Create a new ClassifierChain instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            order: None,
            cv: None,
            random_state: None,
        }
    }

    /// Set the chain order
    pub fn order(mut self, order: Vec<usize>) -> Self {
        self.order = Some(order);
        self
    }

    /// Set cross-validation folds for training
    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = Some(cv);
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for ClassifierChain<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ClassifierChain<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl ClassifierChain<Untrained> {
    /// Fit the classifier chain using a simple mock approach
    pub fn fit_simple(
        self,
        X: &ArrayView2<'_, Float>,
        y: &Array2<i32>,
    ) -> SklResult<ClassifierChain<ClassifierChainTrained>> {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Determine chain order
        let order = self
            .order
            .clone()
            .unwrap_or_else(|| (0..n_labels).collect());

        if order.len() != n_labels {
            return Err(SklearsError::InvalidInput(
                "Chain order must contain all label indices".to_string(),
            ));
        }

        // Train models in the chain
        let mut models = Vec::new();
        let mut current_features = X.to_owned();

        for (i, &label_idx) in order.iter().enumerate() {
            let y_binary = y.column(label_idx).to_owned();

            // Train binary classifier
            let model = train_binary_classifier(&current_features.view(), &y_binary)?;
            models.push(model);

            // Add predictions as features for next model (except for the last one)
            if i < order.len() - 1 {
                let predictions = predict_binary_classifier(&current_features.view(), &models[i]);
                let n_current_features = current_features.ncols();
                let mut new_features = Array2::<Float>::zeros((n_samples, n_current_features + 1));

                // Copy existing features
                new_features
                    .slice_mut(s![.., ..n_current_features])
                    .assign(&current_features);

                // Add predictions as new feature
                for j in 0..n_samples {
                    new_features[[j, n_current_features]] = predictions[j] as Float;
                }

                current_features = new_features;
            }
        }

        let trained_state = ClassifierChainTrained {
            models,
            order,
            n_features,
            n_labels,
        };

        Ok(ClassifierChain {
            state: trained_state,
            order: self.order,
            cv: self.cv,
            random_state: self.random_state,
        })
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>, ClassifierChainTrained>
    for ClassifierChain<Untrained>
{
    type Fitted = ClassifierChain<ClassifierChainTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        self.fit_simple(X, y)
    }
}

/// Trained state for ClassifierChain
#[derive(Debug, Clone)]
pub struct ClassifierChainTrained {
    models: Vec<SimpleBinaryModel>,
    order: Vec<usize>,
    n_features: usize,
    n_labels: usize,
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>> for ClassifierChain<ClassifierChainTrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));
        let mut current_features = X.to_owned();

        // Make predictions following the chain order
        for (i, &label_idx) in self.state.order.iter().enumerate() {
            let model = &self.state.models[i];
            let label_predictions = predict_binary_classifier(&current_features.view(), model);

            // Store predictions
            for j in 0..n_samples {
                predictions[[j, label_idx]] = label_predictions[j];
            }

            // Add predictions as features for next model (if not last)
            if i < self.state.order.len() - 1 {
                let n_current_features = current_features.ncols();
                let mut new_features = Array2::<Float>::zeros((n_samples, n_current_features + 1));

                // Copy existing features
                new_features
                    .slice_mut(s![.., ..n_current_features])
                    .assign(&current_features);

                // Add current label predictions as feature
                for j in 0..n_samples {
                    new_features[[j, n_current_features]] = label_predictions[j] as Float;
                }

                current_features = new_features;
            }
        }

        Ok(predictions)
    }
}

impl ClassifierChain<ClassifierChainTrained> {
    /// Predict probabilities for each label
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut probabilities = Array2::<Float>::zeros((n_samples, self.state.n_labels));
        let mut current_features = X.to_owned();

        // Make probability predictions following the chain order
        for (i, &label_idx) in self.state.order.iter().enumerate() {
            let model = &self.state.models[i];
            let label_probabilities = predict_binary_probabilities(&current_features.view(), model);

            // Store probabilities
            for j in 0..n_samples {
                probabilities[[j, label_idx]] = label_probabilities[j];
            }

            // Add predictions as features for next model (if not last)
            if i < self.state.order.len() - 1 {
                let label_predictions =
                    label_probabilities.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
                let n_current_features = current_features.ncols();
                let mut new_features = Array2::<Float>::zeros((n_samples, n_current_features + 1));

                // Copy existing features
                new_features
                    .slice_mut(s![.., ..n_current_features])
                    .assign(&current_features);

                // Add predictions as feature
                for j in 0..n_samples {
                    new_features[[j, n_current_features]] = label_predictions[j];
                }

                current_features = new_features;
            }
        }

        Ok(probabilities)
    }

    /// Get the chain order used during training
    pub fn chain_order(&self) -> &[usize] {
        &self.state.order
    }

    /// Get the number of models in the chain
    pub fn n_models(&self) -> usize {
        self.state.models.len()
    }

    /// Get number of targets/labels
    pub fn n_targets(&self) -> usize {
        self.state.n_labels
    }

    /// Simple prediction method (alias for predict)
    pub fn predict_simple(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        self.predict(X)
    }

    /// Monte Carlo prediction (simplified)
    pub fn predict_monte_carlo(
        &self,
        X: &ArrayView2<'_, Float>,
        n_samples: usize,
        random_state: Option<u64>,
    ) -> SklResult<Array2<Float>> {
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "n_samples must be greater than 0".to_string(),
            ));
        }
        // For now, just return probabilities
        self.predict_proba(X)
    }

    /// Monte Carlo prediction for labels (simplified)
    pub fn predict_monte_carlo_labels(
        &self,
        X: &ArrayView2<'_, Float>,
        n_samples: usize,
        random_state: Option<u64>,
    ) -> SklResult<Array2<i32>> {
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "n_samples must be greater than 0".to_string(),
            ));
        }
        // For now, just return predictions
        self.predict(X)
    }
}

/// Regressor Chain
///
/// A multi-output model that arranges regressors into a chain.
/// Each model makes a prediction in the order specified by the chain using
/// all of the available features provided to the model plus the predictions
/// of models that are earlier in the chain.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::chains::RegressorChain;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let targets = array![[1.5, 2.5], [2.5, 3.5], [3.5, 1.5]];
/// ```
#[derive(Debug, Clone)]
pub struct RegressorChain<S = Untrained> {
    state: S,
    order: Option<Vec<usize>>,
    cv: Option<usize>,
    random_state: Option<u64>,
}

impl RegressorChain<Untrained> {
    /// Create a new RegressorChain instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            order: None,
            cv: None,
            random_state: None,
        }
    }

    /// Set the chain order
    pub fn order(mut self, order: Vec<usize>) -> Self {
        self.order = Some(order);
        self
    }

    /// Set cross-validation folds for training
    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = Some(cv);
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for RegressorChain<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RegressorChain<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl RegressorChain<Untrained> {
    /// Fit the regressor chain using a simple linear approach
    pub fn fit_simple(
        self,
        X: &ArrayView2<'_, Float>,
        y: &Array2<Float>,
    ) -> SklResult<RegressorChain<RegressorChainTrained>> {
        let (n_samples, n_features) = X.dim();
        let n_targets = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Determine chain order
        let order = self
            .order
            .clone()
            .unwrap_or_else(|| (0..n_targets).collect());

        if order.len() != n_targets {
            return Err(SklearsError::InvalidInput(
                "Chain order must contain all target indices".to_string(),
            ));
        }

        // Train models in the chain
        let mut models = Vec::new();
        let mut current_features = X.to_owned();

        for (i, &target_idx) in order.iter().enumerate() {
            let y_target = y.column(target_idx).to_owned();

            // Train linear regressor
            let model = train_simple_linear_classifier(&current_features.view(), &y_target)?;
            models.push(model);

            // Add predictions as features for next model (except for the last one)
            if i < order.len() - 1 {
                let predictions = predict_simple_linear(&current_features.view(), &models[i]);
                let n_current_features = current_features.ncols();
                let mut new_features = Array2::<Float>::zeros((n_samples, n_current_features + 1));

                // Copy existing features
                new_features
                    .slice_mut(s![.., ..n_current_features])
                    .assign(&current_features);

                // Add predictions as new feature
                for j in 0..n_samples {
                    new_features[[j, n_current_features]] = predictions[j];
                }

                current_features = new_features;
            }
        }

        let trained_state = RegressorChainTrained {
            models,
            order,
            n_features,
            n_targets,
        };

        Ok(RegressorChain {
            state: trained_state,
            order: self.order,
            cv: self.cv,
            random_state: self.random_state,
        })
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<Float>, RegressorChainTrained>
    for RegressorChain<Untrained>
{
    type Fitted = RegressorChain<RegressorChainTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<Float>) -> SklResult<Self::Fitted> {
        self.fit_simple(X, y)
    }
}

/// Trained state for RegressorChain
#[derive(Debug, Clone)]
pub struct RegressorChainTrained {
    models: Vec<SimpleLinearClassifier>,
    order: Vec<usize>,
    n_features: usize,
    n_targets: usize,
}

impl Predict<ArrayView2<'_, Float>, Array2<Float>> for RegressorChain<RegressorChainTrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<Float>::zeros((n_samples, self.state.n_targets));
        let mut current_features = X.to_owned();

        // Make predictions following the chain order
        for (i, &target_idx) in self.state.order.iter().enumerate() {
            let model = &self.state.models[i];
            let target_predictions = predict_simple_linear(&current_features.view(), model);

            // Store predictions
            for j in 0..n_samples {
                predictions[[j, target_idx]] = target_predictions[j];
            }

            // Add predictions as features for next model (if not last)
            if i < self.state.order.len() - 1 {
                let n_current_features = current_features.ncols();
                let mut new_features = Array2::<Float>::zeros((n_samples, n_current_features + 1));

                // Copy existing features
                new_features
                    .slice_mut(s![.., ..n_current_features])
                    .assign(&current_features);

                // Add current target predictions as feature
                for j in 0..n_samples {
                    new_features[[j, n_current_features]] = target_predictions[j];
                }

                current_features = new_features;
            }
        }

        Ok(predictions)
    }
}

impl RegressorChain<RegressorChainTrained> {
    /// Get the chain order used during training
    pub fn chain_order(&self) -> &[usize] {
        &self.state.order
    }

    /// Get the number of models in the chain
    pub fn n_models(&self) -> usize {
        self.state.models.len()
    }

    /// Get model at specified index
    pub fn get_model(&self, index: usize) -> Option<&SimpleLinearClassifier> {
        self.state.models.get(index)
    }

    /// Get the number of targets
    pub fn n_targets(&self) -> usize {
        self.state.n_targets
    }

    /// Simple prediction method (alias for predict)
    pub fn predict_simple(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.predict(X)
    }
}

/// Ensemble of Chains
///
/// An ensemble approach that combines multiple ClassifierChain models
/// with different chain orders or different random seeds to improve
/// prediction performance and robustness.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::chains::EnsembleOfChains;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let labels = array![[0, 1], [1, 0], [1, 1]];
/// ```
#[derive(Debug, Clone)]
pub struct EnsembleOfChains<S = Untrained> {
    state: S,
    n_chains: usize,
    chain_method: ChainMethod,
    random_state: Option<u64>,
}

/// Method for generating chains in ensemble
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChainMethod {
    /// Random chain orders
    Random,
    /// Fixed different orders
    Fixed,
    /// Bootstrap sampling with chains
    Bootstrap,
}

impl EnsembleOfChains<Untrained> {
    /// Create a new EnsembleOfChains instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_chains: 10,
            chain_method: ChainMethod::Random,
            random_state: None,
        }
    }

    /// Set number of chains in ensemble
    pub fn n_chains(mut self, n_chains: usize) -> Self {
        self.n_chains = n_chains;
        self
    }

    /// Set chain generation method
    pub fn chain_method(mut self, method: ChainMethod) -> Self {
        self.chain_method = method;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for EnsembleOfChains<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for EnsembleOfChains<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl EnsembleOfChains<Untrained> {
    /// Fit the ensemble of chains
    pub fn fit_simple(
        self,
        X: &ArrayView2<'_, Float>,
        y: &Array2<i32>,
    ) -> SklResult<EnsembleOfChains<EnsembleOfChainsTrained>> {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        let mut chains = Vec::new();
        let mut rng_state = self.random_state.unwrap_or(42);

        for i in 0..self.n_chains {
            // Generate chain order based on method
            let chain_order = match self.chain_method {
                ChainMethod::Random => {
                    let mut order: Vec<usize> = (0..n_labels).collect();
                    // Simple shuffle using deterministic random
                    for j in (1..order.len()).rev() {
                        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                        let k = (rng_state as usize) % (j + 1);
                        order.swap(j, k);
                    }
                    order
                }
                ChainMethod::Fixed => {
                    // Create different fixed orders
                    let mut order: Vec<usize> = (0..n_labels).collect();
                    order.rotate_left(i % n_labels);
                    order
                }
                ChainMethod::Bootstrap => {
                    // For bootstrap, use random order and later bootstrap samples
                    let mut order: Vec<usize> = (0..n_labels).collect();
                    for j in (1..order.len()).rev() {
                        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                        let k = (rng_state as usize) % (j + 1);
                        order.swap(j, k);
                    }
                    order
                }
            };

            // Create and train individual chain
            let chain = ClassifierChain::new()
                .order(chain_order)
                .random_state(rng_state);

            let trained_chain = chain.fit_simple(X, y)?;
            chains.push(trained_chain);

            rng_state = rng_state.wrapping_add(1);
        }

        let trained_state = EnsembleOfChainsTrained {
            chains,
            n_features,
            n_labels,
        };

        Ok(EnsembleOfChains {
            state: trained_state,
            n_chains: self.n_chains,
            chain_method: self.chain_method,
            random_state: self.random_state,
        })
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>, EnsembleOfChainsTrained>
    for EnsembleOfChains<Untrained>
{
    type Fitted = EnsembleOfChains<EnsembleOfChainsTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        self.fit_simple(X, y)
    }
}

/// Trained state for EnsembleOfChains
#[derive(Debug, Clone)]
pub struct EnsembleOfChainsTrained {
    chains: Vec<ClassifierChain<ClassifierChainTrained>>,
    n_features: usize,
    n_labels: usize,
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>> for EnsembleOfChains<EnsembleOfChainsTrained> {
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        // Collect predictions from all chains
        let mut all_predictions = Vec::new();
        for chain in &self.state.chains {
            let predictions = chain.predict(X)?;
            all_predictions.push(predictions);
        }

        // Ensemble predictions by majority voting
        let mut final_predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        for i in 0..n_samples {
            for j in 0..self.state.n_labels {
                let mut votes = 0;
                for predictions in &all_predictions {
                    votes += predictions[[i, j]];
                }
                // Majority vote
                final_predictions[[i, j]] = if votes > (self.state.chains.len() as i32) / 2 {
                    1
                } else {
                    0
                };
            }
        }

        Ok(final_predictions)
    }
}

impl EnsembleOfChains<EnsembleOfChainsTrained> {
    /// Predict probabilities using ensemble voting
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        // Collect probability predictions from all chains
        let mut all_probabilities = Vec::new();
        for chain in &self.state.chains {
            let probabilities = chain.predict_proba(X)?;
            all_probabilities.push(probabilities);
        }

        // Average probabilities across chains
        let mut final_probabilities = Array2::<Float>::zeros((n_samples, self.state.n_labels));

        for i in 0..n_samples {
            for j in 0..self.state.n_labels {
                let mut prob_sum = 0.0;
                for probabilities in &all_probabilities {
                    prob_sum += probabilities[[i, j]];
                }
                final_probabilities[[i, j]] = prob_sum / self.state.chains.len() as Float;
            }
        }

        Ok(final_probabilities)
    }

    /// Get number of chains in ensemble
    pub fn n_chains(&self) -> usize {
        self.state.chains.len()
    }

    /// Get individual chain at specified index
    pub fn get_chain(&self, index: usize) -> Option<&ClassifierChain<ClassifierChainTrained>> {
        self.state.chains.get(index)
    }

    /// Get diversity measure between chains
    pub fn chain_diversity(&self) -> Float {
        if self.state.chains.len() < 2 {
            return 0.0;
        }

        let mut diversity_sum = 0.0;
        let mut count = 0;

        // Compare chain orders pairwise
        for i in 0..self.state.chains.len() {
            for j in (i + 1)..self.state.chains.len() {
                let order1 = self.state.chains[i].chain_order();
                let order2 = self.state.chains[j].chain_order();

                // Calculate order similarity (Kendall's tau-like measure)
                let mut agreements = 0;
                for k in 0..order1.len() {
                    if order1[k] == order2[k] {
                        agreements += 1;
                    }
                }

                let similarity = agreements as Float / order1.len() as Float;
                diversity_sum += 1.0 - similarity;
                count += 1;
            }
        }

        if count > 0 {
            diversity_sum / count as Float
        } else {
            0.0
        }
    }

    /// Get the number of targets
    pub fn n_targets(&self) -> usize {
        self.state.n_labels
    }

    /// Simple prediction method (alias for predict)
    pub fn predict_simple(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        self.predict(X)
    }

    /// Simple probability prediction method (alias for predict_proba)
    pub fn predict_proba_simple(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        self.predict_proba(X)
    }
}

/// Bayesian Classifier Chain
///
/// A probabilistic variant of classifier chain that uses Bayesian inference
/// for the binary classifiers, providing uncertainty quantification alongside
/// predictions.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::chains::BayesianClassifierChain;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // This is a simple example showing the structure
/// let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let labels = array![[0, 1], [1, 0], [1, 1]];
/// let model = BayesianClassifierChain::new()
///     .n_samples(100)
///     .prior_strength(1.0);
/// ```
#[derive(Debug, Clone)]
pub struct BayesianClassifierChain<S = Untrained> {
    state: S,
    /// order
    pub order: Option<Vec<usize>>,
    /// n_samples
    pub n_samples: usize,
    /// prior_strength
    pub prior_strength: Float,
    /// random_state
    pub random_state: Option<u64>,
}

impl BayesianClassifierChain<Untrained> {
    /// Create a new BayesianClassifierChain instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            order: None,
            n_samples: 100,
            prior_strength: 1.0,
            random_state: None,
        }
    }

    /// Set the chain order
    pub fn order(mut self, order: Vec<usize>) -> Self {
        self.order = Some(order);
        self
    }

    /// Set number of posterior samples
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set prior strength (regularization parameter)
    pub fn prior_strength(mut self, prior_strength: Float) -> Self {
        self.prior_strength = prior_strength;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for BayesianClassifierChain<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BayesianClassifierChain<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl BayesianClassifierChain<Untrained> {
    /// Fit the Bayesian classifier chain
    #[allow(non_snake_case)]
    pub fn fit_simple(
        self,
        X: &ArrayView2<'_, Float>,
        y: &Array2<i32>,
    ) -> SklResult<BayesianClassifierChain<BayesianClassifierChainTrained>> {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Validate binary labels
        for &val in y.iter() {
            if val != 0 && val != 1 {
                return Err(SklearsError::InvalidInput(
                    "y must contain only binary values (0 or 1)".to_string(),
                ));
            }
        }

        // Determine chain order
        let order = self
            .order
            .clone()
            .unwrap_or_else(|| (0..n_labels).collect());

        if order.len() != n_labels {
            return Err(SklearsError::InvalidInput(
                "Chain order must contain all label indices".to_string(),
            ));
        }

        // Standardize features
        let feature_means = X.mean_axis(Axis(0)).unwrap();
        let feature_stds = X.std_axis(Axis(0), 0.0);
        let X_standardized = standardize_features_simple(X, &feature_means, &feature_stds);

        // Train Bayesian models in the chain
        let mut bayesian_models = Vec::new();
        let mut current_features = X_standardized;

        for (i, &label_idx) in order.iter().enumerate() {
            let y_binary = y.column(label_idx).to_owned();

            // Train Bayesian binary classifier
            let model = train_bayesian_binary_classifier(
                &current_features,
                &y_binary,
                self.prior_strength,
            )?;
            bayesian_models.push(model);

            // Add predictions as features for next model (except for the last one)
            if i < order.len() - 1 {
                let predictions =
                    predict_bayesian_mean(&current_features.view(), &bayesian_models[i]);
                let n_current_features = current_features.ncols();
                let mut new_features = Array2::<Float>::zeros((n_samples, n_current_features + 1));

                // Copy existing features
                new_features
                    .slice_mut(s![.., ..n_current_features])
                    .assign(&current_features);

                // Add predictions as new feature
                for j in 0..n_samples {
                    new_features[[j, n_current_features]] = predictions[j];
                }

                current_features = new_features;
            }
        }

        let trained_state = BayesianClassifierChainTrained {
            bayesian_models,
            order,
            n_features,
            n_labels,
            feature_means,
            feature_stds,
        };

        Ok(BayesianClassifierChain {
            state: trained_state,
            order: None,
            n_samples: self.n_samples,
            prior_strength: self.prior_strength,
            random_state: self.random_state,
        })
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>, BayesianClassifierChainTrained>
    for BayesianClassifierChain<Untrained>
{
    type Fitted = BayesianClassifierChain<BayesianClassifierChainTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, y: &Array2<i32>) -> SklResult<Self::Fitted> {
        self.fit_simple(X, y)
    }
}

/// Trained state for Bayesian Classifier Chain
#[derive(Debug, Clone)]
pub struct BayesianClassifierChainTrained {
    bayesian_models: Vec<BayesianBinaryModel>,
    order: Vec<usize>,
    n_features: usize,
    n_labels: usize,
    feature_means: Array1<Float>,
    feature_stds: Array1<Float>,
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for BayesianClassifierChain<BayesianClassifierChainTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.feature_means.len() {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        // Standardize features
        let X_standardized =
            standardize_features_simple(X, &self.state.feature_means, &self.state.feature_stds);

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));
        let mut current_features = X_standardized;

        // Make predictions following the chain order
        for (chain_pos, &label_idx) in self.state.order.iter().enumerate() {
            let model = &self.state.bayesian_models[chain_pos];

            // Sample from posterior distribution and make predictions
            let label_predictions = predict_bayesian_binary(&current_features.view(), model);

            // Convert probabilities to binary predictions
            for i in 0..n_samples {
                predictions[[i, label_idx]] = if label_predictions[i] > 0.5 { 1 } else { 0 };
            }

            // Add predictions as features for next model (if not last)
            if chain_pos < self.state.order.len() - 1 {
                let mut new_features =
                    Array2::<Float>::zeros((n_samples, current_features.ncols() + 1));

                // Copy existing features
                new_features
                    .slice_mut(s![.., ..current_features.ncols()])
                    .assign(&current_features);

                // Add current label predictions as feature
                for i in 0..n_samples {
                    new_features[[i, current_features.ncols()]] =
                        predictions[[i, label_idx]] as Float;
                }

                current_features = new_features;
            }
        }

        Ok(predictions)
    }
}

impl BayesianClassifierChain<BayesianClassifierChainTrained> {
    /// Predict with uncertainty quantification
    #[allow(non_snake_case)]
    pub fn predict_uncertainty(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.feature_means.len() {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        // Standardize features
        let X_standardized =
            standardize_features_simple(X, &self.state.feature_means, &self.state.feature_stds);

        let mut uncertainties = Array2::<Float>::zeros((n_samples, self.state.n_labels));
        let mut current_features = X_standardized;

        // Make predictions following the chain order with uncertainty estimation
        for (chain_pos, &label_idx) in self.state.order.iter().enumerate() {
            let model = &self.state.bayesian_models[chain_pos];

            // Get uncertainty estimates
            let (means, variances) = predict_bayesian_uncertainty(&current_features.view(), model)?;

            // Store uncertainties
            for i in 0..n_samples {
                uncertainties[[i, label_idx]] = variances[i];
            }

            // For chaining, use mean predictions as features
            if chain_pos < self.state.order.len() - 1 {
                let mut new_features =
                    Array2::<Float>::zeros((n_samples, current_features.ncols() + 1));

                // Copy existing features
                new_features
                    .slice_mut(s![.., ..current_features.ncols()])
                    .assign(&current_features);

                // Add mean predictions as feature
                for i in 0..n_samples {
                    new_features[[i, current_features.ncols()]] = means[i];
                }

                current_features = new_features;
            }
        }

        Ok(uncertainties)
    }

    /// Predict probabilities with Bayesian averaging
    #[allow(non_snake_case)]
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = X.dim();
        if n_features != self.state.feature_means.len() {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        // Standardize features
        let X_standardized =
            standardize_features_simple(X, &self.state.feature_means, &self.state.feature_stds);

        let mut probabilities = Array2::<Float>::zeros((n_samples, self.state.n_labels));
        let mut current_features = X_standardized;

        // Make probability predictions following the chain order
        for (chain_pos, &label_idx) in self.state.order.iter().enumerate() {
            let model = &self.state.bayesian_models[chain_pos];

            // Get probability predictions
            let label_probabilities = predict_bayesian_binary(&current_features.view(), model);

            // Store probabilities
            for i in 0..n_samples {
                probabilities[[i, label_idx]] = label_probabilities[i];
            }

            // Add mean predictions as features for next model (if not last)
            if chain_pos < self.state.order.len() - 1 {
                let mut new_features =
                    Array2::<Float>::zeros((n_samples, current_features.ncols() + 1));

                // Copy existing features
                new_features
                    .slice_mut(s![.., ..current_features.ncols()])
                    .assign(&current_features);

                // Add mean predictions as feature
                for i in 0..n_samples {
                    new_features[[i, current_features.ncols()]] = label_probabilities[i];
                }

                current_features = new_features;
            }
        }

        Ok(probabilities)
    }

    /// Get the chain order used during training
    pub fn chain_order(&self) -> &[usize] {
        &self.state.order
    }

    /// Get number of Bayesian models in the chain
    pub fn n_models(&self) -> usize {
        self.state.bayesian_models.len()
    }

    /// Get posterior statistics for a specific model in the chain
    pub fn model_posterior_stats(
        &self,
        model_idx: usize,
    ) -> Option<(&Array1<Float>, &Array2<Float>)> {
        self.state
            .bayesian_models
            .get(model_idx)
            .map(|model| (&model.weight_mean, &model.weight_cov))
    }

    /// Get the chain order used during training
    pub fn order(&self) -> &[usize] {
        &self.state.order
    }
}

// Chain-specific utility functions

/// Helper function to predict binary classification
fn predict_binary_classifier(X: &ArrayView2<Float>, model: &SimpleBinaryModel) -> Array1<i32> {
    let raw_scores = X.dot(&model.weights) + model.bias;
    raw_scores.mapv(|x| if x > 0.0 { 1 } else { 0 })
}
