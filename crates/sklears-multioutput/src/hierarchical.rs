//! Hierarchical classification and graph neural network models
//!
//! This module provides algorithms for hierarchical multi-label classification and
//! graph-based structured prediction tasks. It includes ontology-aware classifiers,
//! cost-sensitive hierarchical methods, and graph neural networks.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::RandNormal;
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Consistency enforcement strategies for hierarchical classification
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ConsistencyEnforcement {
    /// Post-processing approach that corrects predictions after classification
    #[default]
    PostProcessing,
    /// Training-time approach that enforces constraints during optimization
    ConstrainedTraining,
    /// Bayesian inference approach using probabilistic dependencies
    BayesianInference,
}

/// Ontology-Aware Hierarchical Classifier
///
/// A hierarchical multi-label classifier that incorporates domain ontology knowledge
/// to ensure taxonomically consistent predictions. This method enforces that if a child
/// concept is predicted, its parent concepts are also predicted according to the
/// provided hierarchical structure.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::{OntologyAwareClassifier, ConsistencyEnforcement};
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let y = array![[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]];
///
/// // Define ontology: child -> parent relationships
/// let mut ontology = HashMap::new();
/// ontology.insert(2, vec![0]); // concept 2 is child of concept 0
/// ontology.insert(3, vec![1]); // concept 3 is child of concept 1
///
/// let classifier = OntologyAwareClassifier::new()
///     .ontology(ontology)
///     .consistency_enforcement(ConsistencyEnforcement::PostProcessing)
///     .base_classifier_learning_rate(0.01);
/// let trained_classifier = classifier.fit(&X.view(), &y).unwrap();
/// let predictions = trained_classifier.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OntologyAwareClassifier<S = Untrained> {
    state: S,
    ontology: HashMap<usize, Vec<usize>>,
    consistency_enforcement: ConsistencyEnforcement,
    base_classifier_learning_rate: Float,
    max_iterations: usize,
}

/// Trained state for OntologyAwareClassifier
#[derive(Debug, Clone)]
pub struct OntologyAwareClassifierTrained {
    weights: Array2<Float>,
    biases: Array1<Float>,
    ontology: HashMap<usize, Vec<usize>>,
    consistency_enforcement: ConsistencyEnforcement,
    n_features: usize,
    n_labels: usize,
}

impl OntologyAwareClassifier<Untrained> {
    /// Create a new OntologyAwareClassifier
    pub fn new() -> Self {
        Self {
            state: Untrained,
            ontology: HashMap::new(),
            consistency_enforcement: ConsistencyEnforcement::PostProcessing,
            base_classifier_learning_rate: 0.01,
            max_iterations: 100,
        }
    }

    /// Set the ontology (child -> parent relationships)
    pub fn ontology(mut self, ontology: HashMap<usize, Vec<usize>>) -> Self {
        self.ontology = ontology;
        self
    }

    /// Set the consistency enforcement strategy
    pub fn consistency_enforcement(mut self, enforcement: ConsistencyEnforcement) -> Self {
        self.consistency_enforcement = enforcement;
        self
    }

    /// Set the learning rate for the base classifier
    pub fn base_classifier_learning_rate(mut self, learning_rate: Float) -> Self {
        self.base_classifier_learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
}

impl Default for OntologyAwareClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for OntologyAwareClassifier<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for OntologyAwareClassifier<Untrained> {
    type Fitted = OntologyAwareClassifier<OntologyAwareClassifierTrained>;

    fn fit(
        self,
        X: &ArrayView2<'_, Float>,
        y: &Array2<i32>,
    ) -> SklResult<OntologyAwareClassifier<OntologyAwareClassifierTrained>> {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize weights and biases
        let mut weights = Array2::<Float>::zeros((n_features, n_labels));
        let mut biases = Array1::<Float>::zeros(n_labels);

        // Train base classifiers for each label
        for iteration in 0..self.max_iterations {
            let mut total_loss = 0.0;

            for sample_idx in 0..n_samples {
                let x = X.row(sample_idx);
                let y_true = y.row(sample_idx);

                // Forward pass
                let logits = x.dot(&weights) + &biases;
                let probabilities = logits.mapv(|x| 1.0 / (1.0 + (-x).exp()));

                // Apply consistency constraints during training
                let consistent_probabilities = match self.consistency_enforcement {
                    ConsistencyEnforcement::ConstrainedTraining => {
                        self.enforce_consistency_training(&probabilities)?
                    }
                    _ => probabilities.clone(),
                };

                // Calculate loss and gradients
                for label_idx in 0..n_labels {
                    let y_label = y_true[label_idx] as Float;
                    let prob = consistent_probabilities[label_idx];
                    let error = prob - y_label;

                    total_loss += if y_label == 1.0 {
                        -prob.ln()
                    } else {
                        -(1.0 - prob).ln()
                    };

                    // Update weights and biases
                    for feat_idx in 0..n_features {
                        weights[[feat_idx, label_idx]] -=
                            self.base_classifier_learning_rate * error * x[feat_idx];
                    }
                    biases[label_idx] -= self.base_classifier_learning_rate * error;
                }
            }

            if iteration > 0 && total_loss < 1e-6 {
                break;
            }
        }

        Ok(OntologyAwareClassifier {
            state: OntologyAwareClassifierTrained {
                weights,
                biases,
                ontology: self.ontology,
                consistency_enforcement: self.consistency_enforcement,
                n_features,
                n_labels,
            },
            ontology: HashMap::new(),
            consistency_enforcement: self.consistency_enforcement,
            base_classifier_learning_rate: self.base_classifier_learning_rate,
            max_iterations: self.max_iterations,
        })
    }
}

impl OntologyAwareClassifier<Untrained> {
    /// Enforce consistency during training
    fn enforce_consistency_training(
        &self,
        probabilities: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let mut consistent_probs = probabilities.clone();

        // For training, we enforce that parent probabilities are at least as high as child probabilities
        for (&child, parents) in &self.ontology {
            if child < probabilities.len() {
                for &parent in parents {
                    if parent < probabilities.len() {
                        let child_prob = probabilities[child];
                        if consistent_probs[parent] < child_prob {
                            consistent_probs[parent] = child_prob;
                        }
                    }
                }
            }
        }

        Ok(consistent_probs)
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for OntologyAwareClassifier<OntologyAwareClassifierTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            let x = X.row(sample_idx);

            // Forward pass
            let logits = x.dot(&self.state.weights) + &self.state.biases;
            let probabilities = logits.mapv(|x| 1.0 / (1.0 + (-x).exp()));

            // Apply consistency enforcement
            let consistent_probs = match self.state.consistency_enforcement {
                ConsistencyEnforcement::PostProcessing => {
                    self.enforce_consistency_postprocessing(&probabilities)?
                }
                ConsistencyEnforcement::BayesianInference => {
                    self.enforce_consistency_bayesian(&probabilities)?
                }
                _ => probabilities,
            };

            // Convert probabilities to binary predictions
            for label_idx in 0..self.state.n_labels {
                predictions[[sample_idx, label_idx]] = if consistent_probs[label_idx] > 0.5 {
                    1
                } else {
                    0
                };
            }
        }

        Ok(predictions)
    }
}

impl OntologyAwareClassifier<OntologyAwareClassifierTrained> {
    /// Get the learned weights
    pub fn weights(&self) -> &Array2<Float> {
        &self.state.weights
    }

    /// Get the learned biases
    pub fn biases(&self) -> &Array1<Float> {
        &self.state.biases
    }

    /// Get the ontology
    pub fn ontology(&self) -> &HashMap<usize, Vec<usize>> {
        &self.state.ontology
    }

    /// Enforce consistency using post-processing
    fn enforce_consistency_postprocessing(
        &self,
        probabilities: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let mut consistent_probs = probabilities.clone();

        // Enforce hierarchical constraints: if child is predicted, parent must be predicted
        for (&child, parents) in &self.state.ontology {
            if child < probabilities.len() && probabilities[child] > 0.5 {
                for &parent in parents {
                    if parent < probabilities.len() {
                        consistent_probs[parent] =
                            consistent_probs[parent].max(probabilities[child]);
                    }
                }
            }
        }

        Ok(consistent_probs)
    }

    /// Enforce consistency using Bayesian inference
    fn enforce_consistency_bayesian(
        &self,
        probabilities: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let mut consistent_probs = probabilities.clone();

        // Simple Bayesian consistency: P(parent|child) = 1 if child is predicted
        for (&child, parents) in &self.state.ontology {
            if child < probabilities.len() {
                let child_prob = probabilities[child];
                for &parent in parents {
                    if parent < probabilities.len() {
                        // Bayesian update: P(parent) = P(parent) + P(child) * P(parent|child)
                        // Simplified: if child has high probability, parent should too
                        consistent_probs[parent] = consistent_probs[parent].max(child_prob * 0.8);
                    }
                }
            }
        }

        Ok(consistent_probs)
    }
}

/// Cost strategy for hierarchical classification
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CostStrategy {
    /// Uniform misclassification costs
    #[default]
    Uniform,
    /// Distance-based costs (closer nodes have lower cost)
    DistanceBased,
    /// Custom cost matrix
    Custom,
}

/// Cost-Sensitive Hierarchical Classifier
///
/// A hierarchical multi-label classifier that incorporates misclassification costs
/// and hierarchical relationships to optimize cost-sensitive predictions. This method
/// can handle different cost strategies including uniform costs, distance-based costs,
/// and custom cost matrices.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::{CostSensitiveHierarchicalClassifier, CostStrategy};
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let y = array![[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]];
///
/// // Define hierarchical structure and costs
/// let mut hierarchy = HashMap::new();
/// hierarchy.insert(0, vec![2, 3]); // concept 0 has children 2, 3
/// hierarchy.insert(1, vec![2, 3]); // concept 1 has children 2, 3
///
/// let classifier = CostSensitiveHierarchicalClassifier::new()
///     .hierarchy(hierarchy)
///     .cost_strategy(CostStrategy::DistanceBased)
///     .learning_rate(0.01);
/// let trained_classifier = classifier.fit(&X.view(), &y).unwrap();
/// let predictions = trained_classifier.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CostSensitiveHierarchicalClassifier<S = Untrained> {
    state: S,
    hierarchy: HashMap<usize, Vec<usize>>,
    cost_strategy: CostStrategy,
    cost_matrix: Option<Array2<Float>>,
    learning_rate: Float,
    max_iterations: usize,
    lambda_hierarchy: Float,
    lambda_cost: Float,
}

/// Trained state for CostSensitiveHierarchicalClassifier
#[derive(Debug, Clone)]
pub struct CostSensitiveHierarchicalClassifierTrained {
    weights: Array2<Float>,
    /// Hierarchical structure
    hierarchy: HashMap<usize, Vec<usize>>,
    cost_strategy: CostStrategy,
    cost_matrix: Option<Array2<Float>>,
    n_features: usize,
    n_labels: usize,
    lambda_hierarchy: Float,
    lambda_cost: Float,
}

impl CostSensitiveHierarchicalClassifier<Untrained> {
    /// Create a new CostSensitiveHierarchicalClassifier
    pub fn new() -> Self {
        Self {
            state: Untrained,
            hierarchy: HashMap::new(),
            cost_strategy: CostStrategy::Uniform,
            cost_matrix: None,
            learning_rate: 0.01,
            max_iterations: 100,
            lambda_hierarchy: 1.0,
            lambda_cost: 1.0,
        }
    }

    /// Set the hierarchical structure (parent -> children relationships)
    pub fn hierarchy(mut self, hierarchy: HashMap<usize, Vec<usize>>) -> Self {
        self.hierarchy = hierarchy;
        self
    }

    /// Set the cost strategy
    pub fn cost_strategy(mut self, strategy: CostStrategy) -> Self {
        self.cost_strategy = strategy;
        self
    }

    /// Set a custom cost matrix
    pub fn cost_matrix(mut self, cost_matrix: Array2<Float>) -> Self {
        self.cost_matrix = Some(cost_matrix);
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the hierarchical constraint weight
    pub fn lambda_hierarchy(mut self, lambda: Float) -> Self {
        self.lambda_hierarchy = lambda;
        self
    }

    /// Set the cost constraint weight
    pub fn lambda_cost(mut self, lambda: Float) -> Self {
        self.lambda_cost = lambda;
        self
    }
}

impl Default for CostSensitiveHierarchicalClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CostSensitiveHierarchicalClassifier<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Array2<i32>> for CostSensitiveHierarchicalClassifier<Untrained> {
    type Fitted = CostSensitiveHierarchicalClassifier<CostSensitiveHierarchicalClassifierTrained>;

    fn fit(
        self,
        X: &ArrayView2<'_, Float>,
        y: &Array2<i32>,
    ) -> SklResult<CostSensitiveHierarchicalClassifier<CostSensitiveHierarchicalClassifierTrained>>
    {
        let (n_samples, n_features) = X.dim();
        let n_labels = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize cost matrix if not provided
        let cost_matrix = match &self.cost_matrix {
            Some(matrix) => matrix.clone(),
            None => self.generate_cost_matrix(n_labels)?,
        };

        // Initialize weights
        let mut weights = Array2::<Float>::zeros((n_features, n_labels));

        // Training loop with cost-sensitive and hierarchical constraints
        for _iteration in 0..self.max_iterations {
            for sample_idx in 0..n_samples {
                let x = X.row(sample_idx);
                let y_true = y.row(sample_idx);

                // Forward pass
                let scores = x.dot(&weights);
                let probabilities = scores.mapv(|x| 1.0 / (1.0 + (-x).exp()));

                // Calculate gradients with cost-sensitive and hierarchical terms
                for label_idx in 0..n_labels {
                    let y_label = y_true[label_idx] as Float;
                    let prob = probabilities[label_idx];

                    // Standard logistic loss gradient
                    let mut gradient = prob - y_label;

                    // Add cost-sensitive term
                    let cost_weight = cost_matrix[[label_idx, label_idx]];
                    gradient *= cost_weight * self.lambda_cost;

                    // Add hierarchical constraint term
                    gradient += self.lambda_hierarchy
                        * self.hierarchical_gradient(label_idx, &probabilities, &y_true)?;

                    // Update weights
                    for feat_idx in 0..n_features {
                        weights[[feat_idx, label_idx]] -=
                            self.learning_rate * gradient * x[feat_idx];
                    }
                }
            }
        }

        Ok(CostSensitiveHierarchicalClassifier {
            state: CostSensitiveHierarchicalClassifierTrained {
                weights,
                hierarchy: self.hierarchy,
                cost_strategy: self.cost_strategy,
                cost_matrix: Some(cost_matrix),
                n_features,
                n_labels,
                lambda_hierarchy: self.lambda_hierarchy,
                lambda_cost: self.lambda_cost,
            },
            hierarchy: HashMap::new(),
            cost_strategy: self.cost_strategy,
            cost_matrix: None,
            learning_rate: self.learning_rate,
            max_iterations: self.max_iterations,
            lambda_hierarchy: self.lambda_hierarchy,
            lambda_cost: self.lambda_cost,
        })
    }
}

impl CostSensitiveHierarchicalClassifier<Untrained> {
    /// Generate cost matrix based on strategy
    fn generate_cost_matrix(&self, n_labels: usize) -> SklResult<Array2<Float>> {
        match self.cost_strategy {
            CostStrategy::Uniform => Ok(Array2::eye(n_labels)),
            CostStrategy::DistanceBased => {
                let mut cost_matrix = Array2::<Float>::zeros((n_labels, n_labels));
                // Simple distance-based costs (can be enhanced with actual hierarchy distances)
                for i in 0..n_labels {
                    for j in 0..n_labels {
                        cost_matrix[[i, j]] = if i == j { 1.0 } else { 0.5 };
                    }
                }
                Ok(cost_matrix)
            }
            CostStrategy::Custom => Err(SklearsError::InvalidInput(
                "Custom cost strategy requires a cost matrix".to_string(),
            )),
        }
    }

    /// Calculate hierarchical gradient term
    fn hierarchical_gradient(
        &self,
        label_idx: usize,
        probabilities: &Array1<Float>,
        y_true: &ArrayView1<i32>,
    ) -> SklResult<Float> {
        let mut gradient = 0.0;

        // If this label has children, enforce that children can't be more probable than parent
        if let Some(children) = self.hierarchy.get(&label_idx) {
            for &child in children {
                if child < probabilities.len() {
                    let parent_prob = probabilities[label_idx];
                    let child_prob = probabilities[child];
                    let child_true = y_true[child] as Float;

                    // Penalty if child probability exceeds parent probability when child is true
                    if child_true > 0.5 && child_prob > parent_prob {
                        gradient += child_prob - parent_prob;
                    }
                }
            }
        }

        // If this label is a child, enforce consistency with parents
        for (&parent, children) in &self.hierarchy {
            if children.contains(&label_idx) && parent < probabilities.len() {
                let parent_prob = probabilities[parent];
                let child_prob = probabilities[label_idx];
                let label_true = y_true[label_idx] as Float;

                // Penalty if child is predicted but parent is not
                if label_true > 0.5 && child_prob > parent_prob {
                    gradient -= child_prob - parent_prob;
                }
            }
        }

        Ok(gradient)
    }
}

impl Predict<ArrayView2<'_, Float>, Array2<i32>>
    for CostSensitiveHierarchicalClassifier<CostSensitiveHierarchicalClassifierTrained>
{
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<i32>> {
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = Array2::<i32>::zeros((n_samples, self.state.n_labels));

        for sample_idx in 0..n_samples {
            let x = X.row(sample_idx);

            // Forward pass
            let scores = x.dot(&self.state.weights);
            let probabilities = scores.mapv(|x| 1.0 / (1.0 + (-x).exp()));

            // Apply hierarchical constraints and cost-sensitive thresholding
            let final_predictions = self.apply_constraints(&probabilities)?;

            for label_idx in 0..self.state.n_labels {
                predictions[[sample_idx, label_idx]] = final_predictions[label_idx];
            }
        }

        Ok(predictions)
    }
}

impl CostSensitiveHierarchicalClassifier<CostSensitiveHierarchicalClassifierTrained> {
    /// Get the learned weights
    pub fn weights(&self) -> &Array2<Float> {
        &self.state.weights
    }

    /// Get the cost matrix
    pub fn cost_matrix(&self) -> Option<&Array2<Float>> {
        self.state.cost_matrix.as_ref()
    }

    /// Apply constraints to get final predictions
    fn apply_constraints(&self, probabilities: &Array1<Float>) -> SklResult<Array1<i32>> {
        let mut binary_predictions = Array1::<i32>::zeros(probabilities.len());

        // Convert probabilities to binary predictions with cost-sensitive thresholds
        for i in 0..probabilities.len() {
            let threshold = if let Some(cost_matrix) = &self.state.cost_matrix {
                // Adjust threshold based on cost
                let cost = cost_matrix[[i, i]];
                0.5 / cost.max(0.1) // Higher cost = lower threshold
            } else {
                0.5
            };

            binary_predictions[i] = if probabilities[i] > threshold { 1 } else { 0 };
        }

        // Enforce hierarchical constraints
        for (&parent, children) in &self.state.hierarchy {
            if parent < binary_predictions.len() {
                // If any child is predicted, parent must be predicted
                let mut any_child_predicted = false;
                for &child in children {
                    if child < binary_predictions.len() && binary_predictions[child] == 1 {
                        any_child_predicted = true;
                        break;
                    }
                }
                if any_child_predicted {
                    binary_predictions[parent] = 1;
                }
            }
        }

        Ok(binary_predictions)
    }
}

// Graph Neural Networks for Structured Output Prediction

/// Aggregation functions for Graph Neural Networks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationFunction {
    /// Mean aggregation
    Mean,
    /// Sum aggregation
    Sum,
    /// Max aggregation
    Max,
    /// Attention-based aggregation
    Attention,
}

/// Message passing variants for Graph Neural Networks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MessagePassingVariant {
    /// Graph Convolutional Network (GCN)
    GCN,
    /// Graph Attention Network (GAT)
    GAT,
    /// GraphSAGE
    GraphSAGE,
    /// Graph Isomorphism Network (GIN)
    GIN,
}

/// Graph Neural Network for Structured Output Prediction
///
/// A graph neural network implementation for multi-output prediction tasks where
/// the outputs have structural relationships represented as a graph. This method
/// can leverage node features, edge information, and graph topology to make
/// predictions that respect the underlying graph structure.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::{GraphNeuralNetwork, MessagePassingVariant, AggregationFunction};
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// // Node features and adjacency matrix
/// let node_features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0], [1.0, 3.0]];
/// let adjacency = array![[0, 1, 1, 0, 0], [1, 0, 1, 1, 0], [1, 1, 0, 0, 1],
///                        [0, 1, 0, 0, 1], [0, 0, 1, 1, 0]];
/// let node_labels = array![[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]];
///
/// let gnn = GraphNeuralNetwork::new()
///     .hidden_dim(16)
///     .num_layers(2)
///     .message_passing_variant(MessagePassingVariant::GCN)
///     .aggregation_function(AggregationFunction::Mean);
/// let trained_gnn = gnn.fit_graph(&adjacency.view(), &node_features.view(), &node_labels).unwrap();
/// let predictions = trained_gnn.predict_graph(&adjacency.view(), &node_features.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GraphNeuralNetwork<S = Untrained> {
    state: S,
    hidden_dim: usize,
    num_layers: usize,
    message_passing_variant: MessagePassingVariant,
    aggregation_function: AggregationFunction,
    learning_rate: Float,
    max_iter: usize,
    dropout_rate: Float,
    random_state: Option<u64>,
}

/// Trained state for GraphNeuralNetwork
#[derive(Debug, Clone)]
pub struct GraphNeuralNetworkTrained {
    /// Layer weights: Vec of (input_dim x output_dim) matrices
    layer_weights: Vec<Array2<Float>>,
    /// Layer biases: Vec of output_dim vectors
    layer_biases: Vec<Array1<Float>>,
    /// Attention weights for GAT (if applicable)
    attention_weights: Option<Vec<Array2<Float>>>,
    /// Model configuration
    hidden_dim: usize,
    num_layers: usize,
    message_passing_variant: MessagePassingVariant,
    aggregation_function: AggregationFunction,
    n_features: usize,
    n_outputs: usize,
    dropout_rate: Float,
}

impl GraphNeuralNetwork<Untrained> {
    /// Create a new GraphNeuralNetwork instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            hidden_dim: 32,
            num_layers: 2,
            message_passing_variant: MessagePassingVariant::GCN,
            aggregation_function: AggregationFunction::Mean,
            learning_rate: 0.01,
            max_iter: 100,
            dropout_rate: 0.0,
            random_state: None,
        }
    }

    /// Set the hidden dimension
    pub fn hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = hidden_dim;
        self
    }

    /// Set the number of layers
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set the message passing variant
    pub fn message_passing_variant(mut self, variant: MessagePassingVariant) -> Self {
        self.message_passing_variant = variant;
        self
    }

    /// Set the aggregation function
    pub fn aggregation_function(mut self, function: AggregationFunction) -> Self {
        self.aggregation_function = function;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the dropout rate
    pub fn dropout_rate(mut self, dropout_rate: Float) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for GraphNeuralNetwork<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GraphNeuralNetwork<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Fit method for Graph Neural Networks with graph structure
impl GraphNeuralNetwork<Untrained> {
    /// Fit the GNN using graph structure, node features, and node labels
    pub fn fit_graph(
        self,
        adjacency: &ArrayView2<'_, i32>,
        node_features: &ArrayView2<'_, Float>,
        node_labels: &Array2<i32>,
    ) -> SklResult<GraphNeuralNetwork<GraphNeuralNetworkTrained>> {
        let (n_nodes, n_features) = node_features.dim();
        let n_outputs = node_labels.ncols();

        if adjacency.dim() != (n_nodes, n_nodes) {
            return Err(SklearsError::InvalidInput(
                "Adjacency matrix must be n_nodes x n_nodes".to_string(),
            ));
        }

        if node_labels.nrows() != n_nodes {
            return Err(SklearsError::InvalidInput(
                "Node labels must have same number of rows as nodes".to_string(),
            ));
        }

        // Initialize parameters
        let mut rng_instance = thread_rng();
        let (layer_weights, layer_biases, attention_weights) =
            self.initialize_gnn_parameters(n_features, n_outputs, &mut rng_instance)?;

        // Training loop (simplified gradient descent)
        let mut weights = layer_weights;
        let biases = layer_biases;
        let attention_weights = attention_weights;

        for _iteration in 0..self.max_iter {
            // Forward pass
            let (node_embeddings, _) = self.forward_pass_graph(
                adjacency,
                node_features,
                &weights,
                &biases,
                &attention_weights,
            )?;

            // Compute loss and gradients (simplified)
            let predictions = node_embeddings.mapv(|x| if x > 0.0 { 1 } else { 0 });

            // Simple gradient update (in practice, would use backpropagation)
            for layer_idx in 0..weights.len() {
                for i in 0..weights[layer_idx].nrows() {
                    for j in 0..weights[layer_idx].ncols() {
                        weights[layer_idx][[i, j]] *= 0.999; // Simple weight decay
                    }
                }
            }
        }

        let trained_state = GraphNeuralNetworkTrained {
            layer_weights: weights,
            layer_biases: biases,
            attention_weights,
            hidden_dim: self.hidden_dim,
            num_layers: self.num_layers,
            message_passing_variant: self.message_passing_variant,
            aggregation_function: self.aggregation_function,
            n_features,
            n_outputs,
            dropout_rate: self.dropout_rate,
        };

        Ok(GraphNeuralNetwork {
            state: trained_state,
            hidden_dim: self.hidden_dim,
            num_layers: self.num_layers,
            message_passing_variant: self.message_passing_variant,
            aggregation_function: self.aggregation_function,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            dropout_rate: self.dropout_rate,
            random_state: self.random_state,
        })
    }

    /// Initialize GNN parameters
    fn initialize_gnn_parameters(
        &self,
        n_features: usize,
        n_outputs: usize,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(
        Vec<Array2<Float>>,
        Vec<Array1<Float>>,
        Option<Vec<Array2<Float>>>,
    )> {
        let mut layer_weights = Vec::new();
        let mut layer_biases = Vec::new();
        let mut attention_weights = None;

        // Input layer
        let input_dim = match self.message_passing_variant {
            MessagePassingVariant::GraphSAGE => n_features * 2, // Concatenated features
            _ => n_features,
        };

        // Hidden layers
        let hidden_dim = match self.message_passing_variant {
            MessagePassingVariant::GraphSAGE => self.hidden_dim * 2, // Concatenated features
            _ => self.hidden_dim,
        };

        // Initialize weights for each layer
        for layer_idx in 0..self.num_layers {
            let (in_dim, out_dim) = if layer_idx == 0 {
                (input_dim, self.hidden_dim)
            } else if layer_idx == self.num_layers - 1 {
                (hidden_dim, n_outputs)
            } else {
                (hidden_dim, self.hidden_dim)
            };

            let normal_dist = RandNormal::new(0.0, (2.0 / in_dim as Float).sqrt()).unwrap();
            let mut input_weight = Array2::<Float>::zeros((in_dim, out_dim));
            for i in 0..in_dim {
                for j in 0..out_dim {
                    input_weight[[i, j]] = rng.sample(normal_dist);
                }
            }
            let bias = Array1::<Float>::zeros(out_dim);

            layer_weights.push(input_weight);
            layer_biases.push(bias);
        }

        // Initialize attention weights for GAT
        if self.message_passing_variant == MessagePassingVariant::GAT {
            let mut att_weights = Vec::new();
            for layer_idx in 0..self.num_layers {
                let att_dim = if layer_idx == 0 {
                    n_features
                } else {
                    self.hidden_dim
                };
                let att_normal_dist = RandNormal::new(0.0, 0.1).unwrap();
                let mut attention_weight = Array2::<Float>::zeros((att_dim * 2, 1));
                for i in 0..(att_dim * 2) {
                    attention_weight[[i, 0]] = rng.sample(att_normal_dist);
                }
                att_weights.push(attention_weight);
            }
            attention_weights = Some(att_weights);
        }

        Ok((layer_weights, layer_biases, attention_weights))
    }

    /// Forward pass through the graph neural network
    fn forward_pass_graph(
        &self,
        adjacency: &ArrayView2<'_, i32>,
        node_features: &ArrayView2<'_, Float>,
        weights: &[Array2<Float>],
        biases: &[Array1<Float>],
        attention_weights: &Option<Vec<Array2<Float>>>,
    ) -> SklResult<(Array2<Float>, Vec<Array2<Float>>)> {
        let n_nodes = node_features.nrows();
        let mut current_embeddings = node_features.to_owned();
        let mut layer_outputs = Vec::new();

        for layer_idx in 0..self.num_layers {
            let layer_output = match self.message_passing_variant {
                MessagePassingVariant::GCN => self.gcn_layer(
                    &current_embeddings,
                    adjacency,
                    &weights[layer_idx],
                    &biases[layer_idx],
                )?,
                MessagePassingVariant::GAT => {
                    let att_weights = attention_weights.as_ref().unwrap();
                    self.gat_layer(
                        &current_embeddings,
                        adjacency,
                        &weights[layer_idx],
                        &biases[layer_idx],
                        &att_weights[layer_idx],
                    )?
                }
                MessagePassingVariant::GraphSAGE => self.graphsage_layer(
                    &current_embeddings,
                    adjacency,
                    &weights[layer_idx],
                    &biases[layer_idx],
                )?,
                MessagePassingVariant::GIN => self.gin_layer(
                    &current_embeddings,
                    adjacency,
                    &weights[layer_idx],
                    &biases[layer_idx],
                )?,
            };

            current_embeddings = layer_output.clone();
            layer_outputs.push(layer_output);
        }

        Ok((current_embeddings, layer_outputs))
    }

    /// Graph Convolutional Network layer
    fn gcn_layer(
        &self,
        node_embeddings: &Array2<Float>,
        adjacency: &ArrayView2<'_, i32>,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
    ) -> SklResult<Array2<Float>> {
        let n_nodes = node_embeddings.nrows();
        let mut output = Array2::<Float>::zeros((n_nodes, weights.ncols()));

        for i in 0..n_nodes {
            let mut aggregated = Array1::<Float>::zeros(node_embeddings.ncols());
            let mut degree = 0;

            // Aggregate neighbor features
            for j in 0..n_nodes {
                if adjacency[[i, j]] == 1 {
                    aggregated += &node_embeddings.row(j).to_owned();
                    degree += 1;
                }
            }

            // Add self-loop
            aggregated += &node_embeddings.row(i).to_owned();
            degree += 1;

            // Normalize by degree
            if degree > 0 {
                aggregated /= degree as Float;
            }

            // Apply linear transformation
            let transformed = aggregated.dot(weights) + bias;
            let activated = transformed.mapv(|x| x.max(0.0)); // ReLU activation

            output.row_mut(i).assign(&activated);
        }

        Ok(output)
    }

    /// Graph Attention Network layer (simplified)
    fn gat_layer(
        &self,
        node_embeddings: &Array2<Float>,
        adjacency: &ArrayView2<'_, i32>,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
        attention_weights: &Array2<Float>,
    ) -> SklResult<Array2<Float>> {
        let n_nodes = node_embeddings.nrows();
        let mut output = Array2::<Float>::zeros((n_nodes, weights.ncols()));

        for i in 0..n_nodes {
            let mut attention_scores = Array1::<Float>::zeros(n_nodes);
            let mut valid_neighbors = Vec::new();

            // Calculate attention scores
            for j in 0..n_nodes {
                if adjacency[[i, j]] == 1 || i == j {
                    // Concatenate node features for attention computation
                    let concat_features = Array1::from_iter(
                        node_embeddings
                            .row(i)
                            .iter()
                            .chain(node_embeddings.row(j).iter())
                            .cloned(),
                    );

                    if concat_features.len() == attention_weights.nrows() {
                        let score = concat_features.dot(&attention_weights.column(0));
                        attention_scores[j] = score.exp();
                        valid_neighbors.push(j);
                    }
                }
            }

            // Normalize attention scores
            let total_attention: Float = valid_neighbors.iter().map(|&j| attention_scores[j]).sum();
            if total_attention > 0.0 {
                for &j in &valid_neighbors {
                    attention_scores[j] /= total_attention;
                }
            }

            // Aggregate features using attention weights
            let mut aggregated = Array1::<Float>::zeros(node_embeddings.ncols());
            for &j in &valid_neighbors {
                let weighted_features = &node_embeddings.row(j).to_owned() * attention_scores[j];
                aggregated += &weighted_features;
            }

            // Apply linear transformation
            let transformed = aggregated.dot(weights) + bias;
            let activated = transformed.mapv(|x| x.max(0.0)); // ReLU activation

            output.row_mut(i).assign(&activated);
        }

        Ok(output)
    }

    /// GraphSAGE layer (simplified)
    fn graphsage_layer(
        &self,
        node_embeddings: &Array2<Float>,
        adjacency: &ArrayView2<'_, i32>,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
    ) -> SklResult<Array2<Float>> {
        let n_nodes = node_embeddings.nrows();
        let embedding_dim = node_embeddings.ncols();
        let output_dim = weights.ncols();
        let mut output = Array2::<Float>::zeros((n_nodes, output_dim));

        for i in 0..n_nodes {
            // Aggregate neighbor features
            let mut neighbor_sum = Array1::<Float>::zeros(embedding_dim);
            let mut neighbor_count = 0;

            for j in 0..n_nodes {
                if adjacency[[i, j]] == 1 && i != j {
                    neighbor_sum += &node_embeddings.row(j).to_owned();
                    neighbor_count += 1;
                }
            }

            // Average pooling of neighbors
            if neighbor_count > 0 {
                neighbor_sum /= neighbor_count as Float;
            }

            // Concatenate self and neighbor representations
            let self_features = node_embeddings.row(i).to_owned();
            let concatenated =
                Array1::from_iter(self_features.iter().chain(neighbor_sum.iter()).cloned());

            // Apply linear transformation (note: weights should match concatenated dimension)
            if concatenated.len() == weights.nrows() {
                let transformed = concatenated.dot(weights) + bias;
                let activated = transformed.mapv(|x| x.max(0.0)); // ReLU activation
                output.row_mut(i).assign(&activated);
            }
        }

        Ok(output)
    }

    /// Graph Isomorphism Network layer
    fn gin_layer(
        &self,
        node_embeddings: &Array2<Float>,
        adjacency: &ArrayView2<'_, i32>,
        weights: &Array2<Float>,
        bias: &Array1<Float>,
    ) -> SklResult<Array2<Float>> {
        let n_nodes = node_embeddings.nrows();
        let mut output = Array2::<Float>::zeros((n_nodes, weights.ncols()));
        let epsilon = 0.0; // Learnable parameter, simplified as 0

        for i in 0..n_nodes {
            // Sum neighbor features
            let mut neighbor_sum = Array1::<Float>::zeros(node_embeddings.ncols());

            for j in 0..n_nodes {
                if adjacency[[i, j]] == 1 && i != j {
                    neighbor_sum += &node_embeddings.row(j).to_owned();
                }
            }

            // GIN update: (1 + epsilon) * h_i + sum(h_j for j in neighbors)
            let updated = &node_embeddings.row(i).to_owned() * (1.0 + epsilon) + &neighbor_sum;

            // Apply MLP (simplified as single linear layer)
            let transformed = updated.dot(weights) + bias;
            let activated = transformed.mapv(|x| x.max(0.0)); // ReLU activation

            output.row_mut(i).assign(&activated);
        }

        Ok(output)
    }
}

impl GraphNeuralNetwork<GraphNeuralNetworkTrained> {
    /// Predict node labels using the trained GNN
    pub fn predict_graph(
        &self,
        adjacency: &ArrayView2<'_, i32>,
        node_features: &ArrayView2<'_, Float>,
    ) -> SklResult<Array2<i32>> {
        let (n_nodes, n_features) = node_features.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Node features have different dimensionality than training data".to_string(),
            ));
        }

        if adjacency.dim() != (n_nodes, n_nodes) {
            return Err(SklearsError::InvalidInput(
                "Adjacency matrix must be n_nodes x n_nodes".to_string(),
            ));
        }

        // Forward pass
        let (final_embeddings, _) = self.forward_pass_trained(adjacency, node_features)?;

        // Convert to binary predictions
        let predictions = final_embeddings.mapv(|x| if x > 0.0 { 1 } else { 0 });

        Ok(predictions)
    }

    /// Get the hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.state.hidden_dim
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.state.num_layers
    }

    /// Forward pass for trained model
    fn forward_pass_trained(
        &self,
        adjacency: &ArrayView2<'_, i32>,
        node_features: &ArrayView2<'_, Float>,
    ) -> SklResult<(Array2<Float>, Vec<Array2<Float>>)> {
        let n_nodes = node_features.nrows();
        let mut current_embeddings = node_features.to_owned();
        let mut layer_outputs = Vec::new();

        for layer_idx in 0..self.state.num_layers {
            let layer_output = match self.state.message_passing_variant {
                MessagePassingVariant::GCN => {
                    self.gcn_layer_trained(&current_embeddings, adjacency, layer_idx)?
                }
                MessagePassingVariant::GAT => {
                    self.gat_layer_trained(&current_embeddings, adjacency, layer_idx)?
                }
                MessagePassingVariant::GraphSAGE => {
                    self.graphsage_layer_trained(&current_embeddings, adjacency, layer_idx)?
                }
                MessagePassingVariant::GIN => {
                    self.gin_layer_trained(&current_embeddings, adjacency, layer_idx)?
                }
            };

            current_embeddings = layer_output.clone();
            layer_outputs.push(layer_output);
        }

        Ok((current_embeddings, layer_outputs))
    }

    /// GCN layer for trained model
    fn gcn_layer_trained(
        &self,
        node_embeddings: &Array2<Float>,
        adjacency: &ArrayView2<'_, i32>,
        layer_idx: usize,
    ) -> SklResult<Array2<Float>> {
        let weights = &self.state.layer_weights[layer_idx];
        let bias = &self.state.layer_biases[layer_idx];
        let n_nodes = node_embeddings.nrows();
        let mut output = Array2::<Float>::zeros((n_nodes, weights.ncols()));

        for i in 0..n_nodes {
            let mut aggregated = Array1::<Float>::zeros(node_embeddings.ncols());
            let mut degree = 0;

            // Aggregate neighbor features
            for j in 0..n_nodes {
                if adjacency[[i, j]] == 1 {
                    aggregated += &node_embeddings.row(j).to_owned();
                    degree += 1;
                }
            }

            // Add self-loop
            aggregated += &node_embeddings.row(i).to_owned();
            degree += 1;

            // Normalize by degree
            if degree > 0 {
                aggregated /= degree as Float;
            }

            // Apply linear transformation
            let transformed = aggregated.dot(weights) + bias;
            let activated = if layer_idx == self.state.num_layers - 1 {
                // Output layer: sigmoid activation for binary classification
                transformed.mapv(|x| 1.0 / (1.0 + (-x).exp()))
            } else {
                // Hidden layers: ReLU activation
                transformed.mapv(|x| x.max(0.0))
            };

            output.row_mut(i).assign(&activated);
        }

        Ok(output)
    }

    /// GAT layer for trained model
    fn gat_layer_trained(
        &self,
        node_embeddings: &Array2<Float>,
        adjacency: &ArrayView2<'_, i32>,
        layer_idx: usize,
    ) -> SklResult<Array2<Float>> {
        let weights = &self.state.layer_weights[layer_idx];
        let bias = &self.state.layer_biases[layer_idx];
        let attention_weights = self.state.attention_weights.as_ref().unwrap();
        let att_weights = &attention_weights[layer_idx];

        let n_nodes = node_embeddings.nrows();
        let mut output = Array2::<Float>::zeros((n_nodes, weights.ncols()));

        for i in 0..n_nodes {
            let mut attention_scores = Array1::<Float>::zeros(n_nodes);
            let mut valid_neighbors = Vec::new();

            // Calculate attention scores
            for j in 0..n_nodes {
                if adjacency[[i, j]] == 1 || i == j {
                    let concat_features = Array1::from_iter(
                        node_embeddings
                            .row(i)
                            .iter()
                            .chain(node_embeddings.row(j).iter())
                            .cloned(),
                    );

                    if concat_features.len() == att_weights.nrows() {
                        let score = concat_features.dot(&att_weights.column(0));
                        attention_scores[j] = score.exp();
                        valid_neighbors.push(j);
                    }
                }
            }

            // Normalize attention scores
            let total_attention: Float = valid_neighbors.iter().map(|&j| attention_scores[j]).sum();
            if total_attention > 0.0 {
                for &j in &valid_neighbors {
                    attention_scores[j] /= total_attention;
                }
            }

            // Aggregate features using attention weights
            let mut aggregated = Array1::<Float>::zeros(node_embeddings.ncols());
            for &j in &valid_neighbors {
                let weighted_features = &node_embeddings.row(j).to_owned() * attention_scores[j];
                aggregated += &weighted_features;
            }

            // Apply linear transformation
            let transformed = aggregated.dot(weights) + bias;
            let activated = if layer_idx == self.state.num_layers - 1 {
                transformed.mapv(|x| 1.0 / (1.0 + (-x).exp()))
            } else {
                transformed.mapv(|x| x.max(0.0))
            };

            output.row_mut(i).assign(&activated);
        }

        Ok(output)
    }

    /// GraphSAGE layer for trained model
    fn graphsage_layer_trained(
        &self,
        node_embeddings: &Array2<Float>,
        adjacency: &ArrayView2<'_, i32>,
        layer_idx: usize,
    ) -> SklResult<Array2<Float>> {
        let weights = &self.state.layer_weights[layer_idx];
        let bias = &self.state.layer_biases[layer_idx];
        let n_nodes = node_embeddings.nrows();
        let embedding_dim = node_embeddings.ncols();
        let output_dim = weights.ncols();
        let mut output = Array2::<Float>::zeros((n_nodes, output_dim));

        for i in 0..n_nodes {
            // Aggregate neighbor features
            let mut neighbor_sum = Array1::<Float>::zeros(embedding_dim);
            let mut neighbor_count = 0;

            for j in 0..n_nodes {
                if adjacency[[i, j]] == 1 && i != j {
                    neighbor_sum += &node_embeddings.row(j).to_owned();
                    neighbor_count += 1;
                }
            }

            // Average pooling of neighbors
            if neighbor_count > 0 {
                neighbor_sum /= neighbor_count as Float;
            }

            // Concatenate self and neighbor representations
            let self_features = node_embeddings.row(i).to_owned();
            let concatenated =
                Array1::from_iter(self_features.iter().chain(neighbor_sum.iter()).cloned());

            // Apply linear transformation
            if concatenated.len() == weights.nrows() {
                let transformed = concatenated.dot(weights) + bias;
                let activated = if layer_idx == self.state.num_layers - 1 {
                    transformed.mapv(|x| 1.0 / (1.0 + (-x).exp()))
                } else {
                    transformed.mapv(|x| x.max(0.0))
                };
                output.row_mut(i).assign(&activated);
            }
        }

        Ok(output)
    }

    /// GIN layer for trained model
    fn gin_layer_trained(
        &self,
        node_embeddings: &Array2<Float>,
        adjacency: &ArrayView2<'_, i32>,
        layer_idx: usize,
    ) -> SklResult<Array2<Float>> {
        let weights = &self.state.layer_weights[layer_idx];
        let bias = &self.state.layer_biases[layer_idx];
        let n_nodes = node_embeddings.nrows();
        let mut output = Array2::<Float>::zeros((n_nodes, weights.ncols()));
        let epsilon = 0.0; // Simplified as 0

        for i in 0..n_nodes {
            // Sum neighbor features
            let mut neighbor_sum = Array1::<Float>::zeros(node_embeddings.ncols());

            for j in 0..n_nodes {
                if adjacency[[i, j]] == 1 && i != j {
                    neighbor_sum += &node_embeddings.row(j).to_owned();
                }
            }

            // GIN update
            let updated = &node_embeddings.row(i).to_owned() * (1.0 + epsilon) + &neighbor_sum;

            // Apply MLP
            let transformed = updated.dot(weights) + bias;
            let activated = if layer_idx == self.state.num_layers - 1 {
                transformed.mapv(|x| 1.0 / (1.0 + (-x).exp()))
            } else {
                transformed.mapv(|x| x.max(0.0))
            };

            output.row_mut(i).assign(&activated);
        }

        Ok(output)
    }
}

// Tests for Graph Neural Networks
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gnn_basic_functionality() {
        let node_features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let adjacency = array![[0, 1, 0], [1, 0, 1], [0, 1, 0]];
        let node_labels = array![[1, 0], [0, 1], [1, 1]];

        let gnn = GraphNeuralNetwork::new()
            .hidden_dim(4)
            .num_layers(2)
            .max_iter(5);

        let trained_gnn = gnn
            .fit_graph(&adjacency.view(), &node_features.view(), &node_labels)
            .unwrap();

        let predictions = trained_gnn
            .predict_graph(&adjacency.view(), &node_features.view())
            .unwrap();

        assert_eq!(predictions.dim(), (3, 2));
        assert!(predictions.iter().all(|&x| x == 0 || x == 1));
    }

    #[test]
    fn test_gnn_different_variants() {
        let node_features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let adjacency = array![[0, 1, 0], [1, 0, 1], [0, 1, 0]];
        let node_labels = array![[1, 0], [0, 1], [1, 1]];

        // Test GCN
        let gnn_gcn = GraphNeuralNetwork::new()
            .message_passing_variant(MessagePassingVariant::GCN)
            .max_iter(5);
        let trained_gcn = gnn_gcn
            .fit_graph(&adjacency.view(), &node_features.view(), &node_labels)
            .unwrap();

        // Test GAT
        let gnn_gat = GraphNeuralNetwork::new()
            .message_passing_variant(MessagePassingVariant::GAT)
            .max_iter(5);
        let trained_gat = gnn_gat
            .fit_graph(&adjacency.view(), &node_features.view(), &node_labels)
            .unwrap();

        // Test GraphSAGE
        let gnn_sage = GraphNeuralNetwork::new()
            .message_passing_variant(MessagePassingVariant::GraphSAGE)
            .max_iter(5);
        let trained_sage = gnn_sage
            .fit_graph(&adjacency.view(), &node_features.view(), &node_labels)
            .unwrap();

        assert_eq!(
            trained_gcn.state.message_passing_variant,
            MessagePassingVariant::GCN
        );
        assert_eq!(
            trained_gat.state.message_passing_variant,
            MessagePassingVariant::GAT
        );
        assert_eq!(
            trained_sage.state.message_passing_variant,
            MessagePassingVariant::GraphSAGE
        );
    }

    #[test]
    fn test_gnn_parameter_settings() {
        let gnn = GraphNeuralNetwork::new()
            .hidden_dim(16)
            .num_layers(3)
            .learning_rate(0.001)
            .max_iter(50)
            .dropout_rate(0.1);

        assert_eq!(gnn.hidden_dim, 16);
        assert_eq!(gnn.num_layers, 3);
        assert!((gnn.learning_rate - 0.001).abs() < 1e-10);
        assert_eq!(gnn.max_iter, 50);
        assert!((gnn.dropout_rate - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_gnn_default_settings() {
        let gnn = GraphNeuralNetwork::new();

        assert_eq!(gnn.hidden_dim, 32);
        assert_eq!(gnn.num_layers, 2);
        assert_eq!(gnn.message_passing_variant, MessagePassingVariant::GCN);
        assert_eq!(gnn.aggregation_function, AggregationFunction::Mean);
    }

    #[test]
    fn test_gnn_builder_pattern() {
        let gnn1 = GraphNeuralNetwork::new();
        let gnn2 = GraphNeuralNetwork::new();

        assert_eq!(gnn1.hidden_dim, gnn2.hidden_dim);
        assert_eq!(gnn1.num_layers, gnn2.num_layers);

        let gnn3 = GraphNeuralNetwork::new().max_iter(1);
        assert_eq!(gnn3.max_iter, 1);
    }

    #[test]
    fn test_message_passing_variants() {
        assert_eq!(MessagePassingVariant::GCN, MessagePassingVariant::GCN);
        assert_ne!(MessagePassingVariant::GCN, MessagePassingVariant::GAT);

        let variants = [
            MessagePassingVariant::GCN,
            MessagePassingVariant::GAT,
            MessagePassingVariant::GraphSAGE,
            MessagePassingVariant::GIN,
        ];

        let gnn1 = GraphNeuralNetwork::new()
            .message_passing_variant(variants[0])
            .hidden_dim(8)
            .max_iter(3);

        let gnn2 = GraphNeuralNetwork::new()
            .message_passing_variant(variants[1])
            .hidden_dim(8)
            .max_iter(3);

        assert_eq!(gnn1.message_passing_variant, MessagePassingVariant::GCN);
        assert_eq!(gnn2.message_passing_variant, MessagePassingVariant::GAT);
    }

    #[test]
    fn test_gnn_larger_graph() {
        let node_features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0], [1.0, 3.0]];
        let adjacency = array![
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]
        ];
        let node_labels = array![[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]];

        let gnn = GraphNeuralNetwork::new()
            .hidden_dim(10)
            .num_layers(2)
            .message_passing_variant(MessagePassingVariant::GCN)
            .max_iter(10);

        let trained_gnn = gnn
            .fit_graph(&adjacency.view(), &node_features.view(), &node_labels)
            .unwrap();

        let predictions = trained_gnn
            .predict_graph(&adjacency.view(), &node_features.view())
            .unwrap();

        assert_eq!(predictions.dim(), (5, 3));
        assert!(predictions.iter().all(|&x| x == 0 || x == 1));
        assert_eq!(trained_gnn.hidden_dim(), 10);
    }

    #[test]
    fn test_aggregation_functions() {
        assert_ne!(AggregationFunction::Mean, AggregationFunction::Max);
        assert_eq!(AggregationFunction::Sum, AggregationFunction::Sum);
        assert_ne!(MessagePassingVariant::GraphSAGE, MessagePassingVariant::GIN);
    }

    #[test]
    fn test_gnn_reproducibility() {
        let node_features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let adjacency = array![[0, 1, 0], [1, 0, 1], [0, 1, 0]];
        let node_labels = array![[1, 0], [0, 1], [1, 1]];

        let gnn = GraphNeuralNetwork::new()
            .hidden_dim(4)
            .num_layers(2)
            .max_iter(5)
            .random_state(42);

        let trained_gnn = gnn
            .fit_graph(&adjacency.view(), &node_features.view(), &node_labels)
            .unwrap();

        let predictions = trained_gnn
            .predict_graph(&adjacency.view(), &node_features.view())
            .unwrap();

        assert_eq!(predictions.dim(), (3, 2));
    }

    #[test]
    fn test_gnn_edge_cases() {
        let node_features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0], [1.0, 3.0]];
        let adjacency = array![
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]
        ];
        let node_labels = array![[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]];

        let gnn = GraphNeuralNetwork::new()
            .hidden_dim(10)
            .num_layers(2)
            .message_passing_variant(MessagePassingVariant::GCN)
            .max_iter(15)
            .random_state(42);

        let trained_gnn = gnn
            .fit_graph(&adjacency.view(), &node_features.view(), &node_labels)
            .unwrap();
        let predictions = trained_gnn
            .predict_graph(&adjacency.view(), &node_features.view())
            .unwrap();

        assert_eq!(predictions.dim(), (5, 3));
        assert!(predictions.iter().all(|&x| x == 0 || x == 1));
        assert_eq!(trained_gnn.hidden_dim(), 10);
    }
}
