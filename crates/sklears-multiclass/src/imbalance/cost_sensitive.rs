use std::marker::PhantomData;
use crate::core::{MulticlassError, Estimator, Fit, Predict, PredictProba};
use crate::imbalance::ClassDistribution;
use sklears_core::traits::Fit;

#[derive(Debug, Clone)]
pub struct CostMatrix {
    pub costs: HashMap<(usize, usize), f64>,
    pub num_classes: usize,
}

impl CostMatrix {
    pub fn new(num_classes: usize) -> Self {
        CostMatrix {
            costs: HashMap::new(),
            num_classes,
        }
    }
    
    pub fn set_cost(&mut self, true_class: usize, predicted_class: usize, cost: f64) {
        self.costs.insert((true_class, predicted_class), cost);
    }
    
    pub fn get_cost(&self, true_class: usize, predicted_class: usize) -> f64 {
        self.costs.get(&(true_class, predicted_class)).copied().unwrap_or(0.0)
    }
    
    pub fn from_class_weights(class_weights: &HashMap<usize, f64>) -> Self {
        let num_classes = class_weights.len();
        let mut cost_matrix = CostMatrix::new(num_classes);
        
        for (&true_class, &weight) in class_weights {
            for predicted_class in 0..num_classes {
                if true_class != predicted_class {
                    cost_matrix.set_cost(true_class, predicted_class, weight);
                }
            }
        }
        
        cost_matrix
    }
    
    pub fn from_distribution(distribution: &ClassDistribution) -> Self {
        let num_classes = distribution.num_classes;
        let mut cost_matrix = CostMatrix::new(num_classes);
        
        let max_count = *distribution.class_counts.values().max().unwrap_or(&1);
        
        for (&class, &count) in &distribution.class_counts {
            let weight = max_count as f64 / count as f64;
            for predicted_class in 0..num_classes {
                if class != predicted_class {
                    cost_matrix.set_cost(class, predicted_class, weight);
                }
            }
        }
        
        cost_matrix
    }
    
    pub fn balanced_weights(distribution: &ClassDistribution) -> HashMap<usize, f64> {
        let n_samples = distribution.total_samples as f64;
        let n_classes = distribution.num_classes as f64;
        
        distribution.class_counts.iter()
            .map(|(&class, &count)| {
                let weight = n_samples / (n_classes * count as f64);
                (class, weight)
            })
            .collect()
    }
    
    pub fn expected_cost(&self, true_class: usize, probabilities: &[f64]) -> f64 {
        probabilities.iter()
            .enumerate()
            .map(|(predicted_class, &prob)| prob * self.get_cost(true_class, predicted_class))
            .sum()
    }
    
    pub fn total_cost(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
        y_true.iter()
            .zip(y_pred.iter())
            .map(|(&true_class, &pred_class)| self.get_cost(true_class, pred_class))
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct CostSensitiveClassifier<E> {
    base_estimator: E,
    cost_matrix: CostMatrix,
    decision_threshold: f64,
}

impl<E> CostSensitiveClassifier<E> {
    pub fn new(base_estimator: E, cost_matrix: CostMatrix) -> Self {
        CostSensitiveClassifier {
            base_estimator,
            cost_matrix,
            decision_threshold: 0.5,
        }
    }
    
    pub fn builder() -> CostSensitiveClassifierBuilder<E> {
        CostSensitiveClassifierBuilder::new()
    }
    
    pub fn set_decision_threshold(&mut self, threshold: f64) {
        self.decision_threshold = threshold;
    }
}

impl<E> Estimator for CostSensitiveClassifier<E> {
    fn name(&self) -> &'static str {
        "CostSensitiveClassifier"
    }
}

impl<E> Fit<Vec<Vec<f64>>, Vec<usize>> for CostSensitiveClassifier<E>
where
    E: Fit<Vec<Vec<f64>>, Vec<usize>>,
{
    type Fitted = FittedCostSensitiveClassifier<E::Fitted>;
    
    fn fit(&self, x: &Vec<Vec<f64>>, y: &Vec<usize>) -> Result<Self::Fitted, MulticlassError> {
        let distribution = ClassDistribution::new(y)?;
        let class_weights = self.calculate_class_weights(&distribution);
        
        let weighted_estimator = self.apply_class_weights(&self.base_estimator, &class_weights);
        let fitted_estimator = weighted_estimator.fit(x, y)?;
        
        Ok(FittedCostSensitiveClassifier {
            fitted_estimator,
            cost_matrix: self.cost_matrix.clone(),
            decision_threshold: self.decision_threshold,
        })
    }
}

impl<E> CostSensitiveClassifier<E> {
    fn calculate_class_weights(&self, distribution: &ClassDistribution) -> HashMap<usize, f64> {
        CostMatrix::balanced_weights(distribution)
    }
    
    fn apply_class_weights(&self, estimator: &E, _weights: &HashMap<usize, f64>) -> E
    where
        E: Clone,
    {
        estimator.clone()
    }
}

#[derive(Debug, Clone)]
pub struct FittedCostSensitiveClassifier<E> {
    fitted_estimator: E,
    cost_matrix: CostMatrix,
    decision_threshold: f64,
}

impl<E> Estimator for FittedCostSensitiveClassifier<E> {
    fn name(&self) -> &'static str {
        "FittedCostSensitiveClassifier"
    }
}

impl<E> Predict<Vec<Vec<f64>>, Vec<usize>> for FittedCostSensitiveClassifier<E>
where
    E: PredictProba<Vec<Vec<f64>>, Vec<Vec<f64>>>,
{
    fn predict(&self, x: &Vec<Vec<f64>>) -> Result<Vec<usize>, MulticlassError> {
        let probabilities = self.fitted_estimator.predict_proba(x)?;
        let predictions = probabilities.iter()
            .map(|probs| self.cost_sensitive_predict(probs))
            .collect();
        
        Ok(predictions)
    }
}

impl<E> PredictProba<Vec<Vec<f64>>, Vec<Vec<f64>>> for FittedCostSensitiveClassifier<E>
where
    E: PredictProba<Vec<Vec<f64>>, Vec<Vec<f64>>>,
{
    fn predict_proba(&self, x: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, MulticlassError> {
        self.fitted_estimator.predict_proba(x)
    }
}

impl<E> FittedCostSensitiveClassifier<E> {
    fn cost_sensitive_predict(&self, probabilities: &[f64]) -> usize {
        (0..probabilities.len())
            .min_by(|&a, &b| {
                let cost_a = self.cost_matrix.expected_cost(a, probabilities);
                let cost_b = self.cost_matrix.expected_cost(b, probabilities);
                cost_a.partial_cmp(&cost_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct CostSensitiveClassifierBuilder<E> {
    base_estimator: Option<E>,
    cost_matrix: Option<CostMatrix>,
    decision_threshold: f64,
    phantom: PhantomData<E>,
}

impl<E> CostSensitiveClassifierBuilder<E> {
    pub fn new() -> Self {
        CostSensitiveClassifierBuilder {
            base_estimator: None,
            cost_matrix: None,
            decision_threshold: 0.5,
            phantom: PhantomData,
        }
    }
    
    pub fn base_estimator(mut self, estimator: E) -> Self {
        self.base_estimator = Some(estimator);
        self
    }
    
    pub fn cost_matrix(mut self, cost_matrix: CostMatrix) -> Self {
        self.cost_matrix = Some(cost_matrix);
        self
    }
    
    pub fn decision_threshold(mut self, threshold: f64) -> Self {
        self.decision_threshold = threshold;
        self
    }
    
    pub fn build(self) -> Result<CostSensitiveClassifier<E>, MulticlassError> {
        let base_estimator = self.base_estimator.ok_or_else(|| {
            MulticlassError::InvalidConfiguration("Base estimator is required".to_string())
        })?;
        
        let cost_matrix = self.cost_matrix.ok_or_else(|| {
            MulticlassError::InvalidConfiguration("Cost matrix is required".to_string())
        })?;
        
        Ok(CostSensitiveClassifier {
            base_estimator,
            cost_matrix,
            decision_threshold: self.decision_threshold,
        })
    }
}

#[derive(Debug, Clone)]
pub enum CostSensitiveStrategy {
    /// ClassWeighting

    ClassWeighting,
    /// ThresholdMoving

    ThresholdMoving,
    /// MetaCost

    MetaCost,
    /// CostSensitiveDecisionTree

    CostSensitiveDecisionTree,
}

pub fn create_cost_sensitive_wrapper<E>(

    base_estimator: E,

    strategy: CostSensitiveStrategy,

    cost_matrix: Option<CostMatrix>,
    distribution: &ClassDistribution,
) -> Result<CostSensitiveClassifier<E>, MulticlassError> {
    let cost_matrix = cost_matrix.unwrap_or_else(|| CostMatrix::from_distribution(distribution));
    
    match strategy {
        CostSensitiveStrategy::ClassWeighting => {
            CostSensitiveClassifier::builder()
                .base_estimator(base_estimator)
                .cost_matrix(cost_matrix)
                .build()
        }
        _ => {
            CostSensitiveClassifier::builder()
                .base_estimator(base_estimator)
                .cost_matrix(cost_matrix)
                .build()
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::MockClassifier;
    
    #[test]
    fn test_cost_matrix_creation() {
        let mut cost_matrix = CostMatrix::new(3);
        cost_matrix.set_cost(0, 1, 2.0);
        cost_matrix.set_cost(1, 0, 1.5);
        
        assert_eq!(cost_matrix.get_cost(0, 1), 2.0);
        assert_eq!(cost_matrix.get_cost(1, 0), 1.5);
        assert_eq!(cost_matrix.get_cost(0, 0), 0.0);
    }
    
    #[test]
    fn test_cost_matrix_from_distribution() {
        let y = vec![0, 0, 0, 1, 1, 2];
        let distribution = ClassDistribution::new(&y).unwrap();
        let cost_matrix = CostMatrix::from_distribution(&distribution);
        
        assert_eq!(cost_matrix.num_classes, 3);
        assert!(cost_matrix.get_cost(2, 0) > cost_matrix.get_cost(0, 2));
    }
    
    #[test]
    fn test_balanced_weights() {
        let y = vec![0, 0, 0, 1, 1, 2];
        let distribution = ClassDistribution::new(&y).unwrap();
        let weights = CostMatrix::balanced_weights(&distribution);
        
        assert!(weights[&2] > weights[&0]);
        assert!(weights[&1] > weights[&0]);
    }
    
    #[test]
    fn test_expected_cost() {
        let mut cost_matrix = CostMatrix::new(3);
        cost_matrix.set_cost(0, 1, 2.0);
        cost_matrix.set_cost(0, 2, 3.0);
        
        let probabilities = vec![0.5, 0.3, 0.2];
        let expected_cost = cost_matrix.expected_cost(0, &probabilities);
        
        assert_eq!(expected_cost, 2.0 * 0.3 + 3.0 * 0.2);
    }
    
    #[test]
    fn test_cost_sensitive_classifier_builder() {
        let mock_classifier = MockClassifier::new();
        let cost_matrix = CostMatrix::new(3);
        
        let result = CostSensitiveClassifier::builder()
            .base_estimator(mock_classifier)
            .cost_matrix(cost_matrix)
            .decision_threshold(0.4)
            .build();
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_cost_sensitive_wrapper() {
        let mock_classifier = MockClassifier::new();
        let y = vec![0, 0, 0, 1, 1, 2];
        let distribution = ClassDistribution::new(&y).unwrap();
        
        let result = create_cost_sensitive_wrapper(
            mock_classifier,
            CostSensitiveStrategy::ClassWeighting,
            None,
            &distribution,
        );
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_total_cost() {
        let mut cost_matrix = CostMatrix::new(3);
        cost_matrix.set_cost(0, 1, 2.0);
        cost_matrix.set_cost(1, 0, 1.5);
        cost_matrix.set_cost(2, 0, 3.0);
        
        let y_true = vec![0, 1, 2];
        let y_pred = vec![1, 0, 0];
        let total_cost = cost_matrix.total_cost(&y_true, &y_pred);
        
        assert_eq!(total_cost, 2.0 + 1.5 + 3.0);
    }
}