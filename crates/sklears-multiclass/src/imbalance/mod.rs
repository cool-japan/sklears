pub mod cost_sensitive;
pub mod sampling;
pub mod threshold_moving;
pub mod ensemble_imbalance;
pub mod adaptive_sampling;
pub mod evaluation;

use crate::core::MulticlassError;

#[derive(Debug, Clone)]
pub struct ClassDistribution {
    pub class_counts: HashMap<usize, usize>,
    pub total_samples: usize,
    pub num_classes: usize,
}

impl ClassDistribution {
    pub fn new(y: &[usize]) -> Result<Self, MulticlassError> {
        let mut class_counts = HashMap::new();
        
        for &class in y {
            *class_counts.entry(class).or_insert(0) += 1;
        }
        
        let total_samples = y.len();
        let num_classes = class_counts.len();
        
        Ok(ClassDistribution {
            class_counts,
            total_samples,
            num_classes,
        })
    }
    
    pub fn class_frequencies(&self) -> HashMap<usize, f64> {
        self.class_counts.iter()
            .map(|(&class, &count)| (class, count as f64 / self.total_samples as f64))
            .collect()
    }
    
    pub fn imbalance_ratio(&self) -> f64 {
        if self.class_counts.is_empty() {
            return 0.0;
        }
        
        let max_count = *self.class_counts.values().max().unwrap_or(&0);
        let min_count = *self.class_counts.values().min().unwrap_or(&1);
        
        if min_count == 0 {
            return f64::INFINITY;
        }
        
        max_count as f64 / min_count as f64
    }
    
    pub fn majority_class(&self) -> Option<usize> {
        self.class_counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&class, _)| class)
    }
    
    pub fn minority_class(&self) -> Option<usize> {
        self.class_counts.iter()
            .min_by_key(|(_, &count)| count)
            .map(|(&class, _)| class)
    }
    
    pub fn is_imbalanced(&self, threshold: f64) -> bool {
        self.imbalance_ratio() > threshold
    }
    
    pub fn gini_coefficient(&self) -> f64 {
        let n = self.total_samples as f64;
        let k = self.num_classes as f64;
        
        if n == 0.0 || k <= 1.0 {
            return 0.0;
        }
        
        let frequencies: Vec<f64> = self.class_frequencies().values().cloned().collect();
        let sum_squared: f64 = frequencies.iter().map(|&f| f * f).sum();
        
        (k / (k - 1.0)) * (1.0 - sum_squared)
    }
    
    pub fn effective_number_of_classes(&self) -> f64 {
        let frequencies: Vec<f64> = self.class_frequencies().values().cloned().collect();
        let shannon_entropy: f64 = frequencies.iter()
            .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
            .sum();
        
        2.0_f64.powf(shannon_entropy)
    }
}

#[derive(Debug, Clone)]
pub enum ImbalanceStrategy {

    None,
    /// CostSensitive

    CostSensitive,
    /// Sampling

    Sampling,
    /// ThresholdMoving

    ThresholdMoving,
    /// EnsembleImbalance

    EnsembleImbalance,
    /// AdaptiveSampling

    AdaptiveSampling,
    /// Hybrid

    Hybrid(Vec<ImbalanceStrategy>),
}

#[derive(Debug, Clone)]
pub struct ImbalanceConfig {
    pub strategy: ImbalanceStrategy,
    pub imbalance_threshold: f64,
    pub cost_matrix: Option<HashMap<(usize, usize), f64>>,
    pub sampling_ratio: f64,
    pub random_state: Option<u64>,
}

impl Default for ImbalanceConfig {
    fn default() -> Self {
        ImbalanceConfig {
            strategy: ImbalanceStrategy::None,
            imbalance_threshold: 2.0,
            cost_matrix: None,
            sampling_ratio: 1.0,
            random_state: None,
        }
    }
}

impl ImbalanceConfig {
    pub fn builder() -> ImbalanceConfigBuilder {
        ImbalanceConfigBuilder::default()
    }
}

#[derive(Debug, Clone)]
pub struct ImbalanceConfigBuilder {
    config: ImbalanceConfig,
}

impl Default for ImbalanceConfigBuilder {
    fn default() -> Self {
        ImbalanceConfigBuilder {
            config: ImbalanceConfig::default(),
        }
    }
}

impl ImbalanceConfigBuilder {
    pub fn strategy(mut self, strategy: ImbalanceStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }
    
    pub fn imbalance_threshold(mut self, threshold: f64) -> Self {
        self.config.imbalance_threshold = threshold;
        self
    }
    
    pub fn cost_matrix(mut self, cost_matrix: HashMap<(usize, usize), f64>) -> Self {
        self.config.cost_matrix = Some(cost_matrix);
        self
    }
    
    pub fn sampling_ratio(mut self, ratio: f64) -> Self {
        self.config.sampling_ratio = ratio;
        self
    }
    
    pub fn random_state(mut self, state: u64) -> Self {
        self.config.random_state = Some(state);
        self
    }
    
    pub fn build(self) -> ImbalanceConfig {
        self.config
    }
}

pub fn recommend_strategy(distribution: &ClassDistribution) -> ImbalanceStrategy {
    let imbalance_ratio = distribution.imbalance_ratio();
    let num_classes = distribution.num_classes;
    let total_samples = distribution.total_samples;
    
    match (imbalance_ratio, num_classes, total_samples) {
        _ if imbalance_ratio <= 2.0 => ImbalanceStrategy::None,
        _ if imbalance_ratio <= 5.0 && num_classes <= 5 => ImbalanceStrategy::CostSensitive,
        _ if imbalance_ratio <= 10.0 && total_samples >= 1000 => ImbalanceStrategy::Sampling,
        _ if imbalance_ratio <= 20.0 => ImbalanceStrategy::ThresholdMoving,
        _ if num_classes >= 10 => ImbalanceStrategy::EnsembleImbalance,
        _ => ImbalanceStrategy::AdaptiveSampling,
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_class_distribution() {
        let y = vec![0, 0, 0, 1, 1, 2];
        let dist = ClassDistribution::new(&y).unwrap();
        
        assert_eq!(dist.total_samples, 6);
        assert_eq!(dist.num_classes, 3);
        assert_eq!(dist.class_counts[&0], 3);
        assert_eq!(dist.class_counts[&1], 2);
        assert_eq!(dist.class_counts[&2], 1);
        
        assert_eq!(dist.majority_class(), Some(0));
        assert_eq!(dist.minority_class(), Some(2));
        assert_eq!(dist.imbalance_ratio(), 3.0);
    }
    
    #[test]
    fn test_imbalance_detection() {
        let y = vec![0, 0, 0, 0, 1];
        let dist = ClassDistribution::new(&y).unwrap();
        
        assert!(dist.is_imbalanced(3.0));
        assert!(!dist.is_imbalanced(5.0));
    }
    
    #[test]
    fn test_strategy_recommendation() {
        let y_balanced = vec![0, 0, 1, 1, 2, 2];
        let dist_balanced = ClassDistribution::new(&y_balanced).unwrap();
        
        match recommend_strategy(&dist_balanced) {
            ImbalanceStrategy::None => {},
            _ => panic!("Expected None strategy for balanced data"),
        }
        
        let y_imbalanced = vec![0, 0, 0, 0, 1, 2];
        let dist_imbalanced = ClassDistribution::new(&y_imbalanced).unwrap();
        
        match recommend_strategy(&dist_imbalanced) {
            ImbalanceStrategy::CostSensitive => {},
            _ => panic!("Expected CostSensitive strategy for moderate imbalance"),
        }
    }
    
    #[test]
    fn test_gini_coefficient() {
        let y_balanced = vec![0, 0, 1, 1, 2, 2];
        let dist_balanced = ClassDistribution::new(&y_balanced).unwrap();
        let gini_balanced = dist_balanced.gini_coefficient();
        
        let y_imbalanced = vec![0, 0, 0, 0, 1, 2];
        let dist_imbalanced = ClassDistribution::new(&y_imbalanced).unwrap();
        let gini_imbalanced = dist_imbalanced.gini_coefficient();
        
        assert!(gini_imbalanced < gini_balanced);
    }
    
    #[test]
    fn test_effective_number_of_classes() {
        let y_balanced = vec![0, 0, 1, 1, 2, 2];
        let dist_balanced = ClassDistribution::new(&y_balanced).unwrap();
        let enc_balanced = dist_balanced.effective_number_of_classes();
        
        let y_imbalanced = vec![0, 0, 0, 0, 1, 2];
        let dist_imbalanced = ClassDistribution::new(&y_imbalanced).unwrap();
        let enc_imbalanced = dist_imbalanced.effective_number_of_classes();
        
        assert!(enc_balanced > enc_imbalanced);
    }
    
    #[test]
    fn test_builder_pattern() {
        let config = ImbalanceConfig::builder()
            .strategy(ImbalanceStrategy::CostSensitive)
            .imbalance_threshold(5.0)
            .sampling_ratio(0.8)
            .random_state(42)
            .build();
        
        assert_eq!(config.imbalance_threshold, 5.0);
        assert_eq!(config.sampling_ratio, 0.8);
        assert_eq!(config.random_state, Some(42));
    }
}