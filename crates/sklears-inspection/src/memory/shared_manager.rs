//! Shared explanation management with reference counting
//!
//! This module provides reference-counted shared explanation data with automatic lifecycle
//! management, access tracking, and cleanup capabilities for efficient memory usage.

use crate::types::*;
use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// Reference-counted shared explanation data
#[derive(Debug, Clone)]
pub struct SharedExplanation {
    /// Reference-counted explanation data
    inner: Arc<ExplanationData>,
    /// Unique identifier for tracking
    id: String,
    /// Reference count manager
    manager: Option<Arc<SharedExplanationManager>>,
}

/// Internal explanation data
#[derive(Debug)]
pub struct ExplanationData {
    /// Feature importance values
    pub feature_importance: Array1<Float>,
    /// SHAP values (if available)
    pub shap_values: Option<Array2<Float>>,
    /// Partial dependence data (if available)
    pub partial_dependence: Option<Array2<Float>>,
    /// Model metadata
    pub metadata: ExplanationMetadata,
    /// Creation timestamp
    pub created_at: u64,
    /// Last accessed timestamp
    pub last_accessed: Arc<Mutex<u64>>,
    /// Access count
    pub access_count: Arc<Mutex<usize>>,
}

/// Metadata associated with explanations
#[derive(Debug, Clone)]
pub struct ExplanationMetadata {
    /// Model type
    pub model_type: String,
    /// Number of features
    pub n_features: usize,
    /// Number of samples used for explanation
    pub n_samples: usize,
    /// Explanation method used
    pub method: String,
    /// Configuration parameters
    pub config: HashMap<String, String>,
}

/// Manager for shared explanations with reference counting
#[derive(Debug)]
pub struct SharedExplanationManager {
    /// Active explanations (strong references)
    active: Mutex<HashMap<String, Arc<ExplanationData>>>,
    /// Weak references for cleanup tracking
    weak_refs: Mutex<HashMap<String, Weak<ExplanationData>>>,
    /// Configuration
    config: SharedExplanationConfig,
    /// Access statistics
    stats: Mutex<ManagerStats>,
}

/// Configuration for shared explanation manager
#[derive(Debug, Clone)]
pub struct SharedExplanationConfig {
    /// Maximum number of cached explanations
    pub max_cached: usize,
    /// Enable automatic cleanup of unused explanations
    pub auto_cleanup: bool,
    /// Cleanup interval in seconds
    pub cleanup_interval: u64,
    /// Maximum age for unused explanations (seconds)
    pub max_age: u64,
    /// Enable access tracking
    pub track_access: bool,
}

/// Manager statistics
#[derive(Debug, Default)]
pub struct ManagerStats {
    /// Total explanations created
    pub total_created: usize,
    /// Currently active explanations
    pub active_count: usize,
    /// Total memory usage estimate (bytes)
    pub estimated_memory_usage: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
}

impl Default for SharedExplanationConfig {
    fn default() -> Self {
        Self {
            max_cached: 100,
            auto_cleanup: true,
            cleanup_interval: 300, // 5 minutes
            max_age: 3600,         // 1 hour
            track_access: true,
        }
    }
}

impl SharedExplanationManager {
    /// Create a new shared explanation manager
    pub fn new(config: SharedExplanationConfig) -> Self {
        Self {
            active: Mutex::new(HashMap::new()),
            weak_refs: Mutex::new(HashMap::new()),
            config,
            stats: Mutex::new(ManagerStats::default()),
        }
    }

    /// Create a new shared explanation
    pub fn create_explanation(
        &self,
        id: String,
        feature_importance: Array1<Float>,
        shap_values: Option<Array2<Float>>,
        metadata: ExplanationMetadata,
    ) -> SklResult<SharedExplanation> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let explanation_data = Arc::new(ExplanationData {
            feature_importance,
            shap_values,
            partial_dependence: None,
            metadata,
            created_at: now,
            last_accessed: Arc::new(Mutex::new(now)),
            access_count: Arc::new(Mutex::new(0)),
        });

        // Store in active explanations
        {
            let mut active = self.active.lock().unwrap();
            active.insert(id.clone(), explanation_data.clone());
        }

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_created += 1;
            stats.active_count += 1;
            stats.estimated_memory_usage += self.estimate_explanation_size(&explanation_data);
        }

        // Create weak reference for tracking
        {
            let mut weak_refs = self.weak_refs.lock().unwrap();
            weak_refs.insert(id.clone(), Arc::downgrade(&explanation_data));
        }

        Ok(SharedExplanation {
            inner: explanation_data,
            id,
            manager: Some(Arc::new(self.clone())),
        })
    }

    /// Get an existing shared explanation
    pub fn get_explanation(&self, id: &str) -> Option<SharedExplanation> {
        let active = self.active.lock().unwrap();
        if let Some(explanation_data) = active.get(id) {
            // Update access statistics
            if self.config.track_access {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                *explanation_data.last_accessed.lock().unwrap() = now;
                *explanation_data.access_count.lock().unwrap() += 1;
            }

            // Update cache hits
            {
                let mut stats = self.stats.lock().unwrap();
                stats.cache_hits += 1;
            }

            Some(SharedExplanation {
                inner: explanation_data.clone(),
                id: id.to_string(),
                manager: Some(Arc::new(self.clone())),
            })
        } else {
            // Update cache misses
            {
                let mut stats = self.stats.lock().unwrap();
                stats.cache_misses += 1;
            }
            None
        }
    }

    /// Remove an explanation from the manager
    pub fn remove_explanation(&self, id: &str) -> bool {
        let removed = {
            let mut active = self.active.lock().unwrap();
            active.remove(id).is_some()
        };

        if removed {
            let mut weak_refs = self.weak_refs.lock().unwrap();
            weak_refs.remove(id);

            let mut stats = self.stats.lock().unwrap();
            stats.active_count = stats.active_count.saturating_sub(1);
        }

        removed
    }

    /// Cleanup unused explanations
    pub fn cleanup_unused(&self) -> usize {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut removed_count = 0;
        let mut to_remove = Vec::new();

        // Find explanations to remove
        {
            let active = self.active.lock().unwrap();
            for (id, explanation) in active.iter() {
                let last_accessed = *explanation.last_accessed.lock().unwrap();
                let age = now.saturating_sub(last_accessed);

                if age > self.config.max_age {
                    to_remove.push(id.clone());
                }
            }
        }

        // Remove old explanations
        for id in to_remove {
            if self.remove_explanation(&id) {
                removed_count += 1;
            }
        }

        // Also check weak references and clean up dead ones
        {
            let mut weak_refs = self.weak_refs.lock().unwrap();
            weak_refs.retain(|_, weak_ref| weak_ref.strong_count() > 0);
        }

        removed_count
    }

    /// Get manager statistics
    pub fn get_stats(&self) -> ManagerStats {
        let stats = self.stats.lock().unwrap();

        // Update active count from actual data
        let active_count = {
            let active = self.active.lock().unwrap();
            active.len()
        };

        ManagerStats {
            total_created: stats.total_created,
            active_count,
            estimated_memory_usage: stats.estimated_memory_usage,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
        }
    }

    /// Estimate memory usage of an explanation
    fn estimate_explanation_size(&self, explanation: &ExplanationData) -> usize {
        let mut size = 0;

        // Feature importance
        size += explanation.feature_importance.len() * std::mem::size_of::<Float>();

        // SHAP values
        if let Some(shap) = &explanation.shap_values {
            size += shap.len() * std::mem::size_of::<Float>();
        }

        // Partial dependence
        if let Some(pd) = &explanation.partial_dependence {
            size += pd.len() * std::mem::size_of::<Float>();
        }

        // Metadata overhead (rough estimate)
        size += 1024; // Approximate overhead for metadata, timestamps, etc.

        size
    }
}

// Clone implementation for SharedExplanationManager (for Arc wrapping)
impl Clone for SharedExplanationManager {
    fn clone(&self) -> Self {
        // This is a deep clone for creating a new manager with the same config
        Self::new(self.config.clone())
    }
}

impl SharedExplanation {
    /// Get feature importance values
    pub fn feature_importance(&self) -> &Array1<Float> {
        self.update_access();
        &self.inner.feature_importance
    }

    /// Get SHAP values if available
    pub fn shap_values(&self) -> Option<&Array2<Float>> {
        self.update_access();
        self.inner.shap_values.as_ref()
    }

    /// Get partial dependence data if available
    pub fn partial_dependence(&self) -> Option<&Array2<Float>> {
        self.update_access();
        self.inner.partial_dependence.as_ref()
    }

    /// Get explanation metadata
    pub fn metadata(&self) -> &ExplanationMetadata {
        &self.inner.metadata
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> u64 {
        self.inner.created_at
    }

    /// Get last access timestamp
    pub fn last_accessed(&self) -> u64 {
        *self.inner.last_accessed.lock().unwrap()
    }

    /// Get access count
    pub fn access_count(&self) -> usize {
        *self.inner.access_count.lock().unwrap()
    }

    /// Get unique identifier
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get reference count
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Check if this explanation is unique (only one reference)
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Create a copy of the explanation data (breaks sharing)
    pub fn make_unique(&self) -> SklResult<SharedExplanation> {
        let new_data = Arc::new(ExplanationData {
            feature_importance: self.inner.feature_importance.clone(),
            shap_values: self.inner.shap_values.clone(),
            partial_dependence: self.inner.partial_dependence.clone(),
            metadata: self.inner.metadata.clone(),
            created_at: self.inner.created_at,
            last_accessed: Arc::new(Mutex::new(*self.inner.last_accessed.lock().unwrap())),
            access_count: Arc::new(Mutex::new(*self.inner.access_count.lock().unwrap())),
        });

        Ok(SharedExplanation {
            inner: new_data,
            id: format!("{}_copy", self.id),
            manager: None, // Unique copy is not managed
        })
    }

    /// Update access statistics
    fn update_access(&self) {
        if let Some(manager) = &self.manager {
            if manager.config.track_access {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                *self.inner.last_accessed.lock().unwrap() = now;
                *self.inner.access_count.lock().unwrap() += 1;
            }
        }
    }
}

impl Drop for SharedExplanation {
    fn drop(&mut self) {
        // If this is the last reference and we have a manager, potentially cleanup
        if Arc::strong_count(&self.inner) == 1 {
            if let Some(manager) = &self.manager {
                if manager.config.auto_cleanup {
                    // Note: We can't actually remove here due to borrowing rules,
                    // but the manager's cleanup process will handle it
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_shared_explanation_manager_creation() {
        let config = SharedExplanationConfig::default();
        let manager = SharedExplanationManager::new(config);

        let stats = manager.get_stats();
        assert_eq!(stats.total_created, 0);
        assert_eq!(stats.active_count, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    #[test]
    fn test_shared_explanation_creation_and_access() {
        let config = SharedExplanationConfig::default();
        let manager = SharedExplanationManager::new(config);

        // Create test data
        let feature_importance = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let shap_values = array![[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]];
        let metadata = ExplanationMetadata {
            model_type: "RandomForest".to_string(),
            n_features: 5,
            n_samples: 2,
            method: "SHAP".to_string(),
            config: HashMap::new(),
        };

        // Create shared explanation
        let explanation = manager
            .create_explanation(
                "test_explanation".to_string(),
                feature_importance.clone(),
                Some(shap_values.clone()),
                metadata.clone(),
            )
            .unwrap();

        // Check data access
        assert_eq!(explanation.feature_importance().len(), 5);
        assert!(explanation.shap_values().is_some());
        assert_eq!(explanation.shap_values().unwrap().nrows(), 2);
        assert_eq!(explanation.metadata().model_type, "RandomForest");
        assert_eq!(explanation.id(), "test_explanation");

        // Check reference counting
        assert_eq!(explanation.ref_count(), 2); // One in manager, one here
        assert!(!explanation.is_unique());
    }

    #[test]
    fn test_shared_explanation_manager_get_and_cache() {
        let config = SharedExplanationConfig::default();
        let manager = SharedExplanationManager::new(config);

        // Create explanation
        let feature_importance = array![1.0, 2.0, 3.0];
        let metadata = ExplanationMetadata {
            model_type: "LinearRegression".to_string(),
            n_features: 3,
            n_samples: 1,
            method: "Permutation".to_string(),
            config: HashMap::new(),
        };

        let _explanation = manager
            .create_explanation(
                "cached_explanation".to_string(),
                feature_importance,
                None,
                metadata,
            )
            .unwrap();

        // Get explanation (should hit cache)
        let retrieved = manager.get_explanation("cached_explanation");
        assert!(retrieved.is_some());

        // Try to get non-existent explanation (should miss cache)
        let missing = manager.get_explanation("non_existent");
        assert!(missing.is_none());

        // Check statistics
        let stats = manager.get_stats();
        assert_eq!(stats.total_created, 1);
        assert_eq!(stats.active_count, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_shared_explanation_access_tracking() {
        let config = SharedExplanationConfig {
            track_access: true,
            ..SharedExplanationConfig::default()
        };
        let manager = SharedExplanationManager::new(config);

        // Create explanation
        let feature_importance = array![1.0, 2.0];
        let metadata = ExplanationMetadata {
            model_type: "SVM".to_string(),
            n_features: 2,
            n_samples: 1,
            method: "LIME".to_string(),
            config: HashMap::new(),
        };

        let explanation = manager
            .create_explanation(
                "access_tracked".to_string(),
                feature_importance,
                None,
                metadata,
            )
            .unwrap();

        let initial_access_count = explanation.access_count();

        // Access data multiple times
        let _ = explanation.feature_importance();
        let _ = explanation.metadata();
        let _ = explanation.created_at();

        // Access count should have increased
        assert!(explanation.access_count() > initial_access_count);
    }

    #[test]
    fn test_shared_explanation_make_unique() {
        let config = SharedExplanationConfig::default();
        let manager = SharedExplanationManager::new(config);

        // Create explanation
        let feature_importance = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let metadata = ExplanationMetadata {
            model_type: "XGBoost".to_string(),
            n_features: 5,
            n_samples: 1,
            method: "TreeSHAP".to_string(),
            config: HashMap::new(),
        };

        let explanation = manager
            .create_explanation(
                "to_be_unique".to_string(),
                feature_importance.clone(),
                None,
                metadata,
            )
            .unwrap();

        // Original should not be unique (shared with manager)
        assert!(!explanation.is_unique());

        // Create unique copy
        let unique_explanation = explanation.make_unique().unwrap();

        // Unique copy should be unique
        assert!(unique_explanation.is_unique());
        assert_eq!(unique_explanation.ref_count(), 1);

        // Data should be the same
        assert_eq!(unique_explanation.feature_importance().len(), 5);
        for (a, b) in unique_explanation
            .feature_importance()
            .iter()
            .zip(feature_importance.iter())
        {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_shared_explanation_manager_remove() {
        let config = SharedExplanationConfig::default();
        let manager = SharedExplanationManager::new(config);

        // Create explanation
        let feature_importance = array![1.0];
        let metadata = ExplanationMetadata {
            model_type: "DecisionTree".to_string(),
            n_features: 1,
            n_samples: 1,
            method: "Feature Importance".to_string(),
            config: HashMap::new(),
        };

        let _explanation = manager
            .create_explanation(
                "to_be_removed".to_string(),
                feature_importance,
                None,
                metadata,
            )
            .unwrap();

        // Check it exists
        assert!(manager.get_explanation("to_be_removed").is_some());

        // Remove it
        let removed = manager.remove_explanation("to_be_removed");
        assert!(removed);

        // Check it's gone
        assert!(manager.get_explanation("to_be_removed").is_none());

        // Try removing again (should return false)
        let removed_again = manager.remove_explanation("to_be_removed");
        assert!(!removed_again);
    }

    #[test]
    fn test_shared_explanation_manager_cleanup() {
        let config = SharedExplanationConfig {
            max_age: 1, // 1 second for testing
            auto_cleanup: true,
            ..SharedExplanationConfig::default()
        };
        let manager = SharedExplanationManager::new(config);

        // Create explanation
        let feature_importance = array![1.0, 2.0];
        let metadata = ExplanationMetadata {
            model_type: "KNN".to_string(),
            n_features: 2,
            n_samples: 1,
            method: "Distance".to_string(),
            config: HashMap::new(),
        };

        let _explanation = manager
            .create_explanation(
                "short_lived".to_string(),
                feature_importance,
                None,
                metadata,
            )
            .unwrap();

        // Initially should exist
        assert!(manager.get_explanation("short_lived").is_some());

        // Wait for aging
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Cleanup
        let cleaned_up = manager.cleanup_unused();
        assert_eq!(cleaned_up, 1);

        // Should be gone now
        assert!(manager.get_explanation("short_lived").is_none());
    }

    #[test]
    fn test_shared_explanation_manager_stats() {
        let config = SharedExplanationConfig::default();
        let manager = SharedExplanationManager::new(config);

        // Create multiple explanations
        for i in 0..3 {
            let feature_importance = array![i as Float];
            let metadata = ExplanationMetadata {
                model_type: format!("Model{}", i),
                n_features: 1,
                n_samples: 1,
                method: "Test".to_string(),
                config: HashMap::new(),
            };

            let _explanation = manager
                .create_explanation(
                    format!("explanation_{}", i),
                    feature_importance,
                    None,
                    metadata,
                )
                .unwrap();
        }

        let stats = manager.get_stats();
        assert_eq!(stats.total_created, 3);
        assert_eq!(stats.active_count, 3);
        assert!(stats.estimated_memory_usage > 0);
    }
}
