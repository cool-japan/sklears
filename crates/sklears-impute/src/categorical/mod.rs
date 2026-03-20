//! Categorical data imputation methods
//!
//! This module provides imputation strategies specifically designed for categorical data
//! including hot-deck imputation, categorical clustering, association rules, and
//! mixed-type data handling methods.

pub mod association_rules;
pub mod clustering;
pub mod hot_deck;
pub mod random_forest;

// Re-export all the main types for convenience
pub use association_rules::{
    AssociationRule, AssociationRuleImputer, AssociationRuleImputerTrained, Item, Itemset,
};
pub use clustering::{CategoricalClusteringImputer, CategoricalClusteringImputerTrained};
pub use hot_deck::{HotDeckImputer, HotDeckImputerTrained};
pub use random_forest::{
    CategoricalRandomForestImputer, CategoricalRandomForestImputerTrained, CategoricalTree,
};
