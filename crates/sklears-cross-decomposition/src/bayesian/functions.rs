//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{BayesianCCA, HierarchicalBayesianCCA, VariationalPLS};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    #[test]
    fn test_bayesian_cca_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5], [5.5, 6.5]];
        let bcca = BayesianCCA::new(1)
            .n_samples(100)
            .burn_in(20)
            .random_state(42);
        let result = bcca.fit(&x, &y).expect("fit should succeed");
        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0] > 0.5);
        let x_weights = result.x_weights();
        let y_weights = result.y_weights();
        assert_eq!(x_weights.shape(), &[2, 1]);
        assert_eq!(y_weights.shape(), &[2, 1]);
    }
    #[test]
    fn test_bayesian_cca_multiple_components() {
        let x = array![
            [1.0, 2.0, 0.5],
            [2.0, 3.0, 1.0],
            [3.0, 4.0, 1.5],
            [4.0, 5.0, 2.0],
            [5.0, 6.0, 2.5]
        ];
        let y = array![
            [1.5, 2.5, 0.8],
            [2.5, 3.5, 1.2],
            [3.5, 4.5, 1.8],
            [4.5, 5.5, 2.2],
            [5.5, 6.5, 2.8]
        ];
        let bcca = BayesianCCA::new(2)
            .n_samples(50)
            .burn_in(10)
            .random_state(42);
        let result = bcca.fit(&x, &y).expect("fit should succeed");
        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 2);
        let intervals = result.correlation_credible_intervals(0.95);
        assert_eq!(intervals.shape(), &[2, 2]);
        for k in 0..2 {
            assert!(intervals[[k, 0]] <= intervals[[k, 1]]);
        }
    }
    #[test]
    fn test_bayesian_cca_transform() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let bcca = BayesianCCA::new(1)
            .n_samples(30)
            .burn_in(10)
            .random_state(42);
        let result = bcca.fit(&x, &y).expect("fit should succeed");
        let x_transformed = result.transform_x(&x).expect("operation should succeed");
        let y_transformed = result.transform_y(&y).expect("transform_y should succeed");
        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);
    }
    #[test]
    fn test_bayesian_cca_diagnostics() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let bcca = BayesianCCA::new(1)
            .n_samples(50)
            .burn_in(10)
            .random_state(42);
        let result = bcca.fit(&x, &y).expect("fit should succeed");
        let ess = result.effective_sample_size();
        assert_eq!(ess.len(), 1);
        assert!(ess[0] > 0.0);
        let diagnostics = result.mcmc_diagnostics();
        assert!(diagnostics.contains_key("min_ess"));
        assert!(diagnostics.contains_key("mean_ess"));
        assert!(diagnostics.contains_key("n_samples"));
        assert!(diagnostics["n_samples"] >= 20.0 && diagnostics["n_samples"] <= 50.0);
    }
    #[test]
    fn test_bayesian_cca_error_cases() {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];
        let bcca = BayesianCCA::new(1);
        let result = bcca.fit(&x, &y);
        assert!(result.is_err());
        let x_small = array![[1.0, 2.0]];
        let y_small = array![[1.5, 2.5]];
        let bcca2 = BayesianCCA::new(1);
        let result2 = bcca2.fit(&x_small, &y_small);
        assert!(result2.is_err());
    }
    #[test]
    fn test_bayesian_cca_prior_settings() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let bcca = BayesianCCA::new(1)
            .n_samples(30)
            .burn_in(5)
            .thin(2)
            .prior_precision(2.0)
            .random_state(42);
        let result = bcca.fit(&x, &y).expect("fit should succeed");
        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 1);
        let diagnostics = result.mcmc_diagnostics();
        assert!(diagnostics["n_samples"] <= 15.0);
    }
    #[test]
    fn test_variational_pls_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5], [5.5]];
        let vpls = VariationalPLS::new(1).max_iter(50).tolerance(1e-4);
        let result = vpls.fit(&x, &y).expect("fit should succeed");
        let x_loadings = result.x_loadings();
        let y_loadings = result.y_loadings();
        assert_eq!(x_loadings.shape(), &[2, 1]);
        assert_eq!(y_loadings.shape(), &[1, 1]);
        let noise_precision = result.noise_precision();
        assert!(noise_precision > 0.0);
        let feature_relevance = result.feature_relevance();
        assert_eq!(feature_relevance.len(), 2);
        assert!(feature_relevance.iter().all(|&x| x > 0.0));
    }
    #[test]
    fn test_variational_pls_transform() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5]];
        let vpls = VariationalPLS::new(1).max_iter(30).tolerance(1e-4);
        let result = vpls.fit(&x, &y).expect("fit should succeed");
        let x_transformed = result.transform_x(&x).expect("operation should succeed");
        let y_transformed = result.transform_y(&y).expect("transform_y should succeed");
        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);
        let y_pred = result.predict(&x).expect("predict should succeed");
        assert_eq!(y_pred.shape(), &[4, 1]);
    }
    #[test]
    fn test_variational_pls_uncertainty() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5]];
        let vpls = VariationalPLS::new(1).max_iter(30).tolerance(1e-4);
        let result = vpls.fit(&x, &y).expect("fit should succeed");
        let (std_wx, std_wy) = result.loading_standard_deviations();
        assert_eq!(std_wx.shape(), &[2, 1]);
        assert_eq!(std_wy.shape(), &[1, 1]);
        assert!(std_wx.iter().all(|&x| x >= 0.0));
        assert!(std_wy.iter().all(|&x| x >= 0.0));
        let elbo_history = result.elbo_history();
        assert!(!elbo_history.is_empty());
        if elbo_history.len() > 1 {
            let first = elbo_history[0];
            let last = elbo_history[elbo_history.len() - 1];
            assert!(last >= first - 1e-6);
        }
    }
    #[test]
    fn test_variational_pls_convergence() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5]];
        let vpls = VariationalPLS::new(1).max_iter(100).tolerance(1e-8);
        let result = vpls.fit(&x, &y).expect("fit should succeed");
        let elbo_history = result.elbo_history();
        if elbo_history.len() > 10 {
            let last_few = &elbo_history[elbo_history.len() - 3..];
            let variance = last_few
                .iter()
                .map(|&x| (x - last_few.iter().sum::<f64>() / last_few.len() as f64).powi(2))
                .sum::<f64>()
                / last_few.len() as f64;
            assert!(variance < 1e-10);
        }
    }
    #[test]
    fn test_variational_pls_error_cases() {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.5], [2.5], [3.5]];
        let vpls = VariationalPLS::new(1);
        let result = vpls.fit(&x, &y);
        assert!(result.is_err());
        let x_small = array![[1.0, 2.0]];
        let y_small = array![[1.5]];
        let vpls2 = VariationalPLS::new(1);
        let result2 = vpls2.fit(&x_small, &y_small);
        assert!(result2.is_err());
    }
    #[test]
    fn test_variational_pls_hyperparameters() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5], [2.5], [3.5], [4.5]];
        let vpls = VariationalPLS::new(1)
            .max_iter(20)
            .tolerance(1e-5)
            .ard_alpha(1e-4)
            .ard_beta(1e-4);
        let result = vpls.fit(&x, &y).expect("fit should succeed");
        let feature_relevance = result.feature_relevance();
        assert!(feature_relevance.iter().all(|&x| x > 0.0));
        let noise_precision = result.noise_precision();
        assert!(noise_precision > 0.0);
    }
    #[test]
    fn test_hierarchical_bayesian_cca_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];
        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(50)
            .burn_in(10)
            .random_state(42);
        let result = hbcca.fit(&x, &y, &groups).expect("fit should succeed");
        let (pop_x, pop_y) = result.population_effects();
        let (group_x, group_y) = result.group_effects();
        assert_eq!(pop_x.shape(), &[2, 1]);
        assert_eq!(pop_y.shape(), &[2, 1]);
        assert_eq!(group_x.shape(), &[2, 2, 1]);
        assert_eq!(group_y.shape(), &[2, 2, 1]);
        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0] >= -1.0 && correlations[0] <= 1.0);
    }
    #[test]
    fn test_hierarchical_bayesian_cca_transform() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];
        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(30)
            .burn_in(5)
            .random_state(42);
        let result = hbcca.fit(&x, &y, &groups).expect("fit should succeed");
        let x_group0 = result
            .transform_x_group(&x.slice(scirs2_core::ndarray::s![0..2, ..]).to_owned(), 0)
            .expect("operation should succeed");
        let y_group0 = result
            .transform_y_group(&y.slice(scirs2_core::ndarray::s![0..2, ..]).to_owned(), 0)
            .expect("operation should succeed");
        assert_eq!(x_group0.shape(), &[2, 1]);
        assert_eq!(y_group0.shape(), &[2, 1]);
        let x_group1 = result
            .transform_x_group(&x.slice(scirs2_core::ndarray::s![2..4, ..]).to_owned(), 1)
            .expect("operation should succeed");
        let y_group1 = result
            .transform_y_group(&y.slice(scirs2_core::ndarray::s![2..4, ..]).to_owned(), 1)
            .expect("operation should succeed");
        assert_eq!(x_group1.shape(), &[2, 1]);
        assert_eq!(y_group1.shape(), &[2, 1]);
    }
    #[test]
    fn test_hierarchical_bayesian_cca_variance_decomposition() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];
        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(30)
            .burn_in(5)
            .random_state(42);
        let result = hbcca.fit(&x, &y, &groups).expect("fit should succeed");
        let variance_decomp = result.variance_decomposition();
        assert!(variance_decomp.contains_key("between_group_var_x"));
        assert!(variance_decomp.contains_key("between_group_var_y"));
        let between_var_x = &variance_decomp["between_group_var_x"];
        let between_var_y = &variance_decomp["between_group_var_y"];
        assert_eq!(between_var_x.len(), 1);
        assert_eq!(between_var_y.len(), 1);
        assert!(between_var_x[0] >= 0.0);
        assert!(between_var_y[0] >= 0.0);
    }
    #[test]
    fn test_hierarchical_bayesian_cca_credible_intervals() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];
        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(50)
            .burn_in(10)
            .random_state(42);
        let result = hbcca.fit(&x, &y, &groups).expect("fit should succeed");
        let intervals = result.correlation_credible_intervals(0.95);
        assert_eq!(intervals.shape(), &[1, 2]);
        assert!(intervals[[0, 0]] <= intervals[[0, 1]]);
    }
    #[test]
    fn test_hierarchical_bayesian_cca_error_cases() {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];
        let groups = vec![0, 0];
        let hbcca = HierarchicalBayesianCCA::new(1);
        let result = hbcca.fit(&x, &y, &groups);
        assert!(result.is_err());
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];
        let groups = vec![0, 0];
        let hbcca2 = HierarchicalBayesianCCA::new(1);
        let result2 = hbcca2.fit(&x, &y, &groups);
        assert!(result2.is_err());
    }
    #[test]
    fn test_hierarchical_bayesian_cca_hyperparameters() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];
        let groups = vec![0, 0, 1, 1];
        let hbcca = HierarchicalBayesianCCA::new(1)
            .n_samples(20)
            .burn_in(5)
            .population_precision(2.0)
            .group_precision(0.5)
            .random_state(42);
        let result = hbcca.fit(&x, &y, &groups).expect("fit should succeed");
        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0] >= -1.0 && correlations[0] <= 1.0);
    }
}
