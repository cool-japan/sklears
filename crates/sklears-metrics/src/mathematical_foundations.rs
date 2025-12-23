//! Mathematical Foundations and Derivations
//!
//! This module provides comprehensive mathematical derivations and formulations
//! for all metrics implemented in sklears-metrics. Each metric includes:
//! - Mathematical definition with LaTeX notation
//! - Derivation from first principles
//! - Computational complexity analysis
//! - Properties and theoretical guarantees
//!
//! # Organization
//!
//! The documentation is organized by metric category:
//! - Classification metrics
//! - Regression metrics
//! - Clustering metrics
//! - Ranking metrics
//! - Information-theoretic metrics

pub mod classification {
    //! # Classification Metrics - Mathematical Foundations
    //!
    //! ## Accuracy
    //!
    //! ### Definition
    //! Accuracy measures the proportion of correct predictions:
    //!
    //! ```latex
    //! \text{Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(y_i = \hat{y}_i)
    //! ```
    //!
    //! where:
    //! - $n$ is the number of samples
    //! - $y_i$ is the true label
    //! - $\hat{y}_i$ is the predicted label
    //! - $\mathbb{1}(\cdot)$ is the indicator function
    //!
    //! ### Properties
    //! - Range: $[0, 1]$
    //! - Higher is better
    //! - Symmetric with respect to classes
    //! - Can be misleading with imbalanced datasets
    //!
    //! ### Computational Complexity
    //! - Time: $O(n)$
    //! - Space: $O(1)$
    //!
    //! ## Precision
    //!
    //! ### Definition
    //! Precision (Positive Predictive Value) measures the proportion of true positives
    //! among predicted positives:
    //!
    //! ```latex
    //! \text{Precision} = \frac{TP}{TP + FP} = \frac{\sum_{i=1}^{n} \mathbb{1}(y_i = 1 \land \hat{y}_i = 1)}{\sum_{i=1}^{n} \mathbb{1}(\hat{y}_i = 1)}
    //! ```
    //!
    //! where:
    //! - $TP$ = True Positives
    //! - $FP$ = False Positives
    //!
    //! ### Derivation
    //! From the confusion matrix:
    //! ```latex
    //! \begin{bmatrix}
    //! TN & FP \\
    //! FN & TP
    //! \end{bmatrix}
    //! ```
    //!
    //! Precision answers: "Of all samples predicted as positive, how many are actually positive?"
    //!
    //! ### Properties
    //! - Range: $[0, 1]$
    //! - Higher is better
    //! - Undefined when $TP + FP = 0$ (no positive predictions)
    //! - Sensitive to class imbalance
    //!
    //! ### Computational Complexity
    //! - Time: $O(n)$
    //! - Space: $O(1)$
    //!
    //! ## Recall (Sensitivity, True Positive Rate)
    //!
    //! ### Definition
    //! ```latex
    //! \text{Recall} = \frac{TP}{TP + FN} = \frac{\sum_{i=1}^{n} \mathbb{1}(y_i = 1 \land \hat{y}_i = 1)}{\sum_{i=1}^{n} \mathbb{1}(y_i = 1)}
    //! ```
    //!
    //! ### Derivation
    //! Recall answers: "Of all actual positive samples, how many did we correctly identify?"
    //!
    //! ### Properties
    //! - Range: $[0, 1]$
    //! - Higher is better
    //! - Undefined when $TP + FN = 0$ (no positive samples)
    //! - Complements specificity
    //!
    //! ## F1 Score
    //!
    //! ### Definition
    //! The harmonic mean of precision and recall:
    //!
    //! ```latex
    //! F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
    //! ```
    //!
    //! ### Derivation
    //! The harmonic mean is chosen because:
    //! 1. It's more sensitive to low values than arithmetic mean
    //! 2. It balances precision and recall equally
    //!
    //! Proof that it's the harmonic mean:
    //! ```latex
    //! \text{HM}(P, R) = \frac{2}{\frac{1}{P} + \frac{1}{R}} = \frac{2PR}{P + R} = F_1
    //! ```
    //!
    //! ### Properties
    //! - Range: $[0, 1]$
    //! - Higher is better
    //! - Reaches maximum only when Precision = Recall = 1
    //! - More sensitive to low values than arithmetic mean
    //!
    //! ### Generalization: F-Beta Score
    //! ```latex
    //! F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
    //! ```
    //!
    //! where $\beta$ controls the trade-off:
    //! - $\beta < 1$: Emphasizes precision
    //! - $\beta = 1$: Balanced (F1 score)
    //! - $\beta > 1$: Emphasizes recall
    //!
    //! ## Matthews Correlation Coefficient (MCC)
    //!
    //! ### Definition
    //! ```latex
    //! \text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
    //! ```
    //!
    //! ### Derivation
    //! MCC is the Pearson correlation coefficient between predicted and true binary labels.
    //!
    //! For binary variables $X, Y \in \{0, 1\}^n$:
    //! ```latex
    //! \rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
    //! ```
    //!
    //! After expansion and simplification with the confusion matrix, we obtain MCC.
    //!
    //! ### Properties
    //! - Range: $[-1, 1]$
    //! - 1: Perfect prediction
    //! - 0: Random prediction
    //! - -1: Perfect disagreement
    //! - Symmetric measure
    //! - Handles class imbalance well
    //!
    //! ## Cohen's Kappa
    //!
    //! ### Definition
    //! ```latex
    //! \kappa = \frac{p_o - p_e}{1 - p_e}
    //! ```
    //!
    //! where:
    //! - $p_o$ = observed agreement (accuracy)
    //! - $p_e$ = expected agreement by chance
    //!
    //! ### Derivation
    //! For a confusion matrix $C$:
    //! ```latex
    //! p_o = \frac{1}{n} \sum_{i} C_{ii}
    //! ```
    //! ```latex
    //! p_e = \frac{1}{n^2} \sum_{i} \left(\sum_{j} C_{ij}\right) \left(\sum_{j} C_{ji}\right)
    //! ```
    //!
    //! Cohen's Kappa adjusts accuracy for chance agreement.
    //!
    //! ### Properties
    //! - Range: $[-1, 1]$ (though negative values are rare)
    //! - 1: Perfect agreement
    //! - 0: Agreement by chance
    //! - Accounts for class distribution
    //!
    //! ## Log Loss (Cross-Entropy Loss)
    //!
    //! ### Definition
    //! For binary classification:
    //! ```latex
    //! \text{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right]
    //! ```
    //!
    //! For multi-class:
    //! ```latex
    //! \text{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(p_{ic})
    //! ```
    //!
    //! where:
    //! - $p_i$ or $p_{ic}$ are predicted probabilities
    //! - $y_i$ or $y_{ic}$ are true labels (one-hot encoded)
    //!
    //! ### Derivation
    //! Log loss is the negative log-likelihood of the Bernoulli/Multinomial distribution.
    //!
    //! For maximum likelihood estimation:
    //! ```latex
    //! \hat{\theta} = \arg\max_\theta \prod_{i=1}^{n} P(y_i | x_i, \theta)
    //! ```
    //!
    //! Taking logarithm (for numerical stability):
    //! ```latex
    //! \hat{\theta} = \arg\max_\theta \sum_{i=1}^{n} \log P(y_i | x_i, \theta)
    //! ```
    //!
    //! Minimizing negative log-likelihood gives us log loss.
    //!
    //! ### Properties
    //! - Range: $[0, \infty)$
    //! - Lower is better
    //! - Heavily penalizes confident wrong predictions
    //! - Differentiable (useful for optimization)
    //! - Proper scoring rule
    //!
    //! ## Brier Score
    //!
    //! ### Definition
    //! ```latex
    //! \text{BS} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2
    //! ```
    //!
    //! ### Derivation
    //! Brier score is the mean squared error between predicted probabilities and true labels.
    //!
    //! It can be decomposed into:
    //! ```latex
    //! \text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}
    //! ```
    //!
    //! ### Properties
    //! - Range: $[0, 1]$
    //! - Lower is better
    //! - Proper scoring rule
    //! - More lenient than log loss for wrong confident predictions
}

pub mod regression {
    //! # Regression Metrics - Mathematical Foundations
    //!
    //! ## Mean Squared Error (MSE)
    //!
    //! ### Definition
    //! ```latex
    //! \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    //! ```
    //!
    //! ### Derivation
    //! MSE is the expected value of the squared error:
    //! ```latex
    //! \text{MSE} = \mathbb{E}[(Y - \hat{Y})^2]
    //! ```
    //!
    //! Under squared loss, MSE is minimized by the conditional mean:
    //! ```latex
    //! \hat{y} = \mathbb{E}[Y | X]
    //! ```
    //!
    //! ### Bias-Variance Decomposition
    //! ```latex
    //! \text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
    //! ```
    //!
    //! ### Properties
    //! - Range: $[0, \infty)$
    //! - Lower is better
    //! - Sensitive to outliers (quadratic penalty)
    //! - Differentiable
    //! - Units: squared units of target variable
    //!
    //! ### Computational Complexity
    //! - Time: $O(n)$
    //! - Space: $O(1)$
    //!
    //! ## Mean Absolute Error (MAE)
    //!
    //! ### Definition
    //! ```latex
    //! \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
    //! ```
    //!
    //! ### Derivation
    //! MAE is the expected absolute deviation:
    //! ```latex
    //! \text{MAE} = \mathbb{E}[|Y - \hat{Y}|]
    //! ```
    //!
    //! Under absolute loss, MAE is minimized by the conditional median:
    //! ```latex
    //! \hat{y} = \text{median}(Y | X)
    //! ```
    //!
    //! ### Properties
    //! - Range: $[0, \infty)$
    //! - Lower is better
    //! - More robust to outliers than MSE
    //! - Same units as target variable
    //! - Not differentiable at zero
    //!
    //! ## R² Score (Coefficient of Determination)
    //!
    //! ### Definition
    //! ```latex
    //! R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}
    //! ```
    //!
    //! where:
    //! - $SS_{res}$ = Residual sum of squares
    //! - $SS_{tot}$ = Total sum of squares
    //! - $\bar{y}$ = mean of true values
    //!
    //! ### Derivation
    //! R² represents the proportion of variance explained by the model:
    //!
    //! ```latex
    //! \text{Var}(Y) = \text{Var}(\hat{Y}) + \text{Var}(Y - \hat{Y})
    //! ```
    //!
    //! Therefore:
    //! ```latex
    //! R^2 = \frac{\text{Var}(\hat{Y})}{\text{Var}(Y)} = 1 - \frac{\text{Var}(Y - \hat{Y})}{\text{Var}(Y)}
    //! ```
    //!
    //! ### Properties
    //! - Range: $(-\infty, 1]$
    //! - 1: Perfect fit
    //! - 0: Model performs as well as predicting the mean
    //! - Negative: Model is worse than predicting the mean
    //! - Not appropriate for non-linear models without intercept
    //!
    //! ### Adjusted R²
    //! Accounts for number of predictors:
    //! ```latex
    //! R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
    //! ```
    //!
    //! where $p$ is the number of predictors.
    //!
    //! ## Huber Loss
    //!
    //! ### Definition
    //! ```latex
    //! L_\delta(y, \hat{y}) = \begin{cases}
    //! \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
    //! \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
    //! \end{cases}
    //! ```
    //!
    //! ### Derivation
    //! Huber loss is designed to be:
    //! - Quadratic for small errors (like MSE)
    //! - Linear for large errors (like MAE)
    //!
    //! This provides a balance between MSE's sensitivity and MAE's robustness.
    //!
    //! ### Properties
    //! - Continuous and differentiable everywhere
    //! - Robust to outliers
    //! - Parameter $\delta$ controls the transition point
    //!
    //! ## Mean Absolute Percentage Error (MAPE)
    //!
    //! ### Definition
    //! ```latex
    //! \text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
    //! ```
    //!
    //! ### Properties
    //! - Scale-independent
    //! - Interpretable as percentage
    //! - Undefined when $y_i = 0$
    //! - Asymmetric (penalizes over-predictions more than under-predictions)
    //!
    //! ### Symmetric MAPE (sMAPE)
    //! ```latex
    //! \text{sMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
    //! ```
}

pub mod information_theory {
    //! # Information-Theoretic Metrics
    //!
    //! ## Mutual Information
    //!
    //! ### Definition
    //! ```latex
    //! I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
    //! ```
    //!
    //! ### Derivation
    //! Mutual information measures the reduction in uncertainty about $X$ given $Y$:
    //! ```latex
    //! I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
    //! ```
    //!
    //! where $H$ is entropy:
    //! ```latex
    //! H(X) = -\sum_{x \in X} p(x) \log p(x)
    //! ```
    //!
    //! ### Properties
    //! - Range: $[0, \min(H(X), H(Y))]$
    //! - Symmetric: $I(X; Y) = I(Y; X)$
    //! - $I(X; Y) = 0$ iff $X$ and $Y$ are independent
    //! - Non-negative
    //!
    //! ## Kullback-Leibler Divergence
    //!
    //! ### Definition
    //! ```latex
    //! D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
    //! ```
    //!
    //! ### Derivation
    //! KL divergence measures how one probability distribution diverges from a reference distribution.
    //!
    //! It can be derived from the log-likelihood ratio:
    //! ```latex
    //! D_{KL}(P \| Q) = \mathbb{E}_P\left[\log \frac{P(X)}{Q(X)}\right]
    //! ```
    //!
    //! ### Properties
    //! - Range: $[0, \infty)$
    //! - Not symmetric: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
    //! - $D_{KL}(P \| Q) = 0$ iff $P = Q$ almost everywhere
    //! - Not a true metric (doesn't satisfy triangle inequality)
    //!
    //! ## Jensen-Shannon Divergence
    //!
    //! ### Definition
    //! ```latex
    //! \text{JSD}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)
    //! ```
    //!
    //! where $M = \frac{1}{2}(P + Q)$
    //!
    //! ### Properties
    //! - Range: $[0, \log 2]$ for base-2 logarithm
    //! - Symmetric
    //! - Bounded
    //! - Square root of JSD is a metric
}

pub mod ranking {
    //! # Ranking Metrics - Mathematical Foundations
    //!
    //! ## Area Under ROC Curve (AUC-ROC)
    //!
    //! ### Definition
    //! ```latex
    //! \text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x)) \, dx
    //! ```
    //!
    //! where:
    //! - TPR = True Positive Rate (Recall)
    //! - FPR = False Positive Rate
    //!
    //! ### Probabilistic Interpretation
    //! AUC equals the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance:
    //!
    //! ```latex
    //! \text{AUC} = P(\text{score}(x^+) > \text{score}(x^-))
    //! ```
    //!
    //! ### Mann-Whitney U Statistic
    //! AUC can be computed using:
    //! ```latex
    //! \text{AUC} = \frac{U}{n^+ n^-}
    //! ```
    //!
    //! where $U$ is the Mann-Whitney U statistic.
    //!
    //! ### Properties
    //! - Range: $[0, 1]$
    //! - 0.5: Random classifier
    //! - 1.0: Perfect classifier
    //! - Threshold-independent
    //! - Invariant to class imbalance
    //!
    //! ## Normalized Discounted Cumulative Gain (NDCG)
    //!
    //! ### Definition
    //! ```latex
    //! \text{DCG}_k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
    //! ```
    //!
    //! ```latex
    //! \text{NDCG}_k = \frac{\text{DCG}_k}{\text{IDCG}_k}
    //! ```
    //!
    //! where IDCG is the DCG of the ideal ranking.
    //!
    //! ### Derivation
    //! The logarithmic discount factor $\frac{1}{\log_2(i + 1)}$ reflects the decreasing likelihood that users examine items at lower positions.
    //!
    //! The gain $2^{rel_i} - 1$ emphasizes highly relevant items exponentially.
    //!
    //! ### Properties
    //! - Range: $[0, 1]$
    //! - Position-aware
    //! - Handles graded relevance
    //! - Normalized across queries with different numbers of relevant items
    //!
    //! ## Mean Average Precision (MAP)
    //!
    //! ### Definition
    //! ```latex
    //! \text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{m_q} \sum_{k=1}^{n} P(k) \cdot \text{rel}(k)
    //! ```
    //!
    //! where:
    //! - $Q$ is the set of queries
    //! - $m_q$ is the number of relevant documents for query $q$
    //! - $P(k)$ is precision at position $k$
    //! - $\text{rel}(k) = 1$ if item at position $k$ is relevant, 0 otherwise
    //!
    //! ### Properties
    //! - Range: $[0, 1]$
    //! - Position-aware
    //! - Binary relevance
    //! - Emphasizes ranking relevant items higher
}

pub mod clustering {
    //! # Clustering Metrics - Mathematical Foundations
    //!
    //! ## Silhouette Score
    //!
    //! ### Definition
    //! For each sample $i$:
    //! ```latex
    //! s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
    //! ```
    //!
    //! where:
    //! - $a(i)$ = mean distance to other points in the same cluster
    //! - $b(i)$ = mean distance to points in the nearest cluster
    //!
    //! Overall silhouette score:
    //! ```latex
    //! S = \frac{1}{n} \sum_{i=1}^{n} s(i)
    //! ```
    //!
    //! ### Properties
    //! - Range: $[-1, 1]$
    //! - 1: Perfect clustering (samples far from neighboring clusters)
    //! - 0: Samples on the boundary between clusters
    //! - -1: Samples assigned to wrong cluster
    //!
    //! ## Davies-Bouldin Index
    //!
    //! ### Definition
    //! ```latex
    //! \text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{s_i + s_j}{d_{ij}}
    //! ```
    //!
    //! where:
    //! - $s_i$ = average distance of samples in cluster $i$ to centroid
    //! - $d_{ij}$ = distance between centroids of clusters $i$ and $j$
    //!
    //! ### Properties
    //! - Range: $[0, \infty)$
    //! - Lower is better
    //! - Ratio of within-cluster to between-cluster distances
    //!
    //! ## Calinski-Harabasz Index (Variance Ratio Criterion)
    //!
    //! ### Definition
    //! ```latex
    //! \text{CH} = \frac{\text{tr}(B_k)}{\text{tr}(W_k)} \cdot \frac{n - k}{k - 1}
    //! ```
    //!
    //! where:
    //! - $B_k$ = between-cluster dispersion matrix
    //! - $W_k$ = within-cluster dispersion matrix
    //! - $n$ = number of samples
    //! - $k$ = number of clusters
    //!
    //! ### Derivation
    //! This is the ratio of between-cluster variance to within-cluster variance, adjusted for degrees of freedom.
    //!
    //! ### Properties
    //! - Range: $[0, \infty)$
    //! - Higher is better
    //! - Fast to compute
    //! - Based on F-statistic from ANOVA
    //!
    //! ## Adjusted Rand Index (ARI)
    //!
    //! ### Definition
    //! ```latex
    //! \text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}
    //! ```
    //!
    //! where RI is the Rand Index:
    //! ```latex
    //! \text{RI} = \frac{TP + TN}{TP + FP + FN + TN}
    //! ```
    //!
    //! ### Derivation
    //! ARI adjusts for chance using the hypergeometric distribution:
    //! ```latex
    //! \mathbb{E}[\text{RI}] = \frac{\binom{n}{2}^{-1} \sum_{ij} \binom{n_{ij}}{2} \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}}{\binom{n}{2}}
    //! ```
    //!
    //! ### Properties
    //! - Range: $[-1, 1]$ (though negative values are rare)
    //! - 1: Perfect match
    //! - 0: Random labeling
    //! - Accounts for chance agreement
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_mathematical_documentation_exists() {
        // This test ensures the mathematical documentation compiles
        // In practice, these would be rendered in documentation
        assert!(true);
    }
}
