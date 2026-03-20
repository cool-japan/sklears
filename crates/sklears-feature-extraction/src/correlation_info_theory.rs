use crate::SklResult;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::prelude::{Float, SklearsError};

// =============================================================================
// Statistical Correlation and Information Theory Features
// =============================================================================

/// Correlation feature extractor
///
/// Extracts various correlation measures between features or time series.
pub struct CorrelationFeatureExtractor {
    correlation_types: Vec<CorrelationType>,
    lag_range: Option<(isize, isize)>,
    include_auto_correlation: bool,
    include_cross_correlation: bool,
    include_partial_correlation: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum CorrelationType {
    /// Pearson
    Pearson,
    /// Spearman
    Spearman,
    /// Kendall
    Kendall,
    /// Distance
    Distance,
    /// MaximalInformationCoefficient
    MaximalInformationCoefficient,
}

impl CorrelationFeatureExtractor {
    /// Create a new correlation feature extractor
    pub fn new() -> Self {
        Self {
            correlation_types: vec![CorrelationType::Pearson, CorrelationType::Spearman],
            lag_range: None,
            include_auto_correlation: true,
            include_cross_correlation: false,
            include_partial_correlation: false,
        }
    }

    /// Set correlation types to compute
    pub fn correlation_types(mut self, types: Vec<CorrelationType>) -> Self {
        self.correlation_types = types;
        self
    }

    /// Set lag range for time series correlation
    pub fn lag_range(mut self, range: (isize, isize)) -> Self {
        self.lag_range = Some(range);
        self
    }

    /// Include auto-correlation features
    pub fn include_auto_correlation(mut self, include: bool) -> Self {
        self.include_auto_correlation = include;
        self
    }

    /// Include cross-correlation features (for multivariate data)
    pub fn include_cross_correlation(mut self, include: bool) -> Self {
        self.include_cross_correlation = include;
        self
    }

    /// Include partial correlation features
    pub fn include_partial_correlation(mut self, include: bool) -> Self {
        self.include_partial_correlation = include;
        self
    }

    /// Extract correlation features from data
    pub fn extract_features(&self, data: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        if data.nrows() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for correlation analysis".to_string(),
            ));
        }

        let mut features = Vec::new();

        // Auto-correlation features
        if self.include_auto_correlation {
            let autocorr_features = self.extract_autocorrelation_features(data)?;
            features.extend(autocorr_features);
        }

        // Cross-correlation features
        if self.include_cross_correlation && data.ncols() > 1 {
            let crosscorr_features = self.extract_crosscorrelation_features(data)?;
            features.extend(crosscorr_features);
        }

        // Partial correlation features
        if self.include_partial_correlation && data.ncols() > 2 {
            let partial_features = self.extract_partial_correlation_features(data)?;
            features.extend(partial_features);
        }

        Ok(Array1::from_vec(features))
    }

    /// Extract auto-correlation features for each column
    fn extract_autocorrelation_features(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Float>> {
        let mut features = Vec::new();

        for col_idx in 0..data.ncols() {
            let series = data.column(col_idx);

            for &corr_type in &self.correlation_types {
                match corr_type {
                    CorrelationType::Pearson => {
                        let autocorr = self.calculate_autocorrelation(&series, 1)?;
                        features.push(autocorr);
                    }
                    _ => {
                        // For non-Pearson, compute at lag 1
                        let lag1_series = self.lag_series(&series, 1)?;
                        let correlation = self.calculate_correlation(
                            &series.slice(s![1..]),
                            &lag1_series.view(),
                            corr_type,
                        )?;
                        features.push(correlation);
                    }
                }
            }

            // If lag range is specified, compute at multiple lags
            if let Some((min_lag, max_lag)) = self.lag_range {
                for lag in min_lag..=max_lag {
                    if lag != 0 {
                        let autocorr = self.calculate_autocorrelation(&series, lag)?;
                        features.push(autocorr);
                    }
                }
            }
        }

        Ok(features)
    }

    /// Extract cross-correlation features between columns
    fn extract_crosscorrelation_features(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Float>> {
        let mut features = Vec::new();
        let ncols = data.ncols();

        for i in 0..ncols {
            for j in (i + 1)..ncols {
                let series1 = data.column(i);
                let series2 = data.column(j);

                for &corr_type in &self.correlation_types {
                    let correlation = self.calculate_correlation(&series1, &series2, corr_type)?;
                    features.push(correlation);
                }

                // Lagged cross-correlation if specified
                if let Some((min_lag, max_lag)) = self.lag_range {
                    for lag in min_lag..=max_lag {
                        if lag != 0 {
                            let crosscorr =
                                self.calculate_crosscorrelation(&series1, &series2, lag)?;
                            features.push(crosscorr);
                        }
                    }
                }
            }
        }

        Ok(features)
    }

    /// Extract partial correlation features
    fn extract_partial_correlation_features(
        &self,
        data: &ArrayView2<Float>,
    ) -> SklResult<Vec<Float>> {
        let mut features = Vec::new();
        let ncols = data.ncols();

        // Compute partial correlations between all pairs controlling for all other variables
        for i in 0..ncols {
            for j in (i + 1)..ncols {
                let partial_corr = self.calculate_partial_correlation(data, i, j)?;
                features.push(partial_corr);
            }
        }

        Ok(features)
    }

    /// Calculate auto-correlation at specific lag
    fn calculate_autocorrelation(
        &self,
        series: &ArrayView1<Float>,
        lag: isize,
    ) -> SklResult<Float> {
        let n = series.len() as isize;

        if lag.abs() >= n {
            return Ok(0.0);
        }

        let effective_len = (n - lag.abs()) as usize;
        if effective_len < 2 {
            return Ok(0.0);
        }

        let (start1, start2) = if lag > 0 {
            (0, lag as usize)
        } else {
            ((-lag) as usize, 0)
        };

        let series1 = series.slice(s![start1..start1 + effective_len]);
        let series2 = series.slice(s![start2..start2 + effective_len]);

        self.calculate_correlation(&series1, &series2, CorrelationType::Pearson)
    }

    /// Calculate cross-correlation between two series at specific lag
    fn calculate_crosscorrelation(
        &self,
        series1: &ArrayView1<Float>,
        series2: &ArrayView1<Float>,
        lag: isize,
    ) -> SklResult<Float> {
        let n = series1.len().min(series2.len()) as isize;

        if lag.abs() >= n {
            return Ok(0.0);
        }

        let effective_len = (n - lag.abs()) as usize;
        if effective_len < 2 {
            return Ok(0.0);
        }

        let (s1_slice, s2_slice) = if lag > 0 {
            (
                series1.slice(s![0..effective_len]),
                series2.slice(s![lag as usize..lag as usize + effective_len]),
            )
        } else {
            (
                series1.slice(s![(-lag) as usize..(-lag) as usize + effective_len]),
                series2.slice(s![0..effective_len]),
            )
        };

        self.calculate_correlation(&s1_slice, &s2_slice, CorrelationType::Pearson)
    }

    /// Calculate correlation between two series
    fn calculate_correlation(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        corr_type: CorrelationType,
    ) -> SklResult<Float> {
        if x.len() != y.len() || x.len() < 2 {
            return Ok(0.0);
        }

        match corr_type {
            CorrelationType::Pearson => self.pearson_correlation(x, y),
            CorrelationType::Spearman => self.spearman_correlation(x, y),
            CorrelationType::Kendall => self.kendall_correlation(x, y),
            CorrelationType::Distance => self.distance_correlation(x, y),
            CorrelationType::MaximalInformationCoefficient => self.mic_correlation(x, y),
        }
    }

    /// Calculate Pearson correlation
    fn pearson_correlation(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut x_sq_sum = 0.0;
        let mut y_sq_sum = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let x_diff = xi - x_mean;
            let y_diff = yi - y_mean;

            numerator += x_diff * y_diff;
            x_sq_sum += x_diff * x_diff;
            y_sq_sum += y_diff * y_diff;
        }

        let denominator = (x_sq_sum * y_sq_sum).sqrt();

        if denominator < 1e-10 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate Spearman rank correlation
    fn spearman_correlation(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        let x_ranks = self.compute_ranks(x);
        let y_ranks = self.compute_ranks(y);

        let x_ranks_array = Array1::from_vec(x_ranks);
        let y_ranks_array = Array1::from_vec(y_ranks);

        self.pearson_correlation(&x_ranks_array.view(), &y_ranks_array.view())
    }

    /// Calculate Kendall's tau correlation
    fn kendall_correlation(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        let n = x.len();
        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let x_diff = x[i] - x[j];
                let y_diff = y[i] - y[j];

                if (x_diff > 0.0 && y_diff > 0.0) || (x_diff < 0.0 && y_diff < 0.0) {
                    concordant += 1;
                } else if (x_diff > 0.0 && y_diff < 0.0) || (x_diff < 0.0 && y_diff > 0.0) {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs == 0 {
            Ok(0.0)
        } else {
            Ok((concordant - discordant) as Float / total_pairs as Float)
        }
    }

    /// Calculate distance correlation (simplified version)
    fn distance_correlation(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        let n = x.len();

        // Distance matrices
        let mut dx = Array2::zeros((n, n));
        let mut dy = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                dx[[i, j]] = (x[i] - x[j]).abs();
                dy[[i, j]] = (y[i] - y[j]).abs();
            }
        }

        // Double-centered distance matrices
        let dx_centered = self.double_center_matrix(&dx);
        let dy_centered = self.double_center_matrix(&dy);

        // Calculate distance correlation
        let dcov_xy = self.calculate_dcov(&dx_centered, &dy_centered);
        let dcov_xx = self.calculate_dcov(&dx_centered, &dx_centered);
        let dcov_yy = self.calculate_dcov(&dy_centered, &dy_centered);

        let denominator = (dcov_xx * dcov_yy).sqrt();
        if denominator < 1e-10 {
            Ok(0.0)
        } else {
            Ok(dcov_xy / denominator)
        }
    }

    /// Calculate Maximal Information Coefficient (simplified version)
    fn mic_correlation(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> SklResult<Float> {
        // Simplified MIC implementation using binning
        let n_bins = (x.len() as Float).log2().ceil() as usize + 1;
        let n_bins = n_bins.min(10).max(2); // Reasonable bounds

        let x_bins = self.discretize_data(x, n_bins);
        let y_bins = self.discretize_data(y, n_bins);

        // Calculate mutual information
        let mi = self.calculate_mutual_information(&x_bins, &y_bins, n_bins);

        // Normalize by maximum possible MI
        let max_mi = (n_bins as Float).log2().min((x.len() as Float).log2());

        if max_mi > 1e-10 {
            Ok(mi / max_mi)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate partial correlation
    fn calculate_partial_correlation(
        &self,
        data: &ArrayView2<Float>,
        i: usize,
        j: usize,
    ) -> SklResult<Float> {
        let ncols = data.ncols();
        if i >= ncols || j >= ncols || i == j {
            return Ok(0.0);
        }

        // Simple partial correlation: correlation after removing linear effects of other variables
        let mut residuals_i = data.column(i).to_owned();
        let mut residuals_j = data.column(j).to_owned();

        // Remove linear effects of all other variables
        for k in 0..ncols {
            if k != i && k != j {
                let control_var = data.column(k);
                residuals_i = self.remove_linear_effect(&residuals_i.view(), &control_var)?;
                residuals_j = self.remove_linear_effect(&residuals_j.view(), &control_var)?;
            }
        }

        self.pearson_correlation(&residuals_i.view(), &residuals_j.view())
    }

    /// Remove linear effect of one variable from another
    fn remove_linear_effect(
        &self,
        y: &ArrayView1<Float>,
        x: &ArrayView1<Float>,
    ) -> SklResult<Array1<Float>> {
        // Simple linear regression: y = a + b*x, return residuals
        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let x_diff = xi - x_mean;
            numerator += x_diff * (yi - y_mean);
            denominator += x_diff * x_diff;
        }

        let slope = if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        let mut residuals = Array1::zeros(y.len());
        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            let predicted = intercept + slope * xi;
            residuals[i] = yi - predicted;
        }

        Ok(residuals)
    }

    /// Compute ranks for Spearman correlation
    fn compute_ranks(&self, data: &ArrayView1<Float>) -> Vec<Float> {
        let mut indexed_data: Vec<(Float, usize)> =
            data.iter().enumerate().map(|(i, &val)| (val, i)).collect();
        indexed_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; data.len()];
        for (rank, &(_, orig_idx)) in indexed_data.iter().enumerate() {
            ranks[orig_idx] = (rank + 1) as Float;
        }

        ranks
    }

    /// Double center a matrix for distance correlation
    fn double_center_matrix(&self, matrix: &Array2<Float>) -> Array2<Float> {
        let n = matrix.nrows();
        let mut centered = matrix.clone();

        // Row means
        let row_means: Vec<Float> = (0..n)
            .map(|i| matrix.row(i).mean().unwrap_or(0.0))
            .collect();

        // Column means
        let col_means: Vec<Float> = (0..n)
            .map(|j| matrix.column(j).mean().unwrap_or(0.0))
            .collect();

        // Grand mean
        let grand_mean = matrix.mean().unwrap_or(0.0);

        // Double center
        for i in 0..n {
            for j in 0..n {
                centered[[i, j]] = matrix[[i, j]] - row_means[i] - col_means[j] + grand_mean;
            }
        }

        centered
    }

    /// Calculate distance covariance
    fn calculate_dcov(&self, dx: &Array2<Float>, dy: &Array2<Float>) -> Float {
        let n = dx.nrows();
        let mut dcov = 0.0;

        for i in 0..n {
            for j in 0..n {
                dcov += dx[[i, j]] * dy[[i, j]];
            }
        }

        dcov / (n * n) as Float
    }

    /// Discretize data into bins
    fn discretize_data(&self, data: &ArrayView1<Float>, n_bins: usize) -> Vec<usize> {
        let min_val = data.iter().cloned().fold(Float::INFINITY, Float::min);
        let max_val = data.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

        if (max_val - min_val).abs() < 1e-10 {
            return vec![0; data.len()];
        }

        let bin_width = (max_val - min_val) / n_bins as Float;

        data.iter()
            .map(|&val| {
                let bin = ((val - min_val) / bin_width).floor() as usize;
                bin.min(n_bins - 1)
            })
            .collect()
    }

    /// Calculate mutual information between binned data
    fn calculate_mutual_information(
        &self,
        x_bins: &[usize],
        y_bins: &[usize],
        n_bins: usize,
    ) -> Float {
        let n = x_bins.len();

        // Joint and marginal distributions
        let mut joint_counts = vec![vec![0; n_bins]; n_bins];
        let mut x_counts = vec![0; n_bins];
        let mut y_counts = vec![0; n_bins];

        for (&x_bin, &y_bin) in x_bins.iter().zip(y_bins.iter()) {
            joint_counts[x_bin][y_bin] += 1;
            x_counts[x_bin] += 1;
            y_counts[y_bin] += 1;
        }

        // Calculate mutual information
        let mut mi = 0.0;
        for i in 0..n_bins {
            for j in 0..n_bins {
                let joint_prob = joint_counts[i][j] as Float / n as Float;
                let x_prob = x_counts[i] as Float / n as Float;
                let y_prob = y_counts[j] as Float / n as Float;

                if joint_prob > 1e-10 && x_prob > 1e-10 && y_prob > 1e-10 {
                    mi += joint_prob * (joint_prob / (x_prob * y_prob)).log2();
                }
            }
        }

        mi
    }

    /// Create lagged version of series
    fn lag_series(&self, series: &ArrayView1<Float>, lag: isize) -> SklResult<Array1<Float>> {
        let n = series.len();
        let lag_abs = lag.unsigned_abs();

        if lag_abs >= n {
            return Err(SklearsError::InvalidInput(
                "Lag too large for series length".to_string(),
            ));
        }

        let result_len = n - lag_abs;
        let mut lagged = Array1::zeros(result_len);

        if lag > 0 {
            // Positive lag: take from beginning of series
            lagged.assign(&series.slice(s![0..result_len]));
        } else {
            // Negative lag: take from end of series
            lagged.assign(&series.slice(s![lag_abs..]));
        }

        Ok(lagged)
    }
}

impl Default for CorrelationFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Mutual information feature extractor
///
/// Extracts mutual information-based features for dependency analysis.
pub struct MutualInformationExtractor {
    estimator: MIEstimator,
    n_bins: usize,
    include_conditional_mi: bool,
    include_normalized_mi: bool,
    include_transfer_entropy: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum MIEstimator {
    /// Histogram
    Histogram,
    /// KSG
    KSG, // Kraskov-Stögbauer-Grassberger
    /// Gaussian
    Gaussian,
}

impl MutualInformationExtractor {
    /// Create a new mutual information extractor
    pub fn new() -> Self {
        Self {
            estimator: MIEstimator::Histogram,
            n_bins: 10,
            include_conditional_mi: false,
            include_normalized_mi: true,
            include_transfer_entropy: false,
        }
    }

    /// Set MI estimator type
    pub fn estimator(mut self, estimator: MIEstimator) -> Self {
        self.estimator = estimator;
        self
    }

    /// Set number of bins for histogram estimator
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Include conditional mutual information
    pub fn include_conditional_mi(mut self, include: bool) -> Self {
        self.include_conditional_mi = include;
        self
    }

    /// Include normalized mutual information
    pub fn include_normalized_mi(mut self, include: bool) -> Self {
        self.include_normalized_mi = include;
        self
    }

    /// Include transfer entropy (for time series)
    pub fn include_transfer_entropy(mut self, include: bool) -> Self {
        self.include_transfer_entropy = include;
        self
    }

    /// Extract mutual information features
    pub fn extract_features(&self, data: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        if data.nrows() < 2 || data.ncols() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples and 2 features for MI analysis".to_string(),
            ));
        }

        let mut features = Vec::new();
        let ncols = data.ncols();

        // Pairwise mutual information
        for i in 0..ncols {
            for j in (i + 1)..ncols {
                let x = data.column(i);
                let y = data.column(j);

                let mi = self.calculate_mutual_information(&x, &y)?;
                features.push(mi);

                if self.include_normalized_mi {
                    let nmi = self.calculate_normalized_mi(&x, &y, mi)?;
                    features.push(nmi);
                }
            }
        }

        // Conditional mutual information
        if self.include_conditional_mi && ncols > 2 {
            let conditional_features = self.extract_conditional_mi_features(data)?;
            features.extend(conditional_features);
        }

        // Transfer entropy (for time series data)
        if self.include_transfer_entropy {
            let te_features = self.extract_transfer_entropy_features(data)?;
            features.extend(te_features);
        }

        Ok(Array1::from_vec(features))
    }

    /// Calculate mutual information between two variables
    fn calculate_mutual_information(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        match self.estimator {
            MIEstimator::Histogram => self.mi_histogram(x, y),
            MIEstimator::KSG => self.mi_ksg(x, y),
            MIEstimator::Gaussian => self.mi_gaussian(x, y),
        }
    }

    /// Histogram-based MI estimation
    fn mi_histogram(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> SklResult<Float> {
        let n = x.len();

        // Discretize data
        let x_bins = self.discretize_variable(x, self.n_bins);
        let y_bins = self.discretize_variable(y, self.n_bins);

        // Count joint and marginal frequencies
        let mut joint_counts = vec![vec![0; self.n_bins]; self.n_bins];
        let mut x_counts = vec![0; self.n_bins];
        let mut y_counts = vec![0; self.n_bins];

        for (&x_bin, &y_bin) in x_bins.iter().zip(y_bins.iter()) {
            joint_counts[x_bin][y_bin] += 1;
            x_counts[x_bin] += 1;
            y_counts[y_bin] += 1;
        }

        // Calculate MI
        let mut mi = 0.0;
        for i in 0..self.n_bins {
            for j in 0..self.n_bins {
                let joint_prob = joint_counts[i][j] as Float / n as Float;
                let x_prob = x_counts[i] as Float / n as Float;
                let y_prob = y_counts[j] as Float / n as Float;

                if joint_prob > 1e-10 && x_prob > 1e-10 && y_prob > 1e-10 {
                    mi += joint_prob * (joint_prob / (x_prob * y_prob)).log2();
                }
            }
        }

        Ok(mi)
    }

    /// KSG MI estimator (simplified version)
    fn mi_ksg(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> SklResult<Float> {
        // Simplified KSG implementation using k=3 nearest neighbors
        let k = 3;
        let n = x.len();

        if n <= k {
            return Ok(0.0);
        }

        let mut mi_sum = 0.0;

        for i in 0..n {
            // Find k-th nearest neighbor distance in joint space
            let mut distances: Vec<Float> = Vec::new();

            for j in 0..n {
                if i != j {
                    let dx = (x[i] - x[j]).abs();
                    let dy = (y[i] - y[j]).abs();
                    let dist = dx.max(dy); // Chebyshev distance
                    distances.push(dist);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if distances.len() >= k {
                let epsilon = distances[k - 1];

                // Count neighbors within epsilon in marginal spaces
                let nx = distances.iter().filter(|&&d| d <= epsilon).count();
                let ny = nx; // Same for Chebyshev distance

                if nx > 0 && ny > 0 {
                    mi_sum += (k as Float).ln() - (nx as Float).ln() - (ny as Float).ln()
                        + (n - 1) as Float;
                }
            }
        }

        Ok(mi_sum / n as Float)
    }

    /// Gaussian MI estimator
    fn mi_gaussian(&self, x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> SklResult<Float> {
        // For Gaussian variables: MI = -0.5 * ln(1 - r^2)
        let correlation = self.pearson_correlation(x, y)?;
        let r_squared = correlation * correlation;

        if r_squared >= 1.0 - 1e-10 {
            Ok(Float::INFINITY)
        } else {
            Ok(-0.5 * (1.0 - r_squared).ln())
        }
    }

    /// Calculate normalized mutual information
    fn calculate_normalized_mi(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
        mi: Float,
    ) -> SklResult<Float> {
        let h_x = self.calculate_entropy(x)?;
        let h_y = self.calculate_entropy(y)?;

        let joint_entropy = h_x + h_y - mi;

        if joint_entropy > 1e-10 {
            Ok(mi / joint_entropy)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate entropy of a variable
    fn calculate_entropy(&self, x: &ArrayView1<Float>) -> SklResult<Float> {
        let x_bins = self.discretize_variable(x, self.n_bins);
        let n = x.len();

        let mut counts = vec![0; self.n_bins];
        for &bin in &x_bins {
            counts[bin] += 1;
        }

        let mut entropy = 0.0;
        for &count in &counts {
            if count > 0 {
                let prob = count as Float / n as Float;
                entropy -= prob * prob.log2();
            }
        }

        Ok(entropy)
    }

    /// Extract conditional MI features
    fn extract_conditional_mi_features(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Float>> {
        let mut features = Vec::new();
        let ncols = data.ncols();

        // I(X;Y|Z) for all combinations
        for i in 0..ncols {
            for j in (i + 1)..ncols {
                for k in 0..ncols {
                    if k != i && k != j {
                        let cmi = self.calculate_conditional_mi(data, i, j, k)?;
                        features.push(cmi);
                    }
                }
            }
        }

        Ok(features)
    }

    /// Calculate conditional mutual information I(X;Y|Z)
    fn calculate_conditional_mi(
        &self,
        data: &ArrayView2<Float>,
        x_idx: usize,
        y_idx: usize,
        z_idx: usize,
    ) -> SklResult<Float> {
        let x = data.column(x_idx);
        let y = data.column(y_idx);
        let z = data.column(z_idx);

        // I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
        // For simplicity, use: I(X;Y|Z) = I(X,Z;Y) - I(Z;Y)

        // This is a simplified approximation
        let mi_xz_y = self.calculate_joint_mi(&x, &z, &y)?;
        let mi_z_y = self.calculate_mutual_information(&z, &y)?;

        Ok(mi_xz_y - mi_z_y)
    }

    /// Calculate joint MI (simplified approximation)
    fn calculate_joint_mi(
        &self,
        x1: &ArrayView1<Float>,
        x2: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        // Simplified: treat (x1,x2) as a joint variable by concatenating ranks
        let x1_ranks = self.compute_ranks(x1);
        let x2_ranks = self.compute_ranks(x2);

        // Create joint variable by weighted combination
        let joint_var: Array1<Float> = x1_ranks
            .iter()
            .zip(x2_ranks.iter())
            .map(|(&r1, &r2)| r1 + 0.5 * r2) // Simple combination
            .collect();

        self.calculate_mutual_information(&joint_var.view(), y)
    }

    /// Extract transfer entropy features
    fn extract_transfer_entropy_features(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Float>> {
        let mut features = Vec::new();
        let ncols = data.ncols();

        // Transfer entropy from each series to every other series
        for i in 0..ncols {
            for j in 0..ncols {
                if i != j {
                    let te = self.calculate_transfer_entropy(data, i, j)?;
                    features.push(te);
                }
            }
        }

        Ok(features)
    }

    /// Calculate transfer entropy from X to Y
    fn calculate_transfer_entropy(
        &self,
        data: &ArrayView2<Float>,
        x_idx: usize,
        y_idx: usize,
    ) -> SklResult<Float> {
        let lag = 1; // Use lag of 1
        let nrows = data.nrows();

        if nrows <= lag {
            return Ok(0.0);
        }

        let effective_len = nrows - lag;

        // Y(t+1), Y(t), X(t)
        let y_column = data.column(y_idx);
        let x_column = data.column(x_idx);
        let y_future = y_column.slice(s![lag..]);
        let y_present = y_column.slice(s![0..effective_len]);
        let x_present = x_column.slice(s![0..effective_len]);

        // TE(X->Y) = I(Y(t+1); X(t) | Y(t))
        // Simplified as: I(Y(t+1), Y(t); X(t)) - I(Y(t); X(t))

        let mi_joint = self.calculate_joint_mi(&y_future, &y_present, &x_present)?;
        let mi_marginal = self.calculate_mutual_information(&y_present, &x_present)?;

        Ok(mi_joint - mi_marginal)
    }

    /// Discretize a variable into bins
    fn discretize_variable(&self, data: &ArrayView1<Float>, n_bins: usize) -> Vec<usize> {
        let min_val = data.iter().cloned().fold(Float::INFINITY, Float::min);
        let max_val = data.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

        if (max_val - min_val).abs() < 1e-10 {
            return vec![0; data.len()];
        }

        let bin_width = (max_val - min_val) / n_bins as Float;

        data.iter()
            .map(|&val| {
                let bin = ((val - min_val) / bin_width).floor() as usize;
                bin.min(n_bins - 1)
            })
            .collect()
    }

    /// Compute ranks for rank-based methods
    fn compute_ranks(&self, data: &ArrayView1<Float>) -> Vec<Float> {
        let mut indexed_data: Vec<(Float, usize)> =
            data.iter().enumerate().map(|(i, &val)| (val, i)).collect();
        indexed_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; data.len()];
        for (rank, &(_, orig_idx)) in indexed_data.iter().enumerate() {
            ranks[orig_idx] = (rank + 1) as Float;
        }

        ranks
    }

    /// Calculate Pearson correlation
    fn pearson_correlation(
        &self,
        x: &ArrayView1<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut x_sq_sum = 0.0;
        let mut y_sq_sum = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let x_diff = xi - x_mean;
            let y_diff = yi - y_mean;

            numerator += x_diff * y_diff;
            x_sq_sum += x_diff * x_diff;
            y_sq_sum += y_diff * y_diff;
        }

        let denominator = (x_sq_sum * y_sq_sum).sqrt();

        if denominator < 1e-10 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

impl Default for MutualInformationExtractor {
    fn default() -> Self {
        Self::new()
    }
}
