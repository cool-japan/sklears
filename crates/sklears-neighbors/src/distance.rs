//! Distance metrics for neighbor-based algorithms

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float as FloatTrait, Zero};
use sklears_core::types::Float;
use sklears_utils::metrics;

use std::hash::{Hash, Hasher};

/// Distance metric enum
#[derive(Default)]
pub enum Distance {
    /// Euclidean distance (L2 norm)
    #[default]
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Chebyshev distance (L-infinity norm)
    Chebyshev,
    /// Minkowski distance with parameter p
    Minkowski(Float),
    /// Cosine distance
    Cosine,
    /// Hamming distance (for binary data)
    Hamming,
    /// Mahalanobis distance with inverse covariance matrix
    Mahalanobis(Array2<Float>),
    /// Custom distance function
    Custom(Box<dyn Fn(&ArrayView1<Float>, &ArrayView1<Float>) -> Float + Send + Sync>),
    /// RBF (Gaussian) kernel distance
    RbfKernel(Float), // gamma parameter
    /// Polynomial kernel distance
    PolynomialKernel {
        degree: Float,
        gamma: Float,
        coef0: Float,
    },
    /// Sigmoid kernel distance
    SigmoidKernel { gamma: Float, coef0: Float },
    /// Laplacian kernel distance
    LaplacianKernel(Float), // gamma parameter
}

impl PartialEq for Distance {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Distance::Euclidean, Distance::Euclidean) => true,
            (Distance::Manhattan, Distance::Manhattan) => true,
            (Distance::Chebyshev, Distance::Chebyshev) => true,
            (Distance::Minkowski(p1), Distance::Minkowski(p2)) => p1 == p2,
            (Distance::Cosine, Distance::Cosine) => true,
            (Distance::Hamming, Distance::Hamming) => true,
            (Distance::Mahalanobis(m1), Distance::Mahalanobis(m2)) => m1 == m2,
            (Distance::Custom(_), Distance::Custom(_)) => false, // Cannot compare function pointers
            (Distance::RbfKernel(g1), Distance::RbfKernel(g2)) => g1 == g2,
            (
                Distance::PolynomialKernel {
                    degree: d1,
                    gamma: g1,
                    coef0: c1,
                },
                Distance::PolynomialKernel {
                    degree: d2,
                    gamma: g2,
                    coef0: c2,
                },
            ) => d1 == d2 && g1 == g2 && c1 == c2,
            (
                Distance::SigmoidKernel {
                    gamma: g1,
                    coef0: c1,
                },
                Distance::SigmoidKernel {
                    gamma: g2,
                    coef0: c2,
                },
            ) => g1 == g2 && c1 == c2,
            (Distance::LaplacianKernel(g1), Distance::LaplacianKernel(g2)) => g1 == g2,
            _ => false,
        }
    }
}

impl Eq for Distance {}

impl Hash for Distance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Distance::Euclidean => {}
            Distance::Manhattan => {}
            Distance::Chebyshev => {}
            Distance::Minkowski(p) => {
                // Hash f64 as bits to handle NaN consistently
                p.to_bits().hash(state);
            }
            Distance::Cosine => {}
            Distance::Hamming => {}
            Distance::Mahalanobis(m) => {
                // Hash array shape and elements
                m.shape().hash(state);
                for &element in m.iter() {
                    element.to_bits().hash(state);
                }
            }
            Distance::Custom(_) => {
                // For function pointers, we can't hash the function itself,
                // so we just hash a constant that represents the custom type
                42u64.hash(state);
            }
            Distance::RbfKernel(gamma) => {
                gamma.to_bits().hash(state);
            }
            Distance::PolynomialKernel {
                degree,
                gamma,
                coef0,
            } => {
                degree.to_bits().hash(state);
                gamma.to_bits().hash(state);
                coef0.to_bits().hash(state);
            }
            Distance::SigmoidKernel { gamma, coef0 } => {
                gamma.to_bits().hash(state);
                coef0.to_bits().hash(state);
            }
            Distance::LaplacianKernel(gamma) => {
                gamma.to_bits().hash(state);
            }
        }
    }
}

impl std::fmt::Debug for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Distance::Euclidean => write!(f, "Euclidean"),
            Distance::Manhattan => write!(f, "Manhattan"),
            Distance::Chebyshev => write!(f, "Chebyshev"),
            Distance::Minkowski(p) => write!(f, "Minkowski({:?})", p),
            Distance::Cosine => write!(f, "Cosine"),
            Distance::Hamming => write!(f, "Hamming"),
            Distance::Mahalanobis(m) => write!(f, "Mahalanobis({:?})", m),
            Distance::Custom(_) => write!(f, "Custom(<function>)"),
            Distance::RbfKernel(gamma) => write!(f, "RbfKernel({:?})", gamma),
            Distance::PolynomialKernel {
                degree,
                gamma,
                coef0,
            } => {
                write!(
                    f,
                    "PolynomialKernel {{ degree: {:?}, gamma: {:?}, coef0: {:?} }}",
                    degree, gamma, coef0
                )
            }
            Distance::SigmoidKernel { gamma, coef0 } => {
                write!(
                    f,
                    "SigmoidKernel {{ gamma: {:?}, coef0: {:?} }}",
                    gamma, coef0
                )
            }
            Distance::LaplacianKernel(gamma) => write!(f, "LaplacianKernel({:?})", gamma),
        }
    }
}

impl Clone for Distance {
    fn clone(&self) -> Self {
        match self {
            Distance::Euclidean => Distance::Euclidean,
            Distance::Manhattan => Distance::Manhattan,
            Distance::Chebyshev => Distance::Chebyshev,
            Distance::Minkowski(p) => Distance::Minkowski(*p),
            Distance::Cosine => Distance::Cosine,
            Distance::Hamming => Distance::Hamming,
            Distance::Mahalanobis(m) => Distance::Mahalanobis(m.clone()),
            Distance::Custom(_) => {
                // We cannot clone function pointers, so we'll provide a default
                // In practice, custom distance functions should be created fresh
                Distance::Euclidean
            }
            Distance::RbfKernel(gamma) => Distance::RbfKernel(*gamma),
            Distance::PolynomialKernel {
                degree,
                gamma,
                coef0,
            } => Distance::PolynomialKernel {
                degree: *degree,
                gamma: *gamma,
                coef0: *coef0,
            },
            Distance::SigmoidKernel { gamma, coef0 } => Distance::SigmoidKernel {
                gamma: *gamma,
                coef0: *coef0,
            },
            Distance::LaplacianKernel(gamma) => Distance::LaplacianKernel(*gamma),
        }
    }
}

impl Distance {
    /// Calculate distance between two points
    pub fn calculate<F: FloatTrait + Zero + Clone + std::fmt::Debug + 'static>(
        &self,
        x1: &ArrayView1<F>,
        x2: &ArrayView1<F>,
    ) -> F {
        match self {
            Distance::Euclidean => euclidean_distance(x1, x2),
            Distance::Manhattan => manhattan_distance(x1, x2),
            Distance::Chebyshev => chebyshev_distance(x1, x2),
            Distance::Minkowski(p) => minkowski_distance(x1, x2, *p),
            Distance::Cosine => cosine_distance(x1, x2),
            Distance::Hamming => hamming_distance(x1, x2),
            Distance::Mahalanobis(inv_cov) => mahalanobis_distance(x1, x2, inv_cov),
            Distance::Custom(func) => {
                // Convert to Float arrays for the custom function
                let x1_float = x1.mapv(|x| F::to_f64(&x).unwrap() as Float);
                let x2_float = x2.mapv(|x| F::to_f64(&x).unwrap() as Float);
                let result = func(&x1_float.view(), &x2_float.view());
                F::from(result).unwrap()
            }
            Distance::RbfKernel(gamma) => kernel_to_distance(rbf_kernel(x1, x2, *gamma)),
            Distance::PolynomialKernel {
                degree,
                gamma,
                coef0,
            } => kernel_to_distance(polynomial_kernel(x1, x2, *degree, *gamma, *coef0)),
            Distance::SigmoidKernel { gamma, coef0 } => {
                kernel_to_distance(sigmoid_kernel(x1, x2, *gamma, *coef0))
            }
            Distance::LaplacianKernel(gamma) => {
                kernel_to_distance(laplacian_kernel(x1, x2, *gamma))
            }
        }
    }

    /// Alias for calculate method for backward compatibility
    pub fn distance<F: FloatTrait + Zero + Clone + std::fmt::Debug + 'static>(
        &self,
        x1: &ArrayView1<F>,
        x2: &ArrayView1<F>,
    ) -> F {
        self.calculate(x1, x2)
    }

    /// Calculate pairwise distances between all points in two matrices
    pub fn pairwise<F: FloatTrait + Zero + Clone + std::fmt::Debug + 'static>(
        &self,
        x: &ArrayView2<F>,
        y: &ArrayView2<F>,
    ) -> Array2<F> {
        let n_samples_x = x.nrows();
        let n_samples_y = y.nrows();
        let mut distances = Array2::zeros((n_samples_x, n_samples_y));

        for (i, x_row) in x.axis_iter(Axis(0)).enumerate() {
            for (j, y_row) in y.axis_iter(Axis(0)).enumerate() {
                distances[[i, j]] = self.calculate(&x_row, &y_row);
            }
        }

        distances
    }

    /// Calculate distances from a single point to all points in a matrix
    pub fn to_matrix<F: FloatTrait + Zero + Clone + std::fmt::Debug + 'static>(
        &self,
        point: &ArrayView1<F>,
        matrix: &ArrayView2<F>,
    ) -> Array1<F> {
        let n_samples = matrix.nrows();
        let mut distances = Array1::zeros(n_samples);

        for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
            distances[i] = self.calculate(point, &row);
        }

        distances
    }

    /// Create a custom distance function
    pub fn custom<F>(func: F) -> Self
    where
        F: Fn(&ArrayView1<Float>, &ArrayView1<Float>) -> Float + Send + Sync + 'static,
    {
        Distance::Custom(Box::new(func))
    }

    /// Create an RBF (Gaussian) kernel distance
    pub fn rbf_kernel(gamma: Float) -> Self {
        Distance::RbfKernel(gamma)
    }

    /// Create a polynomial kernel distance
    pub fn polynomial_kernel(degree: Float, gamma: Float, coef0: Float) -> Self {
        Distance::PolynomialKernel {
            degree,
            gamma,
            coef0,
        }
    }

    /// Create a sigmoid kernel distance
    pub fn sigmoid_kernel(gamma: Float, coef0: Float) -> Self {
        Distance::SigmoidKernel { gamma, coef0 }
    }

    /// Create a Laplacian kernel distance
    pub fn laplacian_kernel(gamma: Float) -> Self {
        Distance::LaplacianKernel(gamma)
    }
}

/// Calculate Euclidean distance between two points
pub fn euclidean_distance<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
) -> F {
    // Use SIMD-optimized version for Float type (f64 by default)
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<Float>() {
        // Safe transmutation to Float arrays for same-size types
        let a1 = unsafe { std::mem::transmute::<&ArrayView1<F>, &ArrayView1<Float>>(x1) };
        let a2 = unsafe { std::mem::transmute::<&ArrayView1<F>, &ArrayView1<Float>>(x2) };
        let result = metrics::euclidean_distance(&a1.to_owned(), &a2.to_owned());
        // Safe cast back using from conversion
        F::from(result).unwrap_or_else(|| euclidean_distance_fallback(x1, x2))
    } else {
        // Fallback for other float types
        euclidean_distance_fallback(x1, x2)
    }
}

/// Fallback implementation for non-Float types
fn euclidean_distance_fallback<F: FloatTrait + Zero>(x1: &ArrayView1<F>, x2: &ArrayView1<F>) -> F {
    x1.iter()
        .zip(x2.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .fold(F::zero(), |acc, x| acc + x)
        .sqrt()
}

/// Calculate Manhattan distance between two points
pub fn manhattan_distance<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
) -> F {
    // Use SIMD-optimized version for Float type (f64 by default)
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<Float>() {
        // Safe transmutation to Float arrays for same-size types
        let a1 = unsafe { std::mem::transmute::<&ArrayView1<F>, &ArrayView1<Float>>(x1) };
        let a2 = unsafe { std::mem::transmute::<&ArrayView1<F>, &ArrayView1<Float>>(x2) };
        let result = metrics::manhattan_distance(&a1.to_owned(), &a2.to_owned());
        // Safe cast back using from conversion
        F::from(result).unwrap_or_else(|| manhattan_distance_fallback(x1, x2))
    } else {
        // Fallback for other float types
        manhattan_distance_fallback(x1, x2)
    }
}

/// Fallback implementation for non-Float types
fn manhattan_distance_fallback<F: FloatTrait + Zero>(x1: &ArrayView1<F>, x2: &ArrayView1<F>) -> F {
    x1.iter()
        .zip(x2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(F::zero(), |acc, x| acc + x)
}

/// Calculate Chebyshev distance between two points
pub fn chebyshev_distance<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
) -> F {
    x1.iter()
        .zip(x2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(F::zero(), |acc, x| acc.max(x))
}

/// Calculate Minkowski distance between two points
pub fn minkowski_distance<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
    p: Float,
) -> F {
    if p == 1.0 {
        return manhattan_distance(x1, x2);
    }
    if p == 2.0 {
        return euclidean_distance(x1, x2);
    }
    if p.is_infinite() {
        return chebyshev_distance(x1, x2);
    }

    let p_f = F::from(p).expect("Cannot convert p to target float type");
    x1.iter()
        .zip(x2.iter())
        .map(|(&a, &b)| (a - b).abs().powf(p_f))
        .fold(F::zero(), |acc, x| acc + x)
        .powf(F::one() / p_f)
}

/// Calculate cosine distance between two points
pub fn cosine_distance<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
) -> F {
    // Use SIMD-optimized version for Float type (f64 by default)
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<Float>() {
        // Safe transmutation to Float arrays for same-size types
        let a1 = unsafe { std::mem::transmute::<&ArrayView1<F>, &ArrayView1<Float>>(x1) };
        let a2 = unsafe { std::mem::transmute::<&ArrayView1<F>, &ArrayView1<Float>>(x2) };
        let result = metrics::cosine_distance(&a1.to_owned(), &a2.to_owned());
        // Safe cast back using from conversion
        F::from(result).unwrap_or_else(|| cosine_distance_fallback(x1, x2))
    } else {
        // Fallback for other float types
        cosine_distance_fallback(x1, x2)
    }
}

/// Fallback implementation for non-Float types
fn cosine_distance_fallback<F: FloatTrait + Zero>(x1: &ArrayView1<F>, x2: &ArrayView1<F>) -> F {
    let dot_product = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| a * b)
        .fold(F::zero(), |acc, x| acc + x);

    let norm_x1 = x1
        .iter()
        .map(|&x| x * x)
        .fold(F::zero(), |acc, x| acc + x)
        .sqrt();
    let norm_x2 = x2
        .iter()
        .map(|&x| x * x)
        .fold(F::zero(), |acc, x| acc + x)
        .sqrt();

    if norm_x1 == F::zero() || norm_x2 == F::zero() {
        return F::one(); // Maximum distance
    }

    let cosine_similarity = dot_product / (norm_x1 * norm_x2);
    F::one() - cosine_similarity
}

/// Calculate Hamming distance between two points (for binary/categorical data)
pub fn hamming_distance<F: FloatTrait + Zero + PartialEq + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
) -> F {
    let different_count = x1.iter().zip(x2.iter()).filter(|(&a, &b)| a != b).count();

    F::from(different_count).expect("Cannot convert count to target float type")
}

/// Calculate Mahalanobis distance between two points using inverse covariance matrix
pub fn mahalanobis_distance<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
    inv_cov: &Array2<Float>,
) -> F {
    // For non-Float types, we need to convert to Float for matrix operations
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<Float>() {
        // Safe transmutation for same-size types
        let a1 = unsafe { std::mem::transmute::<&ArrayView1<F>, &ArrayView1<Float>>(x1) };
        let a2 = unsafe { std::mem::transmute::<&ArrayView1<F>, &ArrayView1<Float>>(x2) };
        let result = mahalanobis_distance_float(a1, a2, inv_cov);
        F::from(result).unwrap_or_else(|| F::zero())
    } else {
        // Convert to Float for computation
        let x1_float: Array1<Float> = x1.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();
        let x2_float: Array1<Float> = x2.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

        let result = mahalanobis_distance_float(&x1_float.view(), &x2_float.view(), inv_cov);
        F::from(result).unwrap_or_else(|| F::zero())
    }
}

/// Calculate Mahalanobis distance for Float types
fn mahalanobis_distance_float(
    x1: &ArrayView1<Float>,
    x2: &ArrayView1<Float>,
    inv_cov: &Array2<Float>,
) -> Float {
    let diff = x1 - x2;
    let temp = inv_cov.dot(&diff);
    let result = diff.dot(&temp);
    result.sqrt()
}

impl Distance {
    /// Create a Mahalanobis distance metric from training data
    pub fn from_mahalanobis(data: &Array2<Float>) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Cannot compute covariance of empty data".to_string());
        }

        let cov_matrix = compute_covariance_matrix(data)?;
        let inv_cov_matrix = invert_matrix(&cov_matrix)?;

        Ok(Distance::Mahalanobis(inv_cov_matrix))
    }
}

/// Compute the covariance matrix of the data
fn compute_covariance_matrix(data: &Array2<Float>) -> Result<Array2<Float>, String> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples == 0 || n_features == 0 {
        return Err("Data must have non-zero samples and features".to_string());
    }

    // Compute mean for each feature
    let means = data.mean_axis(Axis(0)).unwrap();

    // Center the data
    let mut centered = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            centered[[i, j]] = data[[i, j]] - means[j];
        }
    }

    // Compute covariance matrix
    let cov = centered.t().dot(&centered) / (n_samples - 1) as Float;

    Ok(cov)
}

/// Invert a matrix using simple LU decomposition for small matrices
fn invert_matrix(matrix: &Array2<Float>) -> Result<Array2<Float>, String> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err("Matrix must be square".to_string());
    }

    // For now, implement a simple inversion for small matrices using Gauss-Jordan elimination
    let mut augmented = Array2::zeros((n, 2 * n));

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = matrix[[i, j]];
            augmented[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
        }
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                max_row = k;
            }
        }

        if augmented[[max_row, i]].abs() < 1e-10 {
            return Err("Matrix is singular and cannot be inverted".to_string());
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = augmented[[i, j]];
                augmented[[i, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = temp;
            }
        }

        // Make diagonal element 1
        let pivot = augmented[[i, i]];
        for j in 0..(2 * n) {
            augmented[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = augmented[[k, i]];
                for j in 0..(2 * n) {
                    augmented[[k, j]] -= factor * augmented[[i, j]];
                }
            }
        }
    }

    // Extract inverse from the right half
    let mut inverse = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inverse[[i, j]] = augmented[[i, j + n]];
        }
    }

    Ok(inverse)
}

/// Convert kernel similarity to distance
/// Distance = sqrt(2 * (1 - kernel))
fn kernel_to_distance<F: FloatTrait + Zero + 'static>(kernel_value: F) -> F {
    let two = F::one() + F::one();
    let one = F::one();
    (two * (one - kernel_value)).max(F::zero()).sqrt()
}

/// RBF (Gaussian) kernel function
/// K(x, y) = exp(-gamma * ||x - y||^2)
fn rbf_kernel<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
    gamma: Float,
) -> F {
    let squared_distance = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        })
        .fold(F::zero(), |acc, x| acc + x);

    let gamma_f = F::from(gamma).expect("Cannot convert gamma to target float type");
    (-gamma_f * squared_distance).exp()
}

/// Polynomial kernel function
/// K(x, y) = (gamma * <x, y> + coef0)^degree
fn polynomial_kernel<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
    degree: Float,
    gamma: Float,
    coef0: Float,
) -> F {
    let dot_product = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| a * b)
        .fold(F::zero(), |acc, x| acc + x);

    let gamma_f = F::from(gamma).expect("Cannot convert gamma to target float type");
    let coef0_f = F::from(coef0).expect("Cannot convert coef0 to target float type");
    let degree_f = F::from(degree).expect("Cannot convert degree to target float type");

    (gamma_f * dot_product + coef0_f).powf(degree_f)
}

/// Sigmoid kernel function  
/// K(x, y) = tanh(gamma * <x, y> + coef0)
fn sigmoid_kernel<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
    gamma: Float,
    coef0: Float,
) -> F {
    let dot_product = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| a * b)
        .fold(F::zero(), |acc, x| acc + x);

    let gamma_f = F::from(gamma).expect("Cannot convert gamma to target float type");
    let coef0_f = F::from(coef0).expect("Cannot convert coef0 to target float type");

    (gamma_f * dot_product + coef0_f).tanh()
}

/// Laplacian kernel function
/// K(x, y) = exp(-gamma * ||x - y||)
fn laplacian_kernel<F: FloatTrait + Zero + 'static>(
    x1: &ArrayView1<F>,
    x2: &ArrayView1<F>,
    gamma: Float,
) -> F {
    let manhattan_dist = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(F::zero(), |acc, x| acc + x);

    let gamma_f = F::from(gamma).expect("Cannot convert gamma to target float type");
    (-gamma_f * manhattan_dist).exp()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_euclidean_distance() {
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![4.0, 6.0, 8.0];

        let dist = euclidean_distance(&x1.view(), &x2.view());
        let expected = ((3.0_f64).powi(2) + (4.0_f64).powi(2) + (5.0_f64).powi(2)).sqrt();

        assert_abs_diff_eq!(dist, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_manhattan_distance() {
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![4.0, 6.0, 8.0];

        let dist = manhattan_distance(&x1.view(), &x2.view());
        let expected = 3.0 + 4.0 + 5.0;

        assert_abs_diff_eq!(dist, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_chebyshev_distance() {
        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![4.0, 6.0, 8.0];

        let dist = chebyshev_distance(&x1.view(), &x2.view());
        let expected = 5.0; // max of |1-4|, |2-6|, |3-8|

        assert_abs_diff_eq!(dist, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_cosine_distance() {
        let x1 = array![1.0, 0.0, 0.0];
        let x2 = array![0.0, 1.0, 0.0];

        let dist = cosine_distance(&x1.view(), &x2.view());
        let expected = 1.0; // Orthogonal vectors have cosine similarity 0, so distance is 1

        assert_abs_diff_eq!(dist, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_distance_enum() {
        let x1 = array![1.0, 2.0];
        let x2 = array![4.0, 6.0];

        let euclidean = Distance::Euclidean;
        let manhattan = Distance::Manhattan;

        let dist1 = euclidean.calculate(&x1.view(), &x2.view());
        let dist2 = manhattan.calculate(&x1.view(), &x2.view());

        assert_abs_diff_eq!(dist1, 5.0, epsilon = 1e-7); // sqrt(9 + 16)
        assert_abs_diff_eq!(dist2, 7.0, epsilon = 1e-7); // 3 + 4
    }

    #[test]
    fn test_pairwise_distances() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let distance = Distance::Euclidean;
        let distances = distance.pairwise(&x.view(), &y.view());

        assert_eq!(distances.shape(), &[2, 2]);

        // Distance from [1, 2] to [5, 6] should be sqrt(16 + 16) = sqrt(32)
        let expected_00 = (32.0_f64).sqrt();
        assert_abs_diff_eq!(distances[[0, 0]], expected_00, epsilon = 1e-7);
    }

    #[test]
    fn test_mahalanobis_distance() {
        // Create simple 2D data with known covariance structure
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 4.0, 2.0]).unwrap();

        let mahalanobis_metric = Distance::from_mahalanobis(&data).unwrap();

        let x1 = array![1.0, 2.0];
        let x2 = array![2.0, 3.0];

        let distance = mahalanobis_metric.calculate(&x1.view(), &x2.view());

        // Should compute a valid distance (positive value)
        assert!(distance > 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_mahalanobis_vs_euclidean_identity() {
        // When covariance matrix is identity, Mahalanobis should equal Euclidean
        let identity = Array2::eye(2);
        let mahalanobis_metric = Distance::Mahalanobis(identity);

        let x1 = array![1.0, 2.0];
        let x2 = array![4.0, 6.0];

        let mahalanobis_dist = mahalanobis_metric.calculate(&x1.view(), &x2.view());
        let euclidean_dist = euclidean_distance(&x1.view(), &x2.view());

        assert_abs_diff_eq!(mahalanobis_dist, euclidean_dist, epsilon = 1e-7);
    }

    #[test]
    fn test_covariance_matrix_computation() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();

        let cov = compute_covariance_matrix(&data).unwrap();
        assert_eq!(cov.shape(), &[2, 2]);

        // Covariance matrix should be symmetric
        assert_abs_diff_eq!(cov[[0, 1]], cov[[1, 0]], epsilon = 1e-7);

        // Diagonal elements should be positive (variances)
        assert!(cov[[0, 0]] > 0.0);
        assert!(cov[[1, 1]] > 0.0);
    }

    #[test]
    fn test_matrix_inversion() {
        // Test with a simple 2x2 matrix
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let inverse = invert_matrix(&matrix).unwrap();

        // Test that matrix * inverse = identity
        let product = matrix.dot(&inverse);
        let identity = Array2::eye(2);

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(product[[i, j]], identity[[i, j]], epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn test_mahalanobis_distance_creation_error() {
        let empty_data = Array2::zeros((0, 2));
        let result = Distance::from_mahalanobis(&empty_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_distance() {
        // Create a custom distance that computes the sum of squared differences
        let custom_dist = Distance::custom(|x1, x2| {
            x1.iter()
                .zip(x2.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum()
        });

        let x1 = array![1.0, 2.0, 3.0];
        let x2 = array![4.0, 6.0, 8.0];

        let dist = custom_dist.calculate(&x1.view(), &x2.view());
        let expected = (3.0_f64).powi(2) + (4.0_f64).powi(2) + (5.0_f64).powi(2);

        assert_abs_diff_eq!(dist, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_rbf_kernel_distance() {
        let gamma = 1.0;
        let rbf_dist = Distance::rbf_kernel(gamma);

        let x1 = array![1.0, 0.0];
        let x2 = array![0.0, 1.0];

        let dist = rbf_dist.calculate(&x1.view(), &x2.view());

        // RBF kernel for these points: exp(-gamma * 2) = exp(-2)
        // Distance = sqrt(2 * (1 - exp(-2)))
        let kernel_value = (-2.0_f64).exp();
        let expected = (2.0 * (1.0 - kernel_value)).sqrt();

        assert_abs_diff_eq!(dist, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_polynomial_kernel_distance() {
        let degree = 2.0;
        let gamma = 1.0;
        let coef0 = 1.0;
        let poly_dist = Distance::polynomial_kernel(degree, gamma, coef0);

        let x1 = array![1.0, 2.0];
        let x2 = array![3.0, 4.0];

        let dist = poly_dist.calculate(&x1.view(), &x2.view());

        // Dot product: 1*3 + 2*4 = 11
        // Kernel: (1.0 * 11 + 1.0)^2 = 12^2 = 144
        // Distance = sqrt(2 * (1 - 144)) - but we need to handle this properly
        // Since kernel can be > 1, we ensure non-negative distance
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_sigmoid_kernel_distance() {
        let gamma = 1.0;
        let coef0 = 0.0;
        let sigmoid_dist = Distance::sigmoid_kernel(gamma, coef0);

        let x1 = array![1.0, 0.0];
        let x2 = array![0.0, 1.0];

        let dist = sigmoid_dist.calculate(&x1.view(), &x2.view());

        // Dot product: 1*0 + 0*1 = 0
        // Kernel: tanh(1.0 * 0 + 0.0) = tanh(0) = 0
        // Distance = sqrt(2 * (1 - 0)) = sqrt(2)
        let expected = 2.0_f64.sqrt();

        assert_abs_diff_eq!(dist, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_laplacian_kernel_distance() {
        let gamma = 1.0;
        let laplacian_dist = Distance::laplacian_kernel(gamma);

        let x1 = array![1.0, 0.0];
        let x2 = array![0.0, 1.0];

        let dist = laplacian_dist.calculate(&x1.view(), &x2.view());

        // Manhattan distance: |1-0| + |0-1| = 2
        // Kernel: exp(-1.0 * 2) = exp(-2)
        // Distance = sqrt(2 * (1 - exp(-2)))
        let kernel_value = (-2.0_f64).exp();
        let expected = (2.0 * (1.0 - kernel_value)).sqrt();

        assert_abs_diff_eq!(dist, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_kernel_to_distance_conversion() {
        // Test that kernel_to_distance works correctly
        let kernel_value = 0.5;
        let distance = kernel_to_distance(kernel_value);
        let expected = (2.0 * (1.0 - 0.5)).sqrt();

        assert_abs_diff_eq!(distance, expected, epsilon = 1e-7);
    }

    #[test]
    fn test_distance_enum_partial_eq() {
        let euclidean1 = Distance::Euclidean;
        let euclidean2 = Distance::Euclidean;
        let manhattan = Distance::Manhattan;

        assert_eq!(euclidean1, euclidean2);
        assert_ne!(euclidean1, manhattan);

        let rbf1 = Distance::rbf_kernel(1.0);
        let rbf2 = Distance::rbf_kernel(1.0);
        let rbf3 = Distance::rbf_kernel(2.0);

        assert_eq!(rbf1, rbf2);
        assert_ne!(rbf1, rbf3);

        // Custom distances are never equal
        let custom1 =
            Distance::custom(|x1, x2| x1.iter().zip(x2.iter()).map(|(&a, &b)| (a - b).abs()).sum());
        let custom2 =
            Distance::custom(|x1, x2| x1.iter().zip(x2.iter()).map(|(&a, &b)| (a - b).abs()).sum());
        assert_ne!(custom1, custom2);
    }
}
