//! Statistical utility functions

use scirs2_core::ndarray::{Array1, Array2, Axis, NdFloat};
use scirs2_core::numeric::FromPrimitive;

/// Compute mean along an axis
pub fn mean_axis<'a, D>(array: &'a Array2<D>, axis: Axis) -> Array1<D>
where
    D: NdFloat + FromPrimitive + 'a,
{
    array.mean_axis(axis).unwrap()
}

/// Compute variance along an axis
pub fn var_axis<'a, D>(array: &'a Array2<D>, axis: Axis, ddof: usize) -> Array1<D>
where
    D: NdFloat + FromPrimitive + 'a,
{
    let mean = array.mean_axis(axis).unwrap();
    let n = array.len_of(axis);

    if axis == Axis(0) {
        // Variance along rows (for each column)
        let mut var = Array1::zeros(array.ncols());
        for j in 0..array.ncols() {
            let col = array.column(j);
            let m = mean[j];
            let sum_sq: D = col.mapv(|x| (x - m).powi(2)).sum();
            var[j] = sum_sq / D::from(n - ddof).unwrap();
        }
        var
    } else {
        // Variance along columns (for each row)
        let mut var = Array1::zeros(array.nrows());
        for i in 0..array.nrows() {
            let row = array.row(i);
            let m = mean[i];
            let sum_sq: D = row.mapv(|x| (x - m).powi(2)).sum();
            var[i] = sum_sq / D::from(n - ddof).unwrap();
        }
        var
    }
}

/// Compute standard deviation along an axis
pub fn std_axis<'a, D>(array: &'a Array2<D>, axis: Axis, ddof: usize) -> Array1<D>
where
    D: NdFloat + FromPrimitive + 'a,
{
    var_axis(array, axis, ddof).mapv(|v| v.sqrt())
}

/// Compute covariance matrix
pub fn covariance<D>(x: &Array2<D>, ddof: usize) -> Array2<D>
where
    D: NdFloat + FromPrimitive,
{
    let n_samples = x.nrows();

    // Center the data
    let mean = x.mean_axis(Axis(0)).unwrap();
    let centered = x - &mean;

    // Compute covariance
    let cov = centered.t().dot(&centered) / D::from(n_samples - ddof).unwrap();
    cov
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_axis() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let mean_rows = mean_axis(&x, Axis(0));
        assert_abs_diff_eq!(mean_rows[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean_rows[1], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean_rows[2], 6.0, epsilon = 1e-10);

        let mean_cols = mean_axis(&x, Axis(1));
        assert_abs_diff_eq!(mean_cols[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean_cols[1], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean_cols[2], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_var_axis() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let var_rows = var_axis(&x, Axis(0), 0);
        assert_abs_diff_eq!(var_rows[0], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(var_rows[1], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(var_rows[2], 6.0, epsilon = 1e-10);
    }
}
