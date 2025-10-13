//! Train-test split functionality

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::SliceRandomExt;
use sklears_core::error::Result;

/// Split arrays or matrices into random train and test subsets
#[allow(clippy::type_complexity)]
pub fn train_test_split<X, Y>(
    x: &Array2<X>,
    y: &Array1<Y>,
    test_size: f64,
    random_state: Option<u64>,
) -> Result<(Array2<X>, Array2<X>, Array1<Y>, Array1<Y>)>
where
    X: Clone,
    Y: Clone,
{
    let n_samples = x.nrows();
    let n_test = (n_samples as f64 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    // Generate random indices
    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let train_indices = &indices[..n_train];
    let test_indices = &indices[n_train..];

    // Create train and test arrays
    let x_train = x.select(Axis(0), train_indices);
    let x_test = x.select(Axis(0), test_indices);
    let y_train = y.select(Axis(0), train_indices);
    let y_test = y.select(Axis(0), test_indices);

    Ok((x_train, x_test, y_train, y_test))
}
