//! Serialization implementations for specific manifold learning algorithms.
//!
//! These implementations rely exclusively on the public API of each estimator
//! (public accessors and reconstruction constructors). No private fields are
//! touched, which keeps the serialization logic decoupled from the internal
//! representation of the algorithms.
//!
//! [`RandomProjection`] supports a lossless round-trip: the fitted projection
//! matrix together with its hyperparameters is stored in a [`SerializableModel`]
//! and can be reconstructed into an equivalent fitted model that produces
//! identical transforms. The binary (oxicode) and in-memory representations are
//! bit-for-bit exact; the JSON text format is written exactly but is only
//! reproduced to within one ULP because `serde_json` is configured here without
//! its `float_roundtrip` feature.
//!
//! [`TSNE`] and [`Isomap`] expose their learned embedding for serialization,
//! but they cannot be reconstructed into a *functional* fitted estimator from
//! the public API alone (a working t-SNE/Isomap model needs internal optimizer
//! and neighbor-graph state that is not part of the embedding). For those, the
//! deserialization side returns an honest error instead of fabricating a model.

use crate::serialization::{
    ArrayConverter, ModelMetadata, ModelState, Serializable, SerializableModel,
    SerializableModelBuilder, SerializableParam,
};
use crate::{Isomap, IsomapTrained, RPTrained, RandomProjection, TsneTrained, TSNE};
use scirs2_core::ndarray::Array2;
use sklears_core::error::{Result as SklResult, SklearsError};

/// Algorithm tag stored in the serialized model for [`RandomProjection`].
const RANDOM_PROJECTION_ALGORITHM: &str = "RandomProjection";

impl Serializable for RandomProjection<RPTrained> {
    fn to_serializable(&self) -> SklResult<SerializableModel> {
        let projection_matrix = self.projection_matrix();
        let (n_features, n_components) = projection_matrix.dim();

        let mut builder = SerializableModelBuilder::new(RANDOM_PROJECTION_ALGORITHM)
            .parameter(
                "n_components",
                SerializableParam::Int(self.n_components() as i64),
            )
            .parameter("density", SerializableParam::Float(self.density()))
            .parameter(
                "scaling_factor",
                SerializableParam::Float(self.scaling_factor()),
            )
            .projection_matrix(projection_matrix);

        if let Some(seed) = self.random_state() {
            builder = builder.parameter("random_state", SerializableParam::Int(seed as i64));
        }

        let mut model = builder.build();
        model.metadata = ModelMetadata {
            n_features,
            n_components,
            ..model.metadata
        };

        Ok(model)
    }

    fn from_serializable(serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized,
    {
        if serializable.algorithm != RANDOM_PROJECTION_ALGORITHM {
            return Err(SklearsError::InvalidInput(format!(
                "Expected algorithm '{}' but found '{}'",
                RANDOM_PROJECTION_ALGORITHM, serializable.algorithm
            )));
        }

        let projection_matrix = reconstruct_projection_matrix(&serializable.state)?;

        let n_components = match serializable.parameters.get("n_components") {
            Some(SerializableParam::Int(value)) if *value > 0 => *value as usize,
            Some(_) => {
                return Err(SklearsError::InvalidInput(
                    "Parameter 'n_components' must be a positive integer".to_string(),
                ))
            }
            None => projection_matrix.ncols(),
        };

        let density = match serializable.parameters.get("density") {
            Some(SerializableParam::Float(value)) => *value,
            Some(_) => {
                return Err(SklearsError::InvalidInput(
                    "Parameter 'density' must be a float".to_string(),
                ))
            }
            None => 1.0,
        };

        let random_state = match serializable.parameters.get("random_state") {
            Some(SerializableParam::Int(value)) => Some(*value as u64),
            Some(_) => {
                return Err(SklearsError::InvalidInput(
                    "Parameter 'random_state' must be an integer".to_string(),
                ))
            }
            None => None,
        };

        RandomProjection::from_fitted(projection_matrix, n_components, density, random_state)
    }
}

/// Reconstruct the projection matrix stored in a [`ModelState`].
fn reconstruct_projection_matrix(state: &ModelState) -> SklResult<Array2<f64>> {
    let rows = state.projection_matrix.as_ref().ok_or_else(|| {
        SklearsError::InvalidInput("Serialized model is missing the projection matrix".to_string())
    })?;
    ArrayConverter::vec_to_array2(rows)
}

impl Serializable for TSNE<TsneTrained> {
    fn to_serializable(&self) -> SklResult<SerializableModel> {
        let embedding = self.embedding();
        let (n_samples, n_components) = embedding.dim();

        let mut model = SerializableModelBuilder::new("TSNE")
            .parameter("n_components", SerializableParam::Int(n_components as i64))
            .embedding(embedding)
            .build();

        model.metadata = ModelMetadata {
            n_samples,
            n_components,
            ..model.metadata
        };

        Ok(model)
    }

    fn from_serializable(_serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized,
    {
        Err(SklearsError::InvalidInput(
            "Reconstructing a fitted TSNE model from a serialized embedding is not supported: a \
             functional t-SNE estimator requires internal optimizer state that is not part of the \
             public API"
                .to_string(),
        ))
    }
}

impl Serializable for Isomap<IsomapTrained> {
    fn to_serializable(&self) -> SklResult<SerializableModel> {
        let embedding = self.embedding();
        let (n_samples, n_components) = embedding.dim();

        let mut model = SerializableModelBuilder::new("Isomap")
            .parameter("n_components", SerializableParam::Int(n_components as i64))
            .embedding(embedding)
            .build();

        model.metadata = ModelMetadata {
            n_samples,
            n_components,
            ..model.metadata
        };

        Ok(model)
    }

    fn from_serializable(_serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized,
    {
        Err(SklearsError::InvalidInput(
            "Reconstructing a fitted Isomap model from a serialized embedding is not supported: a \
             functional Isomap estimator requires the neighbor graph and geodesic distances that \
             are not part of the public API"
                .to_string(),
        ))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialization::{ModelSerializer, SerializationFormat};
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Transform};
    use std::env::temp_dir;

    /// Fit a deterministic `RandomProjection` for use in the round-trip tests.
    fn fit_reference_model() -> RandomProjection<RPTrained> {
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [9.0, 10.0, 11.0, 12.0, 13.0],
            [2.0, 4.0, 6.0, 8.0, 10.0]
        ];

        RandomProjection::new()
            .n_components(3)
            .density(1.0)
            .random_state(42)
            .fit(&x.view(), &())
            .expect("fitting RandomProjection should succeed")
    }

    #[test]
    fn test_random_projection_to_serializable_captures_state() {
        let model = fit_reference_model();
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");

        assert_eq!(serialized.algorithm, "RandomProjection");
        assert!(serialized.state.projection_matrix.is_some());

        let projection = model.projection_matrix();
        assert_eq!(serialized.metadata.n_features, projection.nrows());
        assert_eq!(serialized.metadata.n_components, projection.ncols());

        match serialized.parameters.get("n_components") {
            Some(SerializableParam::Int(value)) => assert_eq!(*value, 3),
            other => panic!("unexpected n_components parameter: {:?}", other),
        }
        match serialized.parameters.get("random_state") {
            Some(SerializableParam::Int(value)) => assert_eq!(*value, 42),
            other => panic!("unexpected random_state parameter: {:?}", other),
        }
    }

    #[test]
    fn test_random_projection_round_trip_in_memory() {
        let model = fit_reference_model();

        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        let restored = RandomProjection::from_serializable(&serialized)
            .expect("deserialization should succeed");

        // Hyperparameters must be preserved exactly.
        assert_eq!(restored.n_components(), model.n_components());
        assert_eq!(restored.density(), model.density());
        assert_eq!(restored.random_state(), model.random_state());
        assert_eq!(restored.scaling_factor(), model.scaling_factor());

        // The projection matrix must be reproduced bit-for-bit.
        assert_eq!(restored.projection_matrix(), model.projection_matrix());

        // The reconstructed model must produce identical transforms.
        let x = array![[3.0, 1.0, 4.0, 1.0, 5.0], [9.0, 2.0, 6.0, 5.0, 3.0]];
        let original_transform = model
            .transform(&x.view())
            .expect("transform should succeed");
        let restored_transform = restored
            .transform(&x.view())
            .expect("transform should succeed");
        assert_eq!(original_transform, restored_transform);
    }

    #[test]
    fn test_random_projection_round_trip_json_file() {
        use approx::assert_abs_diff_eq;

        let model = fit_reference_model();
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");

        let mut path = temp_dir();
        path.push(format!("sklears_manifold_rp_{}.json", std::process::id()));

        ModelSerializer::save_to_file(&serialized, &path, SerializationFormat::Json)
            .expect("saving model should succeed");
        let loaded = ModelSerializer::load_from_file(&path, SerializationFormat::Json)
            .expect("loading model should succeed");
        let _ = std::fs::remove_file(&path);

        let restored =
            RandomProjection::from_serializable(&loaded).expect("deserialization should succeed");

        // The JSON text format is written exactly (Ryu) but `serde_json` is built
        // here without its `float_roundtrip` feature, so parsing may differ from
        // the original f64 by up to one ULP. The reconstructed matrix is therefore
        // numerically equal rather than bit-identical; binary/in-memory round-trips
        // (tested separately) are exact.
        let restored_matrix = restored.projection_matrix();
        let original_matrix = model.projection_matrix();
        assert_eq!(restored_matrix.dim(), original_matrix.dim());
        for (restored_value, original_value) in restored_matrix.iter().zip(original_matrix.iter()) {
            assert_abs_diff_eq!(restored_value, original_value, epsilon = 1e-12);
        }

        let x = array![[0.5, 1.5, 2.5, 3.5, 4.5]];
        let original_transform = model
            .transform(&x.view())
            .expect("transform should succeed");
        let restored_transform = restored
            .transform(&x.view())
            .expect("transform should succeed");
        assert_eq!(original_transform.dim(), restored_transform.dim());
        for (restored_value, original_value) in
            restored_transform.iter().zip(original_transform.iter())
        {
            assert_abs_diff_eq!(restored_value, original_value, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_random_projection_round_trip_binary() {
        let model = fit_reference_model();
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");

        let bytes =
            ModelSerializer::to_binary(&serialized).expect("binary encoding should succeed");
        let decoded = ModelSerializer::from_binary(&bytes).expect("binary decoding should succeed");
        let restored =
            RandomProjection::from_serializable(&decoded).expect("deserialization should succeed");

        let x = array![[7.0, 7.0, 7.0, 7.0, 7.0], [1.0, 2.0, 3.0, 4.0, 5.0]];
        let original_transform = model
            .transform(&x.view())
            .expect("transform should succeed");
        let restored_transform = restored
            .transform(&x.view())
            .expect("transform should succeed");
        assert_eq!(original_transform, restored_transform);
    }

    #[test]
    fn test_from_serializable_rejects_wrong_algorithm() {
        let model = fit_reference_model();
        let mut serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        serialized.algorithm = "SomethingElse".to_string();

        let result = RandomProjection::from_serializable(&serialized);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_serializable_rejects_missing_matrix() {
        let model = fit_reference_model();
        let mut serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        serialized.state.projection_matrix = None;

        let result = RandomProjection::from_serializable(&serialized);
        assert!(result.is_err());
    }

    #[test]
    fn test_tsne_isomap_deserialization_is_honest_error() {
        let model = fit_reference_model();
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");

        // The TSNE/Isomap deserialization paths must report an honest error
        // rather than fabricating a fitted model.
        assert!(TSNE::<TsneTrained>::from_serializable(&serialized).is_err());
        assert!(Isomap::<IsomapTrained>::from_serializable(&serialized).is_err());
    }
}
