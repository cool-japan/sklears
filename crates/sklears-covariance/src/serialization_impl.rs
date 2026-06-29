//! Serialization implementations for concrete covariance estimators.
//!
//! Each implementation relies exclusively on the public API of the estimator
//! (the `get_*` accessors and the `from_fitted` reconstruction constructors).
//! No private fields are touched, which keeps the serialization logic decoupled
//! from the internal representation of the estimators.
//!
//! The round-trip is lossless: a fitted estimator is converted into a
//! [`SerializableModel`], persisted (or kept in memory) and reconstructed into an
//! estimator whose stored covariance, precision, location and hyperparameters are
//! identical (bit-for-bit for the binary/MessagePack/in-memory paths; within one
//! ULP for JSON, as documented in [`crate::serialization`]).

use crate::serialization::{
    check_algorithm, float_param, optional_precision, require_covariance, require_location,
    usize_param, ArrayConverter, Serializable, SerializableModel, SerializableModelBuilder,
    SerializableParam,
};
use crate::{
    EmpiricalCovariance, EmpiricalCovarianceTrained, GraphicalLasso, GraphicalLassoTrained,
    LedoitWolf, LedoitWolfTrained, MinCovDet, MinCovDetTrained, OASTrained, ShrunkCovariance,
    ShrunkCovarianceTrained, OAS,
};
use sklears_core::error::{Result as SklResult, SklearsError};

const EMPIRICAL_ALGORITHM: &str = "EmpiricalCovariance";
const SHRUNK_ALGORITHM: &str = "ShrunkCovariance";
const LEDOIT_WOLF_ALGORITHM: &str = "LedoitWolf";
const OAS_ALGORITHM: &str = "OAS";
const GRAPHICAL_LASSO_ALGORITHM: &str = "GraphicalLasso";
const MIN_COV_DET_ALGORITHM: &str = "MinCovDet";

impl Serializable for EmpiricalCovariance<EmpiricalCovarianceTrained> {
    fn to_serializable(&self) -> SklResult<SerializableModel> {
        let mut builder = SerializableModelBuilder::new(EMPIRICAL_ALGORITHM)
            .covariance(self.get_covariance())
            .location(self.get_location());
        if let Some(precision) = self.get_precision() {
            builder = builder.precision(precision);
        }
        Ok(builder.build())
    }

    fn from_serializable(serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized,
    {
        check_algorithm(serializable, EMPIRICAL_ALGORITHM)?;
        let covariance = require_covariance(&serializable.state)?;
        let precision = optional_precision(&serializable.state)?;
        let location = require_location(&serializable.state)?;
        Ok(EmpiricalCovariance::from_fitted(
            covariance, precision, location,
        ))
    }
}

impl Serializable for ShrunkCovariance<ShrunkCovarianceTrained> {
    fn to_serializable(&self) -> SklResult<SerializableModel> {
        let mut builder = SerializableModelBuilder::new(SHRUNK_ALGORITHM)
            .parameter("shrinkage", SerializableParam::Float(self.get_shrinkage()))
            .covariance(self.get_covariance())
            .location(self.get_location());
        if let Some(precision) = self.get_precision() {
            builder = builder.precision(precision);
        }
        Ok(builder.build())
    }

    fn from_serializable(serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized,
    {
        check_algorithm(serializable, SHRUNK_ALGORITHM)?;
        let covariance = require_covariance(&serializable.state)?;
        let precision = optional_precision(&serializable.state)?;
        let location = require_location(&serializable.state)?;
        let shrinkage = float_param(serializable, "shrinkage")?;
        Ok(ShrunkCovariance::from_fitted(
            covariance, precision, location, shrinkage,
        ))
    }
}

impl Serializable for LedoitWolf<LedoitWolfTrained> {
    fn to_serializable(&self) -> SklResult<SerializableModel> {
        let mut builder = SerializableModelBuilder::new(LEDOIT_WOLF_ALGORITHM)
            .parameter("shrinkage", SerializableParam::Float(self.get_shrinkage()))
            .covariance(self.get_covariance())
            .location(self.get_location());
        if let Some(precision) = self.get_precision() {
            builder = builder.precision(precision);
        }
        Ok(builder.build())
    }

    fn from_serializable(serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized,
    {
        check_algorithm(serializable, LEDOIT_WOLF_ALGORITHM)?;
        let covariance = require_covariance(&serializable.state)?;
        let precision = optional_precision(&serializable.state)?;
        let location = require_location(&serializable.state)?;
        let shrinkage = float_param(serializable, "shrinkage")?;
        // `block_size` only affects (re)fitting, never the stored state. When the
        // artifact was written by this crate it is present; otherwise fall back to
        // the documented default used by `LedoitWolf::new`.
        let block_size = match serializable.parameters.get("block_size") {
            Some(SerializableParam::Int(value)) if *value > 0 => *value as usize,
            _ => 1000,
        };
        Ok(LedoitWolf::from_fitted(
            covariance, precision, location, shrinkage, block_size,
        ))
    }
}

impl Serializable for OAS<OASTrained> {
    fn to_serializable(&self) -> SklResult<SerializableModel> {
        let mut builder = SerializableModelBuilder::new(OAS_ALGORITHM)
            .parameter("shrinkage", SerializableParam::Float(self.get_shrinkage()))
            .covariance(self.get_covariance())
            .location(self.get_location());
        if let Some(precision) = self.get_precision() {
            builder = builder.precision(precision);
        }
        Ok(builder.build())
    }

    fn from_serializable(serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized,
    {
        check_algorithm(serializable, OAS_ALGORITHM)?;
        let covariance = require_covariance(&serializable.state)?;
        let precision = optional_precision(&serializable.state)?;
        let location = require_location(&serializable.state)?;
        let shrinkage = float_param(serializable, "shrinkage")?;
        Ok(OAS::from_fitted(covariance, precision, location, shrinkage))
    }
}

impl Serializable for GraphicalLasso<GraphicalLassoTrained> {
    fn to_serializable(&self) -> SklResult<SerializableModel> {
        Ok(SerializableModelBuilder::new(GRAPHICAL_LASSO_ALGORITHM)
            .parameter("alpha", SerializableParam::Float(self.get_alpha()))
            .parameter("n_iter", SerializableParam::Int(self.get_n_iter() as i64))
            .covariance(self.get_covariance())
            .precision(self.get_precision())
            .location(self.get_location())
            .build())
    }

    fn from_serializable(serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized,
    {
        check_algorithm(serializable, GRAPHICAL_LASSO_ALGORITHM)?;
        let covariance = require_covariance(&serializable.state)?;
        let precision = optional_precision(&serializable.state)?.ok_or_else(|| {
            SklearsError::InvalidInput(
                "GraphicalLasso requires a precision matrix in the serialized state".to_string(),
            )
        })?;
        let location = require_location(&serializable.state)?;
        let alpha = float_param(serializable, "alpha")?;
        let n_iter = usize_param(serializable, "n_iter")?;
        // Reconstruct hyperparameters that govern (re)fitting with the documented
        // defaults from `GraphicalLasso::new`; `alpha` and `n_iter` come from the
        // stored fit so the reconstructed state matches exactly.
        Ok(GraphicalLasso::from_fitted(
            covariance,
            precision,
            location,
            alpha,
            n_iter,
            "cd".to_string(),
            1e-4,
            100,
        ))
    }
}

impl Serializable for MinCovDet<MinCovDetTrained> {
    fn to_serializable(&self) -> SklResult<SerializableModel> {
        let mut builder = SerializableModelBuilder::new(MIN_COV_DET_ALGORITHM)
            .covariance(self.get_covariance())
            .location(self.get_location())
            .support(self.get_support())
            .distances(self.get_dist());
        if let Some(precision) = self.get_precision() {
            builder = builder.precision(precision);
        }
        Ok(builder.build())
    }

    fn from_serializable(serializable: &SerializableModel) -> SklResult<Self>
    where
        Self: Sized,
    {
        check_algorithm(serializable, MIN_COV_DET_ALGORITHM)?;
        let covariance = require_covariance(&serializable.state)?;
        let precision = optional_precision(&serializable.state)?;
        let location = require_location(&serializable.state)?;
        let support_values = serializable.state.support.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput(
                "MinCovDet requires a support mask in the serialized state".to_string(),
            )
        })?;
        let support = ArrayConverter::vec_to_array1_bool(support_values);
        let distances = match serializable.state.distances.as_ref() {
            Some(values) => ArrayConverter::vec_to_array1(values),
            None => ArrayConverter::vec_to_array1(&vec![0.0; support.len()]),
        };
        // Hyperparameters that govern (re)fitting are restored with the documented
        // defaults from `MinCovDet::new`; the stored state is preserved exactly.
        Ok(MinCovDet::from_fitted(
            covariance, precision, location, support, distances, None, None, 500, true,
        ))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialization::{ModelSerializer, SerializationFormat};
    use scirs2_core::ndarray::{array, Array2};
    use sklears_core::traits::Fit;
    use std::env::temp_dir;

    fn reference_data() -> Array2<f64> {
        array![
            [1.0, 0.1, 0.2],
            [2.0, 1.9, 0.1],
            [3.0, 2.8, 0.4],
            [4.0, 4.1, 0.3],
            [5.0, 4.9, 0.6],
            [6.0, 5.8, 0.2]
        ]
    }

    fn assert_matrix_eq(left: &Array2<f64>, right: &Array2<f64>) {
        assert_eq!(left.dim(), right.dim());
        for (a, b) in left.iter().zip(right.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_empirical_round_trip_all_formats() {
        let data = reference_data();
        let model = EmpiricalCovariance::new()
            .fit(&data.view(), &())
            .expect("fitting should succeed");
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");

        for format in [
            SerializationFormat::Binary,
            SerializationFormat::MessagePack,
        ] {
            let bytes =
                ModelSerializer::to_bytes(&serialized, format).expect("encoding should succeed");
            let decoded =
                ModelSerializer::from_bytes(&bytes, format).expect("decoding should succeed");
            let restored = EmpiricalCovariance::from_serializable(&decoded)
                .expect("reconstruction should succeed");

            assert_matrix_eq(restored.get_covariance(), model.get_covariance());
            assert_matrix_eq(
                restored.get_precision().expect("precision present"),
                model.get_precision().expect("precision present"),
            );
            assert_eq!(restored.get_location(), model.get_location());
        }
    }

    #[test]
    fn test_empirical_round_trip_json_file() {
        use approx::assert_abs_diff_eq;

        let data = reference_data();
        let model = EmpiricalCovariance::new()
            .fit(&data.view(), &())
            .expect("fitting should succeed");
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");

        let mut path = temp_dir();
        path.push(format!("sklears_cov_empirical_{}.json", std::process::id()));
        ModelSerializer::save_to_file(&serialized, &path, SerializationFormat::Json)
            .expect("saving should succeed");
        let loaded = ModelSerializer::load_from_file(&path, SerializationFormat::Json)
            .expect("loading should succeed");
        let _ = std::fs::remove_file(&path);

        let restored =
            EmpiricalCovariance::from_serializable(&loaded).expect("reconstruction should succeed");

        let restored_cov = restored.get_covariance();
        let original_cov = model.get_covariance();
        assert_eq!(restored_cov.dim(), original_cov.dim());
        for (a, b) in restored_cov.iter().zip(original_cov.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_shrunk_round_trip_preserves_shrinkage() {
        let data = reference_data();
        let model = ShrunkCovariance::new()
            .shrinkage(0.3)
            .fit(&data.view(), &())
            .expect("fitting should succeed");
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        let bytes = ModelSerializer::to_binary(&serialized).expect("encoding should succeed");
        let decoded = ModelSerializer::from_binary(&bytes).expect("decoding should succeed");
        let restored =
            ShrunkCovariance::from_serializable(&decoded).expect("reconstruction should succeed");

        assert_eq!(restored.get_shrinkage(), model.get_shrinkage());
        assert_matrix_eq(restored.get_covariance(), model.get_covariance());
        assert_eq!(restored.get_location(), model.get_location());
    }

    #[test]
    fn test_ledoit_wolf_round_trip() {
        let data = reference_data();
        let model = LedoitWolf::new()
            .fit(&data.view(), &())
            .expect("fitting should succeed");
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        let bytes = ModelSerializer::to_binary(&serialized).expect("encoding should succeed");
        let decoded = ModelSerializer::from_binary(&bytes).expect("decoding should succeed");
        let restored =
            LedoitWolf::from_serializable(&decoded).expect("reconstruction should succeed");

        assert_eq!(restored.get_shrinkage(), model.get_shrinkage());
        assert_matrix_eq(restored.get_covariance(), model.get_covariance());
        assert_matrix_eq(
            restored.get_precision().expect("precision present"),
            model.get_precision().expect("precision present"),
        );
    }

    #[test]
    fn test_oas_round_trip() {
        let data = reference_data();
        let model = OAS::new()
            .fit(&data.view(), &())
            .expect("fitting should succeed");
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        let bytes = ModelSerializer::to_binary(&serialized).expect("encoding should succeed");
        let decoded = ModelSerializer::from_binary(&bytes).expect("decoding should succeed");
        let restored = OAS::from_serializable(&decoded).expect("reconstruction should succeed");

        assert_eq!(restored.get_shrinkage(), model.get_shrinkage());
        assert_matrix_eq(restored.get_covariance(), model.get_covariance());
        assert_eq!(restored.get_location(), model.get_location());
    }

    #[test]
    fn test_graphical_lasso_round_trip() {
        let data = reference_data();
        let model = GraphicalLasso::new()
            .alpha(0.1)
            .fit(&data.view(), &())
            .expect("fitting should succeed");
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        let bytes = ModelSerializer::to_binary(&serialized).expect("encoding should succeed");
        let decoded = ModelSerializer::from_binary(&bytes).expect("decoding should succeed");
        let restored =
            GraphicalLasso::from_serializable(&decoded).expect("reconstruction should succeed");

        assert_eq!(restored.get_alpha(), model.get_alpha());
        assert_eq!(restored.get_n_iter(), model.get_n_iter());
        assert_matrix_eq(restored.get_covariance(), model.get_covariance());
        assert_matrix_eq(restored.get_precision(), model.get_precision());
    }

    #[test]
    fn test_min_cov_det_round_trip_preserves_support() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [1.2, 2.2],
            [0.8, 1.8],
            [50.0, 60.0]
        ];
        let model = MinCovDet::new()
            .support_fraction(0.75)
            .fit(&data.view(), &())
            .expect("fitting should succeed");
        let serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        let bytes = ModelSerializer::to_binary(&serialized).expect("encoding should succeed");
        let decoded = ModelSerializer::from_binary(&bytes).expect("decoding should succeed");
        let restored =
            MinCovDet::from_serializable(&decoded).expect("reconstruction should succeed");

        assert_matrix_eq(restored.get_covariance(), model.get_covariance());
        assert_eq!(restored.get_support(), model.get_support());
        assert_eq!(restored.get_location(), model.get_location());
    }

    #[test]
    fn test_from_serializable_rejects_wrong_algorithm() {
        let data = reference_data();
        let model = EmpiricalCovariance::new()
            .fit(&data.view(), &())
            .expect("fitting should succeed");
        let mut serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        serialized.algorithm = "SomethingElse".to_string();

        assert!(EmpiricalCovariance::from_serializable(&serialized).is_err());
    }

    #[test]
    fn test_from_serializable_rejects_missing_covariance() {
        let data = reference_data();
        let model = EmpiricalCovariance::new()
            .fit(&data.view(), &())
            .expect("fitting should succeed");
        let mut serialized = model
            .to_serializable()
            .expect("serialization should succeed");
        serialized.state.covariance = None;

        assert!(EmpiricalCovariance::from_serializable(&serialized).is_err());
    }
}
