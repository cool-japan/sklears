//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::api_reference_generator::TraitInfo;
use crate::error::{Result, SklearsError};
use scirs2_core::ndarray::{Array, Array1, Array2, Axis};
use scirs2_core::ndarray_ext::{manipulation, matrix, stats};
use scirs2_core::random::{thread_rng, Random};
use scirs2_core::constants::physical;
use scirs2_core::error::CoreError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use super::types::*;
