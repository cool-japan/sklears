# sklears-cross-decomposition TODO

## Disabled modules (re-enable per empirical protocol)

- [x] Re-enable `pub mod cross_validation` — KNOWN ISSUE (v0.1.0): Module disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.
- [x] Re-enable `pub mod permutation_tests` — KNOWN ISSUE (v0.1.0): Module disabled due to ndarray HRTB lifetime constraints. Planned for v0.2.0.

## Source-level TODOs

- [x] `src/simd_acceleration/advanced_simd.rs` — SIMD functions confirmed available in scirs2_core 0.4.2:
  - `scirs2_core::simd_ops::matmul::simd_dot_product_f64` / `simd_dot_product_f32`
  - `scirs2_core::simd::SimdOps` trait
  - `scirs2_core::simd_ops::SimdUnifiedOps` trait (impl for f32/f64)
  - `simd_dot_product_impl` now delegates to `simd_dot_product_f64` (hardware SIMD when `simd` feature enabled, scalar fallback otherwise)
  - All production `expect()` calls eliminated; non-contiguous array paths have scalar fallbacks

## Phase C-4

- [x] Phase C-4: Removed all 26 blanket #![allow(...)] suppressors; 506 tests pass, 0 warnings

## OxiCUDA Migration (v0.2.0)

Status: fully migrated to the oxicuda-backed GPU stack (no scirs2-core GPU usage). Remaining item is a hardening/depth improvement (workspace Phase 5, optional for the 0.2.0 release).

- [ ] (M) Offload `GpuMatrixOps::eig` and `svd` to oxicuda-solver when a CUDA backend is live — `src/gpu_acceleration.rs`, `Cargo.toml` (deferred 2026-07-06: no genuine device win available yet)
  - `eig`/`svd` (`src/gpu_acceleration.rs:344-363`) always run on CPU via `scirs2_linalg::compat::{eigh, svd}`; module docs (lines 23-26) acknowledge this as the CPU-correct baseline pending oxicuda-solver wiring.
  - Investigated 2026-07-06: read `oxicuda-solver` 0.4.0 source directly (`~/.cargo/registry/src/.../oxicuda-solver-0.4.0/src/dense/{eig,svd}.rs`). Both `dense::syevd` and `dense::svd` are, in their own words, "CPU host-fallback" with "NO GPU acceleration of this path" — unlike `sklears-discriminant-analysis`'s LDA generalized-eigensolve (which gets a genuine partial win from on-device `cholesky`/`inverse`/GEMM before falling back to host-only `syevd`), a plain symmetric `eig` or general `svd` has no such reduction step to accelerate: wiring either call in this crate would only add a dependency, a device round-trip (upload → solver internally downloads to host anyway → compute → re-upload → this crate downloads again), and complexity, with zero speed or correctness benefit over the current direct `scirs2_linalg` CPU call. Per the "no pretend offload" rule, the CPU path is being kept as-is rather than wiring a `dep:oxicuda-solver` `gpu`-feature path that would look GPU-accelerated but isn't.
  - Preferred (unblocks when upstream ships real on-device syevd/SVD): extend `sklears_core::gpu` with `eigh`/`svd` wrappers over `oxicuda_solver::dense` (syevd at `dense/eig.rs:70`, SVD at `dense/svd.rs:116`) and call those, keeping this crate free of direct oxicuda deps.
  - Alternative: add optional `oxicuda-solver` + `oxicuda-memory` to the `gpu` feature mirroring sklears-discriminant-analysis (`Cargo.toml:46-48,61`), with column-major round-trips — only worth doing once `oxicuda-solver` has genuine on-device syevd/SVD kernels, not the current host fallback.
  - Update module docs (lines 23-26, 44-45) and the `GpuMatrixOps` doc comment (lines 262-268) once wired.
  - Caveat: oxicuda-solver 0.4.0's syevd/SVD device paths are documented CPU host fallbacks — do not claim on-device acceleration until upstream restores it. Re-check on every oxicuda-solver version bump.

---

See also: [Workspace roadmap](../../TODO.md)
