//! GPU memory-pool telemetry and (when the `gpu` feature is enabled) a real
//! pooled allocator built on `oxicuda-memory`'s stream-ordered `MemoryPool`.
//!
//! [`PoolTelemetry`] is pure, hardware-independent bookkeeping: a handful of
//! atomic counters and two derived ratios (`hit_rate`, `used_fraction`). It
//! has no CUDA/driver dependency and is unit-tested unconditionally (see the
//! `tests` module below), unlike everything else in this crate's `gpu`
//! module, which needs a real device to exercise meaningfully.
//!
//! [`GpuMemoryPool`] (feature = "gpu") is the real allocator: it wraps
//! `oxicuda_memory::MemoryPool` (itself a stream-ordered pool over
//! `cuMemAlloc_v2`/`cuMemFree_v2`) with an application-level free-list of
//! [`PooledHandle`]s so that [`PoolTelemetry::record_hit`]/`record_miss`
//! reflect *this* pool's own reuse decisions precisely: a hit means "we handed
//! back a buffer we were already holding, with no new device allocation call
//! at all," not an estimate or a guess about what the underlying driver did
//! internally.
//!
//! # Why a handle instead of returning `PooledBuffer<T>` directly
//!
//! `oxicuda_memory::PooledBuffer<T>::drop` unconditionally returns its
//! allocation to the *underlying* `MemoryPool`'s own internal free-bins
//! (gated on a stream-ordered recycle event) -- it has no hook for a caller
//! to intercept. That underlying reuse is real, but it is invisible from the
//! outside: nothing in `oxicuda-memory`'s public API reports whether a given
//! `alloc_async` call was actually served from a free bin or required a fresh
//! `cuMemAlloc_v2`. Fabricating a hit/miss split from that opaque signal
//! would violate the "no fabricated telemetry" goal this module exists to
//! fix. [`PooledHandle`] instead keeps the `PooledBuffer<T>` alive in *this*
//! pool's own free-list on drop (see its `Drop` impl), so a hit is something
//! this module can prove: "no `alloc_async` call happened for this
//! acquisition."

// This whole module is gated behind the `gpu` feature at its `mod gpu_pool;`
// declaration in `lib.rs` (there is no pooled-allocator concept at all
// without real GPU support compiled in), so nothing in this file needs its
// own repeated `#[cfg(feature = "gpu")]`.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use oxicuda_memory::{MemoryPool, PooledBuffer};

use crate::NeuralResult;
use sklears_core::error::SklearsError;

/// Pure-logic, hardware-independent GPU memory-pool telemetry.
///
/// Tracks real hit/miss/allocation counts reported by [`GpuMemoryPool`] (when
/// the `gpu` feature is enabled) and derives `hit_rate`/`used_fraction` from
/// them. Every method here is plain atomic-counter arithmetic with no CUDA
/// dependency, so its behaviour is fully covered by the unconditional
/// `tests` module below -- it never needs a GPU to verify.
#[derive(Debug)]
pub struct PoolTelemetry {
    hits: AtomicU64,
    misses: AtomicU64,
    allocated_bytes: AtomicU64,
    capacity_bytes: u64,
}

impl PoolTelemetry {
    /// Creates telemetry for a pool with the given capacity ceiling in bytes.
    ///
    /// `capacity_bytes` is used only to compute [`used_fraction`](Self::used_fraction);
    /// it does not itself reserve or limit anything.
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            allocated_bytes: AtomicU64::new(0),
            capacity_bytes: capacity_bytes as u64,
        }
    }

    /// Records that an acquisition was served by handing back a buffer the
    /// pool was already holding -- no new device allocation was made. This is
    /// a genuine reuse, never an estimate (see the module docs).
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records that an acquisition required allocating `bytes` fresh from the
    /// device because no reusable buffer was on hand.
    pub fn record_miss(&self, bytes: usize) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.allocated_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Records that `bytes` were released back to the device rather than
    /// retained in the pool's own free-list for a future hit.
    pub fn record_free(&self, bytes: usize) {
        // `fetch_update` (rather than `fetch_sub`) so a free that reports
        // more bytes than are currently tracked saturates at zero instead of
        // wrapping around; `AtomicU64` has no built-in saturating-subtract.
        let _ = self
            .allocated_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(bytes as u64))
            });
    }

    /// Fraction of recorded acquisitions served from the free-list without a
    /// fresh device allocation, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` (never `NaN`) when no acquisition has been recorded yet.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total == 0.0 { 0.0 } else { hits / total }
    }

    /// Currently-allocated bytes as a fraction of `capacity_bytes`.
    ///
    /// Clamped to `[0.0, 1.0]`: a pool that has genuinely outgrown its
    /// configured ceiling reports full (`1.0`) utilization rather than a
    /// fraction above one. Returns `0.0` when `capacity_bytes` is `0` (there
    /// is nothing to divide by; this is a deliberate degenerate case, not a
    /// `NaN`/`inf` bug).
    pub fn used_fraction(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        let used = self.allocated_bytes.load(Ordering::Relaxed) as f64;
        (used / self.capacity_bytes as f64).clamp(0.0, 1.0)
    }
}

/// Free-list buckets, keyed by `(element type, size class)`, each holding
/// zero or more previously-released, type-erased `Box<PooledBuffer<T>>`
/// values ready to be downcast back to `T` and handed out as a hit. Factored
/// out into its own alias only to keep [`GpuMemoryPool`]'s field declaration
/// readable (`clippy::type_complexity`); it carries no behavior of its own.
type PoolFreeLists = Mutex<HashMap<(TypeId, usize), Vec<Box<dyn Any>>>>;

/// A real pooled GPU allocator backed by `oxicuda_memory::MemoryPool`.
///
/// Layers an application-level, size-classed free-list of [`PooledHandle`]s
/// on top of the underlying stream-ordered pool so that hit/miss telemetry
/// (via [`PoolTelemetry`]) reflects real reuse decisions this type makes
/// itself, rather than the underlying pool's own (unobservable from here)
/// internal recycling. See the module docs for why this wraps rather than
/// returning `oxicuda_memory::PooledBuffer<T>` directly.
pub struct GpuMemoryPool {
    inner_pool: MemoryPool,
    free_lists: PoolFreeLists,
    telemetry: PoolTelemetry,
}

impl GpuMemoryPool {
    /// Maximum number of released buffers kept on hand per `(type, size
    /// class)` bucket before a further release is freed back to the device
    /// for real. Bounds the free-list's memory footprint; a handful is
    /// enough to smooth out typical acquire/release churn without the pool
    /// growing without limit.
    const MAX_CACHED_PER_CLASS: usize = 4;

    /// Creates a pool bound to `device_ordinal`, with `capacity_bytes` used
    /// only as the ceiling [`PoolTelemetry::used_fraction`] measures against.
    pub fn new(device_ordinal: i32, capacity_bytes: usize) -> NeuralResult<Self> {
        let inner_pool = MemoryPool::new(device_ordinal).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to create GPU memory pool: {}", e))
        })?;
        Ok(Self {
            inner_pool,
            free_lists: Mutex::new(HashMap::new()),
            telemetry: PoolTelemetry::new(capacity_bytes),
        })
    }

    /// Acquires a buffer of `n` elements of `T`, ordered against `stream`.
    ///
    /// Checks this pool's own free-list for a previously-released buffer of
    /// the same element type and size class first (a real, zero-allocation
    /// reuse -- [`PoolTelemetry::record_hit`]); if none is available, performs
    /// a real `oxicuda_memory::PooledBuffer::alloc_async` and records a miss.
    /// Requests are rounded up to the next power of two elements so a
    /// released buffer can serve any later request of the same or smaller
    /// size within its class, not only an exact-length match.
    pub fn acquire<T: Copy + 'static>(
        &self,
        n: usize,
        stream: &oxicuda_driver::Stream,
    ) -> NeuralResult<PooledHandle<'_, T>> {
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "cannot acquire a zero-element pooled GPU buffer".to_string(),
            ));
        }
        let size_class = n.checked_next_power_of_two().unwrap_or(n);
        let key = (TypeId::of::<T>(), size_class);

        let reused = {
            let mut lists = self
                .free_lists
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            lists.get_mut(&key).and_then(|bucket| bucket.pop())
        };

        if let Some(boxed) = reused {
            match boxed.downcast::<PooledBuffer<T>>() {
                Ok(buf) => {
                    self.telemetry.record_hit();
                    return Ok(PooledHandle {
                        buf: Some(*buf),
                        pool: self,
                        size_class,
                    });
                }
                Err(_mismatched) => {
                    // Unreachable in practice: `key` already encodes
                    // `TypeId::of::<T>()`, so any entry stored under it is a
                    // `PooledBuffer<T>`. If this ever trips regardless, drop
                    // the mismatched box here (freeing it back to the device
                    // for real via its own `Drop`) rather than risk a panic,
                    // and fall through to a fresh allocation below.
                }
            }
        }

        let bytes = size_class
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| {
                SklearsError::InvalidInput("pooled allocation size overflow".to_string())
            })?;
        let buf = PooledBuffer::<T>::alloc_async(&self.inner_pool, size_class, stream)
            .map_err(|e| {
                SklearsError::InvalidInput(format!("GPU pooled allocation failed: {}", e))
            })?;
        self.telemetry.record_miss(bytes);
        Ok(PooledHandle {
            buf: Some(buf),
            pool: self,
            size_class,
        })
    }

    /// Returns `buf` to this pool's free-list for `size_class` if there is
    /// room under [`MAX_CACHED_PER_CLASS`](Self::MAX_CACHED_PER_CLASS),
    /// otherwise frees it back to the device for real and records the free.
    fn release<T: Copy + 'static>(&self, buf: PooledBuffer<T>, size_class: usize) {
        let key = (TypeId::of::<T>(), size_class);
        let mut lists = self
            .free_lists
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let bucket = lists.entry(key).or_default();
        if bucket.len() < Self::MAX_CACHED_PER_CLASS {
            bucket.push(Box::new(buf));
        } else {
            drop(lists); // release the lock before the real (slower) free
            let bytes = size_class.saturating_mul(std::mem::size_of::<T>());
            self.telemetry.record_free(bytes);
            drop(buf);
        }
    }

    /// Real `(used_fraction, hit_rate)` telemetry for this pool; see
    /// [`PoolTelemetry::used_fraction`]/[`PoolTelemetry::hit_rate`].
    pub fn telemetry_stats(&self) -> (f64, f64) {
        (self.telemetry.used_fraction(), self.telemetry.hit_rate())
    }
}

/// RAII handle to a buffer acquired via [`GpuMemoryPool::acquire`].
///
/// Wraps a real `oxicuda_memory::PooledBuffer<T>`. Unlike a bare
/// `PooledBuffer` (whose `Drop` unconditionally frees back to the underlying
/// `oxicuda_memory::MemoryPool`), dropping a `PooledHandle` returns the
/// buffer to *this* [`GpuMemoryPool`]'s own free-list (see its `Drop` impl
/// below), making it available to a later `acquire` call as a genuine,
/// zero-allocation hit.
pub struct PooledHandle<'a, T: Copy + 'static> {
    buf: Option<PooledBuffer<T>>,
    pool: &'a GpuMemoryPool,
    size_class: usize,
}

impl<T: Copy + 'static> PooledHandle<'_, T> {
    /// Number of `T` elements available. This is the rounded-up size class,
    /// which is always `>=` the `n` originally requested from
    /// [`GpuMemoryPool::acquire`].
    pub fn len(&self) -> usize {
        self.buf.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    /// `true` if this handle holds zero elements. Only possible transiently
    /// during `Drop`; never observable through a live handle returned by
    /// `acquire` (which always rejects `n == 0` up front).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total allocation size in bytes.
    pub fn byte_size(&self) -> usize {
        self.buf.as_ref().map(|b| b.byte_size()).unwrap_or(0)
    }

    /// Raw device pointer, for feeding into a kernel launch or BLAS call.
    pub fn as_device_ptr(&self) -> oxicuda_driver::CUdeviceptr {
        self.buf.as_ref().map(|b| b.as_device_ptr()).unwrap_or(0)
    }
}

impl<T: Copy + 'static> Drop for PooledHandle<'_, T> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.release(buf, self.size_class);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- `PoolTelemetry`: pure atomic-counter logic, needs no CUDA/GPU
    // hardware to exercise (only the `gpu` Cargo feature to compile, via this
    // module's `#[cfg(feature = "gpu")] mod gpu_pool;` declaration in
    // `lib.rs`). --

    #[test]
    fn hit_rate_matches_recorded_counts() {
        let telemetry = PoolTelemetry::new(1024);
        for _ in 0..3 {
            telemetry.record_hit();
        }
        telemetry.record_miss(64);
        assert!((telemetry.hit_rate() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn hit_rate_is_zero_not_nan_when_untouched() {
        let telemetry = PoolTelemetry::new(1024);
        assert_eq!(telemetry.hit_rate(), 0.0);
        assert!(!telemetry.hit_rate().is_nan());
    }

    #[test]
    fn used_fraction_is_zero_when_nothing_allocated() {
        let telemetry = PoolTelemetry::new(1024);
        assert_eq!(telemetry.used_fraction(), 0.0);
    }

    #[test]
    fn used_fraction_reflects_partial_allocation() {
        let telemetry = PoolTelemetry::new(1000);
        telemetry.record_miss(250);
        assert!((telemetry.used_fraction() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn used_fraction_is_clamped_when_allocation_exceeds_capacity() {
        let telemetry = PoolTelemetry::new(100);
        telemetry.record_miss(150);
        assert_eq!(telemetry.used_fraction(), 1.0);
    }

    #[test]
    fn used_fraction_with_zero_capacity_never_divides_by_zero() {
        let telemetry = PoolTelemetry::new(0);
        telemetry.record_miss(10);
        assert_eq!(telemetry.used_fraction(), 0.0);
        assert!(!telemetry.used_fraction().is_nan());
    }

    #[test]
    fn record_free_reduces_allocated_bytes() {
        let telemetry = PoolTelemetry::new(1000);
        telemetry.record_miss(400);
        assert!((telemetry.used_fraction() - 0.4).abs() < 1e-12);
        telemetry.record_free(400);
        assert_eq!(telemetry.used_fraction(), 0.0);
    }

    #[test]
    fn record_free_saturates_instead_of_underflowing() {
        let telemetry = PoolTelemetry::new(1000);
        telemetry.record_miss(100);
        telemetry.record_free(500); // frees more than was ever allocated
        assert_eq!(telemetry.used_fraction(), 0.0);
    }

    // -- `GpuMemoryPool`/`PooledHandle`: real pool/stream construction and
    // acquire/release bookkeeping. Compiles unconditionally in this module
    // (again, gated only by the `gpu` feature already), but every test below
    // gracefully skips on a host with no actual CUDA driver/device, matching
    // the skip idiom already used throughout `gpu.rs`'s own tests. --

    use std::sync::Arc;

    /// Bootstraps a real CUDA context on device 0 if a driver and device are
    /// actually present. Returns `None` so every test below can skip
    /// gracefully on a no-GPU host (this crate's own dev/CI machines
    /// included) rather than failing -- mirrors the skip idiom already used
    /// throughout `gpu.rs`'s own `#[cfg(feature = "gpu")]` tests.
    fn real_context() -> Option<Arc<oxicuda_driver::Context>> {
        if oxicuda_driver::init().is_err() {
            return None;
        }
        let device = oxicuda_driver::Device::get(0).ok()?;
        oxicuda_driver::Context::new(&device).ok().map(Arc::new)
    }

    #[test]
    fn acquire_after_release_is_a_genuine_hit_not_a_fresh_allocation() {
        let Some(ctx) = real_context() else {
            return;
        };
        let Ok(stream) = oxicuda_driver::Stream::new(&ctx) else {
            return;
        };
        let Ok(pool) = GpuMemoryPool::new(0, 1_048_576) else {
            return;
        };

        {
            let handle = match pool.acquire::<f32>(16, &stream) {
                Ok(handle) => handle,
                Err(_) => return, // real device present but allocation failed; skip
            };
            assert_eq!(handle.len(), 16);
        } // `handle` drops here -> returned to `pool`'s own free-list.

        let (_, hit_rate_after_miss) = pool.telemetry_stats();
        assert_eq!(hit_rate_after_miss, 0.0); // exactly one miss so far, zero hits

        let Ok(_second) = pool.acquire::<f32>(16, &stream) else {
            return;
        };
        let (_, hit_rate_after_hit) = pool.telemetry_stats();
        assert_eq!(hit_rate_after_hit, 0.5); // one hit, one miss
    }
}
