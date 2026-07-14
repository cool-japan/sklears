# TODO - v0.2.1

## Current Status
This crate is part of the sklears v0.2.1 release line (initially shipped in v0.1.0).
No GPU/OxiCUDA code paths exist in this crate (graph algorithms are CPU-parallelized
via Rayon and SIMD-accelerated instead), so it required no changes for the
workspace-wide 0.2.0 GPU-honesty pass.

## Future Enhancements
- Performance optimizations
- Additional scikit-learn API coverage
- Enhanced documentation and examples

See the main [workspace TODO](../../TODO.md) for overall project roadmap.
