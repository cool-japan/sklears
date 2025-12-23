# Makefile for sklears

.PHONY: test test-neural test-neural-all test-neural-heavy test-all build clean

# Default test (fast tests only)
test:
	cargo test --workspace --lib

# Test sklears-neural (fast tests only, ignores heavy tests)
test-neural:
	cargo test --package sklears-neural

# Test sklears-neural including heavy/ignored tests (single-threaded to avoid resource contention)
test-neural-all:
	RUST_TEST_THREADS=1 cargo test --package sklears-neural -- --include-ignored

# Test only heavy/ignored tests in sklears-neural
test-neural-heavy:
	RUST_TEST_THREADS=1 cargo test --package sklears-neural -- --ignored

# Test all packages
test-all:
	cargo test --workspace

# Build all packages
build:
	cargo build --workspace

# Build release
release:
	cargo build --workspace --release

# Clean build artifacts
clean:
	cargo clean

# Run specific package tests
test-%:
	cargo test --package sklears-$*

# Help
help:
	@echo "Available targets:"
	@echo "  test              - Run fast tests (default)"
	@echo "  test-neural       - Run sklears-neural fast tests"
	@echo "  test-neural-all   - Run all sklears-neural tests (includes heavy tests, single-threaded)"
	@echo "  test-neural-heavy - Run only heavy/ignored sklears-neural tests"
	@echo "  test-all          - Run all workspace tests"
	@echo "  build             - Build all packages"
	@echo "  release           - Build release"
	@echo "  clean             - Clean build artifacts"
