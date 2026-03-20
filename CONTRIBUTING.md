# Contributing to sklears

Thank you for your interest in contributing to sklears! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. We expect all contributors to:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Rust 1.70 or later
- Git
- Familiarity with machine learning concepts (helpful but not required)

### Finding Issues

Good places to start:

- Issues labeled `good-first-issue` - ideal for newcomers
- Issues labeled `help-wanted` - where we need community help
- Issues labeled `documentation` - improve our docs
- Issues labeled `bug` - fix bugs in existing code

## Development Setup

1. **Fork and Clone the Repository**

   ```bash
   git clone https://github.com/cool-japan/sklears.git
   cd sklears
   ```

2. **Install Development Tools**

   ```bash
   rustup component add rustfmt clippy
   cargo install cargo-nextest  # Optional but recommended
   ```

3. **Build the Project**

   ```bash
   cargo build --all-features
   ```

4. **Run Tests**

   ```bash
   cargo test --all-features
   # Or using nextest (faster)
   cargo nextest run --all-features
   ```

5. **Check Code Quality**

   ```bash
   cargo fmt --check
   cargo clippy --all-targets --all-features -- -D warnings
   ```

## Making Contributions

### Types of Contributions

We welcome many types of contributions:

1. **Bug Fixes** - Fix issues in existing code
2. **New Features** - Implement new algorithms or capabilities
3. **Documentation** - Improve docs, add examples, fix typos
4. **Tests** - Add test coverage, property-based tests
5. **Benchmarks** - Add performance benchmarks
6. **Examples** - Create examples demonstrating features
7. **Refactoring** - Improve code quality and maintainability

### Before You Start

1. **Check Existing Issues** - Search for existing issues/PRs related to your contribution
2. **Create an Issue** - For significant changes, create an issue first to discuss the approach
3. **Get Feedback** - Wait for maintainer feedback before starting large changes

## Coding Standards

### Rust Style Guide

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` to format all code
- Address all `cargo clippy` warnings
- Use meaningful variable and function names
- Keep functions focused and reasonably sized (< 100 lines when possible)

### SciRS2 Policy

**CRITICAL:** sklears uses SciRS2 as its scientific computing foundation. Follow these rules:

1. **Array Operations:**
   ```rust
   // ✅ CORRECT - Use SciRS2
   use scirs2_core::ndarray::{Array1, Array2};

   // ❌ WRONG - Direct ndarray usage
   use ndarray::{Array1, Array2};
   ```

2. **Random Number Generation:**
   ```rust
   // ✅ CORRECT - Use SciRS2 random
   use scirs2_core::random::{Normal, Uniform};
   use scirs2_core::random::thread_rng;

   // ❌ WRONG - Direct rand usage
   use rand::{Rng, thread_rng};
   ```

3. **See SCIRS2_INTEGRATION_POLICY.md** for complete details

### Code Organization

- Place implementation code in appropriately named modules
- Keep public API surface minimal and well-documented
- Use feature flags for optional functionality
- Follow the workspace policy for dependencies

### Naming Conventions

- **snake_case** for functions, methods, variables
- **CamelCase** for types, traits, enums
- **SCREAMING_SNAKE_CASE** for constants
- Use descriptive names that convey intent

### Error Handling

- Use `Result<T, E>` for fallible operations
- Provide context in error messages
- Use `thiserror` for error types
- Document error conditions in function docs

## Testing

### Test Requirements

All contributions must include appropriate tests:

1. **Unit Tests** - Test individual functions/methods
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_feature_works() {
           // Test code here
       }
   }
   ```

2. **Integration Tests** - Test cross-crate interactions
   - Place in `tests/` directory

3. **Property-Based Tests** - Use `proptest` for invariants
   ```rust
   use proptest::prelude::*;

   proptest! {
       #[test]
       fn property_holds(x in any::<f64>()) {
           // Property test
       }
   }
   ```

4. **Documentation Tests** - Add runnable examples in docs
   ```rust
   /// # Example
   /// ```
   /// use sklears::linear::LinearRegression;
   /// let model = LinearRegression::new();
   /// ```
   ```

### Running Tests

```bash
# All tests
cargo test --all-features

# Specific crate
cargo test -p sklears-linear

# With nextest (parallel execution)
cargo nextest run --all-features

# Documentation tests
cargo test --doc
```

### Test Coverage

- Aim for >80% line coverage for new code
- Test both success and error paths
- Include edge cases and boundary conditions
- Use temporary files from `std::env::temp_dir()` in tests

## Documentation

### Documentation Requirements

1. **Public API** - All public items must have documentation
   ```rust
   /// Trains a linear regression model on the given data.
   ///
   /// # Arguments
   ///
   /// * `x` - Feature matrix of shape (n_samples, n_features)
   /// * `y` - Target vector of shape (n_samples,)
   ///
   /// # Returns
   ///
   /// A trained model ready for prediction
   ///
   /// # Errors
   ///
   /// Returns an error if shapes are incompatible
   ///
   /// # Example
   ///
   /// ```
   /// use sklears::linear::LinearRegression;
   /// use scirs2_core::ndarray::Array2;
   ///
   /// let model = LinearRegression::new();
   /// // ... fit and predict
   /// ```
   pub fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Trained> {
       // ...
   }
   ```

2. **Module Documentation** - Add module-level docs in lib.rs
3. **Examples** - Include at least one example for each major feature
4. **README** - Update README if adding user-facing features

### Documentation Style

- Use complete sentences with proper punctuation
- Explain **why** not just **what**
- Include complexity/performance notes where relevant
- Link to related functions/types with backticks
- Follow [RFC 1574](https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md)

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### Examples

```
feat(linear): add ElasticNet regression

Implements ElasticNet regression with L1 and L2 regularization.
Uses coordinate descent optimization from scirs2-optimize.

Closes #123
```

```
fix(metrics): correct F1 score calculation for multiclass

The F1 score was using macro averaging when micro was specified.
Fixed by properly handling the averaging parameter.

Fixes #456
```

## Pull Request Process

### Before Submitting

1. **Ensure All Tests Pass**
   ```bash
   cargo test --all-features
   cargo clippy --all-targets --all-features -- -D warnings
   cargo fmt --check
   ```

2. **Update Documentation**
   - Update README if needed
   - Add/update doc comments
   - Update CHANGELOG.md

3. **Run Benchmarks** (if performance-related)
   ```bash
   cargo bench
   ```

### Submitting the PR

1. **Create a Branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make Your Changes**
   - Follow coding standards
   - Add tests
   - Update documentation

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

4. **Push to Your Fork**
   ```bash
   git push origin feature/my-new-feature
   ```

5. **Open a Pull Request**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe what changed and why
   - Include test results
   - Add screenshots/examples if applicable

### PR Review Process

1. **Automated Checks** - CI will run tests and lints
2. **Code Review** - Maintainers will review your code
3. **Feedback** - Address any requested changes
4. **Approval** - Once approved, maintainers will merge

### After Merging

- Delete your feature branch
- Update your local main branch
- Check the CHANGELOG for credit

## Additional Guidelines

### Performance Considerations

- Profile before optimizing
- Use `cargo bench` for benchmarks
- Consider SIMD operations for numerical code
- Use `#[inline]` judiciously
- Prefer iteration over indexing

### Memory Safety

- Avoid `unsafe` unless absolutely necessary
- Document all `unsafe` blocks with safety invariants
- Use clippy's unsafe lints

### Feature Flags

- Make expensive features optional
- Document feature flags in Cargo.toml
- Use feature flags for platform-specific code

### Breaking Changes

- Avoid breaking changes when possible
- Clearly document breaking changes
- Discuss breaking changes in issues first
- Follow semver guidelines

## Getting Help

- **Documentation:** https://docs.rs/sklears
- **Issues:** https://github.com/cool-japan/sklears/issues
- **Discussions:** https://github.com/cool-japan/sklears/discussions

## Recognition

Contributors are recognized in:
- CHANGELOG.md
- GitHub Contributors page
- Release notes

Thank you for contributing to sklears! 🦀
