#!/bin/bash
# Helper script to suppress dyld warnings when running cargo commands
# Usage: ./scripts/quiet_cargo.sh build
#        ./scripts/quiet_cargo.sh test
#        ./scripts/quiet_cargo.sh bench

# Suppress dyld warnings for this session
export DYLD_PRINT_WARNINGS=0
export DYLD_LIBRARY_PATH=""
export DYLD_FALLBACK_LIBRARY_PATH=""

# Run cargo with the provided arguments
cargo "$@" 2>&1 | grep -v "dyld\[.*\]: symbol.*missing from root"