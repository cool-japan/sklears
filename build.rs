fn main() {
    // Use macOS Accelerate framework for BLAS/LAPACK and OpenMP
    println!("cargo:rustc-link-search=native=/opt/homebrew/opt/libomp/lib");
    println!("cargo:rustc-link-lib=omp");
    println!("cargo:rustc-link-lib=framework=Accelerate");
}