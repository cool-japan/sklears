fn main() {
    // For macOS: allow undefined symbols (they'll be provided by Python interpreter at runtime)
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
