//! Real system memory statistics.
//!
//! Reads actual memory information from the OS — never fabricates values.
//! On unsupported platforms returns `None` or `Err` rather than inventing numbers.

use crate::error::{Result, SklearsError};

/// System-wide memory statistics in bytes.
#[derive(Debug, Clone)]
pub struct SystemMemory {
    /// Total physical RAM in bytes.
    pub total: u64,
    /// Available (free + reclaimable) RAM in bytes.
    pub available: u64,
    /// Used RAM in bytes (`total - available`).
    pub used: u64,
}

/// Read real system-wide memory statistics.
///
/// # Platform support
/// - **Linux**: parses `/proc/meminfo` (MemTotal / MemAvailable lines).
/// - **Apple**: `sysctl hw.memsize` plus `host_statistics64`.
/// - **Windows**: uses `winapi::um::sysinfoapi::GlobalMemoryStatusEx`.
/// - **FreeBSD family**: `sysconf(_SC_PHYS_PAGES)` plus the `vm.stats.vm`
///   sysctls, which report free and inactive page counts.
/// - **Other Unix**: `libc::sysconf(_SC_PHYS_PAGES)` and `_SC_AVPHYS_PAGES`
///   where that (glibc/Solaris) extension exists.
///
/// Returns `Err` if the platform APIs are unavailable or parsing fails.
pub fn system_memory() -> Result<SystemMemory> {
    system_memory_impl()
}

/// Read this process's Resident Set Size in bytes.
///
/// Returns `None` on unsupported platforms (honest unknown, not zero).
pub fn process_rss_bytes() -> Option<u64> {
    process_rss_impl()
}

// ── Linux implementation ──────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn system_memory_impl() -> Result<SystemMemory> {
    let content = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| SklearsError::InvalidOperation(format!("cannot read /proc/meminfo: {}", e)))?;

    let mut mem_total: Option<u64> = None;
    let mut mem_available: Option<u64> = None;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            mem_total = Some(parse_kb_line(rest)?);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            mem_available = Some(parse_kb_line(rest)?);
        }
        if mem_total.is_some() && mem_available.is_some() {
            break;
        }
    }

    let total = mem_total.ok_or_else(|| {
        SklearsError::InvalidOperation("MemTotal not found in /proc/meminfo".to_string())
    })?;
    let available = mem_available.ok_or_else(|| {
        SklearsError::InvalidOperation("MemAvailable not found in /proc/meminfo".to_string())
    })?;
    let used = total.saturating_sub(available);

    Ok(SystemMemory {
        total,
        available,
        used,
    })
}

#[cfg(target_os = "linux")]
fn parse_kb_line(rest: &str) -> Result<u64> {
    // Format: "   <number> kB"
    let trimmed = rest.trim();
    let kb_str = trimmed
        .split_whitespace()
        .next()
        .ok_or_else(|| SklearsError::InvalidOperation("empty /proc/meminfo value".to_string()))?;
    let kb: u64 = kb_str.parse().map_err(|_| {
        SklearsError::InvalidOperation(format!("cannot parse /proc/meminfo value: {}", kb_str))
    })?;
    Ok(kb * 1024)
}

#[cfg(target_os = "linux")]
fn process_rss_impl() -> Option<u64> {
    // /proc/self/statm: fields space-separated, field index 1 = resident pages
    let content = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages: u64 = content.split_whitespace().nth(1)?.parse().ok()?;
    let page_size = page_size_bytes()?;
    Some(resident_pages * page_size)
}

// ── Non-Linux, non-Apple Unix implementation ─────────────────────────────────

#[cfg(all(
    target_family = "unix",
    not(target_os = "linux"),
    not(target_vendor = "apple")
))]
fn system_memory_impl() -> Result<SystemMemory> {
    let total = unix_sysconf_bytes(libc::_SC_PHYS_PAGES).ok_or_else(|| {
        SklearsError::InvalidOperation("sysconf(_SC_PHYS_PAGES) returned unavailable".to_string())
    })?;
    let available = available_memory_bytes().ok_or_else(|| {
        SklearsError::InvalidOperation(
            "available memory is not reported by this platform".to_string(),
        )
    })?;
    let used = total.saturating_sub(available);
    Ok(SystemMemory {
        total,
        available,
        used,
    })
}

/// Available physical memory in bytes.
///
/// `_SC_AVPHYS_PAGES` is a glibc/Solaris extension that the FreeBSD family,
/// NetBSD and the non-macOS Apple targets do not define, so reading it on
/// "every Unix that is not Linux or macOS" did not compile there. Each family
/// gets its own reader instead and the remaining targets report an honest
/// unknown. Keep the target list here and in the fallback below complementary.
#[cfg(any(
    target_os = "android",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "openbsd",
    target_os = "haiku",
    target_os = "aix",
    target_os = "hurd",
    target_os = "nto",
    target_os = "cygwin"
))]
fn available_memory_bytes() -> Option<u64> {
    unix_sysconf_bytes(libc::_SC_AVPHYS_PAGES)
}

/// FreeBSD family: free and inactive page counts from the `vm.stats.vm`
/// sysctls, mirroring the "free + inactive" definition used on Apple targets.
#[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
fn available_memory_bytes() -> Option<u64> {
    let page_size = page_size_bytes()?;
    let free = sysctl_u32(c"vm.stats.vm.v_free_count")?;
    let inactive = sysctl_u32(c"vm.stats.vm.v_inactive_count")?;
    Some((u64::from(free) + u64::from(inactive)) * page_size)
}

#[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
fn sysctl_u32(name: &std::ffi::CStr) -> Option<u32> {
    let mut value: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    // SAFETY: `name` is NUL terminated and the output buffer matches `len`
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u32 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 {
        Some(value)
    } else {
        None
    }
}

/// Unix targets with no known "available memory" interface (NetBSD, ...):
/// honest unknown rather than an invented number.
#[cfg(all(
    target_family = "unix",
    not(target_os = "linux"),
    not(target_vendor = "apple"),
    not(any(
        target_os = "android",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "openbsd",
        target_os = "haiku",
        target_os = "aix",
        target_os = "hurd",
        target_os = "nto",
        target_os = "cygwin",
        target_os = "freebsd",
        target_os = "dragonfly"
    ))
))]
fn available_memory_bytes() -> Option<u64> {
    None
}

#[cfg(all(
    target_family = "unix",
    not(target_os = "linux"),
    not(target_vendor = "apple")
))]
fn unix_sysconf_bytes(name: libc::c_int) -> Option<u64> {
    // SAFETY: sysconf is safe to call with standard constants
    let pages = unsafe { libc::sysconf(name) };
    if pages < 0 {
        return None;
    }
    let page_size = page_size_bytes()?;
    Some(pages as u64 * page_size)
}

#[cfg(all(
    target_family = "unix",
    not(target_os = "linux"),
    not(target_vendor = "apple")
))]
fn process_rss_impl() -> Option<u64> {
    // rusage.ru_maxrss — on BSDs (non-Apple, non-Linux) this is kilobytes.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: &mut usage is valid, RUSAGE_SELF is a valid constant
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret != 0 {
        return None;
    }
    let rss = usage.ru_maxrss as u64 * 1024; // kB on other BSDs
    Some(rss)
}

// ── Apple implementation (macOS, iOS and the other Darwin targets) ───────────

// Declare mach_host_self directly to avoid the libc deprecation warning;
// the libc crate marks it deprecated in favour of the `mach2` crate, but
// adding a new dependency for a single trap call is unnecessary.
#[cfg(target_vendor = "apple")]
extern "C" {
    fn mach_host_self() -> libc::mach_port_t;
}

#[cfg(target_vendor = "apple")]
fn system_memory_impl() -> Result<SystemMemory> {
    let total = apple_total_memory()
        .ok_or_else(|| SklearsError::InvalidOperation("sysctl hw.memsize failed".to_string()))?;
    let available = apple_available_memory()
        .ok_or_else(|| SklearsError::InvalidOperation("host_statistics64 failed".to_string()))?;
    let used = total.saturating_sub(available);
    Ok(SystemMemory {
        total,
        available,
        used,
    })
}

#[cfg(target_vendor = "apple")]
fn apple_total_memory() -> Option<u64> {
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: sysctlbyname with a well-known constant name; output pointer is valid
    let ret = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 {
        Some(value)
    } else {
        None
    }
}

#[cfg(target_vendor = "apple")]
fn apple_available_memory() -> Option<u64> {
    let page_size = page_size_bytes()?;
    let mut vm_stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count: libc::mach_msg_type_number_t = (std::mem::size_of::<libc::vm_statistics64>()
        / std::mem::size_of::<libc::integer_t>())
        as libc::mach_msg_type_number_t;
    // SAFETY: mach_host_self() is always valid; pointers are properly sized
    let ret = unsafe {
        libc::host_statistics64(
            mach_host_self(),
            libc::HOST_VM_INFO64 as libc::host_flavor_t,
            &mut vm_stats as *mut _ as *mut libc::integer_t,
            &mut count,
        )
    };
    if ret != libc::KERN_SUCCESS {
        return None;
    }
    let free_pages = vm_stats.free_count as u64 + vm_stats.inactive_count as u64;
    Some(free_pages * page_size)
}

#[cfg(target_vendor = "apple")]
fn process_rss_impl() -> Option<u64> {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: getrusage is safe with RUSAGE_SELF and a valid pointer
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret != 0 {
        return None;
    }
    // On Darwin, ru_maxrss is in bytes (unlike Linux where it's kilobytes)
    Some(usage.ru_maxrss as u64)
}

// ── Windows implementation ───────────────────────────────────────────────────

#[cfg(target_os = "windows")]
fn system_memory_impl() -> Result<SystemMemory> {
    use winapi::um::sysinfoapi::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut mem_status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    mem_status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;

    // SAFETY: mem_status is correctly initialised above
    let ok = unsafe { GlobalMemoryStatusEx(&mut mem_status) };
    if ok == 0 {
        return Err(SklearsError::InvalidOperation(
            "GlobalMemoryStatusEx failed".to_string(),
        ));
    }

    let total = mem_status.ullTotalPhys;
    let available = mem_status.ullAvailPhys;
    let used = total.saturating_sub(available);

    Ok(SystemMemory {
        total,
        available,
        used,
    })
}

#[cfg(target_os = "windows")]
fn process_rss_impl() -> Option<u64> {
    use winapi::um::processthreadsapi::GetCurrentProcess;
    use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};

    let mut pmc: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: pmc is zero-initialised, size is correct
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, size) };
    if ok == 0 {
        return None;
    }
    Some(pmc.WorkingSetSize as u64)
}

// ── Unsupported platforms ────────────────────────────────────────────────────

#[cfg(not(any(target_family = "unix", target_os = "windows")))]
fn system_memory_impl() -> Result<SystemMemory> {
    Err(SklearsError::NotImplemented(
        "system_memory() is not implemented on this platform".to_string(),
    ))
}

#[cfg(not(any(target_family = "unix", target_os = "windows")))]
fn process_rss_impl() -> Option<u64> {
    None
}

// ── Shared helpers ──────────────────────────────────────────────────────────

/// Returns the OS page size in bytes, or `None` if unavailable.
#[cfg(target_family = "unix")]
fn page_size_bytes() -> Option<u64> {
    // SAFETY: _SC_PAGESIZE is a valid sysconf constant
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps <= 0 {
        None
    } else {
        Some(ps as u64)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for issue #5.
    ///
    /// `system_memory()` and `process_rss_bytes()` are compiled on every
    /// target, so a platform arm that is missing (or that reads an API the
    /// platform does not provide, as `_SC_AVPHYS_PAGES` on the BSDs) breaks
    /// the build of that platform. Running the API everywhere keeps the
    /// `cfg` arms of this module exhaustive and mutually exclusive.
    #[test]
    fn test_issue_5_system_memory_gates_resolve() {
        match system_memory() {
            Ok(memory) => {
                assert!(memory.total > 0, "total memory must be positive");
                assert!(
                    memory.available <= memory.total,
                    "available ({}) must not exceed total ({})",
                    memory.available,
                    memory.total
                );
                assert_eq!(memory.used, memory.total.saturating_sub(memory.available));
            }
            Err(err) => {
                // Platforms without a memory interface report an honest error
                // instead of inventing numbers.
                assert!(!err.to_string().is_empty());
            }
        }
    }

    /// Companion to the test above for the RSS reader: `None` is an accepted
    /// answer, a fabricated zero is not.
    #[test]
    fn test_issue_5_process_rss_gates_resolve() {
        if let Some(rss) = process_rss_bytes() {
            assert!(rss > 0, "a running process cannot have a zero RSS");
        }
    }
}
