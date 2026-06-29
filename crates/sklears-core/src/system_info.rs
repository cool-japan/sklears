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
/// - **Other Unix**: uses `libc::sysconf(_SC_PHYS_PAGES)` and `_SC_AVPHYS_PAGES`.
/// - **Windows**: uses `winapi::um::sysinfoapi::GlobalMemoryStatusEx`.
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

// ── Non-Linux Unix implementation ─────────────────────────────────────────────

#[cfg(all(
    target_family = "unix",
    not(target_os = "linux"),
    not(target_os = "macos")
))]
fn system_memory_impl() -> Result<SystemMemory> {
    let total = unix_sysconf_bytes(libc::_SC_PHYS_PAGES).ok_or_else(|| {
        SklearsError::InvalidOperation("sysconf(_SC_PHYS_PAGES) returned unavailable".to_string())
    })?;
    let available = unix_sysconf_bytes(libc::_SC_AVPHYS_PAGES).ok_or_else(|| {
        SklearsError::InvalidOperation("sysconf(_SC_AVPHYS_PAGES) returned unavailable".to_string())
    })?;
    let used = total.saturating_sub(available);
    Ok(SystemMemory {
        total,
        available,
        used,
    })
}

#[cfg(all(
    target_family = "unix",
    not(target_os = "linux"),
    not(target_os = "macos")
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
    not(target_os = "macos")
))]
fn process_rss_impl() -> Option<u64> {
    // rusage.ru_maxrss — on BSDs (non-macOS, non-Linux) this is kilobytes.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: &mut usage is valid, RUSAGE_SELF is a valid constant
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret != 0 {
        return None;
    }
    let rss = usage.ru_maxrss as u64 * 1024; // kB on other BSDs
    Some(rss)
}

// ── macOS implementation ──────────────────────────────────────────────────────

// Declare mach_host_self directly to avoid the libc deprecation warning;
// the libc crate marks it deprecated in favour of the `mach2` crate, but
// adding a new dependency for a single trap call is unnecessary.
#[cfg(target_os = "macos")]
extern "C" {
    fn mach_host_self() -> libc::mach_port_t;
}

#[cfg(target_os = "macos")]
fn system_memory_impl() -> Result<SystemMemory> {
    let total = macos_total_memory().ok_or_else(|| {
        SklearsError::InvalidOperation("sysctl hw.memsize failed on macOS".to_string())
    })?;
    let available = macos_available_memory().ok_or_else(|| {
        SklearsError::InvalidOperation("host_statistics64 failed on macOS".to_string())
    })?;
    let used = total.saturating_sub(available);
    Ok(SystemMemory {
        total,
        available,
        used,
    })
}

#[cfg(target_os = "macos")]
fn macos_total_memory() -> Option<u64> {
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

#[cfg(target_os = "macos")]
fn macos_available_memory() -> Option<u64> {
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

#[cfg(target_os = "macos")]
fn process_rss_impl() -> Option<u64> {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: getrusage is safe with RUSAGE_SELF and a valid pointer
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret != 0 {
        return None;
    }
    // On macOS, ru_maxrss is in bytes (unlike Linux where it's kilobytes)
    Some(usage.ru_maxrss as u64)
}

// ── Windows implementation ───────────────────────────────────────────────────

#[cfg(target_os = "windows")]
fn system_memory_impl() -> Result<SystemMemory> {
    use winapi::um::processthreadsapi::GetCurrentProcess;
    use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
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
