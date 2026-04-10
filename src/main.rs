use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const QUEUE_DIR: &str = "/tmp/revolver-queue";
const LOCK_DIR: &str = "/tmp/revolver-lock";
const LOCK_TTL_SECS: u64 = 10;
const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

// --- terminal width via ioctl ---

fn term_width() -> usize {
    #[cfg(unix)]
    {
        use std::mem::zeroed;
        unsafe {
            let mut ws: libc_winsize = zeroed();
            if libc_ioctl(2, TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
                return ws.ws_col as usize;
            }
        }
    }
    80
}

#[cfg(unix)]
#[repr(C)]
struct libc_winsize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
}

#[cfg(unix)]
const TIOCGWINSZ: u64 = 0x5413;

#[cfg(unix)]
unsafe fn libc_ioctl(fd: i32, request: u64, arg: *mut libc_winsize) -> i32 {
    extern "C" {
        fn ioctl(fd: i32, request: u64, ...) -> i32;
    }
    ioctl(fd, request, arg)
}

// --- queue management ---

fn queue_join(requested: usize) -> PathBuf {
    let _ = fs::create_dir_all(QUEUE_DIR);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid = process::id();
    let path = Path::new(QUEUE_DIR).join(format!("{}.{}", nanos, pid));
    fs::write(&path, requested.to_string()).expect("failed to create queue file");
    path
}

fn queue_leave(path: &Path) {
    let _ = fs::remove_file(path);
}

fn pid_alive(pid: u32) -> bool {
    unsafe {
        extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }
        kill(pid as i32, 0) == 0
    }
}

fn queue_position(my_file: &Path) -> usize {
    let my_name = my_file.file_name().unwrap().to_string_lossy().to_string();
    let mut entries: Vec<String> = Vec::new();

    if let Ok(dir) = fs::read_dir(QUEUE_DIR) {
        for entry in dir.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            // Clean up stale entries
            if let Some(pid_str) = name.split('.').nth(1) {
                if let Ok(pid) = pid_str.parse::<u32>() {
                    if !pid_alive(pid) {
                        let _ = fs::remove_file(entry.path());
                        continue;
                    }
                }
            }
            entries.push(name);
        }
    }

    entries.sort();
    entries.iter().position(|e| e == &my_name).unwrap_or(0)
}

// --- GPU locking ---

fn lock_gpus(indices: &[u32]) {
    let _ = fs::create_dir_all(LOCK_DIR);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    for &idx in indices {
        let path = Path::new(LOCK_DIR).join(idx.to_string());
        fs::write(&path, nanos.to_string()).expect("failed to write GPU lock file");
    }
}

fn is_gpu_locked(idx: u32) -> bool {
    let path = Path::new(LOCK_DIR).join(idx.to_string());
    match fs::read_to_string(&path) {
        Ok(contents) => {
            if let Ok(lock_nanos) = contents.trim().parse::<u128>() {
                let now_nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let age_secs = (now_nanos.saturating_sub(lock_nanos)) / 1_000_000_000;
                if age_secs < LOCK_TTL_SECS as u128 {
                    return true;
                }
                // Stale lock, clean up
                let _ = fs::remove_file(&path);
            }
            false
        }
        Err(_) => false,
    }
}

// --- GPU queries ---

fn query_gpu_indices() -> Vec<(u32, String)> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=index,gpu_bus_id", "--format=csv,noheader"])
        .output()
        .expect("failed to run nvidia-smi");
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(2, ", ").collect();
            if parts.len() == 2 {
                let idx = parts[0].trim().parse::<u32>().ok()?;
                let bus_id = parts[1].trim().to_string();
                Some((idx, bus_id))
            } else {
                None
            }
        })
        .collect()
}

fn busy_bus_ids() -> HashSet<String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=gpu_bus_id",
            "--format=csv,noheader",
        ])
        .output()
        .expect("failed to run nvidia-smi");
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().map(|l| l.trim().to_string()).collect()
}

fn find_free_gpus(n: usize) -> Result<Vec<u32>, (usize, usize)> {
    let gpus = query_gpu_indices();
    let busy = busy_bus_ids();
    let total = gpus.len();

    let mut free = Vec::new();

    for (idx, bus_id) in &gpus {
        if busy.contains(bus_id) || is_gpu_locked(*idx) {
            // busy
        } else {
            free.push(*idx);
        }
    }

    free.sort();

    if free.len() >= n {
        Ok(free[..n].to_vec())
    } else {
        Err((free.len(), total))
    }
}

fn format_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else {
        format!("{}m{:02}s", secs / 60, secs % 60)
    }
}

// --- signal handling ---

static mut QUEUE_FILE_PATH: Option<PathBuf> = None;

unsafe extern "C" fn signal_handler(_sig: i32) {
    if let Some(ref path) = QUEUE_FILE_PATH {
        let _ = fs::remove_file(path);
    }
    process::exit(1);
}

fn register_signal_handlers(path: &Path) {
    unsafe {
        QUEUE_FILE_PATH = Some(path.to_path_buf());
        extern "C" {
            fn signal(sig: i32, handler: unsafe extern "C" fn(i32)) -> usize;
        }
        signal(2, signal_handler); // SIGINT
        signal(15, signal_handler); // SIGTERM
    }
}

// --- main ---

fn main() {
    let requested: usize = match std::env::args().nth(1) {
        None => 1,
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => {
                eprintln!("usage: revolver [N]  (N = number of GPUs, default 1)");
                process::exit(1);
            }
        },
    };

    let total_gpus = query_gpu_indices().len();
    if requested > total_gpus {
        eprintln!(
            "error: requested {} GPUs but only {} available",
            requested, total_gpus
        );
        process::exit(1);
    }

    let queue_file = queue_join(requested);
    register_signal_handlers(&queue_file);

    let format_output = |indices: &[u32]| -> String {
        indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };

    // Fast path: we're first in queue and enough GPUs are free
    let pos = queue_position(&queue_file);
    if pos == 0 {
        if let Ok(indices) = find_free_gpus(requested) {
            lock_gpus(&indices);
            queue_leave(&queue_file);
            println!("{}", format_output(&indices));
            return;
        }
    }

    // Blocking path
    let start = Instant::now();
    let mut tick: usize = 0;
    let stderr = io::stderr();
    let mut last_free_count: usize = 0;
    let mut last_total_count: usize = total_gpus;
    let mut last_pos: usize = pos;

    loop {
        thread::sleep(Duration::from_millis(200));
        tick += 1;

        // Poll every 2s
        if tick % 10 == 0 {
            last_pos = queue_position(&queue_file);
            match find_free_gpus(requested) {
                Ok(indices) if last_pos == 0 => {
                    lock_gpus(&indices);
                    let mut err = stderr.lock();
                    let _ = write!(err, "\r\x1b[2K");
                    let _ = err.flush();
                    queue_leave(&queue_file);
                    println!("{}", format_output(&indices));
                    return;
                }
                Ok(_) => {} // GPUs free but not our turn
                Err((free, total)) => {
                    last_free_count = free;
                    last_total_count = total;
                }
            }
        }

        let spinner = SPINNER[tick % SPINNER.len()];
        let elapsed = format_elapsed(start.elapsed());

        let line = if requested == 1 {
            format!(
                "{} #{} in queue, all {} GPUs busy [{}]",
                spinner,
                last_pos + 1,
                last_total_count,
                elapsed,
            )
        } else {
            format!(
                "{} #{} in queue, need {} GPUs but only {} of {} free [{}]",
                spinner,
                last_pos + 1,
                requested,
                last_free_count,
                last_total_count,
                elapsed,
            )
        };
        let width = term_width();
        let truncated: String = line.chars().take(width).collect();

        let mut err = stderr.lock();
        let _ = write!(err, "\r\x1b[2K{}", truncated);
        let _ = err.flush();
    }
}
