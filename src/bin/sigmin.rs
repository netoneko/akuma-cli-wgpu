// throwaway bisect helper — minimal signal-mask repro
use std::time::{Duration, Instant};

extern "C" fn on_sig(_s: libc::c_int) {}

fn main() {
    unsafe {
        // v2: handlers installed, like the demo's install_signal_handlers
        libc::signal(libc::SIGINT, on_sig as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_sig as *const () as libc::sighandler_t);
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGTERM);
        let r = libc::sigprocmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        eprintln!("sigmin: sigprocmask ret={r}");
        let mut old: libc::sigset_t = std::mem::zeroed();
        libc::sigprocmask(0, std::ptr::null(), &mut old);
        eprintln!(
            "sigmin: TERM in current mask = {}",
            libc::sigismember(&old, libc::SIGTERM)
        );
    }
    let t0 = Instant::now();
    let mut n = 0u64;
    loop {
        let _ = t0.elapsed();
        std::thread::sleep(Duration::from_millis(10));
        n += 1;
        if n % 100 == 0 {
            eprintln!("sigmin: alive {}s, TERM still ignored", n / 100);
        }
    }
}
