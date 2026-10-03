// throwaway bisect helper — v3: handlers only (no mask), does the handler run on EINTR-delivery?
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

static HITS: AtomicU32 = AtomicU32::new(0);

extern "C" fn on_sig(_s: libc::c_int) {
    HITS.fetch_add(1, Ordering::SeqCst);
}

fn main() {
    unsafe {
        libc::signal(libc::SIGINT, on_sig as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_sig as *const () as libc::sighandler_t);
    }
    let mut n = 0u64;
    loop {
        std::thread::sleep(Duration::from_millis(10));
        n += 1;
        if n % 100 == 0 {
            eprintln!("sigmin v3: alive {}s, handler hits={}", n / 100, HITS.load(Ordering::SeqCst));
        }
    }
}
