//! Keyboard input, the raw-Linux way: termios raw mode on the console tty,
//! `poll(2)` for non-blocking reads, and a small escape-sequence decoder.
//!
//! The template gets all of this from crossterm (`use-dev-tty`). We inline
//! the ~80 lines that matter, for the same reason rio will not carry
//! crossterm onto this kernel: fewer moving parts between the keystroke and
//! the pixel. The kernel side of this is the console pump (`console.rs`):
//! xHCI keyboard bytes land in the line discipline and come back out here.
//!
//! Best-effort by design: if stdin is not a tty (piped, or a kernel build
//! with the console elsewhere) we run headless — no keys, no raw-mode
//! error — and SIGINT/SIGTERM still stop the screensaver cleanly.

use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};

static QUIT_FLAG: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_sig: libc::c_int) {
    QUIT_FLAG.store(true, Ordering::SeqCst);
}

/// Install SIGINT/SIGTERM handlers. Idempotent; harmless if stdin is not a
/// tty. Returns the initial flag state (always false).
pub fn install_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
}

pub fn quit_requested() -> bool {
    QUIT_FLAG.load(Ordering::SeqCst)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Quit,        // q or Esc
    Left,        // ESC [ D — previous asset
    Right,       // ESC [ C — next asset
    Up,          // ESC [ A — unused, reserved (fps nudge?)
    Down,        // ESC [ B — unused, reserved
    Other,
}

/// Raw-mode guard for stdin. Construct before the render loop; Drop restores
/// the original termios even on panic/early-return (the kernel analogue of
/// the template's LeaveAlternateScreen: leave the console as you found it).
pub struct RawTty {
    fd: RawFd,
    saved: Option<libc::termios>,
    buf: Vec<u8>,
}

impl RawTty {
    pub fn new() -> io::Result<RawTty> {
        let fd = io::stdin().as_raw_fd();
        let mut saved: libc::termios = unsafe { std::mem::zeroed() };
        let is_tty = unsafe { libc::tcgetattr(fd, &mut saved) } == 0;
        if is_tty {
            let mut raw = saved;
            unsafe { libc::cfmakeraw(&mut raw) };
            // cfmakeraw turns off OPOST; we are a fullscreen pixel app and
            // never write text to the tty, so leave output post-processing
            // enabled to avoid surprising the console's ANSI state.
            raw.c_oflag = saved.c_oflag;
            if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(RawTty {
            fd,
            saved: is_tty.then_some(saved),
            buf: Vec::new(),
        })
    }

    /// Drain pending input into `keys`. Never blocks; `poll(2)` timeout 0.
    pub fn poll_keys(&mut self) -> Vec<Key> {
        let mut keys = Vec::new();
        if self.saved.is_none() {
            return keys; // headless
        }
        let mut pfd = libc::pollfd {
            fd: self.fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // A frame's worth of grace for escape sequences to arrive whole.
        loop {
            let r = unsafe { libc::poll(&mut pfd, 1, 0) };
            if r <= 0 || pfd.revents & libc::POLLIN == 0 {
                break;
            }
            let mut chunk = [0u8; 64];
            let n = unsafe { libc::read(self.fd, chunk.as_mut_ptr() as *mut libc::c_void, chunk.len()) };
            if n <= 0 {
                break;
            }
            self.buf.extend_from_slice(&chunk[..n as usize]);
        }
        keys.extend(decode(&mut self.buf));
        keys
    }
}

impl Drop for RawTty {
    fn drop(&mut self) {
        if let Some(saved) = &self.saved {
            unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, saved) };
        }
    }
}

/// Decode keys out of a byte buffer, consuming what it understands.
fn decode(buf: &mut Vec<u8>) -> Vec<Key> {
    let mut keys = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        match buf[i] {
            b'q' | b'Q' => {
                keys.push(Key::Quit);
                i += 1;
            }
            0x1b => {
                // ESC [ <letter> is an arrow; a lone ESC (the last byte of
                // this drain, nothing followed it) is the template's quit key.
                if i + 2 < buf.len() && buf[i + 1] == b'[' {
                    keys.push(match buf[i + 2] {
                        b'A' => Key::Up,
                        b'B' => Key::Down,
                        b'C' => Key::Right,
                        b'D' => Key::Left,
                        _ => Key::Other,
                    });
                    i += 3;
                } else if i + 1 == buf.len() {
                    keys.push(Key::Quit);
                    i += 1;
                } else {
                    keys.push(Key::Other);
                    i += 1;
                }
            }
            0x03 => {
                // ^C — the console turns it into SIGINT via ISIG, but a
                // kernel line discipline might hand it to us raw instead.
                keys.push(Key::Quit);
                i += 1;
            }
            _ => {
                keys.push(Key::Other);
                i += 1;
            }
        }
    }
    buf.clear();
    keys
}
