//! A persistent worker pool for the draw's parallel phases.
//!
//! `std::thread::scope` creates and joins its threads on every call, and on
//! Akuma that costs about 1.5 ms per thread (measured: a vertex stage whose
//! chunks run 1.7 ms each spent 7.9 ms in the scope). A frame has several
//! parallel phases, so the workers are started once and parked on a condvar
//! between jobs.
//!
//! `run(n, f)` calls `f(0..n)` — index 0 on the calling thread, the rest on
//! pool workers — and returns when all have finished, so `f` may borrow from
//! the caller's stack. One job runs at a time.

use std::sync::{Condvar, Mutex, OnceLock};

/// the job as the workers see it: a lifetime-erased borrow that `run` keeps
/// alive until every worker is done with it
#[derive(Clone, Copy)]
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
}

// the pointee is Sync and outlives the job (see `run`)
unsafe impl Send for Job {}

struct State {
    /// bumped for every posted job
    epoch: u64,
    job: Option<Job>,
    /// workers that have finished the current job
    done: usize,
}

struct Pool {
    state: Mutex<State>,
    wake: Condvar,
    finished: Condvar,
    /// serializes callers
    gate: Mutex<()>,
    workers: Mutex<usize>,
}

fn pool() -> &'static Pool {
    static P: OnceLock<Pool> = OnceLock::new();
    P.get_or_init(|| Pool {
        state: Mutex::new(State { epoch: 0, job: None, done: 0 }),
        wake: Condvar::new(),
        finished: Condvar::new(),
        gate: Mutex::new(()),
        workers: Mutex::new(0),
    })
}

fn worker(index: usize) {
    let p = pool();
    let mut seen = 0u64;
    loop {
        let job = {
            let mut st = p.state.lock().unwrap();
            while st.epoch == seen {
                st = p.wake.wait(st).unwrap();
            }
            seen = st.epoch;
            st.job
        };
        if let Some(job) = job {
            if index < job.n {
                // SAFETY: `run` does not return (so the closure stays alive)
                // until every participating worker has bumped `done`
                unsafe { (*job.f)(index) };
                let mut st = p.state.lock().unwrap();
                st.done += 1;
                p.finished.notify_all();
            }
        }
    }
}

/// Run `f(i)` for `i` in `0..n`, `f(0)` on this thread. With `n <= 1` there is
/// no parallelism and no pool.
pub fn run(n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n <= 1 {
        f(0);
        return;
    }
    let p = pool();
    let _gate = p.gate.lock().unwrap();
    {
        let mut have = p.workers.lock().unwrap();
        while *have < n - 1 {
            *have += 1;
            let index = *have;
            std::thread::Builder::new()
                .name(format!("gpu-{index}"))
                .spawn(move || worker(index))
                .expect("gpu worker thread");
        }
    }
    // SAFETY: erase the borrow's lifetime; we wait below for every worker to
    // finish before `f` can go out of scope
    let fp: *const (dyn Fn(usize) + Sync + '_) = f;
    let fp: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(fp) };
    {
        let mut st = p.state.lock().unwrap();
        st.job = Some(Job { f: fp, n });
        st.done = 0;
        st.epoch += 1;
        p.wake.notify_all();
    }
    f(0);
    let mut st = p.state.lock().unwrap();
    while st.done < n - 1 {
        st = p.finished.wait(st).unwrap();
    }
    st.job = None;
}
