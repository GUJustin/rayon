//! Fixed-work folding/reduction workload for comparing worker waiting policies.
//!
//! This models the shrinking parallel regions and dependent challenges in a
//! polynomial prover. Arithmetic is wrapping u64 arithmetic, not cryptography;
//! its timings are not estimates of Jolt arithmetic costs. Unlike a timed busy
//! loop, every configuration performs the same arithmetic and returns the same
//! checksum. Pool construction and input initialization are outside the timer.
//!
//! Build: cargo build --release -p rayon --example folding_workload
//! Run: BENCH_THREADS=16 target/release/examples/folding_workload
//! Compare with BENCH_IDLE_SPIN_US=64, alternating run order. Absence selects
//! the existing Rayon default; zero explicitly selects immediate sleep attempts.
//! BENCH_LOG_N (20), BENCH_BATCHES (8), BENCH_REPEATS (10), BENCH_GRAIN (1024),
//! and BENCH_SERIAL_WORK (256) control the shape. Use an external process CPU
//! measurement as well as elapsed time; less latency can require more CPU.
//! BENCH_MODE=install (default) enters the pool once for the whole workload;
//! BENCH_MODE=regions enters separately for each message and bind, leaving the
//! dependent challenge computation on the driving thread.

use rayon::prelude::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn parameter(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| {
        value.parse().unwrap_or_else(|_| panic!("invalid {name}"))
    })
}

fn mix(value: u64) -> u64 {
    value
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
        .rotate_left(17)
}

fn message(a: &[u64], b: &[u64], grain: usize) -> [u64; 2] {
    let half = a.len() / 2;
    let (a0, a1) = a.split_at(half);
    let (b0, b1) = b.split_at(half);
    a0.par_iter()
        .zip(a1)
        .zip(b0.par_iter().zip(b1))
        .with_min_len(grain)
        .fold(
            || [0_u64; 2],
            |mut sums, ((&a0, &a1), (&b0, &b1))| {
                sums[0] = sums[0].wrapping_add(a0.wrapping_mul(b0));
                let a2 = a1.wrapping_add(a1).wrapping_sub(a0);
                let b2 = b1.wrapping_add(b1).wrapping_sub(b0);
                sums[1] = sums[1].wrapping_add(a2.wrapping_mul(b2));
                sums
            },
        )
        .reduce(
            || [0_u64; 2],
            |left, right| {
                [
                    left[0].wrapping_add(right[0]),
                    left[1].wrapping_add(right[1]),
                ]
            },
        )
}

fn bind(values: &mut Vec<u64>, challenge: u64, grain: usize) {
    let half = values.len() / 2;
    let (left, right) = values.split_at_mut(half);
    left.par_iter_mut()
        .zip(right.par_iter())
        .with_min_len(grain)
        .for_each(|(left, &right)| {
            *left = left.wrapping_add(challenge.wrapping_mul(right.wrapping_sub(*left)));
        });
    values.truncate(half);
}

fn in_region<R: Send>(pool: Option<&rayon::ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

fn proof_shape(
    batches: &mut [(Vec<u64>, Vec<u64>)],
    grain: usize,
    serial_work: usize,
    region_pool: Option<&rayon::ThreadPool>,
) -> u64 {
    let mut transcript = 0x1f83_d9ab_fb41_bd6b;
    for (a, b) in batches {
        while a.len() > 1 {
            let sums = in_region(region_pool, || message(a, b, grain));
            transcript = mix(transcript ^ sums[0] ^ sums[1].rotate_left(31));
            for _ in 0..serial_work {
                transcript = mix(black_box(transcript));
            }
            let challenge = transcript;
            in_region(region_pool, || {
                rayon::join(|| bind(a, challenge, grain), || bind(b, challenge, grain));
            });
        }
        transcript ^= a[0].wrapping_mul(b[0]);
    }
    black_box(transcript)
}

fn main() {
    let mode = std::env::var("BENCH_MODE").unwrap_or_else(|_| "install".to_owned());
    assert!(matches!(mode.as_str(), "install" | "regions"));
    let threads = parameter("BENCH_THREADS", 16);
    let log_n = parameter("BENCH_LOG_N", 20);
    let batch_count = parameter("BENCH_BATCHES", 8);
    let repeats = parameter("BENCH_REPEATS", 10);
    let grain = parameter("BENCH_GRAIN", 1024);
    let serial_work = parameter("BENCH_SERIAL_WORK", 256);
    assert!(threads > 0 && grain > 0 && repeats > 0 && batch_count > 0);
    assert!(
        log_n > 0 && log_n <= 24,
        "keep the diagnostic allocation bounded"
    );
    assert!(batch_count <= 16);
    let n = 1usize << log_n;
    let idle_us = std::env::var("BENCH_IDLE_SPIN_US")
        .ok()
        .map(|value| value.parse::<u64>().expect("invalid BENCH_IDLE_SPIN_US"));
    let mut builder = rayon::ThreadPoolBuilder::new().num_threads(threads);
    if let Some(idle_us) = idle_us {
        builder = builder.idle_spin_timeout(Duration::from_micros(idle_us));
    }
    let pool = builder.build().unwrap();
    let inputs: Vec<_> = (0..batch_count)
        .map(|batch| {
            let a = (0..n).map(|i| mix(i as u64 ^ batch as u64)).collect();
            let b = (0..n).map(|i| mix(i as u64 ^ !(batch as u64))).collect();
            (a, b)
        })
        .collect();
    let mut expected = None;
    for repeat in 0..=repeats {
        let mut values = inputs.clone();
        let start = Instant::now();
        let checksum = if mode == "install" {
            pool.install(|| proof_shape(&mut values, grain, serial_work, None))
        } else {
            proof_shape(&mut values, grain, serial_work, Some(&pool))
        };
        let elapsed = start.elapsed();
        if let Some(expected) = expected {
            assert_eq!(checksum, expected);
        } else {
            expected = Some(checksum);
        }
        if repeat > 0 {
            println!(
                "mode={mode} threads={threads} log_n={log_n} batches={batch_count} grain={grain} serial_work={serial_work} idle_us={idle_us:?} repeat={repeat} ns={} checksum={checksum:016x}",
                elapsed.as_nanos()
            );
        }
    }
}
