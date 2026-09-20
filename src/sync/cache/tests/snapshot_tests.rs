#![cfg(test)]

//! Tests for `Cache::snapshot`.

use super::Cache;
use crate::{
    common::time::Clock,
    notification::RemovalCause,
    snapshot::{EvictionStrategy, MetricAccuracy},
};

use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    thread,
    time::Duration,
};

/// Asserts the invariants every snapshot must satisfy on its own, regardless
/// of whether its estimated values are exact or approximate.
fn assert_internally_consistent(snap: &crate::snapshot::CacheSnapshot) {
    // Both estimates must describe the same maintenance generation and share
    // the same accuracy label.
    assert_eq!(
        snap.entry_count().accuracy(),
        snap.weighted_size().accuracy(),
        "entry_count and weighted_size accuracy must agree"
    );

    match snap.entry_count().accuracy() {
        MetricAccuracy::Unmaintained => {
            assert_eq!(snap.generation(), 0);
            assert_eq!(snap.entry_count().value(), 0);
            assert_eq!(snap.weighted_size().value(), 0);
        }
        MetricAccuracy::Exact => assert!(snap.generation() >= 1),
        MetricAccuracy::Approximate => {
            assert!(snap.generation() >= 1);
            assert!(snap.pending_write_ops() > 0);
        }
    }

    // A cache without a weigher weights every entry by one, so its weighted
    // size must never be below its entry count.
    if snap.entry_count().accuracy() != MetricAccuracy::Unmaintained {
        assert!(
            snap.weighted_size().value() >= snap.entry_count().value(),
            "weighted_size {} must be >= entry_count {}",
            snap.weighted_size().value(),
            snap.entry_count().value()
        );
    }
}

#[test]
fn snapshot_before_first_maintenance_is_unmaintained() {
    let mut cache = Cache::builder().max_capacity(10).build();
    cache.reconfigure_for_testing();
    let cache = cache;

    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert_eq!(snap.generation(), 0);
    assert_eq!(snap.entry_count().accuracy(), MetricAccuracy::Unmaintained);
    assert_eq!(snap.entry_count().value(), 0);
    assert_eq!(snap.max_capacity(), Some(10));
    assert_eq!(snap.eviction_strategy(), EvictionStrategy::TinyLfu);
    assert_eq!(snap.time_to_live(), None);
    assert_eq!(snap.time_to_idle(), None);
    assert_eq!(snap.pending_write_ops(), 0);
    assert!(!snap.is_exact());

    // The entry is visible in the hash table even though maintenance has not
    // observed it yet.
    cache.insert(1, "a");
    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert_eq!(snap.generation(), 0);
    assert_eq!(snap.entry_count().accuracy(), MetricAccuracy::Unmaintained);
}

#[test]
fn snapshot_after_maintenance_is_exact_and_advances_generation() {
    let cache = Cache::new(10);

    cache.insert(1, "a");
    cache.run_pending_tasks();

    let snap1 = cache.snapshot();
    assert_internally_consistent(&snap1);
    assert!(snap1.is_exact());
    assert_eq!(snap1.entry_count().value(), 1);
    assert_eq!(snap1.weighted_size().value(), 1);
    assert_eq!(snap1.pending_write_ops(), 0);
    assert!(snap1.generation() >= 1);

    cache.insert(2, "b");
    cache.run_pending_tasks();

    let snap2 = cache.snapshot();
    assert_internally_consistent(&snap2);
    assert!(snap2.is_exact());
    assert_eq!(snap2.entry_count().value(), 2);
    assert!(snap2.generation() >= snap1.generation());
}

#[test]
fn snapshot_reports_weighted_eviction_self_consistently() {
    let cache: Cache<u32, u32> = Cache::builder()
        .max_capacity(10)
        .weigher(|_k, v| *v)
        .build();

    cache.insert(1, 6);
    cache.insert(2, 6);
    cache.run_pending_tasks();

    // 6 + 6 exceeds 10, so maintenance must have evicted something. We do not
    // pin which entry survives (that is an admission-policy decision); we
    // assert the snapshot values are consistent with the capacity and each
    // other.
    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert!(snap.is_exact());
    assert_eq!(snap.max_capacity(), Some(10));
    assert!(snap.weighted_size().value() <= 10);
    assert!(snap.entry_count().value() <= 2);
    assert!(snap.entry_count().value() >= 1);
    assert!(snap.weighted_size().value() >= 6 * snap.entry_count().value());
}

#[test]
fn snapshot_after_expiration_reports_zero_exactly() {
    let (clock, mock) = Clock::mock();
    let mut cache = Cache::builder()
        .max_capacity(10)
        .time_to_live(Duration::from_secs(10))
        .clock(clock)
        .build();
    cache.reconfigure_for_testing();
    let cache = cache;

    cache.insert(1, "a");
    cache.run_pending_tasks();
    assert_eq!(cache.snapshot().entry_count().value(), 1);

    mock.increment(Duration::from_secs(11));
    cache.run_pending_tasks();

    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert!(snap.is_exact());
    assert_eq!(snap.entry_count().value(), 0);
    assert_eq!(snap.weighted_size().value(), 0);
    assert_eq!(snap.time_to_live(), Some(Duration::from_secs(10)));
}

#[test]
fn snapshot_after_explicit_invalidation() {
    let cache = Cache::new(10);
    cache.insert(1, "a");
    cache.insert(2, "b");
    cache.run_pending_tasks();

    cache.invalidate(&1);
    let pending = cache.snapshot();
    assert_internally_consistent(&pending);

    cache.run_pending_tasks();
    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert!(snap.is_exact());
    assert_eq!(snap.entry_count().value(), 1);
    assert_eq!(snap.weighted_size().value(), 1);
}

#[test]
fn snapshot_generation_is_shared_between_clones() {
    let cache = Cache::new(10);
    let clone = cache.clone();

    cache.insert(1, "a");
    cache.run_pending_tasks();

    let s1 = cache.snapshot();
    let s2 = clone.snapshot();
    assert_eq!(s1.generation(), s2.generation());
    assert!(s1.is_exact());
    assert_eq!(s2.entry_count().value(), s1.entry_count().value());

    clone.insert(2, "b");
    clone.run_pending_tasks();
    assert_eq!(cache.snapshot().generation(), clone.snapshot().generation());
}

#[test]
fn snapshot_generations_are_independent_between_caches() {
    let cache1: Cache<u32, &str> = Cache::new(10);
    let cache2: Cache<u32, &str> = Cache::new(10);

    cache1.insert(1, "a");
    cache1.run_pending_tasks();
    cache1.insert(2, "b");
    cache1.run_pending_tasks();

    // cache2 has done no maintenance of its own, even though cache1 has.
    let s1 = cache1.snapshot();
    let s2 = cache2.snapshot();
    assert!(s1.is_exact());
    assert_eq!(s1.entry_count().value(), 2);
    assert_eq!(s2.generation(), 0);
    assert_eq!(s2.entry_count().accuracy(), MetricAccuracy::Unmaintained);
}

#[test]
fn snapshots_remain_self_consistent_under_concurrent_writes() {
    let cache = Arc::new(Cache::<u32, u32>::new(1000));

    let mut writers = Vec::new();
    for t in 0..4 {
        let cache = Arc::clone(&cache);
        writers.push(thread::spawn(move || {
            for i in 0..500u32 {
                let key = t * 500 + i;
                cache.insert(key, key % 7 + 1);
                if i % 32 == 0 {
                    cache.invalidate(&key);
                }
            }
        }));
    }

    // Snapshot repeatedly while writes and maintenance are in flight. We never
    // pin an approximate field to an exact value; we only assert internal
    // consistency.
    let snapshotter = {
        let cache = Arc::clone(&cache);
        thread::spawn(move || {
            for _ in 0..10_000 {
                let snap = cache.snapshot();
                assert_eq!(
                    snap.entry_count().accuracy(),
                    snap.weighted_size().accuracy()
                );
                assert!(snap.weighted_size().value() >= snap.entry_count().value());
                assert!(snap.weighted_size().value() <= 1000 + 8);
            }
        })
    };

    for w in writers {
        w.join().expect("writer panicked");
    }
    snapshotter.join().expect("snapshotter panicked");

    cache.run_pending_tasks();
    let final_snap = cache.snapshot();
    assert_internally_consistent(&final_snap);
    assert!(final_snap.is_exact());
    assert!(final_snap.weighted_size().value() <= 1000);
}

#[test]
fn snapshot_does_not_invoke_weigher_or_listener() {
    let weigh_calls = Arc::new(AtomicUsize::new(0));
    let listener_calls = Arc::new(AtomicUsize::new(0));

    let weigh_calls2 = Arc::clone(&weigh_calls);
    let listener_calls2 = Arc::clone(&listener_calls);

    let cache: Cache<u32, u32> = Cache::builder()
        .max_capacity(10)
        .weigher(move |_k, v| {
            weigh_calls2.fetch_add(1, Ordering::SeqCst);
            *v
        })
        .eviction_listener(move |_k, _v, _cause| {
            listener_calls2.fetch_add(1, Ordering::SeqCst);
        })
        .build();

    cache.insert(1, 100); // would evict under weight 10, but only at maintenance
    let weigh_after_insert = weigh_calls.load(Ordering::SeqCst);
    assert_eq!(weigh_after_insert, 1);

    for _ in 0..100 {
        let snap = cache.snapshot();
        assert_internally_consistent(&snap);
    }

    assert_eq!(
        weigh_calls.load(Ordering::SeqCst),
        weigh_after_insert,
        "snapshot must not call the weigher"
    );
    assert_eq!(
        listener_calls.load(Ordering::SeqCst),
        0,
        "snapshot must not call the eviction listener"
    );
}

#[test]
fn snapshot_while_maintenance_is_running_does_not_block() {
    // Gate the eviction listener so a maintenance task stays running while we
    // take a snapshot from another thread.
    let entered = Arc::new((Mutex::new(false), Condvar::new()));
    let release = Arc::new((Mutex::new(false), std::sync::Condvar::new()));

    let cache: Cache<u32, u32> = {
        let entered = Arc::clone(&entered);
        let release = Arc::clone(&release);
        Cache::builder()
            .max_capacity(1)
            .eviction_listener(move |_k, _v, cause| {
                if cause == RemovalCause::Size {
                    {
                        let (lock, cvar) = &*entered;
                        let mut entered_g = lock.lock().unwrap();
                        *entered_g = true;
                        cvar.notify_all();
                    }
                    let (lock, cvar) = &*release;
                    let mut released = lock.lock().unwrap();
                    while !*released {
                        released = cvar.wait(released).unwrap();
                    }
                }
            })
            .build()
    };

    cache.insert(1, 1);
    cache.run_pending_tasks();

    let maint = {
        let cache = cache.clone();
        thread::spawn(move || {
            // Inserting a second entry and running maintenance evicts key 1 and
            // blocks inside its listener.
            cache.insert(2, 1);
            cache.run_pending_tasks();
        })
    };

    {
        let (lock, cvar) = &*entered;
        let mut entered_g = lock.lock().unwrap();
        while !*entered_g {
            entered_g = cvar.wait(entered_g).unwrap();
        }
    }

    // Maintenance is now running (and blocked). Snapshot must return promptly
    // with counters from a consistent generation.
    let snap = cache.snapshot();
    assert_eq!(
        snap.entry_count().accuracy(),
        snap.weighted_size().accuracy()
    );
    assert!(snap.generation() >= 1);
    assert!(snap.entry_count().value() >= 1);

    {
        let (lock, cvar) = &*release;
        let mut released = lock.lock().unwrap();
        *released = true;
        cvar.notify_all();
    }
    maint.join().expect("maintenance panicked");

    cache.run_pending_tasks();
    let settled = cache.snapshot();
    assert_internally_consistent(&settled);
    assert!(settled.is_exact());
    assert_eq!(settled.entry_count().value(), 1);
}

#[test]
fn snapshot_works_after_listener_panic() {
    let panic_armed = Arc::new(AtomicBool::new(true));
    let cache: Cache<u32, &str> = {
        let panic_armed = Arc::clone(&panic_armed);
        Cache::builder()
            .max_capacity(2)
            .eviction_listener(move |_k, _v, _cause| {
                if panic_armed.swap(false, Ordering::SeqCst) {
                    panic!("listener panic must be isolated");
                }
            })
            .build()
    };

    // Catch the listener panic so the test thread survives; moka isolates the
    // panic internally via catch_unwind anyway.
    cache.insert(1, "a");
    cache.run_pending_tasks();
    cache.insert(2, "b");
    cache.run_pending_tasks();
    // The third insertion forces a size eviction; the listener panics once and
    // is then automatically disabled.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.insert(3, "c");
        cache.run_pending_tasks();
    }));
    assert!(result.is_ok(), "a listener panic must not propagate");

    // The cache stays usable and snapshots remain self-consistent.
    cache.run_pending_tasks();
    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert!(snap.is_exact());
    assert!(snap.entry_count().value() <= 2);
}
