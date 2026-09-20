#![cfg(test)]

//! Tests for `future::Cache::snapshot`.

use crate::future::Cache;
use crate::{
    common::time::Clock,
    future::FutureExt,
    notification::{ListenerFuture, RemovalCause},
    snapshot::{CacheSnapshot, EvictionStrategy, MetricAccuracy},
};

use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

/// Asserts the invariants every snapshot must satisfy on its own, regardless
/// of whether its estimated values are exact or approximate.
fn assert_internally_consistent(snap: &CacheSnapshot) {
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

    if snap.entry_count().accuracy() != MetricAccuracy::Unmaintained {
        assert!(
            snap.weighted_size().value() >= snap.entry_count().value(),
            "weighted_size {} must be >= entry_count {}",
            snap.weighted_size().value(),
            snap.entry_count().value()
        );
    }
}

#[tokio::test]
async fn snapshot_before_first_maintenance_is_unmaintained() {
    let mut cache = Cache::builder().max_capacity(10).build();
    cache.reconfigure_for_testing().await;
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

    cache.insert(1, "a").await;
    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert_eq!(snap.generation(), 0);
    assert_eq!(snap.entry_count().accuracy(), MetricAccuracy::Unmaintained);
}

#[tokio::test]
async fn snapshot_after_maintenance_is_exact_and_advances_generation() {
    let cache = Cache::new(10);

    cache.insert(1, "a").await;
    cache.run_pending_tasks().await;

    let snap1 = cache.snapshot();
    assert_internally_consistent(&snap1);
    assert!(snap1.is_exact());
    assert_eq!(snap1.entry_count().value(), 1);
    assert_eq!(snap1.weighted_size().value(), 1);
    assert_eq!(snap1.pending_write_ops(), 0);
    assert!(snap1.generation() >= 1);

    cache.insert(2, "b").await;
    cache.run_pending_tasks().await;

    let snap2 = cache.snapshot();
    assert_internally_consistent(&snap2);
    assert!(snap2.is_exact());
    assert_eq!(snap2.entry_count().value(), 2);
    assert!(snap2.generation() >= snap1.generation());
}

#[tokio::test]
async fn snapshot_reports_weighted_eviction_self_consistently() {
    let cache: Cache<u32, u32> = Cache::builder()
        .max_capacity(10)
        .weigher(|_k, v| *v)
        .build();

    cache.insert(1, 6).await;
    cache.insert(2, 6).await;
    cache.run_pending_tasks().await;

    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert!(snap.is_exact());
    assert_eq!(snap.max_capacity(), Some(10));
    assert!(snap.weighted_size().value() <= 10);
    assert!(snap.entry_count().value() <= 2);
    assert!(snap.entry_count().value() >= 1);
    assert!(snap.weighted_size().value() >= 6 * snap.entry_count().value());
}

#[tokio::test]
async fn snapshot_after_expiration_reports_zero_exactly() {
    let (clock, mock) = Clock::mock();
    let mut cache = Cache::builder()
        .max_capacity(10)
        .time_to_live(Duration::from_secs(10))
        .clock(clock)
        .build();
    cache.reconfigure_for_testing().await;
    let cache = cache;

    cache.insert(1, "a").await;
    cache.run_pending_tasks().await;
    assert_eq!(cache.snapshot().entry_count().value(), 1);

    mock.increment(Duration::from_secs(11));
    cache.run_pending_tasks().await;

    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert!(snap.is_exact());
    assert_eq!(snap.entry_count().value(), 0);
    assert_eq!(snap.weighted_size().value(), 0);
    assert_eq!(snap.time_to_live(), Some(Duration::from_secs(10)));
}

#[tokio::test]
async fn snapshot_after_explicit_invalidation() {
    let cache = Cache::new(10);
    cache.insert(1, "a").await;
    cache.insert(2, "b").await;
    cache.run_pending_tasks().await;

    cache.invalidate(&1).await;
    let pending = cache.snapshot();
    assert_internally_consistent(&pending);

    cache.run_pending_tasks().await;
    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert!(snap.is_exact());
    assert_eq!(snap.entry_count().value(), 1);
    assert_eq!(snap.weighted_size().value(), 1);
}

#[tokio::test]
async fn snapshot_generation_is_shared_between_clones() {
    let cache = Cache::new(10);
    let clone = cache.clone();

    cache.insert(1, "a").await;
    cache.run_pending_tasks().await;

    let s1 = cache.snapshot();
    let s2 = clone.snapshot();
    assert_eq!(s1.generation(), s2.generation());
    assert!(s1.is_exact());
    assert_eq!(s2.entry_count().value(), s1.entry_count().value());

    clone.insert(2, "b").await;
    clone.run_pending_tasks().await;
    assert_eq!(cache.snapshot().generation(), clone.snapshot().generation());
}

#[tokio::test]
async fn snapshot_generations_are_independent_between_caches() {
    let cache1: Cache<u32, &str> = Cache::new(10);
    let cache2: Cache<u32, &str> = Cache::new(10);

    cache1.insert(1, "a").await;
    cache1.run_pending_tasks().await;
    cache1.insert(2, "b").await;
    cache1.run_pending_tasks().await;

    let s1 = cache1.snapshot();
    let s2 = cache2.snapshot();
    assert!(s1.is_exact());
    assert_eq!(s1.entry_count().value(), 2);
    assert_eq!(s2.generation(), 0);
    assert_eq!(s2.entry_count().accuracy(), MetricAccuracy::Unmaintained);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshots_remain_self_consistent_under_concurrent_writes() {
    let cache = Arc::new(Cache::<u32, u32>::new(1000));

    let mut writers = Vec::new();
    for t in 0..4u32 {
        let cache = Arc::clone(&cache);
        writers.push(tokio::spawn(async move {
            for i in 0..500u32 {
                let key = t * 500 + i;
                cache.insert(key, key % 7 + 1).await;
                if i % 32 == 0 {
                    cache.invalidate(&key).await;
                }
            }
        }));
    }

    let snapshotter = {
        let cache = Arc::clone(&cache);
        tokio::spawn(async move {
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
        w.await.expect("writer panicked");
    }
    snapshotter.await.expect("snapshotter panicked");

    cache.run_pending_tasks().await;
    let final_snap = cache.snapshot();
    assert_internally_consistent(&final_snap);
    assert!(final_snap.is_exact());
    assert!(final_snap.weighted_size().value() <= 1000);
}

#[tokio::test]
async fn snapshot_does_not_invoke_weigher_or_listener() {
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
        .async_eviction_listener(move |_k, _v, _cause| {
            let listener_calls = Arc::clone(&listener_calls2);
            async move {
                listener_calls.fetch_add(1, Ordering::SeqCst);
            }
            .boxed()
        })
        .build();

    cache.insert(1, 100).await;
    let weigh_after_insert = weigh_calls.load(Ordering::SeqCst);
    assert_eq!(weigh_after_insert, 1);

    for _ in 0..100 {
        let snap = cache.snapshot();
        assert_internally_consistent(&snap);
    }

    assert_eq!(weigh_calls.load(Ordering::SeqCst), weigh_after_insert);
    assert_eq!(listener_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn snapshot_while_maintenance_is_running_does_not_block() {
    use tokio::sync::Notify;

    // Gate the async eviction listener so a maintenance task stays running
    // while we take a snapshot.
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());

    let cache: Cache<u32, u32> = {
        let started = Arc::clone(&started);
        let release = Arc::clone(&release);
        Cache::builder()
            .max_capacity(1)
            .async_eviction_listener(move |_k, _v, cause| {
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                async move {
                    if cause == RemovalCause::Size {
                        started.notify_one();
                        release.notified().await;
                    }
                }
                .boxed()
            })
            .build()
    };

    cache.insert(1, 1).await;
    cache.run_pending_tasks().await;

    let maint = {
        let cache = cache.clone();
        tokio::spawn(async move {
            cache.insert(2, 1).await;
            cache.run_pending_tasks().await;
        })
    };

    // Wait until the maintenance task is blocked in the listener.
    started.notified().await;

    // This call is synchronous and must return immediately even though
    // maintenance is in flight. Its counters must be from one generation.
    let snap = cache.snapshot();
    assert_eq!(
        snap.entry_count().accuracy(),
        snap.weighted_size().accuracy()
    );
    assert!(snap.generation() >= 1);
    assert!(snap.entry_count().value() >= 1);

    release.notify_one();
    maint.await.expect("maintenance panicked");

    cache.run_pending_tasks().await;
    let settled = cache.snapshot();
    assert_internally_consistent(&settled);
    assert!(settled.is_exact());
    assert_eq!(settled.entry_count().value(), 1);
}

#[tokio::test]
async fn snapshot_works_after_listener_panic() {
    let panic_armed = Arc::new(AtomicBool::new(true));
    let cache: Cache<u32, &str> = {
        let panic_armed = Arc::clone(&panic_armed);
        Cache::builder()
            .max_capacity(2)
            .async_eviction_listener(move |_k, _v, _cause| {
                let panic_armed = Arc::clone(&panic_armed);
                async move {
                    if panic_armed.swap(false, Ordering::SeqCst) {
                        panic!("listener panic must be isolated");
                    }
                }
                .boxed()
            })
            .build()
    };

    cache.insert(1, "a").await;
    cache.run_pending_tasks().await;
    cache.insert(2, "b").await;
    cache.run_pending_tasks().await;
    // The third insertion forces a size eviction; the listener panics once and
    // is then automatically disabled.
    cache.insert(3, "c").await;
    cache.run_pending_tasks().await;

    cache.run_pending_tasks().await;
    let snap = cache.snapshot();
    assert_internally_consistent(&snap);
    assert!(snap.is_exact());
    assert!(snap.entry_count().value() <= 2);
}

#[allow(dead_code)]
fn _listener_future_type_used(_f: ListenerFuture) {}
