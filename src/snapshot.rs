//! Read-only, point-in-time policy snapshots of a cache.
//!
//! A [`CacheSnapshot`] is an immutable, self-contained view of the policy state of
//! a cache at a single maintenance generation. It is returned by the `snapshot`
//! method of the synchronous and asynchronous `Cache` types:
//!
//! - `moka::sync::Cache::snapshot`
//! - `moka::future::Cache::snapshot`
//!
//! Taking a snapshot does not run any maintenance work: it does not evict entries,
//! fire eviction listeners, advance the expiration clock, execute a user weigher,
//! or block while the maintenance queues are drained. It only performs a small,
//! fixed number of atomic loads.
//!
//! # Maintenance generations
//!
//! A cache updates its policy counters (`entry_count`, `weighted_size`) in
//! batches inside a maintenance task. Each completed maintenance task is a
//! _generation_, identified by [`CacheSnapshot::generation`].
//!
//! All estimate fields in a single snapshot belong to the same generation: a
//! snapshot never mixes, for example, an entry count from before a maintenance
//! run with a weighted size from after it.
//!
//! Cloning a cache shares the underlying state, so clones of the same cache
//! observe the same generation. Independently created caches have independent
//! generations.

use std::time::Duration;

/// How accurately a [`CacheSnapshot`] field reflects the live state of the
/// cache.
///
/// The accuracy applies to [`entry_count`](CacheSnapshot::entry_count) and
/// [`weighted_size`](CacheSnapshot::weighted_size); both values in a snapshot
/// always share the same accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetricAccuracy {
    /// The value is exact: the last maintenance task has fully processed all
    /// write operations that had been issued when the snapshot was taken, so
    /// the policy counters describe exactly what is in the cache.
    Exact,
    /// The value is an approximation based on a completed maintenance
    /// generation. Some write operations were already applied to the concurrent
    /// hash table but had not been applied to the policy data structures yet
    /// when the snapshot was taken.
    Approximate,
    /// No maintenance task has ever run for this cache. The associated value
    /// is reported as zero, not guessed from the live hash table. Run
    /// maintenance (or keep using the cache) and take another snapshot to get
    /// [`Exact`](Self::Exact) or [`Approximate`](Self::Approximate) values.
    Unmaintained,
}

/// A single numeric field of a [`CacheSnapshot`], paired with its
/// [`MetricAccuracy`].
///
/// Use [`value`](Self::value) to read the number and
/// [`accuracy`](Self::accuracy) to inspect how it should be interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Metric {
    value: u64,
    accuracy: MetricAccuracy,
}

impl Metric {
    pub(crate) fn new(value: u64, accuracy: MetricAccuracy) -> Self {
        Self { value, accuracy }
    }

    /// Returns the numeric value.
    ///
    /// When [`accuracy`](#method.accuracy) is
    /// [`MetricAccuracy::Unmaintained`], this is always zero.
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Returns how accurately [`value`](#method.value) reflects the live cache
    /// state.
    pub fn accuracy(&self) -> MetricAccuracy {
        self.accuracy
    }

    /// Returns `true` if this metric is reported as
    /// [`MetricAccuracy::Exact`].
    pub fn is_exact(&self) -> bool {
        self.accuracy == MetricAccuracy::Exact
    }
}

/// The entry admission and eviction strategy configured on a cache.
///
/// This is a stable, value-based view of the strategy used by the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvictionStrategy {
    /// Entries are admitted based on a TinyLFU frequency estimator and
    /// evicted in least-recently-used order.
    TinyLfu,
    /// Entries are evicted in least-recently-used order without a TinyLFU
    /// admission filter.
    Lru,
}

/// A read-only snapshot of a cache's policy state at one maintenance
/// generation.
///
/// Obtain one via the `snapshot` method of a cache:
///
/// - [`moka::sync::Cache::snapshot`](crate::sync::Cache::snapshot)
/// - [`moka::future::Cache::snapshot`](crate::future::Cache::snapshot)
///
/// A snapshot is a plain value with no reference to the cache, so it can be
/// stored, moved across threads, and examined freely without affecting the
/// cache.
///
/// # Consistency
///
/// The estimated fields are mutually consistent:
/// [`entry_count`](Self::entry_count) and
/// [`weighted_size`](Self::weighted_size) are read as one
/// [generation](Self::generation), so they always have the same
/// [`MetricAccuracy`]. The snapshot is not linearizable with concurrent cache
/// writes; callers that need exact values should allow maintenance to run and
/// check that the returned metrics are exact instead of blocking on the
/// maintenance queues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheSnapshot {
    max_capacity: Option<u64>,
    eviction_strategy: EvictionStrategy,
    time_to_live: Option<Duration>,
    time_to_idle: Option<Duration>,
    entry_count: Metric,
    weighted_size: Metric,
    generation: u64,
    pending_write_ops: usize,
}

impl CacheSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        max_capacity: Option<u64>,
        eviction_strategy: EvictionStrategy,
        time_to_live: Option<Duration>,
        time_to_idle: Option<Duration>,
        entry_count: Metric,
        weighted_size: Metric,
        generation: u64,
        pending_write_ops: usize,
    ) -> Self {
        debug_assert_eq!(entry_count.accuracy, weighted_size.accuracy);
        Self {
            max_capacity,
            eviction_strategy,
            time_to_live,
            time_to_idle,
            entry_count,
            weighted_size,
            generation,
            pending_write_ops,
        }
    }

    /// Returns the configured maximum capacity of the cache, expressed in
    /// number of entries when no weigher is configured, or in total weight
    /// units when a weigher is configured. `None` means the cache is unbounded.
    pub fn max_capacity(&self) -> Option<u64> {
        self.max_capacity
    }

    /// Returns the configured entry admission and eviction strategy.
    pub fn eviction_strategy(&self) -> EvictionStrategy {
        self.eviction_strategy
    }

    /// Returns the configured time-to-live, if any.
    pub fn time_to_live(&self) -> Option<Duration> {
        self.time_to_live
    }

    /// Returns the configured time-to-idle, if any.
    pub fn time_to_idle(&self) -> Option<Duration> {
        self.time_to_idle
    }

    /// Returns the number of entries observed by the policy at this
    /// [generation](Self::generation), with its accuracy.
    pub fn entry_count(&self) -> Metric {
        self.entry_count
    }

    /// Returns the total weight observed by the policy at this
    /// [generation](Self::generation), with its accuracy.
    ///
    /// When the cache has no weigher, this is the number of observed entries
    /// weighted by one and equals [`entry_count`](Self::entry_count).
    pub fn weighted_size(&self) -> Metric {
        self.weighted_size
    }

    /// Returns the logical maintenance generation the estimated fields belong
    /// to.
    ///
    /// Zero means no maintenance task has completed yet. Clones of the same
    /// cache share generations; independent caches do not.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the number of write operations that had been issued to the
    /// cache but had not been applied to the policy data structures yet when
    /// the snapshot was taken.
    ///
    /// A non-zero value means the maintenance queues have work pending, which
    /// is why the estimated fields may be
    /// [`approximate`](MetricAccuracy::Approximate). The value itself is a
    /// best-effort observation and must not be treated as exact.
    pub fn pending_write_ops(&self) -> usize {
        self.pending_write_ops
    }

    /// Returns `true` if the estimated fields are
    /// [`MetricAccuracy::Exact`].
    pub fn is_exact(&self) -> bool {
        self.entry_count.is_exact()
    }
}
