//! Provides a read-only, point-in-time snapshot of a cache's policy and
//! maintenance state.
//!
//! The types in this module are returned by the `snapshot` method of
//! [`sync::Cache`](../sync/struct.Cache.html#method.snapshot) and
//! [`future::Cache`](../future/struct.Cache.html#method.snapshot). They share
//! the same semantics regardless of which cache type produced them.
//!
//! A snapshot is a best-effort, non-linearizable view of a live cache: it does
//! not pause concurrent writers, does not wait for the maintenance queue to
//! drain, and never invokes user code (eviction listeners, weigher closures or
//! expiry calculators). To make the snapshot explainable, every observed
//! counter is paired with a [`SnapshotAccuracy`] and all counter values are
//! guaranteed to come from the same logical maintenance generation.

use crate::policy::{EvictionPolicy, Policy};

/// A read-only, point-in-time view of a cache's capacity/eviction policy and
/// its maintenance state.
///
/// A snapshot answers the following questions without pausing cache writes:
///
/// - Which capacity and eviction policy is the cache configured with?
///   ([`policy`](#method.policy) and [`eviction_policy`](#method.eviction_policy))
/// - How many entries and how much weight have been observed by the cache's
///   maintenance so far? ([`entry_count`](#method.entry_count) and
///   [`weighted_size`](#method.weighted_size))
/// - Is the maintenance queue lagging behind?
///   ([`maintenance`](#method.maintenance))
/// - Which logical maintenance generation does this snapshot come from?
///   ([`maintenance_generation`](#method.maintenance_generation))
///
/// # Consistency
///
/// The snapshot is not linearizable against all concurrent writes, but the
/// `entry_count` and `weighted_size` values are always read from the same
/// maintenance generation: they will never combine a pre-maintenance entry
/// count with a post-maintenance weighted size. When a consistent read is not
/// possible without blocking (e.g. a maintenance run is concurrently
/// publishing new counters), the values are reported with
/// [`SnapshotAccuracy::Approximate`] instead.
#[derive(Clone, Debug)]
pub struct CacheSnapshot {
    policy: Policy,
    eviction_policy: EvictionPolicy,
    entry_count: SnapshotValue<u64>,
    weighted_size: SnapshotValue<u64>,
    maintenance: MaintenanceStatus,
}

impl CacheSnapshot {
    pub(crate) fn new(
        policy: Policy,
        eviction_policy: EvictionPolicy,
        entry_count: SnapshotValue<u64>,
        weighted_size: SnapshotValue<u64>,
        maintenance: MaintenanceStatus,
    ) -> Self {
        Self {
            policy,
            eviction_policy,
            entry_count,
            weighted_size,
            maintenance,
        }
    }

    /// Returns the capacity and expiration policy of the cache.
    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// Returns the eviction (and admission) policy of the cache.
    pub fn eviction_policy(&self) -> &EvictionPolicy {
        &self.eviction_policy
    }

    /// Returns the entry count observed by the cache's maintenance, paired
    /// with its accuracy.
    pub fn entry_count(&self) -> SnapshotValue<u64> {
        self.entry_count
    }

    /// Returns the total weight of the entries observed by the cache's
    /// maintenance, paired with its accuracy.
    pub fn weighted_size(&self) -> SnapshotValue<u64> {
        self.weighted_size
    }

    /// Returns the status of the cache's maintenance at the time of the
    /// snapshot.
    pub fn maintenance(&self) -> MaintenanceStatus {
        self.maintenance
    }

    /// Returns the logical maintenance generation this snapshot was taken
    /// from.
    ///
    /// The generation is the number of maintenance runs completed by the
    /// cache so far. It is shared by all clones of the same cache, but is
    /// never shared between independently created caches. A generation of
    /// `0` means no maintenance run has completed yet.
    pub fn maintenance_generation(&self) -> u64 {
        self.maintenance.generation()
    }
}

/// A value observed in a [`CacheSnapshot`], paired with its accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotValue<T> {
    value: T,
    accuracy: SnapshotAccuracy,
}

impl<T: Copy> SnapshotValue<T> {
    pub(crate) fn new(value: T, accuracy: SnapshotAccuracy) -> Self {
        Self { value, accuracy }
    }

    /// Returns the observed value.
    pub fn value(&self) -> T {
        self.value
    }

    /// Returns the accuracy of the observed value.
    pub fn accuracy(&self) -> SnapshotAccuracy {
        self.accuracy
    }

    /// Returns `true` if the value is exact.
    pub fn is_exact(&self) -> bool {
        self.accuracy == SnapshotAccuracy::Exact
    }
}

/// Indicates how accurate a value in a [`CacheSnapshot`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SnapshotAccuracy {
    /// The value was read consistently from a completed maintenance
    /// generation. It exactly reflects the cache state observed by that
    /// maintenance run (newer writes may still be waiting in the maintenance
    /// queue).
    Exact,
    /// The value could not be read consistently without blocking because a
    /// maintenance run was concurrently publishing new counters. It is a
    /// best-effort estimate that may mix values from adjacent maintenance
    /// generations.
    Approximate,
    /// No maintenance run has completed yet, so the value is the initial
    /// estimate maintained by the cache and likely lags behind the actual
    /// cache contents.
    Unmaintained,
}

/// The status of a cache's maintenance at the time of a [`CacheSnapshot`].
///
/// All values in this status are instantaneous observations; they may already
/// be outdated by the time they are inspected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaintenanceStatus {
    generation: u64,
    pending_read_ops: usize,
    pending_write_ops: usize,
    is_running: bool,
}

impl MaintenanceStatus {
    pub(crate) fn new(
        generation: u64,
        pending_read_ops: usize,
        pending_write_ops: usize,
        is_running: bool,
    ) -> Self {
        Self {
            generation,
            pending_read_ops,
            pending_write_ops,
            is_running,
        }
    }

    /// Returns the logical maintenance generation: the number of maintenance
    /// runs completed by the cache so far.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the number of read operations waiting in the maintenance
    /// queue.
    pub fn pending_read_ops(&self) -> usize {
        self.pending_read_ops
    }

    /// Returns the number of write operations waiting in the maintenance
    /// queue.
    pub fn pending_write_ops(&self) -> usize {
        self.pending_write_ops
    }

    /// Returns `true` if there are any operations waiting in the maintenance
    /// queue, meaning the maintenance is lagging behind the cache writes.
    pub fn has_pending_ops(&self) -> bool {
        self.pending_read_ops > 0 || self.pending_write_ops > 0
    }

    /// Returns `true` if a maintenance run was in progress when the snapshot
    /// was taken.
    pub fn is_running(&self) -> bool {
        self.is_running
    }
}
