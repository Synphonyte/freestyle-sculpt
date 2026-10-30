//! The crate's hash maps and sets.
//!
//! Both aliases use a fixed-seed [`foldhash`] builder, so iteration order is a
//! pure function of the insertion sequence and a weld produces the same result
//! on every run. `hashbrown`'s `default-hasher` feature is disabled in
//! `Cargo.toml`, so a randomly seeded map cannot be introduced by accident - a
//! map or set that does not name the builder here does not compile.

/// A [`hashbrown::HashMap`] with the crate's fixed-seed builder.
pub type HashMap<K, V> = ::hashbrown::HashMap<K, V, foldhash::fast::FixedState>;

/// A [`hashbrown::HashSet`] with the crate's fixed-seed builder.
pub type HashSet<T> = ::hashbrown::HashSet<T, foldhash::fast::FixedState>;
