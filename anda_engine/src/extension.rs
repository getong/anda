//! Built-in tool and agent extensions.
//!
//! Extensions are optional building blocks that can be registered with an
//! [`EngineBuilder`](crate::engine::EngineBuilder) or used directly in tests.
//! They cover common runtime needs such as structured extraction, web fetching,
//! workspace filesystem access, shell execution, notes, skills, todos, and
//! search.
//!
//! # Key Components
//! - [`fetch`]: signed HTTP fetching and resource loading.
//! - [`fs`]: workspace-scoped file read, write, search, and edit tools.
//! - [`note`]: lightweight per-agent note storage.
//! - [`shell`]: native or sandboxed command execution.
//! - [`skill`]: file-backed skill loading and lifecycle management.
//! - [`todo`]: session-scoped task tracking for long-running agents.

pub mod fetch;
pub mod fs;
pub mod note;
pub mod shell;
pub mod skill;
pub mod todo;

#[deprecated(
    since = "0.12.0",
    note = "The `google` extension is deprecated and will be removed in a future release."
)]
pub mod google;

#[deprecated(
    since = "0.12.0",
    note = "The `extractor` extension is deprecated and will be removed in a future release."
)]
pub mod extractor;
