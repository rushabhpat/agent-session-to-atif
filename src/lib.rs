//! PyO3 bindings for the ATIF exporter core.
//!
//! The boundary deliberately passes JSON strings rather than rich Python
//! objects: one `serde_json` serialisation here and one `json.loads` on the
//! Python side keeps the FFI surface to a handful of functions, instead of
//! hand-writing conversions for every nested ATIF type.

mod claude;
mod codex;
mod common;
mod cursor;
mod discover;
mod validate;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Build an ATIF trajectory from a session log. Returns JSON text.
#[pyfunction]
#[pyo3(signature = (path, agent=None))]
fn build(py: Python<'_>, path: &str, agent: Option<&str>) -> PyResult<String> {
    let p = PathBuf::from(path);
    if !p.exists() {
        return Err(PyValueError::new_err(format!("no such log: {path}")));
    }
    let kind = match agent {
        Some(a) => a.to_string(),
        None => discover::sniff_agent(&p),
    };
    // Parsing a large log takes seconds; release the GIL so the caller stays
    // responsive and threaded callers actually run in parallel.
    let traj = py.allow_threads(move || match kind.as_str() {
        "claude" => claude::build(&p),
        "cursor" => cursor::build(&p),
        "codex" => codex::build(&p),
        other => Value::String(format!("__unknown_agent__{other}")),
    });
    if let Some(s) = traj.as_str() {
        if let Some(name) = s.strip_prefix("__unknown_agent__") {
            return Err(PyValueError::new_err(format!("unknown agent: {name}")));
        }
    }
    Ok(traj.to_string())
}

/// Validate a trajectory given as JSON text. Returns the list of errors.
#[pyfunction]
fn validate_json(text: &str) -> PyResult<Vec<String>> {
    let traj: Value = serde_json::from_str(text)
        .map_err(|e| PyValueError::new_err(format!("invalid JSON: {e}")))?;
    Ok(validate::validate(&traj))
}

/// Validate a trajectory file on disk. Returns the list of errors.
#[pyfunction]
fn validate_file(path: &str) -> PyResult<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?;
    validate_json(&text)
}

/// Discover sessions. Returns JSON text: a list of session records.
#[pyfunction]
#[pyo3(signature = (cwd=None, agents=None, titles=false))]
fn discover_sessions(
    py: Python<'_>,
    cwd: Option<String>,
    agents: Option<Vec<String>>,
    titles: bool,
) -> PyResult<String> {
    let agents = agents.unwrap_or_else(|| {
        discover::AGENTS
            .iter()
            .map(|s| (*s).to_string())
            .collect::<Vec<String>>()
    });
    let out = py.allow_threads(move || {
        let found = discover::discover(&agents, cwd.as_deref());
        let items: Vec<Value> = found.iter().map(|f| f.to_json(titles)).collect();
        Value::Array(items).to_string()
    });
    Ok(out)
}

/// Best available human label for one session log.
#[pyfunction]
fn session_title(py: Python<'_>, path: &str, agent: &str) -> String {
    let p = PathBuf::from(path);
    let agent = agent.to_string();
    py.allow_threads(move || discover::session_title(&p, &agent))
}

/// Identify which agent wrote a log.
#[pyfunction]
fn sniff_agent(path: &str) -> String {
    discover::sniff_agent(Path::new(path))
}

/// The session id for a log, extracted the way discovery does it.
///
/// A Codex rollout is named `rollout-<timestamp>-<uuid>`, so the filename stem
/// is not the id; for the other two agents the stem is correct.
#[pyfunction]
fn session_id(path: &str, agent: &str) -> String {
    let p = Path::new(path);
    if agent == "codex" {
        return codex::session_id(p);
    }
    p.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("session")
        .to_string()
}

/// Reproduce an agent's project-directory slug for a working directory.
#[pyfunction]
fn project_slug(cwd: &str) -> String {
    discover::project_slug(cwd)
}

/// Strip harness wrapping from a prompt, for titles.
#[pyfunction]
fn clean_prompt(text: &str) -> String {
    discover::clean_prompt(text)
}

/// Read the cwd a Codex rollout records, from either log generation.
#[pyfunction]
fn codex_log_cwd(path: &str) -> Option<String> {
    codex::log_cwd(Path::new(path))
}

/// Thread names Codex keeps outside the rollout logs, as a JSON object.
#[pyfunction]
fn codex_thread_names() -> String {
    discover::codex_thread_names_json()
}

/// Normalise a timestamp to ISO-8601, or return None if unparseable.
#[pyfunction]
fn iso(raw: &str) -> Option<String> {
    common::iso(raw)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SCHEMA", common::SCHEMA)?;
    m.add("ACCEPTED", common::accepted_versions())?;
    m.add("AGENTS", discover::AGENTS.to_vec())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add_function(wrap_pyfunction!(validate_json, m)?)?;
    m.add_function(wrap_pyfunction!(validate_file, m)?)?;
    m.add_function(wrap_pyfunction!(discover_sessions, m)?)?;
    m.add_function(wrap_pyfunction!(session_title, m)?)?;
    m.add_function(wrap_pyfunction!(sniff_agent, m)?)?;
    m.add_function(wrap_pyfunction!(session_id, m)?)?;
    m.add_function(wrap_pyfunction!(project_slug, m)?)?;
    m.add_function(wrap_pyfunction!(clean_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(codex_log_cwd, m)?)?;
    m.add_function(wrap_pyfunction!(codex_thread_names, m)?)?;
    m.add_function(wrap_pyfunction!(iso, m)?)?;
    Ok(())
}
