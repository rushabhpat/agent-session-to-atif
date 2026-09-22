//! Session discovery and human-readable titles.

use crate::codex;
use crate::common::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const AGENTS: [&str; 3] = ["cursor", "claude", "codex"];

pub struct Found {
    pub agent: String,
    pub path: PathBuf,
    pub cwd: Option<String>,
    pub id: String,
    pub project: Option<String>,
    pub mtime: f64,
    pub size: u64,
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

/// Reproduce how Cursor and Claude Code name a project directory.
///
/// Every non-alphanumeric character becomes '-', so '/home/me/x.dev' and
/// '/home/me/x-dev' collapse to the same slug. The mapping is therefore
/// one-way: we encode the cwd to find its directory and never decode a slug
/// back into a path, which would invent paths that do not exist.
pub fn project_slug(cwd: &str) -> String {
    cwd.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

fn stat(path: &Path) -> (f64, u64) {
    match std::fs::metadata(path) {
        Ok(m) => {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            (mtime, m.len())
        }
        Err(_) => (0.0, 0),
    }
}

/// Read the cwd that a Claude Code log records on its own records.
///
/// Not on the first record: a session opens with title, mode and attachment
/// bookkeeping, and `cwd` only appears once real turns start.
fn claude_log_cwd(path: &Path) -> Option<String> {
    read_jsonl_head(path, 30)
        .iter()
        .find_map(|r| owned_str(r, "cwd"))
}

fn sorted_jsonl(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("jsonl"))
        .collect();
    out.sort();
    out
}

fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out
}

/// Find sessions, newest first. `cwd` restricts results to one project.
pub fn discover(agents: &[String], cwd: Option<&str>) -> Vec<Found> {
    let home = home();
    let mut found: Vec<Found> = Vec::new();
    let want = |name: &str| agents.iter().any(|a| a == name);

    if want("claude") {
        let root = home.join(".claude/projects");
        let dirs = match cwd {
            Some(c) => vec![root.join(project_slug(c))],
            None => subdirs(&root),
        };
        for dir in dirs {
            if !dir.is_dir() {
                continue;
            }
            let project = dir.file_name().and_then(|s| s.to_str()).map(str::to_string);
            for file in sorted_jsonl(&dir) {
                let (mtime, size) = stat(&file);
                let id = file
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                found.push(Found {
                    agent: "claude".into(),
                    cwd: claude_log_cwd(&file),
                    id,
                    project: project.clone(),
                    mtime,
                    size,
                    path: file,
                });
            }
        }
    }

    if want("cursor") {
        let root = home.join(".cursor/projects");
        let dirs = match cwd {
            Some(c) => vec![root.join(project_slug(c).trim_start_matches('-'))],
            None => subdirs(&root),
        };
        for dir in dirs {
            let transcripts = dir.join("agent-transcripts");
            if !transcripts.is_dir() {
                continue;
            }
            let project = dir.file_name().and_then(|s| s.to_str()).map(str::to_string);
            // A transcript that is another session's subagent also gets its own
            // top-level directory; listing both would double-count the same work.
            let mut nested: std::collections::HashSet<String> = std::collections::HashSet::new();
            for sub in subdirs(&transcripts) {
                for f in sorted_jsonl(&sub.join("subagents")) {
                    if let Some(stem) = f.file_stem().and_then(|s| s.to_str()) {
                        nested.insert(stem.to_string());
                    }
                }
            }
            for sub in subdirs(&transcripts) {
                let name = sub.file_name().and_then(|s| s.to_str()).unwrap_or("");
                let file = sub.join(format!("{name}.jsonl"));
                if !file.is_file() || nested.contains(name) {
                    continue;
                }
                let (mtime, size) = stat(&file);
                found.push(Found {
                    agent: "cursor".into(),
                    // Cursor logs record no cwd, but the search already knows
                    // which project directory was matched, so reuse it.
                    cwd: cwd.map(str::to_string),
                    id: name.to_string(),
                    project: project.clone(),
                    mtime,
                    size,
                    path: file,
                });
            }
        }
    }

    if want("codex") {
        let root = home.join(".codex/sessions");
        for file in walk_jsonl(&root) {
            let name = file.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !name.starts_with("rollout-") {
                continue;
            }
            let log = codex::log_cwd(&file);
            if let Some(target) = cwd {
                if log.as_deref() != Some(target) {
                    continue;
                }
            }
            let (mtime, size) = stat(&file);
            found.push(Found {
                agent: "codex".into(),
                id: codex::session_id(&file),
                cwd: log,
                project: None,
                mtime,
                size,
                path: file,
            });
        }
    }

    found.sort_by(|a, b| b.mtime.total_cmp(&a.mtime));
    found
}

fn walk_jsonl(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                out.push(path);
            }
        }
    }
    out
}

/// Identify which agent wrote a log, by location then by content.
///
/// Location is authoritative for logs still in their home directory. For copies
/// and fixtures elsewhere, the first records are inspected: guessing by filename
/// would silently hand a log to the wrong adapter and produce an empty export.
pub fn sniff_agent(path: &Path) -> String {
    let text = path.to_string_lossy();
    for (marker, agent) in [
        (".cursor", "cursor"),
        (".claude", "claude"),
        (".codex", "codex"),
    ] {
        if path.components().any(|c| c.as_os_str() == marker) || text.contains(marker) {
            return agent.to_string();
        }
    }
    for rec in read_jsonl_head(path, 10) {
        let kind = str_at(&rec, "type").unwrap_or("");
        if matches!(
            kind,
            "session_meta"
                | "response_item"
                | "event_msg"
                | "turn_context"
                | "function_call"
                | "function_call_output"
                | "reasoning"
        ) {
            return "codex".into();
        }
        if rec.get("uuid").is_some()
            || rec.get("sessionId").is_some()
            || rec.get("isSidechain").is_some()
        {
            return "claude".into();
        }
        if matches!(str_at(&rec, "role"), Some("user" | "assistant")) && rec.get("message").is_some()
        {
            return "cursor".into();
        }
    }
    "codex".into()
}

// ---------------------------------------------------------------- titles

/// A whole message is injected context if it opens with one of these.
const INJECTED: [&str; 12] = [
    "<recommended_plugins",
    "<recommended-plugins",
    "<environment_context",
    "<user_instructions",
    "<ide_",
    "<attached_files",
    "<system-reminder",
    "# agents.md",
    "# files mentioned by the user",
    "caveat:",
    "<local-command",
    "<command-name",
];

/// Recover the user's actual words from a harness-wrapped prompt.
///
/// Prompts arrive wrapped in `<user_query>` or padded with injected context
/// (plugin catalogues, AGENTS.md, attachment manifests). Titles built from raw
/// message text would read identically across unrelated sessions, so injected
/// messages are rejected outright and the caller tries the next user turn.
pub fn clean_prompt(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let body = match (text.find("<user_query>"), text.find("</user_query>")) {
        (Some(a), Some(b)) if b > a + 12 => &text[a + 12..b],
        _ => text,
    };
    let head = body.trim().to_lowercase();
    if INJECTED.iter().any(|n| head.starts_with(n)) {
        return String::new();
    }
    for raw in body.lines() {
        // A self-contained metadata tag carries no prompt; drop it whole rather
        // than unwrapping it, which would surface the metadata value as a title.
        if is_tag_line(raw) {
            continue;
        }
        let line = strip_tags(raw).trim().to_string();
        if line.is_empty() {
            continue;
        }
        let low = line.to_lowercase();
        if INJECTED.iter().any(|n| low.starts_with(n)) {
            continue;
        }
        if line.starts_with("#")
            || line.starts_with("- ")
            || line.starts_with("* ")
            || line.starts_with("/tmp/")
            || line.starts_with("|")
        {
            continue;
        }
        return line;
    }
    String::new()
}

/// True for a line that is nothing but `<tag>...</tag>`.
fn is_tag_line(raw: &str) -> bool {
    let s = raw.trim();
    if !s.starts_with('<') || !s.ends_with('>') {
        return false;
    }
    let Some(open_end) = s.find('>') else {
        return false;
    };
    let name = &s[1..open_end];
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
        return false;
    }
    s.ends_with(&format!("</{name}>"))
}

fn strip_tags(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut depth = 0usize;
    let mut run = 0usize;
    for c in raw.chars() {
        match c {
            '<' => {
                depth += 1;
                run = 0;
            }
            '>' if depth > 0 => {
                depth -= 1;
                run = 0;
            }
            _ if depth > 0 => {
                // Only short spans are treated as tags, matching the Python
                // exporter's `<[^>]{1,40}>`; longer ones are ordinary text.
                run += 1;
                if run > 40 {
                    depth = 0;
                    out.push(c);
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// Codex keeps thread names outside the rollout logs, in a small index.
fn codex_thread_name(id: &str) -> Option<String> {
    let index = home().join(".codex/session_index.jsonl");
    for rec in read_jsonl(&index) {
        if str_at(&rec, "id") == Some(id) {
            if let Some(name) = str_at(&rec, "thread_name") {
                let trimmed = name.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}

/// The whole thread-name index, as a JSON object of id -> name.
pub fn codex_thread_names_json() -> String {
    let index = home().join(".codex/session_index.jsonl");
    let mut map = serde_json::Map::new();
    for rec in read_jsonl(&index) {
        if let (Some(id), Some(name)) = (str_at(&rec, "id"), str_at(&rec, "thread_name")) {
            if !name.trim().is_empty() {
                map.insert(id.to_string(), json!(name.trim()));
            }
        }
    }
    Value::Object(map).to_string()
}

/// Best available human label for a session.
///
/// Claude Code stores a generated `ai-title` (and a `custom-title` when the user
/// set one); Codex keeps thread names in a separate index that covers only
/// recent sessions; Cursor records no title at all. Whatever is missing falls
/// back to the first real user prompt, so every row gets a label. The scan is
/// bounded because listing must stay fast on logs that reach hundreds of MB.
pub fn session_title(path: &Path, agent: &str) -> String {
    if agent == "codex" {
        if let Some(name) = codex_thread_name(&codex::session_id(path)) {
            return name;
        }
    }
    let mut first = String::new();
    let mut fallback = String::new();
    for rec in read_jsonl_head(path, 400) {
        // A title the user set themselves outranks a generated one.
        if str_at(&rec, "type") == Some("custom-title") {
            if let Some(t) = str_at(&rec, "customTitle") {
                if !t.trim().is_empty() {
                    return t.trim().to_string();
                }
            }
        }
        if str_at(&rec, "type") == Some("ai-title") {
            if let Some(t) = str_at(&rec, "aiTitle") {
                if !t.trim().is_empty() {
                    return t.trim().to_string();
                }
            }
        }
        if !first.is_empty() {
            continue;
        }
        let body = match agent {
            "cursor" => {
                if str_at(&rec, "role") == Some("user") {
                    text_of(rec.get("message").and_then(|m| m.get("content")))
                } else {
                    String::new()
                }
            }
            "claude" => {
                let msg = rec.get("message").filter(|m| m.is_object());
                if str_at(&rec, "type") == Some("user") && msg.is_some() {
                    text_of(msg.and_then(|m| m.get("content")))
                } else {
                    String::new()
                }
            }
            _ => {
                let (_, p) = codex::payload(&rec);
                if str_at(p, "type") == Some("message") && str_at(p, "role") == Some("user") {
                    text_of(p.get("content"))
                } else {
                    String::new()
                }
            }
        };
        let prompt = clean_prompt(&body);
        if prompt.is_empty() {
            continue;
        }
        // A bare slash-command is a poor label; keep it only if nothing better follows.
        if prompt.starts_with('/') && !prompt.contains(' ') {
            if fallback.is_empty() {
                fallback = prompt;
            }
            continue;
        }
        first = prompt;
    }
    if first.is_empty() {
        fallback
    } else {
        first
    }
}

impl Found {
    pub fn to_json(&self, with_title: bool) -> Value {
        json!({
            "agent": self.agent,
            "path": self.path.to_string_lossy(),
            "cwd": self.cwd,
            "id": self.id,
            "project": self.project,
            "mtime": self.mtime,
            "size": self.size,
            "title": if with_title { Value::String(session_title(&self.path, &self.agent)) } else { Value::Null },
        })
    }
}
