//! Cursor adapter: `~/.cursor/projects/<slug>/agent-transcripts/<uuid>/<uuid>.jsonl`.
//!
//! Cursor persists only the model-visible conversation: tool calls are recorded
//! but their results are not, so these trajectories carry no observations. ATIF
//! makes `observation` optional, and the discrepancy is declared in `notes`.

use crate::common::*;
use serde_json::{json, Map, Value};
use std::path::Path;

const NOTES: &str = "Converted from a Cursor agent transcript. Cursor records the model-visible \
conversation only, so tool calls have no stored results and these steps carry no `observation`; \
tool_call_ids are synthesised for intra-step referencing. The log has no timestamps or token \
counts, so steps omit `timestamp` and `metrics`. Subagent transcripts are embedded but not \
referenced from a parent tool call, because the log does not record which delegation produced \
which transcript.";

fn build_steps(path: &Path) -> (Vec<Map<String, Value>>, Vec<String>) {
    let mut buf = Steps::new();
    let mut statuses: Vec<String> = Vec::new();

    for rec in read_jsonl(path) {
        if str_at(&rec, "type") == Some("turn_ended") {
            statuses.push(str_at(&rec, "status").unwrap_or("unknown").to_string());
            continue;
        }
        let role = str_at(&rec, "role").unwrap_or("");
        if !matches!(role, "user" | "assistant" | "system") {
            continue;
        }
        let content = rec.get("message").and_then(|m| m.get("content"));
        if role != "assistant" {
            let source = if role == "user" { "user" } else { "system" };
            buf.add(source, json!(text_of(content)), Map::new());
            continue;
        }
        let mut fields = Map::new();
        fields.insert("llm_call_count".into(), json!(1));
        let step_no = buf.steps.len() + 1;
        let idx = buf.add("agent", json!(text_of(content)), fields);
        if let Some(blocks) = content.and_then(Value::as_array) {
            let mut n = 0usize;
            for block in blocks {
                if str_at(block, "type") != Some("tool_use") {
                    continue;
                }
                n += 1;
                // Cursor stores no tool ids, so they are synthesised; they only
                // need to be unique within this trajectory.
                let id =
                    owned_str(block, "id").unwrap_or_else(|| format!("call_{step_no}_{n}"));
                let name = str_at(block, "name").unwrap_or("unknown").to_string();
                buf.add_call(idx, id, &name, as_object(block.get("input")));
            }
        }
    }
    (buf.steps, statuses)
}

pub fn build(path: &Path) -> Value {
    let parent_name = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("session");
    let session = if parent_name == "agent-transcripts" || parent_name.is_empty() {
        stem.to_string()
    } else {
        parent_name.to_string()
    };

    let mut subagents = Vec::new();
    if let Some(dir) = path.parent() {
        let sub_dir = dir.join("subagents");
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&sub_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("jsonl"))
            .collect();
        files.sort();
        for (i, sub) in files.iter().enumerate() {
            let (steps, _) = build_steps(sub);
            if steps.is_empty() {
                continue;
            }
            let mut meta = Meta::new("cursor-subagent");
            meta.trajectory_id = Some(format!("{session}-subagent-{}", i + 1));
            let sub_stem = sub.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            meta.extra.insert("session_id".into(), json!(sub_stem));
            subagents.push(trajectory(meta, steps));
        }
    }

    let (steps, statuses) = build_steps(path);
    let mut meta = Meta::new("cursor");
    meta.session_id = Some(session);
    meta.notes = Some(NOTES.to_string());
    meta.extra.insert(
        "source_log".into(),
        json!(path.to_string_lossy().to_string()),
    );
    if !statuses.is_empty() {
        meta.extra.insert("turn_statuses".into(), json!(statuses));
    }
    meta.subagents = subagents;
    trajectory(meta, steps)
}
