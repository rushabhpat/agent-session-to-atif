//! Claude Code adapter: `~/.claude/projects/<slug>/<session-uuid>.jsonl`.
//!
//! One LLM response is split across several records that share `message.id`,
//! each carrying an identical copy of `usage`. Grouping on that id restores
//! ATIF's one-step-per-inference convention; summing the repeated usage would
//! inflate token counts several-fold.

use crate::common::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::Path;

const NOTES: &str = "Converted from a Claude Code session log. Assistant records sharing a \
`message.id` are merged into one step, matching ATIF's one-LLM-call-per-step convention. \
`prompt_tokens` is the sum of Anthropic's uncached, cache-write and cache-read input buckets; \
cache-write tokens are kept in `metrics.extra` because they are billed separately.";

/// Anthropic reports uncached input, cache writes and cache reads as three
/// separate numbers; ATIF's `prompt_tokens` is the total of all input tokens
/// with `cached_tokens` a subset of it.
const USAGE_KEYS: [&str; 3] = [
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
];

/// Drop replayed records, keeping the first occurrence.
///
/// Resuming or forking a session re-appends earlier history under the *same*
/// record `uuid`. Those are the same interaction written twice, not two turns,
/// so emitting both would duplicate tool_call_ids and double-count tokens.
pub fn dedupe_by_uuid(records: Vec<Value>) -> Vec<Value> {
    let mut seen = std::collections::HashSet::new();
    records
        .into_iter()
        .filter(|r| match str_at(r, "uuid") {
            Some(uuid) => seen.insert(uuid.to_string()),
            None => true,
        })
        .collect()
}

/// Translate Anthropic's disjoint token buckets into ATIF's nested model.
///
/// Cache-creation tokens are billed at a different rate, so they are preserved
/// in `extra` as the spec directs.
fn metrics(usage: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    if !usage.is_object() {
        return out;
    }
    let prompt: i64 = USAGE_KEYS
        .iter()
        .map(|k| usage.get(k).and_then(Value::as_i64).unwrap_or(0))
        .sum();
    let mut extra = Map::new();
    if let Some(n) = usage.get("cache_creation_input_tokens").and_then(Value::as_i64) {
        if n != 0 {
            extra.insert("cache_creation_input_tokens".into(), json!(n));
        }
    }
    if let Some(n) = usage
        .get("output_tokens_details")
        .and_then(|d| d.get("thinking_tokens"))
        .and_then(Value::as_i64)
    {
        if n != 0 {
            extra.insert("reasoning_tokens".into(), json!(n));
        }
    }
    if prompt != 0 {
        out.insert("prompt_tokens".into(), json!(prompt));
    }
    if let Some(n) = usage.get("output_tokens").and_then(Value::as_i64) {
        out.insert("completion_tokens".into(), json!(n));
    }
    // Kept even when zero: the provider reported the field, and dropping it
    // would lose the distinction between "no cache hits" and "not measured".
    if let Some(n) = usage.get("cache_read_input_tokens").and_then(Value::as_i64) {
        out.insert("cached_tokens".into(), json!(n));
    }
    if !extra.is_empty() {
        out.insert("extra".into(), Value::Object(extra));
    }
    out
}

/// Prefer the structured tool result, falling back to the model-visible copy.
fn result_content(block: &Value, rec: &Value) -> String {
    let content = block.get("content");
    let empty = match content {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        Some(Value::Array(a)) => a.is_empty(),
        _ => false,
    };
    if empty {
        if let Some(detail) = rec.get("toolUseResult") {
            if let Some(obj) = detail.as_object() {
                for key in ["stdout", "stderr"] {
                    if let Some(s) = obj.get(key).and_then(Value::as_str) {
                        if !s.is_empty() {
                            return s.to_string();
                        }
                    }
                }
                return stringify(detail);
            }
            if !detail.is_null() {
                return stringify(detail);
            }
        }
    }
    match content {
        Some(v @ Value::Array(_)) => text_of(Some(v)),
        Some(v) => stringify(v),
        None => String::new(),
    }
}

/// Fold one record's content blocks into the open agent step.
fn merge_blocks(buf: &mut Steps, idx: usize, content: Option<&Value>) {
    let Some(blocks) = content.and_then(Value::as_array) else {
        return;
    };
    for block in blocks {
        match str_at(block, "type") {
            Some("text") => {
                let body = str_at(block, "text").unwrap_or("");
                if body.is_empty() {
                    continue;
                }
                let prev = buf
                    .get(idx, "message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let merged = if prev.is_empty() {
                    body.to_string()
                } else {
                    format!("{prev}\n{body}").trim().to_string()
                };
                buf.set(idx, "message", json!(merged));
            }
            Some("thinking") => {
                let body = str_at(block, "thinking")
                    .or_else(|| str_at(block, "text"))
                    .unwrap_or("");
                if body.is_empty() {
                    continue;
                }
                let prev = buf
                    .get(idx, "reasoning_content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let merged = format!("{prev}\n{body}").trim().to_string();
                buf.set(idx, "reasoning_content", json!(merged));
            }
            Some("tool_use") => {
                let existing = buf
                    .get(idx, "tool_calls")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);
                let id = owned_str(block, "id").unwrap_or_else(|| format!("call_{existing}"));
                let name = str_at(block, "name").unwrap_or("unknown").to_string();
                buf.add_call(idx, id, &name, as_object(block.get("input")));
            }
            _ => {}
        }
    }
}

/// Map a system record, promoting compaction events to ATIF context_management.
fn system_step(buf: &mut Steps, rec: &Value, ts: Option<String>) {
    let body = str_at(rec, "content").unwrap_or("").to_string();
    let subtype = str_at(rec, "subtype");
    // CLI self-narration, not conversation -- see PLUMBING. Checked regardless of
    // subtype: these records carry `subtype: "local_command"`, and only the
    // compaction boundary below is a real event worth keeping.
    if subtype != Some("compact_boundary") && authored_by(&body) == "plumbing" {
        return;
    }
    let mut extra = Map::new();
    if let Some(sub) = subtype {
        extra.insert("subtype".into(), json!(sub));
    }
    if subtype == Some("compact_boundary") {
        let meta = rec.get("compactMetadata");
        let mut cm = Map::new();
        cm.insert("type".into(), json!("compaction"));
        cm.insert("boundary".into(), json!("replace"));
        if let Some(m) = meta {
            if let Some(t) = str_at(m, "trigger") {
                cm.insert("trigger".into(), json!(t));
            }
            if let Some(n) = m.get("preTokens").and_then(Value::as_i64) {
                cm.insert("pre_tokens".into(), json!(n));
            }
            if let Some(n) = m.get("postTokens").and_then(Value::as_i64) {
                cm.insert("post_tokens".into(), json!(n));
            }
        }
        extra.insert("context_management".into(), Value::Object(cm));
    }
    if body.trim().is_empty() && extra.is_empty() {
        return;
    }
    let mut fields = Map::new();
    if let Some(t) = ts {
        fields.insert("timestamp".into(), json!(t));
    }
    if !extra.is_empty() {
        fields.insert("extra".into(), Value::Object(extra));
    }
    buf.add("system", json!(body), fields);
}

/// Convert one Claude conversation (main thread or a single sidechain).
fn build_steps(records: &[Value]) -> (Vec<Map<String, Value>>, Option<String>) {
    let mut buf = Steps::new();
    let mut model: Option<String> = None;
    let mut open: Option<(String, usize)> = None; // (message.id, step index)

    for rec in records {
        let kind = str_at(rec, "type").unwrap_or("");
        let msg = rec.get("message").filter(|m| m.is_object());
        let ts = str_at(rec, "timestamp").and_then(iso);

        if kind == "system" {
            open = None;
            system_step(&mut buf, rec, ts);
            continue;
        }

        if kind == "assistant" {
            let Some(msg) = msg else { continue };
            let mid = str_at(msg, "id").unwrap_or("").to_string();
            let reopen = match &open {
                Some((prev, _)) => *prev != mid,
                None => true,
            };
            if reopen {
                if let Some(m) = str_at(msg, "model") {
                    model = Some(m.to_string());
                }
                let mut fields = Map::new();
                if let Some(t) = &ts {
                    fields.insert("timestamp".into(), json!(t));
                }
                if let Some(m) = str_at(msg, "model") {
                    fields.insert("model_name".into(), json!(m));
                }
                fields.insert("llm_call_count".into(), json!(1));
                if let Some(e) = rec.get("effort").filter(|v| !v.is_null()) {
                    fields.insert("reasoning_effort".into(), e.clone());
                }
                let m = metrics(msg.get("usage").unwrap_or(&Value::Null));
                if !m.is_empty() {
                    fields.insert("metrics".into(), Value::Object(m));
                }
                let idx = buf.add("agent", json!(""), fields);
                open = Some((mid, idx));
            }
            if let Some((_, idx)) = open {
                merge_blocks(&mut buf, idx, msg.get("content"));
            }
            continue;
        }

        // kind == "user": tool results are observations; anything else is a turn.
        let content = msg.and_then(|m| m.get("content"));
        let results: Vec<&Value> = content
            .and_then(Value::as_array)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|b| str_at(b, "type") == Some("tool_result"))
                    .collect()
            })
            .unwrap_or_default();
        let had_results = !results.is_empty();
        for block in results {
            let mut extra = Map::new();
            if block.get("is_error").and_then(Value::as_bool) == Some(true) {
                extra.insert("is_error".into(), json!(true));
            }
            let id = str_at(block, "tool_use_id").unwrap_or("").to_string();
            buf.observe(&id, result_content(block, rec), extra);
        }
        let body = text_of(content);
        if !body.trim().is_empty() || !had_results {
            open = None;
            let source = authored_by(&body);
            if source == "plumbing" {
                continue;
            }
            let mut fields = Map::new();
            if let Some(t) = ts {
                fields.insert("timestamp".into(), json!(t));
            }
            buf.add(source, json!(body), fields);
        }
    }
    (buf.steps, model)
}

/// Split sidechain records into one embedded trajectory per delegated task.
///
/// Sidechain records form parentUuid chains; a record whose parent is not itself
/// a sidechain starts a new subagent conversation.
fn subagents(records: &[Value], parent: &str) -> Vec<Value> {
    let own: std::collections::HashSet<&str> =
        records.iter().filter_map(|r| str_at(r, "uuid")).collect();
    let mut root_of: HashMap<String, String> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<Value>> = HashMap::new();

    for rec in records {
        let parent_uuid = str_at(rec, "parentUuid").unwrap_or("");
        let root = if own.contains(parent_uuid) {
            root_of.get(parent_uuid).cloned()
        } else {
            None
        }
        .unwrap_or_else(|| {
            str_at(rec, "uuid")
                .map(str::to_string)
                .unwrap_or_else(|| format!("sidechain-{}", groups.len()))
        });
        if let Some(uuid) = str_at(rec, "uuid") {
            root_of.insert(uuid.to_string(), root.clone());
        }
        if !groups.contains_key(&root) {
            order.push(root.clone());
        }
        groups.entry(root).or_default().push(rec.clone());
    }

    let mut out = Vec::new();
    for (i, root) in order.iter().enumerate() {
        let group = &groups[root];
        let (steps, model) = build_steps(group);
        if steps.is_empty() {
            continue;
        }
        let mut meta = Meta::new("claude-code-subagent");
        meta.model = model;
        meta.trajectory_id = Some(format!("{parent}-subagent-{}", i + 1));
        meta.extra.insert("sidechain_root_uuid".into(), json!(root));
        out.push(trajectory(meta, steps));
    }
    out
}

pub fn build(path: &Path) -> Value {
    let records: Vec<Value> = dedupe_by_uuid(
        read_jsonl(path)
            .into_iter()
            .filter(|r| matches!(str_at(r, "type"), Some("user" | "assistant" | "system")))
            .collect(),
    );
    let (main, side): (Vec<Value>, Vec<Value>) = records
        .iter()
        .cloned()
        .partition(|r| r.get("isSidechain").and_then(Value::as_bool) != Some(true));

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("session");
    let version = records
        .iter()
        .rev()
        .find_map(|r| owned_str(r, "version"));
    let cwd = records.iter().rev().find_map(|r| owned_str(r, "cwd"));
    let session_id = records
        .iter()
        .find_map(|r| owned_str(r, "sessionId").or_else(|| owned_str(r, "session_id")))
        .unwrap_or_else(|| stem.to_string());

    let (steps, model) = build_steps(&main);
    let mut meta = Meta::new("claude-code");
    meta.version = version;
    meta.model = model;
    meta.session_id = Some(session_id);
    meta.notes = Some(NOTES.to_string());
    meta.extra.insert(
        "source_log".into(),
        json!(path.to_string_lossy().to_string()),
    );
    if let Some(c) = cwd {
        meta.extra.insert("cwd".into(), json!(c));
    }
    meta.subagents = subagents(&side, stem);
    trajectory(meta, steps)
}
