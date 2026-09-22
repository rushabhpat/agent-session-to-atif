//! Codex adapter: `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl`.
//!
//! Two on-disk generations exist: newer records wrap the event in `payload`,
//! older ones are flat. Unwrapping first lets a single pass handle both.

use crate::common::*;
use serde_json::{json, Map, Value};
use std::path::Path;

const NOTES: &str = "Converted from a Codex rollout log. Each `function_call` becomes its own step \
because the log records no response-level grouping; `reasoning` items are attached to the action \
they precede. Per-step metrics come from the turn's `last_token_usage`.";

const TOOL_CALLS: [&str; 4] = [
    "function_call",
    "local_shell_call",
    "custom_tool_call",
    "web_search_call",
];

/// Return (event_type, payload) for either Codex log generation.
pub fn payload(rec: &Value) -> (&str, &Value) {
    match rec.get("payload") {
        Some(p) if p.is_object() => (str_at(rec, "type").unwrap_or(""), p),
        _ => (
            if rec.get("type").is_some() {
                "response_item"
            } else {
                ""
            },
            rec,
        ),
    }
}

/// Map a Codex token_usage block.
///
/// Unlike Anthropic, Codex's `input_tokens` already includes cache hits, which
/// is exactly ATIF's definition of `prompt_tokens`.
fn usage(info: &Value, key: &str) -> Map<String, Value> {
    let mut out = Map::new();
    let Some(u) = info.get(key).filter(|v| v.is_object()) else {
        return out;
    };
    let mut extra = Map::new();
    for (src, dst) in [
        ("reasoning_output_tokens", "reasoning_tokens"),
        ("cache_write_input_tokens", "cache_write_input_tokens"),
    ] {
        if let Some(n) = u.get(src).and_then(Value::as_i64) {
            if n != 0 {
                extra.insert(dst.into(), json!(n));
            }
        }
    }
    for (src, dst) in [
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("cached_input_tokens", "cached_tokens"),
    ] {
        if let Some(n) = u.get(src).and_then(Value::as_i64) {
            out.insert(dst.into(), json!(n));
        }
    }
    if !extra.is_empty() {
        out.insert("extra".into(), Value::Object(extra));
    }
    out
}

/// Build a tool call, normalising Codex's shell/custom variants onto one shape.
fn tool_call(p: &Value, ptype: &str, index: usize) -> (String, String, Value) {
    let (name, args) = match ptype {
        "local_shell_call" => {
            let action = p.get("action").filter(|a| a.is_object());
            (
                "local_shell".to_string(),
                as_object(action.or(Some(p))),
            )
        }
        "web_search_call" => {
            let action = p.get("action").filter(|a| a.is_object()).cloned();
            let args = action.unwrap_or_else(|| json!({ "query": p.get("query") }));
            ("web_search".to_string(), as_object(Some(&args)))
        }
        "custom_tool_call" => (
            str_at(p, "name").unwrap_or(ptype).to_string(),
            as_object(p.get("input")),
        ),
        _ => (
            str_at(p, "name").unwrap_or(ptype).to_string(),
            as_object(p.get("arguments")),
        ),
    };
    let id = owned_str(p, "call_id")
        .or_else(|| owned_str(p, "id"))
        .unwrap_or_else(|| format!("call_{index}"));
    (id, name, args)
}

/// Extract the session uuid from a rollout filename.
pub fn session_id(path: &Path) -> String {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    // rollout-<ISO timestamp>-<uuid>.jsonl; the uuid is the last 5 dash groups.
    let stem = name.strip_suffix(".jsonl").unwrap_or(name);
    let parts: Vec<&str> = stem.split('-').collect();
    if parts.len() >= 5 {
        let tail = &parts[parts.len() - 5..];
        let looks_uuid = tail[0].len() == 8
            && tail[1].len() == 4
            && tail[2].len() == 4
            && tail[3].len() == 4
            && tail[4].len() == 12
            && tail
                .iter()
                .all(|s| s.chars().all(|c| c.is_ascii_hexdigit()));
        if looks_uuid {
            return tail.join("-");
        }
    }
    stem.to_string()
}

/// Read the cwd from a Codex log without parsing the whole rollout.
///
/// Newer logs put it in `session_meta`; older ones only reveal it inside the
/// injected `<environment_context>` message, so both are checked over the first
/// few records rather than reading files that can reach hundreds of megabytes.
pub fn log_cwd(path: &Path) -> Option<String> {
    for rec in read_jsonl_head(path, 60) {
        let (_, p) = payload(&rec);
        if let Some(cwd) = owned_str(p, "cwd") {
            return Some(cwd);
        }
        if p.get("content").is_some() {
            let text = text_of(p.get("content"));
            if let Some(cwd) = extract_cwd(&text) {
                return Some(cwd);
            }
        }
    }
    None
}

/// Pull `<cwd>...</cwd>` out of an injected environment block.
pub fn extract_cwd(text: &str) -> Option<String> {
    let start = text.find("<cwd>")? + 5;
    let end = text[start..].find("</cwd>")? + start;
    Some(text[start..end].trim().to_string())
}

pub fn build(path: &Path) -> Value {
    let mut buf = Steps::new();
    let mut model: Option<String> = None;
    let mut version: Option<String> = None;
    let mut cwd: Option<String> = None;
    let mut meta_seen: Option<Value> = None;
    let mut reasoning: Vec<String> = Vec::new();
    let mut totals: Map<String, Value> = Map::new();
    let mut orphans = 0usize;

    for rec in read_jsonl(path) {
        let (kind, p) = payload(&rec);
        let ts = str_at(&rec, "timestamp")
            .or_else(|| str_at(p, "timestamp"))
            .and_then(iso);
        let ptype = str_at(p, "type").unwrap_or("");

        // `session_meta` in the new format; a bare header record in the old one.
        let is_header = kind == "session_meta"
            || (kind.is_empty() && p.get("id").is_some() && p.get("instructions").is_some());
        if is_header {
            if meta_seen.is_none() {
                meta_seen = Some(p.clone());
            }
            if let Some(m) = owned_str(p, "model") {
                model = Some(m);
            }
            if let Some(v) = owned_str(p, "cli_version") {
                version = Some(v);
            }
            if let Some(c) = owned_str(p, "cwd") {
                cwd = Some(c);
            }
            if let Some(instr) = str_at(p, "instructions") {
                if !instr.trim().is_empty() {
                    let mut fields = Map::new();
                    if let Some(t) = ts {
                        fields.insert("timestamp".into(), json!(t));
                    }
                    buf.add("system", json!(instr), fields);
                }
            }
            continue;
        }

        if kind == "turn_context" {
            if let Some(m) = owned_str(p, "model") {
                model = Some(m);
            }
            continue;
        }

        if kind == "event_msg" {
            if ptype == "token_count" {
                if let Some(info) = p.get("info").filter(|v| v.is_object()) {
                    let t = usage(info, "total_token_usage");
                    if !t.is_empty() {
                        totals = t;
                    }
                    // token_count is emitted more than once per turn; only fill
                    // an empty slot so a later repeat cannot overwrite it.
                    let step_usage = usage(info, "last_token_usage");
                    if !step_usage.is_empty() {
                        if let Some(idx) = buf.last_agent() {
                            if buf.get(idx, "metrics").is_none() {
                                buf.set(idx, "metrics", Value::Object(step_usage));
                            }
                        }
                    }
                }
            }
            continue;
        }

        if ptype == "reasoning" {
            let summary = text_of_kinds(p.get("summary"), &["summary_text", "text"]);
            let body = if summary.is_empty() {
                text_of_kinds(p.get("content"), &["reasoning_text", "text"])
            } else {
                summary
            };
            if !body.is_empty() {
                reasoning.push(body);
            }
            continue;
        }

        if ptype == "message" {
            let body = text_of(p.get("content"));
            if str_at(p, "role") == Some("assistant") {
                let mut fields = Map::new();
                if let Some(t) = ts {
                    fields.insert("timestamp".into(), json!(t));
                }
                if let Some(m) = &model {
                    fields.insert("model_name".into(), json!(m));
                }
                fields.insert("llm_call_count".into(), json!(1));
                let idx = buf.add("agent", json!(body), fields);
                if !reasoning.is_empty() {
                    buf.set(idx, "reasoning_content", json!(reasoning.join("\n\n")));
                    reasoning.clear();
                }
            } else {
                if cwd.is_none() {
                    cwd = extract_cwd(&body);
                }
                let mut fields = Map::new();
                if let Some(t) = ts {
                    fields.insert("timestamp".into(), json!(t));
                }
                // Codex injects environment, skills and delegation context as
                // user-role messages. It really was in the model's context, so
                // it is kept -- but attributed to `system` rather than left
                // looking like something the user typed.
                let source = authored_by(&body);
                if source == "plumbing" {
                    continue;
                }
                buf.add(source, json!(body), fields);
            }
            continue;
        }

        if TOOL_CALLS.contains(&ptype) {
            let (id, name, args) = tool_call(p, ptype, buf.steps.len());
            let mut fields = Map::new();
            if let Some(t) = ts {
                fields.insert("timestamp".into(), json!(t));
            }
            if let Some(m) = &model {
                fields.insert("model_name".into(), json!(m));
            }
            fields.insert("llm_call_count".into(), json!(1));
            let idx = buf.add("agent", json!(""), fields);
            buf.add_call(idx, id, &name, args);
            if !reasoning.is_empty() {
                buf.set(idx, "reasoning_content", json!(reasoning.join("\n\n")));
                reasoning.clear();
            }
            continue;
        }

        if matches!(ptype, "function_call_output" | "custom_tool_call_output") {
            let raw = p.get("output");
            let content = match raw {
                Some(Value::Object(map)) => map
                    .get("content")
                    .map(stringify)
                    .unwrap_or_else(|| stringify(&Value::Object(map.clone()))),
                Some(v) => stringify(v),
                None => String::new(),
            };
            let id = str_at(p, "call_id").unwrap_or("");
            if !buf.observe(id, content, Map::new()) {
                orphans += 1;
            }
        }
    }

    // Trailing reasoning with no following action still belongs to the last turn.
    if !reasoning.is_empty() {
        if let Some(idx) = buf.last_agent() {
            let prev = buf
                .get(idx, "reasoning_content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let mut parts: Vec<String> = Vec::new();
            if !prev.is_empty() {
                parts.push(prev);
            }
            parts.extend(reasoning.iter().cloned());
            buf.set(idx, "reasoning_content", json!(parts.join("\n\n")));
        }
    }

    let n_steps = buf.steps.len();
    let mut meta = Meta::new("codex");
    meta.version = version;
    meta.model = model;
    meta.session_id = Some(
        meta_seen
            .as_ref()
            .and_then(|m| owned_str(m, "session_id").or_else(|| owned_str(m, "id")))
            .unwrap_or_else(|| session_id(path)),
    );
    let mut notes = NOTES.to_string();
    if orphans > 0 {
        notes.push_str(&format!(
            " {orphans} tool output(s) had no matching call in this log and were dropped."
        ));
    }
    meta.notes = Some(notes);
    meta.extra.insert(
        "source_log".into(),
        json!(path.to_string_lossy().to_string()),
    );
    if let Some(c) = cwd {
        meta.extra.insert("cwd".into(), json!(c));
    }
    if let Some(m) = &meta_seen {
        for key in ["model_provider", "originator"] {
            if let Some(v) = owned_str(m, key) {
                meta.extra.insert(key.into(), json!(v));
            }
        }
    }

    let mut traj = trajectory(meta, buf.steps);
    // Codex reports authoritative cumulative counts; prefer them over re-summing.
    if !totals.is_empty() {
        let mut fm = Map::new();
        for (src, dst) in [
            ("prompt_tokens", "total_prompt_tokens"),
            ("completion_tokens", "total_completion_tokens"),
            ("cached_tokens", "total_cached_tokens"),
        ] {
            if let Some(v) = totals.get(src) {
                fm.insert(dst.into(), v.clone());
            }
        }
        if n_steps > 0 {
            fm.insert("total_steps".into(), json!(n_steps));
        }
        if let Some(obj) = traj.as_object_mut() {
            if fm.is_empty() {
                obj.remove("final_metrics");
            } else {
                obj.insert("final_metrics".into(), Value::Object(fm));
            }
        }
    }
    traj
}
