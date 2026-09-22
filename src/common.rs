//! Shared pieces: JSONL reading, JSON helpers, step buffer, trajectory assembly.
//!
//! The adapters all face the same three problems -- logs that may be truncated
//! mid-write, tool arguments that are not objects, and tool results that arrive
//! after the call they belong to -- so those are solved once here.

use serde_json::{json, Map, Value};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// v1.8 only adds audio, which we never emit; v1.7 has the widest server support.
pub const SCHEMA: &str = "ATIF-v1.7";

pub fn accepted_versions() -> Vec<String> {
    (0..=8).map(|n| format!("ATIF-v1.{n}")).collect()
}

/// Parse a JSONL file, skipping blank and malformed lines.
///
/// Live session logs are appended to concurrently, so a truncated final line is
/// normal rather than exceptional; a strict parse would make export flaky.
pub fn read_jsonl(path: &Path) -> Vec<Value> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    BufReader::with_capacity(1 << 20, file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect()
}

/// Read at most `limit` records, for probes that must stay fast on huge logs.
pub fn read_jsonl_head(path: &Path, limit: usize) -> Vec<Value> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    BufReader::with_capacity(1 << 16, file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .take(limit)
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect()
}

pub fn str_at<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key)?.as_str()
}

pub fn owned_str(v: &Value, key: &str) -> Option<String> {
    str_at(v, key).map(str::to_string)
}

/// Flatten a content array (or bare string) into plain text.
pub fn text_of(v: Option<&Value>) -> String {
    let Some(v) = v else { return String::new() };
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    let Some(items) = v.as_array() else {
        return String::new();
    };
    let mut parts: Vec<&str> = Vec::new();
    for item in items {
        if let Some(s) = item.as_str() {
            parts.push(s);
        } else if let Some(kind) = str_at(item, "type") {
            let wanted = matches!(kind, "text" | "output_text" | "input_text");
            if wanted {
                if let Some(t) = str_at(item, "text") {
                    parts.push(t);
                }
            }
        }
    }
    parts.join("\n")
}

/// Flatten content, accepting an explicit set of block types.
pub fn text_of_kinds(v: Option<&Value>, kinds: &[&str]) -> String {
    let Some(v) = v else { return String::new() };
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    let Some(items) = v.as_array() else {
        return String::new();
    };
    let mut parts: Vec<&str> = Vec::new();
    for item in items {
        if let Some(kind) = str_at(item, "type") {
            if kinds.contains(&kind) {
                if let Some(t) = str_at(item, "text") {
                    parts.push(t);
                }
            }
        }
    }
    parts.join("\n")
}

/// Render a value as a string, JSON-encoding anything that is not already text.
///
/// Non-string values use Python's `json.dumps` separators (", " and ": ") so
/// that observation contents are byte-identical to the reference exporter's.
pub fn stringify(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => py_json(other),
    }
}

/// Serialise with Python's default `json.dumps` spacing.
pub fn py_json(v: &Value) -> String {
    let mut out = String::new();
    write_py_json(v, &mut out);
    out
}

fn write_py_json(v: &Value, out: &mut String) {
    match v {
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push_str(": ");
                write_py_json(val, out);
            }
            out.push('}');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// Coerce tool arguments into a JSON object, which ATIF requires.
///
/// Codex stores `arguments` as a JSON *string*; malformed or non-object values
/// are preserved under `_raw` rather than dropped, so nothing is ever lost.
pub fn as_object(v: Option<&Value>) -> Value {
    match v {
        None | Some(Value::Null) => json!({}),
        Some(Value::Object(map)) => Value::Object(map.clone()),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(map)) => Value::Object(map),
            Ok(other) => json!({ "_raw": other }),
            Err(_) => json!({ "_raw": s }),
        },
        Some(other) => json!({ "_raw": other.clone() }),
    }
}

/// Drop null, empty-object and empty-array members.
pub fn prune(map: Map<String, Value>) -> Map<String, Value> {
    map.into_iter()
        .filter(|(_, v)| match v {
            Value::Null => false,
            Value::Object(m) => !m.is_empty(),
            Value::Array(a) => !a.is_empty(),
            _ => true,
        })
        .collect()
}

/// Normalise a timestamp to ISO-8601 with an explicit offset.
///
/// Accepts the forms the three agents actually write (`...Z`, an explicit
/// offset, or a naive local stamp) and rejects anything else, because an
/// unparseable timestamp must not reach the output.
pub fn iso(raw: &str) -> Option<String> {
    let s = raw.trim();
    if s.len() < 19 {
        return None;
    }
    let b = s.as_bytes();
    let digits_ok = b[0..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit)
        && (b[10] == b'T' || b[10] == b' ')
        && b[11..13].iter().all(u8::is_ascii_digit)
        && b[13] == b':'
        && b[14..16].iter().all(u8::is_ascii_digit)
        && b[16] == b':'
        && b[17..19].iter().all(u8::is_ascii_digit);
    if !digits_ok {
        return None;
    }
    // Shape alone is not enough: an out-of-range field means the log is not
    // actually carrying a timestamp, and passing it through would put an
    // impossible date into the output.
    let field = |a: usize, b: usize| s[a..b].parse::<u32>().unwrap_or(u32::MAX);
    let (month, day) = (field(5, 7), field(8, 10));
    let (hour, minute, second) = (field(11, 13), field(14, 16), field(17, 19));
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let (head, tail) = s.split_at(19);
    let head = head.replace(' ', "T");
    // Fractional seconds, then an optional zone.
    let rest = tail.trim_start_matches(|c: char| c == '.' || c.is_ascii_digit());
    let raw_frac = &tail[..tail.len() - rest.len()];
    // Normalise to microseconds, matching `datetime.fromisoformat().isoformat()`:
    // it prints 6 digits, or none at all when the value is zero.
    let frac = if raw_frac.is_empty() {
        String::new()
    } else {
        let digits: String = raw_frac.trim_start_matches('.').chars().take(6).collect();
        let padded = format!("{digits:0<6}");
        if padded == "000000" {
            String::new()
        } else {
            format!(".{padded}")
        }
    };
    let zone = match rest {
        "" => "+00:00".to_string(),
        "Z" | "z" => "+00:00".to_string(),
        z if z.starts_with('+') || z.starts_with('-') => {
            if z.len() == 5 {
                format!("{}:{}", &z[..3], &z[3..]) // +0530 -> +05:30
            } else {
                z.to_string()
            }
        }
        _ => return None,
    };
    Some(format!("{head}{frac}{zone}"))
}

/// Wrappers a CLI writes to narrate itself, which are not conversation.
///
/// `/clear` produces a caveat, a command echo and an empty stdout block: the
/// user typed a slash-command, they did not say those words to the model, and
/// no inference happened. Emitting them as steps invents turns that never
/// occurred, so they are dropped.
const PLUMBING: [&str; 6] = [
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
];

/// Context injected into the prompt that the user did not author.
///
/// Unlike plumbing this really was in the model's context and shaped its
/// behaviour, so it is kept -- but attributed to `system` rather than left
/// masquerading as something the user said.
const INJECTED: [&str; 6] = [
    "<system-reminder>",
    "<task-notification>",
    "<environment_context>",
    "<skills_instructions>",
    "<multi_agent_",
    "<recommended_plugins>",
];

/// Who actually authored a message the log attributes to the user.
///
/// Returns `"user"`, `"system"` for injected context, or `"plumbing"` for CLI
/// self-narration that should not become a step at all.
pub fn authored_by(text: &str) -> &'static str {
    let head = text.trim_start();
    if head.is_empty() {
        return "user";
    }
    let lower = head.to_lowercase();
    if PLUMBING.iter().any(|p| lower.starts_with(p)) {
        return "plumbing";
    }
    if INJECTED.iter().any(|p| lower.starts_with(p)) {
        return "system";
    }
    "user"
}

/// Accumulates steps and resolves tool-call -> observation pairing.
///
/// Every agent writes a tool's result *after* the call, sometimes several
/// records later and sometimes out of order. Recording which step owns each
/// tool_call_id lets observations be attached whenever they arrive, which keeps
/// `source_call_id` references valid -- something the ATIF validator enforces.
#[derive(Default)]
pub struct Steps {
    pub steps: Vec<Map<String, Value>>,
    /// tool_call_id -> index into `steps`
    owner: std::collections::HashMap<String, usize>,
}

impl Steps {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a step; returns its index.
    pub fn add(&mut self, source: &str, message: Value, mut fields: Map<String, Value>) -> usize {
        let mut step = Map::new();
        step.insert("source".into(), json!(source));
        for (k, v) in fields.iter() {
            step.insert(k.clone(), v.clone());
        }
        fields.clear();
        let mut step = prune(step);
        // `message` is required and may legitimately be an empty string, so it
        // is inserted after pruning.
        step.insert("message".into(), message);
        self.steps.push(step);
        self.steps.len() - 1
    }

    /// Attach a tool call to an existing step and record who owns it.
    ///
    /// Ids are made unique on collision so a malformed log can never emit an
    /// invalid trajectory; ATIF requires tool_call_ids to be unambiguous.
    pub fn add_call(&mut self, idx: usize, mut id: String, name: &str, arguments: Value) {
        if self.owner.contains_key(&id) {
            let mut n = 2;
            while self.owner.contains_key(&format!("{id}-{n}")) {
                n += 1;
            }
            id = format!("{id}-{n}");
        }
        self.owner.insert(id.clone(), idx);
        let call = json!({
            "tool_call_id": id,
            "function_name": name,
            "arguments": arguments,
        });
        self.steps[idx]
            .entry("tool_calls")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .expect("tool_calls is an array")
            .push(call);
    }

    /// Attach a result to whichever step issued `call_id`.
    pub fn observe(&mut self, call_id: &str, content: Value, extra: Map<String, Value>) -> bool {
        let Some(&idx) = self.owner.get(call_id) else {
            return false;
        };
        let mut result = Map::new();
        result.insert("source_call_id".into(), json!(call_id));
        let extra = prune(extra);
        if !extra.is_empty() {
            result.insert("extra".into(), Value::Object(extra));
        }
        result.insert("content".into(), content);
        self.steps[idx]
            .entry("observation")
            .or_insert_with(|| json!({ "results": [] }))
            .get_mut("results")
            .and_then(Value::as_array_mut)
            .expect("results is an array")
            .push(Value::Object(result));
        true
    }

    pub fn last_agent(&mut self) -> Option<usize> {
        (0..self.steps.len())
            .rev()
            .find(|&i| self.steps[i].get("source").and_then(Value::as_str) == Some("agent"))
    }

    pub fn set(&mut self, idx: usize, key: &str, value: Value) {
        self.steps[idx].insert(key.into(), value);
    }

    pub fn get<'a>(&'a self, idx: usize, key: &str) -> Option<&'a Value> {
        self.steps[idx].get(key)
    }
}

const STEP_ORDER: [&str; 12] = [
    "step_id",
    "timestamp",
    "source",
    "model_name",
    "reasoning_effort",
    "message",
    "reasoning_content",
    "tool_calls",
    "observation",
    "metrics",
    "llm_call_count",
    "extra",
];

/// Emit step keys in schema order so diffs and manual review read naturally.
fn order_step(step: Map<String, Value>) -> Value {
    let mut out = Map::new();
    for key in STEP_ORDER {
        if let Some(v) = step.get(key) {
            out.insert(key.to_string(), v.clone());
        }
    }
    for (k, v) in step {
        if !STEP_ORDER.contains(&k.as_str()) {
            out.insert(k, v);
        }
    }
    Value::Object(out)
}

pub struct Meta {
    pub agent: String,
    pub version: Option<String>,
    pub model: Option<String>,
    pub session_id: Option<String>,
    pub trajectory_id: Option<String>,
    pub notes: Option<String>,
    pub extra: Map<String, Value>,
    pub subagents: Vec<Value>,
}

impl Meta {
    pub fn new(agent: &str) -> Self {
        Self {
            agent: agent.to_string(),
            version: None,
            model: None,
            session_id: None,
            trajectory_id: None,
            notes: None,
            extra: Map::new(),
            subagents: Vec::new(),
        }
    }
}

/// Assemble a trajectory: number the steps and total the metrics.
/// Split a `data:` URL into its media type and base64 payload.
fn data_url(url: &str) -> Option<(&str, &str)> {
    let (meta, data) = url.strip_prefix("data:")?.split_once(',')?;
    Some((meta.strip_suffix(";base64")?, data))
}

/// The image URL on a content part, whether given flat or nested under `url`.
fn image_url(part: &Value) -> Option<&str> {
    match part.get("image_url")? {
        Value::String(s) => Some(s.as_str()),
        nested => str_at(nested, "url"),
    }
}

/// The base64 payload on a Claude-style image part: `source.{media_type,data}`.
fn nested_image(part: &Value) -> Option<(&str, &str)> {
    if part.get("type")?.as_str()? != "image" {
        return None;
    }
    let source = part.get("source")?;
    Some((str_at(source, "media_type")?, str_at(source, "data")?))
}

/// Convert a provider content array into ATIF content parts, if it holds images.
///
/// Codex returns tool output as a list of parts, and a screenshot arrives as an
/// `input_image` carrying a `data:` URL; Claude uses `source.data` instead.
/// Flattening either to text inlines every screenshot as base64: one real
/// session reached 244 MB, 98% of it image data, which no viewer can render and
/// which the uploader rejects for exceeding its 50 MB per-file cap while
/// reporting it as a missing file.
///
/// ATIF's own answer is an image part pointing at a file, so the payload is
/// carried here as base64 and externalised when the trajectory is written, where
/// the destination directory is known. Text-only output is left alone: it is
/// already faithful as a string, and restructuring it would churn every session
/// for no gain.
pub fn content_parts(v: &Value) -> Option<Value> {
    let arr = v.as_array()?;
    if !arr
        .iter()
        .any(|p| image_url(p).is_some() || nested_image(p).is_some())
    {
        return None;
    }
    let mut out: Vec<Value> = Vec::new();
    for part in arr {
        if let Some((media, data)) = nested_image(part) {
            out.push(json!({
                "type": "image",
                "source": { "media_type": media, "data": data },
            }));
            continue;
        }
        match image_url(part) {
            Some(url) => match data_url(url) {
                Some((media, data)) => out.push(json!({
                    "type": "image",
                    "source": { "media_type": media, "data": data },
                })),
                // A remote image: reference it rather than invent a payload.
                None => out.push(json!({ "type": "image", "source": { "path": url } })),
            },
            None => {
                // `text_of` flattens an array; this is one part, so read it
                // directly. Anything unrecognised is stringified rather than
                // dropped: silently losing a part would be worse than noise.
                let text = match part {
                    Value::String(s) => s.clone(),
                    _ => match str_at(part, "text") {
                        Some(t) => t.to_string(),
                        None => stringify(part),
                    },
                };
                if !text.is_empty() {
                    out.push(json!({ "type": "text", "text": text }));
                }
            }
        }
    }
    Some(Value::Array(out))
}

pub fn trajectory(meta: Meta, steps: Vec<Map<String, Value>>) -> Value {
    let mut numbered = steps;
    for (i, step) in numbered.iter_mut().enumerate() {
        step.insert("step_id".into(), json!(i + 1));
    }

    let mut prompt = 0i64;
    let mut completion = 0i64;
    let mut cached = 0i64;
    let mut cost = 0f64;
    for step in &numbered {
        let Some(m) = step.get("metrics") else { continue };
        prompt += m.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(0);
        completion += m
            .get("completion_tokens")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        cached += m.get("cached_tokens").and_then(Value::as_i64).unwrap_or(0);
        cost += m.get("cost_usd").and_then(Value::as_f64).unwrap_or(0.0);
    }
    let mut final_metrics = Map::new();
    if prompt != 0 {
        final_metrics.insert("total_prompt_tokens".into(), json!(prompt));
    }
    if completion != 0 {
        final_metrics.insert("total_completion_tokens".into(), json!(completion));
    }
    if cached != 0 {
        final_metrics.insert("total_cached_tokens".into(), json!(cached));
    }
    if cost != 0.0 {
        final_metrics.insert("total_cost_usd".into(), json!(cost));
    }
    if !numbered.is_empty() {
        final_metrics.insert("total_steps".into(), json!(numbered.len()));
    }

    let mut agent = Map::new();
    agent.insert("name".into(), json!(meta.agent));
    agent.insert(
        "version".into(),
        json!(meta.version.unwrap_or_else(|| "unknown".into())),
    );
    if let Some(model) = meta.model {
        agent.insert("model_name".into(), json!(model));
    }

    let mut out = Map::new();
    out.insert("schema_version".into(), json!(SCHEMA));
    if let Some(sid) = meta.session_id {
        out.insert("session_id".into(), json!(sid));
    }
    if let Some(tid) = meta.trajectory_id {
        out.insert("trajectory_id".into(), json!(tid));
    }
    out.insert("agent".into(), Value::Object(agent));
    out.insert(
        "steps".into(),
        Value::Array(numbered.into_iter().map(order_step).collect()),
    );
    if let Some(notes) = meta.notes {
        out.insert("notes".into(), json!(notes));
    }
    if !final_metrics.is_empty() {
        out.insert("final_metrics".into(), Value::Object(final_metrics));
    }
    let extra = prune(meta.extra);
    if !extra.is_empty() {
        out.insert("extra".into(), Value::Object(extra));
    }
    if !meta.subagents.is_empty() {
        out.insert("subagent_trajectories".into(), json!(meta.subagents));
    }
    Value::Object(out)
}

#[cfg(test)]
#[path = "common_tests.rs"]
mod tests;
