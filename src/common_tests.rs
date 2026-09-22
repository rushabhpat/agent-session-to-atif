//! Unit tests for internals that are private to the compiled core.
//!
//! The Python suite covers behaviour through the public API; these cover the
//! pieces it can no longer reach directly, plus the parsing edge cases that are
//! cheapest to pin down here.

use super::*;
use serde_json::json;

#[test]
fn authored_by_separates_plumbing_from_injected_context() {
    // CLI self-narration: no turn happened, so it must not become a step.
    for s in [
        "<local-command-caveat>Caveat: the messages below…</local-command-caveat>",
        "<command-name>/clear</command-name>",
        "<local-command-stdout></local-command-stdout>",
    ] {
        assert_eq!(authored_by(s), "plumbing", "{s:?}");
    }
    // Real context the model acted on: kept, but not attributed to the user.
    for s in [
        "<system-reminder>The user named this session…</system-reminder>",
        "<environment_context><cwd>/tmp</cwd></environment_context>",
        "<task-notification><summary>Goal check-in</summary></task-notification>",
    ] {
        assert_eq!(authored_by(s), "system", "{s:?}");
    }
    // Anything the user actually typed stays theirs.
    for s in ["fix the bug", "", "  what does <T> mean?"] {
        assert_eq!(authored_by(s), "user", "{s:?}");
    }
}

#[test]
fn iso_normalises_the_forms_the_agents_write() {
    assert_eq!(iso("2026-01-02T03:04:05Z").unwrap(), "2026-01-02T03:04:05+00:00");
    assert_eq!(iso("2026-01-02 03:04:05").unwrap(), "2026-01-02T03:04:05+00:00");
    assert_eq!(
        iso("2026-01-02T03:04:05.386000+00:00").unwrap(),
        "2026-01-02T03:04:05.386000+00:00"
    );
    // Fractional seconds are padded to microseconds, as Python's isoformat does.
    assert_eq!(iso("2026-01-02T03:04:05.5Z").unwrap(), "2026-01-02T03:04:05.500000+00:00");
    // A zero fraction prints no fraction at all.
    assert_eq!(iso("2026-01-02T03:04:05.000Z").unwrap(), "2026-01-02T03:04:05+00:00");
    assert_eq!(iso("2026-01-02T03:04:05+0530").unwrap(), "2026-01-02T03:04:05+05:30");
}

#[test]
fn iso_rejects_what_it_cannot_parse() {
    // A bad timestamp must never reach the output, so these return None rather
    // than guessing a value.
    for junk in ["", "last tuesday", "2026-01-02", "not-a-date-at-all", "2026-13-99T99:99:99Z"] {
        assert!(iso(junk).is_none(), "{junk:?} should not parse");
    }
}

#[test]
fn as_object_coerces_every_argument_shape() {
    assert_eq!(as_object(Some(&json!({"a": 1}))), json!({"a": 1}));
    // Codex stores arguments as a JSON string.
    assert_eq!(as_object(Some(&json!(r#"{"a": 1}"#))), json!({"a": 1}));
    assert_eq!(as_object(None), json!({}));
    assert_eq!(as_object(Some(&Value::Null)), json!({}));
    // Anything that is not an object is preserved rather than dropped.
    assert_eq!(as_object(Some(&json!("nope"))), json!({"_raw": "nope"}));
    assert_eq!(as_object(Some(&json!("[1,2]"))), json!({"_raw": [1, 2]}));
    assert_eq!(as_object(Some(&json!(7))), json!({"_raw": 7}));
}

#[test]
fn py_json_matches_python_dumps_spacing() {
    // Observation contents are compared byte-for-byte against the reference
    // exporter, so the separators must match `json.dumps` exactly.
    assert_eq!(py_json(&json!({"a": 1, "b": [1, 2]})), r#"{"a": 1, "b": [1, 2]}"#);
    assert_eq!(py_json(&json!([])), "[]");
    assert_eq!(py_json(&json!({})), "{}");
}

#[test]
fn prune_drops_only_empty_containers_and_null() {
    let mut m = Map::new();
    m.insert("keep_zero".into(), json!(0));
    m.insert("keep_false".into(), json!(false));
    m.insert("keep_empty_string".into(), json!(""));
    m.insert("drop_null".into(), Value::Null);
    m.insert("drop_obj".into(), json!({}));
    m.insert("drop_arr".into(), json!([]));
    let out = prune(m);
    let mut keys: Vec<&String> = out.keys().collect();
    keys.sort();
    assert_eq!(keys, ["keep_empty_string", "keep_false", "keep_zero"]);
}

#[test]
fn text_of_flattens_blocks_and_ignores_non_text() {
    assert_eq!(text_of(Some(&json!("bare"))), "bare");
    assert_eq!(
        text_of(Some(&json!([
            {"type": "text", "text": "one"},
            {"type": "tool_use", "name": "Bash"},
            {"type": "text", "text": "two"}
        ]))),
        "one\ntwo"
    );
    assert_eq!(text_of(None), "");
}

#[test]
fn duplicate_tool_call_ids_are_disambiguated() {
    // ATIF requires unambiguous ids, so a malformed log must still produce a
    // valid trajectory rather than two calls that cannot be told apart.
    let mut steps = Steps::new();
    let a = steps.add("agent", json!(""), Map::new());
    steps.add_call(a, "dup".into(), "Bash", json!({}));
    let b = steps.add("agent", json!(""), Map::new());
    steps.add_call(b, "dup".into(), "Bash", json!({}));
    steps.add_call(b, "dup".into(), "Bash", json!({}));

    let ids: Vec<String> = steps
        .steps
        .iter()
        .flat_map(|s| s.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default())
        .map(|c| c["tool_call_id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids, ["dup", "dup-2", "dup-3"]);
}

#[test]
fn observations_attach_to_the_step_that_issued_the_call() {
    let mut steps = Steps::new();
    let first = steps.add("agent", json!(""), Map::new());
    steps.add_call(first, "t1".into(), "Bash", json!({}));
    steps.add("user", json!("interleaved"), Map::new());
    let second = steps.add("agent", json!(""), Map::new());
    steps.add_call(second, "t2".into(), "Read", json!({}));

    // Out of order, and long after the calls were made.
    assert!(steps.observe("t2", "second".into(), Map::new()));
    assert!(steps.observe("t1", "first".into(), Map::new()));
    // An unknown id is reported rather than silently attached to the wrong step.
    assert!(!steps.observe("nope", "orphan".into(), Map::new()));

    let got = |i: usize| -> String {
        steps.steps[i]["observation"]["results"][0]["content"]
            .as_str()
            .unwrap()
            .to_string()
    };
    assert_eq!(got(first), "first");
    assert_eq!(got(second), "second");
}

#[test]
fn trajectory_numbers_steps_and_totals_metrics() {
    let mut steps = Steps::new();
    for n in [10, 20] {
        let mut f = Map::new();
        f.insert(
            "metrics".into(),
            json!({"prompt_tokens": n, "completion_tokens": 1, "cached_tokens": 0}),
        );
        steps.add("agent", json!(""), f);
    }
    let traj = trajectory(Meta::new("test"), steps.steps);
    let out = traj.as_object().unwrap();
    let emitted = out["steps"].as_array().unwrap();
    assert_eq!(emitted[0]["step_id"], json!(1));
    assert_eq!(emitted[1]["step_id"], json!(2));
    let fm = &out["final_metrics"];
    assert_eq!(fm["total_prompt_tokens"], json!(30));
    assert_eq!(fm["total_completion_tokens"], json!(2));
    assert_eq!(fm["total_steps"], json!(2));
    // A zero total is omitted rather than emitted as 0.
    assert!(fm.get("total_cached_tokens").is_none());
    // Version defaults rather than being left missing, which the validator rejects.
    assert_eq!(out["agent"]["version"], json!("unknown"));
}

#[test]
fn step_keys_are_emitted_in_schema_order() {
    let mut steps = Steps::new();
    let mut f = Map::new();
    f.insert("metrics".into(), json!({"prompt_tokens": 1}));
    f.insert("timestamp".into(), json!("2026-01-02T03:04:05+00:00"));
    f.insert("model_name".into(), json!("m"));
    let idx = steps.add("agent", json!("hi"), f);
    steps.add_call(idx, "t1".into(), "Bash", json!({}));
    let traj = trajectory(Meta::new("test"), steps.steps);
    let keys: Vec<&str> = traj["steps"][0]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        ["step_id", "timestamp", "source", "model_name", "message", "tool_calls", "metrics"]
    );
}
