//! ATIF schema validation: the rules Harbor's own validator enforces.

use crate::common::{accepted_versions, iso};
use serde_json::Value;
use std::collections::HashSet;

const SOURCES: [&str; 3] = ["system", "user", "agent"];
const AGENT_ONLY: [&str; 5] = [
    "model_name",
    "reasoning_content",
    "tool_calls",
    "metrics",
    "reasoning_effort",
];

pub fn validate(traj: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    check(traj, "trajectory", &mut errs);
    errs
}

fn check(traj: &Value, path: &str, errs: &mut Vec<String>) {
    let version = traj.get("schema_version").and_then(Value::as_str);
    let accepted = accepted_versions();
    if !version.is_some_and(|v| accepted.iter().any(|a| a == v)) {
        errs.push(format!(
            "{path}: schema_version {:?} not in {:?}",
            version.unwrap_or("<missing>"),
            accepted
        ));
    }

    let agent = traj.get("agent");
    let named = agent
        .and_then(|a| a.get("name"))
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    let versioned = agent
        .and_then(|a| a.get("version"))
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    if !(agent.is_some_and(Value::is_object) && named && versioned) {
        errs.push(format!("{path}: agent requires both name and version"));
    }

    let Some(steps) = traj.get("steps").and_then(Value::as_array) else {
        errs.push(format!("{path}: steps must be a non-empty array"));
        return;
    };
    if steps.is_empty() {
        errs.push(format!("{path}: steps must be a non-empty array"));
        return;
    }

    let mut seen_calls: HashSet<String> = HashSet::new();
    for (i, step) in steps.iter().enumerate() {
        let n = i + 1;
        let where_ = format!("{path}.steps[{n}]");

        if step.get("step_id").and_then(Value::as_u64) != Some(n as u64) {
            errs.push(format!(
                "{where_}: step_id {} should be {n}",
                step.get("step_id").unwrap_or(&Value::Null)
            ));
        }
        let source = step.get("source").and_then(Value::as_str);
        if !source.is_some_and(|s| SOURCES.contains(&s)) {
            errs.push(format!(
                "{where_}: source {:?} not in {SOURCES:?}",
                source.unwrap_or("<missing>")
            ));
        }
        match step.get("message") {
            Some(Value::String(_)) | Some(Value::Array(_)) => {}
            _ => errs.push(format!(
                "{where_}: message is required (may be an empty string)"
            )),
        }
        if let Some(ts) = step.get("timestamp").and_then(Value::as_str) {
            if iso(ts).is_none() {
                errs.push(format!("{where_}: timestamp {ts:?} is not ISO-8601"));
            }
        }
        if source != Some("agent") {
            for field in AGENT_ONLY {
                if step.get(field).is_some_and(|v| !v.is_null()) {
                    errs.push(format!(
                        "{where_}: {field} is only valid when source is 'agent'"
                    ));
                }
            }
        }
        if step.get("llm_call_count").and_then(Value::as_i64) == Some(0)
            && (step.get("metrics").is_some() || step.get("reasoning_content").is_some())
        {
            errs.push(format!(
                "{where_}: llm_call_count 0 forbids metrics and reasoning_content"
            ));
        }

        let mut ids: HashSet<String> = HashSet::new();
        if let Some(calls) = step.get("tool_calls").and_then(Value::as_array) {
            for (j, call) in calls.iter().enumerate() {
                let cid = call.get("tool_call_id").and_then(Value::as_str);
                let name = call.get("function_name").and_then(Value::as_str);
                if !cid.is_some_and(|s| !s.is_empty()) || !name.is_some_and(|s| !s.is_empty()) {
                    errs.push(format!(
                        "{where_}.tool_calls[{j}]: tool_call_id and function_name required"
                    ));
                }
                if !call.get("arguments").is_some_and(Value::is_object) {
                    errs.push(format!(
                        "{where_}.tool_calls[{j}]: arguments must be an object"
                    ));
                }
                if let Some(cid) = cid {
                    if !seen_calls.insert(cid.to_string()) {
                        errs.push(format!(
                            "{where_}.tool_calls[{j}]: duplicate tool_call_id {cid:?}"
                        ));
                    }
                    ids.insert(cid.to_string());
                }
            }
        }

        if let Some(obs) = step.get("observation") {
            match obs.get("results").and_then(Value::as_array) {
                None => errs.push(format!("{where_}.observation: results array is required")),
                Some(results) => {
                    for (j, res) in results.iter().enumerate() {
                        if let Some(reference) = res.get("source_call_id").and_then(Value::as_str) {
                            if !ids.contains(reference) {
                                errs.push(format!(
                                    "{where_}.observation.results[{j}]: source_call_id \
                                     {reference:?} matches no tool_call in this step"
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(subs) = traj.get("subagent_trajectories").and_then(Value::as_array) {
        let mut sub_ids: HashSet<String> = HashSet::new();
        for (i, sub) in subs.iter().enumerate() {
            match sub.get("trajectory_id").and_then(Value::as_str) {
                None => errs.push(format!(
                    "{path}: subagent_trajectories[{i}]: trajectory_id is required on \
                     embedded subagents"
                )),
                Some(tid) => {
                    if !sub_ids.insert(tid.to_string()) {
                        errs.push(format!(
                            "{path}: subagent_trajectories[{i}]: duplicate trajectory_id {tid:?}"
                        ));
                    }
                }
            }
            check(sub, &format!("{path}.subagent_trajectories[{i}]"), errs);
        }
    }
}
