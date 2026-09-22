#!/usr/bin/env python3
"""Tests for the ATIF exporter.

    uv run --with ./tools/atif-rs pytest tools/atif-rs/tests
    # or, against an installed wheel:
    python tools/atif-rs/tests/test_atif.py

Synthetic fixtures cover the mapping rules and edge cases found in real logs.
The corpus check then runs every session on this machine through the converter,
which is what catches format variants no fixture anticipates. The parsing core
under test is compiled Rust; these tests exercise it through the Python API.
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from atif import _core  # noqa: E402
from atif import cli, core, tui  # noqa: E402


class atif:  # noqa: N801 - a namespace so the assertions below read unchanged
    """Flat view over the package, mirroring the original single-module API."""

    SCHEMA = core.SCHEMA
    Path = Path
    # Rust core
    build_claude = staticmethod(lambda p: json.loads(_core.build(str(p), "claude")))
    build_codex = staticmethod(lambda p: json.loads(_core.build(str(p), "codex")))
    build_cursor = staticmethod(lambda p: json.loads(_core.build(str(p), "cursor")))
    validate = staticmethod(core.validate)
    sniff_agent = staticmethod(lambda p: _core.sniff_agent(str(p)))
    project_slug = staticmethod(_core.project_slug)
    clean_prompt = staticmethod(_core.clean_prompt)
    iso = staticmethod(_core.iso)
    session_title = staticmethod(
        lambda p, agent, **kw: _core.session_title(str(p), agent))
    codex_log_cwd = staticmethod(lambda p: _core.codex_log_cwd(str(p)))
    codex_thread_names = staticmethod(lambda: json.loads(_core.codex_thread_names()))
    # Python layer
    discover = staticmethod(core.discover)
    write = staticmethod(core.write)
    main = staticmethod(cli.main)
    ellipsis = staticmethod(tui.ellipsis)
    api_key = staticmethod(tui.api_key)


def write_log(dirpath: Path, name: str, records) -> Path:
    path = dirpath / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def assertValid(self, traj):
        errs = atif.validate(traj)
        self.assertEqual(errs, [], f"expected a valid trajectory, got: {errs}")


# --------------------------------------------------------------- helper units


class TestHelpers(Base):
    """Behaviour that now lives in Rust, asserted through observable output.

    The unit-level internals (`as_object`, `Steps`, `read_jsonl`, `dedupe_by_uuid`)
    are private to the compiled core and covered by `cargo test`; here they are
    exercised end-to-end, which is the contract that actually matters.
    """

    def test_json_string_arguments_become_objects(self):
        """Codex stores arguments as a JSON string; ATIF requires an object."""
        traj = atif.build_codex(write_log(self.tmp, "a.jsonl", [
            {"type": "function_call", "name": "shell", "call_id": "c1",
             "arguments": '{"command": "ls"}'}]))
        call = traj["steps"][0]["tool_calls"][0]
        self.assertEqual(call["arguments"], {"command": "ls"})

    def test_unparseable_arguments_are_preserved_not_dropped(self):
        traj = atif.build_codex(write_log(self.tmp, "b.jsonl", [
            {"type": "function_call", "name": "shell", "call_id": "c1",
             "arguments": "not json"},
            {"type": "function_call", "name": "shell", "call_id": "c2",
             "arguments": "[1, 2]"}]))
        args = [c["arguments"] for s in traj["steps"] for c in (s.get("tool_calls") or [])]
        self.assertEqual(args, [{"_raw": "not json"}, {"_raw": [1, 2]}])

    def test_iso_normalises_and_rejects_junk(self):
        self.assertTrue(atif.iso("2026-01-02T03:04:05Z").startswith("2026-01-02T03:04:05"))
        # Fractional seconds are normalised to microseconds, as
        # `datetime.fromisoformat().isoformat()` does.
        self.assertEqual(atif.iso("2026-01-02T03:04:05.386000+00:00"),
                         "2026-01-02T03:04:05.386000+00:00")
        self.assertEqual(atif.iso("2026-01-02T03:04:05.5Z"),
                         "2026-01-02T03:04:05.500000+00:00")
        self.assertIsNone(atif.iso("last tuesday"))
        self.assertIsNone(atif.iso(""))

    def test_truncated_final_line_is_skipped(self):
        """Live logs are appended to concurrently, so a partial line is normal."""
        path = self.tmp / "partial.jsonl"
        path.write_text(
            '{"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}\n'
            '\n{"role": "assis',
            encoding="utf-8")
        traj = atif.build_cursor(path)
        self.assertEqual(len(traj["steps"]), 1)

    def test_project_slug_is_encode_only(self):
        """Dots become dashes, so slugs must never be decoded back into paths."""
        self.assertEqual(atif.project_slug("/Users/x/my.project"), "-Users-x-my-project")

    def test_replayed_records_are_deduplicated(self):
        """Resuming a session re-appends history under the same record uuid."""
        once = atif.build_claude(write_log(self.tmp, "one.jsonl", claude_records()))
        twice = atif.build_claude(
            write_log(self.tmp, "two.jsonl", claude_records() + claude_records()))
        self.assertEqual(len(twice["steps"]), len(once["steps"]))

    def test_duplicate_tool_call_ids_are_disambiguated(self):
        """A trajectory must never contain two identical tool_call_ids."""
        block = {"type": "tool_use", "id": "dup", "name": "Bash", "input": {}}
        traj = atif.build_claude(write_log(self.tmp, "dup.jsonl", [
            {"type": "assistant", "uuid": "a1", "message": {
                "id": "m1", "role": "assistant", "content": [block]}},
            {"type": "assistant", "uuid": "a2", "message": {
                "id": "m2", "role": "assistant", "content": [block]}}]))
        ids = [c["tool_call_id"] for s in traj["steps"] for c in (s.get("tool_calls") or [])]
        self.assertEqual(ids, ["dup", "dup-2"])
        self.assertValid(traj)

    def test_observation_attaches_across_interleaved_turns(self):
        """A result may arrive several records after the call it belongs to."""
        traj = atif.build_claude(write_log(self.tmp, "late.jsonl", [
            {"type": "assistant", "uuid": "a1", "message": {"id": "m1", "role": "assistant",
             "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]}},
            {"type": "user", "uuid": "u1", "message": {"role": "user", "content": [
                {"type": "text", "text": "an interleaved turn"}]}},
            {"type": "user", "uuid": "u2", "message": {"role": "user", "content": [
                {"tool_use_id": "t1", "type": "tool_result", "content": "done"}]}}]))
        agent = next(s for s in traj["steps"] if s["source"] == "agent")
        self.assertEqual(agent["observation"]["results"],
                         [{"source_call_id": "t1", "content": "done"}])
        self.assertValid(traj)

    def test_metrics_are_totalled_into_final_metrics(self):
        traj = atif.build_claude(write_log(self.tmp, "m.jsonl", claude_records()))
        self.assertEqual(traj["final_metrics"]["total_prompt_tokens"], 1110)
        self.assertEqual(traj["final_metrics"]["total_steps"], len(traj["steps"]))

    def test_ellipsis_collapses_newlines_and_truncates(self):
        self.assertEqual(atif.ellipsis("a\n b   c", 20), "a b c")
        self.assertEqual(atif.ellipsis("x" * 30, 10), "x" * 9 + "\u2026")

    def test_clean_prompt_unwraps_and_rejects_injected_text(self):
        self.assertEqual(atif.clean_prompt("<user_query>\nfix the bug\n</user_query>"),
                         "fix the bug")
        self.assertEqual(atif.clean_prompt("<recommended_plugins>\nHere is a list\n- Airtable"),
                         "", "a plugin catalogue is not a user prompt")
        self.assertEqual(atif.clean_prompt("# Files mentioned by the user:\n## a.jpg: /tmp/x"),
                         "", "an attachment manifest is not a user prompt")
        self.assertEqual(atif.clean_prompt("<timestamp>x</timestamp>\n\nreal question"),
                         "real question")

    def test_title_prefers_custom_then_generated_then_prompt(self):
        base = [{"type": "user", "uuid": "u1", "message": {"role": "user", "content": [
            {"type": "text", "text": "the first real prompt"}]}}]
        self.assertEqual(atif.session_title(write_log(self.tmp, "p.jsonl", base), "claude"),
                         "the first real prompt")
        gen = [{"type": "ai-title", "aiTitle": "Generated"}] + base
        self.assertEqual(atif.session_title(write_log(self.tmp, "g.jsonl", gen), "claude"),
                         "Generated")
        cust = [{"type": "custom-title", "customTitle": "Mine"}] + gen
        self.assertEqual(atif.session_title(write_log(self.tmp, "c.jsonl", cust), "claude"),
                         "Mine")

    def test_title_skips_injected_turns_and_bare_slash_commands(self):
        records = [
            {"type": "user", "uuid": "u1", "message": {"role": "user", "content": [
                {"type": "text", "text": "<recommended_plugins>\nHere is a list\n- Airtable"}]}},
            {"type": "user", "uuid": "u2", "message": {"role": "user", "content": [
                {"type": "text", "text": "/clear"}]}},
            {"type": "user", "uuid": "u3", "message": {"role": "user", "content": [
                {"type": "text", "text": "what does this repo do?"}]}},
        ]
        self.assertEqual(atif.session_title(write_log(self.tmp, "t.jsonl", records), "claude"),
                         "what does this repo do?")

    def test_title_falls_back_to_slash_command_when_nothing_better(self):
        records = [{"type": "user", "uuid": "u1", "message": {"role": "user", "content": [
            {"type": "text", "text": "/model"}]}}]
        self.assertEqual(atif.session_title(write_log(self.tmp, "s.jsonl", records), "claude"),
                         "/model")


# --------------------------------------------------------------- validator


class TestValidator(Base):
    def good(self):
        return {"schema_version": atif.SCHEMA, "agent": {"name": "a", "version": "1"},
                "steps": [{"step_id": 1, "source": "user", "message": "hi"}]}

    def test_accepts_minimal_trajectory(self):
        self.assertValid(self.good())

    def test_rejects_bad_schema_version(self):
        t = self.good()
        t["schema_version"] = "ATIF-v9"
        self.assertTrue(any("schema_version" in e for e in atif.validate(t)))

    def test_rejects_missing_agent_version(self):
        t = self.good()
        t["agent"] = {"name": "a"}
        self.assertTrue(any("version" in e for e in atif.validate(t)))

    def test_rejects_non_sequential_step_ids(self):
        t = self.good()
        t["steps"].append({"step_id": 5, "source": "user", "message": "x"})
        self.assertTrue(any("step_id" in e for e in atif.validate(t)))

    def test_rejects_bad_source(self):
        t = self.good()
        t["steps"][0]["source"] = "assistant"
        self.assertTrue(any("source" in e for e in atif.validate(t)))

    def test_rejects_agent_only_field_on_user_step(self):
        t = self.good()
        t["steps"][0]["tool_calls"] = [{"tool_call_id": "a", "function_name": "f",
                                        "arguments": {}}]
        self.assertTrue(any("only valid when source" in e for e in atif.validate(t)))

    def test_rejects_non_object_arguments(self):
        t = self.good()
        t["steps"] = [{"step_id": 1, "source": "agent", "message": "",
                       "tool_calls": [{"tool_call_id": "a", "function_name": "f",
                                       "arguments": "{}"}]}]
        self.assertTrue(any("arguments must be an object" in e for e in atif.validate(t)))

    def test_rejects_dangling_source_call_id(self):
        """ATIF requires every observation to reference a call in the same step."""
        t = self.good()
        t["steps"] = [{"step_id": 1, "source": "agent", "message": "",
                       "observation": {"results": [{"source_call_id": "nope", "content": "x"}]}}]
        self.assertTrue(any("matches no tool_call" in e for e in atif.validate(t)))

    def test_rejects_embedded_subagent_without_trajectory_id(self):
        t = self.good()
        t["subagent_trajectories"] = [self.good()]
        self.assertTrue(any("trajectory_id is required" in e for e in atif.validate(t)))

    def test_reports_errors_inside_subagents(self):
        sub = self.good()
        sub["trajectory_id"] = "s1"
        sub["steps"][0]["source"] = "bogus"
        t = self.good()
        t["subagent_trajectories"] = [sub]
        self.assertTrue(any("subagent_trajectories[0]" in e for e in atif.validate(t)))


# --------------------------------------------------------------- Claude Code


def claude_records():
    """One user turn, one inference split across four records, then two results."""
    usage = {"input_tokens": 10, "cache_creation_input_tokens": 100,
             "cache_read_input_tokens": 1000, "output_tokens": 50,
             "output_tokens_details": {"thinking_tokens": 20}}

    def assistant(block):
        return {"type": "assistant", "uuid": f"u{block['type']}{block.get('id', '')}",
                "timestamp": "2026-01-01T00:00:01Z", "version": "2.0.1",
                "cwd": "/repo", "sessionId": "sess-1",
                "message": {"id": "msg-1", "role": "assistant", "model": "claude-opus-5",
                            "usage": usage, "content": [block]}}
    return [
        {"type": "user", "uuid": "u0", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/repo",
         "message": {"role": "user", "content": [{"type": "text", "text": "list files"}]}},
        assistant({"type": "thinking", "thinking": "I should look around."}),
        assistant({"type": "text", "text": "Listing now."}),
        assistant({"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}),
        assistant({"type": "tool_use", "id": "t2", "name": "Read", "input": {"path": "a.txt"}}),
        {"type": "user", "uuid": "u5", "message": {"role": "user", "content": [
            {"tool_use_id": "t1", "type": "tool_result", "content": "a.txt"}]}},
        {"type": "user", "uuid": "u6", "message": {"role": "user", "content": [
            {"tool_use_id": "t2", "type": "tool_result", "content": "", "is_error": True}]},
         "toolUseResult": {"stderr": "boom"}},
        {"type": "system", "uuid": "u7", "subtype": "compact_boundary",
         "content": "Conversation compacted",
         "compactMetadata": {"trigger": "manual", "preTokens": 900, "postTokens": 40}},
    ]


class TestClaude(Base):
    def setUp(self):
        super().setUp()
        self.traj = atif.build_claude(write_log(self.tmp, "s.jsonl", claude_records()))

    def test_is_valid(self):
        self.assertValid(self.traj)

    def test_agent_identity_from_log(self):
        self.assertEqual(self.traj["agent"], {"name": "claude-code", "version": "2.0.1",
                                              "model_name": "claude-opus-5"})
        self.assertEqual(self.traj["session_id"], "sess-1")
        self.assertEqual(self.traj["extra"]["cwd"], "/repo")

    def test_one_step_per_inference(self):
        """Four records share message.id, so they must collapse to a single step."""
        agents = [s for s in self.traj["steps"] if s["source"] == "agent"]
        self.assertEqual(len(agents), 1)
        step = agents[0]
        self.assertEqual(step["message"], "Listing now.")
        self.assertEqual(step["reasoning_content"], "I should look around.")
        self.assertEqual([c["function_name"] for c in step["tool_calls"]], ["Bash", "Read"])
        self.assertEqual(step["llm_call_count"], 1)

    def test_prompt_tokens_sum_anthropic_buckets(self):
        """Anthropic reports three disjoint input buckets; ATIF wants their total."""
        m = [s for s in self.traj["steps"] if s["source"] == "agent"][0]["metrics"]
        self.assertEqual(m["prompt_tokens"], 1110)
        self.assertEqual(m["cached_tokens"], 1000)
        self.assertEqual(m["completion_tokens"], 50)
        self.assertEqual(m["extra"], {"cache_creation_input_tokens": 100,
                                      "reasoning_tokens": 20})

    def test_usage_counted_once_per_inference(self):
        """Usage repeats on every record of the inference and must not be summed."""
        self.assertEqual(self.traj["final_metrics"]["total_prompt_tokens"], 1110)

    def test_tool_results_become_observations(self):
        step = [s for s in self.traj["steps"] if s["source"] == "agent"][0]
        results = step["observation"]["results"]
        self.assertEqual([r["source_call_id"] for r in results], ["t1", "t2"])
        self.assertEqual(results[0]["content"], "a.txt")
        self.assertEqual(results[1]["content"], "boom", "empty result falls back to stderr")
        self.assertTrue(results[1]["extra"]["is_error"])

    def test_tool_results_do_not_create_user_turns(self):
        self.assertEqual(sum(1 for s in self.traj["steps"] if s["source"] == "user"), 1)

    def test_compaction_becomes_context_management(self):
        sys_steps = [s for s in self.traj["steps"] if s["source"] == "system"]
        cm = sys_steps[0]["extra"]["context_management"]
        self.assertEqual(cm["type"], "compaction")
        self.assertEqual(cm["boundary"], "replace")
        self.assertEqual((cm["pre_tokens"], cm["post_tokens"]), (900, 40))

    def test_replayed_records_are_not_duplicated(self):
        """Resuming a session re-appends history under the same uuid."""
        doubled = atif.build_claude(
            write_log(self.tmp, "dup.jsonl", claude_records() + claude_records()))
        self.assertEqual(len(doubled["steps"]), len(self.traj["steps"]))
        self.assertValid(doubled)

    def test_sidechains_become_embedded_subagents(self):
        extra = [{"type": "assistant", "uuid": "sc1", "isSidechain": True,
                  "message": {"id": "m9", "role": "assistant", "model": "claude-opus-5",
                              "content": [{"type": "text", "text": "subagent work"}]}}]
        traj = atif.build_claude(write_log(self.tmp, "sc.jsonl", claude_records() + extra))
        subs = traj["subagent_trajectories"]
        self.assertEqual(len(subs), 1)
        self.assertTrue(subs[0]["trajectory_id"])
        self.assertEqual(subs[0]["steps"][0]["message"], "subagent work")
        self.assertValid(traj)


# --------------------------------------------------------------- Codex


def codex_records(legacy=False):
    """Codex logs come in two generations: newer records wrap events in `payload`."""
    items = [
        ("session_meta", {"session_id": "cx-1", "cwd": "/repo", "cli_version": "0.77.0",
                          "model": "gpt-5", "model_provider": "openai",
                          "instructions": "be brief"}),
        ("response_item", {"type": "message", "role": "user",
                           "content": [{"type": "input_text", "text": "fix the bug"}]}),
        ("response_item", {"type": "reasoning",
                           "summary": [{"type": "summary_text", "text": "Inspect first."}]}),
        ("response_item", {"type": "function_call", "name": "shell", "call_id": "c1",
                           "arguments": '{"command": "ls"}'}),
        ("response_item", {"type": "function_call_output", "call_id": "c1", "output": "a.txt"}),
        ("event_msg", {"type": "token_count", "info": {
            "last_token_usage": {"input_tokens": 500, "cached_input_tokens": 400,
                                 "output_tokens": 30, "reasoning_output_tokens": 10},
            "total_token_usage": {"input_tokens": 500, "cached_input_tokens": 400,
                                  "output_tokens": 30}}}),
        ("response_item", {"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": "Fixed."}]}),
    ]
    out = []
    for kind, payload in items:
        if legacy:  # flat records, no session_meta wrapper
            if kind == "session_meta":
                out.append({"id": "cx-1", "timestamp": "2026-01-01T00:00:00Z",
                            "instructions": "be brief"})
            elif kind == "response_item":
                out.append(payload)
            continue
        out.append({"type": kind, "timestamp": "2026-01-01T00:00:00Z", "payload": payload})
    return out


class TestCodex(Base):
    def setUp(self):
        super().setUp()
        self.traj = atif.build_codex(write_log(self.tmp, "r.jsonl", codex_records()))

    def test_is_valid(self):
        self.assertValid(self.traj)

    def test_identity_and_system_prompt(self):
        self.assertEqual(self.traj["agent"]["name"], "codex")
        self.assertEqual(self.traj["agent"]["version"], "0.77.0")
        self.assertEqual(self.traj["session_id"], "cx-1")
        self.assertEqual(self.traj["steps"][0]["source"], "system")
        self.assertEqual(self.traj["steps"][0]["message"], "be brief")

    def test_arguments_parsed_into_object(self):
        call = [c for s in self.traj["steps"] for c in (s.get("tool_calls") or [])][0]
        self.assertEqual(call["arguments"], {"command": "ls"})
        self.assertEqual(call["tool_call_id"], "c1")

    def test_reasoning_attaches_to_following_action(self):
        step = next(s for s in self.traj["steps"] if s.get("tool_calls"))
        self.assertEqual(step["reasoning_content"], "Inspect first.")

    def test_output_becomes_observation(self):
        step = next(s for s in self.traj["steps"] if s.get("tool_calls"))
        self.assertEqual(step["observation"]["results"],
                         [{"source_call_id": "c1", "content": "a.txt"}])

    def test_per_step_metrics_from_last_token_usage(self):
        step = next(s for s in self.traj["steps"] if s.get("metrics"))
        self.assertEqual(step["metrics"]["prompt_tokens"], 500)
        self.assertEqual(step["metrics"]["cached_tokens"], 400)
        self.assertEqual(step["metrics"]["extra"]["reasoning_tokens"], 10)

    def test_final_metrics_use_reported_totals(self):
        self.assertEqual(self.traj["final_metrics"]["total_prompt_tokens"], 500)
        self.assertEqual(self.traj["final_metrics"]["total_cached_tokens"], 400)

    def test_legacy_flat_records(self):
        traj = atif.build_codex(write_log(self.tmp, "old.jsonl", codex_records(legacy=True)))
        self.assertValid(traj)
        call = [c for s in traj["steps"] for c in (s.get("tool_calls") or [])][0]
        self.assertEqual(call["arguments"], {"command": "ls"})
        self.assertEqual(sum(1 for s in traj["steps"]
                             if (s.get("observation") or {}).get("results")), 1)

    def test_cwd_recovered_from_environment_context(self):
        """Older logs only reveal the cwd inside an injected context message."""
        records = [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<environment_context>\n<cwd>/x/y</cwd>\n"
                                           "</environment_context>"}]}]
        path = write_log(self.tmp, "cw.jsonl", records)
        self.assertEqual(atif.codex_log_cwd(path), "/x/y")

    def test_orphan_output_is_declared_not_dropped_silently(self):
        records = [{"type": "function_call_output", "call_id": "zz", "output": "x"}]
        traj = atif.build_codex(write_log(self.tmp, "orphan.jsonl", records))
        self.assertIn("no matching call", traj["notes"])

    def test_shell_and_custom_tool_variants(self):
        records = [
            {"type": "local_shell_call", "call_id": "s1", "action": {"command": ["ls"]}},
            {"type": "custom_tool_call", "call_id": "s2", "name": "grep", "input": '{"q": "x"}'},
        ]
        traj = atif.build_codex(write_log(self.tmp, "var.jsonl", records))
        self.assertValid(traj)
        names = [c["function_name"] for s in traj["steps"] for c in (s.get("tool_calls") or [])]
        self.assertEqual(names, ["local_shell", "grep"])


# --------------------------------------------------------------- Cursor


class TestCursor(Base):
    def setUp(self):
        super().setUp()
        root = self.tmp / "agent-transcripts" / "sess-a"
        self.log = write_log(root, "sess-a.jsonl", [
            {"role": "user", "message": {"content": [{"type": "text", "text": "hello"}]}},
            {"role": "assistant", "message": {"content": [
                {"type": "text", "text": "Looking."},
                {"type": "tool_use", "name": "Shell", "input": {"command": "ls"}},
                {"type": "tool_use", "name": "Read", "input": {"path": "a"}}]}},
            {"type": "turn_ended", "status": "success"},
        ])
        write_log(root / "subagents", "sub-1.jsonl", [
            {"role": "assistant", "message": {"content": [{"type": "text", "text": "sub"}]}}])
        self.traj = atif.build_cursor(self.log)

    def test_is_valid(self):
        self.assertValid(self.traj)

    def test_identity_and_turn_status(self):
        self.assertEqual(self.traj["agent"]["name"], "cursor")
        self.assertEqual(self.traj["session_id"], "sess-a")
        self.assertEqual(self.traj["extra"]["turn_statuses"], ["success"])

    def test_multiple_tool_uses_share_one_step(self):
        step = next(s for s in self.traj["steps"] if s["source"] == "agent")
        self.assertEqual(step["message"], "Looking.")
        self.assertEqual(len(step["tool_calls"]), 2)

    def test_synthesised_ids_are_unique(self):
        """Cursor stores no tool ids, so they are generated and must not collide."""
        ids = [c["tool_call_id"] for s in self.traj["steps"]
               for c in (s.get("tool_calls") or [])]
        self.assertEqual(len(ids), len(set(ids)))

    def test_missing_results_declared_in_notes(self):
        """Cursor does not persist tool results; the gap must be documented."""
        self.assertFalse(any(s.get("observation") for s in self.traj["steps"]))
        self.assertIn("no stored results", self.traj["notes"])

    def test_subagents_embedded_and_valid(self):
        subs = self.traj["subagent_trajectories"]
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["trajectory_id"], "sess-a-subagent-1")
        self.assertValid(subs[0])


# --------------------------------------------------------------- CLI


class TestCLI(Base):
    def test_sniff_agent_identifies_logs_outside_home(self):
        """A misidentified log is parsed by the wrong adapter and exports nothing."""
        self.assertEqual(atif.sniff_agent(write_log(self.tmp, "c.jsonl", claude_records())),
                         "claude")
        self.assertEqual(atif.sniff_agent(write_log(self.tmp, "x.jsonl", codex_records())),
                         "codex")
        self.assertEqual(atif.sniff_agent(write_log(self.tmp, "cu.jsonl", [
            {"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}])),
            "cursor")

    def test_export_writes_and_validates_a_log_path(self):
        log = write_log(self.tmp, "s.jsonl", claude_records())
        out = self.tmp / "out.json"
        rc = atif.main(["export", str(log), "-o", str(out)])
        self.assertEqual(rc, 0)
        traj = json.loads(out.read_text())
        self.assertEqual(traj["agent"]["name"], "claude-code")
        self.assertValid(traj)

    def test_cli_plumbing_is_dropped_and_injected_context_is_not_the_user(self):
        """`/clear` writes three records that are narration, not conversation.

        Emitting them invented user turns that never happened, and attributing
        injected context to the user misstates who authored the prompt.
        """
        records = [
            {"type": "user", "uuid": "u1", "message": {"role": "user", "content": [
                {"type": "text", "text": "<local-command-caveat>Caveat: the messages below"
                                         "</local-command-caveat>"}]}},
            {"type": "user", "uuid": "u2", "message": {"role": "user", "content": [
                {"type": "text", "text": "<command-name>/clear</command-name>"}]}},
            {"type": "system", "uuid": "s1", "subtype": "local_command",
             "content": "<local-command-stdout></local-command-stdout>"},
            {"type": "user", "uuid": "u3", "message": {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>session named x</system-reminder>"}]}},
            {"type": "user", "uuid": "u4", "message": {"role": "user", "content": [
                {"type": "text", "text": "the actual question"}]}},
        ]
        traj = atif.build_claude(write_log(self.tmp, "p.jsonl", records))
        self.assertEqual([(s["source"], s["message"][:17]) for s in traj["steps"]],
                         [("system", "<system-reminder>"), ("user", "the actual questi")])
        self.assertValid(traj)

    def test_publish_menu_offers_only_configured_targets(self):
        """A destination nobody has set up is a dead end, not a choice."""
        shown = []

        def fake_menu(title, rows, **kw):
            shown.append(list(rows))
            return None  # cancel, so nothing is actually run

        # Both available: three actions plus the exit.
        with mock.patch.object(tui, "menu", fake_menu), \
             mock.patch.object(tui, "api_key", return_value="k"), \
             mock.patch.object(tui, "have_trajectories", return_value=True), \
             mock.patch.object(tui, "have_harbor", return_value=True):
            tui.publish_menu(self.tmp, "slug")
        self.assertEqual(len(shown[0]), 4)
        self.assertTrue(any("trajectories.sh" in r for r in shown[0]))
        self.assertTrue(any("Harbor" in r for r in shown[0]))

        # trajectories.sh not installed: its entry disappears entirely.
        shown.clear()
        with mock.patch.object(tui, "menu", fake_menu), \
             mock.patch.object(tui, "api_key", return_value="k"), \
             mock.patch.object(tui, "have_trajectories", return_value=False), \
             mock.patch.object(tui, "have_harbor", return_value=True):
            tui.publish_menu(self.tmp, "slug")
        self.assertFalse(any("trajectories.sh" in r for r in shown[0]), shown[0])

        # Installed but no API key: still hidden, because it could only fail.
        shown.clear()
        with mock.patch.object(tui, "menu", fake_menu), \
             mock.patch.object(tui, "api_key", return_value=None), \
             mock.patch.object(tui, "have_trajectories", return_value=True), \
             mock.patch.object(tui, "have_harbor", return_value=True):
            tui.publish_menu(self.tmp, "slug")
        self.assertFalse(any("trajectories.sh" in r for r in shown[0]), shown[0])

    def test_publish_menu_skipped_when_nothing_is_configured(self):
        """With no targets, an empty menu is useless: say what is missing."""
        with mock.patch.object(tui, "api_key", return_value=None), \
             mock.patch.object(tui, "have_trajectories", return_value=False), \
             mock.patch.object(tui, "have_harbor", return_value=False), \
             mock.patch.object(tui, "menu", side_effect=AssertionError("must not prompt")), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(tui.publish_menu(self.tmp, "slug"), 0)
        msg = err.getvalue()
        self.assertIn("No upload target", msg)
        self.assertIn("trajectories-sh", msg, "should say how to install it")
        self.assertIn("harbor", msg)

    def test_harbor_cmd_prefers_an_installed_binary(self):
        """The git fallback triggers a multi-minute build, so avoid it when able."""
        with mock.patch.object(tui, "which", return_value="/usr/local/bin/harbor"):
            self.assertEqual(tui.harbor_cmd(), ["/usr/local/bin/harbor"])
        with mock.patch.object(tui, "which", return_value=None):
            self.assertEqual(tui.harbor_cmd(), tui.HARBOR_FALLBACK)

    def test_have_harbor_requires_an_install_not_the_fallback(self):
        with mock.patch.object(tui, "which", return_value=None):
            self.assertFalse(tui.have_harbor())
        with mock.patch.object(tui, "which", return_value="/x/harbor"):
            self.assertTrue(tui.have_harbor())

    def test_have_trajectories_accepts_path_or_npx_cache(self):
        """`npx` would install on demand, so detection must mean 'already set up'."""
        with mock.patch.object(tui, "which", return_value="/x/trajectories-sh"):
            self.assertTrue(tui.have_trajectories())
        # Nothing on PATH and an empty cache: not configured.
        home = self.tmp / "fakehome"
        (home / ".npm" / "_npx").mkdir(parents=True)
        with mock.patch.object(tui, "which", return_value=None), \
             mock.patch.object(tui, "HOME", home):
            self.assertFalse(tui.have_trajectories())
        # A resolved npx entry counts as evidence of real use.
        (home / ".npm" / "_npx" / "abc123" / "node_modules" / "trajectories-sh").mkdir(parents=True)
        with mock.patch.object(tui, "which", return_value=None), \
             mock.patch.object(tui, "HOME", home):
            self.assertTrue(tui.have_trajectories())

    def test_bare_atif_on_a_terminal_opens_the_picker(self):
        """`atif` with no arguments is the documented way to export interactively.

        Guards the wiring between "no subcommand" and the picker, which is easy to
        break because argparse's own default is to print help instead.
        """
        called = []
        with mock.patch.object(cli, "interactive", return_value=True), \
             mock.patch.object(cli, "export_interactive",
                               side_effect=lambda a: called.append(a) or 0):
            self.assertEqual(atif.main([]), 0)
        self.assertEqual(len(called), 1, "bare `atif` should reach the picker")
        # It must arrive with export's own defaults, not a half-built namespace.
        self.assertFalse(called[0].all)
        self.assertFalse(called[0].no_input)
        self.assertIsNone(called[0].selector)

    def test_bare_atif_without_a_terminal_prints_help(self):
        """Nothing to prompt with, so help and a non-zero exit is the honest answer."""
        with mock.patch.object(cli, "interactive", return_value=False), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(atif.main([]), 2)
        self.assertIn("usage:", out.getvalue())

    def test_job_layout_satisfies_the_uploader_contract(self):
        """The fields `trajectories upload` validates, pinned by name.

        Read off the uploader's own validator: a missing `stats.evals` or
        `trial_uri` is rejected at upload time, which is far too late to notice.
        """
        log = write_log(self.tmp, "s.jsonl", claude_records())
        job = self.tmp / "job"
        self.assertEqual(atif.main(["export", str(log), "--job", "-o", str(job)]), 0)

        result = json.loads((job / "result.json").read_text())
        for key in ("n_total_trials", "started_at", "finished_at", "stats"):
            self.assertIsNotNone(result.get(key), f"job result.json needs {key}")
        for key in ("n_trials", "n_errors", "evals"):
            self.assertIn(key, result["stats"], f"job stats needs {key}")

        trials = [p for p in job.iterdir() if p.is_dir()]
        self.assertEqual(len(trials), 1, "a job claiming one trial must contain one")
        trial = trials[0]
        data = json.loads((trial / "result.json").read_text())
        for key in ("task_name", "started_at", "finished_at", "agent_info", "trial_uri"):
            self.assertIsNotNone(data.get(key), f"trial result.json needs {key}")
        self.assertTrue(data["agent_info"]["name"])
        self.assertIn("verifier_result", data)
        # A URI, not a bare path, and it must point at this trial.
        self.assertTrue(data["trial_uri"].startswith("file://"))
        self.assertEqual(data["trial_uri"], trial.resolve().as_uri())
        self.assertTrue((trial / "agent" / "trajectory.json").is_file())

    def test_repeated_exports_do_not_pile_trials_into_one_job(self):
        """Two sessions must not land in a job whose result.json claims one trial."""
        out = self.tmp / "jobs"
        for name in ("a", "b"):
            log = write_log(self.tmp, f"{name}.jsonl", claude_records())
            self.assertEqual(atif.main(["export", str(log), "--job", "-o", str(out / name)]), 0)
        for name in ("a", "b"):
            job = out / name
            claimed = json.loads((job / "result.json").read_text())["n_total_trials"]
            actual = len([p for p in job.iterdir() if p.is_dir()])
            self.assertEqual(actual, claimed)

    def test_codex_id_from_an_explicit_path_is_the_uuid_not_the_filename(self):
        """A rollout is `rollout-<timestamp>-<uuid>`, so the stem is not the id.

        Exporting by path bypasses discovery, which is where the id normally comes
        from; getting it wrong names the trial directory `…__rollout-`.
        """
        name = "rollout-2026-04-29T08-18-04-019dd9d1-949c-73a3-b703-b9fb0d4c8e0c.jsonl"
        log = write_log(self.tmp, name, codex_records())
        job = self.tmp / "cx"
        self.assertEqual(atif.main(["export", str(log), "--job", "-o", str(job)]), 0)
        trial = next(p for p in job.iterdir() if p.is_dir())
        self.assertTrue(trial.name.endswith("__019dd9d1"), trial.name)

    def test_validate_command_reports_bad_file(self):
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"schema_version": "nope", "agent": {}, "steps": []}))
        self.assertEqual(atif.main(["validate", str(bad)]), 1)

    def test_validate_command_accepts_good_file(self):
        log = write_log(self.tmp, "s.jsonl", claude_records())
        out = self.tmp / "ok.json"
        atif.main(["export", str(log), "-o", str(out)])
        self.assertEqual(atif.main(["validate", str(out)]), 0)

    def test_unknown_selector_exits_nonzero(self):
        self.assertEqual(atif.main(["export", "definitely-not-a-session-xyz"]), 1)

    def test_list_shows_titles_and_runs_fast(self):
        """`list` must stay responsive: titles come from a bounded scan."""
        import time
        start = time.time()
        self.assertEqual(atif.main(["list", "-a", "-n", "10"]), 0)
        self.assertLess(time.time() - start, 10.0)

    def test_codex_thread_names_are_read_from_the_index(self):
        """Codex stores thread names outside the rollout logs."""
        names = atif.codex_thread_names()
        self.assertIsInstance(names, dict)

    def test_job_layout_meets_upload_preconditions(self):
        """`trajectories upload` takes a job dir and rejects null verifier_result."""
        log = write_log(self.tmp, "s.jsonl", claude_records())
        job = self.tmp / "job"
        self.assertEqual(atif.main(["export", str(log), "--job", "-o", str(job)]), 0)
        trials = [d for d in job.iterdir() if d.is_dir()]
        self.assertEqual(len(trials), 1)
        trial = trials[0]
        self.assertValid(json.loads((trial / "agent" / "trajectory.json").read_text()))
        result = json.loads((trial / "result.json").read_text())
        self.assertIsNotNone(result["verifier_result"])
        self.assertIsNone(result["verifier_result"]["reward"],
                          "an exported session was never graded, so it must claim no score")
        stats = json.loads((job / "result.json").read_text())["stats"]
        self.assertEqual((stats["n_trials"], stats["n_errors"]), (1, 0))
        self.assertIn("name", json.loads((trial / "config.json").read_text())["agent"])


# --------------------------------------------------------------- real corpus


class TestCorpus(unittest.TestCase):
    """Convert every session on this machine. Real logs contain variants no
    fixture predicts, so this is the check that actually protects the exporter."""

    @classmethod
    def setUpClass(cls):
        cls.sessions = atif.discover()
        if not cls.sessions:
            raise unittest.SkipTest("no local agent sessions found")

    def test_every_session_converts_and_validates(self):
        failures, counts = [], {}
        for s in self.sessions:
            counts[s.agent] = counts.get(s.agent, 0) + 1
            try:
                errs = atif.validate(s.build())
                if errs:
                    failures.append(f"{s.agent}/{s.path.name}: {errs[0]}")
            except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
                failures.append(f"{s.agent}/{s.path.name}: {type(exc).__name__}: {exc}")
        print(f"\n  corpus: {len(self.sessions)} sessions {counts}")
        self.assertEqual(failures, [], "\n".join(failures[:10]))

    def test_tool_calls_and_observations_are_not_lost(self):
        """Compare against counts taken from the raw logs, independent of adapters."""
        for agent in ("claude", "codex"):
            session = next((s for s in self.sessions if s.agent == agent), None)
            if session is None:
                continue
            traj = session.build()
            calls = sum(len(s.get("tool_calls") or []) for s in traj["steps"])
            obs = sum(len((s.get("observation") or {}).get("results") or [])
                      for s in traj["steps"])
            self.assertGreaterEqual(calls, obs,
                                    f"{agent}: more observations than tool calls")
            for step in traj["steps"]:
                ids = {c["tool_call_id"] for c in (step.get("tool_calls") or [])}
                for res in (step.get("observation") or {}).get("results") or []:
                    self.assertIn(res["source_call_id"], ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
