"""Session discovery and export, wrapping the compiled core."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import _core

SCHEMA = _core.SCHEMA
AGENTS = tuple(_core.AGENTS)


class Session:
    """A discovered session log, uniformly addressable across the three agents."""

    __slots__ = ("agent", "path", "cwd", "id", "project", "mtime", "size", "_title")

    def __init__(self, record: dict):
        self.agent = record["agent"]
        self.path = Path(record["path"])
        self.cwd = record.get("cwd")
        self.id = record["id"]
        self.project = record.get("project")
        self.mtime = record.get("mtime") or 0.0
        self.size = record.get("size") or 0
        self._title = record.get("title")

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")

    @property
    def where(self) -> str:
        if self.cwd:
            return self.cwd
        # Cursor records no cwd, so fall back to reversing the project slug.
        # Lossy (dots and dashes both encode to '-'), hence display-only: this
        # string is never used to build a path that gets opened.
        if self.project and self.project.lstrip("-").startswith("Users"):
            return "/" + self.project.lstrip("-").replace("-", "/") + " (approx)"
        return self.project or "?"

    @property
    def title(self) -> str:
        """Computed on demand: only listings need it, and it costs a file scan."""
        if self._title is None:
            self._title = _core.session_title(str(self.path), self.agent)
        return self._title

    def build(self) -> dict:
        return json.loads(_core.build(str(self.path), self.agent))

    def slug(self) -> str:
        return f"{self.agent}-{self.id[:8]}"


def discover(agents=AGENTS, cwd: Path | None = None, titles=False) -> list[Session]:
    """Find sessions, newest first. `cwd` restricts results to one project."""
    records = json.loads(
        _core.discover_sessions(str(cwd) if cwd else None, list(agents), titles)
    )
    return [Session(r) for r in records]


def resolve(selector: str | None, cwd: Path, everywhere=False) -> list[Session]:
    """Turn a user selector into sessions. Empty selector means 'the newest one'."""
    p = Path(selector).expanduser() if selector else None
    if p is not None and p.exists():
        target = p / f"{p.name}.jsonl" if p.is_dir() else p
        agent = _core.sniff_agent(str(target))
        return [Session({
            "agent": agent,
            "path": str(target),
            # Not the filename stem: a Codex rollout is named
            # `rollout-<timestamp>-<uuid>`, so the id has to be extracted the
            # same way discovery does it rather than guessed from the path.
            "id": _core.session_id(str(target), agent),
            "mtime": target.stat().st_mtime,
            "size": target.stat().st_size,
        })]
    agents = (selector,) if selector in AGENTS else AGENTS
    rest = None if selector in AGENTS else selector
    sessions = discover(agents, None if everywhere else cwd)
    if not sessions and not everywhere:  # fall back rather than report nothing
        sessions = discover(agents, None)
    if rest:
        sessions = [s for s in sessions if s.id.startswith(rest) or rest in s.path.name]
    return sessions


def validate(traj: dict) -> list[str]:
    """Check a trajectory against the ATIF rules Harbor's validator enforces."""
    return _core.validate_json(json.dumps(traj))


def write(traj: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(traj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def write_job(traj: dict, session: Session, job_dir: Path) -> Path:
    """Write a Harbor job directory, the unit `trajectories upload` accepts.

        <job>/result.json                       job summary; stats needs n_trials + evals
        <job>/<task>__<id>/result.json          trial summary; trial_uri + verifier_result
        <job>/<task>__<id>/config.json          agent identification
        <job>/<task>__<id>/agent/trajectory.json

    `job_dir` is the job root and describes exactly one trial, so each export
    gets its own root; sharing one would leave a job whose result.json claims a
    single trial sitting next to several, which the uploader rejects. The trial
    is named `<task>__<id>` after Harbor's own convention, which also avoids a
    directory repeating its parent's name.

    Required fields were read off the uploader's validator and a real Harbor job.
    A bare trajectory.json is rejected, and the server additionally rejects a
    null `verifier_result`. An exported session was never graded, so the verifier
    reports a null score with `n_metrics: 0` and `evals` stays empty -- recording
    "not evaluated" rather than inventing a reward that would read as a score.
    """
    agent = traj["agent"]
    task = f"{agent['name']}-session"
    trial = f"{task}__{session.id[:8]}"
    steps = traj.get("steps") or []
    started = next((s["timestamp"] for s in steps if s.get("timestamp")), None) \
        or datetime.fromtimestamp(session.mtime, timezone.utc).isoformat()
    finished = next((s["timestamp"] for s in reversed(steps) if s.get("timestamp")), started)
    job_id = traj.get("session_id") or session.slug()
    model = agent.get("model_name")
    trial_dir = job_dir / trial

    write(traj, trial_dir / "agent" / "trajectory.json")
    (trial_dir / "config.json").write_text(json.dumps({
        "trial_name": trial,
        "agent": {"name": agent["name"], "model_name": model},
        "job_id": job_id,
    }, indent=1) + "\n", encoding="utf-8")
    (trial_dir / "result.json").write_text(json.dumps({
        "id": job_id,
        "task_name": task,
        "trial_name": trial,
        # Required, and a URI rather than a path: where this trial was written.
        "trial_uri": trial_dir.resolve().as_uri(),
        "started_at": started,
        "finished_at": finished,
        "agent_info": {"name": agent["name"], "version": agent["version"],
                       "model_info": {"name": model} if model else None},
        # Non-null so the server accepts the trial; scoreless because nothing graded it.
        "verifier_result": {"reward": None, "metrics": {}, "n_metrics": 0,
                            "note": "exported session, not evaluated"},
        "exception_info": None,
        "source_log": (traj.get("extra") or {}).get("source_log"),
    }, indent=1) + "\n", encoding="utf-8")
    (job_dir / "result.json").write_text(json.dumps({
        "id": job_id,
        "started_at": started,
        "updated_at": finished,
        "finished_at": finished,
        "n_total_trials": 1,
        "stats": {"n_trials": 1, "n_errors": 0, "n_completed_trials": 1,
                  "n_errored_trials": 0, "n_running_trials": 0,
                  "n_pending_trials": 0, "n_cancelled_trials": 0, "n_retries": 0,
                  # Required to be present; empty because no verifier ran.
                  "evals": {}},
    }, indent=1) + "\n", encoding="utf-8")
    return job_dir


def summarise(traj: dict) -> str:
    steps = traj.get("steps") or []
    if not steps:
        return "no convertible steps"
    kinds = {k: sum(1 for s in steps if s["source"] == k) for k in ("user", "agent", "system")}
    calls = sum(len(s.get("tool_calls") or []) for s in steps)
    obs = sum(len((s.get("observation") or {}).get("results") or []) for s in steps)
    fm = traj.get("final_metrics") or {}
    bits = [f"{len(steps)} steps ({kinds['user']}u/{kinds['agent']}a/{kinds['system']}s)",
            f"{calls} tool calls", f"{obs} observations"]
    if fm.get("total_prompt_tokens"):
        bits.append(f"{fm['total_prompt_tokens']:,} prompt tokens")
    subs = traj.get("subagent_trajectories")
    if subs:
        bits.append(f"{len(subs)} subagents")
    return ", ".join(bits)
