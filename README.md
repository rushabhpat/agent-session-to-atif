# Agent session to atif

Export your Claude Code, Codex, and Cursor sessions as
[ATIF](https://docs.harborframework.com/core-concepts/agents/atif) trajectories,
ready to view or upload to [Harbor](https://harborframework.com) and
[trajectories.sh](https://trajectories.sh).

Your coding agents already write a detailed record of every session to disk, each
in its own undocumented format. `atif` reads all three and converts them into one
standard format, so you can inspect, share, and analyse them with the same tools.

The parsing core is Rust; the CLI and interactive picker are Python.

```
$ atif
Select a session to export  (my-project)
❯ claude  2026-09-22 08:12     2.4MB  Fix the flaky auth test
  codex   2026-09-22 07:40     0.9MB  Add a retry to the upload path
  cursor  2026-09-21 19:03     0.5MB  Why is this query slow?
  ...
  show all 37 sessions...
  sessions from all projects...

Fix the flaky auth test
  184 steps (23u/159a/2s), 96 tool calls, 96 observations, 1,204,338 prompt tokens
  atif-jobs/claude-a1b2c3d4

What next?
❯ upload to trajectories.sh
  open in the local Harbor viewer
  nothing, I'm done
```

## Install

Requires [uv](https://docs.astral.sh/uv/) and a
[Rust toolchain](https://rustup.rs). The Rust core is compiled at install time.

```sh
uv tool install git+https://github.com/rushabhpat/agent-session-to-atif
```

Or without installing anything permanently:

```sh
uvx --from git+https://github.com/rushabhpat/agent-session-to-atif atif list
```

From a clone, `./install.sh` does the same and checks the prerequisites first.

## Use

```sh
atif                    # pick a session, export it, then choose where to send it
atif list               # recent sessions for this project
atif list -a            # every project
atif export claude      # newest Claude Code session, non-interactively
atif export a1b2c3d4    # a specific session by id prefix
atif export path/to.jsonl
atif export --all       # every session for this directory -> ./atif-out/
atif export --job       # uploadable Harbor job dir -> ./atif-jobs/<session>/
atif validate FILE      # check an existing trajectory.json
```

Run on a terminal with no arguments, `atif` opens a picker: the ten most recent
sessions, expandable to all of them or widened to every project. After exporting
it offers the destinations your machine can actually reach, and says how to set
one up if none is configured. Piped or scripted use never prompts, so this is
safe in automation; `--no-input` forces that path explicitly.

### Uploading

Both uploaders take a **job directory**, not a bare `trajectory.json`, so pass
`--job` whenever the goal is to view a session in Harbor or on trajectories.sh.
A plain `atif export` is for reading and diffing locally.

```sh
atif export --job -o /tmp/myjob
```

**Harbor.** `upload` sends the job to a Harbor server; `view` opens it in the
local viewer without uploading anything, which is the quicker way to check that a
conversion looks right.

```sh
harbor view /tmp/myjob
harbor upload /tmp/myjob
```

Harbor is not on PyPI, so install it from git:

```sh
uv tool install 'harbor @ git+https://github.com/harbor-framework/harbor'
```

**trajectories.sh.**

```sh
npx trajectories-sh upload trajectory /tmp/myjob --slug my-session
```

Set `TRAJECTORIES_API_KEY` in your environment, or put it in
`~/.config/atif/trajectories.env` (`ATIF_ENV_FILE` overrides the location):

```sh
TRAJECTORIES_API_KEY=your-key-here
```

The key is passed to the uploader through its environment only. It is never put
in a command line, logged, or written into an export.

The interactive menu runs exactly these commands for you, and lists only the
destinations it can find. An uninstalled uploader or a missing key is a dead end
rather than a choice, so it is left out and the setup hint is printed instead.
Harbor counts only when `harbor` is on PATH: it can also be run straight from
git, but offering that as a menu entry would invite you to pick what looks
instant and then wait for a compile.

Exported sessions carry `verifier_result.reward: null`. They were never graded,
and a fabricated reward would render as a benchmark score.

## Before you share a session

A trajectory is a verbatim replay of an agent session, so it contains whatever
passed through that terminal: keys echoed by a shell command, tokens in a `curl`
line, customer data in a fixture. The source logs in `~/.claude`, `~/.codex` and
`~/.cursor` were always private; an export is the copy that leaves your machine.

Check before uploading:

```sh
rg -i 'api[_-]?key|secret|password|BEGIN .*PRIVATE KEY' trajectory.json
```

## Where sessions are read from

| Agent | Location |
|---|---|
| Claude Code | `~/.claude/projects/<project>/<session>.jsonl` |
| Codex | `~/.codex/sessions/<date>/rollout-*.jsonl` |
| Cursor | `~/.cursor/projects/<project>/agent-transcripts/<session>/` |

Nothing is written to those directories. `atif` only reads them.

## Fidelity

Each agent records a different amount, and the converter never invents what is
absent. Every trajectory states its own limitations in `notes`.

| | Claude Code | Codex | Cursor |
|---|---|---|---|
| tool results | yes | yes | **not recorded** |
| timestamps | yes | yes | **not recorded** |
| token counts | per inference | per turn | **not recorded** |
| reasoning | thinking blocks | summaries | no |
| subagents | sidechains | n/a | `subagents/` |

Cursor persists only the model-visible conversation, so those trajectories carry
tool calls but no observations. That is a limit of the source log, not the
converter, so do not "fix" it by synthesising results.

Two details that are easy to get wrong, both learned from real logs:

- **Anthropic's token buckets are disjoint.** ATIF's `prompt_tokens` is the sum
  of `input_tokens`, `cache_creation_input_tokens` and
  `cache_read_input_tokens`, with `cached_tokens` the read subset. Treating
  `input_tokens` as the total under-reports prompt size by more than an order of
  magnitude.
- **Claude splits one inference across records** that share a `message.id`,
  repeating `usage` on each. Group by that id and count usage once; summing it
  inflates every token total several-fold.

### Screenshots

Sessions that take screenshots carry them inline as base64. Flattening those into
the trajectory produces a file that is mostly image data: one real codex session
came to 244 MB, 98% of it base64, which no viewer can render.

So images are written as files beside the trajectory and referenced by path,
which is what ATIF expects:

```
trajectory.json
media/<sha256>.png
```

That took the same session to 4.5 MB, and the server then recognised the images
as screenshots. Names are content hashes, so an image repeated across turns is
stored once. Keep `media/` next to `trajectory.json` when moving an export: both
validators resolve those paths relative to the trajectory.

This also explains a confusing upload error. `trajectories upload` skips any file
over 50 MB, then validates against the list of files it kept, so an oversized
trajectory is reported as:

```
✗ Trial "...": missing agent/trajectory.json
```

The file is present. It was too big, and 98% of the bulk was screenshots.

Harness plumbing is dropped rather than exported as conversation. Typing `/clear`
makes Claude Code write three bookkeeping records; emitting them would invent
user turns that never happened. Injected context such as `<environment_context>`
or `<system-reminder>` is kept, because the model did act on it, but attributed
to `system` rather than to you.

## Develop

```sh
cargo test --lib                        # Rust unit tests
cargo clippy --lib                      # lints
uvx maturin build --release             # build a wheel
uv run --with . python tests/test_atif.py
```

The Python suite finishes by converting every session on your machine and
validating each result, which is what catches format variants no fixture
anticipates. Note that `uv run --with .` reuses a cached wheel keyed on version,
so rebuild after changing the Python layer.

To check output against Harbor's own validator rather than the built-in one:

```sh
uv run --with 'harbor @ git+https://github.com/harbor-framework/harbor' \
  --python 3.12 python -m harbor.utils.trajectory_validator trajectory.json
```

## Layout

```
src/            Rust core: JSONL reading, the three adapters, ATIF assembly,
                validation, session discovery and titles
python/atif/    CLI, interactive picker, publish flow
tests/          Python suite, including a full corpus sweep
```

The boundary between them passes JSON strings rather than rich objects: one
`serde_json` serialisation out, one `json.loads` in. That keeps the binding
surface small instead of requiring a conversion for every nested ATIF type.

## License

MIT
