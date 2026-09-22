"""Export Cursor, Claude Code, and Codex sessions as ATIF trajectories.

    atif                      # pick a session, export it, then choose where to send it
    atif list                 # recent sessions across all three agents
    atif export <selector>    # agent name, session-id prefix, or a log path
    atif export --all         # every session for $PWD -> ./atif-out/
    atif export --job         # uploadable Harbor job dir -> ./atif-jobs/<session>/
    atif validate FILE        # check an existing trajectory.json

Interactive only when stdin and stdout are a terminal and no session is named;
piped or scripted use keeps the direct path (or force it with --no-input).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import _core
from .core import (
    AGENTS, Session, discover, resolve, summarise, validate, write, write_job,
)
from .tui import ellipsis, interactive, menu, publish_menu


def session_row(s: Session, *, show_cwd: bool) -> str:
    row = f"{s.agent:<7} {s.when}  {s.size / 1e6:7.1f}MB  {ellipsis(s.title, 44):<44}"
    return f"{row}  \x1b[2m{s.where}\x1b[0m" if show_cwd else row.rstrip()


def cmd_list(args) -> int:
    scope = None if args.everywhere else Path.cwd()
    sessions = discover(cwd=scope, titles=True)
    if not sessions and scope is not None:
        sessions = discover(titles=True)
        if sessions:
            print(f"No sessions for {Path.cwd()}; showing all projects.\n", file=sys.stderr)
            scope = None
    if not sessions:
        print("No sessions found.", file=sys.stderr)
        return 1
    rows = sessions[:args.limit]
    # The cwd column is noise when every row shares it, which is the common case.
    show_cwd = args.everywhere or len({s.where for s in rows}) > 1
    for s in rows:
        print(session_row(s, show_cwd=show_cwd))
    return 0


def pick_session(scope: Path | None) -> Session | None:
    """Choose a session: 10 most recent, widening only when the user asks."""
    sessions = discover(cwd=scope, titles=True)
    scoped = scope is not None
    if not sessions and scoped:
        sessions, scoped = discover(titles=True), False
    if not sessions:
        print("No sessions found.", file=sys.stderr)
        return None
    limit = 10
    while True:
        shown = sessions[:limit]
        show_cwd = not scoped or len({s.where for s in shown}) > 1
        rows = [session_row(s, show_cwd=show_cwd) for s in shown]
        footer, tags = [], []
        if len(sessions) > len(shown):
            footer.append(f"\x1b[2mshow all {len(sessions)} sessions\u2026\x1b[0m")
            tags.append("more")
        if scoped:
            footer.append("\x1b[2msessions from all projects\u2026\x1b[0m")
            tags.append("global")
        label = scope.name if scoped else "all projects"
        choice = menu(f"Select a session to export  \x1b[2m({label})\x1b[0m",
                      rows, footer=footer, page=min(len(rows) + len(footer), 20))
        if choice is None:
            return None
        if choice < len(rows):
            return shown[choice]
        if tags[choice - len(rows)] == "more":
            limit = len(sessions)
        else:
            sessions, scoped, limit = discover(titles=True), False, 10


def export_interactive(args) -> int:
    """Pick a session, export it, then offer to publish it."""
    scope = None if args.everywhere else Path.cwd()
    session = pick_session(scope)
    if session is None:
        print("Cancelled.")
        return 0
    traj = session.build()
    if not traj.get("steps"):
        print(f"{session.path}: no convertible steps.", file=sys.stderr)
        return 1
    errs = validate(traj)

    # Always write the job layout here: it is the only form the uploaders accept,
    # and the plain trajectory.json sits inside it either way. One job root per
    # session, so exporting twice does not leave two trials inside a job whose
    # result.json claims one -- which the uploader rejects.
    root = Path(args.out).expanduser() if args.out else Path("atif-jobs") / session.slug()
    write_job(traj, session, root)
    print(f"\n\x1b[1m{ellipsis(session.title, 60) or session.slug()}\x1b[0m")
    print(f"  {summarise(traj)}")
    print(f"  {root}")
    if errs:
        print(f"  \x1b[31m{len(errs)} schema error(s)\x1b[0m", file=sys.stderr)
        for e in errs[:5]:
            print(f"  ! {e}", file=sys.stderr)
        return 1
    print()
    return publish_menu(root, session.slug())


def cmd_export(args) -> int:
    # Interactive only when a human is plainly present and no session was named.
    # Piped output, CI, `--no-input`, `--all` and an explicit selector all keep
    # the original non-interactive behaviour, so scripts never block on a prompt.
    if not args.selector and not args.all and not args.no_input and interactive():
        return export_interactive(args)

    sessions = resolve(args.selector, Path.cwd(), everywhere=args.everywhere)
    if not sessions:
        print(f"No session matched {args.selector!r}." if args.selector else "No sessions found.",
              file=sys.stderr)
        return 1
    chosen = sessions if args.all else sessions[:1]
    out = Path(args.out).expanduser() if args.out else None
    single = len(chosen) == 1 and not args.all
    failures = 0
    for s in chosen:
        traj = s.build()
        if not traj.get("steps"):
            print(f"{s.path}: no convertible steps, skipped", file=sys.stderr)
            failures += 1
            continue
        errs = validate(traj)
        if args.job:
            # One job root per session: a job's result.json describes one trial.
            root = out or Path("atif-jobs")
            dest = write_job(traj, s, root if single and out else root / s.slug())
        else:
            dest = (out or Path("trajectory.json")) if single else \
                (out or Path("atif-out")) / s.slug() / "trajectory.json"
            write(traj, dest)
        status = "OK" if not errs else f"{len(errs)} SCHEMA ERRORS"
        print(f"{dest}  [{s.agent}] {summarise(traj)}  [{status}]")
        for e in errs[:5]:
            print(f"  ! {e}", file=sys.stderr)
        failures += bool(errs)
    if args.job and not failures:
        print("\nupload with:\n  npx trajectories-sh upload trajectory <dir> --slug <slug>")
    return 1 if failures else 0


def cmd_validate(args) -> int:
    rc = 0
    for name in args.files:
        path = Path(name).expanduser()
        try:
            errs = _core.validate_file(str(path))
        except (OSError, ValueError) as exc:
            print(f"{path}: unreadable ({exc})", file=sys.stderr)
            rc = 1
            continue
        try:
            traj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            traj = {}
        detail = summarise(traj) if traj.get("steps") else ""
        head = f"{path}: {'valid' if not errs else f'{len(errs)} error(s)'}"
        print(f"{head} \u2014 {detail}" if detail else head)
        for e in errs:
            print(f"  ! {e}")
        rc |= bool(errs)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="atif", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version=f"atif {_core.__version__} (rust core, {_core.SCHEMA})")
    subs = ap.add_subparsers(dest="cmd")

    p = subs.add_parser("list", help="list recent sessions")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.add_argument("-a", "--everywhere", action="store_true", help="all projects, not just $PWD")
    p.set_defaults(fn=cmd_list)

    p = subs.add_parser("export", help="export session(s) to ATIF")
    p.add_argument("selector", nargs="?", help="agent name, session-id prefix, or log path")
    p.add_argument("-o", "--out", help="output file (single) or directory (--all)")
    p.add_argument("--all", action="store_true", help="export every matching session")
    p.add_argument("--job", action="store_true",
                   help="write an uploadable Harbor job directory instead of a bare file")
    p.add_argument("--no-input", action="store_true", help="never prompt, even on a terminal")
    p.add_argument("-a", "--everywhere", action="store_true", help="all projects, not just $PWD")
    p.set_defaults(fn=cmd_export)

    p = subs.add_parser("validate", help="validate trajectory.json files")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_validate)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        # Bare `atif` on a terminal is the common case -- "export something" --
        # so treat it as `atif export` rather than printing help at someone who
        # has already said what they want. Without a terminal there is nothing to
        # prompt with, so help (and a non-zero exit) is the honest answer.
        if interactive():
            return cmd_export(ap.parse_args(["export"]))
        ap.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
