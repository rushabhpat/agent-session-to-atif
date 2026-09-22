"""Terminal interaction: a raw-mode list picker and the publish menu.

Only ever entered when both stdin and stdout are a TTY. Every other context --
pipes, CI, an explicit selector -- keeps the non-interactive path, so scripts
never block on a prompt.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ESC = "\x1b"
KEYS = {"\x1b[A": "up", "\x1b[B": "down", "\x1bOA": "up", "\x1bOB": "down",
        "\r": "enter", "\n": "enter", "k": "up", "j": "down",
        "q": "quit", "\x03": "quit", "\x1b": "quit"}

HOME = Path.home()
# Harbor is not on PyPI, so an uninstalled fallback has to come from git. Used
# only when no `harbor` is already on PATH, because resolving this costs a build.
HARBOR_FALLBACK = ["uv", "run", "--with",
                   "harbor @ git+https://github.com/harbor-framework/harbor",
                   "--python", "3.12", "harbor"]


def which(name: str):
    """Absolute path to an executable on PATH, or None."""
    return shutil.which(name)


def harbor_cmd():
    """How to invoke Harbor, preferring an install over building it from git."""
    found = which("harbor")
    return [found] if found else HARBOR_FALLBACK


def have_harbor() -> bool:
    """Is Harbor usable without a multi-minute build?

    Only an installed `harbor` counts. The git fallback still works when chosen
    explicitly, but offering it as a menu entry would invite someone to pick what
    looks like a quick action and wait several minutes for a compile.
    """
    return which("harbor") is not None


def have_trajectories() -> bool:
    """Is the trajectories.sh CLI available without a network install?

    `npx trajectories-sh` would happily download the package on demand, so the
    question is not "can this be run" but "is this someone's actual workflow".
    Evidence of use is a real install: on PATH, installed globally by npm, or
    already resolved into the npx cache. Requiring npm's global prefix to be
    queried is avoided -- it is slow -- unless the cheaper checks fail.
    """
    if which("trajectories-sh"):
        return True
    # npx caches each resolved package under a content-addressed directory.
    npx_cache = HOME / ".npm/_npx"
    try:
        for entry in npx_cache.iterdir():
            if (entry / "node_modules" / "trajectories-sh").exists():
                return True
    except OSError:
        pass
    # A global npm install is the remaining case; `npm root -g` is slow, so the
    # conventional locations are probed directly first.
    for root in (Path("/usr/local/lib/node_modules"), Path("/opt/homebrew/lib/node_modules"),
                 HOME / ".npm-global/lib/node_modules",
                 HOME / ".nvm/versions/node"):
        if root.name == "node" and root.is_dir():
            try:
                if any((v / "lib/node_modules/trajectories-sh").exists()
                       for v in root.iterdir()):
                    return True
            except OSError:
                pass
        elif (root / "trajectories-sh").exists():
            return True
    return False


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def ellipsis(text: str, width: int) -> str:
    text = " ".join((text or "").split())  # titles may contain newlines
    return text if len(text) <= width else text[:width - 1].rstrip() + "\u2026"


class Term:
    """Raw-mode keyboard reader that always restores the terminal.

    A crash or Ctrl-C inside a raw-mode session would otherwise leave the user's
    shell with no echo, so teardown lives in a context manager rather than at the
    end of the read loop.
    """

    def __enter__(self):
        import termios
        import tty
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        sys.stdout.write("\x1b[?25l")  # hide cursor
        sys.stdout.flush()
        return self

    def __exit__(self, *exc):
        import termios
        sys.stdout.write("\x1b[?25h")  # restore cursor
        sys.stdout.flush()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        return False

    def key(self) -> str:
        """Read one keypress, collapsing escape sequences into a single name.

        Reads bytes straight from the fd rather than `sys.stdin.read`: the text
        wrapper buffers, which desynchronises `select` and splits an arrow key
        into three separate "keypresses". A bare ESC must still mean cancel, so
        the continuation bytes are waited for briefly instead of assumed.
        """
        import select
        data = os.read(self.fd, 1)
        if data != b"\x1b":
            return KEYS.get(data.decode("utf-8", "replace"), data.decode("utf-8", "replace"))
        for _ in range(2):
            if not select.select([self.fd], [], [], 0.2)[0]:
                break
            data += os.read(self.fd, 1)
            if data.decode("utf-8", "replace") in KEYS:
                break
        return KEYS.get(data.decode("utf-8", "replace"), "quit")


def menu(title: str, rows: list, *, footer=None, page=10) -> int | None:
    """Arrow-key list picker. Returns the chosen index, or None if cancelled.

    `rows` are pre-rendered strings. `footer` adds terminal entries (such as
    "show all sessions") that are selectable but visually separated.
    """
    options = list(rows) + list(footer or [])
    if not options:
        return None
    pos, top, drawn = 0, 0, 0
    with Term() as term:
        while True:
            view = min(page, len(options))
            top = max(0, min(top, len(options) - view))
            if pos < top:
                top = pos
            elif pos >= top + view:
                top = pos - view + 1
            out = [f"\x1b[1m{title}\x1b[0m"] if title else []
            for i in range(top, top + view):
                if footer and i == len(rows):
                    out.append("  \x1b[2m" + "\u2500" * 58 + "\x1b[0m")
                mark = "\x1b[36m\u276f\x1b[0m" if i == pos else " "
                body = options[i]
                out.append(f"{mark} {body}" if i != pos else f"{mark} \x1b[1m{body}\x1b[0m")
            out.append("  \x1b[2m\u2191\u2193 move \u00b7 enter select \u00b7 q cancel\x1b[0m")
            if drawn:
                sys.stdout.write(f"\x1b[{drawn}A")  # rewind over the last frame
            sys.stdout.write("".join(f"\x1b[2K{line}\n" for line in out))
            sys.stdout.flush()
            drawn = len(out)

            k = term.key()
            if k == "up":
                pos = (pos - 1) % len(options)
            elif k == "down":
                pos = (pos + 1) % len(options)
            elif k == "enter":
                return pos
            elif k == "quit":
                return None
            elif k.isdigit():  # 1-9 jump straight to a row
                n = int(k) - 1
                if 0 <= n < len(options):
                    return n


def env_files():
    """Where a trajectories.sh key may be kept, most specific first.

    `ATIF_ENV_FILE` lets a caller point anywhere; otherwise the tool's own config
    directory is preferred, then the location trajectories.sh itself documents.
    """
    override = os.environ.get("ATIF_ENV_FILE")
    paths = [Path(override).expanduser()] if override else []
    paths += [HOME / ".config/atif/trajectories.env",
              HOME / ".config/trajectories/trajectories.env"]
    return paths


def api_key() -> str | None:
    """Find a trajectories.sh key without ever putting it in argv.

    Checked in order: the current environment, then the files above. The value is
    only ever placed in a child process's environment, never passed as
    `--api-key` and never printed.
    """
    key = os.environ.get("TRAJECTORIES_API_KEY")
    if key:
        return key.strip()
    for env_file in env_files():
        try:
            text = env_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            name, _, value = line.partition("=")
            # Tolerate `export FOO=bar`, which is how these files are often written.
            name = name.strip()
            if name.startswith("export "):
                name = name[7:].strip()
            if name == "TRAJECTORIES_API_KEY":
                return value.strip().strip("'\"") or None
    return None


def run_child(cmd: list, *, env_extra=None) -> int:
    """Run a publish command, streaming its output. Secrets travel by env only."""
    env = dict(os.environ)
    env.update(env_extra or {})
    print(f"\n\x1b[2m$ {' '.join(cmd)}\x1b[0m")
    try:
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        print(f"{cmd[0]}: not found on PATH", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        return 130


def publish_menu(job_dir: Path, slug: str) -> int:
    """Offer only what this machine can actually do with the job directory.

    Every entry is a destination someone has set up: an uninstalled uploader or a
    missing API key is not a choice, it is a dead end that wastes a keystroke and
    then fails. When nothing is configured the menu is skipped entirely and the
    path is printed instead, which is the useful answer in that case.
    """
    key = api_key()
    actions = []
    if key and have_trajectories():
        actions.append(("upload to trajectories.sh", "traj"))
    if have_harbor():
        actions.append(("upload to Harbor", "harbor"))
        actions.append(("open in the local Harbor viewer", "view"))

    if not actions:
        # Say what is missing, and how to get each one, rather than an empty menu.
        print("No upload target is configured. To enable one:", file=sys.stderr)
        if not have_trajectories():
            print("  trajectories.sh  npm i -g trajectories-sh", file=sys.stderr)
        elif not key:
            print("  trajectories.sh  set TRAJECTORIES_API_KEY (or put it in\n"
                  "                   ~/.config/atif/trajectories.env)", file=sys.stderr)
        if not have_harbor():
            print("  Harbor           uv tool install "
                  "'harbor @ git+https://github.com/harbor-framework/harbor'", file=sys.stderr)
        return 0

    actions.append(("nothing, I'm done", "done"))
    choice = menu("What next?", [a[0] for a in actions])
    if choice is None:
        return 0
    tag = actions[choice][1]
    if tag == "done":
        return 0
    if tag == "traj":
        return run_child(["npx", "--yes", "trajectories-sh", "upload", "trajectory",
                          str(job_dir), "--slug", slug],
                         env_extra={"TRAJECTORIES_API_KEY": key})
    if tag == "harbor":
        return run_child(harbor_cmd() + ["upload", str(job_dir)])
    return run_child(harbor_cmd() + ["view", str(job_dir)])
