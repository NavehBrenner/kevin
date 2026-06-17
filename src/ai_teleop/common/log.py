"""Project console logging — a thin wrapper over the stdlib ``logging`` module.

One parent logger (``ai_teleop``) carries the handler configuration; every module
gets a child of it via :func:`get_logger`, so a single :func:`configure_logging`
call (typically from a script's ``main``) styles all of them at once.

Design notes:

- **Stdlib first.** Output is plain ``logging``; ``rich`` is an *optional* prettifier
  used only when it's installed and the stream is a TTY. Without it (or when piped),
  the formatter falls back to a compact ``HH:MM:SS LEVEL [name] message`` line.
- **Console plus an optional file tee.** Logs always go to stderr (keeping stdout
  clean for real program output); ``log_file`` additionally writes them to disk.
- **Layered, not baked in.** Only scripts/harnesses configure and emit logs — the
  per-tick control loop (``run_episode``, the controller) stays logging-free.

Typical use in a script::

    from ai_teleop.common.log import add_logging_arguments, configure_from_args, get_logger

    log = get_logger("datagen")

    def main() -> int:
        parser = argparse.ArgumentParser(...)
        add_logging_arguments(parser)
        args = parser.parse_args()
        configure_from_args(args)
        log.info("generating %d episodes -> %s", n, out_dir)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Soft import: rich is an optional dependency. Absent -> stdlib StreamHandler.
try:
    from rich.console import Console
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    Console = None  # type: ignore[assignment,misc]
    RichHandler = None  # type: ignore[assignment,misc]

# The single parent every project logger descends from. Configuring this logger
# (handlers + level) styles all `get_logger(...)` children at once.
PACKAGE_LOGGER_NAME = "ai_teleop"

# argparse `const` for a bare `--log-file` (auto-name under outputs/logs/), kept
# distinct from an explicit path the user passes.
AUTO_LOG_FILE = "auto"

_LOG_DIR = Path("outputs") / "logs"
_CONSOLE_FORMAT = "%(asctime)s %(levelname)-5s [%(name_short)s] %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-8s [%(name_short)s] %(message)s"
_TIME_FORMAT = "%H:%M:%S"


class _ShortNameFilter(logging.Filter):
    """Add ``name_short`` — the logger name without the ``ai_teleop.`` prefix.

    Lets the formatter print ``[datagen]`` instead of ``[ai_teleop.datagen]``
    while keeping the real hierarchical name on the record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.name_short = record.name.removeprefix(f"{PACKAGE_LOGGER_NAME}.")
        return True


def get_logger(name: str) -> logging.Logger:
    """Return the ``ai_teleop.<name>`` logger (a child of the package logger).

    Thin and side-effect-free: it only names a logger. Output styling comes from
    :func:`configure_logging`; until that runs, the stdlib defaults apply.
    """
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def default_log_path(prog: str | None = None) -> Path:
    """Auto-named per-run log path: ``outputs/logs/<prog>_<UTC-timestamp>.log``.

    ``prog`` defaults to the running script's stem (e.g. ``generate_dataset``).
    """
    if prog is None:
        prog = Path(sys.argv[0]).stem or "ai_teleop"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return _LOG_DIR / f"{prog}_{stamp}.log"


def _console_handler(*, level: int, use_rich: bool) -> logging.Handler:
    """A stderr console handler — RichHandler when available + on a TTY, else stdlib."""
    on_tty = sys.stderr.isatty()
    if use_rich and RichHandler is not None and on_tty:
        handler: logging.Handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_path=False,
            omit_repeated_times=False,
        )
        # RichHandler renders its own time + level columns; the formatter only
        # supplies the tag + message.
        handler.setFormatter(logging.Formatter("[%(name_short)s] %(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_TIME_FORMAT))
    handler.setLevel(level)
    handler.addFilter(_ShortNameFilter())
    return handler


def _file_handler(path: Path, *, level: int) -> logging.Handler:
    """A plain-text file handler (never coloured); creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_TIME_FORMAT))
    handler.setLevel(level)
    handler.addFilter(_ShortNameFilter())
    return handler


def configure_logging(
    *,
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    use_rich: bool = True,
    quiet: bool = False,
) -> None:
    """Configure the package logger's handlers. Idempotent.

    Re-attaches handlers from scratch on every call (so repeated invocation —
    in tests, or a CLI that reconfigures — never stacks duplicates).

    Args:
        level: Threshold for record creation and the console handler (name or int).
        log_file: Also tee logs to this file; ``None`` for console only. Pass
            :data:`AUTO_LOG_FILE` to auto-name one under ``outputs/logs/``.
        use_rich: Prefer ``rich`` for console output when installed and on a TTY.
        quiet: Raise the *console* threshold to ``WARNING`` (the file tee, if any,
            still records at ``level``).
    """
    numeric_level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):  # unknown name -> getLevelName returns a str
        raise ValueError(f"unknown log level: {level!r}")

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(numeric_level)
    # Tear down any handlers a previous call added before re-adding.
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    console_level = logging.WARNING if quiet else numeric_level
    logger.addHandler(_console_handler(level=console_level, use_rich=use_rich))

    if log_file is not None:
        path = default_log_path() if log_file == AUTO_LOG_FILE else Path(log_file)
        logger.addHandler(_file_handler(path, level=numeric_level))


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--log-level`` / ``--quiet`` / ``--log-file`` flags.

    Pairs with :func:`configure_from_args` so every script exposes logging the
    same way without duplicating argparse boilerplate.
    """
    group = parser.add_argument_group("logging")
    group.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity (default: INFO).",
    )
    group.add_argument(
        "--quiet",
        action="store_true",
        help="Only warnings and errors on the console (a --log-file still records everything).",
    )
    group.add_argument(
        "--log-file",
        nargs="?",
        const=AUTO_LOG_FILE,
        default=None,
        metavar="PATH",
        help="Also write logs to a file; bare flag auto-names one under outputs/logs/.",
    )


def configure_from_args(args: argparse.Namespace) -> None:
    """Apply the flags added by :func:`add_logging_arguments`."""
    configure_logging(level=args.log_level, log_file=args.log_file, quiet=args.quiet)
