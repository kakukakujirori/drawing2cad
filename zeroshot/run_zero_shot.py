#!/usr/bin/env python3
"""Zero-shot 2D drawing -> 3D CAD using the installed agent CLIs.

The runner deliberately does not call model-provider SDKs or HTTP APIs.  It
invokes one of ``agy`` (Gemini), ``codex`` (GPT), or ``claude`` directly.

Each task is staged into an opaque, short-lived directory containing only the
three render_3d PNGs and the DXF drawing.  The agent itself sees a minimal
bubblewrap filesystem and a disposable HOME populated with only the credential
file needed by its CLI.  Generated Python is then checked and executed in a
second bubblewrap sandbox with no network access.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


RENDER_INPUT_NAMES = (
    "render_1.png",
    "render_2.png",
    "render_3.png",
)
DXF_INPUT_NAME = "drawing.dxf"
MODEL_NAME = "model.py"
MODEL_ATTEMPT_PATTERN = re.compile(r"model_attempt_(\d+)\.py")
FINAL_SELECTION_NAME = "final_selection.json"
STEP_NAME = "output.step"
REASONING_NAME = "reasoning.md"
SOURCE_USAGE_NAME = "source_usage.json"
ACTION_TRACE_NAME = "action_trace.jsonl"
CAD_ENV_ROOT = "/cad-env"
CAD_ENV_PYTHON = f"{CAD_ENV_ROOT}/bin/python"
INTERMEDIATES_DIR_NAME = "intermediates"
QUARANTINE_DIR_NAME = "quarantine"
AUDIT_DIR_NAME = "audit"
MAX_INTERMEDIATE_FILE_BYTES = 64 * 1024 * 1024
MAX_INTERMEDIATES_TOTAL_BYTES = 256 * 1024 * 1024
MAX_REPORT_FILE_BYTES = 4 * 1024 * 1024
MAX_PYTHON_FILE_BYTES = 1_000_000
RUNNER_CONTRACT_VERSION = 3
# Statuses that produced a validated STEP solid and a canonical model.py.
# "success-with-warnings" additionally carries `compliance_warnings` describing
# how the agent deviated from the requested artifact discipline; the evidence
# for every deviation is retained under artifact_history/.
SUCCESS_STATUSES = frozenset({"success", "success-with-warnings"})

GPT_DISABLED_FEATURES = (
    "apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "enable_mcp_apps",
    "image_generation",
    "plugin_sharing",
    "plugins",
    "skill_mcp_dependency_install",
    "skill_search",
    "standalone_web_search",
    "workspace_dependencies",
)

# Human-facing reasoning-effort labels (the same parenthetical style as the
# Gemini "(High)" convention) mapped to the effort tokens the CLIs understand.
# GPT (Codex `model_reasoning_effort`) and Claude (`--effort`) share this exact
# set of levels.  "Max" is only available on some models; the CLI reports an
# unsupported model/effort pair, so the runner validates the label spelling here
# and leaves the per-model availability check to the CLI.  Gemini does not use
# this map: its variant stays embedded in the label passed verbatim to agy.
EFFORT_LABELS = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "extra high": "xhigh",
    "max": "max",
}

DEFAULT_MODELS = {
    "gemini": "Gemini 3.6 Flash (High)",
    "gpt": "gpt-5.6-sol (Max)",
    "claude": "claude-opus-4-8",
}

PROVIDER_BIN_NAMES = {
    "gemini": "agy",
    "gpt": "codex",
    "claude": "claude",
}

# Only these files cross from a user's persistent agent configuration into a
# disposable HOME.  In particular, histories, transcripts, memories, caches,
# project state, shell snapshots, plugins, and settings are never copied.
AUTH_FILES = {
    "gemini": (
        (Path(".gemini/antigravity-cli/antigravity-oauth-token"), True),
        (Path(".gemini/antigravity-cli/installation_id"), False),
    ),
    "gpt": ((Path(".codex/auth.json"), True),),
    "claude": ((Path(".claude/.credentials.json"), True),),
}
# agy records a complete trajectory - every tool call with its arguments and the
# permission decision - in a per-conversation SQLite store under the disposable
# HOME the runner owns.  It is the action-level audit trail agy's stdout and
# native debug log do not provide.
AGY_CONVERSATION_DIR = Path(".gemini/antigravity-cli/conversations")
AGY_TRAJECTORY_TABLE = "steps"
# Only command- and target-shaped arguments are read out of a tool call.  File
# *contents* are deliberately excluded: reasoning.md prose written through a
# write tool would otherwise be policy-scanned as if it were an action.
AGY_ACTION_KEYS = (
    "AbsolutePath",
    "BypassSandbox",
    "Command",
    "CommandLine",
    "Cwd",
    "Directory",
    "DirectoryPath",
    "FilePath",
    "Pattern",
    "Query",
    "SearchDirectory",
    "SearchPath",
    "SearchQuery",
    "TargetDirectory",
    "TargetFile",
    "Uri",
    "Url",
)
GEMINI_DISPOSABLE_SETTINGS = {
    "toolPermission": "proceed-in-sandbox",
    "enableTerminalSandbox": True,
    "allowNonWorkspaceAccess": False,
    "permissions": {
        # agy matches a command against `command(<prefix>)`, so `command(*)` on
        # its own is not a reliable wildcard; the tools a DXF/CAD workflow needs
        # are named explicitly.  Granting a shell is safe here because commands
        # run inside agy's terminal sandbox, which was measured to have no
        # network route at all (Errno 101 to raw IPs, and no DNS), and because
        # the bwrap jail exposes no dataset, target, repository, or other task.
        "allow": [
            "read_file(/work)",
            "write_file(/work)",
            "command(*)",
            "command(python3)",
            "command(python)",
            "command(/cad-env/bin/python)",
            "command(bash)",
            "command(sh)",
            "command(ls)",
            "command(cat)",
            "command(head)",
            "command(tail)",
            "command(grep)",
            "command(sed)",
            "command(awk)",
            "command(wc)",
            "command(echo)",
            "command(mkdir)",
            "command(cp)",
            "command(mv)",
        ],
        # `unsandboxed(*)` is deliberately absent: it matches *every* command,
        # not only sandbox-bypass requests, and left the agent with no shell at
        # all.  A bypass is caught instead by the trajectory audit, which fails
        # the task on an executed `BypassSandbox: true` and records a refused
        # one as a blocked attempt.
        "deny": [
            "read_file(/home)",
            "write_file(/home)",
            "read_url(*)",
            "execute_url(*)",
            "mcp(*)",
        ],
        "ask": [],
    },
    "enableTelemetry": False,
    "showFeedbackSurvey": False,
    "showTips": False,
    "verbosity": "high",
}

DENIED_IMPORT_ROOTS = {
    "aiohttp",
    "asyncio",
    "builtins",
    "ctypes",
    "cv2",
    "ezdxf",
    "ftplib",
    "glob",
    "hashlib",
    "http",
    "importlib",
    "marshal",
    "matplotlib",
    "multiprocessing",
    "os",
    "paramiko",
    "PIL",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
}
OTHER_CAD_IMPORT_ROOTS = {"OCP", "FreeCAD", "Part", "bpy", "solid", "trimesh"}
DENIED_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
SUSPICIOUS_LITERAL_PARTS = (
    "/home/",
    "/root/",
    "/workspace",
    "../",
    "drawing.dxf",
    "file://",
    "ftp://",
    "git@",
    "github.com",
    "http://",
    "https://",
    "render_1.png",
    "render_2.png",
    "render_3.png",
)
# Agent-CLI failures that say nothing about the model's CAD ability.  These are
# reported separately so a quota wall or a dropped connection is not scored as a
# reconstruction failure, and so the affected IDs can simply be rerun.
AGENT_INFRA_ERROR_PATTERNS = (
    (
        re.compile(
            r"quota reached|quota exceeded|resource[_ ]exhausted|rate limit|"
            r"too many requests|\b429\b|upgrade your subscription",
            re.IGNORECASE,
        ),
        "provider quota or rate limit reached",
    ),
    (
        re.compile(
            r"not logged in|unauthenticated|unauthorized|\b401\b|\b403\b|"
            r"credentials? (?:expired|invalid)|please sign in",
            re.IGNORECASE,
        ),
        "provider authentication failed",
    ),
    (
        re.compile(
            r"connection (?:refused|reset|closed)|network is unreachable|"
            r"dns lookup failed|\b50[0234]\b|service unavailable|"
            r"internal server error",
            re.IGNORECASE,
        ),
        "provider service unavailable",
    ),
    (
        # The agent CLI keeps its own connection to the model service; when that
        # transport dies the run ends with no answer through no fault of the model.
        # Only the CLI's own phrasings are matched so that a model merely writing
        # about a timeout is not mistaken for one.
        re.compile(
            r"network issue connecting to the server|"
            r"(?:read|write|dial) tcp [\d.:>\- ]+: (?:read|write|connect): "
            r"connection timed out|"
            r"i/o timeout|context deadline exceeded",
            re.IGNORECASE,
        ),
        "provider connection timed out",
    ),
)

TRACE_POLICY_PATTERNS = (
    (
        re.compile(r'"(?:BypassSandbox|unsandboxed)"\s*:\s*true', re.IGNORECASE),
        "sandbox bypass request",
    ),
    (
        re.compile(
            r"\b(?:WebSearch|WebFetch|web_search|google_search|search_query|"
            r"browser(?:_use)?|computer_use|chrome|url_open)\b",
            re.IGNORECASE,
        ),
        "web/browser tool",
    ),
    (re.compile(r"(?:https?|ftp)://", re.IGNORECASE), "external URL access"),
    (
        re.compile(
            r"\b(?:curl|wget|aria2c|nc|ncat|socat|ssh|scp)\b[^\n]*",
            re.IGNORECASE,
        ),
        "network command",
    ),
    (
        re.compile(r"\bgit\s+(?:clone|fetch|pull|ls-remote|remote)\b", re.IGNORECASE),
        "remote git command",
    ),
    (re.compile(r"\bgh\s+\S+", re.IGNORECASE), "GitHub CLI access"),
    (
        re.compile(
            r"\b(?:hashlib|md5sum|sha1sum|sha256sum|b2sum|cksum|openssl\s+dgst)\b",
            re.IGNORECASE,
        ),
        "file hashing",
    ),
    (
        re.compile(
            r"\b(?:import|from)\s+(?:aiohttp|httpx|requests|socket|urllib)\b",
            re.IGNORECASE,
        ),
        "network library",
    ),
    (
        # Only the agent's own credential and config stores are flagged.  Paths
        # are NOT evidence of host access: inside the jail `/home` is the empty
        # per-task disposable HOME, `/root` and the repository are not mounted
        # at all, and the reachable tree is the read-only system directories,
        # /work and /cad-env.  Walking any of those cannot reach the dataset,
        # the targets, another task, or the user's files, so environment
        # probing such as `find /home /opt -name "python*"` is legitimate.
        # The mounts are the control; this pattern covers only the one secret
        # the jail must carry, namely the provider credential in the HOME.
        re.compile(
            r"(?:\.ssh|\.aws|\.netrc|\.config/gcloud|"
            r"\.gemini|\.codex|\.claude)\b",
            re.IGNORECASE,
        ),
        "credential store access",
    ),
    (
        re.compile(
            r"\b(?:pip3?|conda|mamba|apt(?:-get)?|npm|yarn|pnpm)\s+"
            r"(?:install|download|add|update)\b",
            re.IGNORECASE,
        ),
        "package/network access",
    ),
    (
        re.compile(
            r"(?:\.codex/auth\.json|\.claude/\.credentials\.json|"
            r"antigravity-oauth-token)",
            re.IGNORECASE,
        ),
        "credential access",
    ),
)


class RunnerError(RuntimeError):
    """A user-actionable inference runner error."""


@dataclass(frozen=True)
class TaskInput:
    task_id: str
    render_paths: tuple[Path, Path, Path]
    render_labels: tuple[str, str, str]
    dxf_path: Path


@dataclass
class AttemptResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False


def significant_lines(content: bytes) -> list[bytes]:
    """Split evidence files into comparable, whitespace-insensitive lines."""
    return [line.strip() for line in content.split(b"\n") if line.strip()]


def dropped_lines(observed: Sequence[bytes], final: bytes) -> list[bytes]:
    """Observed evidence lines that are no longer an ordered subsequence of final.

    Byte-prefix comparison is too strict for real agents: a tool that rewrites a
    whole file, an atomic temp-file rename, or a chunked write all produce
    intermediate states that are not prefixes of one another even though no
    evidence was lost.  Comparing observed lines against the final content
    tolerates those while still catching content that actually disappeared.
    """
    remaining = significant_lines(final)
    cursor = 0
    missing: list[bytes] = []
    for line in observed:
        try:
            cursor = remaining.index(line, cursor) + 1
        except ValueError:
            missing.append(line)
    return missing


def _replaces_content(previous: bytes | None, content: bytes) -> bool:
    """True only when a write discards content the file already held.

    An agent's write tool routinely produces more than one CLOSE_WRITE per
    logical write: create-then-fill leaves a transient empty file, and buffered
    or chunked writers flush a growing prefix.  Neither discards anything, so
    neither counts as an overwrite.
    """
    if previous is None or not previous.strip():
        return False
    return not content.startswith(previous)


class ArtifactAuditMonitor:
    """Use inotify to retain parent-owned filesystem events and artifact revisions.

    The monitor is the runner's evidence recorder, not its judge.  Every
    revision of every tracked artifact is snapshotted into ``history_dir``
    before the agent can replace it, so an in-pass rewrite destroys nothing the
    auditor can see.  Verdicts are therefore reconciled once, in ``stop()``,
    against the final workdir state:

    * ``integrity_violations`` - evidence the runner can no longer reconstruct,
      or content that must never appear (credentials, oversized artifacts).
    * ``compliance_warnings`` - the agent broke the stated immutable/append-only
      discipline but every revision is still on disk and auditable.
    """

    TRACKED_APPEND_ONLY = {REASONING_NAME, ACTION_TRACE_NAME}
    TRACKED_SNAPSHOTS = {
        FINAL_SELECTION_NAME,
        REASONING_NAME,
        SOURCE_USAGE_NAME,
        ACTION_TRACE_NAME,
    }

    def __init__(
        self,
        workdir: Path,
        audit_path: Path,
        history_dir: Path,
        redactions: Sequence[str] = (),
    ):
        self.workdir = workdir
        self.audit_path = audit_path
        self.history_dir = history_dir
        self._redactions = tuple(value.encode() for value in redactions)
        self.violations: list[str] = []
        self.warnings: list[str] = []
        self._committed_candidates = {
            path.name for path in numbered_model_candidates(workdir)
        }
        self._carried_over_candidates = set(self._committed_candidates)
        # Every distinct line the runner ever saw in an append-only artifact,
        # in observation order, reconciled against the final file in stop().
        self._observed_lines: dict[str, list[bytes]] = {}
        for name in self.TRACKED_APPEND_ONLY:
            path = workdir / name
            if path.is_file():
                self._observed_lines[name] = significant_lines(path.read_bytes())
        self._revision_counts: dict[str, int] = {}
        self._last_snapshot_contents: dict[str, bytes] = {}
        self._candidate_contents: dict[str, bytes] = {
            path.name: path.read_bytes() for path in numbered_model_candidates(workdir)
        }
        self._rewritten_candidates: set[str] = set()
        self._sequence = 0
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stderr = ""

    def start(self) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        # revision_000 of a carried-over candidate is the baseline this pass
        # started from, not something the agent wrote during it, so it is a
        # byte-identical copy of the previous pass's final revision by design.
        # Record which names those are so the duplication is not read as a bug.
        for name in sorted(self._committed_candidates):
            path = self.workdir / name
            self._snapshot(name, path.read_bytes())
        if self._carried_over_candidates:
            (self.history_dir / "carried_over.json").write_text(
                json.dumps(
                    {
                        "note": (
                            "revision_000 of these files is the pre-pass baseline "
                            "copied from the previous attempt, not a new write"
                        ),
                        "files": sorted(self._carried_over_candidates),
                    },
                    indent=2,
                )
                + "\n"
            )
        try:
            self._process = subprocess.Popen(
                [
                    "inotifywait",
                    "--monitor",
                    "--recursive",
                    "--quiet",
                    "--event",
                    "create",
                    "--event",
                    "close_write",
                    "--event",
                    "delete",
                    "--event",
                    "moved_to",
                    "--event",
                    "moved_from",
                    "--format",
                    "%e|%w%f",
                    str(self.workdir),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.violations.append(f"filesystem audit failed to start: {exc}")
            return
        assert self._process.stdout is not None
        self._thread = threading.Thread(target=self._read_events, daemon=True)
        self._thread.start()
        # Give inotifywait time to establish its initial recursive watches.
        time.sleep(0.05)

    def stop(self) -> tuple[list[str], list[str]]:
        """Stop the watcher and return (integrity violations, compliance warnings)."""
        if self._process is None:
            self._reconcile()
            return self._verdict()
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._process.stderr is not None:
            self._stderr = self._process.stderr.read().strip()
            self._process.stderr.close()
        if self._process.stdout is not None:
            self._process.stdout.close()
        if self._process.returncode not in (-15, 0):
            detail = f": {self._stderr}" if self._stderr else ""
            self.violations.append(
                f"filesystem audit exited unexpectedly ({self._process.returncode}){detail}"
            )
        self._reconcile()
        return self._verdict()

    def _verdict(self) -> tuple[list[str], list[str]]:
        return (
            list(dict.fromkeys(self.violations)),
            list(dict.fromkeys(self.warnings)),
        )

    def _read_events(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        with self.audit_path.open("a") as output:
            for line in self._process.stdout:
                event_names, separator, raw_path = line.rstrip("\n").partition("|")
                if not separator:
                    continue
                try:
                    relative = Path(raw_path).relative_to(self.workdir).as_posix()
                except ValueError:
                    self.violations.append(
                        f"filesystem audit reported path outside workdir: {raw_path}"
                    )
                    continue
                self._sequence += 1
                record = {
                    "sequence": self._sequence,
                    "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "events": event_names.split(","),
                    "path": relative,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                self._inspect_event(relative, set(event_names.split(",")))

    def _inspect_event(self, relative: str, events: set[str]) -> None:
        path = self.workdir / relative
        root_name = relative if "/" not in relative else ""
        is_candidate = bool(MODEL_ATTEMPT_PATTERN.fullmatch(root_name))
        committed = bool({"CLOSE_WRITE", "MOVED_TO"} & events)

        if is_candidate and committed:
            if path.is_file() and not path.is_symlink():
                try:
                    content = path.read_bytes()
                except OSError:
                    content = None
                if content is not None:
                    previous = self._candidate_contents.get(root_name)
                    if _replaces_content(previous, content):
                        # Recorded, not fatal: the pre-image is already in
                        # artifact_history.  stop() decides the severity.
                        self._rewritten_candidates.add(root_name)
                    self._candidate_contents[root_name] = content
                    self._committed_candidates.add(root_name)
                    self._snapshot(root_name, content)
        # Deletion/rename is only judged in stop(): `mv tmp target` and
        # write-then-replace both emit removal events for files that are intact
        # by the end of the pass.

        if root_name in self.TRACKED_APPEND_ONLY and committed and path.is_file():
            try:
                content = path.read_bytes()
            except OSError:
                return
            self._record_observed_lines(root_name, content)
            self._snapshot(root_name, content)
        elif root_name in self.TRACKED_SNAPSHOTS and committed and path.is_file():
            try:
                self._snapshot(root_name, path.read_bytes())
            except OSError:
                pass

    def _record_observed_lines(self, name: str, content: bytes) -> None:
        observed = self._observed_lines.setdefault(name, [])
        known = set(observed)
        for line in significant_lines(content):
            if line not in known:
                known.add(line)
                observed.append(line)

    def _reconcile(self) -> None:
        """Grade the final workdir state once the agent has exited."""
        for name in sorted(self._committed_candidates):
            path = self.workdir / name
            if not path.is_file() or path.is_symlink():
                self.violations.append(
                    f"Python candidate was deleted or renamed during agent execution: {name}"
                )
            elif name in self._rewritten_candidates & self._carried_over_candidates:
                # A candidate the runner already validated and gave feedback on
                # must never change; that is the output-overwrite guarantee.
                self.violations.append(
                    "Python candidate from an earlier evaluated pass was "
                    f"overwritten: {name}"
                )
            elif name in self._rewritten_candidates:
                self.warnings.append(
                    f"Python candidate was rewritten in place instead of being "
                    f"superseded by a new number: {name} "
                    f"(every revision is retained in artifact_history)"
                )

        for name in sorted(self._observed_lines):
            observed = self._observed_lines[name]
            if not observed:
                continue
            path = self.workdir / name
            if not path.is_file() or path.is_symlink():
                self.violations.append(
                    f"{name} was deleted or renamed during agent execution"
                )
                continue
            try:
                final = path.read_bytes()
            except OSError as exc:
                self.violations.append(f"{name} became unreadable: {exc}")
                continue
            missing = dropped_lines(observed, final)
            if missing:
                self.warnings.append(
                    f"{name} is not strictly append-only: {len(missing)} of "
                    f"{len(observed)} observed line(s) were edited away "
                    f"(every revision is retained in artifact_history)"
                )

    def _snapshot(self, name: str, content: bytes) -> None:
        if not content.strip():
            # The transient empty file a create-then-fill write leaves behind is
            # not a revision worth keeping in the history.
            return
        limit = MAX_PYTHON_FILE_BYTES if name.endswith(".py") else MAX_REPORT_FILE_BYTES
        if len(content) > limit:
            self.violations.append(
                f"generated artifact exceeds audit snapshot limit: {name}"
            )
            return
        if any(secret in content for secret in self._redactions):
            self.violations.append(
                f"credential content detected in generated artifact: {name}"
            )
            return
        if self._last_snapshot_contents.get(name) == content:
            return
        self._last_snapshot_contents[name] = content
        revision = self._revision_counts.get(name, 0)
        self._revision_counts[name] = revision + 1
        source_name = Path(name)
        destination = self.history_dir / (
            f"{source_name.stem}.revision_{revision:03d}{source_name.suffix}"
        )
        destination.write_bytes(content)


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RunnerError(f"Path does not exist: {path}") from exc


def discover_tasks(data_dir: Path) -> list[TaskInput]:
    """Discover IDs from inputs only; never enumerate or read target files."""
    data_dir = _safe_resolve(data_dir)
    render_root = data_dir / "render_3d"
    dxf_root = data_dir / "techdraw" / "dxf"
    if not render_root.is_dir():
        raise RunnerError(f"Missing render directory: {render_root}")
    if not dxf_root.is_dir():
        raise RunnerError(f"Missing DXF directory: {dxf_root}")

    # The challenge checkout currently calls the answer directory target_step;
    # some prepared variants call it target.  Existence is checked as a layout
    # guard, but its contents are intentionally never listed or opened.
    if not any((data_dir / name).is_dir() for name in ("target", "target_step")):
        raise RunnerError(
            f"Expected {data_dir / 'target'} or {data_dir / 'target_step'} "
            "(the runner will not expose or read it)."
        )

    render_dirs = sorted(p for p in render_root.iterdir() if p.is_dir())
    if len(render_dirs) != 3:
        raise RunnerError(
            f"Expected exactly three render_3d variants, found {len(render_dirs)}: "
            + ", ".join(p.name for p in render_dirs)
        )

    render_maps: list[dict[str, Path]] = []
    for directory in render_dirs:
        symlinks = [p for p in directory.glob("*.png") if p.is_symlink()]
        if symlinks:
            raise RunnerError(
                f"Refusing symlinked render inputs in {directory}: "
                + ", ".join(path.name for path in symlinks[:10])
            )
        files = {p.stem: p for p in directory.glob("*.png") if p.is_file()}
        if not files:
            raise RunnerError(f"No PNG inputs found in {directory}")
        render_maps.append(files)
    dxf_symlinks = [p for p in dxf_root.glob("*.dxf") if p.is_symlink()]
    if dxf_symlinks:
        raise RunnerError(
            f"Refusing symlinked DXF inputs in {dxf_root}: "
            + ", ".join(path.name for path in dxf_symlinks[:10])
        )
    dxf_map = {p.stem: p for p in dxf_root.glob("*.dxf") if p.is_file()}

    input_sets = [set(dxf_map), *(set(mapping) for mapping in render_maps)]
    common_ids = set.intersection(*input_sets)
    all_ids = set.union(*input_sets)
    if not common_ids:
        raise RunnerError("No IDs have one DXF and all three render_3d PNGs.")
    if common_ids != all_ids:
        incomplete = sorted(all_ids - common_ids)
        preview = ", ".join(incomplete[:20])
        suffix = " ..." if len(incomplete) > 20 else ""
        raise RunnerError(
            "Input sets are inconsistent; every ID must have one DXF and all "
            f"three renders. Incomplete IDs: {preview}{suffix}"
        )

    labels = tuple(p.name for p in render_dirs)
    return [
        TaskInput(
            task_id=task_id,
            render_paths=tuple(mapping[task_id] for mapping in render_maps),  # type: ignore[arg-type]
            render_labels=labels,  # type: ignore[arg-type]
            dxf_path=dxf_map[task_id],
        )
        for task_id in sorted(common_ids)
    ]


def infer_provider(model: str, explicit: str = "auto") -> str:
    if explicit != "auto":
        return explicit
    value = model.lower()
    if "gemini" in value:
        return "gemini"
    if "claude" in value or any(x in value for x in ("opus", "sonnet", "haiku")):
        return "claude"
    if value.startswith(("gpt-", "o1", "o3", "o4", "codex")):
        return "gpt"
    raise RunnerError(
        f"Cannot infer provider from model {model!r}; pass --provider explicitly."
    )


def parse_model_effort(model: str) -> tuple[str, str | None]:
    """Split a GPT/Claude model label such as 'gpt-5.4-mini (Extra high)' or
    'claude-opus-4-8 (High)' into (base_model, effort), mirroring the Gemini
    '(High)' convention.  Returns effort=None when no parenthetical is present."""
    match = re.fullmatch(r"\s*(.+?)\s*\(([^()]+)\)\s*", model)
    if not match:
        return model.strip(), None
    base, label = match.group(1), match.group(2).strip()
    effort = EFFORT_LABELS.get(label.lower())
    if effort is None:
        raise RunnerError(
            f"Unknown reasoning effort {label!r}; "
            "use one of Low, Medium, High, Extra high, Max"
        )
    return base, effort


def cad_env_site_packages(python_env: Path) -> str:
    """The jail-visible site-packages of the bound CAD environment."""
    matches = sorted((python_env / "lib").glob("python3.*/site-packages"))
    if not matches:
        return f"{CAD_ENV_ROOT}/lib/site-packages"
    return f"{CAD_ENV_ROOT}/lib/{matches[-1].parent.name}/site-packages"


def build_prompt(
    format_name: str,
    render_labels: Sequence[str] = ("style-1", "style-2", "style-3"),
    repair: bool = False,
) -> str:
    if format_name == "cadquery":
        library = "CadQuery (`import cadquery as cq`)"
        export_hint = f"cq.exporters.export(result, {STEP_NAME!r})"
    else:
        library = "build123d (`from build123d import ...`)"
        export_hint = f"export_step(result, {STEP_NAME!r})"

    repair_block = ""
    if repair:
        repair_block = textwrap.dedent(
            """
            This is a repair pass. Read `runner_feedback.txt`, the current
            numbered `model_attempt_*.py` files, and `execution.log` if present.
            Fix the reported issue while preserving correct geometry. Create a
            NEW numbered Python candidate; never edit an existing Python file.
            Append the new rationale to `reasoning.md`, append repair actions to
            `action_trace.jsonl`, and update the final selection/report files.
            """
        )

    return (
        textwrap.dedent(
            f"""
        Reconstruct one mechanical part as a single watertight 3D solid from an
        engineering drawing. Dimensions are in millimetres.

        This is a strict zero-shot evaluation. The only permitted evidence is:
        - `{RENDER_INPUT_NAMES[0]}` (render style: `{render_labels[0]}`)
        - `{RENDER_INPUT_NAMES[1]}` (render style: `{render_labels[1]}`)
        - `{RENDER_INPUT_NAMES[2]}` (render style: `{render_labels[2]}`)
        - `{DXF_INPUT_NAME}` (the technical drawing in DXF format)
        - on a repair pass only: files created for this same attempt

        You MUST inspect all three PNG renders, not merely one of them, and use
        the DXF for exact drawing geometry/dimensions. Do not use the internet,
        web search, GitHub, dataset/repository searches, file hashes, shell
        history, caches, memories, transcripts, other tasks, graders, reference
        volumes, or pre-existing CAD/model files. Do not look outside the current
        working directory. Do not identify or search for the public dataset.

        A Python environment holding {library.split()[0]}, ezdxf, numpy and
        matplotlib is mounted read-only at `{CAD_ENV_ROOT}`. Run every script and
        every check with `{CAD_ENV_PYTHON}`. Your shell's bare `python3` may be
        the system interpreter, which does not have these libraries; that is
        expected and is not a reason to search the filesystem for another
        interpreter or to install anything.

        {repair_block}
        Required outputs in the current working directory:

        1. Numbered Python candidates: first write `model_attempt_01.py`, then
           use `model_attempt_02.py`, `model_attempt_03.py`, etc. for changed
           candidates. NEVER overwrite, edit, rename, or delete an existing
           `.py` file. Each candidate must be standalone Python using {library},
           create a final object named `result`, produce exactly one solid, and
           export it to `{STEP_NAME}` when run, using an equivalent of:
           `{export_hint}`. Use no input files at runtime and no imports for
           networking, subprocesses, filesystem discovery, hashing, or dynamic
           code execution. Use only the requested CAD library; do not import or
           use the alternative CAD library.

        2. `{FINAL_SELECTION_NAME}`: valid JSON of the form
           `{{"model_file": "model_attempt_NN.py"}}`, selecting the latest
           candidate that you consider final. The trusted runner will copy the
           validated selected candidate to the canonical output `{MODEL_NAME}`.

        3. `{REASONING_NAME}`: a detailed, chronological, externally visible
           reasoning log. Record as much of your reasoning process as the CLI
           and provider permit. At each major stage, append the evidence and
           observations from every input, DXF measurements and calculations,
           cross-view correspondences, alternative interpretations considered,
           assumptions and uncertainty, the modeling decision and why it was
           chosen, tool/action summaries and their results, validation outcomes,
           and reasons for every revision. If internal chain-of-thought is not
           available for disclosure, provide a detailed step-by-step engineering
           reasoning trace instead of merely a short final summary. This file is
           APPEND-ONLY: create it once, then only append new dated/numbered stage
           and attempt sections; never truncate or rewrite earlier content.

        4. `{SOURCE_USAGE_NAME}`: valid JSON with this shape:
           {{
             "render_images_used": [
               {{"file": "render_1.png", "observations": "..."}},
               {{"file": "render_2.png", "observations": "..."}},
               {{"file": "render_3.png", "observations": "..."}}
             ],
             "techdraw_used": {{"file": "drawing.dxf", "observations": "..."}}
           }}
           Include each image exactly once. Every `observations` value must be
           a concrete, non-empty description of evidence unique to that input.

        5. `{ACTION_TRACE_NAME}`: append-only JSON Lines, with one JSON object
           per externally reportable action or decision. Start this file before
           inspecting inputs. Every record must contain an increasing integer
           `sequence`, a `stage`, an `action`, and a non-empty `result`. Append
           records as work happens, not as an end-of-run reconstruction. Include
           separate records for inspecting each of the four inputs, extracting
           dimensions, choosing/rejecting interpretations, writing every Python
           candidate, every local execution/render/validation attempt (including
           failures), and selecting the final candidate. Do not include hidden
           private chain-of-thought or credentials; record observable evidence,
           engineering decisions, actions, and outcomes.

        Do not download anything or seek external information. Iterative local
        execution, rendering, and validation are allowed and encouraged. Before
        starting the FIRST validation, however, you MUST write a complete
        provisional answer to `model_attempt_01.py` (and write the report
        files). Never keep a provisional solution only in a shell heredoc,
        stdin snippet, or chat text. Before every later validation, first write
        the improved solution to the NEXT unused numbered Python filename, then
        validate that new file. Python files are immutable; only `reasoning.md`
        may accumulate changes in-place, and those changes must be append-only.
        A separate trusted runner will also execute the selected candidate and
        return a concrete traceback for a repair pass if needed.
        """
        ).strip()
        + "\n"
    )


def stage_task(task: TaskInput, workdir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for generic_name, label, source in zip(
        RENDER_INPUT_NAMES, task.render_labels, task.render_paths
    ):
        shutil.copyfile(source, workdir / generic_name)
        mapping[generic_name] = label
    shutil.copyfile(task.dxf_path, workdir / DXF_INPUT_NAME)
    mapping[DXF_INPUT_NAME] = "techdraw/dxf"
    return mapping


def prepare_disposable_home(
    provider: str, destination: Path, source_home: Path
) -> None:
    copied_required = False
    for relative, required in AUTH_FILES[provider]:
        source = source_home / relative
        target = destination / relative
        if not source.is_file():
            if required:
                raise RunnerError(
                    f"Missing {provider} CLI credential {source}. Sign in with "
                    f"`{PROVIDER_BIN_NAMES[provider]}` first."
                )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o600)
        copied_required = copied_required or required
    if not copied_required:
        raise RunnerError(f"No required credential copied for {provider}.")
    if provider == "gemini":
        # Headless agy cannot ask for Bash confirmation. This fixed, disposable
        # config lets commands proceed only through agy's terminal sandbox,
        # keeps unsandboxed execution denied, and limits file tools to /work.
        # It does not copy the user's settings, permissions, plugins, or history.
        settings_path = destination / ".gemini/antigravity-cli/settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(
            json.dumps(GEMINI_DISPOSABLE_SETTINGS, indent=2) + "\n"
        )
        settings_path.chmod(0o600)


def credential_redactions(provider: str, source_home: Path) -> tuple[str, ...]:
    """Collect credential string values so parent-owned logs never persist them."""
    values: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, str) and len(value.strip()) >= 8:
            values.add(value.strip())
        elif isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    for relative, _required in AUTH_FILES[provider]:
        path = source_home / relative
        if not path.is_file():
            continue
        text = path.read_text(errors="replace").strip()
        if not text:
            continue
        try:
            collect(json.loads(text))
        except ValueError:
            collect(text)
    return tuple(sorted(values, key=len, reverse=True))


def redact_credentials(text: str, redactions: Sequence[str]) -> str:
    for secret in redactions:
        text = text.replace(secret, "[REDACTED_CREDENTIAL]")
    # agy writes the authenticated account address to its native debug log even
    # though that value is not necessarily present in the copied token file.
    text = re.sub(
        r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])",
        "[REDACTED_EMAIL]",
        text,
    )
    return text


def find_credential_leaks(workdir: Path, redactions: Sequence[str]) -> list[str]:
    encoded = [value.encode() for value in redactions]
    if not encoded:
        return []
    leaks: list[str] = []
    for path in workdir.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if path.stat().st_size > MAX_INTERMEDIATE_FILE_BYTES:
                continue
            content = path.read_bytes()
        except OSError:
            continue
        if any(secret in content for secret in encoded):
            leaks.append(path.relative_to(workdir).as_posix())
    return sorted(leaks)


def _bwrap_base(
    *,
    workdir: Path,
    disposable_home: Path,
    executable: Path,
    python_env: Path,
    audit_dir: Path | None = None,
) -> list[str]:
    """Minimal FS jail for the networked CLI control process.

    Network is retained because the CLI itself must contact its model service.
    Provider-specific tool restrictions prevent web tools; generated code is
    run separately with the network namespace removed.
    """
    args = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
    ]
    for system_path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(system_path).exists():
            args += ["--ro-bind", system_path, system_path]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/home",
        "--dir",
        "/home/runner",
        "--bind",
        str(disposable_home),
        "/home/runner",
        "--dir",
        "/work",
        "--bind",
        str(workdir),
        "/work",
        "--dir",
        "/cad-env",
        "--ro-bind",
        str(python_env),
        "/cad-env",
        "--ro-bind",
        str(executable),
        "/agent-bin",
        "--chdir",
        "/work",
        "--clearenv",
        "--setenv",
        "HOME",
        "/home/runner",
        "--setenv",
        "USER",
        "runner",
        "--setenv",
        "LOGNAME",
        "runner",
        "--setenv",
        "PATH",
        "/cad-env/bin:/usr/bin:/bin",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        # The agent CLI runs its shell in its own nested sandbox, which resets
        # PATH, so `python3` there resolves to the system interpreter rather
        # than /cad-env/bin/python.  PYTHONPATH survives and makes that system
        # interpreter find the CAD libraries anyway; without it the agent burns
        # a dozen tool calls hunting for a usable environment.
        "--setenv",
        "PYTHONPATH",
        cad_env_site_packages(python_env),
        "--hostname",
        "zero-shot-jail",
    ]
    if audit_dir is not None:
        args += ["--dir", "/audit", "--bind", str(audit_dir), "/audit"]
    return args


def build_agent_command(
    provider: str,
    model: str,
    prompt: str,
    executable: Path,
    workdir: Path,
    disposable_home: Path,
    python_env: Path,
    timeout_s: int,
    audit_dir: Path | None = None,
) -> tuple[list[str], str | None]:
    base = _bwrap_base(
        workdir=workdir,
        disposable_home=disposable_home,
        executable=executable,
        python_env=python_env,
        audit_dir=audit_dir if provider == "gemini" else None,
    )
    if provider == "gemini":
        command = [
            "/agent-bin",
            "-p",
            prompt,
            "--mode",
            "accept-edits",
            "--sandbox",
            "--add-dir",
            "/work",
            "--add-dir",
            "/cad-env",
            "--log-file",
            "/audit/agy-cli.log",
            "--print-timeout",
            f"{max(1, timeout_s - 30)}s",
        ]
        command += ["--model", model]
        return base + command, None

    if provider == "gpt":
        base_model, effort = parse_model_effort(model)
        command = [
            "/agent-bin",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--json",
            "--strict-config",
            "--color",
            "never",
            "--sandbox",
            "workspace-write",
            "--cd",
            "/work",
            "--model",
            base_model,
            "--config",
            "sandbox_workspace_write.network_access=false",
            "--config",
            'approval_policy="never"',
            # Web search is on by default and runs server-side, outside the
            # sandbox network namespace, so network_access=false does not stop
            # it.  Disabling it here keeps the GPT path networkless like the rest.
            "--config",
            'web_search="disabled"',
            "--output-last-message",
            "/work/final_response.txt",
        ]
        for feature in GPT_DISABLED_FEATURES:
            command += ["--disable", feature]
        if effort:
            command += ["--config", f'model_reasoning_effort="{effort}"']
        for image_name in RENDER_INPUT_NAMES:
            command += ["--image", f"/work/{image_name}"]
        command.append("-")
        # Codex reads auth from CODEX_HOME even with --ignore-user-config.
        base += ["--setenv", "CODEX_HOME", "/home/runner/.codex"]
        return base + command, prompt

    base_model, effort = parse_model_effort(model)
    command = [
        "/agent-bin",
        "--print",
        "--model",
        base_model,
        "--safe-mode",
        "--no-chrome",
        "--disable-slash-commands",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        "{}",
        "--settings",
        json.dumps(
            {
                "sandbox": {
                    "enabled": True,
                    "autoAllowBashIfSandboxed": True,
                    "allowUnsandboxedCommands": False,
                    "network": {"allowedDomains": []},
                }
            },
            separators=(",", ":"),
        ),
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--tools",
        "Read,Write,Edit,Bash",
        "--disallowedTools",
        "WebSearch,WebFetch,NotebookEdit,Task,Skill,"
        "Bash(curl *),Bash(wget *),Bash(git *),Bash(gh *),"
        "Bash(nc *),Bash(ncat *),Bash(socat *),Bash(ssh *),"
        "Bash(scp *),Bash(md5sum *),Bash(sha1sum *),"
        "Bash(sha256sum *),Bash(pip *),Bash(pip3 *)",
        "--output-format",
        "stream-json",
        "--verbose",
        prompt,
    ]
    if effort:
        command += ["--effort", effort]
    base += ["--setenv", "CLAUDE_CONFIG_DIR", "/home/runner/.claude"]
    return base + command, None


def run_agent(
    command: Sequence[str],
    stdin_text: str | None,
    timeout_s: int,
    stream_audit_path: Path,
    redactions: Sequence[str] = (),
) -> AttemptResult:
    """Run an agent while the parent timestamps immutable stdout/stderr events."""
    started = time.monotonic()
    stream_audit_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    write_lock = threading.Lock()
    sequence = 0

    def capture(stream: object, stream_name: str, destination: list[str]) -> None:
        nonlocal sequence
        readline = getattr(stream, "readline")
        with stream_audit_path.open("a") as audit_file:
            while True:
                chunk = readline()
                if chunk == "":
                    break
                chunk = redact_credentials(chunk, redactions)
                destination.append(chunk)
                with write_lock:
                    sequence += 1
                    record = {
                        "sequence": sequence,
                        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "elapsed_s": round(time.monotonic() - started, 6),
                        "stream": stream_name,
                        "text": chunk.rstrip("\n"),
                    }
                    audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    audit_file.flush()

    try:
        proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        message = redact_credentials(str(exc), redactions)
        stream_audit_path.write_text(
            json.dumps(
                {
                    "sequence": 1,
                    "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "elapsed_s": round(time.monotonic() - started, 6),
                    "stream": "runner",
                    "text": message,
                }
            )
            + "\n"
        )
        return AttemptResult(-1, "", message + "\n", time.monotonic() - started)

    assert proc.stdout is not None
    assert proc.stderr is not None
    stdout_thread = threading.Thread(
        target=capture, args=(proc.stdout, "stdout", stdout_parts), daemon=True
    )
    stderr_thread = threading.Thread(
        target=capture, args=(proc.stderr, "stderr", stderr_parts), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    if proc.stdin is not None:
        try:
            if stdin_text is not None:
                proc.stdin.write(stdin_text)
            proc.stdin.close()
        except BrokenPipeError:
            pass

    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    proc.stdout.close()
    proc.stderr.close()
    stderr = "".join(stderr_parts)
    if timed_out:
        stderr += "\nrunner wall-clock timeout\n"
    return AttemptResult(
        -1 if timed_out else proc.returncode,
        "".join(stdout_parts),
        stderr,
        time.monotonic() - started,
        timed_out=timed_out,
    )


def _structured_trace_events(text: str) -> tuple[list[dict[str, object]], list[str]]:
    objects: list[object] = []
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            objects.append(json.loads(line))
        except ValueError as exc:
            errors.append(f"line {line_number}: {exc}")

    events: list[dict[str, object]] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            type_name = str(value.get("type", "")).lower()
            is_event = any(
                marker in type_name
                for marker in (
                    "tool",
                    "command_execution",
                    "file_change",
                    "web_search",
                    "browser",
                    "computer",
                    "mcp",
                    "function_call",
                )
            ) or any(key in value for key in ("tool_name", "toolName"))
            if is_event and "tool_result" not in type_name:
                events.append(value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for value in objects:
        visit(value)
    return events, errors


def harvest_agy_trajectory(disposable_home: Path, destination: Path) -> list[Path]:
    """Copy agy's conversation trajectory stores out of the disposable HOME.

    The disposable HOME is deleted as soon as the agent exits, so the store must
    be consolidated first.  ``Connection.backup`` is used rather than a file copy
    because the live database keeps recent steps in a write-ahead log.
    """
    source_dir = disposable_home / AGY_CONVERSATION_DIR
    if not source_dir.is_dir():
        return []
    destination.mkdir(parents=True, exist_ok=True)
    harvested: list[Path] = []
    for database in sorted(source_dir.glob("*.db")):
        target = destination / database.name
        source_connection = None
        target_connection = None
        try:
            source_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            target_connection = sqlite3.connect(target)
            source_connection.backup(target_connection)
        except sqlite3.Error:
            continue
        finally:
            for connection in (target_connection, source_connection):
                if connection is not None:
                    connection.close()
        harvested.append(target)
    return harvested


# A trajectory step frames a call as `\x12<len>tool_name\x1a<len>{"Arg":...}`.
# The length prefix after \x1a is one to a few bytes and may itself be printable,
# so the separator is matched as opaque bytes up to the start of the JSON object.
_AGY_TOOL_CALL = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]{2,48})\x1a.{1,6}?(?=\{\")", re.DOTALL
)


def _repair_mojibake(value: str) -> str:
    """Recover UTF-8 text read out of a protobuf blob decoded as latin-1."""
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return value


def _extract_agy_tool_calls(blob: bytes) -> list[dict[str, object]]:
    """Pull ``tool_name {json}`` pairs out of a protobuf-encoded trajectory step.

    The blob is decoded as latin-1 so byte offsets survive one-to-one; any
    non-ASCII inside the recovered JSON strings is repaired afterwards.
    """
    text = blob.decode("latin-1")
    decoder = json.JSONDecoder()
    calls: list[dict[str, object]] = []
    for match in _AGY_TOOL_CALL.finditer(text):
        try:
            arguments, _ = decoder.raw_decode(text, match.end())
        except ValueError:
            continue
        if not isinstance(arguments, dict):
            continue
        action: dict[str, object] = {"tool": _repair_mojibake(match.group(1))}
        for key in AGY_ACTION_KEYS:
            if key not in arguments:
                continue
            value = arguments[key]
            action[key] = _repair_mojibake(value) if isinstance(value, str) else value
        calls.append(action)
    return calls


def agy_trajectory_events(
    databases: Sequence[Path], redactions: Sequence[str] = ()
) -> tuple[list[str], list[str]]:
    """Normalize agy trajectory stores into deduplicated tool-action strings.

    agy repeats one call across several rows (the model's proposal, the
    execution, and its result), so calls are keyed by their action and merged:
    the earliest step wins and any permission denial recorded on a later row is
    carried onto it.
    """
    parse_errors: list[str] = []
    merged: dict[str, dict[str, object]] = {}
    for database in databases:
        connection = None
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            rows = connection.execute(
                f"SELECT idx, metadata, error_details, step_payload "
                f"FROM {AGY_TRAJECTORY_TABLE} ORDER BY idx"  # noqa: S608
            ).fetchall()
        except sqlite3.Error as exc:
            parse_errors.append(f"{database.name}: {exc}")
            continue
        finally:
            if connection is not None:
                connection.close()
        for index, metadata, error_details, payload in rows:
            # agy words the refusal after the rule that matched, so it reads
            # "for command(...)" or "for unsandboxed(...)" depending on the case.
            denied = isinstance(error_details, bytes) and (
                b"Permission denied for " in error_details
            )
            for blob in (metadata, payload):
                if not isinstance(blob, bytes):
                    continue
                for action in _extract_agy_tool_calls(blob):
                    key = json.dumps(action, ensure_ascii=False, sort_keys=True)
                    existing = merged.get(key)
                    if existing is None:
                        action["step"] = index
                        action["blocked_by_permission_rule"] = denied
                        merged[key] = action
                    elif denied:
                        existing["blocked_by_permission_rule"] = True

    event_texts = [
        redact_credentials(
            json.dumps(action, ensure_ascii=False, sort_keys=True), redactions
        )
        for action in sorted(merged.values(), key=lambda item: item["step"])  # type: ignore[arg-type,return-value]
    ]
    return event_texts, parse_errors


def _trace_event_action(event: dict[str, object]) -> str:
    action: dict[str, object] = {}
    for key in (
        "type",
        "name",
        "tool_name",
        "toolName",
        "command",
        "input",
        "arguments",
        "args",
        "query",
        "url",
    ):
        if key in event:
            action[key] = event[key]
    return json.dumps(action or event, ensure_ascii=False)


def audit_provider_trace(
    provider: str,
    result: AttemptResult,
    native_log: Path | None,
    event_output: Path,
    redactions: Sequence[str] = (),
    trajectory_databases: Sequence[Path] = (),
) -> dict[str, object]:
    """Normalize provider traces and apply policy only to actions, not prose."""
    event_texts: list[str] = []
    parse_errors: list[str] = []
    coverage = "structured-jsonl"
    complete = provider in {"gpt", "claude"}

    if provider in {"gpt", "claude"}:
        events, parse_errors = _structured_trace_events(result.stdout)
        event_texts = [_trace_event_action(event) for event in events]
        if result.returncode == 0 and not result.stdout.strip():
            parse_errors.append("successful CLI invocation emitted no structured trace")
        if result.returncode == 0 and not events:
            parse_errors.append(
                "successful CLI invocation emitted no tool/action events"
            )
    else:
        # Primary source: agy's own trajectory store, which records every tool
        # call with its arguments.  The native debug log is kept as a fallback
        # for agy builds that do not write a conversation database.
        coverage = "agy-trajectory-store"
        event_texts, parse_errors = agy_trajectory_events(
            trajectory_databases, redactions
        )
        complete = bool(trajectory_databases) and not parse_errors
        if not trajectory_databases:
            coverage = "agy-native-log-and-command-events"
            if result.returncode == 0:
                parse_errors.append(
                    "successful CLI invocation left no agy trajectory store"
                )
        if native_log is not None and native_log.is_file():
            native_text = redact_credentials(
                native_log.read_text(errors="replace"), redactions
            )
            for match in re.finditer(
                r"command_line:(.*?)\s+command_output:", native_text, re.DOTALL
            ):
                event_texts.append(match.group(1).strip())
            for line in native_text.splitlines():
                if re.search(
                    r"\b(?:WebSearch|WebFetch|web_search|browser_use)\b",
                    line,
                    re.IGNORECASE,
                ):
                    event_texts.append(line)
        elif not trajectory_databases:
            parse_errors.append("agy native audit log is missing")

    event_output.parent.mkdir(parents=True, exist_ok=True)
    with event_output.open("w") as output:
        for sequence_number, event_text in enumerate(event_texts, 1):
            output.write(
                json.dumps(
                    {
                        "sequence": sequence_number,
                        "provider": provider,
                        "event": event_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    # An action the provider's permission layer refused never happened, so it
    # cannot have leaked anything.  Policy is enforced on what actually ran;
    # refused attempts are surfaced separately as sandbox evidence rather than
    # failing a task whose reconstruction is otherwise sound.
    executed, blocked = _partition_blocked_events(event_texts)
    violations = detect_trace_policy_violations("\n".join(executed), "")
    blocked_attempts = detect_trace_policy_violations("\n".join(blocked), "")
    return {
        "coverage": coverage,
        "complete": complete,
        "event_count": len(event_texts),
        "blocked_event_count": len(blocked),
        "trajectory_store_count": len(trajectory_databases),
        "parse_errors": parse_errors,
        "policy_violations": violations,
        "blocked_policy_attempts": blocked_attempts,
    }


def _partition_blocked_events(
    event_texts: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split trace events into those that ran and those the provider refused."""
    executed: list[str] = []
    blocked: list[str] = []
    for text in event_texts:
        try:
            event = json.loads(text)
        except ValueError:
            event = None
        if isinstance(event, dict) and event.get("blocked_by_permission_rule"):
            blocked.append(text)
        else:
            executed.append(text)
    return executed, blocked


def classify_agent_failure(result: AttemptResult) -> str | None:
    """Name the infrastructure fault behind a failed CLI run, if there is one."""
    if result.timed_out:
        return "agent CLI exceeded the runner wall-clock timeout"
    text = result.stderr + "\n" + result.stdout
    for pattern, label in AGENT_INFRA_ERROR_PATTERNS:
        if pattern.search(text):
            return label
    return None


def detect_trace_policy_violations(stdout: str, stderr: str) -> list[str]:
    """Flag explicit cheating attempts recorded in provider tool traces."""
    trace = stdout + "\n" + stderr
    return [label for pattern, label in TRACE_POLICY_PATTERNS if pattern.search(trace)]


def numbered_model_candidates(workdir: Path) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    for path in workdir.glob("model_attempt_*.py"):
        match = MODEL_ATTEMPT_PATTERN.fullmatch(path.name)
        if match and path.is_file() and not path.is_symlink():
            candidates.append((int(match.group(1)), path))
    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def selected_model_candidate(workdir: Path) -> tuple[Path | None, list[str]]:
    selection_path = workdir / FINAL_SELECTION_NAME
    try:
        if selection_path.is_symlink():
            raise ValueError(f"{FINAL_SELECTION_NAME} must not be a symlink")
        selection = json.loads(selection_path.read_text())
        filename = selection["model_file"]
        if not isinstance(filename, str) or not MODEL_ATTEMPT_PATTERN.fullmatch(
            filename
        ):
            raise ValueError("model_file must match model_attempt_NN.py")
        candidate = workdir / filename
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"selected candidate does not exist: {filename}")
        return candidate, []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, [f"invalid {FINAL_SELECTION_NAME}: {exc}"]


def immutable_trace_issues(
    workdir: Path,
    previous_python: dict[str, bytes],
    previous_reasoning: bytes | None,
    previous_action_trace: bytes | None = None,
) -> tuple[list[str], list[str]]:
    """Compare the workdir against the state the previous pass was graded on.

    Returns ``(violations, warnings)``.  Mutating a Python candidate the runner
    already executed and reported on is a real overwrite of graded output and
    stays fatal.  A report file that was edited rather than appended to is only
    a discipline warning, because the previous pass's copy is preserved under
    ``artifact_history/agent_attempt_NN/``.
    """
    violations: list[str] = []
    warnings: list[str] = []
    for name, old_content in previous_python.items():
        path = workdir / name
        if not path.is_file():
            violations.append(f"existing Python candidate was deleted: {name}")
        elif path.read_bytes() != old_content:
            violations.append(f"existing Python candidate was overwritten: {name}")
    for name, old_content in (
        (REASONING_NAME, previous_reasoning),
        (ACTION_TRACE_NAME, previous_action_trace),
    ):
        if old_content is None:
            continue
        path = workdir / name
        if not path.is_file():
            violations.append(f"{name} was deleted between agent passes")
        elif dropped_lines(significant_lines(old_content), path.read_bytes()):
            warnings.append(
                f"{name} lost earlier content between agent passes "
                f"(the graded copy is retained in artifact_history)"
            )
    return violations, warnings


def validate_generated_code(path: Path, format_name: str) -> list[str]:
    issues: list[str] = []
    if not path.is_file():
        return [f"missing {path.name}"]
    if path.stat().st_size > MAX_PYTHON_FILE_BYTES:
        return [f"{path.name} exceeds 1 MB"]
    source = path.read_text(errors="replace")
    if source.lstrip().startswith("```"):
        return [f"{path.name} contains a Markdown fence instead of plain Python"]
    try:
        tree = ast.parse(source, filename=path.name)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    required_root = "cadquery" if format_name == "cadquery" else "build123d"
    alternative_root = "build123d" if format_name == "cadquery" else "cadquery"
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DENIED_CALL_NAMES:
                issues.append(f"denied call: {node.func.id}()")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            issues.append(f"denied dunder attribute: {node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            for part in SUSPICIOUS_LITERAL_PARTS:
                if part in lowered:
                    issues.append(f"suspicious path/URL literal containing {part!r}")
                    break
    denied = sorted(imports & DENIED_IMPORT_ROOTS)
    if denied:
        issues.append("denied imports: " + ", ".join(denied))
    if required_root not in imports:
        issues.append(f"expected an import from {required_root}")
    if alternative_root in imports:
        issues.append(
            f"selected format {format_name} must not import {alternative_root}"
        )
    other_cad = sorted(imports & OTHER_CAD_IMPORT_ROOTS)
    if other_cad:
        issues.append(
            f"selected format {format_name} must not import other CAD libraries: "
            + ", ".join(other_cad)
        )
    if not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "result"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
        for node in ast.walk(tree)
    ):
        issues.append("script must assign the final shape to `result`")
    if STEP_NAME not in source:
        issues.append(f"script does not reference {STEP_NAME!r}")
    return list(dict.fromkeys(issues))


def validate_auxiliary_outputs(workdir: Path) -> list[str]:
    issues: list[str] = []
    reasoning = workdir / REASONING_NAME
    if reasoning.is_file() and reasoning.stat().st_size > MAX_REPORT_FILE_BYTES:
        issues.append(f"{REASONING_NAME} exceeds report size limit")
    if (
        reasoning.is_symlink()
        or not reasoning.is_file()
        or len(reasoning.read_text(errors="replace").strip()) < 40
    ):
        issues.append(f"missing or too-short {REASONING_NAME}")

    usage_path = workdir / SOURCE_USAGE_NAME
    try:
        if usage_path.stat().st_size > MAX_REPORT_FILE_BYTES:
            raise ValueError(f"{SOURCE_USAGE_NAME} exceeds report size limit")
        if usage_path.is_symlink():
            raise ValueError(f"{SOURCE_USAGE_NAME} must not be a symlink")
        usage = json.loads(usage_path.read_text())
        image_items = usage["render_images_used"]
        if not isinstance(image_items, list):
            raise TypeError("render_images_used must be a list")
        reported = [item.get("file") for item in image_items]
        if len(reported) != 3 or set(reported) != set(RENDER_INPUT_NAMES):
            issues.append(
                f"{SOURCE_USAGE_NAME} must report all three render images exactly once"
            )
        observations: list[str] = []
        for item in image_items:
            observation = item.get("observations")
            if not isinstance(observation, str) or len(observation.strip()) < 20:
                issues.append(
                    f"{SOURCE_USAGE_NAME} needs a concrete observation for "
                    f"{item.get('file', 'an image')}"
                )
            else:
                observations.append(observation.strip().casefold())
        if len(observations) == 3 and len(set(observations)) != 3:
            issues.append(
                f"{SOURCE_USAGE_NAME} observations must distinguish the three renders"
            )
        techdraw = usage.get("techdraw_used", {})
        if techdraw.get("file") != DXF_INPUT_NAME:
            issues.append(f"{SOURCE_USAGE_NAME} must report {DXF_INPUT_NAME}")
        techdraw_observation = techdraw.get("observations")
        if (
            not isinstance(techdraw_observation, str)
            or len(techdraw_observation.strip()) < 20
        ):
            issues.append(f"{SOURCE_USAGE_NAME} needs a concrete DXF observation")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        issues.append(f"invalid {SOURCE_USAGE_NAME}: {exc}")
    return issues


def validate_action_trace(
    workdir: Path,
    filesystem_audit_path: Path,
    previous_candidates: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """Check the chain-of-thought action trace.

    Returns ``(issues, warnings)``.  Issues describe a trace that cannot be read
    as a log at all and are worth a repair pass; warnings describe a log that is
    thinner or less strictly ordered than requested but still usable evidence,
    and must not discard an otherwise valid reconstruction.
    """
    issues: list[str] = []
    warnings: list[str] = []
    trace_path = workdir / ACTION_TRACE_NAME
    if trace_path.is_symlink() or not trace_path.is_file():
        return [f"missing {ACTION_TRACE_NAME}"], warnings
    if trace_path.stat().st_size > MAX_REPORT_FILE_BYTES:
        return [f"{ACTION_TRACE_NAME} exceeds report size limit"], warnings
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        trace_path.read_text(errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            issues.append(f"invalid {ACTION_TRACE_NAME} line {line_number}: {exc}")
            continue
        if not isinstance(record, dict):
            issues.append(f"{ACTION_TRACE_NAME} line {line_number} must be an object")
            continue
        records.append(record)

    if not records:
        issues.append(f"{ACTION_TRACE_NAME} contains no usable records")
    elif len(records) < 6:
        warnings.append(
            f"{ACTION_TRACE_NAME} contains only {len(records)} records; at least "
            "six chronological records were requested"
        )
    # The prompt asks for an increasing `sequence`, so grade increasing, not
    # dense-from-1: a numbering that skips or restarts across repair passes is
    # still an unambiguous ordering.
    sequences = [record.get("sequence") for record in records]
    if not all(isinstance(value, int) for value in sequences):
        issues.append(f"{ACTION_TRACE_NAME} records need an integer `sequence`")
    elif any(a >= b for a, b in zip(sequences, sequences[1:])):  # type: ignore[operator]
        issues.append(f"{ACTION_TRACE_NAME} sequence values must strictly increase")
    for index, record in enumerate(records, 1):
        for field in ("stage", "action", "result"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                issues.append(
                    f"{ACTION_TRACE_NAME} record {index} needs non-empty {field}"
                )

    serialized = [json.dumps(record, ensure_ascii=False) for record in records]
    for input_name in (*RENDER_INPUT_NAMES, DXF_INPUT_NAME):
        if not any(input_name in record for record in serialized):
            warnings.append(
                f"{ACTION_TRACE_NAME} does not name {input_name} in any record"
            )

    warnings += _action_trace_chronology_warnings(
        filesystem_audit_path, previous_candidates
    )
    return list(dict.fromkeys(issues)), list(dict.fromkeys(warnings))


def _action_trace_chronology_warnings(
    filesystem_audit_path: Path, previous_candidates: Sequence[str]
) -> list[str]:
    """Check the trace was opened before this pass wrote its first candidate.

    Candidates carried over from an earlier pass are excluded: on a repair pass
    they are already on disk when the watcher starts, so comparing against them
    would flag every repair pass regardless of what the agent did.  A late trace
    only means the log may be partly reconstructed after the fact, which is a
    disclosure-quality signal rather than evidence of cheating.
    """
    carried_over = set(previous_candidates)
    try:
        audit_records = [
            json.loads(line)
            for line in filesystem_audit_path.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, ValueError):
        return ["filesystem audit cannot establish action-trace chronology"]
    candidate_first = [
        record["sequence"]
        for record in audit_records
        if MODEL_ATTEMPT_PATTERN.fullmatch(str(record.get("path", "")))
        and record.get("path") not in carried_over
    ]
    if not candidate_first:
        return []
    trace_first = [
        record["sequence"]
        for record in audit_records
        if record.get("path") == ACTION_TRACE_NAME
    ]
    if not trace_first or min(trace_first) >= min(candidate_first):
        return [
            f"{ACTION_TRACE_NAME} was not opened before this pass wrote its "
            "first Python candidate; the log may be partly reconstructed"
        ]
    return []


def _execution_bwrap(workdir: Path, python_env: Path, argv: Sequence[str]) -> list[str]:
    args = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
    ]
    for system_path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(system_path).exists():
            args += ["--ro-bind", system_path, system_path]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--bind",
        str(workdir),
        "/work",
        "--dir",
        "/cad-env",
        "--ro-bind",
        str(python_env),
        "/cad-env",
        "--chdir",
        "/work",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp/empty-home",
        "--setenv",
        "PATH",
        "/cad-env/bin:/usr/bin:/bin",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--hostname",
        "cad-executor",
    ]
    return args + list(argv)


STEP_VERIFY_CODE = r"""
import json
import cadquery as cq

shape = cq.importers.importStep("output.step")
solids = shape.solids().vals()
valid = [solid.isValid() for solid in solids]
print(json.dumps({"solid_count": len(solids), "valid": valid}))
raise SystemExit(0 if len(solids) == 1 and all(valid) else 2)
"""


def execute_and_verify(
    workdir: Path, python_env: Path, timeout_s: int, model_filename: str = MODEL_NAME
) -> tuple[bool, str, dict[str, object] | None]:
    step = workdir / STEP_NAME
    if step.exists():
        step.unlink()
    with tempfile.TemporaryDirectory(prefix="drawing2cad-exec-") as execution_name:
        execution_dir = Path(execution_name)
        shutil.copyfile(workdir / model_filename, execution_dir / model_filename)
        isolated_step = execution_dir / STEP_NAME
        command = _execution_bwrap(
            execution_dir, python_env, ["/cad-env/bin/python", model_filename]
        )
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command, text=True, capture_output=True, timeout=timeout_s
            )
            elapsed = time.monotonic() - started
            log = (
                f"exit={proc.returncode} elapsed={elapsed:.2f}s\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            return (
                False,
                f"execution timeout\n--- stdout ---\n{stdout}\n"
                f"--- stderr ---\n{stderr}\n",
                None,
            )

        if proc.returncode != 0:
            return False, log, None
        if not isolated_step.is_file() or isolated_step.stat().st_size < 100:
            return False, log + f"{STEP_NAME} was not produced or is empty\n", None

        verify_command = _execution_bwrap(
            execution_dir,
            python_env,
            ["/cad-env/bin/python", "-c", STEP_VERIFY_CODE],
        )
        verify = subprocess.run(
            verify_command, text=True, capture_output=True, timeout=timeout_s
        )
        log += (
            f"--- STEP verification (exit={verify.returncode}) ---\n"
            f"{verify.stdout}{verify.stderr}\n"
        )
        verification: dict[str, object] | None = None
        try:
            verification = json.loads(verify.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
        if verify.returncode == 0:
            shutil.copyfile(isolated_step, step)
        return verify.returncode == 0, log, verification


def _write_agent_log(
    path: Path, result: AttemptResult, provider: str, attempt: int
) -> None:
    path.write_text(
        f"provider={provider}\nattempt={attempt}\nexit={result.returncode}\n"
        f"elapsed_s={result.elapsed_s:.3f}\ntimed_out={result.timed_out}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n"
    )


def _copy_artifacts(
    workdir: Path,
    result_dir: Path,
    *,
    success: bool,
    redactions: Sequence[str] = (),
) -> None:
    destination_root = result_dir if success else result_dir / QUARANTINE_DIR_NAME
    destination_root.mkdir(parents=True, exist_ok=True)
    for name in (
        MODEL_NAME,
        FINAL_SELECTION_NAME,
        STEP_NAME,
        REASONING_NAME,
        SOURCE_USAGE_NAME,
        ACTION_TRACE_NAME,
        "execution.log",
        "final_response.txt",
        "runner_feedback.txt",
    ):
        source = workdir / name
        if source.is_file() and not source.is_symlink():
            content = source.read_bytes()
            if any(secret.encode() in content for secret in redactions):
                continue
            shutil.copyfile(source, destination_root / name)


def _copy_intermediate_artifacts(
    workdir: Path, result_dir: Path, redactions: Sequence[str] = ()
) -> dict[str, object]:
    """Preserve agent-created work files without carrying them into later tasks."""
    destination_root = result_dir / INTERMEDIATES_DIR_NAME
    destination_root.mkdir(parents=True, exist_ok=True)
    staged_inputs = {*RENDER_INPUT_NAMES, DXF_INPUT_NAME}
    final_outputs = {
        MODEL_NAME,
        FINAL_SELECTION_NAME,
        STEP_NAME,
        REASONING_NAME,
        SOURCE_USAGE_NAME,
        ACTION_TRACE_NAME,
        "execution.log",
        "final_response.txt",
        "runner_feedback.txt",
    }
    copied: list[str] = []
    skipped: list[str] = []
    total = 0
    for source in sorted(workdir.rglob("*")):
        relative = source.relative_to(workdir)
        relative_text = relative.as_posix()
        if len(relative.parts) == 1 and source.name in staged_inputs | final_outputs:
            continue
        if source.is_symlink() or not source.is_file():
            if source.is_symlink():
                skipped.append(f"{relative_text}: symlink")
            continue
        size = source.stat().st_size
        if size > MAX_INTERMEDIATE_FILE_BYTES:
            skipped.append(f"{relative_text}: exceeds per-file limit")
            continue
        if total + size > MAX_INTERMEDIATES_TOTAL_BYTES:
            skipped.append(f"{relative_text}: exceeds total limit")
            continue
        if redactions:
            try:
                content = source.read_bytes()
            except OSError:
                skipped.append(f"{relative_text}: unreadable")
                continue
            if any(secret.encode() in content for secret in redactions):
                skipped.append(f"{relative_text}: credential content")
                continue
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(relative_text)
        total += size
    manifest = {
        "copied": copied,
        "skipped": skipped,
        "total_bytes": total,
        "original_inputs_excluded": sorted(staged_inputs),
        "reused_by_future_tasks": False,
    }
    (destination_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def _clean_previous_result_artifacts(result_dir: Path) -> None:
    """Remove only files owned by this runner before an explicit rerun."""
    exact_names = {
        MODEL_NAME,
        FINAL_SELECTION_NAME,
        STEP_NAME,
        REASONING_NAME,
        SOURCE_USAGE_NAME,
        ACTION_TRACE_NAME,
        "execution.log",
        "final_response.txt",
        "metadata.json",
    }
    for managed_dir in (
        result_dir / INTERMEDIATES_DIR_NAME,
        result_dir / "python_history",
        result_dir / "artifact_history",
        result_dir / AUDIT_DIR_NAME,
        result_dir / QUARANTINE_DIR_NAME,
    ):
        if managed_dir.is_symlink() or managed_dir.is_file():
            managed_dir.unlink()
        elif managed_dir.is_dir():
            shutil.rmtree(managed_dir)
    for path in result_dir.iterdir():
        if not path.is_file():
            continue
        if (
            path.name in exact_names
            or re.fullmatch(r"agent-attempt-\d+\.log", path.name)
            or re.fullmatch(r"prompt-\d+\.md", path.name)
            or re.fullmatch(r"agent-stream-attempt-\d+\.jsonl", path.name)
        ):
            path.unlink()


def run_task(
    task: TaskInput,
    *,
    provider: str,
    model: str,
    format_name: str,
    out_dir: Path,
    executable: Path,
    python_env: Path,
    source_home: Path,
    timeout_s: int,
    execute_timeout_s: int,
    max_repairs: int,
) -> dict[str, object]:
    result_dir = out_dir / task.task_id
    result_dir.mkdir(parents=True, exist_ok=True)
    _clean_previous_result_artifacts(result_dir)
    started = time.monotonic()
    metadata: dict[str, object] = {
        "runner_contract_version": RUNNER_CONTRACT_VERSION,
        "task_id": task.task_id,
        "provider": provider,
        "model": model,
        "format": format_name,
        "status": "failed",
        "input_contract": {
            "render_count": 3,
            "techdraw_format": "dxf",
            "target_exposed": False,
        },
        "attempts": [],
    }
    redactions = credential_redactions(provider, source_home)

    with tempfile.TemporaryDirectory(prefix="drawing2cad-task-") as temp_name:
        temporary_root = Path(temp_name)
        workdir = temporary_root / secrets.token_hex(12)
        workdir.mkdir(parents=True)
        style_mapping = stage_task(task, workdir)
        metadata["render_style_mapping"] = style_mapping

        for attempt in range(max_repairs + 1):
            previous_python = {
                path.name: path.read_bytes()
                for path in numbered_model_candidates(workdir)
            }
            reasoning_path = workdir / REASONING_NAME
            previous_reasoning = (
                reasoning_path.read_bytes() if reasoning_path.is_file() else None
            )
            action_trace_path = workdir / ACTION_TRACE_NAME
            previous_action_trace = (
                action_trace_path.read_bytes() if action_trace_path.is_file() else None
            )
            prompt = build_prompt(format_name, task.render_labels, repair=attempt > 0)
            (result_dir / f"prompt-{attempt}.md").write_text(prompt)
            native_audit_dir = temporary_root / f"native-audit-{attempt:02d}"
            native_audit_dir.mkdir()
            audit_result_dir = result_dir / AUDIT_DIR_NAME
            audit_result_dir.mkdir(parents=True, exist_ok=True)
            filesystem_audit_path = (
                audit_result_dir / f"filesystem-attempt-{attempt}.jsonl"
            )
            with tempfile.TemporaryDirectory(prefix="drawing2cad-auth-") as home_name:
                disposable_home = Path(home_name)
                prepare_disposable_home(provider, disposable_home, source_home)
                command, stdin_text = build_agent_command(
                    provider,
                    model,
                    prompt,
                    executable,
                    workdir,
                    disposable_home,
                    python_env,
                    timeout_s,
                    native_audit_dir,
                )
                monitor = ArtifactAuditMonitor(
                    workdir,
                    filesystem_audit_path,
                    result_dir / "artifact_history" / f"agent_attempt_{attempt:02d}",
                    redactions,
                )
                monitor.start()
                try:
                    agent_result = run_agent(
                        command,
                        stdin_text,
                        timeout_s,
                        audit_result_dir / f"agent-stream-attempt-{attempt}.jsonl",
                        redactions,
                    )
                finally:
                    live_trace_issues, live_trace_warnings = monitor.stop()
                    # The disposable HOME is deleted when this block exits, so
                    # agy's trajectory store must be consolidated out of it now.
                    trajectory_databases = (
                        harvest_agy_trajectory(
                            disposable_home,
                            audit_result_dir / f"agy-trajectory-attempt-{attempt}",
                        )
                        if provider == "gemini"
                        else []
                    )
            native_log = native_audit_dir / "agy-cli.log"
            if native_log.is_file():
                (audit_result_dir / f"agy-cli-attempt-{attempt}.log").write_text(
                    redact_credentials(
                        native_log.read_text(errors="replace"), redactions
                    )
                )
            _write_agent_log(
                result_dir / f"agent-attempt-{attempt}.log",
                agent_result,
                provider,
                attempt,
            )
            attempt_meta: dict[str, object] = {
                "attempt": attempt,
                "agent_exit": agent_result.returncode,
                "agent_elapsed_s": round(agent_result.elapsed_s, 3),
                "agent_timed_out": agent_result.timed_out,
                "artifact_trace_violations": live_trace_issues,
            }
            # Extended in place below, so the metadata entry stays in sync
            # and is present even on an early break.
            compliance_warnings: list[str] = list(live_trace_warnings)
            attempt_meta["compliance_warnings"] = compliance_warnings
            metadata["attempts"].append(attempt_meta)  # type: ignore[union-attr]

            trace_audit = audit_provider_trace(
                provider,
                agent_result,
                native_log if provider == "gemini" else None,
                audit_result_dir / f"provider-events-attempt-{attempt}.jsonl",
                redactions,
                trajectory_databases,
            )
            attempt_meta["trace_audit"] = trace_audit
            credential_leaks = find_credential_leaks(workdir, redactions)
            attempt_meta["credential_leak_paths"] = credential_leaks
            if credential_leaks:
                attempt_meta["issues"] = [
                    "agent copied credential content into work artifacts: "
                    + ", ".join(credential_leaks)
                ]
                metadata["status"] = "credential-leak"
                break
            policy_violations = trace_audit["policy_violations"]
            attempt_meta["trace_policy_violations"] = policy_violations
            if policy_violations:
                attempt_meta["issues"] = [
                    "agent trace shows prohibited activity: "
                    + ", ".join(policy_violations)
                ]
                metadata["status"] = "policy-violation"
                break

            # A fault that kills the CLI mid-task usually leaves partial work
            # behind, so this is recorded rather than gated on there being no
            # candidate at all.  Judging half-written deliverables would score
            # the provider's outage as the model's reconstruction ability.
            infra_fault = (
                classify_agent_failure(agent_result)
                if agent_result.returncode != 0
                else None
            )
            if infra_fault is not None:
                attempt_meta["infra_fault"] = infra_fault

            candidates = numbered_model_candidates(workdir)
            if agent_result.returncode != 0 and not candidates:
                if infra_fault is not None:
                    attempt_meta["issues"] = [
                        f"agent CLI failed for an infrastructure reason ({infra_fault}); "
                        "rerun this task, its result says nothing about the model"
                    ]
                    metadata["status"] = "infra-error"
                    metadata["infra_fault"] = infra_fault
                else:
                    attempt_meta["issues"] = [
                        "agent CLI failed before producing a numbered Python "
                        "candidate; see agent log"
                    ]
                break

            trace_parse_errors = trace_audit["parse_errors"]
            if trace_parse_errors:
                attempt_meta["issues"] = [
                    "provider trace is incomplete or unparsable: "
                    + "; ".join(trace_parse_errors)
                ]
                metadata["status"] = "audit-failure"
                break

            cross_pass_issues, cross_pass_warnings = immutable_trace_issues(
                workdir,
                previous_python,
                previous_reasoning,
                previous_action_trace,
            )
            immutability_issues = list(live_trace_issues) + cross_pass_issues
            compliance_warnings += cross_pass_warnings
            if (workdir / MODEL_NAME).exists():
                immutability_issues.append(
                    f"agent must not create {MODEL_NAME}; select a numbered candidate"
                )
            if immutability_issues:
                attempt_meta["issues"] = immutability_issues
                metadata["status"] = "immutability-violation"
                break

            new_candidates = [
                path for path in candidates if path.name not in previous_python
            ]
            if not new_candidates:
                # The pass reused an existing candidate rather than superseding
                # it. Nothing was destroyed, so grade the artifacts as they are.
                compliance_warnings.append(
                    "this agent pass created no new model_attempt_NN.py"
                )

            selected_candidate, selection_issues = selected_model_candidate(workdir)
            code_issues = list(selection_issues)
            if selected_candidate is not None:
                code_issues += validate_generated_code(selected_candidate, format_name)
            aux_issues = validate_auxiliary_outputs(workdir)
            action_trace_issues, action_trace_warnings = validate_action_trace(
                workdir, filesystem_audit_path, sorted(previous_python)
            )
            compliance_warnings += action_trace_warnings
            issues = code_issues + aux_issues + action_trace_issues
            execution_ok = False
            verification = None
            if not code_issues and selected_candidate is not None:
                execution_ok, execution_log, verification = execute_and_verify(
                    workdir,
                    python_env,
                    execute_timeout_s,
                    selected_candidate.name,
                )
                (workdir / "execution.log").write_text(execution_log)
                if not execution_ok:
                    issues.append(
                        f"{selected_candidate.name} failed execution or STEP solid validation"
                    )

            attempt_meta["issues"] = issues
            attempt_meta["step_verification"] = verification
            attempt_meta["selected_candidate"] = (
                selected_candidate.name if selected_candidate else None
            )
            last_attempt = attempt == max_repairs
            reportable = issues or compliance_warnings
            if execution_ok and selected_candidate is not None:
                # A verified solid is the deliverable. Report-quality gaps are
                # worth another repair pass, but on the last pass they must not
                # throw away a reconstruction that executed and validated.
                if not reportable or last_attempt:
                    # Create the canonical deliverable once, from the selected
                    # candidate. The agent never writes or overwrites it.
                    shutil.copyfile(selected_candidate, workdir / MODEL_NAME)
                    metadata["status"] = (
                        "success" if not reportable else "success-with-warnings"
                    )
                    metadata["step_verification"] = verification
                    metadata["selected_candidate"] = selected_candidate.name
                    metadata["compliance_warnings"] = compliance_warnings
                    metadata["report_issues"] = issues
                    break

            feedback = (
                "The trusted runner rejected the current artifacts:\n- "
                + "\n- ".join(
                    [*issues, *compliance_warnings]
                    or ["agent did not complete successfully"]
                )
                + "\nRead execution.log when present, then fix the artifacts.\n"
            )
            (workdir / "runner_feedback.txt").write_text(feedback)

        _reclassify_infra_failure(metadata)
        _copy_artifacts(
            workdir,
            result_dir,
            success=metadata["status"] in SUCCESS_STATUSES,
            redactions=redactions,
        )
        metadata["intermediates"] = _copy_intermediate_artifacts(
            workdir, result_dir, redactions
        )

    metadata["elapsed_s"] = round(time.monotonic() - started, 3)
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _reclassify_infra_failure(metadata: dict[str, object]) -> None:
    """Re-label a plain failure whose last attempt died of an infrastructure fault.

    The agent usually has partial deliverables on disk when its CLI loses the
    model service, and judging those would score the outage as the model's
    reconstruction ability.  A policy violation or an audit failure is still
    reported as itself: those are findings about the run, not about the provider.
    """
    if metadata.get("status") != "failed":
        return
    attempts = metadata.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return
    infra_fault = attempts[-1].get("infra_fault")
    if not infra_fault:
        return
    metadata["status"] = "infra-error"
    metadata["infra_fault"] = infra_fault
    attempts[-1]["issues"] = [
        f"agent CLI failed for an infrastructure reason ({infra_fault}) after "
        "producing partial work; rerun this task, its result says nothing "
        "about the model",
        *attempts[-1].get("issues", []),
    ]


def _select_tasks(
    tasks: list[TaskInput], ids: Sequence[str], limit: int
) -> list[TaskInput]:
    if ids:
        requested = set(ids)
        known = {task.task_id for task in tasks}
        missing = sorted(requested - known)
        if missing:
            raise RunnerError("Unknown/incomplete task IDs: " + ", ".join(missing))
        tasks = [task for task in tasks if task.task_id in requested]
    if limit:
        tasks = tasks[:limit]
    return tasks


def _resolve_binary(provider: str, override: str | None) -> Path:
    value = override or shutil.which(PROVIDER_BIN_NAMES[provider])
    if not value:
        raise RunnerError(
            f"{PROVIDER_BIN_NAMES[provider]} is not installed or not on PATH."
        )
    return _safe_resolve(Path(value))


def _python_env_from_interpreter(interpreter: Path) -> Path:
    interpreter = _safe_resolve(interpreter)
    if interpreter.parent.name != "bin":
        raise RunnerError(
            f"--python must be an environment's bin/python executable: {interpreter}"
        )
    env_root = interpreter.parent.parent
    if not env_root.is_dir():
        raise RunnerError(f"Cannot resolve Python environment root for {interpreter}")
    return env_root


def _require_cad_library(python_env: Path, format_name: str) -> None:
    """Fail before any task if the bound environment cannot import the CAD library.

    The agent is told to use this library and the runner executes its script with
    the same interpreter, so a missing import wastes a full agent run per task.
    """
    interpreter = python_env / "bin" / "python"
    probe = subprocess.run(
        [str(interpreter), "-c", f"import {format_name}"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if probe.returncode != 0:
        detail = (probe.stderr.strip() or probe.stdout.strip()).splitlines()[-1:]
        raise RunnerError(
            f"--python environment {python_env} cannot import {format_name}"
            + (f": {detail[0]}" if detail else "")
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="challenge directory containing render_3d, target/target_step, techdraw",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODELS["gemini"],
        help="Gemini, GPT, or Claude model name; provider is inferred",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "gemini", "gpt", "claude"),
        default="auto",
        help="override provider inference from --model",
    )
    parser.add_argument(
        "--format", choices=("cadquery", "build123d"), default="cadquery"
    )
    parser.add_argument("--ids", nargs="*", default=[], metavar="ID")
    parser.add_argument("--limit", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--force", action="store_true", help="rerun successful IDs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--agent-bin", help="override agy/codex/claude executable")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter whose environment has cadquery and build123d",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="agent timeout seconds"
    )
    parser.add_argument(
        "--execute-timeout", type=int, default=180, help="CAD script timeout seconds"
    )
    parser.add_argument("--max-repairs", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0 or args.max_repairs < 0:
        raise RunnerError("--limit and --max-repairs must be non-negative")
    if args.timeout < 60 or args.execute_timeout < 1:
        raise RunnerError("--timeout must be >= 60 and --execute-timeout must be >= 1")
    if shutil.which("bwrap") is None:
        raise RunnerError(
            "bubblewrap (`bwrap`) is required; refusing an unsandboxed run"
        )
    if shutil.which("inotifywait") is None:
        raise RunnerError(
            "inotifywait is required for immutable artifact auditing; "
            "install inotify-tools"
        )

    provider = infer_provider(args.model, args.provider)
    if provider in {"gpt", "claude"}:
        # Reject a mistyped effort label before launching any task rather than
        # failing every task with an opaque CLI model/effort rejection error.
        # Gemini is excluded: it carries its variant verbatim inside the label.
        parse_model_effort(args.model)
    executable = _resolve_binary(provider, args.agent_bin)
    python_env = _python_env_from_interpreter(args.python)
    _require_cad_library(python_env, args.format)
    tasks = _select_tasks(discover_tasks(args.data_dir), args.ids, args.limit)
    if not tasks:
        raise RunnerError("No tasks selected")

    out_dir = args.out_dir.expanduser().resolve()
    if out_dir == Path("/") or out_dir == Path.home().resolve():
        raise RunnerError(f"Refusing unsafe --out_dir: {out_dir}")
    print(
        f"provider={provider} model={args.model!r} format={args.format} "
        f"tasks={len(tasks)}"
    )
    print("isolation=opaque-per-task-home,minimal-fs,no-target,networkless-cad-exec")

    if args.dry_run:
        for task in tasks:
            print(
                f"[dry-run] {task.task_id}: 3 renders "
                f"({', '.join(task.render_labels)}) + DXF; agent={executable}"
            )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    source_home = Path.home().resolve()
    successes = failures = skipped = infra_errors = 0
    summary_rows: list[dict[str, object]] = []
    for index, task in enumerate(tasks, 1):
        result_dir = out_dir / task.task_id
        metadata_path = result_dir / "metadata.json"
        if (
            not args.force
            and metadata_path.is_file()
            and (result_dir / STEP_NAME).is_file()
        ):
            try:
                previous = json.loads(metadata_path.read_text())
            except ValueError:
                previous = {}
            if (
                previous.get("status") in SUCCESS_STATUSES
                and previous.get("runner_contract_version") == RUNNER_CONTRACT_VERSION
            ):
                print(
                    f"[{index}/{len(tasks)}] {task.task_id}: skip (already successful)"
                )
                skipped += 1
                continue
            if previous.get("status") in SUCCESS_STATUSES:
                raise RunnerError(
                    f"Existing successful result {result_dir} predates runner contract "
                    f"{RUNNER_CONTRACT_VERSION}; use a new --out_dir or pass --force "
                    "to replace it explicitly"
                )
        print(f"[{index}/{len(tasks)}] {task.task_id}: running")
        metadata = run_task(
            task,
            provider=provider,
            model=args.model,
            format_name=args.format,
            out_dir=out_dir,
            executable=executable,
            python_env=python_env,
            source_home=source_home,
            timeout_s=args.timeout,
            execute_timeout_s=args.execute_timeout,
            max_repairs=args.max_repairs,
        )
        status = str(metadata["status"])
        warnings_reported = metadata.get("compliance_warnings")
        warnings_reported = (
            list(warnings_reported) if isinstance(warnings_reported, list) else []
        )
        summary_rows.append(
            {
                "task_id": task.task_id,
                "status": status,
                "elapsed_s": metadata["elapsed_s"],
                "compliance_warnings": warnings_reported,
            }
        )
        if status in SUCCESS_STATUSES:
            note = ""
            if status == "success-with-warnings":
                note = f" (with {len(warnings_reported)} compliance warning(s))"
            print(f"[{index}/{len(tasks)}] {task.task_id}: {status}{note}")
            successes += 1
        elif status == "infra-error":
            print(
                f"[{index}/{len(tasks)}] {task.task_id}: infra-error "
                f"({metadata.get('infra_fault')}); rerun this ID",
                file=sys.stderr,
            )
            infra_errors += 1
        else:
            print(
                f"[{index}/{len(tasks)}] {task.task_id}: failed; "
                f"see {result_dir / 'metadata.json'}",
                file=sys.stderr,
            )
            failures += 1

    infra_ids = [
        str(row["task_id"]) for row in summary_rows if row["status"] == "infra-error"
    ]
    summary = {
        "provider": provider,
        "model": args.model,
        "format": args.format,
        "selected": len(tasks),
        "success": successes,
        "failed": failures,
        "infra_error": infra_errors,
        "infra_error_ids": infra_ids,
        "skipped": skipped,
        "results": summary_rows,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"done: success={successes} failed={failures} "
        f"infra_error={infra_errors} skipped={skipped}"
    )
    if infra_ids:
        print(
            "rerun the infrastructure failures with: --ids " + " ".join(infra_ids),
            file=sys.stderr,
        )
    return 1 if failures or infra_errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
