# Zero-shot Drawing2CAD

`run_zero_shot.py` is a zero-shot runner that reconstructs 3D CAD directly from a 2D technical drawing and three 3D renders. It invokes logged-in command-line agents directly, without using model-provider SDKs or APIs.

- Gemini: `agy`
- GPT: `codex`
- Claude: `claude`

All three providers are supported by one script. The provider is inferred from `--model`; use `--provider` only when an explicit override is necessary.

Gemini model labels are passed to agy exactly as written. In particular, labels such as `Gemini 3.6 Flash (High)` already encode their variant and are not combined with a separate `--effort`, which agy rejects for that model. GPT reasoning-effort shorthand is handled separately through suffixes such as `-max`; Claude model names do not affect an effort option.

## Prerequisites

- Linux with `bubblewrap` (`bwrap`) and `inotify-tools` (`inotifywait`)
- A prior login for each agent CLI you intend to use: `agy`, `codex`, or `claude`
- A Python environment that can import `cadquery` and `build123d`
- The dataset placed under `data/eccv2026-cad-challenge-data`

CLI authentication uses an existing subscription/OAuth login. The runner does not use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or provider SDKs.

The expected input layout is:

```text
data/eccv2026-cad-challenge-data/my_codes/test_vlm/
├── render_3d/
│   ├── hlg_perspective/<id>.png
│   ├── hlg_translucent_faces_perspective/<id>.png
│   └── transparent_shaded_edges_perspective/<id>.png
├── techdraw/
│   └── dxf/<id>.dxf
└── target/ or target_step/        # Checked for existence only; contents are never read
```

`render_3d` must contain exactly three directories. Every ID must have all three PNGs and one DXF; inconsistent input sets and symlinked inputs are rejected rather than silently skipped. PDFs, SVGs, CSVs, and target STEP files are not provided as inputs.

## Usage

The following examples are run from the repository root.

```bash
# Gemini (default)
python zero-shot/run_zero_shot.py \
  --data_dir data/eccv2026-cad-challenge-data/my_codes/test_vlm \
  --out_dir outputs/eccv2026-cad-challenge-data/zero_shot_gemini_cadquery \
  --model "Gemini 3.6 Flash (High)" \
  --format cadquery

# GPT. The -max suffix is translated to Codex reasoning effort=max.
python zero-shot/run_zero_shot.py \
  --data_dir data/eccv2026-cad-challenge-data/my_codes/test_vlm \
  --out_dir outputs/eccv2026-cad-challenge-data/zero_shot_gpt_cadquery \
  --model gpt-5.6-sol-max \
  --format cadquery

# Claude
python zero-shot/run_zero_shot.py \
  --data_dir data/eccv2026-cad-challenge-data/my_codes/test_vlm \
  --out_dir outputs/eccv2026-cad-challenge-data/zero_shot_claude_cadquery \
  --model claude-opus-4-8 \
  --format cadquery
```

Use `--format build123d` to generate build123d code instead.

To run a single-case smoke test:

```bash
python zero-shot/run_zero_shot.py \
  --data_dir data/eccv2026-cad-challenge-data/my_codes/test_vlm \
  --out_dir outputs/eccv2026-cad-challenge-data/zero_shot_gemini_cadquery_smoke \
  --model "Gemini 3.6 Flash (High)" \
  --format cadquery \
  --ids 000364
```

Useful options:

- `--ids 000364 000405`: run only the specified IDs
- `--limit 1`: run only the first N tasks after sorting
- `--dry-run`: check input matching and CLI selection without reading credentials, creating output, or invoking a model
- `--force`: rerun tasks that have already succeeded
- `--max-repairs 2`: set the number of repair attempts after a trusted-runner failure
- `--timeout 1800`: set the time limit in seconds for each agent invocation
- `--python /path/to/env/bin/python`: explicitly select the Python environment containing the CAD libraries

A task is skipped automatically only when it already has a successful `metadata.json` with the current hardened runner contract version and an `output.step`. If a successful directory predates this contract, the runner asks for a new `--out_dir` or an explicit `--force`; it does not silently accept or overwrite the old research artifacts.

## Output

```text
<out_dir>/
├── run_summary.json
└── <id>/
    ├── model.py                    # Canonical copy of the selected, validated candidate
    ├── output.step                # Result of running model.py in isolation
    ├── final_selection.json        # Immutable candidate selected by the agent
    ├── reasoning.md                # Append-only, chronological reasoning trace
    ├── action_trace.jsonl          # Agent-reported action/decision ledger
    ├── source_usage.json           # How each of the three renders and the DXF were used
    ├── execution.log               # Trusted execution and STEP-validation log
    ├── agent-attempt-*.log         # Redacted raw CLI output
    ├── prompt-*.md
    ├── audit/                      # Parent-owned stream, filesystem, and tool events
    ├── artifact_history/           # Immutable revisions observed with inotify
    ├── intermediates/              # Crops, renders, SVGs, immutable candidates, etc.
    │   └── manifest.json
    ├── quarantine/                 # Untrusted artifacts; failure cases only
    └── metadata.json
```

### Python candidates and log retention

The agent writes `model_attempt_01.py` before performing its first validation. If it wants to make a change, it creates a new file such as `model_attempt_02.py`; it must never overwrite or delete an existing Python candidate. The agent identifies its final candidate in `final_selection.json`, and the trusted runner copies it to `model.py` exactly once, after validation succeeds.

The runner watches the work directory with Linux inotify while the agent is working. Every create, close-write, move, and delete event is timestamped under `audit/`; tracked report files and Python candidates are snapshotted under `artifact_history/` **before** the agent can replace them.

### Integrity violations versus compliance warnings

The monitor is the evidence recorder, not the judge. Because every revision is captured, an in-pass rewrite destroys nothing the auditor can see, so verdicts are reconciled once against the final work directory and split into two levels:

- **Integrity violations fail the task.** Evidence the runner can no longer reconstruct (a tracked candidate or report file deleted or renamed away), credential content in an artifact, an artifact over the audit snapshot limit, a write outside the work directory, a prohibited action in the provider trace, or a Python candidate that an *earlier, already-graded* pass produced being changed. That last rule is the output-overwrite guarantee: what the runner has executed and reported on can never change underneath it.
- **Compliance warnings are recorded and do not fail the task.** The agent rewrote a candidate in place instead of writing the next number, edited `reasoning.md`/`action_trace.jsonl` rather than strictly appending, wrote no new candidate in a repair pass, or opened the action trace after its first candidate. Every one of these keeps the full revision chain in `artifact_history/`, so the deviation is auditable rather than destructive. They appear in `compliance_warnings` in `metadata.json` and `run_summary.json`, and a task that produces a verified STEP solid despite them ends as `success-with-warnings`.

This split exists because a strict byte-prefix rule produces false positives against real agents rather than catching cheating. A create-then-fill write leaves a transient empty file; a buffered writer flushes a growing prefix; an atomic write replaces a file wholesale; a whole-file write tool reflows a line it is also appending after. None of these lose evidence. Candidate writes are therefore compared only when the new bytes discard content the file already held, and append-only artifacts are graded by whether every line the runner ever observed is still present, in order, in the final file.

Only `reasoning.md` and `action_trace.jsonl` may be appended to in place, during a pass and across repair passes. The action trace should start before the first Python candidate this pass writes (candidates carried over from an earlier pass are excluded, since they already exist when the watcher starts), use strictly increasing sequence numbers, and separately record inspection of all four inputs; a trace that is short, sparsely numbered, late, or missing an input name is warned about rather than discarded. A trace that cannot be read as an ordered log at all — unparsable lines, non-increasing sequences, empty required fields — is still an issue worth a repair pass. It records observable evidence, actions, decisions, and outcomes; it does not claim to expose unavailable private chain-of-thought. Local execution/render/validation is deliberately not mandatory: if the model chooses to validate on its own, those actions and outcomes must be appended and can be studied.

On a repair pass, `artifact_history/agent_attempt_NN/` opens with a `revision_000` snapshot of each candidate carried over from the previous pass. Those bytes are identical to the previous attempt's final revision by design — it is the baseline the pass started from, not a new write. `carried_over.json` in the same directory lists which files this applies to.

Provider stdout/stderr is captured by the parent process as timestamped JSONL outside `/work`, so the agent cannot rewrite it. GPT and Claude structured JSONL is parsed into normalized tool events and an unparsable successful trace fails closed.

Gemini is audited from `agy`'s own trajectory store. `agy` writes every tool call — the tool name, its arguments, the `BypassSandbox` flag, and the permission decision — into a per-conversation SQLite database under `$HOME/.gemini/antigravity-cli/conversations/`. That HOME is the disposable one the runner creates and deletes, so the agent cannot reach or rewrite it. The store is consolidated out with `Connection.backup` (the live database keeps recent steps in a write-ahead log) into `audit/agy-trajectory-attempt-N/` before the HOME is destroyed, and the decoded actions land in `audit/provider-events-attempt-N.jsonl`. A successful run that leaves no trajectory store fails closed, and the older native-log scrape remains as a fallback for `agy` builds that do not write one.

Only command- and target-shaped arguments are read out of a call (`CommandLine`, `Cwd`, `AbsolutePath`, `Query`, `SearchPath`, `Url`, and similar). File *contents* are deliberately excluded, so reasoning prose written through a write tool is never policy-scanned as though it were an action.

**Policy is enforced on what actually ran.** A call the provider's permission layer refused never happened and cannot have leaked anything, so refused calls are reported separately as `blocked_policy_attempts` and `blocked_event_count` in `trace_audit` rather than failing an otherwise sound reconstruction. Those fields are the positive evidence that the sandbox held. A matching call that was *not* refused is a `policy_violation` and fails the task.

Credential string values are redacted from logs and any generated artifact containing them is rejected and never persisted.

Input-derived crops, previews, validation renders, helper scripts, and other files created in the work directory are copied to `intermediates/`. The original three PNGs and DXF are not duplicated. Symlinks and device files are rejected. The default limits are 64 MiB per file and 256 MiB in total. Intermediate artifacts are never provided as inputs to another task.

## Leakage prevention and sandboxing

Each task runs in a random temporary directory whose name does not contain the task ID. The only task data visible to the agent is:

```text
/work/render_1.png
/work/render_2.png
/work/render_3.png
/work/drawing.dxf
```

The main safeguards are:

1. The agent filesystem is a minimally mounted `bwrap` jail. The repository, dataset, targets, grader, other tasks, normal `/home`, and `/root` are not mounted.
2. Persistent agent directories are not bind-mounted. Only the files required for authentication are copied with mode 0600 into a disposable home directory, which is deleted after the run. It contains no history, conversations, memory, cache, shell snapshots, plugins, or previous models.
3. Codex runs with `--ephemeral --ignore-user-config --ignore-rules`, workspace network access disabled, and browser, computer-use, apps, plugins, dependency installation, and external skill discovery features explicitly disabled.
4. Claude runs in safe mode and `dontAsk` permission mode without session persistence, Chrome, slash commands, MCP servers, WebSearch, or WebFetch. Its built-in Bash sandbox uses `allowedDomains=[]`, denies common network commands, and prohibits unsandboxed commands.
5. Gemini runs with agy's `--sandbox --mode accept-edits`; dangerous permission auto-approval is not used. Its disposable `settings.json` selects `toolPermission=proceed-in-sandbox`, permits file tools only under `/work`, and explicitly denies `/home` file access, URL/browser actions, and MCP. Commands are auto-approved only inside agy's Linux `nsjail` terminal sandbox, allowing voluntary local validation without granting a sandbox escape. `/cad-env` is added read-only so CAD execution does not require an escape. The selected Gemini model label is passed through unchanged.

   The allow list names command prefixes individually (`command(python3)`, `command(/cad-env/bin/python)`, …). agy matches a command against `command(<prefix>)`, so `command(*)` alone is not a dependable wildcard. The deny list deliberately omits `unsandboxed(*)`: that rule matches **every** command rather than only sandbox-bypass requests, and with it present the agent had no shell at all — measured across two runs, all 8 of 8 `run_command` calls were refused, including `echo hello`. A sandbox bypass is caught by the trajectory audit instead, which fails the task when a `BypassSandbox: true` call actually executes.

   Granting the shell does not open a network path. Inside agy's terminal sandbox, egress was measured as unreachable at the kernel level: `Errno 101` connecting to raw addresses `1.1.1.1:443`, `8.8.8.8:53`, and `142.250.196.132:443`, and DNS resolution fails outright. The agent CLI's own process does retain network, because it must reach the model service.
6. Provider tool events are checked for sandbox-bypass requests, Web/browser tools, external URLs, network/package commands, remote Git/GitHub access, hashing, and credential-store access. Ordinary path or filesystem references are deliberately not flagged: inside the jail `/home` is the empty per-task disposable HOME, and `/root`, the repository, the dataset and other tasks are not mounted at all, so environment probing such as `find / -name cadquery` cannot reach an answer and must not fail a task. The mounts are the control; these patterns cover only what the mounts cannot express. A violation in an action that actually executed marks the task `policy-violation`; an action the provider refused is recorded as a blocked attempt instead.
7. Generated Python undergoes AST inspection that rejects networking, subprocesses, filesystem exploration, hashing, dynamic execution, dunder/introspection access, the non-selected CAD library, and direct use of other CAD kernels.
8. Generated Python runs in a fresh second `bwrap --unshare-all` jail containing only the selected script, the read-only CAD Python environment, and an empty disposable work directory. The source images, DXF, reports, and agent work files are absent. Only a successfully verified STEP is copied back.
9. The resulting STEP file is imported again and must contain exactly one solid that OpenCASCADE reports as valid.

Canonical `model.py` and `output.step` are copied only when all policy, integrity, code, execution, and STEP checks pass; the status is then `success`, or `success-with-warnings` if compliance warnings or report-quality issues remain after the last repair pass. On failure, agent-created deliverables are stored only under `quarantine/`; a failed task never exposes an untrusted root-level `model.py` or `output.step`.

An agent CLI that dies before writing any candidate is classified: a provider quota wall, an auth failure, a service outage, a lost transport to the model service, or a wall-clock timeout is reported as `infra-error` with an `infra_fault` reason, and `run_summary.json` lists the affected IDs for a rerun. These say nothing about the model's CAD ability and must not be scored as reconstruction failures. The transport patterns match only the CLI's own wording, so a model that merely writes about a timeout is still graded as a model failure.

The environment behind `--python` (default: the interpreter running the runner) is bound read-only at `/cad-env` and placed first on the jail's `PATH`. The agent CLI, however, runs its shell in its own nested sandbox that resets `PATH`, so a bare `python3` there is the system interpreter and has no CAD libraries. Measured on a real run, this cost about twenty tool calls of environment hunting. Two things address it: `PYTHONPATH` is exported to the bound environment's `site-packages`, which the nested shell inherits, and the prompt names `/cad-env/bin/python` explicitly and tells the agent not to go looking for another interpreter. Before the first task the runner imports the requested library in that environment and refuses to start if it is missing, since otherwise every task burns a full agent run before failing.

The target directory is checked only as a layout guard. Its file listing, contents, and hashes are never read, and no output-to-target comparison is performed. This is intentional: the runner records whether the model voluntarily performs its own local validation instead of forcing a reference-based validation step.

## Verification status

- The hardened command construction, audit parsers, inotify history, quarantine behavior, source/format validation, and sandbox construction are covered by local unit tests.
- Live Gemini runs confirm the trajectory audit (45 to 67 tool events per task, `complete: true`), the relaxed immutability judgment, and that the agent now performs its own local validation: it parses the DXF with `ezdxf`, iterates on CadQuery geometry, executes `model_attempt_01.py`, and re-imports `output.step` to check the solid before selecting it.
- Sandbox egress was verified empirically rather than assumed, both for the agent's terminal sandbox and for the CAD execution jail.
- No GPT or Claude model inference was run while validating these changes, per user instruction. The GPT and Claude sandbox settings were not re-measured, so the `agy`-specific findings above do not transfer to them.

Run the local tests with:

```bash
cd zero-shot
python -m unittest -v
ruff check run_zero_shot.py test_run_zero_shot.py
```
