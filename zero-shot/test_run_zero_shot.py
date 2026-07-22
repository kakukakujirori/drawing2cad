from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

import run_zero_shot as runner


class DataDiscoveryTests(unittest.TestCase):
    def test_discovers_complete_four_input_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            for style in ("a", "b", "c"):
                directory = root / "render_3d" / style
                directory.mkdir(parents=True)
                (directory / "000001.png").write_bytes(b"png")
            dxf = root / "techdraw" / "dxf"
            dxf.mkdir(parents=True)
            (dxf / "000001.dxf").write_text("dxf")
            # Its contents are irrelevant and must never be needed.
            (root / "target").mkdir()
            (root / "target" / "secret.step").write_text("answer")

            tasks = runner.discover_tasks(root)

            self.assertEqual([task.task_id for task in tasks], ["000001"])
            self.assertEqual(tasks[0].render_labels, ("a", "b", "c"))

    def test_rejects_inconsistent_input_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            for style in ("a", "b", "c"):
                directory = root / "render_3d" / style
                directory.mkdir(parents=True)
                (directory / "000001.png").write_bytes(b"png")
            (root / "render_3d" / "a" / "incomplete.png").write_bytes(b"png")
            dxf = root / "techdraw" / "dxf"
            dxf.mkdir(parents=True)
            (dxf / "000001.dxf").write_text("dxf")
            (root / "target").mkdir()

            with self.assertRaisesRegex(runner.RunnerError, "inconsistent"):
                runner.discover_tasks(root)


class RoutingTests(unittest.TestCase):
    def test_provider_inference(self) -> None:
        self.assertEqual(runner.infer_provider("Gemini 3.6 Flash (High)"), "gemini")
        self.assertEqual(runner.infer_provider("gpt-5.6-sol-max"), "gpt")
        self.assertEqual(runner.infer_provider("claude-opus-4-8"), "claude")

    def test_gpt_effort_shorthand(self) -> None:
        self.assertEqual(
            runner.split_gpt_model_effort("gpt-5.6-sol-max"),
            ("gpt-5.6-sol", "max"),
        )
        self.assertEqual(
            runner.split_gpt_model_effort("gpt-5.6-sol"),
            ("gpt-5.6-sol", None),
        )


class PromptAndValidationTests(unittest.TestCase):
    def test_prompt_names_every_input_and_forbids_leak_sources(self) -> None:
        prompt = runner.build_prompt("cadquery", ("opaque", "xray", "edges"))
        for name in (*runner.RENDER_INPUT_NAMES, runner.DXF_INPUT_NAME):
            self.assertIn(name, prompt)
        for label in ("opaque", "xray", "edges"):
            self.assertIn(label, prompt)
        self.assertIn("GitHub", prompt)
        self.assertIn("all three PNG", prompt)
        self.assertIn("Before every later validation", prompt)
        self.assertIn("model_attempt_01.py", prompt)
        self.assertIn("as much of your reasoning process", prompt)
        self.assertIn("alternative interpretations considered", prompt)
        self.assertIn(runner.ACTION_TRACE_NAME, prompt)
        # The agent's shell resolves `python3` to the system interpreter, so the
        # prompt must name the environment that actually has the CAD libraries.
        self.assertIn(runner.CAD_ENV_PYTHON, prompt)
        self.assertIn("as work happens", prompt)
        self.assertNotIn("Do not reveal", prompt)

    def test_code_validation_accepts_cadquery_and_rejects_search_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            model = Path(temp_name) / runner.MODEL_NAME
            model.write_text(
                "import cadquery as cq\n"
                "result = cq.Workplane('XY').box(1, 2, 3)\n"
                "cq.exporters.export(result, 'output.step')\n"
            )
            self.assertEqual(runner.validate_generated_code(model, "cadquery"), [])

            model.write_text(
                "import cadquery as cq\nimport os, hashlib\n"
                "result = cq.Workplane('XY').box(1, 2, 3)\n"
                "open('/home/user/model.py').read()\n"
                "cq.exporters.export(result, 'output.step')\n"
            )
            issues = "\n".join(runner.validate_generated_code(model, "cadquery"))
            self.assertIn("denied imports", issues)
            self.assertIn("denied call", issues)
            self.assertIn("suspicious path", issues)

    def test_code_validation_accepts_build123d(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            model = Path(temp_name) / "model_attempt_01.py"
            model.write_text(
                "from build123d import Box, export_step\n"
                "result = Box(1, 2, 3)\n"
                "export_step(result, 'output.step')\n"
            )
            self.assertEqual(runner.validate_generated_code(model, "build123d"), [])

    def test_code_validation_rejects_alternative_cad_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            model = Path(temp_name) / "model_attempt_01.py"
            model.write_text(
                "import cadquery as cq\n"
                "from build123d import Box\n"
                "result = cq.Workplane('XY').box(1, 2, 3)\n"
                "cq.exporters.export(result, 'output.step')\n"
            )
            issues = runner.validate_generated_code(model, "cadquery")
            self.assertTrue(
                any("must not import build123d" in issue for issue in issues)
            )

    def test_auxiliary_report_requires_all_three_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            work = Path(temp_name)
            (work / runner.REASONING_NAME).write_text("engineering summary " * 5)
            (work / runner.SOURCE_USAGE_NAME).write_text(
                json.dumps(
                    {
                        "render_images_used": [
                            {
                                "file": name,
                                "observations": f"Concrete distinct evidence from {name}",
                            }
                            for name in runner.RENDER_INPUT_NAMES
                        ],
                        "techdraw_used": {
                            "file": runner.DXF_INPUT_NAME,
                            "observations": "Measured exact dimensions from DXF entities",
                        },
                    }
                )
            )
            self.assertEqual(runner.validate_auxiliary_outputs(work), [])

    def test_action_trace_requires_inputs_and_precedes_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            work = Path(temp_name) / "work"
            work.mkdir()
            records = [
                {
                    "sequence": index,
                    "stage": "inspection",
                    "action": "inspect",
                    "result": f"observed {name}",
                }
                for index, name in enumerate(
                    (*runner.RENDER_INPUT_NAMES, runner.DXF_INPUT_NAME), 1
                )
            ]
            records += [
                {
                    "sequence": 5,
                    "stage": "modeling",
                    "action": "write candidate",
                    "result": "wrote model_attempt_01.py",
                },
                {
                    "sequence": 6,
                    "stage": "selection",
                    "action": "select candidate",
                    "result": "selected model_attempt_01.py",
                },
            ]
            (work / runner.ACTION_TRACE_NAME).write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            audit = Path(temp_name) / "filesystem.jsonl"
            audit.write_text(
                json.dumps({"sequence": 1, "path": runner.ACTION_TRACE_NAME})
                + "\n"
                + json.dumps({"sequence": 2, "path": "model_attempt_01.py"})
                + "\n"
            )
            self.assertEqual(runner.validate_action_trace(work, audit), ([], []))

    def test_action_trace_tolerates_sparse_but_ordered_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            work = Path(temp_name) / "work"
            work.mkdir()
            records = [
                {
                    "sequence": 10 * index,
                    "stage": "inspection",
                    "action": "inspect",
                    "result": f"observed {name}",
                }
                for index, name in enumerate(
                    (*runner.RENDER_INPUT_NAMES, runner.DXF_INPUT_NAME), 1
                )
            ]
            (work / runner.ACTION_TRACE_NAME).write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            audit = Path(temp_name) / "filesystem.jsonl"
            audit.write_text(
                json.dumps({"sequence": 1, "path": runner.ACTION_TRACE_NAME}) + "\n"
            )
            issues, warnings = runner.validate_action_trace(work, audit)
            # Non-dense sequence numbers and a short log are reported, not fatal.
            self.assertEqual(issues, [])
            self.assertTrue(any("only 4 records" in warning for warning in warnings))

    def test_action_trace_chronology_ignores_carried_over_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            audit = Path(temp_name) / "filesystem.jsonl"
            audit.write_text(
                json.dumps({"sequence": 1, "path": "model_attempt_01.py"})
                + "\n"
                + json.dumps({"sequence": 2, "path": runner.ACTION_TRACE_NAME})
                + "\n"
                + json.dumps({"sequence": 3, "path": "model_attempt_02.py"})
                + "\n"
            )
            self.assertEqual(
                runner._action_trace_chronology_warnings(
                    audit, ["model_attempt_01.py"]
                ),
                [],
            )
            self.assertTrue(runner._action_trace_chronology_warnings(audit, []))

    def test_action_trace_rejects_unordered_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            work = Path(temp_name) / "work"
            work.mkdir()
            (work / runner.ACTION_TRACE_NAME).write_text(
                json.dumps({"sequence": 5, "stage": "a", "action": "b", "result": "c"})
                + "\n"
                + json.dumps(
                    {"sequence": 2, "stage": "a", "action": "b", "result": "c"}
                )
                + "\n"
            )
            audit = Path(temp_name) / "filesystem.jsonl"
            audit.write_text("")
            issues, _ = runner.validate_action_trace(work, audit)
            self.assertTrue(any("strictly increase" in issue for issue in issues))

    def test_selects_numbered_candidate_and_enforces_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            work = Path(temp_name)
            first = work / "model_attempt_01.py"
            first.write_text("import cadquery as cq\nresult = None\n")
            (work / runner.FINAL_SELECTION_NAME).write_text(
                json.dumps({"model_file": first.name})
            )
            selected, issues = runner.selected_model_candidate(work)
            self.assertEqual(selected, first)
            self.assertEqual(issues, [])

            previous = {first.name: first.read_bytes()}
            reasoning = work / runner.REASONING_NAME
            reasoning.write_text("attempt one\n")
            old_reasoning = reasoning.read_bytes()
            first.write_text("overwritten\n")
            reasoning.write_text("replacement\n")
            violations, warnings = runner.immutable_trace_issues(
                work, previous, old_reasoning
            )
            # Rewriting an already-graded candidate stays fatal.
            self.assertTrue(any("overwritten" in issue for issue in violations))
            # Losing rationale between passes is recoverable from artifact_history.
            self.assertTrue(any("lost earlier content" in item for item in warnings))

    def test_cross_pass_reasoning_append_tolerates_reformatting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            work = Path(temp_name)
            reasoning = work / runner.REASONING_NAME
            reasoning.write_text("## Stage 1\n- measured 40 mm\n")
            old = reasoning.read_bytes()
            reasoning.write_text(
                "## Stage 1\n\n- measured 40 mm\n\n## Stage 2\n- extruded\n"
            )
            violations, warnings = runner.immutable_trace_issues(work, {}, old)
            self.assertEqual(violations, [])
            self.assertEqual(warnings, [])

    def test_trace_policy_detector(self) -> None:
        self.assertEqual(runner.detect_trace_policy_violations("normal", ""), [])
        labels = runner.detect_trace_policy_violations(
            "tool: WebSearch; command: md5sum drawing.dxf", ""
        )
        self.assertIn("web/browser tool", labels)
        self.assertIn("file hashing", labels)

    def test_filesystem_detector_targets_secrets_not_environment_probing(
        self,
    ) -> None:
        for benign in (
            # All observed in real runs while locating the CAD interpreter.
            # Nothing reachable from the jail's root holds an answer, so these
            # must not fail a task.
            "which python3; find / -name cadquery 2>/dev/null",
            'conda env list || find /home /opt -name "python*" 2>/dev/null',
            "ls -la /home/runner",
            "ls -la /work",
            "grep -n ENTITIES /work/drawing.dxf",
            "/cad-env/bin/python3 model_attempt_01.py",
        ):
            with self.subTest(command=benign):
                self.assertEqual(runner.detect_trace_policy_violations(benign, ""), [])
        for hunting in (
            "cat /home/runner/.gemini/antigravity-cli/settings.json",
            "grep -r token ~/.ssh",
            "cat ~/.aws/credentials",
        ):
            with self.subTest(command=hunting):
                self.assertIn(
                    "credential store access",
                    runner.detect_trace_policy_violations(hunting, ""),
                )


def _write_agy_trajectory(home: Path, steps: list[tuple[bytes, bytes]]) -> Path:
    """Build a store shaped like agy's: protobuf blobs with embedded tool JSON."""
    directory = home / runner.AGY_CONVERSATION_DIR
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "926618e9-dbb1-4d3b-b497-9e3f2b144c63.db"
    connection = sqlite3.connect(database)
    connection.execute(
        f"CREATE TABLE {runner.AGY_TRAJECTORY_TABLE} "
        "(idx INTEGER, step_type INTEGER, status INTEGER, metadata BLOB, "
        "error_details BLOB, step_payload BLOB)"
    )
    for index, (metadata, error_details) in enumerate(steps):
        connection.execute(
            f"INSERT INTO {runner.AGY_TRAJECTORY_TABLE} VALUES (?,?,?,?,?,?)",
            (index, 21, 7, metadata, error_details, metadata),
        )
    connection.commit()
    connection.close()
    return database


def _agy_step(tool: str, arguments: dict) -> bytes:
    """Frame a tool call the way agy does: \\x12<len>name\\x1a<len>{json}."""
    name = tool.encode()
    payload = json.dumps(arguments, separators=(",", ":")).encode()
    return (
        b"\x0a\x0c\x08\x88\xb2\x86\xd3\x06\x12"
        + bytes([len(name)])
        + name
        + b"\x1a"
        + bytes([len(payload) & 0x7F, len(payload) >> 7])
        + payload
        + b"\x3a\xea\x14\x12\xe7\xff\x00\x4d\x32"
    )


class AgyTrajectoryAuditTests(unittest.TestCase):
    def test_harvest_extracts_commands_and_scopes_policy_to_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            home = root / "home"
            _write_agy_trajectory(
                home,
                [
                    (
                        _agy_step(
                            "run_command",
                            {
                                "BypassSandbox": False,
                                "CommandLine": "python3 parse_dxf.py",
                                "Cwd": "/work",
                                "toolSummary": "Parsing the drawing",
                            },
                        ),
                        b"",
                    ),
                    (
                        # File contents must never reach the policy scanner:
                        # this prose would otherwise read as a web-tool event.
                        _agy_step(
                            "write_file",
                            {
                                "AbsolutePath": "/work/reasoning.md",
                                "CodeContent": (
                                    "I considered a browser and web_search but "
                                    "https://example.com is off limits."
                                ),
                            },
                        ),
                        b"",
                    ),
                ],
            )
            harvested = runner.harvest_agy_trajectory(home, root / "trajectory")
            self.assertEqual(len(harvested), 1)

            events, errors = runner.agy_trajectory_events(harvested)
            self.assertEqual(errors, [])
            joined = "\n".join(events)
            self.assertIn("python3 parse_dxf.py", joined)
            self.assertIn("/work/reasoning.md", joined)
            self.assertNotIn("example.com", joined)
            self.assertEqual(runner.detect_trace_policy_violations(joined, ""), [])

    def test_executed_network_access_is_a_policy_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            home = root / "home"
            _write_agy_trajectory(
                home,
                [
                    (
                        _agy_step(
                            "run_command",
                            {
                                "BypassSandbox": True,
                                "CommandLine": "curl -sS https://example.com/answer.step",
                                "Cwd": "/work",
                            },
                        ),
                        b"",
                    )
                ],
            )
            harvested = runner.harvest_agy_trajectory(home, root / "trajectory")
            audit = runner.audit_provider_trace(
                "gemini",
                runner.AttemptResult(0, "", "", 1.0),
                None,
                root / "events.jsonl",
                (),
                harvested,
            )
            self.assertIn("sandbox bypass request", audit["policy_violations"])
            self.assertIn("external URL access", audit["policy_violations"])
            self.assertIn("network command", audit["policy_violations"])

    def test_refused_attempt_is_evidence_not_a_task_failure(self) -> None:
        """A command the permission layer refused never ran, so nothing leaked."""
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            home = root / "home"
            _write_agy_trajectory(
                home,
                [
                    (
                        _agy_step(
                            "run_command",
                            {
                                "BypassSandbox": True,
                                "CommandLine": "python3 inspect_dxf.py",
                                "Cwd": "/work",
                            },
                        ),
                        b"Permission denied for unsandboxed(python3 inspect_dxf.py). "
                        b"Matches user-configured deny rule.",
                    )
                ],
            )
            harvested = runner.harvest_agy_trajectory(home, root / "trajectory")
            audit = runner.audit_provider_trace(
                "gemini",
                runner.AttemptResult(0, "", "", 1.0),
                None,
                root / "events.jsonl",
                (),
                harvested,
            )
            self.assertEqual(audit["policy_violations"], [])
            self.assertEqual(audit["blocked_event_count"], 1)
            self.assertIn("sandbox bypass request", audit["blocked_policy_attempts"])

    def test_audit_uses_trajectory_store_and_fails_closed_without_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            home = root / "home"
            _write_agy_trajectory(
                home,
                [(_agy_step("run_command", {"CommandLine": "ls /work"}), b"")],
            )
            harvested = runner.harvest_agy_trajectory(home, root / "trajectory")
            ok = runner.AttemptResult(0, "done", "", 1.0)

            audit = runner.audit_provider_trace(
                "gemini", ok, None, root / "events.jsonl", (), harvested
            )
            self.assertEqual(audit["coverage"], "agy-trajectory-store")
            self.assertTrue(audit["complete"])
            self.assertEqual(audit["event_count"], 1)
            self.assertEqual(audit["parse_errors"], [])

            blind = runner.audit_provider_trace(
                "gemini", ok, None, root / "events2.jsonl", (), []
            )
            self.assertFalse(blind["complete"])
            self.assertTrue(
                any("no agy trajectory store" in e for e in blind["parse_errors"])
            )

    def test_credentials_are_redacted_from_trajectory_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            home = root / "home"
            _write_agy_trajectory(
                home,
                [
                    (
                        _agy_step(
                            "run_command",
                            {"CommandLine": "echo ya29.SECRET-TOKEN-VALUE"},
                        ),
                        b"",
                    )
                ],
            )
            harvested = runner.harvest_agy_trajectory(home, root / "trajectory")
            events, _ = runner.agy_trajectory_events(
                harvested, ("ya29.SECRET-TOKEN-VALUE",)
            )
            joined = "\n".join(events)
            self.assertNotIn("ya29.SECRET-TOKEN-VALUE", joined)
            self.assertIn("[REDACTED_CREDENTIAL]", joined)


class SandboxCommandTests(unittest.TestCase):
    def test_cad_library_preflight_runs_before_any_task(self) -> None:
        env_root = Path(sys.executable).resolve().parent.parent
        runner._require_cad_library(env_root, "cadquery")
        with self.assertRaises(runner.RunnerError) as raised:
            runner._require_cad_library(env_root, "definitely_not_a_cad_library")
        self.assertIn("cannot import", str(raised.exception))

    def test_cad_env_site_packages_is_exported_for_the_nested_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            env_root = Path(temp_name) / "venv"
            (env_root / "lib/python3.12/site-packages").mkdir(parents=True)
            self.assertEqual(
                runner.cad_env_site_packages(env_root),
                "/cad-env/lib/python3.12/site-packages",
            )
            command = runner._bwrap_base(
                workdir=Path(temp_name),
                disposable_home=Path(temp_name),
                executable=Path("/bin/true"),
                python_env=env_root,
            )
            index = command.index("PYTHONPATH")
            self.assertEqual(
                command[index + 1], "/cad-env/lib/python3.12/site-packages"
            )

    def test_jailed_interpreter_is_the_bound_environment(self) -> None:
        command = runner._execution_bwrap(
            Path("/tmp/work"), Path("/opt/cad-venv"), ["/cad-env/bin/python", "m.py"]
        )
        # `python3` on PATH must resolve inside the bound environment, not to the
        # system interpreter, or the agent probes a different set of libraries
        # than the one the runner executes its script with.
        self.assertIn("/cad-env/bin:/usr/bin:/bin", command)
        binds = [
            command[index + 1 : index + 3]
            for index, arg in enumerate(command)
            if arg == "--ro-bind"
        ]
        self.assertIn(["/opt/cad-venv", "/cad-env"], binds)

    def test_gemini_disposable_home_uses_fixed_sandbox_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source"
            destination = root / "destination"
            token = source / ".gemini/antigravity-cli/antigravity-oauth-token"
            token.parent.mkdir(parents=True)
            token.write_text("test-oauth-token")
            destination.mkdir()

            runner.prepare_disposable_home("gemini", destination, source)

            settings_path = destination / ".gemini/antigravity-cli/settings.json"
            self.assertEqual(
                json.loads(settings_path.read_text()),
                runner.GEMINI_DISPOSABLE_SETTINGS,
            )
            self.assertEqual(
                runner.GEMINI_DISPOSABLE_SETTINGS["toolPermission"],
                "proceed-in-sandbox",
            )
            self.assertTrue(runner.GEMINI_DISPOSABLE_SETTINGS["enableTerminalSandbox"])
            self.assertFalse(
                runner.GEMINI_DISPOSABLE_SETTINGS["allowNonWorkspaceAccess"]
            )
            permissions = runner.GEMINI_DISPOSABLE_SETTINGS["permissions"]
            for allowed in (
                "read_file(/work)",
                "write_file(/work)",
                # agy matches command prefixes, so the interpreter the agent
                # needs for local validation must be named explicitly.
                "command(python3)",
                "command(/cad-env/bin/python)",
            ):
                self.assertIn(allowed, permissions["allow"])
            for denied in (
                "read_file(/home)",
                "write_file(/home)",
                "read_url(*)",
                "execute_url(*)",
                "mcp(*)",
            ):
                self.assertIn(denied, permissions["deny"])
            # `unsandboxed(*)` matches every command, not just bypass requests,
            # and silently removes the agent's shell. Bypass is caught by the
            # trajectory audit instead.
            self.assertNotIn("unsandboxed(*)", permissions["deny"])

    def test_gemini_command_uses_model_label_without_duplicate_effort(self) -> None:
        for label in ("High", "Medium", "Low"):
            with self.subTest(label=label):
                command, stdin_text = runner.build_agent_command(
                    "gemini",
                    f"Gemini 3.6 Flash ({label})",
                    "prompt",
                    Path("/tmp/agy"),
                    Path("/tmp/work"),
                    Path("/tmp/home"),
                    Path("/tmp/env"),
                    1800,
                    Path("/tmp/audit"),
                )
                self.assertIsNone(stdin_text)
                self.assertNotIn("--effort", command)
                self.assertNotIn("--dangerously-skip-permissions", command)
                self.assertIn("accept-edits", command)
                self.assertIn("/audit/agy-cli.log", command)

    def test_log_redaction_removes_credentials_and_account_email(self) -> None:
        redacted = runner.redact_credentials(
            "token=top-secret-token email=person@example.com",
            ("top-secret-token",),
        )
        self.assertEqual(
            redacted,
            "token=[REDACTED_CREDENTIAL] email=[REDACTED_EMAIL]",
        )

    def test_gpt_disables_external_tool_features(self) -> None:
        command, _ = runner.build_agent_command(
            "gpt",
            "gpt-5.6-sol-max",
            "prompt",
            Path("/tmp/codex"),
            Path("/tmp/work"),
            Path("/tmp/home"),
            Path("/tmp/env"),
            1800,
        )
        for feature in runner.GPT_DISABLED_FEATURES:
            self.assertIn(feature, command)
        self.assertIn("sandbox_workspace_write.network_access=false", command)

    def test_non_gemini_commands_do_not_use_gemini_effort_mapping(self) -> None:
        command, _ = runner.build_agent_command(
            "claude",
            "claude-sonnet-4-6 (Low)",
            "prompt",
            Path("/tmp/claude"),
            Path("/tmp/work"),
            Path("/tmp/home"),
            Path("/tmp/env"),
            1800,
        )
        self.assertNotIn("--effort", command)

    def test_agent_jail_uses_disposable_paths_not_real_config(self) -> None:
        command = runner._bwrap_base(
            workdir=Path("/tmp/task"),
            disposable_home=Path("/tmp/auth"),
            executable=Path("/tmp/agy"),
            python_env=Path("/tmp/env"),
        )
        rendered = " ".join(command)
        self.assertIn("--clearenv", command)
        self.assertIn("/home/runner", command)
        self.assertNotIn("/home/ryotaro/.gemini", rendered)
        self.assertNotIn("/home/ryotaro/.codex", rendered)
        self.assertNotIn("/home/ryotaro/.claude", rendered)

    def test_cad_executor_unshares_network_and_all_namespaces(self) -> None:
        command = runner._execution_bwrap(
            Path("/tmp/task"), Path("/tmp/env"), ["/cad-env/bin/python", "model.py"]
        )
        self.assertIn("--unshare-all", command)
        self.assertIn("--clearenv", command)

    def test_cad_executor_runs_both_formats_in_isolated_directory(self) -> None:
        python_env = runner._python_env_from_interpreter(Path(sys.executable))
        scripts = {
            "cadquery": (
                "import cadquery as cq\n"
                "result = cq.Workplane('XY').box(1, 2, 3)\n"
                "cq.exporters.export(result, 'output.step')\n"
            ),
            "build123d": (
                "from build123d import Box, export_step\n"
                "result = Box(1, 2, 3)\n"
                "export_step(result, 'output.step')\n"
            ),
        }
        for format_name, source in scripts.items():
            with self.subTest(format_name), tempfile.TemporaryDirectory() as temp_name:
                work = Path(temp_name)
                candidate = work / "model_attempt_01.py"
                candidate.write_text(source)
                ok, _log, verification = runner.execute_and_verify(
                    work, python_env, 30, candidate.name
                )
                if not ok and "Failed to create NETLINK_ROUTE socket" in _log:
                    self.skipTest(
                        "outer managed sandbox forbids creating a nested network namespace"
                    )
                self.assertTrue(ok, _log)
                self.assertEqual(verification["solid_count"], 1)
                self.assertTrue((work / runner.STEP_NAME).is_file())


class IntermediateArtifactTests(unittest.TestCase):
    def test_preserves_generated_files_but_not_inputs_or_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            work = root / "work"
            result = root / "result"
            work.mkdir()
            result.mkdir()
            for name in (*runner.RENDER_INPUT_NAMES, runner.DXF_INPUT_NAME):
                (work / name).write_bytes(b"input")
            (work / "drawing_zoom.png").write_bytes(b"derived")
            (work / "notes").mkdir()
            (work / "notes" / "measurements.txt").write_text("42 mm")
            (work / "unsafe-link").symlink_to("/etc/passwd")

            manifest = runner._copy_intermediate_artifacts(work, result)

            saved = result / runner.INTERMEDIATES_DIR_NAME
            self.assertTrue((saved / "drawing_zoom.png").is_file())
            self.assertTrue((saved / "notes" / "measurements.txt").is_file())
            self.assertFalse((saved / runner.RENDER_INPUT_NAMES[0]).exists())
            self.assertFalse((saved / "unsafe-link").exists())
            self.assertIn("unsafe-link: symlink", manifest["skipped"])
            self.assertFalse(manifest["reused_by_future_tasks"])

    def test_artifact_monitor_preserves_and_flags_fast_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            work = root / "work"
            history = root / "history"
            work.mkdir()
            audit = root / "filesystem.jsonl"
            monitor = runner.ArtifactAuditMonitor(work, audit, history)
            monitor.start()
            candidate = work / "model_attempt_01.py"
            candidate.write_text("version one\n")
            time.sleep(0.02)
            candidate.write_text("version two\n")
            time.sleep(0.1)
            issues, warnings = monitor.stop()

            # In-pass rewrites are recorded, not fatal: both revisions survive.
            self.assertEqual(issues, [])
            self.assertTrue(any("rewritten in place" in x for x in warnings))
            self.assertEqual(
                (history / "model_attempt_01.revision_000.py").read_text(),
                "version one\n",
            )
            self.assertEqual(
                (history / "model_attempt_01.revision_001.py").read_text(),
                "version two\n",
            )
            self.assertTrue(audit.is_file())

    def test_artifact_monitor_ignores_duplicate_close_with_identical_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            work = root / "work"
            history = root / "history"
            work.mkdir()
            audit = root / "filesystem.jsonl"
            monitor = runner.ArtifactAuditMonitor(work, audit, history)
            monitor.start()
            candidate = work / "model_attempt_01.py"
            candidate.write_text("same content\n")
            time.sleep(0.05)
            candidate.write_text("same content\n")
            time.sleep(0.1)
            issues, warnings = monitor.stop()

            self.assertEqual(issues, [])
            self.assertEqual(warnings, [])
            revisions = list(history.glob("model_attempt_01.revision_*.py"))
            self.assertEqual(len(revisions), 1)

    def test_artifact_monitor_ignores_create_then_fill_write(self) -> None:
        """Reproduces the empty-then-content write seen in task 000405."""
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            work = root / "work"
            history = root / "history"
            work.mkdir()
            monitor = runner.ArtifactAuditMonitor(
                work, root / "filesystem.jsonl", history
            )
            monitor.start()
            candidate = work / "model_attempt_01.py"
            candidate.write_text("")
            time.sleep(0.05)
            candidate.write_text("import cadquery as cq\n")
            time.sleep(0.05)
            # A buffered writer flushing a growing prefix is one logical write.
            candidate.write_text("import cadquery as cq\nresult = None\n")
            time.sleep(0.1)
            issues, warnings = monitor.stop()

            self.assertEqual(issues, [])
            self.assertEqual(warnings, [])
            # The transient empty file is not kept as a revision.
            self.assertEqual(
                (history / "model_attempt_01.revision_000.py").read_text(),
                "import cadquery as cq\n",
            )

    def test_artifact_monitor_tolerates_atomic_rewrite_of_action_trace(self) -> None:
        """Reproduces the truncate-then-restore trace seen in task 001014."""
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            work = root / "work"
            work.mkdir()
            monitor = runner.ArtifactAuditMonitor(
                work, root / "filesystem.jsonl", root / "history"
            )
            monitor.start()
            trace = work / runner.ACTION_TRACE_NAME
            first = json.dumps({"sequence": 1, "action": "start"}) + "\n"
            rest = json.dumps({"sequence": 2, "action": "inspect"}) + "\n"
            trace.write_text(first)
            time.sleep(0.05)
            trace.write_text(rest)  # transient: new lines only
            time.sleep(0.05)
            trace.write_text(first + rest)  # repaired full file
            time.sleep(0.1)
            issues, warnings = monitor.stop()

            self.assertEqual(issues, [])
            self.assertEqual(warnings, [])

    def test_artifact_monitor_warns_but_does_not_fail_on_edited_reasoning(self) -> None:
        """Reproduces the in-place status-line edit seen in task 000364."""
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            work = root / "work"
            work.mkdir()
            monitor = runner.ArtifactAuditMonitor(
                work, root / "filesystem.jsonl", root / "history"
            )
            monitor.start()
            reasoning = work / runner.REASONING_NAME
            reasoning.write_text("## Stage 1\n- Status: initialized. Next: inspect.\n")
            time.sleep(0.05)
            reasoning.write_text("## Stage 1\n- Status: initialized.\n\n## Stage 2\n")
            time.sleep(0.1)
            issues, warnings = monitor.stop()

            self.assertEqual(issues, [])
            self.assertTrue(any("append-only" in warning for warning in warnings))

    def test_artifact_monitor_fails_on_destroyed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            work = root / "work"
            work.mkdir()
            monitor = runner.ArtifactAuditMonitor(
                work, root / "filesystem.jsonl", root / "history"
            )
            monitor.start()
            candidate = work / "model_attempt_01.py"
            candidate.write_text("result = None\n")
            reasoning = work / runner.REASONING_NAME
            reasoning.write_text("## Stage 1\n")
            time.sleep(0.05)
            candidate.unlink()
            reasoning.unlink()
            time.sleep(0.1)
            issues, _ = monitor.stop()

            self.assertTrue(any("model_attempt_01.py" in issue for issue in issues))
            self.assertTrue(any(runner.REASONING_NAME in issue for issue in issues))

    def test_partial_work_lost_to_an_outage_is_not_a_model_failure(self) -> None:
        # Observed on 000405: the CLI lost its transport after the agent had
        # already written model_attempt_01.py, so the task was graded on
        # deliverables the outage had truncated.
        metadata: dict[str, object] = {
            "status": "failed",
            "attempts": [
                {
                    "infra_fault": "provider connection timed out",
                    "issues": ["invalid final_selection.json: No such file"],
                }
            ],
        }
        runner._reclassify_infra_failure(metadata)
        self.assertEqual(metadata["status"], "infra-error")
        self.assertEqual(metadata["infra_fault"], "provider connection timed out")
        attempts = metadata["attempts"]
        assert isinstance(attempts, list)
        self.assertIn("rerun this task", attempts[0]["issues"][0])

        # Findings about the run itself must survive as themselves.
        for status in ("policy-violation", "audit-failure", "success"):
            with self.subTest(status=status):
                other: dict[str, object] = {
                    "status": status,
                    "attempts": [{"infra_fault": "provider connection timed out"}],
                }
                runner._reclassify_infra_failure(other)
                self.assertEqual(other["status"], status)

        # A model that simply failed stays a failure.
        plain: dict[str, object] = {"status": "failed", "attempts": [{"issues": []}]}
        runner._reclassify_infra_failure(plain)
        self.assertEqual(plain["status"], "failed")

    def test_infra_failures_are_named_and_model_failures_are_not(self) -> None:
        quota = runner.AttemptResult(
            1, "", "Error: Individual quota reached. Please upgrade.", 6.0
        )
        self.assertEqual(
            runner.classify_agent_failure(quota), "provider quota or rate limit reached"
        )
        crash = runner.AttemptResult(1, "", "Traceback: bad model output", 6.0)
        self.assertIsNone(runner.classify_agent_failure(crash))
        # Observed on 000364: the CLI lost its transport to the model service after
        # 992s of real work, which is an infrastructure fault, not a model failure.
        dropped = runner.AttemptResult(
            1,
            "",
            "Error: There was a network issue connecting to the server, "
            "please try again.",
            992.0,
        )
        self.assertEqual(
            runner.classify_agent_failure(dropped), "provider connection timed out"
        )
        # A model merely discussing timeouts must stay a model failure.
        chatter = runner.AttemptResult(
            1, "The pipe connection timed out in the original design.", "", 6.0
        )
        self.assertIsNone(runner.classify_agent_failure(chatter))
        self.assertEqual(
            runner.classify_agent_failure(runner.AttemptResult(-1, "", "", 1.0, True)),
            "agent CLI exceeded the runner wall-clock timeout",
        )

    def test_claude_command_has_no_web_tools_and_sandboxes_bash_network(self) -> None:
        command, stdin_text = runner.build_agent_command(
            "claude",
            "claude-opus-4-8",
            "prompt",
            Path("/tmp/claude"),
            Path("/tmp/work"),
            Path("/tmp/home"),
            Path("/tmp/env"),
            1800,
        )
        rendered = " ".join(command)
        self.assertIsNone(stdin_text)
        self.assertIn("Read,Write,Edit,Bash", command)
        self.assertIn('"allowedDomains":[]', rendered)
        self.assertIn("WebSearch,WebFetch", rendered)
        self.assertIn("--no-session-persistence", command)
        self.assertIn("dontAsk", command)
        self.assertIn("--no-chrome", command)

    def test_failed_artifacts_are_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            work = root / "work"
            result = root / "result"
            work.mkdir()
            result.mkdir()
            (work / runner.MODEL_NAME).write_text("untrusted")
            (work / runner.STEP_NAME).write_bytes(b"untrusted step")

            runner._copy_artifacts(work, result, success=False)

            self.assertFalse((result / runner.MODEL_NAME).exists())
            self.assertFalse((result / runner.STEP_NAME).exists())
            self.assertTrue(
                (result / runner.QUARANTINE_DIR_NAME / runner.MODEL_NAME).is_file()
            )


class AuditTests(unittest.TestCase):
    def test_gemini_native_log_is_included_in_policy_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            native_log = Path(temp_name) / "agy-cli.log"
            native_log.write_text(
                "prefix command_line:curl https://example.com/part.step command_output:blocked\n"
            )
            audit = runner.audit_provider_trace(
                "gemini",
                runner.AttemptResult(0, "done", "", 1.0),
                native_log,
                Path(temp_name) / "events.jsonl",
            )
            self.assertFalse(audit["complete"])
            self.assertIn("network command", audit["policy_violations"])

    def test_structured_trace_extracts_tool_event_and_policy_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            stdout = json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "command": "curl https://example.com/part.step",
                    },
                }
            )
            result = runner.AttemptResult(0, stdout + "\n", "", 1.0)
            audit = runner.audit_provider_trace(
                "gpt", result, None, Path(temp_name) / "events.jsonl"
            )
            self.assertEqual(audit["parse_errors"], [])
            self.assertIn("network command", audit["policy_violations"])

    def test_credentials_are_redacted(self) -> None:
        self.assertEqual(
            runner.redact_credentials("token=very-secret-token", ["very-secret-token"]),
            "token=[REDACTED_CREDENTIAL]",
        )

    def test_credential_content_is_never_copied_from_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            work = root / "work"
            result = root / "result"
            work.mkdir()
            result.mkdir()
            (work / runner.REASONING_NAME).write_text("leaked very-secret-token")
            self.assertEqual(
                runner.find_credential_leaks(work, ["very-secret-token"]),
                [runner.REASONING_NAME],
            )
            runner._copy_artifacts(
                work,
                result,
                success=False,
                redactions=["very-secret-token"],
            )
            self.assertFalse(
                (result / runner.QUARANTINE_DIR_NAME / runner.REASONING_NAME).exists()
            )

    def test_parent_stream_audit_is_timestamped_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            audit = Path(temp_name) / "stream.jsonl"
            result = runner.run_agent(
                ["/bin/sh", "-c", "echo very-secret-token; echo tool-error >&2"],
                None,
                5,
                audit,
                ["very-secret-token"],
            )
            self.assertEqual(result.returncode, 0)
            self.assertNotIn("very-secret-token", result.stdout)
            records = [json.loads(line) for line in audit.read_text().splitlines()]
            self.assertEqual(
                [record["sequence"] for record in records],
                list(range(1, len(records) + 1)),
            )
            self.assertTrue(all(record.get("observed_at") for record in records))


if __name__ == "__main__":
    unittest.main()
