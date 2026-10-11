"""Tests for latest user request parity and optional prompt debug dumping."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from client import AntigravityClient
from debug import _reset_counter, dump_prompt_debug
from prompt import (
    _format_delta_prompt,
    _format_messages_as_prompt,
    _latest_user_request_section,
)

EXPECTED_INSTRUCTION_SUBSTRING = (
    "If the LATEST USER REQUEST comments on, questions, or gives feedback about prior work "
    "rather than asking to continue it, address THAT message and do NOT silently continue the earlier task."
)

MODEL = "gemini-3.8-flash-high"


def _worker_turn_lines(conversation_id: str, text: str) -> list[str]:
    events = [
        {"event": "init", "conversation_id": conversation_id},
        {"event": "step_update", "step_update": {"text_delta": text}},
        {
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "response": text,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    ]
    return [json.dumps(event) + "\n" for event in events] + [""]


def _make_mock_proc(lines: list[str]) -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None
    proc.wait.return_value = 0
    proc.stdout.readline.side_effect = list(lines)
    return proc


class TestLatestUserRequestParity(unittest.TestCase):
    def test_single_source_of_truth_exact_content(self):
        user_text = "What is the capital of France?"
        section = _latest_user_request_section(user_text)
        expected = (
            f"### LATEST USER REQUEST TO ANSWER:\nUser:\n{user_text}\n\n"
            "INSTRUCTION: Respond directly and specifically to the LATEST USER REQUEST above. "
            "Do NOT repeat previous architectural summaries, code reviews, or overview boilerplate unless explicitly asked."
            " If the LATEST USER REQUEST comments on, questions, or gives feedback about prior work rather than asking to continue it, address THAT message and do NOT silently continue the earlier task. If the user's intent is genuinely ambiguous, ask ONE short clarifying question instead of proceeding."
        )
        self.assertEqual(section, expected)
        self.assertIn(EXPECTED_INSTRUCTION_SUBSTRING, section)

    def test_full_prompt_produces_identical_section(self):
        user_text = "Analyze this project"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        full_prompt = _format_messages_as_prompt(messages)
        expected_section = _latest_user_request_section(user_text)
        self.assertTrue(full_prompt.endswith(expected_section))

    def test_full_prompt_sandwich_anchoring_in_multi_turn_history(self):
        user_text = "Write a detailed explanation of TCP congestion control"
        messages = [
            {"role": "user", "content": "Write a python script"},
            {"role": "assistant", "content": "Done with script."},
            {"role": "user", "content": user_text},
        ]
        full_prompt = _format_messages_as_prompt(messages)
        # Sandwich anchor before transcript
        self.assertIn("### ACTIVE OBJECTIVE:\n", full_prompt)
        self.assertIn(f"The user has submitted a new request:\n\"{user_text}\"", full_prompt)
        # Conversation transcript exists
        self.assertIn("Conversation transcript:\n", full_prompt)
        # LATEST USER REQUEST at the end
        expected_section = _latest_user_request_section(user_text)
        self.assertTrue(full_prompt.endswith(expected_section))
        # Active objective appears before Conversation transcript
        active_idx = full_prompt.index("### ACTIVE OBJECTIVE:\n")
        transcript_idx = full_prompt.index("Conversation transcript:\n")
        latest_idx = full_prompt.index("### LATEST USER REQUEST TO ANSWER:\n")
        self.assertTrue(active_idx < transcript_idx < latest_idx)

    def test_delta_prompt_user_produces_identical_section(self):
        user_text = "Analyze this project"
        delta_messages = [{"role": "user", "content": user_text}]
        delta_prompt = _format_delta_prompt(delta_messages)
        expected_section = _latest_user_request_section(user_text)
        self.assertEqual(delta_prompt, expected_section)

    def test_both_routes_share_identical_latest_request_string(self):
        user_text = "Explain the delta"
        full = _format_messages_as_prompt([{"role": "user", "content": user_text}])
        delta = _format_delta_prompt([{"role": "user", "content": user_text}])
        shared_section = _latest_user_request_section(user_text)

        self.assertTrue(full.endswith(shared_section))
        self.assertEqual(delta, shared_section)

    def test_delta_prompt_user_text_appears_only_once(self):
        user_text = "unique_marker_do_not_duplicate_42"
        delta_messages = [
            {"role": "assistant", "content": "Previous task done."},
            {"role": "user", "content": user_text},
        ]
        delta_prompt = _format_delta_prompt(delta_messages)

        # The user text appears only once in the entire delta prompt
        self.assertEqual(delta_prompt.count(user_text), 1)
        # It is inside the LATEST USER REQUEST section
        self.assertIn(f"### LATEST USER REQUEST TO ANSWER:\nUser:\n{user_text}", delta_prompt)
        # Assistant message is rendered before it
        self.assertIn("Assistant:\nPrevious task done.", delta_prompt)
        # Anti-loop instruction is present
        self.assertIn(EXPECTED_INSTRUCTION_SUBSTRING, delta_prompt)

    def test_delta_prompt_single_user_message_no_loose_user_prefix(self):
        user_text = "standalone_user_msg"
        delta_messages = [{"role": "user", "content": user_text}]
        delta_prompt = _format_delta_prompt(delta_messages)

        self.assertEqual(delta_prompt.count(user_text), 1)
        self.assertFalse(delta_prompt.startswith(f"User:\n{user_text}\n\n"))
        self.assertTrue(delta_prompt.startswith("### LATEST USER REQUEST TO ANSWER:"))

    def test_delta_prompt_tool_role_unchanged(self):
        tool_messages = [
            {"role": "tool", "tool_call_id": "call_abc", "content": "Tool output 123"}
        ]
        delta_prompt = _format_delta_prompt(tool_messages)

        self.assertNotIn("### LATEST USER REQUEST TO ANSWER", delta_prompt)
        self.assertNotIn(EXPECTED_INSTRUCTION_SUBSTRING, delta_prompt)
        self.assertIn("Tool Result (call_abc):\nTool output 123", delta_prompt)
        self.assertIn("Continue the conversation from the latest tool result.", delta_prompt)
        self.assertIn("REMINDER: You are in HEADLESS INFERENCE MODE.", delta_prompt)

    def test_delta_prompt_other_role_unchanged(self):
        other_messages = [
            {"role": "assistant", "content": "Assistant only message"}
        ]
        delta_prompt = _format_delta_prompt(other_messages)

        self.assertNotIn("### LATEST USER REQUEST TO ANSWER", delta_prompt)
        self.assertIn("Assistant:\nAssistant only message", delta_prompt)
        self.assertTrue(delta_prompt.endswith("Continue the conversation from the latest message above."))


class TestPromptDebugDump(unittest.TestCase):
    def setUp(self):
        _reset_counter(0)

    def test_dump_disabled_when_unset_or_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for env_val in (None, "", "   "):
                env_patch = {"ANTIGRAVITY_DEBUG_PROMPT_DIR": env_val} if env_val is not None else {}
                with patch.dict(os.environ, env_patch, clear=False):
                    if env_val is None and "ANTIGRAVITY_DEBUG_PROMPT_DIR" in os.environ:
                        del os.environ["ANTIGRAVITY_DEBUG_PROMPT_DIR"]
                    res = dump_prompt_debug(
                        prompt="hello prompt",
                        branch="delta",
                        model=MODEL,
                        messages=[{"role": "user", "content": "hi"}],
                    )
                    self.assertIsNone(res)
            # Verify no files were created in tmp_dir
            self.assertEqual(os.listdir(tmp_dir), [])

    def test_dump_creates_file_with_correct_permissions_posix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_dir = Path(tmp_dir) / "debug_dumps"
            with patch.dict(os.environ, {"ANTIGRAVITY_DEBUG_PROMPT_DIR": str(dump_dir)}):
                res = dump_prompt_debug(
                    prompt="hello prompt",
                    branch="delta",
                    model=MODEL,
                    messages=[{"role": "user", "content": "hi"}],
                )
                self.assertIsNotNone(res)
                self.assertTrue(res.exists())

                if os.name == "posix":
                    dir_stat = os.stat(dump_dir)
                    self.assertEqual(stat.S_IMODE(dir_stat.st_mode), 0o700)
                    file_stat = os.stat(res)
                    self.assertEqual(stat.S_IMODE(file_stat.st_mode), 0o600)

    def test_dump_content_and_header(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_dir = Path(tmp_dir) / "debug_dumps"
            messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "content": "t1"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "u2"},
                {"role": "user", "content": "u3"},
            ]
            with patch.dict(os.environ, {"ANTIGRAVITY_DEBUG_PROMPT_DIR": str(dump_dir)}):
                res = dump_prompt_debug(
                    prompt="RAW PROMPT CONTENT 999",
                    branch="full",
                    model=MODEL,
                    messages=messages,
                )
                self.assertIsNotNone(res)
                content = res.read_text(encoding="utf-8")

                self.assertIn("=== PROMPT DEBUG DUMP ===", content)
                self.assertIn(f"Model: {MODEL}", content)
                self.assertIn("Branch: full", content)
                self.assertIn("Message Count: 7", content)
                # Last 6 roles of 7: u1, a1, t1, a2, u2, u3
                self.assertIn("Last Roles: user, assistant, tool, assistant, user, user", content)
                self.assertIn("RAW PROMPT CONTENT 999", content)

                # Ensure no secrets, tokens or env vars leak into file
                self.assertNotIn("KEYRING", content)
                self.assertNotIn("TOKEN", content.upper().replace("TOTAL_TOKENS", ""))

    def test_dump_filename_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_dir = Path(tmp_dir) / "debug_dumps"
            with patch.dict(os.environ, {"ANTIGRAVITY_DEBUG_PROMPT_DIR": str(dump_dir)}):
                file1 = dump_prompt_debug("p1", branch="delta")
                file2 = dump_prompt_debug("p2", branch="oneshot")

                pattern = r"^\d{8}T\d{6}Z_\d{4}_(delta|oneshot)\.txt$"
                self.assertRegex(file1.name, pattern)
                self.assertRegex(file2.name, pattern)
                self.assertTrue(file1.name.endswith("_0001_delta.txt"))
                self.assertTrue(file2.name.endswith("_0002_oneshot.txt"))

    def test_dump_write_error_does_not_propagate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_dir = Path(tmp_dir) / "debug_dumps"
            with patch.dict(os.environ, {"ANTIGRAVITY_DEBUG_PROMPT_DIR": str(dump_dir)}):
                with patch("os.open", side_effect=OSError("Disk full")):
                    # Must not raise
                    res = dump_prompt_debug("prompt", branch="delta")
                    self.assertIsNone(res)


class TestClientDebugDumpDispatch(unittest.TestCase):
    def setUp(self):
        _reset_counter(0)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.debug_dir = Path(self.tmp_dir.name) / "prompt_dumps"

        patcher_auth = patch("client.is_authenticated", return_value=True)
        patcher_token = patch("process.resolve_real_token_path", return_value=None)
        patcher_cmd = patch("client.resolve_agy_command", return_value="agy")
        patcher_keychains = patch("process._link_macos_keychains")
        patcher_auth.start()
        patcher_token.start()
        patcher_cmd.start()
        patcher_keychains.start()
        self.addCleanup(patcher_auth.stop)
        self.addCleanup(patcher_token.stop)
        self.addCleanup(patcher_cmd.stop)
        self.addCleanup(patcher_keychains.stop)

    def _client(self) -> AntigravityClient:
        return AntigravityClient(cwd=self.tmp_dir.name)

    def test_client_all_4_dispatch_paths(self):
        with patch.dict(os.environ, {"ANTIGRAVITY_DEBUG_PROMPT_DIR": str(self.debug_dir)}):
            # 1. Path: "full" (Worker spawned, empty history)
            client = self._client()
            self.addCleanup(client.close)
            proc_full = _make_mock_proc(_worker_turn_lines("conv-1", "Answer full"))

            with patch("subprocess.Popen", return_value=proc_full):
                res1 = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": "Question 1"}],
                    stream=False,
                )
                self.assertEqual(res1.choices[0].message.content, "Answer full")

            dump_files = sorted(self.debug_dir.glob("*.txt"))
            self.assertEqual(len(dump_files), 1)
            self.assertTrue(dump_files[0].name.endswith("_full.txt"))
            content = dump_files[0].read_text(encoding="utf-8")
            self.assertIn("Branch: full", content)

            # 2. Path: "delta" (Worker continuation with matching history prefix)
            proc_delta = proc_full
            proc_delta.stdout.readline.side_effect = _worker_turn_lines("conv-1", "Answer delta")
            continuation_msgs = [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer full"},
                {"role": "user", "content": "Question 2"},
            ]
            with patch("subprocess.Popen", return_value=proc_delta):
                res2 = client.chat.completions.create(
                    model=MODEL,
                    messages=continuation_msgs,
                    stream=False,
                )
                self.assertEqual(res2.choices[0].message.content, "Answer delta")

            dump_files = sorted(self.debug_dir.glob("*.txt"))
            self.assertEqual(len(dump_files), 2)
            self.assertTrue(dump_files[1].name.endswith("_delta.txt"))
            content_delta = dump_files[1].read_text(encoding="utf-8")
            self.assertIn("Branch: delta", content_delta)

            # 3. Path: "retry" (BrokenPipeError on worker write -> respawns and retries)
            proc_broken = _make_mock_proc([])
            proc_broken.stdin.write.side_effect = BrokenPipeError("Worker died")
            proc_respawned = _make_mock_proc(_worker_turn_lines("conv-respawn", "Answer retry"))

            # Next request: non-continuation or continuation triggers broken pipe
            with patch("subprocess.Popen", side_effect=[proc_broken, proc_respawned]):
                client._worker_proc = proc_broken
                res3 = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": "Retry test"}],
                    stream=False,
                )
                self.assertEqual(res3.choices[0].message.content, "Answer retry")

            dump_files = sorted(self.debug_dir.glob("*.txt"))
            retry_files = [f for f in dump_files if f.name.endswith("_retry.txt")]
            self.assertEqual(len(retry_files), 1)
            content_retry = retry_files[0].read_text(encoding="utf-8")
            self.assertIn("Branch: retry", content_retry)

            # 4. Path: "oneshot" (Worker lock unavailable -> falls back to oneshot)
            proc_oneshot = _make_mock_proc(_worker_turn_lines("conv-oneshot", "Answer oneshot"))
            client._worker_lock.acquire()  # Simulate concurrent turn holding worker lock
            try:
                with patch("subprocess.Popen", return_value=proc_oneshot):
                    res4 = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": "Oneshot test"}],
                        stream=False,
                    )
                    self.assertEqual(res4.choices[0].message.content, "Answer oneshot")
            finally:
                client._worker_lock.release()

            dump_files = sorted(self.debug_dir.glob("*.txt"))
            oneshot_files = [f for f in dump_files if f.name.endswith("_oneshot.txt")]
            self.assertEqual(len(oneshot_files), 1)
            content_oneshot = oneshot_files[0].read_text(encoding="utf-8")
            self.assertIn("Branch: oneshot", content_oneshot)
