"""A bridged native tool step must not throw the conversation away.

agy's native tool steps are re-issued as Hermes tool calls: the process is
killed before it can run them (correct), but the CONVERSATION has to survive so
the tool result comes back as a delta into the resumed session. Dropping it
replayed the whole transcript into a fresh, memoryless session -- the model then
continued the previous command as if the user's latest message never arrived.
"""

import json
import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from client import AntigravityClient  # noqa: E402

TERMINAL_TOOL = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]


class _FakeWorker:
    """A persistent agy process: readline blocks until the test feeds a line."""

    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self.alive = True
        self.proc = MagicMock()
        self.proc.poll.side_effect = lambda: None if self.alive else 0
        self.proc.stdout.readline.side_effect = lambda: self._queue.get()
        self.proc.stdin = MagicMock()

    def feed(self, events):
        for event in events:
            self._queue.put(json.dumps(event) + "\n")
        return self.proc


def _init(conv):
    return {"event": "init", "init": {"conversation_id": conv}}


def _text(text, conv):
    return {"event": "step_update", "step_update": {"text_delta": text, "conversation_id": conv}}


def _done(text, conv):
    return {"event": "result", "result": {"status": "SUCCESS", "response": text, "conversation_id": conv}}


def _native_step(conv):
    return {"event": "step_update", "step_update": {
        "step_type": "tool", "tool_name": "run_command", "step_index": 2,
        "tool_info": {"parameters": {"CommandLine": "echo hi"}}, "conversation_id": conv}}


def _branches(dumps: Path) -> list[str]:
    names = sorted(p.name for p in dumps.glob("*_*.txt"))
    return [n.rsplit("_", 1)[-1].removesuffix(".txt") for n in names]


class BridgeContinuityTests(unittest.TestCase):
    def _client(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        client = AntigravityClient(cwd=tmp.name)
        self.dumps = Path(tmp.name) / "prompts"
        self.dumps.mkdir(exist_ok=True)
        self.conv = "conv-keep"
        conv_dir = client._isolated_gemini_dir / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / f"{self.conv}.db").write_bytes(b"")
        env = patch.dict(os.environ, {  # isolated from the developer's real accounts
            "ANTIGRAVITY_ACCOUNTS_FILE": str(Path(tmp.name) / "accounts.json"),
            "ANTIGRAVITY_ACCOUNTS_DIR": str(Path(tmp.name) / "accounts"),
            "ANTIGRAVITY_ROTATION": "off",
            "ANTIGRAVITY_DEBUG_PROMPT_DIR": str(self.dumps),
        })
        env.start()
        self.addCleanup(env.stop)
        return client

    def _call(self, client, messages):
        return client._execute_chat_completion(
            resolved_model="claude-sonnet-4-6", effort=None,
            messages_list=[dict(m) for m in messages], effective_timeout=20.0,
            tools=TERMINAL_TOOL, tool_choice=None, stream=False,
            home_dir=None, account_label=None)

    def test_bridged_tool_step_keeps_the_conversation_and_the_result_is_a_delta(self):
        client = self._client()
        worker = _FakeWorker()
        worker.feed([_init(self.conv), _text("satu", self.conv), _done("satu", self.conv)])
        respawned = _FakeWorker()

        with patch("subprocess.Popen", return_value=respawned.proc), \
                patch.object(client, "_terminate_process"):
            messages = [{"role": "user", "content": "satu"}]
            with patch("subprocess.Popen", return_value=worker.proc):
                first = self._call(client, messages)
            self.assertEqual(first.choices[0].finish_reason, "stop")
            history_after_turn1 = list(client._conversation_history)
            self.assertTrue(history_after_turn1)

            messages += [{"role": "assistant", "content": "satu"}, {"role": "user", "content": "dua"}]
            worker.feed([_init(self.conv), _native_step(self.conv)])
            bridged = self._call(client, messages)
            call = bridged.choices[0].message.tool_calls[0]

            # The process is gone, the conversation is not.
            self.assertIsNone(client._worker_proc)
            self.assertEqual(client._worker_conversation_id, self.conv)
            self.assertEqual(client._conversation_history, history_after_turn1)

            messages += [
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": call.id, "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments}}]},
                {"role": "tool", "tool_call_id": call.id, "name": "terminal",
                 "content": json.dumps({"output": "hi", "exit_code": 0})},
            ]
            respawned.feed([_init(self.conv), _text("selesai", self.conv), _done("selesai", self.conv)])
            self._call(client, messages)

        self.assertEqual(_branches(self.dumps), ["full", "delta", "delta"])
        delta_prompt = sorted(self.dumps.glob("*_delta.txt"))[-1].read_text(encoding="utf-8")
        self.assertIn("Tool Result (", delta_prompt)          # the result travels as an increment
        self.assertNotIn("Conversation transcript", delta_prompt)  # not a full replay

    def test_dead_worker_without_a_resumable_conversation_falls_back_to_full_prompt(self):
        client = self._client()
        worker = _FakeWorker()
        worker.feed([_init(self.conv), _text("satu", self.conv), _done("satu", self.conv)])
        respawned = _FakeWorker()
        respawned.feed([_init(self.conv), _text("dua", self.conv), _done("dua", self.conv)])

        with patch("subprocess.Popen", return_value=respawned.proc), \
                patch.object(client, "_terminate_process"):
            messages = [{"role": "user", "content": "satu"}]
            with patch("subprocess.Popen", return_value=worker.proc):
                self._call(client, messages)

            # Worker died and its conversation database is gone: a delta would
            # reach a process that knows nothing, so the prompt must be complete.
            worker.alive = False
            client._terminate_worker_locked(keep_conversation=True)
            (client._isolated_gemini_dir / "conversations" / f"{self.conv}.db").unlink()

            messages += [{"role": "assistant", "content": "satu"}, {"role": "user", "content": "dua"}]
            self._call(client, messages)

        self.assertEqual(_branches(self.dumps), ["full", "full"])
        full_prompt = sorted(self.dumps.glob("*_full.txt"))[-1].read_text(encoding="utf-8")
        self.assertIn("dua", full_prompt)


if __name__ == "__main__":
    unittest.main()
