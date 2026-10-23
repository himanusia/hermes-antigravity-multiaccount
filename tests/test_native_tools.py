"""agy-native tool steps become Hermes tool calls; unmappable ones stay neutralized."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import native_tools  # noqa: E402
from client import AntigravityClient  # noqa: E402

TERMINAL_TOOL = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]


def _step(name, params, idx=2):
    return {"step_index": idx, "state": "ACTIVE", "step_type": "tool", "tool_name": name,
            "tool_info": {"name": name, "parameters": params}}


class TranslateTests(unittest.TestCase):
    def test_run_command_maps_to_terminal(self):
        call = native_tools.translate(_step("run_command", {"CommandLine": "echo hi", "Cwd": "/w"}), {"terminal"}, 0)
        self.assertEqual(call.function.name, "terminal")
        self.assertEqual(json.loads(call.function.arguments), {"command": "echo hi", "workdir": "/w"})
        self.assertRegex(call.id, r"^agy_\d+_2_run_command$")

    def test_unavailable_hermes_tool_is_not_mapped(self):
        self.assertIsNone(native_tools.translate(_step("run_command", {"CommandLine": "x"}), {"read_file"}, 0))

    def test_unknown_native_tool_is_not_mapped(self):
        self.assertIsNone(native_tools.translate(_step("browser_click_element", {"x": 1}), {"terminal"}, 0))

    def test_missing_parameters_is_not_mapped(self):
        self.assertIsNone(native_tools.translate({"step_type": "tool", "tool_name": "run_command"}, {"terminal"}, 0))

    def test_view_file_and_search(self):
        c = native_tools.translate(_step("view_file", {"AbsolutePath": "/a.py", "StartLine": 5}), {"read_file"}, 0)
        self.assertEqual(json.loads(c.function.arguments), {"path": "/a.py", "offset": 5})
        c = native_tools.translate(_step("grep_search", {"Query": "foo", "SearchPath": "/r"}), {"search_files"}, 0)
        self.assertEqual(json.loads(c.function.arguments), {"pattern": "foo", "target": "content", "path": "/r"})


class StreamBridgeTests(unittest.TestCase):
    def _run(self, events, tools):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        client = AntigravityClient(cwd=tmp.name)
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdout.readline.side_effect = [json.dumps(e) + "\n" for e in events] + [""] * 50
        proc.stdin = MagicMock()
        # Hermetic: never probe the host keyring (on Linux that probe itself goes through Popen).
        # Hermetic: never probe the host keyring (on Linux that probe itself goes through Popen).
        with patch("client.is_authenticated", return_value=True), \
                patch("subprocess.Popen", return_value=proc), patch.object(client, "_terminate_process") as term:
            chunks = list(client.chat.completions.create(model="claude-sonnet-4-6", tools=tools, stream=True,
                                                         messages=[{"role": "user", "content": "run it"}]))
        return chunks, term, proc

    def test_native_run_command_is_reissued_as_hermes_terminal_call(self):
        events = [{"event": "init", "conversation_id": "c"},
                  {"event": "step_update", "step_update": _step("run_command", {"CommandLine": "echo SOUL_42"})}]
        chunks, term, proc = self._run(events, TERMINAL_TOOL)
        term.assert_called_once_with(proc)  # agy never gets to execute it
        calls = [tc for ch in chunks for tc in (ch.choices[0].delta.tool_calls or [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "terminal")
        self.assertEqual(json.loads(calls[0].function.arguments)["command"], "echo SOUL_42")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

    def test_unmappable_native_tool_still_neutralized_without_call(self):
        events = [{"event": "step_update", "step_update": _step("browser_click_element", {"x": 1})}]
        chunks, term, proc = self._run(events, TERMINAL_TOOL)
        term.assert_called_once_with(proc)
        calls = [tc for ch in chunks for tc in (ch.choices[0].delta.tool_calls or [])]
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
