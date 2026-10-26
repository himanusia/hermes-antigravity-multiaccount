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
WEB_SEARCH_TOOL = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]


def _step(name, params, idx=2):
    return {"step_index": idx, "state": "ACTIVE", "step_type": "tool", "tool_name": name,
            "tool_info": {"name": name, "parameters": params}}


class TranslateTests(unittest.TestCase):
    def test_every_mapped_native_tool_reaches_its_hermes_tool(self):
        # Sample arguments follow the field names agy 1.4.7 actually emits.
        cases = {
            "run_command": ({"CommandLine": "echo hi"}, "terminal", {"command": "echo hi"}),
            "view_file": ({"AbsolutePath": "/a.py"}, "read_file", {"path": "/a.py"}),
            "read_file": ({"Path": "/a.py", "StartLine": 3}, "read_file", {"path": "/a.py", "offset": 3}),
            "write_to_file": ({"TargetFile": "/a.py", "CodeContent": "x"}, "write_file",
                              {"path": "/a.py", "content": "x"}),
            "write_file": ({"TargetFile": "/a.py", "CodeContent": "x"}, "write_file",
                           {"path": "/a.py", "content": "x"}),
            "grep_search": ({"Query": "foo"}, "search_files",
                            {"pattern": "foo", "target": "content"}),
            "find_by_name": ({"Pattern": "*.py"}, "search_files",
                             {"pattern": "*.py", "target": "files"}),
            "search_web": ({"query": "hermes"}, "web_search", {"query": "hermes"}),
            "web_search": ({"query": "hermes"}, "web_search", {"query": "hermes"}),
            "read_url_content": ({"Url": "https://example.com"}, "web_extract",
                                 {"urls": ["https://example.com"]}),
        }
        self.assertEqual(set(cases), set(native_tools.NATIVE_TO_HERMES))
        for native, (params, hermes_name, expected_args) in cases.items():
            with self.subTest(native=native):
                call = native_tools.translate(_step(native, params), {hermes_name}, 0)
                assert call is not None, f"{native} should map to {hermes_name}"
                self.assertEqual(call.function.name, hermes_name)
                self.assertEqual(json.loads(call.function.arguments), expected_args)

    def test_run_command_maps_to_terminal(self):
        call = native_tools.translate(_step("run_command", {"CommandLine": "echo hi", "Cwd": "/w"}), {"terminal"}, 0)
        self.assertEqual(call.function.name, "terminal")
        self.assertEqual(json.loads(call.function.arguments), {"command": "echo hi", "workdir": "/w"})
        self.assertRegex(call.id, r"^agy_\d+_2_run_command$")

    def test_unavailable_hermes_tool_is_not_mapped(self):
        self.assertIsNone(native_tools.translate(_step("run_command", {"CommandLine": "x"}), {"read_file"}, 0))

    def test_search_web_only_maps_when_web_search_is_available(self):
        step = _step("search_web", {"query": "hermes"})
        self.assertIsNone(native_tools.translate(step, {"terminal"}, 0))
        self.assertIsNotNone(native_tools.translate(step, {"web_search"}, 0))

    def test_unknown_native_tool_is_not_mapped(self):
        self.assertIsNone(native_tools.translate(_step("browser_click_element", {"x": 1}), {"terminal"}, 0))

    def test_native_names_absent_from_agy_are_not_mapped(self):
        # agy 1.4.7 has no such native tools; a lookup must not invent a call.
        available = {"web_search", "web_extract", "search_files", "read_file"}
        for native in ("search_google", "fetch_url", "open_url", "browse", "web_fetch"):
            with self.subTest(native=native):
                self.assertIsNone(native_tools.translate(_step(native, {"query": "x", "url": "https://e.com"}), available, 0))

    # Captured verbatim from `agy -p ... --output-format stream-json` on 2026-10-26
    # (agy 1.4.7, gemini-3.8-flash-high): the model called its own search_web.
    def test_search_web_real_payload_maps_to_web_search_with_the_query(self):
        step = _step("search_web", {"query": "Hermes Agent Nous Research GitHub repository"})
        call = native_tools.translate(step, {"web_search"}, 3)
        assert call is not None
        self.assertEqual(call.function.name, "web_search")
        self.assertEqual(json.loads(call.function.arguments),
                         {"query": "Hermes Agent Nous Research GitHub repository"})
        self.assertRegex(call.id, r"^agy_\d+_2_search_web$")

    def test_read_url_content_maps_to_web_extract_with_urls_list(self):
        call = native_tools.translate(_step("read_url_content", {"Url": "https://example.com/a"}),
                                      {"web_extract"}, 0)
        assert call is not None
        self.assertEqual(call.function.name, "web_extract")
        self.assertEqual(json.loads(call.function.arguments), {"urls": ["https://example.com/a"]})
        # A list payload keeps every URL instead of stringifying the list.
        call = native_tools.translate(_step("read_url_content", {"urls": ["https://a.test", "https://b.test"]}),
                                      {"web_extract"}, 0)
        assert call is not None
        self.assertEqual(json.loads(call.function.arguments),
                         {"urls": ["https://a.test", "https://b.test"]})

    def test_missing_parameters_is_not_mapped(self):
        self.assertIsNone(native_tools.translate({"step_type": "tool", "tool_name": "run_command"}, {"terminal"}, 0))

    def test_search_web_without_query_is_not_mapped(self):
        self.assertIsNone(native_tools.translate(_step("search_web", {"domain": "example.com"}), {"web_search"}, 0))

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

    def test_native_search_web_is_reissued_as_hermes_web_search(self):
        # Same event shape agy 1.4.7 streams for a native web search.
        events = [{"event": "init", "conversation_id": "c"},
                  {"event": "step_update",
                   "step_update": _step("search_web", {"query": "Hermes Agent Nous Research GitHub repository"})}]
        chunks, term, proc = self._run(events, WEB_SEARCH_TOOL)
        term.assert_called_once_with(proc)  # agy never searches on its own network path
        calls = [tc for ch in chunks for tc in (ch.choices[0].delta.tool_calls or [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "web_search")
        self.assertEqual(json.loads(calls[0].function.arguments),
                         {"query": "Hermes Agent Nous Research GitHub repository"})
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

    def test_unmappable_native_tool_still_neutralized_without_call(self):
        events = [{"event": "step_update", "step_update": _step("browser_click_element", {"x": 1})}]
        chunks, term, proc = self._run(events, TERMINAL_TOOL)
        term.assert_called_once_with(proc)
        calls = [tc for ch in chunks for tc in (ch.choices[0].delta.tool_calls or [])]
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
