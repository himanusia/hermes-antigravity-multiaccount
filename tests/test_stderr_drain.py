"""Tests for subprocess stderr draining and tail buffering (issue #17)."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add plugin parent dir to sys.path
plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from client import AntigravityClient
from process import StderrDrainer, start_stderr_drainer
from stream import AntigravityStream


class StderrDrainTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher_auth = patch("client.is_authenticated", return_value=True)
        patcher_token = patch("client.resolve_real_token_path", return_value=None)
        patcher_keychains = patch("process._link_macos_keychains")
        patcher_auth.start()
        patcher_token.start()
        patcher_keychains.start()
        self.addCleanup(patcher_auth.stop)
        self.addCleanup(patcher_token.stop)
        self.addCleanup(patcher_keychains.stop)

    def test_real_child_process_large_stderr_does_not_hang(self) -> None:
        """A child writing >=4 MiB to stderr before stdout completes without hanging."""
        # Pipe capacity on Linux is 16 pages: 64 KiB with 4 KiB pages (x86_64),
        # but 1 MiB on systems with 64 KiB pages (e.g. Linux arm64). Writing
        # 4 MiB (4096 lines of 1 KiB) guarantees saturating the pipe buffer
        # across all page sizes, hanging the child process unless stderr is
        # actively drained concurrently.
        script = (
            "import sys\n"
            "chunk = 'X' * 1023 + '\\n'\n"
            "for _ in range(4096):\n"
            "    sys.stderr.write(chunk)\n"
            "sys.stderr.write('FINAL_STDERR_MARKER\\n')\n"
            "sys.stderr.flush()\n"
            "sys.stdout.write('STDOUT_LINE_OK\\n')\n"
            "sys.stdout.flush()\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        drainer = start_stderr_drainer(proc, max_chars=65536)

        try:
            # Read stdout line; without concurrent stderr draining, this blocks/hangs
            stdout_line = proc.stdout.readline() if proc.stdout else ""
            self.assertEqual(stdout_line, "STDOUT_LINE_OK\n")
            returncode = proc.wait(timeout=5.0)
            self.assertEqual(returncode, 0)

            tail = drainer.get_tail(timeout=2.0)
            self.assertIn("FINAL_STDERR_MARKER", tail)
            self.assertLessEqual(len(tail), 65536)
            self.assertEqual(len(tail), 65536)
            self.assertTrue(tail.endswith("FINAL_STDERR_MARKER\n"))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_stderr_tail_respects_limit(self) -> None:
        """Bounded tail buffer strictly adheres to max_chars."""
        limit = 1024
        # Pipe capacity varies by page size (64 KiB with 4 KiB pages, 1 MiB
        # with 64 KiB pages). Writing 4 MiB ensures the pipe is saturated
        # across architectures while verifying the retained tail size.
        script = (
            "import sys\n"
            "chunk = 'A' * 1023 + '\\n'\n"
            "for _ in range(4096):\n"
            "    sys.stderr.write(chunk)\n"
            "sys.stderr.write('TAIL_LIMIT_CHECK_END')\n"
            "sys.stderr.flush()\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        drainer = start_stderr_drainer(proc, max_chars=limit)
        try:
            proc.wait(timeout=5.0)
            tail = drainer.get_tail(timeout=2.0)
            self.assertEqual(len(tail), limit)
            self.assertTrue(tail.endswith("TAIL_LIMIT_CHECK_END"))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_oneshot_error_path_surfaces_stderr_in_message(self) -> None:
        """Oneshot error path includes drained stderr in the raised exception message."""
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        client = AntigravityClient(cwd=tmp_dir.name)
        self.addCleanup(client.close)

        script = (
            "import sys\n"
            "sys.stderr.write('CRITICAL_FAILURE: token revoked by server\\n')\n"
            "sys.stderr.flush()\n"
            "sys.exit(1)\n"
        )
        client._command = sys.executable
        client._args = ["-c", script]

        # Acquire worker lock to force execution down the oneshot path
        self.assertTrue(client._worker_lock.acquire(blocking=False))
        try:
            with self.assertRaises(RuntimeError) as ctx:
                client.chat.completions.create(
                    model="gemini-3.8-flash",
                    messages=[{"role": "user", "content": "test error"}],
                    stream=False,
                )
            self.assertIn("Antigravity execution failed", str(ctx.exception))
            self.assertIn("CRITICAL_FAILURE: token revoked by server", str(ctx.exception))
        finally:
            client._worker_lock.release()

    def test_oneshot_streaming_error_path_surfaces_stderr_in_message(self) -> None:
        """Oneshot streaming error path includes drained stderr in the raised exception message."""
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        client = AntigravityClient(cwd=tmp_dir.name)
        self.addCleanup(client.close)

        script = (
            "import sys\n"
            "sys.stderr.write('STREAM_ERROR_DETAILS: connection refused\\n')\n"
            "sys.stderr.flush()\n"
            "sys.exit(2)\n"
        )
        client._command = sys.executable
        client._args = ["-c", script]

        self.assertTrue(client._worker_lock.acquire(blocking=False))
        try:
            stream = client.chat.completions.create(
                model="gemini-3.8-flash",
                messages=[{"role": "user", "content": "test error"}],
                stream=True,
            )
            with self.assertRaises(RuntimeError) as ctx:
                list(stream)
            self.assertIn("Antigravity execution failed", str(ctx.exception))
            self.assertIn("STREAM_ERROR_DETAILS: connection refused", str(ctx.exception))
        finally:
            client._worker_lock.release()

    def test_real_child_process_multibyte_utf8_not_corrupted(self) -> None:
        """Multibyte UTF-8 characters split across read boundaries are not corrupted into U+FFFD."""
        script = (
            "import sys\n"
            "sys.stderr.reconfigure(encoding='utf-8')\n"
            "pattern = 'ñ€😀'\n"
            "for _ in range(3000):\n"
            "    sys.stderr.write(pattern)\n"
            "sys.stderr.write('\\nMULTIBYTE_STDERR_END\\n')\n"
            "sys.stderr.flush()\n"
            "sys.stdout.write('OK\\n')\n"
            "sys.stdout.flush()\n"
        )
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )
        drainer = start_stderr_drainer(proc, max_chars=65536)
        try:
            line = proc.stdout.readline() if proc.stdout else ""
            self.assertEqual(line, "OK\n")
            proc.wait(timeout=5.0)
            tail = drainer.get_tail(timeout=2.0)
            self.assertIn("MULTIBYTE_STDERR_END", tail)
            self.assertIn("ñ€😀", tail)
            self.assertNotIn("\ufffd", tail)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_tolerates_fake_processes_without_real_stderr(self) -> None:
        """Helper safely tolerates mock processes without stderr or with mock stderr."""
        # Case 1: proc.stderr is None -> start() starts no thread and does not attach drainer
        mock_proc_none = MagicMock(spec=["poll"])
        mock_proc_none.stderr = None
        drainer_none = start_stderr_drainer(mock_proc_none)
        self.assertIsNone(drainer_none._thread)
        self.assertFalse(hasattr(mock_proc_none, "_stderr_drainer"))
        self.assertEqual(drainer_none.get_tail(), "")

        # Case 2: proc.stderr lacks read method -> start() starts no thread and does not attach drainer
        mock_proc_no_read = MagicMock(spec=["stderr"])
        mock_proc_no_read.stderr = 42
        drainer_no_read = start_stderr_drainer(mock_proc_no_read)
        self.assertIsNone(drainer_no_read._thread)
        self.assertFalse(hasattr(mock_proc_no_read, "_stderr_drainer"))
        self.assertEqual(drainer_no_read.get_tail(), "")

        # Case 3: proc.stderr is an unconfigured MagicMock -> read() returns MagicMock (not str), loop breaks immediately
        mock_proc_mock = MagicMock()
        drainer_mock = start_stderr_drainer(mock_proc_mock)
        if drainer_mock._thread is not None:
            drainer_mock._thread.join(timeout=1.0)
            self.assertFalse(drainer_mock._thread.is_alive())
        self.assertEqual(drainer_mock.get_tail(), "")

        # Case 4: proc.stderr is io.StringIO -> drained in thread, attached to proc
        mock_proc_io = MagicMock()
        mock_proc_io.stderr = io.StringIO("in-memory error content\n")
        drainer_io = start_stderr_drainer(mock_proc_io)
        self.assertTrue(hasattr(mock_proc_io, "_stderr_drainer"))
        self.assertEqual(drainer_io.get_tail(), "in-memory error content\n")

    def test_stream_get_stderr_tail_fallback(self) -> None:
        """AntigravityStream falls back cleanly if proc has no _stderr_drainer."""
        mock_proc = MagicMock(spec=["stderr", "poll", "wait"])
        mock_proc.stderr = io.StringIO("direct fallback stderr\n")
        mock_client = MagicMock()

        stream = AntigravityStream(
            proc=mock_proc,
            client=mock_client,
            model="gemini-3.8-flash",
            timeout=5.0,
            is_worker=False,
        )
        self.assertEqual(stream._get_stderr_tail(), "direct fallback stderr\n")

    @unittest.skipIf(
        os.name == "nt",
        "mock agy script with shebang is not executable on Windows",
    )
    def test_worker_turn_with_large_stderr_does_not_hang(self) -> None:
        """Persistent worker emitting >=4 MiB to stderr during a turn completes without hanging."""
        # Pipe capacity on Linux is 16 pages: 64 KiB with 4 KiB pages (x86_64),
        # but 1 MiB on systems with 64 KiB pages (e.g. Linux arm64). Writing
        # 4 MiB (4096 lines of 1 KiB) guarantees saturating the pipe buffer
        # across all page sizes, hanging the child process unless stderr is
        # actively drained concurrently.
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        client = AntigravityClient(cwd=tmp_dir.name)
        self.addCleanup(client.close)

        mock_agy = Path(tmp_dir.name) / "mock_agy"
        mock_agy.write_text(
            f"#!{sys.executable}\n"
            "import sys, json\n"
            "chunk = 'W' * 1023 + '\\n'\n"
            "for line in sys.stdin:\n"
            "    for _ in range(4096):\n"
            "        sys.stderr.write(chunk)\n"
            "    sys.stderr.write('WORKER_STDERR_END\\n')\n"
            "    sys.stderr.flush()\n"
            "    print(json.dumps({'event': 'init', 'conversation_id': 'worker-1'}))\n"
            "    print(json.dumps({'event': 'step_update', 'step_update': {'text_delta': 'worker done'}}))\n"
            "    print(json.dumps({'event': 'result', 'result': {'status': 'SUCCESS', 'response': 'worker done', 'usage': {}}}))\n"
            "    sys.stdout.flush()\n"
        )
        mock_agy.chmod(0o755)
        client._command = str(mock_agy)
        client._args = []

        res = client.chat.completions.create(
            model="gemini-3.8-flash",
            messages=[{"role": "user", "content": "turn 1"}],
            stream=False,
        )
        self.assertEqual(res.choices[0].message.content, "worker done")

        worker_proc = client._worker_proc
        self.assertIsNotNone(worker_proc)
        drainer = getattr(worker_proc, "_stderr_drainer", None)
        self.assertIsNotNone(drainer)
        client._terminate_worker()
        tail = drainer.get_tail(timeout=2.0)
        self.assertIn("WORKER_STDERR_END", tail)


if __name__ == "__main__":
    unittest.main()
