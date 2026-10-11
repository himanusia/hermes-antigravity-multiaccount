"""Tests for non-blocking close() inside a running asyncio event loop.

AntigravityStream.close() is called by Hermes' _close_chunk_stream in the
finally block of _aggregate_chat_stream_async, which runs on the event loop
thread.  The blocking process termination (proc.terminate + proc.wait(2))
and the client RLock acquisition in _terminate_worker would stall the loop
for up to ~2 s if close() ran synchronously there.

The fix detects a running loop (asyncio.get_running_loop) and dispatches the
blocking work to a dedicated daemon thread while returning immediately.
Outside a loop, close() is unchanged: fully synchronous.

These tests pin:
  (a) close() inside a coroutine returns in < 0.2 s with a ~1 s termination,
      and the deferred thread completes the termination and lock release.
  (b) close() outside a loop runs synchronously before returning.
  (c) Idempotency inside the loop: a second close() does not launch another
      thread and does not raise.
  (d) Worker-replaced guard: the deferred thread does not kill the new worker
      when the old worker was replaced between dispatch and execution.
"""

import asyncio
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add plugin parent dir to sys.path
plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from client import AntigravityClient
from stream import AntigravityStream

MODEL = "gemini-3.8-flash-high"
MESSAGES = [{"role": "user", "content": "turn one"}]
TURN_USAGE = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}


def _turn_lines(text: str = "answer one") -> list[str]:
    events = [
        {"event": "init", "conversation_id": "conv-1"},
        {"event": "step_update", "step_update": {"text_delta": text}},
        {"event": "result", "result": {"status": "SUCCESS", "response": text, "usage": TURN_USAGE}},
    ]
    return [json.dumps(e) + "\n" for e in events] + [""]


def _mock_proc(
    lines: list[str],
    *,
    alive: bool = True,
    terminate_delay: float = 0.0,
    wait_delay: float = 0.0,
) -> MagicMock:
    """Create a mock subprocess.Popen.

    *terminate_delay* and *wait_delay* simulate a slow process termination:
    proc.terminate() sleeps for *terminate_delay*, proc.wait() sleeps for
    *wait_delay*.  Both default to instant (0).
    """
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None if alive else 0
    proc.stdout.readline.side_effect = lines

    real_terminate = proc.terminate

    def _slow_terminate(*args: Any, **kwargs: Any) -> None:
        if terminate_delay > 0:
            time.sleep(terminate_delay)
        # Simulate the process dying after terminate.
        proc.poll.return_value = 0

    def _slow_wait(*args: Any, **kwargs: Any) -> int:
        if wait_delay > 0:
            time.sleep(wait_delay)
        return 0

    proc.terminate.side_effect = _slow_terminate
    proc.wait.side_effect = _slow_wait
    return proc


class _PinnedSeamsMixin:
    def setUp(self):
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
        temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_test_")
        self.addCleanup(temp_dir.cleanup)
        client = AntigravityClient(cwd=temp_dir.name)
        self.addCleanup(client.close)
        return client


def _wait_for_close_threads(timeout: float = 3.0) -> None:
    """Wait until all agy-stream-close threads have finished."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        close_threads = [
            t for t in threading.enumerate()
            if t.is_alive() and t.name == "agy-stream-close"
        ]
        if not close_threads:
            return
        time.sleep(0.02)


class NonBlockingCloseTests(_PinnedSeamsMixin, unittest.TestCase):
    """close() must not block a running asyncio event loop."""

    def test_close_inside_loop_returns_fast_with_slow_termination(self):
        """(a) close() called inside a coroutine with a ~1 s termination
        returns in < 0.2 s, and the deferred thread completes the
        termination and lock release afterwards."""
        client = self._client()
        proc = _mock_proc(_turn_lines(), terminate_delay=0.3, wait_delay=0.7)
        with patch("subprocess.Popen", return_value=proc):
            stream = client.chat.completions.create(
                model=MODEL, messages=MESSAGES, stream=True,
            )
            # Consume the first chunk so the stream has started.
            first = next(stream)
            self.assertEqual(first.choices[0].delta.content, "answer one")

        # The stream is a worker stream with the lock held.
        self.assertTrue(stream.is_worker)
        self.assertTrue(stream._worker_lock_held)
        self.assertTrue(client._worker_lock.locked())

        elapsed = None

        async def _close_in_loop() -> float:
            t0 = time.monotonic()
            stream.close()
            return time.monotonic() - t0

        elapsed = asyncio.run(_close_in_loop())

        # close() returned quickly despite the 1 s of blocking work.
        self.assertLess(elapsed, 0.2, f"close() took {elapsed:.3f}s inside the loop")
        # The stream is marked closed immediately.
        self.assertTrue(stream._closed)

        # Wait for the deferred thread to complete.
        _wait_for_close_threads(timeout=5.0)

        # After the deferred thread finishes:
        # - The worker was terminated (terminate was called).
        self.assertTrue(proc.terminate.called, "worker was not terminated by deferred thread")
        # - The worker lock was released.
        self.assertFalse(
            client._worker_lock.locked(),
            "worker lock was not released by deferred thread",
        )
        # - The worker proc was cleared.
        self.assertIsNone(client._worker_proc)

    def test_close_outside_loop_is_synchronous(self):
        """(b) close() outside a loop terminates synchronously before
        returning."""
        client = self._client()
        proc = _mock_proc(_turn_lines(), terminate_delay=0.05, wait_delay=0.05)
        with patch("subprocess.Popen", return_value=proc):
            stream = client.chat.completions.create(
                model=MODEL, messages=MESSAGES, stream=True,
            )
            next(stream)

        self.assertTrue(stream.is_worker)
        self.assertTrue(stream._worker_lock_held)

        # No running loop here: close() should run synchronously.
        stream.close()

        # Everything happened before close() returned:
        self.assertTrue(stream._closed)
        self.assertTrue(proc.terminate.called)
        self.assertFalse(client._worker_lock.locked())
        self.assertIsNone(client._worker_proc)

    def test_close_idempotent_inside_loop(self):
        """(c) A second close() inside the loop does not launch another
        thread and does not raise."""
        client = self._client()
        proc = _mock_proc(_turn_lines(), terminate_delay=0.1, wait_delay=0.1)
        with patch("subprocess.Popen", return_value=proc):
            stream = client.chat.completions.create(
                model=MODEL, messages=MESSAGES, stream=True,
            )
            next(stream)

        close_threads_before: list[str] = []

        async def _double_close() -> None:
            stream.close()
            # Record how many close threads exist after the first close.
            close_threads_before.extend(
                t.name for t in threading.enumerate()
                if t.is_alive() and t.name == "agy-stream-close"
            )
            # Second close: should be a no-op.
            stream.close()

        asyncio.run(_double_close())

        # Still closed.
        self.assertTrue(stream._closed)

        # At most one close thread was ever started (from the first call).
        close_threads_after = [
            t for t in threading.enumerate()
            if t.is_alive() and t.name == "agy-stream-close"
        ]
        # The second close() did not spawn an additional thread.
        self.assertLessEqual(
            len(close_threads_after),
            len(close_threads_before),
            "second close() launched an extra deferred thread",
        )

        _wait_for_close_threads(timeout=3.0)

    def test_deferred_close_does_not_kill_replaced_worker(self):
        """(d) When the worker is replaced between close() dispatch and
        the deferred thread's execution, the deferred thread does not kill
        the new worker."""
        client = self._client()
        old_proc = _mock_proc(_turn_lines(), terminate_delay=0.0, wait_delay=0.0)
        new_proc = MagicMock()
        new_proc.poll.return_value = None  # alive

        with patch("subprocess.Popen", return_value=old_proc):
            stream = client.chat.completions.create(
                model=MODEL, messages=MESSAGES, stream=True,
            )
            next(stream)

        # Simulate the worker being replaced BEFORE the deferred close runs.
        # Replace the client's worker proc so the identity check fails.
        client._worker_proc = new_proc

        barrier = threading.Event()
        original_close_blocking = stream._close_blocking

        def _delayed_close_blocking() -> None:
            # Wait for the test to replace the worker before running.
            barrier.wait(timeout=3.0)
            original_close_blocking()

        stream._close_blocking = _delayed_close_blocking

        async def _run() -> None:
            stream.close()

        asyncio.run(_run())

        # Signal the deferred thread to proceed.
        barrier.set()

        _wait_for_close_threads(timeout=3.0)

        # The new worker must NOT have been terminated.
        new_proc.terminate.assert_not_called()
        # The new worker is still the client's worker.
        self.assertIs(client._worker_proc, new_proc)

    def test_oneshot_close_inside_loop_returns_fast(self):
        """Oneshot streams also benefit from non-blocking close inside a loop."""
        client = self._client()
        proc = _mock_proc(_turn_lines(), alive=False, terminate_delay=0.3, wait_delay=0.5)

        # Force oneshot by holding the worker lock.
        self.assertTrue(client._worker_lock.acquire(blocking=False))
        try:
            with patch("subprocess.Popen", return_value=proc):
                stream = client.chat.completions.create(
                    model=MODEL, messages=MESSAGES, stream=True,
                )
        finally:
            client._worker_lock.release()

        self.assertFalse(stream.is_worker)

        async def _close_in_loop() -> float:
            t0 = time.monotonic()
            stream.close()
            return time.monotonic() - t0

        elapsed = asyncio.run(_close_in_loop())
        self.assertLess(elapsed, 0.2, f"oneshot close() took {elapsed:.3f}s inside the loop")
        self.assertTrue(stream._closed)

        _wait_for_close_threads(timeout=5.0)
        self.assertTrue(proc.terminate.called)


if __name__ == "__main__":
    unittest.main()
