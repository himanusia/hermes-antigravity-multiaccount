"""Tests for the idle bound on the persistent agy worker.

Hermes retires an evicted agent's client without calling ``close()``, so the
persistent worker of a finished conversation used to stay alive for the
lifetime of the gateway. The client now terminates a worker that has not
finished a turn for ``ANTIGRAVITY_WORKER_IDLE_SECONDS`` (default 900).
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add plugin parent dir to sys.path
plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from client import (
    _DEFAULT_WORKER_IDLE_SECONDS,
    _worker_idle_seconds,
    AntigravityClient,
)

TURN_MESSAGES = [{"role": "user", "content": "hello agy"}]
IDLE_ENV = "ANTIGRAVITY_WORKER_IDLE_SECONDS"

# Upper bound for a reap that is expected to happen; a regression fails the
# test instead of hanging it.
REAP_DEADLINE_SECONDS = 5.0


class WorkerIdleTests(unittest.TestCase):
    def setUp(self) -> None:
        patchers = (
            patch("client.is_authenticated", return_value=True),
            patch("process.resolve_real_token_path", return_value=None),
            patch("client.resolve_agy_command", return_value="agy"),
            patch("process._link_macos_keychains"),
            patch("client.terminate_process"),
        )
        for patcher in patchers:
            mock = patcher.start()
            self.addCleanup(patcher.stop)
        self.terminate_process = mock
        temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_test_")
        self.addCleanup(temp_dir.cleanup)
        self.client = AntigravityClient(cwd=temp_dir.name)
        self.addCleanup(self.client.close)

    def _install_worker(self) -> MagicMock:
        proc = MagicMock()
        proc.poll.return_value = None
        self.client._worker_proc = proc
        self.client._active_processes.add(proc)
        return proc

    def _wait_for_reap(self) -> None:
        deadline = time.monotonic() + REAP_DEADLINE_SECONDS
        while self.client._worker_proc is not None and time.monotonic() < deadline:
            time.sleep(0.01)

    def test_idle_worker_is_terminated_after_the_bound(self):
        proc = self._install_worker()
        with patch.dict(os.environ, {IDLE_ENV: "0.05"}):
            self.client._update_worker_history(TURN_MESSAGES)
        self._wait_for_reap()
        self.assertIsNone(self.client._worker_proc)
        self.terminate_process.assert_called_once_with(proc)
        # The next turn must take the full-prompt branch on a fresh worker.
        self.assertEqual(self.client._worker_history, [])
        self.assertNotIn(proc, self.client._active_processes)

    def test_worker_with_a_running_turn_is_not_reaped(self):
        proc = self._install_worker()
        self.assertTrue(self.client._worker_lock.acquire(blocking=False))
        try:
            with patch.dict(os.environ, {IDLE_ENV: "0.05"}):
                self.client._update_worker_history(TURN_MESSAGES)
            timer = self.client._worker_idle_timer
            timer.join(REAP_DEADLINE_SECONDS)
            self.assertFalse(timer.is_alive())
            self.assertIs(self.client._worker_proc, proc)
            self.terminate_process.assert_not_called()
        finally:
            self.client._worker_lock.release()

    def test_finished_turn_restarts_the_countdown(self):
        self._install_worker()
        with patch.dict(os.environ, {IDLE_ENV: "60"}):
            self.client._update_worker_history(TURN_MESSAGES)
            first = self.client._worker_idle_timer
            self.client._update_worker_history(TURN_MESSAGES * 2)
            second = self.client._worker_idle_timer
        self.assertIsNot(first, second)
        self.assertTrue(first.finished.is_set(), "the previous countdown was not cancelled")
        self.assertFalse(second.finished.is_set())

    def test_terminating_the_worker_cancels_the_countdown(self):
        self._install_worker()
        with patch.dict(os.environ, {IDLE_ENV: "60"}):
            self.client._update_worker_history(TURN_MESSAGES)
        timer = self.client._worker_idle_timer
        self.client._terminate_worker()
        self.assertIsNone(self.client._worker_idle_timer)
        self.assertTrue(timer.finished.is_set())

    def test_stale_countdown_does_not_terminate_a_replaced_worker(self):
        stale_proc = MagicMock()
        current_proc = self._install_worker()
        self.client._reap_idle_worker(stale_proc)
        self.assertIs(self.client._worker_proc, current_proc)
        self.terminate_process.assert_not_called()
        self.assertFalse(self.client._worker_lock.locked())

    def test_non_positive_bound_disables_the_countdown(self):
        self._install_worker()
        for value in ("0", "-1"):
            with patch.dict(os.environ, {IDLE_ENV: value}):
                self.client._update_worker_history(TURN_MESSAGES)
            self.assertIsNone(self.client._worker_idle_timer)

    def test_no_countdown_without_a_worker(self):
        with patch.dict(os.environ, {IDLE_ENV: "60"}):
            self.client._update_worker_history(TURN_MESSAGES)
        self.assertIsNone(self.client._worker_idle_timer)

    def test_bound_defaults_when_unset_or_invalid(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(IDLE_ENV, None)
            self.assertEqual(_worker_idle_seconds(), _DEFAULT_WORKER_IDLE_SECONDS)
        with patch.dict(os.environ, {IDLE_ENV: "not-a-number"}):
            self.assertEqual(_worker_idle_seconds(), _DEFAULT_WORKER_IDLE_SECONDS)
        with patch.dict(os.environ, {IDLE_ENV: " 120 "}):
            self.assertEqual(_worker_idle_seconds(), 120.0)


if __name__ == "__main__":
    unittest.main()
