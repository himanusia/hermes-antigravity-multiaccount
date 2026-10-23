"""Probes must never be able to open a browser window.

`agy` opens the system browser when it decides a session needs re-login. Every
read-only probe (list, /usage, status bar, desktop, rotation picks) runs
unattended, so a probe that re-authenticates would pop a browser tab the user
never asked for. These tests pin the guard that prevents it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import accounts  # noqa: E402
import usage as usage_module  # noqa: E402


class BrowserBlockTests(unittest.TestCase):
    def test_probe_env_blocks_browser_by_default(self):
        env = accounts.probe_env("/tmp")
        self.assertEqual(env["BROWSER"], "/usr/bin/false")
        shim_dir = env["PATH"].split(os.pathsep)[0]
        shim = os.path.join(shim_dir, "open")
        self.assertTrue(os.path.isfile(shim))
        self.assertTrue(os.access(shim, os.X_OK))
        self.assertEqual(env["HOME"], str(Path("/tmp").resolve()))

    def test_block_can_be_disabled_for_interactive_sign_in(self):
        with patch.dict(os.environ, {"ANTIGRAVITY_BLOCK_BROWSER": "0"}):
            env = accounts.probe_env("/tmp")
        self.assertNotEqual(env.get("BROWSER"), "/usr/bin/false")
        self.assertNotIn("agy-nobrowser", env.get("PATH", ""))

    def test_shim_refuses_to_launch_a_browser(self):
        # The shim is a POSIX `#!/bin/sh` script (BROWSER=/usr/bin/false is POSIX
        # only too), so there is nothing to execute on Windows.
        if os.name == "nt":
            self.skipTest("no-browser shims are POSIX-only")
        env = accounts.probe_env("/tmp")
        shim = os.path.join(env["PATH"].split(os.pathsep)[0], "open")
        res = subprocess.run([shim, "https://accounts.google.com/"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("blocked browser launch", res.stderr)

    def test_usage_probe_passes_the_blocked_env(self):
        with patch.object(usage_module, "resolve_agy_command", return_value="/usr/bin/agy"), patch(
            "usage.subprocess.run"
        ) as run:
            run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            usage_module._query_agy_usage()
        captured = run.call_args.kwargs.get("env") or {}
        self.assertEqual(captured.get("BROWSER"), "/usr/bin/false")
        self.assertIn("agy-nobrowser", captured.get("PATH", ""))


if __name__ == "__main__":
    unittest.main()


class FileTokenStorageTests(unittest.TestCase):
    """Registered-account agy runs keep the token in its file, never the keychain.

    A registered account's HOME has no keychain, so a keychain write on token refresh
    pops macOS's "A keychain cannot be found" dialog. agy skips the keychain when it
    detects an SSH session; the host default account must keep its real keychain.
    """

    def test_probe_env_selects_file_token_storage(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SSH_CONNECTION", None)
            env = accounts.probe_env("/tmp")
        self.assertEqual(env["SSH_CONNECTION"], accounts.FILE_TOKEN_STORAGE_ENV["SSH_CONNECTION"])

    def test_real_ssh_session_is_left_alone(self):
        with patch.dict(os.environ, {"SSH_CONNECTION": "10.0.0.2 5000 10.0.0.1 22"}):
            env = accounts.probe_env("/tmp")
        self.assertEqual(env["SSH_CONNECTION"], "10.0.0.2 5000 10.0.0.1 22")

    def test_client_env_only_for_registered_account_homes(self):
        import tempfile
        from client import AntigravityClient

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SSH_CONNECTION", None)
            client = AntigravityClient(cwd=tmp)
            self.assertIn("SSH_CONNECTION", client._child_env(home_dir=tmp))
            self.assertNotIn("SSH_CONNECTION", client._child_env())
