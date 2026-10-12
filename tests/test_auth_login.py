"""`hermes auth add` signs in through agy print mode and handles accounts that cannot be used yet."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import pytest  # noqa: E402

from accounts import add_account, list_accounts  # noqa: E402
from __init__ import _LOGIN_PROMPT, _login_command, auth_handler  # noqa: E402
from test_multi_account import create_mock_token_file  # noqa: E402


@pytest.fixture
def tty():
    with patch("sys.stdin.isatty", return_value=True):
        yield


def _fake_login(email="new@example.com"):
    calls = []

    def run(cmd, env=None, **kwargs):
        calls.append((cmd, kwargs))
        create_mock_token_file(Path(env["HOME"]), email=email)
        return SimpleNamespace(returncode=0)

    return run, calls


def test_login_command_is_print_mode_not_the_interactive_tui():
    cmd = _login_command("/bin/agy")
    assert cmd[:3] == ["/bin/agy", "-p", _LOGIN_PROMPT]
    assert "--print-timeout" in cmd


def test_add_signs_in_with_print_mode_in_a_private_cwd(tty, capsys):
    run, calls = _fake_login()
    with patch("subprocess.run", side_effect=run), patch("__init__.resolve_agy_command", return_value="agy"), \
            patch("__init__.probe_account_state", return_value=("ok", None)):
        assert auth_handler("add", SimpleNamespace(label="")) is True
    cmd, kwargs = calls[0]
    assert cmd[1] == "-p"
    assert kwargs["cwd"] and "hermes-agy-login-" in kwargs["cwd"]
    assert [a["email"] for a in list_accounts()] == ["new@example.com"]
    assert "do not need to type /exit" in capsys.readouterr().out


def test_not_eligible_account_is_registered_skipped_and_explained(tty, capsys):
    run, _ = _fake_login("blocked@example.com")
    with patch("subprocess.run", side_effect=run), patch("__init__.resolve_agy_command", return_value="agy"), \
            patch("__init__.probe_account_state", return_value=("not_eligible", None)):
        auth_handler("add", SimpleNamespace(label=""))
    acc = list_accounts()[0]
    assert acc["eligible"] is False
    out = capsys.readouterr().out
    assert "NOT eligible" in out and "browser" in out and "refresh" in out


def test_signed_out_after_login_is_registered_with_a_retry_hint(tty, capsys):
    run, _ = _fake_login("out@example.com")
    with patch("subprocess.run", side_effect=run), patch("__init__.resolve_agy_command", return_value="agy"), \
            patch("__init__.probe_account_state", return_value=("needs_login", None)):
        auth_handler("add", SimpleNamespace(label=""))
    out = capsys.readouterr().out
    assert "login did not complete" in out and "skipped by rotation" in out
    assert len(list_accounts()) == 1


def test_unverifiable_account_is_kept_and_flagged(tty, capsys):
    run, _ = _fake_login("slow@example.com")
    with patch("subprocess.run", side_effect=run), patch("__init__.resolve_agy_command", return_value="agy"), \
            patch("__init__.probe_account_state", return_value=("unknown", None)):
        auth_handler("add", SimpleNamespace(label=""))
    assert "could not verify" in capsys.readouterr().out
    assert len(list_accounts()) == 1


def test_closed_browser_registers_nothing_and_leaves_no_staging_dir(tty, tmp_path):
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=1)), \
            patch("__init__.resolve_agy_command", return_value="agy"):
        with pytest.raises(SystemExit, match="Authentication failed"):
            auth_handler("add", SimpleNamespace(label=""))
    assert list_accounts() == []
    accounts_dir = tmp_path / "accounts"
    assert not accounts_dir.exists() or not any(accounts_dir.iterdir())


def test_refresh_reprobes_and_restores_an_account_that_was_fixed(capsys):
    add_account("fixed@example.com", "/h/fixed", eligible=False, email="fixed@example.com")
    with patch("__init__.probe_account_state", return_value=("ok", None)):
        auth_handler("refresh", SimpleNamespace(target="fixed@example.com", label=None))
    assert list_accounts()[0]["eligible"] is True
    assert "state: ok" in capsys.readouterr().out


def test_refresh_names_what_each_unusable_account_needs(capsys):
    add_account("a@example.com", "/h/a", email="a@example.com")
    add_account("b@example.com", "/h/b", email="b@example.com")
    states = {"/h/a": "needs_login", "/h/b": "not_eligible"}
    with patch("__init__.probe_account_state", side_effect=lambda home, *a, **k: (states[home], None)):
        auth_handler("refresh", SimpleNamespace(target=None, label=None))
    out = capsys.readouterr().out
    assert "a@example.com: signed out" in out and "--label a@example.com" in out
    assert "b@example.com: not eligible" in out


SAMPLE_INELIGIBLE = (
    "error: Eligibility check failed: Your current account is not eligible for Antigravity. "
    "Verify your account to continue.\n\nAlternatively, try signing in with another personal Google account.\n\n"
    "Please verify your account in your browser to continue:\n"
    "https://accounts.google.com/signin/continue?sarp=1&scc=1&continue=https%3A%2F%2Fexample.test%2Fx\n"
)


def test_extract_verify_url_reads_the_link_agy_prints():
    from accounts import extract_verify_url

    url = extract_verify_url(SAMPLE_INELIGIBLE)
    assert url == "https://accounts.google.com/signin/continue?sarp=1&scc=1&continue=https%3A%2F%2Fexample.test%2Fx"
    assert extract_verify_url("error: not eligible, no link here") is None


def _probe_with(output, tmp_path):
    from accounts import get_verify_url, probe_account_state

    res = SimpleNamespace(stdout="", stderr=output, returncode=1)
    with patch("subprocess.run", return_value=res), patch("accounts.resolve_agy_command", return_value="agy"):
        state, _ = probe_account_state(tmp_path)
    return state, get_verify_url(tmp_path)


def test_probe_keeps_the_link_for_not_eligible_and_drops_it_after_a_fix(tmp_path):
    state, url = _probe_with(SAMPLE_INELIGIBLE, tmp_path)
    assert state == "not_eligible" and url and url.startswith("https://accounts.google.com/")
    ok = SimpleNamespace(stdout="{}", stderr="", returncode=0)
    with patch("subprocess.run", return_value=ok), patch("accounts.resolve_agy_command", return_value="agy"):
        from accounts import get_verify_url, probe_account_state

        probe_account_state(tmp_path)
        assert get_verify_url(tmp_path) is None


def test_add_prints_the_verification_link_for_a_not_eligible_account(tty, capsys):
    run, _ = _fake_login("blocked@example.com")

    def probe(home, *a, **k):
        import accounts

        accounts._VERIFY_URLS[str(Path(home).resolve())] = "https://accounts.google.com/signin/continue?x=1"
        return "not_eligible", None

    with patch("subprocess.run", side_effect=run), patch("__init__.resolve_agy_command", return_value="agy"), \
            patch("__init__.probe_account_state", side_effect=probe):
        auth_handler("add", SimpleNamespace(label=""))
    out = capsys.readouterr().out
    assert "sign in as blocked@example.com" in out
    assert "https://accounts.google.com/signin/continue?x=1" in out


def test_refresh_prints_a_link_per_not_eligible_account(capsys):
    import accounts

    add_account("b@example.com", "/h/b", email="b@example.com")
    accounts._VERIFY_URLS[str(Path("/h/b").resolve())] = "https://accounts.google.com/signin/continue?b=1"
    with patch("__init__.probe_account_state", return_value=("not_eligible", None)):
        auth_handler("refresh", SimpleNamespace(target=None, label=None))
    assert "https://accounts.google.com/signin/continue?b=1" in capsys.readouterr().out
