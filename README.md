# Antigravity Subscription DirectSDK Provider for Hermes Agent

[![Hermes Agent Plugin](https://img.shields.io/badge/Hermes%20Agent-Plugin-purple.svg)](https://hermes-agent.nousresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tested with Hermes](https://img.shields.io/badge/Hermes-v0.21%2B-green.svg)](https://github.com/nousresearch/hermes-agent)

Hermes Agent model provider that routes inference through an active Google Antigravity account using the local `agy` binary.

This plugin lets Hermes use Gemini and Claude models through your existing Antigravity quota.

> **Not a fork, a derivative.** This project started from [soyelmismo/hermes-antigravity-subscription](https://github.com/soyelmismo/hermes-antigravity-subscription) (streaming client, process isolation, keyring probes) and takes the native tool bridge and SOUL.md delivery from [neerazz/hermes-antigravity-oauth](https://github.com/neerazz/hermes-antigravity-oauth). Both are MIT. It has diverged enough to stand alone. What it adds:
>
> | Feature | What it does | Origin |
> | :--- | :--- | :--- |
> | Multi-account rotation | Register several Google accounts, each in its own HOME. `quota`, `round_robin` or `off` modes, cooldown on 429, skip of ineligible or logged-out accounts. | this repo |
> | Quota in `/usage` | Per-account 5-hour and weekly remaining quota from `agy -p "/usage"`, which spends no tokens. | this repo |
> | `hermes antigravity ...` CLI | `list`, `run`, `usage`, `use`, `add`, `remove`, `mode`, `ignite` (companion plugin in [`companion/`](companion)). | this repo |
> | Native tool bridge | A model that calls agy's built-in tools gets them re-issued as Hermes tools instead of losing the turn. | neerazz, adapted |
> | SOUL.md as rules | Your Hermes persona is delivered to agy as workspace `GEMINI.md`, written only when a session starts so rotation and resume stay intact. | neerazz, adapted |
> | Core provider | Token streaming, tool-call parsing, worker reuse, model discovery. | soyelmismo |

---

## How It Works

- **Token streaming**: Reads stdout chunks from `agy --output-format stream-json` and yields standard completion deltas.
- **Tool call routing**: The model emits `<tool_call>` tags in text. The plugin parses these into OpenAI function call deltas, so Hermes executes tools on the host instead of `agy`.
- **Tool execution guard**: `agy` registers local system tools by default. This plugin runs `agy` in headless mode without permissions skip flags. If `agy` attempts to execute an internal tool step, the stream closes and terminates the child process; steps that map to a Hermes tool are re-issued as that tool call (see *Native tool bridge*).
- **Thinking effort mapping**: Maps Hermes reasoning effort settings (`low`, `medium`, `high`) to backend model variants (`gemini-3.8-flash-low`, `gemini-3.8-flash-high`).
- **Subagent concurrency**: Each completion turn runs in its own process group (`start_new_session=True`). Multiple Hermes subagents can request completions concurrently without shared state.
- **Filesystem isolation**: Subprocesses run in an isolated, private temporary working directory per client and slash commands are disabled. Project `AGENTS.md` files are not read. The only `GEMINI.md` agy sees is the one this plugin writes into that directory (see *Native tool bridge and workspace rules*).
- **Tool output pruning**: Old tool results (diffs, directory trees, test output) dominate wire payload in long sessions. The prompt builder keeps the last 8 tool results intact and caps older tool outputs at 300 characters. A 20k character safety limit with head/tail preservation guards individual tool calls. In a 336-message session, this cut wire payload from 600 KB to 180 KB (70% reduction).
- **Error classification and recovery**: `classify_api_error` detects `agy` PubSub stalls (`subscriber fell behind updates, stalled for 5s`), trajectory limits, and empty SUCCESS status (returning `context_overflow` with `should_compress: True`). It classifies credit balance exhaustion and billing caps as `billing` for immediate failover, and maps account verification and Terms of Service notices to `auth_permanent` to surface resolution links without retry loops.

---

## Models

| Model | Suffix | LLM Context | Plugin Declared | Supported Efforts |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3.8-flash` | `-low`, `-medium`, `-high` | 1M tokens | 239k tokens | `low`, `medium`, `high` |
| `gemini-3.7-flash` | `-low`, `-medium`, `-high` | 1M tokens | 239k tokens | `low`, `medium`, `high` |
| `gemini-3.6-flash` | `-low`, `-medium`, `-high` | 1M tokens | 239k tokens | `low`, `medium`, `high` |
| `gemini-3.1-pro` | `-low`, `-high` | 2M tokens | 239k tokens | `low`, `high` |
| `gpt-oss-120b` | `-medium` only | 128k tokens | 128k tokens | `medium` |
| `claude-sonnet-4-6` | None | 200k tokens | 239k tokens | None (agy rejects `--effort`) |
| `claude-opus-4-6-thinking` | None | 200k tokens | 239k tokens | None (agy rejects `--effort`) |

**Dynamic discovery per account.** Available models and supported reasoning efforts are discovered dynamically from `agy models`. The picker reflects the exact models available on the user's account and subscription tier (including preview models or custom models configured in `agy` settings). A model that agy lists by bare name takes no `--effort`. A requested effort that the model lacks maps to the nearest supported one, the stronger on a tie (`medium` on `gemini-3.1-pro` becomes `high`). The result is cached for one hour; `models.py` provides an emergency fallback when `agy models` fails. Hermes' own `xhigh` and `max` map to `high`.

> **Claude Model Aliases**: Requests for `claude-sonnet-5-5` or `claude-opus-5-5` resolve dynamically if present in the user's `agy models` catalog; if the account lacks them, they gracefully fall back to `claude-sonnet-4-6` and `claude-opus-4-6-thinking`.

> **Trajectory Persistence & Context Length**: Multi-turn sessions persist across worker restarts via `agy --conversation <id>`, sending only incremental deltas on each turn. This bypasses `agy`'s single-turn 100,000-token prompt trimmer. The plugin declares 239,000 tokens to Hermes (configurable via `HERMES_ANTIGRAVITY_CONTEXT_LENGTH` or `ANTIGRAVITY_CONTEXT_LENGTH`), ensuring Hermes auto-compaction triggers safely below `agy`'s internal checkpoint threshold (239,616 tokens / 256k - 16k output tokens) and preserves sovereign context control. Conversation databases are strictly isolated inside the worker's private temp directory and automatically purged on session resets (`/new`).

### Model Compatibility

`agy` embeds its own system prompt ("you are Antigravity, you have `run_command`, `view_file`, ...") before the plugin's preamble. Some models obey `agy`'s prompt over the plugin's `<tool_call>` protocol and invoke native tools, which headless mode soft-denies. The plugin detects this and terminates the stream, but the turn is lost.

| Model | `<tool_call>` Protocol | Notes |
| :--- | :--- | :--- |
| `gemini-3.8-flash` | Follows | Tested with full toolset (~40 tools) |
| `gemini-3.7-flash` | Follows | Tested with full toolset |
| `gemini-3.6-flash` | Ignores | Goes native even with a single tool schema |
| `gemini-3.1-pro` | Follows | Tested with full toolset |
| `claude-opus-4-6-thinking` | Follows | Tested with full toolset |
| `claude-sonnet-4-6` | Partial | Works with moderate toolsets (tested up to 10 tools); refuses or flips to native with large toolsets (~40 tools) |
| `gpt-oss-120b` | Follows | Tested with full toolset |

Models that ignore the protocol attempt `agy`'s native `RunCommand`/`WriteToFile` steps. The plugin neutralizes these (stream watcher kills the process), but every tool-using turn fails. This is an upstream `agy` limitation: the persona prompt is embedded in the closed binary with no configuration surface to suppress it.

---

## Environment passed to `agy`

Hermes often runs with gateway and dashboard secrets in its environment (`SLACK_BOT_TOKEN`, `TELEGRAM_BOT_TOKEN`, `*_API_KEY`, ...). `agy` is a closed-source binary and does not need any of them, so by default the child environment is the parent environment **minus credential-looking variables**: names containing `TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, `ACCESS_KEY`, `PRIVATE_KEY`, `CREDENTIAL(S)`, `AUTH`, `WEBHOOK`, `DSN` or `COOKIE`, and database/broker URLs (`DATABASE_URL`, `REDIS_URL`, `MONGO_URI`, ...). `SSH_AUTH_SOCK`, proxy and certificate variables are kept.

This default is a **name heuristic**: it cannot know what an arbitrarily named variable holds. For a guarantee of exactly what `agy` sees, use strict mode.

| Variable | Effect |
| :--- | :--- |
| `ANTIGRAVITY_ENV_PASSTHROUGH` | Comma-separated names to keep even if they look like credentials. |
| `ANTIGRAVITY_ENV_STRICT` | `1`, `true`, `yes` or `on`: pass only the baseline below plus `ANTIGRAVITY_ENV_ALLOWLIST` and `ANTIGRAVITY_ENV_PASSTHROUGH`. |
| `ANTIGRAVITY_ENV_ALLOWLIST` | Comma-separated names to add in strict mode. |

Strict baseline: `PATH`, `LANG`, `LANGUAGE`, `LC_*`, `TZ`, `TERM`, `TMPDIR`/`TEMP`/`TMP`, `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` (and their lowercase spellings on POSIX), `SSL_CERT_FILE`, `SSL_CERT_DIR`, `SSH_AUTH_SOCK`, `DBUS_SESSION_BUS_ADDRESS` and `XDG_RUNTIME_DIR` (Linux keyring sign-in), `SYSTEMROOT`, `WINDIR`, `COMSPEC`, `PATHEXT` (Windows), `AGY_CLI_DISABLE_AUTO_UPDATE` and `AGY_CLI_MODEL_API_MAX_RETRIES`. Names are matched case-insensitively on Windows.

---

## Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                        Hermes Agent                         │
│  - Prompts and system instructions                          │
│  - Tool definitions and approvals                           │
│  - Host execution                                           │
└──────────────┬───────────────────────────────▲──────────────┘
               │ Prompt + tool schemas         │ Text with <tool_call> tags
               │                               │ parsed into completion chunks
┌──────────────▼───────────────────────────────┴──────────────┐
│       Antigravity Subscription DirectSDK Plugin             │
│                                                             │
│  1. Preamble tells the model to emit tool tags in text.     │
│  2. Omits --dangerously-skip-permissions to deny agy tools. │
│  3. Stream watcher kills agy if it emits a tool step.       │
│  4. Runs in private temp dir to ignore rules files.         │
└──────────────────────────────┬──────────────────────────────┘
                               │ Stdin / stdout (NDJSON pipes)
┌──────────────────────────────▼──────────────────────────────┐
│               Local `agy` CLI binary (Go)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Requirements

1. **Antigravity CLI**:
   `agy` must be installed and authenticated on the system.
   ```bash
   agy --version
   ```
2. **Hermes Agent**:
   Version `0.21.0` or higher.
   ```bash
   hermes --version
   ```

> **Optional: Linux keyring detection.**
> On Linux with a D-Bus session, `agy` keeps its session in the freedesktop
> Secret Service instead of a token file (credentials `service`/`gemini` +
> `username`/`antigravity`, the go-keyring convention). Detection uses
> `secret-tool`, falling back to `python3` with `secretstorage` when the CLI is
> absent or cannot answer (missing, spawn failure, timeout or non-zero exit).
> `secret-tool` prints the credential on its own stdout, so the plugin closes
> that stream and reads only the attribute lines on stderr; the scripted
> fallback reads attributes and labels and never asks for the secret. Either way
> the plugin never receives the stored secret. Installing `secret-tool` is not
> strictly neutral: the CLI path answers on the exact attribute pair, while the
> scripted path additionally matches a credential whose go-keyring label embeds
> "antigravity". Headless systems without a D-Bus session keep using the token-file
> path, so nothing extra is required there.
> ```bash
> command -v secret-tool   # optional: faster than the python3 fallback
> ```
> Install it with `libsecret-tools` (Debian/Ubuntu) or `libsecret` (Fedora/Arch).

> **SSH session variables.**
> `agy` switches to file-based token storage when it detects `SSH_CONNECTION`,
> `SSH_CLIENT`, or `SSH_TTY`. When no token file exists (credential lives in the
> OS keyring), the plugin strips these variables from the child environment so
> `agy` uses keyring authentication instead of failing. `SSH_AUTH_SOCK` is
> preserved. When a token file exists, the environment is unchanged.

> **macOS keychain.** On macOS `agy` keeps its session in the login keychain
> (a generic password with service `gemini` and account `antigravity`).
> Detection runs `/usr/bin/security find-generic-password` without `-g` or
> `-w`, which prints only the item's attributes, so the plugin never receives
> the secret and no keychain prompt appears. The keychain is found through
> `$HOME/Library/Keychains`, so the plugin links that directory into the
> isolated HOME it gives `agy`; otherwise the child `agy` would ask you to log
> in again. Nothing extra is required. Setting `ANTIGRAVITY_CONFIG_DIR` turns
> both off and keeps the token-file path.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HERMES_ANTIGRAVITY_CONTEXT_LENGTH` | `239000` | Override the declared context window (tokens). Also checks `ANTIGRAVITY_CONTEXT_LENGTH`. Hermes triggers auto-compression based on this value (e.g. at 75% threshold, ~179k tokens). |
| `ANTIGRAVITY_COMMAND` | `agy` | Path to the `agy` binary. Also checks `AGY_CLI_PATH` and `ANTIGRAVITY_CLI_PATH`. |
| `ANTIGRAVITY_ARGS` | (none) | Extra arguments to pass to the `agy` subprocess. |
| `ANTIGRAVITY_CONFIG_DIR` | (none) | Override the config directory, bypassing keyring and token-file detection. |
| `ANTIGRAVITY_ROTATION` | `off` | Multi-account rotation mode: `off` (default), `quota`, or `round_robin`. |
| `ANTIGRAVITY_ACCOUNTS_DIR` | `~/.agy-accounts` | Base directory for multi-account isolated HOME environments. |
| `ANTIGRAVITY_ACCOUNTS_FILE` | `~/.hermes/antigravity-accounts.json` | Path to the multi-account registry file. |
| `ANTIGRAVITY_DEBUG_PROMPT_DIR` | (none) | Directory to dump outgoing prompts sent to `agy` for offline diagnosis. |
| `ANTIGRAVITY_WORKER_IDLE_SECONDS` | `900` | Terminate the persistent `agy` worker after this many seconds without a finished turn. The next turn starts a fresh worker with the full prompt. `0` or a negative value disables the bound. |
| `ANTIGRAVITY_PROXY` | (none) | Proxy URL (e.g. `http://127.0.0.1:8080` or `socks5h://127.0.0.1:1080`) agy's subprocess should use. Sets `HTTP(S)_PROXY` and `ALL_PROXY` (both casings) and adds loopback to `NO_PROXY`. Unset means agy inherits the parent process's proxy environment unchanged. |
| `ANTIGRAVITY_ENV_PASSTHROUGH` | (none) | Comma-separated variable names to preserve in the child environment even if they match secret heuristics. |
| `ANTIGRAVITY_ENV_STRICT` | `false` | When enabled (`1`, `true`), only passes variables from `ANTIGRAVITY_ENV_ALLOWLIST` plus a minimal runtime baseline. |
| `ANTIGRAVITY_ENV_ALLOWLIST` | (none) | Additional variables to pass through when `ANTIGRAVITY_ENV_STRICT` is active. |

---

## Installation

Install through the Hermes plugin manager:

```bash
hermes plugins install https://github.com/himanusia/hermes-antigravity-multiaccount
hermes plugins enable antigravity-subscription-directsdk
```

Or clone into the local plugins folder:

```bash
git clone https://github.com/himanusia/hermes-antigravity-multiaccount.git \
  ~/.hermes/plugins/antigravity-subscription-directsdk

hermes plugins enable antigravity-subscription-directsdk
```

Optional CLI companion (`hermes antigravity ...`). It is a second plugin because Hermes never calls `register(ctx)` on `kind: model-provider` plugins:

```bash
cp -r ~/.hermes/plugins/antigravity-subscription-directsdk/companion ~/.hermes/plugins/antigravity-companion
hermes plugins enable antigravity-companion
```

> **Directory naming tip**: Hermes Agent discovers plugins by scanning directories in `~/.hermes/plugins/` alphabetically. If you keep backup copies (such as `antigravity-subscription-directsdk.backup`), place them outside `~/.hermes/plugins/` so a later-sorting directory does not override the active plugin.

---

## Usage

### Interactive model picker

Open the model picker inside Hermes:

```text
/model
```

Select **Antigravity Subscription DirectSDK**, then pick a model and effort level.

### Command line flag

```bash
hermes --provider antigravity-subscription-directsdk --model gemini-3.8-flash
```

### Configuration file

Add to `~/.hermes/config.yaml`:

```yaml
agent:
  provider: antigravity-subscription-directsdk
  model: gemini-3.8-flash
  reasoning_effort: high
```

### Subscription Quota & Usage Tracking

The provider profile implements `fetch_account_usage()`, exposing subscription quota limits to Hermes `/usage`, the TUI status bar, and the desktop app via `AccountUsageSnapshot`.

Note: Custom plugin tools and slash commands are not available for `kind: model-provider` plugins because Hermes core skips calling `register(ctx)` for model providers (`hermes_cli/plugins_discovery.py:286`, `hermes_cli/plugin_validate.py:253-258`). All quota visibility is provided through the native `fetch_account_usage()` interface. The **companion plugin** (`antigravity-companion`, `kind: standalone`) provides the `hermes antigravity ...` CLI instead.

Quota queries execute `agy -p "/usage" --output-format json` under an isolated HOME environment without consuming any model tokens or inference turns. Results are cached thread-safely for 60 seconds.

### Multi-Account & Quota Rotation

When you have multiple Google accounts with Antigravity / Gemini subscriptions, you can register them and rotate between them automatically.

```bash
# Add an account (label is optional; if omitted, defaults to the account email from id_token)
hermes auth add antigravity-subscription-directsdk

# Or specify an explicit label
hermes auth add antigravity-subscription-directsdk --label work

# Check status and remaining quotas across accounts (includes host default account)
hermes auth status antigravity-subscription-directsdk

# Refresh quota and clear cooldowns
hermes auth refresh antigravity-subscription-directsdk

# Remove an account from registry (preserves credentials and directory)
hermes auth logout antigravity-subscription-directsdk work
```

#### Account Setup & Authentication

- **Optional Label & Safe Directory Names**: `--label` is optional. When omitted, the plugin extracts the email address directly from the JWT `id_token` payload in the newly saved token file. The home directory name is sanitized for cross-platform compatibility (`[A-Za-z0-9._-]`), while the saved label retains the full email address.
- **No manual TUI.** `hermes auth add` runs `agy -p "Reply with exactly: OK" --print-timeout 9m` in the new account's HOME. With no saved session, agy starts its Google sign-in first (browser, or paste the code back over SSH), answers, and exits by itself. Nobody has to type `/exit`. The login turn costs one tiny prompt. (Approach from neerazz/hermes-antigravity-oauth.)
- **Accounts that cannot be used yet.** After sign-in the plugin probes `agy -p "/usage"` (free). A signed-out or not-eligible account is still registered, marked, and skipped by rotation, and the command prints what fixes it. An account agy could not verify is kept and flagged. `hermes auth refresh` re-probes every account, so one that was fixed in the browser comes back without re-adding. For a not-eligible account, `hermes auth add|status|refresh` and `hermes antigravity list|usage` print the Google verification link that agy returns, to open while signed in as that account. The link is minted per check, kept in memory and never saved to the registry.
- **Token File-based Authentication**: Success is determined by the presence of the OAuth token file (`antigravity-oauth-token` or `jetski-standalone-oauth-token`), rather than the process exit code.
- **macOS Keychain Dialog**: A registered account's HOME has no keychain (agy's keychain item has a fixed name, so a shared keychain would let accounts overwrite each other). The plugin runs agy for registered accounts with `SSH_CONNECTION` set, which makes agy keep the token in its file and skip the keychain, so token refreshes no longer pop *"a keychain cannot be found to store 'antigravity'"*. The interactive sign-in (`hermes auth add`) can still show it once; click **Cancel**.
- **Automatic Eligibility Check**: Upon sign-in, the plugin validates subscription eligibility via `agy -p /usage --output-format json`. Ineligible accounts are saved with `eligible: false` and are automatically bypassed during quota rotation.

#### Rotation Modes

Set per session with `ANTIGRAVITY_ROTATION`, or persistently in the registry with `hermes antigravity mode <mode>` (the env var wins for the session):

- `off` (default): always use the host default account or the single active account.
- `quota`: pick the healthiest account per request with base score `f_5h * f_weekly^2`, scaled by replenishment urgency (a window about to reset ranks higher) and divided by `(1 + active leases)`. Ineligible accounts and accounts at 0% in either the 5-hour or weekly window are hard-gated (score = 0.0).
- `round_robin`: cycle through available, eligible accounts sequentially.
- `fixed`: always use the pinned account, set with `hermes antigravity use <label>` or `ANTIGRAVITY_ACCOUNT=<label>` for one session. Fails open to the host default when the pinned account is unavailable.

Session stickiness keeps one account per Hermes session so the prompt prefix cache stays warm (`ANTIGRAVITY_SESSION_STICKINESS=0` disables it). Quota ignition (`hermes antigravity ignite on`) optionally wakes an idle 5-hour window on the selected account.

Model access is per Google account (one account may list Claude 5.5 while another only has 4.6), so rotation only picks accounts whose `agy models` listing has the requested model; with none left the request runs on the host default account.

The account serving requests is recorded in the registry (`serving`), and `/usage` meters that account: the snapshot title ends with the account label (`default` for the host login).

If an account encounters a quota error (HTTP 429 / resource exhausted), it is placed on a cooldown until the window resets and Hermes automatically fails over to the next best available account — including while rotation is `off`, so a depleted account swaps accounts instead of degrading to another model. Transient rate limiting alone does not trigger a swap.

Account credentials are stored under `~/.agy-accounts/<label>/` and the registry file is stored at `~/.hermes/antigravity-accounts.json` (configurable via `ANTIGRAVITY_ACCOUNTS_DIR` and `ANTIGRAVITY_ACCOUNTS_FILE`).

### Native tool bridge and workspace rules

- **Native tool bridge.** When a model ignores the `<tool_call>` protocol and calls an agy built-in, the plugin no longer drops the turn. It kills the agy step and re-issues it as the matching Hermes tool, so Hermes' approvals and sandbox run it: `run_command`→`terminal`, `view_file`→`read_file`, `write_to_file`→`write_file`, `grep_search`/`find_by_name`→`search_files`, `search_web`→`web_search`, `read_url_content`→`web_extract`. It only maps to tools present in the request; any other native step is still neutralized. (`native_tools.py`, idea from neerazz/hermes-antigravity-oauth, MIT.)
- **SOUL.md as workspace rules.** When a fresh agy session starts, the plugin writes `$HERMES_HOME/SOUL.md` (if Hermes left it out), the Hermes system prompt and the tool contract to `GEMINI.md` in agy's workspace, which agy loads as first-class rules. The stdin prompt then carries only tools and conversation. Rules are written before the spawn and never mid-session. Set `ANTIGRAVITY_WORKSPACE_RULES=0` to inline everything in the prompt again. One-shot calls always inline. (`soul.py`)

---

## Tests

Run the test suite:

```bash
PYTHONPATH=/usr/local/lib/hermes-agent:. pytest
```

---

## License

MIT. Copyright (c) soyelmismo, neerazz and himanusia. See [LICENSE](LICENSE).
