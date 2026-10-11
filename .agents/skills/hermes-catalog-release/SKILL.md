---
name: hermes-catalog-release
description: Release protocol and upstream catalog PR workflow for antigravity-subscription-directsdk.
---

# Hermes Catalog Release and Upstream PR Protocol

Release workflow for `antigravity-subscription-directsdk` and catalog pin-bump PR process for upstream `NousResearch/hermes-agent`.

---

## 1. Upstream Catalog Admission Rules

Per `hermes-agent/plugin-catalog/README.md` (Rules 4 and 5):
- Catalog entries require a 40-character Git commit SHA pin (`sha:`). The catalog validator rejects branches and tags.
- Maintainers must review and merge a PR to update an existing pin.
- The plugin repository owner or a major contributor must submit the SHA bump.
- Reviewers inspect the commit range (`vOLD..vNEW`) for security and stability.

---

## 2. Release Protocol (Plugin Repository)

Run these steps in `/root/.hermes/plugins/antigravity-subscription-directsdk`:

### Step 1: Update Documentation (`README.md`)
Review code changes since the previous release (`git log vOLD..HEAD`).
Update `README.md` to document:
- New features, flags, or configuration options.
- Model compatibility matrix updates.
- Workarounds, known limits, or directory naming considerations.

### Step 2: Bump Version and Description in `plugin.yaml`
Update `version:` in `plugin.yaml`:
```yaml
name: antigravity-subscription-directsdk
manifest_version: 2
version: X.Y.Z
```
Check if `description:` needs adjustment for new core functionality.

### Step 3: Run the Test Suite
Confirm tests pass before committing:
```bash
PYTHONPATH=/usr/local/lib/hermes-agent:. pytest
```

### Step 4: Commit the Release
Create the release commit on `main`:
```bash
git commit -am "release: vX.Y.Z"
```

### Step 5: Push `main` to `origin/main` Before Tagging
The release workflow (`.github/workflows/release.yml`) checks:
`git merge-base --is-ancestor "$TAG" origin/main`
The release commit must exist on `origin/main` before you push the tag, or CI fails.
```bash
git push origin main
```

### Step 6: Tag and Push Tag
Tag the commit with semantic versioning (`vX.Y.Z`):
```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

### Step 7: Verify the Release Build
Wait for GitHub Actions to complete:
```bash
gh run list --repo soyelmismo/hermes-antigravity-subscription -L 3
gh run watch <run_id> --repo soyelmismo/hermes-antigravity-subscription
```
Verify the published release:
```bash
gh release view vX.Y.Z
```

### Step 8: Record the Commit SHA
```bash
git rev-parse HEAD
# Example: 3d691dadfdd515083237337c4a9747389a18ca4a
```

---

## 3. Upstream PR Protocol (`NousResearch/hermes-agent`)

Run these steps in `/usr/local/lib/hermes-agent`:

### Step 1: Sync `main` with Upstream
```bash
cd /usr/local/lib/hermes-agent
git fetch origin main
git checkout main
git merge --ff-only origin/main
```

### Step 2: Create a Feature Branch
```bash
git checkout -b catalog/bump-antigravity-pin-v<version_slug>
# Example: catalog/bump-antigravity-pin-v106
```

### Step 3: Update Catalog Entry
Edit `plugin-catalog/antigravity-subscription-directsdk.yaml`:
- Set `sha:` to the 40-character commit SHA from Step 2.8.
- Set `version:` to `"X.Y.Z"`.
- Keep `description:` concise (under 800 characters). Do NOT accumulate append-only changelogs or historical version tags (`v1.0.5`, `since v1.0.3`). Summarize current runtime behavior and keep the unified `Disclosure:` block. Release notes belong in the PR body and GitHub Releases, not in the catalog manifest.
- Preserve existing security disclosures (keyring access, isolated HOME symlinks, token handling).
- Verify `capabilities:` matches actual plugin registrations.

### Step 4: Run Validators
Run both validators:
```bash
python3 scripts/validate_plugin_catalog.py plugin-catalog/antigravity-subscription-directsdk.yaml
# Expected: OK: 1 file(s) valid

hermes plugins validate /root/.hermes/plugins/antigravity-subscription-directsdk
# Expected: Validation passed. (Security scan: safe)
```

### Step 5: Commit and Push to Fork
```bash
git commit -am "catalog: bump antigravity-subscription-directsdk pin to vX.Y.Z"
git push -u fork catalog/bump-antigravity-pin-v<version_slug>
```

### Step 6: Submit Upstream PR
Open the PR against `NousResearch/hermes-agent:main`:
```bash
gh pr create --repo NousResearch/hermes-agent \
  --base main \
  --head soyelmismo:catalog/bump-antigravity-pin-v<version_slug> \
  --title "catalog: bump antigravity-subscription-directsdk pin to vX.Y.Z" \
  --body "<body_content>"
```

---

## 4. PR and Catalog Writing Standards

Maintainers review catalog PRs in high volume. Apply these standards:

### Catalog Description Rules (`plugin-catalog/<name>.yaml`):
1. **Length:** Keep under 800 characters. CLI table views truncate descriptions to 60 characters; web cards show short summaries. Full documentation renders from `README.md`.
2. **No Changelog Bloat:** Never accumulate historical release notes (`v1.0.5 prunes...`, `v1.0.6 strips...`, `since v1.0.3...`).
3. **Current State Only:** Describe current capabilities directly.
4. **Single Unified Disclosure:** Group security disclosures into one `Disclosure:` block (external CLI routing, private temp HOME symlink, keyring presence-only probe, no credential store writes).

### PR Body Rules:
1. **Language:** English only.
2. **Title Format:** `catalog: bump antigravity-subscription-directsdk pin to vX.Y.Z`.
3. **No Puffery:**
   - Avoid buzzwords: "intelligent", "smart", "crucial", "game-changing", "seamlessly", "leverages".
   - Use plain technical terms: "tool pruning", "error classification", "context bound".
4. **No Throat-Clearing:**
   - Cut preamble: "Here is what changed", "In order to", "It is important to note".
   - Lead with facts and measurements.
5. **No Em Dashes:** Use colons or periods.
6. **Active Voice and Concrete Data:**
   - Name changed files (`prompt.py`, `__init__.py`).
   - Cite error signatures, byte counts, and token limits.
   - State measured reductions (e.g. "cuts wire payload from 600 KB to 180 KB").
7. **Fully Qualified Issue and PR References:**
   - Never use bare issue numbers such as `#15` or `(#16)` in PR titles, bodies, or commit summaries. In `NousResearch/hermes-agent`, GitHub auto-links bare numbers to unrelated upstream issues.
   - Always qualify references with the plugin repository: `soyelmismo/hermes-antigravity-subscription#15` or full issue URLs.

### PR Body Template:

```markdown
Follow-up to the catalogued plugin (owner-submitted pin bump, catalog rules 4 and 5).

Pin: <OLD_SHA> -> <NEW_SHA>
Version: <OLD_VERSION> -> <NEW_VERSION>
Release: https://github.com/soyelmismo/hermes-antigravity-subscription/releases/tag/v<NEW_VERSION>

Commits adopted in range (`v<OLD_VERSION>..v<NEW_VERSION>`):

• `<NEW_SHA_SHORT>` release: v<NEW_VERSION>
• `<COMMIT_SHA_SHORT>` <commit message>

### Root Cause / Motivation

<Direct description of failure mode or technical reason for the update. Include exact error message if applicable.>

### Changes in v<NEW_VERSION>

1. **<Component / File 1>**
   - <Exact behavior change, measurements, byte/token counts.>
2. **<Component / File 2>**
   - <Exact behavior change, configuration flags.>

### Validation

• Plugin repo CI: <passed count> passed, <skipped count> skipped against `HERMES_REF v2026.9.21`.
• `python3 scripts/validate_plugin_catalog.py plugin-catalog/antigravity-subscription-directsdk.yaml` passed (1 file valid).
• `hermes plugins validate /root/.hermes/plugins/antigravity-subscription-directsdk` passed (safe, declared capabilities match).
• Manifest version <NEW_VERSION> matches git tag `v<NEW_VERSION>` and catalog entry.
```
