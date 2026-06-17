#!/usr/bin/env bash
# release.sh — one-command release.
#
# Bumps the version in all four manifests IN SYNC (package.json, manifest.json,
# .claude-plugin/plugin.json, server.json — incl. its release-asset URL), commits,
# pushes main, and creates + pushes the tag ONCE. The publish-mcp.yml workflow then
# builds the .mcpb, cuts the GitHub Release, and publishes the new version to the
# official MCP Registry (which PulseMCP/Glama mirror). Run from anywhere in the repo.
#
# Usage:
#   scripts/release.sh                  # ship next PATCH (default)
#   scripts/release.sh patch|minor|major
#   scripts/release.sh 1.4.0            # explicit version
#   scripts/release.sh patch --dry-run  # print the plan, change nothing
set -euo pipefail

DRY=0; BUMP="patch"
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    patch|minor|major) BUMP="$a" ;;
    [0-9]*.[0-9]*.[0-9]*) BUMP="$a" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
REPO="$(git config --get remote.origin.url | sed -E 's#\.git$##; s#^.*[/:]([^/]+/[^/]+)$#\1#')"
CUR="$(python3 -c "import json;print(json.load(open('package.json'))['version'])")"

NEXT="$(python3 - "$CUR" "$BUMP" <<'PY'
import sys
cur, bump = sys.argv[1], sys.argv[2]
if bump in ("patch","minor","major"):
    a,b,c = (int(x) for x in cur.split("."))
    a,b,c = (a+1,0,0) if bump=="major" else (a,b+1,0) if bump=="minor" else (a,b,c+1)
    print(f"{a}.{b}.{c}")
else:
    print(bump)
PY
)"
TAG="v$NEXT"
echo "repo: $REPO"
echo "version: $CUR  ->  $NEXT   (tag $TAG)"

# Guard 1: never re-tag an existing version (re-tagging desyncs the registry sha from the
# release asset and 400s as a duplicate version — the exact bug from v1.0.1).
if git rev-parse "$TAG" >/dev/null 2>&1 || git ls-remote --tags origin "refs/tags/$TAG" 2>/dev/null | grep -q "$TAG"; then
  echo "ERROR: tag $TAG already exists (local or remote). Pick a higher version." >&2; exit 1
fi

echo "Will bump: package.json, manifest.json, .claude-plugin/plugin.json, server.json (version + asset URL)"
if [ "$DRY" = "1" ]; then echo "[dry-run] nothing changed."; exit 0; fi

# Guard 2 (real release only): clean tree, so the release commit is just the version bump.
if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree not clean — commit or stash first:" >&2; git status --short >&2; exit 1
fi

python3 - "$CUR" "$NEXT" <<'PY'
import re, pathlib, sys
cur, nxt = sys.argv[1], sys.argv[2]
for f in ["package.json","manifest.json",".claude-plugin/plugin.json","server.json"]:
    p=pathlib.Path(f); t=p.read_text()
    t=t.replace(f'"version": "{cur}"', f'"version": "{nxt}"')      # all version fields
    if f=="server.json":
        t=re.sub(r'/v\d+\.\d+\.\d+/', f'/v{nxt}/', t)              # release-asset URL
    p.write_text(t)
    import json; json.loads(p.read_text())                        # validate still-parses
print("bumped 4 files")
PY

git add package.json manifest.json .claude-plugin/plugin.json server.json
git commit -q -m "Release $TAG"
git push -q origin HEAD
git tag "$TAG"
git push -q origin "$TAG"
echo "Released $TAG. CI will build the .mcpb, cut the Release, and publish to the MCP Registry."
echo "Watch: gh run watch \"\$(gh run list --workflow='Publish to MCP Registry' -R $REPO --limit 1 --json databaseId --jq '.[0].databaseId')\" -R $REPO"
