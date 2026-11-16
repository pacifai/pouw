#!/usr/bin/env bash
# Check out upstream at the commit in patches/UPSTREAM into the given
# directory, and apply every patch in patches/ in name order.
#
# Usage: fetch_upstream.sh <dir>
#
# Rerunning it discards changes to upstream's tracked files first, so the
# patches apply to clean files.
set -euo pipefail

DIR="$1"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../patches" && pwd)"
COMMIT="$(< "$PATCH_DIR/UPSTREAM")"

if [ ! -d "$DIR/.git" ]; then
  git init -q "$DIR"
  git -C "$DIR" remote add origin https://github.com/pearl-research-labs/pearl
fi
git -C "$DIR" fetch -q --depth 1 origin "$COMMIT"
git -C "$DIR" checkout -q --force FETCH_HEAD
git -C "$DIR" apply "$PATCH_DIR"/*.patch
