#!/usr/bin/env bash
# Setup shared git hooks for the repository.
# Run once after clone: bash scripts/setup_git_hooks.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
git config core.hooksPath "$ROOT/.githooks"
echo "Git hooks enabled: $ROOT/.githooks (pre-commit auto-format with ruff)"
