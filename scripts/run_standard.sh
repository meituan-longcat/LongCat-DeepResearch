#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
question_file="${1:?usage: scripts/run_standard.sh <question-file> [run-root]}"
run_root="${2:-$repo_root/runs}"

cd "$repo_root"
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_root/.env"
  set +a
fi

exec python3 -m longcat_deepresearch \
  --question-file "$question_file" \
  --run-root "$run_root"
