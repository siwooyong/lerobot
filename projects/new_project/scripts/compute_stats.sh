#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

if [[ -f "$repo_root/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$repo_root/venv/bin/activate"
fi

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  compute_stats.sh <dataset-root> [additional compute_stats.py args]

Examples:
  ./projects/new_project/scripts/compute_stats.sh projects/new_project/data/atomic
  ./projects/new_project/scripts/compute_stats.sh projects/new_project/data/composite
  ./projects/new_project/scripts/compute_stats.sh projects/new_project/data/pretrain_human
EOF
  exit 2
fi

dataset_root="$1"
shift

echo "[compute_stats] repo_root: $repo_root"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[compute_stats] git: $(git branch --show-current) @ $(git rev-parse --short HEAD)"
fi
echo "[compute_stats] dataset: $dataset_root"

python "$repo_root/projects/new_project/scripts/compute_stats.py" \
  --dataset-root "$dataset_root" \
  "$@"
