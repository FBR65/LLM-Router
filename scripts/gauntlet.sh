#!/usr/bin/env bash
# One command that reruns every gauntlet layer.
#
#   uv run bash scripts/gauntlet.sh
#
# Layers: lint, format, types, tests, changed-line coverage, mutation, real run.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8099}"
fail=0

step() { printf '\n=== %s ===\n' "$1"; }

step "lint"
uv run ruff check . || fail=1

step "format"
uv run ruff format --check . || fail=1

step "types"
uv run mypy router.py tests scripts/mutate.py || fail=1

step "tests + coverage (randomized order, 3 seeds)"
for seed in 1 2 3; do
  uv run pytest -q -p randomly -p "no:cacheprovider" --randomly-seed="$seed" \
    --cov=router --cov-report=term-missing --cov-branch || fail=1
done

step "mutation"
uv run python scripts/mutate.py || fail=1

step "real run against three separate local OpenAI-compatible servers"
servers=()
for offset in 0 1 2; do
  uv run python scripts/fake_openai_server.py "$((PORT + offset))" &
  servers+=($!)
done
trap 'for pid in "${servers[@]}"; do kill "$pid" 2>/dev/null || true; done' EXIT
sleep 1
OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" \
OPENAI_API_KEY=router-key \
ROUTER_MODEL=router-model \
FAST_BASE_URL="http://127.0.0.1:$((PORT + 1))/v1" \
FAST_API_KEY=fast-key \
FAST_MODEL=luna \
POWERFUL_BASE_URL="http://127.0.0.1:$((PORT + 2))/v1" \
POWERFUL_API_KEY=powerful-key \
POWERFUL_MODEL=sol \
uv run python router.py || fail=1

step "result"
if [ "$fail" -eq 0 ]; then echo "all layers green"; else echo "FAILURES"; fi
exit "$fail"
