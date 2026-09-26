"""Manual mutation check: break router.py on purpose, prove the suite notices.

    uv run python scripts/mutate.py

Every mutant is a plausible bug, not a random character change. The script
restores router.py after each run and exits non-zero if a mutant survived.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "router.py"

MUTANTS: list[tuple[str, str, str]] = [
    (
        "fail-open-route",
        '_FALLBACK_ROUTE: Route = "fast"',
        '_FALLBACK_ROUTE: Route = "powerful"',
    ),
    (
        "inverted-logprob-comparison",
        "label = max(scores.items(), key=lambda item: item[1])[0]",
        "label = min(scores.items(), key=lambda item: item[1])[0]",
    ),
    (
        "label-map-swapped",
        "_ROUTE_BY_LABEL: dict[str, Route] = dict(zip(_LABELS, _ROUTES, strict=True))",
        "_ROUTE_BY_LABEL: dict[str, Route] = dict(zip(_ROUTES, _LABELS, strict=True))",
    ),
    (
        "env-not-stripped",
        'os.environ.get(name, "").strip() for name in _REQUIRED_VARS',
        'os.environ.get(name, "") for name in _QUESTIONS',
    ),
    (
        "empty-task-guard-uses-blocks",
        "if not text.strip():\n        raise ValueError",
        "if not parts:\n        raise ValueError",
    ),
    (
        "router-max-tokens-dropped",
        "                max_tokens=1,\n                logprobs=True,",
        "                logprobs=True,",
    ),
    (
        "classifier-retries-silently",
        '            http_client=cast("Any", http_client),\n            max_retries=0,',
        '            http_client=cast("Any", http_client),\n            max_retries=2,',
    ),
    (
        "routes-share-one-endpoint",
        '    def endpoint_for(self, route: Route) -> Endpoint:\n        return self.fast if route == "fast" else self.powerful',
        "    def endpoint_for(self, route: Route) -> Endpoint:\n        return self.fast",
    ),
    (
        "routes-reuse-router-key",
        "                    api_key=endpoint.api_key,",
        "                    api_key=self._settings.router_api_key,",
    ),
    (
        "routes-reuse-router-base-url",
        "                    base_url=endpoint.base_url,",
        "                    base_url=self._settings.router_base_url,",
    ),
    (
        "laya-checkpoint-reverts-to-english",
        'laya_model=os.environ.get("LAYA_MODEL", "").strip() or "multilingual"',
        'laya_model=os.environ.get("LAYA_MODEL", "").strip() or "english"',
    ),
    (
        "laya-max-len-unchecked",
        "    if value <= 0:",
        "    if False:",
    ),
    (
        "laya-max-len-dropped",
        '            laya_max_len=_positive_int(os.environ, "LAYA_MAX_LEN", 1024),',
        "            laya_max_len=1024,",
    ),
    (
        "content-fallback-never-parses",
        '    match = _LABEL_PATTERN.search(content or "")',
        "    match = None",
    ),
]


def run_suite() -> int:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly"],
        cwd=TARGET.parent,
        capture_output=True,
        check=False,
    ).returncode


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")
    survivors: list[str] = []
    killed: list[str] = []

    try:
        for name, old, new in MUTANTS:
            if original.count(old) != 1:
                print(
                    f"SKIP  {name}: anchor not unique ({original.count(old)} matches)"
                )
                survivors.append(f"{name} (anchor not found)")
                continue

            TARGET.write_text(original.replace(old, new), encoding="utf-8")
            try:
                code = run_suite()
            finally:
                TARGET.write_text(original, encoding="utf-8")

            if code == 0:
                survivors.append(name)
                print(f"ALIVE {name}")
            else:
                killed.append(name)
                print(f"KILL  {name}")
    finally:
        TARGET.write_text(original, encoding="utf-8")

    print(f"\n{len(killed)}/{len(MUTANTS)} mutants killed")
    for name in survivors:
        print(f"survivor: {name}")

    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
