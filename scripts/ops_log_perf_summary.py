from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return round(ordered[index], 3)


def parse_perf_line(line: str) -> dict[str, Any] | None:
    if "[performance]" not in line:
        return None
    bracket_values = re.findall(r"\[([^\]]+)\]", line)
    seconds = [float(value) for value in re.findall(r"([0-9]+\.[0-9]+)s", line)]
    rows_match = re.search(r"(?:记录数|输出行数)=([0-9]+)", line)
    if len(bracket_values) < 2 or not seconds:
        return None
    return {
        "stage": bracket_values[1],
        "mode": bracket_values[2] if len(bracket_values) >= 3 else "",
        "rows": int(rows_match.group(1)) if rows_match else 0,
        "seconds": seconds[-1],
        "line": line.strip(),
    }


def summarize_performance_logs(log_root: str | Path) -> dict[str, Any]:
    root = Path(log_root)
    records: list[dict[str, Any]] = []
    candidate_paths = sorted(root.glob("*.log")) if root.exists() else []
    if root != Path("."):
        candidate_paths.extend(sorted(Path(".").glob("*.log")))

    for path in candidate_paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            parsed = parse_perf_line(line)
            if parsed:
                parsed["file"] = str(path)
                records.append(parsed)

    groups: dict[str, list[float]] = {}
    for record in records:
        key = f"{record['stage']}|{record['mode']}".strip("|")
        groups.setdefault(key, []).append(float(record["seconds"]))

    summary = {
        key: {
            "count": len(values),
            "p50_seconds": percentile(values, 0.50),
            "p95_seconds": percentile(values, 0.95),
            "max_seconds": round(max(values), 3),
        }
        for key, values in sorted(groups.items())
    }
    return {"record_count": len(records), "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs")
    args = parser.parse_args(argv)
    print(json.dumps(summarize_performance_logs(args.logs), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
