from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request


def wait_for_http(url: str, timeout_seconds: float) -> float:
    start = time.perf_counter()
    deadline = start + timeout_seconds
    last_error = ""
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return time.perf_counter() - start
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.25)
    raise TimeoutError(last_error or f"not ready within {timeout_seconds}s")


def build_streamlit_command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8591)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-cold-start", type=float, default=15.0)
    args = parser.parse_args(argv)

    url = f"http://127.0.0.1:{args.port}"
    process = subprocess.Popen(
        build_streamlit_command(args.port),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready_seconds = wait_for_http(url, args.timeout)
        ok = ready_seconds <= args.max_cold_start
        print(
            json.dumps(
                {"ok": ok, "url": url, "cold_start_seconds": round(ready_seconds, 3)},
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0 if ok else 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
