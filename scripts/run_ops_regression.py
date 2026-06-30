from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_TEST_MODULES = ("tests.test_dgb_multiring", "tests.test_ops_scripts")


def db_family_paths(db_path: str | Path) -> list[Path]:
    resolved = Path(db_path)
    return [
        resolved,
        resolved.with_name(resolved.name + "-wal"),
        resolved.with_name(resolved.name + "-shm"),
    ]


def snapshot_db_family(db_path: str | Path, backup_dir: str | Path) -> dict[str, str | None]:
    backup_root = Path(backup_dir)
    backup_root.mkdir(parents=True, exist_ok=True)
    snapshot: dict[str, str | None] = {}
    for path in db_family_paths(db_path):
        if path.exists():
            backup_path = backup_root / path.name
            shutil.copy2(path, backup_path)
            snapshot[str(path)] = str(backup_path)
        else:
            snapshot[str(path)] = None
    return snapshot


def restore_db_family(db_path: str | Path, snapshot: dict[str, str | None]) -> None:
    for path in db_family_paths(db_path):
        backup_path = snapshot.get(str(path))
        if path.exists():
            path.unlink()
        if backup_path:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, path)


def build_unittest_command(test_modules: list[str] | tuple[str, ...]) -> list[str]:
    return [sys.executable, "-m", "unittest", *test_modules]


def run_regression(
    *,
    db_path: str | Path = "cost_monitor_data.db",
    test_modules: list[str] | tuple[str, ...] = DEFAULT_TEST_MODULES,
    cwd: str | Path = ".",
) -> int:
    root = Path(cwd).resolve()
    resolved_db_path = Path(db_path)
    if not resolved_db_path.is_absolute():
        resolved_db_path = root / resolved_db_path

    with tempfile.TemporaryDirectory(prefix="cost-monitor-regression-") as tmp_dir:
        snapshot = snapshot_db_family(resolved_db_path, Path(tmp_dir) / "db-snapshot")
        try:
            completed = subprocess.run(build_unittest_command(test_modules), cwd=root)
            return completed.returncode
        finally:
            restore_db_family(resolved_db_path, snapshot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="cost_monitor_data.db")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("tests", nargs="*", default=list(DEFAULT_TEST_MODULES))
    args = parser.parse_args(argv)

    print("Running isolated unit regression suite...")
    exit_code = run_regression(db_path=args.db, test_modules=args.tests, cwd=args.cwd)
    if exit_code == 0:
        print("Regression suite passed and DB files were restored from the pre-test snapshot.")
    else:
        print(f"Regression suite failed with exit code {exit_code}; DB files were restored from the pre-test snapshot.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
