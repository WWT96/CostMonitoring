from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _message_box(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, "CostMonitoring Launcher", 0x10)


def main() -> int:
    app_dir = _app_dir()
    launcher = app_dir / "launcher.bat"
    if not launcher.exists():
        _message_box(f"launcher.bat was not found:\n{launcher}")
        return 1

    subprocess.Popen(
        ["cmd.exe", "/c", str(launcher)],
        cwd=str(app_dir),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
