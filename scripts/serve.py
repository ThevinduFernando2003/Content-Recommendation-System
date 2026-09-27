"""Start the API and the Streamlit UI together.

``make run`` calls this so the same command works in PowerShell and in bash.
Ctrl+C stops both processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _spawn(args: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(args, cwd=ROOT, env=env)


def main() -> None:
    api_port = os.environ.get("PORT", "8000")
    ui_port = os.environ.get("UI_PORT", "8501")
    env = os.environ.copy()
    env.setdefault("API_BASE_URL", f"http://127.0.0.1:{api_port}")
    env.setdefault("PYTHONPATH", str(ROOT))
    api = _spawn(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            api_port,
        ],
        env,
    )
    ui = _spawn(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            ui_port,
            "--server.headless",
            "true",
        ],
        env,
    )
    print(f"API http://127.0.0.1:{api_port}  UI http://127.0.0.1:{ui_port}")
    processes = (api, ui)
    try:
        while True:
            if any(proc.poll() is not None for proc in processes):
                break
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
    codes = [proc.returncode for proc in processes]
    if any(code not in (0, None, -15, 1) for code in codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
