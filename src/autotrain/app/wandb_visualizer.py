"""Utilities for managing a background W&B LEET process for the web UI."""

import os
import subprocess
import threading
from collections import deque
from typing import Deque, Dict, Optional

from autotrain import logger


class WandbVisualizerManager:
    """Launch and monitor a `wandb beta leet` process for the web UI."""

    def __init__(self, max_lines: int = 500):
        self._max_lines = max_lines
        self._lines: Deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._run_dir: Optional[str] = None
        self._command: Optional[str] = None
        self._error: Optional[str] = None

    def start(self, run_dir: str, wandb_token: Optional[str] = None) -> bool:
        """Start the LEET process if possible."""
        run_dir = os.path.abspath(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        env = os.environ.copy()
        # Pin all W&B paths to the run directory to avoid writing in repo root
        env["WANDB_DIR"] = run_dir
        env["WANDB_CACHE_DIR"] = run_dir
        env["WANDB_DATA_DIR"] = run_dir
        if wandb_token:
            env["WANDB_API_KEY"] = wandb_token

        import sys

        cmd = [sys.executable, "-m", "wandb", "beta", "leet", run_dir]
        command_display = f'WANDB_DIR="{run_dir}" {sys.executable} -m wandb beta leet "{run_dir}"'

        with self._lock:
            self._stop_locked()
            self._lines.clear()
            self._run_dir = run_dir
            self._command = command_display
            self._error = None

            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            except FileNotFoundError:
                msg = "`wandb` executable not found. Install wandb>=0.23.0 to stream LEET."
                logger.warning(msg)
                self._lines.append(msg)
                self._error = "wandb_not_found"
                self._process = None
                return False
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to launch W&B visualizer: {exc}")
                self._lines.append(f"Failed to launch W&B visualizer: {exc}")
                self._error = "spawn_error"
                self._process = None
                return False

            self._stdout_thread = threading.Thread(target=self._reader, args=(self._process.stdout, ""), daemon=True)
            self._stderr_thread = threading.Thread(
                target=self._reader, args=(self._process.stderr, "[ERR] "), daemon=True
            )
            self._stdout_thread.start()
            self._stderr_thread.start()
            return True

    def stop(self) -> None:
        """Stop the LEET process."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None

    def _reader(self, pipe, prefix: str) -> None:
        """Read process output and store it for UI consumption."""
        if pipe is None:
            return
        try:
            for line in pipe:
                if not line:
                    break
                text = line.rstrip()
                if text:
                    with self._lock:
                        self._lines.append(f"{prefix}{text}")
        finally:
            try:
                pipe.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    def status(self) -> Dict[str, Optional[object]]:
        """Return current status for the frontend."""
        with self._lock:
            active = self._process is not None and self._process.poll() is None
            return {
                "active": active,
                "run_dir": self._run_dir,
                "command": self._command,
                "lines": list(self._lines),
                "error": self._error,
            }

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None
