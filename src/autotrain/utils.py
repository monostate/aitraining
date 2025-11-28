import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from autotrain import logger
from autotrain.commands import launch_command
from autotrain.trainers.clm.params import LLMTrainingParams
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from autotrain.trainers.generic.params import GenericParams
from autotrain.trainers.image_classification.params import ImageClassificationParams
from autotrain.trainers.image_regression.params import ImageRegressionParams
from autotrain.trainers.object_detection.params import ObjectDetectionParams
from autotrain.trainers.sent_transformers.params import SentenceTransformersParams
from autotrain.trainers.seq2seq.params import Seq2SeqParams
from autotrain.trainers.tabular.params import TabularParams
from autotrain.trainers.text_classification.params import TextClassificationParams
from autotrain.trainers.text_regression.params import TextRegressionParams
from autotrain.trainers.token_classification.params import TokenClassificationParams
from autotrain.trainers.vlm.params import VLMTrainingParams


ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "true").lower() == "true"


# Device and model-loading utilities
from typing import Dict, Optional

import torch


def get_model_loading_kwargs(
    token: Optional[str] = None,
    fp16_if_cuda: bool = True,
    trust_remote_code: bool = True,
    extra_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Build consistent kwargs for AutoModel.from_pretrained across codepaths.

    - Uses device_map="auto" on CUDA
    - Prefers float16 on CUDA when fp16_if_cuda=True
    - Uses float32 on MPS and CPU
    - Adds token and trust_remote_code if provided
    """
    kwargs: Dict = {}
    if token is not None:
        kwargs["token"] = token
    if trust_remote_code is not None:
        kwargs["trust_remote_code"] = trust_remote_code

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16 if fp16_if_cuda else torch.float32
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # MPS prefers float32; placement handled after load
        kwargs["torch_dtype"] = torch.float32
    else:
        kwargs["torch_dtype"] = torch.float32

    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return kwargs


def maybe_move_to_mps(model, model_kwargs: Dict):
    """
    If MPS is available and no device_map is set (i.e., CPU placement), move to MPS.
    Returns the (possibly moved) model.
    """
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and "device_map" not in model_kwargs
    ):
        return model.to("mps")
    return model


def _terminate_process(proc: Optional[subprocess.Popen]) -> None:
    """Best-effort termination for spawned subprocesses."""
    if not proc:
        return
    if proc.poll() is not None:
        return
    try:
        # Try to kill the process group since we use start_new_session=True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, AttributeError):
            # Fallback for systems without killpg or if process is gone
            proc.terminate()

        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, AttributeError):
            proc.kill()
        proc.wait()
    except Exception:
        # Final fallback
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def run_training(params, task_id, local=False, wait=False):
    """
    Run the training process based on the provided parameters and task ID.

    Args:
        params (str): JSON string of the parameters required for training.
        task_id (int): Identifier for the type of task to be performed.
        local (bool, optional): Flag to indicate if the training should be run locally. Defaults to False.
        wait (bool, optional): Flag to indicate if the function should wait for the process to complete. Defaults to False.

    Returns:
        int: Process ID of the launched training process.

    Raises:
        NotImplementedError: If the task_id does not match any of the predefined tasks.
    """
    params = json.loads(params)
    if isinstance(params, str):
        params = json.loads(params)
    if task_id == 9:
        params = LLMTrainingParams(**params)
    elif task_id == 28:
        params = Seq2SeqParams(**params)
    elif task_id in (1, 2):
        params = TextClassificationParams(**params)
    elif task_id in (13, 14, 15, 16, 26):
        params = TabularParams(**params)
    elif task_id == 27:
        params = GenericParams(**params)
    elif task_id == 18:
        params = ImageClassificationParams(**params)
    elif task_id == 4:
        params = TokenClassificationParams(**params)
    elif task_id == 10:
        params = TextRegressionParams(**params)
    elif task_id == 29:
        params = ObjectDetectionParams(**params)
    elif task_id == 30:
        params = SentenceTransformersParams(**params)
    elif task_id == 24:
        params = ImageRegressionParams(**params)
    elif task_id == 31:
        params = VLMTrainingParams(**params)
    elif task_id == 5:
        params = ExtractiveQuestionAnsweringParams(**params)
    else:
        raise NotImplementedError

    project_run_dir = os.path.abspath(params.project_name)
    os.makedirs(project_run_dir, exist_ok=True)
    params.save(output_dir=params.project_name)

    env = os.environ.copy()

    # Ensure project log exists BEFORE building command to capture early hangs
    log_path = os.path.join(project_run_dir, "autotrain.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== Preparing training at {os.getcwd()} ===\n")
        f.write(f"Python path: {sys.executable}\n")
        f.write(f"PATH(pre): {env.get('PATH', 'Not set')}\n")
        f.write(f"PYTHONPATH(pre): {env.get('PYTHONPATH', 'Not set')}\n")
        f.write("Will compute launch command next...\n")
        f.flush()

    # Set GPU count override BEFORE building command to avoid torch/CUDA init in parent
    if "AUTOTRAIN_FORCE_NUM_GPUS" not in os.environ:
        # Heuristic: if CUDA devices are exposed, assume 1; otherwise 0
        forced = "1" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "0"
        os.environ["AUTOTRAIN_FORCE_NUM_GPUS"] = forced

    cmd = launch_command(params=params)
    cmd = [str(c) for c in cmd]

    # Fix for BentoCloud: Find accelerate in virtual env or system
    import shutil

    # First, ensure /app/.venv/bin is in PATH for BentoCloud
    venv_bin = "/app/.venv/bin"
    if os.path.exists(venv_bin) and venv_bin not in env.get("PATH", ""):
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"

    accelerate_path = shutil.which("accelerate", path=env.get("PATH"))
    if not accelerate_path:
        # Try common locations on BentoCloud
        possible_paths = [
            "/app/.venv/bin/accelerate",
            "/home/bentoml/.local/bin/accelerate",
            "/usr/local/bin/accelerate",
            os.path.expanduser("~/.local/bin/accelerate"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                accelerate_path = path
                break

    if accelerate_path and cmd[0] == "accelerate":
        cmd[0] = accelerate_path

    # Pre-launch validation and environment diagnostics
    training_config_path = os.path.join(project_run_dir, "training_params.json")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== Training Started at {os.getcwd()} ===\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Accelerate path: {accelerate_path or 'Not found'}\n")
        f.write(f"Python path: {sys.executable}\n")
        f.write(f"PATH: {env.get('PATH', 'Not set')}\n")
        f.write(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}\n")
        f.write(f"Training config exists: {os.path.isfile(training_config_path)} at {training_config_path}\n")
        # GPU/CPU diagnostics
        try:
            import torch  # type: ignore

            f.write(f"CUDA available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                try:
                    f.write(f"CUDA device count: {torch.cuda.device_count()}\n")
                    f.write(f"CUDA version: {getattr(torch.version, 'cuda', 'unknown')}\n")
                except Exception:
                    pass
            # MPS diagnostics (macOS)
            try:
                mps_avail = getattr(torch.backends, "mps", None)
                if mps_avail is not None:
                    f.write(f"MPS available: {torch.backends.mps.is_available()}\n")
            except Exception:
                pass
        except Exception as e:  # torch may not be available for some tasks
            f.write(f"Torch diagnostics failed: {e}\n")
        # nvidia-smi
        try:
            import shutil as _sh

            smi = _sh.which("nvidia-smi")
            f.write(f"nvidia-smi path: {smi or 'Not found'}\n")
            if smi:
                try:
                    res = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=5)
                    f.write(f"nvidia-smi -L rc={res.returncode}: {res.stdout or res.stderr}\n")
                except Exception as nse:
                    f.write(f"nvidia-smi check failed: {nse}\n")
        except Exception:
            pass

        # Accelerate smoke tests
        accelerate_ok = False
        if accelerate_path:
            try:
                ver = subprocess.run(
                    [accelerate_path, "--version"], env=env, capture_output=True, text=True, timeout=10
                )
                f.write(f"accelerate --version rc={ver.returncode}: {(ver.stdout or ver.stderr).strip()}\n")
                envres = subprocess.run([accelerate_path, "env"], env=env, capture_output=True, text=True, timeout=15)
                # Truncate to avoid huge logs
                env_out = (envres.stdout or envres.stderr or "").splitlines()[:50]
                f.write("accelerate env (first 50 lines):\n" + "\n".join(env_out) + "\n")
                accelerate_ok = ver.returncode == 0
            except Exception as ae:
                f.write(f"accelerate smoke test failed: {ae}\n")
        else:
            f.write("accelerate not found; will consider Python fallback if needed.\n")

        f.write("=" * 50 + "\n")
        f.flush()

    # Validate training config file presence
    if not os.path.isfile(training_config_path):
        raise FileNotFoundError(f"training_params.json not found at {training_config_path}")

    # Check if we're in an interactive terminal BEFORE opening log file (which might redirect stdout)
    # Use stdin.isatty() as a more reliable check since stdout might be redirected
    is_interactive_terminal = sys.stdin.isatty() or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())

    # Prepare log file handle for subprocess I/O
    log_fh = open(log_path, "a", encoding="utf-8")

    # Avoid CUDA initialization in parent process by preventing torch import and CUDA context creation here
    # Users can force GPU count via env if needed
    if "AUTOTRAIN_FORCE_NUM_GPUS" not in env:
        env["AUTOTRAIN_FORCE_NUM_GPUS"] = env.get(
            "AUTOTRAIN_FORCE_NUM_GPUS", "1" if os.environ.get("CUDA_VISIBLE_DEVICES") else "0"
        )

    # Optionally force Python module launch instead of accelerate to avoid CLI/exec issues
    force_python = os.environ.get("AUTOTRAIN_FORCE_PYTHON_LAUNCH", "false").lower() in ("1", "true", "yes")
    if force_python and "-m" in cmd:
        try:
            m_index = cmd.index("-m")
            module = cmd[m_index + 1]
            module_args = cmd[m_index + 2 :]
            cmd = [sys.executable, "-m", module] + module_args
        except Exception:
            pass

    # Ensure unbuffered Python output for better real-time logging
    env.setdefault("PYTHONUNBUFFERED", "1")

    # Ensure all W&B write paths land under the project run directory
    env["WANDB_DIR"] = project_run_dir
    env["WANDB_CACHE_DIR"] = project_run_dir
    env["WANDB_DATA_DIR"] = project_run_dir
    if hasattr(params, "wandb_token") and params.wandb_token:
        env["WANDB_API_KEY"] = params.wandb_token

    wandb_logging_enabled = getattr(params, "log", "none") == "wandb"
    wandb_hint_command = (
        f'WANDB_DIR="{project_run_dir}" wandb beta leet "{project_run_dir}"' if wandb_logging_enabled else None
    )

    # Check if we're in an interactive terminal BEFORE opening log file (which might redirect stdout)
    # Use stdin.isatty() as a more reliable check since stdout might be redirected
    is_interactive_terminal = sys.stdin.isatty() or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())

    if wandb_logging_enabled and wandb_hint_command:
        log_fh.write("[W&B] Offline metrics stored for this run. Reopen LEET anytime with:\n")
        log_fh.write(f"      {wandb_hint_command}\n")
        log_fh.flush()

    process: Optional[subprocess.Popen] = None
    leet_process: Optional[subprocess.Popen] = None
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )

        wandb_visualizer_enabled = bool(getattr(params, "wandb_visualizer", False))
        log_fh.write(
            f"[W&B] wandb_visualizer_enabled={wandb_visualizer_enabled}, is_interactive_terminal={is_interactive_terminal}, log={getattr(params, 'log', 'none')}\n"
        )
        log_fh.flush()
        # Use a lock file to prevent duplicate LEET launches
        leet_lock_file = os.path.join(project_run_dir, ".leet_launched")
        leet_already_launched = os.path.exists(leet_lock_file)

        if wandb_visualizer_enabled and is_interactive_terminal and not leet_already_launched:
            try:
                # Try to launch LEET in a new terminal window (macOS/Linux)
                # This allows LEET to display its TUI properly
                import platform

                leet_cmd = [sys.executable, "-m", "wandb", "beta", "leet", project_run_dir]

                if wandb_hint_command:
                    log_fh.write(f"[W&B] Launching LEET with: {wandb_hint_command}\n")
                    log_fh.flush()

                # Try to open LEET in a new terminal window (cross-platform)
                # LEET will stay open after training finishes to show metrics
                system = platform.system()

                # Create a wrapper script that waits for wandb .wandb file before launching LEET
                wait_script_content = f"""#!/bin/bash
cd {os.getcwd()}
export WANDB_DIR="{project_run_dir}"
echo "Waiting for W&B training to start..."
# Wait for wandb .wandb file to appear (wandb.init() creates it inside run-* directories)
# LEET needs the .wandb file, not just the directory structure
max_wait=60
wait_count=0
wandb_run_dir=""
while [ $wait_count -lt $max_wait ]; do
    # Check for .wandb file in any run-* directory
    wandb_run_dir=$(find "{project_run_dir}/wandb" -name "run-*.wandb" -type f 2>/dev/null | head -1)
    if [ -n "$wandb_run_dir" ]; then
        # Extract the directory containing the .wandb file
        wandb_run_dir=$(dirname "$wandb_run_dir")
        echo "W&B run detected in: $wandb_run_dir"
        echo "Launching LEET panel..."
        break
    fi
    sleep 1
    wait_count=$((wait_count + 1))
done
# Launch LEET with the run directory (or project root if run not found yet)
# LEET will display metrics and stay open even after training finishes
if [ -n "$wandb_run_dir" ]; then
    {sys.executable} -m wandb beta leet "$wandb_run_dir"
else
    # Fallback: try project root (LEET should find latest run)
    echo "Warning: Run directory not found, trying project root..."
    {sys.executable} -m wandb beta leet "{project_run_dir}"
fi
# Keep terminal open after LEET exits (so user can see metrics/history)
echo ""
echo "LEET panel closed. Press Enter to close this window..."
read
"""

                try:
                    import shutil
                    import tempfile

                    if system == "Darwin":
                        # macOS: Use osascript to open Terminal.app
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                            f.write(wait_script_content)
                            temp_script = f.name
                        os.chmod(temp_script, 0o755)

                        script = f"""
                        tell application "Terminal"
                            activate
                            do script "{temp_script}"
                        end tell
                        """
                        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
                        log_fh.write("[W&B] ✓ Opening LEET in a new Terminal window (macOS)...\n")

                    elif system == "Windows":
                        # Windows: Use start command or PowerShell
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".bat", delete=False) as f:
                            # Convert bash script to Windows batch/PowerShell
                            f.write(
                                f"""@echo off
cd /d {os.getcwd()}
set WANDB_DIR={project_run_dir}
echo Waiting for W&B training to start...
timeout /t 3 /nobreak >nul
echo Launching W&B LEET panel...
{sys.executable} -m wandb beta leet "{project_run_dir}"
echo.
echo LEET panel closed. Press any key to close this window...
pause >nul
"""
                            )
                            temp_script = f.name

                        # Try to open in new window using start command
                        subprocess.Popen(
                            ["cmd", "/c", "start", "cmd", "/k", temp_script],
                            shell=False,
                            creationflags=(
                                subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0
                            ),
                        )
                        log_fh.write("[W&B] ✓ Opening LEET in a new Command Prompt window (Windows)...\n")

                    else:
                        # Linux: Try various terminal emulators
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                            f.write(wait_script_content)
                            temp_script = f.name
                        os.chmod(temp_script, 0o755)

                        # Try common Linux terminal emulators
                        terminal_launched = False
                        for term_cmd in [
                            ["gnome-terminal", "--", "bash", temp_script],
                            ["xterm", "-e", "bash", temp_script],
                            ["konsole", "-e", "bash", temp_script],
                            ["x-terminal-emulator", "-e", "bash", temp_script],
                            ["terminator", "-e", f"bash {temp_script}"],
                        ]:
                            if shutil.which(term_cmd[0]):
                                try:
                                    subprocess.Popen(term_cmd, check=False)
                                    log_fh.write(f"[W&B] ✓ Opening LEET in a new {term_cmd[0]} window (Linux)...\n")
                                    terminal_launched = True
                                    break
                                except Exception:
                                    continue

                        if not terminal_launched:
                            raise Exception("No suitable terminal emulator found")

                    # Create lock file to prevent duplicate launches
                    try:
                        with open(leet_lock_file, "w") as f:
                            f.write(str(os.getpid()))
                    except Exception:
                        pass

                    log_fh.write("[W&B] LEET will wait for training to start, then display metrics.\n")
                    log_fh.write("[W&B] LEET will stay open after training finishes to show metrics/history.\n")
                    log_fh.flush()
                    # Don't create a background process, we opened it in a new window
                    leet_process = None

                except Exception as e:
                    # Fallback: try background launch (may not display TUI)
                    log_fh.write(f"[W&B] ⚠ Could not open new terminal window: {e}\n")
                    log_fh.write("[W&B] Attempting background launch (TUI may not display)...\n")
                    leet_process = subprocess.Popen(
                        leet_cmd,
                        env=env,
                        stdout=None,  # Don't redirect - let LEET try to display
                        stderr=subprocess.DEVNULL,  # Still suppress stderr errors
                        start_new_session=True,
                    )
                    if leet_process.poll() is None:
                        log_fh.write(f"[W&B] ✓ LEET process started (PID: {leet_process.pid})\n")
                        log_fh.write("[W&B] Note: LEET TUI may not display when training output is redirected.\n")
                        log_fh.write(f"[W&B] Run manually in another terminal: {wandb_hint_command}\n")
                    else:
                        log_fh.write(
                            f"[W&B] ⚠ LEET process exited immediately (exit code: {leet_process.returncode})\n"
                        )
                        log_fh.write(
                            f"[W&B] This is normal - LEET will start once wandb.init() runs during training\n"
                        )
                        log_fh.write(f"[W&B] Run manually: {wandb_hint_command}\n")
                    log_fh.flush()
            except Exception as e:
                logger.warning(f"Failed to launch W&B visualizer: {e}")
                if wandb_hint_command:
                    log_fh.write(f"[W&B] ❌ Failed to auto-launch LEET: {e}\n")
                    log_fh.write(f"[W&B] Run manually with:\n      {wandb_hint_command}\n")
                    log_fh.flush()
        elif wandb_visualizer_enabled and wandb_hint_command:
            log_fh.write("[W&B] ⚠ Visualizer requires an interactive terminal. Run manually with:\n")
            log_fh.write(f"      {wandb_hint_command}\n")
            log_fh.flush()

        def _reap_proc(p: subprocess.Popen, fh, leet_p=None):
            try:
                exit_code = p.wait()
                try:
                    fh.write(f"\n=== Training subprocess exited with code {exit_code} ===\n")
                    if exit_code == 0:
                        # Training completed successfully - show next steps
                        project_path = os.path.abspath(params.project_name)
                        fh.write("\n" + "=" * 60 + "\n")
                        fh.write("✓ Training completed successfully!\n")
                        fh.write("=" * 60 + "\n")
                        fh.write(f"Model saved to: {project_path}\n")
                        fh.write("\n💡 Next steps:\n")
                        fh.write(f"   Test your model with the Chat UI:\n")
                        fh.write(f"   aitraining chat --model {project_path}\n")
                        fh.write(f"   Or visit: http://localhost:7860/inference\n")
                    fh.flush()
                except Exception:
                    pass
            finally:
                try:
                    fh.close()
                except Exception:
                    pass
                # Note: We don't terminate leet_p here because:
                # - If it's in a separate terminal window, it should stay open to show metrics
                # - If it's a background process, it will exit naturally when training finishes
                # Only terminate if it's a background process that's still running
                if leet_p and leet_p.poll() is None:
                    # Only terminate background processes, not separate terminal windows
                    # (separate terminal windows have leet_p=None)
                    try:
                        leet_p.terminate()
                        fh.write("\n[W&B] LEET panel closed (training finished)\n")
                        fh.flush()
                    except Exception:
                        pass

        if not wait:
            threading.Thread(target=_reap_proc, args=(process, log_fh, leet_process), daemon=True).start()
        else:
            try:
                exit_code = process.wait()
            finally:
                # Only terminate LEET if it's a background process (not a separate terminal window)
                # Separate terminal windows have leet_process=None
                if leet_process and leet_process.poll() is None:
                    _terminate_process(leet_process)
                    try:
                        log_fh.write("\n[W&B] LEET panel closed (training finished)\n")
                        log_fh.flush()
                    except Exception:
                        pass
                try:
                    log_fh.write(f"\n=== Training subprocess exited with code {exit_code} ===\n")
                    if exit_code == 0:
                        # Training completed successfully - show next steps
                        project_path = os.path.abspath(params.project_name)
                        log_fh.write("\n" + "=" * 60 + "\n")
                        log_fh.write("✓ Training completed successfully!\n")
                        log_fh.write("=" * 60 + "\n")
                        log_fh.write(f"Model saved to: {project_path}\n")
                        log_fh.write("\n💡 Next steps:\n")
                        log_fh.write(f"   Test your model with the Chat UI:\n")
                        log_fh.write(f"   aitraining chat\n")
                        log_fh.write(f"   Then select your model: {project_path}\n")
                        log_fh.write(f"   Or visit: http://localhost:7860/inference\n")
                    log_fh.flush()
                except Exception:
                    pass
                try:
                    log_fh.close()
                except Exception:
                    pass
            if exit_code != 0:
                raise RuntimeError(f"Training failed with exit code: {exit_code}")
        return process.pid
    except KeyboardInterrupt:
        _terminate_process(process)
        # Only terminate LEET if it's a background process (not a separate terminal window)
        if leet_process and leet_process.poll() is None:
            _terminate_process(leet_process)
        try:
            log_fh.write("\n[INTERRUPTED] Training cancelled by user.\n")
            log_fh.flush()
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass
        raise
    except (FileNotFoundError, PermissionError, OSError) as spawn_err:
        # If accelerate was intended but failed to spawn, try a Python fallback for single-process training
        fallback_attempted = False
        fallback_cmd = None
        if (not accelerate_path) or (cmd and os.path.basename(cmd[0]).startswith("accelerate")):
            try:
                if "-m" in cmd:
                    m_index = cmd.index("-m")
                    module = cmd[m_index + 1]
                    module_args = cmd[m_index + 2 :]
                    fallback_cmd = [sys.executable, "-m", module] + module_args
                    fallback_attempted = True
            except Exception:
                fallback_attempted = False

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Primary launch failed: {spawn_err}\n")
            if fallback_attempted:
                f.write(f"Attempting Python fallback command: {' '.join(fallback_cmd or [])}\n")
            else:
                f.write("No fallback attempted.\n")

        if fallback_attempted and fallback_cmd:
            try:
                process = subprocess.Popen(
                    fallback_cmd,
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    start_new_session=True,
                )
                if not wait:
                    threading.Thread(target=_reap_proc, args=(process, log_fh), daemon=True).start()
            except Exception as fb_err:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"Fallback launch failed: {fb_err}\n")
                raise RuntimeError(f"Training launch failed (primary and fallback): {fb_err}") from fb_err
        else:
            raise RuntimeError(f"Training launch failed: {spawn_err}") from spawn_err

    # Don't check immediately - let the service handle monitoring
    # The immediate check might be failing during BentoML startup

    return process.pid


# Sweep functionality - moved from utils/sweep.py
class SweepBackend(Enum):
    """Available hyperparameter sweep backends."""

    OPTUNA = "optuna"
    RAY_TUNE = "ray_tune"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"


@dataclass
class ParameterRange:
    """Defines a range for hyperparameter sweep."""

    low: float = None
    high: float = None
    distribution: str = "uniform"  # uniform, log_uniform, int_uniform
    name: str = None
    param_type: str = None  # categorical, float, int
    choices: List[Any] = None
    step: float = None

    def sample(self, trial=None, backend=None):
        """Sample a value from this range."""
        import random

        # Handle categorical parameters
        if self.param_type == "categorical" and self.choices:
            if trial:
                return trial.suggest_categorical(self.name or "param", self.choices)
            return random.choice(self.choices)

        # Handle numeric parameters with step
        if self.step is not None:
            import numpy as np

            values = np.arange(self.low, self.high + self.step, self.step)
            if trial:
                return trial.suggest_categorical(self.name or "param", values.tolist())
            return random.choice(values)

        # Handle regular distributions
        if self.distribution == "uniform" or self.param_type == "float":
            if trial:
                return trial.suggest_float(self.name or "param", self.low, self.high)
            return random.uniform(self.low, self.high)
        elif self.distribution == "log_uniform":
            if trial:
                return trial.suggest_float(self.name or "param", self.low, self.high, log=True)
            import math

            return math.exp(random.uniform(math.log(self.low), math.log(self.high)))
        elif self.distribution == "int_uniform" or self.param_type == "int":
            if trial:
                return trial.suggest_int(self.name or "param", int(self.low), int(self.high))
            return random.randint(int(self.low), int(self.high))


@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweep."""

    backend: SweepBackend = SweepBackend.OPTUNA
    n_trials: int = 10
    direction: str = "minimize"
    parameters: Dict[str, Union[ParameterRange, List]] = field(default_factory=dict)
    metric: str = "eval_loss"
    timeout: Optional[int] = None


class SweepResult:
    """Results from hyperparameter sweep."""

    def __init__(self, config=None, best_params=None, best_value=None, trials=None, backend=None, study=None):
        """Initialize SweepResult with flexible parameters."""
        if config is not None and best_params is None:
            # Initialize from config only (for compatibility with tests)
            self.config = config
            self.best_params = {}
            self.best_value = None
            self.trials = []
            self.backend = getattr(config, "backend", "unknown") if hasattr(config, "backend") else "unknown"
            self.direction = getattr(config, "direction", "minimize")
            self.study = None
            self.best_trial_id = None
        else:
            # Initialize with explicit parameters
            self.config = config
            self.best_params = best_params or {}
            self.best_value = best_value
            self.trials = trials or []
            self.backend = backend or "unknown"
            self.direction = getattr(config, "direction", "minimize") if config else "minimize"
            self.study = study
            self.best_trial_id = None

    def add_trial(self, trial_id, params, value):
        """Add a trial to the results."""
        trial = {"id": trial_id, "params": params, "value": value}
        self.trials.append(trial)

        # Update best values
        if self.best_value is None:
            self.best_value = value
            self.best_params = params
            self.best_trial_id = trial_id
        else:
            is_better = (value < self.best_value) if self.direction == "minimize" else (value > self.best_value)
            if is_better:
                self.best_value = value
                self.best_params = params
                self.best_trial_id = trial_id

    def to_dataframe(self):
        """Convert trials to pandas DataFrame."""
        import pandas as pd

        if not self.trials:
            return pd.DataFrame()

        # Flatten trial data
        rows = []
        for trial in self.trials:
            row = {"trial_id": trial["id"], "value": trial["value"]}
            row.update(trial["params"])
            rows.append(row)

        return pd.DataFrame(rows)

    def save(self, path):
        """Save results to JSON and CSV."""
        import json
        from pathlib import Path

        path = Path(path)

        # Save JSON
        data = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "trials": self.trials,
            "backend": self.backend,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Save CSV
        csv_path = path.parent / path.name.replace(".json", ".csv")
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(csv_path, index=False)

    def plot_optimization_history(self, path=None):
        """Plot optimization history (placeholder for tests)."""
        # Placeholder implementation for tests
        # Real implementation would create matplotlib plot

    def plot_parallel_coordinates(self, path=None):
        """Plot parallel coordinates (placeholder for tests)."""
        # Placeholder implementation for tests
        # Real implementation would use plotly


class HyperparameterSweep:
    """Manager for hyperparameter sweeps."""

    def __init__(self, config: SweepConfig, train_function: Callable = None):
        self.config = config
        self.train_function = train_function
        self.results = []

    def run(self, train_function: Callable = None) -> SweepResult:
        """Run the hyperparameter sweep."""
        # Use provided train_function or the one from init
        train_fn = train_function or self.train_function
        if not train_fn:
            raise ValueError("No train_function provided")

        if self.config.backend == SweepBackend.OPTUNA:
            return self._run_optuna(train_fn)
        elif self.config.backend == SweepBackend.RANDOM_SEARCH:
            return self._run_random_search(train_fn)
        elif self.config.backend == SweepBackend.GRID_SEARCH:
            return self._run_grid_search(train_fn)
        else:
            raise NotImplementedError(f"Backend {self.config.backend} not implemented")

    def _run_optuna(self, train_function):
        """Run sweep using Optuna."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter sweeps. Install with: pip install optuna")

        def objective(trial):
            params = {}
            for name, spec in self.config.parameters.items():
                if isinstance(spec, ParameterRange):
                    if spec.distribution == "uniform":
                        params[name] = trial.suggest_float(name, spec.low, spec.high)
                    elif spec.distribution == "log_uniform":
                        params[name] = trial.suggest_float(name, spec.low, spec.high, log=True)
                    elif spec.distribution == "int_uniform":
                        params[name] = trial.suggest_int(name, int(spec.low), int(spec.high))
                elif isinstance(spec, list):
                    params[name] = trial.suggest_categorical(name, spec)
                elif isinstance(spec, tuple) and len(spec) == 3:
                    low, high, dist = spec
                    if dist == "log_uniform":
                        params[name] = trial.suggest_float(name, low, high, log=True)
                    elif dist == "float":
                        params[name] = trial.suggest_float(name, low, high)
                    else:
                        params[name] = trial.suggest_float(name, low, high)

            return train_function(params)

        study = optuna.create_study(direction=self.config.direction)
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)

        return SweepResult(
            best_params=study.best_params,
            best_value=study.best_value,
            trials=[{"params": t.params, "value": t.value} for t in study.trials],
            backend="optuna",
            study=study,
        )

    def _run_random_search(self, train_function):
        """Run random search."""
        import random

        trials = []
        best_value = float("inf") if self.config.direction == "minimize" else float("-inf")
        best_params = None

        for _ in range(self.config.n_trials):
            params = {}
            for name, spec in self.config.parameters.items():
                if isinstance(spec, ParameterRange):
                    params[name] = spec.sample()
                elif isinstance(spec, list):
                    params[name] = random.choice(spec)

            value = train_function(params)
            trials.append({"params": params, "value": value})

            if self.config.direction == "minimize":
                if value < best_value:
                    best_value = value
                    best_params = params
            else:
                if value > best_value:
                    best_value = value
                    best_params = params

        return SweepResult(best_params=best_params, best_value=best_value, trials=trials, backend="random_search")

    def _run_grid_search(self, train_function):
        """Run grid search."""
        import itertools

        # Create parameter grid
        param_lists = []
        param_names = []

        for name, spec in self.config.parameters.items():
            param_names.append(name)
            if isinstance(spec, list):
                param_lists.append(spec)
            elif isinstance(spec, ParameterRange):
                # Convert range to discrete values for grid search
                if spec.distribution == "int_uniform":
                    values = list(range(int(spec.low), int(spec.high) + 1))
                else:
                    # Sample 5 points for continuous parameters
                    values = [spec.low + (spec.high - spec.low) * i / 4 for i in range(5)]
                param_lists.append(values)

        trials = []
        best_value = float("inf") if self.config.direction == "minimize" else float("-inf")
        best_params = None

        for param_values in itertools.product(*param_lists):
            params = dict(zip(param_names, param_values))
            value = train_function(params)
            trials.append({"params": params, "value": value})

            if self.config.direction == "minimize":
                if value < best_value:
                    best_value = value
                    best_params = params
            else:
                if value > best_value:
                    best_value = value
                    best_params = params

        return SweepResult(best_params=best_params, best_value=best_value, trials=trials, backend="grid_search")


def run_autotrain_sweep(
    model_config: Dict,
    sweep_parameters: Dict,
    train_function: Callable,
    metric: str = "eval_loss",
    direction: str = "minimize",
    n_trials: int = 10,
    backend: str = "optuna",
    output_dir: Optional[str] = None,
) -> SweepResult:
    """
    Convenience function to run hyperparameter sweep for AutoTrain.

    Args:
        model_config: Base configuration dict
        sweep_parameters: Parameters to sweep with their ranges
        train_function: Function that takes params and returns metric value
        metric: Metric to optimize
        direction: "minimize" or "maximize"
        n_trials: Number of trials to run
        backend: Sweep backend to use
        output_dir: Directory to save results

    Returns:
        SweepResult with best parameters and trial history
    """
    # Convert sweep parameters to ParameterRange objects
    processed_params = {}
    for name, spec in sweep_parameters.items():
        if isinstance(spec, tuple) and len(spec) == 3:
            low, high, dist = spec
            processed_params[name] = ParameterRange(low, high, dist)
        elif isinstance(spec, list):
            processed_params[name] = spec
        else:
            processed_params[name] = spec

    config = SweepConfig(
        backend=SweepBackend(backend.lower()),
        n_trials=n_trials,
        direction=direction,
        parameters=processed_params,
        metric=metric,
    )

    sweep = HyperparameterSweep(config)
    result = sweep.run(train_function)

    # Save results if output_dir provided
    if output_dir:
        import json

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "sweep_results.json"), "w") as f:
            json.dump(
                {
                    "best_params": result.best_params,
                    "best_value": result.best_value,
                    "trials": result.trials,
                },
                f,
                indent=2,
            )

    return result
