import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from argparse import ArgumentParser

import requests

from autotrain import logger

from . import BaseAutoTrainCommand


def handle_output(stream, log_file):
    while True:
        line = stream.readline()
        if not line:
            break
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()


def run_chat_command_factory(args):
    return RunAutoTrainChatCommand(args.port, args.host)


class RunAutoTrainChatCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_chat_parser = parser.add_parser(
            "chat",
            description="✨ Run AutoTrain Chat / Inference UI",
        )
        run_chat_parser.add_argument(
            "--port",
            type=int,
            default=7860,
            help="Port to run the app on",
            required=False,
        )
        run_chat_parser.add_argument(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Host to run the app on",
            required=False,
        )
        run_chat_parser.set_defaults(func=run_chat_command_factory)

    def __init__(self, port, host):
        self.port = port
        self.host = host

    def run(self):
        # Ensure HF_TOKEN is not strictly required by setting environment if missing
        # But we handled this in api_routes.py / ui_routes.py

        # Add src to PYTHONPATH to prioritize local changes over installed package
        env = os.environ.copy()
        src_path = os.path.abspath("src")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = src_path

        # Use sys.executable to ensure we use the same python environment (e.g. venv)
        command = (
            f"{sys.executable} -m uvicorn autotrain.app.app:app --host {self.host} --port {self.port} --workers 1"
        )

        # Open browser when server is ready
        url = f"http://{self.host}:{self.port}/inference"
        logger.info(f"Starting AutoTrain Chat at {url}")

        def open_browser():
            timeout = 30
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        logger.info(f"Server ready. Opening browser at {url}")
                        webbrowser.open(url)
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(0.5)
            else:
                logger.warning("Timed out waiting for server to start.")

        threading.Thread(target=open_browser, daemon=True).start()

        with open("autotrain.log", "w", encoding="utf-8") as log_file:
            if sys.platform == "win32":
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            else:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                    env=env,
                )

            output_thread = threading.Thread(target=handle_output, args=(process.stdout, log_file))
            output_thread.start()

            try:
                process.wait()
                output_thread.join()
            except KeyboardInterrupt:
                logger.warning("Attempting to terminate the process...")
                if sys.platform == "win32":
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                logger.info("Process terminated by user")
