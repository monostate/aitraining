"""Command runner for executing AITraining commands."""

import asyncio
import os
import subprocess
from typing import List, Optional

from textual.widgets import RichLog


class CommandRunner:
    """Handles execution of AITraining commands."""

    def __init__(self, dry_run: bool = False):
        """Initialize the command runner."""
        self.dry_run = dry_run
        self.current_process: Optional[subprocess.Popen] = None

    async def run_command(self, command: List[str], log_widget: RichLog) -> int:
        """Run a command and stream output to the log widget."""
        if self.dry_run:
            log_widget.write("[yellow]DRY RUN[/yellow] - Command would be executed:")
            log_widget.write(f"[dim]{' '.join(command)}[/dim]")
            return 0

        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = "./src" + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"

        try:
            # Start the process
            self.current_process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            log_widget.write("[cyan]Process started[/cyan]\n")

            # Create tasks for reading stdout and stderr
            async def read_stream(stream, prefix=""):
                """Read from a stream and write to log widget."""
                while True:
                    line = await stream.readline()
                    if not line:
                        break

                    text = line.decode("utf-8", errors="replace").rstrip()
                    if text:
                        if prefix:
                            log_widget.write(f"{prefix}{text}")
                        else:
                            log_widget.write(text)

            # Read both streams concurrently
            await asyncio.gather(
                read_stream(self.current_process.stdout),
                read_stream(self.current_process.stderr, "[yellow]ERR:[/yellow] "),
            )

            # Wait for process to complete
            returncode = await self.current_process.wait()

            if returncode == 0:
                log_widget.write("\n[green]✓[/green] Process completed successfully")
            else:
                log_widget.write(f"\n[red]✗[/red] Process exited with code {returncode}")

            return returncode

        except asyncio.CancelledError:
            # Handle cancellation
            if self.current_process:
                log_widget.write("\n[yellow]Stopping process...[/yellow]")
                self.current_process.terminate()
                try:
                    await asyncio.wait_for(self.current_process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    log_widget.write("[red]Force killing process...[/red]")
                    self.current_process.kill()
                    await self.current_process.wait()
            raise

        except Exception as e:
            log_widget.write(f"\n[red]Error running command:[/red] {e}")
            return 1

        finally:
            self.current_process = None

    async def stop_current_process(self) -> None:
        """Stop the currently running process."""
        if self.current_process:
            self.current_process.terminate()
            try:
                await asyncio.wait_for(self.current_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.current_process.kill()
                await self.current_process.wait()
            self.current_process = None

    def is_running(self) -> bool:
        """Check if a process is currently running."""
        return self.current_process is not None and self.current_process.returncode is None
