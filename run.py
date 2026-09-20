import json
import subprocess
import sys
import threading
from pathlib import Path

import webview


PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD = PROJECT_ROOT / "ecosystem_dashboard.html"


class EcosimAPI:
    def __init__(self):
        self.lock = threading.Lock()

        self.running = False
        self.progress = 0
        self.step = 0
        self.total_steps = 0
        self.status = "Ready"
        self.error = None

    def get_progress(self):
        """Return the current simulation progress to the dashboard."""
        with self.lock:
            return {
                "running": self.running,
                "progress": self.progress,
                "step": self.step,
                "total_steps": self.total_steps,
                "status": self.status,
                "error": self.error,
            }

    def run_simulation(self, seed, steps):
        """
        Start a new simulation in the background.

        The JavaScript dashboard polls get_progress() while this runs.
        """
        if self.running:
            return {
                "success": False,
                "message": "A simulation is already running."
            }

        try:
            seed = int(seed)
            steps = int(steps)

            if steps < 1:
                raise ValueError("Steps must be greater than 0.")

            if steps > 100000:
                raise ValueError("Steps must be 100,000 or fewer.")

        except (TypeError, ValueError) as exc:
            return {
                "success": False,
                "message": str(exc)
            }

        # Reset progress state.
        with self.lock:
            self.running = True
            self.progress = 0
            self.step = 0
            self.total_steps = steps
            self.status = "Starting simulation..."
            self.error = None

        # Run the actual work in a background thread so the WebView
        # remains responsive while the simulation is running.
        thread = threading.Thread(
            target=self._simulation_worker,
            args=(seed, steps),
            daemon=True,
        )
        thread.start()

        return {
            "success": True,
            "message": "Simulation started."
        }

    def _simulation_worker(self, seed, steps):
        try:
            # ---------------------------------------------------------
            # 1. Run the simulation
            # ---------------------------------------------------------
            command = [
                sys.executable,
                "-u",
                "-m",
                "ecosim.simulate",
                "--steps",
                str(steps),
                "--seed",
                str(seed),
                "--out",
                "run",
            ]

            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # The simulation currently prints progress at regular
            # intervals, e.g.:
            #
            # step   200 | plants ...
            #
            # We read those lines and turn them into dashboard progress.
            for line in process.stdout:
                line = line.strip()

                if not line:
                    continue

                # Keep the most useful status text visible.
                if line.startswith("Starting simulation"):
                    with self.lock:
                        self.status = "Starting simulation..."

                elif line.startswith("step"):
                    try:
                        parts = line.split()
                        step_index = parts.index("step")
                        current_step = int(parts[step_index + 1])

                        current_step = max(
                            0,
                            min(current_step, steps)
                        )

                        # Reserve the final 5% for dashboard building.
                        simulation_progress = (
                            current_step / steps
                        ) * 95

                        with self.lock:
                            self.step = current_step
                            self.progress = int(simulation_progress)
                            self.status = (
                                f"Simulating ecosystem — "
                                f"step {current_step:,} / {steps:,}"
                            )

                    except (ValueError, IndexError):
                        pass

            return_code = process.wait()

            if return_code != 0:
                raise RuntimeError(
                    f"Simulation exited with code {return_code}."
                )

            # ---------------------------------------------------------
            # 2. Build dashboard
            # ---------------------------------------------------------
            with self.lock:
                self.progress = 96
                self.status = "Building dashboard..."

            build_command = [
                sys.executable,
                "-u",
                "build_dashboard.py",
                "--prefix",
                "run",
                "--out",
                "ecosystem_dashboard.html",
            ]

            build_process = subprocess.run(
                build_command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )

            if build_process.returncode != 0:
                error_message = (
                    build_process.stderr.strip()
                    or build_process.stdout.strip()
                    or "Dashboard build failed."
                )
                raise RuntimeError(error_message)

            # ---------------------------------------------------------
            # 3. Finished
            # ---------------------------------------------------------
            with self.lock:
                self.step = steps
                self.progress = 100
                self.status = "Simulation complete."
                self.error = None

        except Exception as exc:
            with self.lock:
                self.running = False
                self.error = str(exc)
                self.status = "Simulation failed."

        else:
            # Give the dashboard a moment to see 100%.
            import time
            time.sleep(0.4)

            with self.lock:
                self.running = False


def build_initial_dashboard():
    """
    Build the initial dashboard when the application starts.
    """
    command = [
        sys.executable,
        "-m",
        "ecosim.simulate",
        "--steps",
        "3000",
        "--seed",
        "42",
        "--out",
        "run",
    ]

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    build_command = [
        sys.executable,
        "build_dashboard.py",
        "--prefix",
        "run",
        "--out",
        "ecosystem_dashboard.html",
    ]

    subprocess.run(
        build_command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    # Always rebuild the initial dashboard when launching the app.
    build_initial_dashboard()

    api = EcosimAPI()

    window = webview.create_window(
        "Ecosim — Evolutionary Simulation",
        DASHBOARD.as_uri(),
        js_api=api,
        width=1600,
        height=1000,
        min_size=(1100, 700),
        resizable=True,
        fullscreen=False,
        frameless=False,
        text_select=True,
    )
    gui_backend = "edgechromium" if sys.platform.startswith("win") else None

    webview.start(
        gui=gui_backend,
        debug=False,
        http_server=True,
    )


if __name__ == "__main__":
    main()