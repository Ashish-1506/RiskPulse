"""One-command demo runner for the RiskPulse Platform.

Usage:
    python run_demo.py [--offline] [--no-ingestion]

Features:
- Starts FastAPI backend service on port 8000 (auto-detects next free port if occupied).
- Health-checks and awaits API readiness.
- Optionally starts real-time replay ingestion in background.
- Starts Streamlit dashboard on port 8501 (auto-detects next free port).
- Gracefully terminates all background child processes upon Ctrl+C.
- Provides seamless offline autonomy fallback.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import requests

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

CHILD_PROCESSES: List[subprocess.Popen] = []


def find_free_port(start_port: int) -> int:
    """Finds the next available local port starting from start_port."""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return start_port


def cleanup() -> None:
    """Cleanly terminates all spawned child subprocesses."""
    for proc in CHILD_PROCESSES:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def signal_handler(sig, frame) -> None:
    print("\n[RiskPulse] Shutting down all demo services cleanly...")
    cleanup()
    sys.exit(0)


def wait_for_api_healthy(api_url: str, timeout_sec: float = 20.0) -> bool:
    """Polls the FastAPI /health endpoint until it returns HTTP 200."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            resp = requests.get(f"{api_url}/health", timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            time.sleep(0.3)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="RiskPulse Live Demo Runner")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in strict offline mode using local datasets without external network requests.",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Initial port for FastAPI backend (default: 8000).",
    )
    parser.add_argument(
        "--st-port",
        type=int,
        default=8501,
        help="Initial port for Streamlit dashboard (default: 8501).",
    )
    args = parser.parse_args()

    # Register process cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 70)
    print("🛡️  RiskPulse AI Platform: Initializing Live Demo Environment")
    print("=" * 70)

    # 1. Resolve Ports
    api_port = find_free_port(args.api_port)
    if api_port != args.api_port:
        print(f"[Notice] Port {args.api_port} is busy. Switched API port to {api_port}.")

    st_port = find_free_port(args.st_port)
    if st_port != args.st_port:
        print(f"[Notice] Port {args.st_port} is busy. Switched Streamlit port to {st_port}.")

    api_url = f"http://127.0.0.1:{api_port}"
    env = os.environ.copy()
    env["API_URL"] = api_url
    env["PORT"] = str(api_port)
    env["PYTHONPATH"] = str(BASE_DIR)
    if args.offline:
        env["OFFLINE_MODE"] = "1"
        env["DISABLE_EXTERNAL_FEEDS"] = "1"

    # 2. Launch FastAPI backend
    print(f"[1/3] Launching FastAPI service on port {api_port}...")
    fastapi_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(api_port),
        "--log-level",
        "warning",
    ]
    fastapi_proc = subprocess.Popen(fastapi_cmd, env=env, cwd=str(BASE_DIR))
    CHILD_PROCESSES.append(fastapi_proc)

    # 3. Wait for API Health
    print(f"      Awaiting /health readiness at {api_url}...")
    if not wait_for_api_healthy(api_url, timeout_sec=15.0):
        print("[Warning] API did not respond within timeout. Dashboard will operate in direct offline mode.")
    else:
        print("      ✓ FastAPI service is ONLINE and healthy.")

    # 4. Launch Streamlit Dashboard
    print(f"[2/3] Launching Streamlit dashboard on port {st_port}...")
    st_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/dashboard/app.py",
        "--server.port",
        str(st_port),
        "--server.headless",
        "true",
        "--server.runOnSave",
        "false",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]
    st_proc = subprocess.Popen(st_cmd, env=env, cwd=str(BASE_DIR))
    CHILD_PROCESSES.append(st_proc)

    time.sleep(2.0)

    # 5. Summary Info
    mode_str = "OFFLINE REPLAY (Local Datasets Only)" if args.offline else "HYBRID ONLINE (FastAPI + Live Ingestion)"
    print("\n" + "=" * 70)
    print("🚀  RISKPULSE DEMO ENVIRONMENT READY!")
    print("=" * 70)
    print(f"  📊 Streamlit Dashboard:  http://localhost:{st_port}")
    print(f"  📑 Interactive API Docs:  {api_url}/docs")
    print(f"  💓 Service Health Probe:  {api_url}/health")
    print(f"  🔒 Execution Mode:        {mode_str}")
    print("=" * 70)
    print("Press Ctrl+C at any time to shut down all processes cleanly.\n")

    # Monitor subprocesses
    try:
        while True:
            time.sleep(1.0)
            for proc in CHILD_PROCESSES:
                if proc.poll() is not None:
                    # One process terminated
                    cleanup()
                    sys.exit(0)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
