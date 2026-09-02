#!/usr/bin/env python3
"""
Single entry point for the NEXYGEN project.

Usage:
    python run.py --env local  --app both        # uvicorn + streamlit as local subprocesses
    python run.py --env local  --app fastapi      # backend only, local
    python run.py --env local  --app streamlit    # frontend only, local (expects a backend already running)
    python run.py --env docker --app both         # docker compose up --build (both services)
    python run.py --env docker --app fastapi      # docker compose up --build backend
    python run.py --env docker --app streamlit    # docker compose up --build frontend (brings up backend too, via depends_on)
    python run.py --env docker --down             # docker compose down

Local mode runs processes directly with the interpreter/tools already on
PATH -- install backend/requirements.txt and
frontend/requirements.txt first. Docker mode only requires
Docker itself; dependencies are installed inside the images.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
COMPOSE_DIR = ROOT

BACKEND_PORT = "8000"
FRONTEND_PORT = "8501"


def _check_tool(name: str, hint: str):
    if shutil.which(name) is None:
        print(f"ERROR: '{name}' not found on PATH. {hint}", file=sys.stderr)
        sys.exit(1)


def run_local(app: str):
    procs = []
    try:
        if app in ("fastapi", "both"):
            _check_tool("uvicorn", "Install backend deps: pip install -r backend/requirements.txt")
            print(f"Starting FastAPI backend on :{BACKEND_PORT} ...")
            procs.append(subprocess.Popen(
                ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", BACKEND_PORT],
                cwd=str(BACKEND_DIR),
            ))
        if app in ("streamlit", "both"):
            _check_tool("streamlit", "Install frontend deps: pip install -r frontend/requirements.txt")
            print(f"Starting Streamlit frontend on :{FRONTEND_PORT} ...")
            # streamlit_app.py's own default for API_BASE_URL is
            # http://backend:8000 -- correct only inside Docker Compose's
            # network, where "backend" resolves via Docker's internal DNS.
            # There's no such hostname on a bare local machine, so local
            # mode needs to point it at localhost instead. setdefault (not
            # a hard override) means an explicitly-set API_BASE_URL --
            # from the shell or a loaded .env file -- still wins.
            frontend_env = os.environ.copy()
            frontend_env.setdefault("API_BASE_URL", f"http://localhost:{BACKEND_PORT}")
            procs.append(subprocess.Popen(
                ["streamlit", "run", "streamlit_app.py", "--server.port", FRONTEND_PORT, "--server.address", "0.0.0.0"],
                cwd=str(FRONTEND_DIR),
                env=frontend_env,
            ))

        print("Press Ctrl+C to stop." if procs else "Nothing to start.")
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def run_docker(app: str, down: bool):
    _check_tool("docker", "Install Docker: https://docs.docker.com/get-docker/")

    if down:
        subprocess.run(["docker", "compose", "down"], cwd=str(COMPOSE_DIR), check=False)
        return

    if app == "both":
        cmd = ["docker", "compose", "up", "--build"]
    else:
        # 'streamlit' also brings up 'backend' automatically (depends_on: service_healthy)
        service = "backend" if app == "fastapi" else "frontend"
        cmd = ["docker", "compose", "up", "--build", service]

    print(f"Running: {' '.join(cmd)} (in {COMPOSE_DIR})")
    subprocess.run(cmd, cwd=str(COMPOSE_DIR), check=False)


def main():
    parser = argparse.ArgumentParser(description="Run the NEXYGEN project")
    parser.add_argument("--env", choices=["local", "docker"], default="local")
    parser.add_argument("--app", choices=["fastapi", "streamlit", "both"], default="both")
    parser.add_argument("--down", action="store_true", help="(docker only) tear the stack down instead of starting it")
    args = parser.parse_args()

    if args.env == "local":
        if args.down:
            print("--down only applies to --env docker", file=sys.stderr)
            sys.exit(1)
        run_local(args.app)
    else:
        run_docker(args.app, args.down)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
