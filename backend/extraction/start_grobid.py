"""
start_grobid.py — starts the Grobid server via Docker.

Usage:
  python start_grobid.py           # start and wait until ready
  python start_grobid.py --stop    # stop the container
  python start_grobid.py --status  # print whether Grobid is reachable

Grobid is used by heuristic.py for citation extraction.  Run this once
before running the extraction pipeline; the container stays alive until
you stop it or reboot.
"""

import argparse
import subprocess
import sys
import time

import requests

GROBID_IMAGE     = "lfoppiano/grobid:0.8.1"
CONTAINER_NAME   = "knowledgeBank-grobid"
GROBID_HOST      = "http://localhost:8070"
HEALTH_ENDPOINT  = f"{GROBID_HOST}/api/isalive"
READY_TIMEOUT_S  = 120
POLL_INTERVAL_S  = 3


def is_running() -> bool:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def container_exists() -> bool:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return CONTAINER_NAME in result.stdout


def start() -> None:
    if is_running():
        print(f"[grobid] Already running at {GROBID_HOST}")
        return

    if container_exists():
        print(f"[grobid] Container '{CONTAINER_NAME}' exists but is stopped — starting it ...")
        subprocess.run(["docker", "start", CONTAINER_NAME], check=True)
    else:
        print(f"[grobid] Pulling {GROBID_IMAGE} and starting container ...")
        subprocess.run([
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", "8070:8070",
            "-p", "8071:8071",
            GROBID_IMAGE,
        ], check=True)

    print(f"[grobid] Waiting for Grobid to be ready (up to {READY_TIMEOUT_S}s) ...")
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if is_running():
            print(f"[grobid] Ready at {GROBID_HOST}")
            return
        time.sleep(POLL_INTERVAL_S)

    print("[grobid] ERROR: Grobid did not become ready in time.", file=sys.stderr)
    sys.exit(1)


def stop() -> None:
    if container_exists():
        subprocess.run(["docker", "stop", CONTAINER_NAME], check=True)
        print(f"[grobid] Container '{CONTAINER_NAME}' stopped.")
    else:
        print(f"[grobid] Container '{CONTAINER_NAME}' not found.")


def status() -> None:
    if is_running():
        print(f"[grobid] Running — {GROBID_HOST}")
    else:
        print(f"[grobid] Not reachable at {GROBID_HOST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the Grobid Docker container")
    parser.add_argument("--stop",   action="store_true", help="Stop the running container")
    parser.add_argument("--status", action="store_true", help="Check whether Grobid is reachable")
    args = parser.parse_args()

    if args.stop:
        stop()
    elif args.status:
        status()
    else:
        start()
