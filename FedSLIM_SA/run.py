import os
import sys
import time
import subprocess
from pathlib import Path
import argparse


# =========================
# Configuration
# =========================
parser = argparse.ArgumentParser(description="Federated SLIM Server with Mutual TLS")
parser.add_argument("--dataset", type=str, required=True, help="which dataset?")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument("--max_time", type=int, default=86400, help="Maximum runtime in seconds (default: 86400 = 1 day)")
parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
parser.add_argument("--folder_name", type=str, default="none", help="Folder Name")
 
args = parser.parse_args()

DATASET = args.dataset
PORT = args.port
HOST = args.host
MAX_TIME = args.max_time
NUM_CLIENTS = args.num_clients
DATA_FOLDER = args.folder_name

# DATASET = "chess_trial"
# NUM_CLIENTS = 3

BASE_DIR = Path(__file__).resolve().parent

SERVER_SCRIPT = "server.py"
CLIENT_SCRIPT = "client.py"

DATA_DIR = Path(f"./data/{DATA_FOLDER}")
# DATA_DIR = Path(f"./data/scalability_iid/{NUM_CLIENTS}")
# DATA_DIR = Path(f"./data/uneven_IID_10cl/{DATASET}")
# DATA_DIR = Path(f"./data/iid_10clients/{DATASET}")
# DATA_DIR = Path(f"./data/{DATASET}")
LOG_DIR = BASE_DIR / "logs"

PYTHON = sys.executable  # uses current Python interpreter

# =========================
# Setup
# =========================
LOG_DIR.mkdir(parents=True, exist_ok=True)

processes = []

try:
    # =========================
    # Start server
    # =========================
    # server_log = open(LOG_DIR / f"fed_server_{DATASET}.log", "w")
    server_log = open(LOG_DIR / f"fed_server_{DATA_FOLDER}.log", "w")

    server_cmd = [
        PYTHON, 
        str(BASE_DIR / SERVER_SCRIPT), 
        "--num_clients", str(NUM_CLIENTS),
        "--dataset", DATASET,
        "--host", HOST,
        "--port", str(PORT),
        "--max_time", str(MAX_TIME)
    ]

    print("Starting server...")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        cwd=BASE_DIR
    )
    processes.append((server_proc, server_log))

    # give server time to start
    time.sleep(20)

    # =========================
    # Start clients
    # =========================
    for cid in range(NUM_CLIENTS):
        data_file = DATA_DIR / f"cl{cid}.dat"
        # client_log = open(LOG_DIR / f"fed_cl{cid}_{DATASET}.log", "w")
        client_log = open(LOG_DIR / f"fed_cl{cid}_{DATA_FOLDER}.log", "w")


        client_cmd = [
            PYTHON, 
            str(BASE_DIR / CLIENT_SCRIPT), 
            "--cid", str(cid),
            "--data", str(data_file),
            "--host", HOST,
            "--port", str(PORT)
        ]

        print(f"Starting client {cid} with {data_file}")
        p = subprocess.Popen(
            client_cmd,
            stdout=client_log,
            stderr=subprocess.STDOUT,
            cwd=BASE_DIR
        )

        processes.append((p, client_log))

    # =========================
    # Wait for all processes
    # =========================
    for p, _ in processes:
        p.wait()

except KeyboardInterrupt:
    print("\nInterrupted. Terminating all processes...")

finally:
    # =========================
    # Cleanup
    # =========================
    for p, log in processes:
        if p.poll() is None:
            p.terminate()
        log.close()

    print("All processes terminated. Logs saved in:", LOG_DIR.resolve())