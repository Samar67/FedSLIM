import os
import ssl
import socket
import pickle
import struct
import argparse
import numpy as np
import pandas as pd

from functools import lru_cache, reduce
from sortedcontainers import SortedDict
from collections import defaultdict
from pyroaring import BitMap as Bitmap

from sklearn.base import BaseEstimator
from skmine.base import TransformerMixin

# =============================================================================
# Certificate Paths
# =============================================================================

CERTS_DIR = "certs"
CA_CERT = os.path.join(CERTS_DIR, "ca.crt")


def get_client_cert_paths(cid):
    """Get certificate and key paths for a specific client."""
    return (
        os.path.join(CERTS_DIR, f"client{cid}.crt"),
        os.path.join(CERTS_DIR, f"client{cid}.key")
    )


def get_client_ssl_context(cid):
    client_cert, client_key = get_client_cert_paths(cid)

    if not os.path.exists(client_cert) or not os.path.exists(client_key):
        raise FileNotFoundError(
            f"Client {cid} certificates not found at {client_cert} and {client_key}.\n"
            f"Please run the server first with --numClients {cid + 1} (or higher) to generate certificates."
        )

    if not os.path.exists(CA_CERT):
        raise FileNotFoundError(
            f"CA certificate not found at {CA_CERT}.\n"
            f"Please run the server first to generate certificates."
        )

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_cert_chain(client_cert, client_key)
    context.load_verify_locations(CA_CERT)
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = False
    return context


# =============================================================================
# Message Protocol
# =============================================================================

import zlib

MSG_REGISTER = "REGISTER"
MSG_REQUEST_CANDIDATES = "REQUEST_CANDIDATES"
MSG_CANDIDATES = "CANDIDATES"
MSG_EVALUATE_CANDIDATE = "EVALUATE_CANDIDATE"
MSG_USAGE_RESULT = "USAGE_RESULT"
MSG_ACCEPT_CANDIDATE = "ACCEPT_CANDIDATE"
MSG_TERMINATE = "TERMINATE"

CHUNK_SIZE = 65536
COMPRESSION_LEVEL = 6
MAX_MESSAGE_SIZE = 1 * 1024 * 1024 * 1024


def recv_all_chunked(sock, n):
    if n > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message size {n} exceeds maximum allowed {MAX_MESSAGE_SIZE}")

    chunks = []
    bytes_received = 0
    while bytes_received < n:
        chunk_size = min(CHUNK_SIZE, n - bytes_received)
        chunk = sock.recv(chunk_size)
        if not chunk:
            return None
        chunks.append(chunk)
        bytes_received += len(chunk)
    return b''.join(chunks)


def send_all_chunked(sock, data):
    total_sent = 0
    data_len = len(data)
    while total_sent < data_len:
        chunk_end = min(total_sent + CHUNK_SIZE, data_len)
        sent = sock.send(data[total_sent:chunk_end])
        if sent == 0:
            raise RuntimeError("Socket connection broken during send")
        total_sent += sent


def send_message(sock, msg_type, payload):
    pickled_data = pickle.dumps((msg_type, payload), protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = zlib.compress(pickled_data, COMPRESSION_LEVEL)

    original_size = len(pickled_data)
    compressed_size = len(compressed_data)

    if original_size > 1024 * 1024:
        ratio = (1 - compressed_size / original_size) * 100
        print(f"  [Network] Sending {msg_type}: {original_size / 1024 / 1024:.2f}MB -> {compressed_size / 1024 / 1024:.2f}MB ({ratio:.1f}% compression)")

    length_prefix = struct.pack('>I', compressed_size)
    sock.sendall(length_prefix)
    send_all_chunked(sock, compressed_data)


def recv_message(sock):
    raw_len = recv_all_chunked(sock, 4)
    if not raw_len:
        return None, None

    compressed_size = struct.unpack('>I', raw_len)[0]
    if compressed_size > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message size {compressed_size} exceeds maximum allowed {MAX_MESSAGE_SIZE}")

    compressed_data = recv_all_chunked(sock, compressed_size)
    if not compressed_data:
        return None, None

    try:
        pickled_data = zlib.decompress(compressed_data)
    except zlib.error as e:
        raise ValueError(f"Failed to decompress message: {e}")

    msg_type, payload = pickle.loads(pickled_data)
    return msg_type, payload


# =============================================================================
# Data Loading
# =============================================================================

def spaced_data(data_dir):
    with open(data_dir, 'r') as f:
        transactions = [line.strip().split() for line in f]
    return transactions


# =============================================================================
# SLIM_CLIENT Class
# =============================================================================

class SLIM_CLIENT(BaseEstimator, TransformerMixin):
    def __init__(self, cid, data, sprt_thr_prcntg=0.01, pruning=True, items=None):
        self.cid = cid
        self.data = data

        self.sprt_thr_prcntg = sprt_thr_prcntg
        self.sprt_thr = 1
        self.nmbr_transaction = 0

        self.singletons = dict()
        self.standard_codetable_ = pd.Series(dtype='object')
        self.codetable_ = SortedDict()
        self.usage_ = pd.Series(dtype=np.uint32)

        self.items = items
        self.pruning = pruning
        self._eval_cache = {}  # candidate -> CTc Bitmaps; avoids re-running _out_cover on acceptance

    def get_clientID(self):
        return self.cid

    def get_usage(self) -> pd.Series:
        return self.usage_

    def _to_vertical(self, D, stop_items=None, return_len=False):
        if stop_items is None:
            stop_items = set()
        res = defaultdict(Bitmap)
        idx = 0
        for idx, transaction in enumerate(D):
            for e in transaction:
                if e not in stop_items:
                    res[e].add(idx)
        if return_len:
            return dict(res), idx + 1
        return dict(res)

    @lru_cache(maxsize=1024)
    def _get_support(self, *items):
        a = items[-1]
        tids = self.standard_codetable_[a]
        if len(items) > 1:
            return tids & self._get_support(*items[:-1])
        else:
            return tids

    def _standard_cover_key_C(self, itemset):
        return (
            -len(itemset),
            -len(self._get_support(*itemset)),
            tuple(sorted(itemset))
        )

    def start(self):
        self.singletons, self.nmbr_transaction = self._to_vertical(self.data, return_len=True)

        self.standard_codetable_ = pd.Series(self.singletons)

        usage = self.standard_codetable_.map(len).astype(np.uint32)
        usage_redundant = usage.nlargest(len(self.standard_codetable_))
        self.standard_codetable_ = self.standard_codetable_[usage_redundant.index]

        usage.index = usage.index.map(lambda x: frozenset([x]))

        sorted_items = sorted(usage.index, key=self._standard_cover_key_C)

        usage = usage.loc[sorted_items]
        self.usage_ = usage

        ct_it = ((frozenset([e]), tids) for e, tids in self.standard_codetable_.items())
        self.codetable_ = SortedDict(self._standard_cover_key_C, ct_it)

    def generate_candidates(self):
        codetable = self.codetable_  # direct reference — generate_candidates only reads, never writes
        candidates = []

        stack = set(self.codetable_.keys())

        for idx, (x, x_usage) in enumerate(codetable.items()):
            Y = codetable.items()[idx + 1:]

            old_usage_X = len(codetable[x])
            if old_usage_X == 0:
                continue

            for y, y_usage in Y:
                XY = x.union(y)
                if XY in stack:
                    continue

                old_usage_Y = len(codetable[y])
                if old_usage_Y == 0:
                    continue

                new_usage_XY = y_usage.intersection_cardinality(x_usage)
                if new_usage_XY < self.sprt_thr:
                    continue

                new_usage_X = old_usage_X - new_usage_XY
                new_usage_Y = old_usage_Y - new_usage_XY

                stack.add(XY)

                candidates.append((XY, new_usage_XY, x, new_usage_X, y, new_usage_Y))

        return sorted(candidates, key=lambda e: e[1], reverse=True)

    def _out_cover(self, sct: dict, itemsets: list) -> dict:
        covers = dict()
        for iset in itemsets:
            it = [sct[i] for i in iset]
            usage = reduce(Bitmap.intersection, it).copy() if it else Bitmap()
            covers[iset] = usage
            for k in iset:
                sct[k] -= usage
        return covers

    def evaluate_candidate(self, candidate):
        idx = self.codetable_.bisect(candidate)

        ct = list(self.codetable_)
        ct.insert(idx, candidate)

        D = {k: v.copy() for k, v in self.standard_codetable_.items()}
        CTc = self._out_cover(D, ct)

        # Cache the full Bitmap dict so insert_new_candidate can reuse it
        # without re-running _out_cover with identical inputs.
        self._eval_cache[candidate] = CTc

        isets, usages = zip(*((_[0], len(_[1])) for _ in CTc.items() if len(_[1]) > 0 or len(_[0]) == 1))

        return pd.Series(data=usages, index=isets, dtype=np.uint32)

    def insert_new_candidate(self, candidate):
        if candidate in self._eval_cache:
            CTc = self._eval_cache.pop(candidate)
        else:
            idx = self.codetable_.bisect(candidate)
            ct = list(self.codetable_)
            ct.insert(idx, candidate)
            D = {k: v.copy() for k, v in self.standard_codetable_.items()}
            CTc = self._out_cover(D, ct)

        pruned_CTc = {
            k: v for k, v in CTc.items()
            if len(v) > 0 or len(k) == 1
        }

        self.codetable_.clear()
        self.codetable_.update(pruned_CTc)

        isets, usages = zip(*((k, len(v)) for k, v in self.codetable_.items()))
        self.usage_ = pd.Series(data=usages, index=isets, dtype=np.uint32)


# =============================================================================
# Client Communication Loop
# =============================================================================

def run_client(cid, data_path, host, port):
    print(f"Client {cid}: Loading data from {data_path}...")
    data = spaced_data(data_path)
    sl_client = SLIM_CLIENT(cid, data)
    sl_client.start()
    print(f"Client {cid}: Initialized with {sl_client.nmbr_transaction} transactions")

    print(f"Client {cid}: Loading certificates...")
    ssl_context = get_client_ssl_context(cid)
    client_cert, _ = get_client_cert_paths(cid)
    print(f"Client {cid}: Using certificate: {client_cert}")

    print(f"Client {cid}: Connecting to server at {host}:{port}...")
    raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ssl_socket = ssl_context.wrap_socket(raw_socket, server_hostname=host)
    ssl_socket.connect((host, port))

    server_cert = ssl_socket.getpeercert()
    if server_cert:
        subject = dict(x[0] for x in server_cert.get('subject', []))
        print(f"Client {cid}: Connected to server: {subject.get('commonName', 'Unknown')}")

    # ==========================================================================
    # Registration: Send exact initial usage
    # ==========================================================================
    send_message(ssl_socket, MSG_REGISTER, {
        'cid': cid,
        'usage': sl_client.get_usage(),
        'nmbr_transaction': sl_client.nmbr_transaction
    })
    print(f"Client {cid}: Registered with server (exact usage sent)")

    # ==========================================================================
    # Main Communication Loop
    # ==========================================================================
    while True:
        msg_type, payload = recv_message(ssl_socket)

        if msg_type is None:
            print(f"Client {cid}: Connection closed by server")
            break

        if msg_type == MSG_TERMINATE:
            print(f"Client {cid}: Received termination signal")
            break

        # ----------------------------------------------------------------------
        # Request for candidates with exact usages
        # ----------------------------------------------------------------------
        elif msg_type == MSG_REQUEST_CANDIDATES:
            candidates = sl_client.generate_candidates()
            send_message(ssl_socket, MSG_CANDIDATES, candidates)
            print(f"Client {cid}: Sent {len(candidates)} candidates")

        # ----------------------------------------------------------------------
        # Evaluate candidate — send exact result
        # ----------------------------------------------------------------------
        elif msg_type == MSG_EVALUATE_CANDIDATE:
            candidate = payload['candidate']
            usage_result = sl_client.evaluate_candidate(candidate)
            send_message(ssl_socket, MSG_USAGE_RESULT, usage_result)
            print(f"Client {cid}: Evaluated candidate {sorted(list(candidate))}")

        # ----------------------------------------------------------------------
        # Accept candidate — insert locally, no reply needed.
        # The server already cached the evaluation result as the updated usage.
        # ----------------------------------------------------------------------
        elif msg_type == MSG_ACCEPT_CANDIDATE:
            candidate = payload['candidate']
            sl_client.insert_new_candidate(candidate)
            print(f"Client {cid}: Accepted candidate {sorted(list(candidate))}")

    ssl_socket.close()
    print(f"Client {cid}: Disconnected")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Federated SLIM Client — Exact Usage Communication")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--data", type=str, required=True, help="Path to local data file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=6666, help="Server port")

    args = parser.parse_args()

    run_client(args.cid, args.data, args.host, args.port)


if __name__ == "__main__":
    main()
