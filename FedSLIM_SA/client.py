import os
import ssl
import socket
import pickle
import struct
import argparse
import numpy as np
import pandas as pd
import zlib

from functools import lru_cache, reduce
from sortedcontainers import SortedDict
from collections import defaultdict
from pyroaring import BitMap as Bitmap

from sklearn.base import BaseEstimator
from skmine.base import TransformerMixin

from secure_aggregation import (
    generate_keypair,
    derive_all_pairwise_seeds,
    SecureAggregator
)

# =============================================================================
# Certificate Paths
# =============================================================================

CERTS_DIR = "certs"
CA_CERT = os.path.join(CERTS_DIR, "ca.crt")


def get_client_cert_paths(cid):
    return (
        os.path.join(CERTS_DIR, f"client{cid}.crt"),
        os.path.join(CERTS_DIR, f"client{cid}.key")
    )


def get_client_ssl_context(cid):
    client_cert, client_key = get_client_cert_paths(cid)
    
    if not os.path.exists(client_cert) or not os.path.exists(client_key):
        raise FileNotFoundError(
            f"Client {cid} certificates not found at {client_cert} and {client_key}.\n"
            f"Please run the server first to generate certificates."
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

# Message Types
MSG_REGISTER = "REGISTER"
MSG_ALL_KEYS = "ALL_KEYS"
MSG_REQUEST_INDEX = "REQUEST_INDEX"
MSG_INDEX_RESPONSE = "INDEX_RESPONSE"
MSG_GLOBAL_INDEX = "GLOBAL_INDEX"
MSG_PADDED_USAGE = "PADDED_USAGE"
MSG_REQUEST_CANDIDATE_ITEMSETS = "REQUEST_CANDIDATE_ITEMSETS"
MSG_CANDIDATE_ITEMSETS = "CANDIDATE_ITEMSETS"
MSG_GLOBAL_CANDIDATE_INDEX = "GLOBAL_CANDIDATE_INDEX"
MSG_PADDED_CANDIDATES = "PADDED_CANDIDATES"
MSG_EVALUATE_CANDIDATE = "EVALUATE_CANDIDATE"
MSG_USAGE_RESULT = "USAGE_RESULT"
MSG_ACCEPT_CANDIDATE = "ACCEPT_CANDIDATE"
MSG_REJECT_CANDIDATE = "REJECT_CANDIDATE"
MSG_TERMINATE = "TERMINATE"

# Protocol constants
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
    
    length_prefix = struct.pack('>I', len(compressed_data))
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
# SLIM_CLIENT Class (based on fed_slim.py, with Two-Round SecAgg)
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

        self.generated_candidates = []
        self.generated_candidate_set = set()  # O(1) existence check (see _candidate_exist)
        self._eval_cache = {}  # candidate -> CTc Bitmaps; avoids re-running _out_cover on acceptance

        self.items = items
        self.pruning = pruning

        
        self.private_key = None
        self.public_key_bytes = None
        self.pairwise_seeds = None
        self.secure_aggregator = None

    def get_clientID(self):
        return self.cid
    
    def setup_secagg(self, all_public_keys):
        self.pairwise_seeds = derive_all_pairwise_seeds(
            self.cid, self.private_key, all_public_keys
        )
        self.secure_aggregator = SecureAggregator(self.cid, self.pairwise_seeds)
        print(f"Client {self.cid}: SecAgg setup complete with {len(self.pairwise_seeds)} peers")

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
        """Get support from an itemset."""
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
        # Generate ECDH keypair for SecAgg
        self.private_key, self.public_key_bytes = generate_keypair()
        
        # Initialize data structures (from fed_slim.py)
        self.singletons, self.nmbr_transaction = self._to_vertical(self.data, return_len=True)
        self.sprt_thr = 1  

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

    def get_local_index(self):
        return list(self.usage_.index)

    def get_padded_masked_usage(self, global_index, round_number):
        padded_series = self.usage_.reindex(global_index, fill_value=0).astype(np.int64)
        masked = self.secure_aggregator.compute_masked_value(padded_series, round_number)
        return masked

    def generate_candidate(self):
        codetable = self.codetable_  # direct reference — generate_candidate only reads, never writes
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

        self.generated_candidates = sorted(candidates, key=lambda e: e[1], reverse=True)
        self.generated_candidate_set = {c[0] for c in self.generated_candidates}
        return self.generated_candidates

    def get_candidate_itemsets(self):
        candidates = self.generate_candidate()
        return [(c[0], c[2], c[4]) for c in candidates]  # (XY, X, Y)

    def get_padded_masked_candidates(self, global_candidate_index, round_number):
        # Build lookup from local candidates
        local_candidate_map = {}  # XY -> (usage_XY, usage_X, usage_Y)
        for (XY, usage_XY, X, usage_X, Y, usage_Y) in self.generated_candidates:
            local_candidate_map[XY] = (usage_XY, usage_X, usage_Y)
        
        # Create padded array
        n_candidates = len(global_candidate_index)
        padded_array = np.zeros((n_candidates, 3), dtype=np.int64)
        
        for i, (XY, X, Y) in enumerate(global_candidate_index):
            if XY in local_candidate_map:
                usage_XY, usage_X, usage_Y = local_candidate_map[XY]
                padded_array[i, 0] = usage_XY
                padded_array[i, 1] = usage_X
                padded_array[i, 2] = usage_Y
            # else: zeros (already initialized)
        
        # Flatten for masking, then reshape
        flat_series = pd.Series(padded_array.flatten(), dtype=np.int64)
        masked_flat = self.secure_aggregator.compute_masked_value(flat_series, round_number)
        masked_array = np.array(masked_flat).reshape((n_candidates, 3))
        
        return masked_array

    def _out_cover(self, sct: dict, itemsets: list) -> dict:
        covers = dict()
        for iset in itemsets:
            it = [sct[i] for i in iset]
            usage = reduce(Bitmap.intersection, it).copy() if it else Bitmap()
            covers[iset] = usage
            for k in iset:
                sct[k] -= usage
        return covers

    def _candidate_exist(self, candidate):
        return candidate in self.generated_candidate_set

    def evaluate_candidate(self, candidate):
        if not self._candidate_exist(candidate):
            return self.usage_

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

    def evaluate_candidate_padded_masked(self, candidate, eval_index, round_number):
        usage_series = self.evaluate_candidate(candidate)
        padded_series = usage_series.reindex(eval_index, fill_value=0).astype(np.int64)
        masked = self.secure_aggregator.compute_masked_value(padded_series, round_number)
        return masked

    def insert_new_candidate(self, candidate):
        if not self._candidate_exist(candidate):
            return

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
    # Load data and initialize client
    print(f"Client {cid}: Loading data from {data_path}...")
    data = spaced_data(data_path)
    sl_client = SLIM_CLIENT(cid, data)
    sl_client.start()
    print(f"Client {cid}: Initialized with {sl_client.nmbr_transaction} transactions")
    print(f"Client {cid}: Generated ECDH keypair for SecAgg")
    
    # Setup SSL context
    ssl_context = get_client_ssl_context(cid)
    
    # Connect to server
    print(f"Client {cid}: Connecting to server at {host}:{port}...")
    raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ssl_socket = ssl_context.wrap_socket(raw_socket, server_hostname=host)
    ssl_socket.connect((host, port))
    print(f"Client {cid}: Connected to server")
    
    # ==========================================================================
    # Phase 1: Registration with ECDH Public Key
    # ==========================================================================
    send_message(ssl_socket, MSG_REGISTER, {
        'cid': cid,
        'public_key': sl_client.public_key_bytes,
        'nmbr_transaction': sl_client.nmbr_transaction
    })
    print(f"Client {cid}: Sent registration with ECDH public key")
    
    # ==========================================================================
    # Phase 2: Receive All Public Keys and Setup SecAgg
    # ==========================================================================
    msg_type, payload = recv_message(ssl_socket)
    if msg_type == MSG_ALL_KEYS:
        all_public_keys = payload['all_public_keys']
        sl_client.setup_secagg(all_public_keys)
    
    # ==========================================================================
    # Phase 3: Main Communication Loop
    # ==========================================================================
    client_global_index = None  # synchronised copy of server's global_index
    
    while True:
        msg_type, payload = recv_message(ssl_socket)
        
        if msg_type is None:
            print(f"Client {cid}: Connection closed by server")
            break
        
        if msg_type == MSG_TERMINATE:
            print(f"Client {cid}: Received termination signal")
            break
        
        # ----------------------------------------------------------------------
        # Initial Usage: Round 1 - Send index only
        # ----------------------------------------------------------------------
        elif msg_type == MSG_REQUEST_INDEX:
            local_index = sl_client.get_local_index()
            send_message(ssl_socket, MSG_INDEX_RESPONSE, local_index)
            print(f"Client {cid}: Sent local index ({len(local_index)} itemsets)")
        
        # ----------------------------------------------------------------------
        # Initial Usage: Round 2 - Send padded masked usage
        # ----------------------------------------------------------------------
        elif msg_type == MSG_GLOBAL_INDEX:
            global_index = payload['global_index']
            round_number = payload['round']
            
            client_global_index = global_index  # initialise local copy
            padded_masked = sl_client.get_padded_masked_usage(global_index, round_number)
            send_message(ssl_socket, MSG_PADDED_USAGE, padded_masked)
            print(f"Client {cid}: Sent padded masked usage (global size: {len(global_index)})")
        
        # ----------------------------------------------------------------------
        # Candidates: Round 1 - Send itemsets only
        # ----------------------------------------------------------------------
        elif msg_type == MSG_REQUEST_CANDIDATE_ITEMSETS:
            candidate_itemsets = sl_client.get_candidate_itemsets()
            send_message(ssl_socket, MSG_CANDIDATE_ITEMSETS, candidate_itemsets)
            print(f"Client {cid}: Sent {len(candidate_itemsets)} candidate itemsets")
        
        # ----------------------------------------------------------------------
        # Candidates: Round 2 - Send padded masked values
        # ----------------------------------------------------------------------
        elif msg_type == MSG_GLOBAL_CANDIDATE_INDEX:
            global_candidate_index = payload['global_candidate_index']
            round_number = payload['round']
            
            padded_masked = sl_client.get_padded_masked_candidates(global_candidate_index, round_number)
            send_message(ssl_socket, MSG_PADDED_CANDIDATES, padded_masked)
            print(f"Client {cid}: Sent padded masked candidates (global size: {len(global_candidate_index)})")
        
        # ----------------------------------------------------------------------
        # Evaluate candidate
        # ----------------------------------------------------------------------
        elif msg_type == MSG_EVALUATE_CANDIDATE:
            candidate = payload['candidate']
            round_number = payload['round']
            
            # Reconstruct eval_index locally — avoids server sending the full list
            # each round-trip. Uses the same sort key as the server.
            eval_index = list(client_global_index)
            if candidate not in eval_index:
                eval_index.append(candidate)
            eval_index = sorted(eval_index, key=lambda x: (-len(x), tuple(sorted(x))))
            
            masked_result = sl_client.evaluate_candidate_padded_masked(candidate, eval_index, round_number)
            send_message(ssl_socket, MSG_USAGE_RESULT, masked_result)
            
            exists_locally = sl_client._candidate_exist(candidate)
            status = "evaluated" if exists_locally else "not local"
            print(f"Client {cid}: {status} candidate {sorted(list(candidate))}")
        
        # ----------------------------------------------------------------------
        # Accept candidate
        # ----------------------------------------------------------------------
        elif msg_type == MSG_ACCEPT_CANDIDATE:
            candidate = payload['candidate']
            client_global_index = payload['new_global_index']  # keep local copy in sync
            
            sl_client.insert_new_candidate(candidate)
            # No reply needed; server reuses the evaluation aggregate as updated usage.
            
            exists_locally = sl_client._candidate_exist(candidate)
            status = "inserted" if exists_locally else "ignored"
            print(f"Client {cid}: {status} candidate {sorted(list(candidate))}")
    
    ssl_socket.close()
    print(f"Client {cid}: Disconnected")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Federated SLIM Client with Two-Round Secure Aggregation")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--data", type=str, required=True, help="Path to local data file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Federated SLIM Client {args.cid} with Two-Round Secure Aggregation")
    print("=" * 60)
    print("\nPrivacy guarantees:")
    print("- All usage values are SecAgg-masked before sending")
    print("- Global index padding ensures masks cancel correctly")
    print("- Server only sees aggregated sums")
    print()
    
    run_client(args.cid, args.data, args.host, args.port)


if __name__ == "__main__":
    main()
