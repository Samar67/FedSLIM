import os
import ssl
import socket
import pickle
import argparse
import struct
import time
import math
import ipaddress
import numpy as np
import pandas as pd
import zlib

from collections import Counter, defaultdict
from itertools import chain

from sklearn.base import BaseEstimator
from skmine.base import TransformerMixin

from secure_aggregation import (
    generate_keypair,
    aggregate_masked_values
)
from privacy_utils import int64_to_uint32_safe


# =============================================================================
# Certificate Generation (kept from original)
# =============================================================================

CERTS_DIR = "certs"
CA_CERT = os.path.join(CERTS_DIR, "ca.crt")
CA_KEY = os.path.join(CERTS_DIR, "ca.key")
SERVER_CERT = os.path.join(CERTS_DIR, "server.crt")
SERVER_KEY = os.path.join(CERTS_DIR, "server.key")


def generate_ca(ca_cert_path, ca_key_path):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from datetime import datetime, timedelta
    
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedSLIM"),
        x509.NameAttribute(NameOID.COMMON_NAME, "FedSLIM Root CA"),
    ])
    
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                key_encipherment=False, content_commitment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    
    os.makedirs(os.path.dirname(ca_cert_path) if os.path.dirname(ca_cert_path) else ".", exist_ok=True)
    
    with open(ca_key_path, "wb") as f:
        f.write(ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    with open(ca_cert_path, "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
    
    print(f"Generated CA certificate: {ca_cert_path}")
    return ca_key, ca_cert

def load_ca(ca_cert_path, ca_key_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    
    with open(ca_key_path, "rb") as f:
        ca_key = serialization.load_pem_private_key(f.read(), password=None)
    with open(ca_cert_path, "rb") as f:
        ca_cert = x509.load_pem_x509_certificate(f.read())
    return ca_key, ca_cert

def generate_server_cert(ca_key, ca_cert, cert_path, key_path):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from datetime import datetime, timedelta
    
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    
    server_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedSLIM"),
        x509.NameAttribute(NameOID.COMMON_NAME, "FedSLIM Server"),
    ])
    
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    
    with open(key_path, "wb") as f:
        f.write(server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    with open(cert_path, "wb") as f:
        f.write(server_cert.public_bytes(serialization.Encoding.PEM))
    
    print(f"Generated server certificate: {cert_path}")

def generate_client_cert(ca_key, ca_cert, client_id, cert_path, key_path):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from datetime import datetime, timedelta
    
    client_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    
    client_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedSLIM"),
        x509.NameAttribute(NameOID.COMMON_NAME, f"FedSLIM Client {client_id}"),
    ])
    
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(client_name)
        .issuer_name(ca_cert.subject)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    
    with open(key_path, "wb") as f:
        f.write(client_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    with open(cert_path, "wb") as f:
        f.write(client_cert.public_bytes(serialization.Encoding.PEM))
    
    print(f"Generated client {client_id} certificate: {cert_path}")

def ensure_certificates(num_clients):
    os.makedirs(CERTS_DIR, exist_ok=True)
    
    if not os.path.exists(CA_CERT) or not os.path.exists(CA_KEY):
        print("CA certificates not found. Generating...")
        ca_key, ca_cert = generate_ca(CA_CERT, CA_KEY)
    else:
        print("Loading existing CA certificates...")
        ca_key, ca_cert = load_ca(CA_CERT, CA_KEY)
    
    if not os.path.exists(SERVER_CERT) or not os.path.exists(SERVER_KEY):
        print("Server certificates not found. Generating...")
        generate_server_cert(ca_key, ca_cert, SERVER_CERT, SERVER_KEY)
    
    for cid in range(num_clients):
        client_cert = os.path.join(CERTS_DIR, f"client{cid}.crt")
        client_key = os.path.join(CERTS_DIR, f"client{cid}.key")
        if not os.path.exists(client_cert) or not os.path.exists(client_key):
            print(f"Client {cid} certificates not found. Generating...")
            generate_client_cert(ca_key, ca_cert, cid, client_cert, client_key)
    
    print("All certificates ready.\n")

def get_server_ssl_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(SERVER_CERT, SERVER_KEY)
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


# =============================================================================
# Byte Tracking
# =============================================================================

class ByteTracker:
    def __init__(self):
        # With headers (4-byte length prefix + compressed data)
        self.bytes_sent_with_headers = 0
        self.bytes_received_with_headers = 0
        # Payload only (compressed data without header)
        self.payload_sent = 0
        self.payload_received = 0
        # Message counts
        self.messages_sent = 0
        self.messages_received = 0
    
    def add_sent(self, header_bytes, compressed_bytes, payload_bytes):
        self.bytes_sent_with_headers += header_bytes + compressed_bytes
        self.payload_sent += payload_bytes
        self.messages_sent += 1
    
    def add_received(self, header_bytes, compressed_bytes, payload_bytes):
        self.bytes_received_with_headers += header_bytes + compressed_bytes
        self.payload_received += payload_bytes
        self.messages_received += 1
    
    def print_statistics(self):
        total_with_headers = self.bytes_sent_with_headers + self.bytes_received_with_headers
        total_payload = self.payload_sent + self.payload_received
        header_overhead = total_with_headers - total_payload
        
        print("\n" + "=" * 55)
        print("COMMUNICATION STATISTICS")
        print("=" * 55)
        print()
        print("[Scenario 1] Total bytes WITH 4-byte headers:")
        print(f"  Bytes sent:     {self.bytes_sent_with_headers:,}")
        print(f"  Bytes received: {self.bytes_received_with_headers:,}")
        print(f"  TOTAL:          {total_with_headers:,}")
        print()
        print("[Scenario 2] Payload bytes ONLY (no headers):")
        print(f"  Payload sent:     {self.payload_sent:,}")
        print(f"  Payload received: {self.payload_received:,}")
        print(f"  TOTAL:            {total_payload:,}")
        print()
        print("[Summary]")
        print(f"  Messages sent:     {self.messages_sent}")
        print(f"  Messages received: {self.messages_received}")
        print(f"  Total Messages: {self.messages_sent + self.messages_received}")
        print(f"  Header overhead:   {header_overhead:,} bytes")
        print("=" * 55)

# Global byte tracker instance
byte_tracker = ByteTracker()


# =============================================================================
# Time Profiling
# =============================================================================

class TimeProfiler:
    def __init__(self):
        self.communication_time = 0.0  # send_message + recv_message
        self.secagg_time = 0.0         # aggregate_masked_values
        self.algorithm_time = 0.0       # SLIM computations
        self.total_time = 0.0           # Total wall-clock time
        
        # For tracking current operation
        self._current_start = None
        self._current_category = None
    
    def start(self, category: str):
        self._current_start = time.time()
        self._current_category = category
    
    def stop(self):
        if self._current_start is None:
            return 0.0
        
        elapsed = time.time() - self._current_start
        
        if self._current_category == 'communication':
            self.communication_time += elapsed
        elif self._current_category == 'secagg':
            self.secagg_time += elapsed
        elif self._current_category == 'algorithm':
            self.algorithm_time += elapsed
        
        self._current_start = None
        self._current_category = None
        return elapsed
    
    def add_communication(self, elapsed: float):
        self.communication_time += elapsed
    
    def add_secagg(self, elapsed: float):
        self.secagg_time += elapsed
    
    def add_algorithm(self, elapsed: float):
        self.algorithm_time += elapsed
    
    def set_total(self, total: float):
        self.total_time = total
    
    def print_statistics(self):
        # Calculate other/overhead time
        tracked_time = self.communication_time + self.secagg_time + self.algorithm_time
        other_time = max(0, self.total_time - tracked_time)
        
        print("\n" + "=" * 60)
        print("TIME PROFILING STATISTICS")
        print("=" * 60)
        print()
        print("[Time Breakdown]")
        print(f"  Communication (send/recv):  {self.communication_time:>10.2f}s  ({self._pct(self.communication_time)}%)")
        print(f"  SecAgg (aggregation):       {self.secagg_time:>10.2f}s  ({self._pct(self.secagg_time)}%)")
        print(f"  Algorithm (SLIM compute):   {self.algorithm_time:>10.2f}s  ({self._pct(self.algorithm_time)}%)")
        print(f"  Other (setup/overhead):     {other_time:>10.2f}s  ({self._pct(other_time)}%)")
        print(f"  ─────────────────────────────────────────")
        print(f"  TOTAL wall-clock time:      {self.total_time:>10.2f}s  (100.00%)")
        print()
        print("[For Paper Comparison]")
        print(f"  Pure algorithm time:        {self.algorithm_time:>10.2f}s")
        print(f"  FL overhead (comm+secagg):  {self.communication_time + self.secagg_time:>10.2f}s")
        print(f"  Overhead percentage:        {self._pct(self.communication_time + self.secagg_time):>10}%")
        print("=" * 60)
    
    def _pct(self, value: float) -> str:
        if self.total_time > 0:
            return f"{(value / self.total_time) * 100:.2f}"
        return "0.00"


# Global time profiler instance
time_profiler = TimeProfiler()

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
    comm_start = time.time()
    
    pickled_data = pickle.dumps((msg_type, payload), protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = zlib.compress(pickled_data, COMPRESSION_LEVEL)
    
    length_prefix = struct.pack('>I', len(compressed_data))
    sock.sendall(length_prefix)
    send_all_chunked(sock, compressed_data)
    
    # Track bytes: header (4 bytes) + compressed data, and payload (compressed only)
    byte_tracker.add_sent(4, len(compressed_data), len(compressed_data))
    
    # Track communication time
    time_profiler.add_communication(time.time() - comm_start)

def recv_message(sock):
    comm_start = time.time()
    
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
    
    # Track bytes: header (4 bytes) + compressed data, and payload (compressed only)
    byte_tracker.add_received(4, compressed_size, compressed_size)
    
    # Track communication time
    time_profiler.add_communication(time.time() - comm_start)
    
    return msg_type, payload

# =============================================================================
# SLIM_SERVER Class (based on fed_slim.py, with privacy modifications)
# =============================================================================

class SLIM_SERVER(BaseEstimator, TransformerMixin):
    def __init__(self, sprt_thr_prcntg=0.05, max_time=-1, pruning=True, items=None):
        self.global_usagetable = pd.Series(dtype="uint32")
        self.global_index = []  # Ordered list of all itemsets (global)
        self.converged = False

        self.sprt_thr_prcntg = sprt_thr_prcntg
        self.sprt_thr = 0
        self.total_trans = 0
        self.clients_num = 0

        self.tested_candidates = set()

        self.pruning = pruning
        self.max_time = max_time

        self.baseline_L = 0.0
        self.final_L = 0.0
        self.model_size_ = 0.0
        self.data_size_ = 0.0

        self.items = items

    def _standard_cover_key_S(self, itemset):
        return (
            -len(itemset),
            -int(self.global_usagetable.get(itemset, 0)),
            tuple(sorted(itemset))
        )

    def _log2(self, values) -> pd.Series:
        res_index = values.index if isinstance(values, pd.Series) else None
        res = np.zeros(len(values), dtype=np.float32)
        positive_mask = values > 0
        if np.any(positive_mask):
            res[positive_mask] = np.log2(values[positive_mask]).astype(np.float32)
        return pd.Series(res, index=res_index)

    def start(self, aggregated_usage, total_transactions, num_clients):
        self.clients_num = num_clients
        self.total_trans = total_transactions

        self.sprt_thr = math.ceil(self.sprt_thr_prcntg * self.total_trans)
        self.sprt_thr = 0 

        # Use aggregated usage directly (already summed via SecAgg)
        self.global_usagetable = aggregated_usage.astype("uint32")
        sorted_itemsets = sorted(self.global_usagetable.index, key=self._standard_cover_key_S)
        self.global_usagetable = self.global_usagetable.loc[sorted_itemsets]
        self.global_index = list(self.global_usagetable.index)

        codes = -self._log2(self.global_usagetable / self.global_usagetable.sum())
        self._starting_codes = codes
        self.model_size_ = 2 * codes.sum()
        self.data_size_ = (codes * self.global_usagetable).sum()
        self.baseline_L = self.model_size_ + self.data_size_

        print(f"  Initial model_size: {self.model_size_:.2f}")
        print(f"  Initial data_size: {self.data_size_:.2f}")
        print(f"  Initial baseline_L: {self.baseline_L:.2f}")

        flat_index = [
            list(idx)[0] if isinstance(idx, (tuple, frozenset, set)) else idx
            for idx in self._starting_codes.index
        ]

        tempo = self._starting_codes.copy()
        tempo.index = flat_index
        self._starting_codes_dict = tempo.to_dict()

    def process_aggregated_candidates(self, aggregated_values, global_candidate_index):
        final_results = []
        
        for i, (XY, X, Y) in enumerate(global_candidate_index):
            usage_XY = int(aggregated_values[i, 0])
            usage_X = int(aggregated_values[i, 1])
            usage_Y = int(aggregated_values[i, 2])
            
            if usage_XY > self.sprt_thr:
                final_results.append((XY, usage_XY, X, usage_X, Y, usage_Y))
        
        final_results.sort(key=lambda x: (-len(x[0]), -x[1], tuple(sorted(x[0]))))
        
        if len(final_results) == 0:
            self.converged = True
            self.final_L = self.model_size_ + self.data_size_

        return self._estimate_gain(final_results)

    def _estimate_gain(self, candidates):
        res = []

        total_old_countsum = int(self.global_usagetable.sum())
        total_old_num_codes = int((self.global_usagetable > 0).sum())

        for candi in candidates:
            XY, usage_XY, X, new_usage_X, Y, new_usage_Y = candi

            usage_XY = int(usage_XY)
            new_usage_X = int(new_usage_X)
            new_usage_Y = int(new_usage_Y)
            old_usage_X = int(self.global_usagetable.get(X, 0))
            old_usage_Y = int(self.global_usagetable.get(Y, 0))

            # Adjust for clients who didn't generate this candidate
            x_diff = old_usage_X - (usage_XY + new_usage_X)
            new_usage_X += x_diff
            y_diff = old_usage_Y - (usage_XY + new_usage_Y)
            new_usage_Y += y_diff

            new_countsum = total_old_countsum - usage_XY
            new_num_codes_with_non_zero_usage = total_old_num_codes + 1 \
                - (1 if new_usage_X == 0 else 0) - (1 if new_usage_Y == 0 else 0)

            log_values = self._log2(np.array([old_usage_X, old_usage_Y, usage_XY,
                                              new_usage_X, new_usage_Y, total_old_countsum, new_countsum]))

            gain_db_XY = -1 * (
                -usage_XY * log_values[2] - new_usage_X * log_values[3] + old_usage_X * log_values[0]
                - new_usage_Y * log_values[4] + old_usage_Y * log_values[1] + new_countsum *
                log_values[6] - total_old_countsum * log_values[5])

            gain_ct_XY = -log_values[2]
            old_Y_size_code = sum(self._starting_codes_dict.get(item, 0) for item in Y)
            old_X_size_code = sum(self._starting_codes_dict.get(item, 0) for item in X)

            if new_usage_X != old_usage_X:
                if new_usage_X != 0 and old_usage_X != 0:
                    gain_ct_XY -= log_values[3]
                    gain_ct_XY += log_values[0]
                elif old_usage_X == 0:
                    gain_ct_XY += old_X_size_code
                    gain_ct_XY -= log_values[3]
                elif new_usage_X == 0:
                    gain_ct_XY -= old_X_size_code
                    gain_ct_XY += log_values[0]

            if new_usage_Y != old_usage_Y:
                if new_usage_Y != 0 and old_usage_Y != 0:
                    gain_ct_XY -= log_values[4]
                    gain_ct_XY += log_values[1]
                elif old_usage_Y == 0:
                    gain_ct_XY += old_Y_size_code
                    gain_ct_XY -= log_values[4]
                elif new_usage_Y == 0:
                    gain_ct_XY += old_Y_size_code
                    gain_ct_XY -= log_values[1]

            gain_ct_XY += new_num_codes_with_non_zero_usage * log_values[6]
            gain_ct_XY -= total_old_num_codes * log_values[5]

            gain_XY = gain_db_XY - gain_ct_XY - min(old_X_size_code, old_Y_size_code)

            if gain_XY > 0:
                res.append((XY, gain_XY))

        return sorted(res, key=lambda e: e[1], reverse=True)

    def compute_sizes(self, usages):
        non_zero_usages = usages[usages > 0]

        if len(non_zero_usages) == 0:
            return 0.0, 0.0

        itemsets = non_zero_usages.index
        usages_arr = np.array(non_zero_usages, dtype=np.uint32)
        codes = -self._log2(usages_arr / usages_arr.sum())

        counts = Counter(chain(*itemsets))
        stand_codes_sum = sum(
            self._starting_codes_dict.get(item, 0) * ctr
            for item, ctr in counts.items()
        )

        model_size = stand_codes_sum + codes.sum()
        data_size = (codes * usages_arr).sum()

        return data_size, model_size

    def update(self, total_usages):
        """Update global usage table and index."""
        self.global_usagetable = total_usages.astype(np.uint32)
        sorted_itemsets = sorted(self.global_usagetable.index, key=self._standard_cover_key_S)
        self.global_usagetable = self.global_usagetable.loc[sorted_itemsets]
        self.global_index = list(self.global_usagetable.index)


# =============================================================================
# Server Orchestration
# =============================================================================

def run_server(host, port, num_clients, max_time = 86400):
    start_time = time.time()
    
    ensure_certificates(num_clients)
    ssl_context = get_server_ssl_context()
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(num_clients)
    
    print(f"Server listening on {host}:{port}")
    print(f"Waiting for {num_clients} clients to connect...")
    
    # ==========================================================================
    # Phase 1: Accept Connections and Collect ECDH Public Keys
    # ==========================================================================
    client_sockets = {}
    client_public_keys = {}
    client_transactions = {}
    
    while len(client_sockets) < num_clients:
        client_socket, addr = server_socket.accept()
        try:
            ssl_socket = ssl_context.wrap_socket(client_socket, server_side=True)
            
            msg_type, payload = recv_message(ssl_socket)
            if msg_type == MSG_REGISTER:
                cid = payload['cid']
                public_key = payload['public_key']
                nmbr_transaction = payload['nmbr_transaction']
                
                client_sockets[cid] = ssl_socket
                client_public_keys[cid] = public_key
                client_transactions[cid] = nmbr_transaction
                
                print(f"Client {cid} connected from {addr} ({len(client_sockets)}/{num_clients})")
        except ssl.SSLError as e:
            print(f"SSL Error from {addr}: {e}")
            client_socket.close()
    
    print(f"\nAll {num_clients} clients connected.")
    
    # ==========================================================================
    # Phase 2: Distribute All Public Keys (for SecAgg pairwise key derivation)
    # ==========================================================================
    print("Distributing ECDH public keys to all clients...")
    for cid, sock in client_sockets.items():
        send_message(sock, MSG_ALL_KEYS, {
            'all_public_keys': client_public_keys,
            'my_cid': cid
        })
    
    # ==========================================================================
    # Phase 3: Two-Round Initial Usage Collection
    # ==========================================================================
    print("\n--- Initial Usage: Two-Round SecAgg ---")
    
    # Round 1: Collect indexes only
    print("  Round 1: Collecting indexes...")
    for cid, sock in client_sockets.items():
        send_message(sock, MSG_REQUEST_INDEX, {'round': 0})
    
    all_indexes = []
    for cid, sock in client_sockets.items():
        msg_type, payload = recv_message(sock)
        if msg_type == MSG_INDEX_RESPONSE:
            all_indexes.extend(payload)  # List of frozensets
    
    # Compute global index (union of all, sorted)
    algo_start = time.time()
    global_index = sorted(set(all_indexes), key=lambda x: (-len(x), tuple(sorted(x))))
    time_profiler.add_algorithm(time.time() - algo_start)
    print(f"  Global index size: {len(global_index)}")
    
    # Round 2: Send global index, receive padded masked usages
    print("  Round 2: Collecting padded masked usages...")
    for cid, sock in client_sockets.items():
        send_message(sock, MSG_GLOBAL_INDEX, {
            'global_index': global_index,
            'round': 0
        })
    
    masked_usages = []
    for cid, sock in client_sockets.items():
        msg_type, payload = recv_message(sock)
        if msg_type == MSG_PADDED_USAGE:
            masked_usages.append(payload)  # pd.Series with global_index
    
    # Aggregate (SecAgg: masks cancel)
    secagg_start = time.time()
    aggregated_usage = aggregate_masked_values(masked_usages)
    aggregated_usage = int64_to_uint32_safe(aggregated_usage)
    time_profiler.add_secagg(time.time() - secagg_start)
    
    # Initialize server
    algo_start = time.time()
    total_transactions = sum(client_transactions.values())
    sl_server = SLIM_SERVER(max_time = max_time)
    sl_server.start(aggregated_usage, total_transactions, num_clients)
    time_profiler.add_algorithm(time.time() - algo_start)
    
    connection_setup = time.time()
    print(f"\nSetup complete. Time: {connection_setup - start_time:.2f}s")
    print("\nStarting federated SLIM algorithm...\n")
    
    # ==========================================================================
    # Phase 4: Main Algorithm Loop
    # ==========================================================================
    iteration = 0
    round_number = 1  # Start from 1 (0 was used for initial)
    
    while True:
        iteration += 1
        print(f"--- Iteration {iteration} ---")
        
        # ======================================================================
        # Step 1: Two-Round Candidate Generation
        # ======================================================================
        
        # Round 1: Collect candidate itemsets only (no values)
        print("  Candidates Round 1: Collecting itemsets...")
        for cid, sock in client_sockets.items():
            send_message(sock, MSG_REQUEST_CANDIDATE_ITEMSETS, {'round': round_number})
        
        all_candidate_structures = []  # List of (XY, X, Y) tuples
        for cid, sock in client_sockets.items():
            msg_type, payload = recv_message(sock)
            if msg_type == MSG_CANDIDATE_ITEMSETS:
                all_candidate_structures.extend(payload)
        
        # Compute global candidate index (unique (XY, X, Y) tuples)
        # Use XY as the key for deduplication
        algo_start = time.time()
        candidate_map = {}  # XY -> (X, Y)
        for (XY, X, Y) in all_candidate_structures:
            if XY not in candidate_map:
                candidate_map[XY] = (X, Y)
        
        global_candidate_index = [(XY, X, Y) for XY, (X, Y) in candidate_map.items()]
        global_candidate_index.sort(key=lambda x: (-len(x[0]), tuple(sorted(x[0]))))
        time_profiler.add_algorithm(time.time() - algo_start)
        
        if len(global_candidate_index) == 0:
            print("  No candidates generated. Converged.")
            sl_server.converged = True
            sl_server.final_L = sl_server.model_size_ + sl_server.data_size_
            break
        
        print(f"  Global candidate index size: {len(global_candidate_index)}")
        
        # Round 2: Send global candidate index, receive padded masked values
        print("  Candidates Round 2: Collecting padded masked values...")
        round_number += 1
        for cid, sock in client_sockets.items():
            send_message(sock, MSG_GLOBAL_CANDIDATE_INDEX, {
                'global_candidate_index': global_candidate_index,
                'round': round_number
            })
        
        masked_candidate_values = []
        for cid, sock in client_sockets.items():
            msg_type, payload = recv_message(sock)
            if msg_type == MSG_PADDED_CANDIDATES:
                masked_candidate_values.append(payload)  # np.array of shape (n_candidates, 3)
        
        # Aggregate candidate values (SecAgg: masks cancel)
        secagg_start = time.time()
        aggregated_candidates = np.sum(masked_candidate_values, axis=0)
        time_profiler.add_secagg(time.time() - secagg_start)
        
        # Process aggregated candidates
        algo_start = time.time()
        max_gain_sorted_candidates = sl_server.process_aggregated_candidates(
            aggregated_candidates, global_candidate_index
        )
        time_profiler.add_algorithm(time.time() - algo_start)
        
        if sl_server.converged:
            print("  No positive gain candidates. Converged.")
            break
        
        # ======================================================================
        # Step 2: Evaluate candidates one by one
        # ======================================================================
        accepted = False
        
        for candidate, gain in max_gain_sorted_candidates:
            # if candidate in sl_server.tested_candidates:
            #     continue
            sl_server.tested_candidates.add(candidate)
            
            round_number += 1
            
            # Broadcast evaluation request — client reconstructs eval_index locally
            # from its synchronised global_index copy, saving the full list per message.
            for cid, sock in client_sockets.items():
                send_message(sock, MSG_EVALUATE_CANDIDATE, {
                    'candidate': candidate,
                    'round': round_number
                })
            
            # Collect masked evaluation results
            evaluation_results = []
            for cid, sock in client_sockets.items():
                msg_type, payload = recv_message(sock)
                if msg_type == MSG_USAGE_RESULT:
                    evaluation_results.append(payload)
            
            # Aggregate (SecAgg: masks cancel)
            secagg_start = time.time()
            total_usages = aggregate_masked_values(evaluation_results)
            total_usages = int64_to_uint32_safe(total_usages)
            time_profiler.add_secagg(time.time() - secagg_start)
            
            # Calculate sizes and check gain
            algo_start = time.time()
            data_size, model_size = sl_server.compute_sizes(total_usages)
            diff = (sl_server.model_size_ + sl_server.data_size_) - (data_size + model_size)
            time_profiler.add_algorithm(time.time() - algo_start)
            
            if diff > 0:
                # Accept candidate
                algo_start = time.time()
                sl_server.update(total_usages)
                sl_server.model_size_ = model_size
                sl_server.data_size_ = data_size
                time_profiler.add_algorithm(time.time() - algo_start)
                
                # Notify clients to insert and update their local global_index copy.
                # No response expected — server already holds total_usages from evaluation.
                for cid, sock in client_sockets.items():
                    send_message(sock, MSG_ACCEPT_CANDIDATE, {
                        'candidate': candidate,
                        'new_global_index': sl_server.global_index
                    })
                
                print(f"  Accepted: {sorted(list(candidate))} (gain={diff:.2f})")
                print(f"  Total size = {sl_server.model_size_+sl_server.data_size_}")
                accepted = True
                break
            # Rejected candidates need no notification — clients have no state to update.
        
        if not accepted:
            sl_server.converged = True
            sl_server.final_L = sl_server.model_size_ + sl_server.data_size_
            print("  No candidate improved the model. Converged.")
            break
        
        if sl_server.max_time != -1 and time.time() - connection_setup > sl_server.max_time:
            print("Max time reached.")
            break
    
    # ==========================================================================
    # Termination
    # ==========================================================================
    print("\nTerminating clients...")
    for cid, sock in client_sockets.items():
        send_message(sock, MSG_TERMINATE, None)
        sock.close()
    
    server_socket.close()
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    l = 0
    for key, value in sl_server.global_usagetable.items():
        if value != 0:
            string_order = sorted(list(key))
            print(f"{l} - {string_order} : {value}")
            l += 1
    
    print(f"\nModel size = {sl_server.model_size_:.2f}")
    print(f"Data size = {sl_server.data_size_:.2f}")
    sl_server.final_L = sl_server.model_size_ + sl_server.data_size_
    print(f"Total size = {sl_server.final_L:.2f}")
    print(f"Total candidates evaluated = {len(sl_server.tested_candidates)}")
    
    if sl_server.baseline_L > 0:
        L_percentage = sl_server.final_L / sl_server.baseline_L * 100
        print(f"L% = {L_percentage:.2f}%")
    
    total_elapsed_time = time.time() - start_time
    run_time = time.time() - connection_setup
    print(f"\nTotal time: {total_elapsed_time:.2f} seconds")
    print(f"Run time: {run_time:.2f} seconds")
    
    # Print network communication statistics
    byte_tracker.print_statistics()
    
    # Print time profiling statistics
    time_profiler.set_total(run_time)
    time_profiler.print_statistics()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Federated SLIM Server with Mutual TLS")
    parser.add_argument("--num_clients", type=int, required=True, help="Number of clients to wait for")
    parser.add_argument("--dataset", type=str, required=True, help="which dataset?")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--max_time", type=int, default=86400, help="Maximum runtime in seconds (default: 86400 = 1 day)")

    args = parser.parse_args()
    
    print("=" * 60)
    print("Federated SLIM Server with Two-Round Secure Aggregation")
    print("=" * 60)
    print("\nPrivacy guarantees:")
    print("- Server does NOT know which client generated which candidate")
    print("- Server only sees aggregated (summed) usage values")
    print("- Two-round protocol ensures SecAgg masks cancel correctly")
    print()


    run_server(args.host, args.port, args.num_clients, args.max_time)


if __name__ == "__main__":
    main()
