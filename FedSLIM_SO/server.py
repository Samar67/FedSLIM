import os
import ssl
import socket
import pickle
import struct
import argparse
import time
import math
import ipaddress
import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from itertools import chain

from sklearn.base import BaseEstimator
from skmine.base import TransformerMixin

# =============================================================================
# Mutual TLS Certificate Generation
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


# =============================================================================
# Byte Tracking
# =============================================================================

class ByteTracker:
    def __init__(self):
        self.bytes_sent_with_headers = 0
        self.bytes_received_with_headers = 0
        self.payload_sent = 0
        self.payload_received = 0
        self.messages_sent = 0
        self.messages_received = 0

    def add_sent(self, payload_size):
        self.payload_sent += payload_size
        self.bytes_sent_with_headers += payload_size + 4
        self.messages_sent += 1

    def add_received(self, payload_size):
        self.payload_received += payload_size
        self.bytes_received_with_headers += payload_size + 4
        self.messages_received += 1

    def report(self):
        print("\n" + "=" * 55)
        print("COMMUNICATION STATISTICS")
        print("=" * 55)

        total_with_headers = self.bytes_sent_with_headers + self.bytes_received_with_headers
        print("\n[Scenario 1] Total bytes WITH 4-byte headers:")
        print(f"  Bytes sent:     {self.bytes_sent_with_headers:,}")
        print(f"  Bytes received: {self.bytes_received_with_headers:,}")
        print(f"  TOTAL:          {total_with_headers:,}")

        total_payload = self.payload_sent + self.payload_received
        print("\n[Scenario 2] Payload bytes ONLY (no headers):")
        print(f"  Payload sent:     {self.payload_sent:,}")
        print(f"  Payload received: {self.payload_received:,}")
        print(f"  TOTAL:            {total_payload:,}")

        header_overhead = (self.messages_sent + self.messages_received) * 4
        print(f"\n[Summary]")
        print(f"  Messages sent:     {self.messages_sent}")
        print(f"  Messages received: {self.messages_received}")
        print(f"  Total Messages: {self.messages_sent + self.messages_received}")
        print(f"  Header overhead:   {header_overhead:,} bytes")
        print("=" * 55)


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


def send_message(sock, msg_type, payload, tracker=None):
    pickled_data = pickle.dumps((msg_type, payload), protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = zlib.compress(pickled_data, COMPRESSION_LEVEL)

    original_size = len(pickled_data)
    compressed_size = len(compressed_data)

    if tracker:
        tracker.add_sent(compressed_size)

    if original_size > 1024 * 1024:
        ratio = (1 - compressed_size / original_size) * 100
        print(f"  [Network] Sending {msg_type}: {original_size / 1024 / 1024:.2f}MB -> {compressed_size / 1024 / 1024:.2f}MB ({ratio:.1f}% compression)")

    length_prefix = struct.pack('>I', compressed_size)
    sock.sendall(length_prefix)
    send_all_chunked(sock, compressed_data)


def recv_message(sock, tracker=None):
    raw_len = recv_all_chunked(sock, 4)
    if not raw_len:
        return None, None

    compressed_size = struct.unpack('>I', raw_len)[0]
    if compressed_size > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message size {compressed_size} exceeds maximum allowed {MAX_MESSAGE_SIZE}")

    if tracker:
        tracker.add_received(compressed_size)

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
# SLIM_SERVER Class
# =============================================================================

class SLIM_SERVER(BaseEstimator, TransformerMixin):
    def __init__(self, sprt_thr_prcntg=0.05, max_time=-1, pruning=True, items=None):
        self.global_usagetable = pd.Series(dtype="uint32")
        self.converged = False

        self.sprt_thr_prcntg = sprt_thr_prcntg
        self.sprt_thr = 0
        self.total_trans = 0
        self.clients_num = 0
        self.clients_usages = []
        self.clients_IDs = set()

        self.tested_candidated = dict()

        self.pruning = pruning
        self.max_time = max_time

        self.baseline_L = 0.0
        self.final_L = 0.0

        self.items = items

    def _standard_cover_key_S(self, itemset):
        return (
            -len(itemset),
            -int(self.global_usagetable[itemset]),
            tuple(sorted(itemset))
        )

    def _log2(self, values) -> pd.Series:
        res_index = values.index if isinstance(values, pd.Series) else None
        res = np.zeros(len(values), dtype=np.float32)
        positive_mask = values > 0
        if np.any(positive_mask):
            res[positive_mask] = np.log2(values[positive_mask]).astype(np.float32)
        return pd.Series(res, index=res_index)

    def start(self, clients_data):
        self.clients_num = len(clients_data)
        self.clients_usages = [None] * self.clients_num

        for cid, usage_, nmbr_transaction in clients_data:
            self.clients_usages[cid] = usage_
            self.clients_IDs.add(cid)
            self.total_trans += nmbr_transaction

        self.sprt_thr = math.ceil(self.sprt_thr_prcntg * self.total_trans)
        self.sprt_thr = 0

        self.global_usagetable = pd.concat(self.clients_usages, axis=1).sum(axis=1).astype("uint32")
        sorted_itemsets = sorted(self.global_usagetable.index, key=self._standard_cover_key_S)
        self.global_usagetable = self.global_usagetable.loc[sorted_itemsets]

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

    def agg_n_estimate_candidates(self, local_candidates):
        agg_map = defaultdict(lambda: {'xy_usage': 0, 'clients': set(), 'parents': defaultdict(int)})

        for cid, candidates in local_candidates:
            for entry in candidates:
                XY, local_usage_XY, X, local_usage_X, Y, local_usage_Y = entry

                agg_map[XY]['xy_usage'] += local_usage_XY
                agg_map[XY]['clients'].add(cid)
                agg_map[XY]['parents'][X] += local_usage_X
                agg_map[XY]['parents'][Y] += local_usage_Y

        final_results = []

        for XY, data in agg_map.items():
            total_usage_XY = data['xy_usage']

            if total_usage_XY > self.sprt_thr:
                parent_dict = data['parents']
                sorted_parents = sorted(parent_dict.keys(), key=lambda p: (-len(p), tuple(sorted(p))))

                if len(sorted_parents) == 2:
                    P1 = sorted_parents[0]
                    P2 = sorted_parents[1]

                    reconstructed_entry = (
                        XY,
                        total_usage_XY,
                        P1,
                        parent_dict[P1],
                        P2,
                        parent_dict[P2],
                        data['clients']
                    )
                    final_results.append(reconstructed_entry)

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
            XY, usage_XY, X, new_usage_X, Y, new_usage_Y, clients = candi

            usage_XY = int(usage_XY)
            new_usage_X = int(new_usage_X)
            new_usage_Y = int(new_usage_Y)
            old_usage_X = int(self.global_usagetable[X])
            old_usage_Y = int(self.global_usagetable[Y])

            if len(clients) < self.clients_num:
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
                res.append((XY, gain_XY, clients))

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
        self.global_usagetable = total_usages.astype(np.uint32)
        sorted_itemsets = sorted(self.global_usagetable.index, key=self._standard_cover_key_S)
        self.global_usagetable = self.global_usagetable.loc[sorted_itemsets]


# =============================================================================
# Server Orchestration
# =============================================================================

def run_server(host, port, num_clients, max_time = 86400):
    start_time = time.time()

    tracker = ByteTracker()

    ensure_certificates(num_clients)
    ssl_context = get_server_ssl_context()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(num_clients)

    print(f"Server listening on {host}:{port}")
    print(f"Waiting for {num_clients} clients to connect...")

    # ==========================================================================
    # Phase 1: Accept Connections and Registration
    # ==========================================================================
    client_sockets = {}
    clients_data = []  # List of (cid, exact_usage, nmbr_transaction)

    while len(client_sockets) < num_clients:
        client_socket, addr = server_socket.accept()
        try:
            ssl_socket = ssl_context.wrap_socket(client_socket, server_side=True)

            client_cert = ssl_socket.getpeercert()
            if client_cert:
                subject = dict(x[0] for x in client_cert.get('subject', []))
                cert_cn = subject.get('commonName', 'Unknown')
                print(f"Client authenticated via certificate: {cert_cn}")

            msg_type, payload = recv_message(ssl_socket, tracker)
            if msg_type == MSG_REGISTER:
                cid = payload['cid']
                usage = payload['usage']
                nmbr_transaction = payload['nmbr_transaction']

                client_sockets[cid] = ssl_socket
                clients_data.append((cid, usage, nmbr_transaction))
                print(f"Client {cid} connected from {addr} ({len(client_sockets)}/{num_clients})")
        except ssl.SSLError as e:
            print(f"SSL Error from {addr}: {e}")
            client_socket.close()

    connection_setup = time.time()
    connection_time = connection_setup - start_time
    print(f"\nAll {num_clients} clients connected. Time: {connection_time:.2f}s")

    sl_server = SLIM_SERVER(max_time=max_time)
    sl_server.start(clients_data)

    print("\nStarting federated SLIM algorithm...\n")

    # ==========================================================================
    # Phase 2: Main Algorithm Loop
    # ==========================================================================
    iteration = 0

    while True:
        iteration += 1
        print(f"--- Iteration {iteration} ---")

        # Step 1: Request candidates
        for cid, sock in client_sockets.items():
            send_message(sock, MSG_REQUEST_CANDIDATES, {'round': iteration}, tracker)

        local_candidates = []
        for cid, sock in client_sockets.items():
            msg_type, payload = recv_message(sock, tracker)
            if msg_type == MSG_CANDIDATES:
                local_candidates.append((cid, payload))

        max_gain_sorted_candidates = sl_server.agg_n_estimate_candidates(local_candidates)

        if sl_server.converged:
            print("No more candidates. Converged.")
            break

        accepted = False

        # Step 2: Evaluate candidates
        for candidate, gain, participant_clients in max_gain_sorted_candidates:
            sl_server.tested_candidated[candidate] = participant_clients

            for cid in participant_clients:
                send_message(client_sockets[cid], MSG_EVALUATE_CANDIDATE, {'candidate': candidate}, tracker)

            # Collect exact evaluation results from participating clients
            evaluation_results = {}
            for cid in participant_clients:
                msg_type, payload = recv_message(client_sockets[cid], tracker)
                if msg_type == MSG_USAGE_RESULT:
                    evaluation_results[cid] = payload

            # Aggregate results from participants
            participant_aggregate = pd.Series(dtype=np.float64)
            for cid, usage in evaluation_results.items():
                participant_aggregate = participant_aggregate.add(usage, fill_value=0)

            # Add cached usages from non-participating clients
            missing_clients = sl_server.clients_IDs - participant_clients
            total_usages = participant_aggregate.copy()

            for cid in missing_clients:
                if sl_server.clients_usages[cid] is not None:
                    total_usages = total_usages.add(sl_server.clients_usages[cid], fill_value=0)

            total_usages = total_usages.astype(np.uint32)
            data_size, model_size = sl_server.compute_sizes(total_usages)
            diff = (sl_server.model_size_ + sl_server.data_size_) - (data_size + model_size)

            if diff > 0:
                sl_server.update(total_usages)
                sl_server.model_size_ = model_size
                sl_server.data_size_ = data_size

                # Notify participating clients to accept (fire-and-forget — no reply expected)
                for cid in participant_clients:
                    send_message(client_sockets[cid], MSG_ACCEPT_CANDIDATE, {'candidate': candidate}, tracker)

                # Reuse evaluation results as the updated per-client usage tables.
                # evaluate_candidate() and insert_new_candidate() run the same cover
                # computation, so the evaluation result IS the post-insertion usage.
                for cid in participant_clients:
                    sl_server.clients_usages[cid] = evaluation_results[cid]

                print(f"  Accepted: {sorted(list(candidate))} (gain={diff:.2f})")
                print(f"  Total size = {sl_server.model_size_ + sl_server.data_size_}")
                accepted = True
                break

        if not accepted:
            sl_server.converged = True
            sl_server.final_L = sl_server.model_size_ + sl_server.data_size_
            print("No candidate improved the model. Converged.")
            break

        if sl_server.max_time != -1 and time.time() - connection_setup > sl_server.max_time:
            print("Max time reached.")
            break

    # ==========================================================================
    # Termination
    # ==========================================================================
    print("\nTerminating clients...")
    for cid, sock in client_sockets.items():
        send_message(sock, MSG_TERMINATE, None, tracker)
        sock.close()

    server_socket.close()

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
    print(f"Total candidates evaluated = {len(sl_server.tested_candidated)}")

    if sl_server.baseline_L > 0:
        L_percentage = sl_server.final_L / sl_server.baseline_L * 100
        print(f"L% = {L_percentage:.2f}%")
    else:
        print(f"L% = N/A (baseline_L = 0)")

    total_elapsed_time = time.time() - start_time
    run_time = time.time() - connection_setup
    print(f"\nTotal time: {total_elapsed_time:.2f} seconds")
    print(f"Run time: {run_time:.2f} seconds")

    tracker.report()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Federated SLIM Server — Exact Usage Communication")
    parser.add_argument("--numClients", type=int, required=True, help="Number of clients to wait for")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=6666, help="Server port")
    parser.add_argument("--max_time", type=int, default=86400, help="Maximum runtime in seconds (default: 86400 = 1 day)")
    
    args = parser.parse_args()

    run_server(args.host, args.port, args.numClients, args.max_time)


if __name__ == "__main__":
    main()
