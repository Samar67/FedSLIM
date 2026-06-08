import os
import hashlib
import hmac
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend


CURVE = ec.SECP256R1()  # NIST P-256 curve
SEED_LENGTH = 32  # 256 bits
HASH_LENGTH = 32  # SHA-256 output


# =============================================================================
# ECDH Key Exchange
# =============================================================================

def generate_keypair() -> Tuple[ec.EllipticCurvePrivateKey, bytes]:
    private_key = ec.generate_private_key(CURVE, default_backend())
    public_key_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.CompressedPoint
    )
    return private_key, public_key_bytes


def deserialize_public_key(public_key_bytes: bytes) -> ec.EllipticCurvePublicKey:
    return ec.EllipticCurvePublicKey.from_encoded_point(CURVE, public_key_bytes)


def derive_shared_secret(my_private_key: ec.EllipticCurvePrivateKey, 
                         their_public_key_bytes: bytes) -> bytes:
    their_public_key = deserialize_public_key(their_public_key_bytes)
    shared_key = my_private_key.exchange(ec.ECDH(), their_public_key)
    
    # Derive a uniform seed using HKDF
    seed = HKDF(
        algorithm=hashes.SHA256(),
        length=SEED_LENGTH,
        salt=b"FedSLIM_SecAgg_v1",
        info=b"pairwise_seed",
        backend=default_backend()
    ).derive(shared_key)
    
    return seed


def derive_all_pairwise_seeds(my_cid: int, 
                               my_private_key: ec.EllipticCurvePrivateKey,
                               all_public_keys: Dict[int, bytes]) -> Dict[int, bytes]:
    seeds = {}
    for other_cid, pk_bytes in all_public_keys.items():
        if other_cid != my_cid:
            seeds[other_cid] = derive_shared_secret(my_private_key, pk_bytes)
    return seeds


# =============================================================================
# Pseudorandom Generator (PRG)
# =============================================================================

def expand_seed(seed: bytes, round_number: int, length: int) -> np.ndarray:
    # Create round-specific key
    round_key = hmac.new(
        seed, 
        f"round_{round_number}".encode(), 
        hashlib.sha256
    ).digest()
    
    # Generate enough bytes for the required length
    # Each int64 needs 8 bytes
    bytes_needed = length * 8
    blocks_needed = (bytes_needed + 31) // 32  # SHA256 outputs 32 bytes
    
    output_bytes = b""
    for counter in range(blocks_needed):
        block = hmac.new(
            round_key,
            counter.to_bytes(4, 'big'),
            hashlib.sha256
        ).digest()
        output_bytes += block
    
    # Convert to int64 array
    # Use int64 to allow for negative values after subtraction
    mask_array = np.frombuffer(output_bytes[:length * 8], dtype=np.int64)
    
    # Scale to reasonable range to avoid overflow when summing
    # Use modular arithmetic with a large prime
    PRIME = 2**31 - 1  # Mersenne prime
    mask_array = mask_array % PRIME
    
    return mask_array


def expand_seed_for_series(seed: bytes, round_number: int, 
                           index: pd.Index) -> pd.Series:
    mask_array = expand_seed(seed, round_number, len(index))
    return pd.Series(mask_array, index=index, dtype=np.int64)


# =============================================================================
# Commitment Scheme
# =============================================================================

def generate_nonce() -> bytes:
    return os.urandom(SEED_LENGTH)


def compute_commitment(data: bytes, nonce: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(data)
    h.update(nonce)
    return h.digest()


def verify_commitment(data: bytes, nonce: bytes, commitment: bytes) -> bool:
    expected = compute_commitment(data, nonce)
    return hmac.compare_digest(expected, commitment)


def serialize_for_commitment(obj: Any) -> bytes:
    import pickle
    # Use protocol 4 for deterministic output
    return pickle.dumps(obj, protocol=4)


# =============================================================================
# Masking Operations
# =============================================================================

class SecureAggregator:
    """
    Handles secure aggregation masking for a single client.
    
    The masking scheme ensures that when all masked values are summed,
    the masks cancel out and only the true aggregate remains.
    
    For clients i and j where i < j:
    - Client i ADDS mask_ij
    - Client j SUBTRACTS mask_ij
    
    When summed: mask_ij - mask_ij = 0
    """
    
    def __init__(self, my_cid: int, pairwise_seeds: Dict[int, bytes]):
        self.my_cid = my_cid
        self.pairwise_seeds = pairwise_seeds
        self.all_cids = sorted([my_cid] + list(pairwise_seeds.keys()))
    
    def compute_masked_value(self, value: pd.Series, round_number: int, 
                             sub_round: int = 0) -> pd.Series:
        # Start with the actual value, converted to int64 for masking
        masked = value.astype(np.int64).copy()
        
        # Combined round identifier
        combined_round = round_number * 1000 + sub_round
        
        # Apply masks for each pair
        for other_cid, seed in self.pairwise_seeds.items():
            mask = expand_seed_for_series(seed, combined_round, value.index)
            
            if self.my_cid < other_cid:
                # I'm the "lower" client, ADD the mask
                masked = masked + mask
            else:
                # I'm the "higher" client, SUBTRACT the mask
                masked = masked - mask
        
        return masked
    
    def compute_masked_tuple(self, values: Tuple, round_number: int,
                             sub_round: int = 0) -> Tuple:
        # Create a small Series for the tuple values
        index = pd.Index(range(len(values)))
        value_series = pd.Series(values, index=index, dtype=np.int64)
        
        masked_series = self.compute_masked_value(value_series, round_number, sub_round)
        
        return tuple(masked_series.values)


def aggregate_masked_values(masked_values: List[pd.Series]) -> pd.Series:
    if not masked_values:
        return pd.Series(dtype=np.int64)
    
    # Sum all masked values
    result = masked_values[0].copy()
    for mv in masked_values[1:]:
        result = result.add(mv, fill_value=0)
    
    # Convert back to uint32 for usage counts (masks should have canceled)
    # The result might be slightly off due to int64 arithmetic, but should be close
    return result.astype(np.int64)


def aggregate_masked_tuples(masked_tuples: List[Tuple]) -> Tuple:
    if not masked_tuples:
        return ()
    
    n_values = len(masked_tuples[0])
    result = [0] * n_values
    
    for mt in masked_tuples:
        for i, v in enumerate(mt):
            result[i] += v
    
    return tuple(result)


# =============================================================================
# Utility Functions
# =============================================================================

def create_client_key_message(cid: int, public_key_bytes: bytes) -> Dict:
    return {
        'cid': cid,
        'public_key': public_key_bytes
    }


def create_commitment_message(commitment: bytes) -> Dict:
    return {
        'commitment': commitment
    }


def create_masked_value_message(masked_value: pd.Series, nonce: bytes) -> Dict:
    return {
        'masked_value': masked_value,
        'nonce': nonce
    }
