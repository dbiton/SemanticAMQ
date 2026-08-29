"""
Small deterministic-hashing helpers shared by the LSH-based AMQ baselines
(DSBFFilter, LSBFFilter, ...). SHA-256-based so results are independent of
PYTHONHASHSEED, matching the approach already proven by
`VacuumFilter._hash_to_int`.
"""

import hashlib
import struct
import numpy as np


def stable_hash_to_int(key) -> int:
    """
    Deterministic 32-bit hash of `key`.

    `key` is normally a str, but LSH signatures can also arrive as raw bytes
    or numpy.void scalars (e.g. from SimHashLSH with L > 1), so any of those
    are accepted directly.
    """
    if isinstance(key, str):
        key_bytes = key.encode()
    elif isinstance(key, bytes):
        key_bytes = key
    elif isinstance(key, np.void):
        key_bytes = key.tobytes()
    else:
        key_bytes = bytes(key)
    digest = hashlib.sha256(key_bytes).digest()
    return struct.unpack('<I', digest[:4])[0]


def stable_bit_index(key, num_bits: int) -> int:
    """Map `key` to a bit position in [0, num_bits)."""
    return stable_hash_to_int(key) % num_bits


def signature_to_key(signature):
    """Convert an LSH signature tuple into a single filter key (a scalar
    inputs to stable_hash_to_int can hash directly, or a colon-joined string
    for multi-element tuples)."""
    if isinstance(signature, tuple):
        if len(signature) == 1:
            return signature[0]
        return ":".join(map(str, signature))
    return signature
