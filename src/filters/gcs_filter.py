import math
from typing import List, Optional, Iterator
import numpy as np
from filters.filter import Filter
from lsh import LSH

class BitStreamWriter:
    """Fast bitwise packer to avoid massive Python lists during construction."""
    def __init__(self):
        self.data = bytearray()
        self.buffer = 0
        self.bits_in_buffer = 0

    def write_unary(self, quotient: int):
        for _ in range(quotient):
            self.buffer = (self.buffer << 1) | 1
            self.bits_in_buffer += 1
            if self.bits_in_buffer == 8:
                self.data.append(self.buffer)
                self.buffer = 0
                self.bits_in_buffer = 0
        # Write the terminating 0
        self.buffer <<= 1
        self.bits_in_buffer += 1
        if self.bits_in_buffer == 8:
            self.data.append(self.buffer)
            self.buffer = 0
            self.bits_in_buffer = 0

    def write_binary(self, value: int, p: int):
        for i in range(p - 1, -1, -1):
            self.buffer = (self.buffer << 1) | ((value >> i) & 1)
            self.bits_in_buffer += 1
            if self.bits_in_buffer == 8:
                self.data.append(self.buffer)
                self.buffer = 0
                self.bits_in_buffer = 0

    def flush(self):
        if self.bits_in_buffer > 0:
            self.buffer <<= (8 - self.bits_in_buffer)
            self.data.append(self.buffer)
            self.bits_in_buffer = 0

class BitStreamReader:
    """Fast bitwise reader for sequentially decoding the GCS."""
    def __init__(self, data: bytearray):
        self.data = data
        self.byte_idx = 0
        self.bit_idx = 7

    def read_bit(self) -> Optional[int]:
        if self.byte_idx >= len(self.data):
            return None
        bit = (self.data[self.byte_idx] >> self.bit_idx) & 1
        self.bit_idx -= 1
        if self.bit_idx < 0:
            self.bit_idx = 7
            self.byte_idx += 1
        return bit

    def read_binary(self, p: int) -> Optional[int]:
        val = 0
        for _ in range(p):
            b = self.read_bit()
            if b is None: return None
            val = (val << 1) | b
        return val


class GCSFilter(Filter):
    """
    A Golomb-Coded Set (GCS) filter: a space-optimal static filter.
    """
    def __init__(
        self,
        dim: int,
        lsh: Optional[LSH] = None,
        m: int = 256,
        max_memory: Optional[int] = None,
    ):
        super().__init__(dim, lsh, max_memory=max_memory)
        
        self._m = max(m, 2)
        self._p = 0
        self._n = 0
        self._modulus = 0
        self._constructed = False
        self._encoded_data: Optional[bytearray] = None

    def get_exact_matching_fpr(self) -> float:
        return 1.0 / self._m

    @staticmethod
    def _hash_to_range(key: str, modulus: int) -> int:
        combined = hash(f"{key}:gcs:0x9E3779B9")
        return abs(combined) % modulus

    def construct(self, vectors: np.ndarray, is_hashed: bool = False) -> None:
        hashes = self._normalize_hashes(vectors) if is_hashed else (self._normalize_hashes(self.lsh.hash(vectors)) if self.lsh else [[str(v)] for v in vectors])
        keys = list({key for signature_keys in hashes for key in signature_keys})
        self._n = len(keys)

        if self._n == 0:
            self._constructed = True
            self._encoded_data = bytearray()
            return

        # --- Dynamic max_memory enforcement ---
        if self.max_memory is not None:
            total_bits = self.max_memory * 8
            bits_per_item = total_bits / self._n
            if bits_per_item < 1.5:
                raise ValueError("max_memory is too small to store the dataset using GCS.")
            # Cap to avoid float/int overflow for very generous budgets relative
            # to n; an FPR of 1/2^62 is already far beyond any practical use.
            bits_per_item = min(bits_per_item, 62.0)
            # Reverse Golomb math: P = 2^(b - 1.4427)
            self._m = max(2, int(2 ** (bits_per_item - 1.4427)))

        self._modulus = self._n * self._m
        self._p = max(1, math.ceil(math.log2(self._m))) # Corrected Rice Math

        # Hash and Sort
        raw_hashes = sorted([self._hash_to_range(k, self._modulus) for k in keys])

        # Write directly to bitstream, NO saving raw_hashes to self
        writer = BitStreamWriter()
        prev = 0
        for h in raw_hashes:
            diff = (h - prev) % self._modulus
            prev = h
            
            quotient = diff >> self._p
            remainder = diff & ((1 << self._p) - 1)
            
            writer.write_unary(quotient)
            writer.write_binary(remainder, self._p)

        writer.flush()
        self._encoded_data = writer.data
        self._constructed = True

    def query(self, vectors: np.ndarray, is_hashed: bool = False) -> List[bool]:
        if not self._constructed:
            raise ValueError("GCSFilter has not been constructed.")

        if self._n == 0:
            return [False] * len(vectors)

        hashes = self._normalize_hashes(vectors) if is_hashed else (self._normalize_hashes(self.lsh.hash(vectors)) if self.lsh else [[str(v)] for v in vectors])
        results = []

        for signature_keys in hashes:
            found = False
            for key in signature_keys:
                target_val = self._hash_to_range(key, self._modulus)
                
                # We must decompress on the fly to save memory. 
                # This is O(N). GCS is meant for space, not speed.
                reader = BitStreamReader(self._encoded_data)
                current_val = 0
                
                for _ in range(self._n):
                    q = 0
                    while reader.read_bit() == 1:
                        q += 1
                        
                    r = reader.read_binary(self._p)
                    if r is None: break
                        
                    delta = (q << self._p) | r
                    current_val = (current_val + delta) % self._modulus
                    
                    if current_val == target_val:
                        found = True
                        break
                    elif current_val > target_val:
                        break # Because values were sorted
                if found:
                    break
            results.append(found)

        return results

    def add(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("GCS filters are static.")

    def remove(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("GCS filters are static.")

    def get_bytesize(self) -> int:
        if not self._constructed: return 0
        return len(self._encoded_data)