"""Decode vLLM's Marlin repacked weight/scale layout, and prove the formula.

Establishes, for 4-bit weights with 16-bit activations, exactly which (k, n)
each nibble of each packed uint32 holds -- which is what the GEMV kernel needs
in order to read the Marlin-repacked weights in place.
"""

import numpy as np
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
    marlin_weights,
)
from vllm.scalar_type import scalar_types

K, N = 256, 256
GROUP = 128


def predicted(word_in_row, nibble):
    """(k_local, n_in_block) for a nibble of word `word_in_row` of a k-tile row.

    word_in_row indexes uint32 within one packed row; rows are k-tiles of 16.
    Returns k_local in [0,16) and the absolute n.
    """
    group = word_in_row // 128          # each group of 128 words == 64 n
    g = word_in_row % 128
    i, j = g // 4, g % 4
    col = i // 4                        # 0..7
    kset = i % 4
    klocs = [2 * kset, 2 * kset + 1, 2 * kset + 8, 2 * kset + 9]
    n_tile = 4 * group + j              # already absolute within the row
    n0 = 16 * n_tile + col
    n1 = n0 + 8
    # interleave [0,2,4,6,1,3,5,7] applied to
    #   [ (k0,n0),(k1,n0),(k2,n0),(k3,n0), (k0,n1),(k1,n1),(k2,n1),(k3,n1) ]
    table = [(klocs[0], n0), (klocs[2], n0), (klocs[0], n1), (klocs[2], n1),
             (klocs[1], n0), (klocs[3], n0), (klocs[1], n1), (klocs[3], n1)]
    return table[nibble]


def main():
    perm = get_weight_perm(4, False)
    print("perm.numel() =", perm.numel())

    # --- index-tracing through the exact permutation pipeline ---------------
    idx = torch.arange(K * N).reshape(K, N)
    idx = idx.reshape(K // 16, 16, N // 16, 16).permute(0, 2, 1, 3)
    idx = idx.reshape(K // 16, N * 16)
    idx = idx.reshape((-1, perm.numel()))[:, perm].reshape(K // 16, N * 16)
    # idx[r, c] == original flat index k*N + n

    words_per_row = N * 16 // 8
    print("rows =", K // 16, " words/row =", words_per_row,
          " (== 2N:", words_per_row == 2 * N, ")")

    bad = 0
    # words repeat their (k_local, n-offset-within-64-block) pattern every 128
    for r in range(K // 16):
        for w in range(words_per_row):
            for nib in range(8):
                flat = idx[r, w * 8 + nib].item()
                k, n = flat // N, flat % N
                pk, pn = predicted(w, nib)
                pk_abs, pn_abs = 16 * r + pk, pn
                if (k, n) != (pk_abs, pn_abs):
                    if bad < 8:
                        print("MISMATCH r=%d w=%d nib=%d: real(k=%d,n=%d) "
                              "pred(k=%d,n=%d)" % (r, w, nib, k, n, pk_abs, pn_abs))
                    bad += 1
    print("weight-layout mismatches:", bad)

    # --- end-to-end: pack real values, unpack with the formula --------------
    g = torch.Generator().manual_seed(0)
    q = torch.randint(0, 16, (K, N), generator=g, dtype=torch.int32)
    packed = marlin_weights(q.clone(), K, N, 4, perm).numpy().astype(np.uint32)
    print("packed shape", packed.shape, "expect", (K // 16, 2 * N))

    rec = np.zeros((K, N), dtype=np.int32)
    for r in range(K // 16):
        for w in range(words_per_row):
            word = packed[r, w]
            for nib in range(8):
                v = (word >> (4 * nib)) & 0xF
                pk, pn = predicted(w, nib)
                rec[16 * r + pk, pn] = v
    print("unpack matches original q_w:", bool((rec == q.numpy()).all()))

    # --- scales -------------------------------------------------------------
    s = torch.arange(
        (K // GROUP) * N, dtype=torch.float32).reshape(K // GROUP, N)
    sp = marlin_permute_scales(s.clone(), K, N, GROUP)
    ok = True
    for row in range(K // GROUP):
        for n in range(N):
            # scale_perm is the 8x8 transpose within each 64-wide chunk, and it
            # is its own inverse
            p = (n // 64) * 64 + (n % 64 % 8) * 8 + (n % 64) // 8
            if sp[row, p].item() != s[row, n].item():
                ok = False
                break
    print("scale index formula  n -> (n/64)*64 + (n%%8)*8 + (n%%64)/8 :", ok)


if __name__ == "__main__":
    main()
