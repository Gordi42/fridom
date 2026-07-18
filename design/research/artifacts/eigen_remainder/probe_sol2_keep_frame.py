"""Solution 2 (keep-frame) micro-verification: pure numpy, single process.

Keep the half spectrum on the SHARDED axis b; a is the local axis.
Frame here: a = axis 0 (local, full spectrum), b = axis 1 (sharded, the
rfft half). We do NOT build a shard_map -- the collectives are emulated
on plain arrays (all_to_all is a no-op reshuffle in one process; the
collective_permute a-flip is (-ka) % na indexing).

Two claims:
(1) FORWARD: rfftn(u) == (fftn then slice the half axis), regardless of
    per-axis application order (a-first or b-first).
(2) BACKWARD via Hermitian reconstruction (option i): reconstruct the
    upper b-half as conj(C[(-ka)%na, nb-kb]) -- a FLIP across the
    sharded a-coefficient axis -- then local ifft_b, ifft_a, real part.
    The full forward->contract->backward chain must reproduce the
    replicated reference (rfftn / contract / irfftn) to < 1e-13, even
    for a NON-Hermitian (non-closed) contraction.
"""
import numpy as np

rng = np.random.default_rng(0)


def hermitian_reconstruct_a_flip(out_half, na, nb):
    """Fill the full b spectrum from the stored half via the a-flip.

    out_half: (na, nb//2+1, D) complex. Returns (na, nb, D): the upper
    b-half (kb = nb//2+1 .. nb-1) is conj(out_half[(-ka)%na, nb-kb]).
    This is exactly the conjugate symmetry irfftn uses internally
    (C[ka,kb] = conj(C[(-ka)%na, (-kb)%nb])).
    """
    na_idx = (-np.arange(na)) % na          # the collective_permute
    out_full = np.zeros((na, nb, out_half.shape[-1]), dtype=complex)
    out_full[:, : nb // 2 + 1] = out_half
    for kb in range(nb // 2 + 1, nb):
        out_full[:, kb] = np.conj(out_half[na_idx, nb - kb])
    return out_full


def run(na, nb, ndof, *, closed):
    u = rng.standard_normal((na, nb, ndof))          # real field
    # per-plane basis and weights (the contraction Q diag(w) Q^H M z)
    metric = rng.uniform(0.5, 1.5, ndof)
    q = (rng.standard_normal((na, nb // 2 + 1, ndof, ndof))
         + 1j * rng.standard_normal((na, nb // 2 + 1, ndof, ndof)))
    if closed:
        w = rng.standard_normal((na, nb // 2 + 1, ndof))  # real -> closed
    else:
        w = (rng.standard_normal((na, nb // 2 + 1, ndof))
             + 1j * rng.standard_normal((na, nb // 2 + 1, ndof)))

    def contract(z):
        amp = np.einsum("...dj,d,...d->...j", np.conj(q), metric, z)
        return np.einsum("...dj,...j->...d", q, w * amp)

    # -------- replicated reference (engine single-device path) --------
    z_ref = np.fft.rfftn(u, axes=(0, 1))             # half on axis 1 (b)
    out_ref = contract(z_ref)
    synth_ref = np.fft.irfftn(out_ref, s=(na, nb), axes=(0, 1)).real

    # -------- claim (1): fftn-then-slice == rfftn, order-free ----------
    ab = np.fft.fft(np.fft.fft(u, axis=0), axis=1)[:, : nb // 2 + 1]
    ba = np.fft.fft(np.fft.fft(u, axis=1), axis=0)[:, : nb // 2 + 1]
    fwd_a_first = np.abs(ab - z_ref).max()
    fwd_b_first = np.abs(ba - z_ref).max()

    # -------- Solution-2 keep-frame chain -----------------------------
    # forward: full fft on a (local), all_to_all (noop), full fft on b,
    # slice b to the half extent
    ca = np.fft.fft(u, axis=0)
    cab = np.fft.fft(ca, axis=1)
    z_half = cab[:, : nb // 2 + 1]
    out_half = contract(z_half)
    # backward: Hermitian reconstruct (a-flip), ifft_b, all_to_all
    # (noop), ifft_a, real part
    out_full = hermitian_reconstruct_a_flip(out_half, na, nb)
    ib = np.fft.ifft(out_full, axis=1)
    iba = np.fft.ifft(ib, axis=0)
    synth_sol2 = iba.real

    chain = np.abs(synth_sol2 - synth_ref).max()
    return fwd_a_first, fwd_b_first, chain


for na, nb in ((6, 8), (5, 8), (6, 7), (7, 9)):
    for closed in (True, False):
        fa, fb, ch = run(na, nb, 4, closed=closed)
        tag = "closed " if closed else "nonclos"
        print(f"na={na} nb={nb} {tag}: "
              f"fwd(a-first)={fa:.2e} fwd(b-first)={fb:.2e} "
              f"chain={ch:.2e} "
              f"{'PASS' if max(fa, fb, ch) < 1e-13 else 'FAIL'}")
