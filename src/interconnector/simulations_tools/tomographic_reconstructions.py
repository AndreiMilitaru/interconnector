"""
Author: Andrei Militaru
Description: This module contains tools for simulating tomographic reconstructions with readout confusion.
date: April 2026
"""

import numpy as np
import qutip as qt


def _gell_mann_basis(d):
    """
    Return the d^2 generalised Gell-Mann matrices for dimension d,
    with the identity as the zeroth element.

    The basis is orthogonal under the Hilbert-Schmidt inner product:
        Tr(F_a F_b) = n_a * delta_ab
    with n_0 = d (identity) and n_a = 2 for a >= 1.
    """
    basis = [np.eye(d, dtype=complex)]
    # Symmetric off-diagonal: |j><k| + |k><j|
    for j in range(d):
        for k in range(j + 1, d):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = 1.0
            m[k, j] = 1.0
            basis.append(m)
    # Antisymmetric off-diagonal: -i|j><k| + i|k><j|
    for j in range(d):
        for k in range(j + 1, d):
            m = np.zeros((d, d), dtype=complex)
            m[j, k] = -1j
            m[k, j] = 1j
            basis.append(m)
    # Diagonal: sqrt(2/(l(l+1))) * (sum_{m<l} |m><m| - l*|l><l|)
    for l in range(1, d):
        m = np.zeros((d, d), dtype=complex)
        c = np.sqrt(2.0 / (l * (l + 1)))
        for j in range(l):
            m[j, j] = c
        m[l, l] = -l * c
        basis.append(m)
    return basis


def apply_readout_confusion(rho, confusion_matrix, confused_subsystem=0,
                            trace_out=None):
    """
    Apply readout confusion to a density matrix, returning the state that
    a standard tomography experiment would reconstruct.

    This function implements confusion via Gell-Mann tomographic
    reconstruction.  The idea mirrors what an experimentalist actually
    does:

      1. Choose an operator basis {F_a} for d x d Hermitian matrices
         (generalised Gell-Mann matrices + identity).

      2. For each F_a with spectral decomposition F_a = U_a^dag D_a U_a,
         rotate the state to the eigenbasis and extract the diagonal:

             p_j^(a) = (U_a rho U_a^dag)_jj                -- Eq. (2)

      3. Apply the confusion matrix to the outcome probabilities:

             p~_k^(a) = sum_j  C[k,j] p_j^(a)              -- Eq. (3)

      4. Compute the apparent (measured) expectation value:

             <F_a>_meas = sum_k  e_k^(a) * p~_k^(a)        -- Eq. (4)

         where e_k^(a) are the eigenvalues of F_a.

      5. Reconstruct the measured density matrix:

             rho_meas = sum_a  <F_a>_meas / n_a  *  F_a     -- Eq. (5)

         where n_a = Tr(F_a^2).

    For multi-partite systems the confusion map is applied as a local
    superoperator on the specified subsystem:  Phi_C (x) id.

    Parameters
    ----------
    rho : qt.Qobj
        Input density matrix (single- or multi-partite).
    confusion_matrix : np.ndarray
        d x d column-stochastic matrix.  C[k,j] = P(report k | true j).
        Each column must sum to 1.  d must match the confused subsystem.
    confused_subsystem : int, default 0
        Index of the subsystem to confuse.  If trace_out is used, this
        refers to the index AFTER tracing.
    trace_out : list of int, optional
        Subsystem indices to trace out before applying confusion.
        Indices refer to the original rho.

    Returns
    -------
    qt.Qobj
        The confused density matrix with the same dims as the (possibly
        reduced) input state.
    """
    # -- optional partial trace ------------------------------------------------
    if trace_out is not None:
        n_sub = len(rho.dims[0])
        keep = [i for i in range(n_sub) if i not in trace_out]
        rho = qt.ptrace(rho, keep)

    dims = rho.dims[0]
    N = len(dims)
    s = confused_subsystem
    d_s = dims[s]
    C = np.asarray(confusion_matrix, dtype=float)
    assert C.shape == (d_s, d_s), (
        f"confusion_matrix shape {C.shape} does not match "
        f"subsystem {s} dimension {d_s}")

    # -- build single-subsystem superoperator from Gell-Mann basis -------------
    basis = _gell_mann_basis(d_s)
    norms = np.array([np.trace(F @ F).real for F in basis])  # n_0=d, n_{a>=1}=2

    S = np.zeros((d_s**2, d_s**2), dtype=complex)
    for col_pq in range(d_s**2):
        p_idx, q_idx = divmod(col_pq, d_s)
        rho_in = np.zeros((d_s, d_s), dtype=complex)
        rho_in[p_idx, q_idx] = 1.0

        rho_out = np.zeros((d_s, d_s), dtype=complex)
        for a, F_a in enumerate(basis):
            eigenvalues, U = np.linalg.eigh(F_a)
            diag_rot = np.diag(U.conj().T @ rho_in @ U)
            confused_diag = C @ diag_rot
            exp_meas = np.dot(eigenvalues, confused_diag)
            rho_out += (exp_meas / norms[a]) * F_a

        S[:, col_pq] = rho_out.ravel()

    # -- apply to the (multi-partite) density matrix ---------------------------
    D = int(np.prod(dims))
    rho_arr = rho.full()

    if N == 1:
        rho_new = (S @ rho_arr.ravel()).reshape(d_s, d_s)
    else:
        T = rho_arr.reshape(list(dims) + list(dims))
        # Move confused bra (axis s) and ket (axis N+s) to the last two positions
        T_moved = np.moveaxis(T, [s, N + s], [-2, -1])
        other_shape = T_moved.shape[:-2]
        # Flatten spectator dims, vectorise subsystem-s blocks, apply S
        T_flat = T_moved.reshape(-1, d_s * d_s)
        T_out_flat = (S @ T_flat.T).T
        # Restore shape and axis ordering
        T_out_moved = T_out_flat.reshape(other_shape + (d_s, d_s))
        T_new = np.moveaxis(T_out_moved, [-2, -1], [s, N + s])
        rho_new = T_new.reshape(D, D)

    return qt.Qobj(rho_new, dims=rho.dims)
