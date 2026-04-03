"""
Author: Andrei Militaru
Description: This module contains tools for simulating the microwave dynamics during 
the time bin entanglement experiment.
date: March 2026
"""

from __future__ import annotations
import numpy as np
import qutip as qt
from dataclasses import dataclass, field
from typing import Optional
from .interconnector.simulations_tools.tomographic_reconstructions import apply_readout_confusion


def build_operators(p: SystemParams) -> dict:
    """Construct all QuTiP operators for a qutrit  x  resonator Hilbert space."""
    Nq, Ntrunc = p.Nq, p.Ntrunc

    # -- qubit basis states ----------------------------------------------------
    g = qt.basis(Nq, 0)
    e = qt.basis(Nq, 1)
    f = qt.basis(Nq, 2)

    # -- resonator basis states ------------------------------------------------
    vac = qt.basis(Ntrunc, 0)
    one = qt.basis(Ntrunc, 1)

    # -- qubit single-space operators ------------------------------------------
    X_q = e*g.dag() + g*e.dag()
    Y_q = 1j*(e*g.dag() - g*e.dag())
    Z_q = -e*e.dag() + g*g.dag()
    ef_q = e*f.dag() + f*e.dag()

    # -- drive Hamiltonians (tensor product, resonator identity) ---------------
    Hd_ge_X = qt.tensor(X_q, qt.identity(Ntrunc))
    Hd_ge_Y = qt.tensor(Y_q, qt.identity(Ntrunc))
    Hd_ef_X = qt.tensor(ef_q, qt.identity(Ntrunc))

    # -- BSB interaction: |g,0> <-> |e,1> --------------------------------------
    bsb_transition = (qt.tensor(e*g.dag(), one*vac.dag()) +
                      qt.tensor(g*e.dag(), vac*one.dag()))
    Hd_bsb = bsb_transition

    # -- full-space ladder operators -------------------------------------------
    b  = qt.tensor(qt.destroy(Nq), qt.identity(Ntrunc))   # qubit ladder
    a  = qt.tensor(qt.identity(Nq), qt.destroy(Ntrunc))   # cavity ladder

    # -- projectors (full space) -----------------------------------------------
    sigz = qt.tensor(Z_q, qt.identity(Ntrunc))

    return dict(
        g=g, e=e, f=f, vac=vac, one=one,
        X_q=X_q, Y_q=Y_q, Z_q=Z_q,
        Hd_ge_X=Hd_ge_X, Hd_ge_Y=Hd_ge_Y, Hd_ef_X=Hd_ef_X, Hd_bsb=Hd_bsb,
        b=b, a=a, sigz=sigz,
    )


def build_augmented_operators(p: SystemParams) -> dict:
    """
    Construct all QuTiP operators for the augmented Hilbert space:
        qubit(0)  x  resonator(1)  x  early_TLS(2)  x  late_TLS(3)

    The early and late TLS modes are tracked as c_ops (L_early, L_late)
    so that qubit-time-bin entanglement is preserved while resonator decay
    remains fully irreversible (Lindblad, not beamsplitter oscillations).
    """
    Nq, Ntrunc = p.Nq, p.Ntrunc

    # -- qubit basis states ----------------------------------------------------
    g = qt.basis(Nq, 0)
    e = qt.basis(Nq, 1)
    f = qt.basis(Nq, 2)

    # -- resonator basis states ------------------------------------------------
    vac = qt.basis(Ntrunc, 0)
    one = qt.basis(Ntrunc, 1)

    # -- qubit single-space operators ------------------------------------------
    X_q = e*g.dag() + g*e.dag()
    Y_q = 1j*(e*g.dag() - g*e.dag())
    Z_q = -e*e.dag() + g*g.dag()
    ef_q = e*f.dag() + f*e.dag()

    # -- drive Hamiltonians (4 subsystems) -------------------------------------
    Hd_ge_X = qt.tensor(X_q,  qt.identity(Ntrunc), qt.identity(2), qt.identity(2))
    Hd_ge_Y = qt.tensor(Y_q,  qt.identity(Ntrunc), qt.identity(2), qt.identity(2))
    Hd_ef_X = qt.tensor(ef_q, qt.identity(Ntrunc), qt.identity(2), qt.identity(2))

    # -- BSB interaction: |g,0> <-> |e,1> --------------------------------------
    bsb_transition = (qt.tensor(e*g.dag(), one*vac.dag(), qt.identity(2), qt.identity(2)) +
                      qt.tensor(g*e.dag(), vac*one.dag(), qt.identity(2), qt.identity(2)))
    Hd_bsb = bsb_transition

    # -- TLS raising operator --------------------------------------------------
    tls_g = qt.basis(2, 0)
    tls_e = qt.basis(2, 1)
    sigma_plus = tls_e * tls_g.dag()   # |e><g|

    # Correlated c_ops  L = a  x  sigma+  (non-Hermitian, used as collapse operators)
    # Under D[L]rho = LrhoLdag - 1/2{LdagL, rho} the jump term LrhoLdag coherently maps
    # |q, 1_res, g_TLS> -> |q, 0_res, e_TLS> preserving the qubit index q,
    # giving irreversible exponential decay with full entanglement tracking.
    # NOTE: this Lindblad channel destroys inter-branch coherences; see
    #       H_swap_early/H_swap_late for the coherence-preserving alternative.
    L_early = qt.tensor(qt.identity(Nq), qt.destroy(Ntrunc), sigma_plus, qt.identity(2))
    L_late  = qt.tensor(qt.identity(Nq), qt.destroy(Ntrunc), qt.identity(2), sigma_plus)

    # -- SWAP Hamiltonians: coherence-preserving emission ----------------------
    # H_swap = g * (-i adag sigma-_TLS + i a sigma+_TLS)  is a sigma_y-type beam-splitter
    # in the {|1_res,g_TLS>, |0_res,e_TLS>} subspace.  A pi/2 rotation swaps
    # |1_res,g_TLS> -> |0_res,e_TLS> WITHOUT the -i phase that a sigma_x-type
    # coupling would introduce, matching the phase-free Lindblad jump L~asigma+.
    # The coupling strength g is set at runtime so that g * T_window = pi/2.
    sigma_minus = tls_g * tls_e.dag()  # |g><e|
    H_swap_early = (-1j * qt.tensor(qt.identity(Nq), qt.destroy(Ntrunc).dag(),
                                    sigma_minus, qt.identity(2)) +
                     1j * qt.tensor(qt.identity(Nq), qt.destroy(Ntrunc),
                                    sigma_plus, qt.identity(2)))
    H_swap_late  = (-1j * qt.tensor(qt.identity(Nq), qt.destroy(Ntrunc).dag(),
                                    qt.identity(2), sigma_minus) +
                     1j * qt.tensor(qt.identity(Nq), qt.destroy(Ntrunc),
                                    qt.identity(2), sigma_plus))

    assert H_swap_early.dag() == H_swap_early
    assert H_swap_late.dag() == H_swap_late

    # -- full-space ladder operators -------------------------------------------
    b = qt.tensor(qt.destroy(Nq), qt.identity(Ntrunc), qt.identity(2), qt.identity(2))  # qubit
    a = qt.tensor(qt.identity(Nq), qt.destroy(Ntrunc), qt.identity(2), qt.identity(2))  # cavity

    # -- projectors (full space) -----------------------------------------------
    sigz = qt.tensor(Z_q, qt.identity(Ntrunc), qt.identity(2), qt.identity(2))

    return dict(
        g=g, e=e, f=f, vac=vac, one=one, X_q=X_q, Y_q=Y_q, Z_q=Z_q,
        Hd_ge_X=Hd_ge_X, Hd_ge_Y=Hd_ge_Y, Hd_ef_X=Hd_ef_X, Hd_bsb=Hd_bsb,
        b=b, a=a, sigz=sigz, tls_g=tls_g, tls_e=tls_e, sigma_plus=sigma_plus,
        L_early=L_early, L_late=L_late,
        H_swap_early=H_swap_early, H_swap_late=H_swap_late,
    )


@dataclass
class CalibrationResult:
    A_pi_ge:  float = 0.0
    A_pi_ef:  float = 0.0
    A_pi_bsb: float = 0.0


def Hd_X_coeff_fn(A, sig, T, cen):
    """Return a closure: f(t, args) = A * Gaussian(t, sig, T, cen)."""
    def f(t, args):
        return A * np.exp(-((t - cen) / 2 / sig)**2) * (np.abs(t - cen) < T/2)
    return f


def Hbsb_coeff_fn(A, T, cen):
    """Return a closure: f(t, args) = A * rect(t, T, cen)."""
    def f(t, args):
        return A * (np.abs(t - cen) < T/2)
    return f


def calibrate(p: SystemParams, ops: dict,
              n_ge: int = 201, n_ef: int = 101, n_bsb: int = 101,
              verbose: bool = True) -> CalibrationResult:
    """
    Sweep Rabi amplitude for ge, ef, and BSB transitions.
    Returns CalibrationResult with the pi-pulse amplitudes.
    """
    g, e, f = ops['g'], ops['e'], ops['f']
    vac, one = ops['vac'], ops['one']
    b, a     = ops['b'],   ops['a']
    Hd_ge_X  = ops['Hd_ge_X']
    Hd_ef_X  = ops['Hd_ef_X']
    Hd_bsb   = ops['Hd_bsb']

    c_ops_cal = [np.sqrt(p.gamma_1) * b,
                 np.sqrt(p.gamma_phi) * b.dag() * b,
                 np.sqrt(p.kappa) * a]

    rho0 = qt.tensor(g * g.dag(), vac * vac.dag())
    tlist_cal = np.arange(0, 200e-3, 1e-3)

    # -- ge calibration --------------------------------------------------------
    amp_max_ge  = 2 / (p.sigma_ge * np.sqrt(np.pi))
    amp_arr_ge  = np.linspace(0, amp_max_ge, n_ge)
    cen_ge = 50e-3 + p.T_ge / 2

    pg_ge = []
    for amp in amp_arr_ge:
        H = [[Hd_ge_X, Hd_X_coeff_fn(amp, p.sigma_ge, p.T_ge, cen_ge)]]
        out = qt.mesolve(H, rho0, tlist_cal, c_ops_cal,
                         options=qt.Options(max_step=p.T_ge/4))
        rho_q = qt.ptrace(out.states[-1], 0)
        pg_ge.append(qt.expect(g * g.dag(), rho_q))

    A_pi_ge = amp_arr_ge[np.argmin(pg_ge)]

    # -- ef calibration (start from |e>) --------------------------------------
    rho0_e = qt.tensor(e * e.dag(), vac * vac.dag())
    amp_max_ef = 2 / (p.sigma_ef * np.sqrt(np.pi))
    amp_arr_ef = np.linspace(0, amp_max_ef, n_ef)
    cen_ef = 50e-3 + p.T_ef / 2

    pe_ef = []
    for amp in amp_arr_ef:
        H = [[ops['Hd_ef_X'], Hd_X_coeff_fn(amp, p.sigma_ef, p.T_ef, cen_ef)]]
        out = qt.mesolve(H, rho0_e, tlist_cal, c_ops_cal,
                         options=qt.Options(max_step=p.T_ge/4))
        rho_q = qt.ptrace(out.states[-1], 0)
        pe_ef.append(qt.expect(e * e.dag(), rho_q))

    A_pi_ef = amp_arr_ef[np.argmin(pe_ef)]

    # -- BSB calibration (start from |g,0>) -----------------------------------
    amp_max_bsb = np.pi / p.T_bsb
    amp_arr_bsb = np.linspace(0, amp_max_bsb, n_bsb)
    cen_bsb = 10e-3 + p.T_bsb / 2
    g0_proj = qt.tensor(g * g.dag(), vac * vac.dag())

    pg0_bsb = []
    for amp in amp_arr_bsb:
        H = [[Hd_bsb, Hbsb_coeff_fn(amp, p.T_bsb, cen_bsb)]]
        out = qt.mesolve(H, rho0, tlist_cal, c_ops_cal,
                         options=qt.Options(max_step=p.T_ge/4))
        pg0_bsb.append(qt.expect(g0_proj, out.states[-1]))

    A_pi_bsb = amp_arr_bsb[np.argmin(pg0_bsb)]

    cal = CalibrationResult(A_pi_ge=A_pi_ge, A_pi_ef=A_pi_ef, A_pi_bsb=A_pi_bsb)
    if verbose:
        print(f"  A_pi_ge  = {A_pi_ge:.4f}")
        print(f"  A_pi_ef  = {A_pi_ef:.4f}")
        print(f"  A_pi_bsb = {A_pi_bsb:.4f}")
    return cal


def gen_hamiltonian_for_pulse_sequence(pulses, p: SystemParams,
                                       cal: CalibrationResult,
                                       ops: dict):
    """
    Build a QuTiP time-dependent Hamiltonian list from a pulse schedule.

    Parameters
    ----------
    pulses : list of [type, value]
        type in {'ge', 'ef', 'bsb', 'wait'}
        value : rotation angle in units of pi  (ge/ef/bsb), or duration in ns (wait)
    p, cal, ops : system parameters, calibrated amplitudes, operators

    Returns
    -------
    H    : QuTiP Hamiltonian list  [[Op, coeff_fn], ...]
    tmax : float  - end time in us
    t_bsb_centers : list of float - centre times of each BSB pulse (us)
    """
    H = []
    t_now = 0.0
    t_bsb_centers = []

    for pulse in pulses:
        kind, value = pulse[0], pulse[1]

        if kind == 'wait':
            t_now += value * 1e-3  # ns -> us

        elif kind == 'ge':
            A   = cal.A_pi_ge * value
            cen = t_now + p.T_ge / 2
            H.append([ops['Hd_ge_X'],
                       Hd_X_coeff_fn(A, p.sigma_ge, p.T_ge, cen)])
            t_now += p.T_ge

        elif kind == 'ef':
            A   = cal.A_pi_ef * value
            cen = t_now + p.T_ef / 2
            H.append([ops['Hd_ef_X'],
                       Hd_X_coeff_fn(A, p.sigma_ef, p.T_ef, cen)])
            t_now += p.T_ef

        elif kind == 'bsb':
            A   = cal.A_pi_bsb * value
            cen = t_now + p.T_bsb / 2
            H.append([ops['Hd_bsb'],
                       Hbsb_coeff_fn(A, p.T_bsb, cen)])
            t_bsb_centers.append(cen)
            t_now += p.T_bsb

        else:
            raise ValueError(f"Unknown pulse type: {kind!r}")

    return H, t_now, t_bsb_centers


def default_timebin_sequence(delta_t_bins_ns: float, p: SystemParams) -> list:
    """
    Return the canonical pulse list.

    Parameters
    ----------
    delta_t_bins_ns : float
        Empty wait between the end of the first BSB pulse and the start of the
        ef/ge pulses that prepare the second emission.  Unit: nanoseconds.
        At the experimental default this is  600 - T_bsb*1e3  ~ 444 ns.
    """
    T_bsb_ns = p.T_bsb * 1e3   # us -> ns

    return [
        ['wait', 752],                  # initialisation
        ['ge',   0.5],                  # pi/2 on qubit
        ['wait', 4],
        ['ef',   1],                    # ef pi on qubit
        ['wait', 4],
        ['bsb',  1],                    # entangling gate
        ['wait', 600 - T_bsb_ns + delta_t_bins_ns],       # early photon leakage 
        ['ef',   1],                    # take back shelved population
        ['wait', 4],
        ['ge',   1],                    # e->g
        ['wait', 4],
        ['bsb',  1],                    # <- LATE photon
        ['wait', 600 - T_bsb_ns],      # symmetric tail window
        ['ge',   1],                    # readout rotation
        ['wait', 4],
        ['ef',   1],                    # readout rotation
        ['wait', 4],
    ]


def get_rho_manifold(rho: qt.Qobj, ops: dict, p: SystemParams, apply_confusion_matrix: Optional[bool] = False) -> qt.Qobj:
    """
    Extract the 4x4 logical density matrix in the single-photon manifold of
    qubit  x  time-bin from the full density matrix in the augmented space.

    The logical basis states are:
    |g, late> = |g>_qubit  x  |early_g, late_e>_TLS
    |g, early> = |g>_qubit  x  |early_e, late_g>_TLS
    |e, late> = |e>_qubit  x  |early_g, late_e>_TLS
    |e, early> = |e>_qubit  x  |early_e, late_g>_TLS

    This is used for state tomography and visualisation of the density matrix.
    """
    rho_reduced = qt.ptrace(rho, [0, 2, 3])  # dims [[Nq,2,2],[Nq,2,2]]
    if apply_confusion_matrix:
        rho_reduced = apply_readout_confusion(
            rho_reduced, p.confusion_matrix, confused_subsystem=0)

    # Logical 2-qubit basis (qubit  x  time-bin) embedded in the reduced space
    #   qubit: |g> = 0, |e> = 1
    #   time-bin: |late>  = |early_g, late_e>  (early TLS=0, late TLS=1)
    #            |early> = |early_e, late_g>  (early TLS=1, late TLS=0)
    gg = qt.tensor(qt.basis(p.Nq, 0), qt.basis(2, 0), qt.basis(2, 1))  # |g, late>
    ge = qt.tensor(qt.basis(p.Nq, 0), qt.basis(2, 1), qt.basis(2, 0))  # |g, early>
    eg = qt.tensor(qt.basis(p.Nq, 1), qt.basis(2, 0), qt.basis(2, 1))  # |e, late>
    # Phase correction: the canonical pulse sequence accumulates a net +i
    # relative phase between the |g,late> and |e,early> branches due to
    # the -i factors from each sigma_x-type Hamiltonian gate (ge, ef, BSB).
    # Multiplying by -i absorbs this so the ideal Bell state has real coherences.
    ee = -1j * qt.tensor(qt.basis(p.Nq, 1), qt.basis(2, 1), qt.basis(2, 0))  # |e, early> (phase-corrected)

    basis_states = [gg, ge, eg, ee]

    # Build the 4x4 logical density matrix with elements rho_mn = <m| rho_reduced |n>.
    rho4 = np.zeros((4, 4), dtype=complex)
    for m, ket_m in enumerate(basis_states):
        for n, ket_n in enumerate(basis_states):
            rho4[m, n] = complex(ket_m.dag() * rho_reduced * ket_n)

    rho_manifold = qt.Qobj(rho4, dims=[[2, 2], [2, 2]])
    tr = rho_manifold.tr()
    if np.isclose(tr, 0.0):
        return 0.0
    rho_manifold /= tr  # renormalise to the single-photon logical subspace
    return rho_manifold



def fidelity(rho: qt.Qobj, ops: dict, p: SystemParams, apply_confusion_matrix: Optional[bool] = False) -> float:
    """Compute the Bell-state fidelity F = <psi+|rho|psi+> from the full density matrix in the augmented space."""
    rho_manifold = get_rho_manifold(rho, ops, p, apply_confusion_matrix=apply_confusion_matrix)
    # Target Bell state: (|g,late> + |e,early>)/sqrt2  -> (|00> + |11>)/sqrt2 in this basis
    psi_ideal = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
                 qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
    rho_ideal = psi_ideal * psi_ideal.dag()

    return qt.expect(rho_ideal, rho_manifold)


@dataclass
class SystemParams:
    """All physical parameters.  Times in us, frequencies in rad/us (2pi MHz)."""

    # -- qubit / cavity frequencies --------------------------------------------
    wq:         float = 2*np.pi*6566.6   # qubit frequency  [rad/us]
    wres:       float = 2*np.pi*8.906      # resonator frequency [rad/us]
    alpha:      float = 2*np.pi*257.24   # anharmonicity    [rad/us]

    # -- decoherence rates  (1/T in us**(-1)) ------------------------------------
    gamma_1:    float = 1/19             # qubit energy relaxation
    gamma_phi:  float = 1/14             # qubit pure dephasing
    kappa:      float = 2*np.pi*1.43     # total cavity linewidth
    kappa_frac: float = 1.00/1.43             # kappa_ext / kappa

    # -- thermal photon numbers ------------------------------------------------
    nth:        float = 0.0           # thermal qubit excitation, used to be 0.07
    nwg:        float = 0.0             # waveguide thermal photons, used to be 0.1

    # -- Hilbert space truncations ---------------------------------------------
    Nq:         int   = 3               # qutrit (g, e, f)
    Ntrunc:     int   = 2               # resonator Fock truncation

    # -- pulse shape parameters (us) ------------------------------------------
    sigma_ge:   float = 8e-3            # Gaussian sigma for ge pulse
    sigma_ef:   float = 8e-3            # Gaussian sigma for ef pulse
    T_bsb:      float = 156e-3          # BSB square-pulse duration

    # -- confusion matrix (ideal = identity) ----------------------------------
    confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.array([[91.6, 2.3,  6.1 ],
                                          [7.0,  81.4, 11.6],
                                          [12.5, 14.1, 73.4]]).T / 100)

    def __post_init__(self):
        self.T_ge = self.sigma_ge * 5
        self.T_ef = self.sigma_ef * 5
        self.kappa_ext = self.kappa * self.kappa_frac

    @classmethod
    def from_experiment(cls, **overrides):
        """Return default experimental values, optionally overriding any field."""
        return cls(**overrides)


class QubitCavitySimulation:
    """
    One-stop class for the qubit-cavity time-bin entanglement simulation.

    Usage
    -----
    sim   = QubitCavitySimulation(params, cal, ops)
    out_u = sim.run_unconditional(delta_t_bins_ns=444)
    out_s = sim.run_sme(delta_t_bins_ns=444, ntraj=500)
    """

    def __init__(self, p: SystemParams, cal: CalibrationResult, ops: dict):
        self.p   = p
        self.cal = cal
        self.ops = ops

        # cached results
        self._uncond_cache: dict = {}
        self._sme_cache:    dict = {}

    # -------------------------------------------------------------------------
    def _initial_rho(self):
        p = self.p
        return qt.tensor(
            qt.thermal_dm(p.Nq,     p.nth),
            qt.thermal_dm(p.Ntrunc,
                          ((p.kappa - p.kappa_ext)*p.nth + p.kappa_ext*p.nwg)
                          / p.kappa),
        )

    def _build_c_ops_unconditional(self):
        """Deterministic collapse operators for unconditional simulations."""
        p, ops = self.p, self.ops
        a, b   = ops['a'], ops['b']
        kappa_int = p.kappa - p.kappa_ext

        if p.nth == 0. and p.nwg == 0.:
            return [
                np.sqrt(p.gamma_1) * b,
                np.sqrt(p.gamma_phi) * b.dag() * b,
                np.sqrt(kappa_int + p.kappa_ext) * a,
            ]
        
        elif p.nth == 0. and p.nwg != 0.:
            return [
                np.sqrt(p.gamma_1) * b,
                np.sqrt(p.gamma_phi) * b.dag() * b,
                np.sqrt(kappa_int + p.kappa_ext*(p.nwg+1)) * a,   # internal loss
                np.sqrt(self.p.kappa_ext*p.nwg) * a.dag(),
            ]

        else:
            return [
                np.sqrt(p.gamma_1*(p.nth+1)) * b,
                np.sqrt(p.gamma_1*p.nth) * b.dag(),
                np.sqrt(p.gamma_phi) * b.dag() * b,
                np.sqrt(kappa_int*(p.nth+1) + p.kappa_ext*(p.nwg+1)) * a,   # internal loss
                np.sqrt(kappa_int*p.nth + self.p.kappa_ext*p.nwg) * a.dag(),
            ]

    def _build_c_ops(self):
        """Deterministic collapse operators (no heterodyne channel)."""
        p, ops = self.p, self.ops
        a, b   = ops['a'], ops['b']
        kappa_int = p.kappa - p.kappa_ext

        if p.nth == 0. and p.nwg == 0.:
            return [
                np.sqrt(p.gamma_1) * b,
                np.sqrt(p.gamma_phi) * b.dag() * b,
                np.sqrt(kappa_int) * a,   # internal loss
            ]
        
        elif p.nth == 0. and p.nwg != 0.:
            return [
                np.sqrt(p.gamma_1) * b,
                np.sqrt(p.gamma_phi) * b.dag() * b,
                np.sqrt(kappa_int) * a,   # internal loss
                np.sqrt(self.p.kappa_ext*p.nwg) * a.dag(),
            ]
        
        else:
            return [
                np.sqrt(p.gamma_1*(p.nth+1)) * b,
                np.sqrt(p.gamma_1*p.nth) * b.dag(),
                np.sqrt(p.gamma_phi) * b.dag() * b,
                np.sqrt(kappa_int*(p.nth+1)) * a,   # internal loss
                np.sqrt(kappa_int*p.nth + self.p.kappa_ext*p.nwg) * a.dag(),
                #np.sqrt(self.p.kappa_ext*p.nwg) * a.dag()
            ]

    def _build_sc_ops(self):
        """Stochastic (measured) collapse operators for heterodyne."""
        return [np.sqrt(self.p.kappa_ext * (self.p.nwg + 1)) * self.ops['a']]
                #np.sqrt(self.p.kappa_ext*p.nwg) * self.ops['a'].dag()]

    # -------------------------------------------------------------------------
    def run_unconditional(self, delta_t_bins_ns: float,
                          dt_us: float = 1e-3) -> dict:
        """
        Unconditional (Lindblad) dynamics.  Returns a dict with keys:
        'tlist', 'tmax', 'one_pop', 'e_pop', 'bsb_centers'.
        Results are cached keyed on delta_t_bins_ns.
        """
        key = delta_t_bins_ns
        if key in self._uncond_cache:
            return self._uncond_cache[key]

        p, ops = self.p, self.ops
        pulses = default_timebin_sequence(delta_t_bins_ns, p)
        Hsolve, tmax, bsb_ctrs = gen_hamiltonian_for_pulse_sequence(
            pulses, p, self.cal, ops)

        tlist = np.arange(0, tmax + 40e-3, dt_us)
        rho0  = self._initial_rho()
        c_ops = self._build_c_ops_unconditional()

        out = qt.mesolve(Hsolve, rho0, tlist, c_ops,
                         options=qt.Options(max_step=p.T_ge/4))

        one      = ops['one']
        e_state  = ops['e']
        rhos_res = [qt.ptrace(rho, 1) for rho in out.states]
        rhos_qub = [qt.ptrace(rho, 0) for rho in out.states]

        one_pop = np.array([qt.expect(one * one.dag(), r) for r in rhos_res])
        e_pop   = np.array([qt.expect(e_state * e_state.dag(), r) for r in rhos_qub])

        result = dict(tlist=tlist, tmax=tmax, one_pop=one_pop,
                      e_pop=e_pop, bsb_centers=bsb_ctrs, states=out.states)
        self._uncond_cache[key] = result
        return result

    # -------------------------------------------------------------------------
    def run_sme(self, delta_t_bins_ns: float, ntraj: int = 500,
                dt_us: float = 1e-3, force_rerun: bool = False) -> dict:
        """
        Stochastic Master Equation with heterodyne detection on kappa_ext*a.
        Returns dict with 'sme_out', 'tlist', 'tmax', 'bsb_centers'.
        """
        key = (delta_t_bins_ns, ntraj)
        if key in self._sme_cache and not force_rerun:
            return self._sme_cache[key]

        p, ops = self.p, self.ops
        pulses = default_timebin_sequence(delta_t_bins_ns, p)
        Hsolve, tmax, bsb_ctrs = gen_hamiltonian_for_pulse_sequence(
            pulses, p, self.cal, ops)

        tlist = np.arange(0, tmax + 40e-3, dt_us)
        rho0  = self._initial_rho()
        c_ops = self._build_c_ops()
        sc_ops = self._build_sc_ops()

        sme_out = qt.smesolve(
            Hsolve, rho0, tlist,
            c_ops=c_ops, sc_ops=sc_ops,
            ntraj=ntraj,
            method='heterodyne',
            store_measurement=True,
            options=qt.Options(store_states=True),
        )

        result = dict(sme_out=sme_out, tlist=tlist, tmax=tmax,
                      bsb_centers=bsb_ctrs)
        self._sme_cache[key] = result
        return result

    # -------------------------------------------------------------------------
    def run_mcsolve(self, delta_t_bins_ns: float, ntraj: int = 2000,
                    dt_us: float = 1e-3) -> dict:
        """
        Monte-Carlo photon-counting simulation.
        All collapse ops (including external-port photon) are incoherent.
        """
        p, ops = self.p, self.ops
        pulses = default_timebin_sequence(delta_t_bins_ns, p)
        Hsolve, tmax, bsb_ctrs = gen_hamiltonian_for_pulse_sequence(
            pulses, p, self.cal, ops)

        tlist = np.arange(0, tmax + 40e-3, dt_us)
        rho0  = self._initial_rho()
        # full model: internal + external loss
        a = ops['a'];  b = ops['b']
        c_ops_full = self._build_c_ops() + self._build_sc_ops()

        mc_out = qt.mcsolve(
            Hsolve, rho0, tlist, c_ops_full, ntraj=ntraj,
            options=qt.Options(store_states=True, store_final_state=True))

        return dict(mc_out=mc_out, tlist=tlist, tmax=tmax,
                    bsb_centers=bsb_ctrs)


class QubitCavityAugmentedSimulation:
    """
    One-stop class for the qubit-cavity time-bin entanglement simulation.

    Usage
    -----
    sim   = QubitCavityAugmentedSimulation(params, cal, ops)
    out_u = sim.run(delta_t_bins_ns=444)
    """

    def __init__(self, p: SystemParams, cal: CalibrationResult, ops: dict):
        self.p   = p
        self.cal = cal
        self.ops = ops

    # -------------------------------------------------------------------------
    def _initial_rho(self):
        p = self.p
        # Space: qubit(0)  x  resonator(1)  x  early_TLS(2)  x  late_TLS(3)
        return qt.tensor(
            qt.thermal_dm(p.Nq,     p.nth),
            qt.thermal_dm(p.Ntrunc,
                          ((p.kappa - p.kappa_ext)*p.nth + p.kappa_ext*p.nwg)
                          / p.kappa),
            qt.thermal_dm(2, 0),  # early mode TLS (subsystem 2)
            qt.thermal_dm(2, 0),  # late mode TLS  (subsystem 3)
        )

    def _build_c_ops(self):
        """Time-independent collapse operators (qubit decay + internal resonator loss).
        Time-windowed external decay (L_early / L_late) is appended in run()."""
        p, ops = self.p, self.ops
        b = ops['b']

        if p.nth == 0. and p.nwg == 0.:
            return [
                np.sqrt(p.gamma_1)   * b,
                np.sqrt(p.gamma_phi) * b.dag() * b,
            ]
        else:
            return [
                np.sqrt(p.gamma_1*(p.nth+1))   * b,
                np.sqrt(p.gamma_1*p.nth)        * b.dag(),
                np.sqrt(p.gamma_phi)            * b.dag() * b,
            ]

    # -------------------------------------------------------------------------
    def run(self, delta_t_bins_ns: float,
                          dt_us: float = 1e-3,
                          emission_mode: str = 'shaped_swap') -> dict:
        """
        Unconditional (Lindblad) dynamics.  Returns a dict with keys:
        'tlist', 'tmax', 'one_pop', 'e_pop', 'bsb_centers', 'rho_final'.

        Parameters
        ----------
        delta_t_bins_ns : float
            Extra delay between time bins (ns).
        dt_us : float
            Time step (us).
        emission_mode : str, default 'swap'
            'lindblad'     -- use D[sqrt(kappa) * a x sigma+] Lindblad channel to route
                             photons into the TLS modes.  Correct Z-basis
                             populations but destroys inter-branch coherences.
            'swap'         -- use a constant Hamiltonian beam-splitter coupling
                             g*(adagsigma- + asigma+) with g = pi/(2*T_window) to
                             unitarily swap the photon into the TLS mode.
                             Preserves all coherences.  Produces cos^2 decay.
            'shaped_swap'  -- like 'swap' but with time-dependent coupling
                             g(t) = alpha*kappa/2 * e^{-kappat/2}/sqrt(1-e^{-kappat}) chosen
                             so the resonator population follows e^{-kappat}.
                             Strong initial coupling, smoothly ramped down.

        Memory-efficient: expectation values are computed on the fly via e_ops
        so no full states are stored.  Only the final state is retained to
        extract the reduced density matrix for qubit x early_mode x late_mode
        (subsystem indices [0, 3, 4] in the full space).
        """
        if emission_mode not in ('lindblad', 'swap', 'shaped_swap'):
            raise ValueError(f"emission_mode must be 'lindblad', 'swap', or 'shaped_swap', got {emission_mode!r}")

        p, ops = self.p, self.ops
        pulses = default_timebin_sequence(delta_t_bins_ns, p)
        Hsolve, tmax, bsb_ctrs = gen_hamiltonian_for_pulse_sequence(
            pulses, p, self.cal, ops)

        # -- time window boundaries --------------------------------------------
        T_0              = 0.0
        T_end_noise_1    = bsb_ctrs[0] - p.T_bsb/2   # end of pre-early noise window
        T_end_early_mode = bsb_ctrs[0] + 600e-3       # end of early collection window
        T_end_noise_2    = bsb_ctrs[1] - p.T_bsb/2   # end of inter-BSB noise window
        T_end_late_mode  = bsb_ctrs[1] + 600e-3       # end of late collection window

        # Capture locally to avoid late-binding closure issues
        _t1, _t2, _t3, _t4 = T_end_noise_1, T_end_early_mode, T_end_noise_2, T_end_late_mode
        sqrt_kext = np.sqrt(p.kappa)

        c_ops = self._build_c_ops()

        if emission_mode == 'lindblad':
            # -- Lindblad emission: time-windowed collapse operators -------
            # During noise windows: decay into unmeasured waveguide.
            # During mode windows:  D[sqrt(kappa)*a x sigma+] routes photons into TLS.
            c_ops += [
                [sqrt_kext * ops['a'],       lambda t, args: 1.0 if (T_0 < t <= _t1 or _t2 < t <= _t3) else 0.0],
                [sqrt_kext * ops['L_early'], lambda t, args: 1.0 if _t1 < t <= _t2 else 0.0],
                [sqrt_kext * ops['L_late'],  lambda t, args: 1.0 if _t3 < t <= _t4 else 0.0],
            ]

        elif emission_mode == 'swap':
            # -- SWAP emission: time-windowed Hamiltonian coupling ---------
            # g = pi/(2*T_window) so that a full pi/2 rotation swaps the photon.
            T_window_early = _t2 - _t1
            T_window_late  = _t4 - _t3
            g_early = np.pi / (2 * T_window_early)
            g_late  = np.pi / (2 * T_window_late)

            Hsolve += [
                [g_early * ops['H_swap_early'],
                 lambda t, args: 1.0 if _t1 < t <= _t2 else 0.0],
                [g_late * ops['H_swap_late'],
                 lambda t, args: 1.0 if _t3 < t <= _t4 else 0.0],
            ]
            # Off-window: resonator decays into unmeasured waveguide
            c_ops += [
                [sqrt_kext * ops['a'],
                 lambda t, args: 1.0 if (T_0 < t <= _t1 or _t2 < t <= _t3 or t > _t4) else 0.0],
            ]

        elif emission_mode == 'shaped_swap':
            # -- Shaped SWAP: regularised exact formula --------------------
            # The coupling starts AFTER the BSB pulse ends so that the full
            # rotation budget is used to swap the already-created photon.
            # During the BSB pulse the resonator has only waveguide loss.
            #
            # Exact solution for cos^2(Theta(tau)) = e^{-kappatau} gives
            #   g(tau) = (kappa/2) e^{-kappatau/2} / sqrt(1 - e^{-kappatau})
            # Regularised with delta to cap the 1/sqrttau singularity.  A numerically
            # computed alpha ensures the total rotation equals pi/2.
            _kappa = p.kappa
            _delta = 0.05     # regularisation parameter

            # Emission windows start after each BSB pulse ends
            _e1 = bsb_ctrs[0] + p.T_bsb / 2   # early swap start
            _e2 = _t2                            # early swap end (same as before)
            _l1 = bsb_ctrs[1] + p.T_bsb / 2   # late swap start
            _l2 = _t4                            # late swap end

            def _shaped_swap_coeff(t_start, t_end, kappa, delta):
                T_w = t_end - t_start
                # Numerically compute alpha so that int_0^{T_w} g(tau) dtau = pi/2
                n_quad = 5000
                tau_arr = np.linspace(0, T_w, n_quad + 1)
                kt = kappa * tau_arr
                g_raw = (kappa / 2) * np.exp(-kt / 2) / np.sqrt(
                    1 - np.exp(-kt) + delta**2)
                total_angle = np.trapz(g_raw, tau_arr)
                alpha = (np.pi / 2) / total_angle

                def f(t, args):
                    if t_start < t <= t_end:
                        tau = t - t_start
                        kt_val = kappa * tau
                        return alpha * (kappa / 2) * np.exp(-kt_val / 2) / np.sqrt(
                            1 - np.exp(-kt_val) + delta**2)
                    return 0.0
                return f

            Hsolve += [
                [ops['H_swap_early'],
                 _shaped_swap_coeff(_e1, _e2, _kappa, _delta)],
                [ops['H_swap_late'],
                 _shaped_swap_coeff(_l1, _l2, _kappa, _delta)],
            ]
            # Off-window: resonator decays into unmeasured waveguide
            # (includes the BSB pulse periods)
            c_ops += [
                [sqrt_kext * ops['a'],
                 lambda t, args: 1.0 if (t <= _e1 or (_e2 < t <= _l1) or t > _l2) else 0.0],
            ]

        tlist = np.arange(0, tmax + 40e-3, dt_us)
        rho0  = self._initial_rho()

        # Full-space projectors for on-the-fly expectation values (4 subsystems)
        one     = ops['one']
        e_state = ops['e']
        tls_g   = ops['tls_g']
        tls_e   = ops['tls_e']
        e_op_one = qt.tensor(qt.identity(p.Nq), one * one.dag(),
                             qt.identity(2), qt.identity(2))
        e_op_e   = qt.tensor(e_state * e_state.dag(), qt.identity(p.Ntrunc),
                             qt.identity(2), qt.identity(2))
        early_pop = qt.tensor(qt.identity(p.Nq), qt.identity(p.Ntrunc),
                             tls_e*tls_e.dag(), qt.identity(2))
        late_pop = qt.tensor(qt.identity(p.Nq), qt.identity(p.Ntrunc),
                             qt.identity(2), tls_e*tls_e.dag())

        # Shaped swap needs finer steps to resolve the strong initial coupling
        _max_step = 1e-3 if emission_mode == 'shaped_swap' else p.T_ge / 4

        out = qt.mesolve(Hsolve, rho0, tlist, c_ops,
                         e_ops=[e_op_one, e_op_e, early_pop, late_pop],
                         options=qt.Options(max_step=_max_step,
                                            store_states=False,
                                            store_final_state=True))

        one_pop = np.array(out.expect[0])
        e_pop   = np.array(out.expect[1])
        early_pop = np.array(out.expect[2])
        late_pop = np.array(out.expect[3])

        return dict(tlist=tlist, tmax=tmax, one_pop=one_pop, early_pop=early_pop, late_pop=late_pop,
                    e_pop=e_pop, bsb_centers=bsb_ctrs, rho_final=out.final_state)
    
    