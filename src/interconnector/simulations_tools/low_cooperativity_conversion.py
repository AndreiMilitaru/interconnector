"""
Author: Andrei Militaru
Description: This module contains tools for simulating the microwave dynamics during 
the time bin entanglement experiment.
date: March 2026
"""

from __future__ import annotations
from copyreg import pickle
from zipfile import Path
import numpy as np
import qutip as qt
from dataclasses import dataclass, field
from typing import Optional
from numba import njit
from matplotlib import pyplot as plt
from .interconnector.visualization import set_ax
from scipy.ndimage import gaussian_filter1d
from scipy.constants import hbar


# -- Input-output cavity response ----------------------------------------------
def H_res(omega, delta, kappa, eta_ext):
    """
    Input-output cavity amplitude response.
    Returns ?(? ?_ext) / (i(? - ?) + ?/2)
    omega, delta, kappa in consistent units (rad/?s or rad/s).
    """
    return np.sqrt(kappa * eta_ext) / (1j * (omega - delta) + kappa / 2)


# -- Numba JIT: time-domain upconversion kernel --------------------------------
@njit()
def timed_conversion(t, f, g, ke, ko, etae, etao, detuning, flip_time=0.0):
    """
    Compute the upconverted optical amplitude from:
      f  : microwave input field amplitude (time domain, with losses applied)
      g  : time-dependent coupling g(t) = g0 * internal_pump(t)
      ke : EO microwave cavity decay rate
      ko : EO optical cavity decay rate
      etae/etao : coupling efficiencies
      detuning  : pump detuning (rad/s)
      flip_time : time of MW sign flip (s); 0 = disabled
    All time arrays in seconds (SI).
    """
    dt = t[1] - t[0]
    n = len(t)
    f = np.where(t < flip_time, f, -f) if flip_time > 0.0 else f
    amp_out = np.zeros(n, dtype=np.complex128)
    prefactor = np.sqrt(ko * etao * ke * etae) * dt * dt

    f_exp = f * np.exp((ke / 2 - 1j * detuning) * t)

    cumsum_f = np.zeros(n, dtype=np.complex128)
    cumsum_f[0] = f_exp[0]
    for idx in range(1, n):
        cumsum_f[idx] = cumsum_f[idx - 1] + f_exp[idx]

    for i in range(n):
        tt = t[i]
        sum_val = 0.0 + 0.0j
        for j in range(i):
            tau = t[j]
            sum_val += (cumsum_f[j] * g[j] *
                        np.exp((-ke / 2 + 1j * detuning) * tau - ko / 2 * (tt - tau)))
        amp_out[i] = sum_val * prefactor
    return amp_out


# -- Signal filtering -----------------------------------------------------------
def apply_cavity_filter(signal_fft, freqs, kappa, eta=0.5, n_cascades=1):
    """Apply n_cascades identical ideal cavity filters to signal FFT."""
    H = np.sqrt(kappa * eta) * H_res(freqs * 2 * np.pi, 0, kappa, eta)
    return signal_fft * H**n_cascades


def apply_jittery_filter(signal_fft, freqs, kappa, jitter, eta=0.5, n_pts=100):
    """
    Gaussian-jittered cavity filter. Jitter is the fractional width ? = kappa*jitter.
    Averages the cavity response over a Gaussian distribution of resonance frequencies.
    """
    sig    = kappa * jitter
    dets   = np.linspace(-5 * sig, 5 * sig, n_pts)
    ddet   = dets[1] - dets[0]
    H_sum  = np.zeros_like(signal_fft)
    for det in dets:
        H = np.sqrt(kappa * eta) * H_res(freqs * 2 * np.pi, det, kappa, eta)
        H_sum += signal_fft * H * np.exp(-det**2 / 2 / sig**2) / np.sqrt(2 * np.pi * sig**2) * ddet
    return H_sum


def plot_mw_wavepacket(res: PropagationResult, save=False, plotfolder=Path('./plots')):
    """Normalized microwave photon time envelope (QuTiP output)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(res.t_qutip, res.fock_env, color='C0', lw=1, label='MW photon flux')
    ax.fill_between(res.t_qutip, res.fock_env, alpha=0.15, color='C0')
    set_ax(ax, xlabel=r'time ($\mu$s)', ylabel='Normalised amplitude',
           title='Microwave photon wavepacket', legend=True, grid_alpha=0.3)
    plt.tight_layout()
    if save: fig.savefig(plotfolder / 'mw_wavepacket.pdf')
    plt.show()


def plot_qutip_populations(res: PropagationResult, save=False, plotfolder=Path('./plots')):
    """Resonator |1> population from QuTiP simulation."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(res.t_qutip, res.one_pop, color='C0', lw=1)
    set_ax(ax, xlabel=r'time ($\mu$s)', ylabel=r'$\langle|1\rangle\langle 1|\rangle$',
           title='Qubit-resonator |1> population', grid_alpha=0.3)
    plt.tight_layout()
    if save: fig.savefig(plotfolder / 'qutip_one_pop.pdf')
    plt.show()


def plot_mw_cavity(res: PropagationResult, xlim_us=None, save=False, plotfolder=Path('./plots')):
    """Reflected and internal MW cavity fields."""
    t = res.t_prop
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.2))
    axes[0].plot(t, np.abs(res.fock_padded)**2/1e6, '--', color='C0', lw=0.7,
                 label='incident', alpha=0.7)
    axes[0].plot(t, np.abs(res.reflected_mw)**2/1e6, color='C2', lw=1,
                 label='reflected')
    set_ax(axes[0], xlabel=r'time ($\mu$s)', ylabel=r'flux ($10^6$ ph/s)',
           title='MW cavity: reflected', legend=True, grid_alpha=0.3,
           xlim=xlim_us)
    axes[1].plot(t, np.abs(res.internal_mw)**2, color='C0', lw=1)
    set_ax(axes[1], xlabel=r'time ($\mu$s)', ylabel='population',
           title='MW cavity: internal field', grid_alpha=0.3, xlim=xlim_us)
    plt.tight_layout()
    if save: fig.savefig(plotfolder / 'mw_cavity.pdf')
    plt.show()


def plot_pump(res: PropagationResult, xlim_ns=None, save=False, plotfolder=Path('./plots')):
    """Input pump, reflected pump, and internal pump fields."""
    p = res.params
    t_ns = res.t_prop / 1e-3   # ?s ? ns

    d_ns   = p.delay_pump * 1e9
    dur_ns = p.duration_pump * 1e9

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.2))
    sc = 1e16
    axes[0].plot(t_ns, np.abs(res.pump_input)**2 / sc, '--', color='C0',
                 lw=0.7, label='incident', alpha=0.7)
    axes[0].plot(t_ns, np.abs(res.reflected_pump)**2 / sc, color='C2',
                 lw=1, label='reflected')
    xl = xlim_ns if xlim_ns else [d_ns - 50, d_ns + dur_ns + 100]
    set_ax(axes[0], xlabel='time (ns)', ylabel=r'power ($10^{16}$ ph/s)',
           title='Optical pump: reflected', legend=True, grid_alpha=0.3,
           xlim=xl)
    axes[1].plot(t_ns, np.abs(res.internal_pump)**2 / 1e6, color='C0', lw=1)
    set_ax(axes[1], xlabel='time (ns)', ylabel=r'$n_p$ ($10^6$)',
           title='Optical pump: internal', grid_alpha=0.3, xlim=xl)
    plt.tight_layout()
    if save: fig.savefig(plotfolder / 'pump_fields.pdf')
    plt.show()


def plot_conversion_timing(res: PropagationResult, xlim_us=None, save=False, plotfolder=Path('./plots')):
    """MW wavepacket and pump internal field overlaid (conversion timing)."""
    t = res.t_prop
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.plot(t, np.abs(res.internal_mw)**2, color='C0', lw=1, label='MW internal')
    ip_norm = np.abs(res.internal_pump)**2
    ip_norm = ip_norm / max(ip_norm.max(), 1e-300) * (np.abs(res.internal_mw)**2).max()
    ax.plot(t, ip_norm, color='C1', lw=0.7, ls='--', label='pump internal (norm.)')
    set_ax(ax, xlabel=r'time ($\mu$s)', ylabel='population',
           title='Conversion timing', legend=True, grid_alpha=0.3, xlim=xlim_us)
    plt.tight_layout()
    if save: fig.savefig(plotfolder / 'conversion_timing.pdf')
    plt.show()


def plot_upconverted(res: PropagationResult, xlim_ns=None, save=False, plotfolder=Path('./plots')):
    """Upconverted optical signal before and after filtering."""
    p = res.params
    t_ns = res.t_prop / 1e-3
    d_ns   = p.delay_pump * 1e9
    dur_ns = p.duration_pump * 1e9
    xl = xlim_ns if xlim_ns else [d_ns - 50, d_ns + dur_ns + 150]

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.2))
    axes[0].plot(t_ns, np.abs(res.upconv_fock)**2, color='C0', lw=1,
                 label='upconverted')
    set_ax(axes[0], xlabel='time (ns)', ylabel='flux (ph/s)',
           title=f'Upconverted  P={res.p_converted:.4f}',
           legend=True, grid_alpha=0.3, xlim=xl)
    axes[1].plot(t_ns, np.abs(res.upconv_fock)**2, color='C0', lw=0.7,
                 alpha=0.5, label='pre-filter')
    axes[1].plot(t_ns, np.abs(res.filtered_fock)**2, color='C1', lw=1,
                 label='filtered')
    set_ax(axes[1], xlabel='time (ns)', ylabel='flux (ph/s)',
           title=f'Filtered  P={res.p_detected:.4f}',
           legend=True, grid_alpha=0.3, xlim=xl)
    plt.tight_layout()
    if save: fig.savefig(plotfolder / 'upconverted.pdf')
    plt.show()


def plot_filter_transfer(res: PropagationResult, xlim_MHz=(-15, 15), save=False, plotfolder=Path('./plots')):
    """Spectral content of upconverted field vs filter cavities."""
    p = res.params
    freqs_MHz = res.freqs / 1e6
    upconv_fft = np.fft.fft(res.upconv_fock)
    spec_norm   = np.abs(upconv_fft)**2 / (np.abs(upconv_fft)**2).max()

    H_normal = np.abs(apply_cavity_filter(
        np.ones_like(upconv_fft), res.freqs, p.kappa_filter_normal))**2
    H_narrow = np.abs(apply_cavity_filter(
        np.ones_like(upconv_fft), res.freqs, p.kappa_filter_narrow))**2
    H_jitter = np.abs(apply_jittery_filter(
        np.ones_like(upconv_fft), res.freqs, p.kappa_filter_narrow,
        p.filter_jitter))**2

    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.plot(freqs_MHz, spec_norm,  color='C0', lw=1, label='upconv. spectrum')
    ax.plot(freqs_MHz, H_normal / H_normal.max(), '--', color='C2', lw=0.8, label='wide filter')
    ax.plot(freqs_MHz, H_narrow / H_narrow.max(), '--', color='C1', lw=0.8, label='narrow filter')
    ax.plot(freqs_MHz, H_jitter / H_jitter.max(), ':',  color='C3', lw=0.8, label='jittery filter')
    set_ax(ax, xlabel='frequency (MHz)', ylabel='(norm.)',
           title='Filter transfer functions', legend=True, grid_alpha=0.3,
           xlim=xlim_MHz)
    plt.tight_layout()
    if save: fig.savefig(plotfolder / 'filter_transfer.pdf')
    plt.show()


def plot_all(res: PropagationResult, save=False):
    """Convenience: call all plot functions for a given result."""
    res.summary()
    plot_mw_wavepacket(res, save=save)
    plot_qutip_populations(res, save=save)
    plot_mw_cavity(res, save=save)
    plot_pump(res, save=save)
    plot_conversion_timing(res, save=save)
    plot_upconverted(res, save=save)
    plot_filter_transfer(res, save=save)


def save_result(res: PropagationResult, path: str):
    """Pickle a full PropagationResult to disk."""
    with open(path, 'wb') as fh:
        pickle.dump(res, fh)
    print(f"Saved result ? {path}")


def load_result(path: str) -> PropagationResult:
    """Load a pickled PropagationResult from disk."""
    with open(path, 'rb') as fh:
        res = pickle.load(fh)
    print(f"Loaded result ? {path}")
    return res


@dataclass
class TransducerParams:
    """
    All physical parameters.
    Times: ?s.  Rates: rad/?s.  Optical frequencies: rad/s.
    """
    # -- qubit / qubit-resonator -----------------------------------------------
    gamma_1:    float = 1/19          # qubit relaxation rate (?s^-1)
    gamma_phi:  float = 1/14          # qubit dephasing rate  (?s^-1)
    kappa:      float = 2*np.pi*1.427 # resonator total decay (rad/?s)
    kappa_ext:  float = 2*np.pi*1.0   # resonator ext. coupling (rad/?s)

    # -- BSB pulse -------------------------------------------------------------
    T_bsb:      float = 156e-3        # BSB square pulse duration (?s)
    A_pi_bsb:   float = 11.479        # BSB ?-pulse amplitude
    delay_bsb:  float = 0e-3          # wait before BSB pulse (?s)
    tend_qutip: float = 2.0           # total QuTiP simulation duration (?s)
    dt_qutip:   float = 1e-3          # QuTiP time step (?s)

    # -- Hilbert space ---------------------------------------------------------
    Ntrunc:     int   = 3
    Nq:         int   = 3

    # -- EO transducer ? microwave cavity -------------------------------------
    kappa_eo_mw:  float = 2*np.pi*1.2   # EO MW cavity linewidth (rad/?s)
    eta_e:        float = 0.49           # EO MW coupling efficiency

    # -- EO transducer ? optical cavity ---------------------------------------
    kappa_eo_opt: float = 2*np.pi*11.7  # EO optical cavity linewidth (rad/?s)
    eta_o:        float = 0.306          # optical coupling efficiency
    Lam2:         float = 0.45           # EO input mode matching
    g0:           float = 2*np.pi*6.0   # electro-optic coupling   (rad/s)

    # -- optical pump ---------------------------------------------------------
    w_pump:         float = 2*np.pi*193.5e12   # pump frequency (rad/s)
    power_pump:     float = 1e-3               # pump power (W)
    duration_pump:  float = 200e-9             # pump pulse duration (s ? used in s units)
    delay_pump:     float = 175e-9             # pump pulse delay   (s ? used in s units)
    pump_rise_time: float = 20e-9              # pump rise time (s)
    pump_detuning:  float = 0.0                # pump detuning (rad/s)

    # -- sign flip ? phase gate on MW cavity ----------------------------------
    sign_flip_time: float = 0           # sign flip time (s).  0 = disabled.

    # -- filter cavities -------------------------------------------------------
    kappa_filter_normal: float = 2*np.pi*50e6  # wide filter cavity  (rad/s)
    kappa_filter_narrow: float = 2*np.pi*5e6   # narrow filter cavity (rad/s)
    n_normal_cavities:   int   = 3             # number of cascaded normal cavities
    filter_jitter:       float = 0.05           # narrow cavity jitter (fraction of kappa)

    @classmethod
    def from_experiment(cls, **overrides):
        """Return default experimental parameters, optionally overriding any field."""
        return cls(**overrides)


@dataclass
class TransducerLosses:
    """
    Extrinsic loss factors along the pickup chain.
    Each factor multiplies the *amplitude* at the corresponding stage.
    Default: ideal (no losses, all factors = 1.0).
    """
    qubit_to_eo:       float = 1.0  # cable + mismatch qubit cavity ? EO
    qubit_outcoupling: float = 1.0  # kappa_ext/kappa  (set automatically if desired)
    fridge_outcoupling:float = 1.0  # cable loss fridge ? room temperature
    sig_cavity:        float = 1.0  # signal path cavity coupling loss
    aom:               float = 1.0  # AOM loss

    @classmethod
    def ideal(cls):
        """All losses = 1 (ideal, lossless chain)."""
        return cls()

    @classmethod
    def from_experiment(cls, p: 'TransducerParams'):
        """Return the experimentally measured loss values from the original notebook."""
        return cls(
            qubit_to_eo       = 1.45,
            qubit_outcoupling = p.kappa_ext / p.kappa,
            fridge_outcoupling= np.sqrt(0.27),
            sig_cavity        = 0.44,
            aom               = 0.5,
        )

    def total(self) -> float:
        return (self.qubit_to_eo * self.qubit_outcoupling *
                self.fridge_outcoupling * self.sig_cavity * self.aom)
    

@dataclass
class PropagationResult:
    """
    All intermediate and final fields for one simulation run.
    Axes convention: time in microseconds unless noted (_s = seconds).
    """
    # -- time axes -------------------------------------------------------------
    t_qutip:       np.ndarray = field(default_factory=lambda: np.array([]))  # ?s
    t_prop:        np.ndarray = field(default_factory=lambda: np.array([]))  # ?s (padded)
    freqs:         np.ndarray = field(default_factory=lambda: np.array([]))  # Hz

    # -- microwave photon (QuTiP output) --------------------------------------
    one_pop:       np.ndarray = field(default_factory=lambda: np.array([]))  # resonator |1> pop
    fock_env:      np.ndarray = field(default_factory=lambda: np.array([]))  # normalised MW amplitude
    fock_padded:   np.ndarray = field(default_factory=lambda: np.array([]))  # zero-padded for FFT

    # -- MW cavity stage -------------------------------------------------------
    internal_mw:   np.ndarray = field(default_factory=lambda: np.array([]))  # EO MW inner field
    reflected_mw:  np.ndarray = field(default_factory=lambda: np.array([]))  # EO MW reflected

    # -- optical pump ---------------------------------------------------------
    pump_input:    np.ndarray = field(default_factory=lambda: np.array([]))
    reflected_pump:np.ndarray = field(default_factory=lambda: np.array([]))
    internal_pump: np.ndarray = field(default_factory=lambda: np.array([]))

    # -- conversion ------------------------------------------------------------
    upconv_fock:   np.ndarray = field(default_factory=lambda: np.array([]))  # optical amplitude

    # -- filtered output -------------------------------------------------------
    filtered_fock: np.ndarray = field(default_factory=lambda: np.array([]))

    # -- scalar metrics --------------------------------------------------------
    p_mw_in:          float = 0.0   # photon probability at MW input
    p_converted:      float = 0.0   # conversion probability (before filtering)
    p_detected:       float = 0.0   # after signal filtering
    pump_contrast:    float = 0.0   # reflected pump contrast at chosen time
    params:           Optional['TransducerParams'] = None
    losses:           Optional['TransducerLosses'] = None

    def summary(self):
        print(f"  MW input           : {self.p_mw_in:.4f} photons")
        print(f"  Conversion prob.   : {self.p_converted:.4f}")
        print(f"  After filtering    : {self.p_detected:.4f}")
        print(f"  Pump contrast      : {self.pump_contrast:.4f}")


class PhotonPropagationSim:
    """
    Simulate the full chain: qubit-cavity ? EO transducer ? optical detection.

    Usage
    -----
    sim = PhotonPropagationSim(params, losses)
    res = sim.run()                          # single run
    res = sim.run(delay_pump=200e-9)         # override pump delay for this run
    """

    def __init__(self, p: TransducerParams, losses: TransducerLosses = None):
        self.p      = p
        self.losses = losses if losses is not None else TransducerLosses.ideal()
        self._wavepacket_cache: Optional[dict] = None

    # -------------------------------------------------------------------------
    # Stage 1: QuTiP mesolve ? microwave photon wavepacket
    # -------------------------------------------------------------------------
    def _run_qutip(self, force_rerun: bool = False) -> dict:
        """Run (or return cached) QuTiP BSB simulation."""
        if self._wavepacket_cache is not None and not force_rerun:
            return self._wavepacket_cache

        p = self.p
        Ntrunc, Nq = p.Ntrunc, p.Nq

        # Hilbert space
        vac  = qt.basis(Ntrunc, 0)
        one  = qt.basis(Ntrunc, 1)
        g    = qt.basis(Nq, 0)
        e    = qt.basis(Nq, 1)
        b    = qt.tensor(qt.destroy(Nq), qt.identity(Ntrunc))
        a    = qt.tensor(qt.identity(Nq), qt.destroy(Ntrunc))

        bsb_transition = (qt.tensor(e * g.dag(), one * vac.dag()) +
                          qt.tensor(g * e.dag(), vac * one.dag()))
        Hd_bsb = bsb_transition

        # BSB pulse coefficient (closure captures A, T, cen)
        def make_bsb_coeff(A, T, cen):
            def f(t, args):
                return A * (np.abs(t - cen) < T / 2)
            return f

        cen  = p.delay_bsb + p.T_bsb / 2
        H    = [[Hd_bsb, make_bsb_coeff(p.A_pi_bsb, p.T_bsb, cen)]]
        c_ops= [np.sqrt(p.gamma_1) * b,
                np.sqrt(p.gamma_phi) * b.dag() * b,
                np.sqrt(p.kappa) * a]
        tlist = np.arange(0, p.tend_qutip, p.dt_qutip)
        rho0  = qt.tensor(g * g.dag(), vac * vac.dag())

        out     = qt.mesolve(H, rho0, tlist, c_ops,
                             options=qt.Options(max_step=p.T_bsb / 4))
        rhos_res = [qt.ptrace(rho, 1) for rho in out.states]
        one_pop  = np.array([qt.expect(one * one.dag(), r) for r in rhos_res])

        dt        = tlist[1] - tlist[0]
        fock_env  = np.sqrt(np.abs(one_pop))
        fock_env /= np.sqrt(np.sum(np.abs(one_pop)) * dt * 1e-6)   # normalise to 1 photon

        self._wavepacket_cache = dict(tlist=tlist, one_pop=one_pop,
                                      fock_env=fock_env, dt=dt)
        return self._wavepacket_cache

    # -------------------------------------------------------------------------
    # Stage 2?5: propagation chain (all stages after QuTiP)
    # -------------------------------------------------------------------------
    def _build_time_grid(self, fock_env, dt):
        """Build zero-padded time grid for FFT-based propagation."""
        N = len(fock_env)
        N_pad = int(N / 4)
        t_prop = np.linspace(-dt * N_pad, dt * (N - 1), N + N_pad)   # ?s
        fock_padded = np.pad(fock_env, (N_pad, 0), constant_values=0)
        freqs = np.fft.fftfreq(len(t_prop), dt * 1e-6)                # Hz
        return t_prop, fock_padded, freqs

    def run(self, delay_pump: Optional[float] = None,
            duration_pump: Optional[float] = None,
            sign_flip_time: Optional[float] = None,
            pump_detuning: Optional[float] = None,
            force_qutip_rerun: bool = False) -> PropagationResult:
        """
        Run the full propagation chain.

        Parameters
        ----------
        delay_pump      : override p.delay_pump (seconds)
        duration_pump   : override p.duration_pump (seconds)
        sign_flip_time  : override p.sign_flip_time (seconds); 0 = disabled
        pump_detuning   : override p.pump_detuning (rad/s)
        force_qutip_rerun : rerun the QuTiP stage even if cached

        Returns
        -------
        PropagationResult with all intermediate and final fields.
        """
        p = self.p
        L = self.losses

        # -- apply per-call overrides ------------------------------------------
        d_pump  = delay_pump     if delay_pump     is not None else p.delay_pump
        dur_pump= duration_pump  if duration_pump  is not None else p.duration_pump
        sft     = sign_flip_time if sign_flip_time is not None else p.sign_flip_time
        detuning= pump_detuning  if pump_detuning  is not None else p.pump_detuning

        # -- Stage 1: microwave wavepacket -------------------------------------
        wv   = self._run_qutip(force_rerun=force_qutip_rerun)
        tlist, one_pop, fock_env, dt = (wv['tlist'], wv['one_pop'],
                                         wv['fock_env'], wv['dt'])

        # -- Build propagation grid --------------------------------------------
        t_prop, fock_padded, freqs = self._build_time_grid(fock_env, dt)
        fock_fft = np.fft.fft(fock_padded)

        # -- Stage 2: MW cavity (EO microwave port) ----------------------------
        omega = freqs * 2 * np.pi      # rad/s ? but kappa_eo_mw is in rad/?s
        # Convert: freqs in Hz, kappa_eo_mw in rad/?s ? both to rad/s
        H_mw = H_res(omega, 0,
                     p.kappa_eo_mw * 1e6,    # rad/?s ? rad/s
                     p.eta_e)
        # sign flip (phase gate): flip sign of padded signal after sft
        fock_padded_flipped = fock_padded.copy()
        if sft > 0:
            fock_padded_flipped[t_prop * 1e-6 >= sft] *= -1
        fock_fft_flip = np.fft.fft(fock_padded_flipped)
        sqrt_ke_etae = np.sqrt(p.kappa_eo_mw * 1e6 * p.eta_e)
        reflected_mw = np.fft.ifft((1 - sqrt_ke_etae * H_mw) * fock_fft_flip)
        internal_mw  = np.fft.ifft(H_mw * fock_fft)

        # -- Stage 3: optical pump ---------------------------------------------
        Npump_flux  = np.sqrt(p.power_pump / hbar / p.w_pump)
        H_opt = H_res(omega, 0,
                      p.kappa_eo_opt * 1e6,   # rad/?s ? rad/s
                      p.eta_o * np.sqrt(p.Lam2))

        idx1 = np.searchsorted(t_prop * 1e-6, d_pump)
        idx2 = np.searchsorted(t_prop * 1e-6, d_pump + dur_pump)
        pump_input = np.zeros(len(t_prop), dtype=complex)
        pump_input[idx1:idx2] = (Npump_flux *
            np.exp(1j * detuning * t_prop[idx1:idx2] * 1e-6))
        sig_smooth = max(1, int(p.pump_rise_time / 5 / (dt * 1e-6)))
        pump_input     = gaussian_filter1d(pump_input, sig_smooth)
        pump_fft       = np.fft.fft(pump_input)
        sqrt_ko_etao   = np.sqrt(p.kappa_eo_opt * 1e6 * p.eta_o * np.sqrt(p.Lam2))
        reflected_pump = np.fft.ifft((1 - sqrt_ko_etao * H_opt) * pump_fft)
        internal_pump  = np.fft.ifft(H_opt * pump_fft)

        # pump contrast at 180 ns after pulse start
        idx_ref = np.searchsorted(t_prop * 1e-6, d_pump + 180e-9)
        pump_contrast = (np.abs(reflected_pump[idx_ref])**2 /
                         max(np.max(np.abs(reflected_pump)**2), 1e-300))

        # -- Stage 4: electro-optic upconversion -------------------------------
        loss_amp = np.sqrt(L.total())    # loss applied as amplitude scaling
        g_pump   = internal_pump * p.g0   # rad/?s ? rad/s

        upconv_fock = timed_conversion(
            t_prop * 1e-6,                            # s
            fock_padded_flipped * loss_amp,            # MW amplitude (with losses)
            g_pump.copy(),                             # coupling g(t)
            p.kappa_eo_mw * 1e6,                      # rad/s
            p.kappa_eo_opt * 1e6,                      # rad/s
            p.eta_e,
            p.eta_o * np.sqrt(p.Lam2),
            detuning,
            sft,                                       # flip time (s), 0 = disabled
        )

        # -- Stage 5: signal filtering -----------------------------------------
        upconv_fft = np.fft.fft(upconv_fock)
        # 3? wide cavity, then narrow jittery cavity
        med_fft    = apply_cavity_filter(upconv_fft, freqs,
                                         p.kappa_filter_normal,
                                         n_cascades=p.n_normal_cavities)
        final_fft  = apply_jittery_filter(np.fft.fft(np.fft.ifft(med_fft)),
                                          freqs,
                                          p.kappa_filter_narrow,
                                          p.filter_jitter)
        filtered_fock = np.fft.ifft(final_fft)

        # -- Scalar metrics ----------------------------------------------------
        p_mw_in     = float(np.sum(np.abs(fock_padded)**2) * dt * 1e-6)
        p_converted = float(np.sum(np.abs(upconv_fock)**2) * dt * 1e-6)
        p_detected  = float(np.sum(np.abs(filtered_fock)**2) * dt * 1e-6)

        return PropagationResult(
            t_qutip=tlist, t_prop=t_prop, freqs=freqs,
            one_pop=one_pop, fock_env=fock_env, fock_padded=fock_padded,
            internal_mw=internal_mw, reflected_mw=reflected_mw,
            pump_input=pump_input, reflected_pump=reflected_pump,
            internal_pump=internal_pump,
            upconv_fock=upconv_fock, filtered_fock=filtered_fock,
            p_mw_in=p_mw_in, p_converted=p_converted, p_detected=p_detected,
            pump_contrast=float(np.real(pump_contrast)),
            params=p, losses=L,
        )
