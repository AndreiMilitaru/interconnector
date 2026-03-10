"""
Photon Counting Data Analysis
Author: Andrei Militaru with GitHub Copilot
Organization: Institute of Science and Technology Austria (ISTA)
Date: March 2026

Description:
    Object-oriented data analysis tools for photon counting HDF5 data files.
    Supports data collected by both photon_counting.py (SNR / single-channel)
    and Z_basis_photon_counting.py (Z-basis / two-channel) acquisition modules.

    Both modules save data to an HDF5 file named 'data' with the following structure:
        - Datasets named '0', '1', '2', ... each containing a 2D array of photon
          rows with columns:
              [seconds_count, sync_number, channel, marker_type, bin_index, unix_timestamp]
        - A 'lock_log' dataset containing one total-photon-count value per second.

    Marker conventions:
        photon_counting.py   -> 0: no marker, 1: marker_1 (ch 128), 2: marker_2 (ch 131)
        Z_basis_photon_counting.py -> 0: no marker, 1: Noise, 2: G state, 3: E state
"""

import numpy as np
import h5py
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# Column indices in the photon-row arrays
_COL_SECONDS  = 0
_COL_SYNC     = 1
_COL_CHANNEL  = 2
_COL_MARKER   = 3
_COL_BIN      = 4
_COL_TIME     = 5


# ============================================================================
# PHOTON COUNTING ANALYZER  (photon_counting.py data)
# ============================================================================

class PhotonCountingAnalyzer:
    """
    Analyzes HDF5 data files produced by photon_counting.py.

    Each photon row stores: [seconds_count, sync_number, channel,
                              marker, bin_index, unix_timestamp]

    Marker convention: 0 = no marker, 1 = marker_1 (ID 128), 2 = marker_2 (ID 131)

    Parameters
    ----------
    input_path : str, Path, or list of str/Path
        A single HDF5 file, a list of HDF5 files, or a directory containing
        HDF5 files named 'data' (no extension) or '*.h5'.
    bin_size : float, optional
        Timetrace bin width in picoseconds. Default: 250 ps.
    window_start : float, optional
        Start of the detection window relative to the sync pulse, in
        picoseconds. Default: 4 000 000 ps (= 4 µs).
    window_length : float, optional
        Length of the detection window in picoseconds. Default: 4 500 000 ps.
    """

    def __init__(
        self,
        input_path: Union[str, Path, List],
        bin_size: float = 250,
        window_start: float = 4_000_000,
        window_length: float = 4_500_000,
    ):
        self.bin_size = bin_size          # ps
        self.window_start = window_start  # ps
        self.window_length = window_length  # ps

        self._timestamp_mask: Optional[np.ndarray] = None
        self._accepted_seconds: Optional[set] = None

        self.data_files: List[Path] = []
        self.parsed_data: Dict[str, Dict] = {}

        # Resolve file list
        if isinstance(input_path, (str, Path)):
            path = Path(input_path)
            if path.is_dir():
                # Accept files named 'data' (no extension) or *.h5
                self.data_files = sorted(
                    list(path.glob("data")) + list(path.glob("*.h5"))
                )
            elif path.is_file():
                self.data_files = [path]
            else:
                raise FileNotFoundError(f"Path not found: {input_path}")
        elif isinstance(input_path, list):
            self.data_files = [Path(f) for f in input_path]
        else:
            raise TypeError("input_path must be a str, Path, or list.")

        if not self.data_files:
            raise ValueError(f"No HDF5 data files found at: {input_path}")

        self._parse_all_files()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_all_files(self) -> None:
        """Parse all HDF5 files and store results in self.parsed_data."""
        for f in self.data_files:
            self.parsed_data[str(f)] = self._parse_file(f)

    def _parse_file(self, file_path: Path) -> Dict:
        """
        Parse a single HDF5 data file.

        Returns a dictionary with:
            'datasets'  : dict mapping dataset index (int) -> photon array (N×6)
            'lock_log'  : 1-D array of total photon counts per second
            'n_datasets': number of photon datasets found
        """
        parsed = {
            "datasets": {},
            "lock_log": np.array([]),
            "n_datasets": 0,
        }

        with h5py.File(file_path, "r") as f:
            # Load numbered photon datasets
            dataset_keys = sorted(
                [k for k in f.keys() if k.isdigit()], key=int
            )
            for k in dataset_keys:
                arr = np.array(f[k])
                if arr.ndim == 2 and arr.shape[1] == 6:
                    parsed["datasets"][int(k)] = arr

            # Load lock log
            if "lock_log" in f:
                parsed["lock_log"] = np.array(f["lock_log"])

        parsed["n_datasets"] = len(parsed["datasets"])
        return parsed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_rows(self) -> np.ndarray:
        """Return all photon rows concatenated across all files and datasets,
        respecting any active timestamp mask."""
        parts = []
        for file_data in self.parsed_data.values():
            for ds_idx, arr in file_data["datasets"].items():
                if len(arr) == 0:
                    continue
                if self._accepted_seconds is not None:
                    seconds = arr[:, _COL_SECONDS].astype(int)
                    keep = np.zeros(len(arr), dtype=bool)
                    for s in np.unique(seconds):
                        if (ds_idx, int(s)) in self._accepted_seconds:
                            keep |= (seconds == s)
                    arr = arr[keep]
                if len(arr) > 0:
                    parts.append(arr)
        if not parts:
            return np.empty((0, 6))
        return np.vstack(parts)

    def _per_second_timestamps(self) -> np.ndarray:
        """
        Build a timestamp array aligned to the lock_log (one value per second)
        using the unix_timestamp recorded for the first photon in each unique
        measurement second.  Always returns the full unmasked array.

        The lock_log is appended across datasets; the returned array has the
        same length.
        """
        timestamps = []
        for file_data in self.parsed_data.values():
            for ds_idx in sorted(file_data["datasets"]):
                arr = file_data["datasets"][ds_idx]
                if len(arr) == 0:
                    continue
                # Group by seconds_count and take first unix_timestamp
                seconds = arr[:, _COL_SECONDS].astype(int)
                for s in np.unique(seconds):
                    mask = seconds == s
                    ts = arr[mask, _COL_TIME][0]
                    timestamps.append((ds_idx, int(s), ts))

        # Sort by (dataset_index, seconds_count) to respect acquisition order
        timestamps.sort(key=lambda x: (x[0], x[1]))
        if not timestamps:
            return np.array([])
        return np.array([t[2] for t in timestamps])

    def set_timestamp_mask(self, mask) -> None:
        """
        Restrict all analysis to seconds where *mask* is True.

        Parameters
        ----------
        mask : array-like of bool
            Boolean array of the same length as the unmasked
            ``get_timestamps()`` output.  True = include, False = exclude.
        """
        ts_info = []
        for file_data in self.parsed_data.values():
            for ds_idx in sorted(file_data["datasets"]):
                arr = file_data["datasets"][ds_idx]
                if len(arr) == 0:
                    continue
                seconds = arr[:, _COL_SECONDS].astype(int)
                for s in np.unique(seconds):
                    row_mask = seconds == s
                    ts = arr[row_mask, _COL_TIME][0]
                    ts_info.append((ds_idx, int(s), ts))
        ts_info.sort(key=lambda x: (x[0], x[1]))

        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(ts_info):
            raise ValueError(
                f"Mask length ({len(mask)}) does not match the number of "
                f"timestamps ({len(ts_info)})."
            )
        self._timestamp_mask = mask
        self._accepted_seconds = {
            (info[0], info[1]) for info, m in zip(ts_info, mask) if m
        }

    def clear_timestamp_mask(self) -> None:
        """Remove any active timestamp mask and restore full data access."""
        self._timestamp_mask = None
        self._accepted_seconds = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_timestamps(self) -> np.ndarray:
        """
        Return an array of unix timestamps, one per measurement second,
        sorted in acquisition order.  If a timestamp mask is active, only
        the accepted seconds are returned.

        Returns
        -------
        np.ndarray
            Unix timestamps (float) for each second of data.
        """
        ts = self._per_second_timestamps()
        if self._timestamp_mask is not None:
            return ts[self._timestamp_mask[:len(ts)]]
        return ts

    def get_lock_signal(self) -> Dict:
        """
        Return the lock signal (total photon count per second) with
        corresponding unix timestamps.  Respects any active timestamp mask.

        Returns
        -------
        dict :
            timestamps (np.ndarray): unix timestamp for each second
            values (np.ndarray)    : total photon counts recorded per second
        """
        lock_parts = []
        for file_data in self.parsed_data.values():
            if len(file_data["lock_log"]) > 0:
                lock_parts.append(file_data["lock_log"])

        lock_log = np.concatenate(lock_parts) if lock_parts else np.array([])
        timestamps = self._per_second_timestamps()

        # Align lengths (a file without photons still contributes to lock_log)
        min_len = min(len(timestamps), len(lock_log))
        timestamps = timestamps[:min_len]
        values = lock_log[:min_len]

        if self._timestamp_mask is not None:
            m = self._timestamp_mask[:min_len]
            timestamps = timestamps[m]
            values = values[m]

        return {"timestamps": timestamps, "values": values}

    def get_timetrace(self) -> Dict:
        """
        Reconstruct the time-trace histograms accumulated over all loaded data.

        Returns
        -------
        dict :
            time_ns (np.ndarray)  : bin centres in nanoseconds (from window start)
            total (np.ndarray)    : all photons regardless of marker
            marker_1 (np.ndarray) : photons associated with marker 1
            marker_2 (np.ndarray) : photons associated with marker 2
            n_syncs (int)         : number of unique sync pulses in the data
        """
        rows = self._all_rows()
        if len(rows) == 0:
            empty = np.array([])
            return {"time_ns": empty, "total": empty,
                    "marker_1": empty, "marker_2": empty, "n_syncs": 0}

        syncs = rows[:, _COL_SYNC].astype(int)
        n_syncs = int(syncs.max() - syncs.min() + 1)

        bin_indices = rows[:, _COL_BIN].astype(int)
        markers    = rows[:, _COL_MARKER].astype(int)

        num_bins = int(bin_indices.max()) + 1
        time_ns = np.arange(num_bins) * self.bin_size / 1e3  # ns from window start

        total    = np.zeros(num_bins, dtype=int)
        marker_1 = np.zeros(num_bins, dtype=int)
        marker_2 = np.zeros(num_bins, dtype=int)

        np.add.at(total,    bin_indices, 1)
        np.add.at(marker_1, bin_indices[markers == 1], 1)
        np.add.at(marker_2, bin_indices[markers == 2], 1)

        return {
            "time_ns": time_ns,
            "total": total,
            "marker_1": marker_1,
            "marker_2": marker_2,
            "n_syncs": n_syncs,
        }

    def get_count_rates(self) -> Dict:
        """
        Return per-second photon counts split by marker type.

        Returns
        -------
        dict :
            timestamps (np.ndarray): unix timestamp per second
            total (np.ndarray)     : total photons per second (inside window)
            marker_1 (np.ndarray)  : marker-1 photons per second
            marker_2 (np.ndarray)  : marker-2 photons per second
            n_syncs (np.ndarray)   : unique sync pulses detected per second
        """
        parts_ts, parts_total, parts_m1, parts_m2, parts_syncs = [], [], [], [], []

        for file_data in self.parsed_data.values():
            for ds_idx in sorted(file_data["datasets"]):
                arr = file_data["datasets"][ds_idx]
                if len(arr) == 0:
                    continue
                seconds   = arr[:, _COL_SECONDS].astype(int)
                markers   = arr[:, _COL_MARKER].astype(int)

                for s in np.unique(seconds):
                    if self._accepted_seconds is not None and \
                            (ds_idx, int(s)) not in self._accepted_seconds:
                        continue
                    mask = seconds == s
                    parts_ts.append(arr[mask, _COL_TIME][0])
                    parts_total.append(np.sum(mask))
                    parts_m1.append(np.sum(mask & (markers == 1)))
                    parts_m2.append(np.sum(mask & (markers == 2)))
                    s_col = arr[mask, _COL_SYNC].astype(int)
                    parts_syncs.append(int(s_col.max() - s_col.min() + 1))

        if not parts_ts:
            empty = np.array([])
            return {"timestamps": empty, "total": empty,
                    "marker_1": empty, "marker_2": empty, "n_syncs": empty}

        order = np.argsort(parts_ts)
        return {
            "timestamps": np.array(parts_ts)[order],
            "total":      np.array(parts_total)[order],
            "marker_1":   np.array(parts_m1)[order],
            "marker_2":   np.array(parts_m2)[order],
            "n_syncs":    np.array(parts_syncs)[order],
        }


# ============================================================================
# Z-BASIS PHOTON COUNTING ANALYZER  (Z_basis_photon_counting.py data)
# ============================================================================

class ZBasisPhotonCountingAnalyzer:
    """
    Analyzes HDF5 data files produced by Z_basis_photon_counting.py.

    Each photon row stores: [seconds_count, sync_number, channel,
                              marker_type, bin_index, unix_timestamp]

    Channels:  1 = photon channel 1,  2 = photon channel 2
    Marker convention:
        0 = no marker
        1 = Noise  (marker_1 only, ID 128)
        2 = G state (marker_2 only, ID 131)
        3 = E state (both markers detected)

    Parameters
    ----------
    input_path : str, Path, or list of str/Path
        A single HDF5 file, a list of HDF5 files, or a directory containing
        HDF5 files named 'data' (no extension) or '*.h5'.
    bin_size : float, optional
        Timetrace bin width in picoseconds. Default: 250 ps.
    start_offset : float, optional
        Start of the detection window relative to the sync pulse, in ps.
        Default: 4 000 000 ps.
    detection_length : float, optional
        Length of the detection window in picoseconds. Default: 4 500 000 ps.
    bin1_offset : float, optional
        Start of the first temporal bin relative to start_offset, in ps.
        Default: 1 500 000 ps.
    bin1_length : float, optional
        Length of the first temporal bin in ps. Default: 500 000 ps.
    bin2_offset : float, optional
        Start of the second temporal bin relative to start_offset, in ps.
        Default: 2 500 000 ps.
    bin2_length : float, optional
        Length of the second temporal bin in ps. Default: 500 000 ps.
    """

    def __init__(
        self,
        input_path: Union[str, Path, List],
        bin_size: float = 250,
        start_offset: float = 4_000_000,
        detection_length: float = 4_500_000,
        bin1_offset: float = 1_500_000,
        bin1_length: float = 500_000,
        bin2_offset: float = 2_500_000,
        bin2_length: float = 500_000,
    ):
        self.bin_size        = bin_size
        self.start_offset    = start_offset
        self.detection_length = detection_length
        self.bin1_offset     = bin1_offset
        self.bin1_length     = bin1_length
        self.bin2_offset     = bin2_offset
        self.bin2_length     = bin2_length

        # Derived bin boundaries (in units of bin_size = bin index space)
        self._bin1_start_idx = int(bin1_offset // bin_size)
        self._bin1_end_idx   = int((bin1_offset + bin1_length) // bin_size)
        self._bin2_start_idx = int(bin2_offset // bin_size)
        self._bin2_end_idx   = int((bin2_offset + bin2_length) // bin_size)

        self._timestamp_mask: Optional[np.ndarray] = None
        self._accepted_seconds: Optional[set] = None

        self.data_files: List[Path] = []
        self.parsed_data: Dict[str, Dict] = {}

        # Resolve file list
        if isinstance(input_path, (str, Path)):
            path = Path(input_path)
            if path.is_dir():
                self.data_files = sorted(
                    list(path.glob("data")) + list(path.glob("*.h5"))
                )
            elif path.is_file():
                self.data_files = [path]
            else:
                raise FileNotFoundError(f"Path not found: {input_path}")
        elif isinstance(input_path, list):
            self.data_files = [Path(f) for f in input_path]
        else:
            raise TypeError("input_path must be a str, Path, or list.")

        if not self.data_files:
            raise ValueError(f"No HDF5 data files found at: {input_path}")

        self._parse_all_files()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_all_files(self) -> None:
        """Parse all HDF5 files and store results in self.parsed_data."""
        for f in self.data_files:
            self.parsed_data[str(f)] = self._parse_file(f)

    def _parse_file(self, file_path: Path) -> Dict:
        """
        Parse a single HDF5 data file.

        Returns a dictionary with:
            'datasets'  : dict mapping dataset index (int) -> photon array (N×6)
            'lock_log'  : 1-D array of total photon counts per second
            'n_datasets': number of photon datasets found
        """
        parsed = {
            "datasets": {},
            "lock_log": np.array([]),
            "n_datasets": 0,
        }

        with h5py.File(file_path, "r") as f:
            dataset_keys = sorted(
                [k for k in f.keys() if k.isdigit()], key=int
            )
            for k in dataset_keys:
                arr = np.array(f[k])
                if arr.ndim == 2 and arr.shape[1] == 6:
                    parsed["datasets"][int(k)] = arr

            if "lock_log" in f:
                parsed["lock_log"] = np.array(f["lock_log"])

        parsed["n_datasets"] = len(parsed["datasets"])
        return parsed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_rows(self) -> np.ndarray:
        """Return all photon rows concatenated across all files and datasets,
        respecting any active timestamp mask."""
        parts = []
        for file_data in self.parsed_data.values():
            for ds_idx, arr in file_data["datasets"].items():
                if len(arr) == 0:
                    continue
                if self._accepted_seconds is not None:
                    seconds = arr[:, _COL_SECONDS].astype(int)
                    keep = np.zeros(len(arr), dtype=bool)
                    for s in np.unique(seconds):
                        if (ds_idx, int(s)) in self._accepted_seconds:
                            keep |= (seconds == s)
                    arr = arr[keep]
                if len(arr) > 0:
                    parts.append(arr)
        if not parts:
            return np.empty((0, 6))
        return np.vstack(parts)

    def _per_second_timestamps(self) -> np.ndarray:
        """
        Build a timestamp array aligned to the lock_log using the unix_timestamp
        of the first photon recorded in each unique measurement second.
        Always returns the full unmasked array.
        """
        timestamps = []
        for file_data in self.parsed_data.values():
            for ds_idx in sorted(file_data["datasets"]):
                arr = file_data["datasets"][ds_idx]
                if len(arr) == 0:
                    continue
                seconds = arr[:, _COL_SECONDS].astype(int)
                for s in np.unique(seconds):
                    mask = seconds == s
                    ts = arr[mask, _COL_TIME][0]
                    timestamps.append((ds_idx, int(s), ts))

        timestamps.sort(key=lambda x: (x[0], x[1]))
        if not timestamps:
            return np.array([])
        return np.array([t[2] for t in timestamps])

    def set_timestamp_mask(self, mask) -> None:
        """
        Restrict all analysis to seconds where *mask* is True.

        Parameters
        ----------
        mask : array-like of bool
            Boolean array of the same length as the unmasked
            ``get_timestamps()`` output.  True = include, False = exclude.
        """
        ts_info = []
        for file_data in self.parsed_data.values():
            for ds_idx in sorted(file_data["datasets"]):
                arr = file_data["datasets"][ds_idx]
                if len(arr) == 0:
                    continue
                seconds = arr[:, _COL_SECONDS].astype(int)
                for s in np.unique(seconds):
                    row_mask = seconds == s
                    ts = arr[row_mask, _COL_TIME][0]
                    ts_info.append((ds_idx, int(s), ts))
        ts_info.sort(key=lambda x: (x[0], x[1]))

        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(ts_info):
            raise ValueError(
                f"Mask length ({len(mask)}) does not match the number of "
                f"timestamps ({len(ts_info)})."
            )
        self._timestamp_mask = mask
        self._accepted_seconds = {
            (info[0], info[1]) for info, m in zip(ts_info, mask) if m
        }

    def clear_timestamp_mask(self) -> None:
        """Remove any active timestamp mask and restore full data access."""
        self._timestamp_mask = None
        self._accepted_seconds = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_timestamps(self) -> np.ndarray:
        """
        Return an array of unix timestamps, one per measurement second,
        sorted in acquisition order.  If a timestamp mask is active, only
        the accepted seconds are returned.

        Returns
        -------
        np.ndarray
            Unix timestamps (float) for each second of data.
        """
        ts = self._per_second_timestamps()
        if self._timestamp_mask is not None:
            return ts[self._timestamp_mask[:len(ts)]]
        return ts

    def get_lock_signal(self) -> Dict:
        """
        Return the lock signal (total photon count per second) with
        corresponding unix timestamps.  Respects any active timestamp mask.

        Returns
        -------
        dict :
            timestamps (np.ndarray): unix timestamp for each second
            values (np.ndarray)    : total photon counts (ch1 + ch2) per second
        """
        lock_parts = []
        for file_data in self.parsed_data.values():
            if len(file_data["lock_log"]) > 0:
                lock_parts.append(file_data["lock_log"])

        lock_log = np.concatenate(lock_parts) if lock_parts else np.array([])
        timestamps = self._per_second_timestamps()

        min_len = min(len(timestamps), len(lock_log))
        timestamps = timestamps[:min_len]
        values = lock_log[:min_len]

        if self._timestamp_mask is not None:
            m = self._timestamp_mask[:min_len]
            timestamps = timestamps[m]
            values = values[m]

        return {"timestamps": timestamps, "values": values}

    def get_timetrace(self, channel: Optional[int] = None) -> Dict:
        """
        Reconstruct time-trace histograms accumulated over all loaded data,
        split by marker type (Noise / Signal = G + E combined).

        Parameters
        ----------
        channel : int or None, optional
            If 1 or 2, restrict to that channel only.
            If None (default), both channels are summed together.

        Returns
        -------
        dict :
            time_ns (np.ndarray) : bin centres in ns (from detection-window start)
            noise (np.ndarray)   : photons with Noise marker (type 1)
            g_state (np.ndarray) : photons with G-state marker (type 2)
            e_state (np.ndarray) : photons with E-state marker (type 3)
            total (np.ndarray)   : all photons regardless of marker
            n_syncs (int)        : number of unique sync pulses in the data
                                   (computed before channel filtering, as sync
                                   pulses are shared across both channels)
        """
        rows = self._all_rows()
        if len(rows) == 0:
            empty = np.array([])
            return {"time_ns": empty, "noise": empty,
                    "g_state": empty, "e_state": empty, "total": empty,
                    "n_syncs": 0}

        # Count syncs before channel filtering: sync pulses are global
        syncs = rows[:, _COL_SYNC].astype(int)
        n_syncs = int(syncs.max() - syncs.min() + 1)

        if channel is not None:
            rows = rows[rows[:, _COL_CHANNEL].astype(int) == channel]

        if len(rows) == 0:
            empty = np.array([])
            return {"time_ns": empty, "noise": empty,
                    "g_state": empty, "e_state": empty, "total": empty,
                    "n_syncs": n_syncs}

        bin_indices = rows[:, _COL_BIN].astype(int)
        markers     = rows[:, _COL_MARKER].astype(int)

        num_bins = int(bin_indices.max()) + 1
        time_ns  = np.arange(num_bins) * self.bin_size / 1e3  # ns from window start

        total   = np.zeros(num_bins, dtype=int)
        noise   = np.zeros(num_bins, dtype=int)
        g_state = np.zeros(num_bins, dtype=int)
        e_state = np.zeros(num_bins, dtype=int)

        np.add.at(total,   bin_indices, 1)
        np.add.at(noise,   bin_indices[markers == 1], 1)
        np.add.at(g_state, bin_indices[markers == 2], 1)
        np.add.at(e_state, bin_indices[markers == 3], 1)

        return {
            "time_ns": time_ns,
            "noise":   noise,
            "g_state": g_state,
            "e_state": e_state,
            "total":   total,
            "n_syncs": n_syncs,
        }

    def get_bin_counts(self, channel: Optional[int] = None) -> Dict:
        """
        Return cumulative photon counts in the two temporal bins
        (early and late) for each marker type.

        The six bins follow the convention [N1, N2, G1, G2, E1, E2]:
            N1/N2 = Noise photons in early/late bin
            G1/G2 = G-state photons in early/late bin
            E1/E2 = E-state photons in early/late bin

        Parameters
        ----------
        channel : int or None, optional
            Restrict to channel 1 or 2. Default (None) sums both.

        Returns
        -------
        dict :
            labels  (list)      : ['N1','N2','G1','G2','E1','E2']
            counts  (np.ndarray): cumulative counts for each bin label
            E_z_g   (float)     : E_z discriminator for G state = G2/(G1+G2)
            E_z_e   (float)     : E_z discriminator for E state = E1/(E1+E2)
        """
        rows = self._all_rows()
        if len(rows) == 0:
            return {
                "labels": ["N1", "N2", "G1", "G2", "E1", "E2"],
                "counts": np.zeros(6, dtype=int),
                "E_z_g": np.nan,
                "E_z_e": np.nan,
            }

        if channel is not None:
            rows = rows[rows[:, _COL_CHANNEL].astype(int) == channel]

        bin_indices = rows[:, _COL_BIN].astype(int)
        markers     = rows[:, _COL_MARKER].astype(int)

        def _in_temporal_bin(bin_idx, start, end):
            return (bin_idx >= start) & (bin_idx < end)

        in_b1 = _in_temporal_bin(bin_indices,
                                  self._bin1_start_idx, self._bin1_end_idx)
        in_b2 = _in_temporal_bin(bin_indices,
                                  self._bin2_start_idx, self._bin2_end_idx)

        counts = np.zeros(6, dtype=int)
        for i, (mk, temporal_mask) in enumerate(
            [(1, in_b1), (1, in_b2),
             (2, in_b1), (2, in_b2),
             (3, in_b1), (3, in_b2)]
        ):
            counts[i] = np.sum((markers == mk) & temporal_mask)

        g1, g2, e1, e2 = counts[2], counts[3], counts[4], counts[5]
        E_z_g = float(g2) / (g1 + g2) if (g1 + g2) > 0 else np.nan
        E_z_e = float(e1) / (e1 + e2) if (e1 + e2) > 0 else np.nan

        return {
            "labels": ["N1", "N2", "G1", "G2", "E1", "E2"],
            "counts": counts,
            "E_z_g": E_z_g,
            "E_z_e": E_z_e,
        }

    def get_count_rates(self) -> Dict:
        """
        Return per-second photon counts split by marker type (both channels
        combined), with unix timestamps.

        Returns
        -------
        dict :
            timestamps (np.ndarray): unix timestamp per second
            total (np.ndarray)     : total photons per second (inside window)
            noise (np.ndarray)     : Noise-marker photons per second
            g_state (np.ndarray)   : G-state photons per second
            e_state (np.ndarray)   : E-state photons per second
            n_syncs (np.ndarray)   : unique sync pulses detected per second
        """
        parts_ts, parts_total = [], []
        parts_noise, parts_g, parts_e, parts_syncs = [], [], [], []

        for file_data in self.parsed_data.values():
            for ds_idx in sorted(file_data["datasets"]):
                arr = file_data["datasets"][ds_idx]
                if len(arr) == 0:
                    continue
                seconds = arr[:, _COL_SECONDS].astype(int)
                markers = arr[:, _COL_MARKER].astype(int)

                for s in np.unique(seconds):
                    if self._accepted_seconds is not None and \
                            (ds_idx, int(s)) not in self._accepted_seconds:
                        continue
                    mask = seconds == s
                    parts_ts.append(arr[mask, _COL_TIME][0])
                    parts_total.append(np.sum(mask))
                    parts_noise.append(np.sum(mask & (markers == 1)))
                    parts_g.append(np.sum(mask & (markers == 2)))
                    parts_e.append(np.sum(mask & (markers == 3)))
                    s_col = arr[mask, _COL_SYNC].astype(int)
                    parts_syncs.append(int(s_col.max() - s_col.min() + 1))

        if not parts_ts:
            empty = np.array([])
            return {"timestamps": empty, "total": empty,
                    "noise": empty, "g_state": empty, "e_state": empty,
                    "n_syncs": empty}

        order = np.argsort(parts_ts)
        return {
            "timestamps": np.array(parts_ts)[order],
            "total":      np.array(parts_total)[order],
            "noise":      np.array(parts_noise)[order],
            "g_state":    np.array(parts_g)[order],
            "e_state":    np.array(parts_e)[order],
            "n_syncs":    np.array(parts_syncs)[order],
        }

    def get_z_discriminator_vs_time(
        self,
        window_seconds: int = 10,
        channel: Optional[int] = None,
    ) -> Dict:
        """
        Compute the Z-basis discriminator values E_z_g and E_z_e over time
        using a rolling accumulation window.

        Parameters
        ----------
        window_seconds : int, optional
            Number of seconds over which to accumulate counts for each
            discriminator estimate. Default: 10.
        channel : int or None, optional
            Restrict to channel 1 or 2. Default sums both.

        Returns
        -------
        dict :
            timestamps (np.ndarray): centre unix timestamp for each window
            E_z_g (np.ndarray)     : G-state discriminator G2/(G1+G2)
            E_z_e (np.ndarray)     : E-state discriminator E1/(E1+E2)
        """
        rows = self._all_rows()
        if len(rows) == 0:
            empty = np.array([])
            return {"timestamps": empty, "E_z_g": empty, "E_z_e": empty}

        if channel is not None:
            rows = rows[rows[:, _COL_CHANNEL].astype(int) == channel]

        # Sort by unix timestamp
        order = np.argsort(rows[:, _COL_TIME])
        rows = rows[order]

        timestamps_all = rows[:, _COL_TIME]
        bin_indices    = rows[:, _COL_BIN].astype(int)
        markers        = rows[:, _COL_MARKER].astype(int)

        in_b1 = (bin_indices >= self._bin1_start_idx) & \
                (bin_indices <  self._bin1_end_idx)
        in_b2 = (bin_indices >= self._bin2_start_idx) & \
                (bin_indices <  self._bin2_end_idx)

        t_start = timestamps_all[0]
        t_end   = timestamps_all[-1]
        step    = window_seconds

        out_ts, out_ezg, out_eze = [], [], []

        t = t_start
        while t < t_end:
            t_next = t + window_seconds
            win = (timestamps_all >= t) & (timestamps_all < t_next)
            if np.any(win):
                mk_w = markers[win]
                b1_w = in_b1[win]
                b2_w = in_b2[win]

                g1 = np.sum((mk_w == 2) & b1_w)
                g2 = np.sum((mk_w == 2) & b2_w)
                e1 = np.sum((mk_w == 3) & b1_w)
                e2 = np.sum((mk_w == 3) & b2_w)

                E_z_g = float(g2) / (g1 + g2) if (g1 + g2) > 0 else np.nan
                E_z_e = float(e1) / (e1 + e2) if (e1 + e2) > 0 else np.nan

                out_ts.append(0.5 * (t + t_next))
                out_ezg.append(E_z_g)
                out_eze.append(E_z_e)
            t = t_next

        return {
            "timestamps": np.array(out_ts),
            "E_z_g": np.array(out_ezg),
            "E_z_e": np.array(out_eze),
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path

    # -----------------------------------------------------------------------
    # Example 1 – SNR / single-channel photon counting
    # -----------------------------------------------------------------------
    snr_data_dir = Path("photon_counting_data/2026_02_10_1651_SNR_1kHz_200ns_opt_6p6mV")

    analyzer = PhotonCountingAnalyzer(snr_data_dir / "data")

    # Timestamps (one unix float per acquisition second)
    timestamps = analyzer.get_timestamps()
    print(f"[SNR] {len(timestamps)} seconds of data loaded.")
    if len(timestamps):
        t0 = datetime.fromtimestamp(timestamps[0])
        t1 = datetime.fromtimestamp(timestamps[-1])
        print(f"      From {t0:%Y-%m-%d %H:%M:%S}  to  {t1:%Y-%m-%d %H:%M:%S}")

    # Lock signal (total photon flux used as a proxy for lock quality)
    lock = analyzer.get_lock_signal()
    print(f"[SNR] Lock signal: mean = {lock['values'].mean():.1f} counts/s")

    # Time-trace histograms
    tt = analyzer.get_timetrace()
    print(f"[SNR] Timetrace: {len(tt['time_ns'])} bins, "
          f"total photons = {tt['total'].sum()}")

    # Per-second count rates split by marker
    rates = analyzer.get_count_rates()
    print(f"[SNR] Marker-1 avg rate: {rates['marker_1'].mean():.2f} counts/s")
    print(f"[SNR] Marker-2 avg rate: {rates['marker_2'].mean():.2f} counts/s")

    print()

    # -----------------------------------------------------------------------
    # Example 2 – Z-basis photon counting
    # -----------------------------------------------------------------------
    zbasis_data_dir = Path("photon_counting_data/2026_02_10_1705_ZZ_100Hz_65mV")

    z_analyzer = ZBasisPhotonCountingAnalyzer(zbasis_data_dir / "data")

    timestamps_z = z_analyzer.get_timestamps()
    print(f"[Z]   {len(timestamps_z)} seconds of data loaded.")

    # Lock signal
    lock_z = z_analyzer.get_lock_signal()
    print(f"[Z]   Lock signal: mean = {lock_z['values'].mean():.1f} counts/s")

    # Time-trace histograms (both channels combined)
    tt_z = z_analyzer.get_timetrace()
    print(f"[Z]   Timetrace: {len(tt_z['time_ns'])} bins, "
          f"noise = {tt_z['noise'].sum()}, "
          f"G = {tt_z['g_state'].sum()}, "
          f"E = {tt_z['e_state'].sum()}")

    # Binned counts and discriminators
    bins = z_analyzer.get_bin_counts()
    print(f"[Z]   Bin counts: {dict(zip(bins['labels'], bins['counts'].tolist()))}")
    print(f"[Z]   E_z_g = {bins['E_z_g']:.3f},  E_z_e = {bins['E_z_e']:.3f}")

    # Discriminator vs time (10-second windows)
    ez_vs_t = z_analyzer.get_z_discriminator_vs_time(window_seconds=10)
    print(f"[Z]   E_z_g over time: {len(ez_vs_t['timestamps'])} points, "
          f"mean = {np.nanmean(ez_vs_t['E_z_g']):.3f}")
