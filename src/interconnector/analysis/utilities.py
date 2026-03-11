import numpy as np
from typing import Optional

_PC_COL_SECONDS = 0
_PC_COL_MARKER  = 3
_PC_COL_BIN     = 4


def analyze_below_threshold(timestamps, values, threshold, below=True):
    """
    Analyze when values fall below a threshold and calculate statistics.
    
    Parameters:
    -----------
    timestamps : array-like
        Array of timestamps (datetime objects or numeric values)
    values : array-like
        Array of values to analyze
    threshold : float
        Threshold value to compare against
    below: bool, optional
        If True, returning data below, if False data above. Defaults to True.
    Returns:
    --------
    dict : Dictionary containing:
        - 'below_threshold_mask': boolean array indicating where values < threshold
        - 'total_time': total time (in same units as timestamps)
        - 'fraction_below': fraction of total time below threshold (0 to 1)
    """
    # Convert to numpy arrays
    timestamps = np.asarray(timestamps)
    values = np.asarray(values)
    
    # Create mask for values below threshold
    below_threshold_mask = (values < threshold) if below else (values > threshold)
    
    # Calculate time differences between consecutive points
    if len(timestamps) > 1:
        # Calculate time intervals (assuming uniform sampling, or use actual differences)
        time_diffs = np.diff(timestamps)
        # Assign each time interval to the preceding point
        # For the last point, use the average interval
        avg_interval = np.mean(time_diffs)
        time_intervals = np.append(time_diffs, avg_interval)
        
        # Total time below threshold
        total_time_below = np.sum(time_intervals[below_threshold_mask])
        
        # Total duration
        total_duration = timestamps[-1] - timestamps[0]
        
        # Fraction and percentage
        fraction_below = total_time_below / total_duration
    else:
        total_time_below = 0
        total_duration = 0
        fraction_below = 0
    
    return  below_threshold_mask, total_duration, fraction_below


def transfer_mask(timestamps_1, mask_1, timestamps_2):
    """
    Transfer a boolean mask defined on one timestamp vector to a second
    timestamp vector using the strictest possible criterion.

    For each index i in `timestamps_2`, the output mask is True only if the
    entire interval [timestamps_2[i], timestamps_2[i+1]] is covered by a
    region where `mask_1` is exclusively True — meaning every entry of
    `mask_1` whose timestamp falls within that interval (including the
    bounding entries on either side) is True.

    The last element of the output is always False because there is no
    following timestamp to define its interval.

    In all other cases — the interval extends outside the range of
    `timestamps_1`, or any `mask_1` value that covers any part of the
    interval is False — the output is False.

    Parameters
    ----------
    timestamps_1 : array-like
        Reference timestamp vector (must be sorted in ascending order).
    mask_1 : array-like of bool
        Boolean mask of the same length as `timestamps_1`.
    timestamps_2 : array-like
        Target timestamp vector for which a new mask is produced
        (must be sorted in ascending order).

    Returns
    -------
    np.ndarray of bool
        Boolean mask of the same length as `timestamps_2`.

    Examples
    --------
    >>> import numpy as np
    >>> ts1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> m1  = np.array([True, True, False, True, True])
    >>> ts2 = np.array([0.2, 0.8, 1.5, 3.2])
    >>> transfer_mask(ts1, m1, ts2)
    array([ True, False, False, False])
    """
    timestamps_1 = np.asarray(timestamps_1)
    mask_1       = np.asarray(mask_1, dtype=bool)
    timestamps_2 = np.asarray(timestamps_2)
    
    # Convert datetime objects to numeric values if necessary
    if timestamps_1.dtype.kind == 'O':  # Object dtype (likely datetime)
        timestamps_1 = np.array([t.timestamp() if hasattr(t, 'timestamp') else float(t) 
                                 for t in timestamps_1])
    if timestamps_2.dtype.kind == 'O':  # Object dtype (likely datetime)
        timestamps_2 = np.array([t.timestamp() if hasattr(t, 'timestamp') else float(t) 
                                 for t in timestamps_2])

    if len(timestamps_1) != len(mask_1):
        raise ValueError(
            f"timestamps_1 (len {len(timestamps_1)}) and mask_1 "
            f"(len {len(mask_1)}) must have the same length."
        )

    n = len(timestamps_2)
    output_mask = np.zeros(n, dtype=bool)

    # The last element has no successor interval — always False.
    for i in range(n - 1):
        t_lo = timestamps_2[i]
        t_hi = timestamps_2[i + 1]

        # Last index in timestamps_1 with value <= t_lo (left boundary)
        j_left = np.searchsorted(timestamps_1, t_lo, side="right") - 1
        # First index in timestamps_1 with value >= t_hi (right boundary)
        j_right = np.searchsorted(timestamps_1, t_hi, side="left")

        # Reject if the interval extends outside timestamps_1
        if j_left < 0 or j_right >= len(timestamps_1):
            continue

        # All mask_1 entries from j_left to j_right (inclusive) must be True
        output_mask[i] = bool(np.all(mask_1[j_left: j_right + 1]))

    return output_mask


def bootstrap_timetrace(
    analyzer,
    N: int = 500,
    channel: Optional[int] = None,
    random_state=None,
) -> dict:
    """
    Bootstrap the timetrace histogram of a photon counting analyzer.

    For each of the *N* bootstrap replicates, the set of per-second row
    groups is resampled with replacement (same number of seconds as in the
    original data), and the timetrace histogram is recomputed from the
    resampled rows.  Any timestamp mask that is active on the analyzer is
    respected automatically.

    Parameters
    ----------
    analyzer : PhotonCountingAnalyzer or ZBasisPhotonCountingAnalyzer
        An already-loaded analyzer instance.  Any active timestamp mask is
        forwarded to the bootstrap (only accepted seconds are resampled).
    N : int, optional
        Number of bootstrap replicates.  Default: 500.
    channel : int or None, optional
        Passed to ``get_timetrace()`` for ``ZBasisPhotonCountingAnalyzer``
        instances to restrict to channel 1 or 2.  Ignored for
        ``PhotonCountingAnalyzer`` instances.  Default: None (both channels).
    random_state : int, np.random.Generator, or None, optional
        Seed or Generator for reproducibility.  Default: None.

    Returns
    -------
    dict :
        time_ns (np.ndarray, shape [B])       : bin centres in nanoseconds,
            identical to the ``time_ns`` key of ``get_timetrace()``.

        For ``PhotonCountingAnalyzer``:
            total    (np.ndarray, shape [B, N])
            marker_1 (np.ndarray, shape [B, N])
            marker_2 (np.ndarray, shape [B, N])

        For ``ZBasisPhotonCountingAnalyzer``:
            total   (np.ndarray, shape [B, N])
            noise   (np.ndarray, shape [B, N])
            g_state (np.ndarray, shape [B, N])
            e_state (np.ndarray, shape [B, N])

        Each column ``result['total'][:, k]`` is the timetrace histogram
        for the k-th bootstrap replicate, so statistics across replicates
        are computed along axis 1 (e.g. ``np.std(result['total'], axis=1)``).

    Notes
    -----
    The ``channel`` parameter is used **only** to decide which detected
    photons go into each histogram bin on each replicate.  It does **not**
    affect which seconds are resampled, so *n_syncs* (the total number of
    sync pulses) is the same for every replicate.
    """
    rng = np.random.default_rng(random_state)

    # -----------------------------------------------------------------------
    # Detect analyzer flavour by duck-typing
    # -----------------------------------------------------------------------
    is_zbasis = hasattr(analyzer, "bin1_offset")

    if is_zbasis:
        marker_keys = ("noise", "g_state", "e_state")  # marker values 1, 2, 3
        marker_values = (1, 2, 3)
    else:
        marker_keys = ("marker_1", "marker_2")          # marker values 1, 2
        marker_values = (1, 2)

    # -----------------------------------------------------------------------
    # Gather per-second row groups, respecting any active timestamp mask
    # -----------------------------------------------------------------------
    second_groups = []
    for file_data in analyzer.parsed_data.values():
        for ds_idx in sorted(file_data["datasets"]):
            arr = file_data["datasets"][ds_idx]
            if len(arr) == 0:
                continue
            seconds = arr[:, _PC_COL_SECONDS].astype(int)
            for s in np.unique(seconds):
                if analyzer._accepted_seconds is not None and \
                        (ds_idx, int(s)) not in analyzer._accepted_seconds:
                    continue
                second_groups.append(arr[seconds == s])

    n_seconds = len(second_groups)

    # -----------------------------------------------------------------------
    # Determine time_ns and num_bins from a single full timetrace call
    # -----------------------------------------------------------------------
    if is_zbasis:
        tt_ref = analyzer.get_timetrace(channel=channel)
    else:
        tt_ref = analyzer.get_timetrace()

    time_ns = tt_ref["time_ns"]
    num_bins = len(time_ns)

    if n_seconds == 0 or num_bins == 0:
        empty = np.zeros((num_bins, N), dtype=int)
        result = {"time_ns": time_ns, "total": empty}
        for key in marker_keys:
            result[key] = empty.copy()
        return result

    # -----------------------------------------------------------------------
    # Bootstrap loop
    # -----------------------------------------------------------------------
    out_total = np.zeros((num_bins, N), dtype=np.int64)
    out_markers = {key: np.zeros((num_bins, N), dtype=np.int64)
                   for key in marker_keys}

    for b in range(N):
        # Resample seconds with replacement
        indices = rng.integers(0, n_seconds, size=n_seconds)
        resampled_rows = np.vstack([second_groups[i] for i in indices])

        bin_idx = resampled_rows[:, _PC_COL_BIN].astype(int)
        markers = resampled_rows[:, _PC_COL_MARKER].astype(int)

        # Restrict to photons with valid bin indices
        valid = bin_idx < num_bins
        bin_idx = bin_idx[valid]
        markers = markers[valid]

        # Apply channel filter if requested (ZBasis only)
        if channel is not None and is_zbasis:
            from .photon_counting_analysis import _COL_CHANNEL
            ch_col = resampled_rows[valid, _COL_CHANNEL].astype(int)
            ch_mask = ch_col == channel
            bin_idx = bin_idx[ch_mask]
            markers = markers[ch_mask]

        np.add.at(out_total[:, b], bin_idx, 1)
        for key, mv in zip(marker_keys, marker_values):
            sel = bin_idx[markers == mv]
            np.add.at(out_markers[key][:, b], sel, 1)

    result = {"time_ns": time_ns, "total": out_total}
    result.update(out_markers)
    return result


def summarize_bootstrap(bootstrap_result: dict, detailed: bool = False) -> dict:
    """
    Compute summary statistics from the output of ``bootstrap_timetrace``.

    Parameters
    ----------
    bootstrap_result : dict
        The dictionary returned by ``bootstrap_timetrace``.
    detailed : bool, optional
        If True, include the full 2-D histogram arrays (shape ``[B, N]``)
        under the key ``'histograms'`` in each per-quantity sub-dict.
        Default: False.

    Returns
    -------
    dict :
        time_ns (np.ndarray, shape [B]) : bin centres in nanoseconds.

        For each quantity (``total``, ``marker_1``/``marker_2`` or
        ``noise``/``g_state``/``e_state``), a sub-dict containing:

            mean (np.ndarray, shape [B]) : mean over bootstrap replicates.
            std  (np.ndarray, shape [B]) : standard deviation over replicates.
            histograms (np.ndarray, shape [B, N]) : full 2-D array,
                only present when ``detailed=True``.

    Examples
    --------
    >>> bt = bootstrap_timetrace(analyzer, N=500)
    >>> s  = summarize_bootstrap(bt)
    >>> s['total']['mean']          # mean timetrace, shape [num_bins]
    >>> s['total']['std']           # uncertainty per bin, shape [num_bins]

    >>> s = summarize_bootstrap(bt, detailed=True)
    >>> s['marker_1']['histograms'] # shape [num_bins, 500]
    """
    time_ns = bootstrap_result["time_ns"]
    keys = [k for k in bootstrap_result if k != "time_ns"]

    out = {"time_ns": time_ns}
    for key in keys:
        arr = bootstrap_result[key]          # shape [B, N]
        sub = {
            "mean": arr.mean(axis=1),
            "std":  arr.std(axis=1),
        }
        if detailed:
            sub["histograms"] = arr
        out[key] = sub

    return out


def decimate(vec, Ndec):
    """
    Decimate a 1-D vector by averaging non-overlapping blocks of length Ndec.
    If the length of vec is not an integer multiple of Ndec, the excess elements
    at the end are discarded.
    Parameters
    ----------
    vec : np.ndarray
        Input 1-D array to be decimated.
    Ndec : int
        Number of elements to average in each block.

    Returns
    -------
    np.ndarray
        Decimated 1-D array.
    """
    return vec[:len(vec) - len(vec) % Ndec].reshape(-1, Ndec).mean(axis=-1)