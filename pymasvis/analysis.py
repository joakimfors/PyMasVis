import logging

import numpy as np
import scipy.signal as signal
from numpy.lib.stride_tricks import as_strided

from .params import ap_coeffs, fir_coeffs, kfilter_coeffs
from .utils import Steps, Timer

log = logging.getLogger(__package__)


def rms(data, axis=0):
    return np.sqrt((data**2).mean(axis))


def db(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c = 20 * np.log10(c)
        if isinstance(c, np.ndarray):
            c[c == -np.inf] = -128.0
        elif isinstance(c, np.float64) and c == -np.inf:
            c = -128.0
        c = np.nan_to_num(c)
    return c


def itu1770(data, fs, gated=False):
    nc = data.shape[0]
    ns = data.shape[1]
    g = np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41])  # FL+FR+FC+LFE+BL+BR
    g = g[0:nc].reshape(nc, 1)
    b, a = kfilter_coeffs(fs)
    data_k = signal.lfilter(b, a, data, 1)
    if gated:
        ns_gate = int(fs * 0.4)
        ns_step = int((1 - 0.75) * ns_gate)
        steps = int((ns - ns_gate) / (ns_step)) + 1
        z = np.zeros((nc, steps), dtype=float)
        for i in range(steps):
            j = i * ns_step
            z[:, i : i + 1] = (data_k[:, j : j + ns_gate] ** 2).mean(1, keepdims=True)
        with np.errstate(divide='ignore'):
            l = -0.691 + 10.0 * np.log10((g * z).sum(0))  # noqa
        gamma_a = -70
        j_a = np.flatnonzero(l > gamma_a)
        gamma_r = (
            -0.691
            + 10.0 * np.log10((g * (np.take(z, j_a, 1).mean(1, keepdims=1))).sum())
            - 10
        )
        j_r = np.flatnonzero(l > gamma_r)
        l_kg = -0.691 + 10.0 * np.log10(
            (g * (np.take(z, j_r, 1).mean(1, keepdims=1))).sum()
        )
        return l_kg
    else:
        z = (data_k**2).mean(1, keepdims=1)
        with np.errstate(divide='ignore'):
            l_k = -0.691 + 10.0 * np.log10((g * z).sum())
        return l_k


def analyze(track, callback=None):
    data = track['data']['float']
    raw_data = track['data']['fixed']
    ns = track['samples']
    fs = track['samplerate']
    nc = track['channels']
    bits = track['bitdepth']
    track['duration']
    name = track['metadata']['filename']
    track['metadata']['extension']
    enc = track['metadata']['encoding']
    bps = track['metadata']['bps']

    log.info("Processing %s", name)
    log.debug(
        (
            "\tsample rate: %d\n\tchannels: %d\n\tframes: %d\n"
            "\tbits: %d\n\tencoding: %s\n\tbitrate: %s"
        ),
        fs,
        nc,
        ns,
        bits,
        enc,
        bps,
    )

    # Peak / RMS
    with Timer('Calculating peak and RMS...', Steps.calc_pr, callback):
        data_rms = rms(data, 1)
        data_total_rms = rms(data.reshape(1, -1), 1)
        data_peak = np.abs(data).max(1)

        # Peak dBFS
        peak_dbfs = db(data_peak, 1.0)

        # RMS dBFS
        rms_dbfs = db(data_rms, 1.0)

        # Crest dB
        crest_db = db(data_peak, data_rms)
        crest_total_db = db(data_peak.max(), data_total_rms)

    # Loudest
    with Timer('Calculating loudest...', Steps.calc_loud, callback):
        window = fs // 50
        peak = data_peak.max()
        c_max, s_max, ns_cur, ns_max = 0, 0, 0, 0
        for c in range(nc):
            # Find the indices where the sample value
            # is 95% of track peak value
            peaks = np.flatnonzero(np.abs(data[c]) > 0.95 * peak)
            if len(peaks) == 0:
                continue
            ns_cur = 0
            it = np.nditer(peaks, flags=['buffered', 'c_index'], op_flags=['readonly'])
            for s in it:
                i = it.iterindex
                # Count the number of samples (indices) within the window
                ns_cur = (peaks[i : i + window] < s + window).sum()
                if ns_cur > ns_max:
                    c_max = c
                    ns_max = ns_cur
                    s_max = s
        w_max = (s_max - fs // 20, s_max + fs // 20)
        if w_max[0] < 0:
            w_max = (0, fs // 10)
        if w_max[1] > ns:
            w_max = (ns - fs // 10, ns)

    # True peaks
    with Timer('Calculating true peaks...', Steps.calc_tp, callback):
        fir = fir_coeffs()
        fir_phases, fir_size = fir.shape
        d_size = data.itemsize
        strides = ns - fir_size
        true_peak = np.copy(data_peak)
        steps = int((ns - 3 * fs) / fs) + 1
        sttp = np.zeros((nc, steps))
        for c in range(nc):
            fir_strides = as_strided(
                data[c], (strides, fir_size, 1), (d_size, d_size, d_size)
            )
            peaks = np.abs(np.dot(fir, fir_strides))
            peak = peaks.max()
            if peak > true_peak[c]:
                true_peak[c] = peak
            peaks_strided = as_strided(
                peaks, (steps, fir_phases * 3 * fs), (fir_phases * fs, d_size)
            )
            sttp[c, :] = peaks_strided.max(1)
        true_peak_dbtp = db(true_peak, 1.0)

    # EBU R.128
    with Timer('Calculating EBU R 128...', Steps.calc_ebur128, callback):
        l_kg = itu1770(data, fs, gated=True)
        steps = int((ns - 3 * fs) / fs) + 1
        stl = np.zeros(steps)
        for i in range(steps):
            j = i * fs
            stl[i] = itu1770(data[:, j : j + 3 * fs], fs, gated=False)
        stl_abs = stl[stl >= -70.0]
        stl_power = (10.0 ** (stl_abs / 10.0)).mean()
        stl_int = 10 * np.log10(stl_power)
        stl_rel = stl_abs[stl_abs >= stl_int - 20.0]
        stl_rel_sort = np.sort(stl_rel)
        n_stl = stl_rel.size - 1
        stl_low = stl_rel_sort[int(round(n_stl * 0.1))]
        stl_high = stl_rel_sort[int(round(n_stl * 0.95))]
        lra = stl_high - stl_low

    # PLR
    with Timer('Calculatin PLR...', Steps.calc_plr, callback):
        plr_lu = true_peak_dbtp.max() - l_kg
        stplr_lu = db(sttp.max(0), 1.0) - stl

    # Spectrum
    with Timer('Calculating spectrum...', Steps.calc_spec, callback):
        frames = ns // fs
        wfunc = np.blackman(fs)
        norm_spec = np.zeros((nc, fs))
        for c in range(nc):
            for i in np.arange(0, frames * fs, fs):
                norm_spec[c] += (
                    np.abs(np.fft.fft(np.multiply(data[c, i : i + fs], wfunc), fs)) / fs
                ) ** 2
            norm_spec[c] = 20 * np.log10(
                (np.sqrt(norm_spec[c] / frames)) / (data_rms[c])
            )

    # Allpass
    with Timer('Calculating allpass...', Steps.calc_ap, callback):
        ap_freqs = np.array([20, 60, 200, 600, 2000, 6000, 20000])
        ap_crest = np.zeros((len(ap_freqs), nc))
        ap_rms = np.zeros((len(ap_freqs), nc))
        ap_peak = np.zeros((len(ap_freqs), nc))
        for i in range(len(ap_freqs)):
            fc = ap_freqs[i]
            b, a = ap_coeffs(fc, fs)
            y = signal.lfilter(b, a, data, 1)
            ap_peak[i] = y.max(1)
            ap_rms[i] = rms(y, 1)
            ap_crest[i] = db(ap_peak[i], ap_rms[i])

    # Histogram
    with Timer('Calculating histogram...', Steps.calc_hist, callback):
        hbits = bits
        if bits > 16:
            hbits = 18
        hist = np.zeros((nc, 2**hbits))
        hist_bins = np.zeros((nc, 2**hbits + 1))
        for c in range(nc):
            hist[c], hist_bins[c] = np.histogram(
                raw_data[c],
                bins=2**hbits,
                range=(-(2.0 ** (bits - 1)), 2.0 ** (bits - 1) - 1),
            )
        hist_bits = np.log2((hist > 0).sum(1))
        if bits > hbits:
            # fake but counting 2**24 bins take way too long to be worth it
            hist_bits *= bits / float(hbits)

    # Peak vs RMS
    with Timer('Calculating peak vs RMS...', Steps.calc_pvsr, callback):
        n_1s = ns // fs
        peak_1s_dbfs = np.zeros((nc, n_1s))
        rms_1s_dbfs = np.zeros((nc, n_1s))
        crest_1s_db = np.zeros((nc, n_1s))
        for c in range(nc):
            for i in range(n_1s):
                a = data[c, i * fs : (i + 1) * fs].max()
                b = rms(data[c, i * fs : (i + 1) * fs])
                peak_1s_dbfs[c][i] = db(a, 1.0)
                rms_1s_dbfs[c][i] = db(b, 1.0)
                crest_1s_db[c][i] = db(a, b)

    # DR
    with Timer('Calculating DR...', Steps.calc_dr, callback):
        dr_blocks = int(ns / (3.0 * fs))
        dr_ns = dr_blocks * 3 * fs
        dr_tail = ns - dr_ns
        dr_data = data[:, :dr_ns].reshape(nc, -1, 3 * fs)
        dr_rms = np.sqrt(2 * ((dr_data**2).mean(2)))
        dr_peak = np.absolute(dr_data).max(2)
        if dr_tail > 0:
            dr_rms = np.append(
                dr_rms, np.sqrt(2 * ((data[:, dr_ns:] ** 2).mean(1, keepdims=True))), 1
            )
            dr_peak = np.append(
                dr_peak, np.absolute(data[:, dr_ns:]).max(1, keepdims=True), 1
            )
        dr_rms.sort()
        dr_peak.sort()
        dr_20 = int(round(dr_rms.shape[1] * 0.2))
        if dr_20 < 1:
            log.warning('WARNING: Too few DR blocks')
            dr_20 = 1
        dr_ch = -20 * np.log10(
            np.sqrt((dr_rms[:, -dr_20:] ** 2).mean(1, keepdims=True)) / dr_peak[:, [-2]]
        )
        dr = int(round(dr_ch.mean()))

    with Timer('Calculating checksum...', Steps.calc_csum, callback):
        checksum = (raw_data.astype('uint32') ** 2).sum()

    #
    return {
        'crest_db': crest_db,
        'crest_total_db': crest_total_db,
        'rms_dbfs': rms_dbfs,
        'peak_dbfs': peak_dbfs,
        'true_peak_dbtp': true_peak_dbtp,
        'c_max': c_max,
        'w_max': w_max,
        's_max': s_max,
        'ns_max': ns_max,
        'norm_spec': norm_spec,
        'frames': frames,
        'n_1s': n_1s,
        'ap_freqs': ap_freqs,
        'ap_crest': ap_crest,
        'hist': hist,
        'hist_bits': hist_bits,
        'rms_1s_dbfs': rms_1s_dbfs,
        'peak_1s_dbfs': peak_1s_dbfs,
        'crest_1s_db': crest_1s_db,
        'checksum': checksum,
        'l_kg': l_kg,
        'stl': stl,
        'lra': lra,
        'dr': dr,
        'plr_lu': plr_lu,
        'stplr_lu': stplr_lu,
    }
