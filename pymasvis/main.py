# -*- coding: utf-8 -*-
"""
Copyright 2012 Joakim Fors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__version__ = '1.6.0'
__author__ = 'Joakim Fors'

import collections
import io
import json
import locale
import logging
import operator
import os
import re
import subprocess
import sys
import time

import matplotlib
import numpy as np

matplotlib.use('AGG')  # noqa
from os.path import basename
from subprocess import CalledProcessError

import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib import gridspec, rc
from matplotlib.pyplot import (
    plot,
    semilogx,
    semilogy,
    setp,
    subplot,
    text,
    title,
    xlabel,
    xlim,
    xticks,
    ylabel,
    ylim,
    yticks,
)
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter
from numpy.lib.stride_tricks import as_strided
from PIL import Image

VERSION = __version__
DPI = 72
DEBUG = False
R128_OFFSET = 23

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['figure.titlesize'] = 'medium'
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.dashed_pattern'] = [6, 6]
matplotlib.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
matplotlib.rcParams['lines.dotted_pattern'] = [1, 3]
matplotlib.rcParams['lines.scale_dashes'] = False
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

log = logging.getLogger('pymasvis')
lh = logging.StreamHandler(sys.stdout)
lh.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(lh)
log.setLevel(logging.WARNING)
overviews = collections.OrderedDict()


class Timer:
    def __init__(self, description=None, tid=None, callback=None):
        self.description = description
        self.callback = callback
        self.tid = tid

    def __enter__(self):
        if self.callback:
            self.callback('start', self.tid, desc=self.description)
        elif self.description:
            log.info(self.description)
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000
        if self.callback:
            self.callback('stop', self.tid, secs=self.secs)
        else:
            log.debug('%7.1f ms', self.msecs)


class Steps:
    steps = 23
    (
        total,
        calc_pr,
        calc_loud,
        calc_tp,
        calc_ebur128,
        calc_plr,
        calc_spec,
        calc_ap,
        calc_hist,
        calc_pvsr,
        calc_dr,
        calc_csum,
        draw_plot,
        draw_ch,
        draw_loud,
        draw_spec,
        draw_ap,
        draw_hist,
        draw_pvsr,
        draw_stc,
        draw_ebur128,
        draw_overview,
        save,
    ) = range(steps)
    times = [0] * steps
    descs = [''] * steps

    @classmethod
    def callback(cls, event, tid, desc=None, secs=None):
        if event == 'start':
            cls.start(tid, desc)
        elif event == 'stop':
            cls.stop(tid, secs)

    @classmethod
    def start(cls, tid, desc=None):
        if desc:
            cls.descs[tid] = desc
            log.info(desc)

    @classmethod
    def stop(cls, tid, secs):
        log.debug('%7.1f ms', secs * 1000)
        cls.times[tid] = secs

    @classmethod
    def report(cls):
        for desc, t in zip(cls.descs, cls.times):
            log.info("%s took %5.1f %%", desc, 100 * t / cls.times[0])


class Supervisor(object):
    pass


def find_bin(name):
    paths = [d for d in sys.path if 'Contents/Resources' in d]
    paths.extend(os.getenv('PATH').split(os.pathsep))
    for ospath in paths:
        for binext in ['', '.exe']:
            binpath = os.path.join(ospath, name) + binext
            if os.path.isfile(binpath):
                log.debug('Found %s at %s', name, binpath)
                return binpath
    return None


def load_file(infile, inbuffer=None):
    convs = {
        8: {'format': 's8', 'codec': 'pcm_s8', 'dtype': np.dtype('i1')},
        16: {'format': 's16le', 'codec': 'pcm_s16le', 'dtype': np.dtype('<i2')},
        24: {'format': 's32le', 'codec': 'pcm_s32le', 'dtype': np.dtype('<i4')},
        32: {'format': 's32le', 'codec': 'pcm_s32le', 'dtype': np.dtype('<i4')},
    }
    src = 'file'
    name = os.path.splitext(basename(infile))[0]
    ext = os.path.splitext(infile)[1][1:].strip().lower()
    fmt = None
    title = None
    artist = None
    date = None
    album = None
    track = None
    bps = 1411000
    bits = 16
    ffprobe_bin = find_bin('ffprobe')
    if not ffprobe_bin:
        log.warning("ffprobe not found")
        return 1
    ffmpeg_bin = find_bin('ffmpeg')
    if not ffmpeg_bin:
        log.warning("ffmpeg not found")
        return 1
    log.info("Probing file")
    _infile = infile
    if inbuffer:
        _infile = '-'
    try:
        ffprobe = subprocess.Popen(
            [
                ffprobe_bin,
                '-print_format',
                'json',
                '-show_format',
                '-show_streams',
                '-select_streams',
                'a',
                _infile,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        output, error = ffprobe.communicate(inbuffer)
        log.debug(output)
    except CalledProcessError as e:
        log.warning('Could not probe %s', src)
        return e.returncode
    if ffprobe.returncode > 0:
        log.warning("Failed to probe file %s", infile)
        log.debug(error)
        return ffprobe.returncode
    probe = json.loads(output)
    if 'streams' not in probe:
        log.warning("No streams found in %s", infile)
        return 2
    if not probe['streams']:
        log.warning("No audio stream found in %s", infile)
        return 2
    container = probe['format']
    stream = probe['streams'][0]
    if 'tags' in container:
        tags = {k.lower(): v for k, v in container['tags'].items()}
    else:
        tags = {}

    if 'format_name' in container:
        fmts = container['format_name'].split(',')
        if ext in fmts:
            fmt = ext
        else:
            fmt = fmts[0]
    if 'bit_rate' in container:
        bps = int(container['bit_rate'])
    if 'bit_rate' in stream:
        bps = int(stream['bit_rate'])
    if 'duration_ts' in stream:
        ns = int(stream['duration_ts'])
    if 'sample_rate' in stream:
        fs = int(stream['sample_rate'])
    if 'channels' in stream:
        nc = stream['channels']
    if 'bits_per_raw_sample' in stream:
        bits = int(stream['bits_per_raw_sample'])
    if 'bits_per_sample' in stream and int(stream['bits_per_sample']) > 0:
        bits = int(stream['bits_per_sample'])
    if 'duration' in stream:
        sec = float(stream['duration'])
    if 'codec_name' in stream:
        enc = stream['codec_name']
        if 'pcm_' == enc[0:4]:
            enc = fmt
    if 'artist' in tags:
        artist = tags['artist']
    if 'title' in tags:
        title = tags['title']
    if 'album' in tags:
        album = tags['album']
    if 'track' in tags:
        track = int(tags['track'].split('/')[0])
    if 'date' in tags:
        date = tags['date']
    conv = convs[bits]
    log.info("Converting using ffmpeg")
    command = [
        ffmpeg_bin,
        '-y',
        '-i',
        _infile,
        '-vn',
        '-f',
        conv['format'],
        '-acodec',
        conv['codec'],
        '-flags',
        'bitexact',
        '-',
    ]
    try:
        ffmpeg = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        outbuf, error = ffmpeg.communicate(inbuffer)
        log.debug(error)
    except CalledProcessError as e:
        log.warning('Could not convert %s', infile)
        return e.returncode
    raw_data = np.frombuffer(outbuf, dtype=conv['dtype'])
    raw_data = raw_data.reshape((nc, -1), order='F').copy(order='C')
    log.debug(raw_data.shape)
    ns = raw_data[0].shape[0]
    sec = ns / float(fs)
    if bits == 24:
        raw_data /= 2**8
    data = raw_data.astype('float')
    data /= 2 ** (bits - 1)
    if not fmt:
        fmt = ext
    if artist and title:
        name = '%s - %s' % (artist, title)
    output = {
        'data': {'fixed': raw_data, 'float': data},
        'samples': ns,
        'samplerate': fs,
        'channels': nc,
        'bitdepth': bits,
        'duration': sec,
        'format': fmt,
        'metadata': {
            'source': src,
            'filename': basename(infile),
            'extension': ext,
            'encoding': enc,
            'name': name,
            'artist': artist,
            'title': title,
            'album': album,
            'track': track,
            'date': date,
            'bps': bps,
        },
    }
    return output


def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)


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


class MaxNLocatorMod(MaxNLocator):
    def __init__(self, *args, **kwargs):
        super(MaxNLocatorMod, self).__init__(*args, **kwargs)

    def tick_values(self, vmin, vmax):
        ticks = super(MaxNLocatorMod, self).tick_values(vmin, vmax)
        span = vmax - vmin
        if ticks[-1] > vmax - 0.05 * span:
            ticks = ticks[0:-1]
        return ticks


def positions(nc=1):
    w = 606.0
    h = 1060.0
    h_single = 81.976010101
    h_sep = 31.61931818181818181240
    h = round(h + (nc - 2) * (h_single + h_sep))
    left = 45.450
    right = 587.82
    top = 95.40
    bottom = 37.100
    header_y = 8.480
    subheader_y = 26.500
    footer_y = 7.420
    hr = [1] * nc + [1, 2, 2, 1, 1]
    n = len(hr)
    return {
        'w': w,
        'h': h,
        'left': left / w,
        'right': right / w,
        'top': (h - top) / h,
        'bottom': bottom / h,
        'header_y': (h - header_y) / h,
        'subheader_y': (h - subheader_y) / h,
        'footer_y': footer_y / h,
        'hspace': (h_sep * n) / (h - top - bottom - h_sep * (n - 1)),
        'hr': hr,
        'hn': n,
    }


def render(
    track, analysis, header, r128_unit='LUFS', render_overview=False, callback=None
):
    #
    # Plot
    #
    nc = track['channels']
    fs = track['samplerate']
    crest_db = analysis['crest_db']
    crest_total_db = analysis['crest_total_db']
    dr = analysis['dr']
    l_kg = analysis['l_kg']
    lra = analysis['lra']
    plr = analysis['plr_lu']
    checksum = analysis['checksum']
    lufs_to_lu = 23.0
    if r128_unit == 'LUFS':
        r128_offset = 0
    else:
        r128_offset = R128_OFFSET
    c_color = ['b', 'r', 'g', 'y', 'c', 'm']
    c_name = ['left', 'right', 'center', 'LFE', 'surr left', 'surr right']
    nc_max = len(c_color)
    with Timer("Drawing plot...", Steps.draw_plot, callback):
        subtitle_analysis = (
            'Crest: %.2f dB,  DR: %d,  L$_K$: %.1f %s,  ' 'LRA: %.1f LU,  PLR: %.1f LU'
        ) % (crest_total_db, dr, l_kg + r128_offset, r128_unit, lra, plr)
        subtitle_source = (
            'Encoding: %s,  Channels: %d,  Bits: %d,  '
            'Sample rate: %d Hz,  Bitrate: %s kbps,  '
            'Source: %s'
        ) % (
            track['metadata']['encoding'],
            track['channels'],
            track['bitdepth'],
            fs,
            int(round(track['metadata']['bps'] / 1000.0)),
            track['metadata']['source'],
        )
        subtitle_meta = []
        if track['metadata']['album']:
            subtitle_meta.append('Album: %.*s' % (50, track['metadata']['album']))
        if track['metadata']['track']:
            subtitle_meta.append('Track: %s' % track['metadata']['track'])
        if track['metadata']['date']:
            subtitle_meta.append('Date: %s' % track['metadata']['date'])
        subtitle_meta = ',  '.join(subtitle_meta)
        subtitle = '\n'.join([subtitle_analysis, subtitle_source, subtitle_meta])
        pos = positions(nc)
        fig_d = plt.figure(
            'detailed',
            figsize=(pos['w'] / DPI, pos['h'] / DPI),
            facecolor='white',
            dpi=DPI,
        )
        fig_d.suptitle(header, fontsize='medium', y=pos['header_y'])
        fig_d.text(
            0.5,
            pos['subheader_y'],
            subtitle,
            fontsize='small',
            horizontalalignment='center',
            verticalalignment='top',
            linespacing=1.6,
        )
        fig_d.text(
            pos['left'],
            pos['footer_y'],
            ('Checksum (energy): %d' % checksum),
            fontsize='small',
            va='bottom',
            ha='left',
        )
        fig_d.text(
            pos['right'],
            pos['footer_y'],
            ('PyMasVis %s' % (VERSION)),
            fontsize='small',
            va='bottom',
            ha='right',
        )
        rc('lines', linewidth=0.5, antialiased=True)
        gs = gridspec.GridSpec(
            pos['hn'],
            2,
            width_ratios=[2, 1],
            height_ratios=pos['hr'],
            hspace=pos['hspace'],
            wspace=0.2,
            left=pos['left'],
            right=pos['right'],
            bottom=pos['bottom'],
            top=pos['top'],
        )

    # Channels
    data = track['data']['float']
    sec = track['duration']
    rms_dbfs = analysis['rms_dbfs']
    peak_dbfs = analysis['peak_dbfs']
    true_peak_dbtp = analysis['true_peak_dbtp']
    c_max = analysis['c_max']
    w_max = analysis['w_max']
    with Timer("Drawing channels...", Steps.draw_ch, callback):
        ax_ch = []
        c = 0
        while c < nc and c < nc_max:
            if c == 0:
                ax_ch.append(subplot(gs[c, :]))
            else:
                ax_ch.append(subplot(gs[c, :], sharex=ax_ch[0]))
            new_data, new_ns, new_range = pixelize(
                data[c], ax_ch[c], which='both', oversample=2
            )
            new_fs = new_ns / sec
            new_range = np.arange(0.0, new_ns, 1) / new_fs
            plot(new_range, new_data, color=c_color[c], linestyle='-')
            xlim(0, round(sec))
            ylim(-1.0, 1.0)
            title(
                (
                    u"%s: Crest=%0.2f dB, RMS=%0.2f dBFS, Peak=%0.2f dBFS, "
                    u"True Peak≈%0.2f dBTP"
                )
                % (
                    c_name[c].capitalize(),
                    crest_db[c],
                    rms_dbfs[c],
                    peak_dbfs[c],
                    true_peak_dbtp[c],
                ),
                fontsize='small',
                loc='left',
            )
            yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
            if c_max == c:
                mark_span(ax_ch[c], (w_max[0] / float(fs), w_max[1] / float(fs)))
            if c + 1 == nc or c + 1 == nc_max:
                ax_ch[c].xaxis.set_major_locator(MaxNLocatorMod(prune='both'))
                ax_ch[c].xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                xlabel('s', fontsize='small')
            else:
                setp(ax_ch[c].get_xticklabels(), visible=False)
            axis_defaults(ax_ch[c])
            c += 1
    spi = c - 1

    # Loudest
    s_max = analysis['s_max']
    ns_max = analysis['ns_max']
    with Timer("Drawing loudest...", Steps.draw_loud, callback):
        ax_max = subplot(gs[spi + 1, :])
        plot(
            np.arange(*w_max) / float(fs),
            data[c_max][np.arange(*w_max)],
            c_color[c_max],
        )
        ylim(-1.0, 1.0)
        xlim(w_max[0] / float(fs), w_max[1] / float(fs))
        title(
            ("Loudest part (%s ch, %d samples > 95%% " "during 20 ms at %0.2f s)")
            % (c_name[c_max], ns_max, s_max / float(fs)),
            fontsize='small',
            loc='left',
        )
        yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
        ax_max.xaxis.set_major_locator(MaxNLocatorMod(nbins=5, prune='both'))
        ax_max.xaxis.set_major_formatter(FormatStrFormatter("%0.2f"))
        xlabel('s', fontsize='small')
        axis_defaults(ax_max)

    # Spectrum
    norm_spec = analysis['norm_spec']
    frames = analysis['frames']
    with Timer("Drawing spectrum...", Steps.draw_spec, callback):
        ax_norm = subplot(gs[spi + 2, 0])
        semilogx(
            [0.02, 0.06],
            [-80, -90],
            'k-',
            [0.02, 0.2],
            [-70, -90],
            'k-',
            [0.02, 0.6],
            [-60, -90],
            'k-',
            [0.02, 2.0],
            [-50, -90],
            'k-',
            [0.02, 6.0],
            [-40, -90],
            'k-',
            [0.02, 20.0],
            [-30, -90],
            'k-',
            [0.02, 20.0],
            [-20, -80],
            'k-',
            [0.02, 20.0],
            [-10, -70],
            'k-',
            [0.06, 20.0],
            [-10, -60],
            'k-',
            [0.20, 20.0],
            [-10, -50],
            'k-',
            [0.60, 20.0],
            [-10, -40],
            'k-',
            [2.00, 20.0],
            [-10, -30],
            'k-',
            [6.00, 20.0],
            [-10, -20],
            'k-',
            base=10,
        )
        for c in range(nc):
            new_spec, new_n, new_r = pixelize(
                norm_spec[c],
                ax_norm,
                which='max',
                oversample=1,
                method='log10',
                span=(20, 20000),
            )
            semilogx(new_r / 1000.0, new_spec, color=c_color[c], linestyle='-', base=10)
        ylim(-90, -10)
        xlim(0.02, 20)
        ax_norm.yaxis.grid(True, which='major', linestyle=':', color='k', linewidth=0.5)
        ax_norm.xaxis.grid(True, which='both', linestyle='-', color='k', linewidth=0.5)
        ylabel('dB', fontsize='small', verticalalignment='top', rotation=0)
        xlabel('kHz', fontsize='small', horizontalalignment='right')
        title(
            "Normalized average spectrum, %d frames" % (frames),
            fontsize='small',
            loc='left',
        )
        ax_norm.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7], minor=False)
        ax_norm.set_xticks(
            [
                0.03,
                0.04,
                0.06,
                0.07,
                0.08,
                0.09,
                0.3,
                0.4,
                0.6,
                0.7,
                0.8,
                0.9,
                6,
                8,
                9,
                10,
            ],
            minor=True,
        )
        ax_norm.set_xticklabels([0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7], minor=False)
        ax_norm.set_xticklabels([], minor=True)
        yticks(np.arange(-90, 0, 10), ('', -80, -70, -60, -50, -40, -30, '', ''))
        axis_defaults(ax_norm)

    # Allpass
    ap_freqs = analysis['ap_freqs']
    ap_crest = analysis['ap_crest']
    with Timer("Drawing allpass...", Steps.draw_ap, callback):
        ax_ap = subplot(gs[spi + 2, 1])
        for c in range(nc):
            semilogx(
                ap_freqs / 1000.0,
                crest_db[c] * np.ones(len(ap_freqs)),
                color=c_color[c],
                linestyle='--',
                base=10,
            )
            semilogx(
                ap_freqs / 1000.0,
                ap_crest.swapaxes(0, 1)[c],
                color=c_color[c],
                linestyle='-',
                base=10,
            )
        ylim(0, 30)
        xlim(0.02, 20)
        title("Allpassed crest factor", fontsize='small', loc='left')
        yticks(np.arange(0, 30, 5), ('', 5, 10, 15, 20, ''))
        xticks([0.1, 1, 2], (0.1, 1, 2))
        xlabel('kHz', fontsize='small')
        ylabel('dB', fontsize='small', rotation=0)
        axis_defaults(ax_ap)

    # Histogram
    hist = analysis['hist']
    hist_bits = analysis['hist_bits']
    hist_title_bits = []
    with Timer("Drawing histogram...", Steps.draw_hist, callback):
        ax_hist = subplot(gs[spi + 3, 0])
        for c in range(nc):
            new_hist, new_n, new_range = pixelize(
                hist[c], ax_hist, which='max', oversample=2
            )
            new_hist[(new_hist == 1.0)] = 1.3
            new_hist[(new_hist < 1.0)] = 1.0
            semilogy(
                np.arange(new_n) * 2.0 / new_n - 1.0,
                new_hist,
                color=c_color[c],
                linestyle='-',
                base=10,
                drawstyle='steps',
            )
            hist_title_bits.append('%0.1f' % hist_bits[c])
        xlim(-1.1, 1.1)
        ylim(1, 50000)
        xticks(
            np.arange(-1.0, 1.2, 0.2),
            (-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1),
        )
        yticks([10, 100, 1000], (10, 100, 1000))
        hist_title = 'Histogram, "bits": %s' % '/'.join(hist_title_bits)
        title(hist_title, fontsize='small', loc='left')
        ylabel('n', fontsize='small', rotation=0)
        axis_defaults(ax_hist)

    # Peak vs RMS
    rms_1s_dbfs = analysis['rms_1s_dbfs']
    peak_1s_dbfs = analysis['peak_1s_dbfs']
    with Timer("Drawing peak vs RMS...", Steps.draw_pvsr, callback):
        ax_pr = subplot(gs[spi + 3, 1])
        plot(
            [-50, 0],
            [-50, 0],
            'k-',
            [-50, -10],
            [-40, 0],
            'k-',
            [-50, -20],
            [-30, 0],
            'k-',
            [-50, -30],
            [-20, 0],
            'k-',
            [-50, -40],
            [-10, 0],
            'k-',
        )
        text_style = {
            'fontsize': 'x-small',
            'rotation': 45,
            'va': 'bottom',
            'ha': 'left',
        }
        text(-48, -45, '0 dB', **text_style)
        text(-48, -35, '10', **text_style)
        text(-48, -25, '20', **text_style)
        text(-48, -15, '30', **text_style)
        text(-48, -5, '40', **text_style)
        for c in range(nc):
            plot(
                rms_1s_dbfs[c],
                peak_1s_dbfs[c],
                linestyle='',
                marker='o',
                markerfacecolor='w',
                markeredgecolor=c_color[c],
                markeredgewidth=0.7,
            )
        xlim(-50, 0)
        ylim(-50, 0)
        title("Peak vs RMS level", fontsize='small', loc='left')
        xlabel('dBFS', fontsize='small')
        ylabel('dBFS', fontsize='small', rotation=0)
        xticks([-50, -40, -30, -20, -10, 0], ('', -40, -30, -20, '', ''))
        yticks([-50, -40, -30, -20, -10, 0], ('', -40, -30, -20, -10, ''))
        axis_defaults(ax_pr)

    # Shortterm crest
    crest_1s_db = analysis['crest_1s_db']
    n_1s = analysis['n_1s']
    with Timer("Drawing short term crest...", Steps.draw_stc, callback):
        ax_1s = subplot(gs[spi + 4, :])
        for c in range(nc):
            plot(
                np.arange(n_1s) + 0.5,
                crest_1s_db[c],
                linestyle='',
                marker='o',
                markerfacecolor='w',
                markeredgecolor=c_color[c],
                markeredgewidth=0.7,
            )
        ylim(0, 30)
        xlim(0, n_1s)
        yticks([10, 20], (10, ''))
        ax_1s.yaxis.grid(True, which='major', linestyle=':', color='k', linewidth=0.5)
        title("Short term (1 s) crest factor", fontsize='small', loc='left')
        xlabel('s', fontsize='small')
        ylabel('dB', fontsize='small', rotation=0)
        ax_1s.xaxis.set_major_locator(MaxNLocatorMod(prune='both'))
        ax_1s.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axis_defaults(ax_1s)

    # EBU R 128
    stl = analysis['stl']
    stplr = analysis['stplr_lu']
    with Timer("Drawing EBU R 128 loudness...", Steps.draw_ebur128, callback):
        ax_ebur128 = subplot(gs[spi + 5, :])
        plot(
            np.arange(stl.size) + 1.5,
            stl + r128_offset,
            'ko',
            markerfacecolor='w',
            markeredgecolor='k',
            markeredgewidth=0.7,
        )
        ylim(-41 + r128_offset, -5 + r128_offset)
        xlim(0, n_1s)
        yticks(
            [-33 + r128_offset, -23 + r128_offset, -13 + r128_offset],
            (-33 + r128_offset, -23 + r128_offset, ''),
        )
        title("EBU R 128 Short term loudness", fontsize='small', loc='left')
        title("Short term PLR", fontsize='small', loc='right', color='grey')
        xlabel('s', fontsize='small')
        ylabel('%s' % r128_unit, fontsize='small', rotation=0)
        ax_ebur128.yaxis.grid(
            True, which='major', linestyle=':', color='k', linewidth=0.5
        )
        ax_ebur128_stplr = ax_ebur128.twinx()
        plot(
            np.arange(stplr.size) + 1.5,
            stplr,
            'o',
            markerfacecolor='w',
            markeredgecolor='grey',
            markeredgewidth=0.7,
        )
        xlim(0, n_1s)
        ylim(0, 36)
        yticks([0, 18], (0, 18))
        for tl in ax_ebur128_stplr.get_yticklabels():
            tl.set_color('grey')
        ax_ebur128.xaxis.set_major_locator(MaxNLocatorMod(prune='both'))
        ax_ebur128.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axis_defaults(ax_ebur128)
        axis_defaults(ax_ebur128_stplr)
        ax_ebur128_stplr.tick_params(
            axis='y', which='major', labelsize='xx-small', length=0
        )

    # Overview
    with Timer("Drawing overview...", Steps.draw_overview, callback):
        if render_overview:
            w_o = 606.0
            h_o = 64.0
            fig_o = plt.figure(
                'overview', figsize=(w_o / DPI, h_o / DPI), facecolor='white', dpi=DPI
            )
            ax_o = fig_o.add_subplot(111)
            ax_o.set_position([12 / w_o, 8 / h_o, 464 / w_o, 40 / h_o])
            ax_o.set_xticks([])
            ax_o.set_yticks([])
            header_o = "%s  [%s, %d ch, %d bits, %d Hz, %d kbps]" % (
                header,
                track['metadata']['encoding'],
                track['channels'],
                track['bitdepth'],
                fs,
                int(round(track['metadata']['bps'] / 1000.0)),
            )
            ax_o.set_title(header_o, fontsize='small', loc='left')
            w_buf = round(ax_o.bbox.bounds[2])
            h_buf = round(ax_o.bbox.bounds[3])
            info_o = (
                u"Crest = %0.1f dB\nPeak = %0.1f dBFS\nDR = %d,  " u"L$_k$ = %.1f LU"
            ) % (crest_total_db, peak_dbfs.max(), dr, l_kg + lufs_to_lu)
            fig_o.text(
                482 / w_o,
                28 / h_o,
                info_o,
                fontsize='small',
                verticalalignment='center',
                snap=False,
            )
            fig_buf = plt.figure(
                'buffer', figsize=(w_buf / DPI, h_buf / DPI), facecolor='white', dpi=DPI
            )
            w, h = fig_buf.canvas.get_width_height()
            fig_buf.patch.set_visible(False)
            ax_buf = plt.gca()
            img_buf = np.zeros((h, w, 4), np.uint8)
            img_buf[:, :, 0:3] = 255
            for i, ch in enumerate(data):
                ax_buf.clear()
                ax_buf.axis('off')
                ax_buf.set_position([0, 0, 1, 1])
                ax_buf.set_ylim(-1, 1)
                ax_buf.set_xticks([])
                ax_buf.set_yticks([])
                new_ch, new_n, new_r = pixelize(ch, ax_buf, which='both', oversample=2)
                ax_buf.plot(range(len(new_ch)), new_ch, color=c_color[i])
                ax_buf.set_xlim(0, len(new_ch))
                fig_buf.canvas.draw()
                img = np.frombuffer(fig_buf.canvas.buffer_rgba(), np.uint8).reshape(
                    h, w, -1
                )
                img_buf[:, :, 0:3] = img[:, :, 0:3] * (img_buf[:, :, 0:3] / 255.0)
                img_buf[:, :, -1] = np.maximum(img[:, :, -1], img_buf[:, :, -1])
            img_buf[:, :, 0:3] = (img_buf[:, :, 3:4] / 255.0) * img_buf[:, :, 0:3] + (
                255 - img_buf[:, :, 3:4]
            )
            img_buf[:, :, -1] = 255
            plt.figure('overview')
            plt.imshow(img_buf, aspect='auto', interpolation='none')
            overview = io.BytesIO()
            plt.savefig(overview, format='png', dpi=DPI, transparent=False)
            plt.close(fig_o)
        else:
            overview = None

    # Save
    with Timer("Saving...", Steps.save, callback):
        plt.figure('detailed')
        detailed = io.BytesIO()
        plt.savefig(detailed, format='png', dpi=DPI, transparent=False)
        plt.close(fig_d)

    return detailed, overview


def xpixels(ax):
    return np.round(ax.bbox.bounds[2])


def pixelize(x, ax, method='linear', which='both', oversample=1, span=None):
    if not span:
        span = (0, len(x))
        if method == 'log10':
            span = (1, len(x) + 1)
    pixels = xpixels(ax)
    minmax = 1
    if which == 'both':
        minmax = 2
    nw = int(pixels * oversample)
    w = (span[1] - span[0]) / (pixels * oversample)
    n = nw * minmax
    y = np.zeros(n)
    r = np.zeros(n)
    for i in range(nw):
        if method == 'linear':
            j = int(np.round(i * w + span[0]))
            k = int(np.round(j + w + span[0]))
        elif method == 'log10':
            a = np.log10(span[1]) - np.log10(span[0])
            b = np.log10(span[0])
            j = int(np.round(10 ** (i / float(nw) * a + b)) - 1)
            k = int(np.round(10 ** ((i + 1) / float(nw) * a + b)))
        if i == nw - 1 and k != span[1]:
            log.debug('pixelize tweak k')
            k = span[1]
        r[i] = k
        if which == 'max':
            y[i] = x[j:k].max()
        elif which == 'min':
            y[i] = x[j:k].min()
        elif which == 'both':
            y[i * minmax] = x[j:k].max()
            y[i * minmax + 1] = x[j:k].min()
    return (y, n, r)


def mark_span(ax, span):
    ax.axvspan(
        *span,
        edgecolor='0.2',
        facecolor='0.98',
        fill=False,
        linestyle='dotted',
        linewidth=0.8,
        zorder=10,
    )


def axis_defaults(ax):
    ax.tick_params(direction='in', top='off', right='off')
    ax.tick_params(axis='both', which='major', labelsize='small')
    ax.tick_params(axis='both', which='minor', labelsize='small')
    xpad = ax.xaxis.labelpad
    ypad = ax.yaxis.labelpad
    xpos = ax.transAxes.transform((1.0, 0.0))
    xpos[1] -= xpad
    xpos = ax.transAxes.inverted().transform(xpos)
    ypos = ax.transAxes.transform((0.0, 1.0))
    ypos[0] -= ypad
    ypos = ax.transAxes.inverted().transform(ypos)
    ax.xaxis.set_label_coords(*xpos)
    ax.yaxis.set_label_coords(*ypos)
    ax.xaxis.get_label().set_ha('right')
    ax.xaxis.get_label().set_va('top')
    ax.yaxis.get_label().set_ha('right')
    ax.yaxis.get_label().set_va('top')


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


def ap_coeffs(fc, fs):
    '''
    Discrete first order allpass
    https://ccrma.stanford.edu/realsimple/DelayVar/Phasing_First_Order_Allpass_Filters.html
    http://users.abo.fi/htoivone/courses/sbappl/asp_chapter1.pdf

    T = 1.0/fs
    w_b = 2*np.pi*fc
    p_d = (1 - np.tan(w_b*T/2)) / (1 + np.tan(w_b*T/2))
    b = [p_d, -1.0]
    a = [1.0, -p_d]
    '''
    if fc > fs / 2.0001:
        fc = fs / 2.0001
    rho_b = np.tan(np.pi * fc / fs)
    p_d = (1 - rho_b) / (1 + rho_b)
    b = [p_d, -1.0]
    a = [1.0, -p_d]
    return (b, a)


def kfilter_coeffs(fs):
    # Pre filter
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196

    K = np.tan(np.pi * f0 / fs)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh**0.4996667741545416

    a0 = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0
    b1 = 2.0 * (K * K - Vh) / a0
    b2 = (Vh - Vb * K / Q + K * K) / a0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 - K / Q + K * K) / a0

    b_pre = [b0, b1, b2]
    a_pre = [1.0, a1, a2]

    # Highpass filter
    f0 = 38.13547087602444
    Q = 0.5003270373238773
    K = np.tan(np.pi * f0 / fs)

    a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
    a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)

    b_hp = [1.0, -2.0, 1.0]
    a_hp = [1.0, a1, a2]

    b = signal.convolve(b_pre, b_hp)
    a = signal.convolve(a_pre, a_hp)
    return (b, a)


def fir_coeffs():
    return np.array(
        [
            [
                -0.001780280772489,
                0.003253283030257,
                -0.005447293390376,
                0.008414568116553,
                -0.012363296099675,
                0.017436805871070,
                -0.024020143876810,
                0.032746828420101,
                -0.045326602900760,
                0.066760686868173,
                -0.120643370377371,
                0.989429605248410,
                0.122160009958442,
                -0.046376232812786,
                0.022831393004364,
                -0.011580897261667,
                0.005358105753167,
                -0.001834671998839,
                -0.000103681038815,
                0.001002216283171,
                -0.001293611238062,
                0.001184842429930,
                -0.000908719377960,
                0.002061304229100,
            ],
            [
                -0.001473218555432,
                0.002925336766866,
                -0.005558126468508,
                0.009521159741206,
                -0.015296028027209,
                0.023398977482278,
                -0.034752051245281,
                0.050880967772373,
                -0.075227488678419,
                0.116949442543490,
                -0.212471239510148,
                0.788420616540440,
                0.460788819545818,
                -0.166082211358253,
                0.092555759769552,
                -0.057854829231334,
                0.037380809681132,
                -0.024098441541823,
                0.015115653825711,
                -0.009060645712669,
                0.005033299068467,
                -0.002511544062471,
                0.001030723665756,
                -0.000694079453823,
            ],
            [
                -0.000694079453823,
                0.001030723665756,
                -0.002511544062471,
                0.005033299068467,
                -0.009060645712669,
                0.015115653825711,
                -0.024098441541823,
                0.037380809681132,
                -0.057854829231334,
                0.092555759769552,
                -0.166082211358253,
                0.460788819545818,
                0.788420616540440,
                -0.212471239510148,
                0.116949442543490,
                -0.075227488678419,
                0.050880967772373,
                -0.034752051245281,
                0.023398977482278,
                -0.015296028027209,
                0.009521159741206,
                -0.005558126468508,
                0.002925336766866,
                -0.001473218555432,
            ],
            [
                0.002061304229100,
                -0.000908719377960,
                0.001184842429930,
                -0.001293611238062,
                0.001002216283171,
                -0.000103681038815,
                -0.001834671998839,
                0.005358105753167,
                -0.011580897261667,
                0.022831393004364,
                -0.046376232812786,
                0.122160009958442,
                0.989429605248410,
                -0.120643370377371,
                0.066760686868173,
                -0.045326602900760,
                0.032746828420101,
                -0.024020143876810,
                0.017436805871070,
                -0.012363296099675,
                0.008414568116553,
                -0.005447293390376,
                0.003253283030257,
                -0.001780280772489,
            ],
        ]
    )


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


def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]


def get_data_base(arr):
    """For a given Numpy array, finds the
    base array that "owns" the actual data."""
    base = arr
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base


def arrays_share_data(x, y):
    return get_data_base(x) is get_data_base(y)


def file_formats():
    foo = re.compile(r'\s+DE?\s+(\S+)\s+\S+')
    formats = []
    try:
        result = subprocess.check_output(
            ['ffprobe', '-v', 'quiet', '-formats'], stderr=subprocess.STDOUT
        )
    except CalledProcessError as e:
        log.debug(e)
        return formats
    for line in result.split('\n')[4:]:
        bar = foo.match(line)
        if bar:
            formats += bar.group(1).split(',')
    for foo in ['mjpeg', 'gif', 'vobsub']:
        if foo in formats:
            formats.remove(foo)
    return formats


def run(
    infile,
    outfile=None,
    overviewfile=None,
    fmt='png',
    destdir='',
    update=True,
    header=None,
    r128_unit='LUFS',
):
    loader = None
    loader_args = []
    destdir = os.path.join(os.path.dirname(infile), destdir)
    if destdir and not os.path.isdir(destdir):
        log.debug("Creating destdir %s", destdir)
        os.mkdir(destdir)
    if os.path.isfile(infile):
        log.debug("Selecting file loader")
        loader = load_file
        loader_args = [infile]
        if not outfile:
            filename = os.path.basename(infile)
            filename = "%s-pymasvis.%s" % (filename, fmt)
            outfile = os.path.join(destdir, filename)
        if os.path.isfile(outfile):
            if update == 'no':
                log.warning("Destination file %s already exists", outfile)
                return
            elif update == 'outdated' and os.path.getmtime(outfile) > os.path.getmtime(
                infile
            ):
                log.warning(
                    "Destination file %s exist and is newer than %s", outfile, infile
                )
                return
        if not os.path.splitext(outfile)[1][1:] in ['png', 'jpg']:
            log.warning("Only png and jpg supported as output format")
            return
    else:
        log.warning("Unable to open input %s", infile)
        return
    track = loader(*loader_args)
    if type(track) is int:
        return
    if not header:
        header = "%s" % (track['metadata']['name'])
    with Timer('Running...', Steps.total, Steps.callback):
        with Timer('Analyzing...'):
            analysis = analyze(track, callback=Steps.callback)
        with Timer('Rendering...'):
            render_overview = False
            if overviewfile:
                render_overview = True
            detailed, overview = render(
                track,
                analysis,
                header,
                r128_unit=r128_unit,
                render_overview=render_overview,
                callback=Steps.callback,
            )
            img = Image.open(detailed)
            log.info("Writing %s", outfile)
            if fmt == 'png':
                img = img.convert(mode='P', palette='ADAPTIVE', colors=256)
                img.save(outfile, 'PNG', optimize=True)
            elif fmt == 'jpg':
                img.save(outfile, 'JPEG', quality=80, optimize=True)
            if overview:
                img_o = Image.open(overview)
                if overviewfile:
                    overviewindex = os.path.join(destdir, overviewfile)
                    if overviewindex not in overviews:
                        overviews[overviewindex] = []
                    overviews[overviewindex].append(img_o)
    Steps.report()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze audio file or Spotify URI.")
    parser.add_argument('--version', action='version', version="PyMasVis " + VERSION)
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose messages")
    parser.add_argument('-d', '--debug', action='store_true', help="debug info")
    parser.add_argument('-u', '--username', metavar='username', help="Spotify username")
    parser.add_argument('-p', '--password', metavar='password', help="Spotify password")
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help="recurse directory tree if input is a directory",
    )
    parser.add_argument(
        '--destdir',
        metavar='destdir',
        type=str,
        default='',
        help="destination directory to store analysis in",
    )
    parser.add_argument(
        '--update',
        default='yes',
        type=str,
        choices=['yes', 'no', 'outdated'],
        help="choose wheter to update a result image or not, default: yes",
    )
    parser.add_argument(
        '--format',
        default='png',
        type=str,
        choices=['png', 'jpg'],
        help="selects output format, default: png",
    )
    parser.add_argument(
        '--overview',
        action='store_const',
        const='overview-pymasvis',
        help="generate overview",
    )
    parser.add_argument(
        '--overview-mode',
        default='dir',
        type=str,
        choices=['dir', 'flat'],
        help=(
            "generate an overview file per directory or one file for "
            "all inputs, default: dir"
        ),
    )
    parser.add_argument(
        '--lu',
        dest='r128_unit',
        default='LUFS',
        action='store_const',
        const='LU',
        help="Use LU instead of LUFS when displaying R128 values",
    )
    parser.add_argument(
        'inputs',
        metavar='input',
        type=str,
        nargs='+',
        help=(
            'a file, directory or Spotify URI (track, album or playlist) ' 'to analyze'
        ),
    )
    args = parser.parse_args()
    if args.verbose:
        log.setLevel(logging.INFO)
    if args.debug:
        DEBUG = True
        log.setLevel(logging.DEBUG)
    if args.overview and args.update != 'yes':
        log.error("Update must be set to 'yes' to enable overviews")
        exit(1)
    if args.overview:
        args.overview += '.%s' % args.format
    formats = file_formats()
    infiles = []
    for f in args.inputs:
        if os.path.isfile(f):
            infiles.append(f)
            continue
        if os.path.isdir(f):
            for root, dirs, files in os.walk(f):
                for name in files:
                    if os.path.splitext(name)[1][1:] in formats:
                        infiles.append(os.path.join(root, name))
                if not args.recursive:
                    del dirs[:]
    if len(infiles) == 0:
        log.warning("No valid files for analysis found: " + " ".join(args.inputs))
    for fsenc in [sys.getfilesystemencoding(), locale.getdefaultlocale()[1]]:
        if fsenc:
            encoding = fsenc
            break
    for infile in infiles:
        outfile = None
        header = None
        log.warning(infile)
        run(
            infile,
            outfile,
            args.overview,
            args.format,
            args.destdir,
            args.update,
            header,
            args.r128_unit,
        )
    if args.overview:
        if args.overview_mode == 'flat':
            if args.destdir:
                if not os.path.isdir(args.destdir):
                    log.debug("Creating destdir %s", args.destdir)
                    os.mkdir(args.destdir)
                args.overview = os.path.join(args.destdir, args.overview)
            overviews = {args.overview: reduce(operator.add, overviews.itervalues())}
        for overviewfile, images in overviews.iteritems():
            w, h = images[0].size
            n = len(images)
            out = Image.new('RGBA', (w, h * n))
            for i, image in enumerate(images):
                out.paste(image, (0, h * i))
            log.info("Writing overview %s", overviewfile)
            if args.format == 'png':
                out = out.convert(mode='P', palette='ADAPTIVE', colors=256)
                out.save(overviewfile, 'PNG', optimize=True)
            elif args.format == 'jpg':
                out.save(overviewfile, 'JPEG', quality=80, optimize=True)
