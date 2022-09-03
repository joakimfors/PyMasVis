import io
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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

from . import __version__
from .utils import Steps, Timer

log = logging.getLogger(__package__)

VERSION = __version__
DPI = 72
R128_OFFSET = 23

matplotlib.use('agg')
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
