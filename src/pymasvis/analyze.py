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

import os
import sys
import subprocess
import math
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.image as mpimg
import scipy.signal as signal

from os.path import basename
from scipy.io import wavfile
from matplotlib import rc, gridspec
from matplotlib.pyplot import plot, axis, subplot, subplots, figure, ylim, xlim, xlabel, ylabel, yticks, xticks, title, semilogx, semilogy, loglog, hold, setp, hlines, text, tight_layout, axvspan
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter, FormatStrFormatter

VERSION="0.3.2"

def analyze(infile, outfile=None, name=None):
	if not outfile:
		outfile = "%s-pymasvis.png" % infile
	if not name:
		name = basename(infile)
	ext = os.path.splitext(infile)[1][1:].strip().lower()
	tmpfile = None
	if ext != "wav":
		ffmpeg_bin = None
		paths = [d for d in sys.path if 'Contents/Resources' in d]
		paths.extend(os.getenv('PATH').split(os.pathsep))
		for ospath in paths:
			for binext in ['', '.exe']:
				binpath = os.path.join(ospath, 'ffmpeg') + binext
				if os.path.isfile(binpath):
					print 'Found ffmpeg', binpath
					ffmpeg_bin = binpath
					found = True
					break
			if ffmpeg_bin: break
		if not ffmpeg_bin:
			print "Could not find ffmpeg"
			return 1
		print "Converting using ffmpeg"
		tmpfile = "%s.%s" % (os.tempnam(), 'wav')
		retval = subprocess.call([ffmpeg_bin, '-i', infile, tmpfile])
		if retval == 0:
			infile = tmpfile
			ext = 'wav'
		else:
			print 'Could not convert %s' % infile
			return retval

	fs, raw_data = wavfile.read(infile)
	nc = raw_data.shape[1]
	enc = str(raw_data.dtype)
	nf = raw_data.shape[0]
	sec = nf/fs
	bits = 16
	for b in [8, 16, 24, 32, 64]:
		if enc.find(str(b)) > -1:
			bits = b
			break

	print "Processing %s" % name
	print "\tfs: %d, ch: %d, enc: %s, frames: %d, bits: %d" % (fs, nc, enc, nf, bits)

	raw_data = raw_data.swapaxes(0,1)
	data = raw_data.astype('float')
	data /= 2**(bits-1)
	if tmpfile and os.path.isfile(tmpfile):
		os.remove(tmpfile)

	# Peak / RMS
	print 'Calculating peak and RMS...'
	data_rms = rms(data, 1) # [np.sqrt(np.mean(data[0]**2)), np.sqrt(np.mean(data[1]**2))]
	data_peak = np.abs(data).max(1)

	# Peak dBFS
	peak_dbfs = db(data_peak, 1.0) #[ 20*np.log10(np.abs(data[0]).max()), 20*np.log10(np.abs(data[1]).max()) ]

	# RMS dBFS
	rms_dbfs = db(data_rms, 1.0) # [ 20*np.log10(rms[0]), 20*np.log10(rms[1]) ]

	# Crest dB
	crest_db = db(data_peak, data_rms) # [ 20*np.log10(np.abs(data[0]).max() / rms[0]), 20*np.log10(np.abs(data[1]).max() / rms[1]) ]

	# Loudest
	print 'Calculating loudest...'
	window = int(fs / 50)
	peak = data_peak.max()
	sn_max = 0
	pos_max = 0
	c_max = 0
	sn_cur = 0
	c_max, f_max, nf_cur, nf_max = 0, 0, 0, 0
	for c in range(nc):
		peaks = np.flatnonzero(np.abs(data[c]) > 0.95*peak)
		if len(peaks) == 0:
			continue
		nf_cur = 0
		it = np.nditer(peaks, flags=['buffered','c_index'], op_flags=['readonly'])
		for e in it:
			i = it.iterindex
			#nf_cur = np.count_nonzero(peaks[i:] < e + window)
			nf_cur = (peaks[i:i+window] < e + window).sum()
			if nf_cur > nf_max:
				c_max = c
				nf_max = nf_cur
				f_max = e
	w_max = (f_max - fs/20, f_max + fs/20)
	if w_max[0] < 0:
		w_max = (0, fs/10)
	if w_max[1] > nf:
		w_max = (nf - fs/10, nf)

	# Spectrum
	print 'Calculating spectrum...'
	frames = nf/fs
	wfunc = np.blackman(fs)
	norm_spec = np.zeros((nc,fs))
	for c in range(nc):
		for i in np.arange(0, frames*fs, fs):
			norm_spec[c] += (np.abs(np.fft.fft(np.multiply(data[c,i:i+fs], wfunc), fs))/fs)**2
		norm_spec[c] = 20*np.log10( (np.sqrt(norm_spec[c]/frames)) / (data_rms[c]) )

	# Allpass
	print 'Calculating allpass...'
	ap_freqs = np.array([20, 60, 200, 600, 2000, 20000])
	ap_crest = np.zeros((len(ap_freqs),nc))
	ap_rms = np.zeros((len(ap_freqs),nc))
	ap_peak = np.zeros((len(ap_freqs),nc))
	for i in range(len(ap_freqs)):
		fc = ap_freqs[i]
		b, a = allpass(fc, fs)
		y = signal.lfilter(b, a, data, 1)
		ap_peak[i] = y.max(1)
		ap_rms[i] = rms(y, 1)
		ap_crest[i] = db(ap_peak[i], ap_rms[i])

	# Histogram
	print 'Calculating histogram...'
	hbits = bits
	if bits > 16:
		hbits = 16
	hist = np.zeros((nc, 2**hbits))
	hist_bins = np.zeros((nc, 2**hbits+1))
	for c in range(nc):
		hist[c], hist_bins[c] = np.histogram(raw_data[c], bins=2**hbits, range=(-2.0**(hbits-1), 2.0**(hbits-1)-1))
	hist_bits = np.log2((hist > 0).sum(1))
	if bits > hbits:
		hist_bits *= bits / float(hbits) # fake but counting 2**24 bins take way too long to be worth it

	# Peak vs RMS
	print 'Calculating peak vs RMS...'
	n_1s = nf/fs
	peak_1s_dbfs = np.zeros((nc, n_1s))
	rms_1s_dbfs = np.zeros((nc, n_1s))
	crest_1s_db = np.zeros((nc, n_1s))
	#print n_1s
	for c in range(nc):
		for i in range(n_1s):
			a = data[c,i*fs:(i+1)*fs].max()
			b = rms(data[c,i*fs:(i+1)*fs])
			peak_1s_dbfs[c][i] = db(a, 1.0)
			rms_1s_dbfs[c][i] = db(b, 1.0)
			crest_1s_db[c][i] = db(a, b)

	print 'Calculating checksum...'
	checksum = (raw_data.astype('uint32')**2).sum()

	#
	# Plot
	#
	print "Drawing plot..."
	c_color = ['b', 'r']
	c_name = ['left', 'right']
	fig = plt.figure(figsize=(8.3, 11.7), facecolor='white', dpi=74)
	fig.suptitle(name, fontweight='bold')
	fig.text(0.095, 0.01, ('Checksum (energy): %d' % checksum), fontsize='small', va='bottom', ha='left')
	fig.text(0.95, 0.01, ('PyMasVis %s' % (VERSION)), fontsize='small', va='bottom', ha='right')
	rc('lines', linewidth=0.5, antialiased=True)
	gs = gridspec.GridSpec(6, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1, 2, 2, 1], hspace=0.3, wspace=0.2, left=0.1, right=0.95, bottom=0.04, top=0.94)

	# Left channel
	print "Drawing left channel..."
	ax_lch = subplot(gs[0,:])
	new_data, new_nf, new_range = pixelize(data[0], ax_lch, which='both', oversample=2)
	new_fs = new_nf/sec
	new_range = np.arange(0.0, new_nf, 1)/new_fs
	plot(new_range, new_data, 'b-')
	xlim(0, sec)
	ylim(-1.0, 1.0)
	title("Left: Crest=%0.2f dB, RMS=%0.2f dBFS, Peak=%0.2f dBFS" % (crest_db[0], rms_dbfs[0], peak_dbfs[0]), fontsize='small')
	setp(ax_lch.get_xticklabels(), visible=False)
	yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
	if c_max == 0:
		mark_span(ax_lch, (w_max[0]/float(fs), w_max[1]/float(fs)))

	# Right channel
	print "Drawing right channel..."
	ax_rch = subplot(gs[1,:], sharex=ax_lch)
	new_data, new_nf, new_range = pixelize(data[1], ax_lch, which='both', oversample=2)
	new_fs = new_nf/sec
	new_range = np.arange(0.0, new_nf, 1)/new_fs
	plot(new_range, new_data, 'r-')
	xlim(0, sec)
	ylim(-1.0, 1.0)
	title("Right: Crest=%0.2f dB, RMS=%0.2f dBFS, Peak=%0.2f dBFS" % (crest_db[1], rms_dbfs[1], peak_dbfs[1]), fontsize='small')
	yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
	ax_rch.xaxis.set_major_locator(MaxNLocator(prune='both'))
	ax_rch.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
	xlabel('s', fontsize='small')
	if c_max == 1:
		mark_span(ax_rch, (w_max[0]/float(fs), w_max[1]/float(fs)))
	axis_defaults(ax_lch)
	axis_defaults(ax_rch)

	# Loudest
	print "Drawing loudest..."
	ax_max = subplot(gs[2,:])
	plot(np.arange(*w_max)/float(fs), data[c_max][np.arange(*w_max)], c_color[c_max])
	ylim(-1.0, 1.0)
	xlim(w_max[0]/float(fs), w_max[1]/float(fs))
	title("Loudest part (%s ch, %d samples > 95%% during 20 ms at %0.2f s)" % (c_name[c_max], nf_max, f_max/float(fs)), fontsize='small')
	yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
	ax_max.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
	ax_max.xaxis.set_major_formatter(FormatStrFormatter("%0.2f"))
	xlabel('s', fontsize='small')
	axis_defaults(ax_max)

	# Spectrum
	print "Drawing spectrum..."
	ax_norm = subplot(gs[3,0])
	semilogx(
		[0.02, 0.06], [-80, -90], 'k-',
		[0.02,  0.2], [-70, -90], 'k-',
		[0.02,  0.6], [-60, -90], 'k-',
		[0.02,  2  ], [-50, -90], 'k-',
		[0.02,  6  ], [-40, -90], 'k-',
		[0.02, 20  ], [-30, -90], 'k-',
		[0.02, 20  ], [-20, -80], 'k-',
		[0.02, 20  ], [-10, -70], 'k-',
		[0.06, 20  ], [-10, -60], 'k-',
		[0.2 , 20  ], [-10, -50], 'k-',
		[0.6 , 20  ], [-10, -40], 'k-',
		[2   , 20  ], [-10, -30], 'k-',
		[6   , 20  ], [-10, -20], 'k-',
		basex=10
	)
	new_spec, new_n, new_r = pixelize(norm_spec[0], ax_norm, which='max', oversample=1, method='log10', span=(20,20000))
	semilogx(new_r/1000, new_spec, 'b-', basex=10)
	new_spec, new_n, new_r = pixelize(norm_spec[1], ax_norm, which='max', oversample=1, method='log10', span=(20,20000))
	semilogx(new_r/1000, new_spec, 'r-', basex=10) # must sample log10icaly
	ylim(-90, -10)
	xlim(0.02, 20)
	ax_norm.yaxis.grid(True, which='major', linestyle=':', color='k', linewidth=0.5)
	ax_norm.xaxis.grid(True, which='both', linestyle='-', color='k', linewidth=0.5)
	ylabel('dB', fontsize='small', verticalalignment='top', rotation=0)
	xlabel('kHz', fontsize='small', horizontalalignment='right')
	title("Normalized average spectrum, %d frames" % (frames), fontsize='small')
	ax_norm.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7], minor=False)
	ax_norm.set_xticks([0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 6, 8, 9, 10], minor=True)
	ax_norm.set_xticklabels([0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7])
	yticks(np.arange(-90, -10, 10), ('', -80, -70, -60, -50, -40, -30, '', 'dB'))
	axis_defaults(ax_norm)

	# Allpass
	print "Drawing allpass..."
	ax_ap = subplot(gs[3,1])
	semilogx(ap_freqs/1000.0, crest_db[0]*np.ones(len(ap_freqs)), 'b--', basex=10)
	semilogx(ap_freqs/1000.0, crest_db[1]*np.ones(len(ap_freqs)), 'r--', basex=10)
	semilogx(ap_freqs/1000.0, ap_crest.swapaxes(0,1)[0], 'b-', basex=10)
	semilogx(ap_freqs/1000.0, ap_crest.swapaxes(0,1)[1], 'r-', basex=10)
	ylim(0,30)
	xlim(0.02, 20)
	title("Allpassed crest factor", fontsize='small')
	yticks(np.arange(0, 30, 5), ('', 5, 10, 15, 20, ''))
	xlabel('kHz', fontsize='small')
	ylabel('dB', fontsize='small', rotation=0)
	ax_ap.set_xticklabels([0, 0.1, 1], minor=False)
	xt = np.repeat([''], 17)
	xt[-1] = 2
	ax_ap.set_xticklabels(xt, minor=True)
	axis_defaults(ax_ap)

	# Histogram
	print "Drawing histogram..."
	ax_hist = subplot(gs[4,0])
	new_hist, new_n, new_range = pixelize(hist[0], ax_hist, which='max', oversample=2)
	new_hist[(new_hist == 1.0)] = 1.3
	new_hist[(new_hist < 1.0)] = 1.0
	semilogy(np.arange(new_n)*2.0/new_n-1.0, new_hist, 'b-', basey=10, drawstyle='steps')
	new_hist, new_n, new_range = pixelize(hist[1], ax_hist, which='max', oversample=2)
	new_hist[(new_hist == 1.0)] = 1.3
	new_hist[(new_hist < 1.0)] = 1.0
	semilogy(np.arange(new_n)*2.0/new_n-1.0, new_hist, 'r-', basey=10, drawstyle='steps')
	xlim(-1.1, 1.1)
	ylim(1,50000)
	xticks(np.arange(-1.0, 1.2, 0.2))
	title('Histogram, "bits": %0.1f/%0.1f' % (hist_bits[0], hist_bits[1]), fontsize='small')
	ylabel('n', fontsize='small', rotation=0)
	ax_hist.set_yticklabels(('', 10, 100, 1000), minor=False)
	axis_defaults(ax_hist)

	# Peak vs RMS
	print "Drawing peak vs RMS..."
	ax_pr = subplot(gs[4,1])
	plot(
		[-50,    0], [-50, 0], 'k-',
		[-50,  -10], [-40, 0], 'k-',
		[-50,  -20], [-30, 0], 'k-',
		[-50,  -30], [-20, 0], 'k-',
		[-50,  -40], [-10, 0], 'k-',
	)
	plot(rms_1s_dbfs[0], peak_1s_dbfs[0], 'bo', markerfacecolor='w', markeredgecolor='b', markeredgewidth=0.7)
	plot(rms_1s_dbfs[1], peak_1s_dbfs[1], 'ro', markerfacecolor='w', markeredgecolor='r', markeredgewidth=0.7)
	text(-48, -45, '0 dB', fontsize='x-small', rotation=45, va='bottom', ha='left')
	text(-48, -35, '10', fontsize='x-small', rotation=45, va='bottom', ha='left')
	text(-48, -25, '20', fontsize='x-small', rotation=45, va='bottom', ha='left')
	text(-48, -15, '30', fontsize='x-small', rotation=45, va='bottom', ha='left')
	text(-48, -5, '40', fontsize='x-small', rotation=45, va='bottom', ha='left')
	xlim(-50, 0)
	ylim(-50, 0)
	title("Peak vs RMS level", fontsize='small')
	xlabel('dBFS', fontsize='small')
	ylabel('dBFS', fontsize='small', rotation=0)
	xticks([-50, -40, -30, -20, -10, 0], ('', -40, -30, -20, '', ''))
	yticks([-50, -40, -30, -20, -10, 0], ('', -40, -30, -20, -10, ''))
	axis_defaults(ax_pr)

	# Shortterm crest
	print "Drawing short term crest..."
	ax_1s = subplot(gs[5,:])
	plot(np.arange(n_1s)+0.5, crest_1s_db[0], 'bo', markerfacecolor='w', markeredgecolor='b', markeredgewidth=0.7)
	plot(np.arange(n_1s)+0.5, crest_1s_db[1], 'ro', markerfacecolor='w', markeredgecolor='r', markeredgewidth=0.7)
	ylim(0,30)
	xlim(0,n_1s)
	yticks([10, 20], (10,))
	ax_1s.yaxis.grid(True, which='major', linestyle=':', color='k', linewidth=0.5)
	title("Short term (1 s) crest factor", fontsize='small')
	xlabel('s', fontsize='small')
	ylabel('dB', fontsize='small', rotation=0)
	ax_1s.xaxis.set_major_locator(MaxNLocator(prune='both'))
	ax_1s.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
	axis_defaults(ax_1s)

	# Save
	print "Saving analysis to %s" % outfile
	plt.savefig(outfile, format='png', dpi=74)
	return outfile


def xpixels(ax):
	return np.round(ax.bbox.bounds[2])


def pixelize(x, ax, method='linear', which='both', oversample=1, span=None):
	if not span:
		span = (0, len(x))
		if method is 'log10':
			span = (1, len(x) + 1)
	pixels = xpixels(ax)
	minmax = 1
	if which is 'both':
		minmax = 2
	nw = int(pixels*oversample)
	w = (span[1]-span[0])/(pixels*oversample)
	n = nw*minmax
	y = np.zeros(n)
	r = np.zeros(n)
	for i in range(nw):
		if method is 'linear':
			j = int(np.round(i*w + span[0]))
			k = int(np.round(j+w + span[0]))
		elif method is 'log10':
			a = np.log10(span[1]) - np.log10(span[0])
			b = np.log10(span[0])
			j = int(np.round(10**( i / float(nw) * a + b )) - 1)
			k = int(np.round(10**( (i+1) / float(nw) * a + b )))
		if i == nw - 1 and k != span[1]:
			print 'tweak k'
			k = span[1]
		r[i] = k
		if which is 'max':
			y[i] = x[j:k].max()
		elif which is 'min':
			y[i] = x[j:k].min()
		elif which is 'both':
			y[i*minmax] = x[j:k].max()
			y[i*minmax+1] = x[j:k].min()
	return (y, n, r)


def mark_span(ax, span):
	ax.axvspan(*span, edgecolor='0.2', facecolor='0.98', fill=False, linestyle='dotted', linewidth=0.8, zorder=10)


def axis_defaults(ax):
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


def rms(data, axis = 0):
	return np.sqrt((data**2).mean(axis))


def db(a, b):
	return 20*np.log10(np.divide(a, b))


def allpass(fc, fs):
	T = 1.0/fs
	w_b = 2*np.pi*fc
	p_d = (1 - np.tan(w_b*T/2)) / (1 + np.tan(w_b*T/2))
	return ([p_d, -1.0], [1.0, -p_d])


def rolling_window(a, window):
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


if __name__ == "__main__":
	if len(sys.argv) == 2:
		filename = sys.argv[1]
	else:
		print "Usage: %s filename" % sys.argv[0]
		exit(1)
	if not os.path.isfile(filename):
		print "File %s not found" % filename
		exit(1)

	infile = filename
	outfile = "%s-%s" % (filename, 'pymasvis.png')
	name = basename(filename)

	analyze(infile, outfile, name)