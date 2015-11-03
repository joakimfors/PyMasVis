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

import os
import io
import sys
import locale
import subprocess
import math
import re
import time
import logging
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.image as mpimg
import scipy.signal as signal

from os.path import basename
from tempfile import mkstemp
from subprocess import CalledProcessError
from scipy.io import wavfile
from matplotlib import rc, gridspec
from matplotlib.pyplot import plot, axis, subplot, subplots, figure, ylim, xlim, xlabel, ylabel, yticks, xticks, title, semilogx, semilogy, loglog, hold, setp, hlines, text, tight_layout, axvspan
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter, FormatStrFormatter
from PIL import Image


VERSION="0.9.2"

FIR = [
	[-0.001780280772489, 0.003253283030257, -0.005447293390376, 0.008414568116553, -0.012363296099675, 0.017436805871070, -0.024020143876810, 0.032746828420101, -0.045326602900760, 0.066760686868173, -0.120643370377371, 0.989429605248410, 0.122160009958442, -0.046376232812786, 0.022831393004364, -0.011580897261667, 0.005358105753167, -0.001834671998839, -0.000103681038815, 0.001002216283171, -0.001293611238062, 0.001184842429930, -0.000908719377960, 0.002061304229100],
	[-0.001473218555432, 0.002925336766866, -0.005558126468508, 0.009521159741206, -0.015296028027209, 0.023398977482278, -0.034752051245281, 0.050880967772373, -0.075227488678419, 0.116949442543490, -0.212471239510148, 0.788420616540440, 0.460788819545818, -0.166082211358253, 0.092555759769552, -0.057854829231334, 0.037380809681132, -0.024098441541823, 0.015115653825711, -0.009060645712669, 0.005033299068467, -0.002511544062471, 0.001030723665756, -0.000694079453823],
	[-0.000694079453823, 0.001030723665756, -0.002511544062471, 0.005033299068467, -0.009060645712669, 0.015115653825711, -0.024098441541823, 0.037380809681132, -0.057854829231334, 0.092555759769552, -0.166082211358253, 0.460788819545818, 0.788420616540440, -0.212471239510148, 0.116949442543490, -0.075227488678419, 0.050880967772373, -0.034752051245281, 0.023398977482278, -0.015296028027209, 0.009521159741206, -0.005558126468508, 0.002925336766866, -0.001473218555432],
	[0.002061304229100, -0.000908719377960, 0.001184842429930, -0.001293611238062, 0.001002216283171, -0.000103681038815, -0.001834671998839, 0.005358105753167, -0.011580897261667, 0.022831393004364, -0.046376232812786, 0.122160009958442, 0.989429605248410, -0.120643370377371, 0.066760686868173, -0.045326602900760, 0.032746828420101, -0.024020143876810, 0.017436805871070, -0.012363296099675, 0.008414568116553, -0.005447293390376, 0.003253283030257, -0.001780280772489]
]

class Timer(object):
	def __init__(self, verbose=False):
		self.verbose = verbose

	def __enter__(self):
		self.start = time.time()
		return self

	def __exit__(self, *args):
		self.end = time.time()
		self.secs = self.end - self.start
		self.msecs = self.secs * 1000  # millisecs
		if self.verbose:
			print '  %f ms' % self.msecs

def load_file(infile):
	src = 'file'
	name = os.path.splitext(basename(infile))[0]
	ext = os.path.splitext(infile)[1][1:].strip().lower()
	fmt = None
	title = None
	artist = None
	date = None
	album = None
	track = None
	bps = '1411 kbps'
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
		tmpfd, tmpfile = mkstemp(suffix='.wav')
		try:
			output = subprocess.check_output(
				[ffmpeg_bin, '-y', '-i', infile, '-vn', '-map_metadata', '-1:g', '-map_metadata', '-1:s', '-flags', 'bitexact', tmpfile],
				stderr=subprocess.STDOUT
			)
			infile = tmpfile
			print output
			for line in output.splitlines():
				match = re.match('^Output.*', line)
				if match:
					print "Parsed metadata"
					break
				match = re.match('^Input.*?, (.*?),.*', line, flags=re.I)
				if match:
					print "fmt", match.groups()
					fmt = match.group(1)
				match = re.match('\s*title\s*?: (.*)', line, flags=re.I)
				if match:
					print "title", match.groups()
					title = match.group(1)
				match = re.match('\s*artist\s*?: (.*)', line, flags=re.I)
				if match:
					print "artist", match.groups()
					artist = match.group(1)
				match = re.match('\s*album\s*?: (.*)', line, flags=re.I)
				if match:
					print "album", match.groups()
					album = match.group(1)
				match = re.match('\s*track\s*?: (.*)', line, flags=re.I)
				if match:
					print "track", match.groups()
					track = match.group(1)
				match = re.match('\s*date\s*?: (.*)', line, flags=re.I)
				if match:
					print "date", match.groups()
					date = match.group(1)
				match = re.match('\s*Duration:.*bitrate: (\d*?) kb/s', line)
				if match:
					print "bps", match.groups()
					bps = match.group(1) + ' kbps'
				match = re.match('\s*Stream.*?Audio: (.*?),.*, (\d*?) kb/s', line)
				if match:
					print "fmt, bps", match.groups()
					stream_fmt, bps = match.group(1, 2)
					bps = bps + ' kbps'
		except CalledProcessError as e:
			print 'Could not convert %s' % infile
			exit(e.returncode)
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
	raw_data = raw_data.swapaxes(0,1)
	data = raw_data.astype('float')
	data /= 2**(bits-1)
	if tmpfile and os.path.isfile(tmpfile):
		os.close(tmpfd)
		os.remove(tmpfile)

	if not fmt:
		fmt = ext
	if artist and title:
		name = '%s - %s' % (artist, title)
	return {
		'data': {
			'fixed': raw_data,
			'float': data
		},
		'frames': nf,
		'samplerate': fs,
		'channels': nc,
		'bitdepth': bits,
		'duration': sec,
		'metadata': {
			'source': src,
			'filename': basename(infile),
			'extension': ext,
			'format': fmt,
			'name': name,
			'artist': artist,
			'title': title,
			'album': album,
			'track': track,
			'date': date,
			'bps': bps
		}
	}


def load_spotify(link, username, password):
	dumper = SpotiDump(username, password)
	track = dumper.dump(link)
	return track


def analyze(track):
	data = track['data']['float']
	raw_data = track['data']['fixed']
	nf = track['frames']
	fs = track['samplerate']
	nc = track['channels']
	bits = track['bitdepth']
	sec = track['duration']
	name = track['metadata']['filename']
	ext = track['metadata']['extension']
	bps = track['metadata']['bps']

	print "Processing %s" % name
	print "\tsample rate: %d\n\tchannels: %d\n\tframes: %d\n\tbits: %d\n\tformat: %s\n\tbitrate: %s" % (fs, nc, nf, bits, ext, bps)

	# Peak / RMS
	with Timer(True) as t:
		print 'Calculating peak and RMS...'
		data_rms = rms(data, 1)
		data_peak = np.abs(data).max(1)

		# Peak dBFS
		peak_dbfs = db(data_peak, 1.0)

		# RMS dBFS
		rms_dbfs = db(data_rms, 1.0)

		# Crest dB
		crest_db = db(data_peak, data_rms)

	# Loudest
	with Timer(True) as t:
		print 'Calculating loudest...'
		window = int(fs / 50)
		peak = data_peak.max()
		sn_max = 0
		pos_max = 0
		c_max = 0
		sn_cur = 0
		c_max, f_max, nf_cur, nf_max = 0, 0, 0, 0
		for c in range(nc):
			# Find the indices where the sample value is 95% of track peak value
			peaks = np.flatnonzero(np.abs(data[c]) > 0.95*peak)
			if len(peaks) == 0:
				continue
			nf_cur = 0
			it = np.nditer(peaks, flags=['buffered','c_index'], op_flags=['readonly'])
			for e in it:
				i = it.iterindex
				# Count the number of samples (indices) within the window
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

	# True peak
	with Timer(True) as t:
		print 'Calculating true peak...'
		fir_phases = np.array(FIR)
		#print 'Data peak', data_peak
		true_peak = np.copy(data_peak)
		for c in range(nc):
			for i in range(len(data[c])-24):
				peak = np.abs(np.dot(fir_phases, data[c][i:i+24])).max()
				if peak > true_peak[c]:
					true_peak[c] = peak
		#print 'True peaks', true_peak
		true_peak_dbfs = db(true_peak, 1.0)

	# Spectrum
	with Timer(True) as t:
		print 'Calculating spectrum...'
		frames = nf/fs
		wfunc = np.blackman(fs)
		norm_spec = np.zeros((nc,fs))
		for c in range(nc):
			for i in np.arange(0, frames*fs, fs):
				norm_spec[c] += (np.abs(np.fft.fft(np.multiply(data[c,i:i+fs], wfunc), fs))/fs)**2
			norm_spec[c] = 20*np.log10( (np.sqrt(norm_spec[c]/frames)) / (data_rms[c]) )

	# Allpass
	with Timer(True) as t:
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
	with Timer(True) as t:
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
	with Timer(True) as t:
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

	with Timer(True) as t:
		print 'Calculating checksum...'
		checksum = (raw_data.astype('uint32')**2).sum()

	#
	return {
		'crest_db': crest_db,
		'rms_dbfs': rms_dbfs,
		'peak_dbfs': peak_dbfs,
		'true_peak_dbfs': true_peak_dbfs,
		'c_max': c_max,
		'w_max': w_max,
		'f_max': f_max,
		'nf_max': nf_max,
		'norm_spec': norm_spec,
		'frames': frames,
		'n_1s': n_1s, # FIXME: same as frames?
		'ap_freqs': ap_freqs,
		'ap_crest': ap_crest,
		'hist': hist,
		'hist_bits': hist_bits,
		'rms_1s_dbfs': rms_1s_dbfs,
		'peak_1s_dbfs': peak_1s_dbfs,
		'crest_1s_db': crest_1s_db,
		'checksum': checksum
	}


class MaxNLocatorMod(MaxNLocator):
	def __init__(self, *args, **kwargs):
		super(MaxNLocatorMod, self).__init__(*args, **kwargs)

	def tick_values(self, vmin, vmax):
		ticks = super(MaxNLocatorMod, self).tick_values(vmin, vmax)
		span = vmax - vmin
		if ticks[-1] > vmax - 0.05*span:
			ticks = ticks[0:-1]
		return ticks


def render(track, analysis, header):
	#
	# Plot
	#
	checksum = analysis['checksum']
	with Timer(True) as t:
		print "Drawing plot..."
		c_color = ['b', 'r']
		c_name = ['left', 'right']
		subtitle1 = 'Encoding: %s  Bitrate: %s  Source: %s' % (track['metadata']['format'], track['metadata']['bps'], track['metadata']['source'])
		subtitle2 = []
		if track['metadata']['album']:
			subtitle2.append('Album: %.*s' % (50, track['metadata']['album']))
		if track['metadata']['track']:
			subtitle2.append('Track: %s' % track['metadata']['track'])
		if track['metadata']['date']:
			subtitle2.append('Date: %s' % track['metadata']['date'])
		subtitle2 = '  '.join(subtitle2)
		dpi = 72
		fig = plt.figure(figsize=(606.0/dpi, 946.0/dpi), facecolor='white', dpi=dpi)
		fig.suptitle(header, fontsize='medium')
		fig.text(0.5, 0.95, subtitle1, fontsize='small', horizontalalignment='center')
		fig.text(0.5, 0.93, subtitle2, fontsize='small', horizontalalignment='center')
		fig.text(0.075, 0.01, ('Checksum (energy): %d' % checksum), fontsize='small', va='bottom', ha='left')
		fig.text(0.975, 0.01, ('PyMasVis %s' % (VERSION)), fontsize='small', va='bottom', ha='right')
		rc('lines', linewidth=0.5, antialiased=True)
		gs = gridspec.GridSpec(6, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1, 2, 2, 1], hspace=0.3, wspace=0.2, left=0.075, right=0.975, bottom=0.04, top=0.90)

	# Left channel
	data = track['data']['float']
	sec = track['duration']
	crest_db = analysis['crest_db']
	rms_dbfs = analysis['rms_dbfs']
	peak_dbfs = analysis['peak_dbfs']
	true_peak_dbfs = analysis['true_peak_dbfs']
	c_max = analysis['c_max']
	w_max = analysis['w_max']
	fs = track['samplerate']
	with Timer(True) as t:
		print "Drawing left channel..."
		ax_lch = subplot(gs[0,:])
		new_data, new_nf, new_range = pixelize(data[0], ax_lch, which='both', oversample=2)
		new_fs = new_nf/float(sec)
		new_range = np.arange(0.0, new_nf, 1)/new_fs
		plot(new_range, new_data, 'b-')
		xlim(0, sec)
		ylim(-1.0, 1.0)
		title(u"Left: Crest=%0.2f dB, RMS=%0.2f dBFS, Peak=%0.2f dBFS, True peak≈%0.2f dBFS" % (crest_db[0], rms_dbfs[0], peak_dbfs[0], true_peak_dbfs[0]), fontsize='small', loc='left')
		setp(ax_lch.get_xticklabels(), visible=False)
		yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
		if c_max == 0:
			mark_span(ax_lch, (w_max[0]/float(fs), w_max[1]/float(fs)))

	# Right channel
	with Timer(True) as t:
		print "Drawing right channel..."
		ax_rch = subplot(gs[1,:], sharex=ax_lch)
		new_data, new_nf, new_range = pixelize(data[1], ax_lch, which='both', oversample=2)
		new_fs = new_nf/float(sec)
		new_range = np.arange(0.0, new_nf, 1)/new_fs
		plot(new_range, new_data, 'r-')
		xlim(0, sec)
		ylim(-1.0, 1.0)
		title(u"Right: Crest=%0.2f dB, RMS=%0.2f dBFS, Peak=%0.2f dBFS, True peak≈%0.2f dBFS" % (crest_db[1], rms_dbfs[1], peak_dbfs[1], true_peak_dbfs[1]), fontsize='small', loc='left')
		yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
		ax_rch.xaxis.set_major_locator(MaxNLocatorMod(prune='both'))
		ax_rch.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
		xlabel('s', fontsize='small')
		if c_max == 1:
			mark_span(ax_rch, (w_max[0]/float(fs), w_max[1]/float(fs)))
		axis_defaults(ax_lch)
		axis_defaults(ax_rch)

	# Loudest
	f_max = analysis['f_max']
	nf_max = analysis['nf_max']
	with Timer(True) as t:
		print "Drawing loudest..."
		ax_max = subplot(gs[2,:])
		plot(np.arange(*w_max)/float(fs), data[c_max][np.arange(*w_max)], c_color[c_max])
		ylim(-1.0, 1.0)
		xlim(w_max[0]/float(fs), w_max[1]/float(fs))
		title("Loudest part (%s ch, %d samples > 95%% during 20 ms at %0.2f s)" % (c_name[c_max], nf_max, f_max/float(fs)), fontsize='small', loc='left')
		yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
		ax_max.xaxis.set_major_locator(MaxNLocatorMod(nbins=5, prune='both'))
		ax_max.xaxis.set_major_formatter(FormatStrFormatter("%0.2f"))
		xlabel('s', fontsize='small')
		axis_defaults(ax_max)

	# Spectrum
	norm_spec = analysis['norm_spec']
	frames = analysis['frames']
	with Timer(True) as t:
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
		semilogx(new_r/1000.0, new_spec, 'b-', basex=10)
		new_spec, new_n, new_r = pixelize(norm_spec[1], ax_norm, which='max', oversample=1, method='log10', span=(20,20000))
		semilogx(new_r/1000.0, new_spec, 'r-', basex=10)
		ylim(-90, -10)
		xlim(0.02, 20)
		ax_norm.yaxis.grid(True, which='major', linestyle=':', color='k', linewidth=0.5)
		ax_norm.xaxis.grid(True, which='both', linestyle='-', color='k', linewidth=0.5)
		ylabel('dB', fontsize='small', verticalalignment='top', rotation=0)
		xlabel('kHz', fontsize='small', horizontalalignment='right')
		title("Normalized average spectrum, %d frames" % (frames), fontsize='small', loc='left')
		ax_norm.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7], minor=False)
		ax_norm.set_xticks([0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 6, 8, 9, 10], minor=True)
		ax_norm.set_xticklabels([0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7])
		yticks(np.arange(-90, -10, 10), ('', -80, -70, -60, -50, -40, -30, '', 'dB'))
		axis_defaults(ax_norm)

	# Allpass
	ap_freqs = analysis['ap_freqs']
	ap_crest = analysis['ap_crest']
	with Timer(True) as t:
		print "Drawing allpass..."
		ax_ap = subplot(gs[3,1])
		semilogx(ap_freqs/1000.0, crest_db[0]*np.ones(len(ap_freqs)), 'b--', basex=10)
		semilogx(ap_freqs/1000.0, crest_db[1]*np.ones(len(ap_freqs)), 'r--', basex=10)
		semilogx(ap_freqs/1000.0, ap_crest.swapaxes(0,1)[0], 'b-', basex=10)
		semilogx(ap_freqs/1000.0, ap_crest.swapaxes(0,1)[1], 'r-', basex=10)
		ylim(0,30)
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
	with Timer(True) as t:
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
		xticks(np.arange(-1.0, 1.2, 0.2), (-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1))
		yticks([10, 100, 1000], (10, 100, 1000))
		title('Histogram, "bits": %0.1f/%0.1f' % (hist_bits[0], hist_bits[1]), fontsize='small', loc='left')
		ylabel('n', fontsize='small', rotation=0)
		axis_defaults(ax_hist)

	# Peak vs RMS
	rms_1s_dbfs = analysis['rms_1s_dbfs']
	peak_1s_dbfs = analysis['peak_1s_dbfs']
	with Timer(True) as t:
		print "Drawing peak vs RMS..."
		ax_pr = subplot(gs[4,1])
		plot(
			[-50,    0], [-50, 0], 'k-',
			[-50,  -10], [-40, 0], 'k-',
			[-50,  -20], [-30, 0], 'k-',
			[-50,  -30], [-20, 0], 'k-',
			[-50,  -40], [-10, 0], 'k-',
		)
		text(-48, -45, '0 dB', fontsize='x-small', rotation=45, va='bottom', ha='left')
		text(-48, -35, '10', fontsize='x-small', rotation=45, va='bottom', ha='left')
		text(-48, -25, '20', fontsize='x-small', rotation=45, va='bottom', ha='left')
		text(-48, -15, '30', fontsize='x-small', rotation=45, va='bottom', ha='left')
		text(-48, -5, '40', fontsize='x-small', rotation=45, va='bottom', ha='left')
		plot(rms_1s_dbfs[0], peak_1s_dbfs[0], 'bo', markerfacecolor='w', markeredgecolor='b', markeredgewidth=0.7)
		plot(rms_1s_dbfs[1], peak_1s_dbfs[1], 'ro', markerfacecolor='w', markeredgecolor='r', markeredgewidth=0.7)
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
	with Timer(True) as t:
		print "Drawing short term crest..."
		ax_1s = subplot(gs[5,:])
		plot(np.arange(n_1s)+0.5, crest_1s_db[0], 'bo', markerfacecolor='w', markeredgecolor='b', markeredgewidth=0.7)
		plot(np.arange(n_1s)+0.5, crest_1s_db[1], 'ro', markerfacecolor='w', markeredgecolor='r', markeredgewidth=0.7)
		ylim(0,30)
		xlim(0,n_1s)
		yticks([10, 20], (10,))
		ax_1s.yaxis.grid(True, which='major', linestyle=':', color='k', linewidth=0.5)
		title("Short term (1 s) crest factor", fontsize='small', loc='left')
		xlabel('s', fontsize='small')
		ylabel('dB', fontsize='small', rotation=0)
		ax_1s.xaxis.set_major_locator(MaxNLocatorMod(prune='both'))
		ax_1s.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
		axis_defaults(ax_1s)

	# Save
	with Timer(True) as t:
		print "Saving..."
		f = io.BytesIO()
		plt.savefig(f, format='png', dpi=dpi, transparent=False)
	return f


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


def rms(data, axis = 0):
	return np.sqrt((data**2).mean(axis))


def db(a, b):
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide(a,b)
		c = 20*np.log10(c)
		if isinstance(c, np.ndarray):
			c[c == -np.inf] = -128.0
		elif isinstance(c, np.float64) and c == -np.inf:
			c = -128.0
		c = np.nan_to_num(c)
	return c


def allpass(fc, fs):
	T = 1.0/fs
	w_b = 2*np.pi*fc
	p_d = (1 - np.tan(w_b*T/2)) / (1 + np.tan(w_b*T/2))
	return ([p_d, -1.0], [1.0, -p_d])


def rolling_window(a, window):
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def run(infile, outfile=None, header=None, username=None, password=None):
	loader = None
	loader_args = []
	spotify = False
	if os.path.isfile(infile):
		print "Selecting file loader"
		loader = load_file
		loader_args = [infile]
	elif infile.startswith('spotify:'):
		print "Selecting Spotify loader"
		loader = load_spotify
		loader_args = [infile, username, password]
		spotify = True
	track = loader(*loader_args)
	if not outfile and spotify:
		outfile = "%s.spotify-pymasvis.png" % track['metadata']['name']
	elif not outfile:
		outfile = "%s-pymasvis.png" % infile
	if not header:
		header = "%s" % (track['metadata']['name'])
	analysis = analyze(track)
	picture = render(track, analysis, header)
	img = Image.open(picture)
	img = img.convert(mode='P', palette='ADAPTIVE', colors=256)
	img.save(outfile, 'PNG')


if __name__ == "__main__":
	import optparse
	import glob
	usage = "usage: %prog [options] arg"
	op = optparse.OptionParser(usage)
	op.add_option("-u", "--username", help="Spotify username")
	op.add_option("-p", "--password", help="Spotify password")
	(options, args) = op.parse_args()
	if len(args) == 0:
		op.print_help()
		exit(0)
	candidates = []
	for arg in args:
		if arg.startswith('spotify:'):
			from spotidump import SpotiDump
			candidates.append(arg)
			continue
		for filename in glob.glob(os.path.expanduser(arg)):
			candidates.append(filename)
	language, encoding = locale.getdefaultlocale()
	if len(candidates) == 0:
		print "No valid candidates for analysation found: " + " ".join(args)
	for candidate in candidates:
		infile = candidate.decode(encoding)
		outfile = None
		name = None
		run(infile, outfile, name, options.username, options.password)
