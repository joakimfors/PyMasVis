import sys
import inspect
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
from scikits.audiolab import Format, Sndfile
from matplotlib import rc, gridspec
from matplotlib.pyplot import plot, axis, subplot, subplots, figure, ylim, xlim, xlabel, ylabel, yticks, xticks, title, semilogx, semilogy, loglog, hold, setp, hlines, text, tight_layout, axvspan

VERSION="0.0.1"

def analyze(filename):
	f = Sndfile(filename, 'r')
	fs = f.samplerate
	nc = f.channels
	enc = f.encoding
	nf = f.nframes

	print basename(filename)

	print "Hz: %d ch: %d enc: %s frames: %d" % (fs, nc, enc, nf)
	#print inspect.getmembers(f)

	#data = f.read_frames(n, np.dtype('i2')).swapaxes(0,1)
	data = f.read_frames(nf).swapaxes(0,1)
	print np.abs(data).max()
	#print "max: %d %d val: %d %d" % (np.argmax(data[:,0]), np.argmax(data[:,1]), data[np.argmax(data[:,0]), 0], data[np.argmax(data[:,1]), 1])
	#print data[fs*2]
	#print data[fs*2,1]

	data_rms = rms(data, 1) # [np.sqrt(np.mean(data[0]**2)), np.sqrt(np.mean(data[1]**2))]
	data_peak = np.abs(data).max(1)
	#plt.plot(data)
	#plt.show()
	#print data[0,100000:100010]
	#print data[0,100000:100010]**2
	#print np.mean(data[0,100000:100010]**2)
	#print np.sqrt(np.mean(data[0,100000:100010]**2))

	# Peak dBFS
	peak_dbfs = db(data_peak, 1.0) #[ 20*np.log10(np.abs(data[0]).max()), 20*np.log10(np.abs(data[1]).max()) ]

	# RMS dBFS
	rms_dbfs = db(data_rms, 1.0) # [ 20*np.log10(rms[0]), 20*np.log10(rms[1]) ]

	# Crest dB
	crest_db = db(data_peak, data_rms) # [ 20*np.log10(np.abs(data[0]).max() / rms[0]), 20*np.log10(np.abs(data[1]).max() / rms[1]) ]

	print "RMS", data_rms, "Peak", data_peak, "Peak dBFS", peak_dbfs, "RMS dBFS", rms_dbfs, "Crest dB", crest_db

	# Loudest
	window = int(fs / 50)
	peak = data_peak.max()
	sn_max = 0
	pos_max = 0
	c_max = 0
	sn_cur = 0
	"""for c in range(nc):
		for i, e in zip(range(n), np.abs(data[c])):
			if e > 0.95*peak:
				sn_cur = 0
				for j in range(window):
					if i+j < n and np.abs(data[c,i+j]) > peak*0.95:
						sn_cur += 1
				if sn_cur > sn_max:
					sn_max = sn_cur
					c_max = c
					pos_max = i
					print "intermediate max", sn_max, c_max, pos_max, pos_max/float(fs)
	print "final max", sn_max, c_max, pos_max, pos_max/float(fs)"""
	c_max, f_max, nf_cur, nf_max = 0, 0, 0, 0
	for c in range(nc):
		peaks = np.flatnonzero(np.abs(data[c]) > 0.95*peak)
		print(len(peaks))

		nf_cur = 0
		#for i, e in zip(range(len(peaks)), peaks):
		it = np.nditer(peaks, flags=['buffered','c_index'], op_flags=['readonly'])
		for e in it:
			i = it.iterindex
			#nf_cur = np.count_nonzero(peaks[i:] < e + window)
			nf_cur = (peaks[i:i+window] < e + window).sum()
			if nf_cur > nf_max:
				c_max = c
				nf_max = nf_cur
				f_max = e
	print c_max, nf_max, f_max, f_max/float(fs)

	w_max = (f_max - fs/20, f_max + fs/20)
	if w_max[0] < 0:
		w_max = (0, fs/10)
	if w_max[1] > nf:
		w_max = (nf - fs/10, nf)

	#peaks = (np.abs(data) > 0.05*peak)
	#print np.sum(rolling_window(peaks, window), -1).max()

	# Spectrum
	print nf/float(fs)
	frames = nf/fs
	wfunc = np.blackman(fs)
	norm_spec = np.zeros((nc,fs))
	X = np.zeros((nc, frames, fs))
	tmp = np.zeros(fs)
	for c in range(nc):
		for i in np.arange(0, frames*fs, fs):
			#norm_spec += np.abs(np.fft.fft(np.multiply(data[0,i:i+fs], wfunc), fs))
			#tmp = np.abs(np.fft.fft(np.multiply(data[c,i:i+fs], wfunc), fs))
			#p = (data[c,i:i+fs]**2).sum()
			X[c][i/fs] = np.abs(np.fft.fft(np.multiply(data[c,i:i+fs], wfunc), fs))
			#P = (X[c][i/fs]**2).sum()
			#print "powah", p, P
			#norm_spec[c] += 20*np.log10(tmp/tmp.max())
		X_max = X[c].max()
		norm_spec[c] = 20*np.log10(X[c]/X_max).mean(0)
		#norm_spec[c] = 20*np.log10(X[c]/np.sqrt(fs)).mean(0)
	#print 'X fft max/ch', X.max(1)
	#print 'X fft max/ch', X.max(1).max(1)
	#X_max = X.max(1).max(1)
	#norm_spec = 20*np.log10(tmp/X_max)
	#plt.plot(20*np.log10(norm_spec[np.arange(fs/2)]/norm_spec.max()))
	#plt.show()
	#plt.semilogx(np.arange(fs/2), norm_spec[0:fs/2], basex=10)
	#plt.show()

	# Allpass
	ap_freqs = np.array([20, 60, 200, 600, 2000, 20000])
	ap_crest = np.zeros((len(ap_freqs),nc))
	ap_rms = np.zeros((len(ap_freqs),nc))
	ap_peak = np.zeros((len(ap_freqs),nc))
	"""x = np.zeros(16).reshape(4,2)#data.view().swapaxes(0,1)
	x[0,0] =
	y = np.zeros(x.shape)
	print 'x', x.flags
	print 'y', y.flags
	for freq in ap_freqs:
		b, a = allpass(freq, fs)
		y[0] = b[0]*x[0]
		print x[0]
		print y[0]
		it = np.nditer([x, y], flags=['buffered', 'c_index'], op_flags=[['readonly'],['readwrite']], order='C')
		print it.itersize
		it.iternext()
		for x_i, y_i in it:
			#y[i] = b[0]*x[i] + b[1]*x[i-1] - a[1]*y[i-1]
			i = it.iterindex
			y_i = b[0]*x_i + b[1]*x[i-1] - a[1]*y[i-1]
			if i <= 1 or i >= nf-2:
				print y[0:1], x[0:1], i"""
	for i in range(len(ap_freqs)):
		fc = ap_freqs[i]
		b, a = allpass(fc, fs)
		y = signal.lfilter(b, a, data, 1)
		ap_peak[i] = y.max(1)
		ap_rms[i] = rms(y, 1)
		ap_crest[i] = db(ap_peak[i], ap_rms[i])
	print 'AP Crest', ap_crest

	# Histogram
	hist = np.zeros((nc, 2**16))
	hist_bins = np.zeros((nc, 2**16+1))
	for c in range(nc):
		hist[c], hist_bins[c] = np.histogram(data[c], bins=2**16, range=(-1.0, 1.0))
	print hist.shape
	hist_bits = np.log2((hist > 0).sum(1))
	print hist_bits


	# Peak vs RMS
	n_1s = nf/fs
	peak_1s_dbfs = np.zeros((nc, n_1s))
	rms_1s_dbfs = np.zeros((nc, n_1s))
	crest_1s_db = np.zeros((nc, n_1s))
	print n_1s
	for c in range(nc):
		for i in range(n_1s):
			a = data[c,i*fs:(i+1)*fs].max()
			b = rms(data[c,i*fs:(i+1)*fs])
			peak_1s_dbfs[c][i] = db(a, 1.0)
			rms_1s_dbfs[c][i] = db(b, 1.0)
			crest_1s_db[c][i] = db(a, b)

	print peak_1s_dbfs
	print rms_1s_dbfs
	print crest_1s_db


	#
	# Plot
	#
	c_color = ['b', 'r']
	c_name = ['left', 'right']



	fig = plt.figure(figsize=(8.3, 11.7), facecolor='white')
	fig.suptitle(basename(filename), fontweight='bold')
	fig.text(0.95, 0.01, ('PyMasVis %s' % (VERSION)), fontsize='small', va='bottom', ha='right')

	"""cc_img = mpimg.imread('cc.png')
	pd_img = mpimg.imread('pd.png')
	fig.figimage(cc_img, 16, fig.bbox.ymax - cc_img.shape[1] - 16)
	fig.figimage(pd_img, 16 + cc_img.shape[0] + 5, fig.bbox.ymax - pd_img.shape[1] - 16)"""

	rc('lines', linewidth=0.5, antialiased=True)

	gs = gridspec.GridSpec(6, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1, 2, 2, 1], hspace=0.3, wspace=0.2, left=0.1, right=0.95, bottom=0.04, top=0.94)

	data_d = signal.decimate(data, 18, n=1, ftype='iir', axis=1)
	fs_d = fs/18
	nf_d = nf/18

	# Left channel
	ax_lch = subplot(gs[0,:]) #subplot(4,2,1)
	#simplify(plot(np.arange(nf)/float(fs), data[0], 'b-', rasterized=True, lod=True))
	#xlim(0, nf/fs)
	simplify(plot(np.arange(nf_d)/float(fs_d), data_d[0], 'b-', rasterized=True, lod=True))
	xlim(0, nf_d/fs_d)
	ylim(-1.0, 1.0)
	title("Left: Crest=%0.2f dB, RMS=%0.2f dBFS, Peak=%0.2f dBFS" % (crest_db[0], rms_dbfs[0], peak_dbfs[0]), fontsize='small')
	setp(ax_lch.get_xticklabels(), visible=False)
	yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
	if c_max == 0:
		mark_span(ax_lch, (w_max[0]/float(fs), w_max[1]/float(fs)))

	# Right channel
	ax_rch = subplot(gs[1,:], sharex=ax_lch)
	#simplify(plot(np.arange(nf)/float(fs), data[1], 'r-', rasterized=True, lod=True))
	#xlim(0, nf/fs)
	simplify(plot(np.arange(nf_d)/float(fs_d), data_d[1], 'r-', rasterized=True, lod=True))
	xlim(0, nf_d/fs_d)
	ylim(-1.0, 1.0)
	title("Right: Crest=%0.2f dB, RMS=%0.2f dBFS, Peak=%0.2f dBFS" % (crest_db[1], rms_dbfs[1], peak_dbfs[1]), fontsize='small')
	yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
	ax_rch.get_xticklabels()[-1].set_visible(False)
	xlabel('s', fontsize='small')
	if c_max == 1:
		print w_max
		mark_span(ax_rch, (w_max[0]/float(fs), w_max[1]/float(fs)))


	axis_defaults(ax_lch)
	axis_defaults(ax_rch)

	# Loudest
	ax_max = subplot(gs[2,:])
	simplify(plot(np.arange(*w_max)/float(fs), data[c_max][np.arange(*w_max)], c_color[c_max], lod=True))
	ylim(-1.0, 1.0)
	xlim(w_max[0]/float(fs), w_max[1]/float(fs))
	title("Loudest part (%s ch, %d samples > 95%% during 20 ms at %0.2f s)" % (c_name[c_max], nf_max, f_max/float(fs)), fontsize='small')
	yticks([1, -0.5, 0, 0.5, 1], ('', -0.5, 0, '', ''))
	ax_max.get_xticklabels()[-1].set_visible(False)
	xlabel('s', fontsize='small')

	#print ax_max.get_xaxis_text1_transform()
	#for label in ax_max.get_xmajorticklabels():
	#	print 'tick pos', ax_max.labelpad #label.get_text(), label._pad

	axis_defaults(ax_max)

	# Normalized
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
	simplify(semilogx(np.arange(0,fs/2.0)/1000, norm_spec[0,0:fs/2], 'b-', basex=10, lod=True))
	simplify(semilogx(np.arange(0,fs/2.0)/1000, norm_spec[1,0:fs/2], 'r-', basex=10, lod=True))
	ylim(-90, -10)
	#xlim(0.02, (fs/2.0)/1000)
	xlim(0.02, 20)
	ax_norm.yaxis.grid(True, which='major', linestyle=':', color='k', linewidth=0.5)
	ax_norm.xaxis.grid(True, which='both', linestyle='-', color='k', linewidth=0.5)
	ylabel('dB', fontsize='small', verticalalignment='top', rotation=0)
	xlabel('kHz', fontsize='small', horizontalalignment='right')
	title("Normalized average spectrum, %d frames" % (frames), fontsize='small')
	#ax_norm.ticklabel_format(axis='x', style='plain')
	ax_norm.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7], minor=False)
	ax_norm.set_xticks([0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 6, 8, 9, 10], minor=True)
	ax_norm.set_xticklabels([0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7])
	yticks(np.arange(-90, -10, 10), ('', -80, -70, -60, -50, -40, -30, '', 'dB'))

	axis_defaults(ax_norm)



	# Allpass
	ax_ap = subplot(gs[3,1])
	semilogx(ap_freqs/1000.0, crest_db[0]*np.ones(len(ap_freqs)), 'b--', basex=10, rasterized=True, lod=True)
	semilogx(ap_freqs/1000.0, crest_db[1]*np.ones(len(ap_freqs)), 'r--', basex=10, rasterized=True, lod=True)
	semilogx(ap_freqs/1000.0, ap_crest.swapaxes(0,1)[0], 'b-', basex=10, rasterized=True, lod=True)
	semilogx(ap_freqs/1000.0, ap_crest.swapaxes(0,1)[1], 'r-', basex=10, rasterized=True, lod=True)
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
	ax_hist = subplot(gs[4,0])
	print hist.shape
	simplify(semilogy(np.arange(2**16)/2.0**15-1.0, hist[0], 'b-', basey=10, rasterized=True, lod=True, drawstyle='step-mid'))
	simplify(semilogy(np.arange(2**16)/2.0**15-1.0, hist[1], 'r-', basey=10, rasterized=True, lod=True, drawstyle='step-mid'))
	xlim(-1.1, 1.1)
	ylim(0,50000)
	#xticks([0, 2**15, 2**16], [-1, 0, 1])
	xticks(np.arange(-1.0, 1.2, 0.2))
	title('Histogram, "bits": %0.1f/%0.1f' % (hist_bits[0], hist_bits[1]), fontsize='small')
	ylabel('n', fontsize='small', rotation=0)
	ax_hist.set_yticklabels(('', 10, 100, 1000), minor=False)

	axis_defaults(ax_hist)

	# Peak vs RMS
	ax_pr = subplot(gs[4,1])
	plot(
		[-50,    0], [-50, 0], 'k-',
		[-50,  -10], [-40, 0], 'k-',
		[-50,  -20], [-30, 0], 'k-',
		[-50,  -30], [-20, 0], 'k-',
		[-50,  -40], [-10, 0], 'k-',
	)
	plot(rms_1s_dbfs[0], peak_1s_dbfs[0], 'bo', rasterized=True, lod=True)
	plot(rms_1s_dbfs[1], peak_1s_dbfs[1], 'ro', rasterized=True, lod=True)
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
	ax_1s = subplot(gs[5,:])
	plot(np.arange(n_1s)+0.5, crest_1s_db[0], 'bo')
	plot(np.arange(n_1s)+0.5, crest_1s_db[1], 'ro')
	ylim(0,30)
	xlim(0,n_1s)
	yticks([10, 20], (10,))
	ax_1s.yaxis.grid(True, which='major', linestyle=':', color='k', linewidth=0.5)
	title("Short term (1 s) crest factor", fontsize='small')
	xlabel('s', fontsize='small')
	ylabel('dB', fontsize='small', rotation=0)
	ax_1s.get_xticklabels()[-1].set_visible(False)

	axis_defaults(ax_1s)

	#plt.show()
	out_file = "%s.png" % filename
	print "Saving analysis to %s" % out_file
	plt.savefig(out_file, format='png', dpi=74)


def mark_span(ax, span):
	ax.axvspan(*span, edgecolor='0.5', facecolor='0.98', linestyle='dotted', linewidth=0.5)

def simplify(paths):
	for p in paths:
		path = p.get_path()
		transform = p.get_transform()
		path = transform.transform_path(path)
		#simplified = list(path.iter_segments(simplify=(800, 600)))
		simplified = list(path.iter_segments(simplify=True))
		print "Original length: %d, simplified length: %d" % (len(path.vertices), len(simplified))

def axis_defaults(ax):
	ax.tick_params(axis='both', which='major', labelsize='small')
	ax.tick_params(axis='both', which='minor', labelsize='small')
	"""for tick in ax.axis.get_major_ticks():
		tick.label.set_fontsize('small')
		# specify integer or one of preset strings, e.g.
		#tick.label.set_fontsize('x-small')
		tick.label.set_rotation('vertical')"""
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
	print 'foo', ax.transAxes.transform((0.0, 1.0))

def rms(data, axis = 0):
	return np.sqrt((data**2).mean(axis))

def db(a, b):
	return 20*np.log10(np.divide(a, b))

def allpass(fc, fs):
	T = 1.0/fs
	w_b = 2*np.pi*fc
	p_d = (1 - np.tan(w_b*T/2)) / (1 + np.tan(w_b*T/2))
	return ([p_d, -1], [1, -p_d])

def rolling_window(a, window):
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


if __name__ == "__main__":
	if len(sys.argv) == 2:
		filename = sys.argv[1]
	else:
		print "Usage: %s filename" % sys.argv[0]
		exit()

	analyze(filename)