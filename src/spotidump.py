# -*- coding: utf8 -*-

import os
import io
import sys
import time
import spotify
import logging
import threading
import numpy as np

from binascii import unhexlify

log = logging.getLogger('pymasvis')

class SpotiDump:
	user_agent = "PyMasVis"
	app_key = unhexlify('015EE684B140D4A942EF4214B4F6D515BA7DAF9DB9C7AD987F8920E8A57CDA697765FE6E104D859C7342063D68D874F8F7866C63084B59E3D1F89A03569CDBC6118B6597BB32A774C2A221399D0365287603B0034DD04A91E16C925F3349E3ED8B08AB5B5989644BBB6A53DBD34ABC437EF6AA6A72149F64CD281DD6C96E47361D26A2BB04607BB9FBC98A9173726CBFC56B9B2D4697FBF3A80C5A2CECB96CAD8DB479E2787197F6B98EEA53D1BA6DD7DFF3E85984391D57C382C72BBEB0B75A2BCBBED04DB9264D5288EB92103B4C54D3F32C2B6F3BFEA5B14126BDEFB66CEFA35FCA88B38265E15403E137877B518AC347410A76A579258453DB32C1049E28C8806DC939CECB4CA4F992F91CDB4F0C83D00458BF3C9E2B646A457D853924F8758BD1F33D798DE2537646885952C3B3D79F14032AA431022BE4D302F34FAEEF32')

	def __init__(self, username, password):
		self.username = username
		self.password = password
		config = spotify.Config()
		config.cache_location = "/tmp/spotidump"
		config.settings_location = "/tmp/spotidump"
		config.application_key = self.app_key
		config.user_agent = self.user_agent
		session = spotify.Session(config=config)
		session.on(spotify.SessionEvent.CONNECTION_STATE_UPDATED, self.on_connection_state_updated)
		session.on(spotify.SessionEvent.END_OF_TRACK, self.on_end_of_track)
		session.on(spotify.SessionEvent.LOGGED_IN, self.on_logged_in)
		session.on(spotify.SessionEvent.MUSIC_DELIVERY, self.on_music_delivery)
		session.on(spotify.SessionEvent.GET_AUDIO_BUFFER_STATS, self.on_get_audio_buffer_stats)
		self.session = session
		self.loop = spotify.EventLoop(self.session)
		self.loop.start()
		self.logged_in = threading.Event()
		self.end_of_track = threading.Event()

	def on_connection_state_updated(self, session):
		if session.connection.state is spotify.ConnectionState.LOGGED_IN:
			self.logged_in.set()

	def on_logged_in(self, session, error_type):
		if error_type is spotify.ErrorType.OK:
			log.debug('Logged in as %s', session.user)
		else:
			log.critical('Login failed: %s', error_type)

	def on_end_of_track(self, session):
		self.end_of_track.set()

	def on_music_delivery(self, session, audio_format, frames, num_frames):
		self.size += audio_format.frame_size() * num_frames
		self.frames += num_frames
		self.channels = audio_format.channels
		self.bitdepth = audio_format.frame_size()*4
		self.sample_rate  = audio_format.sample_rate
		self.buf.write(bytearray(frames))
		if self.count % 100 == 0:
			pos = self.frames / float(self.sample_rate)
			pct = 100.0 * pos / self.duration
			real = time.time() - self.start
			speed = pos / real
			log.debug("\t%3d %%, %5.1f s, %5.1f s, %5.1fx", pct, pos, real, speed)
		self.count += 1
		return num_frames

	def on_get_audio_buffer_stats(self, session):
		return spotify.AudioBufferStats(samples=0, stutter=0)

	def dump(self, uri):
		self.buf = io.BytesIO()
		self.size = 0
		self.frames = 0
		self.channels = 0
		self.bitdepth = 0
		self.sample_rate = 0
		self.count = 0
		self.session.volume_normalization = False
		self.session.preferred_bitrate(spotify.Bitrate.BITRATE_320k)
		self.session.preferred_offline_bitrate(spotify.Bitrate.BITRATE_320k)
		self.login()
		link = self.session.get_link(uri)
		if not link.type is spotify.LinkType.TRACK:
			log.error('Only link type track is supported, got', link.type)
			return 1
		track = link.as_track()
		track.load()
		artists = []
		for artist in track.artists:
			artists.append(artist.name)
		artist = ', '.join(artists)
		title = track.name
		album = track.album.name
		date = track.album.year
		tracknumber = track.index
		name = '%s - %s' % (artist, title)
		self.duration = duration = track.duration / 1000.0
		self.start = time.time()
		log.info('Dumping %s', name)
		self.session.player.prefetch(track)
		self.session.player.load(track)
		self.session.player.play()
		try:
			while not self.end_of_track.wait(0.1):
				pass
		except KeyboardInterrupt:
			pass
		self.session.player.unload()
		self.end_of_track.clear()
		raw_data = np.frombuffer(self.buf.getvalue(), dtype=np.int16).reshape(2, -1, order='F').copy(order='C')
		data = raw_data.astype('float')
		data /= 2**(self.bitdepth-1)
		return {
			'data': {
				'buffer': self.buf,
				'fixed': raw_data,
				'float': data
			},
			'samples': self.frames,
			'samplerate': self.sample_rate,
			'channels': self.channels,
			'bitdepth': self.bitdepth,
			'duration': duration,
			'format': 'vorbis',
			'metadata': {
				'source': 'Spotify',
				'id': track.link.uri.split(':')[-1],
				'filename': name + '.ogg',
				'extension': 'ogg',
				'encoding': 'vorbis',
				'name': name,
				'artist': artist,
				'title': title,
				'album': album,
				'track': tracknumber,
				'date': date,
				'bps': 320000
			}
		}

	def get_tracks(self, uri):
		tracks = []
		self.login()
		link = self.session.get_link(uri)
		log.debug(link)
		if link.type is spotify.LinkType.ALBUM:
			album = link.as_album()
			browser = album.browse()
			browser.load()
			log.debug(browser.tracks)
			for track in browser.tracks:
				tracks.append(track.link.uri)
		elif link.type is spotify.LinkType.PLAYLIST:
			playlist = link.as_playlist()
			playlist.load()
			for track in playlist.tracks:
				tracks.append(track.link.uri)
		else:
			log.warning('Unknown link type', link.type)
		return tracks

	def login(self):
		if self.session.connection.state is not spotify.ConnectionState.LOGGED_IN:
			self.session.login(self.username, self.password)
			self.logged_in.wait()

	def logout(self):
		self.session.logout()


if __name__ == '__main__':
	import optparse
	import wave
	usage = "usage: %prog [options] arg"
	op = optparse.OptionParser(usage)
	op.add_option("-u", "--username", help="Spotify username")
	op.add_option("-p", "--password", help="Spotify password")
	op.add_option("-b", "--bitrate", type="int", help="Spotify bitrate (96, 160 or 320)")
	op.add_option("-v", "--verbose", help="Show debug information",
		dest="verbose", action="store_true")
	(options, args) = op.parse_args()
	if len(args) != 1:
		op.error("Missing spotify link")
	lh = logging.StreamHandler(sys.stdout)
	lh.setFormatter(logging.Formatter("%(message)s"))
	log.addHandler(lh)
	log.setLevel(logging.WARNING)
	if options.verbose:
		log.setLevel(logging.DEBUG)
		#slog = logging.getLogger('spotify')
		#slog.addHandler(lh)
		#slog.setLevel(logging.DEBUG)
	dumper = SpotiDump(options.username, options.password)
	if args[0].startswith('spotify:track:'):
		track = dumper.dump(args[0])
		log.debug(track['metadata'])
	elif args[0].startswith('spotify:album:') or args[0].startswith('spotify:user:'):
		tracks = dumper.get_tracks(args[0])
		log.debug(tracks)
		exit(0)
	try:
		wavfile = wave.open(track['metadata']['name'] + '.wav', 'wb')
		wavfile.setparams((track['channels'], 2, track['samplerate'], track['samples'], 'NONE', 'NONE'))
		wavfile.writeframes(track['data']['buffer'].getvalue())
		wavfile.close()
	except wave.Error as e:
		log.critical(e)
