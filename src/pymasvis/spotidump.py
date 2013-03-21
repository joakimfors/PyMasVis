# -*- coding: utf8 -*-

import cmd
import logging
import os
import sys
import threading
import time
import numpy as np

from io import BytesIO, StringIO
from binascii import unhexlify
from spotify import Link, SpotifyError
from spotify.manager import SpotifySessionManager

class DumpThread(threading.Thread):
    def __init__(self, manager):
        threading.Thread.__init__(self)
        self.manager = manager

    def run(self):
        print "Hello from DumpThread... dumping"
        self.manager.dump_track()

class DumpManager(SpotifySessionManager):
    cache_location = '/tmp/spotidump'
    settings_location = '/tmp/spotidump'
    #appkey_file = 'spotify_appkey.key'
    application_key = unhexlify('015EE684B140D4A942EF4214B4F6D515BA7DAF9DB9C7AD987F8920E8A57CDA697765FE6E104D859C7342063D68D874F8F7866C63084B59E3D1F89A03569CDBC6118B6597BB32A774C2A221399D0365287603B0034DD04A91E16C925F3349E3ED8B08AB5B5989644BBB6A53DBD34ABC437EF6AA6A72149F64CD281DD6C96E47361D26A2BB04607BB9FBC98A9173726CBFC56B9B2D4697FBF3A80C5A2CECB96CAD8DB479E2787197F6B98EEA53D1BA6DD7DFF3E85984391D57C382C72BBEB0B75A2BCBBED04DB9264D5288EB92103B4C54D3F32C2B6F3BFEA5B14126BDEFB66CEFA35FCA88B38265E15403E137877B518AC347410A76A579258453DB32C1049E28C8806DC939CECB4CA4F992F91CDB4F0C83D00458BF3C9E2B646A457D853924F8758BD1F33D798DE2537646885952C3B3D79F14032AA431022BE4D302F34FAEEF32')
    user_agent = 'PyMasVis'

    spotify_link = ''
    bitrate = 1
    trackname = "NA"
    filename = "dump.pcm"
    buf = None
    size = 0
    frames = 0
    count = 0


    def __init__(self, username, password, spotify_link, bitrate=320):
        SpotifySessionManager.__init__(self, username, password)
        self.spotify_link = spotify_link
        if bitrate in [160, 320, 96]:
            self.bitrate = [160, 320, 96].index(bitrate)
        self.thread = DumpThread(self)

    def logged_in(self, session, error):
        if error:
            print error
            return
        print 'Logged in'
        session.set_preferred_bitrate(self.bitrate)
        if not self.thread.is_alive():
            self.thread.start()
        #self.dump_track()

    def dump_track(self):
        line = self.spotify_link
        try:
            if line.startswith("spotify:"):
                # spotify url
                l = Link.from_string(line)
                if not l.type() == Link.LINK_TRACK:
                    print "You can only play tracks!"
                    return
                self.load_track(l.as_track())
            else:
                print "Please provide a spotify link"
                return
        except SpotifyError as e:
            print "Unable to load track:", e
            return
        self.filepath = os.path.join(self.cache_location, self.trackname + ".pcm_" + str(self.bitrate))
        print "Dumping frames"
        #exit(1)
        self.buf = BytesIO() #open(self.filepath, 'w')
        self.session.play(1)

    def load_track(self, track):
        print u"Loading track..."
        while not track.is_loaded():
            time.sleep(0.1)
        if track.is_autolinked(): # if linked, load the target track instead
            print "Autolinked track, loading the linked-to track"
            return self.load_track(track.playable())
        if track.availability() != 1:
            print "Track not available (%s)" % track.availability()
        self.session.load(track)
        self.trackname = "%s - %s" % (track.artists()[0].name(), track.name())
        print "Loaded track: %s" % self.trackname

    def music_delivery_safe(self, session, frames, frame_size, num_frames, sample_type, sample_rate, channels):
        #print frame_size, num_frames, sample_type, sample_rate, channels
        #print str(type(frames))
        #self.end_of_track(session)
        self.size += frame_size * num_frames
        self.frames += num_frames
        self.channels = channels
        self.bitdepth = frame_size*4
        self.sample_rate  = sample_rate
        self.buf.write(bytearray(frames))
        if self.count % 100 == 0:
            print "%.2f s" % (self.frames / (sample_rate * 1.0))
        self.count += 1
        return num_frames

    def end_of_track(self, session):
        #self.output.seek(0)
        #npa = np.frombuffer(self.output.getvalue(), dtype=np.int16)
        #print "Nupy array shape and size ", npa.shape, npa.size, npa.itemsize
        #real_file = open(self.filepath, 'w')
        #real_file.write(self.output.getvalue())
        #real_file.close()
        time.sleep(0.1)
        self.session.play(0)
        #print ""
        #print "Wrote %d bytes to %s" % (self.size, self.filepath)
        #print "ch: %d, bits: %d, fs: %d" % (self.channels, self.bitdepth, self.sample_rate)
        self.session.logout()
        self.disconnect()

    def get_nparray(self):
        ar = np.frombuffer(self.buf.getvalue(), dtype=np.int16)
        return ar.reshape(2, -1, order='F')

    def get_data(self):
        return self.buf.getvalue()

    def dump(self):
        self.connect()


if __name__ == '__main__':
    import optparse
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
    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    dumper = DumpManager(options.username, options.password, args[0], options.bitrate)
    dumper.dump()
    print "DONE DUMPING!"