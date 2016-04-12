# -*- coding: utf-8 -*-

import io
import time
import random
import string
import logging
import resource
import multiprocessing
import atexit
import flask
import eventlet
import eventlet.wsgi
import analyze as pymasvis

from eventlet.green import zmq
from flask import Flask, render_template, session, request, send_file, abort, flash
from flask_socketio import SocketIO, emit, join_room, rooms
from PIL import Image


app = Flask(__name__)
app.secret_key = 'Oo1Shu*nae|daisi7kaePh#ae7ef6ooG'
socketio = SocketIO(app, async_mode='eventlet')
pool = None
thread = None


@app.before_first_request
def init():
	global thread
	if not thread:
		thread = eventlet.spawn(worker)


@app.before_request
def before_request():
	if 'uid' not in session:
		session['uid'] = ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(32)])


@app.route('/')
def index():
	return render_template('index.html', debug=app.debug)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
	global pool
	print 'XHR?', request.is_xhr
	if request.method == 'POST':
		try:
			f = request.files['file']
		except KeyError as e:
			flash('Missing file parameter')
			print e
			abort(500)
		if not f:
			flash('No file found in request')
			print f
			abort(500)
		infile = f.filename
		inbuffer = f.stream.read()
		result = pool.apply_async(render, (infile, inbuffer, session['uid']))
		while not result.ready():
			eventlet.sleep(0.5)
		imgbuf = result.get(timeout=1)
		if type(imgbuf) is not io.BytesIO:
			flash('Failed to analyze file')
			abort(500)
		return send_file(imgbuf, as_attachment=True, attachment_filename="%s-pymasvis.png" % infile, mimetype='image/png')


@app.route('/status')
def status():
	result = "Memory self: %6d KiB, children: %6d KiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024)
	print result
	return result, 200


@socketio.on('connect')
def on_connect():
	print('Client connected', request.sid, session)
	emit('event', {'data': 'Connected', 'count': 0})


@socketio.on('disconnect')
def on_disconnect():
	print('Client disconnected', request.sid)


@socketio.on('join')
def on_join(message):
	if 'room' in message:
		join_room(message['room'])


def pool_cleaner():
	global pool
	pool.close()
	pool.join()


def worker():
	ctx = zmq.Context()
	sock = ctx.socket(zmq.PULL)
	sock.bind('tcp://127.0.0.1:64646')
	while True:
		msg = sock.recv_pyobj()
		socketio.emit('progress', {'message': msg[1], 'status': msg[2]}, room=msg[0])


def render(infile, inbuffer, uid):
	print 'Render req from uid:', uid
	ctx = zmq.Context()
	sock = ctx.socket(zmq.PUSH)
	sock.connect('tcp://127.0.0.1:64646')
	sock.send_pyobj([uid, 'Analysing...', 'start'])
	def callback(event, tid, desc=None, secs=None):
		if event == 'start':
			sock.send_pyobj([uid, desc, 'inprogress'])
	track = pymasvis.load_file(infile, inbuffer)
	if type(track) is int:
		sock.send_pyobj([uid, 'Failed to find audiostream in file', 'finished'])
		eventlet.sleep(2)
		return 0
	result = pymasvis.analyze(track, callback=callback)
	detailed, overview = pymasvis.render(track, result, track['metadata']['name'], render_overview=False, callback=callback)
	img = Image.open(detailed)
	img = img.convert(mode='P', palette='ADAPTIVE', colors=256)
	imgbuf = io.BytesIO()
	img.save(imgbuf, 'PNG', optimize=True)
	detailed.close()
	img.close()
	imgbuf.seek(0)
	sock.send_pyobj([uid, 'Done!', 'finished'])
	return imgbuf


def main(host=None, port=None):
	global pool
	pool = multiprocessing.Pool()
	atexit.register(pool_cleaner)
	socketio.run(app, host=host, port=port, debug=app.debug)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="Web app for PyMasVis")
	parser.add_argument('-d', '--debug', action='store_true', help="debug info")
	parser.add_argument('--host', type=str, default=None, help="listen address")
	parser.add_argument('--port', type=int, default=None, help="listen port")
	args = parser.parse_args()
	if args.debug:
		app.debug = True
		pymasvis.log.setLevel(logging.INFO)
	else:
		app.debug = False
	main(args.host, args.port)
