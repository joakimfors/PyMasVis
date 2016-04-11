# -*- coding: utf-8 -*-

import io
import time
import random
import string
import logging
import multiprocessing
import atexit
import flask
import eventlet
import eventlet.wsgi
import analyze as pymasvis

import resource

from eventlet.green import zmq
#from werkzeug.serving import run_with_reloader
#from werkzeug.debug import DebuggedApplication
from flask import Flask, render_template, session, request, send_file, abort
from flask_socketio import SocketIO, emit, join_room, rooms
from PIL import Image



app = Flask(__name__)
app.debug = True
app.secret_key = 'foobar'
socketio = SocketIO(app, async_mode='eventlet')
pymasvis.log.setLevel(logging.INFO)

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
	#print dir(request)
	#print('/', request.sid)
	print 'Hell that session user:', session
	return render_template('index.html', count=0)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
	global pool
	if request.method == 'POST':
		if 'file' not in request.files:
			abort(500)
		f = request.files['file']
		print f
		infile = f.filename
		print infile
		inbuffer = f.stream.read()
		result = pool.apply_async(render, (infile, inbuffer, session['uid']))
		while not result.ready():
			#gevent.sleep(0.5)
			eventlet.sleep(0.5)
		imgbuf = result.get(timeout=1)
		r = send_file(imgbuf, as_attachment=True, attachment_filename="%s-pymasvis.png" % infile, mimetype='image/png')
		print r
		print dir(r)
		return r


@socketio.on('connect')
def on_connect():
	print('Client connected', request.sid, session)
	emit('event', {'data': 'Connected', 'count': 0})


@socketio.on('disconnect')
def on_disconnect():
	print('Client disconnected', request.sid)


@socketio.on('join')
def on_join(message):
	print('Hej', message)
	if 'room' in message:
		join_room(message['room'])


def cleanup():
	global pool
	pool.close()
	pool.join()


def worker():
	"""Example of how to send server generated events to clients."""
	print "Boom!"
	ctx = zmq.Context()
	sock = ctx.socket(zmq.PULL)
	sock.bind('tcp://127.0.0.1:1234')
	while True:
		print "I'm in a loop!"
		#eventlet.sleep(10)
		msg = sock.recv_pyobj()
		print 'Got:', msg
		#sock.send('OK')
		print "Memory self: %6d KiB, children: %6d KiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024)
		socketio.emit('event', {'data': 'Server generated event', 'message': msg[1]}, room=msg[0])


def render(infile, inbuffer, uid):
	print 'Render req from uid:', uid
	ctx = zmq.Context()
	sock = ctx.socket(zmq.PUSH)
	sock.connect('tcp://127.0.0.1:1234')
	sock.send_pyobj([uid, 'Hello, world!'])
	#sock.recv()
	def callback(event, tid, desc=None, secs=None):
		if event == 'start':
			sock.send_pyobj([uid, desc])
	track = pymasvis.load_file(infile, inbuffer)
	result = pymasvis.analyze(track, callback=callback)
	detailed, overview = pymasvis.render(track, result, track['metadata']['name'], render_overview=False, callback=callback)
	img = Image.open(detailed)
	img = img.convert(mode='P', palette='ADAPTIVE', colors=256)
	imgbuf = io.BytesIO()
	img.save(imgbuf, 'PNG', optimize=True)
	detailed.close()
	img.close()
	imgbuf.seek(0)
	sock.send_pyobj([uid, 'Goodbye, world!'])
	#sock.recv()
	return imgbuf


#@run_with_reloader
def main():
	global pool
	pool = multiprocessing.Pool()
	atexit.register(cleanup)
	#gevent.wsgi.WSGIServer(('', 5000), DebuggedApplication(app)).serve_forever()
	#eventlet.wsgi.server(eventlet.listen(('', 5000)), DebuggedApplication(app))
	socketio.run(app, debug=True)

if __name__ == '__main__':
	main()