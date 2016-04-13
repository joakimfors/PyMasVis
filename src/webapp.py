# -*- coding: utf-8 -*-

import io
import time
import random
import string
import pprint
import logging
import resource
import multiprocessing
import atexit
import flask
import eventlet
import eventlet.wsgi
import analyze as pymasvis

from eventlet.green import zmq
from flask import Flask, Markup, render_template, session, request, send_file, abort, flash, jsonify, url_for, redirect
from werkzeug.exceptions import HTTPException
from flask_socketio import SocketIO, emit, join_room, rooms
from PIL import Image


app = Flask(__name__)
app.secret_key = 'Oo1Shu*nae|daisi7kaePh#ae7ef6ooG'
app.config['TRAP_HTTP_EXCEPTIONS'] = True
socketio = SocketIO(app, async_mode='eventlet')
pool = None
thread = None
storage = {}


@app.before_first_request
def init():
	global thread
	if not thread:
		thread = eventlet.spawn(msg_router)


@app.before_request
def before_request():
	if 'uid' not in session:
		session['uid'] = rand_str(32)


@app.errorhandler(HTTPException)
def error_handler(error):
	app.logger.debug('Exception code: %d, name: %s, description: %s, message: %s', error.code, error.name, error.description, error.message)
	if request.is_xhr:
		response = jsonify(error=error.description, code=error.code, name=error.name, flash=flask.get_flashed_messages())
		response.status_code = error.code
		return response
	return render_template('error.html', error=error)


@app.route('/test')
def test():
	return redirect(url_for('index'))


@app.route('/')
def index():
	links = []
	for imgid, img in storage.iteritems():
		if img['uid'] == session['uid']:
			links.append('<a href="%s">%s</a> ' % (img['imgurl'], img['imgname']))
	if links:
		flash(Markup("You have the following images available for download: %s" % ', '.join(links)))
	return render_template('index.html', debug=app.debug)


@app.route('/analyze', defaults={'imgid': None}, methods=['POST'])
@app.route('/analyze/<imgid>', methods=['GET'])
def analyze(imgid):
	global pool
	if request.method == 'GET':
		img = storage.pop(imgid, None)
		if not img:
			flash('Analysis already downloaded or removed')
			app.logger.error(str(e))
			abort(410)
	elif request.method == 'POST':
		try:
			f = request.files['file']
		except KeyError as e:
			flash('Missing file parameter')
			app.logger.error(str(e))
			abort(400)
		if not f:
			flash('No file found in request')
			app.logger.error(str(f))
			abort(400)
		socketio.emit('progress', {'message': 'Uploaded...', 'status': 'inprogress'}, room=session['uid'])
		infile = f.filename
		inbuffer = f.stream.read()
		imgid = rand_str(8)
		imgname = "%s-pymasvis.png" % infile
		imgurl = url_for('analyze', imgid=imgid)
		if request.is_xhr:
			def callback(result):
				if 'error' not in result:
					storage[result['imgid']] = result
		else:
			callback = None
		result = pool.apply_async(process, (infile, inbuffer, imgid, imgname, imgurl, session['uid']), callback=callback)
		if request.is_xhr:
			return jsonify({
				'message': 'OK'
			})
		while not result.ready():
			eventlet.sleep(0.5)
		img = result.get()
		if 'error' in img:
			flash(img['error'])
			abort(415)
	return send_file(img['imgbuf'], as_attachment=True, attachment_filename=img['imgname'], mimetype='image/png')


@app.route('/status')
def status():
	result = "Memory self: %6d KiB, children: %6d KiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024)
	app.logger.info(result)
	return result, 200


@app.route('/stored')
def pawned():
	return pprint.pformat(storage, depth=2)


@socketio.on('connect')
def on_connect():
	app.logger.info('Client connected with sid: %s', request.sid)
	emit('event', {'message': 'Connected'})


@socketio.on('disconnect')
def on_disconnect():
	app.logger.info('Client disconnected with sid: %s', request.sid)


@socketio.on('join')
def on_join(message):
	if 'room' in message:
		join_room(message['room'])


def rand_str(length):
	return ''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(length)])


def msg_router():
	ctx = zmq.Context()
	sock = ctx.socket(zmq.PULL)
	sock.bind('tcp://127.0.0.1:64646')
	while True:
		msg = sock.recv_pyobj()
		uid = msg.pop('uid')
		socketio.emit('progress', msg, room=uid)


def process(infile, inbuffer, imgid, imgname, imgurl, uid):
	app.logger.info('Render req from uid: %s', uid)
	ctx = zmq.Context()
	sock = ctx.socket(zmq.PUSH)
	sock.connect('tcp://127.0.0.1:64646')
	sock.send_pyobj({
		'uid': uid,
		'message': 'Reading...',
		'status': 'inprogress'
	})
	def callback(event, tid, desc=None, secs=None):
		if event == 'start':
			sock.send_pyobj({
				'uid': uid,
				'message': desc,
				'status': 'inprogress'
			})
	track = pymasvis.load_file(infile, inbuffer)
	if type(track) is int:
		sock.send_pyobj({
			'uid': uid,
			'message': 'Failed to find audio in file',
			'status': 'error'
		})
		eventlet.sleep(2)
		return {
			'imgid': imgid,
			'error': 'Failed to find audio in file'
		}
	sock.send_pyobj({
		'uid': uid,
		'message': 'Analysing...',
		'status': 'inprogress'
	})
	result = pymasvis.analyze(track, callback=callback)
	detailed, overview = pymasvis.render(track, result, track['metadata']['name'], render_overview=False, callback=callback)
	img = Image.open(detailed)
	img = img.convert(mode='P', palette='ADAPTIVE', colors=256)
	imgbuf = io.BytesIO()
	img.save(imgbuf, 'PNG', optimize=True)
	detailed.close()
	img.close()
	imgbuf.seek(0)
	sock.send_pyobj({
		'uid': uid,
		'message': 'Done!',
		'url': imgurl,
		'status': 'finished'
	})
	sock.close()
	return {
		'imgid': imgid,
		'imgbuf': imgbuf,
		'imgname': imgname,
		'imgurl': imgurl,
		'uid': uid,
		'ts': time.time()
	}


def killer():
	global pool
	#global thread
	#if thread:
	#	thread.kill()
	if pool:
		pool.terminate()
		pool.join()


def main(host=None, port=None):
	global pool
	pool = multiprocessing.Pool(maxtasksperchild=10)
	atexit.register(killer)
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
		multiprocessing.log_to_stderr().setLevel(multiprocessing.SUBDEBUG)
	else:
		app.debug = False
	main(args.host, args.port)
