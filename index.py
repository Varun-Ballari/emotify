import sys, os
import logging
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit, join_room, disconnect

from cv.predictor import Predictor
from utils import Queue

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.config['DEBUG'] = True
socketio = SocketIO(app)
cvSocket = None
predictor = Predictor()
queue = Queue(predictor)

trueEmotion = None


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    queue.enqueue_input(input)


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")
    emit('my response', {'data': 'Connected', 'count': 0})


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


def _change_song(emotion, user):
    global trueEmotion

    if emotion is None:
        return

    if trueEmotion == emotion:
        return

    trueEmotion = emotion

    if emotion == 'happy':
        song = 'MOWDb2TBYDg'
    elif emotion == 'sad':
        song = 'd-diB65scQU'
    elif emotion == 'angry':
        song = 'Zv479MCnThA'
    elif emotion == 'disgust':
        song = 'jofNR_WkoCE'
    elif emotion == 'fear':
        song = '4V90AmXnguw'
    elif emotion == 'surprise':
        song = 'gkBvuhBBhvA'
    elif emotion == 'neutral':
        song = 'ymHBUyui_ws'
    else:
        song = 'ymHBUyui_ws'

    socketio.emit('message', {'emotion': emotion, 'user': user, 'song': song}, namespace='/test')


def gen():
    """Video streaming generator function."""

    app.logger.info("starting to generate frames!")
    while True:
        output_img, emotion, user = queue.dequeue()
        _change_song(emotion, user)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + output_img + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        socketio.run(app)
    except KeyboardInterrupt:
        os._exit(1)
