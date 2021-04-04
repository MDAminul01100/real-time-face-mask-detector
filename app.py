import cv2
from flask import Flask, render_template, request, redirect, Response
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)


class ApplicationHandler(Resource):
    def get(self):
        return 'bla bla'


api.add_resource(ApplicationHandler, "/applicationHandler")


@app.route('/', methods=['GET', 'POSt'])
def home():
    return render_template('index.html')


@app.route('/result', methods=['POSt'])
def result():
    flexRadioDefault = request.form['flexRadioDefault']
    if flexRadioDefault == 'realTimeRadioChecked':
        hello_world = 'hello world'
    return render_template('result.html', flexRadioDefault=flexRadioDefault)


def realTimeGenerator():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen():
    img = cv2.imread("resources/resources_for_testing/test2.jpg")
    img = cv2.resize(img, (0, 0), fx=1, fy=1)
    frame = cv2.imencode('.jpg', img)[1].tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
