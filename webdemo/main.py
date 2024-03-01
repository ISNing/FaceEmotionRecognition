from flask import Flask, render_template, Response
import cv2
import atexit
from inference import Classifier
from camera import PiCamera
from face_detect import FaceDetector

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 0  # pixels
_TOP_MARGIN = 2  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1.5
_FONT_THICKNESS = 2
_FPS_AVERAGE_FRAME_COUNT = 10

camera = PiCamera()
camera.start()
faceDetector = FaceDetector()
classifier = Classifier("../pretrained/lite_fer_model_efficientnetv2-b0_metadata.tflite", 3, 0.0, 4)

app = Flask(__name__)


@atexit.register
def cleanup():
    camera.close()


def gen_frames():
    while True:
        try:
            # Capture image from the camera and run inference
            image = camera.capture()
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for (x, y, w, h) in faceDetector.detect(image):
                face = image[y:y + h, x:x + w]
                cv2.rectangle(image, (x, y), (x + w, y + h), _TEXT_COLOR, 2)

                categories = classifier.inference(face)
                # Show classification results on the image
                for idx, category in enumerate(categories.classifications[0].categories):
                    category_name = category.category_name
                    score = round(category.score, 2)
                    result_text = category_name + ' (' + str(score) + ')'
                    text_location = (_LEFT_MARGIN + x, _TOP_MARGIN + (idx + 1) * _ROW_SIZE + y + h)
                    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

            ret, buffer = cv2.imencode('.jpg', image)
            if ret:
                image = buffer.tobytes()
                yield (b'--frame\r\n' +
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print(f'Exception generated: {e}')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)
