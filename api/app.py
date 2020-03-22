import io
from json import dumps

import cv2
import numpy as np
from flask import Flask, Response, send_file, request
from .model import Covid19Net
from .model import heatmap

app = Flask(__name__)
model = Covid19Net.load_model()
h_map = heatmap.HeatmapGenerator()


@app.route('/')
def hello_world():
    return 'Hello, World!'


def encode_image(_bytes):
    """
    Converts the image into a numpy array for further processing by the model.
    :param _bytes:
    :return: 3D ndarray containing the image
    """
    ndarray = np.fromstring(_bytes, np.uint8)
    return cv2.imdecode(ndarray, cv2.IMREAD_COLOR)


@app.route('/api/v1/classify/', methods=['POST'])
def classify():
    """
    Classifies objects detected in the provided image.
    :return: Class score.
    """
    if request.content_length > 4194304:
        return Response(dumps({'error': 'Exceeded maximal content size'}), status=413,
                        mimetype='application/json')
    try:
        _bytes = request.get_data()
        image = encode_image(_bytes)
        data_loader = model.load_image(image)
        y_pred = Covid19Net.predict(data_loader, model)
        # TODO here get heatmap
        img_out =  h_map.generate()
        _, img_encoded = cv2.imencode('.jpg', image)

        # Return json with b64 or bytes encoded image and probabilities when possible
        # Otherwise we need individual endpoint for heatmaps and prediction
        # return send_file(io.BytesIO(img_encoded), as_attachment=True, attachment_filename='image_detected.jpg',
        #                  mimetype='image/jpg')

    return Response(dumps({'prediction': 99.7}), status=200, mimetype='application/json')


if __name__ == '__main__':
    # Set debug=False and threaded=False otherwise the model throws exceptions
    # Set debug=True for developing the API without using the model
    app.run(debug=True, threaded=False, host='localhost', port='8080')
