import io
from json import dumps

import cv2
import torch
import numpy as np
from flask import Flask, Response, send_file, request
from api.model import Covid19Net
from api.model import heatmap
from PIL import Image

app = Flask(__name__)
model = Covid19Net.load_model('res/model.pth.tar', device=torch.device('cpu'))


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
    # _bytes = request.get_data()
    # image = encode_image(_bytes)
    # TODO dummy image
    image = Image.open('res/image.jpeg').convert('RGB')
    prediction = Covid19Net.predict(model, image)

    # TODO here get heatmap
    # img_out =  h_map.generate()
    # _, img_encoded = cv2.imencode('.jpg', image)

    return Response(dumps({'prediction': prediction}), status=200, mimetype='application/json')


if __name__ == '__main__':
    # Set debug=False and threaded=False otherwise the model throws exceptions
    # Set debug=True for developing the API without using the model
    app.run(debug=True, threaded=False, host='localhost', port='8080')