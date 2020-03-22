import base64
from json import dumps

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, Response, request
from flask_cors import CORS, cross_origin

from api.model import Covid19Net

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = Covid19Net.load_model('res/model.pth.tar', device=torch.device('cpu'))

_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224)
])


def decode_image(_bytes):
    """
    Converts byte array into image for further processing by the model.
    :param _bytes: byte array
    :return: PIL Image
    """
    ndarray = np.fromstring(_bytes, np.uint8)
    ndarray = cv2.imdecode(ndarray, cv2.IMREAD_COLOR)
    image = Image.fromarray(ndarray)
    return image


def image_array_to_base64(image_array):
    """
    :param image_array: Numpy array (height, width, channels)
    :return: base64 string
    """
    _, encoded_image = cv2.imencode('.jpg', image_array)
    return base64.b64encode(encoded_image).decode('utf-8')


def generate_heatmap_image(image):
    """
    :param image: PIL Image
    :return: Numpy array (height, width, channels)
    """
    # resize and center crop the input image
    image = _transform(image)
    # generate heatmap using the network
    heatmap = Covid19Net.generate_heatmap(model, image).numpy()
    # normalize heatmap
    heatmap = heatmap / np.max(heatmap)
    # resize to match input image dimensions
    heatmap = cv2.resize(heatmap, (224, 224))
    # apply color map to the heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    # apply heatmap to the input image
    image_array = np.array(image)
    heatmap_image = cv2.addWeighted(image_array, 1, heatmap, 0.35, 0)
    heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
    return heatmap_image


@app.route('/api/v1/classify/', methods=['POST'])
@cross_origin()
def classify():
    """
    Classifies objects detected in the provided image.
    :return: Class score.
    """
    bytes_ = request.files['file'].stream.read()
    image = decode_image(bytes_)
    prediction = Covid19Net.predict(model, image)
    heatmap_image = generate_heatmap_image(image)
    heatmap_image_base64 = image_array_to_base64(heatmap_image)
    response_json = dumps({'prediction': prediction,
                           'heatmap': heatmap_image_base64})
    return Response(response_json, status=200, mimetype='application/json')


if __name__ == '__main__':
    # Set debug=False and threaded=False otherwise the model throws exceptions
    # Set debug=True for developing the API without using the model
    app.run(debug=True, threaded=False, host='localhost', port='8000')
