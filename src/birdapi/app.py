from flask import Flask, render_template, request, redirect, url_for, make_response
import numpy as np
import keras.models
from keras.preprocessing import image
import re
import sys
import os
import base64
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def init():
    model = load_model("model/bird_pred.h5")
    return model

global model, filtered_classes
model = init()
filtered_classes = ['barswa', 'cohmar1', 'combuz1', 'comsan', 'eaywag1', 'eubeat1', 'litegr', 'thrnig1', 'wlwwar', 'woosan']

app = Flask(__name__)

@app.route('/')
def index_view():
    return redirect(url_for('static', filename='index.html'))

def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/prediction', methods=['GET'])
def prediction():
    img_width, img_height = 884, 322
    filename=request.args.get('filename')
    x = image.load_img('static/content/' + filename, target_size=(img_height, img_width))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x /= 255.

    pred = model.predict(x)
    #response = np.array_str(np.argmax(pred,  axis=1))
    #print(response, flush=True)
    #print(filtered_classes[np.argmax(pred)], flush=True)
    return filtered_classes[np.argmax(pred)]
