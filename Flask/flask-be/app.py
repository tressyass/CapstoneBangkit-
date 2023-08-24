from PIL import Image
from keras.models import load_model
from keras_facenet import FaceNet
import numpy as np
from numpy import asarray
from numpy import expand_dims
import pickle
import cv2
import os
from io import BytesIO

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

from gevent.pywsgi import WSGIServer
from dotenv import dotenv_values

app = Flask(__name__)

env_file = '.env'
if os.path.exists(env_file):
    config = dotenv_values('.env')
    isEnvFile = True
else:
    isEnvFile = False

# get config
def getConfig(key):
    if (isEnvFile == True):
        return config[key]
    else:
        return os.environ.get(key)

# get app config
appPort = getConfig('APP_PORT')
isDevelopment = eval(getConfig('IS_DEVELOPMENT').capitalize())

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

model = open("dataset.pkl", "rb")
database = pickle.load(model)
model.close()

# ===========================================
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the 'file' parameter exists in the request
    if 'image' not in request.files:
        return 'No file part in the request', 400

    file = request.files['image']

    # Check if a file was uploaded
    if file.filename == '':
        return 'No file selected', 400

    # Validate if the uploaded file is an image
    try:
        image_data = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_ANYCOLOR)
        face = HaarCascade.detectMultiScale(image,1.1,4)
        
        if len(face)>0:
            x1,y1,w,h = face[0]
        else:
            x1,y1,w,h = 1,1,10,10
                
        x1,y1 = abs(x1), abs(y1)
        x2,y2 = x1 + w, y1 + h
            
        wajah = image[y1:y2, x1:x2]
        wajah = Image.fromarray(wajah)
        wajah = wajah.resize((160,160))
        wajah = asarray(wajah)
            
        wajahTs = expand_dims(wajah, axis=0)
        signature = MyFaceNet.embeddings(wajahTs)
        
        min_dist = 100
        identity = ' '
        for key, value in database.items():
            dist = np.linalg.norm(value-signature)
            if dist < min_dist:
                min_dist = dist
                identity = key
        
        return jsonify({
            'success': True,
            'identity': identity
        });

    except: 
        return 'Internal server error', 500

@app.route('/', methods=['GET'])
def welcome():
    return "Hello World!"

if __name__ == '__main__':
    if(isDevelopment == True):
        app.run(host='0.0.0.0', port=int(appPort), debug=True)
    else:
        http_server = WSGIServer(('', int(appPort)), app)
        http_server.serve_forever()
