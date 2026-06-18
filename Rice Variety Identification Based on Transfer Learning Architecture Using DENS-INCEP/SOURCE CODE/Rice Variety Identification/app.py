#!/usr/bin/env python
from flask import Flask, render_template, request
from markupsafe import Markup
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
 
app = Flask(__name__)

# Class labels
_classes = ['Arborio','Basmati','Ipsala','Jasmine','Karacadag']


# Load models
mobilenet = load_model('MobileNet.h5')
DenseNet121 = load_model('DenseNet121.h5')

def predict_label(img_path):
    test_image = image.load_img(img_path, target_size=(224,224))
    test_image = image.img_to_array(test_image)/255.0
    test_image = test_image.reshape(1, 224, 224, 3)

    predict_x = DenseNet121.predict(test_image) 
    classes_x = np.argmax(predict_x, axis=1)
    return _classes[classes_x[0]]

def predict_labels(img_path):
    test_image = image.load_img(img_path, target_size=(224,224))
    test_image = image.img_to_array(test_image)/255.0
    test_image = test_image.reshape(1, 224, 224, 3)

    predict_x = mobilenet.predict(test_image) 
    classes_x = np.argmax(predict_x, axis=1)
    return _classes[classes_x[0]]

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')
    
@app.route("/login")
def login():
    return render_template('login.html')    

@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files.get('my_image')
        model = request.form.get('model')
        
        # Ensure both image and model were provided
        if img and model:
            img_path = "static/tests/" + img.filename	
            img.save(img_path)

            if model == 'DenseNet121':
                predict_result = predict_label(img_path)
            elif model == 'MobileNet':
                predict_result = predict_labels(img_path)
            else:
                predict_result = "Unknown model selected"
            
            print(predict_result)
            return render_template("prediction.html", prediction=predict_result, img_path=img_path, model=model)
        
        # Handle case if image or model is missing
        return "Image or model selection is missing. Please try again.", 400  # 400: Bad Request
    
    # Fallback if the request method is not POST
    return "Invalid request method. Please submit the form correctly.", 405  # 405: Method Not Allowed

if __name__ == '__main__':
    app.run(debug=True)
