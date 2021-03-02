from flask import Flask, render_template, request, redirect, flash, url_for

import urllib.request
#from app import app
from werkzeug.utils import secure_filename

import os

import sys
sys.path.append('../src/')
from model import *
from inference import *


UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load model and label key
label_key = pd.read_json('../data/label_num_to_disease_map.json', typ='series')
model = EfficientNetB4Model(load_fp='../models/cutmix-efficientNetB4.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            label, acc = predict_img(model=model, label_key=label_key, img_file='uploads/' + filename)
            flash(label)
            flash(acc)
            flash(filename)
            return redirect('/')


if __name__ == "__main__":
    app.run(debug=True, port=8000)