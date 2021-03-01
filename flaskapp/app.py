from flask import Flask

UPLOAD_FOLDER = '/Users/melanierbutler/Desktop/Springboard/Github/cassava/flaskapp/uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
