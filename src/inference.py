import numpy as np
from pandas import read_json
from seaborn import heatmap
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import *



def predict_img(model=None, label_key=None, img_file=None):
    
    """Returns the prediction and probability for a single image given the image filepath"""
    if img_file is None:
        img_file = input('Please enter an image filepath: ')
    img = load_img(img_file, target_size=(model.img_size,model.img_size))
    input_arr = np.array([img_to_array(img) / 255])
    if model is None:
        model = EfficientNetB4Model(load_fp='../models/cutmix-efficientNetB4.h5')
    if label_key is None:
        label_key = pd.read_json('../data/label_num_to_disease_map.json', typ='series')
    probs = model.model.predict(input_arr)
    max_prob = np.round(np.max(probs * 100))
    pred = label_key[np.argmax(probs)]
        
    print('Predicted Class:', pred, '\t//   Model Probability:', max_prob)
    
    return str(pred), str(max_prob)



if __name__ == '__main__':
    predict_img()
