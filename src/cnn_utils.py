# import packages
import numpy as np
import os
import pandas as pd 
import pathlib
import PIL
import PIL.Image
import matplotlib.pyplot  as plt
from matplotlib import rcParams
from sklearn.utils import shuffle

import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator




def set_plot(size):
    
    """Sets style preferences and text sizes for matplotlib plots."""
    
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['axes.grid']=False
    rcParams['xtick.minor.visible']=True
    rcParams['ytick.minor.visible']=True
    rcParams['xtick.direction']='in'
    rcParams['ytick.direction']='in'

    plt.rc('axes', titlesize=size)  # fontsize of the axes title
    plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size*0.8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size*0.8)    # fontsize of the tick labels
    
    
def balance_df(df, label='label', class_size=1000):
    
    """Resamples data frame containing class labels so that every class has an equal class size. 
    Classes are sampled with replacement if they exceed the desired class size, and without 
    replacement if they do not.
        
    Inputs
    - df : name of dataframe containing sample data. 
    - label : name of column containing one-hot encoded class labels.
    - class_size : desired size of each class after resampling.

    Returns
    balanced_df - a dataframe with balanced classes."""
    
    balanced_df = pd.DataFrame()
    n_classes = df[label].nunique()
    
    for i in range(n_classes):
        one_class = df[df[label] == i]
        if len(one_class) >= 2000:
            replace=False
        else:
            replace=True
        
        idx = np.random.choice(df[df[label] == i].index, size=class_size, replace=replace)
        temp = df.iloc[idx]
        balanced_df = pd.concat([balanced_df, temp])

    balanced_df = balanced_df.sample(frac=1).reset_index().rename(columns={'index':'old_index'})

    return balanced_df


def plot_batch(dfiterator, label_key=None, cutmix=False):
    
    """Plots the next batch of images and labels in a keras dataframe iterator.
    Inputs:
    -dfiterator: keras dataframe iterator
    -label_key: series or dictionary such that label_key[image_label] returns a class string.
    -cutmix: whether or not the generator is a cutmix generator."""
    
    bs = dfiterator.batch_size
    images, labels = next(dfiterator)
    if cutmix:
        cols = 3
    else:
        cols = 4
    rows = int(bs / cols) + int(bs % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = axes.flatten()
    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img)
        axes[i].axis('off')
        
        if label_key is not None:
            if cutmix:
                max_class = np.argsort(label)[-1]
                second_class = np.argsort(label)[-2]
                max_pct = round(label[max_class] * 100)
                second_pct = round(label[second_class] * 100)
                title = label_key[max_class]+ ' / ' + str(max_pct) + '%\n' + label_key[second_class] + ' / ' + str(second_pct) + '%'
            else:
                title = label_key[np.argmax(label)]
        axes[i].set_title(title)