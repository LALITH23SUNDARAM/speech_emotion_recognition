from librosa import display
import librosa
import numpy as np
from numpy import newaxis
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import pickle

# load json and create model
json_file = open('saved_model_object\LSTM_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_model_object\LSTM_model_weights.h5")

# load train mean
with open('saved_model_object/train_mean', 'rb') as fp:
    train_mean = pickle.load(fp)

# load train variance
with open('saved_model_object/train_var', 'rb') as fp1:
    train_variance = pickle.load(fp1)


def transformation(file_up):
    # extract mfcc
    #This function extracts mfcc features and obtain the mean of each dimension
    #Input : path_to_wav_file
    #Output: mfcc_features'''
    y, sr = librosa.load(file_up,duration=3
                                  ,offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    
    return mfccs

def normalizer(mfcc):
    numerator = mfcc - train_mean
    denominator =  train_variance
    z = numerator / denominator
    return z

def expand_dimension(z):
    a = z.reshape(1,-1)
    b = a[:, :, newaxis]
    return b


# lets predict 
def predict(z):
    #Making prediction
    y_pred=loaded_model.predict(z)
    y_pred = np.argmax(y_pred)
    class_names = ['Men in angry', 'Men in calm', 'Men in disgust', 'Men in fearful', 'Men_happy',
                    'Men in neutral', 'Men in sad', 'Men in surprised', 'Women in angry',
                    'Women in calm', 'Women in disgust', 'Women in fearful', 'Women in happy',
                    'Women in neutral', 'Women in sad', 'Women in surprised']
    output = class_names[y_pred]
    return output





    
