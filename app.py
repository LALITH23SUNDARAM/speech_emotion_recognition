"""Create an Audio Classification Web App using  Streamlit."""
# import libraries
import numpy as np
from numpy import newaxis
from tranformation import predict, transformation,plt,normalizer,expand_dimension
import streamlit as st
import keras
from PIL import Image
import librosa
from librosa import display
# set title of app
st.set_page_config(page_title="my app",layout="wide")


st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>Emotion Recognition Engine</h1>", unsafe_allow_html=True)
image = Image.open("image.png")
st.image(image, use_column_width=True)


# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an audio", type = ["wav","mp3"])
try:
    y, sr = librosa.load(librosa.ex('trumpet'))
    length = librosa.get_duration(y=y, sr=sr)
except:
    print('oops !.Error occured')

def length_(file_up):
    if length > 5.0:
        st.write('file should be less than 5 sec')

st.write('Model takes audio files for 5 sec, if files larger its truncated to 5 sec, if smaller it padded by silence')

if file_up is not None:

    if st.button("Recognize"):
        try: 
            features = transformation(file_up)
            normalized_features = normalizer(features)
            expanded_features = expand_dimension(normalized_features)
            emotions = predict(expanded_features)
            st.write(f'{emotions}')
        except:
            st.write('oops !. Error Occurred..')

    if st.button('Get waveform'):
        fig = plt.figure(figsize=(3,1 ))
        data, sample_rate = librosa.load(file_up)
        librosa.display.waveplot(data, sr= sample_rate)
        plt.title('waveform')
        st.pyplot(fig)