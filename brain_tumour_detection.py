import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

prediction_decoded = {0 : 'No Tumour', 1: 'Glioma Tumour', 2: 'Meningioma Tumour', 3: 'Pituitary Tumour'}

st.title('Brain Tumor Detection and Classification')
st.image('1651474032780.jpg')
img = st.file_uploader('Upload your image', type=['jpg'])
model = tf.keras.models.load_model('TUMOR_FINAL_MODEL.h5')

if img is not None:
    st.image(img)

if st.button('Predict'):
    img = Image.open(img)
    img_array = np.array(img)
    img_array = tf.image.resize(img_array, (128, 128)) 
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred = np.argmax(pred, axis=1)
    pred = prediction_decoded[pred[0]]
    st.info(pred)
