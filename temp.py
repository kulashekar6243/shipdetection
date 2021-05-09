import streamlit as st
st.title("Image Classification")
st.header("Boat classification")
st.text("Upload a boat for image classification as type of boat floating")

import tensorflow.keras as keras

from PIL import Image, ImageOps
import numpy as np

boats = {0:"Buoy",1:"cruise",2:"sail boat",
         3:"inflatable boart",4:"freight boat",
         5:"kayak"}

def teachable_machine_classification(img, weights_file):
    # Load the model
   
    model = keras.models.load_model(weights_file)



    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

uploaded_file = st.file_uploader("Choose a boat image...", type="jpg")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded boat image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'keras_model.h5')
        st.write(boats[label])
        













