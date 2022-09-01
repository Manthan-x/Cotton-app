import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


model =load_model("model/model.h5")

st.image('static\images\h.png')

st.title('Cotton Plant Diesease Model')

user_upload = st.file_uploader('')
 

if st.button('Upload'):
    test_image = load_img(user_upload, target_size = (150, 150))
    test_image = img_to_array(test_image)/255
    test_image = np.expand_dims(test_image, axis = 0) 
    result = model.predict(test_image).round(3) 
    pred = np.argmax(result) 
    if pred == 0:
     st.header(' " Burned Cotton Plant " ')
     st.header(' : - Although the chemical fertilizer has fallen on the leaves So the leaves are burnt .')
    elif pred == 1:
     st.header(' " Diseased Cotton Plant " ')
     st.header(' : - It Happen Due to Attack of Leaf Sucking and Chewing Pests.')
    elif pred == 2:
     st.header(' " Healthy Cotton Leaf " :' )
     st.header(' : - Good Condition of This Leaf.')
    else:
     st.header(' "  Cotton Plant " :' )
     st.header(' : - Good Condition of This Cotton Plant .')
    st.image(user_upload) 
     
    

    
    
       
    
        
    