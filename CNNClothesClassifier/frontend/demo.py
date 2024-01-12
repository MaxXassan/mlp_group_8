import streamlit as st
import os
import sys
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

#Import the model and useful functions
from backend.utils import device, model, model_weights_path, transform_image, get_prediction

#Streamlit demo
if __name__== "__main__":
    message = """
    # Clothing Oracle v.1
    Welcome! I am the clothing oracle, show me a piece of clothing and I will tell you what it is!
"""
    st.write(message)
    #Load model
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    
    #Upload image
    st.write("# Upload your photo")
    uploaded_files = st.file_uploader("Choose an image file!", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()

        #Transform image
        transformed_image = transform_image(bytes_data)
        trial_image = T.transforms.ToPILImage()(transformed_image.squeeze(dim=0))
        
        #Make prediction
        prediction  = get_prediction(transformed_image)

        #View prediction, original and transformed image
        st.write("# Predicted class:", prediction)
        st.image(uploaded_file, width= 200)
        st.image(trial_image, width= 200)
        

