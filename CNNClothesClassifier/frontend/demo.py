import streamlit as st
import os
import sys
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
#from backend.utils import device, model, model_weights_path, classes, transform_image, get_prediction

# Add the path to the backend folder to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from backend.utils import device, model, model_weights_path, transform_image, get_prediction
from data.preprocessing import get_transformation

# Now you can import from the backend folder
#from utils import device, model, model_weights_path, classes, transform_image, get_prediction


if __name__== "__main__":
    message = """
    # Clothing Oracle v.1
    Welcome! I am the clothing oracle, show me a piece of clothing and I will tell you what is!
"""

    st.write(message)

    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    st.write("# Upload your photo")
    uploaded_files = st.file_uploader("Choose an image file (.jpg)", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        #st.write(bytes_data)
        transformed_image = transform_image(bytes_data)
        transformer = get_transformation()
        trial_image = T.transforms.ToPILImage()(transformed_image.squeeze(dim=0))

        prediction  = get_prediction(transformed_image)
        st.write("# Predicted class:", prediction)
        st.image(uploaded_file, width= 200)
        st.image(trial_image, width= 200)
        

