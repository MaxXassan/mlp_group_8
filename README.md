# Clothing Classification ML Model
This project is a machine learning model that classifies clothing articles using a Flask API. 
It leverages Docker for seamless deployment and scalability. For demonstration purposes, there is a streamlit demo included as well. 

# Overview
The Clothing Classification ML Model is designed to identify and classify different types 
of clothing articles, such as shirts, pants, dresses, and shoes. It utilizes machine learning 
techniques to make accurate predictions based on input images.

# Key features:

* Image classification of clothing articles
* Flask API for easy integration into web applications
* Docker containerization for deployment and scalability
* Streamlit demo

# Prerequisites
Before you begin, ensure you have met the following requirements:

* Docker: https://docs.docker.com/desktop/

* Python >= 3.8: https://www.python.org/downloads/

# Installation
Follow these steps to set up and run the Clothing Classification ML Model:

* git clone https://github.com/yourusername/clothing-classification.git

* cd MLP_GROUP_8

Then to run the application:
* docker-compose up --build

# Demo
To use the streamlit demo follow this link: 
* https://fashion-mnist-clothing-classifier.streamlit.app/
or run it locally or over a network through: 
* going to the "mlp_group_8/CNNClothesClassifier/frontend" directory and typing "python -m streamlit run demo.py