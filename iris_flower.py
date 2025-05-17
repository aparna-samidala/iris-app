import streamlit as st
from keras.models import load_model 
import numpy as np 
import pandas as pd
from sklearn.datasets import load_iris                                                                                                           


model = load_model("model.h5")
label = np.load("label.npy")



st.markdown(
    "<h1 style='text-align: center; font-size: 80px;'>🌸</h1>",
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: black;'>Iris  Flower  Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 15px;'><strong><i>Enter the flower measurements to predict the species</i></strong></p>", unsafe_allow_html=True)


st.image("iris_classi.png")

a = float(st.number_input("sepal length in cm"))
b = float(st.number_input("sepal width in cm"))
c = float(st.number_input("petal length in cm"))
d = float(st.number_input("petal width in cm"))

# Custom styled button using markdown
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: red;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Predict"):

    prediction = model.predict(np.array([a,b,c,d]).reshape(1,-1))
    pred = label[np.argmax(prediction)]
    st.subheader(pred)
    
    
    if pred == "Iris-setosa":
        st.image("Iris-setosa.jpg")
    elif pred == "Iris-versicolor":
        st.image("iris-versicolor.jpg")
    elif pred == "Iris-virginica":
        st.image("iris_virginica.jpg")