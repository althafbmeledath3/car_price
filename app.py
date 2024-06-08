import streamlit as st
import pickle
import numpy as np
import sklearn

from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LinearRegression
df = pickle.load(open("df.pkl",'rb'))
with open("pipeline.pkl", 'rb') as f:
    pipeline = pickle.load(f)

st.title("Car Price Predictor")

name = st.selectbox("name",df['name'].unique())
year = st.selectbox("year",df['year'].unique())
km = st.number_input("Ënter the km driven")
fuel = st.selectbox("fuel",df['fuel'].unique())
seller_type = st.selectbox("seller_type",df['seller_type'].unique())
transmission = st.selectbox("Transmission",df['transmission'].unique())
owner = st.selectbox("owner",df['owner'].unique())
mileage = st.number_input("Ënter the Mileage")
engine = st.number_input("Ënter the engine power in between 50 to 3600 Horse Power")
seats = st.selectbox("seats",df['seats'].unique())


query = np.array([name,year,km,fuel,seller_type,transmission,owner,mileage,engine,seats],dtype='object').reshape(1,10)


if st.button("Predict Price"):
    out = pipeline.predict(query)
    st.title(f'The Price is {np.round(out[0],decimals=2)} rs')



