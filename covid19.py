
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import seaborn as sns
import numpy as np



st.write("""
# Machine Learning Covid-19 death Prediction App

This app predicts the risk of **Death during Covid-19 ** 

By Pefura-Yone et al. 
""")

st.header('User Input Parameters(please select patients features here)')


def user_input_features():
    age = st.selectbox('Age(0 - 17 years=1, 18 to 49 years=2, 50-64 years=3,  ≥65 years=4)', ('1','2', '3','4'))
    race = st.selectbox('Ethnicity (White=1, Black=2, Asian=3, Multiple/Other=4, Unknown=5)', ('1', '2', '3','4','5'))
    exposure = st.selectbox('Exposure (Yes=1, No/Unknown=2)', ('1', '2'))
    symptoms = st.selectbox('Symptoms(Asymptomatic=0, Symptomatic=1,Unknown=2)', ('0', '1','2'))
    hospitalisation = st.selectbox('Hospitalisation(No=0, Yes=1, Unknown=2)', ('0', '1', '2'))
    intensive_care = st.selectbox('Intensive care(No=0, Yes=1, Unknown=2)', ('0', '1', '2'))
    comorbidities= st.selectbox('comorbidities(No=0, Yes=1, Unknown=2)', ('0', '1', '2'))
    data = {'age': age,
            'race': race,
            'exposure': exposure,
            'symptoms': symptoms,
            'hospitalisation':hospitalisation,
            'intensive_care':intensive_care,
            'comorbidities':comorbidities}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


st.subheader('User Input parameters confirmation')
st.write(df)

