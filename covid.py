
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

death_covid = pd.read_csv('C:/Users/DDD/Desktop/data/Covid et vaccin/covid_cleaned_1_coded_ML.csv', header=0)
death_covid=death_covid.astype(str)
death_covid.head(3)



# Slectionner les prédicteurs et la variable réponse
y = death_covid['death']
X = death_covid.drop('death', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Model fit


encoder = OneHotEncoder()

model =  make_pipeline(encoder,SGDClassifier(loss='log', penalty="l2", max_iter=1000))
model.fit(X_train, y_train)
model.score(X_test, y_test)


# Prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
prediction_proba_percent = prediction_proba * 100
proba = prediction_proba[:, 1]
prediction_proba_percent = proba * 100

st.subheader('Logistic Regression Probability of death(%)')
st.write(prediction_proba_percent)

probability_risk= {"low risk":"probability < 5%", "moderate risk": "probability  5-15%", "high risk": "probability >15%"}
#### Prédiction Naive Bayes
NBA =  CategoricalNB()
NBA.fit(X_train, y_train)
NBA.score(X_test, y_test)


#st.write(prediction)
st.subheader('Risk of death')
st.write( '''
low risk: probability < 5%

intermediate risk: probability 5-15%

high risk: probability >15% ''')