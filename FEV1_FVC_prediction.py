import streamlit as st
import pandas as pd
import numpy as np

st.title("FEV1/FVC prediction App")
st.write("By Pefura-Yone ")

st.sidebar.header('User Input Parameters(please select patients features here)')

age = st.sidebar.number_input('age, years', 4.0, 89.0)
height = st.sidebar.number_input('height, cm', 104.0,188.0 )

data_input = {'age': age,
            'height': height }

df = pd.DataFrame(data_input, index=[0])
row_names = {0:'values'}

df = df.rename(index = row_names)
st.subheader('User Input parameters')
st.table(df)

dataset = pd.read_csv("https://raw.githubusercontent.com/pefura/IFPERA/main/Cameroon_lung_function.csv", sep=";")
dataset_female = dataset.query('sex==2')
data = dataset_female[['age', 'height', 'fevfvc']]

# Selectionner les prédicteurs et la variable réponse

y = data['fevfvc']
X = data.drop(columns =['fevfvc'])

# Entrainer les modèles selectionnés
from sklearn.ensemble import GradientBoostingRegressor

## Fonction de calcul
def prediction_FEV_FVC (age, height):
    # Prediction espérance = médiane
    fit_median = GradientBoostingRegressor(loss="quantile", alpha=0.50, random_state=0)
    fit_median.fit(X, y)
    # Prédiction LLN
    fit_LLN= GradientBoostingRegressor(loss="quantile", alpha=0.05, random_state=0)
    fit_LLN.fit(X, y)
    # Prédiction ULN
    fit_ULN= GradientBoostingRegressor(loss="quantile", alpha=0.95, random_state=0)
    fit_ULN.fit(X, y)
    var = {'age':[age],
        'height':[height]}
    X1 = pd.DataFrame (var)
    pred_median = fit_median.predict(X1)
    LLN = fit_LLN.predict(X1)
    ULN = fit_ULN.predict(X1)
    table = pd.DataFrame([LLN [0],pred_median[0], ULN[0]]).T
    table.columns = ["LLN", "median", "ULN"]
    return table
# Prédiction pour les valeurs spécifiques de l'âge et de la taille
pred = prediction_FEV_FVC (age=age, height=height).T
pred.columns = ["values"]
predicted_values = pred.T

# Prediction
st.subheader('FEV1/FVC predicted')
st.table(predicted_values)
st.write("LLN: lower limit of normal; ULN: upper limit of normal")
