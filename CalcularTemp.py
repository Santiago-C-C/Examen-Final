import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de temperatura  ''')
st.image("mexico.jpg", caption="Predicción de la temperatura en una ciudad.")

st.header('Datos')

def user_input_features():
  # Entrada
  Año = st.number_input('Año:', min_value=1900, max_value=2100, value=2020, step=1)
  Ciudad = st.number_input('Ciudad (ID numérico):', min_value=1, max_value=1000, value=1, step=1)
  Mes = st.number_input('Mes (1-12):', min_value=1, max_value=12, value=1, step=1)

  user_input_data = {'Year': Año,
                     'City': Ciudad,
                     'Month': Mes}

  features = pd.DataFrame(user_input_data, index=[0])
  return features

df = user_input_features()

# === Cargar datos ===
M = pd.read_csv('Temperatura.csv', encoding='latin-1')

# Variables independientes y dependientes
X = M[["Year", "Month", "City"]]
y = M['AverageTemperature']

# === Entrenar modelo ===
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1614175)

LR = LinearRegression()
LR.fit(X_train, y_train)

b = LR.coef_
b0 = LR.intercept_

prediccion = b0 + b[0]*df['Year'] + b[1]*df['City'] + b[2]*df['Month']

st.subheader('Cálculo de la temperatura estimada')
st.write('La temperatura estimada es: ', prediccion.values[0], "°C")
