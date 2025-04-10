import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sklearn as sk  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def create_model(data):
  X = data.drop(['diagnosis'], axis=1)
  
  y = data['diagnosis']
  
  scaler= StandardScaler()
  
  X = scaler.fit_transform(X)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  
  model = LogisticRegression()
  
  model.fit(X_train, y_train)
  
  y_pred = model.predict(X_test)
  
  accuracy = accuracy_score(y_test, y_pred)
  
  return model, scaler


def test_model(data, model, scaler):
  st.write("Test the model")
  
  col1, col2 = st.columns(2)
  
  with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    
    
  with col2:
    st.write("### Input Parameters")
    st.write(f"Radius Mean: {radius_mean}")
    st.write(f"Texture Mean: {texture_mean}")
    st.write(f"Perimeter Mean: {perimeter_mean}")

def get_clean_data():
  print(data.head())
  data = pd.read_csv("dataset/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

def main():
  data = get_clean_data ()
  
  model, scaler = create_model(data)
  
  test_model(data, model, scaler)
  
  with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
  
  with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)






if __name__ == '__main__':
  main()