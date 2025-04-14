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
    area_mean = st.number_input("Area Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    radius_se = st.number_input("Radius Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    texture_se = st.number_input("Texture Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    perimeter_se = st.number_input("Perimeter Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    area_se = st.number_input("Area Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    smoothness_se = st.number_input("Smoothness Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    compactness_se = st.number_input("Compactness Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    concavity_se = st.number_input("Concavity Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)    
    concave_points_se = st.number_input("Concave Points Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    symmetry_se = st.number_input("Symmetry Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    fractal_dimension_se = st.number_input("Fractal Dimension Standard Error", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    radius_worst = st.number_input("Radius Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    texture_worst = st.number_input("Texture Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    area_worst = st.number_input("Area Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    compactness_worst = st.number_input("Compactness Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    concavity_worst = st.number_input("Concavity Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)    
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    diagnosis = st.number_input("Diagnosis", min_value=0, max_value=1, value=0, step=1)
    
    
  with col2:
    st.write("### Input Parameters")
    st.write(f"Radius Mean: {radius_mean}")
    st.write(f"Texture Mean: {texture_mean}")
    st.write(f"Perimeter Mean: {perimeter_mean}")
    st.write(f"Area Mean: {area_mean}")
    st.write(f"Smoothness Mean: {smoothness_mean}")
    st.write(f"Compactness Mean: {compactness_mean}")
    st.write(f"Concavity Mean: {concavity_mean}")
    st.write(f"Concave Points Mean: {concave_points_mean}")
    st.write(f"Symmetry Mean: {symmetry_mean}")
    st.write(f"Fractal Dimension Mean: {fractal_dimension_mean}")
    st.write(f"Radius Standard Error: {radius_se}")
    st.write(f"Texture Standard Error: {texture_se}")
    st.write(f"Perimeter Standard Error: {perimeter_se}")
    st.write(f"Area Standard Error: {area_se}")
    st.write(f"Smoothness Standard Error: {smoothness_se}")
    st.write(f"Compactness Standard Error: {compactness_se}")
    st.write(f"Concavity Standard Error: {concavity_se}")
    st.write(f"Concave Points Standard Error: {concave_points_se}")
    st.write(f"Symmetry Standard Error: {symmetry_se}")
    st.write(f"Fractal Dimension Standard Error: {fractal_dimension_se}")
    st.write(f"Radius Worst: {radius_worst}")
    st.write(f"Texture Worst: {texture_worst}")
    st.write(f"Perimeter Worst: {perimeter_worst}")
    st.write(f"Area Worst: {area_worst}")
    st.write(f"Smoothness Worst: {smoothness_worst}")
    st.write(f"Compactness Worst: {compactness_worst}")
    st.write(f"Concavity Worst: {concavity_worst}")
    st.write(f"Concave Points Worst: {concave_points_worst}")
    st.write(f"Symmetry Worst: {symmetry_worst}")
    st.write(f"Fractal Dimension Worst: {fractal_dimension_worst}")
    st.write(f"Diagnosis: {diagnosis}")

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