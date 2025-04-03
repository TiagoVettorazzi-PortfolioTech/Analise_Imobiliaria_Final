# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from pydantic import BaseModel, Field
import os
os.chdir(os.path.abspath(os.curdir))

# Carregando o modelo treinado
model = joblib.load('models/modelo3.pkl')
#kmeans = joblib.load('models/modelo3.pkl')

# Instanciando o app
app = FastAPI(title="API de Previsão de Preço de Imóvel")

# MINHAS COLUNAS
# 'longitude', 'latitude', 'housing_median_age', 'total_rooms',
# 'total_bedrooms', 'population', 'households', 'median_income',
# 'median_house_value', 'ocean_proximity_INLAND',
# 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
# 'ocean_proximity_NEAR OCEAN'

# Definindo o formato dos dados de entrada
class InputData(BaseModel):
       # Variáveis numéricas para a previsão
    aream2 : float 
    Quartos: int
    banheiros: int
    vagas: int
    condominio: float
    latitude: float
    longitude:float
    idh_longevidade: float 
    area_renda:float
    distancia_centro: float
    cluster_geo: int
     
@app.post("/predict")
def predict(data: InputData):
    # Convertendo os dados para um array 2D
    # input_array = np.array([[ 
    #     data.longitude, data.latitude, data.housing_median_age,
    #     data.total_rooms, data.total_bedrooms, data.population,
    #     data.households, data.median_income, data.ocean_proximity_INLAND, 
    #     data.ocean_proximity_ISLAND, data.ocean_proximity_NEAR_BAY, 
    #     data.ocean_proximity_NEAR_OCEAN
    # ]])

    print("Recebendo os dados:", data)
    # Convertendo os dados para um array 2D
    # input_array = np.array([[ 
    #     data.aream2 , data.Quartos, data.banheiros, data.vagas, 
    #     data.condominio, data.latitude, data.longitude, data.idh_longevidade,
    #     data.area_renda, data.distancia_centro, data.cluster_geo  
    # ]])
        
    input_array = np.array([[ 
        100 , 2, 2, 3, 
        1500, 0, 0, 0.92,
        34, 0, 2  
    ]])  
    # Fazendo a predição
    try:
        prediction = model.predict(input_array)
        return {"predicted_house_value": round(float(prediction[0]), 2)}

    except Exception as e:
        print(e)
# More Fast
# Carregando o modelo e as colunas
# model = joblib.load('models/model_rf.pkl')
# model_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
#                 'total_bedrooms', 'population', 'households', 'median_income',
#                 'median_house_value', 'ocean_proximity_INLAND',
#                 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
#                 'ocean_proximity_NEAR OCEAN']


# class InputData(BaseModel):
#     features: list[float]  # Espera os dados já no formato final

# @app.post("/predict")
# def predict(data: InputData):
#     input_array = np.array(data.features).reshape(1, -1)
#     prediction = model.predict(input_array)
#     return {"predicted_house_value": round(float(prediction[0]), 2)}