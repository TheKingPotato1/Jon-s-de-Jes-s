# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 09:12:44 2025

@author: Jonás de Jesús Contreras Cerecedo """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os

os.chdir('C:/Users/MI PC/Documents/Proyecto 1')

#Limpieza y observación
df = pd.read_csv('DatasetTesla.csv')
print(df.head())
print(df.describe())

print(df.isna().sum())
print(df.dtypes) 
print(len(df))
print(df.columns)
print(df.info()) 


df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

real_data = df[(df['Date'] >= '2010-06-01') & (df['Date'] <= '2020-03-22')] 
df = df.drop(real_data.index)
df.set_index('Date', inplace=True)
real_data.set_index('Date', inplace=True)
print(df.dtypes) 


#Empezamos
train_data_or = df[['High','Low','Open']]  

imputer = SimpleImputer(strategy='mean')
train_data_or = pd.DataFrame(imputer.fit_transform(train_data_or), columns=train_data_or.columns, index=train_data_or.index)

scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_data_or)

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset)-look_back): 
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


X_train, y_train = create_dataset(train_scaled, 60)


model = Sequential([
    LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)


real_data = real_data.drop(['Close', 'Volume'], axis=1)  
real_data = pd.DataFrame(imputer.transform(real_data), columns=real_data.columns, index=real_data.index)  
real_scaled = scaler.transform(real_data) 


X_real, _ = create_dataset(np.vstack([train_scaled[-60:], real_scaled]), 60)
predictions_scaled = model.predict(X_real)
predictions = scaler.inverse_transform(np.concatenate([predictions_scaled, np.zeros((len(predictions_scaled), 2))], axis=1))[:, 0]
real_demand = scaler.inverse_transform(real_scaled)[:, 0]

plt.figure(figsize=(14, 7))
plt.plot(real_data.index, real_demand, label='Datos Reales', color='black')
plt.plot(real_data.index[:len(predictions)], predictions, label='Predicciones', color='blue')
plt.title('Predicciones vs Datos Reales Tesla')
plt.xlabel('Fechas')
plt.ylabel('Close')
plt.legend()
plt.grid(True)
plt.show()



