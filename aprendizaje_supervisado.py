import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_laps = pd.read_csv('/Users/administrador/Desktop/TFE/df/dflaps_df.csv')

def time_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

df_laps = df_laps[~df_laps['Circuito'].isin(['Japan', 'Monaco', 'Singapore'])]

# Convertir la columna "Tiempo_vuelta" a segundos para poder realizar cálculos con ella
df_laps['Tiempo_vuelta_secs'] = df_laps['Tiempo_vuelta'].apply(time_to_seconds)

# Crear la columna para calcular la media de los tiempos de las próximas 10 vueltas
df_laps['Avg_Next_10_Laps'] = df_laps.groupby('Piloto')['Tiempo_vuelta_secs'].rolling(window=10, min_periods=1).mean().reset_index(0, drop=True)

# Crear la columna objetivo "Tire_Change_Optimal"
df_laps['Tire_Change_Optimal'] = ((df_laps['Tiempo_vuelta_secs'] - df_laps['Avg_Next_10_Laps']) > 1).astype(int)

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X = df_laps.drop(['Tire_Change_Optimal', 'Tiempo_vuelta', 'Piloto', 'Circuito', 'Neumaticos'], axis=1)
y = df_laps['Tire_Change_Optimal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar el número de características
num_features = len(X_train.keys())

# Verificar y manejar NaN e Inf en los datos antes de la división y la normalización
if X.isnull().values.any() or np.isinf(X.select_dtypes(include=[np.number]).values).any():
    print('Existen valores NaN o Inf en el conjunto de datos')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

if y.isnull().values.any() or np.isinf(y.values).any():
    print('Existen valores NaN o Inf en el conjunto de etiquetas')
    y.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.fillna(y.median(), inplace=True)

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir el modelo
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X_train[0])]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    epochs=100, validation_split = 0.2)

# Evaluar los resultados
test_results = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_results[1])

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred = [1 if p > 0.5 else 0 for p in y_pred]

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
plt.show()


df_laps.to_csv('/Users/administrador/Desktop/TFE/df/laps_df_postEntreno.csv', index=False)