import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df_laps = pd.read_csv('/Users/administrador/Desktop/TFE/df/dflaps_df.csv')

def time_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

df_laps = df_laps[~df_laps['Circuito'].isin(['Japan', 'Monaco', 'Singapore'])]

# Convierto la columna "Tiempo_vuelta" a segundos para poder realizar cálculos con ella
df_laps['Tiempo_vuelta_secs'] = df_laps['Tiempo_vuelta'].apply(time_to_seconds)

# Creo la columna para calcular la media de mis tiempos de las próximas 10 vueltas
df_laps['Avg_Next_10_Laps'] = df_laps.groupby('Piloto')['Tiempo_vuelta_secs'].rolling(window=10, min_periods=1).mean().reset_index(0, drop=True)

# Creo la columna objetivo "Tire_Change_Optimal"
df_laps['Tire_Change_Optimal'] = ((df_laps['Tiempo_vuelta_secs'] - df_laps['Avg_Next_10_Laps']) > 1).astype(int)

# Divido los datos en un conjunto de entrenamiento y un conjunto de prueba
X = df_laps.drop(['Tire_Change_Optimal', 'Tiempo_vuelta', 'Piloto', 'Circuito', 'Neumaticos'], axis=1)
y = df_laps['Tire_Change_Optimal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardo el número de características
num_features = len(X_train.keys())

# Verifico y manejo NaN e Inf en los datos antes de la división y la normalización
if X.isnull().values.any() or np.isinf(X.select_dtypes(include=[np.number]).values).any():
    print('Existen valores NaN o Inf en mi conjunto de datos')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

if y.isnull().values.any() or np.isinf(y.values).any():
    print('Existen valores NaN o Inf en mi conjunto de etiquetas')
    y.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.fillna(y.median(), inplace=True)

# Divido los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizo los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construyo el modelo de Regresión Logística
model = LogisticRegression()

# Entreno el modelo
model.fit(X_train, y_train)

# Evalúo los resultados
test_accuracy = model.score(X_test, y_test)
print('\nTest accuracy:', test_accuracy)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Hago predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calculo la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizo la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
plt.show()


# Obtengo los coeficientes de la Regresión Logística
coeficients = model.coef_[0]

# Obtengo el nombre de las variables
feature_names = X.columns

# Creo un DataFrame para mostrar los coeficientes
coef_df = pd.DataFrame({'Variable': feature_names, 'Coeficiente': coeficients})
coef_df['Coeficiente_absoluto'] = abs(coef_df['Coeficiente'])  # Valor absoluto de los coeficientes
coef_df = coef_df.sort_values(by='Coeficiente_absoluto', ascending=False)

# Muestro los coeficientes ordenados por su valor absoluto
print(coef_df)


df_laps.to_csv('/Users/administrador/Desktop/TFE/df/laps_df_postEntreno.csv', index=False)

