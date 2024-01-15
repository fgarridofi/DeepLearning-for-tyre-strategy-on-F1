import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

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

# Construyo mi modelo de Regresión Logística
linear_model = LogisticRegression()

# Entreno mi modelo
linear_model.fit(X_train, y_train)

# Creo y entreno un modelo Random Forest
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

# Evalúo los resultados
test_accuracy_linear = linear_model.score(X_test, y_test)
test_accuracy_random_forest = random_forest_model.score(X_test, y_test)
print('\nExactitud en la prueba (Lineal):', test_accuracy_linear)
print('Exactitud en la prueba (Random Forest):', test_accuracy_random_forest)

# Hago predicciones en el conjunto de prueba
y_pred_linear = linear_model.predict(X_test)
y_pred_random_forest = random_forest_model.predict(X_test)

# Obtengo los coeficientes de la Regresión Logística
coeficients_linear = linear_model.coef_[0]

# Obtengo el nombre de las variables
feature_names = X.columns

# Creo un DataFrame para mostrar los coeficientes de la Regresión Logística
coef_df_linear = pd.DataFrame({'Variable': feature_names, 'Coeficiente': coeficients_linear})
coef_df_linear['Coeficiente_absoluto'] = abs(coef_df_linear['Coeficiente'])  # Valor absoluto de los coeficientes
coef_df_linear = coef_df_linear.sort_values(by='Coeficiente_absoluto', ascending=False)

# Muestro los coeficientes ordenados por su valor absoluto
print('Coeficientes de la Regresión Logística:')
print(coef_df_linear)

# Obtengo las características más importantes del modelo Random Forest
importances_random_forest = random_forest_model.feature_importances_

# Creo un DataFrame para mostrar las características más importantes del modelo Random Forest
importances_df_random_forest = pd.DataFrame({'Variable': feature_names, 'Importancia': importances_random_forest})
importances_df_random_forest = importances_df_random_forest.sort_values(by='Importancia', ascending=False)

# Muestro las características más importantes del modelo Random Forest
print('\nImportancia de las características del modelo Random Forest:')
print(importances_df_random_forest)

# Calculo las predicciones de probabilidad en el conjunto de prueba
y_pred_prob_linear = linear_model.predict_proba(X_test)[:, 1]
y_pred_prob_random_forest = random_forest_model.predict_proba(X_test)[:, 1]

# Calculo el área bajo la curva ROC en el conjunto de prueba
roc_auc_linear = roc_auc_score(y_test, y_pred_prob_linear)
roc_auc_random_forest = roc_auc_score(y_test, y_pred_prob_random_forest)

print('\nÁrea bajo la curva ROC (Lineal):', roc_auc_linear)
print('Área bajo la curva ROC (Random Forest):', roc_auc_random_forest)

# Calculo la matriz de confusión
cm_linear = confusion_matrix(y_test, y_pred_linear)
cm_random_forest = confusion_matrix(y_test, y_pred_random_forest)

# Visualizo la matriz de confusión para el modelo de Regresión Logística
plt.figure(figsize=(8, 6))
sns.heatmap(cm_linear, annot=True, fmt='d')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión (Lineal)')
plt.show()

# Visualizo la matriz de confusión para el modelo Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(cm_random_forest, annot=True, fmt='d')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión (Random Forest)')
plt.show()

# Guardar los resultados en un DataFrame comparativo
results_df = pd.DataFrame({'Modelo': ['Linear', 'Random Forest'],
                           'Accuracy': [test_accuracy_linear, test_accuracy_random_forest],
                           'AUC ROC': [roc_auc_linear, roc_auc_random_forest]})
print('\nResultados comparativos:')
print(results_df)
