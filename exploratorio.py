import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



df_laps = pd.read_csv('/Users/administrador/Desktop/TFE/df/dflaps_df.csv')

####################
#TIPOS DE VARIABLES#
####################
#print(df_laps.dtypes)


###############
#CORRELACIONES#
###############

numeric_columns = df_laps.select_dtypes(include=[np.number]).columns
correlation_matrix = df_laps[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.show()




##################################################
#TABLAS DE CONTINGENCIA DE VARIABLES CUALITATIVAS#
##################################################

qualitative_columns = ['Piloto', 'Circuito', 'Neumaticos']

#Tablas de contingencia para cada par de variables cualitativas
for i in range(len(qualitative_columns)):
    for j in range(i + 1, len(qualitative_columns)):
        contingency_table = pd.crosstab(df_laps[qualitative_columns[i]], df_laps[qualitative_columns[j]])
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_table, annot=True, cmap='coolwarm')
        plt.xlabel(qualitative_columns[j])
        plt.ylabel(qualitative_columns[i])
        plt.title(f'Tabla de contingencia: {qualitative_columns[i]} vs {qualitative_columns[j]}')
        plt.show()

#####################################################################
#CALCULAR CORRELACIONES ENTRE VARIABLES CUANTITATIVAS Y CUALITATIVAS#
#####################################################################

quantitative_columns = ['Vel_Media', 'Num_vuelta']
qualitative_columns = ['Piloto', 'Circuito', 'Neumaticos']

#Calculamos las correlaciones entre cada par de variables
for quantitative_column in quantitative_columns:
    for qualitative_column in qualitative_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=qualitative_column, y=quantitative_column, data=df_laps)
        plt.xlabel(qualitative_column)
        plt.ylabel(quantitative_column)
        plt.title(f'Relación entre {qualitative_column} y {quantitative_column}')
        plt.show()



###################################
#COMPARACION DEL USO DE NEUMATICOS#
###################################

#plt.bar(df_laps['Neumaticos'].value_counts().index, df_laps['Neumaticos'].value_counts().values)
#plt.xlabel('Tipo de neumático')
#plt.ylabel('Nº de vueltas ')
#plt.title('Gráfico de barras de uso de los neumáticos')
#plt.show()


############################################################       
#GRAFICO RELACION DEL TIPO DE NEUMATICO CON VELOCIDAD MEDIA#
############################################################

#df_miami = df_laps[df_laps['Circuito'] == 'Italy']

#sns.violinplot(x='Neumaticos', y='Vel_Media', data=df_miami)
#plt.xlabel('Tipo de neumáticos')
#plt.ylabel('Velocidad media')
#plt.title('Gráfico de violín de neumáticos por velocidad media  -  Circuito de Italia')
#plt.show()

########################################################
#GRAFICO RELACION DEL TIPO DE NEUMATICO CON SU DURACION#
########################################################

# Filtramos los datos para incluir solo las filas correspondientes al circuito de 'Spain'.
#df_spain = df_laps[df_laps['Circuito'] == 'Spain']

# Agrupamos los datos por tipo de neumático y calculamos el promedio de vueltas que duraron.
#grouped_spain = df_spain.groupby('Neumaticos')['Num_vuelta'].mean().reset_index()

# Creamos el gráfico de barras.
#plt.figure(figsize=(10,6))
#sns.barplot(x='Neumaticos', y='Num_vuelta', data=grouped_spain)

#plt.title('Promedio de vueltas antes de cambiar neumáticos por tipo de neumático (Circuito de España)')
#plt.xlabel('Tipo de Neumaticos')
#plt.ylabel('Promedio de Vueltas')

#plt.show()

############
#CLUSTERING#
############

# Creamos un DataFrame con información agregada sobre el uso de neumáticos en cada circuito.
df_circuit_tires = df_laps.groupby(['Circuito', 'Neumaticos'])['Num_vuelta'].agg(['mean', 'min', 'max']).reset_index()

# Pasamos a formato largo a ancho
df_circuit_tires = df_circuit_tires.pivot_table(index='Circuito', columns='Neumaticos', aggfunc={'mean': 'first', 'min': 'first', 'max': 'first'})
df_circuit_tires.fillna(0, inplace=True)  # Rellenamos los valores nulos con 0

# Aplanamos las columnas
df_circuit_tires.columns = ['_'.join(col).strip() for col in df_circuit_tires.columns.values]

# Escalamos los datos
scaler = StandardScaler()
df_circuit_tires_scaled = scaler.fit_transform(df_circuit_tires)

# Realizamos el clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(df_circuit_tires_scaled)

# Añadimos las etiquetas de los clusters al DataFrame original
df_circuit_tires['cluster'] = kmeans.labels_

# Visualizamos los clusters utilizando PCA para reducir la dimensionalidad a 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_circuit_tires_scaled)

plt.figure(figsize=(10,6))
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=kmeans.labels_)

# Añadimos los nombres de los circuitos en cada punto
for i, circuit in enumerate(df_circuit_tires.index):
    plt.text(principalComponents[i, 0], principalComponents[i, 1], circuit)

plt.title('Visualización de Clusters')
plt.show()

        
        
        
  