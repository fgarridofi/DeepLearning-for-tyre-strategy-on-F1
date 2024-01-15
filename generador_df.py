import pandas as pd
import fastf1 as ff1

# Inicializo una lista vacía para almacenar los datos
data = []

# Especifica las carreras que quiero analizar
for race_number in range(1, 23):  
    session = ff1.get_session(2022, race_number, 'R')
    session.load()
    laps = session.laps

    

    # Extraemos la información necesaria para cada vuelta
    for i, lap in laps.iterrows():
        car_data = lap.get_car_data()
        
        # Calculo las medias por vuelta de los sensores de telemetria
        rpm_mean = car_data['RPM'].mean()
        speed_mean = car_data['Speed'].mean()
        nGear_mean = car_data['nGear'].mean()
        throttle_mean = car_data['Throttle'].mean()
        brake_mean = car_data['Brake'].mean()
        drs_mean = car_data['DRS'].mean()


        weather_data = lap.get_weather_data()


        lap_data = {
            "Circuito": session.event.Country,
            "Piloto": lap.Driver,
            "Num_vuelta": lap.LapNumber,
            "Tiempo_vuelta": lap.LapTime,
            "Neumaticos": lap.Compound,
            "Neumaticos_usados": lap.Stint,
            "RPM_Medias": rpm_mean,
            "Vel_Media": speed_mean,
            "nMarcha_Media": nGear_mean,
            "Acelerador_Media": throttle_mean,
            "Freno_Media": brake_mean,
            "DRS_Media": drs_mean,
            "Temperatura": weather_data["AirTemp"],
            "Humedad": weather_data["Humidity"],
            "Presion": weather_data["Pressure"],
            "LLuvia": weather_data["Rainfall"],
            "Temp_pista": weather_data["TrackTemp"],
            "Direccion_viento": weather_data["WindDirection"],
            "Velocidad_viento": weather_data["WindSpeed"],
            
        }
        data.append(lap_data)

# Conviertos los datos en un DataFrame
df_laps = pd.DataFrame(data)

df_laps = df_laps[df_laps['Tiempo_vuelta'].notna()]
df_laps['Tiempo_vuelta'] = (df_laps['Tiempo_vuelta'].dt.total_seconds()).apply(lambda x: '{:02d}:{:06.3f}'.format(int(x // 60), x % 60))


df_laps.to_csv('/Users/administrador/Desktop/TFE/df/dflaps_df.csv', index=False)