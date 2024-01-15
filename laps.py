import fastf1
import fastf1.plotting
import pandas as pd
from matplotlib import pyplot as plt


session = fastf1.get_session(2022, 1, 'R')
session.load()

#laps = session.laps
#print(laps)

#laps_selected = laps[["Time", "Driver", "LapTime", "LapNumber"]]
#print(laps_selected)

laps = session.laps.pick_driver('PER').get_car_data()
print(laps)


laps.to_csv('/Users/administrador/Desktop/TFE/df/laps_df.csv', index=False)