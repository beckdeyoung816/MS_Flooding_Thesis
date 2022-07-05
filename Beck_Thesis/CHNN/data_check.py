import pandas as pd
import os

stations = pd.read_csv('Coast_orientation/Selected_Stations_w_Data.csv')

for station in stations['Station']:
    ex = os.path.exists('Input_sst_data/' + station + '_sst.nc')
    print(f'Station: {station} - {ex}')
    if not ex:
        print("\nNo data for this station\n")