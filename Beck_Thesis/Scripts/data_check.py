import pandas as pd
import os
import to_learning

stations = pd.read_csv('Coast_orientation/Selected_Stations_w_Data.csv')

japan = stations[stations['Coast'] == "Japan"]

# for station in stations['Station']:
#     ex = os.path.exists('Input_sst_data/' + station + '_sst.nc')
#     print(f'Station: {station} - {ex}')
#     if not ex:
#         print("\nNo data for this station\n")
        
        
for station in japan['Station']:
    df, ds, dir = to_learning.load_file(station, input_dir = '../Input_nc_detrend_sst')
    print(station)
    print(df.describe())
    print('')