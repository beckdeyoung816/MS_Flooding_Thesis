import numpy as np
import pandas as pd
import to_learning
import os

# os.chdir('..')

def get_date_range(station):
    print(f'Getting Dates for Station: {station["Station"]}\n')
    
    df, ds, dir = to_learning.load_file(station['Station'], input_dir = 'Input_nc')
    # Get the start and end dates of the dataset to get those same dates for the SST data
    start = np.min(np.array(ds.time))
    start = pd.to_datetime(str(start)).strftime('%Y-%m-%d')
    
    end = np.max(np.array(ds.time)) + pd.DateOffset(1) # Add one day after end because filtering is exclusive
    end = pd.to_datetime(str(end)).strftime('%Y-%m-%d')
    
    return start, end

stations = pd.read_csv('Coast_orientation/Selected_Stations.csv')
# stations = stations[stations['Coast'] == 'NE_Atlantic_Yellow']

stations['start_date'], stations['end_date'] = zip(*stations.apply(get_date_range, axis=1))

stations.to_csv('Coast_orientation/Selected_Stations_dates.csv')