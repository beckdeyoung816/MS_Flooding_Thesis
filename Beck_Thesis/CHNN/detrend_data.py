
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

def detrend_data(station: str):
    """Annually detrend all variables in the dataset. 

    Args:
        station (str): Name of station
    """
    # Load in file
    ds = xr.open_dataset('../Input_nc_sst/' + station + '.nc')
    df = (ds.to_dataframe()
          .drop(['uquad', 'vquad'], axis=1)) # Remove quadratic terms before detrending
    
    df_levels = df.index.get_level_values
    start_date = df_levels('time')[0]
    end_date = df_levels('time')[-1]

    # Calculate the annual mean for each year, and unique pairing of latitude and longitude
    df_annual_mean = df.groupby([pd.Grouper(level='time', freq='A'),
                                        pd.Grouper(level='latitude'), 
                                        pd.Grouper(level='longitude') 
                                ]
          ).mean()
    
    cols = df_annual_mean.columns.tolist() 
    df_annual_levels = df_annual_mean.index.get_level_values
    
    # Create a dataframe of unique latitudes and longitudes
    unique_loc = pd.DataFrame([df_annual_levels('latitude'),df_annual_levels('longitude')], index = ['latitude', 'longitude']).T.drop_duplicates()

    def get_helper_df(vals, date, cols, unique_loc):
        """Creates a helper dataframe with place holder values to augment resampled data. Has the same columns as the original df, and has placeholders for a single date and for each unique location.

        Args:
            vals (_type_): row value place holders
            date (_type_): date to assign
            cols (_type_): columns of the original dataframe
            unique_loc (_type_): unique locations
        """
        temp = pd.DataFrame([vals for x in range(unique_loc.shape[0])], 
                            columns = cols)
        temp['time'] = pd.to_datetime(date)
        return pd.concat([unique_loc, temp], axis=1)
    
    # Create helper dataframe with place holder values for one year before the start date and one year after the end date
    # This is to ensure when interpolating with backfill
    # that the entire date range in the original dataframe is covered. 
    # We have to do this because resampling to get the annual mean puts the date at the end of the year
    # So we lose the whole year before the last date when resampling back to hourly.
    # We have to add the extra year at the end because of the backfill interpolation.
    # We assign this extra year to be the last values in the averaged dataframe.
    start = get_helper_df(vals = [0] * len(cols), 
                          date = start_date - pd.DateOffset(years=1), 
                          cols = cols, unique_loc=unique_loc)
    
    end = get_helper_df(vals = df_annual_mean.iloc[-1,:].to_list(), 
                        date = end_date + pd.DateOffset(years=1), 
                        cols = cols, unique_loc=unique_loc)
    
    # Add the start and end helper dataframes to the original dataframe and reset the indices
    df_avg_helper = pd.concat([start, df_annual_mean.reset_index(), end]).set_index('time').sort_index()
    
    # Loop through each unique location, resample the average dataframe hourly
    # Then interpolate with backfill so for every hour in a given year, it has the mean of the year
    # Join all of these dataframes together
    for index, row in unique_loc.iterrows():
        print(f'{index=} | lat: {row["latitude"]} | lon: {row["longitude"]}\n')
        df_avg_loc = (df_avg_helper[(df_avg_helper['latitude'] == row['latitude']) & 
                                    (df_avg_helper['longitude'] == row['longitude'])]
                      .resample('H').interpolate(method='bfill', limit_direction = 'backward')
                      .loc[start_date:end_date]) # Only keep the date ranges in the original dataframe
        if index == 0:
            df_avg = df_avg_loc
        else:
            df_avg = pd.concat([df_avg, df_avg_loc]).sort_index()
    
    df_avg = df_avg.reset_index().set_index(['time', 'latitude', 'longitude']).sort_index() # Reset multiindex
    
    # Detrend data by subtracting the annual mean of each variable from the original dataframe
    df_full = pd.DataFrame()
    for col in cols:
        # Here all of the missing data in the original dataframe will override the interpolated averages values
        # So the result in the detrended dataset will still be a missing value
        df_full[col] = df[col] - df_avg[col]

    # Calculate quadratic terms on detrended data
    df_full['uquad'] = df_full['u10'] ** 2
    df_full['vquad'] = df_full['v10'] ** 2
    
    # Save to netcdf
    df_full.to_xarray().to_netcdf('../Input_nc_sst_detrend/' + station + '.nc')


# # Load in list of selected stations
# # We drop the stations that do not have data 
stations = pd.read_csv('../Coast_orientation/Selected_Stations_dates.csv').dropna()

for index, station in stations.iterrows():
    print(f'Detrending Data for Station: {station["Station"]}\n')
    
    detrend_data(station['Station'])
