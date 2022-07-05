# -*- coding: utf-8 -*-
"""
Preparing input for machine learning

Timothy Tiggeloven and Anaïs Couasnon
"""

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
import sys
import time
import xarray as xr


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def reframe_scale(df, timesteps, scaler_type='MinMax', year='last', prior=False, scaler_op=True):
    df2 = df.copy(deep=True)
    
    #Removing the testing data before transformation
    if year == 'random' or year == 'last':
        dates, _ = draw_sample(df2, 'residual', timesteps, threshold=0, year=year)
    else:
        dates = df2.iloc[-timesteps:].index.values
        
    i_dates = np.array([df2.index.get_loc(date) for date in dates])
    all_index = df2.reset_index().index.values
    train_index = np.delete(all_index, i_dates)
    
    if not prior:
        cols = df2.columns.tolist()
        df2 = df2[cols[1:] + cols[:1]]
    values = df2.values

    # ensure all data is float
    values = values.astype('float32')

    # normalize features on training data only
    if scaler_op == True:
        train = values[train_index, :]        
        # n_train_hours = int(values.shape[0] * tt_value)
        # train = values[:n_train_hours, :]

        if scaler_type == 'MinMax':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler = scaler.fit(train)
        elif scaler_type == 'std_normal':
            scaler = StandardScaler()
            scaler = scaler.fit(train)
        elif scaler_type == 'yeo-johnson':
            #Cannot use Yeo-johnson as is because of a bug in scipy so making this small way around - Using https://github.com/scikit-learn/scikit-learn/issues/14959
            #preprocessor = make_pipeline(QuantileTransformer(output_distribution='uniform'),PowerTransformer(standardize=True))
            preprocessor = make_pipeline(QuantileTransformer(output_distribution='normal'),PowerTransformer(standardize=True))
            scaler = preprocessor.fit(train)
        else:
            print('Could not read your choice, going with MinMax Scaler')
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler = scaler.fit(train)

        scaled = scaler.transform(values)
    else:
        scaled = values

    # frame as supervised learning
    if prior:
        reframed = series_to_supervised(scaled, 1, 1)
        # drop columns we don't want to predict
        reframed.drop(reframed.columns[5:], axis=1, inplace=True)
    else:
        columns = ['var{}(t)'.format(i + 1) for i in range(len(df2.columns) - 1)]
        columns.append('values(t)')
        reframed = pd.DataFrame(scaled, columns=columns)
    
    return reframed, scaler, scaled, dates, i_dates

def resample_rolling(df, lat_list, lon_list, variables, resample, resample_method, make_univariate):
    # df = df.rolling('12H').mean()
    # df['residual'] = df['residual'].rolling('12H').mean().values
    df = df.copy()
    if resample == 'hourly' and resample_method == 'raw':
        step = 24
        df = df.copy()
    elif resample == 'hourly' and resample_method == 'rolling_mean':
        step = 24
        df = df.rolling('12H').mean()
        # df['residual'] = df['residual'].rolling('12H').mean().values
    elif resample == 'daily' and resample_method == 'res_max':
        step = 1
        index_var_name = 'msl_{}_{}'.format(lat_list[int(len(lat_list)/2)], lon_list[int(len(lon_list)/2)])
        index = df[index_var_name].loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:, 0]].index
        df_res = df['residual'].resample('24H').max().to_frame()
        for var in variables:
            df_var = df.loc[:, df.columns.str.startswith(var)]
            if var != 'msl' and var != 'grad':
                df_var  = df_var.loc[index, :]
            # df_var  = df_var.loc[index, :]
            df_var = df_var.resample('24H').max()
            df_res = df_res.merge(df_var, how='left', right_index=True, left_index=True)
        df = df_res.copy()
    elif resample == 'daily' and resample_method == 'max':
        step = 1
        df = df.resample('24H').max()
        
    if make_univariate == True:
        for var in variables:
            print(var)
            var_cols = [col for col in df.columns if var in col]
            df.loc[:,var] = df.loc[:,var_cols].max(axis = 1)
            df.drop(var_cols, axis = 1, inplace = True)            
    
    return df, step

def load_file(station, input_dir):

    if station == 'Cuxhaven':
        station_name = 'cuxhaven-cuxhaven-germany-bsh'
    elif station == 'Hoek van Holland':
        station_name = 'hoekvanholla-hvh-nl-rws'
    elif station == 'Puerto Armuelles':
        station_name = 'puerto_armuelles_b-304b-panama-uhslc'
    else:
        station_name = station
        # sys.exit('Station name not found!')
    filename = os.path.join(input_dir, station_name + '.nc')

    df_dir = pd.read_excel(os.path.join('Coast_orientation', 'stations.xlsx'))
    # direction = df_dir['Direction'][df_dir['Station'] ==  station_name].values[0]
    direction = 'N'

    # read in variables
    ds = xr.open_dataset(filename)
    df = ds[['gesla_swl', 'tide_wtrend', 'residual']].to_dataframe()
    # df = df.rolling('12H').mean()
    return df, ds, direction

def select_extremes(df, percent):
    extremes = df.nlargest(round(percent*len(df)), 'residual').copy()
    extremes.sort_index(inplace = True)
    extreme_dates = extremes.index.values
    return extreme_dates, df.loc[extreme_dates,:]

def calc_na_notna(df, col):
    
    # df_test = pd.DataFrame(df['residual'])
    df_test = pd.DataFrame(df[col])
    
    #Count nb of null
    count_df = pd.concat([df_test, (df_test.loc[:,col].isnull().astype(int).groupby(df.loc[:,col].notnull().astype(int).cumsum()).cumsum().to_frame('na_consec_count'))], axis=1)    
    # compute mask where np.nan = True
    mask = pd.isna(df_test).astype(bool)    
    # compute cumsum across rows fillna with ffill
    cumulative = df_test.astype(bool).cumsum().fillna(method='ffill', axis=0).fillna(0)    
    # get the values of cumulative where nan is True use the same method
    restart = cumulative[mask].fillna(method='ffill', axis=0).fillna(0)    
    # set the result
    result = (cumulative - restart)
    result[mask] = np.nan
    result.rename(columns = {result.columns[0]:'notna_consec_count'}, inplace = True)
    
    count_df = pd.concat([count_df, result], axis = 1)
    count_df.reset_index(inplace = True)
    return count_df

def select_testing_year(count_df, step = 1):       
    #Selecting the testing year
    valid_year_indexes = count_df[count_df['notna_consec_count'].values.astype(int) % (365*step) == 0]
    sel_valid_year_index = np.random.choice(valid_year_indexes.index, 1, replace=False)     
    validation_year = count_df.iloc[np.int((sel_valid_year_index - (365*step) + 1)) : np.int(sel_valid_year_index),:].set_index('time').index
    
    return validation_year

def select_ensemble(df, col, ML, batch, tt_value=0.7, frac_ens=0.5, mask_val=-999, NaN_threshold=0):
    df = df.copy()
    pool = df.index.values
    if ML == 'CNN' or ML == 'ANN':
        df = df[df[col].notnull()]
        random_draw = np.random.choice(len(df), size=int(len(df)*frac_ens), replace=False)
        random_pool = pool[random_draw]
        df_draw = df.iloc[random_pool,:]
        n_train = int(len(df_draw) * tt_value)
        df_draw = df_draw.reset_index(drop = True)
        validation_data = None
        sequences = None
    elif ML == 'LSTM' or ML == 'TCN':
        # count consecutive NaN
        na_count = df[col].isnull().astype(int).groupby(df[col].notnull().astype(int).cumsum()).cumsum()
        na_count[na_count > 0] = np.nan  # 7*24

        # binary na identification
        na_binary = na_count.copy()
        na_binary[na_binary > 0] = 1

        # randomly select start of batch
        start = np.random.randint(0, batch)
        end = batch + start
        sequences = []
        for i in range(int(len(df) / batch) - 1):
            if np.all(na_count[start:end].notnull()) and na_binary[start:end].sum() < batch * 0.25:
                sequences.append(pool[start:end])
            start += batch
            end += batch
        sequences = np.array(sequences)
        if len(sequences) < 20: #26
            print(f'Number of random sequences found was lower than 20 with: {len(sequences)}')
            #sys.exit(0)
        random_draw = np.random.choice(len(sequences), size=int(len(sequences)*frac_ens), replace=False)
        random_sequences = sequences[random_draw]
        random_sequences_train = random_sequences[:int(len(random_sequences)*tt_value)].flatten()
        random_sequences_valid = random_sequences[int(len(random_sequences)*tt_value):].flatten()
        df_train = df.loc[random_sequences_train]
        df_valid = df.loc[random_sequences_valid]
        n_train = len(df_train)
        df_draw = pd.concat([df_train, df_valid])
        df_draw[df_draw.isnull()] = mask_val
    return df_draw, n_train

def draw_sample(df, col, timesteps, threshold=0, year='last'):
    """ 
    select from pool of dates where there are no consecutive NaN of more than 7 days
    """
    # count timesteps in 5 years and insert ID's
    df = df.copy()    
    df.insert(0, 'ID', range(0, len(df)))
    
    df_dates = df.index
    df = df.reset_index(drop= True)

    # count consecutive NaN
    na_count = df[col].isnull().astype(int).groupby(df[col].notnull().astype(int).cumsum()).cumsum()

    # count consecutive timesteps without consecutive NaN of more than 'threshold' days
    na_count[na_count > threshold] = np.nan
    value_count = na_count.notnull().astype(int).groupby(na_count.isnull().astype(int).cumsum()).cumsum()

    # binary na identification
    na_binary = na_count.copy()
    na_binary[na_binary > 0] = 1

    # create pool of dates and randomly select dates interval
    pool = value_count[value_count > timesteps].index.values
    if len(pool) < 1:
        sys.exit(f'No consecutive {timesteps} timesteps found without specified NaN interval')
    
    if year == 'random':
        # check if at least has 75% values
        check, count = False, 0
        while check == False:
            if count > 20:
                sys.exit(f'No consecutive {timesteps} timesteps found with less than 25% NaN')
            end_date = pool[np.random.randint(0, len(pool))]
            end_ID = df.ID.loc[end_date]
            begin_ID = end_ID - timesteps
            if np.all(na_count[begin_ID:end_ID].notnull()) and na_binary[begin_ID:end_ID].sum() < timesteps * 0.25:
                check = True
            count += 1
    elif year == 'last':
        for end_date in reversed(pool):
            end_ID = df.ID.loc[end_date]
            begin_ID = end_ID - timesteps
            if np.all(na_count[begin_ID:end_ID].notnull()) and na_binary[begin_ID:end_ID].sum() < timesteps * 0.25:
                break
    dates = df_dates.values[begin_ID:end_ID]
    #dates = df.index.values[begin_ID:end_ID]
    df = df.iloc[begin_ID:end_ID].copy()
    df.drop(['ID'], axis=1, inplace=True)
    df.set_index(dates, inplace = True)
    return dates, df

def prescreening(station_list, input_dir, batch, threshold=0):
    df_screening = pd.DataFrame(columns=['station', 'years', 'years_rand', 'available'])
    cons_sequences_threshold = 2600 / batch
    batch = batch * 24
    for station in station_list:
        # station = 'abashiri-abashiri-japan-jma'
        station = os.path.split(station)[1][:-3]
        print(station)
        # load dataset
        df, ds, direction = load_file(station, input_dir)

        # insert ID's
        df = df.copy()
        df.insert(0, 'ID', range(0, len(df)))

        try:
            # check NaN in dataset (75% not NaN)
            if df['residual'].isnull().astype(int).sum() > 0:
                # count consecutive NaN
                na_count = df['residual'].isnull().astype(int).groupby(df['residual'].notnull().astype(int).cumsum()).cumsum()

                # count consecutive timesteps without consecutive NaN of more than 'threshold' days
                na_count[na_count > threshold] = np.nan
                value_count = na_count.notnull().astype(int).groupby(na_count.isnull().astype(int).cumsum()).cumsum()

                # find local max in series
                local_max = value_count[(value_count.shift(1) < value_count) & (value_count.shift(-1) < value_count)]
                if value_count[-1:].values[0] > 0:
                    new_data = pd.DataFrame(value_count[-1:].values, index=value_count.index.values[-1:])
                    local_max = local_max.append(new_data)

                # count consecutive years in series
                cons_years = local_max.copy()
                try:
                    cons_years = np.floor(cons_years / (365 * 24)).sum().values[0]
                except:
                    cons_years = np.floor(cons_years / (365 * 24)).sum()

                # count consecutive sequences in series
                cons_sequences = local_max.copy()
                try:
                    cons_sequences = np.floor(cons_sequences / (batch)).sum().values[0]
                except:
                    cons_sequences = np.floor(cons_sequences / (batch)).sum()
            
                # # count random consecutive years in series
                # pool = df.index.values
                # start = np.random.randint(0, 365 * 24)
                # end = 365 * 24 + start
                # cons_years_rand = 0
                # for i in range(int(len(df) / (365 * 24)) - 1):
                #     if df['residual'].loc[pool[start:end]].isnull().sum() < 365 * 24 * 0.25:
                #         cons_years_rand += 1
                #     start += 365 * 24
                #     end += 365 * 24

                # # count random consecutive sequences in series
                # pool = df.index.values
                # start = np.random.randint(0, batch)
                # end = batch + start
                # cons_sequences_rand = 0
                # for i in range(int(len(df) / batch) - 1):
                #     if df['residual'].loc[pool[start:end]].isnull().sum() < batch * 0.25:
                #         cons_sequences_rand += 1
                #     start += batch
                #     end += batch
            else:
                # dataset has no NaN values
                # count consecutive years in series
                cons_years = np.floor(len(df['residual']) / (365 * 24)).sum()

                # count consecutive sequences in series
                cons_sequences = np.floor(len(df['residual']) / (batch)).sum()

            # check if station suitable
            available = True if cons_years >= 1 and cons_sequences >= cons_sequences_threshold else False

            new_row = {'station':station, 'years':cons_years, 'sequences':cons_sequences,
                       'available':available}
            df_screening = df_screening.append(new_row, ignore_index=True)
        except AttributeError as e:
            new_row = {'station':station, 'years':0, 'sequences':0, 'available':False}
            df_screening = df_screening.append(new_row, ignore_index=True)
    
    print(df_screening)
    print(df_screening['available'].sum())
    return df_screening, df

def spatial_to_column(df, ds, variables, selected_dates, n_ncells):
    df = df.copy()
    df0 = df.copy()
    if len(selected_dates)>0:
        df = df.loc[selected_dates,:].copy()

    middle = int(len(ds.longitude)/2)
    lon_list = ds.longitude.values[middle - n_ncells: middle + n_ncells + 1]
    lat_list = ds.latitude.values[middle - n_ncells: middle + n_ncells + 1]

    # extract all gridded data to columns
    for lat in lat_list:
        df2 = df0.copy()
        for lon in lon_list:    
            if len(selected_dates)>0:
                dfi = ds[variables].sel(latitude=lat, longitude=lon, time=selected_dates).to_dataframe().drop(['latitude', 'longitude'], axis = 1)
            else:
                dfi = ds[variables].sel(latitude=lat, longitude=lon).to_dataframe().drop(['latitude', 'longitude'], axis = 1)
            dfi.columns = [col + '_{}_{}'.format(lat, lon) for col in dfi.columns]            
            df2 = df2.merge(dfi, how='left', right_index=True, left_index=True)
            del dfi
        df = df.merge(df2.drop(['tide_wtrend', 'gesla_swl','residual'], axis = 1), how='left', right_index=True, left_index=True)
    df.drop(['tide_wtrend', 'gesla_swl'], axis=1, inplace=True)
    return df, lat_list, lon_list

def column_to_spatial(reframed, columns, lat_list, lon_list, variables, ML, direction):
    reframed = reframed.copy()
    # rename columns
    cols = columns[1:]
    cols = np.append(cols, columns[0])
    reframed.columns = cols

    # create gridded data per variables
    reframed_list = []
    reframed_list.append(reframed['residual'].values)
    if ML == 'CNN' or ML == 'CNN_LSTM':
        reframed_empty = np.zeros((len(reframed), len(lat_list), len(lat_list), 1))  # time_train, lat, lon, 1
        rot_axes = (1, 2)
    elif ML == 'ConvLSTM':
        reframed_empty = np.zeros((len(reframed), 1, len(lat_list), len(lat_list), 1))  # time_train, 1, lat, lon, 1
        rot_axes = (2, 3)
    for var in variables:
        # find columns of variable
        df_var = reframed[list(reframed.filter(regex=var))]
        reframed_var = reframed_empty.copy()

        # store in empty grid
        for i, lat in enumerate(lat_list):
            for j, lon in enumerate(lon_list):
                if ML == 'CNN' or ML == 'CNN_LSTM':
                    reframed_var[:, i, j, 0] = df_var['{}_{}_{}'.format(var, lat, lon)].values
                elif ML == 'ConvLSTM':
                    reframed_var[:, 0, i, j, 0] = df_var['{}_{}_{}'.format(var, lat, lon)].values
        if direction != 'N':
            if direction == 'W':
                k = 1
            elif direction == 'S':
                k = 2
            elif direction == 'E':
                k = -1
            reframed_var = np.rot90(reframed_var, k=k, axes=rot_axes)
        reframed_list.append(reframed_var)
    return reframed_list

def split_tt(reframed, ML, tt_value, n_train):
    if ML == 'ANN':
        values = reframed.values
        if not n_train:
            n_train = int(values.shape[0] * tt_value)
        train = values[:n_train, :]
        test = values[n_train:, :]

        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
    
    if ML == 'LSTM' or ML == 'TCN':
        values = reframed.values
        if not n_train:
            n_train = int(values.shape[0] * tt_value)
            if tt_value == 1:
                n_train = n_train - 2
        train = values[:n_train, :]
        test = values[n_train:, :]

        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    elif ML == 'CNN' or ML == 'ConvLSTM' or ML == 'CNN_LSTM':
        if not n_train:
            n_train = int(reframed[0].shape[0] * tt_value)
            if tt_value == 1:
                n_train = n_train - 2

        # split into input and outputs
        train_y, test_y = reframed[0][:n_train], reframed[0][n_train:]
        train_X = [var[:n_train, :, :, :] for var in reframed[1:]]
        test_X = [var[n_train:, :, :, :] for var in reframed[1:]]

    return train_X, train_y, test_X, test_y, n_train

def prepare_station(station, variables, ML, input_dir, resample, resample_method,
                    cluster_years=5, extreme_thr=0.02, sample=False, make_univariate=False,
                    scaler_type='std_normal', year = 'last', scaler_op=True, n_ncells=2, mask_val=-999, logger=False):
    start = time.time()    

    # read in variables
    df, ds, direction = load_file(station, input_dir)
    # print('Station is loaded')

    if sample == 'cluster':
        print(f'Selecting {cluster_years} years of data')
        selected_dates, df = draw_sample(df, cluster_years * 365 * 24, 24, threshold=7)
    elif sample == 'extreme':
        print(f'Selecting top {extreme_thr} of data')
        selected_dates, df = select_extremes(df, extreme_thr)
    else:
        selected_dates = []

    df, lat_list, lon_list = spatial_to_column(df, ds, variables, selected_dates, n_ncells)
    # print('df to spatial done')    

    # Drop missing values
    df.dropna(inplace=True)
    
    # resample or rolling mean
    df, step = resample_rolling(df, lat_list, lon_list, variables, resample, resample_method, make_univariate)
    # df = df[df['residual'].notna()].copy()
    # print('Resampling done')  
    
    timesteps = int(365 * step)

    # reframe and scale data
    reframed, scaler, scaled, dates, i_dates = reframe_scale(df, timesteps, scaler_type=scaler_type, year = year, scaler_op=scaler_op)
    # reframed_df = reframed.copy()

    if logger:
        logger.info(f'done preparing data for {station}: {time.time()-start: .2f} sec')
    else:
        print(f'done preparing data: {time.time()-start: .2f} sec\n')
    return df, lat_list, lon_list, direction, scaler, reframed, dates, i_dates

def splitting_learning(reframed, df, tt_value, ML, variables, direction, lat_list, lon_list, batch, n_train=False):
    # start = time.time()
        
    # df to 2d
    if ML == 'CNN' or ML == 'ConvLSTM' or ML == 'CNN_LSTM':
        reframed = column_to_spatial(reframed, df.columns, lat_list, lon_list, variables, ML, direction)

    # split into train and test sets
    train_X, train_y, test_X, test_y, n_train = split_tt(reframed, ML, tt_value, n_train)

    return train_X, train_y, test_X, test_y, n_train
