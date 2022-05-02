# -*- coding: utf-8 -*-
"""
Using machine learning to predict coastal hydrographs

Timothy Tiggeloven and Anaïs Couasnon
"""
import logging
import logging.handlers
import os
import sys
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import time

from CHNN_parametrization import ANN
from CHNN_parametrization import to_learning
from CHNN_parametrization import performance


def set_logger(arg_start, arg_end, verbose=True):
    """
    Set-up the logging system, exit if this fails
    """
    # assign logger file name and output directory
    datelog = time.ctime()
    datelog = datelog.replace(':', '_')
    reference = f'ML_stormsurges_loop_{arg_start}-{arg_end}'


    logfilename = ('logger' + os.sep + reference + '_logfile_' +
                   str(datelog.replace(' ', '_')) + '.log')

    # create output directory if not exists
    if not os.path.exists('logger'):
        os.makedirs('logger')

    # create logger and set threshold level, report error if fails
    try:
        logger = logging.getLogger(reference)
        logger.setLevel(logging.DEBUG)
    except IOError:
        sys.exit('IOERROR: Failed to initialize logger with: ' + logfilename)

    # set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s -'
                                  '%(levelname)s - %(message)s')

    # assign logging handler to report to .log file
    ch = logging.handlers.RotatingFileHandler(logfilename,
                                              maxBytes=10*1024*1024,
                                              backupCount=5)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # assign logging handler to report to terminal
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # start up log message
    logger.info('File logging to ' + logfilename)

    return logger, ch


def ensemble_handler(arg_start, arg_end, resample, resample_method, variables, tt_value, year, scaler_type, n_ncells, epochs, frac_ens, batch, batch_normalization, loop, neurons, filters, n_layers, activation, loss, optimizer, dropout, drop_value, l1, l2, ML, mask_val, input_dir, fn_exp):

    logger, ch = set_logger(arg_start, arg_end)

    # load prescreening check
    df_prescreening = pd.read_csv('prescreening_station_parametrization.csv')
    station_list = df_prescreening['station'][df_prescreening['available'] == True].values
    arg_count = arg_start

    print_summary(resample, resample_method, variables, tt_value, scaler_type, n_ncells, epochs, loop, frac_ens, batch, batch_normalization, neurons,
                  filters, n_layers, activation, loss, dropout, drop_value, l1, l2, input_dir, fn_exp, mask_val, year)

    for station in station_list[arg_start: arg_end]:
        logger.info(f'Start ML ensemble run for station {arg_count}: {station}')
        ensemble(station, variables, ML, tt_value, input_dir, resample, resample_method, scaler_type,
                batch, n_layers, neurons, filters, dropout, drop_value, activation, optimizer,
                batch_normalization, loss, epochs, loop=loop, n_ncells=n_ncells, l1=l1, l2=l2, frac_ens = frac_ens, logger=logger,
                year=year, fn_exp=fn_exp, arg_count=arg_count, verbose=0, mask_val=mask_val)
        arg_count += 1



def print_summary(resample, resample_method, variables, tt_value, scaler_type, n_ncells, epochs, loop, frac_ens, batch, batch_normalization, neurons,
                  filters, n_layers, activation, loss, dropout, dropout_value, l1, l2, input_dir, fn_exp, mask_val, year):

    summary = {'timestep': resample,
               'resample_method': resample_method,
               'selected_variables': list(variables),
               'split_train_validation': tt_value,
               'n_ncells': n_ncells,
               'scaling': scaler_type,
               'epochs': epochs,
               'loop': loop,
               'frac_ens': frac_ens,
               'batch_size_days': batch,
               'batch_normalization': batch_normalization,
               'neurons': neurons,
               'filters': filters,
               'n_layers': n_layers,
               'activation': activation,
               'loss': loss,
               'dropout': dropout,
               'dropout_value': dropout_value,
               'l1': l1,
               'l2': l2,
               'input_nc_dir': input_dir,
               'mask_val': mask_val,
               'testing_selection': year}

    summary_df = pd.DataFrame.from_dict(summary, orient = 'index', columns = ['value'])
    summary_df.to_csv(os.path.join(fn_exp, 'summary_experiment_parametrization.csv'),  index_label='parameter')

def ensemble(station, variables, ML, tt_value, input_dir, resample, resample_method, scaler_type,
             batch, n_layers, neurons, filters, dropout, drop_value, activation, optimizer,
             batch_normalization, loss, epochs, loop=5, n_ncells=2, l1=0.00, l2=0.01, frac_ens = 0.5, logger=False,
             year='last', fn_exp='Models', arg_count=0, verbose=2, mask_val=-999):

    start1 = time.time()
    if ML.lower() == 'all':
        ML_list = ['ANN', 'CNN', 'LSTM', 'ConvLSTM']
    else:
        ML_list = [ML]

    df, lat_list, lon_list, direction, scaler, reframed, test_dates, i_test_dates = to_learning.prepare_station(station, variables, ML, input_dir, resample, resample_method,
                                                                                                                cluster_years=5, extreme_thr=0.02, sample=False, make_univariate=False,
                                                                                                                scaler_type=scaler_type, year = year, scaler_op=True, n_ncells=n_ncells, mask_val=mask_val, logger=logger)
    if resample == 'hourly':
        batch = batch * 24

    # split testing phase year
    test_year = reframed.iloc[i_test_dates].copy()

    #NaN masking the complete test year
    reframed.iloc[i_test_dates] = np.nan

    for ML in ML_list:
        if not logger:
            print(f'\nStart ensemble run for {ML}\n')
        start2 = time.time()

        # create model output directory
        #model_dir = os.path.join('Models', 'Ensemble_run', station, ML)
        model_dir = os.path.join(fn_exp, 'Ensemble_run', station, ML)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # reframe ML station data
        test_year.loc[test_year.iloc[:,-1].isna(),'values(t)'] = mask_val #Changing all NaN values in residual testing year to masking_val
        _, _, test_X, test_y, _ = to_learning.splitting_learning(test_year, df, 0, ML, variables, direction, lat_list, lon_list, batch, n_train=False)

        result_all = dict()
        result_all['data'] = dict()
        result_all['train_loss'] = dict()
        result_all['test_loss'] = dict()
        for i in range(loop):
            if not logger:
                print(f'\nEnsemble loop: {i + 1}\n')
            tf.keras.backend.clear_session()
            name_model = f'{ML}_ensemble_{i + 1}'

            # shuffle df
            reframed_ensemble = reframed.copy()
            reframed_draw, n_train = to_learning.select_ensemble(reframed_ensemble, 'values(t)', ML, batch, tt_value=tt_value, frac_ens = frac_ens, mask_val=mask_val)

            # We modify the input data so that it is masked
            reframed_draw = reframed_draw.reset_index(drop=True).copy()
            reframed_draw[reframed_draw.iloc[:,-1]==mask_val] = mask_val
            print('There are so many Nan: ', sum(reframed_draw.iloc[:,-1]==mask_val))

            # reframe ML station data
            train_X, train_y, val_X, val_y, n_train = to_learning.splitting_learning(reframed_draw, df, tt_value, ML, variables, direction, lat_list, lon_list, batch, n_train=n_train)

            # design network
            model = ANN.design_network(n_layers, neurons, filters, train_X, dropout, drop_value, variables,
                                       batch_normalization, name_model, ML=ML, loss=loss,
                                       optimizer=optimizer, activation=activation, l1=l1, l2=l2, mask_val=mask_val)

            # fit network
            model, train_loss, test_loss = ANN.train_model(model, epochs, batch, train_X, train_y, val_X,
                                                           val_y, ML, name_model, model_dir, validation='select', verbose=verbose)
            result_all['train_loss'][i] = train_loss
            result_all['test_loss'][i] = test_loss

            # make a prediction
            inv_yhat, inv_y = ANN.predict(model, test_X, test_year.replace(to_replace=mask_val, value=np.nan), scaler, 0)

            # plot results
            df_all = performance.store_result(inv_yhat, inv_y)
            df_all = df_all.set_index(df.iloc[i_test_dates,:].index, drop = True)

            result_all['data'][i] = df_all.copy()

            if os.path.exists(os.path.join(model_dir, name_model)):
                os.remove(os.path.join(model_dir, name_model))
            model.save(os.path.join(model_dir, name_model), include_optimizer=True) # , overwrite=True
            del(model)

            tf.keras.backend.clear_session()
            keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()

        #model = keras.models.load_model(os.path.join(model_dir, name_model))

        # plot results
        df_result, df_train, df_test =  performance.ensemble_handler(result_all, station, neurons, epochs,
                                                                    batch, resample, tt_value, len(variables),
                                                                    model_dir, layers=n_layers, ML=ML,
                                                                    test_on='ensemble', plot=True, save=True)

        if logger:
            logger.info(f'{arg_count}: {ML} - {station} - {round((time.time() - start2) / 60, 2)} min')
        else:
            print(f'\ndone ensemble run for {ML}: {round((time.time() - start2) / 60, 2)} min\n')

    if logger:
        logger.info(f'{arg_count}: Done - {station} - {round((time.time() - start2) / 60, 2)} min')
        return None
    else:
        print(f'\ndone ensemble runs for {ML_list}: {round((time.time() - start1) / 60, 2)} min\n')
        return df_result, df_train, df_test