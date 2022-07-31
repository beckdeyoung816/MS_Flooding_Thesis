import logging
import logging.handlers
import os
from posixpath import dirname
import sys
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import time
from datetime import date
import xarray as xr
import random

import sys
sys.path.append(os.path.join(sys.path[0], r'./Scripts/'))

from Scripts import to_learning as tl, performance
from Scripts.Coastal_Model import Coastal_Model
from Scripts.station import Station

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def ensemble(coast, variables, ML, tt_value, input_dir, resample, resample_method, scaler_type,
             batch, n_layers, neurons, filters, dropout, drop_value, activation, optimizer,
             batch_normalization, loss, epochs, loop=5, n_ncells=2, l1=0.00, l2=0.01, frac_ens=0.5, logger=False, complexity=False,
             year='last', fn_exp='Models', arg_count=0, verbose=2, mask_val=-999, hyper_opt=False, NaN_threshold=0, validation='split', gamma=1.1, note=''):

    start1 = time.time()

    if isinstance(ML, list): # If ML is already a list of MLs, then keep it as a list
        ML_list = ML
    elif ML.lower() == 'all':
        ML_list = ['ANN', 'LSTM', 'TCN', 'TCN-LSTM']
    else:
        ML_list = [ML]
        
    print('ML_list is:', ML_list)
    
    if resample == 'hourly':                            
        batch = batch * 24
    
    for ML in ML_list: # Loop over each type of ML model
        
        if not logger:
            print(f'\nStart ensemble run for {ML}\n')
            print('\n\n************************************************************************************\n\n')
        print(f'\nStart ensemble run for {ML}\n')
        print('\n\n************************************************************************************\n\n')
        start2 = time.time()

        # create model output directory
        model_dir = os.path.join(fn_exp, 'Ensemble_run', coast, ML)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        #stations = tl.get_all_station_data(coast, variables, ML, input_dir, resample, resample_method, batch, scaler_type, year, n_ncells, mask_val, tt_value, frac_ens, NaN_threshold, logger)
        train_stations, test_stations = tl.get_coast_stations(coast)
        random.shuffle(train_stations)      
                             
        for i in range(loop): # Loop is number of models in the ensemble
            if not logger:
                print(f'\nEnsemble loop: {i + 1}\n')
            tf.keras.backend.clear_session()
            name_model = f'{ML}_{loss}_ensemble_{i + 1}'

            # Initiliaze and run the model
            sherpa_output=None # Quick fix to ignore hyperparameter tuning
            model = Coastal_Model(train_stations, test_stations, ML, loss, n_layers, neurons, activation, dropout, drop_value,
                                      hyper_opt, validation, optimizer, epochs, batch, verbose, model_dir, filters,
                                      variables, batch_normalization, sherpa_output, logger, name_model,
                                      alpha=None, s=None, gamma=gamma, l1=l1, l2=l2, mask_val=mask_val, n_ncells=n_ncells)
                             
            model.design_network()
            model.compile()
            model.train_model(ensemble_loop=i)
            model.predict(ensemble_loop=i)
            
            # Store and delete model from memory
            model.model.save(os.path.join(model_dir, name_model), include_optimizer=True , overwrite=True)
            del(model.model)
            
            tf.keras.backend.clear_session()
            keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
        
        if not hyper_opt:
            # plot results for each station
            for station in model.station_inputs.values():
                df_result, df_train, df_test =  performance.ensemble_handler(station, station.result_all, station.name, neurons, epochs, 
                                                                            batch, resample, tt_value, len(variables), 
                                                                            model_dir, layers=n_layers, ML=ML, 
                                                                            test_on='ensemble', plot=True, save=True, loss=loss)
            
            # Get results for the entire coastline
            all_stations = model.station_inputs.values()
            coast_train_results = performance.get_coastline_results([station for station in all_stations if station.train_test == 'Train'])
            coast_test_results = performance.get_coastline_results([station for station in all_stations if station.train_test == 'Test'])
            print(f'\nCoastline train results:\n {coast_train_results}')
            print(f'\n\nCoastline test results:\n {coast_test_results}')
            
            # Store coastline results
            # os.makedirs(os.path.join(model_dir, 'Results'), exist_ok=True)
            # coast_train_results.to_csv(os.path.join(model_dir, f'Results/train_coast_results_{ML}_{loss}.csv'))
            # coast_test_results.to_csv(os.path.join(model_dir, f'Results/test_coast_results_{ML}_{loss}.csv'))
            
            
            res = os.path.join(fn_exp, 'Results')
            os.makedirs(res, exist_ok=True)
            res_path = f'{res}/results-{coast}-{date.today().strftime("%m-%d")}-({note}).xlsx'
            if not os.path.exists(res_path):
                writer=pd.ExcelWriter(res_path, mode='w')
            else:
                writer = pd.ExcelWriter(res_path, mode='a', if_sheet_exists='replace')
            coast_test_results.to_excel(writer, sheet_name=f'{ML}_{loss}_test')
            coast_train_results.to_excel(writer, sheet_name=f'{ML}_{loss}_train')
            writer.save()
            
        if logger:
            logger.info(f'{arg_count}: {ML} - {coast} - {round((time.time() - start2) / 60, 2)} min')
        else:
            print(f'\ndone ensemble run for {ML}: {round((time.time() - start2) / 60, 2)} min\n')
    
    if logger:
        logger.info(f'{arg_count}: Done - {coast} - {round((time.time() - start1) / 60, 2)} min')
        return None
    else:
        print(f'\ndone ensemble runs for {ML_list}: {round((time.time() - start1) / 60, 2)} min\n')
        return df_result, df_train, df_test
