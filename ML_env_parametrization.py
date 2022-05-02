# -*- coding: utf-8 -*-
"""
Using machine learning to predict coastal hydrographs

Timothy Tiggeloven and Anaïs Couasnon
"""
import os
import sys

from CHNN_parametrization import ANN
from CHNN_parametrization import to_learning
from CHNN_parametrization import performance
from CHNN_parametrization import model_run

#Reading from the bash script - stations
arg_start = int(sys.argv[1])
arg_end = int(sys.argv[2])

# parameters and variables
resample = sys.argv[3]           #'hourly' # 'hourly' 'daily'
resample_method = sys.argv[4]     #'raw'  # 'max' 'res_max' 'rolling_mean' ## res_max for daily and rolling_mean for hourly
tt_value = float(sys.argv[5])     #0.70  # train-test value
year = sys.argv[6]               # 'last'
scaler_type = sys.argv[7]       #'std_normal'  # std_normal, MinMax
n_ncells = int(sys.argv[8])     #2
epochs =  int(sys.argv[9])     #150
frac_ens = float(sys.argv[10])   # 1.0
batch = int(sys.argv[11])         # 100
batch_normalization = [False, True][sys.argv[12] == 'True']  #False 
loop = int(sys.argv[13])          # 1
neurons = int(sys.argv[14])      # 48
filters = int(sys.argv[15])      # 8
n_layers = int(sys.argv[16])      # 1
activation = sys.argv[17]         #'relu'  # 'relu', 'swish', 'Leaky ReLu', 'sigmoid', 'tanh'
loss = sys.argv[18]              # 'mae'  # 'mae', 'mean_squared_logarithmic_error', 'mean_squared_error'
optimizer = sys.argv[19]         #'adam'  # SGD(lr=0.01, momentum=0.9), 'adam'
dropout = [False, True][sys.argv[20] == 'True']  #True
drop_value = float(sys.argv[21]) # 0.2
l1 = float(sys.argv[22]) 
l2 = float(sys.argv[23])           #0.01
ML = sys.argv[24]               # 'ALL'  # 'LSTM', 'CNN', 'ConvLSTM', 'ANN', 'ALL'
mask_val = int(sys.argv[25])    # -999
input_dir = sys.argv[26]         # 'Input_nc_linear'
fn_exp = sys.argv[27]            # 'Models_Exp30'
variables = sys.argv[28:]           #['msl', 'grad', 'u10', 'v10', 'rho']  # 'grad', 'rho', 'phi', 'u10', 'v10', 'uquad', 'vquad'

model_res = os.path.join('Sensitivity',fn_exp)
print(model_res)

#if model_res == "Sensitivity/Models_Exp1":
#    print('This IS working')
#else:
#    print("This is NOT working")
    
if not os.path.exists(model_res):
    os.makedirs(model_res)

model_run.ensemble_handler(arg_start, arg_end, resample, resample_method, variables, tt_value, year, scaler_type, n_ncells, epochs, frac_ens, batch, batch_normalization, loop, neurons, filters, n_layers, activation, loss, optimizer, dropout, drop_value, l1, l2, ML, mask_val, input_dir, model_res)

sys.exit(0)
