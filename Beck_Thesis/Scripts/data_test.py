import sys
import os
sys.path.append(os.path.join(sys.path[0], r'./Scripts/'))

from Scripts import to_learning as tl, model_run_coast as mrc

resample = 'hourly' # 'hourly' 'daily'
resample_method = 'rolling_mean'  # 'max' 'res_max' 'rolling_mean' ## res_max for daily and rolling_mean for hourly
variables = ['msl', 'grad', 'u10', 'v10', 'rho', 'sst']  # 'grad', 'rho', 'phi', 'u10', 'v10', 'uquad', 'vquad'
tt_value = 0.67  # train-test value
scaler = 'std_normal'  # std_normal, MinMax
n_ncells = 2 # 2 = 5x5, 3=7x7
epochs = 100
batch = 10
batch_normalization = False
neurons = 48
filters = 8
n_layers = 3  
activation = 'relu'  # 'relu', 'swish', 'Leaky ReLu', 'sigmoid', 'tanh'
optimizer = 'adam'  # SGD(lr=0.01, momentum=0.9), 'adam'
dropout = True
drop_value = 0.2
l1, l2 = 0, 0.01

model_dir = os.path.join(os.getcwd(), 'Models')
name_model = f'{coast}_{ML}_{loss}'
input_dir = 'Input_nc_detrend_sst' # 'Input_nc_detrend_sst'
output_dir = 'Models'
figures_dir = 'Figures'
year = 'last'
frac_ens = 0.5

loop = 2
gamma = 1.2

logger, ch = mrc.set_logger(loop, n_ncells)

ML = 'ANN'

for coast in ['NE_Atlantic_2', "NE_Pacific"]:
    stations = tl.get_all_station_data(coast, variables, ML, input_dir, resample, resample_method, batch, scaler, year, n_ncells, -999, tt_value, frac_ens, 0, logger, model_dir)