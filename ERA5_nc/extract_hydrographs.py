# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:49:25 2019

@author: ttn430
"""

#import cdsapi
import glob
import logging
import logging.handlers
import matplotlib.dates as mdates
from multiprocessing import Process, Manager, Pool
import numpy as np
import pandas as pd
import os
import sys
import time
#import utide
import xarray as xr

import warnings
warnings.filterwarnings('ignore')


def set_logger(verbose=True):
    """
    Set-up the logging system, exit if this fails
    """
    # assign logger file name and output directory
    datelog = time.ctime()
    datelog = datelog.replace(':', '_')
    reference = 'Coastal_hydrographs'


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


def GESLA_data():
    mismatch = []
    for file in glob.glob('GESLA/*'):
        print(file)
        start = time.time()
        df = pd.read_table(file, header=None, encoding="ISO-8859-1")
        data = pd.DataFrame(df[0].iloc[32:].str.split().tolist(), columns=['date', 'time',
                            'Sea level', 'QC flag', 'EA flag'])

        df = df.iloc[:32]
        meta = {}
        meta['Site name'] = df[df[0].str.contains("SITE NAME")].values.flatten()[0].split()[3:][0]
        meta['Country'] = df[df[0].str.contains("COUNTRY")].values.flatten()[0].split()[2:][0]
        meta['Latitude'] = df[df[0].str.contains("LATITUDE")].values.flatten()[0].split()[2:][0]
        meta['Longitude'] = df[df[0].str.contains("LONGITUDE")].values.flatten()[0].split()[2:][0]
        meta['CS'] = df[df[0].str.contains("COORDINATE SYSTEM")].values.flatten()[0].split()[3:][0]
        meta['Tz'] = df[df[0].str.contains("TIME ZONE HOURS")].values.flatten()[0].split()[4:][0]
        meta['St'] = df[df[0].str.contains("START DATE/TIME")].values.flatten()[0].split()[3:4][0]
        meta['End'] = df[df[0].str.contains("END DATE/TIME")].values.flatten()[0].split()[3:4][0]

        if data['date'].iloc[0] != meta['St']:
            mismatch.append(file)

        data['Timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data = data.set_index('Timestamp')
        data = data.drop(['date', 'time'], axis=1)
        data = data.astype(float)

        flagged = data['QC flag'] > 2.0
        data['Sea level'][flagged] = np.nan
        data['anomaly'] = data['Sea level'] - data['Sea level'].mean()
        # data['anomaly'] = data['anomaly'].interpolate()
        t = mdates.date2num(data.index.to_pydatetime())
        print('{} points were flagged 3-5'.format(flagged.sum()))

        coef = utide.solve(t, data['anomaly'].values, lat=float(meta['Latitude']), method='ols',
                           conf_int='linear', constit='auto', trend=True, phase='Greenwich',
                           nodal=True)
        tide = utide.reconstruct(t, coef)
        data['tide'] = tide.h
        data['residue'] = data.anomaly - tide.h
        data = data.resample('h').mean()
        data.to_csv('Utide' + os.sep + f'Utide_{os.path.split(file)[1]}.csv')
        print(f'Calculation took {time.time() - start} seconds')

    df = pd.DataFrame(mismatch[:], columns=['Mismatch starting date'])
    datelog = time.ctime().replace(':', '-').replace(' ', '_')
    df.to_csv(f'Mismatch_starting_date_{datelog}.csv')


def ERA5_prep():
    ds_fids = []
    for file in glob.glob('GESLA/*'):
        start = time.time()
        print(file)
        df = pd.read_table(file, header=None, encoding="ISO-8859-1")
        data = pd.DataFrame(df[0].iloc[32:].str.split().tolist(), columns=['date', 'time',
                            'Sea level', 'QC flag', 'EA flag'])

        df = df.iloc[:32]
        meta = {}
        meta['Latitude'] = df[df[0].str.contains("LATITUDE")].values.flatten()[0].split()[2:][0]
        meta['Longitude'] = df[df[0].str.contains("LONGITUDE")].values.flatten()[0].split()[2:][0]

        data['Timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data = data.set_index('Timestamp')
        data = data.drop(['date', 'time'], axis=1)
        data = data.astype(float)

        pr_lon = PartialRound(float(meta['Longitude']), 0.25)
        pr_lat = PartialRound(float(meta['Latitude']), 0.25)
        area = [pr_lat, pr_lon, pr_lat, pr_lon]

        years = data.index.year.unique().values.tolist()
        years = [i for i in years if i >= 1979]
        years = list(map(str, years))
        months = data.index.month.unique().values.tolist()
        months.sort()
        months = [str(i).zfill(2) for i in months]
        year_slice = 4
        for i in range(len(years) // year_slice):
            trunc_years = years[i * year_slice: min(year_slice + i * year_slice, len(years))]
            ds_trans = xr.Dataset({'Station': os.path.split(file)[1] + f'_{i}',
                                   'years': (['yrs'], trunc_years),
                                   'months': (['mths'], months),
                                   'area': (['coords'], area)})
            ds_fids.append(ds_trans)
        print(f'Preparation took {time.time() - start} seconds')

    ds = xr.concat(ds_fids, 'fid')
    ds.to_netcdf('ERA5_requests.nc')


def ERA5_requests_per_site(fid, failed):
    station = ds.isel(fid=fid)['Station'].values.tolist()
    years = ds.isel(fid=fid)['years'].values.tolist()
    months = ds.isel(fid=fid)['months'].values.tolist()
    area = ds.isel(fid=fid)['area'].values.tolist()
    era5_file = 'ERA5' + os.sep + f'ERA-5_{station}.nc'
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind',
                             'mean_sea_level_pressure'],
                'year': years,
                'month': months,
                'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                        '13', '14', '15',  '16', '17', '18', '19', '20', '21', '22', '23', '24',
                        '25', '26', '27', '28', '29', '30', '31'],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
                         '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
                         '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                'area': area,
            },
            era5_file)
        print(f'Finished fid {fid}: {station}')
    except:
        print(f'Failed fid {fid}: {station}')
        failed.append(station)


def ERA5_requests_per_site_pool(fid):
    station = ds.isel(fid=fid)['Station'].values.tolist()
    years = ds.isel(fid=fid)['years'].values.tolist()
    months = ds.isel(fid=fid)['months'].values.tolist()
    area = ds.isel(fid=fid)['area'].values.tolist()
    era5_file = 'ERA5' + os.sep + f'ERA-5_{station}.nc'
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind',
                             'mean_sea_level_pressure'],
                'year': years,
                'month': months,
                'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                        '13', '14', '15',  '16', '17', '18', '19', '20', '21', '22', '23', '24',
                        '25', '26', '27', '28', '29', '30', '31'],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
                         '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
                         '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                'area': area,
            },
            era5_file)
        print(f'Finished fid {fid}: {station}')
        failed = False
    except:
        print(f'Failed fid {fid}: {station}')
        failed = True
    return [fid, failed]


def ERA5_requests_full(years, var):
    era5_file = 'U:\\ERA5_full' + os.sep + f'ERA-5_full_{years[0]}-{years[-1]}_{var}.nc'
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [var],
                'year': years,
                'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                        '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                        '25', '26', '27', '28', '29', '30', '31'],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
                         '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
                         '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
            },
            era5_file)
        print(f'Finished years: {years}')
        failed = False
    except:
        print(f'Failed years: {years}')
        failed = True
    return [years, failed]


def Merge_data(file):
    ds = xr.open_dataset('ERA5' + os.sep + f'ERA-5_{os.path.split(file)[1]}.nc')
    gesla = pd.read_csv(f'Utide/Utide_{os.path.split(file)[1]}.csv')
    era_data = pd.to_datetime(ds['time'].values).to_frame(name='msl')
    era_data.truncate(before=gesla.index.min(), after=gesla.index.max())
    era_data['msl'] = ds['msl'].values.flatten()
    era_data = era_data.truncate(before=gesla.index.min(), after=gesla.index.max())
    merged = gesla.merge(era_data, left_index=True, right_index=True)
    merged.to_csv('Merged' + os.sep + f'merged_{os.path.split(file)[1]}.csv')


def PartialRound(value, resolution):
    return round(value / resolution) * resolution


def ERA5_per_site_parallel():
    starttime = time.time()
    num = 10
    ds_blocks = []
    for block in range(29, int(round(len(ds['fid']) / num) + 1)):
        fid_list = range(block * num, min(block * num + num, len(ds['fid']) + 1))
        print('--------------------------------------')
        print(f'FID BLOCK: {fid_list[0]}-{fid_list[-1]}')
        print('--------------------------------------')
        with Manager() as manager:
            failed = manager.list([])
            processes = []
            for i in fid_list:
                p = Process(target=ERA5_requests_per_site, args=(i, failed))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()

            ds_blocks.extend(failed[:])

    df = pd.DataFrame(ds_blocks, columns=['Failed station requests'])
    datelog = time.ctime().replace(':', '-').replace(' ', '_')
    df.to_csv(f'Failed_station_requests_{datelog}.csv')

    print('That took {} hours'.format((time.time() - starttime) / 3600))


def ERA5_per_site_parallel_pool():
    with Pool(5) as p:
        reslist = [p.apply_async(ERA5_requests_per_site_pool,
                                 (r,)) for r in range(1506, len(ds['fid']))]
        out = []
        for result in reslist:
            out.append(result.get())
        df = pd.DataFrame()
        df['FID'] = [i[0] for i in out]
        df['Failed'] = [i[1] for i in out]


def ERA5_full_parallel():
    starttime = time.time()
    year_inst = [['1979', '1980', '1981', '1982'],
                 ['1983', '1984', '1985', '1986', '1987', '1988'],
                 ['1989', '1990', '1991', '1992', '1993', '1994'],
                 ['1995', '1996', '1997', '1998', '1999', '2000'],
                 ['2001', '2002', '2003', '2004', '2005', '2006'],
                 ['2007', '2008', '2009', '2010', '2011', '2012'],
                 ['2013', '2014', '2015', '2016', '2017', '2018']]
    u10 = ['10m_u_component_of_wind'] * len(year_inst)
    v10 = ['10m_v_component_of_wind'] * len(year_inst)
    msl = ['mean_sea_level_pressure'] * len(year_inst)
    var_list = []
    var_list.extend(u10)
    var_list.extend(v10)
    var_list.extend(msl)
    year_list = []
    year_list.extend(year_inst)
    year_list.extend(year_inst)
    year_list.extend(year_inst)
    print('----------------------------------------------------')
    print(f'Start full requests: {year_list[0]}-{year_list[-1]}')
    print('----------------------------------------------------')
    with Pool(5) as p:
        reslist = [p.apply_async(ERA5_requests_full, (r, j,)) for r, j in zip(year_list, var_list)]
        out = []
        for result in reslist:
            out.append(result.get())
        print(out, '\n')
        df = pd.DataFrame()
        # df['FID'] = [i[0] for i in out]
        # df['Failed'] = [i[1] for i in out]

    print('That took {} hours'.format((time.time() - starttime) / 3600))


def ERA5_requests_update():
    # for duplicate area and years
    num = 4
    df = pd.DataFrame()
    df['lon'] = [i[0] for i in ds.area.values]
    df['lat'] = [i[1] for i in ds.area.values]
    df['start'] = [int(i[0]) for i in ds.years.values]
    df['end'] = [int(i[-1]) for i in ds.years.values]
    start_list = df.groupby(['lat', 'lon']).agg({'start': min})
    end_list = df.groupby(['lat', 'lon']).agg({'end': max})
    area_list = []
    year_list = []
    for i in start_list.index:
        area = [i[1], i[0], i[1], i[0]]
        for j in range(start_list.loc[i]['start'], end_list.loc[i]['end'], num):
            area_list.append(area)
            year_tolist = []
            for k in range(j, end_list.loc[i]['end'] - j):
                year_tolist.append(str(k))
            year_list.append(year_tolist)
    print(len(area_list))


def merge_era5_utide():
    years_file = ['1979-1982', '1983-1988', '1989-1994', '1995-2000', '2001-2006', '2007-2012',
                  '2013-2018']
    years_range = ['1979-1980-1981-1982', '1983-1984-1985-1986-1987-1988',
                   '1989-1990-1991-1992-1993-1994', '1995-1996-1997-1998-1999-2000',
                   '2001-2002-2003-2004-2005-2006', '2007-2008-2009-2010-2011-2012',
                   '2013-2014-2015-2016-2017-2018']
    var_short_list = ['msl', 'u10', 'v10']
    var_long_list = ['mean_sea_level_pressure', '10m_u_component_of_wind',
                     '10m_v_component_of_wind']
    
    # for u_file in glob.glob('Utide/*'):
    for u_file in ['Utide_abidjan_vridi-230a-ivory_coast-uhslc.csv']:
        starttime = time.time()
        logger.info(f'Start merging data for {u_file[6:-4]}')
        df = pd.read_csv(os.path.join('Utide', u_file), parse_dates=True, index_col='Timestamp')
        yearmin = df.index.year.min()
        yearmax = df.index.year.max()
        indexmin = np.where([f'{yearmin}' in i for i in years_range])[0][0]
        indexmax = np.where([f'{yearmax}' in i for i in years_range])[0][0]
        logger.info(f'  -- Year range total: {yearmin}-{yearmax}')
    
        gesla_file = os.path.join('GESLA', u_file[6:-4])
        gesla = pd.read_table(gesla_file, header=None, encoding="ISO-8859-1").head(32)
        meta = {}
        meta['Lat'] = gesla[gesla[0].str.contains("LATITUDE")].values.flatten()[0].split()[2:][0]
        meta['Lon'] = gesla[gesla[0].str.contains("LONGITUDE")].values.flatten()[0].split()[2:][0]
        pr_lon = PartialRound(float(meta['Lon']), 0.25) + 180
        pr_lat = PartialRound(float(meta['Lat']), 0.25)
        
        for var_short in var_short_list:
            df[var_short] = np.nan
        
        for i in range(indexmin, indexmax + 1):
            years = years_file[i]
            logger.info(f'  -- Year range iteration: {years}')
            for var_short, var_long in zip(var_short_list, var_long_list):
                logger.info(f'  -- ERA var: {var_short}')
                era5_file = 'ERA5_nc' + os.sep + f'ERA-5_full_{years}_{var_long}.nc'
                era5 = xr.open_dataset(era5_file)
                df_years = df.truncate(before=pd.Timestamp(f'{years[:4]}-01-01'),
                                       after=pd.Timestamp(f'{years[5:]}-12-31 23:00:00')).index
                starttime1 = time.time()
                var = era5[var_short].sel(latitude=pr_lat, longitude=pr_lon, 
                                          time=slice(df_years.min(), df_years.max()))
                condition =(df.index >= df_years.min()) & (df.index <= df_years.max())
                df[var_short][condition] = var.values
                
                logger.info('  -- loading and writer took {} minutes'
                            .format((time.time() - starttime1) / 60))
        df.to_csv(os.path.join('Merged', f'Merged_{u_file[6:-4]}.csv'))
        logger.info('  -- Station finished in {} minutes'.format((time.time() - starttime) / 60))
        logger.info('#################################################')

#c = cdsapi.Client()
logger, ch = set_logger()
ds = xr.open_dataset('ERA5_requests.nc')

if __name__ == '__main__':
    merge_era5_utide()
    # ERA5_full_parallel()
    # ERA5_per_site_parallel_pool()
