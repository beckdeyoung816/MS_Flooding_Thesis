'''
Figure Making
'''
# %%

from cProfile import label
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import to_learning
import json
import os

os.chdir('/Users/beck/My Drive/VU/Thesis/Scripts/Beck_Thesis/')

import Scripts.performance as performance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
os.chdir('..')

# %%
coast = 'NE_Atlantic_1'
train_stations, test_stations = to_learning.get_coast_stations(coast)

# %%
def get_coast_results(coast):
    results = pd.DataFrame()
    for ML in ['ANN', 'LSTM', 'TCN', 'TCN-LSTM']:
        for loss in ['Gumbel', 'mse']:
            for train in ['train', 'test']:
                try:
                    tmp = (pd.read_excel(f'Models/Results/results-{coast}-(5b5_gam1.1).xlsx', 
                                sheet_name=f'{ML}_{loss}_{train}', index_col = 0)
                        .reset_index().rename(columns={'index':'Metric', 'Median':'Value'})
                        .loc[:, ['Metric', 'Value']])
                    # tmp['Metric'].loc[tmp['Metric'] == 'Rel RMSE\nExtremes',:] = 'Rel RMSE Ext'
                    tmp['Value'].loc[tmp['Metric'].str.contains('Rel')] = tmp['Value'].loc[tmp['Metric'].str.contains('Rel')] / 100
                    tmp = tmp[tmp['Metric'].isin(['Recall', 'Precision', 'F_beta', 'Rel RMSE\nExtremes', 'Rel_RMSE'])]
                    
                    tmp['Train'] = train
                    tmp['Model'] = f'{ML}_{loss}'
                    
                    results = pd.concat([results, tmp], axis=0)
                except:
                    print('No results yet')
    return results.reset_index(drop=True)
# %%

def get_station_results(coast, station):
    # Load in results
    res = {}
    for loss in ['Gumbel', 'mse']:
        res[loss] = {}
        for ML in ['ANN', 'LSTM', 'TCN', 'TCN-LSTM']:
            #preds[station][loss][ML] = pd.read_csv(f'Models/Ensemble_run/{coast}/{ML}/Data/{station}_{ML}_{loss}_prediction.csv', index_col=0)
            with open(f'Models/Ensemble_run/{coast}/{ML}/Data/{station}_{ML}_{loss}_result_all.json', 'r') as fp:
                results = json.load(fp)
                for key in ['train_loss', 'test_loss', 'rmse', 'rmse_ext']:
                    results.pop(key)

                results = pd.DataFrame(results)
                results['rel_rmse'] = results['rel_rmse'] / 100
                results['rel_rmse_ext'] = results['rel_rmse_ext'] / 100
            res[loss][ML] = results.median(axis=0)
            
            
    return res

# %%

def transform_to_top_models(res, station = True, station_name = None):
    if station:
        gum = pd.DataFrame(res['Gumbel'])
        gum.columns = [col + '_Gumbel' for col in gum.columns]

        mse = pd.DataFrame(res['mse'])
        mse.columns = [col + '_mse' for col in mse.columns]

        full = pd.concat([gum, mse], axis=1)

        full = (full.T.melt(ignore_index=False,var_name = 'Metric', value_name = 'Value').reset_index()
            .rename(columns={'index':'Model'})
            .sort_values(by=['Metric', 'Value'], ascending=False))
        
        full.to_csv(f'Models/Results/Figure_making/{station_name}_results.csv')
    else:
        full = res.sort_values(by=['Metric', 'Value'], ascending=False)

    best = full[~full['Metric'].str.lower().str.contains('rel')].groupby('Metric').head(3)
    best = pd.concat([best, full[full['Metric'].str.lower().str.contains('rel')].groupby('Metric').tail(3)], axis=0).reset_index(drop=True)

    #best = pd.concat([best, full[~full['Metric'].str.lower().str.contains('rel')].groupby('Metric').tail(1)], axis=0).reset_index(drop=True)
    #best = pd.concat([best, full[full['Metric'].str.lower().str.contains('rel')].groupby('Metric').head(1)], axis=0)
    
    return best

def plot_bar_top_models(best, name, save=True):
    
    palette ={"ANN_Gumbel": "C0",
          'ANN_mse' : "C1",
          'LSTM_Gumbel': "C2",
          'LSTM_mse': "C3",
          'TCN_Gumbel': "C4",
          'TCN_mse': "C5",
          'TCN-LSTM_Gumbel': "C6",
          'TCN-LSTM_mse': "C7"}
    
    # fig = plt.figure(figsize=(10,5))
    sns.barplot(x='Metric', y='Value', hue='Model', data=best, palette=palette)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize =14)
    # plt.title(f'Best 3 Models for {name}', fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(ticks = [0,1,2,3,4], labels = ['Recall', 'Precision', 'Fbeta', 'Rel RMSE Ext', 'Rel RMSE'], fontsize=14, rotation=45)
    plt.ylabel('Value', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    plt.tight_layout()
    
    if save:
    # Increase font size of legend to 
        plt.savefig(f'Models/Results/Figures/{name}_top_models.png', facecolor='white')
        plt.show()
        plt.close()

# def plot_bar_top_models_coast(best, name):
# %%


palette ={"ANN_Gumbel": "C0",
          'ANN_mse' : "C1",
          'LSTM_Gumbel': "C2",
          'LSTM_mse': "C3",
          'TCN_Gumbel': "C4",
          'TCN_mse': "C5",
          'TCN-LSTM_Gumbel': "C6",
          'TCN-LSTM_mse': "C7"}

# %%
# g = sns.FacetGrid(best, col="Train", hue='Model', height=5, aspect=1)
# g.map(sns.barplot, 'Metric', 'Value', order=['Recall', 'Precision', 'F_beta', 'Rel RMSE\nExtremes', 'Rel_RMSE'])
# g.add_legend()

# for ax in g.axes.flat:
#     ax.set_xticklabels(labels = ['Recall', 'Precision', 'Fbeta', 'Rel RMSE Ext', 'Rel RMSE'], fontsize=14, rotation=45)
# %%
# ML = 'ANN'
# loss = 'Gumbel'
station = 'cuxhaven-cuxhaven-germany-bsh'
coast = 'Cux_Station'
# station = 'calais-calais-france-refmar'
# station = 'delfzijl-del-nl-rws'

res = get_station_results(coast, station)
best = transform_to_top_models(res, station_name=station)
fig = plt.figure(figsize=(10,5))
plot_bar_top_models(best, station)

# %%
res = get_coast_results(coast)
best_train = transform_to_top_models(res[res['Train'] == 'train'], False).sort_values(by=['Model'])
best_test = transform_to_top_models(res[res['Train'] == 'test'], False).sort_values(by=['Model'])
# %%
fig, axes = plt.subplots(2,1, figsize=(10,8))
axes = axes.ravel()
sns.barplot(x='Metric', y='Value', hue='Model', data=best_train, ax = axes[0], palette=palette,
            order=['Recall', 'Precision', 'F_beta', 'Rel RMSE\nExtremes', 'Rel_RMSE'])
axes[0].get_legend().remove()
axes[0].tick_params(axis='both', which='major', labelsize=14)

axes[0].set_ylabel('Value', fontsize=14)
axes[0].set_xlabel('', fontsize=14)
axes[0].set_title('Train Stations', fontsize=18)

sns.barplot(x='Metric', y='Value', hue='Model', data=best_test, ax = axes[1], palette=palette, 
            order=['Recall', 'Precision', 'F_beta', 'Rel RMSE\nExtremes', 'Rel_RMSE'])
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize =14)
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].set_ylabel('Value', fontsize=14)
axes[1].set_xlabel('Metric', fontsize=14)
axes[1].set_title('Test Stations', fontsize=18)
fig.tight_layout()

plt.setp(axes, xticks=[0,1,2,3,4], 
         xticklabels=['Recall', 'Precision', 'Fbeta', 'Rel RMSE Ext', 'Rel RMSE'])

plt.savefig(f'Models/Results/Figures/{coast}_top_models.png', facecolor='white')
fig.show()

# %%   
# %%
# cux['Highest'], cux['Lowest']  = cux.idxmax(axis=1), cux.idxmin(axis=1)

# # Set a new column "Best" to either the Highest or Lowest depending on the index name
# # If the index is rel_rmse or rel_rmse_ext, then the lowest is the best
# # If the index is precision_ext or recall_ext or fbeta_ext, then the highest is the best

# # %%
# cux['Best'] = (cux['Lowest'] * cux.index.str.contains('rmse')) + cux['Highest'] * (~cux.index.str.contains('rmse'))
# # %%

# # select numeric variables from cux and plot a bar chart
# cux[sorted(cux.columns)].select_dtypes(include=['number']).plot(kind='bar')
# # Put the legend outside the plot
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
def load_station(coast, station, ML, loss):
    df = pd.read_csv(f'Models/Ensemble_run/{coast}/{ML}/Data/{station}_{ML}_{loss}_prediction.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    return df
# %%
def plot_ts(station, df, ML, loss):
    fig = plt.figure(figsize=(8,5))
    performance.plot_ensemble_testing_ts(df, ax=fig.add_subplot())
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Surge height (m)', fontsize=14)
    plt.savefig(f'Models/Results/Figures/{station}_{ML}_{loss}_ts.png', facecolor='white')
    plt.show()
    
def plot_max_ts(station, df, ML, loss):
    fig = plt.figure(figsize=(8,5))
    performance.plot_ensemble_testing_max_ts(df, ax=fig.add_subplot(), resample='hourly')
    plt.xticks(fontsize=14, rotation=30)
    plt.yticks(fontsize=14)
    plt.ylabel('Surge height (m)', fontsize=14)
    plt.savefig(f'Models/Results/Figures/{station}_{ML}_{loss}_max_ts.png', facecolor='white')
    plt.show()
# %%
coast = "NE_Atlantic_1"
# coast = "Japan"
station = 'cuxhaven-cuxhaven-germany-bsh'
station = 'delfzijl-del-nl-rws'
ML = 'LSTM'
loss = 'mse'

for ML in ['LSTM', 'TCN-LSTM', 'TCN']:
    for loss in ['Gumbel', 'mse']:
        print(f'{ML} {loss}')
        df = load_station(coast, station, ML, loss)
        plot_ts(station,df, ML, loss )
        plot_max_ts(station, df, ML, loss)
# %%

# station = 'calais-calais-france-refmar'
# station = 'brest-brest-france-refmar'
#coast = "Cux_Station"
coast = 'Japan'
station = 'miyakejima-357a-japan-uhslc'
#coast = "NE_Pacific"
#station = "yakutat,ak-570a-usa-uhslc"
#station = 'cuxhaven-cuxhaven-germany-bsh'
ML = "TCN-LSTM"
loss = 'Gumbel'
df = load_station(coast, station, ML, loss)
plot_ts(station,df, ML, loss )
plot_max_ts(station, df, ML, loss)

# %%

coast = 'NE_Pacific'
train_stations, test_stations = to_learning.get_coast_stations(coast)

# %%
def get_desc_stats_mse_gum(station):
    station_name = station.split('-')[0].split(',')[0]
    mse_df = load_station(coast, station, ML, 'mse')
    df = pd.DataFrame(mse_df['median'].describe()).rename(columns={'median': 'mse'})
    gum_df = load_station(coast, station, ML, 'Gumbel')
    df = pd.concat([df, pd.DataFrame(gum_df['median'].describe()).rename(columns={'median': 'Gumbel'})], axis=1)
    df['Observed'] = pd.Series(mse_df['Observed'].describe())
    df['Difference'] = df['Gumbel'] - df['mse']
    df[f'{station_name}'] = np.round(abs(df['Difference']) / abs(df['mse']) * 100,2)
    
    # plt.scatter(df.index[1:], df['Observed'][df.index != 'count'], label='Observed')
    # plt.scatter(df.index[1:], df['Gumbel'][df.index != 'count'], label='Gumbel')
    # plt.scatter(df.index[1:], df['mse'][df.index != 'count'], label='MSE')
    # plt.legend()
    # plt.title(f'{station_name}')
    # plt.show()
    return df#[f'{station_name}']

# %%
coast = "NE_Pacific"
ML = 'TCN'
station = 'yakutat,ak-570a-usa-uhslc'

train_df = pd.DataFrame()

for station in train_stations:
    get_desc_stats_mse_gum(station)
    train_df = pd.concat([train_df, get_desc_stats_mse_gum(station)], axis=1)

test_df = pd.DataFrame()
for station in test_stations:
    get_desc_stats_mse_gum(station)
    test_df = pd.concat([test_df, get_desc_stats_mse_gum(station)], axis=1)

# %%
train_df.to_csv(f'Models/Results/{coast}_TCN_train_stats_gum_mse.csv')
test_df.to_csv(f'Models/Results/{coast}_TCN_test_stats_gum_mse.csv')







# %%
results_df[['precision_ext', 'recall_ext', 'fbeta_ext']].boxplot(grid=False)
# %%
res[station]['Gumbel']['TCN-LSTM'][['precision_ext', 'recall_ext', 'fbeta_ext']].boxplot(grid=False)
# %%
res[station]['Gumbel']['TCN-LSTM'][['rel_rmse', 'rel_rmse_ext']].boxplot(grid=False)
# %%




# %%
categories = ['Precision', 'Recall', 'FBeta','Rel RMSE','rel RMSE Extremes',]

fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type": "polar"}, {"type": "polar"}]])

# Add traces for each ML
fig.add_trace(go.Scatterpolar(
      r=res['Gumbel']['ANN'],
      theta=categories,
      fill='toself',
      name='ANN',
      opacity=.5),
              row = 1, col = 1)

fig.add_trace(go.Scatterpolar(
      r=res['Gumbel']['LSTM'],
      theta=categories,
      fill='toself',
      name='LSTM',
      opacity=.5),
              row = 1, col = 1)

fig.add_trace(go.Scatterpolar(
      r=res['Gumbel']['TCN'],
      theta=categories,
      fill='toself',
      name='TCN',
      opacity=.5),
              row = 1, col = 1)

fig.add_trace(go.Scatterpolar(
      r=res['Gumbel']['TCN-LSTM'],
      theta=categories,
      fill='toself',
      name='TCN-LSTM',
      opacity=.5),
              row = 1, col = 1)

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True,
  title=f'{station} {loss}'
)


fig.add_trace(go.Scatterpolar(
      r=res['mse']['ANN'],
      theta=categories,
      fill='toself',
      name='ANN',
      opacity=.5),
              row = 1, col = 2)

fig.add_trace(go.Scatterpolar(
      r=res['mse']['LSTM'],
      theta=categories,
      fill='toself',
      name='LSTM',
      opacity=.5),
              row = 1, col = 2)

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True,
  title=f'{station} {loss}'
)


fig.show()
# %%



# %%
coast = "Cux_Test"
# coast = "NE_Atlantic_1"
station = 'cuxhaven-cuxhaven-germany-bsh'
ML = 'LSTM'
loss = 'Gumbel'
with open(f'Models/Ensemble_run/{coast}/{ML}/Data/{station}_{ML}_{loss}_result_all.json', 'r') as fp:
    results = json.load(fp)
    for key in ['train_loss', 'test_loss', 'rmse', 'rmse_ext']:
        results.pop(key)

    results = pd.DataFrame(results)
    results['rel_rmse'] = results['rel_rmse'] / 100
    results['rel_rmse_ext'] = results['rel_rmse_ext'] / 100
#results = results.median(axis=0)
# %%