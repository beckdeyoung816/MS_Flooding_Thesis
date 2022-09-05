'''
This class stores all the information (e.g., train/test data & metrics) for a given station
'''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score, recall_score, fbeta_score
import performance
import os
import keras
import sys
import json
import copy

class Station():
    
    def __init__(self, station_name, train_test, train_X, train_Y, test_X, test_Y, val_X, val_Y, scaler, df, reframed_df, n_train_final, test_dates, test_year, model_dir, ML, loss):
        self.train_X = train_X
        self.name = station_name
        self.train_y = train_Y
        self.test_X = test_X
        self.test_y = test_Y
        self.val_X = val_X
        self.val_y = val_Y
        self.scaler = scaler
        self.df = df
        self.reframed_df = reframed_df
        self.n_train_final = n_train_final
        self.test_dates = test_dates
        self.test_year = test_year
        self.train_test = train_test
        self.loss = loss
        
        
        
        # Initialize Storage of results
        self.result_all = dict()
        self.result_all['data'] = dict()
        self.result_all['train_loss'] = dict()
        self.result_all['test_loss'] = dict()
        self.result_all['rmse'] = dict()
        self.result_all['rel_rmse'] = dict()
        self.result_all['rmse_ext'] = dict()
        self.result_all['rel_rmse_ext'] = dict()
        self.result_all['precision_ext'] = dict()
        self.result_all['recall_ext'] = dict()
        self.result_all['fbeta_ext'] = dict()
        
        
        # Store data and remove from memory
        self.model_dir = model_dir
        self.data_path = os.path.join(model_dir, 'Data_storage', self.name, ML)
        os.makedirs(self.data_path, exist_ok=True)
        self.store_and_delete_data(store=True)
    
    def store_and_delete_data(self, store=False):
        """Store the data for the station and remove it from memory to save space
        """
        if store:
            with open(f'{self.data_path}/data.npy', 'wb') as f:
                np.save(f, self.train_X)
                np.save(f, self.train_y)
                np.save(f, self.test_X)
                np.save(f, self.test_y)
                np.save(f, self.val_X)
                np.save(f, self.val_y)
                np.save(f, self.reframed_df)
                np.save(f, self.test_year)
            
            pd.DataFrame(self.reframed_df).to_csv(f'{self.data_path}/{self.name}_reframed_df.csv')
            pd.DataFrame(self.test_year).to_csv(f'{self.data_path}/{self.name}_test_year.csv')
            
        # Delete variables
        del self.train_X
        del self.train_y
        del self.test_X
        del self.test_y
        del self.val_X
        del self.val_y
        del self.reframed_df
        del self.test_year
    
    def reload_data(self):
        with open(f'{self.data_path}/data.npy', 'rb') as f:
            self.train_X = np.load(f)
            self.train_y = np.load(f)
            self.test_X = np.load(f)
            self.test_y = np.load(f)
            self.val_X = np.load(f)
            self.val_y = np.load(f)
            self.reframed_df = np.load(f)
            self.test_year = np.load(f)
        
        self.reframed_df = pd.read_csv(f'{self.data_path}/{self.name}_reframed_df.csv', index_col=0)
        self.test_year = pd.read_csv(f'{self.data_path}/{self.name}_test_year.csv', index_col=0)
        
    def predict(self, model: keras.Model, ensemble_loop, mask_val):
        """Make predictions for a given station
        """
        self.reload_data()
        # Replace masking values
        temp_df = self.test_year.replace(to_replace=mask_val, value=np.nan)[self.n_train_final:].copy()

        # make a prediction
        self.test_preds = model.predict(self.test_X)
        
        # invert scaling for observed surge
        self.inv_test_y = self.scaler.inverse_transform(temp_df.values)[:,-1]

        # invert scaling for modelled surge
        temp_df.loc[:,'values(t)'] = self.test_preds
        self.inv_test_preds = self.scaler.inverse_transform(temp_df.values)[:,-1]
        
         # Get evaluation metrics
        self.evaluate_model(ensemble_loop)
        
        self.store_and_delete_data(store=False)
        
    def evaluate_model(self, ensemble_loop):
        """Get evaluation metrics for model predictions. RMSE, Rel_RMSE, Precision, Recall, FBeta.
        """
        # RMSE
        self.rmse = np.sqrt(mse(self.inv_test_y, self.inv_test_preds))
        self.rel_rmse = self.rmse/np.mean(self.inv_test_y)
        
        # Get the values that are deemed as extremes
        extremes = (pd.DataFrame(self.inv_test_y)
                    .nlargest(round(.10*len(self.inv_test_y)), 0) # Largest 10%
                    .sort_index())
        min_ext = extremes.iloc[:,0].min() # Minimum of Largest 10% to use as threshold lower bound
        extremes_indices = extremes.index.values
                            
        self.inv_test_y_ext = self.inv_test_y[extremes_indices]
        self.inv_test_preds_ext = self.inv_test_preds[extremes_indices]
        
        # RMSE for Extremes
        self.rmse_ext = np.sqrt(mse(self.inv_test_y_ext, self.inv_test_preds_ext))
        self.rel_rmse_ext = self.rmse_ext / self.inv_test_y.mean()
        
        # Precision and recall and fbeta score for extremes
        
        # Turn into binary classification
        ext_df = pd.DataFrame([self.inv_test_y, self.inv_test_preds], index = ['Obs', 'Pred']).T
        ext_df['Extreme_obs'] = ext_df['Obs'] >= min_ext
        ext_df['Extreme_pred'] = ext_df['Pred'] >= min_ext
        
        self.precision_ext = precision_score(ext_df['Extreme_obs'], ext_df['Extreme_pred'])
        self.recall_ext = recall_score(ext_df['Extreme_obs'], ext_df['Extreme_pred'])
        self.fbeta_ext = fbeta_score(ext_df['Extreme_obs'], ext_df['Extreme_pred'], beta=2)
        
        # Store Results
        df_all = performance.store_result(self.inv_test_preds, self.inv_test_y)
        df_all = df_all.set_index(self.df.iloc[self.test_dates,:].index, drop = True)                                                                    
        
        self.result_all['rmse'][ensemble_loop] = self.rmse
        self.result_all['rel_rmse'][ensemble_loop] = self.rel_rmse
        self.result_all['rmse_ext'][ensemble_loop] = self.rmse_ext
        self.result_all['rel_rmse_ext'][ensemble_loop] = self.rel_rmse_ext
        
        self.result_all['precision_ext'][ensemble_loop] = self.precision_ext
        self.result_all['recall_ext'][ensemble_loop] = self.recall_ext
        self.result_all['fbeta_ext'][ensemble_loop] = self.fbeta_ext
        
        self.result_all['data'][ensemble_loop] = df_all.copy()
        
        # Save dictionary result_all to a json file
        # with open(f'{self.model_dir}/Data/{self.name}_{self.loss}_result_all.json', 'w') as fp:
        #     res = copy.deepcopy(self.result_all)
        #     res.pop('data') # Remove the data from the dictionary
        #     json.dump(res, fp)
        
        
        
    