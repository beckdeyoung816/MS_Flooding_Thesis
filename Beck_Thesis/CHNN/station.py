import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import performance
import matplotlib.pyplot as plt

class Station():
    
    def __init__(self, station_name, train_X, train_Y, test_X, test_Y, val_X, val_Y, scaler, df, reframed_df, n_train_final, test_dates, test_year):
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
        
        # Initialize Storage of results
        self.result_all = dict()
        self.result_all['data'] = dict()
        self.result_all['train_loss'] = dict()
        self.result_all['test_loss'] = dict()
        
    def predict(self, model, ensemble_loop, mask_val):
        
        print(f'\nPredicting for {self.name}\n')
        
        temp_df = self.test_year.replace(to_replace=mask_val, value=np.nan)[self.n_train_final:].copy()

        # make a prediction
        self.test_preds = model.predict(self.test_X)

        # invert scaling for observed surge
        self.inv_test_y = self.scaler.inverse_transform(temp_df.values)[:,-1]

        # invert scaling for modelled surge
        temp_df.loc[:,'values(t)'] = self.test_preds
        self.inv_test_preds = self.scaler.inverse_transform(temp_df.values)[:,-1]
        
        # print("\nNUM NAs Y: ", np.count_nonzero(np.isnan(self.test_y)))
        # print(self.test_y[0:5])
        # print("\nNUM NAs Preds: ", np.count_nonzero(np.isnan(self.test_preds)))
        # print(self.test_preds[0:5])
        # print("\nNUM NAs Trans Y: ", np.count_nonzero(np.isnan(self.inv_test_y)))
        # print(self.inv_test_y[0:5])
        # print("\nNUM NAs Trans Preds: ", np.count_nonzero(np.isnan(self.inv_test_preds)))
        # print(self.inv_test_preds[0:5])
        
        # plt.plot(self.inv_test_y, color='blue', alpha=.5)
        # plt.plot(self.inv_test_preds, color='red', alpha=.5)
        # plt.title(self.name)
        # plt.legend(['True', 'Pred'])
        # plt.show()
        
        self.rmse = np.sqrt(mean_squared_error(self.inv_test_y, self.inv_test_preds))
        self.rel_rmse = self.rmse/np.mean(self.inv_test_y)
        print(f'RMSE: {self.rmse: .2f}\n')
        print(f'Relative RMSE: {self.rel_rmse: .2f}\n')
        
        extremes_indices = (pd.DataFrame(self.inv_test_y)
                            .nlargest(round(.1*len(self.inv_test_y)), 0)
                            .sort_index()
                            .index.values)
                            

        self.inv_test_y_ext = self.inv_test_y[extremes_indices]
        self.inv_test_preds_ext = self.inv_test_preds[extremes_indices]

        self.rmse_ext = np.sqrt(mean_squared_error(self.inv_test_y_ext, self.inv_test_preds_ext))
        self.rel_rsme_ext = self.rmse_ext / self.inv_test_y.mean()
        
        print(f'\nRMSE Extremes: {self.rmse_ext: .2f}\n')
        print(f'Relative RMSE Extremes: {self.rel_rsme_ext: .2f}\n')
        
        # Store Results
        df_all = performance.store_result(self.inv_test_preds, self.inv_test_y)
        df_all = df_all.set_index(self.df.iloc[self.test_dates,:].index, drop = True)                                                                    
        
        self.result_all['data'][ensemble_loop] = df_all.copy()
        
        
    def select_extremes(df, percent):
        extremes = df.nlargest(round(percent*len(df)), 'residual').copy()
        extremes.sort_index(inplace = True)
        extreme_dates = extremes.index.values
        return extreme_dates, df.loc[extreme_dates,:]
            

    