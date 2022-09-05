# Coastal_Flooding_Thesis

Coastline_Clustering: Contains the information from Enriquez 2020 about the clustering of the coastlines

Beck_Thesis: Main code for the thesis
    - ML_Beck.sh : Bash script to run all models for all coastlines
    - ML_env_Coast.py : Parameters for the models
    
    - Scripts : Main model code
         - model_run_coast.py : Script to automate pipeline from preprocessing, training, to evaluation
         - Coastal_Model.py : Class with functions to set up, train, and predict NNs on a coastal level
         - to_learning.py : Functions to prepare the input data
         - performance.py : Functions to evaluate model performance and generate figures
         - performance2.py : Functions to generate figures
         - station.py : Class that stores data and results for individual stations
         - The rest are testing scripts or for getting sea surface temperature data
    - Stations : Contains information and scripts about the selected stations
         - Results_to_Latex.Rmd : Generate descriptive tables and turn them into latex
         - Selected_Stations_w_Data.csv : Official csv with selected stations and information
        
  Other folders and files are not necessary, they just hold some back up information
