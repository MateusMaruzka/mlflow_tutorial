# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:39:57 2023

@author: Maruzka
"""

import mlflow 

from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor

def create_experiment():
    # Provide an Experiment description that will appear in the UI
    client = MlflowClient(tracking_uri="http://localhost:5000")

    experiment_description = (
        "This is an MLflow example based on mlflow docs. \
         This ml project has a main tag called diabetes-linear-regression.\
         Each experiments uses a different part of the diabetes sklearn dataset,\
         i.e, age or sex."
    )
    
    # Provide searchable tags that define characteristics of the Runs that
    # will be in this Experiment
    experiment_tags = {
        "project_name": "diabetes-linear-regression",
        # "store_dept": "produce",
        # "team": "stores-ml",
        "project_quarter": "Q4-2023",
        "mlflow.note.content": experiment_description,
    }
    
    # Create the Experiment, providing a unique name
    produce_exp = client.create_experiment(
        name="Age_Models", tags=experiment_tags
    )
    

def main():
    
    # Use search_experiments() to search on the project_name tag key
    client = MlflowClient(tracking_uri="http://localhost:5000")

    exp = client.search_experiments(
        filter_string="tags.`project_name` = 'diabetes-linear-regression'"
    )
    

if __name__ == "__main__":
    main()