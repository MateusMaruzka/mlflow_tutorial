# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import matplotlib.pyplot as plt


def main():
    
    
    mlflow.set_tracking_uri("http://localhost:5000")
    
    mlflow.set_experiment("Blood_Sugar_Models")
    run_name = "bs_linreg_test"
    artifact_path = "bs_linreg"
    
    
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    
    # From sklearn docs: 
    # age age in years
    # sex
    # bmi body mass index
    # bp average blood pressure
    # s1 tc, total serum cholesterol
    # s2 ldl, low-density lipoproteins
    # s3 hdl, high-density lipoproteins
    # s4 tch, total cholesterol / HDL
    # s5 ltg, possibly log of serum triglycerides level
    # s6 glu, blood sugar level
    
    # Use only one feature
    # diabetes_X = diabetes_X[:, np.newaxis, 0] # Getting the age column
    diabetes_X = diabetes_X[:, np.newaxis, 9] # Getting the blood sugar level column

    print(diabetes_X.shape)
    print(diabetes_y.shape)

    # Split the data into training/testing sets
    
    X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size = 0.25)
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X_train, y_train)
    
    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(X_test)
    
    # The coefficients
    print("Coefficients: \n", regr.coef_)
    
    # The mean squared error
    # The coefficient of determination: 1 is perfect prediction

    print("MSE_train: %.2f" % (mse_train := \
                               mean_squared_error(y_train, \
                                                  diabetes_y_train_pred := regr.predict(X_train))))
    print("R2 Train %.2f" % (r2_train := r2_score(y_train, diabetes_y_train_pred)))
    
    print("MSE Test: %.2f" % (mse_test := mean_squared_error(y_test, diabetes_y_pred)))
    print("R2 Test %.2f" % (r2_test := r2_score(y_test, diabetes_y_pred)))
    
    # Plot outputs
    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, diabetes_y_pred, color="blue", linewidth=3)
    
    plt.show()
    
    print(params := regr.get_params())
    metrics = {"mse_test": mse_test, "r2_test": r2_test, 
               "r2_train": r2_train, "mse_train":mse_train}

    # Initiate the MLflow run context
    with mlflow.start_run(run_name=run_name):
         # Log the parameters used for the model fit
          mlflow.log_params(params)
    
         # Log the error metrics that were calculated during validation
          mlflow.log_metrics(metrics)
    
         # Log an instance of the trained model for later use
         # TODO: using test df. going to change to validation
          mlflow.sklearn.log_model(sk_model=regr, input_example=X_test, artifact_path=artifact_path)
        
    
if __name__ == "__main__":
    main()