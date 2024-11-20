import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.preprocessing import OneHotEncoder

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            # para=param[list(models.keys())[i]]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

from sklearn.preprocessing import OneHotEncoder

def get_encoded_values(train_df, test_df):
    preprocessor_path=os.path.join("artifacts","preprocessing.pkl")
    preprocessor=load_object(file_path=preprocessor_path)
    train_df_X=train_df.drop(["Calories_Burned"],axis=1)
    train_df_X_scaled=preprocessor.fit_transform(train_df_X)
    feature_names=preprocessor.get_feature_names_out()
    train_df_X_scaled_df=pd.DataFrame(train_df_X_scaled,columns=feature_names)

    test_df_X=test_df.drop(["Calories_Burned"],axis=1)
    test_df_X_scaled=preprocessor.fit_transform(test_df_X)
    feature_names=preprocessor.get_feature_names_out()
    test_df_X_scaled_df=pd.DataFrame(test_df_X_scaled,columns=feature_names)

    return (
        train_df_X_scaled_df,
        test_df_X_scaled_df
    )





    


def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


