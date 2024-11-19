import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd
from dataclasses import dataclass
from catboost import CatBoostRegressor
from src.utils import get_encoded_values


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrained:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trained(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            train_df=pd.read_csv("artifacts\\train.csv")
            test_df=pd.read_csv("artifacts\\test.csv")
            train_df, test_df = get_encoded_values(train_df=train_df, test_df=test_df)
            X_train,y_train,X_test,y_test=(
                train_df.drop(columns=["Calories_Burned"],axis=1).values,
                train_df[["Calories_Burned"]].values.ravel(),
                test_df.drop(columns=["Calories_Burned"],axis=1).values,
                test_df[["Calories_Burned"]].values.ravel()

            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            

        except Exception as e:
            raise CustomException(e,sys)
