import sys
import os 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessing_path=os.path.join("arifacts","preprocessing.pkl")
            print("Before loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessing_path)
            print("After Loading")

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
