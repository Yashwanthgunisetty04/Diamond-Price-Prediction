import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

import pickle

from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException    

@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path: str = os.path.join("Artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation(self):
        
        try:
            logging.info("Data Transformation Started")
            
            num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cat_cols = ['cut', 'color', 'clarity']
            
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            
            num_pipeline = Pipeline(steps = [
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline(steps = [
                ('ordinal_encoder', OrdinalEncoder(categories = [cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())
            ])
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols)
            ])
            
            logging.info("Data Transformation Completed")
            return preprocessor
            
        except Exception as e:
            logging.info("Error occurred in Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initiating Data Transformation")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading Train and Test Data Completed")
            
            train_df = train_df.drop(columns=['id'])
            test_df = test_df.drop(columns=['id'])
            logging.info("Dropping ID column from Train and Test Data Completed")
            
            target_col = 'price'
            
            X_train = train_df.drop(columns = [target_col])
            Y_train = train_df[target_col]
            
            X_test = test_df.drop(columns = [target_col])
            Y_test = test_df[target_col]
            
            preprocessor = self.get_data_transformation()
            
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            logging.info("Applying Preprocessor on Train and Test Data Completed")  
            
            with open(self.data_transformation_config.preprocessor_object_file_path, 'wb') as file_obj:
                pickle.dump(preprocessor, file_obj)
                
            logging.info("Preprocessor Object Saved")
            
            return (
                X_train_transformed,
                Y_train,
                X_test_transformed,
                Y_test,
                self.data_transformation_config.preprocessor_object_file_path
            )
            
        except Exception as e:
            logging.info("Error occurred in Initiating Data Transformation")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataTransformation()
    train_path = "Artifacts/train.csv"
    test_path = "Artifacts/test.csv"
    obj.initiate_data_transformation(train_path, test_path)