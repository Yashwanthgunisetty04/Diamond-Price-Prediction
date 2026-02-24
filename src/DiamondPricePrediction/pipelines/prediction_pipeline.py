import os
import sys
import pickle
import pandas as pd
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("Artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            # Load model and preprocessor
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            with open(self.preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)

            # Transform input features
            scaled_features = preprocessor.transform(features)

            # Predict price
            prediction = model.predict(scaled_features)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, carat, cut, color, clarity, depth, table, x, y, z):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

    def get_data_as_dataframe(self):
        try:
            data = {
                "carat": [self.carat],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z]
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)