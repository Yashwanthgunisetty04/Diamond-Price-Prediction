import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.DiamondPricePrediction.logger import logging 
from src.DiamondPricePrediction.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('Artifacts', 'train.csv')
    test_data_path: str = os.path.join('Artifacts', 'test.csv')
    raw_data_path: str = os.path.join('Artifacts', 'raw.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            df = pd.read_csv("C:/Users/yashw/Documents/Projects/Diamond-Price-Prediction/Data/raw/train.csv")
            logging.info("Data Ingested Successfully") 
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            
            logging.info("Train Test Split Initiated")
            
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False)
            logging.info("Ingestion of Data Completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Error occurred in Data Ingestion")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()