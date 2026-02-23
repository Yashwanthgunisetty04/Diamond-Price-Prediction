import sys
from src.DiamondPricePrediction.components.data_ingestion import DataIngestion
from src.DiamondPricePrediction.components.data_transformation import DataTransformation
from src.DiamondPricePrediction.components.model_trainer import ModelTrainer
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException

try:
    
    data_ingestion = DataIngestion()
    
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    logging.info(f"Data Ingestion Completed: Train Data Path: {train_path}, Test Data Path: {test_path}")
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")
    
     # Step 2 - Data Transformation
    data_transformation = DataTransformation()
    X_train, Y_train, X_test, Y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    print("Data Transformation completed!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")
    
    #step 3 - Model Training
    model_trainer = ModelTrainer()
    best_model_name, best_model_score = model_trainer.initiate_model_training(X_train, Y_train, X_test, Y_test)
    print("Model Training completed!")
    
    
except Exception as e:
    logging.info("Error occurred in Training Pipeline")
    raise CustomException(e, sys)