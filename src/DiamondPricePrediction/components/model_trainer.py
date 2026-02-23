import os
import sys
import pickle
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import CustomException


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("Artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        try:
            report = {}

            for name, model in models.items():
                # Start MLflow run for each model
                with mlflow.start_run(run_name=name):
                    # Train the model
                    model.fit(X_train, y_train)

                    # Predict on test data
                    y_pred = model.predict(X_test)

                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)

                    # Log metrics to MLflow
                    mlflow.log_metric("r2_score", r2)
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("mse", mse)

                    # Log model to MLflow
                    mlflow.sklearn.log_model(model, name)

                    report[name] = r2
                    logging.info(f"{name} -> R2: {r2:.4f}, MAE: {mae:.4f}")

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model Training Started")

            # Set MLflow experiment name
            mlflow.set_experiment("Diamond Price Prediction")

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

            # Evaluate all models
            model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models)

            # Print all model scores
            print("\n--- Model Performance Report ---")
            for model_name, score in model_report.items():
                print(f"{model_name}: R2 Score = {score:.4f}")

            # Find best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            print(f"\nBest Model: {best_model_name}")
            print(f"Best R2 Score: {best_model_score:.4f}")

            # Save best model as pickle
            with open(self.model_trainer_config.trained_model_file_path, 'wb') as f:
                pickle.dump(best_model, f)

            logging.info(f"Best model {best_model_name} saved successfully")

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)