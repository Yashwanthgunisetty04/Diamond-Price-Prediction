# 💎 Diamond Price Prediction

An end-to-end Machine Learning project that predicts diamond prices using MLOps best practices.

## 🎯 Project Overview

This project builds a complete ML pipeline to predict diamond prices based on features like carat, cut, color, clarity, and dimensions. It follows professional ML engineering practices with modular code, experiment tracking, and containerization.

## 📊 Dataset

- **Source:** Kaggle - Playground Series S3E8
- **Size:** 193,573 diamonds
- **Features:** carat, cut, color, clarity, depth, table, x, y, z
- **Target:** price (USD)

## 🏗️ Project Structure
```
Diamond-Price-Prediction/
├── src/
│   └── DiamondPricePrediction/
│       ├── components/         # ML pipeline components
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   └── model_trainer.py
│       ├── pipelines/          # Training and prediction pipelines
│       │   ├── training_pipeline.py
│       │   └── prediction_pipeline.py
│       ├── logger.py           # Logging configuration
│       └── exception.py        # Custom exception handling
├── Notebook_Experiments/       # EDA notebooks
├── templates/                  # Flask HTML templates
├── app.py                      # Flask web application
├── Dockerfile                  # Docker configuration
├── dvc.yaml                    # DVC pipeline stages
└── requirements.txt            # Project dependencies
```

## 🔑 Key Findings from EDA

- **Carat** has the strongest correlation with price (0.94)
- **x, y, z** dimensions are also strongly correlated (0.90)
- **Depth** has almost no impact on price (0.00)
- Price is right-skewed — most diamonds are affordable
- Better cut doesn't always mean higher price — carat dominates!

## 🤖 Models Trained

| Model | R2 Score |
|-------|----------|
| Linear Regression | 0.9373 |
| Ridge Regression | 0.9373 |
| Lasso Regression | 0.9373 |
| Decision Tree | 0.9575 |
| **Random Forest** | **0.9772 ✅** |
| Gradient Boosting | 0.9761 |

**Winner: Random Forest with 97.72% accuracy!**

## 🛠️ Tech Stack

- **ML:** Scikit-learn, Pandas, NumPy
- **Experiment Tracking:** MLflow
- **Pipeline Orchestration:** DVC
- **Web App:** Flask
- **Containerization:** Docker
- **Version Control:** Git + GitHub

## 🚀 Getting Started

### Option 1 — Run Locally
```bash
# Clone the repository
git clone https://github.com/Yashwanthgunisetty04/Diamond-Price-Prediction.git
cd Diamond-Price-Prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
dvc repro

# Start the web app
python app.py
```

Open browser at `http://127.0.0.1:8080`

### Option 2 — Run with Docker
```bash
# Pull and run
docker build -t diamond-price-app .
docker run -p 8080:8080 diamond-price-app
```

Open browser at `http://127.0.0.1:8080`

## 📱 Web App

Enter diamond features and get instant price predictions!

![Diamond Price Prediction App](static/app_screenshot.png)

## 📈 MLflow Experiment Tracking

Track all experiments with MLflow:
```bash
mlflow ui --port 5001
```

Open browser at `http://127.0.0.1:5001`

## 🔄 DVC Pipeline

Run the complete ML pipeline with one command:
```bash
dvc repro
```

DVC intelligently reruns only the stages that changed!

## 👤 Author

**Yashwanth**
- GitHub: [@Yashwanthgunisetty04](https://github.com/Yashwanthgunisetty04)
- LinkedIn: [Your LinkedIn URL]

## 📝 License

This project is open source and available under the [MIT License](LICENSE).