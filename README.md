# Toxic Comment Detection

This project implements a machine learning pipeline to detect toxic comments, featuring data analysis, model training with Keras/Scikit-Learn, and a deployemnt with FastAPI service.

## Project Overview

### 1. EDA & Data Preparation
Scripts are available to explore the dataset and prepare it for training. This includes analyzing text statistics, class balance, and cleaning the raw text data to ensure high-quality input for the model.

### 2. Model Training
The model training process in (`train_model.ipynb`) utilizes a **Scikit-Learn Pipeline** that combines:
- **TfidfVectorizer**: For transforming text data into numerical features.
- **KerasClassifier**: A wrapper around a TensorFlow/Keras Sequential neural network. With Dense layers with ReLU activation and a Sigmoid output layer for binary classification.

The trained pipeline is saved as a pickle file for deployment.

## Running the API

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Start the Uvicorn server:**
   ```bash
   uv run uvicorn fastapi_service.main:app --reload
   ```

## API Usage Example

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "comments": [
    "You are cool!",   
    "You are stupid"
  ]
}'
```