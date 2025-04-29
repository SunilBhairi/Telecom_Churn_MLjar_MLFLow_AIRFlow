import mlflow
import os
import pandas as pd
from supervised.automl import AutoML
from datetime import datetime

# --- Load Test Data ---
X_test = pd.read_csv('/Users/I353375/Downloads/MLOps/airflow/X_test.csv')
y_test = pd.read_csv('/Users/I353375/Downloads/MLOps/airflow/y_test.csv')  # Actual targetvalues
OUTPUT_DIR = "/Users/I353375/Downloads/MLOps/airflow/"

def load_model_and_predict():
    # --- Load the MLJAR Model from MLflow ---
    results_path = "/Users/I353375/Downloads/MLOps/mlflow_server/mljar/automl_results"
    automl_loaded = AutoML(results_path=results_path)

    # --- Predict on Test Data ---
    predictions = automl_loaded.predict(X_test)

    # --- Combine Actual and Predicted ---
    results_df = X_test.copy()
    results_df["Actual"] = y_test.values.flatten()  # Adding actual target values
    results_df["Predicted"] = predictions  # Adding predicted values
    # --- Save to CSV ---
    output_file = os.path.join(OUTPUT_DIR, f"predictions.csv")
    results_df[["Actual", "Predicted"]].to_csv(output_file, index=False)
    print("✅ Predictions saved to 'predictions.csv'.")

# Run the function if needed standalone
if __name__ == "__main__":
    load_model_and_predict()

    
