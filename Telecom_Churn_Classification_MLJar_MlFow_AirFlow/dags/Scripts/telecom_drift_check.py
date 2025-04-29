import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
import pandas as pd
from datetime import datetime
import json

# --- Paths ---
baseline_path = '/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/X_train.csv'
current_path = '/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/X_test.csv'
report_output_dir = '/Users/I353375/Downloads/MLOps/airflow/reports'
os.makedirs(report_output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file_path = os.path.join(report_output_dir, f"telecom_data_drift_report_{timestamp}.json")

# --- Drift Monitoring Function ---
def telecom_monitor_drift():
    # Load baseline and current datasets
    baseline = pd.read_csv(baseline_path)
    current = pd.read_csv(current_path)

    # Create and run Report
    report = Report(metrics=[DataDriftPreset()])   # <-- Important: Give metrics here during init
    report.run(reference_data=baseline, current_data=current)

    # Delete the old drift report if it exists
    if os.path.exists(report_file_path):
        os.remove(report_file_path)
        print(f"🗑️ Existing drift report deleted: {report_file_path}")

    # Save Report as JSON
    result_dict = report.as_dict()
    with open(report_file_path, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)

    # Analyze results
    try:
        drift_info = result_dict['metrics'][0]['result']
        n_drifted_features = drift_info['number_of_drifted_columns']
        total_features = drift_info['number_of_columns']
        drift_percentage = (n_drifted_features / total_features) * 100

        if drift_percentage > 3:
            print(f"🚨 Drift detected: {n_drifted_features}/{total_features} features drifted ({drift_percentage:.2f}%). HIGH ALERT!")
        else:
            print("✅ No drift detected. All features are stable.")
    except (KeyError, IndexError) as e:
        print("❌ Error parsing drift results:", str(e))

    print(f"📄 Drift report saved as JSON to: {report_file_path}")

# --- Run standalone
if __name__ == "__main__":
    yelecom_monitor_drift()
