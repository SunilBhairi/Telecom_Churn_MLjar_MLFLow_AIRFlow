import os
import json

# --- Helper: Check Drift from the Evidently Report ---
def telecom_retrain_and_log_model():
    report_dir = '/Users/I353375/Downloads/MLOps/airflow/reports'  # Update with your reports folder path

    # Find the latest drift report JSON file
    report_files = sorted([f for f in os.listdir(report_dir) if f.startswith("telecom_data_drift") and f.endswith(".json")])

    if not report_files:
        print("❌ No drift report files found.")
        return False

    latest_report_file = report_files[-1]  # Get the most recent report
    latest_report_path = os.path.join(report_dir, latest_report_file)

    print(f"🔍 Reading drift report: {latest_report_path}")

    with open(latest_report_path, 'r') as f:
        report_data = json.load(f)

    try:
        # Extract drift information from the JSON
        drift_info = report_data['metrics'][0]['result']
        n_drifted_features = drift_info['number_of_drifted_columns']
        total_features = drift_info['number_of_columns']
        drift_percentage = (n_drifted_features / total_features) * 100

        if drift_percentage > 3:
            print(f"🚨 High Alert! Drift detected: {n_drifted_features}/{total_features} features drifted ({drift_percentage:.2f}%).")
            return True  # Drift is detected
        else:
            print(f"✅ No drift detected. ({drift_percentage:.2f}% features drifted)")
            return False  # No drift

    except (KeyError, IndexError) as e:
        print("❌ Error parsing drift report:", str(e))
        return False

# --- Run the drift check ---
if __name__ == "__main__":
    telecom_retrain_and_log_model()

