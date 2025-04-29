import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_data():
    # Load data
    df = pd.read_csv('/Users/I353375/Downloads/MLOps/airflow/day.csv', 
                 dtype={'season': 'int64', 'mnth': 'int64', 'weekday': 'int64', 'weathersit': 'int64'})

    # Define mappings
    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    weather_map = {1: 'good', 2: 'moderate', 3: 'bad', 4: 'severe'}
    month_map = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
             7: 'jul', 8: 'aug', 9: 'sept', 10: 'oct', 11: 'nov', 12: 'dec'}
    weekday_map = {0: 'sun', 1: 'mon', 2: 'tue', 3: 'wed', 4: 'thu', 5: 'fri', 6: 'sat'}

    # Map correctly
    df['season'] = df['season'].map(season_map)
    df['weathersit'] = df['weathersit'].map(weather_map)
    df['mnth'] = df['mnth'].map(month_map)
    df['weekday'] = df['weekday'].map(weekday_map)

    # Drop unwanted column
    df = df.drop('dteday', axis=1)

    # Get dummies (now it will work correctly)
    df_new = pd.get_dummies(data=df, columns=['weathersit', 'season', 'mnth', 'weekday'], dtype=int)
    
    # Train/test split
    df_train, df_test = train_test_split(df_new, train_size=0.7, random_state=100)
    
    # Initialize scaler and scale only once
    scaler = MinMaxScaler()
    numerical_vars = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
    
    # Transform in one operation
    df_train[numerical_vars] = scaler.fit_transform(df_train[numerical_vars])
    df_test[numerical_vars] = scaler.transform(df_test[numerical_vars])
    
    # Split features and target
    X_train = df_train.drop(columns=['cnt'])
    X_test = df_test.drop(columns=['cnt'])
    y_train = df_train['cnt']
    y_test = df_test['cnt']
    
    # Save files with compression for faster I/O
    X_train.to_csv('/Users/I353375/Downloads/MLOps/airflow/X_train.csv', index=False)
    X_test.to_csv('/Users/I353375/Downloads/MLOps/airflow/X_test.csv', index=False)
    y_train.to_csv('/Users/I353375/Downloads/MLOps/airflow/y_train.csv', index=False)
    y_test.to_csv('/Users/I353375/Downloads/MLOps/airflow/y_test.csv', index=False)
    
    print("Preprocessing completed efficiently")

if __name__ == "__main__":
    preprocess_data()
