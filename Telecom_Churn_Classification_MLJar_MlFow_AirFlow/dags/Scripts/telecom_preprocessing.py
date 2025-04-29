import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def telecom_preprocess_data():
    ### MLJar for Telecom Churn project

    ## Proprocesing steps

    #Sklearn
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    #Load the data
    df_train = pd.read_csv("/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/telecom_train.csv")
    df_test = pd.read_csv("/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/telecom_test.csv")
    sample = pd.read_csv("/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/telecom_sample.csv")
    data_dict = pd.read_csv("/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/telecom_data_dictionary.csv")

    #impute with zeroes on both df_train and df_test
    nullvalues=df_train.isnull().mean()*100 
    nullreplacezero_columns_train =  df_train.columns[df_train.columns.str.contains('_amt|count_|fb_user|_pck|total_|max_|_mou|_others')]
    nullreplacezero_columns_test =  df_test.columns[df_test.columns.str.contains('_amt|count_|fb_user|_pck|total_|max_|_mou|_others')]

    #After looking at the min values and other desrciptive stats about each of the columns 
    # we can be certain about replacing NAs with zeroes in these columns
    for col in nullreplacezero_columns_train:
        df_train[col] = df_train[col].replace(np.NaN,0.0)
    
    for col in nullreplacezero_columns_test:
        df_test[col] = df_test[col].replace(np.NaN,0.0)

    # getting rid of columns with just 1 unique value
    single_val_cols_train = df_train.columns[df_train.nunique() == 1]
    df_train.drop(single_val_cols_train,axis=1,inplace=True)

    # getting rid of columns with just 1 unique value
    single_val_cols_test = df_test.columns[df_test.nunique() == 1]
    df_test.drop(single_val_cols_test,axis=1,inplace=True)

    # Removing date columns because these wouldn't give any valuable insights about churn.
    date_columns =  df_train.columns[df_train.columns.str.contains('date')]
    df_train.drop(date_columns,axis=1,inplace=True)
    df_test.drop(date_columns,axis=1,inplace=True)

    # #Dropping the arpu columns because nullvalues are percentages are > 70%, and there is no correct way to impute these
    # df_train.drop((nullvalues[nullvalues > 0].index).tolist(),axis=1, inplace= True)
    # df_test.drop((nullvalues[nullvalues > 0].index).tolist(),axis=1, inplace= True)

    # new column Total_Recharge_Amount_Data = av_rech_amt*total_rech_data for each month
    df_train['total_rech_amt_data_6'] = df_train.av_rech_amt_data_6 * df_train.total_rech_data_6
    df_train['total_rech_amt_data_7'] = df_train.av_rech_amt_data_7 * df_train.total_rech_data_7
    df_train['total_rech_amt_data_8'] = df_train.av_rech_amt_data_8 * df_train.total_rech_data_8

    df_test['total_rech_amt_data_6'] = df_test.av_rech_amt_data_6 * df_test.total_rech_data_6
    df_test['total_rech_amt_data_7'] = df_test.av_rech_amt_data_7 * df_test.total_rech_data_7
    df_test['total_rech_amt_data_8'] = df_test.av_rech_amt_data_8 * df_test.total_rech_data_8

    df_train['total_avg_rech_amt_6_7'] = (df_train.total_rech_amt_6 + df_train.total_rech_amt_data_6 + df_train.total_rech_amt_7+ df_train.total_rech_amt_data_7)/2
    df_test['total_avg_rech_amt_6_7'] = (df_test.total_rech_amt_6 + df_test.total_rech_amt_data_6 + df_test.total_rech_amt_7+ df_test.total_rech_amt_data_7)/2

    ## Some more columns to delete
    list_total_ic_cols = df_train.columns[df_train.columns.str.contains('total_ic_mou|std_ic_mou|loc_ic_mou',regex=True)]
    df_train.drop(list_total_ic_cols,axis=1,inplace=True)
    list_total_ic_cols.tolist()

    list_total_ic_cols = df_test.columns[df_test.columns.str.contains('total_ic_mou|std_ic_mou|loc_ic_mou',regex=True)]
    df_test.drop(list_total_ic_cols,axis=1,inplace=True)
    list_total_ic_cols.tolist()

    list_total_og_cols = df_train.columns[df_train.columns.str.contains('total_og_mou|std_og_mou|loc_og_mou',regex=True)]
    df_train.drop(list_total_og_cols,axis=1,inplace=True)
    list_total_og_cols.tolist()

    list_total_og_cols = df_test.columns[df_test.columns.str.contains('total_og_mou|std_og_mou|loc_og_mou',regex=True)]
    df_test.drop(list_total_og_cols,axis=1,inplace=True)
    list_total_og_cols.tolist()

    ##handeling outliers

    def remove_outliers_iqr(df, column):
        # Calculate Q1 (25th percentile) and Q3 (99th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.99)
        # Calculate IQR
        IQR = Q3 - Q1  
        # Calculate lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR   
        # Filter the DataFrame
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]   
        return filtered_df

    df_train=remove_outliers_iqr(df_train,df_train.columns)

    df_test=remove_outliers_iqr(df_test,df_test.columns)

    ## Apply the KNN Imputer to both train and test data to impute the Null values post removal of outliers 
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)  
    imputer.fit(df_train)
    X_imputed = imputer.transform(df_train)

    df_train = pd.DataFrame(X_imputed, columns=df_train.columns)

    imputer = KNNImputer(n_neighbors=5)  
    imputer.fit(df_test)
    X_imputed = imputer.transform(df_test)

    df_test = pd.DataFrame(X_imputed, columns=df_test.columns)

    ## Filtering High-Value Customers.

    high_value_filter = df_train.total_avg_rech_amt_6_7.quantile(0.7)
    high_val_cust_df_train = df_train[df_train.total_avg_rech_amt_6_7 > high_value_filter]

    high_value_filter = df_test.total_avg_rech_amt_6_7.quantile(0.7)
    high_val_cust_df_test = df_test[df_test.total_avg_rech_amt_6_7 > high_value_filter]

    df_train=high_val_cust_df_train.copy()

    df_unseen_test=df_test.copy()
    
    ## MINMAX Scaling

    from sklearn.preprocessing import MinMaxScaler
    data2=df_train.drop(['id','churn_probability'],axis=1)
    data3=df_test.drop('id',axis=1)
    #initiate a object
    scaler=MinMaxScaler()

    ## Create a list of Numerical vars
    var1=data2.columns
    var2=data3.columns
    #Fit the method
    df_train[var1]= scaler.fit_transform(df_train[var1])
    df_test[var2]= scaler.transform(df_test[var2])

    df_train['churn_probability']=df_train['churn_probability'].replace({'No churn': 0, 'Churn': 1})

    df_train.id = df_train.id.astype(int)
    df_train.churn_probability = df_train.churn_probability.astype(int)
    df_test.id = df_test.id.astype(int)

    df_unseen_test=df_test.copy()

    ## Train test Split

    from sklearn.model_selection import train_test_split 
    df_train,df_test = train_test_split(df_train, train_size=0.7,random_state=100)

    y_train=df_train['churn_probability']
    X_train=df_train.drop(['id','churn_probability'],axis=1)
 
    y_test=df_test['churn_probability']
    X_test=df_test.drop(['id','churn_probability'],axis=1)

    ## Lets use SMOTE to Remove the data imbalance
    from imblearn.datasets import make_imbalance
    from imblearn.under_sampling import NearMiss
    from imblearn.pipeline import make_pipeline
    from imblearn.metrics import classification_report_imbalanced

    from imblearn.over_sampling import SMOTE
    from collections import Counter


    # oversampling the train dataset using SMOTE
    smt = SMOTE()
    X_train, y_train = smt.fit_resample(X_train, y_train)
    X_test, y_test = smt.fit_resample(X_test, y_test)

    # Save files with compression for faster I/O
    X_train.to_csv('/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/X_train.csv', index=False)
    X_test.to_csv('/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/X_test.csv', index=False)
    y_train.to_csv('/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/y_train.csv', index=False)
    y_test.to_csv('/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/y_test.csv', index=False)

    # Load data
    
    X_train = pd.read_csv('/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/X_train.csv')
    X_test = pd.read_csv('/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/X_test.csv')
    y_train = pd.read_csv('/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/y_train.csv')
    y_test = pd.read_csv('/Users/I353375/Downloads/MLOps/airflow/Telecom_Churn/y_test.csv')
    
    print("Preprocessing completed efficiently")

if __name__ == "__main__":
    telecom_preprocess_data()
