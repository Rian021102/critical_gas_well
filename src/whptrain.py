import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from basetrain import base_train, mlp_reg
from eda import eda_train
from rfwhp import train_whp



def load_data(path):
    df = pd.read_csv(path)
    df.columns = [col.replace(' ', '_') if ' ' in col else col for col in df.columns]
    
    # Sort values
    df = df.sort_values(by=['WELL_NAME', 'TEST_DATE'])
    
    # Convert TEST_DATE to datetime
    df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE'])
        
    # Convert negative values to positive for numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        df[col] = df[col].abs()
    
    df['GAS']=df['FM_GAS']/1000

    if 'FM_GAS' in df.columns:
        if 'OIL' in df.columns:
            df['CGR'] = df['OIL'] / (df['FM_GAS'] / 1000)
        else:
            raise KeyError("Column 'OIL' not found in DataFrame")
    else:
        raise KeyError("Column 'FM_GAS' not found in DataFrame")
    
    # Drop unwanted columns
    columns_to_drop = ['Unnamed:_0', 'Unnamed:1', 'Unnamed:_11', 'Unnamed:_13', 'Unnamed:_18', 'Unnamed:_19']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df

def process_data(X_train,X_test):
    #select features
    X_train=X_train[['CSG','SEP','CHK']]
    X_test=X_test[['CSG','SEP','CHK']]
    return X_train,X_test

def main():
    path='/Users/rianrachmanto/pypro/project/gas-well-mon/data/sapi06_well_test.csv'
    df=load_data(path)
    if 'LF_GAS' in df.columns:
         df_gas = df[df['LF_GAS'] == 0]
    else:
        raise KeyError("Columns 'LF_GAS' not found in DataFrame")
    
        #defining X and Y
    X = df_gas.drop(columns=['WHP'])
    y = df_gas['WHP']

    # Split the data and print the shapes to verify
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    # df_train=eda_train(X_train,y_train)
    X_train,X_test=process_data(X_train,X_test)
    # scaler = StandardScaler()
    # X_train_scaled=scaler.fit_transform(X_train)
    # X_test_scaled=scaler.transform(X_test)
    # base_train(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)
    # mlp_reg(X_train_scaled, y_train, X_test_scaled, y_test)
    mse, r2, mae, y_pred, feature_importances = train_whp(X_train, y_train, X_test, y_test)
    
    print('Mean Squared Error: ', mse)
    print('R2 score: ',r2)
    print('mae: ', mae)
    #plot predictions
    plt.figure(figsize=(10,6))
    plt.scatter(y_test,y_pred, alpha=0.3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual WHP vs Predicted WHP')
    plt.show()
    #plot feature importances
    plt.figure(figsize=(10,6))
    sns.barplot(x=feature_importances, y=X_train.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.show()

    #

if __name__ == "__main__":
    main()