import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import pickle

model_path = '/Users/rianrachmanto/pypro/project/gas-well-mon/model/modelwhp.pkl/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

def load_data(path):
    df=pd.read_excel(path,skiprows=5)
    df.columns = [col.replace(' ', '_') if ' ' in col else col for col in df.columns]
    df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE'])
    df['WHT'] = 100
    
    # Convert negative values to positive for numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        df[col] = df[col].abs()
    
    # Drop unwanted columns
    columns_to_drop = ['Unnamed:0', 'Unnamed:1', 'Unnamed:_11', 'Unnamed:_13', 'Unnamed:_18', 'Unnamed:_19']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df

def main():
    path='/Users/rianrachmanto/pypro/project/gas-well-mon/data/test_model.xlsx'
    df=load_data(path)
    print(df.head())
    df_sel=df[['CSG','SEP','CHK']]
    df_sel['PRED_WHP']=model.predict(df_sel)
    #insert df['GAS'] to df_sel
    df_sel['WHP']=df['WHP']
    print(df_sel)

if __name__ == '__main__':
    main()