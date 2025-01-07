import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    df = pd.read_excel(path)
    df.reset_index(drop=True, inplace=True)
    df.columns = [col.replace(' ', '_') if ' ' in col else col for col in df.columns]
    
    # Sort values
    df = df.sort_values(by=['WELL_NAME', 'TEST_DATE'])
    
    # Convert TEST_DATE to datetime
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

def add_feat(df):
    if 'FM_GAS' in df.columns:
        if 'OIL' in df.columns:
            df['CGR'] = df['OIL'] / (df['FM_GAS'] / 1000)
        else:
            raise KeyError("Column 'OIL' not found in DataFrame")
    else:
        raise KeyError("Column 'FM_GAS' not found in DataFrame")
    return df

def coleman(df):
    td=2.992
    sg_gas=0.7
    a=(3.14*(td**2)/(4*144))
    wht=100
    ppc=677+(15*sg_gas)-(37.5*(sg_gas**2))
    tpc=168+(325*sg_gas)-(12.5*(sg_gas**2))
    if 'WHP' in df.columns:
        df['v_gas'] = 5.34 * ((67-0.0031*df['WHP'])**0.25/(0.0031*df['WHP'])**0.5)
        df['ppr'] = df['WHP']/ppc   
    else:
        raise KeyError("Column 'WHP' not found in DataFrame")
    df['tpr']=(df['WHT']+460)/tpc
    df['z_gas']=1+((0.274*df['ppr']**2)/(10**(0.8157*df['tpr'])))-((3.53*df['ppr'])/(10**(0.9813*df['tpr'])))
    df['qgc']=(3.06*df['WHP']*a*df['v_gas'])/((df['WHT']+460)*df['z_gas'])
    # if df['FM_GAS'] > df['qgc']:
    #     df['Above_Critical'] = 'No'
    # else:
    #     df['Above_Critical'] = 'Yes'
    return df


def main():
    path='P:\project\pythonpro\myvenv\gas-well-mon\data\df_comb.xlsx'
    df=load_data(path)
    #combine df_attaka and df_nib
    df=add_feat(df)
    df.to_excel('P:/project/pythonpro/myvenv/gas-well-mon/data/df.xlsx')
    print(df.describe())
    if 'LF_GAS' in df.columns:
        df_gas = df[df['LF_GAS'] == 0]
    else:
        raise KeyError("Columns 'CGR' or 'LF_GAS' not found in DataFrame")
    df_gas = coleman(df_gas)
    print('Length of Data:',len(df_gas))
    print(df_gas.columns)
    print(df_gas.head())
    print(df_gas.dtypes)
    df.to_csv('P:/project/pythonpro/myvenv/gas-well-mon/data/sapi06_well_test.csv')
if __name__ == "__main__":
    main()