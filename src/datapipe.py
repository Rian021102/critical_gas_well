import pandas as pd

def load_data(path):
    df=pd.read_excel(path,skiprows=5)
    df.reset_index(drop=True, inplace=True)
    df.columns = [col.replace(' ', '_') if ' ' in col else col for col in df.columns]
    df = df.sort_values(by=['WELL_NAME', 'TEST_DATE'])
    # Ensure TEST_DATE is in datetime format
    df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE']) 
    df['WHT']=100
    return df

def main ():
    path_attaka='P:/project/pythonpro/myvenv/gas-well-mon/data/attaka.xlsx'
    path_nib='P:/project/pythonpro/myvenv/gas-well-mon/data/nib.xlsx'
    path_seping='P:/project/pythonpro/myvenv/gas-well-mon/data/sepinggan.xlsx'
    path_yakin='P:/project/pythonpro/myvenv/gas-well-mon/data/yakin.xlsx'
    df_attaka=load_data(path_attaka)
    df_nib=load_data(path_nib)
    df_seping=load_data(path_seping)
    df_yakin=load_data(path_yakin)
    #combine df_attaka and df_nib
    df= pd.concat([df_attaka, df_nib,df_seping,df_yakin], ignore_index=True)
    print(df.head())
    #save df as csv
    df.to_csv('P:/project/pythonpro/myvenv/gas-well-mon/data/well_test.csv')
    

if __name__ == "__main__":
    main()
