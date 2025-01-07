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

def main():
    path='P:\project\pythonpro\myvenv\gas-well-mon\data\WellTest A_SHELF_NORTH-FC2_NIB-SAPI-SAPI_SAPI-6RD1-1Jan2020-20Nov2024.xlsx'
    df=load_data(path)
    print(df.head())
    df.to_csv('P:/project/pythonpro/myvenv/gas-well-mon/data/sapi06_well_test.csv')



if __name__ == "__main__":
    main()
