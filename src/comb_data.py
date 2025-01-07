import os
import pandas as pd
 
def load_excel_to_df(directory, suffix_strip=None):
    df_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            excel_file = pd.read_excel(file_path, skiprows=5)
            if suffix_strip and filename.endswith(suffix_strip + '.xlsx'):
                df_name = filename.replace(suffix_strip + '.xlsx', '')
            else:
                df_name = filename.replace('.xlsx', '')
            df_dict[df_name] = excel_file  # Add DataFrame to dictionary
            print(f'Loaded {df_name}')
 
    return df_dict
 
field_path = 'P:\project\pythonpro\myvenv\gas-well-mon\data'
df_field_dict = load_excel_to_df(field_path, suffix_strip='data')
 
# If you want to concatenate all DataFrames into a single DataFrame
df_concatenated = pd.concat(df_field_dict.values(), ignore_index=True)
df_concatenated.reset_index(drop=True,inplace=True)
df_concatenated.columns = [col.replace(' ', '_') if ' ' in col else col for col in df_concatenated.columns]
df_concatenated = df_concatenated.sort_values(by=['WELL_NAME', 'TEST_DATE'])
# Ensure TEST_DATE is in datetime format
df_concatenated['TEST_DATE'] = pd.to_datetime(df_concatenated['TEST_DATE']) 
df_concatenated['WHT']=100
print(df_concatenated.head())
df_concatenated.to_excel('P:/project/pythonpro/myvenv/gas-well-mon/data/df_comb.xlsx')
