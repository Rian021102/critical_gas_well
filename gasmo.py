import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import pickle

st.title("Smart Gas Well Monitoring for Liquid Load Up Problem")
st.header("Well Data")
# Load the model
model_path = '/Users/rianrachmanto/pypro/project/gas-well-mon/model/model_rate.pkl/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

def load_data(path):
    df = pd.read_csv(path)
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
    
    df['GAS']=df['FM_GAS']/1000
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
    return df

def visualize(df):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot FM_GAS and qgc
    ax.plot(df['TEST_DATE'], df['GAS'], label='Field Measured Gas Rate', marker='o')
    ax.plot(df['TEST_DATE'], df['qgc'], label='Critical Gas Rate', marker='s')
    
    # Customize plot
    ax.set_xlabel('Test Date')
    ax.set_ylabel('Gas Rate (MSCFD)')
    ax.set_title(f'Gas Rate vs Critical Rate for {df["WELL_NAME"].iloc[0]}')
    ax.grid(True)
    ax.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig


def main():
    path='/Users/rianrachmanto/pypro/project/gas-well-mon/data/sapi06_well_test.csv'
    df=load_data(path)
    df=add_feat(df)
    if 'CGR' in df.columns and 'LF_GAS' in df.columns:
        df_gas = df[(df['CGR'] <= 50) & (df['LF_GAS'] == 0)]
    else:
        raise KeyError("Columns 'CGR' or 'LF_GAS' not found in DataFrame")

    well_name=st.sidebar.selectbox('Select Well',df_gas['WELL_NAME'].unique())

    df_well=df_gas[df_gas['WELL_NAME']==well_name][['STRING_NAME','WELL_NAME',
                                                      'TEST_DATE','ZONE','SAND','HRS','CHK','GL','WHP','WHT','CSG',
                                                      'SEP','OIL_GRV','BS&W','SAND.1','EML','GOR','GLR','GLOR',
                                                      'GROSS','WTR','OIL','GAS','FM_GAS','LF_GAS','TL_GAS','CGR']]
    
    df_well=coleman(df_well)
    df_well['STATUS'] = np.where(df_well['GAS'] > df_well['qgc'], 'Above critical', 'Below critical')

    st.write(df_well.tail(10))
    if not df_well.empty:
        fig = visualize(df_well)
        st.pyplot(fig)
    
    #get the newest data from df_well 
    latest_data = df_well.iloc[-1]
    st.write("\nLatest Well Data:")
    columns = st.columns(7)

    headers = ['Date', 'WHP (PSI)', 'Choke', 'Gas (MMSCFD)', 'Oil (BOPD)', 'WTR (BWPD)', 'Crit Gas (MMSCFD)']
    values = [
        latest_data['TEST_DATE'].strftime('%Y-%m-%d'),
        f"{latest_data['WHP']:.2f}",
        str(latest_data['CHK']),
        f"{latest_data['GAS']:.2f}",
        f"{latest_data['OIL']:.2f}",
        f"{latest_data['WTR']:.2f}",
        f"{latest_data['qgc']:.2f}"
    ]

    for col, header, value in zip(columns, headers, values):
        with col:
            st.subheader(header, divider='red')
            st.subheader(value)

    if latest_data['GAS'] < latest_data['qgc']:
        optimal_chk = latest_data['CHK']
        while optimal_chk <= 256:
            df_predict = latest_data[['WHP', 'CSG', 'SEP', 'BS&W', 'GOR', 'WTR', 'OIL']].to_frame().transpose()
            df_predict['CHK'] = optimal_chk
            predicted_gas = model.predict(df_predict)[0]
            if predicted_gas > latest_data['qgc']:
                st.write(f"Required Choke Setting: {optimal_chk} to achieve Gas Production: {predicted_gas:.2f} MMSCFD")
                break
            optimal_chk += 2
        if optimal_chk > 256:
            st.write("Maximum Choke Setting reached without achieving desired gas production.")
    else:
        st.write("Current gas production is above the critical rate.")

if __name__ == '__main__':
    main()
    