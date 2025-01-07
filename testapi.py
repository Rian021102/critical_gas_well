import pandas as pd
import requests  # Import the requests library

def load_data(path):
    df = pd.read_excel(path, skiprows=5)
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
    path = '/Users/rianrachmanto/pypro/project/gas-well-mon/data/test_model.xlsx'
    df = load_data(path)
    print(df.head())
    df_sel = df[['WHP', 'CSG', 'SEP', 'GOR', 'OIL', 'CHK']].fillna(0)  # Ensure no NaN values

    # Prepare the data for the API call
    data = {
        "data": df_sel.to_dict(orient='records')  # This converts the DataFrame to a list of dictionaries
    }

    # Define the API endpoint
    url = 'http://127.0.0.1:8000/predict'

    # Make the prediction by calling the API
    response = requests.post(url, json=data)
    if response.status_code == 200:
        predictions = response.json()
        print("Predictions:", predictions)
    #make the prediction result as a new column of the dataframe
        df['PRED_RATE'] = predictions['predictions']
        print(df)
    else:
        print("Failed to get predictions", response.status_code, response.text)

if __name__ == "__main__":
    main()

