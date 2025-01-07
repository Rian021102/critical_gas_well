from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict
import joblib

#make fast api that predict dataframe
app = FastAPI()

# Load the trained model
model=joblib.load("/Users/rianrachmanto/pypro/project/gas-well-mon/model/model_rate.pkl/model.pkl")

def make_prediction(data: pd.DataFrame) -> List[int]:
    # Make predictions using the loaded model
    predictions = model.predict(data)
    return predictions.tolist()
class DataInput(BaseModel):
    data: List[Dict[str, float]]

#make endpoint to predict from dataframe without having to upload csv file
@app.post("/predict")
async def predict(data_input: DataInput):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame(data_input.data)
        # Make predictions
        predictions = make_prediction(data)

        return JSONResponse(content={"predictions": predictions}, status_code=200)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}",
        )
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)