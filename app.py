from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import statsmodels.api as sm
from contextlib import asynccontextmanager
import pandas as pd
from typing import  Literal

from urllib3 import request


models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the pre-trained models
    models['scope1'] = sm.load('scope1_model.pkl')
    models['scope2'] = sm.load('scope2_model.pkl')
    yield
    # Cleanup code if needed
    models.clear()


app = FastAPI(
        title  ="NEXYGEN API",
        description="This is an API endpoint that forcat scope 1 & scope 2 emissions for NEXYGEN based on pre-trained SARIMA models",
        version= "0.1.0",
        lifespan=lifespan)
    

# Define a structure/validation for our request and response models
class ForecastRequest(BaseModel):
    emission_type: Literal['scope1', 'scope2']
    steps : int


class ForecastResponse(BaseModel):
    emission_type: str
    forecast: list[float]
    dates: list[str]
    last_training_date: str


    @app.get("/")
    def health_check():
        return {"Status": "O.K", 
                "message": "NEXYGEN API is up and running!"
                 }

@app.post("/forecast", response_model=ForecastResponse)
def forecast_emissions(request: ForecastRequest):
    key = 'scope1' if request.emission_type == 'scope1' else 'scope2'
    model = models.get(key)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        pred = model.forecast(steps=request.steps)
        print(pred)
        
        # Use fixed last training date for monthly forecasts
        last_training_date = '2024-10-01'
        
        # Generate monthly forecast dates
        last_date = pd.to_datetime(last_training_date)
        dates = [(last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m-%d') for i in range(request.steps)]

        return ForecastResponse(
            emission_type=request.emission_type,
            forecast=pred.tolist(),
            dates=dates,
            last_training_date=last_training_date
        )
    except Exception as e:
        print(f"Error in forecast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

        #dates=sts[(d) for d in pred.index]

#last_date = model.data.endog.index[-1]
#forecast_dates = pd.date_range(start=last_date, periods=request.steps + 1, freq='M')[1:]