from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load the model from the base Docker image
with open('pipeline_v2.bin', 'rb') as f:
    model = pickle.load(f)

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/")
def predict(client: Client):
    X = [[
        client.lead_source,
        client.number_of_courses_viewed,
        client.annual_income
    ]]
    proba = model.predict_proba(X)[0, 1]
    return {"subscription_probability": float(proba)}