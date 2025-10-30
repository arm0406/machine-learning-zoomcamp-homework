import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained pipeline
with open('pipeline_v1.bin', 'rb') as f:
    model = pickle.load(f)

# Define input data schema
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

@app.post("/")
def predict(client: Client):
    data = client.dict()
    probability = model.predict_proba([data])[0, 1]
    return {"subscription_probability": probability}