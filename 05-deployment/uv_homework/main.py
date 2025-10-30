import pickle
from fastapi import FastAPI
from pydantic import BaseModel

with open('pipeline_v1.bin', 'rb') as f:
    model = pickle.load(f)

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

@app.post("/")
def predict(client: Client):
    x = [client.dict()]
    proba = model.predict_proba(x)[0, 1]
    return {"subscription_probability": proba}
