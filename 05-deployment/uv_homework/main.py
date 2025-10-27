def main():
    print("Hello from uv-homework!")


if __name__ == "__main__":
    main()

from fastapi import FastAPI
app = FastAPI()

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/")
def predict_subscription(client: Client):
    # Example scoring (replace this logic with your real model)
    # Here is a dummy probability calculation:
    prob = 0.334 if client.lead_source == "organic_search" else 0.534
    return {"subscription_probability": prob}