from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml import model, data
import pandas as pd


# Instantiate the app.
app = FastAPI(
    description="API to predict income > $50K",
    version="1.0.0",
)

# Define a GET on the specified endpoint.


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 42,
                "workclass": "Private",
                "fnlgt": 159449,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 5178,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


# Load model
predictor = model.model_load()


@app.post("/income_prediction/")
async def income_pred(input_parameters: Data):
    to_process = pd.DataFrame(
        data=[
            input_parameters.dict(
                by_alias=True)],
        index=[0])
    X, _, _, _ = data.process_data(to_process, training=False)
    prediction = model.inference(predictor, X)

    if (prediction[0] == 0):
        return 'Income will be probably lower than $50k'
    else:
        return 'Income will be probably higher than $50k'
