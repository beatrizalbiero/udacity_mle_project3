from fastapi.testclient import TestClient


# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_api_locally_post_class_higher_income():
    higher_example = {
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

    r = client.post("/income_prediction/", json=higher_example)
    assert r.status_code == 200
    assert r.json() == "Income will be probably higher than $50k"


def test_api_locally_post_class_lower_income():
    higher_example = {
                      "age": 39,
                      "workclass": "State-gov",
                      "fnlgt": 77516,
                      "education": "Bachelors",
                      "education-num": 13,
                      "marital-status": "Never-married",
                      "occupation": "Adm-clerical",
                      "relationship": "Not-in-family",
                      "race": "White",
                      "sex": "Male",
                      "capital-gain": 2174,
                      "capital-loss": 0,
                      "hours-per-week": 40,
                      "native-country": "United-States"
                    }

    r = client.post("/income_prediction/", json=higher_example)
    assert r.status_code == 200
    assert r.json() == "Income will be probably lower than $50k"
