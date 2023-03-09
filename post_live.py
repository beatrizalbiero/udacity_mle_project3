import json # You need this to pass in your row data
import requests

row = {
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

response = requests.post(
    # Notice the url ends at the inference endpoint
    url="https://udacity-deployment-m9su.onrender.com/income_prediction/",
    json=row # Notice the change here also
)

print(response.status_code)
print(response.json())
