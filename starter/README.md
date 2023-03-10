# Udacity Project: Deploying an Income Classification Model API
### Owner: Beatriz Albiero

## Folder Structure

```
.
└── starter
    ├── data                       # Folder containing the data used in the model
    │   └── census.csv
    ├── main.py                    # Script that creates the API
    ├── model                      
    ├── model_card.md              # Model Card
    ├── modelling.ipynb           # Notebook used to implement the model
    ├── post_live.py              # Post Live file
    ├── README.md
    ├── requirements.txt
    ├── sanitycheck.py
    ├── screenshots
    │   ├── continuous_deployment.png
    │   ├── example.png
    │   ├── income_by_race_test_count.png
    │   ├── income_by_race_test.png
    │   ├── income_by_race_train_count.png
    │   ├── income_by_race_train.png
    │   ├── income_by_sex_test_count.png
    │   ├── income_by_sex_test.png
    │   ├── income_by_sex_train_count.png
    │   ├── income_by_sex_train.png
    │   ├── live_get.png
    │   └── live_post.png
    ├── setup.py
    ├── slice_output.txt
    ├── starter
    │   ├── __init__.py
    │   ├── ml
    │   │   ├── clf_model.sav            # Saved model
    │   │   ├── data.py                  # Preprocess Data
    │   │   ├── encoder.pkl              # Saved encoder
    │   │   ├── model.py                 # Script that contains functions to assist in the model training
    │   │   ├── notebook
    │   │   │   └── modelling.ipynb
    │   │   └── tests
    │   │       └── unit_tests.py          # Tests functions in model.py
    │   └── train_model.py                 # Script to train the ML model
    └── test_main.py                    # Test main.py

```

## API

This api is running in the website [https://udacity-deployment-m9su.onrender.com/](https://udacity-deployment-m9su.onrender.com/).

You can test it by sending it a dictionary of features to [https://udacity-deployment-m9su.onrender.com/income_prediction/](https://udacity-deployment-m9su.onrender.com/income_prediction/)

Here is an example of dictionary:
```
example = {
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
  ```

  This API is expected to return the sentences:
  - 'Income will be probably lower than $50k' given a prediction of class 0 or
  - 'Income will be probably higher than $50k' given a prediction of class 1

  The script ```post_live.py``` executes this exact test. You can check the response in ```screenshots/live_post.png```


  ## Model

  As the performance of the model is not central to the purposes of this exercise, I chose to develop a simple Logistic Regression model. It is not tuned.

  You can check the code for the model in ```starter/starter/ml/model.py```.

  The model is saved in the file  ```starter/starter/ml/clf_model.sav```

  This script also saved the encoder pickle for future preprocessing of the data in the file ```starter/starter/ml/encoder.pkl```.

  You can also check a model card for this project in the file ```starter/model_card.md``` and check the output of sliced performance in the file ```starter/slice_output.txt```

  ### Tests
  This model gets tested in the file ```starter/starter/ml/unit_tests.py```.


  ## Continuous deployment
  The current build is available [here](https://github.com/beatrizalbiero/udacity_mle_project3/actions/runs/4386885863/jobs/7681500625) and a screenshot of the 6 tests is available in the folder ```starter/screenshots/```
