# -*- coding: utf-8 -*-


"""### Packages imports"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import autosklearn.regression

import plotly.express as px

from joblib import dump

import shap

import datetime

import logging

import matplotlib.pyplot as plt

external_data_path = "../data/external/"
processed_data_path = "../data/processed/"
reports_path = "../reports/"

model_path = "../models/"

timesstr = str(datetime.datetime.now()).replace(' ', '_')

log_config = {
    "version": 1,
    "root": {
        "handlers": ["console"],
        "level": "DEBUG"
    },
    "handlers": {
        "console": {
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        }
    },
    "formatters": {
        "std_out": {
            "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : (Process Details : (%(process)d, %(processName)s), Thread Details : (%(thread)d, %(threadName)s))\nLog : %(message)s",
            "datefmt": "%d-%m-%Y %I:%M:%S"
        }
    },
}

logging.config.dictConfig(log_config)

"""Please Download the data from [this source](https://drive.google.com/file/d/1MUZrfW214Pv9p5cNjNNEEosiruIlLUXz/view?usp=sharing), and upload it on your Introduction2DataScience/data google drive folder.

<a id='P1' name="P1"></a>
## [Loading Data and Train-Test Split](#P0)
"""

df = pd.read_csv(f'{external_data_path}california_housing.csv')

test_size = 0.2
random_state = 0

train, test = train_test_split(df, test_size=test_size, random_state=random_state)

logging.info(f'train test split with test_size={test_size} and random state={random_state}')

train.to_csv(f'{processed_data_path}CaliforniaTrain.csv', index=False)

train = train.copy()

test.to_csv(f'{processed_data_path}CaliforniaTest.csv', index=False)

test = test.copy()

"""<a id='P2' name="P2"></a>
## [Modelling](#P0)
"""

X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]

total_time = 600
per_run_time_limit = 30

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=total_time,
    per_run_time_limit=per_run_time_limit,
    logging_config=log_config
)
automl.fit(X_train, y_train)

logging.info(
    f'Ran autosklearn regressor for a total time of {total_time} seconds, with a maximum of {per_run_time_limit} seconds per model run')

dump(automl, f'{model_path}model{timesstr}.pkl')

logging.info(f'Saved regressor model at {model_path}model{timesstr}.pkl ')

logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())

"""<a id='P2' name="P2"></a>
## [Model Evluation and Explainability](#P0)

Let's separate our test dataframe into a feature variable (X_test), and a target variable (y_test):
"""

X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

"""#### Model Evaluation

Now, we can attempt to predict the median house value from our test set. To do that, we just use the .predict method on the object "automl" that we created and trained in the last sections:
"""

y_pred = automl.predict(X_test)

"""Let's now evaluate it using the mean_squared_error function from scikit learn:"""

logging.info(
    f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test, y_test)}")

"""we can also plot the y_test vs y_pred scatter:"""

df = pd.DataFrame(np.concatenate((X_test, y_test.to_numpy().reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1))

df.columns = ['longitude', 'latitude', 'housing_median_age', 'households',
              'median_income', 'bedroom_per_room',
              'rooms_per_household', 'population_per_household', 'True Target', 'Predicted Target']

fig = px.scatter(df, x='Predicted Target', y='True Target')
fig.write_html(f"{reports_path}residualfig_{timesstr}.html")

logging.info(f"Figure of residuals saved as {reports_path}residualfig_{timesstr}.html")

"""#### Model Explainability"""

explainer = shap.KernelExplainer(model=automl.predict, data=X_test.iloc[:50, :], link="identity")

# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X=X_test.iloc[X_idx:X_idx + 1, :], nsamples=100)
X_test.iloc[X_idx:X_idx + 1, :]
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(base_value=explainer.expected_value,
                shap_values=shap_value_single,
                features=X_test.iloc[X_idx:X_idx + 1, :],
                show=False,
                matplotlib=True
                )
plt.savefig(f"{reports_path}shap_example_{timesstr}.png")
logging.info(f"Shapley example saved as {reports_path}shap_example_{timesstr}.png")

shap_values = explainer.shap_values(X=X_test.iloc[0:50, :], nsamples=100)

fig = shap.summary_plot(shap_values=shap_values,
                        features=X_test.iloc[0:50, :],
                        show=False)
plt.savefig(f"{reports_path}shap_summary_{timesstr}.png")
logging.info(f"Shapley summary saved as {reports_path}shap_summary_{timesstr}.png")
