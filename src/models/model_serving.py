from flask import Flask, jsonify
from flask import request
import pandas as pd
from joblib import load
import sklearn

app = Flask(__name__)
automl = load('models/model2022-04-07_23:26:27.161837.pkl')


@app.route("/", methods=['POST'])
def home():
    json_data = request.get_json()
    value = list(automl.predict(
        pd.DataFrame([json_data['features']], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                                                       'total_bedrooms', 'population', 'households', 'median_income'])))
    return jsonify(prediction=value[0], json=True)


app.run(host='0.0.0.0', port=5000, daemon=True)
