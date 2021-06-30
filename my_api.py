import pandas as pd
from flask import Flask, jsonify, request
import json
import pickle
import numpy as np

myapp = Flask(__name__)

## Load modeliing artefacts
filename = 'final_rf_model.pkl'
load_model = pickle.load(open(filename, 'rb'))
filename = 'ohe_encoder.pkl'
load_encoder = pickle.load(open(filename, 'rb'))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def categorical_encoding(encoder, input_df):
    X_encoded = pd.DataFrame(encoder.transform(input_df[['driver_gender', 'driver_marital_status', 'driver_license_emirates_issued', 
                                                         'driver_nationality','veh_body_type', 'veh_reg_type', 'veh_make']]))
    X_encoded.columns = encoder.get_feature_names(['gender', 'marital_status', 'licence_emirates', 'nationality','body_type', 'reg_type', 'make'])
    input_df.drop(['driver_gender', 'driver_marital_status', 'driver_license_emirates_issued', 
                                 'driver_nationality','veh_body_type', 'veh_reg_type', 'veh_make'], axis = 1, inplace=True)
    X = pd.concat([input_df, X_encoded], axis = 1)
    return X

## Routing requests to /api
@myapp.route("/api", methods=['POST'])
def return_pred_proba():
    # Get user inputs as json
    input_json = request.get_json(force = True)
    # Convert json to dataframe
    input_json.update((x, [y]) for x, y in input_json.items())
    input_df = pd.DataFrame.from_dict(input_json)
    # Data pre-processing
    processed_df = categorical_encoding(load_encoder, input_df)
    pred_proba_list = load_model.predict_proba(processed_df).tolist()
    pred_proba_json = json.dumps({'results' : pred_proba_list}, cls = NumpyEncoder)
    return pred_proba_json
    

if __name__ == '__main__':
    myapp.run(port = 5000, debug = True)