from flask import Flask
import pickle
import numpy as np

app = Flask(__name__)

## Load modeliing artefacts
filename = 'final_rf_model.sav'
load_model = pickle.load(open(filename, 'rb'))
filename = 'ohe_encoder.pkl'
load_encoder = pickle.load(open(filename, 'rb'))

def categorical_encoding(encoder, input_df):
    
    X_encoded = pd.DataFrame(encoder.transform(input_df[['driver_gender', 'driver_marital_status', 'driver_license_emirates_issued', 
                                                         'driver_nationality','veh_body_type', 'veh_reg_type', 'veh_make']]))
    X_encoded.columns = encoder.get_feature_names(['gender', 'marital_status', 'licence_emirates', 'nationality','body_type', 'reg_type', 'make'])
    input_df.drop(['driver_gender', 'driver_marital_status', 'driver_license_emirates_issued', 
                                 'driver_nationality','veh_body_type', 'veh_reg_type', 'veh_make'], axis = 1, inplace=True)
    X = pd.concat([input_df, X_encoded], axis = 1)
    return X

## Routing requests to /api
@app.route("/api", methods=['POST'])
def return_pred_proba(model, input_df):
    # Get user inputs as json
    input_json = request.get_json(force = True)
    # Convert json to dataframe
    input_json.update((x, [y]) for x,y in data.items())
    input_df = pd.DataFrame.from_dict(input_json)
    # Data pre-processing
    processed_df = categorical_encoding(input_df, load_encoder)
    return model.predict_proba(processed_df)
    

if __name__ == '__main__':
    app.run(port =5000, debug = True)