import json
import pickle
import numpy as np
from joblib import load
from flask import Flask, request, url_for, redirect, render_template, jsonify
#

flask_app = Flask(__name__)

#ML model path
model_path = "ML_Model/model.pkl"
model = load(model_path)


@flask_app.route('/')
def home():
    return render_template("home_death.html")

@flask_app.route('/predict',methods=['POST'])
def predict():
    text_feature = np.array([float(x) for x in request.form.values()]).reshape(1,-1)
    prediction = np.max(model.predict_proba(text_feature))

    return render_template('death_after.html',data=prediction)


if __name__ == "__main__":
    flask_app.run(host ='0.0.0.0', debug=True)