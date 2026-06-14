from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():


    age = float(request.form["age"])

    if age < 18 or age > 60 :
        return render_template(
            "index.html",
            prediction_text = "age must be between 18 and 60"
        )

    sex = int(request.form["sex"])
    bmi = float(request.form["bmi"])

    if bmi < 16 or bmi > 55:
        return render_template(
            "index.html",
            prediction_text = "BMI must be between 16 and 55"
        )

    children = int(request.form["children"])
    smoker = int(request.form["smoker"])

    features = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi' :[bmi],
        'children': [children],
        'smoker' :[smoker],
    })

    prediction = model.predict(features)

    return render_template(
      "index.html",
      prediction_text=f"Estimated Insurance Charges: ₹ {prediction[0]:,.2f}"
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
