from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("final_model.pkl", "rb"))

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
    app.run(debug=True)
