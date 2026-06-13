from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():


    age = float(request.form["age"])
    sex = int(request.form["sex"])
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = int(request.form["smoker"])

    features = np.array([[age, sex, bmi, children, smoker]])

    prediction = model.predict(features)

    return render_template(
      "index.html",
      prediction_text=f"Estimated Insurance Charges: ₹ {prediction[0]:,.2f}"
)


if __name__ == "__main__":
    app.run(debug=True)
