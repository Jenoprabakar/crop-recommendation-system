from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputs = [float(request.form.get(x)) for x in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        prediction = model.predict([inputs])[0]
        return render_template("index.html", result=prediction)
    except:
        return render_template("index.html", result="Error in input")

if __name__ == "__main__":
    app.run(debug=True)
