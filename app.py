import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model_knn = pickle.load(open("model_knn.pkl", "rb"))
model_dt = pickle.load(open("model_dt.pkl", "rb"))
model_rfc = pickle.load(open("model_rfc.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction_knn = model_knn.predict(features)
    prediction_dt = model_dt.predict(features)
    prediction_rfc = model_rfc.predict(features)
    return render_template("index.html", 
                           prediction_text_knn = "KNN: The localization_site is {}".format(prediction_knn),
                           prediction_text_dt = "Decision tree: The localization_site is {}".format(prediction_dt),
                           prediction_text_rfc = "Random forest: The localization_site is {}".format(prediction_rfc))

if __name__ == "__main__":
    flask_app.run(debug=True)