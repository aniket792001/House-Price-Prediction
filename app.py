from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Load pipeline (preprocessor + model)
with open("best_random_forest.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        data = {
            "area": [float(request.form["area"])],
            "bedrooms": [int(request.form["bedrooms"])],
            "bathrooms": [int(request.form["bathrooms"])],
            "stories": [int(request.form["stories"])],
            "parking": [int(request.form["parking"])],
            "mainroad": [request.form["mainroad"]],
            "guestroom": [request.form["guestroom"]],
            "basement": [request.form["basement"]],
            "hotwaterheating": [request.form["hotwaterheating"]],
            "airconditioning": [request.form["airconditioning"]],
            "prefarea": [request.form["prefarea"]],
            "furnishingstatus": [request.form["furnishingstatus"]]
        }

        input_df = pd.DataFrame(data)

        # Make prediction
        prediction = pipeline.predict(input_df)[0]

        return render_template("index.html", prediction_text=f"🏠 Estimated Price: ₹ {prediction:,.0f}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
