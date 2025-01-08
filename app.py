from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Load the pre-trained models
MODEL1_PATH = "yield_strength_regressor.pkl"
MODEL2_PATH = "tensile_strength_regressor.pkl"
MODEL3_PATH = "elongation_classifier.pkl"  # New model path

model1 = joblib.load(MODEL1_PATH)
model2 = joblib.load(MODEL2_PATH)
model3 = joblib.load(MODEL3_PATH)  # Load the third model

# Define the feature names
FEATURE_NAMES = [
    "fe", "c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"
]

app = Flask(__name__)

# Set CORS to allow requests only from your Vercel frontend URL
frontend_url = os.environ.get("FRONTEND_URL", "https://cc-landing-plum.vercel.app")  # Use environment variable or hardcode
CORS(app, origins=[frontend_url])

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        # This is the pre-flight CORS request, respond with 200 OK
        return jsonify({"message": "CORS pre-flight successful"}), 200

    try:
        # Parse input data
        data = request.json.get("values")

        # Validate input
        if not data or len(data) != len(FEATURE_NAMES):
            return jsonify({"error": f"Invalid input. Please provide {len(FEATURE_NAMES)} values."}), 400

        # Convert input to DataFrame with proper feature names
        input_df = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Make predictions
        prediction_model1 = model1.predict(input_df)[0]
        prediction_model2 = model2.predict(input_df)[0]
        prediction_model3 = model3.predict(input_df)[0]  # Predict with the third model

        # Return predictions as JSON
        return jsonify({
            "prediction_model1": prediction_model1,
            "prediction_model2": prediction_model2,
            "prediction_model3": prediction_model3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Flask will bind to port 5000 in development, or use the PORT environment variable in production (Heroku)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
