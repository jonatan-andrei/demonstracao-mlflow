# flask_app.py
from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_name = "XGBoost"

model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        data = request.get_json()
        instances = data.get("instances", [])
        input_df = pd.DataFrame(instances)

        prediction = model.predict(input_df)
        return jsonify({"predictions": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)