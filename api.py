from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load("linear.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame(data, index=[0])
        prediction = model.predict(input_data)
        return jsonify({
            "prediction":round(prediction[0])
        })
        
    except:
        print("API Failed") 


if __name__=="__main__":
    app.run(
        debug=True,
    )