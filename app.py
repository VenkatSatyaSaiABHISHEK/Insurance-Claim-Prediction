from flask import Flask, request, render_template, jsonify
import joblib
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load First Model (insurance_claim_model.pkl)
try:
    model1 = joblib.load("model/insurance_claim_model.pkl")
    scaler1 = joblib.load("model/scaler.pkl")
    print("✅ Model 1 (Insurance Claim) loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Model 1: {e}")

# Load Second Model (hel_claim_model.pkl)
try:
    with open("model/hel_claim_model.pkl", "rb") as file:
        model2 = pickle.load(file)
    print("✅ Model 2 (Health Claim) loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Model 2: {e}")

# Dictionary to map regions to numeric values (for Model 1)
region_map = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

# Define feature names for Model 2
numerical_features = ["Age", "BMI", "No. of Hospital Visits", "Claim Amount Requested (₹)"]
categorical_features = ["Gender", "Smoke", "Health Problem"]

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/house/predict', methods=['GET', 'POST'])
def predict_house():
    if request.method == 'GET':
        return render_template("hou.html")  # Show the form when accessed via GET

    try:
        print("Received Form Data:", request.form)

        # Fetch data safely
        property_age = request.form.get("property_age", "").strip()
        location_risk = request.form.get("location_risk", "").strip()
        disaster_history = request.form.get("disaster_history", "").strip()
        construction_type = request.form.get("construction_type", "").strip()
        property_value = request.form.get("property_value", "").strip()
        owner_age = request.form.get("owner_age", "").strip()
        insurance_type = request.form.get("insurance_type", "").strip()
        past_claims = request.form.get("past_claims", "").strip()

        # Validate input
        if not all([property_age, location_risk, disaster_history, construction_type, property_value, owner_age, insurance_type, past_claims]):
            return render_template("hou.html", result="Error: All fields are required!")

        # Convert values
        property_age = int(property_age)
        location_risk = int(location_risk)
        disaster_history = int(disaster_history)
        construction_type = int(construction_type)
        property_value = float(property_value)
        owner_age = int(owner_age)
        insurance_type = int(insurance_type)
        past_claims = int(past_claims)

        # **Manual If-Else Conditions**
        if past_claims > 3:
            result = "Claim Rejected ❌ (Too many past claims)"
        elif location_risk == 2:  # High risk
            result = "Claim Rejected ❌ (High-risk location)"
        elif disaster_history == 1 and property_value > 50:
            result = "Claim Rejected ❌ (High-value property in disaster area)"
        elif owner_age < 25:
            result = "Claim Rejected ❌ (Owner too young)"
        elif property_age > 50:
            result = "Claim Rejected ❌ (Property too old)"
        else:
            result = "Claim Approved ✅"

        return render_template("hou.html", result=result)

    except ValueError:
        return render_template("hou.html", result="Error: Invalid input! Please enter valid numbers.")
    except Exception as e:
        return render_template("hou.html", result=f"Error: {str(e)}")

@app.route('/home')
def home():
    return render_template('landing.html')  # Main index page for Model 1

@app.route('/health')
def health():
    return render_template('life.html')  # Page for Model 2

# ✅ Prediction for Model 1 (insurance_claim_model.pkl)
@app.route('/predict', methods=['POST'])
def predict_insurance():
    try:
        # Get form data
        age = float(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Convert categorical inputs to numerical
        gender = 1 if gender == 'Male' else 0
        smoker = 1 if smoker == 'Yes' else 0
        region = region_map.get(region, 0)  # Default to 0 if not found

        # Prepare input data
        input_data = np.array([[age, gender, bmi, children, smoker, region]])

        # Scale input
        input_data_scaled = scaler1.transform(input_data)

        # Make prediction
        prediction = model1.predict(input_data_scaled)

        # Convert result
        result = "✅ CLAIM APPROVED*" if prediction[0] == 1 else "❌ CLAIM REJECTED*\n\n" + "\n".join([
            "1️⃣ DUE TO MINOR AGE or YOUR HEALTH IS NICE TAKE 'LIFE' INSURANCE",
            "2️⃣ GIVEN WRONG INPUTS or .......",
        ])

        return render_template('result.html', result=result)

    except Exception as e:
        return f"Error: {e}"

# ✅ Prediction for Model 2 (hel_claim_model.pkl)
@app.route("/predict_health", methods=["POST"])
def predict_health():
    try:
        # Get input data from form
        data = {
            "Age": [int(request.form["age"])],
            "BMI": [float(request.form["bmi"])],
            "No. of Hospital Visits": [int(request.form["visits"])],
            "Claim Amount Requested (₹)": [float(request.form["claim_amount"])],
            "Gender": [request.form["gender"]],
            "Smoke": [request.form["smoke"]],
            "Health Problem": [request.form["health_problem"]]
        }

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Extract preprocessor and classifier properly
        preprocessor = model2.steps[0][1]  # First step: preprocessing
        classifier = model2.steps[1][1]  # Second step: classifier

        # Transform input data
        X_processed = preprocessor.transform(df)

        # Predict
        prediction = classifier.predict(X_processed)[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
