from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the Logistic Regression model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic_model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Assessment Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        body {
            background: radial-gradient(circle at top left, #0f172a, #1e1b4b, #020617);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 32px 16px;
        }
        .container {
            background: rgba(30, 41, 59, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            width: 100%;
            max-width: 900px;
            padding: 40px 36px;
        }
        .header {
            text-align: center;
            margin-bottom: 32px;
        }
        .header h1 {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .header p {
            color: #94a3b8;
            font-size: 1rem;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 16px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(1, 1fr);
            gap: 18px;
        }
        @media (min-width: 640px) {
            .form-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (min-width: 860px) {
            .form-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        input, select {
            width: 100%;
            padding: 11px 13px;
            background-color: rgba(15, 23, 42, 0.85);
            border: 1px solid #334155;
            border-radius: 10px;
            color: #ffffff;
            font-size: 0.92rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus, select:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }
        .btn-submit {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #4f46e5 0%, #0284c7 100%);
            border: none;
            border-radius: 10px;
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            margin-top: 24px;
            transition: transform 0.15s ease, opacity 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
        }
        .btn-submit:hover {
            opacity: 0.95;
            box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.55);
        }
        .btn-submit:active {
            transform: scale(0.98);
        }
        .result-box {
            margin-top: 32px;
            padding: 24px;
            border-radius: 14px;
            text-align: center;
            background: rgba(15, 23, 42, 0.9);
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        }
        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #a5b4fc;
            margin-bottom: 6px;
        }
        .result-score {
            font-size: 2.2rem;
            font-weight: 800;
        }
        .status-low {
            color: #34d399;
            border: 1px solid #059669;
        }
        .status-moderate {
            color: #fbbf24;
            border: 1px solid #d97706;
        }
        .status-high {
            color: #f87171;
            border: 1px solid #dc2626;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Health Risk Assessment Predictor</h1>
            <p>Fill out the demographic, lifestyle, and clinical health metrics below</p>
        </div>
        
        <form action="/predict" method="POST">
            <div class="section-title">Demographic & General Info</div>
            <div class="form-grid" style="margin-bottom: 24px;">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" step="1" min="1" max="120" placeholder="e.g. 45" required>
                </div>
                <div class="form-group">
                    <label for="gender">Gender (Encoded)</label>
                    <input type="number" id="gender" name="gender" step="any" placeholder="e.g. 0 or 1" required>
                </div>
                <div class="form-group">
                    <label for="city">City (Encoded)</label>
                    <input type="number" id="city" name="city" step="any" placeholder="e.g. 1" required>
                </div>
                <div class="form-group">
                    <label for="income_bracket">Income Bracket (Encoded)</label>
                    <input type="number" id="income_bracket" name="income_bracket" step="any" placeholder="e.g. 2" required>
                </div>
            </div>

            <div class="section-title">Lifestyle & Daily Habits</div>
            <div class="form-grid" style="margin-bottom: 24px;">
                <div class="form-group">
                    <label for="physical_activity_level">Physical Activity Level</label>
                    <input type="number" id="physical_activity_level" name="physical_activity_level" step="any" placeholder="e.g. 1, 2, or 3" required>
                </div>
                <div class="form-group">
                    <label for="diet_type">Diet Type (Encoded)</label>
                    <input type="number" id="diet_type" name="diet_type" step="any" placeholder="e.g. 0" required>
                </div>
                <div class="form-group">
                    <label for="smoking_status">Smoking Status (Encoded)</label>
                    <input type="number" id="smoking_status" name="smoking_status" step="any" placeholder="e.g. 0" required>
                </div>
                <div class="form-group">
                    <label for="alcohol_consumption">Alcohol Consumption</label>
                    <input type="number" id="alcohol_consumption" name="alcohol_consumption" step="any" placeholder="e.g. 0" required>
                </div>
                <div class="form-group">
                    <label for="hours_sleep_per_night">Sleep Hours / Night</label>
                    <input type="number" id="hours_sleep_per_night" name="hours_sleep_per_night" step="0.1" min="0" max="24" placeholder="e.g. 7.5" required>
                </div>
                <div class="form-group">
                    <label for="stress_level">Stress Level</label>
                    <input type="number" id="stress_level" name="stress_level" step="any" placeholder="e.g. 3" required>
                </div>
            </div>

            <div class="section-title">Clinical & Medical Measurements</div>
            <div class="form-grid">
                <div class="form-group">
                    <label for="bmi">BMI</label>
                    <input type="number" id="bmi" name="bmi" step="0.1" placeholder="e.g. 24.5" required>
                </div>
                <div class="form-group">
                    <label for="family_history_diabetes">Family Diabetes History</label>
                    <select id="family_history_diabetes" name="family_history_diabetes" required>
                        <option value="1">Yes (1)</option>
                        <option value="0">No (0)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="fasting_blood_sugar">Fasting Blood Sugar</label>
                    <input type="number" id="fasting_blood_sugar" name="fasting_blood_sugar" step="0.1" placeholder="e.g. 95.0" required>
                </div>
                <div class="form-group">
                    <label for="hba1c_level">HbA1c Level</label>
                    <input type="number" id="hba1c_level" name="hba1c_level" step="0.1" placeholder="e.g. 5.6" required>
                </div>
                <div class="form-group">
                    <label for="blood_pressure_systolic">Systolic BP</label>
                    <input type="number" id="blood_pressure_systolic" name="blood_pressure_systolic" step="0.1" placeholder="e.g. 120" required>
                </div>
                <div class="form-group">
                    <label for="blood_pressure_diastolic">Diastolic BP</label>
                    <input type="number" id="blood_pressure_diastolic" name="blood_pressure_diastolic" step="0.1" placeholder="e.g. 80" required>
                </div>
                <div class="form-group">
                    <label for="waist_circumference_cm">Waist Circumference (cm)</label>
                    <input type="number" id="waist_circumference_cm" name="waist_circumference_cm" step="0.1" placeholder="e.g. 82.0" required>
                </div>
            </div>
            
            <button type="submit" class="btn-submit">Predict Assessment Level</button>
        </form>

        {% if prediction_text %}
            <div class="result-box status-{{ prediction_text|lower }}">
                <div class="result-title">Predicted Health Risk Level</div>
                <div class="result-score">{{ prediction_text }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Match feature order:
        # ['age', 'gender', 'city', 'bmi', 'family_history_diabetes', 'physical_activity_level',
        #  'diet_type', 'smoking_status', 'alcohol_consumption', 'hours_sleep_per_night',
        #  'stress_level', 'fasting_blood_sugar', 'hba1c_level', 'blood_pressure_systolic',
        #  'blood_pressure_diastolic', 'waist_circumference_cm', 'income_bracket']
        
        input_data = pd.DataFrame([{
            'age': float(request.form['age']),
            'gender': float(request.form['gender']),
            'city': float(request.form['city']),
            'bmi': float(request.form['bmi']),
            'family_history_diabetes': float(request.form['family_history_diabetes']),
            'physical_activity_level': float(request.form['physical_activity_level']),
            'diet_type': float(request.form['diet_type']),
            'smoking_status': float(request.form['smoking_status']),
            'alcohol_consumption': float(request.form['alcohol_consumption']),
            'hours_sleep_per_night': float(request.form['hours_sleep_per_night']),
            'stress_level': float(request.form['stress_level']),
            'fasting_blood_sugar': float(request.form['fasting_blood_sugar']),
            'hba1c_level': float(request.form['hba1c_level']),
            'blood_pressure_systolic': float(request.form['blood_pressure_systolic']),
            'blood_pressure_diastolic': float(request.form['blood_pressure_diastolic']),
            'waist_circumference_cm': float(request.form['waist_circumference_cm']),
            'income_bracket': float(request.form['income_bracket'])
        }])
        
        prediction = model.predict(input_data)[0]
        return render_template_string(HTML_TEMPLATE, prediction_text=str(prediction))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
