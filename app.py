from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        population = float(request.form['population'])
        capacity = float(request.form['capacity'])
        
        # Make prediction using POPULATION and CAPACITY
        features = np.array([[population, capacity]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Calculate additional metrics
        available_slots = capacity - population
        utilization = (population / capacity) * 100 if capacity > 0 else 0
        
        # Determine result
        if prediction == 1:
            result = "OVERcrowded"
            result_class = "overcrowded"
            if probability > 0.7:
                risk = "HIGH"
            elif probability > 0.4:
                risk = "MEDIUM"
            else:
                risk = "LOW"
        else:
            result = "NOT overcrowded"
            result_class = "safe"
            if probability > 0.7:
                risk = "HIGH"
            elif probability > 0.4:
                risk = "MEDIUM"
            else:
                risk = "LOW"
        
        return render_template('index.html', 
                               prediction=result,
                               probability=f"{probability*100:.1f}",
                               risk=risk,
                               population=int(population),
                               capacity=int(capacity),
                               available_slots=int(available_slots),
                               utilization=f"{utilization:.1f}",
                               result_class=result_class)
    
    except Exception as e:
        return render_template('index.html', error="Please enter valid numbers")

if __name__ == '__main__':
    app.run(debug=True)