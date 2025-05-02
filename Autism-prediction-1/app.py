from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize LabelEncoders (match notebook's unique values)
le_gender = LabelEncoder().fit(['f', 'm'])
le_ethnicity = LabelEncoder().fit(['White-European', 'Latino', 'Others', 'Black', 'Asian', 'Middle Eastern', 
                                   'Pasifika', 'South Asian', 'Hispanic', 'Turkish'])
le_jaundice = LabelEncoder().fit(['no', 'yes'])
le_austim = LabelEncoder().fit(['no', 'yes'])
le_country = LabelEncoder().fit(['United States', 'Brazil', 'Spain', 'New Zealand', 'Sri Lanka', 'United Kingdom', 
                                 'Italy', 'Iceland', 'Canada', 'Australia', 'Austria', 'Vietnam', 'Jordan', 
                                 'Kazakhstan', 'Ireland', 'Malaysia', 'India', 'Afghanistan', 'Iran', 'France', 
                                 'Aruba', 'Ukraine', 'Pakistan', 'Bangladesh', 'China', 'Mexico', 'Netherlands', 
                                 'Turkey', 'South Africa', 'Romania', 'Russia', 'Azerbaijan', 'Ethiopia', 'Belgium', 
                                 'Germany', 'Sweden', 'Bahamas', 'Bolivia', 'Angola', 'Saudi Arabia', 
                                 'United Arab Emirates', 'Egypt', 'Oman', 'Iraq', 'Niger', 'Cyprus'])
le_used_app = LabelEncoder().fit(['no', 'yes'])
le_relation = LabelEncoder().fit(['Self', 'Others'])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Collect form data
        data = {
            'A1_Score': int(request.form['A1_Score']),
            'A2_Score': int(request.form['A2_Score']),
            'A3_Score': int(request.form['A3_Score']),
            'A4_Score': int(request.form['A4_Score']),
            'A5_Score': int(request.form['A5_Score']),
            'A6_Score': int(request.form['A6_Score']),
            'A7_Score': int(request.form['A7_Score']),
            'A8_Score': int(request.form['A8_Score']),
            'A9_Score': int(request.form['A9_Score']),
            'A10_Score': int(request.form['A10_Score']),
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'ethnicity': request.form['ethnicity'],
            'jaundice': request.form['jaundice'],
            'austim': request.form['austim'],
            'contry_of_res': request.form['contry_of_res'],
            'used_app_before': request.form['used_app_before'],
            'relation': request.form['relation']
        }

        # Calculate result as sum of A1–A10 scores
        data['result'] = sum([data[f'A{i}_Score'] for i in range(1, 11)])

        # Encode categorical variables (19 features)
        encoded_data = [
            data['A1_Score'], data['A2_Score'], data['A3_Score'], data['A4_Score'], 
            data['A5_Score'], data['A6_Score'], data['A7_Score'], data['A8_Score'], 
            data['A9_Score'], data['A10_Score'], data['age'],
            le_gender.transform([data['gender']])[0],
            le_ethnicity.transform([data['ethnicity']])[0],
            le_jaundice.transform([data['jaundice']])[0],
            le_austim.transform([data['austim']])[0],
            le_country.transform([data['contry_of_res']])[0],
            le_used_app.transform([data['used_app_before']])[0],
            data['result'],
            le_relation.transform([data['relation']])[0]
        ]

        # Convert to numpy array and reshape for prediction
        input_data = np.array(encoded_data).reshape(1, -1)

        # Make prediction
        pred = model.predict(input_data)[0]
        prediction = 'Autism Detected' if pred == 1 else 'No Autism Detected'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
