from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('xgboost_model.pkl')  # Make sure the model file is in the same folder

# Initialize Flask app
app = Flask(__name__)

# Mappings for preprocessing
mappings = {
    'gender': {'male': 1, 'female': 0},
    'home_language': {'English': 0, 'Urdu': 2, 'Punjabi': 1},
    'Fath_occupation': {'Government job': 1, 'Self-employed': 4, 'Agriculture': 0,
                        'Public sector': 3, 'Teacher': 5, 'Private job': 2},
    'Fath_edu': {'Postsecondary': 1, 'Graduated': 0, 'Primary': 2, 'Secondary': 3},
    'Family_size': {'Three Children': 1, 'Four Children': 2, 'Five Children': 0, 'More than five': 4, 'Two Children': 5},
    'means_To_School': {'Walk': 3, 'Bicycle/motorbike': 0, 'Private car': 1, 'Public transport': 2},
    'Grade': {'A': 0, 'B': 2, 'C': 3, 'D': 4, 'D+': 5, 'A-': 1}
}

# Preprocess input data
def preprocess(raw_input):
    processed_data = pd.DataFrame([{
        'Gender': mappings['gender'][raw_input['gender']],
        'Home_Language': mappings['home_language'][raw_input['home_language']],
        'Father_occupation': mappings['Fath_occupation'][raw_input['Fath_occupation']],
        'Father_qualification': mappings['Fath_edu'][raw_input['Fath_edu']],
        'Family_size': mappings['Family_size'][raw_input['Family_size']],
        'School_distance_km': float(raw_input['school_distanceKm'].replace(' km', '')),
        'Age': raw_input['Stu_age'],
        'Education': mappings['Fath_edu'][raw_input['Education']],
        'Means_To_School': mappings['means_To_School'][raw_input['means_To_School']],
        'Grade': mappings['Grade'][raw_input['Grade']]
    }])
    return processed_data

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Receive JSON input
        processed_data = preprocess(data)  # Preprocess input
        prediction = model.predict(processed_data)  # Make prediction
        return jsonify({'prediction': int(prediction[0])})  # Send response
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
