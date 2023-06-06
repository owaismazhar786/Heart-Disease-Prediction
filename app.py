from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('Heart_Prediction model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = float(request.form.get('trestbps'))
    chol = float(request.form.get('chol'))
    fbs = int(request.form.get('fbs'))
    restecg = int(request.form.get('restecg'))
    thalach = float(request.form.get('thalach'))
    exang = int(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slope = int(request.form.get('slope'))
    ca = int(request.form.get('ca'))
    thal = int(request.form.get('thal'))

    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Make predictions using the pre-trained model
    prediction = model.predict(input_data)

    # Display the predicted income category
    if prediction[0] == 0:
        patient_category = 'Heart Disease'
    else:
        patient_category = 'NO Heart Disease'

    return render_template('result.html', prediction_text='Predicted Patient category: {}'.format(patient_category))

if __name__ == '__main__':
    app.run(debug=True)
