from flask import Flask, request, render_template, jsonify
import pickle 
import numpy as np

app = Flask(__name__)

# Define the paths to your models
task1_model1_path = r"models\task1\logisitic_regression_model.pkl"
task1_model2_path = r"models\task1\decision_tree_model.pkl"
task1_model3_path = r"models\task1\random_forest_model.pkl"
task1_model4_path = r"models\task1\k-nearest_neighbors_model.pkl"

# Load models using pickle.load
model_task1_1 = pickle.load(open(task1_model1_path, 'rb'))
model_task1_2 = pickle.load(open(task1_model2_path, 'rb'))
model_task1_3 = pickle.load(open(task1_model3_path, 'rb'))
model_task1_4 = pickle.load(open(task1_model4_path, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from the request form
        age = float(request.form.get('age'))
        sex = float(request.form.get('sex'))  # Assuming 0 for male and 1 for female
        cp = float(request.form.get('cp'))
        trestbps = float(request.form.get('trestbps'))
        chol = float(request.form.get('chol'))
        fbs = float(request.form.get('fbs'))
        restecg = float(request.form.get('restecg'))
        thalach = float(request.form.get('thalach'))
        exang = float(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = float(request.form.get('slope'))
        ca = float(request.form.get('ca'))
        thal = float(request.form.get('thal'))

        # Make prediction using the loaded models
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])
        test_1 = model_task1_1.predict(features)
        test_2 = model_task1_2.predict(features)
        test_3 = model_task1_3.predict(features)
        test_4 = model_task1_4.predict(features)

        return jsonify({'Test_1': test_1[0], 'Test_2': test_2[0], 'Test_3': test_3[0], 'Test_4': test_4[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/hypertension')
def hypertension_form():
    return render_template('hypertension_form.html')

if __name__ == '__main__':
    app.run(debug=True)






















'''

from flask import Flask, render_template
import pickle 
app = Flask(__name__)
# Load your machine learning model during application startup
# Define the paths to your models using raw string literals (r"...")
task1_model1_path = r"models\task1\logisitic_regression_model.pkl"
task1_model2_path = r"models\task1\decision_tree_model.pkl"
task1_model3_path = r"models\task1\random_forest_model.pkl"
task1_model4_path = r"models\task1\k-nearest_neighbors_model.pkl"

# Load models using pickle.load
model_task1_1 = pickle.load(open(task1_model1_path, 'rb'))
model_task1_2 = pickle.load(open(task1_model2_path, 'rb'))
model_task2_3 = pickle.load(open(task1_model3_path, 'rb'))
model_task2_4 = pickle.load(open(task1_model4_path, 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from the request form
        age = float(request.form.get('age'))
        sex = float(request.form.get('sex'))  # Assuming 0 for male and 1 for female
        cp = float(request.form.get('cp'))
        trestbps = float(request.form.get('trestbps'))
        chol = float(request.form.get('chol'))
        fbs = float(request.form.get('fbs'))
        restecg = float(request.form.get('restecg'))
        thalach = float(request.form.get('thalach'))
        exang = float(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))

        # Make prediction using the loaded model
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])
        Test_1 = model_task1_1.predict(features)
        Test_2 = model_task1_2.predict(features)
        Test_3 = model_task2_3.predict(features)
        Test_4 = model_task2_4.predict(features)

        return jsonify({'Test_1': Test_1[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    
@app.route('/hypertension')
def hypertension_form():
    return render_template('hypertension_form.html')

if __name__ == '__main__':
    app.run(debug=True)

'''
























































'''
from flask import Flask, render_template, request, jsonify
import pickle # or import pickle if you used pickle to save the model

app = Flask(__name__)

# Load your machine learning model during application startup
# Define the paths to your models using raw string literals (r"...")
task1_model1_path = r"models\task1\logisitic_regression_model.pkl"
task1_model2_path = r"models\task1\decision_tree_model.pkl"
task1_model3_path = r"models\task1\random_forest_model.pkl"
task1_model4_path = r"models\task1\k-nearest_neighbors_model.pkl"

# Load models using pickle.load
model_task1_1 = pickle.load(open(task1_model1_path, 'rb'))
model_task1_2 = pickle.load(open(task1_model2_path, 'rb'))
model_task2_3 = pickle.load(open(task1_model3_path, 'rb'))
model_task2_4 = pickle.load(open(task1_model4_path, 'rb'))


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from the request form
        bedrooms = float(request.form.get('bedrooms'))
        bathrooms = float(request.form.get('bathrooms'))
        toilets = float(request.form.get('toilets'))
        total_rooms = float(request.form.get('total_rooms'))

        # Make prediction using the loaded model
        features = np.array([[bedrooms, bathrooms, toilets, total_rooms]])
        prediction = model.predict(features)

        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

'''





'''
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stroke', methods=['POST'])
def predict_stroke():
    # Handle form data for stroke prediction
    age = int(request.form['age'])
    sex = request.form['sex']
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    # Add more form data processing as needed
    
    # Placeholder for ML prediction logic
    # This is where you would integrate your machine learning model
    # For now, just returning a mock response
    prediction = 0.7  # Placeholder value
    
    return jsonify({'prediction': prediction})

@app.route('/hypertension', methods=['POST'])
def predict_hypertension():
    # Handle form data for hypertension prediction
    age = int(request.form['age'])
    sex = request.form['sex']
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    # Add more form data processing as needed
    
    # Placeholder for ML prediction logic
    # This is where you would integrate your machine learning model
    # For now, just returning a mock response
    prediction = 0.8  # Placeholder value
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)








'''


















'''

from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
# Load models for task 1
model_task1_1 = pickle.load(open(r"logisitic_regression_model.pkl", 'rb'))
model_task1_2 = pickle.load(open(r"logisitic_regression_model.pkl", 'rb'))
model_task1_3 = pickle.load(open(r"logisitic_regression_model.pkl", 'rb'))
model_task1_4 = pickle.load(open(r"logisitic_regression_model.pkl", 'rb'))



# Load models for task 2
model_task2_1 = pickle.load(open('models/task2/model1.pkl', 'rb'))
model_task2_2 = pickle.load(open('models/task2/model2.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_task1', methods=['POST'])  
def predict_task1():
    # Getnput data from the form or request
    # Example assumes input is in JSON format
    
    # Process data as needed (e.g., convert to DataFrame)

    # Make predictions using the loaded models
    prediction1 = model_task1_1.predict(data)
    prediction2 = model_task1_2.predict(data)

    # Return predictions as JSON response
    return {'prediction1': prediction1.tolist(), 'prediction2': prediction2.tolist()}

@app.route('/predict_task2', methods=['POST'])
#def predict_task2():
    # Similar to predict_task1, but use models for task 2


'''