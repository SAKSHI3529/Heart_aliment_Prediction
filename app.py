import joblib
from flask import Flask, render_template, request, redirect
import pickle
from flask_pymongo import PyMongo
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeClassifier

from pyexpat import model

load_dotenv()
password_pred = os.getenv("passwor_pred")

# model = pickle.load(open('heartweb.pkl', 'rb'))
# Load the trained model and scaler
model_filename = 'heart-disease-prediction-rf-model.pkl'
scaler_filename = 'scaler.pkl'

model = pickle.load(open(model_filename, 'rb'))
scaler = pickle.load(open(scaler_filename, 'rb'))
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about_heart")
def about_heart():
    return render_template("about_heart.html")

@app.route('/prediction')
def man():
    return render_template('prediction.html')

@app.route('/contact')
def mancon():
    return render_template('contact.html')


@app.route("/prediction", methods=['POST'] , endpoint='prediction')
def prediction():
    if request.method == 'POST':
        # Get form values and convert to appropriate data types
        name = request.form['name']# Keep name as a string
        age = int(request.form['age'])
        sex = int(request.form['sex'])  # Male = 1, Female = 0
        cp = int(request.form['cp'])  # Chest Pain Type (0-3)
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])  # 1 for Fasting blood sugar > 120, 0 otherwise
        restecg = int(request.form['restecg'])  # Resting ECG (0-2)
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])  # Exercise-induced Angina (1=Yes, 0=No)
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])  # Slope of peak exercise ST segment (0-2)
        ca = int(request.form['ca'])  # Number of major vessels (0-4)
        thal = int(request.form['thal'])  # Thalassemia (0-2)

        # Prepare the data for prediction
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Scale the data using the loaded scaler
        data_scaled = scaler.transform(data)

        # Make the prediction
        prediction = model.predict(data_scaled)

        print(prediction)
        # Interpret the prediction result
        # if prediction[0] == 1:
        #     result = "Heart Disease"
        # else:
        #     result = "No Heart Disease"
        mongo = PyMongo()
        app.config["MONGO_URI"] = "mongodb://localhost:27017/Heart-ailment"
        mongo.init_app(app)
        user = {
            "name": name,
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,

        }
        mongo.db.users.insert_one(user)

        # Render the result page with the prediction
        return render_template('after.html', prediction=prediction)

@app.route("/prevention")
def prevention():
     return render_template("prevention.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=['POST'])
def contact():
    # Retrieve data from the form
    contdata1 = request.form['nam']
    contdata2 = request.form['ema']
    contdata3 = request.form['mes']
    contdata4 = request.form.get('sub', '')  # Subject is optional, provide a default empty string

    # Configure MongoDB connection
    app.config["MONGO_URI"] = "mongodb://localhost:27017/Heart-ailment"
    mongo = PyMongo(app)

    # Insert the contact data into the 'contacts' collection
    contact = {
        "name": contdata1,
        "email": contdata2,
        "subject": contdata4,
        "message": contdata3
    }
    mongo.db.contacts.insert_one(contact)

    return render_template("contact.html", success=True)
if __name__ == "__main__":
    app.run(debug=True, port=2000)