from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates")
classifier = pickle.load(open("./models/SVC_Classifier.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
@cross_origin()
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        if my_prediction<0:
            return render_template('index.html',prediction_texts="Great you are safe . You dont have  diabetse!!")
        else:
            return render_template('index.html',prediction_text=" Unsafe. You have got the diabetese !! ")
    else:
        return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)

