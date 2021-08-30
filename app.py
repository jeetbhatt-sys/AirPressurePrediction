import pickle

from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import Ridge,LassoCV,LinearRegression
import requests
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen as uReq

app = Flask(__name__)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    df = pd.read_csv("LinearRegressioncsv.csv").head(10)
    result = df.to_html()
    return render_template("index.html",table= result)

@app.route('/model',methods=['POST','GET']) # route to show the review comments in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #'Process_temperature','Rotational_speed', 'Torque', 'Tool_wear', 'Machine_failure'
            Process_temperature = request.form['Process_temperature']
            Rotational_speed = request.form['Rotational_speed']
            Torque = request.form['Torque']
            Tool_wear = request.form['Tool_wear']
            Machine_failure = request.form['Machine_failure']
            df = pd.read_csv("LinearRegressioncsv.csv")
            dataset_features = df[['Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear', 'Machine_failure']]
            scalar = StandardScaler()
            arr = scalar.fit_transform(dataset_features)
            model = pickle.load(open('linear_reg_lasso.sav','rb'))
            test_arr = scalar.transform([[Process_temperature,Rotational_speed,Torque,Tool_wear,Machine_failure]])
            res = model.predict(test_arr)

            return render_template('results.html', reviews=res)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')

@app.route('/report',methods=['POST','GET']) # route to show the review comments in a web UI
def report():
    if request.method == 'POST':
        try:
            return render_template('report.html')
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True)
