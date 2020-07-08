import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
    

@app.route('/predict',methods=['POST'])
def predict():

    AveragAmount_transaction_day = float(request.form['AverageAmount_transaction_day'])
    Transaction_Amount = float(request.form['Transaction_amount'])
    IsDeclined = int(request.form['IsDeclined'])
    TotalNumberOfDeclines_day = int(request.form['TotalNumberOfDeclines_day'])
    isForeignTransaction = int(request.form['isForeignTransaction'])
    isHighRiskCountry = int(request.form['isHighRiskCountry'])
    Daily_chargeback_avg_amt = int(request.form['Daily_chargeback_avg_amt'])
    six_month_avg_chbk_amt = float(request.form['6_month_avg_chbk_amt'])
    six_month_chbk_freq = int(request.form['6_month_chbk_freq'])
    
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

    int_features = [np.array([AveragAmount_transaction_day, Transaction_Amount, IsDeclined, TotalNumberOfDeclines_day, isForeignTransaction, isHighRiskCountry, Daily_chargeback_avg_amt, six_month_avg_chbk_amt, six_month_chbk_freq])]
    final_features =pd.DataFrame(int_features)
    fea=final_features.values
    
    output = model.predict(fea)
    
    if output==1:
        result='Is'
    elif output==0:
        result='Is Not'

    return render_template('index.html', prediction_text='Transaction {} fradulent'.format(result))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)