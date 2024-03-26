import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 
app=Flask(__name__)
model=pickle.load(open('classifymodel.pkl','rb'))
sc=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home1.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    #data_list=list(data.values())
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=sc.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    a='\U0001F603'

    return jsonify(int(output[0]))
    """if(output==0):
        return jsonify("Tomorrow will be a SUNNY Day"+a)
    else:
        return jsonify("Tomorrow will be a RAINY day"+a)"""

@app.route('/prediction',methods=['POST'])
def prediction():
    data=[float(x)for x in request.form.values()]
    final_input=sc.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    if(output==0):
       return render_template("home1.html",prediction_text="Tomorrow will be a SUNNYDAY")
    else:
        return render_template("home1.html",prediction_text="Tomorrow will be a RAINY DAY")
    

    
if __name__=="__main__":
    app.run(debug=True)