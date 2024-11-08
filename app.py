from flask import Flask,request,render_template
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_integer=[float(x) for x in request.form.values()]
    final_integer=[np.array(int_integer)]
    prediction=model.predict(final_integer)
    print(prediction[0])

    return render_template('home.html',prediction_text="the flight price is {}".format(prediction[0]))

if __name__=='__main__':
    app.run(debug=True)
    