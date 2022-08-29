from flask import Flask, redirect, url_for, render_template, request 
import numpy as np
import pickle
import joblib
import os

app = Flask(__name__)

filename = '05 - Final_RandomForest.pickle'

model = pickle.load(open(filename, 'rb'))    # load the model

@app.route('/')

def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])  # The user input is processed here

def predict():
    Lyrics = request.form['lyrics']

    #Conver all of text to lowercase
    Lyrics_Lower = Lyrics.lower()
    
                               
    #Use model to predict cancer type
    pred = model.predict([Lyrics_Lower])
    
    #predict returns an array of 1 prediction.
    #index to grab the value of the prediction
    pred = pred[0]
    
    #Convert prediction to string
    pred_label_dict = {0:'Country', 1:'International', 2:'Pop', 3:'R&B', 4:'Rap/Hip-Hop', 5:'Rock'}
    
    #save output to variable
    output = pred_label_dict[pred]
    
    return render_template('index.html', predict=str(output))




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run( host='0.0.0.0', port=port)
