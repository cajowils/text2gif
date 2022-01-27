import time
from flask import Flask,render_template,request
from bigText import *
from flask import url_for
import pandas as pd
import random
app = Flask(__name__)

print("Loading GloVe Model")
ts = time.time()
glove_model = pickle.load( open( "glove_model.p", "rb" ) )
model_time = time.time() - ts
print("Model loaded in {} seconds".format(model_time))

@app.route('/')
def form():
    return render_template('form.html')
@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        link_df, emotion_scores = big(form_data["Tweet"], glove_model)
        link0 = link_df.iloc[0]['link']
        link1 = link_df.iloc[1]['link']
        link2 = link_df.iloc[2]['link']
        link3 = link_df.iloc[3]['link']
        link4 = link_df.iloc[4]['link']
        link5 = link_df.iloc[5]['link']
        link6 = link_df.iloc[6]['link']
        link7 = link_df.iloc[7]['link']
        link8 = link_df.iloc[8]['link']
        link9 = link_df.iloc[9]['link']
        

        return render_template('data.html',form_data = form_data, link0 = link0, link1=link1, link2=link2, link3=link3, link4=link4, link5=link5, link6=link6, link7=link7, link8=link8, link9=link9, emotion_scores = emotion_scores)
@app.route('/About/')
def About():
    return render_template('About.html')
app.run(host='localhost', port=5000)