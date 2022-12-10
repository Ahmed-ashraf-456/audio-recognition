from flask import Flask, render_template, send_file, request, redirect
from routes import *
import os
from wtforms.validators import InputRequired
import librosa



app = Flask(__name__)


@app.route('/', methods=['GET', "POST", "put"])
def index():
    dict_sliders = {}
    # if request.method == "POST":
    return render_template('index.html')



@app.route('/saveRecord',methods =['POST'])
def save_record():
    if request.method =='POST':
        print('-------------------------before request---------------------')
        file=request.files['AudioFile']
        print('-------------------------after request---------------------')
        file_path='code/voice_recognition/static/assets/recordedAudio.wav'
        print('-------------------------before save---------------------')
        file.save(os.path.join(file_path))
        print('-------------------------after save---------------------')
        scores_1,scores_2,scores_3,scores_4=comparing(file_path)
        print(scores_1)
        print(scores_2)
        print(scores_3)
        print(scores_4)
    return '<h1>sucess</h1>'

if __name__ == '__main__':
    app.run(debug=True)