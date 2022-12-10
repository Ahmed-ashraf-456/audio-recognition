from voice_recognition import app
from flask import request
import os
import numpy as np
from scipy.io import wavfile
import pickle
from sklearn import preprocessing
import librosa
import librosa.display
import python_speech_features as mfcc
import os
import uuid
from flask import Flask, flash, redirect, render_template,request,session,url_for

from flask import Flask, flash, request, redirect


def calculate_delta(array):
	
    rows,cols = array.shape
    print(rows)
    print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
             first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas


def extract_features(file_path):
    audio , sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc_feature = mfcc.mfcc(audio,sample_rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def comparing(file_path):
    test = extract_features(file_path)
    call_Ahmed_model=pickle.load(open('code/voice_recognition/models/Ahmed.gmm','rb'))
    call_yahia_model=pickle.load(open('code/voice_recognition/models/mostafa_gmm','rb'))
    call_mostafa_model=pickle.load(open('code/voice_recognition/models/yahia.gmm','rb'))
    call_magdy_model=pickle.load(open('code/voice_recognition/models/magdy.gmm','rb'))
    call_mahmoud_model=pickle.load(open('code/voice_recognition/models/mahmoud_gmm.gmm','rb'))
    call_mayar_model=pickle.load(open('code/voice_recognition/models/mayar.gmm','rb'))
    call_mina_model=pickle.load(open('code/voice_recognition/models/mina.gmm','rb'))

    scores_1 = np.array(call_Ahmed_model.score(test))
    scores_2 = np.array(call_yahia_model.score(test))
    scores_3 = np.array(call_mostafa_model.score(test))
    scores_4 = np.array(call_magdy_model.score(test))
    scores_5 = np.array(call_mahmoud_model.score(test))
    scores_6 = np.array(call_mayar_model.score(test))
    scores_7 = np.array(call_mina_model.score(test))
    return scores_1,scores_2,scores_3,scores_4,scores_5,scores_6,scores_7


@app.route('/saveRecord',methods =['POST'])
def save_record():
    if request.method =='POST':
        file=request.files['AudioFile']
        if 'id' not in session:
            session['id'] = 0
        id = session['id']
        session['id'] += 1
        file_path='code/voice_recognition/static/assets/recordedAudio.wav'
        file.save(os.path.join(file_path))
        scores_1,scores_2,scores_3,scores_4,scores_5,scores_6,scores_7=comparing(file_path)
        print(scores_1)
        print(scores_2)
        print(scores_3)
        print(scores_4)
        print(scores_5)
        print(scores_6)
        print(scores_7)

        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)
        # file = request.files['file']
        # # if user does not select file, browser also
        # # submit an empty part without filename
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)
        # file_name =  "record.wav"
        # full_file_name = os.path.join("static/voice_recognition/static/assets", file_name)
        # print(full_file_name)
        # file.save(full_file_name)

    return []

        # if len(audio.shape)>1:
        #     audio=audio[:,0]
