from flask import request
import numpy as np
import pickle
from sklearn import preprocessing
import librosa
import librosa.display
import python_speech_features as mfcc

def calculate_delta(array):
    rows,cols = array.shape
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
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def comparing(file_path):
    test = extract_features(file_path)
    call_mostafa_model=pickle.load(open('Voice_Fingerprint/code/voice_recognition/models/mostafa.gmm','rb'))
    call_magdy_model=pickle.load(open('Voice_Fingerprint/code/voice_recognition/models/magdy.gmm','rb'))
    call_mayar_model=pickle.load(open('Voice_Fingerprint/code/voice_recognition/models/mayar.gmm','rb'))
    call_mina_model=pickle.load(open('Voice_Fingerprint/code/voice_recognition/models/mina.gmm','rb'))

    scores_1 = np.array(call_mostafa_model.score(test))
    scores_2 = np.array(call_magdy_model.score(test))
    scores_3 = np.array(call_mayar_model.score(test))
    scores_4 = np.array(call_mina_model.score(test))
    return scores_1,scores_2,scores_3,scores_4



