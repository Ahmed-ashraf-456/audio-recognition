from flask import Flask, redirect, url_for, request,render_template
import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from scipy.io.wavfile import read
import pickle
model= pickle.load(open('voiceWithOthers.sav','rb'))
import python_speech_features as mfcc
from sklearn import preprocessing

import librosa
import os
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize



app = Flask(__name__)

@app.route('/')
def index():
   if 'msg' in request.args:
      msg = request.args['msg']
   else:
      msg = -1
   print(msg)
   return render_template('index.html',msg = msg)




def extractWavFeatures(soundFile, csvFileName):
   #print("The features of the files in the folder "+soundFilesFolder+" will be saved to "+csvFileName)
   header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'  
   for i in range(1, 41):
      header += f' mfcc{i}'  #making headers of csv file
   header += ' label'     
   header = header.split()
   #print('CSV Header: ', header)
   file = open(csvFileName, 'w', newline='')
   #with file:
   writer = csv.writer(file)
   writer.writerow(header)

   y, sr = librosa.load(soundFile, mono=True, duration=30)
   # remove leading and trailing silence
   y, index = librosa.effects.trim(y)
   chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
   rmse = librosa.feature.rms(y=y)
   spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
   spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
   rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
   zcr = librosa.feature.zero_crossing_rate(y)
   mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 40)
   to_append = f'{soundFile} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
   for e in mfcc:
      to_append += f' {np.mean(e)}'
   writer.writerow(to_append.split())
   file.close()


#Reading a dataset and convert file name to corresbonding umnber
def preProcessData(csvFileName):
   data = pd.read_csv(csvFileName)
   data = data.drop(['filename'],axis=1)
   data = data.drop(['label'],axis=1)
   data = data.drop(['chroma_stft'],axis=1)
   # for i in range(1, 41):
   #      data = data.drop([f'mfcc{i}'],axis=1)
   #data = normalize( data )
   return data

   
def showScore():
   
   csvFileName = "Speaker_File1.csv"
   extractWavFeatures (r"testing_set/sample.wav" ,csvFileName )
   data = preProcessData(csvFileName)
   #print(data)
   mode=model.predict(data)
   print('DCT value')
   print (mode)
   return mode


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

def extract_features_GMM(file_path):
    audio , sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc_feature = mfcc.mfcc(audio,sample_rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined


def loadModelsGMM(filePath):
   testFeatures = extract_features_GMM(filePath)
   gmmModels = {
   'Michael': 'Michael.gmm',
   'Belal': 'Belal.gmm',
   'Ahmed_Ashraf':'Ahmed_Ashraf.gmm',
   'others':'others.gmm'
   }
   scores = {}
   mx = -100
   result = " "
   for name,modelName in gmmModels.items():
      gmmModels[name] = pickle.load(open(modelName,'rb'))
      scores[name] = gmmModels[name].score(testFeatures)
      if scores[name] > mx:
         mx = scores[name]
         result = name
   print(scores)
   return result

@app.route('/record',methods=['GET','POST'])
def record():
   FORMAT = pyaudio.paInt16
   CHANNELS = 1
   RATE = 44100
   CHUNK = 512
   RECORD_SECONDS = 2
   device_index = 2
   audio = pyaudio.PyAudio()
   print("----------------------record device list---------------------")
   info = audio.get_host_api_info_by_index(0)
   numdevices = info.get('deviceCount')
   for i in range(0, numdevices):
      if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
         print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
   print("-------------------------------------------------------------")
   # index = int(input())	
   index = 1
   print("recording via index "+str(index))
   stream = audio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,input_device_index = index,
            frames_per_buffer=CHUNK)
   print ("recording started")
   Recordframes = []
   for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      data = stream.read(CHUNK)
      Recordframes.append(data)
   print ("recording stopped")
   stream.stop_stream()
   stream.close()
   audio.terminate()
   OUTPUT_FILENAME="sample.wav"
   WAVE_OUTPUT_FILENAME=os.path.join("testing_set",OUTPUT_FILENAME)
   trainedfilelist = open("testing_set_addition.txt", 'a')
   trainedfilelist.write(OUTPUT_FILENAME+"\n")
   waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
   waveFile.setnchannels(CHANNELS)
   waveFile.setsampwidth(audio.get_sample_size(FORMAT))
   waveFile.setframerate(RATE)
   waveFile.writeframes(b''.join(Recordframes))
   waveFile.close()
   result = showScore()
   resultsGMM = loadModelsGMM("testing_set/sample.wav")
   return redirect(url_for('index', msg=resultsGMM))


if __name__ == '__main__':
   app.run(debug = True)
