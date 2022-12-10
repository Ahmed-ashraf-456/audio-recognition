import pickle
model= pickle.load(open('final_model_test.sav','rb'))
import librosa
import os
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize



csvFileName = "Speaker_File1.csv"


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
# def preProcessData(csvFileName):
#     header_name_list = ['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'mfcc21', 'mfcc22', 'mfcc23', 'mfcc24', 'mfcc25', 'mfcc26', 'mfcc27', 'mfcc28', 'mfcc29', 'mfcc30', 'mfcc31', 'mfcc32', 'mfcc33', 'mfcc34', 'mfcc35', 'mfcc36', 'mfcc37', 'mfcc38', 'mfcc39', 'mfcc40', 'label']
#     print(csvFileName+ " will be preprocessed")
#     data =  pd.read_csv(csvFileName )
#     print(data.shape)
#     #Dropping unnecessary columns
#     # data = data.drop(['filename'],axis=1)
#     # data = data.drop(['label'],axis=1)
#     data = data.drop(['chroma_stft'],axis=1)
#     print(data)
#     return data


#Reading a dataset and convert file name to corresbonding umnber
def preProcessData(csvFileName):
    data = pd.read_csv(csvFileName)
    data = data.drop(['filename'],axis=1)
    data = data.drop(['label'],axis=1)
    data = data.drop(['chroma_stft'],axis=1)
    #data = normalize( data )
    return data


if __name__ == "__main__":
    extractWavFeatures (r"testing_set/sample.wav" ,csvFileName )
    data = preProcessData(csvFileName)
    #print(data)
    mode=model.predict(data)
    print (mode)