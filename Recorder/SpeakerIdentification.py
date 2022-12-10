import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from scipy.io.wavfile import read

warnings.filterwarnings("ignore")

def record_audio_train():
	Name =(input("Please Enter Your Name:"))
	SAMPLES_NUMBER = 10
	for count in range(10):
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
		index = int(input())	
		# index = 1	
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
		OUTPUT_FILENAME=Name+"-sample"+str(count)+".wav"
		WAVE_OUTPUT_FILENAME=os.path.join("training_set",OUTPUT_FILENAME)
		print(WAVE_OUTPUT_FILENAME)
		waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		waveFile.setnchannels(CHANNELS)
		waveFile.setsampwidth(audio.get_sample_size(FORMAT))
		waveFile.setframerate(RATE)
		waveFile.writeframes(b''.join(Recordframes))
		waveFile.close()

def record_audio_test():
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


while True:
	choice=int(input("\n 1.Record audio for training \n 2.Record audio for testing \n "))
	if(choice==1):
		record_audio_train()
	elif(choice==2):
		record_audio_test()
	if(choice>2):
		exit()
