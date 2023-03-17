#import modules 

import pyaudio
import numpy
import wave 
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import os,sys 
import speech_recognition as sr
import pandas as pd

#define constants 
RATE=44100
RECORD_SECONDS = 4
CHUNKSIZE = 1024
CHANNELS=1
FORMAT = pyaudio.paInt16

def record_wav():
    enterName = input("Enter name:  ") 
    WAVE_OUTPUT_FILENAME = enterName+"_noiselevel_master.wav"
    p = pyaudio.PyAudio()
    print("Recording. Speak Now...")
    stream = p.open(format=FORMAT, 
                channels=CHANNELS, 
                rate=RATE, 
                input=True, 
                frames_per_buffer=CHUNKSIZE)
    
    frames = [] # A python-list of chunks(numpy.ndarray)
    for i in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
        data = stream.read(CHUNKSIZE)
        frames.append(numpy.frombuffer(data, dtype=numpy.int16))
        
    audioVector = numpy.hstack(frames)


    # close stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    #plot vector

   

            
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
##def wavToVector(inputFile):
##    
##   # Read file to get buffer                                                                                               
##    ifile = wave.open(inputFile)
##    samples = ifile.getnframes()
##    audio = ifile.readframes(samples)
##
##    #convert buffer to int16 using numpy
##    audio_as_np_int16 = numpy.frombuffer(audio, dtype=numpy.int16)
##
##    return audio_as_np_int16

##    print(audio_as_np_int16) 
    
def wavToText(inputSoundFile):
    
    
    r = sr.Recognizer()

    hellow=sr.AudioFile(inputSoundFile)
    with hellow as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        print("Text: "+s)
        return s 
    except Exception as e:
        print("Exception: "+str(e))
        return 
        
        
def WAVFILE_to_arr(file):
    data = wave.open(file)
    samples = data.getnframes()
    audio = data.readframes(samples)

    arr = numpy.frombuffer(audio, dtype=numpy.int16)
    return arr


def WAVFILE_to_FFT_arr(file, rate=44100):
    arr = WAVFILE_to_arr(file)
    
    #Number of samples in normalized_tone
    N = len(arr)
    print("N is length: %d" % N)

    yf = rfft(arr)
    xf = rfftfreq(N, 1/rate)

    return xf,yf


def plotAudio(inputbuffer):
    fig, tx = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
    plt.plot(inputbuffer,color="blue")
    tx.set_xlim((0,len(inputbuffer)))
    plt.show()

def plot_WAVFILE(file):
    arr = WAVFILE_to_arr(file)
    #print(arr)
    plotAudio(arr)


def plot_WAVFILE_FFT(file):
    xf, yf = WAVFILE_to_FFT_arr(file)

    plt.plot(xf, numpy.abs(yf))
    plt.show()


    

def runAnalysis():
    dirs = os.listdir()

    print("------FILE MENU------")
    
    for s in range(len(dirs)):
        print("["+str(s)+"]"+ " " + dirs[s])
    print("---------------------")

    inputFileNumber = int(input("Enter File Number: "))

    if os.path.isdir(dirs[inputFileNumber]):
        newdirs = os.listdir()
        print(newdirs)
    else:
        print("not a folder")

    print(WAVFILE_to_arr(dirs[inputFileNumber]))
    WAVFILE_to_FFT_arr(dirs[inputFileNumber])
    plot_WAVFILE(dirs[inputFileNumber])
    plot_WAVFILE_FFT(dirs[inputFileNumber])
    stringList = wavToText(dirs[inputFileNumber]).split(' ')
    stringList = numpy.vstack(stringList)
    
 
    stringDataFrame = pd.DataFrame(stringList)
     
    stringDataFrame.transpose().to_excel('output.xlsx', header=False, index=False)

    

        
def main():

    enterChoice = input("Record (Press R) or Run Analysis (Press A): ")
    if enterChoice == "R":
        record_wav()
    elif enterChoice == "A":
        runAnalysis() 
    else:
        print("Invalid Response") 
##    record_wav()
##    
##
##    #pazth = "C:\Users\Vinay\OneDrive\1st Semester Senior Year and College Apps\Sound Files\"
##    dirs = os.listdir()
##
##    print("------FILE MENU------")
##    for s in range(len(dirs)):
##        
##        print("["+str(s)+"]"+ " " + dirs[s])
##    print("---------------------")
##        listSound= s.split('FILE')
##    print(listSound) 
        
  
    
        
    
##    wavToVector(input("Enter file: "))
    
##    inputFileNumber = int(input("Enter File Number: "))
##    
##    print(WAVFILE_to_arr(dirs[inputFileNumber]))
##    WAVFILE_to_FFT_arr(dirs[inputFileNumber])
##    plot_WAVFILE(dirs[inputFileNumber])
##    plot_WAVFILE_FFT(dirs[inputFileNumber])
##
##
####    WAVFILE_to_FFT_arr(input("Enter file: "))
####    plot_WAVFILE_FFT(input("Enter file: "))
##    
##  
##
##    stringList = wavToText(dirs[inputFileNumber]).split(' ')
##    stringList = numpy.vstack(stringList)
##     
##    pd.DataFrame(stringList.transpose()).to_excel('output.xlsx', header=False, index=False)
    
##    
##    #open text file
##    text_file = open("data.txt", "w")
## 
##    #write string to file
##    text_file.write(str(stringList))
## 
##    #close file
##    text_file.close()
##    
##    df = pd.read_table('data.txt')
##    df.to_excel('output.xlsx', 'Sheet1')
##    
    
main() 

#intialize id 
##enterName = input("Enter name:  ") 
##WAVE_OUTPUT_FILENAME = enterName+".wav"
##
### initialize portaudio
##p = pyaudio.PyAudio()
##print("Recording. Speak Now...")
##stream = p.open(format=pyaudio.paInt16, 
##                channels=1, 
##                rate=RATE, 
##                input=True, 
##                frames_per_buffer=CHUNKSIZE)
##
##frames = [] # A python-list of chunks(numpy.ndarray)
##for i in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
##    data = stream.read(CHUNKSIZE)
##    frames.append(numpy.frombuffer(data, dtype=numpy.int16))
##
#####Convert the list of numpy-arrays into a 1D array (column-wise)
####audioVector = numpy.hstack(frames)
####
####print("Stop speaking. Done recording")
##### close stream
####
####stream.stop_stream()
####stream.close()
####p.terminate()
####
####print(audioVector)
####print("Vector has length of %d" % len(audioVector))
##
###save audio file
##
##wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
##wf.setnchannels(CHANNELS)
##wf.setsampwidth(p.get_sample_size(FORMAT))
##wf.setframerate(RATE)
##wf.writeframes(b''.join(frames))
##wf.close()
##
##
###initialize numbers from audio file 
####def plotAudio(inputbuffer):
####    fig, tx = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
####    plt.plot(inputbuffer,color="blue")
####    tx.set_xlim((0,len(inputbuffer)))
####    plt.show()
####
####plotAudio(audioVector)
####
##### Number of samples in normalized_tone
####N = RATE * RECORD_SECONDS
####print("N is length: %d" % N)
######print("1/RATE is: %f" % (1/RATE))
####
######yf = rfft(audioVector)
######xf = rfftfreq(N, 1/RATE)
######
######plt.plot(xf[1:(len(xf)-67)], numpy.abs(yf))
######plt.show()
