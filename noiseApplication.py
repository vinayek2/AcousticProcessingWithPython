# import modules

import pyaudio
import numpy as np
from numpy.fft import rfft,rfftfreq
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq,irfft
# import scipy.fft as fft
from scipy.io.wavfile import write
import wave
import os 

RATE            = 44100
RECORD_SECONDS  = 2
CHUNKSIZE       = 1024
CHANNELS        = 1
FORMAT          = pyaudio.paInt16

# # initialize portaudio
# p = pyaudio.PyAudio()
# stream = p.open(format=FORMAT, 
#                 channels=CHANNELS, 
#                 rate=RATE, 
#                 input=True, 
#                 frames_per_buffer=CHUNKSIZE)


# frames = [] # A python-list of chunks(numpy.ndarray)
# for i in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
#     data = stream.read(CHUNKSIZE)
#     frames.append(numpy.frombuffer(data, dtype=numpy.int16))


# #Convert the list of numpy-arrays into a 1D array (colums)
# audioVector = numpy.hstack(frames)


# # close stream
# stream.stop_stream()
# stream.close()
# p.terminate()

# #print(audioVector)
# print("Vector has length of %d" % len(audioVector))

def WAVFILE_to_arr(file,verbose=False):
    data = wave.open(file)
    samples = data.getnframes()
    audio = data.readframes(samples)
    if verbose == True:
        print( "Number of channels",data.getnchannels())
        print ( "Sample width",data.getsampwidth())
        print ( "Frame rate.",data.getframerate())
        print ("Number of frames",data.getnframes())
        print ( "parameters:",data.getparams())

    arr = np.frombuffer(audio, dtype=np.int16)
    arr = arr.astype(np.float32)
    return arr


def WAVFILE_to_FFT_arr(file, rate=RATE,plot=False):
    arr = WAVFILE_to_arr(file)
    if plot:
        plt.plot(arr)
        plt.title("Numpy Array")
        plt.show()

    #Number of samples in normalized_tone
    N = len(arr)
    print("N is length: %d" % N)

    yf = np.fft.rfft(arr)
    xf = np.fft.rfftfreq(N, 1./rate)

    return xf,yf


def arr_to_WAVFILE(arr,filename, rate=44100):
    arr = arr.astype(np.int16)
    with wave.open(filename,"w") as f:
        f.setnchannels(CHANNELS)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(arr.tobytes())


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

    plt.plot(xf, np.abs(yf))
    plt.show()


def add_noise(arr,db,plot=False):

    #Find peak-to-peak of input signal
    print(max(arr))
    print(min(arr)) 
    arr_pk2pk = max(arr)-min(arr)
    print("[Signal] Peak-to-Peak = " + str(arr_pk2pk))

    #Calculate the needed noise peak-to-peak value for input SNR_dB
    noise_pk2pk = ((arr_pk2pk)/(10 ** (db/20)))
    print("[Noise] Peak-to-Peak = " + str(noise_pk2pk))
    
    #Caluate actual Ratio of dB
    ratio_val = 10 ** (db/20)
    print("[Noise] Ratio = " + str(ratio_val))

    #Reverse the np.random.normal's standard deviation spread 
    # ~ approximately [ 1 : (+4 to -4) ]
    noise_std = noise_pk2pk / 8
    #print("[Noise] Standard Deviation = " + str(noise_std))

    #Create Random Norma Distribution Array
    noise = np.random.normal(loc=0,scale=noise_std,size=len(arr))
    print("[Noise] Measured Peak-to-Peak = " + str(max(noise)-min(noise)))

    noise_signal = np.asarray(arr + noise)



    if plot == True:
        plt.plot(noise)
        plt.title("NOISE")
        plt.show()

        N = len(noise)
        print("N is length: %d" % N)

        yf = np.fft.rfft(noise)
        xf = np.fft.rfftfreq(N, 1./RATE)

        plt.plot(xf,abs(yf))
        plt.title("noise Freq")
        plt.show()

    return noise_signal

# Start with .wav
# Convert to arr and plot
# Convert to freq and plot
# Add noise and plot

# newArr = WAVFILE_to_arr("tone_5000Hz.wav")
# plt.plot(newArr)
# plt.title("FFT of  tone_5000Hz.wav")
# plt.show()
dirs = os.listdir()

print("------FILE MENU------")
for s in range(len(dirs)):
    
    print("["+str(s)+"]"+ " " + dirs[s])
print("---------------------")

inputNumber = int(input("Choose Master File Number : "))

    
FILE = dirs[inputNumber]
fileList = FILE.split('_')
NEW_FOLDER = str(fileList[0])+"_wavs"
if not os.path.exists(NEW_FOLDER):
    try:
        os.makedirs(NEW_FOLDER)
    except:
        print("Cannot make new folder: " + str(NEW_FOLDER))
        exit
    

signal = WAVFILE_to_arr(FILE,verbose=False)

plt.plot(signal)
plt.title(FILE)
plt.show()

noise_levels = [100, 50, 10, 5, 4, 3, 2, 1 , 0,-1, -2, -3, -4, -5, -20]

for lvl in noise_levels:
    print("---------------------START---------------------")
    print("Noise Level Set to:  " + str(lvl) + "dB")
    
    output_signal = add_noise(signal,lvl,plot=False)

    plt.plot(output_signal)
    plt.title(FILE + " + NOISE[" + str(lvl) + "]")
    plt.show()

    OUTPUT_FILE = NEW_FOLDER + "/maya_noiselevel_db" + str(lvl) + ".wav"
    arr_to_WAVFILE(output_signal,OUTPUT_FILE)
    print("----------------------END----------------------\n\n")
