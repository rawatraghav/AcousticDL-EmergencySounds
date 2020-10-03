import os
import wave
import pylab
import struct

def graph_spectrogram(wav_file,i):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(0.28, 0.28))
    pylab.subplot(111) 
    pylab.axis('off')
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig("sound_"+str(i)+".png",transparent=True)
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

for i in range(201,401):
     graph_spectrogram("sound_"+str(i)+".wav",i)

