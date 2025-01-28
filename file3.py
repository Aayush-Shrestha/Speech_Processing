# Python 3.8 on Macbook Pro M1 2020
import numpy as np
import struct 
import librosa # librosa==0.9.1
import webrtcvad # webrtcvad==2.0.10

# load data
file_path = 'ted.wav'

# load wav file (librosa)
y, sr = librosa.load(file_path, sr=16000)
# convert the file to int if it is in float (Py-WebRTC requirement)
if y.dtype.kind == 'f':
    # convert to int16
    y = np.array([ int(s*32768) for s in y])
    # bound
    y[y > 32767] = 32767
    y[y < -32768] = -32768

# create raw sample in bit
raw_samples = struct.pack("%dh" % len(y), *y)

# define webrtcvad VAD
vad = webrtcvad.Vad(2) # set aggressiveness from 0 to 3
window_duration = 0.03 # duration in seconds
samples_per_window = int(window_duration * sr + 0.5)
bytes_per_sample = 2 # for int16

# Start classifying chunks of samples
# var to hold segment wise report
segments = []
# iterate over the audio samples
for i, start in enumerate(np.arange(0, len(y), samples_per_window)):
    stop = min(start + samples_per_window, len(y))
    loc_raw_sample = raw_samples[start * bytes_per_sample: stop * bytes_per_sample]
    try:
        is_speech = vad.is_speech(loc_raw_sample, 
                              sample_rate = sr)
        segments.append(dict(
                start = start,
                stop = stop,
                is_speech = is_speech))
    except Exception as e:
        print(f"Failed for step {i}, reason: {e}")

import pandas as pd
df = pd.DataFrame(segments) # result of classification
print(df)