import torch
import matplotlib.pyplot as plt
import numpy as np

torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

wav = read_audio('nep.wav', sampling_rate=16000)
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, visualize_probs=True, return_seconds=True)

time = np.linspace(0, len(wav) / 16000, len(wav))

plt.figure(figsize=(12, 4))
plt.plot(time, wav, color='blue')

for timestamp in speech_timestamps:
    start = timestamp['start']
    end = timestamp['end']
    
    start_idx = int(start * 16000)
    end_idx = int(end * 16000)
    
    plt.axvspan(time[start_idx], time[end_idx], alpha=0.3, color='red')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio with Speech Timestamps')
plt.show()

