import torch
import matplotlib.pyplot as plt
import numpy as np
import time

#number of threads for PyTorch
torch.set_num_threads(1)

# Load the model 
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

# Read audio file
wav = read_audio('nep.wav', sampling_rate=16000)
sampling_rate = 16000
chunk_duration = 40  # duration of each chunk in seconds

chunk_size = chunk_duration * sampling_rate  # size of each chunk in samples
num_chunks = len(wav) // chunk_size + (1 if len(wav) % chunk_size > 0 else 0)

# Prepare to store timestamps
all_timestamps = []

with open('nep_speech_timestamps.txt', 'a') as f:
    f.write("Start,End\n")

# Process each chunk
for i in range(num_chunks):
    print("Processing chunk ", i)
    start_sample = i * chunk_size
    end_sample = min((i + 1) * chunk_size, len(wav))
    chunk = wav[start_sample:end_sample]
    
    # Get speech timestamps for the chunk
    speech_timestamps = get_speech_timestamps(chunk, model, sampling_rate=sampling_rate, visualize_probs=False, return_seconds=True)
    
    # Store timestamps in the list
    all_timestamps.append({'chunk': i + 1, 'timestamps': speech_timestamps})
    
    # Save timestamps to a text file
    with open('nep_speech_timestamps.txt', 'a') as f:
        for timestamp in speech_timestamps:
            f.write(f"{timestamp['start'] + 40 * i:.2f}, {timestamp['end'] + 40 * i:.2f}\n")

    # Wait for few seconds before processing the next chunk
    if i < num_chunks - 1:
        print(f'Waiting for 7 seconds before processing the next chunk...')
        time.sleep(7)


# Read timestamps from the file
timestamps = []
with open('nep_speech_timestamps.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:  # Skip the header row
        start, end = map(float, line.strip().split(','))
        timestamps.append({'start': start, 'end': end})

# Plot the entire audio file
time = np.linspace(0, len(wav) / sampling_rate, len(wav))

plt.figure(figsize=(12, 4))
plt.plot(time, wav, color='blue')

for timestamp in timestamps:
    start = timestamp['start']
    end = timestamp['end']
    
    start_idx = int(start * sampling_rate)
    end_idx = int(end * sampling_rate)
    
    plt.axvspan(time[start_idx], time[end_idx], alpha=0.3, color='red')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio with Speech Timestamps')
plt.legend()
plt.savefig('speech_timestamps.png')
plt.close()

print('Processing complete. Timestamps saved to speech_timestamps.txt and plots saved as PNG files.')