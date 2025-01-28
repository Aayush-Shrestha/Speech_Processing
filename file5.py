import torch
import matplotlib.pyplot as plt
import numpy as np

# Set the number of threads for PyTorch
torch.set_num_threads(1)

# Load the Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

# Function to split audio into 1-minute chunks
def split_audio_into_chunks(wav, chunk_duration=60, sampling_rate=16000):
    chunk_size = chunk_duration * sampling_rate
    return [wav[i:i + chunk_size] for i in range(0, len(wav), chunk_size)]

# Read the audio file
wav = read_audio('tnc.wav', sampling_rate=16000)

# Split the audio into 1-minute chunks
chunks = split_audio_into_chunks(wav)

# Process each chunk
for chunk_index, chunk in enumerate(chunks):
    speech_timestamps = get_speech_timestamps(chunk, model, sampling_rate=16000, visualize_probs=True, return_seconds=True)

    # Create time array for the chunk
    time = np.linspace(0, len(chunk) / 16000, len(chunk))

    # Plot the audio chunk
    plt.figure(figsize=(12, 4))
    plt.plot(time, chunk, color='blue')

    # Highlight speech timestamps
    for timestamp in speech_timestamps:
        start = timestamp['start']
        end = timestamp['end']
        
        start_idx = int(start * 16000)
        end_idx = int(end * 16000)
        
        plt.axvspan(time[start_idx], time[end_idx], alpha=0.3, color='red')

    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Audio Chunk {chunk_index + 1} with Speech Timestamps')
    plt.legend()
    plt.show()