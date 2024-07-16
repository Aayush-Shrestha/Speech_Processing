import torch
from pydub import AudioSegment
import os

audio_file = 'nep.wav'

torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

wav = read_audio(audio_file, sampling_rate=16000)
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, visualize_probs=True, return_seconds=True)

print(speech_timestamps)

subdir = os.path.splitext(audio_file)[0]
if not os.path.exists(subdir):
    os.makedirs(subdir)

audio = AudioSegment.from_wav(audio_file)

speech_timestamps_ms = [{'start': int(start * 1000), 'end': int(end * 1000)} for start, end in [(t['start'], t['end']) for t in speech_timestamps]]

for i, timestamp in enumerate(speech_timestamps_ms):
    chunk = audio[timestamp['start']:timestamp['end']]
    chunk.export(os.path.join(subdir, f'chunk_{i+1}.wav'), format='wav')

print(f"Audio chunks have been saved in the directory: {subdir}")