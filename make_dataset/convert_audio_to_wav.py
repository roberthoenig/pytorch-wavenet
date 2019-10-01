import subprocess
import os

for i, filename in enumerate(os.listdir(os.getcwd())):
    if filename.endswith(".m4a"):
        wav_filename = f"{i}.wav"
        print(f"Converting file {filename} to file {wav_filename}")
        subprocess.run(["ffmpeg", "-i", filename, '-ac', '1', '-ar', '16000', wav_filename])