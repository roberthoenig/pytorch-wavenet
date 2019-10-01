import os
import torch
import torchaudio

# Load wav files into tensors
wav_directory = os.path.abspath("../wavs/vctk_wavs")
wav_tensors = []
mu_law_encode = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
for i, filename in enumerate(os.listdir(wav_directory)):
    if i%1000 == 0:
        print(i)
    if filename.endswith(".wav"):
        # wav_tensor has dimensions (n_channels, length), where channels corresponds to
        # audio recording channels
        wav_tensor, sample_rate = torchaudio.load_wav(os.path.join(wav_directory, filename))
        # We only support a single audio channel.
        assert wav_tensor.size(0) == 1
        # Rescale wav_tensor to [-1, 1]
        wav_tensor /= 2**15
        # Rescale wav_tensor to [256] with mu law
        mu_wav_tensor = mu_law_encode(wav_tensor).type(torch.int)
        wav_tensors.append(mu_wav_tensor)
# Concatenate tensors
output_wav_tensor = torch.cat(wav_tensors, dim=1).squeeze()

torch.save(output_wav_tensor, 'wav_tensor.pt')
