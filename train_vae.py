import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

import time
import argparse
import json
import math
import os

from wavenet_model import *
from wavenet_modules import *
from audio_data import WavenetDataset, AudioDataset
from vae import VAE, WaveNetEncoder, WaveNetDecoder, MultimodalWaveNetEncoder, OneHotConvolutionalEncoder, BetterWaveNetDecoder, OneHotUnstridedConvolutionalEncoder
from vae_gaussian import GaussianVAE, GaussianWaveNetEncoder, GaussianWaveNetDecoder
from convolutional_encoder import ConvolutionalEncoder
from mfcc_convolutional_encoder import MFCCConvolutionalEncoder

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from wavenet_utils import parameter_count, get_max_free_kl_divergence

from apex import amp

dtype = torch.FloatTensor
ltype = torch.LongTensor

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    step = checkpoint_dict['step']
    # optimizer.load_state_dict()
    model.load_state_dict(checkpoint_dict['model'])
    print("Loaded checkpoint '{}' (step {})" .format(checkpoint_path, step))
    return model, checkpoint_dict['optimizer'], step

def save_checkpoint(model, optimizer, step, filepath):
    torch.save({'model': model.state_dict(),
                'step': step,
                'optimizer': optimizer.state_dict(),
                }, filepath)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str,
                    help='JSON file for configuration', required=True)
args = parser.parse_args()
with open(args.config) as f:
    data = f.read()
config = json.loads(data)
encoder_wavenet_args = config["encoder_wavenet_args"]
decoder_wavenet_args = config["decoder_wavenet_args"]
train_args = config["train_args"]
batch_size = train_args["batch_size"]
epochs = train_args["epochs"]
continue_training_at_step = train_args["continue_training_at_step"]
snapshot_name = os.path.splitext(os.path.basename(args.config))[0]
snapshot_path = f"snapshots/{snapshot_name}"
snapshot_interval = train_args["snapshot_interval"]
weight_decay  = train_args["weight_decay"]
lr = train_args["lr"]
free_bits_per_dimension = train_args["free_bits_per_dimension"]
penalize_free_bits_linearly = train_args.get("penalize_free_bits_linearly", False)
device_name = config["device"]
gpu_index = config["gpu_index"]
dataset_path = config["dataset_path"]
load_path = config["load_path"]
n_z_samples = config["n_z_samples"]
type = config.get("type", "categorical")
flip_decoder = config.get("flip_decoder", True)
use_apex = config.get("use_apex", False)
data_length = config.get("data_length", None)

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
    print(f"creating path {snapshot_path}")

if type in {"categorical", "multimodal", "convolutional", "betterwavenet", "mfcc_betterwavenet", "unstrided_betterwavenet"}:
    ar_factor = train_args.get("ar_factor", None)
    hard_gumbel_softmax = train_args.get("hard_gumbel_softmax", False)
    gumbel_softmax_temperature = train_args["gumbel_softmax_temperature"]
    posterior_entropy_penalty_coeff = train_args["posterior_entropy_penalty_coeff"]
    if not hard_gumbel_softmax:
        temp_min = train_args["temp_min"]
        anneal_rate = train_args["anneal_rate"]

if type == "categorical":
    model = VAE(
        WaveNetEncoder(encoder_wavenet_args),
        WaveNetDecoder(decoder_wavenet_args, flip_decoder),
        hard_gumbel_softmax=hard_gumbel_softmax,
    )
elif type == "multimodal":
    model = VAE(
        MultimodalWaveNetEncoder(encoder_wavenet_args),
        WaveNetDecoder(decoder_wavenet_args, flip_decoder),
        hard_gumbel_softmax=hard_gumbel_softmax,
    ) 
elif type == "convolutional":
    model = VAE(
        OneHotConvolutionalEncoder(encoder_wavenet_args),
        WaveNetDecoder(decoder_wavenet_args, flip_decoder),
        hard_gumbel_softmax=hard_gumbel_softmax,
    ) 
elif type == "betterwavenet":
    model = VAE(
        OneHotConvolutionalEncoder(encoder_wavenet_args),
        BetterWaveNetDecoder(decoder_wavenet_args),
        hard_gumbel_softmax=hard_gumbel_softmax,
    ) 
elif type == "mfcc_betterwavenet":
    model = VAE(
        MFCCConvolutionalEncoder(**encoder_wavenet_args),
        BetterWaveNetDecoder(decoder_wavenet_args),
        hard_gumbel_softmax=hard_gumbel_softmax,
    ) 
elif type == "unstrided_betterwavenet":
    model = VAE(
        OneHotUnstridedConvolutionalEncoder(encoder_wavenet_args),
        BetterWaveNetDecoder(decoder_wavenet_args),
        hard_gumbel_softmax=hard_gumbel_softmax,
    ) 
elif type == "gaussian":
    use_continuous_one_hot = config["use_continuous_one_hot"]
    model = GaussianVAE(
        GaussianWaveNetEncoder(encoder_wavenet_args, use_continuous_one_hot),
        GaussianWaveNetDecoder(decoder_wavenet_args, use_continuous_one_hot, flip_decoder),
        n_z_samples = n_z_samples
    )
else:
    raise Exception(f"Network type {type} not implemented.")

if data_length is None:
    receptive_field_closest_power_of_3 = int(3**math.ceil(math.log(model.decoder.wavenet.receptive_field, 3)))
    data_length = receptive_field_closest_power_of_3 * 3
dataset = AudioDataset(dataset_path, data_length)        

print(f"the receptive field length is {model.decoder.wavenet.receptive_field}")
print(f'the dataset has {str(len(dataset))}  items')
print(f'each item has length {dataset.len_sample}')
n_encoder_params = parameter_count(model.encoder)
n_decoder_params = parameter_count(model.decoder)
n_parameters = n_encoder_params + n_decoder_params
print(f'The WaveNetVAE has {n_parameters} parameters')
print(f'The encoder has {n_encoder_params} parameters')
print(f'The decoder has {n_decoder_params} parameters')
print(f'The maximum free kl divergence is {get_max_free_kl_divergence(n_inputs=dataset.len_sample, free_bits_per_dimension=free_bits_per_dimension)} nats.')

print('start training...')
clip = None

pin_memory = False
device = torch.device(device_name)
if not device_name == "cpu":
    # torch.cuda.set_device(gpu_index)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = model.to(device)
    pin_memory = True

optimizer=optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
if "optimizer_state_dict" in globals():
    optimizer.load_state_dict(optimizer_state_dict)

### APEX MIXED-PRECISION
if use_apex:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

model = nn.DataParallel(model)
if not load_path == "":
    (_, optimizer_state_dict, continue_training_at_step) = load_checkpoint(load_path, model)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=pin_memory,
)

writer = SummaryWriter(comment="_"+snapshot_name)
step = continue_training_at_step
model.train()


print("epochs: ", epochs)
for current_epoch in range(epochs):
    print("\nepoch\n", current_epoch)
    tic = time.time()
    for x in iter(dataloader):
        x = x.long().to(device)
        
        with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
            if type in {"categorical", "multimodal", "convolutional", "betterwavenet", "mfcc_betterwavenet", "unstrided_betterwavenet"}:
                if step % 100 == 0 and not hard_gumbel_softmax:
                    gumbel_softmax_temperature = np.maximum(1. - anneal_rate * step, temp_min)
                p_x, q_z = model(x, gumbel_softmax_temperature)
                posterior_entropy_penalty_coeff_annealed = (
                    posterior_entropy_penalty_coeff if step < 10_000 else
                    posterior_entropy_penalty_coeff * max(0, 1 - (step-10_000)/10_000))
                d = model.module.loss(
                    p_x, x, q_z, posterior_entropy_penalty_coeff_annealed,
                    ar_factor, free_bits_per_dimension, penalize_free_bits_linearly
                )
                loss = d['loss']
                cross_entropy = d['cross_entropy']
                kl_divergence = d['kl_divergence']
                posterior_entropy_penalty = d['posterior_entropy_penalty']
            elif type == "gaussian":
                p_x, mu, logvar = model(x)
                loss, cross_entropy, kl_divergence = model.module.loss(p_x, x, mu, logvar)
            else:
                raise Exception(f"No loss calculation is implemented for type {type}")
            
            optimizer.zero_grad()
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            loss = loss.item()
            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            step += 1
        if prof is not None:
            prof.export_chrome_trace("chrome_trace")

        if step == 100:
            toc = time.time()
            print("one training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

        if step % snapshot_interval == 0:
            if snapshot_path is None:
                continue
            path = snapshot_path + '/' + snapshot_name + '_' + str(step)
            save_checkpoint(model, optimizer, step, path)
        if step % 10 == 0:
            bits_per_dimension = ((kl_divergence+cross_entropy)/math.log(2)) / x.size(-1)
            writer.add_scalar('Loss/train', loss, global_step=step)
            writer.add_scalar('bits/dimension', bits_per_dimension, global_step=step)
            writer.add_scalar('KL divergence', kl_divergence, global_step=step)
            writer.add_scalar('cross entropy', cross_entropy, global_step=step)
            writer.add_scalar('posterior entropy penalty', posterior_entropy_penalty, global_step=step)
            print(f"\rstep {step}, bits/dimension {bits_per_dimension}".ljust(70), end='\r')

print("finished")
